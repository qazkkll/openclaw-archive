#!/usr/bin/env python3
"""V7.5 今日最终买入信号"""
import json, pickle, numpy as np, pandas as pd, xgboost as xgb, minishare as ms
BASE='/home/hermes/.hermes/openclaw-archive'; ML=f'{BASE}/ml'; MD=f'{BASE}/data/models'
api=ms.pro_api('Jarvne6fmgArRa46Xfon0e1kw55E6hes5IB2Fy2X0ndqnvrL48jsVOtTbf014f06')

df=pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str']=df['date'].astype(str).str[:10]
model=xgb.Booster()
model.load_model(f'{MD}/us_v7_5.json')
cal=pickle.load(open(f'{MD}/us_v7_5_calibrator.pkl','rb'))
report=json.load(open(f'{MD}/us_v7_5_report.json'))
FEATS=report['features']
feat_cols=[f for f in FEATS if f in df.columns]
for f in feat_cols: df[f]=pd.to_numeric(df[f],errors='coerce').fillna(0).clip(-1e6,1e6)
df=df.replace([np.inf,-np.inf],0)
latest_date=sorted(df['date_str'].unique())[-1]
latest=df[df['date_str']==latest_date].copy()
fl=json.load(open(f'{ML}/us_filtered_syms.json'))
valid=set(fl['syms'])
latest=latest[latest['sym'].isin(valid)].copy()

X=np.nan_to_num(latest[feat_cols].values.astype(np.float32),nan=0)
raw=model.predict(xgb.DMatrix(X,feature_names=feat_cols))
calib=cal.predict_proba(raw.reshape(-1,1))[:,1]
latest['prob_5pct']=calib
latest=latest.sort_values('prob_5pct',ascending=False)

# close_idx
open_idx,close_idx=pickle.load(open(f'{ML}/us_v75_close_idx_v4.pkl','rb'))

# 实时价
high_cands=latest[latest['prob_5pct']>=0.34]['sym'].tolist()
sym_list=high_cands+['SPY','QQQ','IWM','VIX']
sym_list=list(dict.fromkeys(sym_list))
print(f'拉取{len(sym_list)}只...', flush=True)
rt=api.rt_us_k(ts_code=','.join(sym_list[:50]),extFields='date')
rt_df=pd.DataFrame(rt)
rt_df['sym']=rt_df['ts_code'].str.upper()
live={r['sym']:float(r['close']) for _,r in rt_df.iterrows()}

avg_p=latest.head(50)['prob_5pct'].mean()
print(f'评分日:{latest_date} 市场热度:{avg_p:.3f} {"(热市,仓位<60%)" if avg_p>0.33 else "(正常)"}')
print(f'{"代码":>6s} {"评分":>7s} {"昨收":>7s} {"实时":>7s} {"涨幅":>7s}')
print('-'*38)
for _,r in latest.head(30).iterrows():
    sym=r['sym']
    cp=close_idx.get(sym,{}).get(latest_date,0)
    lp=live.get(sym,0)
    if lp and lp>0:
        chg=(lp-cp)/cp*100 if cp>0 else 0
        print(f'{sym:>6s} {r["prob_5pct"]:>6.1%} ${cp:>5.2f} ${lp:>5.2f} {chg:>+5.1f}%')
    else:
        print(f'{sym:>6s} {r["prob_5pct"]:>6.1%} ${cp:>5.2f} {"N/A":>7s}')

# 指数
print(f'\n--- 大盘 ---')
for x in ['SPY','QQQ','IWM','VIX']:
    if x in live:
        print(f'  {x}: ${live[x]:.2f}')

# 买入建议
print(f'\n--- 买入建议(候选7只,上限4只) ---')
buyable=[]
for sym in high_cands[:7]:
    cp=close_idx.get(sym,{}).get(latest_date,0)
    lp=live.get(sym,0)
    chg=(lp-cp)/cp*100 if lp and cp>0 else 0
    prob=float(latest[latest['sym']==sym].iloc[0]['prob_5pct'])
    if chg>5:
        rec='追涨不追'
    elif lp>0:
        rec='可买入'
        buyable.append(sym)
    else:
        rec='无实时价'
    print(f'  {sym} prob={prob:.1%} 昨收${cp:.2f}->${lp:.2f}({chg:+.1f}%) {rec}')

# 更新持仓
PORT='/home/hermes/.hermes/openclaw-project/data/portfolio_v75_extreme.json'
portfolio=[]
for sym in buyable[:4]:  # 最多4只(热市)
    lp=live.get(sym,0)
    prob=float(latest[latest['sym']==sym].iloc[0]['prob_5pct'])
    portfolio.append({'sym':sym,'action':'hold','days_held':1,'bp':lp,'last_price':lp,'prob':prob,'ret_pct':0.0})
json.dump(portfolio,open(PORT,'w'),indent=2)
print(f'\n持仓已写入: {PORT} ({len(portfolio)}只)')
