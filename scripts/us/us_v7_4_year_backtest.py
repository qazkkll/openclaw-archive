#!/usr/bin/env python3
"""
v7.4 全周期回测 — 2022~2024 验证 + 2025测试
用已经算好的概率，分年度看稳定性
"""
import sys,os,json,pickle,time,warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd,numpy as np,xgboost as xgb

BASE='/home/hermes/.hermes/openclaw-archive'; ML_DIR=f'{BASE}/ml'; MD=f'{BASE}/data/models'; VER='us_v7_4'
print('='*70,flush=True); print('v7.4 全周期分年回测',flush=True); print('='*70,flush=True)
T0=time.time()

# 1. 加载
print('\n[1] 加载...',flush=True)
model=xgb.Booster(); model.load_model(f'{MD}/{VER}.json')
cal=pickle.load(open(f'{MD}/{VER}_calibrator.pkl','rb'))
df=pd.read_parquet(f'{ML_DIR}/us_ml_feats_v71_v19.parquet')
FEATS=['price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
    'ret1','ret5','ret20','ret60','macd','macd_signal','macd_hist',
    'vol_ratio','ma_bias20','vol5','trend_accel',
    'short_ratio','short_pct','market_cap',
    'sector_etf_ret5','spy_ret5','qqq_ret5','iwm_ret5','sc']

df=df.replace([np.inf,-np.inf],np.nan)
df=df.dropna(subset=FEATS+['label_5d_5class','label_5d_pct'])
for f in FEATS: df[f]=pd.to_numeric(df[f],errors='coerce').fillna(0).clip(-1e6,1e6)
df=df.sort_values(['sym','date']).reset_index(drop=True)

dates=sorted(df['date'].unique())
print(f'  数据: {len(df):,}行, {df.sym.nunique()}只',flush=True)
print(f'  日期范围: {dates[0]} ~ {dates[-1]}',flush=True)

# 2. 全量概率计算（分批）
print('\n[2] 全量概率计算...',flush=True)
probs={}
n_batch=5000
for i in range(0,len(df),n_batch):
    pct=100*i//len(df)
    if pct%20==0: print(f'  {pct}%...',flush=True)
    batch=df.iloc[i:i+n_batch].dropna(subset=FEATS)
    if len(batch)==0: continue
    X=np.nan_to_num(batch[FEATS].values.astype(np.float32),nan=0)
    raw=model.predict(xgb.DMatrix(X))
    if raw.ndim>1: raw=raw[:,4]
    calib=cal.predict_proba(raw.reshape(-1,1))[:,1]
    for j,idx in enumerate(batch.index): probs[idx]=float(calib[j])
print(f'  {len(probs):,}行',flush=True)

# 3. 构建每日候选索引
print('\n[3] 构建索引...',flush=True)
day_cands={}
for d in dates:
    dd=str(d)[:10]
    day=df[df['date']==d].copy()
    if len(day)<30: continue
    day['p']=day.index.map(lambda i:probs.get(i,0))
    day=day.dropna(subset=['p'])
    cands=[(r['sym'],r['p'],float(r['price'])) for _,r in day.iterrows() if r['p']>0]
    cands.sort(key=lambda x:-x[1])
    day_cands[dd]=cands

# 日价格索引
date_prices={str(d)[:10]:{} for d in dates}
for _,r in df.iterrows():
    d=str(r['date'])[:10]
    if d in date_prices:
        date_prices[d][r['sym']]=float(r['price'])
print(f'  日索引: {len(day_cands)}天',flush=True)

# 4. 按年度回测
print('\n[4] 按年度回测 (策略: T10_H10_S15_R5)...',flush=True)

YEARS={'2022':'2022-01-01~2022-12-31','2023':'2023-01-01~2023-12-31',
       '2024':'2024-01-01~2024-12-31','2025':'2025-01-01~2025-12-31',
       '2026':'2026-01-01~2026-12-31'}
STRATEGY={'top_n':10,'hold':10,'stop':15}

year_results = {}

for year,yrange in YEARS.items():
    ystart,ystart_date = yrange.split('~')
    yend = ystart_date if '2026' not in year else '2026-06-11'
    
    # 找在范围内的日期
    year_days=[d for d in sorted(day_cands.keys()) if d>=ystart and d<=yend]
    if len(year_days)<50:
        year_results[year]={'days':len(year_days),'status':'不足50天,跳过'}
        continue
    
    cap=10000.0; cash=cap; portfolio={}; trades=0; wins=0; curve=[cap]
    sl=0.15; top_n=10; hold=10; rebal=5
    
    for day_idx,d in enumerate(year_days):
        prices_today=date_prices.get(d,{})
        
        # 止损+到期
        for sym in list(portfolio.keys()):
            pos=portfolio[sym]
            cp=prices_today.get(sym)
            if cp is None: continue
            days_h=(pd.to_datetime(d)-pd.to_datetime(str(pos['bd'])[:10])).days
            ret=(cp-pos['bp'])/pos['bp']
            if ret<=-sl or days_h>=hold:
                cash+=pos['qty']*cp; trades+=1
                if cp>=pos['bp']: wins+=1
                del portfolio[sym]
        
        # 调仓 (每5天)
        if day_idx%rebal==0:
            cands=[c for c in day_cands.get(d,[]) if c[0] not in portfolio]
            buys=cands[:top_n]
            for sym,prob,price in buys:
                qty=cash/max(top_n,1)/max(price,0.01)
                if qty<1: continue
                portfolio[sym]={'bd':d,'bp':price,'qty':qty}
                cash-=qty*price
        
        # 净资产
        pv=sum(p['qty']*prices_today.get(s,p['bp']) for s,p in portfolio.items())
        curve.append(cash+pv)
    
    # 清仓
    final=cash+sum(p['qty']*p['bp'] for p in portfolio.values())
    ec=np.array(curve)
    tr=(final/10000-1)*100
    yrs=len(year_days)/252
    an=((final/10000)**(1/max(yrs,0.01))-1)*100 if yrs>0 else 0
    peak=np.maximum.accumulate(ec)
    mdd=(ec-peak).min()/peak.max()*100 if peak.max()>0 else 0
    dr=np.diff(ec)/(ec[:-1]+1e-10)
    sh=(dr.mean()/max(dr.std(),1e-6))*np.sqrt(252) if len(dr)>20 else 0
    
    year_results[year]={
        'days':len(year_days),
        'total_return':round(tr,1),
        'annualized':round(an,1),
        'sharpe':round(sh,2),
        'max_dd':round(mdd,1),
        'wins':wins,
        'trades':trades,
        'win_rate':round(wins/max(trades,1),3),
        'final':round(final,0)
    }
    print(f'  {year}: {year_days[0]}~{year_days[-1]} ({len(year_days)}天) '
          f'收益{tr:>6.1f}% 年化{an:>6.1f}% 夏普{sh:>5.2f} 回撤{mdd:>5.1f}%',flush=True)

# 5. 汇总
print('\n[5] 汇总',flush=True)
print('='*60)
print(f'{"年度":>6s} {"天数":>5s} {"收益":>7s} {"年化":>7s} {"夏普":>6s} {"回撤":>7s} {"胜率":>6s} {"交易":>5s}')
print('-'*60)
for yr in ['2022','2023','2024','2025','2026']:
    r=year_results.get(yr,{})
    if isinstance(r,dict) and 'annualized' in r:
        print(f'{yr:>6s} {r["days"]:>5d} {r["total_return"]:>6.1f}% {r["annualized"]:>6.1f}% '
              f'{r["sharpe"]:>6.2f} {r["max_dd"]:>6.1f}% {r["win_rate"]:>5.1%} {r["trades"]:>5}')
    else:
        print(f'{yr:>6s} - {r.get("status","无数据")}')

# 全周期
print('\n--- 全周期 (2022~2025) ---')
total_days=sum(r.get('days',0) for r in year_results.values() if isinstance(r,dict) and 'days' in r)
print(f'总天数: {total_days}天 (~{round(total_days/252,1)}年)')

# 保存
json.dump({'timestamp':'2026-06-11 10:35','model':VER,
    'strategy':'T10_H10_S15_R5','results':year_results,
},open(f'{MD}/us_v7_4_year_backtest.json','w'),indent=2)
print(f'\n保存: us_v7_4_year_backtest.json')
print(f'耗时: {time.time()-T0:.0f}s')
print('='*70)
