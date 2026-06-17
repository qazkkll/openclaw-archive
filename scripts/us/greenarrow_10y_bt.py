# -*- coding: utf-8 -*-
"""
绿箭极致 vs 原版 — 10年完整回测（2016-2026）
参数:
  极致: T7_H10_S20_R5
  原版: T5_H10_S15_R10
  
使用V7.5模型（训练期2020-2024）
回测期: 2016-10-01 ~ 2026-06-10
"""
import os, json, pickle, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, xgboost as xgb

print('绿箭极致 vs 原版 — 10年回测'); print('='*60); t0=time.time()
BASE='/home/hermes/.hermes/openclaw-archive'; ML=f'{BASE}/ml'; MD=f'{BASE}/data/models'

model=xgb.Booster(); model.load_model(f'{MD}/us_v7_5.json')
cal=pickle.load(open(f'{MD}/us_v7_5_calibrator.pkl','rb'))
report=json.load(open(f'{MD}/us_v7_5_report.json'))
FEATS=report['features']

df=pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str']=df['date'].astype(str).str[:10]
for f in FEATS:
    if f in df.columns: df[f]=pd.to_numeric(df[f],errors='coerce').fillna(0).clip(-1e6,1e6)
df=df.replace([np.inf,-np.inf],np.nan)

fl=json.load(open(f'{ML}/us_filtered_syms_v5.json'))
pool=set(fl['syms'])
df=df[df['sym'].isin(pool)].copy()

all_dates=sorted(df['date_str'].unique())
# 截取2016-10-01后
bt_dates=[d for d in all_dates if d>='2016-10-01']
print(f'回测期: {len(bt_dates)}天 [{bt_dates[0]}..{bt_dates[-1]}]')
print(f'候选池: {df.sym.nunique()}只')

idx_path=f'{ML}/us_v75_close_idx_v4.pkl'
open_idx,close_idx=pickle.load(open(idx_path,'rb'))

# 评分
print('评分...')
daily_probs={}
n_batch=20000
for i in range(0,len(df),n_batch):
    chunk=df.iloc[i:i+n_batch]
    X=np.nan_to_num(chunk[FEATS].values.astype(np.float32),nan=0)
    raw=model.predict(xgb.DMatrix(X,feature_names=FEATS))
    calib=cal.predict_proba(raw.reshape(-1,1))[:,1]
    for j,(_,r) in enumerate(chunk.iterrows()):
        d=r['date_str']
        if d not in daily_probs: daily_probs[d]={}
        daily_probs[d][r['sym']]=float(calib[j])

def build_cands(dates):
    cands={}
    for di,d in enumerate(dates):
        nxt_d=dates[di+1] if di+1<len(dates) else None
        if nxt_d is None: continue
        probs=daily_probs.get(d,{})
        if len(probs)<30: continue
        picks=[]
        for sym,p in probs.items():
            if sym not in pool: continue
            nxt_price=open_idx.get(sym,{}).get(nxt_d)
            if nxt_price is None or nxt_price<=0: continue
            picks.append((sym,p,float(nxt_price)))
        picks.sort(key=lambda x:-x[1])
        cands[d]=picks
    return cands

def bt(label, dates, cands, T, H, S, R):
    cap=10000.0; cash=cap; port={}; trds=0; curve=[cap]; eq_peak=cap
    sl=S/100.0
    for di,d in enumerate(dates):
        # 卖出
        for sym in list(port.keys()):
            pos=port[sym]; cp=close_idx.get(sym,{}).get(d)
            if cp is None: continue
            ret=(cp-pos['bp'])/pos['bp']
            if ret<=-sl or (di-pos['di'])>=H:
                cash+=pos['qty']*cp; trds+=1
                del port[sym]
        # 买入
        if di%R==0 or len(port)<T:
            picks=cands.get(d,[])
            if len(picks)>=3:
                slots=T-len(port)
                for sym,p,price in picks[:slots]:
                    if sym in port: continue
                    budget=cash/max(1,len(port)+1)
                    qty=int(budget/price)
                    if qty<=0: continue
                    cash-=qty*price
                    port[sym]={'qty':qty,'bp':price,'di':di}
        # 记录
        pv=cash+sum(port[s]['qty']*close_idx.get(s,{}).get(d,0) for s in port)
        curve.append(pv)
        if pv>eq_peak: eq_peak=pv
    
    eq=np.array(curve)
    ann=(eq[-1]/cap)**(252/len(eq))-1
    rets=(eq[1:]-eq[:-1])/eq[:-1]; v=rets.std()*np.sqrt(252)
    sh=ann/max(v,1e-8)
    
    # 最大回撤
    peak_cum=np.maximum.accumulate(eq/cap)
    dd=eq/cap/peak_cum-1
    mdd=float(dd.min())
    
    # 年度分解
    years={}
    eq_df=pd.Series(eq,index=[None]+dates)
    eq_df.index=pd.to_datetime([bt_dates[0]]+dates)
    # 按年
    yr_groups=eq_df.resample('YE')
    prev=cap
    for yr,grp in yr_groups:
        yr_str=yr.strftime('%Y')
        if len(grp)<10: continue
        val=float(grp.iloc[-1])
        r=val/prev-1
        years[yr_str]=round(r,4)
        prev=val
    
    return {'label':label,'ann':round(ann,4),'sh':round(sh,4),'mdd':round(mdd,4),
            'final':round(float(eq[-1]),2),'trades':trds,'total_days':len(dates),
            'years':years,'win_rate':round(float((rets>0).mean()),4)}

print('构建候选...')
cands=build_cands(bt_dates)
print(f'候选: {sum(len(v) for v in cands.values()):,}条')

# 按市场周期拆分
print()
print('回测中...')
periods=[('全部(2016-2026)',bt_dates[:])]

# 手动分段
def date_slice(start,end):
    return [d for d in bt_dates if d>=start and d<end]

periods+=[('2017-2019(慢牛)',date_slice('2017-01-01','2020-01-01')),
          ('2020(疫情崩+反弹)',date_slice('2020-01-01','2021-01-01')),
          ('2021(牛市)',date_slice('2021-01-01','2022-01-01')),
          ('2022(加息熊市)',date_slice('2022-01-01','2023-01-01')),
          ('2023(反弹)',date_slice('2023-01-01','2024-01-01')),
          ('2024(AI牛市)',date_slice('2024-01-01','2025-01-01')),
          ('2025-2026(大涨)',date_slice('2025-01-01','2027-01-01'))]

all_res=[]
for label,ds in periods:
    if len(ds)<50: continue
    r1=bt(f'极致_T7_H10_S20_R5',ds,cands,T=7,H=10,S=20,R=5)
    r2=bt(f'原版_T5_H10_S15_R10',ds,cands,T=5,H=10,S=15,R=10)
    all_res+=[r1,r2]
    r1['period']=label; r2['period']=label
    print(f'{label}:')
    print(f'  极致: +{r1["ann"]*100:>7.1f}% 夏普={r1["sh"]:>5.2f} 回撤={r1["mdd"]*100:>6.1f}% 交易={r1["trades"]:>4}')
    print(f'  原版: +{r2["ann"]*100:>7.1f}% 夏普={r2["sh"]:>5.2f} 回撤={r2["mdd"]*100:>6.1f}% 交易={r2["trades"]:>4}')
    print()

# ===== 10年完整结果 =====
print('='*80)
print('10年完整回测结果')
print('='*80)
print(f'{"策略":<22} {"年化":>8} {"夏普":>6} {"回撤":>7} {"胜率":>6} {"交易":>4} {"终值(万)":>9}')
print('-'*70)
for r in all_res:
    if r['period']=='全部(2016-2026)':
        print(f'{r["label"]:<22} {r["ann"]*100:>7.1f}% {r["sh"]:>6.2f} {r["mdd"]*100:>7.1f}% {r["win_rate"]*100:>5.1f}% {r["trades"]:>4} ${r["final"]/10000:>8,.1f}')

# 按年输出
print()
print('='*80)
print('分年度收益')
print('='*80)
years_set=sorted(set(y for r in all_res if r['period']=='全部(2016-2026)' for y in r['years'].keys()))
yr_labels=[y for y in years_set if y!='2026' or y in ['2025','2026']]

# 找极致和原版的10年结果
full_e=[r for r in all_res if r['period']=='全部(2016-2026)' and '极致' in r['label']][0]
full_o=[r for r in all_res if r['period']=='全部(2016-2026)' and '原版' in r['label']][0]

all_yrs=sorted(full_e['years'].keys())
print(f'{"年度":<8} {"极致_T7_H10_S20_R5":>16} {"原版_T5_H10_S15_R10":>16}')
print('-'*42)
for yr in all_yrs:
    ey=full_e['years'].get(yr)
    oy=full_o['years'].get(yr)
    es=f'+{ey*100:.1f}%' if ey and ey>0 else f'{ey*100:.1f}%' if ey else 'N/A'
    os=f'+{oy*100:.1f}%' if oy and oy>0 else f'{oy*100:.1f}%' if oy else 'N/A'
    print(f'{yr:<8} {es:>16} {os:>16}')

# 分阶段对比
print()
print('='*80)
print('分市场周期对比')
print('='*80)
print(f'{"阶段":<22} {"极致_年化":>9} {"极致_夏普":>9} {"极致_回撤":>9} {"原版_年化":>9} {"原版_夏普":>9} {"原版_回撤":>9}')
print('-'*80)
for label,ds in periods:
    if len(ds)<50: continue
    es=[r for r in all_res if r['period']==label and '极致' in r['label']]
    os=[r for r in all_res if r['period']==label and '原版' in r['label']]
    if not es or not os: continue
    e=es[0]; o=os[0]
    print(f'{label:<22} {e["ann"]*100:>8.1f}% {e["sh"]:>8.2f} {e["mdd"]*100:>8.1f}% {o["ann"]*100:>8.1f}% {o["sh"]:>8.2f} {o["mdd"]*100:>8.1f}%')

# 保存
json.dump({'results':all_res,'time':time.strftime('%Y-%m-%d %H:%M')},
          open(f'{MD}/greenarrow_10y_backtest.json','w'),indent=2)
print(f'\n完成({time.time()-t0:.0f}s)')
