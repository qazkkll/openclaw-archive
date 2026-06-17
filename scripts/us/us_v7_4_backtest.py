#!/usr/bin/env python3
"""
v7.4 快速时序回测 v3
预索引所有数据到字典，用字典查找替代DataFrame过滤
"""
import sys,os,json,pickle,time,itertools,warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd,numpy as np,xgboost as xgb

BASE='/home/hermes/.hermes/openclaw-archive'; ML_DIR=f'{BASE}/ml'; MD=f'{BASE}/data/models'; VER='us_v7_4'
print('='*70,flush=True); print('v7.4 快速回测 v3',flush=True); print('='*70,flush=True)
T0=time.time()

# 1. 加载
print('\n[1] 加载索引...',flush=True)
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

# 索引: date→{sym→price} 和 sym→{date→price}
dates=sorted(df['date'].unique())
btd=[d for d in dates if str(d)>='2025-01-01']
print(f'  数据: {len(df):,}行, {len(btd)}回测天',flush=True)

# 构建快速索引
print('  索引构建...',flush=True)
date_prices={d:{} for d in btd}
sym_prices={s:{} for s in df['sym'].unique()}
for _,r in df[df['date'].isin(btd)].iterrows():
    d=str(r['date'])[:10]
    s=r['sym']
    p=float(r['price'])
    date_prices[d][s]=p
    sym_prices[s][d]=p
print(f'  日索引: {len(date_prices)}天, 股索引: {len(sym_prices)}只',flush=True)

# 2. 概率
print('\n[2] 概率计算...',flush=True)
dt=df[df['date'].isin(btd)].copy()
probs={}
n_batch=5000
for i in range(0,len(dt),n_batch):
    pct=100*i//len(dt) if len(dt) else 0
    if pct%25==0: print(f'  {pct}%...',flush=True)
    batch=dt.iloc[i:i+n_batch].dropna(subset=FEATS)
    if len(batch)==0: continue
    X=np.nan_to_num(batch[FEATS].values.astype(np.float32),nan=0)
    raw=model.predict(xgb.DMatrix(X))
    if raw.ndim>1: raw=raw[:,4]
    calib=cal.predict_proba(raw.reshape(-1,1))[:,1]
    for j,idx in enumerate(batch.index): probs[idx]=float(calib[j])
print(f'  完成: {len(probs):,}行',flush=True)

# 建立日→候选股索引（sym+概率+价格）
print('  日候选索引...',flush=True)
day_cands={}
for d in btd:
    dd=str(d)[:10]
    day=df[df['date']==d].copy()
    if len(day)<30: continue
    day['p']=day.index.map(lambda i:probs.get(i,0))
    day=day.dropna(subset=['p'])
    day_cands[dd]=[(r['sym'],r['p'],float(r['price'])) for _,r in day.iterrows() if r['p']>0]
    day_cands[dd].sort(key=lambda x:-x[1])
print(f'  {len(day_cands)}天有候选股',flush=True)

# 3. 回测
print('\n[3] 参数回测...',flush=True)
PARAM_TOP=[5,10,15]; PARAM_HOLD=[5,10]; PARAM_STOP=[5,10,15]; PARAM_REB=[5,10]
results=[]

for top_n,hold,stop,rebal in itertools.product(PARAM_TOP,PARAM_HOLD,PARAM_STOP,PARAM_REB):
    if hold<rebal: continue
    cap=10000.0; cash=cap; portfolio={}; trades=0; wins=0; curve=[cap]
    sl=stop/100.0; day_list=sorted(day_cands.keys())
    
    for day_idx,d in enumerate(day_list):
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
        
        # 调仓
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
    yrs=len(day_list)/252
    an=((final/10000)**(1/max(yrs,0.01))-1)*100
    peak=np.maximum.accumulate(ec)
    mdd=(ec-peak).min()/peak.max()*100 if peak.max()>0 else 0
    dr=np.diff(ec)/(ec[:-1]+1e-10)
    sh=(dr.mean()/max(dr.std(),1e-6))*np.sqrt(252) if len(dr)>20 else 0
    wr=wins/max(trades,1)
    
    tag=f'T{top_n}_H{hold}_S{stop}_R{rebal}'
    results.append({'tag':tag,'tr':round(tr,1),'an':round(an,1),
        'sh':round(sh,2),'mdd':round(mdd,1),'wr':round(wr,3),'trades':trades})

# 4. 输出
print('\n[4] 结果',flush=True)
rdf=pd.DataFrame(results).sort_values('sh',ascending=False)
print(f'{"参数":20s} {"收益":>7s} {"年化":>7s} {"夏普":>6s} {"回撤":>7s} {"胜率":>6s} {"交易":>6s}')
print('-'*60)
for _,r in rdf.iterrows():
    print(f'{r["tag"]:20s} {r["tr"]:>6.1f}% {r["an"]:>6.1f}% {r["sh"]:>6.2f} {r["mdd"]:>6.1f}% {r["wr"]:>5.1%} {r["trades"]:>6}')

print('\n=== 夏普Top5 ===')
for _,r in rdf.head(5).iterrows():
    print(f'  {r["tag"]:20s} 年化{r["an"]:>5.1f}% 夏普{r["sh"]:>5.2f} 回撤{r["mdd"]:>5.1f}%')
print('\n=== 年化Top5 ===')
for _,r in rdf.sort_values('an',ascending=False).head(5).iterrows():
    print(f'  {r["tag"]:20s} 年化{r["an"]:>5.1f}% 夏普{r["sh"]:>5.2f} 回撤{r["mdd"]:>5.1f}%')

# 5. 保存
json.dump({
    'timestamp':'2026-06-11 10:30','model':VER,'capital':10000,
    'range':f'{btd[0]}~{btd[-1]}','days':len(btd),
    'all':rdf.to_dict('records'),
    'sharpe_top':rdf.head(5).to_dict('records'),
},open(f'{MD}/us_v7_4_backtest.json','w'),indent=2)
print(f'\n[5] 保存: us_v7_4_backtest.json')
print(f'耗时: {time.time()-T0:.0f}s')
print('='*70)
