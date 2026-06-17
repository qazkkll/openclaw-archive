# 整理V7.5回测参数 + 修正前视偏差重跑
import sys, os, json, pickle, time, itertools, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, xgboost as xgb

BASE='/home/hermes/.hermes/openclaw-archive'; ML=f'{BASE}/ml'; MD=f'{BASE}/data/models'; VER='us_v7_5'
print('='*70,flush=True); print('V7.5 回测 v4 — 次日开盘买卖',flush=True); print('='*70,flush=True)
T0=time.time()

model=xgb.Booster(); model.load_model(f'{MD}/{VER}.json')
cal=pickle.load(open(f'{MD}/{VER}_calibrator.pkl','rb'))
report=json.load(open(f'{MD}/{VER}_report.json'))
FEATS=report['features']

# 特征
df=pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str']=df['date'].astype(str).str[:10]
df['target']=(df['fwd_5d_ret']>0.05).astype(int)
for f in FEATS:
    if f in df.columns: df[f]=pd.to_numeric(df[f],errors='coerce').fillna(0).clip(-1e6,1e6)
df=df.replace([np.inf,-np.inf],np.nan)
del df['date']
print(f'  特征: {len(df):,}行, {df.sym.nunique()}只',flush=True)

BTD=sorted(df['date_str'].unique())
BTD=[d for d in BTD if d>='2022-01-01']
print(f'  回测: {len(BTD)}天 ({BTD[0]}~{BTD[-1]})',flush=True)

# 价格索引（含Open/Close，用于次日开盘模拟）
idx_path=f'{ML}/us_v75_close_idx_v4.pkl'
if os.path.exists(idx_path):
    open_idx=pickle.load(open(idx_path,'rb'))
    close_idx=open_idx  # 同文件
    print(f'  价格缓存: {len(open_idx)}只',flush=True)
else:
    main=pd.read_parquet(f'{ML}/us_hist_yf_10y.parquet', columns=['ticker','date','open','close'])
    main.rename(columns={'ticker':'sym'},inplace=True)
    mega=pd.read_parquet(f'{ML}/us_hist_megacap_10y.parquet', columns=['sym','date','open','close'])
    all_v=pd.concat([main,mega],ignore_index=True).drop_duplicates(subset=['sym','date'])
    all_v['ds']=all_v['date'].astype(str).str[:10]
    # 构建 {sym: {date: {open, close}}}
    open_idx={}; close_idx={}
    for s,g in all_v.groupby('sym'):
        g=g.sort_values('ds')
        open_idx[s]=dict(zip(g['ds'].values,g['open'].values.astype(float)))
        close_idx[s]=dict(zip(g['ds'].values,g['close'].values.astype(float)))
    pickle.dump((open_idx,close_idx),open(idx_path,'wb'))
    print(f'  open/close索引: {len(open_idx)}只, {time.time()-T0:.0f}s',flush=True)

# 概率
print(f'\n概率计算...',flush=True)
probs={}
n_batch=10000
n_total=len(df)
for i in range(0,n_total,n_batch):
    pct=100*i//n_total if n_total else 0
    if pct%20==0: print(f'  {pct}%...',flush=True)
    chunk=df.iloc[i:i+n_batch]
    chunk=chunk[chunk['date_str'].isin(BTD)]
    if len(chunk)==0: continue
    X=np.nan_to_num(chunk[FEATS].values.astype(np.float32),nan=0)
    raw=model.predict(xgb.DMatrix(X,feature_names=FEATS))
    calib=cal.predict_proba(raw.reshape(-1,1))[:,1]
    for j,(_,r) in enumerate(chunk.iterrows()):
        probs[r.name]=float(calib[j])
print(f'  完成: {len(probs):,}行',flush=True)

# 日候选（用D日收盘概率，待次日开盘买入）
print('  候选索引...',flush=True)
day_cands={}
for d_str in BTD:
    day=df[df['date_str']==d_str]
    if len(day)<30: continue
    day=day.copy()
    day['p']=day.index.map(lambda i:probs.get(i,0))
    day=day[day['p']>0]
    cands=[]
    for _, r in day.iterrows():
        # 候选买入价是次日开盘价，但需要下个交易日
        # 找下个交易日的开盘价
        nxt_idx=BTD.index(d_str)+1 if d_str in BTD else -1
        if nxt_idx>=len(BTD): continue
        nxt_d=BTD[nxt_idx]
        nxt_price=open_idx.get(r['sym'],{}).get(nxt_d)
        if nxt_price is None or np.isnan(nxt_price): continue
        cands.append((r['sym'],r['p'],float(nxt_price),d_str))
    cands.sort(key=lambda x:-x[1])
    day_cands[d_str]=cands

# 统计
total=sum(len(v) for v in day_cands.values())
print(f'  候选: {total:,}条, {sum(1 for v in day_cands.values() if len(v)>=30)}天>=30只',flush=True)

# 回测 - 用次日开盘价买卖
print(f'\n参数回测...',flush=True)
PARAMS=[
    ('T5_H10_S15_R10',5,10,15,10),
    ('T5_H10_S10_R10',5,10,10,10),
    ('T5_H10_S5_R10',5,10,5,10),
    ('T10_H10_S15_R10',10,10,15,10),
    ('T10_H10_S15_R5',10,10,15,5),
    ('T15_H10_S15_R5',15,10,15,5),
    ('T5_H5_S15_R5',5,5,15,5),
    ('T5_H5_S15_R10',5,5,15,10),
]

results=[]
for tag,top_n,hold,stop,rebal in PARAMS:
    cap=10000.0; cash=cap; portfolio={}; trades=0; wins=0; curve=[cap]
    sl=stop/100.0
    
    for day_idx,d in enumerate(BTD):
        # 持仓股用当日收盘价估值
        for sym in list(portfolio.keys()):
            pos=portfolio[sym]
            cp=close_idx.get(sym,{}).get(d)
            if cp is None: continue
            days_h=day_idx-pos['di']
            # 当日收盘价 vs 买入价（买入价是当时的开盘价）
            ret=(cp-pos['bp'])/pos['bp']
            if ret<=-sl or days_h>=hold:
                # 卖出用当日收盘价
                cash+=pos['qty']*cp
                trades+=1
                if cp>=pos['bp']: wins+=1
                del portfolio[sym]
        
        # 调仓（买入用次日开盘价——已经在候选索引里算好了）
        if day_idx%rebal==0:
            cands=[c for c in day_cands.get(d,[]) if c[0] not in portfolio]
            buys=cands[:top_n]
            for sym,prob,price,_ in buys:
                qty=cash/max(top_n,1)/max(price,0.01)
                if qty<1: continue
                # 买入价=price=次日开盘价
                portfolio[sym]={'bp':price,'qty':qty,'di':day_idx}
                cash-=qty*price
        
        pv=sum(p['qty']*close_idx.get(s,{}).get(d,p['bp']) for s,p in portfolio.items())
        curve.append(cash+pv)
    
    final=cash+sum(p['qty']*p['bp'] for p in portfolio.values())
    ec=np.array(curve)
    tr=(final/10000-1)*100
    yrs=len(BTD)/252
    an=((final/10000)**(1/max(yrs,0.01))-1)*100
    peak=np.maximum.accumulate(ec)
    mdd=(ec-peak).min()/peak.max()*100 if peak.max()>0 else 0
    dr=np.diff(ec)/(ec[:-1]+1e-10)
    sh=(dr.mean()/max(dr.std(),1e-6))*np.sqrt(252) if len(dr)>20 else 0
    wr=wins/max(trades,1)
    results.append({'tag':tag,'tr':round(tr,1),'an':round(an,1),
        'sh':round(sh,2),'mdd':round(mdd,1),'wr':round(wr,3),'trades':trades})
    print(f'  {tag}: 年化{an:.1f}% 夏普{sh:.2f} 回撤{mdd:.1f}%',flush=True)

# 输出
print('\n' + '='*70)
print(f'{"参数":20s} {"收益":>7s} {"年化":>7s} {"夏普":>6s} {"回撤":>7s} {"胜率":>6s} {"交易":>6s}')
print('-'*60)
rdf=pd.DataFrame(results).sort_values('sh',ascending=False)
for _,r in rdf.iterrows():
    print(f'{r["tag"]:20s} {r["tr"]:>6.1f}% {r["an"]:>6.1f}% {r["sh"]:>6.2f} {r["mdd"]:>6.1f}% {r["wr"]:>5.1%} {r["trades"]:>6}')

json.dump({'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),'model':VER,
    'range':f'{BTD[0]}~{BTD[-1]}','days':len(BTD),'method':'次日开盘买卖',
    'all':rdf.to_dict('records')},open(f'{MD}/us_v7_5_backtest_v4.json','w'),indent=2)
print(f'\n保存: us_v7_5_backtest_v4.json')
print(f'耗时: {(time.time()-T0)/60:.1f}分钟')
print('='*70)
