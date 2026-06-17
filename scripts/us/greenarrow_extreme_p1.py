# -*- coding: utf-8 -*-
"""
绿箭极致版P1 — 参数扫描（基于V7.5回测v4管线）
只改参数组合，不改回测逻辑
"""
import sys, os, json, pickle, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, xgboost as xgb

print('绿箭极致P1 — 参数扫描'); print('='*60); t0=time.time()
BASE='/home/hermes/.hermes/openclaw-archive'; ML=f'{BASE}/ml'; MD=f'{BASE}/data/models'; VER='us_v7_5'

model=xgb.Booster(); model.load_model(f'{MD}/{VER}.json')
cal=pickle.load(open(f'{MD}/{VER}_calibrator.pkl','rb'))
report=json.load(open(f'{MD}/{VER}_report.json'))
FEATS=report['features']
print(f'特征: {len(FEATS)}')

df=pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str']=df['date'].astype(str).str[:10]
df['target']=(df['fwd_5d_ret']>0.05).astype(int)
for f in FEATS:
    if f in df.columns: df[f]=pd.to_numeric(df[f],errors='coerce').fillna(0).clip(-1e6,1e6)
df=df.replace([np.inf,-np.inf],np.nan)

# 过滤
fl=json.load(open(f'{ML}/us_filtered_syms_v5.json'))
pool=set(fl['syms'])
df=df[df['sym'].isin(pool)].copy()
print(f'过滤后: {len(df)}行, {df.sym.nunique()}只')

BTD=sorted(df['date_str'].unique())
BTD=[d for d in BTD if d>='2022-01-01']
print(f'{len(BTD)}天 ({BTD[0]}~{BTD[-1]})')

# 价格索引
idx_path=f'{ML}/us_v75_close_idx_v4.pkl'
if os.path.exists(idx_path):
    open_idx,close_idx=pickle.load(open(idx_path,'rb'))
else:
    main=pd.read_parquet(f'{ML}/us_hist_yf_10y.parquet',columns=['ticker','date','open','close'])
    main.rename(columns={'ticker':'sym'},inplace=True)
    all_v=main.drop_duplicates(subset=['sym','date'])
    all_v['ds']=all_v['date'].astype(str).str[:10]
    open_idx={}; close_idx={}
    for s,g in all_v.groupby('sym'):
        g=g.sort_values('ds')
        open_idx[s]=dict(zip(g['ds'].values,g['open'].values.astype(float)))
        close_idx[s]=dict(zip(g['ds'].values,g['close'].values.astype(float)))
    pickle.dump((open_idx,close_idx),open(idx_path,'wb'))

# 评分
print('评分...')
probs={}; n_batch=10000; n_total=len(df)
for i in range(0,n_total,n_batch):
    chunk=df.iloc[i:i+n_batch]
    chunk=chunk[chunk['date_str'].isin(BTD)]
    if len(chunk)==0: continue
    X=np.nan_to_num(chunk[FEATS].values.astype(np.float32),nan=0)
    raw=model.predict(xgb.DMatrix(X,feature_names=FEATS))
    calib=cal.predict_proba(raw.reshape(-1,1))[:,1]
    for j,(_,r) in enumerate(chunk.iterrows()):
        probs[r.name]=float(calib[j])

# 候选（D日收盘用概率，次日开盘买入）
print('候选索引...')
day_cands={}
for d_str in BTD:
    day=df[df['date_str']==d_str]
    if len(day)<30: continue
    day=day.copy()
    day['p']=day.index.map(lambda i:probs.get(i,0))
    day=day[day['p']>0]
    cands=[]
    nxt_idx=BTD.index(d_str)+1 if d_str in BTD else -1
    if nxt_idx>=len(BTD): continue
    nxt_d=BTD[nxt_idx]
    for _,r in day.iterrows():
        nxt_price=open_idx.get(r['sym'],{}).get(nxt_d)
        if nxt_price is None or np.isnan(nxt_price): continue
        cands.append((r['sym'],r['p'],float(nxt_price),d_str))
    cands.sort(key=lambda x:-x[1])
    day_cands[d_str]=cands

print(f'候选: {sum(len(v) for v in day_cands.values()):,}条, {sum(1 for v in day_cands.values() if len(v)>=10)}天>=10只')

# ========= 全覆盖参数扫描 =========
TV=[3,5,7,10]; HV=[5,7,10,14]; SV=[5,10,15,20]; RV=[5,10,15]  # S现在是百分比5%=5
print(f'\n参扫: {len(TV)}x{len(HV)}x{len(SV)}x{len(RV)}={len(TV)*len(HV)*len(SV)*len(RV)}')
results=[]
if os.path.exists('/home/hermes/.hermes/openclaw-project/data/models/greenarrow_extreme_p1_v2.json'):
    print('发现已有结果，加载继续...')
    existing=json.load(open('/home/hermes/.hermes/openclaw-project/data/models/greenarrow_extreme_p1_v2.json'))
    done=set(r['label'] for r in existing['results'])
    results=existing['results']
    print(f'已有: {len(results)}个')
else:
    done=set()

for T in TV:
    for H in HV:
        for S in SV:
            for R in RV:
                tag=f'T{T}_H{H}_S{S}_R{R}'
                if tag in done: continue
                
                cap=10000.0; cash=cap; port={}; trds=0; wins=0; curve=[cap]
                sl=S/100.0
                
                for di,d in enumerate(BTD):
                    # 卖出
                    for sym in list(port.keys()):
                        pos=port[sym]
                        cp=close_idx.get(sym,{}).get(d)
                        if cp is None: continue
                        ret=(cp-pos['bp'])/pos['bp']
                        if ret<=-sl or (di-pos['di'])>=H:
                            cash+=pos['qty']*cp; trds+=1
                            if cp>=pos['bp']: wins+=1
                            del port[sym]
                    
                    # 买入（每R天或持仓<top_n）
                    if di%R==0 or len(port)<T:
                        cands=day_cands.get(d,[])
                        if len(cands)>3:
                            slots=T-len(port)
                            for sym,p,price,_ in cands[:slots]:
                                if sym in port: continue
                                budget=cash/max(1,len(port)+1)
                                qty=int(budget/price)
                                if qty<=0: continue
                                cash-=qty*price
                                port[sym]={'qty':qty,'bp':price,'di':di}
                    
                    curve.append(cash+sum(port[s]['qty']*close_idx.get(s,{}).get(d,0) for s in port))
                
                eq=np.array(curve)
                if eq[-1]<0: continue
                ann=(eq[-1]/cap)**(252/len(eq))-1
                rets=(eq[1:]-eq[:-1])/eq[:-1]; v=rets.std()*np.sqrt(252)
                sh=ann/max(v,1e-8)
                cum=eq/cap; peak=np.maximum.accumulate(cum); dd=cum/peak-1
                res={'label':tag,'T':T,'H':H,'S':S,'R':R,
                     'ann':round(ann,4),'sh':round(sh,4),'mdd':round(float(dd.min()),4),
                     'win':round(float((rets>0).mean()),4),'final':round(float(eq[-1]),2),
                     'trades':trds}
                results.append(res)
                print(f'  {tag}: +{ann*100:.1f}%, sh={sh:.2f}, dd={dd.min()*100:.1f}%',flush=True)
                
                # 每10个保存
                if len(results)%10==0:
                    json.dump({'results':results,'time':time.strftime('%Y-%m-%d %H:%M')},
                              open('/home/hermes/.hermes/openclaw-project/data/models/greenarrow_extreme_p1_v2.json','w'))

results.sort(key=lambda r:-r['sh'])
json.dump({'results':results,'time':time.strftime('%Y-%m-%d %H:%M')},
          open('/home/hermes/.hermes/openclaw-project/data/models/greenarrow_extreme_p1_v2.json','w'))

print(f'\n{"="*60}')
print(f'Top 20 (共{len(results)}个)')
print('-'*60)
for r in results[:20]:
    print(f'{r["label"]:<25} {r["ann"]*100:>7.1f}% {r["sh"]:>7.2f} {r["mdd"]*100:>7.1f}% {r["win"]*100:>6.1f}% ${r["final"]:>8,.0f} {r["trades"]:>5}t')

ddok=[r for r in results if r['mdd']>-0.15]
print(f'\n低回撤(>-15%) {len(ddok)}个:')
for r in sorted(ddok,key=lambda r:-r['sh'])[:10]:
    print(f'{r["label"]:<25} {r["ann"]*100:>7.1f}% {r["sh"]:>7.2f} {r["mdd"]*100:>7.1f}%')

print(f'\n对照: V7.5原版 T5_H10_S15_R10: +193.5%,夏普2.26,回撤-23.3%')
if results: b=results[0]; print(f'\n新最佳: {b["label"]} +{b["ann"]*100:.1f}%,夏普{b["sh"]:.2f},回撤-{b["mdd"]*100:.1f}%')
if ddok: b2=ddok[0]; print(f'低回撤最佳(>-15%): {b2["label"]} +{b2["ann"]*100:.1f}%,夏普{b2["sh"]:.2f},回撤-{b2["mdd"]*100:.1f}%')
print(f'\n完成({time.time()-t0:.0f}s)')
