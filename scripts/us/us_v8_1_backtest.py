# -*- coding: utf-8 -*-
import os, json, pickle, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, xgboost as xgb

print('V8.1 backtest'); t0=time.time()
ML='/home/hermes/.hermes/openclaw-archive/scripts/system'; MD='/home/hermes/.hermes/openclaw-project/data/models'

model_v75=xgb.Booster(); model_v75.load_model(f'{MD}/us_v7_5.json')
cal_v75=pickle.load(open(f'{MD}/us_v7_5_calibrator.pkl','rb'))
model_v81=xgb.Booster(); model_v81.load_model(f'{MD}/us_v8_1.json')
cal_v81=pickle.load(open(f'{MD}/us_v8_1_calibrator.pkl','rb'))
report=json.load(open(f'{MD}/us_v7_5_report.json'))
FEATS=report['features']

df=pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
fl=json.load(open(f'{ML}/us_filtered_syms_v5.json'))
pool=set(fl['syms'])
df=df[df['sym'].isin(pool)].copy()
print(f'Rows: {len(df)}, Stocks: {df.sym.nunique()}')

# Date column - use 'dt' NOT 'd' (d is a feature!)
df['dt']=df['date'].astype(str).str[:10]
df['dt']=[str(v) if isinstance(v,(int,float)) else v for v in df['dt'].values]

# Feature cleaning - exclude 'dt' and 'sym' and 'date' and 'target'
feat_cols_clean=[f for f in FEATS if f in df.columns and f not in ('d','date','target')]
# Also add 'd' back if it's a feature column
feat_cols=[f for f in FEATS if f in df.columns]
print(f'Feat cols: {len(feat_cols)}')
for f in feat_cols_clean:
    if f not in df.columns: continue
    df[f]=pd.to_numeric(df[f],errors='coerce').fillna(0)
df[feat_cols_clean]=df[feat_cols_clean].replace([np.inf,-np.inf],0)

# Dates
all_dates=sorted(df['dt'].unique().tolist())
val_dates=[d for d in all_dates if d>='2025-01-01']
full_dates=[d for d in all_dates if d>='2016-10-18']
print(f'Val: {len(val_dates)}d [{val_dates[0] if val_dates else "N/A"}..{val_dates[-1] if val_dates else "N/A"}]')
print(f'Full: {len(full_dates)}d [{full_dates[0] if full_dates else "N/A"}..{full_dates[-1] if full_dates else "N/A"}]')

# Price idx
open_idx,close_idx=pickle.load(open(f'{ML}/us_v75_close_idx_v4.pkl','rb'))

def score_all(model, cal, use_proba=True):
    probs={}
    n_batch=20000
    for i in range(0,len(df),n_batch):
        chunk=df.iloc[i:i+n_batch]
        X=np.nan_to_num(chunk[feat_cols].values.astype(np.float32),nan=0)
        raw=model.predict(xgb.DMatrix(X,feature_names=feat_cols))
        if use_proba:
            calib=cal.predict_proba(raw.reshape(-1,1))[:,1]
        else:
            calib=cal.predict(raw.reshape(-1,1))
        for j,(_,r) in enumerate(chunk.iterrows()):
            probs[r.name]=float(calib[j])
    return probs

def build_cands(dates, probs):
    cands={}
    for di,d in enumerate(dates):
        nxt_d=dates[di+1] if di+1<len(dates) else None
        if nxt_d is None: continue
        picks=[]
        day=df[df['dt']==d]
        for idx,r in day.iterrows():
            p=probs.get(idx,0)
            if p<=0.05: continue
            nxt_price=open_idx.get(r['sym'],{}).get(nxt_d)
            if nxt_price is None or nxt_price<=0: continue
            picks.append((r['sym'],p,float(nxt_price)))
        picks.sort(key=lambda x:-x[1])
        cands[d]=picks
    return cands

# Score
print('\nScoring V7.5...')
probs_v75=score_all(model_v75,cal_v75,True)
print('Scoring V8.1...')
probs_v81=score_all(model_v81,cal_v81,False)

# Build cands
print('Building cands...')
cands_v75_val=build_cands(val_dates,probs_v75)
cands_v81_val=build_cands(val_dates,probs_v81)
cands_v75_full=build_cands(full_dates,probs_v75)
cands_v81_full=build_cands(full_dates,probs_v81)

# DEBUG check
print('\n=== CANDIDATE CHECK ===')
for name,cands in [('v75_val',cands_v75_val),('v81_val',cands_v81_val),
                   ('v75_full',cands_v75_full),('v81_full',cands_v81_full)]:
    nz=sum(1 for v in cands.values() if len(v)>0)
    tot=len(cands)
    avg=sum(len(v) for v in cands.values())/max(tot,1)
    print(f'{name}: {nz}/{tot} days w/ picks, avg {avg:.1f}/day')

def bt(label, dates, cands, T=7, H=10, S=20, R=5):
    cap=10000.0; cash=cap; port={}; trds=0; curve=[cap]
    sl=S/100.0
    for di,d in enumerate(dates):
        for sym in list(port.keys()):
            pos=port[sym]; cp=close_idx.get(sym,{}).get(d)
            if cp is None: continue
            ret=(cp-pos['bp'])/pos['bp']
            if ret<=-sl or (di-pos['di'])>=H:
                cash+=pos['qty']*cp; trds+=1
                del port[sym]
        if di%R==0 or len(port)<T:
            picks=cands.get(d,[])
            if len(picks)>=3:
                slots=T-len(port)
                for sym,p,price in picks[:slots]:
                    if sym in port: continue
                    budget=cash/max(1,T)
                    qty=int(budget/price)
                    if qty<=0: continue
                    cash-=qty*price
                    port[sym]={'qty':qty,'bp':price,'di':di}
        equity=cash+sum(port[s]['qty']*close_idx.get(s,{}).get(d,0) for s in port)
        curve.append(equity)
    
    eq=np.array(curve)
    ann=(eq[-1]/cap)**(252/max(len(eq),1))-1
    rets=(eq[1:]-eq[:-1])/eq[:-1]; v=rets.std()*np.sqrt(252)
    sh=ann/max(v,1e-8); peak=np.maximum.accumulate(eq/cap); dd=eq/cap/peak-1
    return {'label':label,'ann':round(ann,4),'sh':round(sh,4),'mdd':round(float(dd.min()),4),
            'final':round(float(eq[-1]),2),'trades':trds,'days':len(dates),
            'win_rate':round(float((rets>0).mean()),4)}

# Backtest
params=[(7,10,20,5),(5,10,15,10)]
all_res=[]
print('\n--- Validation (2025-2026) ---')
for mn,cands in [('V7.5',cands_v75_val),('V8.1',cands_v81_val)]:
    for T,H,S,R in params:
        lbl=f'{mn} T{T}_H{H}_S{S}_R{R}'
        res=bt(lbl,val_dates,cands,T,H,S,R)
        res['period']='validation'
        all_res.append(res)
        print(f'  {lbl}: ann={res["ann"]*100:>7.1f}% sharpe={res["sh"]:>5.2f} mdd={res["mdd"]*100:>6.1f}% fin=${res["final"]:>8,.0f}')

print('\n--- Full (2016-2026) ---')
for mn,cands in [('V7.5',cands_v75_full),('V8.1',cands_v81_full)]:
    for T,H,S,R in params:
        lbl=f'{mn} T{T}_H{H}_S{S}_R{R}'
        res=bt(lbl,full_dates,cands,T,H,S,R)
        res['period']='full'
        all_res.append(res)
        print(f'  {lbl}: ann={res["ann"]*100:>7.1f}% sharpe={res["sh"]:>5.2f} mdd={res["mdd"]*100:>6.1f}% fin=${res["final"]:>8,.0f}')

# Summary
print('\n'+'='*80)
print('V8.1 vs V7.5 Backtest')
print('='*80)
print(f'{"Version":<30} {"Period":<10} {"Ann%":>8} {"Sharpe":>7} {"MDD%":>7} {"Win%":>6} {"Trd":>4} {"Fin($K)":>8}')
print('-'*80)
for r in sorted(all_res,key=lambda x:(x['period'],-x['ann'])):
    print(f'{r["label"]:<30} {r["period"]:<10} {r["ann"]*100:>7.1f}% {r["sh"]:>7.2f} {r["mdd"]*100:>7.1f}% {r["win_rate"]*100:>5.1f}% {r["trades"]:>4} ${r["final"]/10000:>6,.1f}')

json.dump({'results':all_res,'time':time.strftime('%Y-%m-%d %H:%M')},
          open(f'{MD}/us_v8_1_backtest.json','w'))
print(f'\nDone ({time.time()-t0:.0f}s) -> {MD}/us_v8_1_backtest.json')
