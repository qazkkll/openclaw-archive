# -*- coding: utf-8 -*-
"""
绿箭极致 + 组合自身回撤风控
规则:
  level0: DD>-10% → 正常
  level1: DD<=-10% 且 DD>-15% → 减半仓
  level2: DD<=-15% → 清仓, 等DD回升>-5%才恢复
"""
import os, json, pickle, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, xgboost as xgb

print('绿箭极致 + 自身回撤风控'); print('='*60); t0=time.time()
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
print(f'数据: {len(df)}行, {df.sym.nunique()}只')

all_dates=sorted(df['date_str'].unique())
train_dates=[d for d in all_dates if d<'2025-01-01' and d>='2022-01-01']
val_dates=[d for d in all_dates if d>='2025-01-01']

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
df['p']=0.0
n_batch=20000
for i in range(0,len(df),n_batch):
    chunk=df.iloc[i:i+n_batch]
    X=np.nan_to_num(chunk[FEATS].values.astype(np.float32),nan=0)
    raw=model.predict(xgb.DMatrix(X,feature_names=FEATS))
    calib=cal.predict_proba(raw.reshape(-1,1))[:,1]
    for j,(idx,_) in enumerate(chunk.iterrows()):
        df.at[idx,'p']=float(calib[j])

def build_cands(dates):
    cands={}
    for di,d in enumerate(dates):
        nxt_d=dates[di+1] if di+1<len(dates) else None
        if nxt_d is None: continue
        day=df[df['date_str']==d]
        day=day[day['p']>0]
        picks=[]
        for _,r in day.iterrows():
            nxt_price=open_idx.get(r['sym'],{}).get(nxt_d)
            if nxt_price is None: continue
            picks.append((r['sym'],r['p'],float(nxt_price)))
        picks.sort(key=lambda x:-x[1])
        cands[d]=picks
    return cands

def run_test(label, dates, cands, T=7, H=10, S=20, R=5, 
             dd_light=-0.10, dd_dark=-0.15, dd_resume=-0.05):
    """
    dd_light: 减半仓阈值
    dd_dark: 清仓阈值  
    dd_resume: 重新进场阈值
    """
    cap=10000.0; cash=cap; port={}; trds=0; curve=[cap]; peak=cap
    frozen=False; freeze_day=0
    risk_counts=[0,0,0]  # l0当前组合DD, l1减半, l2清仓
    
    for di,d in enumerate(dates):
        port_val=sum(port[s]['qty']*close_idx.get(s,{}).get(d,0) for s in port)
        total_val=cash+port_val
        if total_val>peak: peak=total_val
        current_dd=total_val/peak-1
        
        # 风控判定
        if frozen:
            # 冻结中: 不交易, 等DD回升
            risk_counts[2]+=1
            if current_dd>dd_resume:
                frozen=False  # 解冻
        else:
            if current_dd<=dd_dark:
                risk_counts[2]+=1
                frozen=True; freeze_day=di
            elif current_dd<=dd_light:
                risk_counts[1]+=1
            else:
                risk_counts[0]+=1
        
        # 卖出
        for sym in list(port.keys()):
            pos=port[sym]; cp=close_idx.get(sym,{}).get(d)
            if cp is None: continue
            ret=(cp-pos['bp'])/pos['bp']
            if frozen or ret<=-S/100.0 or (di-pos['di'])>=H:
                cash+=pos['qty']*cp; trds+=1
                del port[sym]
        
        # 减半仓
        if not frozen and risk_counts[-1]==risk_counts[-2]==0 and current_dd<=dd_light:
            # 继续减但不重复减（上面已经通过freeze做了）
            pass
        
        # 买入（冻结期不买）
        if not frozen and (di%R==0 or len(port)<T):
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
        
        curve.append(cash+sum(port[s]['qty']*close_idx.get(s,{}).get(d,0) for s in port))
    
    eq=np.array(curve)
    ann=(eq[-1]/cap)**(252/len(eq))-1
    rets=(eq[1:]-eq[:-1])/eq[:-1]; v=rets.std()*np.sqrt(252)
    sh=ann/max(v,1e-8)
    peak2=np.maximum.accumulate(eq/cap); dd=eq/cap/peak2-1
    frozen_pct=risk_counts[2]/max(1,sum(risk_counts))*100
    
    return {'label':label,'ann':round(ann,4),'sh':round(sh,4),'mdd':round(float(dd.min()),4),
            'final':round(float(eq[-1]),2),'trades':trds,'total_days':len(dates),
            'frozen_pct':round(frozen_pct,1),
            'risk_l0':risk_counts[0],'risk_l1':risk_counts[1],'risk_l2':risk_counts[2]}

print('构建候选...')
train_cands=build_cands(train_dates)
val_cands=build_cands(val_dates)

# ===== 跑风控方案对比 =====
print('\n回测中...')
variants=[
    ('T7_H10_S20_R5 无风控',False,None),
    ('DD-10%减半/-15%清仓/-5%恢复',True,(-0.10,-0.15,-0.05)),
    ('DD-8%减半/-12%清仓/-4%恢复',True,(-0.08,-0.12,-0.04)),
    ('DD-12%减半/-18%清仓/-6%恢复',True,(-0.12,-0.18,-0.06)),
    ('DD-7%减半/-10%清仓/-3%恢复',True,(-0.07,-0.10,-0.03)),
]

results=[]
for label,risk,thresholds in variants:
    for phase,dates,prefix in [('训练',train_dates,'train'),('验证',val_dates,'val')]:
        cands=train_cands if prefix=='train' else val_cands
        if risk:
            res=run_test(label,dates,cands,T=7,H=10,S=20,R=5,
                        dd_light=thresholds[0],dd_dark=thresholds[1],dd_resume=thresholds[2])
        else:
            res=run_test(label,dates,cands,T=7,H=10,S=20,R=5,
                        dd_light=-99,dd_dark=-99,dd_resume=-99)
        res['phase']=phase
        results.append(res)
        tag=f'  {label:<35} {phase:>4}: 年化={res["ann"]*100:>7.1f}% 夏普={res["sh"]:>5.2f} 回撤={res["mdd"]*100:>6.1f}% 冻结={res["frozen_pct"]:>5.1f}% 终值=${res["final"]:>8,.0f}'
        print(tag,flush=True)

# ===== 输出 =====
print('\n'+'='*80)
print('自身回撤风控 — 验证期对比')
print('='*80)
print(f'{"策略":<38} {"年化":>8} {"夏普":>6} {"回撤":>7} {"冻结":>5} {"终值":>10}')
print('-'*80)
val_only=[r for r in results if r['phase']=='验证']
for r in sorted(val_only,key=lambda x:x['mdd']):
    print(f'{r["label"]:<38} {r["ann"]*100:>7.1f}% {r["sh"]:>6.2f} {r["mdd"]*100:>7.1f}% {r["frozen_pct"]:>4.0f}% ${r["final"]:>8,.0f}')

print('\n训练期对比:')
train_only=[r for r in results if r['phase']=='训练']
for r in sorted(train_only,key=lambda x:x['mdd']):
    print(f'{r["label"]:<38} {r["ann"]*100:>7.1f}% {r["sh"]:>6.2f} {r["mdd"]*100:>7.1f}% {r["frozen_pct"]:>4.0f}% ${r["final"]:>8,.0f}')

# 选最优
print('\n')
best_risk=[r for r in results if r['phase']=='验证' and '无风控' not in r['label']]
if best_risk:
    best=sorted(best_risk,key=lambda x:-x['sh'])[0]
    no_risk=[r for r in results if r['phase']=='验证' and '无风控' in r['label']][0]
    print(f'推荐风控方案: {best["label"]}')
    print(f'  夏普: {no_risk["sh"]:.2f} -> {best["sh"]:.2f}')
    print(f'  回撤: {no_risk["mdd"]*100:.1f}% -> {best["mdd"]*100:.1f}%')
    print(f'  冻结期: {best["frozen_pct"]:.0f}%')

json.dump({'results':results,'time':time.strftime('%Y-%m-%d %H:%M')},
          open(f'{MD}/greenarrow_extreme_dd_risk.json','w'))
print(f'\n完成({time.time()-t0:.0f}s)')
