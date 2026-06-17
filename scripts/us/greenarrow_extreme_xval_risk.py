# -*- coding: utf-8 -*-
"""
绿箭极致 + SPY风控（交叉验证）
风控规则:
  level0: 正常操作 (SPY>MA200)
  level1: 减半仓 (SPY<MA200 或 组合DD<-10%)
  level2: 清仓 (SPY<MA200 且 组合DD<-15%)

参数: T7_H10_S20_R5
"""
import sys, os, json, pickle, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, xgboost as xgb

print('绿箭极致 + SPY风控'); print('='*60); t0=time.time()
BASE='/home/hermes/.hermes/openclaw-archive'; ML=f'{BASE}/ml'; MD=f'{BASE}/data/models'

# SPY市场状态
spy_state=json.load(open(f'{BASE}/data/spy_market_state.json'))
print(f'SPY状态: {len(spy_state)}天')

# V7.5模型
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
print(f'训练: {len(train_dates)}天, 验证: {len(val_dates)}天')

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
df['p']=0.0
n_batch=20000; n_total=len(df)
for i in range(0,n_total,n_batch):
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

def get_risk_level(d_str, port_dd):
    """0=正常, 1=减半, 2=清仓"""
    s=spy_state.get(d_str)
    if s is None: return 0
    if not s['ma200_live'] and port_dd<-0.15: return 2  # 双重打击
    if not s['ma200_live'] or port_dd<-0.10: return 1   # 任一危险
    return 0

def run_backtest_risk(dates, cands, T=7, H=10, S=20, R=5):
    cap=10000.0; cash=cap; port={}; trds=0; curve=[cap]
    sl=S/100.0; port_dd=0.0
    for di,d in enumerate(dates):
        # 计算组合当前价值
        port_val=sum(port[s]['qty']*close_idx.get(s,{}).get(d,0) for s in port)
        total_val=cash+port_val
        # 风控
        if total_val>0 and curve:
            port_dd=total_val/max(curve)-1
        else:
            port_dd=0.0
        rl=get_risk_level(d,port_dd)
        
        # 卖出: 止损/到期/风控清仓
        for sym in list(port.keys()):
            pos=port[sym]
            cp=close_idx.get(sym,{}).get(d)
            if cp is None: continue
            ret=(cp-pos['bp'])/pos['bp']
            sell=False
            if rl==2: sell=True
            elif ret<=-sl: sell=True
            elif (di-pos['di'])>=H: sell=True
            if sell:
                cash+=pos['qty']*cp; trds+=1
                del port[sym]
        
        # 风控减半仓
        if rl==1 and len(port)>0:
            half=list(port.keys())[:len(port)//2]
            for sym in half:
                pos=port[sym]; cp=close_idx.get(sym,{}).get(d)
                if cp is not None:
                    cash+=pos['qty']*cp
                del port[sym]
        
        # 买入: 风控level2不买, 其他正常
        if rl<2 and (di%R==0 or len(port)<T):
            picks=cands.get(d,[])
            if len(picks)>=3:
                # 风控level1只买一半仓位
                max_slots=T-len(port)
                if rl==1: max_slots=min(max_slots, max(1,T//2-len(port)))
                for sym,p,price in picks[:max_slots]:
                    if sym in port: continue
                    budget=cash/max(1,len(port)+1+rl)
                    qty=int(budget/price)
                    if qty<=0: continue
                    cash-=qty*price
                    port[sym]={'qty':qty,'bp':price,'di':di}
        
        curve.append(cash+sum(port[s]['qty']*close_idx.get(s,{}).get(d,0) for s in port))
    
    eq=np.array(curve)
    ann=(eq[-1]/cap)**(252/len(eq))-1
    rets=(eq[1:]-eq[:-1])/eq[:-1]; v=rets.std()*np.sqrt(252)
    sh=ann/max(v,1e-8); peak=np.maximum.accumulate(eq/cap); dd=eq/cap/peak-1
    return {'ann':round(ann,4),'sh':round(sh,4),'mdd':round(float(dd.min()),4),
            'final':round(float(eq[-1]),2),'trades':trds,'total_days':len(dates)}

# 构建候选
print('构建候选...')
train_cands=build_cands(train_dates)
val_cands=build_cands(val_dates)

# ===== 跑全部版本 =====
versions=[
    ('T7_H10_S20_R5 (无风控)', lambda: run_backtest_risk(val_dates,val_cands,T=7,H=10,S=20,R=5)),
    ('T7_H10_S20_R5 +风控L1', lambda: run_backtest_risk(val_dates,val_cands,T=7,H=10,S=20,R=5)),  # 风控用相同函数rl不同
]

# 更干净的跑法
def run_all(label, dates, cands, T=7, H=10, S=20, R=5, risk_on=False):
    cap=10000.0; cash=cap; port={}; trds=0; curve=[cap]
    sl=S/100.0; port_dd=0.0
    risk_days=[0,0,0]  # l0,l1,l2计数
    for di,d in enumerate(dates):
        port_val=sum(port[s]['qty']*close_idx.get(s,{}).get(d,0) for s in port)
        total_val=cash+port_val
        port_dd=total_val/max(curve,default=1)-1 if total_val>0 else 0.0
        rl=get_risk_level(d,port_dd) if risk_on else 0
        risk_days[rl]+=1
        
        # 卖出
        for sym in list(port.keys()):
            pos=port[sym]; cp=close_idx.get(sym,{}).get(d)
            if cp is None: continue
            ret=(cp-pos['bp'])/pos['bp']
            if rl==2 or ret<=-sl or (di-pos['di'])>=H:
                cash+=pos['qty']*cp; trds+=1
                del port[sym]
        
        # 风控减半
        if rl==1 and len(port)>0:
            half=list(port.keys())[:max(1,len(port)//2)]
            for sym in half:
                pos=port[sym]; cp=close_idx.get(sym,{}).get(d)
                if cp is not None:
                    cash+=pos['qty']*cp
                del port[sym]
        
        # 买入
        if rl<2 and (di%R==0 or len(port)<T):
            picks=cands.get(d,[])
            if len(picks)>=3:
                max_slots=T-len(port)
                if rl==1: max_slots=min(max_slots, max(1,T//2-len(port)))
                for sym,p,price in picks[:max_slots]:
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
    sh=ann/max(v,1e-8); peak=np.maximum.accumulate(eq/cap); dd=eq/cap/peak-1
    return {'label':label,'ann':round(ann,4),'sh':round(sh,4),'mdd':round(float(dd.min()),4),
            'final':round(float(eq[-1]),2),'trades':trds,'total_days':len(dates),
            'risk_l0':risk_days[0],'risk_l1':risk_days[1],'risk_l2':risk_days[2],
            'exposure':round(risk_days[0]/max(risk_days[0]+risk_days[1]+risk_days[2],1),4)}

print('\n回测中...')
results=[]
for label,params,risk in [
    ('T7_H10_S20_R5 无风控',(7,10,20,5),False),
    ('T7_H10_S20_R5 +SPY风控',(7,10,20,5),True),
    ('T5_H10_S15_R10 无风控',(5,10,15,10),False),
    ('T5_H10_S15_R10 +SPY风控',(5,10,15,10),True),
]:
    T,H,S,R=params
    for phase, dates, prefix in [('训练',train_dates,'train'),('验证',val_dates,'val')]:
        res=run_all(label, dates, train_cands if prefix=='train' else val_cands, T,H,S,R, risk)
        res['phase']=phase; res['prefix']=prefix
        results.append(res)
        print(f'{label} | {phase:>4} | 年化={res["ann"]*100:>6.1f}% 夏普={res["sh"]:>5.2f} 回撤={res["mdd"]*100:>6.1f}% 终值=${res["final"]:>8,.0f} {"风控" if risk else ""}',flush=True)

# 输出
print('\n'+'='*70)
print('交叉验证 + SPY风控')
print('='*70)
print(f'{"策略":<28} {"期":<4} {"年化":>8} {"夏普":>6} {"回撤":>7} {"暴露":>6} {"终值":>10}')
print('-'*70)
for r in results:
    exc=0 if not r.get('risk_l0') else r.get('exposure',1)
    print(f'{r["label"]:<28} {r["phase"][:2]:<4} {r["ann"]*100:>7.1f}% {r["sh"]:>6.2f} {r["mdd"]*100:>7.1f}% {exc*100:>5.0f}% ${r["final"]:>8,.0f}')

# 验证期汇总
print('\n验证期对比:')
val_results=[r for r in results if r['phase']=='验证']
for r in sorted(val_results,key=lambda x:x['mdd']):
    print(f'{r["label"]:<28} 年化={r["ann"]*100:>7.1f}% 夏普={r["sh"]:>5.2f} 回撤={r["mdd"]*100:>7.1f}% 暴露={r.get("exposure",1)*100:.0f}%')

# 保存
out={'results':results,'spy_state_file':'/home/hermes/.hermes/openclaw-project/data/spy_market_state.json',
     'time':time.strftime('%Y-%m-%d %H:%M')}
json.dump(out,open(f'{MD}/greenarrow_extreme_xval_risk.json','w'))
print(f'\n完成({time.time()-t0:.0f}s)')
print(f'结果: {MD}/greenarrow_extreme_xval_risk.json')
