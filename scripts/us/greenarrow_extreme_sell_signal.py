# -*- coding: utf-8 -*-
"""
绿箭极致 — 主动卖点（基于评分追踪）

核心逻辑：
- 买入: 每天评分前T只（同原版）
- 卖出: 持仓票的最新评分低于阈值 或 排名掉出前N

参数: T7_H10_S20_R5
卖点规则组合:
  a) 分数掉 < 0.25 → 卖
  b) 分数掉 < 买入时分数 * 0.5 → 卖（分数砍半）
  c) 每天排名后25% → 卖（持仓中评分最低的1/4强制赎回）
  d) 都加上
"""
import os, json, pickle, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, xgboost as xgb

print('绿箭极致 — 主动卖出测试'); print('='*60); t0=time.time()
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
open_idx,close_idx=pickle.load(open(idx_path,'rb'))

# 评分（每日每只评分）
print('逐日评分...')
# 构建 {date: {sym: prob}} 索引
from collections import defaultdict
daily_probs=defaultdict(dict)
n_batch=20000
for i in range(0,len(df),n_batch):
    chunk=df.iloc[i:i+n_batch]
    X=np.nan_to_num(chunk[FEATS].values.astype(np.float32),nan=0)
    raw=model.predict(xgb.DMatrix(X,feature_names=FEATS))
    calib=cal.predict_proba(raw.reshape(-1,1))[:,1]
    for j,(_,r) in enumerate(chunk.iterrows()):
        daily_probs[r['date_str']][r['sym']]=float(calib[j])
print(f'完成: {min(daily_probs.keys())}~{max(daily_probs.keys())}')

def build_cands(dates):
    cands={}
    for di,d in enumerate(dates):
        nxt_d=dates[di+1] if di+1<len(dates) else None
        if nxt_d is None: continue
        probs=daily_probs.get(d,{})
        if len(probs)<30: continue
        picks=[(sym,p,float(open_idx.get(sym,{}).get(nxt_d,0)))
               for sym,p in probs.items() 
               if sym in pool and p>0 and open_idx.get(sym,{}).get(nxt_d,0)>0]
        picks.sort(key=lambda x:-x[1])
        cands[d]=picks
    return cands

def run_test(label, dates, cands, T=7, H=10, S=20, R=5,
             sell_threshold=None, sell_half=None, sell_bottom=None):
    """
    sell_threshold: 分数低于此值就卖（float）
    sell_half: 分数低于买入时一半就卖（bool）
    sell_bottom: 每天强制卖出持仓中排名后25%（int=比例, 0.25=后25%）
    """
    cap=10000.0; cash=cap; port={}; trds=0; curve=[cap]
    sl=S/100.0
    
    for di,d in enumerate(dates):
        # 当前持仓票在今天的评分
        today_probs=daily_probs.get(d,{})
        
        # === 卖出 ===
        for sym in list(port.keys()):
            pos=port[sym]
            cp=close_idx.get(sym,{}).get(d)
            if cp is None: continue
            
            ret=(cp-pos['bp'])/pos['bp']
            need_sell=False
            
            # 1. 硬止损
            if ret<=-sl: need_sell=True
            # 2. 持有到期
            elif (di-pos['di'])>=H: need_sell=True
            # 3. 评分掉到阈值以下
            elif sell_threshold is not None and sym in today_probs and today_probs[sym]<sell_threshold:
                need_sell=True
            # 4. 评分砍半
            elif sell_half and sym in today_probs and today_probs[sym]<pos['buy_prob']*0.5:
                need_sell=True
            # 5. bottom比例的会在下面统一处理
            
            if need_sell:
                cash+=pos['qty']*cp; trds+=1
                del port[sym]
        
        # 统一处理bottom卖出（持仓中评分最低的N个）
        if sell_bottom is not None and len(port)>2:
            port_scores=[(s,port[s],today_probs.get(s,0)) for s in port]
            port_scores.sort(key=lambda x:x[2])
            n_sell=max(1,int(len(port_scores)*sell_bottom))
            for sym,pos,sc in port_scores[:n_sell]:
                if sc<=0: continue  # 没评分的才不主动卖
                if sc<port[sym].get('buy_prob',0.5):  # 至少分数比买入时低才卖
                    cp=close_idx.get(sym,{}).get(d)
                    if cp is not None:
                        cash+=pos['qty']*cp; trds+=1
                        del port[sym]
        
        # === 买入 ===
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
                    port[sym]={'qty':qty,'bp':price,'di':di,'buy_prob':p}
        
        curve.append(cash+sum(port[s]['qty']*close_idx.get(s,{}).get(d,0) for s in port))
    
    eq=np.array(curve)
    ann=(eq[-1]/cap)**(252/len(eq))-1
    rets=(eq[1:]-eq[:-1])/eq[:-1]; v=rets.std()*np.sqrt(252)
    sh=ann/max(v,1e-8); peak=np.maximum.accumulate(eq/cap); dd=eq/cap/peak-1
    
    return {'label':label,'ann':round(ann,4),'sh':round(sh,4),'mdd':round(float(dd.min()),4),
            'final':round(float(eq[-1]),2),'trades':trds}

print('构建候选...')
val_cands=build_cands(val_dates)
train_cands=build_cands(train_dates)

# ===== 各种卖点组合 =====
print('\n回测中...')
tests=[
    ('无卖点(纯止损+到期)',{'sell_threshold':None,'sell_half':False,'sell_bottom':None}),
    ('分数<0.25卖',{'sell_threshold':0.25,'sell_half':False,'sell_bottom':None}),
    ('分数<0.30卖',{'sell_threshold':0.30,'sell_half':False,'sell_bottom':None}),
    ('分数<买入时50%卖',{'sell_threshold':None,'sell_half':True,'sell_bottom':None}),
    ('每天卖后25%',{'sell_threshold':None,'sell_half':False,'sell_bottom':0.25}),
    ('分数<0.25+卖后25%',{'sell_threshold':0.25,'sell_half':False,'sell_bottom':0.25}),
    ('分数<0.30+卖后25%',{'sell_threshold':0.30,'sell_half':False,'sell_bottom':0.25}),
    ('分数<0.25+砍半+后25%',{'sell_threshold':0.25,'sell_half':True,'sell_bottom':0.25}),
    ('分数<0.20卖',{'sell_threshold':0.20,'sell_half':False,'sell_bottom':None}),
    ('分数<0.20+卖后25%',{'sell_threshold':0.20,'sell_half':False,'sell_bottom':0.25}),
]

results=[]
for label,kwargs in tests:
    res=run_test(label, val_dates, val_cands, T=7, H=10, S=20, R=5, **kwargs)
    res['phase']='验证'
    results.append(res)
    print(f'  {label:<30} 年化={res["ann"]*100:>7.1f}% 夏普={res["sh"]:>5.2f} 回撤={res["mdd"]*100:>7.1f}% 交易={res["trades"]:>4}',flush=True)

# 训练期跑最好和最差
print('\n训练期(最优组合)...')
for label,kwargs in [('无卖点',{'sell_threshold':None,'sell_half':False,'sell_bottom':None}),
                     ('分数<0.25+卖后25%',{'sell_threshold':0.25,'sell_half':False,'sell_bottom':0.25})]:
    res=run_test(label, train_dates, train_cands, T=7, H=10, S=20, R=5, **kwargs)
    res['phase']='训练'
    results.append(res)
    print(f'  {label:<30} 年化={res["ann"]*100:>7.1f}% 夏普={res["sh"]:>5.2f} 回撤={res["mdd"]*100:>7.1f}%')

# 输出
print('\n'+'='*80)
print('主动卖出策略 — 验证期(2025-2026)')
print('='*80)
print(f'{"策略":<32} {"年化":>8} {"夏普":>6} {"回撤":>7} {"交易":>5} {"终值":>10}')
print('-'*80)
val_only=[r for r in results if r['phase']=='验证']
for r in sorted(val_only,key=lambda x:-x['sh']):
    print(f'{r["label"]:<32} {r["ann"]*100:>7.1f}% {r["sh"]:>6.2f} {r["mdd"]*100:>7.1f}% {r["trades"]:>4} ${r["final"]:>8,.0f}')

json.dump({'results':results,'time':time.strftime('%Y-%m-%d %H:%M')},
          open(f'{MD}/greenarrow_extreme_sell_signals.json','w'))
print(f'\n完成({time.time()-t0:.0f}s)')
print(f'结果: {MD}/greenarrow_extreme_sell_signals.json')
