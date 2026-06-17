# -*- coding: utf-8 -*-
"""
绿箭极致 — 交叉验证 v2
先一次性算所有probs，再分段回测
"""
import sys, os, json, pickle, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, xgboost as xgb

print('绿箭极致 — 交叉验证 v2'); print('='*60); t0=time.time()
BASE='/home/hermes/.hermes/openclaw-archive'; ML=f'{BASE}/ml'; MD=f'{BASE}/data/models'; VER='us_v7_5'

model=xgb.Booster(); model.load_model(f'{MD}/{VER}.json')
cal=pickle.load(open(f'{MD}/{VER}_calibrator.pkl','rb'))
report=json.load(open(f'{MD}/{VER}_report.json'))
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
print(f'训练: {len(train_dates)}天 [{train_dates[0]}..{train_dates[-1]}]')
print(f'验证: {len(val_dates)}天 [{val_dates[0]}..{val_dates[-1]}]')

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

# 一次性算所有probs
print('评分...')
probs={}
n_batch=20000; n_total=len(df)
for i in range(0,n_total,n_batch):
    chunk=df.iloc[i:i+n_batch]
    X=np.nan_to_num(chunk[FEATS].values.astype(np.float32),nan=0)
    raw=model.predict(xgb.DMatrix(X,feature_names=FEATS))
    calib=cal.predict_proba(raw.reshape(-1,1))[:,1]
    for j,(_,r) in enumerate(chunk.iterrows()):
        probs[r.name]=float(calib[j])
df['p']=df.index.map(lambda i:probs.get(i,0))
print(f'评分完成: {sum(1 for v in probs.values() if v>0):,}条>0')

def build_cands(dates):
    """构建某时间段内的日候选"""
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

def run_backtest(dates, cands, T=7, H=10, S=20, R=5):
    cap=10000.0; cash=cap; port={}; trds=0; wins=0; curve=[cap]
    sl=S/100.0
    for di,d in enumerate(dates):
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
        curve.append(cash+sum(port[s]['qty']*close_idx.get(s,{}).get(d,0) for s in port))
    eq=np.array(curve)
    ann=(eq[-1]/cap)**(252/len(eq))-1
    rets=(eq[1:]-eq[:-1])/eq[:-1]; v=rets.std()*np.sqrt(252)
    sh=ann/max(v,1e-8); peak=np.maximum.accumulate(eq/cap); dd=eq/cap/peak-1
    return {'ann':round(ann,4),'sh':round(sh,4),'mdd':round(float(dd.min()),4),
            'final':round(float(eq[-1]),2),'trades':trds,'win_rate':round(trds and wins/trds,4),
            'total_days':len(dates),'avg_pos':sum(1 for v in curve if v>0)/len(curve)}

# 构建候选
print('构建训练候选...')
train_cands=build_cands(train_dates)
print(f'  候选天数: {len(train_cands)}, 候选数: {sum(len(v) for v in train_cands.values()):,}')
print('构建验证候选...')
val_cands=build_cands(val_dates)
print(f'  候选天数: {len(val_cands)}, 候选数: {sum(len(v) for v in val_cands.values()):,}')

# === T7_H10_S20_R5 ===
print('\n训练回测(T7_H10_S20_R5)...')
tr=run_backtest(train_dates,train_cands)
print('验证回测(T7_H10_S20_R5)...')
vr=run_backtest(val_dates,val_cands)

# === T5_H10_S15_R10 ===
print('训练回测(T5_H10_S15_R10)...')
tro=run_backtest(train_dates,train_cands,T=5,H=10,S=15,R=10)
print('验证回测(T5_H10_S15_R10)...')
vro=run_backtest(val_dates,val_cands,T=5,H=10,S=15,R=10)

# === 输出 ===
print('\n'+'='*60)
print('交叉验证结果')
print('='*60)
print(f'{"策略":<20} {"期段":<6} {"年化":>8} {"夏普":>6} {"回撤":>8} {"交易":>5} {"终值":>10}')
print('-'*70)

# 重排输出方式
print(f'{"T7_H10_S20_R5":<20} {"训练":<6} {tr["ann"]*100:>7.1f}% {tr["sh"]:>6.2f} {tr["mdd"]*100:>7.1f}% {tr["trades"]:>5} ${tr["final"]:>8,.0f}')
print(f'{"T7_H10_S20_R5":<20} {"验证":<6} {vr["ann"]*100:>7.1f}% {vr["sh"]:>6.2f} {vr["mdd"]*100:>7.1f}% {vr["trades"]:>5} ${vr["final"]:>8,.0f}')
print(f'{"差值":<20} {" ":<6} {tr["ann"]*100-vr["ann"]*100:>7.1f}% {tr["sh"]-vr["sh"]:>6.2f} {tr["mdd"]*100-vr["mdd"]*100:>7.1f}%')
print()
print(f'{"T5_H10_S15_R10":<20} {"训练":<6} {tro["ann"]*100:>7.1f}% {tro["sh"]:>6.2f} {tro["mdd"]*100:>7.1f}% {tro["trades"]:>5} ${tro["final"]:>8,.0f}')
print(f'{"T5_H10_S15_R10":<20} {"验证":<6} {vro["ann"]*100:>7.1f}% {vro["sh"]:>6.2f} {vro["mdd"]*100:>7.1f}% {vro["trades"]:>5} ${vro["final"]:>8,.0f}')
print(f'{"差值":<20} {" ":<6} {tro["ann"]*100-vro["ann"]*100:>7.1f}% {tro["sh"]-vro["sh"]:>6.2f} {tro["mdd"]*100-vro["mdd"]*100:>7.1f}%')

# 过拟合判定
diff_sh=tr['sh']-vr['sh']
print('\n'+'='*60)
print(f'过拟合判定 (T7_H10_S20_R5): 训练夏普{tr["sh"]:.2f} -> 验证夏普{vr["sh"]:.2f} = 差{diff_sh:.2f}')
if diff_sh<0.5: print('  ✅ 低过拟合 — 可信，直接用')
elif diff_sh<1.5: print('  ⚠️ 中等过拟合 — 可用但留个心眼')
else: print('  ❌ 严重过拟合 — 训练期好验证期崩了')

diff_orig=tro['sh']-vro['sh']
print(f'原版判定 (T5_H10_S15_R10): 训练夏普{tro["sh"]:.2f} -> 验证夏普{vro["sh"]:.2f} = 差{diff_orig:.2f}')

# 保存
out={'T7_H10_S20_R5':{'train':tr,'val':vr},
     'T5_H10_S15_R10':{'train':tro,'val':vro},
     'param':'T7_H10_S20_R5','time':time.strftime('%Y-%m-%d %H:%M')}
json.dump(out,open(f'{MD}/greenarrow_extreme_xval.json','w'))
print(f'\n完成({time.time()-t0:.0f}s)')
