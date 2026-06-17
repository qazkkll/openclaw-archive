#!/usr/bin/env python3
"""
美股ML v12通过 合成夏普0.900
v13: 把v12的模型中"趋势加速"特征和v9的"时间衰减"做最佳组合
集成: v12趋势模型 + v9时间衰减模型 的预测概率做平均
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import json, time, math, pandas as pd, numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
import _paths

T0 = time.time()
print("═══ v13 Ensemble: 趋势模型 + 动量模型 ═══")

df = pd.read_parquet(_paths.US_ML_FEATS)

# ====== 两套特征 ======
# 模型A: v9风格 — 基础+vol20特征+时间衰减
base = ['ma5','ma10','ma20','ma60','rsi14','vol20','p52',
        'ret1','ret5','ret20','ret60',
        'macd','macd_signal','macd_hist','vol_ratio','ma_bias20',
        'price','volume']
df['vol20_ma20'] = df['vol20'].fillna(0) - (df['volume']/df['volume'].rolling(20).mean()).fillna(0)
df['vol20_ma60'] = df['vol20'].fillna(0) - (df['volume']/df['volume'].rolling(60).mean()).fillna(0)
featsA = base + ['vol20_ma20','vol20_ma60']

# 模型B: v12风格 — 基础+加速/动量特征
df['ret60_vol20'] = df['ret60'].fillna(0)*df['vol20'].fillna(0)
df['trend_accel'] = df['ret20'].fillna(0)-df['ret5'].fillna(0)
df['macd_accel'] = df['macd_hist'].fillna(0)-df['macd_hist'].shift(1).fillna(0)
df['vol_price_div'] = df['ret1'].fillna(0)*df['vol_ratio'].fillna(0)
df['vol_surge'] = (df['vol20'].fillna(0)-df['vol20'].shift(5).fillna(0))/df['vol20'].shift(5).fillna(df['vol20']).replace(0,np.nan).fillna(1)
df['p52_momentum'] = df['ret1'].fillna(0)*(df['p52'].fillna(50)-50)
featsB = base + ['ret60_vol20','trend_accel','macd_accel','vol_price_div','vol_surge','p52_momentum']

def bucket3(p):
    if p>2: return 2
    if p>-2: return 1
    return 0
df['label_bucket'] = df['label_pct'].apply(bucket3)

classes = np.array([0,1,2])
weights = compute_class_weight('balanced', classes=classes, y=df['label_bucket'])
wd={i:w for i,w in enumerate(weights)}

# Walk-Forward 3窗口
wfs=[('WF1',0,0.60,0.60,0.75,0.75,0.85),
     ('WF2',0.15,0.65,0.65,0.80,0.80,0.90),
     ('WF3',0.30,0.70,0.70,0.85,0.85,1.00)]

all_A, all_B, all_act = [], [], []

for name,ts,te,vs,ve,tst,tste in wfs:
    n=len(df)
    XtrA = df[featsA].values[int(ts*n):int(te*n)]
    XtrB = df[featsB].values[int(ts*n):int(te*n)]
    ytr = df['label_bucket'].values[int(ts*n):int(te*n)]
    XvaA = df[featsA].values[int(vs*n):int(ve*n)]
    XvaB = df[featsB].values[int(vs*n):int(ve*n)]
    yva = df['label_bucket'].values[int(vs*n):int(ve*n)]
    XteA = df[featsA].values[int(tst*n):int(tste*n)]
    XteB = df[featsB].values[int(tst*n):int(tste*n)]
    yte_act = df['label_pct'].values[int(tst*n):int(tste*n)]
    
    sw1=np.array([wd[yi] for yi in ytr])
    sw2=np.copy(sw1)
    decay=np.linspace(0.3,1.0,len(sw1))
    sw1*=decay  # 模型A有时间衰减
    # 模型B不用时间衰减（动量特征本身含时间维度）
    _=sw2  # 不用了
    
    # A: 时间衰减模型
    mA = xgb.XGBClassifier(n_estimators=500, max_depth=5, lr=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0, num_class=3, device='cuda')
    mA.fit(XtrA, ytr, sample_weight=sw1, 
           eval_set=[(XvaA, yva)], verbose=0)
    
    # B: 动量模型
    mB = xgb.XGBClassifier(n_estimators=500, max_depth=5, lr=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=43, n_jobs=-1, verbosity=0, num_class=3, device='cuda')
    mB.fit(XtrB, ytr, eval_set=[(XvaB, yva)], verbose=0)
    
    probaA = mA.predict_proba(XteA)[:,2]
    probaB = mB.predict_proba(XteB)[:,2]
    
    all_A.extend(probaA.tolist())
    all_B.extend(probaB.tolist())
    all_act.extend(yte_act.tolist())
    
    proba_ensemble = (probaA + probaB) / 2
    top10 = proba_ensemble >= np.percentile(proba_ensemble, 90)
    r = yte_act[top10]
    sp = r.mean()/r.std()*math.sqrt(252) if r.std()>0 else 0
    wr = (r>0).mean()
    print(f"  {name}: 测{len(XteA)}行, Ensemble夏普={sp:.3f}, 胜率={wr:.1%}", flush=True)

# 合成 — Ensemble
preds_ens = (np.array(all_A) + np.array(all_B)) / 2
acts = np.array(all_act)
te = preds_ens >= np.percentile(preds_ens, 90)
r=acts[te]
sp=r.mean()/r.std()*math.sqrt(252) if r.std()>0 else 0
wr=(r>0).mean()
print(f"\n  Ensemble合成: 夏普={sp:.3f}, 胜率={wr:.1%}")

# 单独看A和B
for tag, preds in [("时间衰减A",all_A), ("动量模型B",all_B)]:
    p=np.array(preds)
    te=p>=np.percentile(p,90)
    r=acts[te]
    sp2=r.mean()/r.std()*math.sqrt(252) if r.std()>0 else 0
    wr2=(r>0).mean()
    print(f"  {tag}单独: 夏普={sp2:.3f}, 胜率={wr2:.1%}")

print(f"\n  >>> Ensemble夏普={sp:.3f} vs v12单独夏普=0.900")

# 全量训练
print("\n[全量 模型A+B Ensemble]")
train_end = int(len(df)*0.85)
XtrA_full, XtrB_full = df[featsA].values[:train_end], df[featsB].values[:train_end]
ytr_full = df['label_bucket'].values[:train_end]
XteA_full, XteB_full = df[featsA].values[train_end:], df[featsB].values[train_end:]
yte_full = df['label_pct'].values[train_end:]

sw_a = np.array([wd[yi] for yi in ytr_full])
decay=np.linspace(0.3,1.0,train_end)
sw_a*=decay

mA_final = xgb.XGBClassifier(n_estimators=500, max_depth=5, lr=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0, num_class=3, device='cuda')
mA_final.fit(XtrA_full, ytr_full, sample_weight=sw_a, eval_set=[(XteA_full, df['label_bucket'].values[train_end:])], verbose=0)

mB_final = xgb.XGBClassifier(n_estimators=500, max_depth=5, lr=0.1, subsample=0.8, colsample_bytree=0.8, random_state=43, n_jobs=-1, verbosity=0, num_class=3, device='cuda')
mB_final.fit(XtrB_full, ytr_full, eval_set=[(XteB_full, df['label_bucket'].values[train_end:])], verbose=0)

pA = mA_final.predict_proba(XteA_full)[:,2]
pB = mB_final.predict_proba(XteB_full)[:,2]
pEns = (pA+pB)/2
te=pEns>=np.percentile(pEns,90)
r=yte_full[te]
sp_f=r.mean()/r.std()*math.sqrt(252) if r.std()>0 else 0
wr_f=(r>0).mean()
print(f"  全量测试: Ensemble夏普={sp_f:.3f}, 胜率={wr_f:.1%}")

# 今日预测
latest = df.dropna(subset=featsA+featsB).drop_duplicates(subset='sym', keep='last')
X_latestA = latest[featsA].values
X_latestB = latest[featsB].values
p_la = mA_final.predict_proba(X_latestA)[:,2]
p_lb = mB_final.predict_proba(X_latestB)[:,2]
p_l_ens = (p_la + p_lb) / 2

results=[]
for i,(_,row) in enumerate(latest.iterrows()):
    results.append({'sym':row['sym'],'price':float(row['price']),
                    'prob_up':float(p_l_ens[i]),'prob_up_A':float(p_la[i]),'prob_up_B':float(p_lb[i])})
results.sort(key=lambda x:-x['prob_up'])

print(f"\n{'═'*65}")
print(f"{'排名':>3} {'代码':>8} {'价格':>8} {'Ensemble':>8} {'模型A':>7} {'模型B':>7}")
print(f"{'─'*65}")
for i,r in enumerate(results[:20]):
    print(f"{i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {r['prob_up']*100:>6.1f}% {r['prob_up_A']*100:>5.1f}% {r['prob_up_B']*100:>5.1f}%")

# 保存ensemble模型
mA_final.save_model(_paths.MODEL_DIR+"/us_xgb_v13_A.json")
mB_final.save_model(_paths.MODEL_DIR+"/us_xgb_v13_B.json")

print(f"\n✅ v13 完成! ({time.time()-T0:.0f}s)")
