#!/usr/bin/env python3
"""
美股ML v12 — 加速/动量特征（热力图指标）
尝试突破最新窗口WF3（当前0.654）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import warnings; warnings.filterwarnings('ignore')
import json, time, math, pandas as pd, numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import _paths

T0 = time.time()
print("═══ v12 加速/动量特征 ═══")

df = pd.read_parquet(_paths.US_ML_FEATS)
base = ['ma5','ma10','ma20','ma60','rsi14','vol20','p52',
        'ret1','ret5','ret20','ret60',
        'macd','macd_signal','macd_hist','vol_ratio','ma_bias20',
        'price','volume']

# v9有效的新特征
df['vol20_ma20'] = df['vol20'].fillna(0) - (df['volume'] / df['volume'].rolling(20).mean()).fillna(0)
df['vol20_ma60'] = df['vol20'].fillna(0) - (df['volume'] / df['volume'].rolling(60).mean()).fillna(0)
df['ret60_vol20'] = df['ret60'].fillna(0) * df['vol20'].fillna(0)

# v12新特征: 动量加速
# ret20 - ret5 = 趋势是否在加速（正=持续加速）
df['trend_accel'] = df['ret20'].fillna(0) - df['ret5'].fillna(0)

# macd趋势加速: macd_hist变化率
df['macd_accel'] = df['macd_hist'].fillna(0) - df['macd_hist'].shift(1).fillna(0)

# 量价背离: ret1与vol_ratio方向是否一致
df['vol_price_div'] = df['ret1'].fillna(0) * df['vol_ratio'].fillna(0)

# 波动率变化: vol20的变化率
df['vol_surge'] = (df['vol20'].fillna(0) - df['vol20'].shift(5).fillna(0)) / df['vol20'].shift(5).fillna(df['vol20']).replace(0, np.nan).fillna(1)

# 相对p52的位置变化: ret1*(p52-50) 超买区域加速上=危险
df['p52_momentum'] = df['ret1'].fillna(0) * (df['p52'].fillna(50) - 50)

new_feats = ['trend_accel', 'macd_accel', 'vol_price_div', 
             'vol_surge', 'p52_momentum']

all_features = base + ['vol20_ma20', 'vol20_ma60', 'ret60_vol20'] + new_feats
print(f"  特征: {len(all_features)}列 (新增: {new_feats})")

df = df.dropna(subset=['label_pct'] + all_features).copy()

def bucket3(p):
    if p > 2: return 2
    if p > -2: return 1
    return 0
df['label_bucket'] = df['label_pct'].apply(bucket3)

X = df[all_features].values
y_b = df['label_bucket'].values
y_a = df['label_pct'].values

classes = np.array([0,1,2])
weights = compute_class_weight('balanced', classes=classes, y=y_b)
wd = {i:w for i,w in enumerate(weights)}

# Walk-Forward 3窗口
wfs = [('WF1',0,0.60,0.60,0.75,0.75,0.85),
       ('WF2',0.15,0.65,0.65,0.80,0.80,0.90),
       ('WF3',0.30,0.70,0.70,0.85,0.85,1.00)]

all_p, all_a = [], []
for name,ts,te,vs,ve,tst,tste in wfs:
    n=len(df)
    Xtr, ytr = X[int(ts*n):int(te*n)], y_b[int(ts*n):int(te*n)]
    Xte = X[int(tst*n):int(tste*n)]
    yte_act = y_a[int(tst*n):int(tste*n)]
    
    sw = np.array([wd[yi] for yi in ytr])
    decay = np.linspace(0.3, 1.0, len(sw))
    sw *= decay
    
    m = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', early_stopping_rounds=20,
        random_state=42, n_jobs=-1, verbosity=0, num_class=3, device='cuda')
    m.fit(Xtr, ytr, sample_weight=sw,
          eval_set=[(X[int(vs*n):int(ve*n)], y_b[int(vs*n):int(ve*n)])], verbose=0)
    
    proba = m.predict_proba(Xte)
    all_p.extend(proba[:,2].tolist())
    all_a.extend(yte_act.tolist())
    
    top10 = proba[:,2] >= np.percentile(proba[:,2], 90)
    r = yte_act[top10]
    sp = r.mean()/r.std()*math.sqrt(252) if r.std()>0 else 0
    wr=(r>0).mean()
    print(f"  {name}: 测{len(Xte)}行, 夏普={sp:.3f}, 胜率={wr:.1%}", flush=True)

preds, acts = np.array(all_p), np.array(all_a)
t10=preds>=np.percentile(preds,90)
r=acts[t10]
wf_sharpe = r.mean()/r.std()*math.sqrt(252) if r.std()>0 else 0
wf_wr = (r>0).mean()
print(f"\n  合成: 夏普={wf_sharpe:.3f}, 胜率={wf_wr:.1%}")

# 全量
print("\n[全量]")
train_end=int(len(df)*0.85)
sw_full=np.array([wd[yi] for yi in y_b[:train_end]])
decay=np.linspace(0.3,1.0,train_end)
sw_full*=decay

final=xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', early_stopping_rounds=20,
    random_state=42, n_jobs=-1, verbosity=0, num_class=3, device='cuda')
final.fit(X[:train_end], y_b[:train_end], sample_weight=sw_full,
          eval_set=[(X[train_end:], y_b[train_end:])], verbose=0)

y_p=final.predict_proba(X[train_end:])
pu=y_p[:,2]
ta=y_a[train_end:]
t10f=pu>=np.percentile(pu,90)
rf=ta[t10f]
sharpe_f = rf.mean()/rf.std()*math.sqrt(252) if rf.std()>0 else 0
wr_f = (rf>0).mean()
acc = accuracy_score(y_b[train_end:], final.predict(X[train_end:]))
print(f"  测试: acc={acc:.3f}, 夏普={sharpe_f:.3f}, 胜率={wr_f:.1%}")

# 今日预测
latest = df.dropna(subset=all_features).drop_duplicates(subset='sym', keep='last')
X_latest = latest[all_features].values
y_p_latest = final.predict_proba(X_latest)

results=[]
for i,(_,row) in enumerate(latest.iterrows()):
    probs=y_p_latest[i].tolist()
    results.append({'sym':row['sym'],'price':float(row['price']),
                    'prob_up':probs[2],'prob_flat':probs[1],'prob_down':probs[0]})
results.sort(key=lambda x:-x['prob_up'])

print(f"\n{'═'*60}")
print(f"{'排名':>3} {'代码':>8} {'价格':>8} {'涨>2%':>7} {'平±2%':>7} {'跌>2%':>7}")
print(f"{'─'*60}")
for i,r in enumerate(results[:20]):
    print(f"{i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {r['prob_up']*100:>6.1f}% "
          f"{r['prob_flat']*100:>6.1f}% {r['prob_down']*100:>6.1f}%")

# 保存到D盘统一路径
final.save_model(_paths.MODEL_DIR + "/us_xgb_v12.json")
output = {
    'timestamp': str(__import__('datetime').datetime.now(__import__('datetime').timezone(__import__('datetime').timedelta(hours=8)))),
    'model': 'us_xgb_v12',
    'features': all_features,
    'wf_sharpe': round(wf_sharpe,4), 'wf_win_rate': round(wf_wr,4),
    'final_sharpe': round(sharpe_f,4), 'final_win_rate': round(wr_f,4),
    'predictions': [{'rank':i+1,**r} for i,r in enumerate(results[:50])],
}
with open(_paths.MODEL_DIR + "/us_xgb_v12_prediction.json", 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ v12 完成! ({time.time()-T0:.0f}s)")
print(f"  模型: {_paths.win(_paths.MODEL_DIR + '/us_xgb_v12.json')}")
