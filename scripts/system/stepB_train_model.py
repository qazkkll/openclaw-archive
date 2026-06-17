"""
stepB_train_model.py — 从缓存加载特征 + 训练XGBoost

用法: python stepB_train_model.py
"""
import json, sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

CACHE = '/home/hermes/.hermes/openclaw-project/data/a_ml_feats_cache.json'
MODEL = '/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v1.json'

t0 = time.time()
print('加载特征...', flush=True)
with open(CACHE, 'r') as f:
    d = json.load(f)

X = np.array(d['X'], dtype=np.float32)
y = np.array(d['y'], dtype=np.float32)
print(f'  X: {X.shape}, 正例率: {y.mean():.2%}', flush=True)

# 训练
print('训练XGBoost...', flush=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    eval_metric='logloss', random_state=42, n_jobs=-1)
m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

# 评估
p = m.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, m.predict(X_test))
auc = roc_auc_score(y_test, p)
print(f'\n✅ 评估: Acc={acc:.4f} AUC={auc:.4f}', flush=True)

# 校准
cal = CalibratedClassifierCV(m, method='sigmoid', cv='prefit')
cal.fit(X_test, y_test)
cp = cal.predict_proba(X_test)[:, 1]
print(f'校准后: {cp.mean():.4f}', flush=True)

# 分桶校准
bins = np.linspace(0, 1, 11)
for i in range(10):
    mask = (cp >= bins[i]) & (cp < bins[i+1])
    if mask.sum() > 5:
        actual = y_test[mask].mean()
        print(f'  [{bins[i]:.1f}-{bins[i+1]:.1f}] actual={actual:.3f} n={mask.sum()}', flush=True)

# 保存
os.makedirs(os.path.dirname(MODEL), exist_ok=True)
m.save_model(MODEL)
print(f'✅ 模型: {MODEL}', flush=True)

fn = ['r1','r5','r20','d5','d20','d60','align','v5','v20','rsi','macd','vr','pos','c/m60']
imp = m.feature_importances_
for n, i in sorted(zip(fn, imp), key=lambda x: -x[1]):
    print(f'  {n}: {i:.4f}', flush=True)

print(f'耗时: {(time.time()-t0)/60:.1f}分钟', flush=True)
