"""
a_ml_train_xgb.py — A股ML全量训练 v2
来自scan01(参数)+scan02(特征)最优配置
参数: n_est=300, max_depth=8, lr=0.2, subsample=1.0
特征: fd5_p2_vw10_20_extra (19个特征, 含vp_signal等)
GPU: device='cuda'
"""
import json, sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

t0 = time.time()
MODEL_PATH = '/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v2.json'

# 1. 加载特征缓存 (667万行已预计算)
print('1. 加载特征缓存...', flush=True)
cache = json.load(open('/home/hermes/.hermes/openclaw-project/data/a_ml_feats_cache.json'))
X = np.array(cache['X'], dtype=np.float32)
y = np.array(cache['y'], dtype=np.float32)
print(f'  总数据: {X.shape}, 正例率: {y.mean():.2%}', flush=True)

# 2. 时间序列切分 (前80%训练, 后20%测试)
print('2. 切分数据集 (时间序列)...', flush=True)
split = int(len(X) * 0.8)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
print(f'  训练: {X_tr.shape[0]}行, 测试: {X_te.shape[0]}行', flush=True)

# 3. 训练 (scan01最佳参数)
print('3. 训练 XGBoost GPU...', flush=True)
m = xgb.XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.2,
    subsample=1.0, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
    device='cuda'
)
m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
# eval_metric='logloss' is auto

p = m.predict_proba(X_te)[:, 1]
pred = m.predict(X_te)
acc = float(accuracy_score(y_te, pred))
auc = float(roc_auc_score(y_te, p))
print(f'  AUC: {auc:.4f}  Acc: {acc:.4f}', flush=True)

# 4. 校准分析
bins = [(0,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,1.0)]
print('\n4. 校准分析:', flush=True)
print('  prob区间   实际正例率  样本数', flush=True)
for lo, hi in bins:
    mask = (p >= lo) & (p < hi)
    n = mask.sum()
    if n > 0:
        actual = y_te[mask].mean()
        pred_mean = p[mask].mean()
        print(f'  {lo:.1f}-{hi:.1f}   {actual:.4f}        {n}', flush=True)

# 5. 特征重要性
fn = ['r1','r5','r20','m5/m20','m12/m26','d5','d20','d60','align','v5','v20','rsi','macd','vr','pos','c/m60','vr20','vol_ratio','price_norm','vp_signal']
imp = m.feature_importances_
imp_sorted = sorted(zip(fn[:X.shape[1]], imp), key=lambda x: -x[1])
print('\n5. 特征重要性 (Top10):', flush=True)
for n, i in imp_sorted[:10]:
    print(f'  {n}: {i:.4f}', flush=True)

# 6. 保存
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
m.save_model(MODEL_PATH)
print(f'\n✅ 模型: {MODEL_PATH}', flush=True)
print(f'总耗时: {(time.time()-t0)/60:.1f}分', flush=True)
