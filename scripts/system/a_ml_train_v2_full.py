"""
a_ml_train_v2_full.py — A股ML全量训练 v2 (正式版)
最优配置来自: scan01(参数n_est=300,depth=8,lr=0.2) + scan02(特征fd5_p2_vw10_20_extra19feat)
GPU: device='cuda'
"""
import json, sys, time, os, numpy as np, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()
LOG = open('/home/hermes/.hermes/openclaw-project/scripts/system/train_v2_full_log.txt', 'w', encoding='utf-8')
def log(s): print(s, flush=True); LOG.write(s+'\n'); LOG.flush()

log('1. 加载特征缓存...')
cache = json.load(open('/home/hermes/.hermes/openclaw-project/data/a_ml_feats_cache.json'))
X = np.array(cache['X'], dtype=np.float32)
y = np.array(cache['y'], dtype=np.float32)
log(f'   总数据: {X.shape}, 正例率: {y.mean():.2%}')

split = int(len(X) * 0.8)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
log(f'   训练: {X_tr.shape[0]}行, 测试: {X_te.shape[0]}行')

log('2. 训练 XGBoost GPU...')
m = xgb.XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.2,
    subsample=1.0, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, device='cuda'
)
m.fit(X_tr, y_tr)

p = m.predict_proba(X_te)[:, 1]
pred = m.predict(X_te)
acc = float(accuracy_score(y_te, pred))
auc = float(roc_auc_score(y_te, p))
log(f'   AUC: {auc:.4f}  Acc: {acc:.4f}')

bins = [(0,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.55),(0.55,0.6),(0.6,0.7),(0.7,1.0)]
log('\n3. 校准分析:')
log('  prob区间    实际正例率  样本数')
for lo, hi in bins:
    mask = (p >= lo) & (p < hi)
    n = mask.sum()
    if n > 0:
        actual = y_te[mask].mean()
        pred_mean = p[mask].mean()
        bias = (actual - pred_mean) * 100
        log(f'  {lo:.1f}-{hi:.1f}   actual={actual:.4f} pred={pred_mean:.4f} bias={bias:+.2f}%  n={n}')

# 纠正bias → 校准输出
corrections = {}
for lo, hi in bins:
    mask = (p >= lo) & (p < hi)
    if mask.sum() > 50:
        corrections[f'{lo:.1f}-{hi:.1f}'] = y_te[mask].mean()

os.makedirs('/home/hermes/.hermes/openclaw-project/data/models', exist_ok=True)
m.save_model('/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v2_full.json')
log(f'\n✅ 模型: a_xgb_tech_v2_full.json')

calib_info = {
    'calibration_bias': corrections,
    'note': '使用时对预测概率做校正: 用对应bin的actual替换raw概率'
}
json.dump(calib_info, open('/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v2_calib.json','w'), indent=2)
log(f'✅ 校准表: a_xgb_tech_v2_calib.json')

fn = ['r1','r5','r20','m5/m20','m12/m26','d5','d20','d60','align','v5','v20','rsi','macd','vr','pos','c/m60','vr20','vol_ratio','price_norm','vp_signal']
imp = m.feature_importances_
imp_sorted = sorted(zip(fn[:X.shape[1]], imp), key=lambda x: -x[1])
log('\n4. 特征重要性 (Top10):')
for n, i in imp_sorted[:10]:
    log(f'  {n}: {i:.4f}')

log(f'\n总耗时: {(time.time()-t0)/60:.1f}分')
LOG.close()
