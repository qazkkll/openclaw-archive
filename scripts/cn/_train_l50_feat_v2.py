"""
L50 + 6个潜伏期新特征 重新训练 v2
完全匹配原L50参数：train < 2025-01-01, target_50, 相同pos_weight
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

t0 = time.time()
print('='*60)
print('L50 + 6Feat 重训 v2 (匹配原L50参数)')
print('='*60)

MD = '/home/hermes/.hermes/openclaw-project/data/models'
ML = '/home/hermes/.hermes/openclaw-archive/scripts/system'

# ===== 1. 加载 =====
print('\n[1/6] 加载特征集...')
df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
print(f'  总行数: {len(df):,}')

# ===== 2. 过滤 =====
print('\n[2/6] 过滤 $1-$10...')
lt = df[(df['ma5'] >= 1.0) & (df['ma5'] <= 10.0)].copy()
print(f'  保留: {len(lt):,} 行')
lt['fwd_5d_ret'] = lt['fwd_5d_ret'].replace([np.inf, -np.inf], np.nan)

# ===== 3. 标签 =====
print('\n[3/6] 标签 target_50...')
lt['target_50'] = (lt['fwd_5d_ret'] > 0.50).astype(int)
rate = lt['target_50'].mean()
print(f'  target_>50%: {rate:.5f} ({int(rate*len(lt)):,} 正样本)')
TARGET = 'target_50'

# ===== 4. 特征 =====
price_feats = ['ma5','ma5_ratio','ma20_ratio','ma60_ratio']
tech_feats = [
    'vol5','vol20','vol_ratio',
    'ema12','ema26','macd','macd_signal','macd_hist',
    'rsi14','k','d','j',
    'bb_upper','bb_lower','bb_width','bb_position',
    'vol_ratio_ma5','vol_ratio_ma20',
    'adx','plus_di','minus_di',
    'price_position','price_position_60','cmf',
    'vix_close',
]
BASE = [c for c in price_feats + tech_feats if c in lt.columns]
print(f'  基础特征: {len(BASE)}')

# 交叉特征
lt['close_log'] = np.log1p(lt['ma5'].clip(lower=0.01))
lt['close_x_vol'] = lt['ma5'] * lt['vol_ratio']
lt['plus_di_x_low_vol'] = lt['plus_di'] * (1 / (1 + lt['vol_ratio']))
lt['adx_x_rsi'] = lt['adx'] * lt['rsi14']
lt['bb_x_vol'] = lt['bb_width'] * lt['vol_ratio']
lt['rsi_x_kdj'] = lt['rsi14'] * (lt['k'] + lt['d']) / 100
lt['low_price'] = (lt['ma5'] < 3.0).astype(float)
CROSS = ['close_log', 'close_x_vol', 'plus_di_x_low_vol',
         'adx_x_rsi', 'bb_x_vol', 'rsi_x_kdj', 'low_price']

# 6个新特征
lt = lt.sort_values(['sym', 'date']).copy()
lt['ma5_prev'] = lt.groupby('sym')['ma5'].shift(1)
lt['pct_chg_1d'] = (lt['ma5'] / lt['ma5_prev'] - 1).fillna(0).clip(-0.3, 0.3)
lt['ma5_5d_ago'] = lt.groupby('sym')['ma5'].shift(5)
lt['pct_chg_5d'] = (lt['ma5'] / lt['ma5_5d_ago'] - 1).fillna(0).clip(-0.5, 0.5)
lt['rsi_plus_di_cross'] = ((lt['rsi14'] < 50) & (lt['plus_di'] > lt['minus_di']) & (lt['plus_di'] > 15)).astype(float)
lt['vol_surge_signal'] = ((lt['vol_ratio'] > 0.8) & (lt['vol_ratio'] < 1.5)).astype(float)
lt['bb_width_ma20'] = lt.groupby('sym')['bb_width'].transform(lambda x: x.rolling(20, min_periods=5).mean())
lt['bb_squeeze'] = (lt['bb_width'] < lt['bb_width_ma20'] * 0.8).astype(float)
lt['price_reversal'] = ((lt['price_position'] < 0.3) & (lt['rsi14'] > 35) & (lt['rsi14'] < 55)).astype(float)

NEW = ['pct_chg_1d', 'pct_chg_5d', 'rsi_plus_di_cross',
       'vol_surge_signal', 'bb_squeeze', 'price_reversal']

ALL_FEATS = BASE + CROSS + NEW
print(f'  最终特征: {len(ALL_FEATS)} (29+7+6)')

# ===== 5. 时间切分（完全匹配原L50）=====
print('\n[5/6] 时间切分 (train<2025-01)...')
train_mask = lt['date'].astype(str).str[:10] < '2025-01-01'
val_mask = (lt['date'].astype(str).str[:10] >= '2025-01-01') & \
           (lt['date'].astype(str).str[:10] < '2026-05-01')
test_mask = lt['date'].astype(str).str[:10] >= '2026-05-01'

train = lt[train_mask].copy()
val = lt[val_mask].copy()
test = lt[test_mask].copy()

print(f'  训练: {len(train):,} ({len(train)/len(lt)*100:.1f}%)')
print(f'  验证: {len(val):,} ({len(val)/len(lt)*100:.1f}%)')
print(f'  测试: {len(test):,} ({len(test)/len(lt)*100:.1f}%)')

X_train = train[ALL_FEATS].fillna(0).values.astype(np.float32)
y_train = train[TARGET].values.astype(float)
X_val = val[ALL_FEATS].fillna(0).values.astype(np.float32)
y_val = val[TARGET].values.astype(float)
X_test = test[ALL_FEATS].fillna(0).values.astype(np.float32)
y_test = test[TARGET].values.astype(float)

pos_rate = y_train.mean()
neg_count = int((y_train == 0).sum())
pos_count = int(y_train.sum())
scale = neg_count / max(pos_count, 1)
print(f'  正例率: {pos_rate:.5f}')
print(f'  scale_pos_weight: {scale:.1f}')

# ===== 6. 训练 =====
print(f'\n[6/6] 训练 (GPU)...')
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'scale_pos_weight': scale,
    'min_child_weight': 2,
    'gamma': 0.1,
    'reg_alpha': 0.3,
    'reg_lambda': 2,
    'tree_method': 'hist', 'device': 'cuda',
    'random_state': 42,
}

dtrain = xgb.DMatrix(X_train, y_train, feature_names=ALL_FEATS)
dval = xgb.DMatrix(X_val, y_val, feature_names=ALL_FEATS)
dtest = xgb.DMatrix(X_test, y_test, feature_names=ALL_FEATS)

t1 = time.time()
model = xgb.train(
    params, dtrain, num_boost_round=2000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=80, verbose_eval=50,
)
print(f'\n  训练完成: {time.time()-t1:.1f}s, best={model.best_iteration}, auc={model.best_score:.4f}')

# ===== 评估 =====
val_preds = model.predict(dval)
test_preds = model.predict(dtest)
val_auc = roc_auc_score(y_val, val_preds)
test_auc = roc_auc_score(y_test, test_preds)
print(f'\n  Val AUC: {val_auc:.4f}')
print(f'  Test AUC: {test_auc:.4f}')

for name, preds, y_true in [('Val', val_preds, y_val), ('Test', test_preds, y_test)]:
    for topk in [10, 20, 50]:
        top_idx = np.argsort(-preds)[:topk]
        hits = int(y_true[top_idx].sum())
        print(f'  {name} precision@{topk}: {hits}/{topk} = {hits/topk:.2f}')

# 特征重要性
print('\n特征重要性 (Top 20):')
gain = model.get_score(importance_type='gain')
sorted_feats = sorted(gain.items(), key=lambda x: x[1], reverse=True)[:20]
for i, (feat, g) in enumerate(sorted_feats, 1):
    marker = ' (NEW)' if feat in NEW else ''
    print(f'  {i:2d}. {feat:>22s}: {g:.1f}{marker}')

print('\n新特征gain排名:')
for feat in NEW:
    rank = next((j+1 for j, (f, _) in enumerate(sorted(gain.items(), key=lambda x: -x[1])) if f == feat), 'N/A')
    print(f'  {feat:>22s}: gain={gain.get(feat, 0):.1f}, rank={rank}')

# ===== 保存 =====
print('\n保存模型...')
model_path = f'{MD}/us_v7_5_l50_feat.json'
model.save_model(model_path)

report = {
    'model': 'us_v7_5_l50_feat',
    'model_path': model_path,
    'params': params,
    'features': ALL_FEATS,
    'num_features': len(ALL_FEATS),
    'new_feature_names': NEW,
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'val_auc': float(val_auc),
    'test_auc': float(test_auc),
    'train_auc': float(model.best_score),
    'best_iteration': int(model.best_iteration),
    'pos_weight': round(float(scale), 2),
    'pos_rate': round(float(pos_rate), 5),
    'date_trained': time.strftime('%Y-%m-%d %H:%M:%S'),
}
with open(f'{MD}/us_v7_5_l50_feat_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f'  模型: {model_path}')
print(f'⏱️ 总耗时: {time.time()-t0:.1f}s')
