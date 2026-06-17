"""
L50 + 6个潜伏期新特征 重新训练
目标：看命中率能否从17.1%提升
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from collections import defaultdict

t0 = time.time()
print('='*60)
print('L50 + 6个潜伏期特征 重新训练')
print('='*60)

MD = '/home/hermes/.hermes/openclaw-project/data/models'
ML = '/home/hermes/.hermes/openclaw-archive/scripts/system'

# ===== 1. 加载数据 =====
print('\n[1/7] 加载特征集...')
df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
print(f'  总行数: {len(df):,}')
print(f'  时间: {df.date.min()} ~ {df.date.max()}')

# ===== 2. 过滤低价股 =====
print('\n[2/7] 过滤低价股 ($1-$10)...')
lt = df[(df['ma5'] >= 1.0) & (df['ma5'] <= 10.0)].copy()
print(f'  保留: {len(lt):,} 行 ({len(lt)/len(df)*100:.1f}%)')
print(f'  sym数: {lt.sym.nunique():,}')

# 清理fwd_5d_ret
lt['fwd_5d_ret'] = lt['fwd_5d_ret'].replace([np.inf, -np.inf], np.nan)

# ===== 3. 生成标签 =====
print('\n[3/7] 生成标签 target_50 (fwd_5d_ret > 0.50)...')
lt['target_50'] = (lt['fwd_5d_ret'] > 0.50).astype(int)
rate = lt['target_50'].mean()
print(f'  target_>50%: {rate:.5f} ({int(rate*len(lt)):,} 正样本)')

TARGET = 'target_50'

# ===== 4. 特征选择（同L50原配方）=====
print('\n[4/7] 特征裁剪...')

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
BASE_FEATS = price_feats + tech_feats
avail = [c for c in BASE_FEATS if c in lt.columns]
print(f'  基础特征: {len(avail)} 可用')

# ===== 5. 生成交叉特征（同L50原配方）=====
print('\n[5/7] 生成交叉特征...')
lt = lt.sort_values(['sym', 'date']).copy()
lt['close_log'] = np.log1p(lt['ma5'].clip(lower=0.01))
lt['close_x_vol'] = lt['ma5'] * lt['vol_ratio']
lt['plus_di_x_low_vol'] = lt['plus_di'] * (1 / (1 + lt['vol_ratio']))
lt['adx_x_rsi'] = lt['adx'] * lt['rsi14']
lt['bb_x_vol'] = lt['bb_width'] * lt['vol_ratio']
lt['rsi_x_kdj'] = lt['rsi14'] * (lt['k'] + lt['d']) / 100
lt['low_price'] = (lt['ma5'] < 3.0).astype(float)

CROSS_FEATS = ['close_log', 'close_x_vol', 'plus_di_x_low_vol',
               'adx_x_rsi', 'bb_x_vol', 'rsi_x_kdj', 'low_price']

# ===== 5b. 生成6个新特征（潜伏期信号）=====
print('\n[5b/7] 生成6个潜伏期新特征...')

# 1. pct_chg_1d - 单日变化率（用ma5 shift）
lt['ma5_prev'] = lt.groupby('sym')['ma5'].shift(1)
lt['pct_chg_1d'] = (lt['ma5'] / lt['ma5_prev'] - 1).fillna(0).clip(-0.3, 0.3)

# 2. pct_chg_5d - 5日变化率
lt['ma5_5d_ago'] = lt.groupby('sym')['ma5'].shift(5)
lt['pct_chg_5d'] = (lt['ma5'] / lt['ma5_5d_ago'] - 1).fillna(0).clip(-0.5, 0.5)

# 3. rsi_plus_di_cross — RSI<50但plus_di上穿minus_di
lt['rsi_plus_di_cross'] = ((lt['rsi14'] < 50) & (lt['plus_di'] > lt['minus_di']) & (lt['plus_di'] > 15)).astype(float)

# 4. vol_surge_signal — 量能在低位温和放大
lt['vol_surge_signal'] = ((lt['vol_ratio'] > 0.8) & (lt['vol_ratio'] < 1.5)).astype(float)

# 5. bb_squeeze — 布林带宽窄（量化紧缩）
lt['bb_width_ma20'] = lt.groupby('sym')['bb_width'].transform(lambda x: x.rolling(20, min_periods=5).mean())
lt['bb_squeeze'] = (lt['bb_width'] < lt['bb_width_ma20'] * 0.8).astype(float)

# 6. price_reversal — 价格低位+RSI回升
lt['price_reversal'] = ((lt['price_position'] < 0.3) & (lt['rsi14'] > 35) & (lt['rsi14'] < 55)).astype(float)

NEW_FEATS = ['pct_chg_1d', 'pct_chg_5d', 'rsi_plus_di_cross',
             'vol_surge_signal', 'bb_squeeze', 'price_reversal']

ALL_FEATS = avail + CROSS_FEATS + NEW_FEATS
print(f'  基础特征: {len(avail)}')
print(f'  交叉特征: {len(CROSS_FEATS)}')
print(f'  新特征: {len(NEW_FEATS)}')
print(f'  最终特征: {len(ALL_FEATS)}')
print('  新特征列表:')
for f in NEW_FEATS:
    print(f'    + {f}')

# ===== 6. 时间切分（同L50原方式）=====
print('\n[6/7] 时间切分...')

# L50原训练方式：<2025-01训练，2025-01到2026-05验证，2026-05+测试
# 但因为我们需要验证top5前瞻，分成train/val/test
# 训练集喂到2025-06，验证集2025-07到2026-04，保留2026-05+给最终test
# 实际上L50原脚本把所有<2025-01当train，但那主要是因为L30标签
# 对于target_50，更合适的是尽量多数据训练，设一个late val

# 做一个更大的训练集：直到2025-12，验证2026-01到2026-04
train_mask = lt['date'].astype(str).str[:10] < '2026-01-01'
val_mask = (lt['date'].astype(str).str[:10] >= '2026-01-01') & \
           (lt['date'].astype(str).str[:10] < '2026-05-01')
test_mask = lt['date'].astype(str).str[:10] >= '2026-05-01'

train = lt[train_mask].copy()
val = lt[val_mask].copy()
test = lt[test_mask].copy()

print(f'  训练(≤2025-12): {len(train):,} ({len(train)/len(lt)*100:.1f}%)')
print(f'  验证(2026-01~04): {len(val):,} ({len(val)/len(lt)*100:.1f}%)')
print(f'  测试(2026-05+): {len(test):,} ({len(test)/len(lt)*100:.1f}%)')

# 填充NaN
for f in ALL_FEATS:
    train[f] = train[f].fillna(0)
    val[f] = val[f].fillna(0)
    test[f] = test[f].fillna(0)

X_train = train[ALL_FEATS].values.astype(np.float32)
y_train = train[TARGET].values.astype(float)
X_val = val[ALL_FEATS].values.astype(np.float32)
y_val = val[TARGET].values.astype(float)
X_test = test[ALL_FEATS].values.astype(np.float32)
y_test = test[TARGET].values.astype(float)

pos_rate = y_train.mean()
print(f'  训练正例率: {pos_rate:.5f}')
neg_count = int((y_train == 0).sum())
pos_count = int(y_train.sum())
scale = neg_count / max(pos_count, 1)
print(f'  权重: {scale:.1f}')

# ===== 7. 训练 =====
print(f'\n[7/7] 训练 (GPU模式)...')

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
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=80,
    verbose_eval=50,
)
print(f'\n  训练完成: {time.time()-t1:.1f}s, best_iter={model.best_iteration}, best_auc={model.best_score:.4f}')

# ===== 评估 =====
print('\n评估...')
val_preds = model.predict(dval)
test_preds = model.predict(dtest)
val_auc = roc_auc_score(y_val, val_preds)
test_auc = roc_auc_score(y_test, test_preds)
print(f'  Val AUC: {val_auc:.4f}')
print(f'  Test AUC: {test_auc:.4f}')

# precision@topK
for name, preds, y_true in [('Val', val_preds, y_val), ('Test', test_preds, y_test)]:
    for topk in [10, 20, 50]:
        top_idx = np.argsort(-preds)[:topk]
        top_true = y_true[top_idx].sum()
        print(f'  {name} precision@{topk}: {top_true}/{topk} = {top_true/topk:.2f}')
        lottery_total = int(y_true.sum())
        lottery_caught = int(y_true[top_idx].sum())
        print(f'    lottery_capture@{topk}: {lottery_caught}/{lottery_total} = {lottery_caught/max(lottery_total,1)*100:.1f}%')

# ===== 特征重要性 =====
print('\n特征重要性 (Top 20):')
gain = model.get_score(importance_type='gain')
sorted_feats = sorted(gain.items(), key=lambda x: x[1], reverse=True)[:20]
for i, (feat, g) in enumerate(sorted_feats, 1):
    marker = ' ★' if feat in NEW_FEATS else ''
    print(f'  {i:2d}. {feat:>22s}: {g:.1f}{marker}')

# 新特征单独统计
print('\n新特征重要性 (gain排名):')
for feat in NEW_FEATS:
    rank = next((i for i, (f, _) in enumerate(sorted(gain.items(), key=lambda x: -x[1])) if f == feat), None)
    g = gain.get(feat, 0)
    print(f'  {feat:>22s}: gain={g:.1f}, 排名={rank+1 if rank is not None else "N/A"}')

# ===== 保存模型 =====
print('\n保存模型...')
model_path = f'{MD}/us_v7_5_l50_feat.json'
model.save_model(model_path)

report = {
    'model': 'us_v7_5_l50_feat',
    'model_path': model_path,
    'data_file': f'{ML}/us_ml_feats_v75.parquet',
    'params': params,
    'features': ALL_FEATS,
    'num_features': len(ALL_FEATS),
    'base_features': len(avail),
    'cross_features': len(CROSS_FEATS),
    'new_features': len(NEW_FEATS),
    'new_feature_names': NEW_FEATS,
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'val_auc': float(val_auc),
    'test_auc': float(test_auc),
    'train_auc': float(model.best_score),
    'lottery_threshold': '50%+ 5日涨幅',
    'best_iteration': int(model.best_iteration),
    'price_filter': '$1-$10',
    'pos_weight': round(float(scale), 2),
    'pos_rate': round(float(pos_rate), 5),
    'new_features_gain': {f: float(gain.get(f, 0)) for f in NEW_FEATS},
    'top20_features': [f for f, _ in sorted_feats],
    'date_trained': time.strftime('%Y-%m-%d %H:%M:%S'),
}

report_path = f'{MD}/us_v7_5_l50_feat_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f'  模型: {model_path}')
print(f'  报告: {report_path}')
print(f'⏱️ 总耗时: {time.time()-t0:.1f}s')
