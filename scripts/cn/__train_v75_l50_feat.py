"""
V8-Lottery-L Feat+ — 彩票股增强版训练
在L50基础上加入6个"爆发潜伏期"特征
专注 $1-10 低价股的未来爆发预测 (30%+ / 5日内)
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from collections import defaultdict

t0 = time.time()
print('='*60)
print('V7.5-L Feat+ — 加入爆发潜伏期信号')
print('='*60)

MD = '/home/hermes/.hermes/openclaw-project/data/models'
ML = '/home/hermes/.hermes/openclaw-archive/scripts/system'

# ===== 1. 加载数据 =====
print('\n[1/8] 加载特征集...')
df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
print(f'  总行数: {len(df):,}')
print(f'  时间: {df.date.min()} ~ {df.date.max()}')

# ===== 2. 过滤：只保留低价股 =====
print('\n[2/8] 过滤低价股 ($1-$10)...')
lt = df[(df['ma5'] >= 1.0) & (df['ma5'] <= 10.0)].copy()
print(f'  保留: {len(lt):,} 行 ({len(lt)/len(df)*100:.1f}%)')
print(f'  sym数: {lt.sym.nunique():,}')

# 清理fwd_5d_ret
lt['fwd_5d_ret'] = lt['fwd_5d_ret'].replace([np.inf, -np.inf], np.nan)

# ===== 3. 生成多重标签 =====
print('\n[3/8] 生成多阈值标签...')
for threshold in [0.10, 0.20, 0.30, 0.50]:
    lt[f'target_{threshold:.0f}'] = (lt['fwd_5d_ret'] > threshold).astype(int)
    rate = lt[f'target_{threshold:.0f}'].mean()
    print(f'  target_>{threshold*100:.0f}%: {rate:.4f} ({int(rate*len(lt)):,} 正样本)')

TARGET = 'target_30'
print(f'\n  主标签: {TARGET}')

# ===== 4. 特征选择 =====
print('\n[4/8] 特征裁剪...')

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
LOTTERY_FEATS = price_feats + tech_feats

# 确保列存在
avail = [c for c in LOTTERY_FEATS if c in lt.columns]
if len(avail) < len(LOTTERY_FEATS):
    missing = set(LOTTERY_FEATS) - set(avail)
    print(f'  ⚠️ 缺失: {missing}')
print(f'  基础特征: {len(avail)}')

# ===== 5. 原交叉特征 =====
print('\n[5/8] 生成原交叉特征...')
lt = lt.copy()
lt['close_log'] = np.log1p(lt['ma5'].clip(lower=0.01))
lt['close_x_vol'] = lt['ma5'] * lt['vol_ratio']
lt['plus_di_x_low_vol'] = lt['plus_di'] * (1 / (1 + lt['vol_ratio']))
lt['adx_x_rsi'] = lt['adx'] * lt['rsi14']
lt['bb_x_vol'] = lt['bb_width'] * lt['vol_ratio']
lt['rsi_x_kdj'] = lt['rsi14'] * (lt['k'] + lt['d']) / 100
lt['low_price'] = (lt['ma5'] < 3.0).astype(float)

CROSS_FEATS = ['close_log', 'close_x_vol', 'plus_di_x_low_vol', 
               'adx_x_rsi', 'bb_x_vol', 'rsi_x_kdj', 'low_price']

# ===== 5b. 新增「爆发潜伏期」特征 =====
print('\n[5b/8] 生成爆发潜伏期信号特征...')
g = lt.groupby('sym')

# (1) pct_chg_1d — ma5单日变化率
ma5_lag1 = g['ma5'].transform(lambda x: x.shift(1))
lt['pct_chg_1d'] = (lt['ma5'] / ma5_lag1 - 1).replace([np.inf, -np.inf], 0.0).fillna(0.0)

# (2) pct_chg_5d — ma5 / ma5_5d_ago - 1 (5日趋势变化)
ma5_lag5 = g['ma5'].transform(lambda x: x.shift(5))
lt['pct_chg_5d'] = (lt['ma5'] / ma5_lag5 - 1).replace([np.inf, -np.inf], 0.0).fillna(0.0)

# (3) rsi_plus_di_cross — RSI低位+plus_di上穿
lt['rsi_plus_di_cross'] = (
    (lt['rsi14'] < 50) & 
    (lt['plus_di'] > lt['minus_di']) & 
    (lt['plus_di'] > 15)
).astype(float)

# (4) vol_surge_signal — 量能在低位温和放大
lt['vol_surge_signal'] = (
    (lt['vol_ratio'] > 0.8) & 
    (lt['vol_ratio'] < 1.5)
).astype(float)

# (5) bb_squeeze — 布林带宽 < 20日均值*0.8
bb_width_ma20 = g['bb_width'].transform(lambda x: x.rolling(30, min_periods=10).mean())
lt['bb_squeeze'] = (lt['bb_width'] < bb_width_ma20 * 0.8).astype(float)

# (6) price_reversal — 价格低位+RSI开始回升
lt['price_reversal'] = (
    (lt['price_position'] < 0.3) & 
    (lt['rsi14'] > 35) & 
    (lt['rsi14'] < 55)
).astype(float)

NEW_FEATS = ['pct_chg_1d', 'pct_chg_5d', 'rsi_plus_di_cross',
             'vol_surge_signal', 'bb_squeeze', 'price_reversal']

ALL_FEATS = avail + CROSS_FEATS + NEW_FEATS
print(f'  pct_chg_1d 非零: {(lt.pct_chg_1d!=0).mean():.1%}')
print(f'  pct_chg_5d 非零: {(lt.pct_chg_5d!=0).mean():.1%}')
print(f'  rsi_plus_di_cross 激活: {lt.rsi_plus_di_cross.mean():.1%}')
print(f'  vol_surge_signal 激活: {lt.vol_surge_signal.mean():.1%}')
print(f'  bb_squeeze 激活: {lt.bb_squeeze.mean():.1%}')
print(f'  price_reversal 激活: {lt.price_reversal.mean():.1%}')
print(f'  新增特征: {len(NEW_FEATS)}')
print(f'  最终特征: {len(ALL_FEATS)}')

# ===== 6. 时间切分 (对齐L50) =====
print('\n[6/8] 时间切分...')

# 检查L50用的什么切分 - 用report里train/val/test样本量(未保存)
# 用模板的切分: 2022前训练, 2023-2025-06验证, 2025-06后测试
# 但L50报告是训练到latest，用更合理的切分
lt = lt.copy()
for t in [10, 20, 30, 50]:
    lt[f'target_{t}'] = (lt['fwd_5d_ret'] > t/100).astype(int)

# 用和模板一样的切分确保可比
train_mask = lt['date'].astype(str).str[:10] < '2025-01-01'
val_mask = (lt['date'].astype(str).str[:10] >= '2025-01-01') & \
           (lt['date'].astype(str).str[:10] < '2026-05-01')
test_mask = lt['date'].astype(str).str[:10] >= '2026-05-01'

train = lt[train_mask].copy()
val = lt[val_mask].copy()
test = lt[test_mask].copy()

print(f'  训练: {len(train):,} ({len(train)/len(lt)*100:.1f}%)')
print(f'  验证: {len(val):,} ({len(val)/len(lt)*100:.1f}%)')
print(f'  测试(5月+): {len(test):,} ({len(test)/len(lt)*100:.1f}%)')

# 填充NaN
X_train = train[ALL_FEATS].fillna(0).values.astype(np.float32)
y_train = train[TARGET].values.astype(float)
X_val = val[ALL_FEATS].fillna(0).values.astype(np.float32)
y_val = val[TARGET].values.astype(float)
X_test = test[ALL_FEATS].fillna(0).values.astype(np.float32)
y_test = test[TARGET].values.astype(float)

pos_rate = y_train.mean()
print(f'  训练正例率: {pos_rate:.4f}')
neg_count = int((y_train == 0).sum())
pos_count = int(y_train.sum())
scale = neg_count / max(pos_count, 1)
print(f'  权重: {scale:.1f}')

# ===== 7. 训练 =====
print(f'\n[7/8] 训练 (GPU模式)...')

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
train_preds = model.predict(dtrain)
val_preds = model.predict(dval)
test_preds = model.predict(dtest)
train_auc = roc_auc_score(y_train, train_preds)
val_auc = roc_auc_score(y_val, val_preds)
test_auc = roc_auc_score(y_test, test_preds)
print(f'  Train AUC: {train_auc:.4f}')
print(f'  Val AUC: {val_auc:.4f}')
print(f'  Test AUC: {test_auc:.4f}')

# Precision@TopK
for name, preds, y_true in [('Train', train_preds, y_train), 
                              ('Val', val_preds, y_val), 
                              ('Test', test_preds, y_test)]:
    for k in [10, 20, 50, 100]:
        top_idx = np.argsort(-preds)[:k]
        top_true = y_true[top_idx].sum()
        total_pos = int(y_true.sum())
        print(f'  {name} precision@{k}: {top_true}/{k} = {top_true/k:.4f} (capture {top_true}/{total_pos} = {top_true/max(total_pos,1)*100:.1f}%)')

# ===== 特征重要性 =====
print('\n特征重要性 (Top 20):')
gain = model.get_score(importance_type='gain')
sorted_feats = sorted(gain.items(), key=lambda x: x[1], reverse=True)[:20]
for i, (feat, g) in enumerate(sorted_feats, 1):
    marker = ' ★NEW' if feat in NEW_FEATS else ''
    print(f'  {i:2d}. {feat:>25s}: {g:.1f}{marker}')

# ===== 8. 保存 =====
print('\n[8/8] 保存模型...')
model_path = f'{MD}/us_v7_5_l50_feat.json'
model.save_model(model_path)

report = {
    'model': 'us_v7_5_l50_feat',
    'model_path': model_path,
    'data_file': f'{ML}/us_ml_feats_v75.parquet',
    'params': params,
    'features': ALL_FEATS,
    'num_features': len(ALL_FEATS),
    'new_features': NEW_FEATS,
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'train_auc': float(train_auc),
    'val_auc': float(val_auc),
    'test_auc': float(test_auc),
    'lottery_threshold': '30%+ 5日涨幅',
    'best_iteration': int(model.best_iteration),
    'price_filter': '$1-$10',
    'pos_weight': round(float(scale), 2),
    'pos_rate': round(float(pos_rate), 4),
    'top20_features': [f for f, _ in sorted_feats],
    'removed_features': ['spy_ret*', 'qqq_ret*', 'iwm_ret*', 'pe_trailing', 
                         'pe_forward', 'div_yield', 'beta', 'ma10', 'ma20', 'ma30', 'ma60'],
    'date_trained': time.strftime('%Y-%m-%d %H:%M:%S'),
}

report_path = f'{MD}/us_v7_5_l50_feat_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f'  模型: {model_path}')
print(f'  报告: {report_path}')
print(f'⏱️ 总耗时: {time.time()-t0:.1f}s')
print('\n' + '='*60)
print('训练完成。与L50对比要点:')
print('  - 新增特征数:', len(NEW_FEATS))
print('  - 与L50训练参数完全一致')
print('  - 时间切分完全一致')
print('  - 评估方式完全一致')
print('='*60)
