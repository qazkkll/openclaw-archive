"""
V8-Lottery-L — 彩票股专属模型训练
专注 $1-10 低价股的未来爆发预测 (30%+ / 20天内)
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
from collections import defaultdict

t0 = time.time()
print('='*60)
print('V7.5-Lottery 训练')
print('='*60)

MD = '/home/hermes/.hermes/openclaw-project/data/models'
ML = '/home/hermes/.hermes/openclaw-archive/scripts/system'

# ===== 1. 加载数据 =====
print('\n[1/7] 加载特征集...')
df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
print(f'  总行数: {len(df):,}')
print(f'  时间: {df.date.min()} ~ {df.date.max()}')

# ===== 2. 过滤：只保留低价股 =====
print('\n[2/7] 过滤低价股 ($1-$10)...')
# ma5是5日均价 = 近似收盘价
lt = df[(df['ma5'] >= 1.0) & (df['ma5'] <= 10.0)].copy()
print(f'  保留: {len(lt):,} 行 ({len(lt)/len(df)*100:.1f}%)')
print(f'  sym数: {lt.sym.nunique():,}')

# 清理fwd_5d_ret
lt['fwd_5d_ret'] = lt['fwd_5d_ret'].replace([np.inf, -np.inf], np.nan)

# 生成多重标签 — 必须先copy后再生成
lt = lt.copy()

# ===== 3. 生成多重标签 =====
print('\n[3/7] 生成多阈值标签...')
# 用 fwd_5d_ret 是5日涨幅
# 目标是涨30%+ -> 用 fwd_5d_ret 的极端分位数
# 但更准确的: 用 us_hist_clean.parquet 算20天涨幅
# 目前先用5日涨幅的强阈值作为代理

for threshold in [0.10, 0.20, 0.30, 0.50]:
    lt[f'target_{threshold:.0f}'] = (lt['fwd_5d_ret'] > threshold).astype(int)
    rate = lt[f'target_{threshold:.0f}'].mean()
    print(f'  target_>{threshold*100:.0f}%: {rate:.4f} ({int(rate*len(lt)):,} 正样本)')

# 最终用 >30% 做主要标签  
TARGET = 'target_30'
print(f'\n  主标签: {TARGET}')

# ===== 4. 特征选择 =====
print('\n[4/7] 特征裁剪...')

# 保留的价格特征
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
# 去掉: spy/qqq/iwm大盘因子, pe/pe_forward/div_yield/beta基本面
# 去掉: ma10, ma20, ma30, ma60 (跟ma5_ratio/cross重复)
LOTTERY_FEATS = price_feats + tech_feats

# 新增交叉特征 (训练时动态生成)
print(f'  基础特征: {len(LOTTERY_FEATS)}')
print(f'  去掉: spy/qqq/iwm 大盘因子 (12个), pe/forward/beta/div_yield (4个)')

# 确保列存在
avail = [c for c in LOTTERY_FEATS if c in lt.columns]
if len(avail) < len(LOTTERY_FEATS):
    missing = set(LOTTERY_FEATS) - set(avail)
    print(f'  ⚠️ 缺失: {missing}')
print(f'  可用: {len(avail)}')

# ===== 5. 生成交叉特征 =====
print('\n[5/7] 生成交叉特征...')
lt = lt.copy()
lt['close_log'] = np.log1p(lt['ma5'].clip(lower=0.01))
lt['close_x_vol'] = lt['ma5'] * lt['vol_ratio']
lt['plus_di_x_low_vol'] = lt['plus_di'] * (1 / (1 + lt['vol_ratio']))
lt['adx_x_rsi'] = lt['adx'] * lt['rsi14']
lt['bb_x_vol'] = lt['bb_width'] * lt['vol_ratio']
lt['rsi_x_kdj'] = lt['rsi14'] * (lt['k'] + lt['d']) / 100
lt['low_price'] = (lt['ma5'] < 3.0).astype(float)  # 超低价flag

CROSS_FEATS = ['close_log', 'close_x_vol', 'plus_di_x_low_vol', 
               'adx_x_rsi', 'bb_x_vol', 'rsi_x_kdj', 'low_price']

ALL_FEATS = avail + CROSS_FEATS
print(f'  交叉特征: {len(CROSS_FEATS)}')
print(f'  最终特征: {len(ALL_FEATS)}')

# ===== 6. 时间切分 =====
print('\n[6/7] 时间切分...')

# 先生成target列再切片 (避免pandas视图问题)
lt = lt.copy()  # 确保是独立副本
for t in [10, 20, 30, 50]:
    lt[f'target_{t}'] = (lt['fwd_5d_ret'] > t/100).astype(int)

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
print(f'\n[7/7] 训练 (GPU模式)...')

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 8,
    'learning_rate': 0.05,        # 稍高一点，彩票模式需要更快反应
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

# 关注precision@top20
# 验证集=2025~2026-04, 测试=2026-05+
for name, preds, y_true, X_mtx in [('Val', val_preds, y_val, X_val), ('Test', test_preds, y_test, X_test)]:
    topk = 20
    top_idx = np.argsort(-preds)[:topk]
    top_true = y_true[top_idx].sum()
    print(f'  {name} precision@{topk}: {top_true}/{topk} = {top_true/topk:.2f}')
    # 彩票捕捉率（>30%的样本里，top20抓到了多少）
    lottery_total = int(y_true.sum())
    lottery_caught = int(y_true[top_idx].sum())
    print(f'  {name} lottery_capture@{topk}: {lottery_caught}/{lottery_total} = {lottery_caught/max(lottery_total,1)*100:.1f}%')

# ===== 特征重要性 =====
print('\n特征重要性 (Top 15):')
gain = model.get_score(importance_type='gain')
sorted_feats = sorted(gain.items(), key=lambda x: x[1], reverse=True)[:15]
for i, (feat, g) in enumerate(sorted_feats, 1):
    print(f'  {i:2d}. {feat:>20s}: {g:.1f}')

# ===== 保存 =====
print('\n保存模型...')
model_path = f'{MD}/us_v7_5_lottery.json'
model.save_model(model_path)

report = {
    'model': 'us_v7_5_lottery',
    'model_path': model_path,
    'data_file': f'{ML}/us_ml_feats_v75.parquet',
    'params': params,
    'features': ALL_FEATS,
    'num_features': len(ALL_FEATS),
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
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

report_path = f'{MD}/us_v7_5_lottery_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f'  模型: {model_path}')
print(f'  报告: {report_path}')
print(f'⏱️ 总耗时: {time.time()-t0:.1f}s')
