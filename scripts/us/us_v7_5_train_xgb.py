#!/usr/bin/env python3
"""
us_v7_5_train_xgb.py — V8-Lottery XGBoost模型训练
输入: /home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v75.parquet (51列, 548万行)
输出: /home/hermes/.hermes/openclaw-project/data/models/us_v7_5.json + 校准器 + 报告

训练策略:
  1. 时间切分: 2016-10~2023 (训练) / 2024 (验证) / 2025~2026 (测试)
  2. 数据量大（500万+），用历史+随机采样控制训练规模
  3. label=1 (涨>2%) / -1 (跌>2%) / 0 (横盘) → 二分类: label 1 vs 非1
  4. XGBoost binary:logistic, early stop
  5. 输出校准曲线 + 特征重要性
"""
import sys, os, json, time, pickle, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd, numpy as np

T0 = time.time()
print('='*60)
print('V8-Lottery XGBoost训练')
print('='*60)

# ===== 配置 =====
FEATS_PATH = r'/home/hermes/.hermes/openclaw-archive/scripts/system\us_ml_feats_v75.parquet'
MODEL_DIR = r'/home/hermes/.hermes/openclaw-archive/data\models'
MODEL_NAME = 'us_v7_5'

# 时间切分（留足2022熊市训练，空2024验证，2025-2026测试）
TRAIN_END = '2023-12-31'
VAL_END = '2024-12-31'

# 特征列（非文本的数值特征）
FEAT_COLS = [
    # 均线
    'ma5','ma10','ma20','ma30','ma60',
    'ma5_ratio','ma20_ratio','ma60_ratio',
    # 波动率
    'vol5','vol20','vol_ratio',
    # MACD
    'ema12','ema26','macd','macd_signal','macd_hist',
    # RSI
    'rsi14',
    # KDJ
    'k','d','j',
    # Bollinger
    'bb_upper','bb_lower','bb_width','bb_position',
    # 成交量
    'vol_ratio_ma5','vol_ratio_ma20',
    # 趋势
    'adx','plus_di','minus_di',
    # 价格位置
    'price_position','price_position_60',
    # 资金流
    'cmf',
    # 大盘ETF因子
    'spy_ret1','spy_ret5','spy_ret20','spy_ret60',
    'qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60',
    'iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60',
    # 恐慌因子
    'vix_close',
    # 基本面
    'pe_trailing','pe_forward','div_yield','beta',
    # 行业ETF因子
    'sector_etf_ret5','sc',
]
print(f'  特征数: {len(FEAT_COLS)}')

# ===== 1. 加载数据 =====
print(f'\n[1/6] 加载特征集...')
df = pd.read_parquet(FEATS_PATH)
print(f'  总行数: {len(df):,}')
print(f'  总列数: {len(df.columns)}')
print(f'  时间: {df.date.min()} ~ {df.date.max()}')

# ===== 2. 标签+过滤 =====
print(f'\n[2/6] 生成标签+过滤...')
# 尝试>5%涨幅标签（和v7.4一样），信号更强
# fwd_5d_ret是5日涨幅，取>5%作为正样本
df['target'] = (df['fwd_5d_ret'] > 0.05).astype(int)
print(f'  正样本(>5%涨): {df.target.sum():,} ({df.target.mean()*100:.1f}%)')
print(f'  负样本: {len(df)-df.target.sum():,}')

# 过滤无效标签
df = df[df['fwd_5d_ret'].notna()].copy()
print(f'  有效标签行: {len(df):,}')

# ===== 3. 时间切分 =====
print(f'\n[3/6] 时间切分...')
train = df[df['date'] < TRAIN_END].copy()
val = df[(df['date'] >= TRAIN_END) & (df['date'] < VAL_END)].copy()
test = df[df['date'] >= VAL_END].copy()

print(f'  训练集: {len(train):,} ({(len(train)/len(df)*100):.0f}%)  目标: {train.target.mean()*100:.1f}%')
print(f'  验证集: {len(val):,} ({(len(val)/len(df)*100):.0f}%)  目标: {val.target.mean()*100:.1f}%')
print(f'  测试集: {len(test):,} ({(len(test)/len(df)*100):.0f}%)  目标: {test.target.mean()*100:.1f}%')

# ===== 4. 特征矩阵 =====
print(f'\n[4/6] 构建特征矩阵...')

# 确保所有特征列都存在
available = [c for c in FEAT_COLS if c in df.columns]
missing = [c for c in FEAT_COLS if c not in df.columns]
if missing:
    print(f'  ⚠️ 缺失特征: {missing}')
print(f'  可用特征: {len(available)}')

# 全量训练（不采样），用hist算法加速
X_train = train[available].fillna(0).astype(np.float32)
y_train = train['target'].values.astype(int)
X_val = val[available].fillna(0).astype(np.float32)
y_val = val['target'].values.astype(int)
X_test = test[available].fillna(0).astype(np.float32)
y_test = test['target'].values.astype(int)

print(f'  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}')

# 正负样本权重
pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
scale_pos_weight = neg_count / max(pos_count, 1)
print(f'  正样本: {pos_count:,}, 负样本: {neg_count:,}, 权重: {scale_pos_weight:.2f}')

# ===== 5. 训练XGBoost =====
print(f'\n[5/6] 训练XGBoost...')
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# 从v7.4读取最优参数作为基线
try:
    v74_params = json.load(open(f'{MODEL_DIR}/us_v7_4.json_meta', 'r'))
    print(f'  使用v7.4参数')
    params_base = v74_params.get('params', {})
except:
    params_base = {}

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': params_base.get('max_depth', 6),
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale_pos_weight,
    'min_child_weight': 3,
    'gamma': 0.2,
    'reg_alpha': 0.5,
    'reg_lambda': 3,
    'random_state': 42,
    'n_estimators': 1000,
    'early_stopping_rounds': 100,
    'tree_method': 'hist',
}

dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)

print('  训练中...')
model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=30,
    verbose_eval=50,
)

# ===== 验证 =====
print(f'\n验证集评估...')
y_val_prob = model.predict(dval)
y_val_pred = (y_val_prob > 0.5).astype(int)
val_auc = roc_auc_score(y_val, y_val_prob)
val_acc = accuracy_score(y_val, y_val_pred)
print(f'  Acc: {val_acc:.4f}  AUC: {val_auc:.4f}')

# 测试集评估
dtest = xgb.DMatrix(X_test)
y_test_prob = model.predict(dtest)
y_test_pred = (y_test_prob > 0.5).astype(int)
test_auc = roc_auc_score(y_test, y_test_prob)
test_acc = accuracy_score(y_test, y_test_pred)
print(f'\n测试集评估...')
print(f'  Acc: {test_acc:.4f}  AUC: {test_auc:.4f}')

# ===== 校准 =====
print(f'\n校准...')
from sklearn.linear_model import LogisticRegression

cal_size = min(len(X_test) // 2, 100000)
X_cal = X_test.iloc[:cal_size]
y_cal = y_test[:cal_size]
X_test_hold = X_test.iloc[cal_size:]
y_test_hold = y_test[cal_size:]

dcal = xgb.DMatrix(X_cal)
y_cal_prob = model.predict(dcal)

calibrator = LogisticRegression(C=1.0, solver='lbfgs')
calibrator.fit(y_cal_prob.reshape(-1, 1), y_cal)
y_cal_calib = calibrator.predict_proba(y_cal_prob.reshape(-1, 1))[:, 1]

print(f'  校准后AUC: {roc_auc_score(y_cal, y_cal_calib):.4f}')
print(f'\n  校准曲线（等距分箱）:')
for lb in np.arange(0, 1.0, 0.1):
    mask = (y_cal_calib >= lb) & (y_cal_calib < lb + 0.1)
    if mask.sum() > 50:
        actual = y_cal[mask].mean()
        print(f'    prob {lb:.1f}-{lb+0.1:.1f} → actual {actual:.3f} (n={mask.sum()})')

# 测试保留集
dtest_hold = xgb.DMatrix(X_test_hold)
y_test_hold_prob = model.predict(dtest_hold)
y_test_hold_calib = calibrator.predict_proba(y_test_hold_prob.reshape(-1, 1))[:, 1]
test_hold_auc = roc_auc_score(y_test_hold, y_test_hold_calib)
print(f'\n  测试保留集校准后AUC: {test_hold_auc:.4f}')

# ===== 特征重要性 =====
print(f'\n特征重要性 (Top 20):')
importance = model.get_score(importance_type='gain')
sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
for i, (feat, gain) in enumerate(sorted_feats, 1):
    print(f'  {i:2d}. {feat:20s}: {gain:.4f}')

# ===== 6. 保存 =====
print(f'\n[6/6] 保存模型...')
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}.json')
model.save_model(model_path)

cal_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_calibrator.pkl')
with open(cal_path, 'wb') as f:
    pickle.dump(calibrator, f)

report = {
    'model': MODEL_NAME,
    'training_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    'data_file': FEATS_PATH,
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'features': available,
    'num_features': len(available),
    'val_accuracy': round(float(val_acc), 4),
    'val_auc': round(float(val_auc), 4),
    'test_accuracy': round(float(test_acc), 4),
    'test_auc': round(float(test_auc), 4),
    'test_hold_auc': round(float(test_hold_auc), 4),
    'params': params,
    'pos_weight': round(scale_pos_weight, 4),
    'top20_features': [f for f, _ in sorted_feats],
    'model_path': model_path,
    'calibrator_path': cal_path,
}

report_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_report.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f'  模型: {model_path}')
print(f'  报告: {report_path}')
print(f'\n{"="*60}')
total = (time.time()-T0)/60
print(f'🎉 V8-Lottery训练完成! 总耗时: {total:.0f}分钟')
print(f'  验证集AUC: {val_auc:.4f}')
print(f'  测试集AUC: {test_auc:.4f}')
print(f'  测试保留集校准AUC: {test_hold_auc:.4f}')
print(f'{"="*60}')
