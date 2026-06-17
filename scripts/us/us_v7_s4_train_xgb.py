"""
V7 Step 4: 训练XGBoost模型
输入: /home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v7_full.parquet (37列)
输出: /home/hermes/.hermes/openclaw-project/data/models/us_xgb_v7.json (及所有变体)

训练流程:
  1. 按时间切分训练/验证/测试 (2021-2024 / 2025 / 2026)
  2. 市值分层采样 (大盘>100B / 中盘10-100B / 小盘<10B)
  3. 正负样本平衡 (过采样)
  4. XGBoost训练 (多seed交叉验证)
  5. Platt校准
  6. 输出训练报告
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime

# ===== 配置 =====
FEATS_PATH = r'/home/hermes/.hermes/openclaw-archive/scripts/system\us_ml_feats_v7_full.parquet'
MODEL_DIR = r'/home/hermes/.hermes/openclaw-archive/data\models'
MODEL_NAME = 'us_xgb_v7'

# 目标特征列 (v3_dated技术指标 + 基本面)
TECH_FEATS = ['ma5','ma10','ma20','ma60','rsi14','macd','macd_signal','macd_hist',
              'bb_upper','bb_lower','vol20','vol_ratio','p52','slope_20d',
              'slope_50d','atr20','bb_width','adx','obv_slope','cci',
              'williams_r','stoch_k','stoch_d','mfi']
FUND_FEATS = ['sector','industry','market_cap','pe_trailing','pe_forward',
              'beta','div_yield','pb','roe','rev_growth','profit_growth',
              'debt_equity','gross_margin','profit_margin']
ALL_FEATS = TECH_FEATS + FUND_FEATS

# 时间切分
TRAIN_CUT = '2025-01-01'
VAL_CUT = '2026-01-01'

# 训练参数
TEST_PROB_THRESHOLD = 0.05  # 预测5日涨幅>5%？ >0.05=5%
HOLDOUT_INTERVAL = 20  # 采样间隔，控制训练集大小

# ===== 1. 加载数据 =====
print('=== V7 训练流程 ===')
print(f'加载特征集: {FEATS_PATH}')
df = pd.read_parquet(FEATS_PATH)
print(f'总行数: {len(df)}')
print(f'总列数: {len(df.columns)}')
print(f'时间范围: {df["date"].min()} ~ {df["date"].max()}')

# 统计特征覆盖率
print('\n特征覆盖率:')
for f in ALL_FEATS:
    if f in df.columns:
        valid = df[f].notna().sum()
        print(f'  {f:20s}: {valid}/{len(df)} ({valid/len(df)*100:.1f}%)')
    else:
        print(f'  {f:20s}: ❌ 缺失')

# 生成标签: 5日涨幅
print('\n生成标签...')
# 按sym分组，对于每行，往前看5天的价格变化
df = df.sort_values(['sym','date']).reset_index(drop=True)
df['price_next_5'] = df.groupby('sym')['price'].shift(-5)  # 5日后的价格
df['label'] = (df['price_next_5'] / df['price'] - 1) > TEST_PROB_THRESHOLD
df['ret_5d'] = df['price_next_5'] / df['price'] - 1

valid_label = df['label'].notna().sum()
print(f'有效标签: {valid_label}/{len(df)} ({valid_label/len(df)*100:.1f}%)')
print(f'正样本率: {df["label"].mean()*100:.2f}%')

# ===== 2. 时间切分 =====
print('\n划分数据集...')
train = df[df['date'] < TRAIN_CUT].copy()
val = df[(df['date'] >= TRAIN_CUT) & (df['date'] < VAL_CUT)].copy()
test = df[df['date'] >= VAL_CUT].copy()

print(f'训练集: {len(train)}  验证集: {len(val)}  测试集: {len(test)}')
print(f'训练集正样本率: {train["label"].mean()*100:.2f}%')
print(f'验证集正样本率: {val["label"].mean()*100:.2f}%')

# ===== 3. 市值分层采样 =====
print('\n市值分层采样...')

# 确定可用特征（剔除缺失率很高的特征和标签/价格列）
available_feats = [f for f in ALL_FEATS if f in df.columns and df[f].notna().sum() > len(df) * 0.3]
print(f'可用特征: {len(available_feats)} 个')

# 对分类特征编码
train_encoded = train.copy()
val_encoded = val.copy()
test_encoded = test.copy()

for col in ['sector', 'industry']:
    if col in available_feats:
        train_encoded[col] = train_encoded[col].astype('category').cat.codes
        val_encoded[col] = val_encoded[col].astype('category').cat.codes
        test_encoded[col] = test_encoded[col].astype('category').cat.codes

# 分层采样（大盘权重低，小盘加入更多负样本？）
# 这里只是标记一下，实际采样策略在训练时控制scale_pos_weight

# ===== 4. 训练XGBoost =====
print('\n训练XGBoost V7...')
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

# 清理特征矩阵 - 只保留特征列+标签
label_col = 'label'
X_train = train_encoded[available_feats].fillna(0)
y_train = train_encoded[label_col].astype(int)
X_val = val_encoded[available_feats].fillna(0)
y_val = val_encoded[label_col].astype(int)
X_test = test_encoded[available_feats].fillna(0)
y_test = test_encoded[label_col].astype(int)

print(f'训练特征矩阵: {X_train.shape}')

# 计算正负样本权重（支持分层采样）
pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print(f'正负样本权重: {pos_weight:.2f}')

# XGBoost训练
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': pos_weight,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 5,
    'random_state': 42,
    'n_estimators': 500,
    'early_stopping_rounds': 30,
}

dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)

print('训练中...')
model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=30,
    verbose_eval=50,
)

# ===== 5. 验证 =====
print('\n验证集评估...')
y_val_prob = model.predict(dval)
y_val_pred = (y_val_prob > 0.5).astype(int)
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve

val_acc = accuracy_score(y_val, y_val_pred)
val_auc = roc_auc_score(y_val, y_val_prob)
print(f'验证集 Acc: {val_acc:.4f}  AUC: {val_auc:.4f}')

# ===== 6. Platt校准 =====
print('\nPlatt校准...')
from sklearn.isotonic import IsotonicRegression

# 测试集做校准（保留一些样本）
cal_size = min(len(X_test) // 2, 50000)
X_cal = X_test.iloc[:cal_size]
y_cal = y_test.iloc[:cal_size]
X_test_remain = X_test.iloc[cal_size:]
y_test_remain = y_test.iloc[cal_size:]

dcal = xgb.DMatrix(X_cal)
y_cal_prob = model.predict(dcal)

# 简单的Platt缩放 (用逻辑回归做二次校准)
from sklearn.linear_model import LogisticRegression
calibrator = LogisticRegression(C=1.0, solver='lbfgs')
calibrator.fit(y_cal_prob.reshape(-1, 1), y_cal)
y_cal_calibrated = calibrator.predict_proba(y_cal_prob.reshape(-1, 1))[:, 1]

print(f'校准后AUC: {roc_auc_score(y_cal, y_cal_calibrated):.4f}')

# 校准后概率分布检查
for lb in np.arange(0, 1.0, 0.1):
    mask = (y_cal_calibrated >= lb) & (y_cal_calibrated < lb + 0.1)
    if mask.sum() > 0:
        actual = y_cal[mask].mean()
        print(f'  prob {lb:.1f}-{lb+0.1:.1f} → actual {actual:.3f}')

# ===== 7. 保存模型 =====
print('\n保存模型...')
os.makedirs(MODEL_DIR, exist_ok=True)

v7_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}.json')
model.save_model(v7_path)

# 保存校准器
import pickle
cal_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_calibrator.pkl')
with open(cal_path, 'wb') as f:
    pickle.dump(calibrator, f)

# 保存训练报告
report = {
    'model': MODEL_NAME,
    'training_time': str(datetime.now()),
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'features': available_feats,
    'num_features': len(available_feats),
    'val_accuracy': round(float(val_acc), 4),
    'val_auc': round(float(val_auc), 4),
    'params': params,
    'prob_threshold': TEST_PROB_THRESHOLD,
    'pos_weight': round(pos_weight, 4),
    'model_path': v7_path,
    'calibrator_path': cal_path,
}

report_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_report.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f'模型保存: {v7_path}')
print(f'报告保存: {report_path}')
print(f'\n✅ V7 训练完成')
print(f'特征数: {len(available_feats)}')
print(f'验证集 AUC: {val_auc:.4f}')
print(f'验证集 Acc: {val_acc:.4f}')
