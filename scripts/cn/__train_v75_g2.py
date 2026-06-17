"""
绿箭V8-Lottery-G2 重训练脚本
改动: 加入close/close_norm价格特征 + max_depth提升到8 + gpu_hist
"""
import sys, json, os, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import xgboost as xgb

t0 = time.time()
print('='*60)
print('V7.5-G2 重训练')
print('='*60)

# ==================== 1. 加载数据 ====================
print('\n[1] 加载 parquet...', flush=True)
df = pd.read_parquet('/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v75.parquet')
print(f'  原始数据: {len(df)} rows, {len(df.columns)} cols')

# 读取旧模型报告
with open('/home/hermes/.hermes/openclaw-project/data/models/us_v7_5_report.json') as f:
    old_report = json.load(f)
old_feats = old_report['features']
print(f'  旧模型特征: {len(old_feats)}')

# ==================== 2. 加入price特征 ====================
print('\n[2] 加入价格特征...', flush=True)

# ma5 就是价格 (close的MA值)
print('\n[3] 从ma5生成价格衍生特征...', flush=True)
# 使用ma5作为价格代理 (ma5已经包含了close信息)
df['close'] = df['ma5'].clip(lower=0.01)

# 价格log变换 (捕捉低价股非线性效应)
df['close_log'] = np.log1p(df['ma5'].clip(lower=0.01))

# 价格 * 成交量交叉 (捕捉冷门低价股模式)
df['close_x_vol_ratio'] = df['ma5'] * df['vol_ratio']

# plus_di * 低成交量 交叉 (正向趋势+冷门=彩票特征)
df['plus_di_x_low_vol'] = df['plus_di'] * (1 / (1 + df['vol_ratio']))

# ADX * RSI 交叉 (趋势强度+超卖)
df['adx_x_rsi'] = df['adx'] * df['rsi14']

print(f'  新增特征: close_log, close_x_vol_ratio, plus_di_x_low_vol, adx_x_rsi')

# ==================== 3. 准备训练数据 ====================
print('\n[4] 准备训练/验证/测试集...', flush=True)

# 新特征列表
new_feats = old_feats + ['close_log', 'close_x_vol_ratio', 'plus_di_x_low_vol', 'adx_x_rsi']
base_feats = [c for c in new_feats if c in df.columns]
print(f'  训练特征: {len(base_feats)}')

# 按时间分割
# 先转日期字符串
print(f'  日期范围: {df["date"].min()} ~ {df["date"].max()}')
print(f'  label值范围: {df["label"].min()} ~ {df["label"].max()}')
print(f'  label样例: {df["label"].iloc[:5].tolist()}')

# 旧模型训练时保留了 -1 作为负样本
# label 1 = 涨5%+, label -1/0 = 没涨
# 转换成: 1=正, 0=负
df['label_bin'] = (df['label'] == 1).astype(float)
print(f'  正例率: {df["label_bin"].mean():.4f}')

train_mask = df['date'].astype(str).str[:10] < '2026-05-01'
val_mask = (df['date'].astype(str).str[:10] >= '2026-05-01') & (df['date'].astype(str).str[:10] < '2026-06-01')
test_mask = df['date'].astype(str).str[:10] >= '2026-06-01'

X_train = df.loc[train_mask, base_feats].values.astype(np.float32)
y_train = df.loc[train_mask, 'label_bin'].values.astype(np.float32)
X_val = df.loc[val_mask, base_feats].values.astype(np.float32)
y_val = df.loc[val_mask, 'label_bin'].values.astype(np.float32)
X_test = df.loc[test_mask, base_feats].values.astype(np.float32)
y_test = df.loc[test_mask, 'label_bin'].values.astype(np.float32)

print(f'  训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}')
print(f'  正例率: 训练{y_train.mean():.4f} 验证{y_val.mean():.4f} 测试{y_test.mean():.4f}')

# ==================== 4. 训练 ====================
print('\n[5] 开始训练 (GPU模式)...', flush=True)

# 计算pos_weight
neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())
pos_weight = neg_count / max(pos_count, 1)
print(f'  pos_weight = {pos_weight:.2f}')

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 8,              # 旧: 6 → 8 (捕捉价格交互)
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': pos_weight,
    'min_child_weight': 3,
    'gamma': 0.2,
    'reg_alpha': 0.5,
    'reg_lambda': 3,
    'tree_method': 'hist', 'device': 'cuda',    # 旧: hist → gpu_hist
    'random_state': 42,
    'n_estimators': 2000,
    'early_stopping_rounds': 100,
}

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=base_feats)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=base_feats)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=base_feats)

t1 = time.time()
model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=100,
    verbose_eval=50,
)
print(f'\n  训练完成: {time.time()-t1:.1f}s, best_iter={model.best_iteration}, best_auc={model.best_score:.4f}')

# ==================== 5. 评估 ====================
print('\n[6] 评估...', flush=True)

val_preds = model.predict(dval)
test_preds = model.predict(dtest)

from sklearn.metrics import roc_auc_score
val_auc = roc_auc_score(y_val, val_preds)
test_auc = roc_auc_score(y_test, test_preds)
print(f'  Val AUC: {val_auc:.4f}')
print(f'  Test AUC: {test_auc:.4f}')

# 5月过滤回测: 同旧模型
from sklearn.metrics import accuracy_score
# 验证集上的标签分布
# 使用接近正例率的阈值 = 训练集正例率
thresh = y_train.mean()
y_val_hat = (val_preds > thresh).astype(int)
print(f'  验证集准确率: {accuracy_score(y_val, y_val_hat):.4f}')
print(f'  验证集正例率: {y_val_hat.mean():.4f} (实际: {y_val.mean():.4f})')

# ==================== 6. 保存 ====================
print('\n[7] 保存模型...', flush=True)

model_path = '/home/hermes/.hermes/openclaw-project/data/models/us_v7_5_g2.json'
model.save_model(model_path)

# 特征重要性
gain = model.get_score(importance_type='gain')
top20 = sorted(gain.items(), key=lambda x:-x[1])[:20]

# 原报告对比
print('\n  原模型Top20 vs 新模型Top20:')
for i, (f, v) in enumerate(top20, 1):
    print(f'    {i:>2d}. {f:>20s}: gain={v:>10.1f}')

# 保存报告
report = {
    'model': 'us_v7_5_g2',
    'model_path': model_path,
    'params': params,
    'features': base_feats,
    'num_features': len(base_feats),
    'val_auc': float(val_auc),
    'test_auc': float(test_auc),
    'best_iteration': int(model.best_iteration),
    'train_samples': int(len(X_train)),
    'val_samples': int(len(X_val)),
    'test_samples': int(len(X_test)),
    'pos_weight': pos_weight,
    'top20_features': top20,
    'training_time': time.time() - t0,
    'date_trained': time.strftime('%Y-%m-%d %H:%M:%S'),
}

report_path = '/home/hermes/.hermes/openclaw-project/data/models/us_v7_5_g2_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f'\n✅ 模型已保存: {model_path}')
print(f'✅ 报告已保存: {report_path}')
print(f'⏱️ 总耗时: {time.time()-t0:.1f}s')
