#!/usr/bin/env python3
"""
us_ml_03_scan_params.py — XGBoost参数扫描 + GPU加速
对us_ml_feats_v4.parquet进行参数搜索，找最优配置
分批训练，每组合写checkpoint断点续传
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

INPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v4.parquet'
OUTPUT = '/home/hermes/.hermes/openclaw-project/data/models/us/scan_v4_results.json'
CKPT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_scan_v4_ckpt.json'

print("us_ml_03: XGBoost GPU参数扫描...")

print("读特征数据...")
df = pd.read_parquet(INPUT)
print(f"  {len(df):,}行, 股票{df['ticker'].nunique()}, 标签: {df['label'].value_counts().to_dict()}")

# 特征列（去除非特征）
exclude = {'ticker', 'date', 'label', 'fwd_5d_ret'}
feat_cols = [c for c in df.columns if c not in exclude]
print(f"  特征数: {len(feat_cols)}")

X = df[feat_cols].values.astype(np.float32)
y = df['label'].values + 1  # XGBoost需要0,1,2 (对应-1,0,1)

# 时间序列切分（按顺序，不shuffle）
n = len(X)
split_idx = int(n * 0.7)
val_idx = int(n * 0.85)
X_train, X_val, X_test = X[:split_idx], X[split_idx:val_idx], X[val_idx:]
y_train, y_val, y_test = y[:split_idx], y[split_idx:val_idx], y[val_idx:]

del X, y, df  # 释放内存

print(f"  训练: {len(X_train):,}, 验证: {len(X_val):,}, 测试: {len(X_test):,}")

# 参数网格
param_grid = [
    {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8},  # 基准
    {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8},  # 加深
    {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8},  # 浅
    {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.10, 'subsample': 0.8, 'colsample_bytree': 0.8},  # 快学
    {'n_estimators': 800, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8},  # 慢学多树
    {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.7},  # 更保守
    {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9},  # 更激进
    {'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7},  # 深+正则
]

base_params = {
    'objective': 'multi:softprob', 'num_class': 3,
    'eval_metric': ['mlogloss', 'merror'],
    'tree_method': 'hist', 'device': 'cuda',
    'random_state': 42,
    'early_stopping_rounds': 20,
}

# 检查断点
all_results = {}
start_idx = 0
if os.path.exists(CKPT):
    cp = json.load(open(CKPT))
    all_results = cp.get('results', {})
    start_idx = cp.get('completed_to', 0)
    print(f"断点: 已训练 {start_idx}/{len(param_grid)} 组合")

T0 = time.time()
for i in range(start_idx, len(param_grid)):
    params = param_grid[i]
    t0 = time.time()
    
    xgb_params = {**base_params, **params}
    print(f"\n组合 {i+1}/{len(param_grid)}: max_depth={params['max_depth']}, lr={params['learning_rate']}, "
          f"n={params['n_estimators']}, subsample={params['subsample']}", flush=True)
    
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)
    
    model = xgb.train(
        xgb_params, dtrain, num_boost_round=params['n_estimators'],
        evals=[(dtrain, 'train'), (dval, 'val')],
        verbose_eval=False,
    )
    
    # 测试集评估
    dtest = xgb.DMatrix(X_test, y_test)
    y_pred = model.predict(dtest)
    y_pred_class = np.argmax(y_pred, axis=1)
    acc = np.mean(y_pred_class == y_test)
    
    # 校准分析（1类概率 vs 实际）
    prob_1 = y_pred[:, 1]  # label=0的概率
    actual_gain = np.mean(y_test == 1)
    pred_gain = np.mean(prob_1)
    calib_bias = pred_gain - actual_gain
    
    sec = time.time() - t0
    print(f"  Acc: {acc:.4f}, Calib偏差: {calib_bias*100:.2f}%, {sec:.0f}s", flush=True)
    
    # 特征重要性
    importance = model.get_score(importance_type='gain')
    total_gain = sum(importance.values()) or 1
    top_feats = sorted(importance.items(), key=lambda x: -x[1])[:5]
    top_str = ', '.join([f'{f}:{v/total_gain*100:.0f}%' for f, v in top_feats])
    
    key = f"md{params['max_depth']}_lr{params['learning_rate']}_n{params['n_estimators']}_ss{params['subsample']}"
    all_results[key] = {
        **params,
        'acc': round(float(acc), 4),
        'calib_bias_pct': round(float(calib_bias*100), 2),
        'time_sec': round(sec, 1),
        'top_feats': {f: round(float(v/total_gain*100), 1) for f, v in top_feats},
    }
    
    # 断点存
    json.dump({'results': all_results, 'completed_to': i+1}, open(CKPT, 'w'))
    
    # 释放显存
    del model, dtrain, dval, dtest, y_pred, y_pred_class, prob_1
    
    total = (time.time() - T0)/60
    pct = (i+1)/len(param_grid)*100
    print(f"  进度: {pct:.0f}%, 总耗时{total:.0f}分钟", flush=True)

# 最终结果排序
sorted_results = sorted(all_results.items(), key=lambda x: -x[1]['acc'])
print(f"\n{'='*60}")
print(f"参数扫描完成! 总耗时{(time.time()-T0)/60:.1f}分钟")
print(f"{'='*60}")
print(f"{'配置':<30} {'Acc':>6} {'Calib':>8} {'时间':>6}")
print("-"*60)
for key, res in sorted_results:
    print(f"{key:<30} {res['acc']:.4f} {res['calib_bias_pct']:>6.2f}% {res['time_sec']:>5.0f}s")

best_key = sorted_results[0][0]
best_res = sorted_results[0][1]
print(f"\n最佳配置: {best_key}")
print(f"  Acc: {best_res['acc']:.4f}")
print(f"  Calib偏差: {best_res['calib_bias_pct']:.2f}%")
print(f"  特征: {best_res['top_feats']}")

# 保存结果
json.dump({'best_config': best_key, 'best_metrics': best_res, 'all_results': all_results}, 
          open(OUTPUT, 'w'), indent=2)
print(f"\n结果保存: {OUTPUT}")
if os.path.exists(CKPT):
    os.remove(CKPT)
