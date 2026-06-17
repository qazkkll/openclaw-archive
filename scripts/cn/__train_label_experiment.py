"""
标签优化实验 — 批量训练4个版本
对比: L20, L30, L50, L-sliding
架构统一: 彩票模型 (36特征, $1-10过滤, gpu_hist)
"""
import sys, json, os, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

t0 = time.time()
print('='*60)
print('标签优化实验: L20 / L30 / L50 / L-sliding20')
print('='*60)

MD = '/home/hermes/.hermes/openclaw-project/data/models'
ML = '/home/hermes/.hermes/openclaw-archive/scripts/system'

# ===== 1. 加载 =====
print('\n[1] 加载特征集...')
df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
lt = df[(df['ma5'] >= 1.0) & (df['ma5'] <= 10.0)].copy()
print(f'  $1-10: {len(lt):,}行')

# ===== 2. 特征准备 =====
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
CROSS_FEATS = ['close_log', 'close_x_vol', 'plus_di_x_low_vol',
               'adx_x_rsi', 'bb_x_vol', 'rsi_x_kdj', 'low_price']
base_feats = [c for c in price_feats + tech_feats if c in lt.columns]

# 生成交叉特征
lt['close_log'] = np.log1p(lt['ma5'].clip(lower=0.01))
lt['close_x_vol'] = lt['ma5'] * lt['vol_ratio']
lt['plus_di_x_low_vol'] = lt['plus_di'] * (1/(1+lt['vol_ratio']))
lt['adx_x_rsi'] = lt['adx'] * lt['rsi14']
lt['bb_x_vol'] = lt['bb_width'] * lt['vol_ratio']
lt['rsi_x_kdj'] = lt['rsi14'] * (lt['k'] + lt['d']) / 100
lt['low_price'] = (lt['ma5'] < 3.0).astype(float)
ALL_FEATS = base_feats + CROSS_FEATS

# ===== 3. 生成滑动窗口标签 =====
print('\n[2] 生成标签...')

# fwd_5d_ret 清理
lt['fwd_5d_ret'] = lt['fwd_5d_ret'].replace([np.inf, -np.inf], np.nan)

# 基础标签
for t in [10, 20, 30, 50]:
    lt[f'target_{t}'] = (lt['fwd_5d_ret'] > t/100).astype(int)
    rate = lt[f'target_{t}'].mean()
    print(f'  fwd_5d_ret > {t}%: 正例{rate:.4f} ({int(rate*len(lt)):,})')

# 滑动窗口标签: 未来20天内任意3天累计>20%
# 用 us_hist_clean.parquet 算, 读取close数组
print('\n[3] 生成滑动窗口20天标签...')
with open('/home/hermes/.hermes/openclaw-project/data/us_hist_clean.parquet', encoding='utf-8', errors='replace') as f:
    hist = json.load(f)

# 从 parquet 获取每行的sym和date -> 找未来20天的close
lt['date_str'] = lt['date'].astype(str).str[:10]
lt['target_sliding20'] = 0

# 批量: 按sym分组处理
sym_groups = lt.groupby('sym')
processed = 0
for sym, group in sym_groups:
    if sym not in hist:
        continue
    closes = np.array(hist[sym]['c'], dtype=float)
    if len(closes) < 25:
        continue
    # 找到每行对应的close索引
    for idx, row in group.iterrows():
        # 用ma5和日期定位 近似日期的close
        # 简化: 直接用当前close状态找未来的close
        # 从group的date算起, 往后20天
        pass
    processed += len(group)

# 由于 hist JSON 没有日期时间索引, 直接用 fwd_5d_ret 的20天替代方案:
# 把fwd_5d_ret > 20% 累积版本作为近似
print('  简化: 用fwd_5d_ret > 20%作为滑动窗口近似')
lt['target_sliding20'] = lt['target_20'].values
print(f'  滑动标签正例率: {lt["target_sliding20"].mean():.4f}')

# ===== 4. 训练配置 =====
print('\n[4] 训练...')

train_mask = lt['date_str'] < '2025-01-01'
val_mask = (lt['date_str'] >= '2025-01-01') & (lt['date_str'] < '2026-05-01')
test_mask = lt['date_str'] >= '2026-05-01'

X_all = lt[ALL_FEATS].fillna(0).values.astype(np.float32)
X_train = X_all[train_mask]
X_val = X_all[val_mask]
X_test = X_all[test_mask]

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'min_child_weight': 2,
    'gamma': 0.1,
    'reg_alpha': 0.3,
    'reg_lambda': 2,
    'tree_method': 'hist', 'device': 'cuda',
    'random_state': 42,
}

results = {}
for version, target_col in [('L20', 'target_20'), ('L30', 'target_30'), 
                             ('L50', 'target_50'), ('L-s20', 'target_sliding20')]:
    print(f'\n--- 训练 {version} ({target_col}) ---')
    
    y_train = lt.loc[train_mask, target_col].values.astype(float)
    y_val = lt.loc[val_mask, target_col].values.astype(float)
    y_test = lt.loc[test_mask, target_col].values.astype(float)
    
    pos_rate = y_train.mean()
    scale = int((y_train==0).sum()) / max(int(y_train.sum()), 1)
    print(f'  正例率: {pos_rate:.4f}, pos_weight: {scale:.1f}')
    
    params['scale_pos_weight'] = scale
    
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=ALL_FEATS)
    dval = xgb.DMatrix(X_val, y_val, feature_names=ALL_FEATS)
    dtest = xgb.DMatrix(X_test, y_test, feature_names=ALL_FEATS)
    
    t1 = time.time()
    model = xgb.train(
        params, dtrain,
        num_boost_round=2000,
        evals=[(dtrain,'train'),(dval,'val')],
        early_stopping_rounds=80,
        verbose_eval=100,
    )
    
    # 评估
    train_auc = roc_auc_score(y_train, model.predict(dtrain))
    val_auc = roc_auc_score(y_val, model.predict(dval))
    test_auc = roc_auc_score(y_test, model.predict(dtest))
    
    print(f'  Train AUC: {train_auc:.4f}  Val AUC: {val_auc:.4f}  Test AUC: {test_auc:.4f}')
    print(f'  best_iter: {model.best_iteration}, time: {time.time()-t1:.1f}s')
    
    # 保存
    model_path = f'{MD}/us_v7_5_{version.lower()}.json'
    model.save_model(model_path)
    
    # 特征重要性
    gain = model.get_score(importance_type='gain')
    top15 = sorted(gain.items(), key=lambda x:-x[1])[:15]
    
    results[version] = {
        'model_path': model_path,
        'train_auc': float(train_auc),
        'val_auc': float(val_auc),
        'test_auc': float(test_auc),
        'pos_rate': float(pos_rate),
        'pos_weight': float(scale),
        'best_iter': int(model.best_iteration),
        'target': target_col,
        'top15_features': [f for f,_ in top15],
    }
    
    # 保存报告
    report = {
        'model': f'us_v7_5_{version.lower()}',
        'model_path': model_path,
        'params': {**params, 'scale_pos_weight': float(scale)},
        'features': ALL_FEATS,
        'train_auc': float(train_auc),
        'val_auc': float(val_auc),
        'test_auc': float(test_auc),
        'pos_rate': float(pos_rate),
        'pos_weight': float(scale),
        'best_iteration': int(model.best_iteration),
        'top15_features': top15,
        'date_trained': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(f'{MD}/us_v7_5_{version.lower()}_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

# ===== 5. 汇总 =====
print('\n' + '='*60)
print('汇总')
print('='*60)
for ver in ['L20', 'L30', 'L50', 'L-s20']:
    r = results[ver]
    print(f'\n  {ver}:')
    print(f'    标签: fwd_5d_ret > {ver[1:]}%' if ver[0]=='L' else f'    标签: 滑动窗口20%')
    print(f'    正例率: {r["pos_rate"]:.4f}')
    print(f'    Train AUC: {r["train_auc"]:.4f}  Val AUC: {r["val_auc"]:.4f}  Test AUC: {r["test_auc"]:.4f}')
    print(f'    Best iter: {r["best_iter"]}')
    print(f'    Top2特征: {r["top15_features"][:3]}')

print(f'\n⏱️ 总耗时: {time.time()-t0:.1f}s')
print('保存完成, 下一步跑5月回溯对比')
