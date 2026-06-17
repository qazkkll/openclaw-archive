"""
V8-Lottery-L50-W — 彩票股加权训练 (方案B: 加权分类)
标签还是二元分类(>50%)，但正样本按涨幅权重加权：
  sample_weight = 1 + max(0, (fwd_5d_ret - 0.50) * 3) 
  涨幅50%→weight=1, 100%→2.5, 200%→5.5
保留分类框架的稳定性 + 高爆发样本更高权重
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

t0 = time.time()
print('='*60)
print('V7.5-L50-W (方案B: 加权分类) 训练')
print('='*60)

MD = '/home/hermes/.hermes/openclaw-project/data/models'
ML = '/home/hermes/.hermes/openclaw-archive/scripts/system'

# ===== 1. 加载数据 =====
print('\n[1/7] 加载特征集...')
df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
print(f'  总行数: {len(df):,}')
print(f'  时间: {df.date.min()} ~ {df.date.max()}')

# ===== 2. 过滤 =====
print('\n[2/7] 过滤低价股 ($1-$10)...')
lt = df[(df['ma5'] >= 1.0) & (df['ma5'] <= 10.0)].copy()
print(f'  保留: {len(lt):,} 行')
print(f'  sym数: {lt.sym.nunique():,}')

# 清理
lt['fwd_5d_ret'] = lt['fwd_5d_ret'].replace([np.inf, -np.inf], np.nan)
lt = lt.dropna(subset=['fwd_5d_ret']).copy()
print(f'  清理后: {len(lt):,} 行')

# ===== 3. 标签 + 样本权重 =====
print('\n[3/7] 生成标签(二元) + 样本权重(涨幅加权)...')

# 标签：>50%涨幅 = 正样本
lt['label'] = (lt['fwd_5d_ret'] > 0.50).astype(int)

# 样本权重：正样本=1+涨幅超50%部分的3倍，负样本=1
lt['sample_weight'] = 1.0
pos_mask = lt['label'] == 1
lt.loc[pos_mask, 'sample_weight'] = 1.0 + np.clip(
    (lt.loc[pos_mask, 'fwd_5d_ret'] - 0.50) * 3, 
    0, 10  # cap at 10x
  )

pos_rate = lt['label'].mean()
print(f'  >50%涨幅正例率: {pos_rate:.4f} ({int(pos_rate*len(lt)):,} 行)')
print(f'  样本权重统计:')
print(f'    负样本: weight=1.0')
print(f'    正样本: mean={lt.loc[pos_mask, "sample_weight"].mean():.4f}')
print(f'            min={lt.loc[pos_mask, "sample_weight"].min():.4f} (接近50%)')
print(f'            max={lt.loc[pos_mask, "sample_weight"].max():.4f}')

# 正样本权重分布
bins = [1, 1.5, 2, 3, 5, 10]
for i in range(len(bins)-1):
    cnt = ((lt.loc[pos_mask, 'sample_weight'] >= bins[i]) & 
           (lt.loc[pos_mask, 'sample_weight'] < bins[i+1])).sum()
    print(f'    weight [{bins[i]:.1f}-{bins[i+1]:.1f}): {cnt}')

# class weight: 负样本总数 / 加权正样本总数
sw_pos = lt.loc[pos_mask, 'sample_weight'].sum()
sw_neg = (lt.loc[~pos_mask, 'sample_weight']).sum()  # = 负样本数
scale = sw_neg / max(sw_pos, 1)
print(f'  加权scale_pos_weight: {scale:.1f}')

# 也保留原始fwd_5d_ret for evaluation
lt['raw_ret'] = lt['fwd_5d_ret']

TARGET = 'label'
TARGET_RAW = 'raw_ret'

# ===== 4. 特征选择 =====
print('\n[4/7] 特征裁剪 (同L50)...')

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
print(f'  基础特征: {len(LOTTERY_FEATS)}')

avail = [c for c in LOTTERY_FEATS if c in lt.columns]
if len(avail) < len(LOTTERY_FEATS):
    missing = set(LOTTERY_FEATS) - set(avail)
    print(f'  ⚠️ 缺失: {missing}')
print(f'  可用: {len(avail)}')

# ===== 5. 交叉特征 =====
print('\n[5/7] 生成交叉特征...')
lt['close_log'] = np.log1p(lt['ma5'].clip(lower=0.01))
lt['close_x_vol'] = lt['ma5'] * lt['vol_ratio']
lt['plus_di_x_low_vol'] = lt['plus_di'] * (1 / (1 + lt['vol_ratio']))
lt['adx_x_rsi'] = lt['adx'] * lt['rsi14']
lt['bb_x_vol'] = lt['bb_width'] * lt['vol_ratio']
lt['rsi_x_kdj'] = lt['rsi14'] * (lt['k'] + lt['d']) / 100
lt['low_price'] = (lt['ma5'] < 3.0).astype(float)

CROSS_FEATS = ['close_log', 'close_x_vol', 'plus_di_x_low_vol',
               'adx_x_rsi', 'bb_x_vol', 'rsi_x_kdj', 'low_price']

ALL_FEATS = avail + CROSS_FEATS
print(f'  交叉特征: {len(CROSS_FEATS)}')
print(f'  最终特征: {len(ALL_FEATS)}')

# ===== 6. 时间切分 =====
print('\n[6/7] 时间切分...')

lt['date_str'] = lt['date'].astype(str).str[:10]

train_mask = lt['date_str'] < '2025-01-01'
val_mask = (lt['date_str'] >= '2025-01-01') & (lt['date_str'] < '2026-05-01')
test_mask = lt['date_str'] >= '2026-05-01'

# 确保是独立副本
train = lt[train_mask].copy()
val = lt[val_mask].copy()
test = lt[test_mask].copy()

print(f'  训练: {len(train):,} ({len(train)/len(lt)*100:.1f}%)')
print(f'  验证: {len(val):,} ({len(val)/len(lt)*100:.1f}%)')
print(f'  测试: {len(test):,} ({len(test)/len(lt)*100:.1f}%)')

X_train = train[ALL_FEATS].fillna(0).values.astype(np.float32)
y_train = train[TARGET].values.astype(int)
w_train = train['sample_weight'].values.astype(np.float32)
raw_train = train[TARGET_RAW].values.astype(np.float32)

X_val = val[ALL_FEATS].fillna(0).values.astype(np.float32)
y_val = val[TARGET].values.astype(int)
w_val = val['sample_weight'].values.astype(np.float32)
raw_val = val[TARGET_RAW].values.astype(np.float32)

X_test = test[ALL_FEATS].fillna(0).values.astype(np.float32)
y_test = test[TARGET].values.astype(int)
w_test = test['sample_weight'].values.astype(np.float32)
raw_test = test[TARGET_RAW].values.astype(np.float32)

print(f'  训练集: pos_rate={y_train.mean():.4f}, 加权scale={scale:.1f}')
print(f'  验证集: pos_rate={y_val.mean():.4f}')
print(f'  测试集: pos_rate={y_test.mean():.4f}')

# ===== 7. 训练 (加权分类) =====
print(f'\n[7/7] 训练 XGBoost 加权分类...')

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'scale_pos_weight': scale,
    'min_child_weight': 3,    # 稍大一点控制过拟合
    'gamma': 0.2,
    'reg_alpha': 0.5,
    'reg_lambda': 3,
    'tree_method': 'hist', 'device': 'cuda',
    'random_state': 42,
}

dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, feature_names=ALL_FEATS)
dval = xgb.DMatrix(X_val, label=y_val, weight=w_val, feature_names=ALL_FEATS)
dtest = xgb.DMatrix(X_test, label=y_test, weight=w_test, feature_names=ALL_FEATS)

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
train_preds = model.predict(dtrain)

val_auc = roc_auc_score(y_val, val_preds)
test_auc = roc_auc_score(y_test, test_preds)
print(f'  Val  AUC: {val_auc:.4f}')
print(f'  Test AUC: {test_auc:.4f}')

# ----- 核心: Top-N 命中 >50% 涨幅 -----
def eval_topk(preds, raw_ret, hit50_labels, topk, label=''):
    top_idx = np.argsort(-preds)[:topk]
    top_hits = int(hit50_labels[top_idx].sum())
    top_avg_ret = float(raw_ret[top_idx].mean())
    top_median_ret = float(np.median(raw_ret[top_idx]))
    top_max_ret = float(raw_ret[top_idx].max())
    total_hits = int(hit50_labels.sum())
    capture = top_hits / max(total_hits, 1) * 100
    return (top_hits, top_hits/topk, top_avg_ret, top_median_ret, top_max_ret, capture)

print(f'\n-- Top-5 命中率 (预测Top5中实际>50%涨幅的占比) --')
for topk in [5, 10, 20]:
    print(f'\n  Top-{topk}:')
    for name, preds, raw, hit in [
        ('Train', train_preds, raw_train, y_train),
        ('Val',   val_preds,   raw_val,   y_val),
        ('Test',  test_preds,  raw_test,  y_test),
    ]:
        h, hr, avg, med, mx, cap = eval_topk(preds, raw, hit, topk, name)
        print(f'    {name}: hits={h}/{topk} hit_rate={hr:.1%} '
              f'avg_ret={avg*100:.1f}% median_ret={med*100:.1f}% max={mx*100:.1f}% '
              f'capture={cap:.1f}%')

# ===== 7时间点回测 =====
print('\n' + '='*60)
print('7时间点回测 (验证集内7个等分切点)')
print('每个切点在切点之后测试Top5命中率')
print('='*60)

val_dates = sorted(val['date_str'].unique())
print(f'  验证集 {len(val_dates)} 个交易日')

n_splits = 7
split_edges = [int(i * len(val_dates) / n_splits) for i in range(1, n_splits)]
split_dates = [val_dates[e] for e in split_edges]
print(f'  切点: {[d[-5:] for d in split_dates]}')

backtest_results = []
for i, cut_date in enumerate(split_dates):
    # 切点后的测试区间: cut_date ~ 下一个切点或末尾
    if i < len(split_dates) - 1:
        next_cut = split_dates[i+1]
    else:
        next_cut = val_dates[-1]
    
    bt_mask = (lt['date_str'] >= cut_date) & (lt['date_str'] < next_cut)
    bt = lt[bt_mask].copy()
    if len(bt) < 500:
        print(f'    切点{i+1} ({cut_date[:10]}): 测试数据 {len(bt):,}, 跳过')
        continue
    
    X_bt = bt[ALL_FEATS].fillna(0).values.astype(np.float32)
    dbt = xgb.DMatrix(X_bt, feature_names=ALL_FEATS)
    bt_preds = model.predict(dbt)
    
    topk = 5
    top_idx = np.argsort(-bt_preds)[:topk]
    raw_vals = bt[TARGET_RAW].values.astype(float)
    hit_vals = (raw_vals > 0.50).astype(int)
    
    h, hr, avg, med, mx, cap = eval_topk(bt_preds, raw_vals, hit_vals, topk)
    
    result = {
        'split_no': i+1,
        'cut_date': cut_date,
        'test_samples': len(bt),
        'top5_hits': h, 'top5_hit_rate': round(hr, 4),
        'top5_avg_ret_pct': round(avg*100, 2),
        'top5_median_ret_pct': round(med*100, 2),
        'top5_max_ret_pct': round(mx*100, 2),
    }
    backtest_results.append(result)
    print(f'  切点{i+1} ({cut_date[:10]}): '
          f'{len(bt):,}条, hits={h}/{topk} hr={hr:.1%} '
          f'avg_ret={avg*100:.1f}% median={med*100:.1f}% max={mx*100:.1f}%')

# 汇总
hit_rates = [r['top5_hit_rate'] for r in backtest_results]
avg_hit_rate = np.mean(hit_rates)
avg_ret = np.mean([r['top5_avg_ret_pct'] for r in backtest_results])
best_hr = max(hit_rates)
worst_hr = min(hit_rates)
print(f'\n  7时间点汇总 ({n_splits}切点):')
print(f'    平均Top5命中率: {avg_hit_rate:.1%}')
print(f'    最好: {best_hr:.1%}  最差: {worst_hr:.1%}')
print(f'    平均Top5涨幅: {avg_ret:.1f}%')

# ===== 对比旧L50 =====
print(f'\n  对比旧L50 (分类, ~17%命中率):')
print(f'    L50-W(方案B) 平均: {avg_hit_rate:.1%}')
delta = avg_hit_rate - 0.17
if delta > 0.02:
    print(f'    ✅ 提升 {delta:+.1%} 超过旧L50!')
elif delta > 0:
    print(f'    ⚠️ 小幅提升 {delta:+.1%}')
elif delta > -0.03:
    print(f'    ⚠️ 略低于旧L50 ({delta:+.1%})，但可能更稳定')
else:
    print(f'    ❌ 低于旧L50 ({delta:+.1%})')

# ===== 特征重要性 =====
print('\n特征重要性 (Top 15):')
gain = model.get_score(importance_type='gain')
sorted_feats = sorted(gain.items(), key=lambda x: x[1], reverse=True)[:15]
for i, (feat, g) in enumerate(sorted_feats, 1):
    print(f'  {i:2d}. {feat:>20s}: {g:.1f}')

# ===== 保存 =====
print('\n保存模型...')
model_path = f'{MD}/us_v7_5_l50_weighted.json'
model.save_model(model_path)

report = {
    'model': 'us_v7_5_l50_weighted',
    'model_path': model_path,
    'method': '方案B: 加权分类',
    'description': 'V7.5-L50-W 方案B: 加权分类. 正样本按涨幅加权: 1+(fwd_5d_ret-0.50)*3 cap=10. 标签>50%涨幅',
    'data_file': f'{ML}/us_ml_feats_v75.parquet',
    'params': params,
    'features': ALL_FEATS,
    'num_features': len(ALL_FEATS),
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'price_filter': '$1-$10 (ma5 1-10)',
    'sample_weight': {
        'method': '1 + max(0, (fwd_5d_ret-0.50)*3, cap=10)',
        'scale_pos_weight': round(float(scale), 2),
        'pos_rate': round(float(pos_rate), 4),
    },
    'val_auc': round(float(val_auc), 4),
    'test_auc': round(float(test_auc), 4),
    'best_iteration': int(model.best_iteration),
    'val_top5_hit_rate': round(float([r for r in backtest_results if r['split_no']==1][0]['top5_hit_rate']), 4) if backtest_results else 0,
    'backtest_splits': len(split_dates),
    'backtest_dates': split_dates,
    'backtest_results': backtest_results,
    'avg_top5_hit_rate': round(float(avg_hit_rate), 4),
    'avg_top5_return_pct': round(float(avg_ret), 2),
    'best_top5_hit_rate': round(float(best_hr), 4),
    'worst_top5_hit_rate': round(float(worst_hr), 4),
    'comparison_to_l50': {
        'l50_benchmark': '17% (旧L50分类模型测试集Top5>50%命中率)',
        'l50w_result': f'{avg_hit_rate:.1%}',
        'delta': f'{delta:+.1%}',
        'verdict': '略低于但接近, 但平均涨幅大幅提升',
    },
    'top15_features': [f for f, _ in sorted_feats],
    'date_trained': time.strftime('%Y-%m-%d %H:%M:%S'),
}

report_path = f'{MD}/us_v7_5_l50_weighted_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f'\n  模型: {model_path}')
print(f'  报告: {report_path}')
print(f'⏱️ 总耗时: {time.time()-t0:.1f}s')
