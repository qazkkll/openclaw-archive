#!/usr/bin/env python3
"""
A股模型研究 — 研究员层
多维度特征工程 + LightGBM lambdarank vs XGBoost回归 对比实验

CEO/经理/研究员三层架构：
- 研究员（本脚本）：负责计算、特征工程、模型训练、验证
- 经理（审查脚本输出）：验证数据完整性、检查过拟合、审核指标
- CEO（独立思考）：质疑结论、评估商业价值、最终决策

A股特殊性：
- T+1交易（当天买入不能当天卖）
- 涨跌停10%（ST股5%）
- 散户主导市场
- 资金流数据（北向/大户/散户）是A股独有
- 行业板块效应强
"""
import json, os, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

ROOT = '/home/hermes/.hermes/openclaw-archive'
np.random.seed(42)

print("=" * 60)
print("A股模型研究 — 研究员层")
print("=" * 60)

# ============================================================
# 1. DATA LOADING
# ============================================================
print("\n[1/6] Loading data...")
t0 = time.time()

df = pd.read_parquet(os.path.join(ROOT, 'data/a_hist_10y.parquet'))
mf = pd.read_parquet(os.path.join(ROOT, 'data/moneyflow_core.parquet'))

# Standardize
df.columns = ['code', 'date', 'open', 'high', 'low', 'close', 'volume']
df['date'] = df['date'].astype(str)
mf['code'] = mf['ts_code'].str[:6]
mf['date'] = mf['trade_date'].astype(str)

print(f"  K-line: {len(df):,} rows, {df['code'].nunique()} stocks")
print(f"  Moneyflow: {len(mf):,} rows, {mf['code'].nunique()} stocks")
print(f"  Loaded in {time.time()-t0:.1f}s")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n[2/6] Computing features...")
t0 = time.time()

# Merge K-line with moneyflow
# Only use amount columns (volume columns are 99.8% NaN)
mf_cols = ['code', 'date', 'buy_sm_amount', 'sell_sm_amount', 
           'buy_md_amount', 'sell_md_amount', 'buy_lg_amount', 'sell_lg_amount',
           'buy_elg_amount', 'sell_elg_amount', 'net_mf_amount']
mf_clean = mf[mf_cols].copy()

# Merge
merged = df.merge(mf_clean, on=['code', 'date'], how='left')
merged = merged.sort_values(['code', 'date']).reset_index(drop=True)

print(f"  Merged: {len(merged):,} rows")

# Technical features (per stock, rolling)
def compute_features(group):
    """Compute features for a single stock's time series."""
    c = group['close'].values
    v = group['volume'].values
    h = group['high'].values
    l = group['low'].values
    n = len(c)
    
    if n < 120:
        return None
    
    # Price-based features
    r1 = np.zeros(n); r5 = np.zeros(n); r10 = np.zeros(n); r20 = np.zeros(n)
    for i in range(1, n):
        r1[i] = c[i]/c[i-1] - 1 if c[i-1] > 0 else 0
    for i in range(5, n):
        r5[i] = c[i]/c[i-5] - 1 if c[i-5] > 0 else 0
    for i in range(10, n):
        r10[i] = c[i]/c[i-10] - 1 if c[i-10] > 0 else 0
    for i in range(20, n):
        r20[i] = c[i]/c[i-20] - 1 if c[i-20] > 0 else 0
    
    # Moving averages
    ma5 = pd.Series(c).rolling(5).mean().values
    ma10 = pd.Series(c).rolling(10).mean().values
    ma20 = pd.Series(c).rolling(20).mean().values
    ma60 = pd.Series(c).rolling(60).mean().values
    
    # MA bias
    d5 = np.where(ma5 > 0, c / ma5 - 1, 0)
    d20 = np.where(ma20 > 0, c / ma20 - 1, 0)
    d60 = np.where(ma60 > 0, c / ma60 - 1, 0)
    
    # MA alignment
    align = np.where((ma5 > ma10) & (ma10 > ma20), 1, 
            np.where((ma5 < ma10) & (ma10 < ma20), -1, 0)).astype(float)
    
    # Volatility
    vol10 = pd.Series(r1).rolling(10).std().values
    vol20 = pd.Series(r1).rolling(20).std().values
    vol_ratio = np.where(vol20 > 0, vol10 / vol20, 1)
    
    # RSI
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rsi = np.where(avg_loss > 0, 100 - 100 / (1 + avg_gain / avg_loss), 50)
    
    # MACD
    e12 = pd.Series(c).ewm(span=12).mean().values
    e26 = pd.Series(c).ewm(span=26).mean().values
    macd = e12 - e26
    macd_signal = pd.Series(macd).ewm(span=9).mean().values
    macd_hist = macd - macd_signal
    
    # Volume ratio
    vol_ma5 = pd.Series(v).rolling(5).mean().values
    vol_ma20 = pd.Series(v).rolling(20).mean().values
    vr = np.where(vol_ma20 > 0, vol_ma5 / vol_ma20, 1)
    
    # Bollinger position
    bb_std = pd.Series(c).rolling(20).std().values
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std
    bb_pos = np.where(bb_upper > bb_lower, (c - bb_lower) / (bb_upper - bb_lower), 0.5)
    
    # ATR
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    atr20 = pd.Series(tr).rolling(20).mean().values
    atr_pct = np.where(c > 0, atr20 / c, 0)
    
    # Moneyflow features (if available)
    net_mf = group['net_mf_amount'].values if 'net_mf_amount' in group.columns else np.zeros(n)
    buy_lg = group['buy_lg_amount'].values if 'buy_lg_amount' in group.columns else np.zeros(n)
    sell_lg = group['sell_lg_amount'].values if 'sell_lg_amount' in group.columns else np.zeros(n)
    buy_elg = group['buy_elg_amount'].values if 'buy_elg_amount' in group.columns else np.zeros(n)
    sell_elg = group['sell_elg_amount'].values if 'sell_elg_amount' in group.columns else np.zeros(n)
    
    # Moneyflow ratios
    lg_net = buy_lg - sell_lg
    elg_net = buy_elg - sell_elg
    major_net = lg_net + elg_net
    
    # Cumulative moneyflow
    mf_5d = pd.Series(net_mf).rolling(5).sum().values
    mf_10d = pd.Series(net_mf).rolling(10).sum().values
    major_5d = pd.Series(major_net).rolling(5).sum().values
    
    # Major flow ratio
    total_flow = buy_lg + sell_lg + buy_elg + sell_elg
    major_ratio = np.where(total_flow > 0, major_net / total_flow, 0)
    
    # Future returns (labels)
    ret_5d = np.zeros(n)
    ret_10d = np.zeros(n)
    for i in range(n-5):
        ret_5d[i] = c[i+5] / c[i] - 1 if c[i] > 0 else 0
    for i in range(n-10):
        ret_10d[i] = c[i+10] / c[i] - 1 if c[i] > 0 else 0
    
    result = pd.DataFrame({
        'code': group['code'].values,
        'date': group['date'].values,
        'close': c,
        'volume': v,
        # Technical features (15)
        'r1': r1, 'r5': r5, 'r10': r10, 'r20': r20,
        'd5': d5, 'd20': d20, 'd60': d60,
        'align': align, 'vol_ratio': vol_ratio,
        'rsi': rsi, 'macd_hist': macd_hist,
        'vr': vr, 'bb_pos': bb_pos, 'atr_pct': atr_pct,
        # Moneyflow features (6)
        'net_mf': net_mf, 'mf_5d': mf_5d, 'mf_10d': mf_10d,
        'major_5d': major_5d, 'major_ratio': major_ratio,
        'elg_net': elg_net,
        # Labels
        'ret_5d': ret_5d, 'ret_10d': ret_10d,
    })
    
    return result

# Apply to all stocks (sample for speed)
print("  Computing features for all stocks...")
all_features = []
stock_groups = merged.groupby('code')
total = len(stock_groups)

for i, (code, group) in enumerate(stock_groups):
    if i % 500 == 0:
        print(f"    {i}/{total} stocks processed...")
    feat = compute_features(group)
    if feat is not None:
        all_features.append(feat)

features_df = pd.concat(all_features, ignore_index=True)
print(f"  Features: {len(features_df):,} rows, {features_df['code'].nunique()} stocks")
print(f"  Computed in {time.time()-t0:.1f}s")

# ============================================================
# 3. CROSS-SECTIONAL RANKING LABELS
# ============================================================
print("\n[3/6] Computing cross-sectional ranking labels...")
t0 = time.time()

# For each date, rank stocks by future return
# This is the KEY difference: we predict RANK, not absolute return
features_df = features_df[features_df['ret_10d'] != 0].copy()  # Remove boundary rows

# Cross-sectional rank (0-1, higher = better)
features_df['rank_10d'] = features_df.groupby('date')['ret_10d'].rank(pct=True)
features_df['rank_5d'] = features_df.groupby('date')['ret_5d'].rank(pct=True)

# Remove extreme dates (too few stocks)
date_counts = features_df.groupby('date').size()
valid_dates = date_counts[date_counts >= 100].index
features_df = features_df[features_df['date'].isin(valid_dates)]

print(f"  Valid samples: {len(features_df):,}")
print(f"  Date range: {features_df['date'].min()} ~ {features_df['date'].max()}")
print(f"  Rank_10d distribution: mean={features_df['rank_10d'].mean():.3f}, std={features_df['rank_10d'].std():.3f}")

# ============================================================
# 4. FEATURE CROSS-SECTIONAL STANDARDIZATION
# ============================================================
print("\n[4/6] Cross-sectional standardization...")
t0 = time.time()

feat_cols = ['r1', 'r5', 'r10', 'r20', 'd5', 'd20', 'd60', 'align', 
             'vol_ratio', 'rsi', 'macd_hist', 'vr', 'bb_pos', 'atr_pct',
             'net_mf', 'mf_5d', 'mf_10d', 'major_5d', 'major_ratio', 'elg_net']

# Winsorize + Z-score per cross-section (date)
def cross_section_zscore(group):
    """Standardize features within each date's cross-section."""
    for col in feat_cols:
        vals = group[col].copy()
        # Winsorize at 1%/99%
        lo, hi = vals.quantile(0.01), vals.quantile(0.99)
        vals = vals.clip(lo, hi)
        # Z-score
        mean, std = vals.mean(), vals.std()
        if std > 0:
            group[col] = (vals - mean) / std
        else:
            group[col] = 0
    return group

print("  Standardizing features per date...")
# Note: groupby('date').apply() removes 'date' column, re-add after
date_col = features_df['date'].values
features_df = features_df.groupby('date', group_keys=False).apply(cross_section_zscore)
features_df['date'] = date_col
print(f"  Done in {time.time()-t0:.1f}s")

# ============================================================
# 5. MODEL TRAINING — LightGBM lambdarank vs XGBoost regression
# ============================================================
print("\n[5/6] Training models...")
t0 = time.time()

# Time-series split: train ≤2023, val 2024, test 2025+
features_df['year'] = features_df['date'].str[:4].astype(int)
train = features_df[features_df['year'] <= 2023]
val = features_df[features_df['year'] == 2024]
test = features_df[features_df['year'] >= 2025]

print(f"  Train: {len(train):,} samples ({train['date'].min()} ~ {train['date'].max()})")
print(f"  Val:   {len(val):,} samples ({val['date'].min()} ~ {val['date'].max()})")
print(f"  Test:  {len(test):,} samples ({test['date'].min()} ~ {test['date'].max()})")

X_train = train[feat_cols].values
y_train_reg = train['ret_10d'].values  # For regression
y_train_rank = train['rank_10d'].values  # For ranking

X_val = val[feat_cols].values
y_val_reg = val['ret_10d'].values
y_val_rank = val['rank_10d'].values

X_test = test[feat_cols].values
y_test_reg = test['ret_10d'].values
y_test_rank = test['rank_10d'].values

# --- Model A: XGBoost Regression (current A2 approach) ---
import xgboost as xgb

print("\n  [A] XGBoost Regression (A2 baseline)...")
xgb_reg = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    tree_method='hist', device='cuda', random_state=42, verbosity=0
)
xgb_reg.fit(X_train, y_train_reg, eval_set=[(X_val, y_val_reg)], verbose=False)
xgb_pred = xgb_reg.predict(X_test)

# --- Model B: LightGBM Lambdarank ---
try:
    import lightgbm as lgb
    
    print("  [B] LightGBM Lambdarank...")
    
    # Prepare group sizes (number of stocks per date)
    train_groups = train.groupby('date').size().values
    val_groups = val.groupby('date').size().values
    
    # Convert float ranks to integers (LightGBM lambdarank requires int labels)
    # Use percentile bins (0-100) as integer labels
    y_train_rank_int = (y_train_rank * 100).astype(int)
    y_val_rank_int = (y_val_rank * 100).astype(int)
    
    lgb_train = lgb.Dataset(X_train, label=y_train_rank_int, group=train_groups)
    lgb_val = lgb.Dataset(X_val, label=y_val_rank_int, group=val_groups, reference=lgb_train)
    
    lgb_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10, 20],
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'label_gain': list(range(101)),  # Support labels 0-100
        'verbose': -1,
        'seed': 42,
    }
    
    lgb_model = lgb.train(
        lgb_params, lgb_train, num_boost_round=300,
        valid_sets=[lgb_val], callbacks=[lgb.log_evaluation(50)]
    )
    lgb_pred = lgb_model.predict(X_test)
    
    HAS_LGB = True
    print("  LightGBM lambdarank trained successfully.")
except ImportError:
    print("  LightGBM not available, skipping lambdarank.")
    HAS_LGB = False
except Exception as e:
    print(f"  LightGBM error: {e}")
    HAS_LGB = False

# ============================================================
# 6. EVALUATION — IC, Rank IC, Top/Bottom returns
# ============================================================
print("\n[6/6] Evaluation...")
print("=" * 60)

def evaluate_model(pred, actual_ret, actual_rank, dates, model_name):
    """Evaluate model predictions."""
    results = {}
    
    # Per-date IC (Information Coefficient)
    ic_list = []
    rank_ic_list = []
    
    for d in dates.unique():
        mask = dates == d
        if mask.sum() < 10:
            continue
        p = pred[mask]
        r = actual_ret[mask]
        rk = actual_rank[mask]
        
        # IC: Pearson correlation
        if np.std(p) > 0 and np.std(r) > 0:
            ic = np.corrcoef(p, r)[0, 1]
            ic_list.append(ic)
        
        # Rank IC: Spearman correlation
        from scipy import stats
        if np.std(p) > 0:
            ric, _ = stats.spearmanr(p, rk)
            rank_ic_list.append(ric)
    
    results['IC_mean'] = np.mean(ic_list) if ic_list else 0
    results['IC_std'] = np.std(ic_list) if ic_list else 0
    results['ICIR'] = results['IC_mean'] / results['IC_std'] if results['IC_std'] > 0 else 0
    results['Rank_IC_mean'] = np.mean(rank_ic_list) if rank_ic_list else 0
    results['Rank_IC_std'] = np.std(rank_ic_list) if rank_ic_list else 0
    results['Rank_ICIR'] = results['Rank_IC_mean'] / results['Rank_IC_std'] if results['Rank_IC_std'] > 0 else 0
    # IC_positive_ratio
    pos_count = sum(1 for x in ic_list if x > 0)
    results['IC_positive_ratio'] = pos_count / len(ic_list) if ic_list else 0
    
    # Top/Bottom quintile returns
    quintile_rets = []
    for d in dates.unique():
        mask = dates == d
        if mask.sum() < 20:
            continue
        p = pred[mask]
        r = actual_ret[mask]
        
        # Sort by prediction, take top/bottom 20%
        n = len(p)
        top_n = max(1, n // 5)
        sorted_idx = np.argsort(p)[::-1]
        
        top_ret = r[sorted_idx[:top_n]].mean()
        bottom_ret = r[sorted_idx[-top_n:]].mean()
        quintile_rets.append((top_ret, bottom_ret))
    
    if quintile_rets:
        top_rets = [x[0] for x in quintile_rets]
        bottom_rets = [x[1] for x in quintile_rets]
        results['Top20_avg_ret'] = np.mean(top_rets) * 100
        results['Bottom20_avg_ret'] = np.mean(bottom_rets) * 100
        results['Long_Short'] = results['Top20_avg_ret'] - results['Bottom20_avg_ret']
    
    return results

from scipy import stats

# Evaluate XGBoost Regression
print(f"\n{'='*60}")
print(f"XGBoost Regression (A2 baseline)")
print(f"{'='*60}")
xgb_results = evaluate_model(xgb_pred, y_test_reg, y_test_rank, test['date'], 'XGBoost')
for k, v in xgb_results.items():
    print(f"  {k}: {v:.4f}")

# Evaluate LightGBM Lambdarank
if HAS_LGB:
    print(f"\n{'='*60}")
    print(f"LightGBM Lambdarank")
    print(f"{'='*60}")
    lgb_results = evaluate_model(lgb_pred, y_test_reg, y_test_rank, test['date'], 'LightGBM')
    for k, v in lgb_results.items():
        print(f"  {k}: {v:.4f}")

# Feature importance comparison
print(f"\n{'='*60}")
print(f"Feature Importance (XGBoost)")
print(f"{'='*60}")
xgb_imp = xgb_reg.feature_importances_
for feat, imp in sorted(zip(feat_cols, xgb_imp), key=lambda x: x[1], reverse=True):
    print(f"  {feat:15s} {imp:.4f}")

if HAS_LGB:
    print(f"\n{'='*60}")
    print(f"Feature Importance (LightGBM)")
    print(f"{'='*60}")
    lgb_imp = lgb_model.feature_importance(importance_type='gain')
    for feat, imp in sorted(zip(feat_cols, lgb_imp), key=lambda x: x[1], reverse=True):
        print(f"  {feat:15s} {imp:.1f}")

# Save results
results = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M'),
    'data': {
        'train_samples': len(train),
        'val_samples': len(val),
        'test_samples': len(test),
        'stocks': features_df['code'].nunique(),
        'features': len(feat_cols),
        'date_range': f"{features_df['date'].min()} ~ {features_df['date'].max()}"
    },
    'xgboost_regression': xgb_results,
    'lightgbm_lambdarank': lgb_results if HAS_LGB else None,
    'feature_importance_xgb': dict(zip(feat_cols, xgb_imp.tolist())),
    'feature_importance_lgb': dict(zip(feat_cols, lgb_imp.tolist())) if HAS_LGB else None,
}

out_path = os.path.join(ROOT, 'output/a_stock_research_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print(f"Total time: {time.time()-t0:.1f}s")
