#!/usr/bin/env python3
"""
绿箭V12 LightGBM 重训练 + IC/ICIR验证
Walk-Forward 5折 + OOS 2024-2026
6 configs: baseline, with_flow, 3d_hold, 10d_hold, hp_tuned, classifier
"""
import json, os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings('ignore')
ROOT = '/home/hermes/.hermes/openclaw-archive'
DATA_DIR = os.path.join(ROOT, 'data', 'us')
MODEL_DIR = os.path.join(ROOT, 'models', 'us')
EXP_DIR = os.path.join(ROOT, 'data', 'experiments')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

def compute_tech_features(g):
    c = g['close']
    g['ma5'] = c.rolling(5).mean()
    g['ma20'] = c.rolling(20).mean()
    g['ma60'] = c.rolling(60).mean()
    g['ma_bias20'] = (c - g['ma20']) / g['ma20']
    g['ma_align'] = ((c > g['ma5']).astype(int) + (g['ma5'] > g['ma20']).astype(int))
    mn60 = c.rolling(60).min()
    mx60 = c.rolling(60).max()
    g['price_position'] = (c - mn60) / (mx60 - mn60 + 1e-10)
    g['ret1'] = c.pct_change(1)
    g['ret5'] = c.pct_change(5)
    g['ret20'] = c.pct_change(20)
    g['ret60'] = c.pct_change(60)
    g['momentum_6m'] = c.pct_change(126)
    g['momentum_1m'] = c.pct_change(21)
    g['mom_divergence'] = g['momentum_1m'] - g['ret20']
    g['trend_accel'] = g['ret5'] - g['ret5'].shift(5)
    dr = c.pct_change(1)
    g['vol20'] = dr.rolling(20).std()
    g['vol5'] = dr.rolling(5).std()
    g['vol_ratio'] = g['volume'] / g['volume'].rolling(20).mean()
    g['vol_change'] = g['vol20'] / g['vol20'].shift(20)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    g['rsi14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    g['rsi_change'] = g['rsi14'].diff(5)
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    g['macd'] = ema12 - ema26
    g['macd_signal'] = g['macd'].ewm(span=9).mean()
    g['macd_hist'] = g['macd'] - g['macd_signal']
    g['bb_std'] = dr.rolling(20).std()
    bb_mid = c.rolling(20).mean()
    g['bb_width'] = 4 * g['bb_std'] * bb_mid / (bb_mid + 1e-10)
    g['bb_pos'] = (c - (bb_mid - 2 * c.rolling(20).std())) / (4 * c.rolling(20).std() + 1e-10)
    ret_pos = dr.clip(lower=0).rolling(20).mean()
    ret_neg = (-dr).clip(lower=0).rolling(20).mean()
    g['ret_quality'] = ret_pos / (ret_pos + ret_neg + 1e-10)
    g['price'] = c
    g['range_pct'] = (g['high'] - g['low']) / (c + 1e-10)
    return g

def compute_flow_features(g):
    """CMF-20 and OBV slope-20 (fast vectorized)"""
    h, l, c, v = g['high'], g['low'], g['close'], g['volume']
    mf_vol = v * (2*c - l - h) / (h - l + 1e-10)
    pos_mf = mf_vol.clip(lower=0)
    neg_mf = (-mf_vol).clip(lower=0)
    g['cmf_20'] = (pos_mf.rolling(20).sum() - neg_mf.rolling(20).sum()) / (v.rolling(20).sum() + 1e-10)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    # Fast rolling slope via numpy stride tricks
    obv_arr = obv.values.astype(np.float64)
    n = len(obv_arr)
    W = 20
    if n >= W:
        # Use sliding_window_view for vectorized computation
        try:
            windows = np.lib.stride_tricks.sliding_window_view(obv_arr, W)
        except AttributeError:
            # Fallback for older numpy
            shape = (n - W + 1, W)
            strides = (obv_arr.strides[0], obv_arr.strides[0])
            windows = np.lib.stride_tricks.as_strided(obv_arr, shape=shape, strides=strides)
        x = np.arange(W, dtype=np.float64)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        y_means = windows.mean(axis=1)
        cov = ((windows - y_means[:, None]) * (x - x_mean)).sum(axis=1)
        slope_full = np.zeros(n)
        slope_full[W-1:] = cov / x_var
        vol_ma20 = v.rolling(20).mean().values + 1e-10
        g['obv_slope_20'] = slope_full / vol_ma20
    else:
        g['obv_slope_20'] = 0.0
    return g

# Feature sets
BASE_FEATS = [
    'ma5', 'ma20', 'ma60', 'ma_bias20', 'ma_align', 'price_position',
    'ret1', 'ret5', 'ret20', 'ret60', 'momentum_6m', 'momentum_1m',
    'mom_divergence', 'trend_accel', 'vol20', 'vol5', 'vol_ratio', 'vol_change',
    'rsi14', 'rsi_change', 'macd', 'macd_signal', 'macd_hist',
    'bb_std', 'bb_width', 'bb_pos', 'ret_quality', 'price', 'range_pct',
    'vix_close', 'spy_ret1', 'spy_ret5', 'spy_ret20', 'spy_ret60'
]
FLOW_FEATS = ['cmf_20', 'obv_slope_20']
ALL_FEATS = BASE_FEATS + FLOW_FEATS  # 36 features

log('=' * 60)
log('绿箭V12 LightGBM 重训练 (Walk-Forward + IC/ICIR)')
log('=' * 60)

# ── Step 1: Load data ──
log('Step 1: 加载数据...')
t0 = time.time()
df = pd.read_parquet(os.path.join(DATA_DIR, 'us_hist_full_10y.parquet'))
df['date'] = pd.to_datetime(df['date'])
df = df.dropna(subset=['close', 'volume'])
df = df[(df['close'] > 0.5) & (df['volume'] > 0)]

# VIX
vix_df = pd.read_parquet(os.path.join(DATA_DIR, 'vix_10y.parquet'))
if isinstance(vix_df.columns, pd.MultiIndex):
    vix_df.columns = [c[0] for c in vix_df.columns]
vix_df = vix_df.reset_index()
vix_df.columns = [c.lower().replace('ticker', '') for c in vix_df.columns]
if 'date' not in vix_df.columns:
    vix_df = vix_df.rename(columns={vix_df.columns[0]: 'date'})
vix_df['date'] = pd.to_datetime(vix_df['date'])
vix_close_col = [c for c in vix_df.columns if 'close' in c.lower()]
if vix_close_col:
    vix_df = vix_df.rename(columns={vix_close_col[0]: 'vix_close'})
vix_df = vix_df[['date', 'vix_close']].dropna()

# SPY only (task spec only has spy_*)
spy = df[df['sym'] == 'SPY'][['date', 'close']].copy()
macro = pd.DataFrame({'date': spy['date']})
macro['spy_ret1'] = spy['close'].pct_change(1)
macro['spy_ret5'] = spy['close'].pct_change(5)
macro['spy_ret20'] = spy['close'].pct_change(20)
macro['spy_ret60'] = spy['close'].pct_change(60)
macro = macro.merge(vix_df, on='date', how='left')
macro['vix_close'] = macro['vix_close'].ffill().fillna(20)
log(f'  宏观: {len(macro)}天, SPY/VIX')

# Filter $1-$10
last_prices = df.groupby('sym')['close'].last()
valid_syms = last_prices[(last_prices >= 1) & (last_prices <= 10)].index
df = df[df['sym'].isin(valid_syms)]
log(f'  绿箭: {df["sym"].nunique()}只, {len(df):,}行')

# ── Step 2: Sample if needed ──
n_syms = df['sym'].nunique()
if n_syms > 3000:
    log(f'  采样 3000/{n_syms} 只股票...')
    sample_syms = np.random.RandomState(42).choice(df['sym'].unique(), size=3000, replace=False)
    df = df[df['sym'].isin(sample_syms)]
    log(f'  采样后: {df["sym"].nunique()}只')

# ── Step 3: Technical features ──
log('Step 2: 计算技术特征...')
t0 = time.time()
df = df.sort_values(['sym', 'date'])
groups = []
for sym, group in df.groupby('sym'):
    if len(group) < 80:
        continue
    g = group.copy()
    g = compute_tech_features(g)
    g = compute_flow_features(g)
    groups.append(g)
df = pd.concat(groups, ignore_index=True)
log(f'  完成: {time.time()-t0:.1f}s, {len(df):,}行')

# ── Step 4: Merge macro ──
df = df.merge(macro, on='date', how='left')
for col in ALL_FEATS:
    if col not in df.columns:
        df[col] = 0
    df[col] = df[col].ffill().fillna(0)

# ── Step 5: Labels ──
log('Step 3: 创建标签...')
def calc_fwd_return_inplace(dfg, hold=5):
    """Compute fwd_ret for each group in-place. Groups must be sorted by date."""
    result = []
    for sym, g in dfg.groupby('sym'):
        g = g.sort_values('date').copy()
        g['fwd_ret'] = g['close'].shift(-hold) / g['close'] - 1
        result.append(g)
    return pd.concat(result, ignore_index=True)

# Exclude ETFs
etf_syms = {'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'IVV', 'VEA', 'VWO',
            'BND', 'AGG', 'TLT', 'GLD', 'SLV', 'USO', 'XLE', 'XLF', 'XLK', 'XLV'}
df = df[~df['sym'].isin(etf_syms)]

# ── Helper: Walk-forward + OOS evaluation ──
def run_experiment(name, feat_cols, hold_days, is_classifier=False, lgb_params=None, extra_name=''):
    log(f'\n{"="*50}')
    log(f'实验: {name} (hold={hold_days}d, feats={len(feat_cols)}, {"classifier" if is_classifier else "regression"})')
    log(f'{"="*50}')

    # Create labels
    df_exp = df.copy()
    df_exp = calc_fwd_return_inplace(df_exp, hold=hold_days)
    df_exp = df_exp.reset_index(drop=True)
    df_exp = df_exp.dropna(subset=['fwd_ret'])

    if is_classifier:
        df_exp['label'] = (df_exp['fwd_ret'] > 0.02).astype(int)
        label_col = 'label'
        obj = 'binary'
        metric = 'auc'
    else:
        label_col = 'fwd_ret'
        obj = 'regression'
        metric = 'l2'

    # Feature completeness filter
    feat_present = df_exp[feat_cols].notna().sum(axis=1)
    df_exp = df_exp[feat_present >= len(feat_cols) * 0.8]
    for col in feat_cols:
        if df_exp[col].isna().any():
            df_exp[col] = df_exp[col].fillna(df_exp[col].median())

    log(f'  数据: {len(df_exp):,}行, {df_exp["sym"].nunique()}只, {len(feat_cols)}特征')

    # Walk-forward
    df_exp = df_exp.sort_values('date')
    dates = np.sort(df_exp['date'].unique())
    oos_start = pd.Timestamp('2024-01-01')
    train_dates = dates[dates < oos_start]
    oos_dates = dates[dates >= oos_start]
    log(f'  训练: {pd.Timestamp(train_dates[0]).date()} ~ {pd.Timestamp(train_dates[-1]).date()} ({len(train_dates)}天)')
    log(f'  OOS: {pd.Timestamp(oos_dates[0]).date()} ~ {pd.Timestamp(oos_dates[-1]).date()} ({len(oos_dates)}天)')

    default_lgb_params = {
        'objective': obj,
        'num_leaves': 63,
        'learning_rate': 0.03,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1
    }
    if lgb_params:
        default_lgb_params.update(lgb_params)

    n_folds = 5
    fold_size = len(train_dates) // n_folds
    wf_results = []

    for fold in range(n_folds):
        train_end_idx = (fold + 1) * fold_size
        val_start_idx = train_end_idx
        val_end_idx = min(val_start_idx + fold_size, len(train_dates))
        if val_end_idx <= val_start_idx:
            continue

        train_mask = df_exp['date'].isin(train_dates[:train_end_idx])
        val_mask = df_exp['date'].isin(train_dates[val_start_idx:val_end_idx])

        X_train = np.nan_to_num(df_exp.loc[train_mask, feat_cols].values, nan=0, posinf=0, neginf=0).astype(np.float32)
        y_train = df_exp.loc[train_mask, label_col].values
        X_val = np.nan_to_num(df_exp.loc[val_mask, feat_cols].values, nan=0, posinf=0, neginf=0).astype(np.float32)
        y_val = df_exp.loc[val_mask, label_col].values

        if len(X_train) < 1000 or len(X_val) < 100:
            continue

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feat_cols, reference=train_data)

        callbacks = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0)
        ]

        model = lgb.train(
            default_lgb_params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=callbacks
        )
        pred = model.predict(X_val)

        val_df = df_exp.loc[val_mask].copy()
        val_df['pred'] = pred

        # IC (use pred for regression; for classifier, use pred as-is since it's probability)
        ic_values = []
        for d, group in val_df.groupby('date'):
            if len(group) < 20:
                continue
            ic, _ = stats.spearmanr(group['pred'], group['fwd_ret'])
            if not np.isnan(ic):
                ic_values.append(ic)

        if ic_values:
            ic_mean = np.mean(ic_values)
            icir = ic_mean / (np.std(ic_values) + 1e-10)
            ic_pos = np.mean([x > 0 for x in ic_values])

            spreads = []
            for d, group in val_df.groupby('date'):
                if len(group) < 20:
                    continue
                n5 = max(1, int(len(group) * 0.05))
                n20 = max(1, int(len(group) * 0.20))
                top5 = group.nlargest(n5, 'pred')['fwd_ret'].mean()
                bot20 = group.nsmallest(n20, 'pred')['fwd_ret'].mean()
                spreads.append(top5 - bot20)

            best_iter = model.best_iteration if hasattr(model, 'best_iteration') else default_lgb_params['n_estimators']

            wf_results.append({
                'fold': fold, 'ic': round(ic_mean, 4), 'icir': round(icir, 3),
                'ic_pos': round(ic_pos, 3), 'spread': round(np.mean(spreads) * 100, 2),
                'n_days': len(ic_values), 'best_iter': best_iter
            })
            log(f'  Fold {fold}: IC={ic_mean:.4f} ICIR={icir:.3f} IC>0={ic_pos*100:.0f}% spread={np.mean(spreads)*100:+.2f}%')

    if not wf_results:
        log(f'  WARNING: No valid folds for {name}')
        return None

    avg_ic = np.mean([r['ic'] for r in wf_results])
    avg_icir = np.mean([r['icir'] for r in wf_results])
    avg_ic_pos = np.mean([r['ic_pos'] for r in wf_results])
    avg_spread = np.mean([r['spread'] for r in wf_results])
    best_iter = max(int(np.median([r['best_iter'] for r in wf_results])), 50)
    log(f'  WF汇总: IC={avg_ic:.4f} ICIR={avg_icir:.3f} IC>0={avg_ic_pos*100:.0f}% spread={avg_spread:+.2f}% best_iter={best_iter}')

    # ── OOS evaluation ──
    train_mask = df_exp['date'] < oos_start
    oos_mask = df_exp['date'] >= oos_start
    X_train = np.nan_to_num(df_exp.loc[train_mask, feat_cols].values, nan=0, posinf=0, neginf=0).astype(np.float32)
    y_train = df_exp.loc[train_mask, label_col].values
    X_oos = np.nan_to_num(df_exp.loc[oos_mask, feat_cols].values, nan=0, posinf=0, neginf=0).astype(np.float32)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols)
    final_params = default_lgb_params.copy()
    final_params['n_estimators'] = best_iter
    final_model = lgb.train(
        final_params,
        train_data,
        num_boost_round=best_iter,
        callbacks=[lgb.log_evaluation(0)]
    )
    pred = final_model.predict(X_oos)

    oos_df = df_exp.loc[oos_mask].copy()
    oos_df['pred'] = pred

    oos_ic_values = []
    oos_spreads = []
    oos_top5_rets = []
    for d, group in oos_df.groupby('date'):
        if len(group) < 20:
            continue
        ic, _ = stats.spearmanr(group['pred'], group['fwd_ret'])
        if not np.isnan(ic):
            oos_ic_values.append(ic)
        n5 = max(1, int(len(group) * 0.05))
        n20 = max(1, int(len(group) * 0.20))
        top5 = group.nlargest(n5, 'pred')['fwd_ret'].mean()
        bot20 = group.nsmallest(n20, 'pred')['fwd_ret'].mean()
        oos_spreads.append(top5 - bot20)
        oos_top5_rets.append(top5)

    oos_ic = np.mean(oos_ic_values)
    oos_icir = oos_ic / (np.std(oos_ic_values) + 1e-10)
    oos_ic_pos = np.mean([x > 0 for x in oos_ic_values])

    log(f'  OOS: IC={oos_ic:.4f} ICIR={oos_icir:.3f} IC>0={oos_ic_pos*100:.0f}%')
    log(f'  Top5%={np.mean(oos_top5_rets)*100:+.2f}% spread={np.mean(oos_spreads)*100:+.2f}%')
    if oos_top5_rets:
        log(f'  Top5 win={np.mean([r > 0 for r in oos_top5_rets])*100:.1f}%')

    # Feature importance
    importance = final_model.feature_importance(importance_type='gain')
    total_imp = importance.sum()
    feat_imp = {feat_cols[i]: round(importance[i] / total_imp * 100, 2)
                for i in np.argsort(-importance)}
    top3 = list(feat_imp.keys())[:3]
    log(f'  Top3 features: {top3}')

    # Save model
    model_path = os.path.join(MODEL_DIR, f'arrow_v12_{name}.txt')
    final_model.save_model(model_path)
    log(f'  模型保存: {model_path}')

    result = {
        'experiment': name,
        'config': {
            'price_range': 'arrow',
            'hold_days': hold_days,
            'n_features': len(feat_cols),
            'features': feat_cols,
            'objective': obj,
            'is_classifier': is_classifier
        },
        'n_samples': len(df_exp),
        'wf': wf_results,
        'wf_summary': {
            'ic': round(avg_ic, 4), 'icir': round(avg_icir, 3),
            'ic_pos': round(avg_ic_pos, 3), 'spread': round(avg_spread, 2)
        },
        'oos': {
            'ic': round(oos_ic, 4), 'icir': round(oos_icir, 3),
            'ic_pos': round(oos_ic_pos, 3),
            'spread': round(np.mean(oos_spreads) * 100, 2),
            'top5_avg': round(np.mean(oos_top5_rets) * 100, 2),
            'top5_win': round(np.mean([r > 0 for r in oos_top5_rets]) * 100, 1) if oos_top5_rets else 0,
            'n_days': len(oos_ic_values)
        },
        'best_iter': best_iter,
        'params': {k: v for k, v in default_lgb_params.items() if k != 'verbose'},
        'feature_importance': feat_imp,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    return result

# ═══════════════════════════════════════════════════
# Experiment 1: Baseline (no flow, 5d hold, regression)
# ═══════════════════════════════════════════════════
r1 = run_experiment('lgb_v12_baseline_5d', BASE_FEATS, hold_days=5, is_classifier=False)

# ═══════════════════════════════════════════════════
# Experiment 2: With Flow (5d hold, regression)
# ═══════════════════════════════════════════════════
r2 = run_experiment('lgb_v12_with_flow_5d', ALL_FEATS, hold_days=5, is_classifier=False)

# ═══════════════════════════════════════════════════
# Experiment 3: 3-day hold with flow
# ═══════════════════════════════════════════════════
r3 = run_experiment('lgb_v12_3d_hold', ALL_FEATS, hold_days=3, is_classifier=False)

# ═══════════════════════════════════════════════════
# Experiment 4: 10-day hold with flow
# ═══════════════════════════════════════════════════
r4 = run_experiment('lgb_v12_10d_hold', ALL_FEATS, hold_days=10, is_classifier=False)

# ═══════════════════════════════════════════════════
# Experiment 5: Hyperparameter tuned (grid search)
# ═══════════════════════════════════════════════════
log('\n' + '=' * 60)
log('实验5: 超参数搜索...')
log('=' * 60)

# Quick grid search on a subset for speed
hp_grid = {
    'num_leaves': [31, 63, 127],
    'learning_rate': [0.01, 0.03],
    'n_estimators': [200, 500]
}
hp_combos = list(itertools.product(*hp_grid.values()))
log(f'  搜索空间: {len(hp_combos)} 组合')

# Use a random subset for HP search (faster)
np.random.seed(42)
hp_sample = df.copy()
hp_sample = calc_fwd_return_inplace(hp_sample, hold=5)
hp_sample = hp_sample.reset_index(drop=True)
hp_sample = hp_sample.dropna(subset=['fwd_ret'])
hp_sample = hp_sample[~hp_sample['sym'].isin(etf_syms)]

# Use 20% of dates for HP search
hp_dates = np.sort(hp_sample['date'].unique())
hp_oos_start = pd.Timestamp('2024-01-01')
hp_train_dates = hp_dates[hp_dates < hp_oos_start]
hp_val_dates = hp_train_dates[int(len(hp_train_dates)*0.85):]
hp_train_sub = hp_train_dates[:int(len(hp_train_dates)*0.85)]

# Feature completeness
feat_present = hp_sample[ALL_FEATS].notna().sum(axis=1)
hp_sample = hp_sample[feat_present >= len(ALL_FEATS) * 0.8]
for col in ALL_FEATS:
    if hp_sample[col].isna().any():
        hp_sample[col] = hp_sample[col].fillna(hp_sample[col].median())

best_hp_score = -999
best_hp_params = None

for combo in hp_combos:
    nl, lr, ne = combo
    params = {
        'objective': 'regression',
        'num_leaves': nl,
        'learning_rate': lr,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1
    }

    train_mask = hp_sample['date'].isin(hp_train_sub)
    val_mask = hp_sample['date'].isin(hp_val_dates)

    X_tr = np.nan_to_num(hp_sample.loc[train_mask, ALL_FEATS].values, nan=0, posinf=0, neginf=0).astype(np.float32)
    y_tr = hp_sample.loc[train_mask, 'fwd_ret'].values
    X_vl = np.nan_to_num(hp_sample.loc[val_mask, ALL_FEATS].values, nan=0, posinf=0, neginf=0).astype(np.float32)
    y_vl = hp_sample.loc[val_mask, 'fwd_ret'].values

    if len(X_tr) < 1000 or len(X_vl) < 100:
        continue

    tr_data = lgb.Dataset(X_tr, label=y_tr, feature_name=ALL_FEATS)
    vl_data = lgb.Dataset(X_vl, label=y_vl, feature_name=ALL_FEATS, reference=tr_data)

    params['n_estimators'] = ne
    m = lgb.train(params, tr_data, num_boost_round=ne, valid_sets=[vl_data],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    pred = m.predict(X_vl)

    val_df = hp_sample.loc[val_mask].copy()
    val_df['pred'] = pred
    ics = []
    for d, grp in val_df.groupby('date'):
        if len(grp) < 20:
            continue
        ic, _ = stats.spearmanr(grp['pred'], grp['fwd_ret'])
        if not np.isnan(ic):
            ics.append(ic)
    if ics:
        score = np.mean(ics) / (np.std(ics) + 1e-10)
        if score > best_hp_score:
            best_hp_score = score
            best_hp_params = {'num_leaves': nl, 'learning_rate': lr, 'n_estimators': ne}
    log(f'  HP: leaves={nl} lr={lr} n_est={ne} -> ICIR={score:.3f}')

log(f'  最佳: {best_hp_params} ICIR={best_hp_score:.3f}')

r5 = run_experiment('lgb_v12_hp_tuned', ALL_FEATS, hold_days=5, is_classifier=False,
                     lgb_params=best_hp_params)

# ═══════════════════════════════════════════════════
# Experiment 6: Classifier (fwd_ret > 2%)
# ═══════════════════════════════════════════════════
r6 = run_experiment('lgb_v12_classifier', ALL_FEATS, hold_days=5, is_classifier=True)

# ═══════════════════════════════════════════════════
# Save all results
# ═══════════════════════════════════════════════════
log('\n' + '=' * 60)
log('汇总结果')
log('=' * 60)

all_results = {}
for r in [r1, r2, r3, r4, r5, r6]:
    if r is not None:
        all_results[r['experiment']] = r
        name = r['experiment']
        wf_icir = r['wf_summary']['icir']
        oos_icir = r['oos']['icir']
        oos_ic = r['oos']['ic']
        log(f'  {name}: WF_ICIR={wf_icir:.3f} OOS_ICIR={oos_icir:.3f} OOS_IC={oos_ic:.4f}')

# Save experiment results
results_path = os.path.join(EXP_DIR, 'lgb_v12_all_results.json')
with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
log(f'\n实验结果: {results_path}')

# Find best config by OOS ICIR
best_name = max(all_results.keys(), key=lambda k: all_results[k]['oos']['icir'])
best = all_results[best_name]
log(f'\n🏆 最佳: {best_name} (OOS ICIR={best["oos"]["icir"]:.3f})')

# Save best model meta
best_model_path = os.path.join(MODEL_DIR, 'arrow_v12_lgb.txt')
meta_path = os.path.join(MODEL_DIR, 'arrow_v12_lgb_meta.json')

meta = {
    'version': 'arrow_v12_lgb',
    'algorithm': 'LightGBM',
    'best_config': best_name,
    'features': best['config']['features'],
    'n_features': best['config']['n_features'],
    'hold_days': best['config']['hold_days'],
    'is_classifier': best['config']['is_classifier'],
    'top_n': 5,
    'universe': f'$1-$10',
    'params': best['params'],
    'n_trees': best['best_iter'],
    'trained_on': str(df['date'].min().date()) + ' ~ ' + str(df['date'].max().date()),
    'train_end': str(df['date'].max().date()),
    'n_train_samples': best['n_samples'],
    'feature_importance': best.get('feature_importance', {}),
    'validation': {
        'method': 'Walk-Forward 5折 + OOS 2024-2026',
        'wf_ic': best['wf_summary']['ic'],
        'wf_icir': best['wf_summary']['icir'],
        'wf_ic_pos': best['wf_summary']['ic_pos'],
        'wf_spread': best['wf_summary']['spread'],
        'oos_ic': best['oos']['ic'],
        'oos_icir': best['oos']['icir'],
        'oos_ic_pos': best['oos']['ic_pos'],
        'oos_top5_avg': best['oos']['top5_avg'],
        'oos_spread': best['oos']['spread'],
        'oos_n_days': best['oos']['n_days'],
    },
    'all_experiments': {
        k: {'wf_icir': v['wf_summary']['icir'], 'oos_icir': v['oos']['icir'],
            'oos_ic': v['oos']['ic'], 'hold_days': v['config']['hold_days']}
        for k, v in all_results.items()
    },
    'created': time.strftime('%Y-%m-%d %H:%M'),
    'engine': 'LightGBM',
    'note': 'Replaces XGBoost v12 which segfaults on WSL2. Same walk-forward protocol.'
}

with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
log(f'Meta: {meta_path}')

# Also copy best model to canonical name
import shutil
best_model_src = os.path.join(MODEL_DIR, f'arrow_v12_{best_name}.txt')
if os.path.exists(best_model_src):
    shutil.copy2(best_model_src, best_model_path)
    log(f'最佳模型复制: {best_model_src} -> {best_model_path}')

log('\n' + '=' * 60)
log('绿箭V12 LightGBM 重训练完成!')
log('=' * 60)

# Final summary table
log('\n╔══════════════════════════════════════════════════════╗')
log('║ 配置                    │ WF_ICIR │ OOS_ICIR │ OOS_IC ║')
log('╠══════════════════════════════════════════════════════╣')
for name, res in all_results.items():
    wf = res['wf_summary']['icir']
    oos_i = res['oos']['icir']
    oos_c = res['oos']['ic']
    marker = ' 🏆' if name == best_name else '   '
    log(f'║ {name:<22s} │ {wf:6.3f}  │ {oos_i:7.3f}  │ {oos_c:6.4f}{marker}║')
log('╚══════════════════════════════════════════════════════╝')
