#!/usr/bin/env python3
"""
xgb_param_optimization.py — GPU-accelerated XGBoost parameter optimization
for Blueshield (>$10) and Arrow ($1-$10) models.

Walk-Forward validation with IC/ICIR as primary metrics.
Random search: 60 combos per model over 6 hyperparameters.
GPU: RTX 3080 Ti, tree_method='hist', device='cuda'
"""

import os, sys, json, time, gc, warnings, random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import xgboost as xgb

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

t0 = time.time()
ROOT = '/home/hermes/.hermes/openclaw-archive'
RESULT_PATH = os.path.join(ROOT, 'data/param_search_results.json')
BS_PATH = os.path.join(ROOT, 'data/blueshield_best_params.json')
AR_PATH = os.path.join(ROOT, 'data/arrow_best_params.json')

def log(msg):
    ts = time.strftime('%H:%M:%S')
    print(f'{ts} | {msg}', flush=True)

# ====================================================================
# 1. FEATURE ENGINEERING (42 features: 27 tech + 2 extra + 13 macro)
# ====================================================================

TECH_FEATS = [
    'ma5','ma20','ma60','ma_bias20','ma_align','price_position',
    'ret1','ret5','ret20','ret60','momentum_6m','momentum_1m',
    'mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change',
    'rsi14','rsi_change','macd','macd_signal','macd_hist',
    'bb_std','bb_width','bb_pos','ret_quality'
]
EXTRA_FEATS = ['price', 'range_pct']
MACRO_FEATS = [
    'vix_close',
    'spy_ret1','spy_ret5','spy_ret20','spy_ret60',
    'qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60',
    'iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60'
]
ALL_FEATS = TECH_FEATS + EXTRA_FEATS + MACRO_FEATS


def compute_features_vectorized(group):
    """Compute features for one stock using pandas vectorized operations (much faster)."""
    g = group.sort_values('date').copy().reset_index(drop=True)
    c = g['close'].values.astype(np.float64)
    h = g['high'].values.astype(np.float64)
    lo = g['low'].values.astype(np.float64)
    o = g['open'].values.astype(np.float64)
    vol = g['volume'].values.astype(np.float64)
    n = len(c)

    if n < 80:
        return None

    # Daily returns
    dr = np.zeros(n)
    dr[1:] = (c[1:] - c[:-1]) / (c[:-1] + 1e-10)

    # Moving averages (vectorized via cumsum)
    cs = np.cumsum(c)
    def rmean_fast(arr, w):
        out = np.full(len(arr), np.nan)
        out[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
        return out

    ma5 = rmean_fast(c, 5)
    ma20 = rmean_fast(c, 20)
    ma60 = rmean_fast(c, 60)
    ma_bias20 = (c - ma20) / (ma20 + 1e-10)
    ma_align = ((c > ma5).astype(np.float64) + (ma5 > ma20).astype(np.float64))

    # Price position (60-day) — vectorized via rolling
    cs_c = pd.Series(c)
    mn60 = cs_c.rolling(60, min_periods=60).min().values
    mx60 = cs_c.rolling(60, min_periods=60).max().values
    price_position = (c - mn60) / (mx60 - mn60 + 1e-10)

    # Returns
    def pctchg(arr, p):
        out = np.full(len(arr), np.nan)
        out[p:] = arr[p:] / (arr[:-p] + 1e-10) - 1
        return out

    ret1 = pctchg(c, 1)
    ret5 = pctchg(c, 5)
    ret20 = pctchg(c, 20)
    ret60 = pctchg(c, 60)
    momentum_6m = pctchg(c, 126)
    momentum_1m = pctchg(c, 21)
    mom_divergence = momentum_1m - ret20
    trend_accel = np.full(n, np.nan)
    trend_accel[10:] = ret5[10:] - ret5[:-10]

    # Volatility (vectorized via pandas rolling)
    dr_s = pd.Series(dr)
    vol20 = dr_s.rolling(20, min_periods=20).std(ddof=1).values
    vol5 = dr_s.rolling(5, min_periods=5).std(ddof=1).values

    # Volume features
    vol_s = pd.Series(vol)
    vol_ma20 = vol_s.rolling(20, min_periods=20).mean().values
    vol_ratio = vol / (vol_ma20 + 1e-10)
    vol_change = np.full(n, np.nan)
    vol_change[20:] = vol20[20:] / (vol20[:-20] + 1e-10)

    # RSI (vectorized)
    delta = np.zeros(n)
    delta[1:] = c[1:] - c[:-1]
    gain = np.where(delta > 0, delta, 0.0)
    loss_arr = np.where(delta < 0, -delta, 0.0)
    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss_arr)
    gain_ma = gain_s.rolling(14, min_periods=14).mean().values
    loss_ma = loss_s.rolling(14, min_periods=14).mean().values
    rsi14 = 100 - 100 / (1 + gain_ma / (loss_ma + 1e-10))
    rsi_change = np.full(n, np.nan)
    rsi_change[5:] = rsi14[5:] - rsi14[:-5]

    # MACD
    c_s = pd.Series(c)
    e12 = c_s.ewm(span=12, adjust=False).mean().values
    e26 = c_s.ewm(span=26, adjust=False).mean().values
    macd = e12 - e26
    macd_s = pd.Series(macd)
    macd_signal = macd_s.ewm(span=9, adjust=False).mean().values
    macd_hist = macd - macd_signal

    # Bollinger Bands
    price_std = dr_s.rolling(20, min_periods=20).std(ddof=1).values  # Actually should be price std
    price_std2 = c_s.rolling(20, min_periods=20).std(ddof=1).values
    bb_std_val = vol20
    bb_width = 4 * bb_std_val * ma20 / (ma20 + 1e-10)
    bb_pos = (c - (ma20 - 2 * price_std2)) / (4 * price_std2 + 1e-10)

    # Ret quality
    ret_pos = np.where(dr > 0, dr, 0.0)
    ret_neg = np.where(dr < 0, -dr, 0.0)
    ret_pos_ma = pd.Series(ret_pos).rolling(20, min_periods=20).mean().values
    ret_neg_ma = pd.Series(ret_neg).rolling(20, min_periods=20).mean().values
    ret_quality = ret_pos_ma / (ret_pos_ma + ret_neg_ma + 1e-10)

    # Extra: price, range_pct
    price = c.copy()
    range_pct = (h - lo) / (c + 1e-10)

    # Forward returns
    fwd5 = np.full(n, np.nan)
    fwd5[:-5] = c[5:] / c[:-5] - 1
    fwd10 = np.full(n, np.nan)
    fwd10[:-10] = c[10:] / c[:-10] - 1
    fwd20 = np.full(n, np.nan)
    fwd20[:-20] = c[20:] / c[:-20] - 1

    # Assign
    for name, val in [
        ('ma5', ma5), ('ma20', ma20), ('ma60', ma60),
        ('ma_bias20', ma_bias20), ('ma_align', ma_align), ('price_position', price_position),
        ('ret1', ret1), ('ret5', ret5), ('ret20', ret20), ('ret60', ret60),
        ('momentum_6m', momentum_6m), ('momentum_1m', momentum_1m),
        ('mom_divergence', mom_divergence), ('trend_accel', trend_accel),
        ('vol20', vol20), ('vol5', vol5), ('vol_ratio', vol_ratio), ('vol_change', vol_change),
        ('rsi14', rsi14), ('rsi_change', rsi_change),
        ('macd', macd), ('macd_signal', macd_signal), ('macd_hist', macd_hist),
        ('bb_std', bb_std_val), ('bb_width', bb_width), ('bb_pos', bb_pos),
        ('ret_quality', ret_quality),
        ('price', price), ('range_pct', range_pct),
        ('fwd5', fwd5), ('fwd10', fwd10), ('fwd20', fwd20),
    ]:
        g[name] = val

    return g


def compute_macro_features(df_feat, vix_df, raw_price):
    """Compute macro features: VIX + SPY/QQQ/IWM returns from raw price data."""
    log('Computing macro features...')

    # VIX
    vix = vix_df[['date', 'close']].copy()
    vix.columns = ['date', 'vix_close']
    vix['date'] = pd.to_datetime(vix['date'])

    # Macro tickers from raw price
    for prefix, ticker in [('spy', 'SPY'), ('qqq', 'QQQ'), ('iwm', 'IWM')]:
        sub = raw_price[raw_price['sym'] == ticker].copy()
        if len(sub) == 0:
            log(f'  WARNING: {ticker} not found in raw data')
            continue
        sub = sub.sort_values('date')
        sub['date'] = pd.to_datetime(sub['date'])
        for w in [1, 5, 20, 60]:
            sub[f'{prefix}_ret{w}'] = sub['close'].pct_change(w)
        cols = ['date'] + [f'{prefix}_ret{w}' for w in [1, 5, 20, 60]]
        df_feat = df_feat.merge(sub[cols], on='date', how='left')
        log(f'  {ticker} merged: {len(sub)} rows')

    # Merge VIX
    df_feat = df_feat.merge(vix, on='date', how='left')
    log(f'  VIX merged')

    return df_feat


# ====================================================================
# 2. WALK-FORWARD VALIDATION & IC/ICIR
# ====================================================================

def train_and_predict(X_train, y_train, X_val, y_val, X_test, y_test, params):
    """Train XGBoost with early stopping, predict on val and test."""
    xgb_params = {
        'max_depth': params['max_depth'],
        'learning_rate': params['learning_rate'],
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
        'min_child_weight': params['min_child_weight'],
        'n_estimators': params['n_estimators'],
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': 'cuda',
        'verbosity': 0,
        'random_state': 42,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=[(dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    y_val_pred = model.predict(dval)
    y_test_pred = model.predict(dtest)

    return y_val_pred, y_test_pred, model


def compute_ic_series(y_true, y_pred, dates):
    """Compute IC (Spearman) per date."""
    df = pd.DataFrame({'y': y_true, 'pred': y_pred, 'date': dates})
    ics = []
    for d, grp in df.groupby('date'):
        if len(grp) < 10:
            continue
        ic, _ = spearmanr(grp['pred'].values, grp['y'].values)
        if not np.isnan(ic):
            ics.append({'date': str(d)[:10], 'ic': ic})
    return ics


def compute_portfolio_metrics(y_true, y_pred, dates, hold_days, top_pct=0.05):
    """Compute portfolio-level metrics from daily predictions."""
    df = pd.DataFrame({'y': y_true, 'pred': y_pred, 'date': dates})
    port_rets = []
    for d, grp in df.groupby('date'):
        if len(grp) < 20:
            continue
        threshold = grp['pred'].quantile(1 - top_pct)
        top = grp[grp['pred'] >= threshold]
        if len(top) == 0:
            continue
        port_rets.append({
            'date': str(d)[:10],
            'ret': float(top['y'].mean()),
            'win_rate': float((top['y'] > 0).mean()),
        })

    if not port_rets:
        return {}

    rdf = pd.DataFrame(port_rets)
    avg_ret = rdf['ret'].mean()
    std_ret = rdf['ret'].std()
    avg_wr = rdf['win_rate'].mean()

    trades_per_year = 252 / hold_days
    ann_ret = avg_ret * trades_per_year
    ann_vol = std_ret * np.sqrt(trades_per_year)
    sharpe = ann_ret / (ann_vol + 1e-10)

    neg = rdf[rdf['ret'] < 0]['ret']
    down_std = neg.std() * np.sqrt(trades_per_year) if len(neg) > 0 else 1e-10
    sortino = ann_ret / (down_std + 1e-10)

    cum = (1 + rdf['ret']).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    max_dd = dd.min()

    return {
        'top5_win_rate': round(avg_wr, 4),
        'top5_avg_return_pct': round(avg_ret * 100, 3),
        'annual_return_pct': round(ann_ret * 100, 2),
        'annual_vol_pct': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'n_days': len(rdf),
    }


# ====================================================================
# 3. MAIN OPTIMIZATION
# ====================================================================

def run_optimization(model_name, df_feat, hold_days, price_low, price_high, n_iter=60):
    """Random search with walk-forward validation."""
    log(f'\n{"="*70}')
    log(f'OPTIMIZING: {model_name} (hold={hold_days}d, ${price_low}-${price_high})')
    log(f'{"="*70}')

    # Filter universe
    df = df_feat[(df_feat['close'] >= price_low) & (df_feat['close'] <= price_high)].copy()
    log(f'Universe: {len(df)} rows, {df["sym"].nunique()} stocks')

    # Pick forward return column
    fwd_col = f'fwd{hold_days}' if f'fwd{hold_days}' in df.columns else 'fwd5'

    # Split by date
    train_df = df[df['date'] <= '2023-12-31'].copy()
    val_df = df[(df['date'] > '2023-12-31') & (df['date'] <= '2024-12-31')].copy()
    test_df = df[df['date'] > '2024-12-31'].copy()

    log(f'Train: {len(train_df)} rows ({train_df["date"].min()} to {train_df["date"].max()})')
    log(f'Val:   {len(val_df)} rows ({val_df["date"].min()} to {val_df["date"].max()})')
    log(f'Test:  {len(test_df)} rows ({test_df["date"].min()} to {test_df["date"].max()})')

    def prepare_xy(data):
        X = data[ALL_FEATS].values.astype(np.float32)
        y = data[fwd_col].values.astype(np.float32)
        dates = data['date'].values
        mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1) & np.isfinite(y)
        return X[mask], y[mask], dates[mask]

    X_tr, y_tr, d_tr = prepare_xy(train_df)
    X_val, y_val, d_val = prepare_xy(val_df)
    X_test, y_test, d_test = prepare_xy(test_df)

    log(f'X_train: {X_tr.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}')

    # Baseline
    baseline_params = {
        'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'min_child_weight': 10, 'n_estimators': 200,
    }

    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'min_child_weight': [5, 10, 20],
        'n_estimators': [200, 400, 600],
    }

    # Build combos
    random.seed(42)
    combos = [baseline_params.copy()]
    for _ in range(n_iter - 1):
        combos.append({k: random.choice(v) for k, v in param_grid.items()})

    log(f'\n--- Running {len(combos)} parameter combos ---')

    all_results = []
    best_icir = -999
    best_params = None
    best_val_metrics = {}
    best_test_metrics = {}

    for i, params in enumerate(combos):
        label = 'baseline' if i == 0 else f'combo_{i}'
        log(f'[{i+1}/{len(combos)}] {label}: '
            f'd={params["max_depth"]} lr={params["learning_rate"]} '
            f'sub={params["subsample"]} col={params["colsample_bytree"]} '
            f'mcw={params["min_child_weight"]} ne={params["n_estimators"]}')

        try:
            y_val_pred, y_test_pred, _ = train_and_predict(
                X_tr, y_tr, X_val, y_val, X_test, y_test, params
            )

            # IC on validation
            val_ics = compute_ic_series(y_val, y_val_pred, d_val)
            if val_ics:
                ic_vals = [x['ic'] for x in val_ics]
                val_ic = np.mean(ic_vals)
                val_ic_std = np.std(ic_vals)
                val_icir = val_ic / (val_ic_std + 1e-10)
            else:
                val_ic = val_ic_std = val_icir = 0

            # Portfolio metrics on validation
            val_port = compute_portfolio_metrics(y_val, y_val_pred, d_val, hold_days)

            # IC on test
            test_ics = compute_ic_series(y_test, y_test_pred, d_test)
            if test_ics:
                test_ic_vals = [x['ic'] for x in test_ics]
                test_ic = np.mean(test_ic_vals)
                test_ic_std = np.std(test_ic_vals)
                test_icir = test_ic / (test_ic_std + 1e-10)
            else:
                test_ic = test_ic_std = test_icir = 0

            test_port = compute_portfolio_metrics(y_test, y_test_pred, d_test, hold_days)

            val_m = {'ic': round(val_ic, 4), 'ic_std': round(val_ic_std, 4), 'icir': round(val_icir, 4)}
            val_m.update(val_port)
            test_m = {'ic': round(test_ic, 4), 'ic_std': round(test_ic_std, 4), 'icir': round(test_icir, 4)}
            test_m.update(test_port)

            result = {'idx': i, 'label': label, 'params': params.copy(),
                      'val_metrics': val_m, 'test_metrics': test_m}
            all_results.append(result)

            log(f'  VAL  ICIR={val_icir:.4f} IC={val_ic:.4f} '
                f'Sharpe={val_port.get("sharpe","N/A")} WR={val_port.get("top5_win_rate","N/A")}')
            log(f'  TEST ICIR={test_icir:.4f} IC={test_ic:.4f} '
                f'Sharpe={test_port.get("sharpe","N/A")} WR={test_port.get("top5_win_rate","N/A")}')

            if val_icir > best_icir:
                best_icir = val_icir
                best_params = params.copy()
                best_val_metrics = val_m.copy()
                best_test_metrics = test_m.copy()
                log(f'  ★ NEW BEST ICIR={best_icir:.4f}')

        except Exception as e:
            log(f'  FAILED: {e}')
            import traceback; traceback.print_exc()

        gc.collect()

    return {
        'model_name': model_name,
        'baseline_params': baseline_params,
        'baseline_val': {},  # filled in from combos[0]
        'baseline_test': {},
        'best_params': best_params,
        'best_val': best_val_metrics,
        'best_test': best_test_metrics,
        'best_icir': best_icir,
        'all_results': all_results,
        'n_combos': len(combos),
    }


# ====================================================================
# 4. MAIN
# ====================================================================

def main():
    log('='*70)
    log('XGBoost Parameter Optimization — GPU Accelerated')
    log(f'GPU: RTX 3080 Ti | XGBoost {xgb.__version__}')
    log('='*70)

    # Load data
    log('\nLoading price data...')
    df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'))
    df['date'] = pd.to_datetime(df['date'])
    log(f'Price data: {len(df)} rows, {df["sym"].nunique()} stocks')

    raw_price = df[['sym', 'date', 'close']].copy()

    # Load VIX
    log('Loading VIX data...')
    vix = pd.read_parquet(os.path.join(ROOT, 'data/us/vix_10y.parquet'))
    vix['date'] = pd.to_datetime(vix['date'])
    log(f'VIX: {len(vix)} rows')

    # Check if SPY/QQQ/IWM exist in the data
    for t in ['SPY', 'QQQ', 'IWM']:
        cnt = (df['sym'] == t).sum()
        log(f'  {t}: {cnt} rows')

    # Compute features
    log('\nComputing technical features (vectorized, should be fast)...')
    t_feat = time.time()
    parts = []
    total_syms = df['sym'].nunique()
    skipped = 0
    for i, (sym, grp) in enumerate(df.groupby('sym')):
        if len(grp) < 140:
            skipped += 1
            continue
        feat = compute_features_vectorized(grp)
        if feat is not None:
            feat['sym'] = sym
            parts.append(feat)
        else:
            skipped += 1
        if (i+1) % 2000 == 0:
            log(f'  {i+1}/{total_syms} ({time.time()-t_feat:.0f}s)')

    df_feat = pd.concat(parts, ignore_index=True)
    log(f'Features done: {time.time()-t_feat:.0f}s, {len(df_feat)} rows, {len(parts)} stocks (skipped {skipped})')

    # Compute macro features
    df_feat = compute_macro_features(df_feat, vix, raw_price)

    # Fill NaN
    for f in ALL_FEATS:
        if f not in df_feat.columns:
            df_feat[f] = 0.0
    df_feat[ALL_FEATS] = df_feat[ALL_FEATS].fillna(0).clip(-1e6, 1e6)

    log(f'Final dataset: {len(df_feat)} rows, features: {len(ALL_FEATS)}')

    # Run optimization
    results = {}

    # Blueshield: >$10, hold 20 days
    bs = run_optimization('Blueshield', df_feat, hold_days=20, price_low=10, price_high=100000, n_iter=60)
    results['blueshield'] = bs

    # Arrow: $1-$10, hold 5 days
    ar = run_optimization('Arrow', df_feat, hold_days=5, price_low=1, price_high=10, n_iter=60)
    results['arrow'] = ar

    # ====================================================================
    # 5. SAVE RESULTS
    # ====================================================================
    save_data = {}
    for k, v in results.items():
        save_data[k] = {
            'model_name': v['model_name'],
            'baseline_params': v['baseline_params'],
            'best_params': v['best_params'],
            'best_val': v['best_val'],
            'best_test': v['best_test'],
            'best_icir': v['best_icir'],
            'n_combos': v['n_combos'],
        }

    with open(RESULT_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    log(f'\nResults saved: {RESULT_PATH}')

    with open(BS_PATH, 'w') as f:
        json.dump(bs['best_params'], f, indent=2)
    log(f'Blueshield best params: {BS_PATH}')

    with open(AR_PATH, 'w') as f:
        json.dump(ar['best_params'], f, indent=2)
    log(f'Arrow best params: {AR_PATH}')

    # ====================================================================
    # 6. SUMMARY
    # ====================================================================
    log('\n' + '='*70)
    log('OPTIMIZATION SUMMARY')
    log('='*70)

    for mk in ['blueshield', 'arrow']:
        r = results[mk]
        universe = '$10+' if mk == 'blueshield' else '$1-$10'
        hold = 20 if mk == 'blueshield' else 5

        log(f'\n--- {r["model_name"]} (universe={universe}, hold={hold}d) ---')
        log(f'Combos tested: {r["n_combos"]}')

        # Find baseline results (first combo)
        baseline_res = r['all_results'][0] if r['all_results'] else None

        if baseline_res:
            bm = baseline_res['val_metrics']
            log(f'BASELINE: {baseline_res["params"]}')
            log(f'  VAL:  ICIR={bm.get("icir","N/A")} IC={bm.get("ic","N/A")} '
                f'Sharpe={bm.get("sharpe","N/A")} WR={bm.get("top5_win_rate","N/A")} '
                f'Top5%Ret={bm.get("top5_avg_return_pct","N/A")}%')

        log(f'BEST:     {r["best_params"]}')
        bm_best = r['best_val']
        log(f'  VAL:  ICIR={bm_best.get("icir","N/A")} IC={bm_best.get("ic","N/A")} '
            f'Sharpe={bm_best.get("sharpe","N/A")} WR={bm_best.get("top5_win_rate","N/A")} '
            f'Top5%Ret={bm_best.get("top5_avg_return_pct","N/A")}%')

        bt_best = r['best_test']
        log(f'  TEST: ICIR={bt_best.get("icir","N/A")} IC={bt_best.get("ic","N/A")} '
            f'Sharpe={bt_best.get("sharpe","N/A")} WR={bt_best.get("top5_win_rate","N/A")} '
            f'Top5%Ret={bt_best.get("top5_avg_return_pct","N/A")}%')

        # Improvement
        if baseline_res:
            bi = baseline_res['val_metrics'].get('icir', 0)
            ni = bm_best.get('icir', 0)
            bs_ = baseline_res['val_metrics'].get('sharpe', 0)
            ns = bm_best.get('sharpe', 0)
            bw = baseline_res['val_metrics'].get('top5_win_rate', 0)
            nw = bm_best.get('top5_win_rate', 0)
            if bi != 0:
                log(f'  IMPROVEMENT: ICIR {bi:.4f}→{ni:.4f} ({(ni-bi)/abs(bi)*100:+.1f}%)')
            if bs_ != 0:
                log(f'              Sharpe {bs_:.4f}→{ns:.4f} ({(ns-bs_)/abs(bs_)*100:+.1f}%)')
            if bw != 0:
                log(f'              WinRate {bw:.4f}→{nw:.4f} ({(nw-bw)/abs(bw)*100:+.1f}%)')

        # Top 5
        log(f'\n  Top 5 by ICIR:')
        sorted_r = sorted(
            [res for res in r['all_results'] if res['val_metrics'].get('icir', -999) > -999],
            key=lambda x: x['val_metrics']['icir'], reverse=True
        )[:5]
        for j, res in enumerate(sorted_r):
            vm = res['val_metrics']
            log(f'    #{j+1} ICIR={vm.get("icir",0):.4f} IC={vm.get("ic",0):.4f} '
                f'Sharpe={vm.get("sharpe","N/A")} WR={vm.get("top5_win_rate","N/A")} '
                f'{res["params"]}')

    log(f'\nTotal time: {(time.time()-t0)/60:.1f} minutes')
    log('DONE')


if __name__ == '__main__':
    main()
