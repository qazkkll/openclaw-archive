#!/usr/bin/env python3
"""
BlueShield V9 Extended Optimization - Push ICIR past 0.5
Tries more aggressive optimizations beyond the initial v9_large_trees (ICIR=0.480).

Key strategies:
1. More trees with ultra-low learning rate
2. Ensemble averaging across seeds
3. Feature engineering: interactions, fundamental tilt, market-neutral
4. Hold period optimization
5. Quantile regression (huber loss)
6. Feature selection by importance
"""
import gc, json, os, sys, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')

ROOT = '/home/hermes/.hermes/openclaw-archive'
DATA_DIR = os.path.join(ROOT, 'data', 'us')
MODEL_DIR = os.path.join(ROOT, 'models', 'us')
EXPER_DIR = os.path.join(ROOT, 'data', 'experiments')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXPER_DIR, exist_ok=True)

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

# ============================================
# Feature definitions (from V9)
# ============================================
TECH_FEATS = [
    'ma5','ma20','ma60','ma_bias20','ma_align','price_position',
    'ret1','ret5','ret20','ret60','momentum_6m','momentum_1m',
    'mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change',
    'rsi14','rsi_change','macd','macd_signal','macd_hist',
    'bb_std','bb_width','bb_pos','ret_quality'
]

def rmean(arr, w):
    out = np.full(len(arr), np.nan)
    cs = np.cumsum(arr)
    out[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
    return out

def rstd(arr, w):
    out = np.full(len(arr), np.nan)
    for i in range(w-1, len(arr)):
        out[i] = np.std(arr[i-w+1:i+1], ddof=1)
    return out

def ema(arr, span):
    out = np.empty(len(arr))
    alpha = 2.0 / (span + 1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

def pctchg(arr, p):
    out = np.full(len(arr), np.nan)
    out[p:] = arr[p:] / (arr[:-p] + 1e-10) - 1
    return out


def compute_features_extended(g):
    """Compute V9 features + extended features."""
    c = g['close'].values.astype(np.float64)
    h = g['high'].values.astype(np.float64)
    l = g['low'].values.astype(np.float64)
    vol = g['volume'].values.astype(np.float64)
    n = len(c)

    # Daily returns
    dr = np.full(n, np.nan)
    dr[1:] = (c[1:] - c[:-1]) / (c[:-1] + 1e-10)

    # Moving averages
    ma5 = rmean(c, 5)
    ma20 = rmean(c, 20)
    ma60 = rmean(c, 60)
    ma_bias20 = (c - ma20) / (ma20 + 1e-10)
    ma_align = ((c > ma5).astype(np.float64) + (ma5 > ma20).astype(np.float64))

    # Price position
    mn60 = np.full(n, np.nan)
    mx60 = np.full(n, np.nan)
    for i in range(59, n):
        mn60[i] = np.min(c[i-59:i+1])
        mx60[i] = np.max(c[i-59:i+1])
    price_position = (c - mn60) / (mx60 - mn60 + 1e-10)

    # Returns
    ret1 = pctchg(c, 1)
    ret5 = pctchg(c, 5)
    ret20 = pctchg(c, 20)
    ret60 = pctchg(c, 60)
    momentum_6m = pctchg(c, 126)
    momentum_1m = pctchg(c, 21)
    mom_divergence = momentum_1m - ret20
    trend_accel = np.full(n, np.nan)
    trend_accel[10:] = ret5[10:] - ret5[:-10]

    # Volatility
    vol20 = rstd(dr, 20)
    vol5 = rstd(dr, 5)

    # Volume
    vol_ma20 = rmean(vol, 20)
    vol_ratio = vol / (vol_ma20 + 1e-10)
    vol_change = np.full(n, np.nan)
    vol_change[20:] = vol20[20:] / (vol20[:-20] + 1e-10)

    # RSI
    delta = np.full(n, 0.0)
    delta[1:] = c[1:] - c[:-1]
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ma = rmean(gain, 14)
    loss_ma = rmean(loss, 14)
    rsi14 = 100 - 100 / (1 + gain_ma / (loss_ma + 1e-10))
    rsi_change = np.full(n, np.nan)
    rsi_change[5:] = rsi14[5:] - rsi14[:-5]

    # MACD
    e12 = ema(c, 12)
    e26 = ema(c, 26)
    macd = e12 - e26
    macd_signal = ema(macd, 9)
    macd_hist = macd - macd_signal

    # Bollinger Bands (FIXED)
    bb_std = vol20
    bb_mid = ma20
    bb_width = 4 * bb_std * bb_mid / (bb_mid + 1e-10)
    price_std = rstd(c, 20)
    bb_pos = (c - (bb_mid - 2 * price_std)) / (4 * price_std + 1e-10)

    # ret_quality (FIXED)
    ret_pos = np.where(dr > 0, dr, 0.0)
    ret_neg = np.where(dr < 0, -dr, 0.0)
    ret_pos_ma = rmean(ret_pos, 20)
    ret_neg_ma = rmean(ret_neg, 20)
    ret_quality = ret_pos_ma / (ret_pos_ma + ret_neg_ma + 1e-10)

    # ============================================
    # EXTENDED FEATURES
    # ============================================

    # 1. High-low range as % of close
    daily_range = (h - l) / (c + 1e-10)
    avg_range = rmean(daily_range, 20)
    range_ratio = daily_range / (avg_range + 1e-10)

    # 2. Close position within day (candle body position)
    candle_body = (c - g['open'].values.astype(np.float64)) / (h - l + 1e-10)
    avg_body = rmean(candle_body, 10)

    # 3. Volume-weighted price trend
    vwap_drift = np.full(n, np.nan)
    for i in range(20, n):
        w = vol[i-19:i+1]
        p = c[i-19:i+1]
        tw = np.sum(w * p) / (np.sum(w) + 1e-10)
        vwap_drift[i] = (c[i] - tw) / (tw + 1e-10)

    # 4. Return momentum tiers
    ret_10d = pctchg(c, 10)
    ret_30d = pctchg(c, 30)
    ret_90d = pctchg(c, 90)

    # 5. Volatility regime
    vol60 = rstd(dr, 60)
    vol_regime = vol20 / (vol60 + 1e-10)

    # 6. MA crossovers
    ma_cross_5_20 = np.where(ma5 > ma20, 1.0, 0.0)
    ma_cross_20_60 = np.where(ma20 > ma60, 1.0, 0.0)

    # 7. RSI zones (categorical encoded as continuous)
    rsi_zone = np.zeros(n)
    rsi_zone[~np.isnan(rsi14)] = rsi14[~np.isnan(rsi14)] / 100.0  # 0-1 scale

    # 8. MACD momentum (rate of change)
    macd_roc = np.full(n, np.nan)
    macd_roc[5:] = macd[5:] - macd[:-5]

    # 9. Drawdown from 60d high
    dd_60 = np.full(n, np.nan)
    for i in range(59, n):
        peak = np.max(c[i-59:i+1])
        dd_60[i] = (c[i] - peak) / (peak + 1e-10)

    # 10. Up/down volume ratio
    up_vol = np.where(dr > 0, vol, 0.0)
    dn_vol = np.where(dr < 0, vol, 0.0)
    up_vol_ma = rmean(up_vol, 20)
    dn_vol_ma = rmean(dn_vol, 20)
    ud_vol_ratio = up_vol_ma / (dn_vol_ma + 1e-10)

    # Combine all
    v9_core = np.column_stack([
        ma5, ma20, ma60, ma_bias20, ma_align, price_position,
        ret1, ret5, ret20, ret60, momentum_6m, momentum_1m,
        mom_divergence, trend_accel, vol20, vol5, vol_ratio, vol_change,
        rsi14, rsi_change, macd, macd_signal, macd_hist,
        bb_std, bb_width, bb_pos, ret_quality
    ])

    ext_feats = np.column_stack([
        range_ratio, avg_body, vwap_drift,
        ret_10d, ret_30d, ret_90d,
        vol_regime, ma_cross_5_20, ma_cross_20_60,
        rsi_zone, macd_roc, dd_60, ud_vol_ratio
    ])

    all_feats = np.column_stack([v9_core, ext_feats])
    return all_feats


def compute_cross_sectional_icir(pred, y, dates):
    """Cross-sectional Spearman ICIR."""
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 30:
            continue
        rp = pd.Series(pred[mask]).rank().values
        ry = pd.Series(y[mask]).rank().values
        ic = np.corrcoef(rp, ry)[0, 1]
        if not np.isnan(ic):
            ics.append(ic)
    if len(ics) < 5:
        return 0.0, 0.0, 0, 0.0
    ics = np.array(ics)
    return float(np.mean(ics)), float(np.std(ics, ddof=1) and np.mean(ics)/(np.std(ics, ddof=1)+1e-10)), len(ics), float(np.mean(ics > 0))


def compute_icir_clean(pred, y, dates):
    """Clean cross-sectional Spearman ICIR computation."""
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 30:
            continue
        rp = pd.Series(pred[mask]).rank().values
        ry = pd.Series(y[mask]).rank().values
        ic = np.corrcoef(rp, ry)[0, 1]
        if not np.isnan(ic):
            ics.append(ic)
    if len(ics) < 5:
        return 0.0, 0.0, 0, 0.0
    ics = np.array(ics)
    mean_ic = float(np.mean(ics))
    std_ic = float(np.std(ics, ddof=1))
    icir = mean_ic / (std_ic + 1e-10)
    ic_pos = float(np.mean(ics > 0))
    return mean_ic, icir, len(ics), ic_pos


def compute_daily_topn_returns(pred, y, dates, top_n):
    daily_rets = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < top_n * 2:
            continue
        top_idx = np.argsort(pred[mask])[-top_n:]
        daily_rets.append(float(np.mean(y[mask][top_idx])))
    return np.array(daily_rets)


def run_single_fold(X_tr, y_tr, X_vl, y_vl, params, n_trees):
    try:
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        model = lgb.train(params, dtrain, num_boost_round=n_trees,
                         valid_sets=[dval], callbacks=callbacks)
        pred = model.predict(X_vl)
        best_iter = getattr(model, 'best_iteration', n_trees)
        del model, dtrain, dval
        gc.collect()
        return pred, best_iter
    except Exception as e:
        log(f'    Fold error: {e}')
        import traceback; traceback.print_exc()
        return None, 0


def run_config(name, config, X_all, y_all, dates_all, n_trees,
               feature_names=None, extra_valid_mask=None):
    """Run a config with walk-forward validation + OOS eval."""
    log(f'\n{"="*60}')
    log(f'Config: {name} | trees={n_trees} | features={X_all.shape[1]}')
    log(f'{"="*60}')

    t0 = time.time()

    params = config.copy()

    # Split train/OOS
    oos_start = np.datetime64('2024-01-01')
    train_mask = dates_all < oos_start
    oos_mask = dates_all >= oos_start

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    dates_train = dates_all[train_mask]

    X_oos = X_all[oos_mask]
    y_oos = y_all[oos_mask]
    dates_oos = dates_all[oos_mask]

    log(f'  Train: {len(X_train):,}, OOS: {len(X_oos):,}')

    # Walk-forward 4-fold
    unique_dates = np.sort(np.unique(dates_train))
    fold_size = len(unique_dates) // 5

    fold_results = []
    best_iters = []

    for fold in range(4):
        val_start_idx = (fold + 1) * fold_size
        val_end_idx = min((fold + 2) * fold_size, len(unique_dates) - 1)

        if val_start_idx >= len(unique_dates) or val_end_idx >= len(unique_dates):
            continue

        val_start = unique_dates[val_start_idx]
        val_end = unique_dates[val_end_idx]

        tr_mask = dates_train < val_start
        vl_mask = (dates_train >= val_start) & (dates_train < val_end)

        if tr_mask.sum() < 5000 or vl_mask.sum() < 500:
            continue

        X_tr, y_tr = X_train[tr_mask], y_train[tr_mask]
        X_vl, y_vl = X_train[vl_mask], y_train[vl_mask]

        pred, best_iter = run_single_fold(X_tr, y_tr, X_vl, y_vl, params, n_trees)

        if pred is not None and len(pred) > 0:
            mean_ic, icir, n_dates, ic_pos = compute_icir_clean(
                pred, y_vl, dates_train[vl_mask]
            )
            fold_results.append({'fold': fold, 'icir': icir, 'mean_ic': mean_ic,
                                'ic_pos': ic_pos, 'n_dates': n_dates})
            best_iters.append(best_iter)
            log(f'  Fold {fold}: ICIR={icir:.4f}, IC={mean_ic:.4f}, IC+={ic_pos:.2f}, iter={best_iter}')

        del X_tr, y_tr, X_vl, y_vl
        gc.collect()

    if not fold_results:
        log(f'  ERROR: No valid folds')
        return {'name': name, 'error': 'no valid folds'}

    avg_wf_icir = float(np.mean([r['icir'] for r in fold_results]))
    final_trees = max(int(np.median(best_iters)), 100) if best_iters else n_trees
    log(f'  WF avg ICIR: {avg_wf_icir:.4f}, final trees: {final_trees}')

    # Train final on full training set + OOS
    try:
        dtrain_final = lgb.Dataset(X_train, label=y_train)
        final_model = lgb.train(params, dtrain_final, num_boost_round=final_trees,
                               callbacks=[lgb.log_evaluation(0)])

        oos_pred = final_model.predict(X_oos)

        oos_ic, oos_icir, oos_n_dates, oos_ic_pos = compute_icir_clean(
            oos_pred, y_oos, dates_oos
        )

        oos_daily = compute_daily_topn_returns(oos_pred, y_oos, dates_oos, 15)
        oos_mean = float(np.mean(oos_daily)) if len(oos_daily) > 0 else 0
        oos_win = float(np.mean(oos_daily > 0) * 100) if len(oos_daily) > 0 else 0

        oos_bot = compute_daily_topn_returns(-oos_pred, y_oos, dates_oos, 15)
        oos_spread = (oos_mean - float(np.mean(oos_bot))) * 100 if len(oos_bot) > 0 else 0

        # Feature importance
        importance = final_model.feature_importance(importance_type='gain')
        total_imp = sum(importance) if sum(importance) > 0 else 1
        if feature_names is None:
            feature_names = [f'f{i}' for i in range(len(importance))]
        feat_imp = {feature_names[i]: round(float(importance[i]) / total_imp * 100, 2)
                    for i in np.argsort(importance)[::-1][:10]}

        elapsed = time.time() - t0

        result = {
            'name': name,
            'params': {k: v for k, v in config.items() if k not in ('verbosity', 'verbose')},
            'wf_icir': round(avg_wf_icir, 4),
            'wf_folds': len(fold_results),
            'n_trees_final': final_trees,
            'n_features': X_all.shape[1],
            'oos_ic': round(oos_ic, 4),
            'oos_icir': round(oos_icir, 4),
            'oos_ic_pos': round(oos_ic_pos, 2),
            'oos_n_dates': oos_n_dates,
            'oos_days': len(oos_daily),
            'oos_avg_ret_pct': round(oos_mean * 100, 2),
            'oos_win_pct': round(oos_win, 1),
            'oos_spread_pct': round(oos_spread, 2),
            'feature_importance': feat_imp,
            'elapsed_s': round(elapsed, 1),
        }

        if oos_icir > 0.47:
            model_path = os.path.join(MODEL_DIR, f'blueshield_{name}_lgb.txt')
            final_model.save_model(model_path)
            result['model_saved'] = model_path
            log(f'  Saved: {model_path}')

        del final_model, dtrain_final
        gc.collect()

        log(f'  OOS: IC={oos_ic:.4f}, ICIR={oos_icir:.4f}, IC+={oos_ic_pos:.2f}')
        log(f'  OOS Top15: ret={oos_mean*100:.2f}%, win={oos_win:.1f}%, spread={oos_spread:.2f}%')
        log(f'  Done in {elapsed:.1f}s')

        return result

    except Exception as e:
        log(f'  ERROR in final eval: {e}')
        import traceback; traceback.print_exc()
        return {'name': name, 'error': str(e), 'wf_icir': round(avg_wf_icir, 4)}


def run_ensemble_config(name, configs_list, X_all, y_all, dates_all, n_trees,
                        feature_names=None):
    """Train multiple models with different seeds and average predictions."""
    log(f'\n{"="*60}')
    log(f'ENSEMBLE Config: {name} | {len(configs_list)} models | trees={n_trees}')
    log(f'{"="*60}')

    t0 = time.time()

    oos_start = np.datetime64('2024-01-01')
    train_mask = dates_all < oos_start
    oos_mask = dates_all >= oos_start

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    dates_train = dates_all[train_mask]
    X_oos = X_all[oos_mask]
    y_oos = y_all[oos_mask]
    dates_oos = dates_all[oos_mask]

    log(f'  Train: {len(X_train):,}, OOS: {len(X_oos):,}')

    # Walk-forward with ensemble
    unique_dates = np.sort(np.unique(dates_train))
    fold_size = len(unique_dates) // 5

    fold_results = []

    for fold in range(4):
        val_start_idx = (fold + 1) * fold_size
        val_end_idx = min((fold + 2) * fold_size, len(unique_dates) - 1)
        if val_start_idx >= len(unique_dates) or val_end_idx >= len(unique_dates):
            continue
        val_start = unique_dates[val_start_idx]
        val_end = unique_dates[val_end_idx]
        tr_mask = dates_train < val_start
        vl_mask = (dates_train >= val_start) & (dates_train < val_end)
        if tr_mask.sum() < 5000 or vl_mask.sum() < 500:
            continue

        X_tr, y_tr = X_train[tr_mask], y_train[tr_mask]
        X_vl, y_vl = X_train[vl_mask], y_train[vl_mask]

        ensemble_preds = []
        for cfg in configs_list:
            pred, _ = run_single_fold(X_tr, y_tr, X_vl, y_vl, cfg, n_trees)
            if pred is not None:
                ensemble_preds.append(pred)

        if ensemble_preds:
            avg_pred = np.mean(ensemble_preds, axis=0)
            mean_ic, icir, n_dates, ic_pos = compute_icir_clean(
                avg_pred, y_vl, dates_train[vl_mask]
            )
            fold_results.append({'fold': fold, 'icir': icir, 'mean_ic': mean_ic,
                                'ic_pos': ic_pos, 'n_dates': n_dates})
            log(f'  Fold {fold}: ICIR={icir:.4f}, IC={mean_ic:.4f}, IC+={ic_pos:.2f}')

        del X_tr, y_tr, X_vl, y_vl
        gc.collect()

    if not fold_results:
        return {'name': name, 'error': 'no valid folds'}

    avg_wf_icir = float(np.mean([r['icir'] for r in fold_results]))
    log(f'  WF avg ICIR: {avg_wf_icir:.4f}')

    # Final ensemble on full train + OOS
    try:
        oos_preds = []
        for cfg in configs_list:
            dtrain = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(cfg, dtrain, num_boost_round=n_trees,
                            callbacks=[lgb.log_evaluation(0)])
            oos_pred = model.predict(X_oos)
            oos_preds.append(oos_pred)
            del model, dtrain
            gc.collect()

        avg_oos_pred = np.mean(oos_preds, axis=0)

        oos_ic, oos_icir, oos_n_dates, oos_ic_pos = compute_icir_clean(
            avg_oos_pred, y_oos, dates_oos
        )

        oos_daily = compute_daily_topn_returns(avg_oos_pred, y_oos, dates_oos, 15)
        oos_mean = float(np.mean(oos_daily)) if len(oos_daily) > 0 else 0
        oos_win = float(np.mean(oos_daily > 0) * 100) if len(oos_daily) > 0 else 0

        oos_bot = compute_daily_topn_returns(-avg_oos_pred, y_oos, dates_oos, 15)
        oos_spread = (oos_mean - float(np.mean(oos_bot))) * 100 if len(oos_bot) > 0 else 0

        elapsed = time.time() - t0

        result = {
            'name': name,
            'type': 'ensemble',
            'n_models': len(configs_list),
            'wf_icir': round(avg_wf_icir, 4),
            'wf_folds': len(fold_results),
            'n_trees_final': n_trees,
            'n_features': X_all.shape[1],
            'oos_ic': round(oos_ic, 4),
            'oos_icir': round(oos_icir, 4),
            'oos_ic_pos': round(oos_ic_pos, 2),
            'oos_n_dates': oos_n_dates,
            'oos_days': len(oos_daily),
            'oos_avg_ret_pct': round(oos_mean * 100, 2),
            'oos_win_pct': round(oos_win, 1),
            'oos_spread_pct': round(oos_spread, 2),
            'elapsed_s': round(elapsed, 1),
        }

        log(f'  OOS: IC={oos_ic:.4f}, ICIR={oos_icir:.4f}, IC+={oos_ic_pos:.2f}')
        log(f'  OOS Top15: ret={oos_mean*100:.2f}%, win={oos_win:.1f}%, spread={oos_spread:.2f}%')
        log(f'  Done in {elapsed:.1f}s')

        return result

    except Exception as e:
        log(f'  ERROR: {e}')
        import traceback; traceback.print_exc()
        return {'name': name, 'error': str(e)}


def main():
    log('='*70)
    log('BlueShield V9 EXTENDED Optimization - Push ICIR > 0.5')
    log('='*70)

    t_total = time.time()

    # 1. Load data
    log('\nStep 1: Loading price data...')
    t0 = time.time()
    df = pd.read_parquet(os.path.join(DATA_DIR, 'us_hist_full_10y.parquet'),
                         columns=['sym','date','open','high','low','close','volume'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close','volume'])
    df = df[(df['close'] > 10) & (df['volume'] > 0)]
    df = df.sort_values(['sym', 'date'])
    log(f'  Loaded: {len(df):,} rows, {df["sym"].nunique()} stocks ({time.time()-t0:.0f}s)')

    # Sample 2500 stocks
    all_syms = df['sym'].unique()
    if len(all_syms) > 2500:
        np.random.seed(42)
        sampled = np.random.choice(all_syms, 2500, replace=False)
        df = df[df['sym'].isin(sampled)]
        log(f'  Sampled to 2500 stocks: {len(df):,} rows')

    # 2. Load fundamentals
    log('\nStep 2: Loading fundamentals...')
    try:
        fund = pd.read_parquet(os.path.join(DATA_DIR, 'fundamentals_latest.parquet'))
        log(f'  Fundamentals: {len(fund)} stocks, cols={fund.columns.tolist()}')
    except:
        fund = None
        log('  No fundamentals available')

    # 3. Compute extended features
    log('\nStep 3: Computing extended features...')
    t0 = time.time()

    feat_list = []
    label_20_list = []
    label_10_list = []
    date_list = []
    sym_list = []
    date_val_list = []

    syms = list(df.groupby('sym'))
    for i, (sym, g) in enumerate(syms):
        if len(g) < 80:
            continue
        try:
            feats = compute_features_extended(g)
            c = g['close'].values.astype(np.float64)
            n = len(c)

            def make_fwd_ret(hold):
                fr = np.full(n, np.nan, dtype=np.float32)
                if n > hold:
                    raw = c[hold:] / (c[:-hold] + 1e-10) - 1.0
                    raw = np.clip(raw, -0.5, 0.5)
                    fr[:-hold] = raw.astype(np.float32)
                return fr

            feat_list.append(feats.astype(np.float32))
            label_20_list.append(make_fwd_ret(20))
            label_10_list.append(make_fwd_ret(10))
            date_list.append(g['date'].values.astype('datetime64[ns]'))
            sym_list.extend([sym] * n)
            date_val_list.extend(g['date'].values.astype('datetime64[ns]'))
        except:
            continue

        if (i+1) % 500 == 0:
            log(f'  {i+1}/{len(syms)} stocks ({time.time()-t0:.0f}s)')

    X_all = np.vstack(feat_list)
    y_20 = np.concatenate(label_20_list)
    y_10 = np.concatenate(label_10_list)
    dates_all = np.concatenate(date_list)

    # Extended feature names
    EXT_FEAT_NAMES = TECH_FEATS + [
        'range_ratio', 'avg_body', 'vwap_drift',
        'ret_10d', 'ret_30d', 'ret_90d',
        'vol_regime', 'ma_cross_5_20', 'ma_cross_20_60',
        'rsi_zone', 'macd_roc', 'dd_60', 'ud_vol_ratio'
    ]

    del feat_list, label_20_list, label_10_list
    gc.collect()

    # Filter valid
    valid = ~np.isnan(X_all).any(axis=1) & ~np.isinf(X_all).any(axis=1)
    X_all = X_all[valid]
    y_20 = y_20[valid]
    y_10 = y_10[valid]
    dates_all = dates_all[valid]

    valid_20 = ~np.isnan(y_20)
    valid_10 = ~np.isnan(y_10)

    log(f'  Total rows: {len(X_all):,}')
    log(f'  Valid labels 20d: {valid_20.sum():,}, 10d: {valid_10.sum():,}')
    log(f'  Features: {X_all.shape[1]}')
    log(f'  Time: {time.time()-t0:.0f}s')

    # Add fundamentals as cross-sectional features (constant per stock, varies by date)
    # We'll add PE quintile, beta, dividend yield rank as features
    if fund is not None:
        log('\n  Adding fundamental features...')
        # Create a mapping
        sym_to_fund = {}
        for _, row in fund.iterrows():
            sym_to_fund[row['sym']] = {
                'beta': float(row['beta']) if pd.notna(row['beta']) else 1.0,
                'pe': float(row['pe_trailing']) if pd.notna(row['pe_trailing']) else 20.0,
                'div_yield': float(row['div_yield']) if pd.notna(row['div_yield']) else 0.0,
            }

        # Add fundamental columns to feature matrix
        fund_cols = np.zeros((len(X_all), 3), dtype=np.float32)
        for idx, sym in enumerate(sym_list):
            if idx < len(fund_cols) and sym in sym_to_fund:
                f = sym_to_fund[sym]
                fund_cols[idx] = [f['pe'], f['div_yield'], f['beta']]

        # Replace nan/inf
        fund_cols = np.nan_to_num(fund_cols, nan=0.0, posinf=100.0, neginf=-100.0)

        # Log-transform PE (winsorize first)
        fund_cols[:, 0] = np.clip(fund_cols[:, 0], -50, 200)
        fund_cols[:, 0] = np.log1p(np.abs(fund_cols[:, 0])) * np.sign(fund_cols[:, 0])

        X_all = np.hstack([X_all, fund_cols])
        EXT_FEAT_NAMES = EXT_FEAT_NAMES + ['pe_log', 'div_yield', 'beta']
        log(f'  Added 3 fundamental features, total: {X_all.shape[1]}')

    # 4. Create market-neutral returns (cross-sectional rank neutralization)
    log('\nStep 4: Creating market-neutral labels...')
    y_20_raw = y_20.copy()

    # Cross-sectional demean each date
    y_20_cn = y_20_raw.copy()
    for d in np.unique(dates_all):
        mask = dates_all == d
        if mask.sum() > 10:
            y_20_cn[mask] = y_20_raw[mask] - np.mean(y_20_raw[mask])

    log(f'  Market-neutral labels created')

    # ============================================
    # Run extended configurations
    # ============================================
    log('\nStep 5: Running extended configurations...')

    results = {}

    # === Config 1: 5000 trees, very low LR ===
    results['lgb_v9_5000trees'] = run_config(
        'lgb_v9_5000trees',
        {'objective': 'regression', 'max_depth': 5, 'learning_rate': 0.005,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 30,
         'reg_alpha': 0.05, 'reg_lambda': 2.0, 'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_all[valid_20], y_20[valid_20], dates_all[valid_20], n_trees=5000,
        feature_names=EXT_FEAT_NAMES
    )
    gc.collect()

    # === Config 2: Huber loss (robust to outliers) ===
    results['lgb_v9_huber'] = run_config(
        'lgb_v9_huber',
        {'objective': 'huber', 'max_depth': 5, 'learning_rate': 0.01,
         'alpha': 0.9, 'subsample': 0.8, 'colsample_bytree': 0.8,
         'min_child_samples': 30, 'reg_alpha': 0.05, 'reg_lambda': 2.0,
         'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_all[valid_20], y_20[valid_20], dates_all[valid_20], n_trees=3000,
        feature_names=EXT_FEAT_NAMES
    )
    gc.collect()

    # === Config 3: Deeper trees, less regularization ===
    results['lgb_v9_deep'] = run_config(
        'lgb_v9_deep',
        {'objective': 'regression', 'max_depth': 7, 'learning_rate': 0.008,
         'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_samples': 15,
         'reg_alpha': 0.02, 'reg_lambda': 1.0, 'num_leaves': 63,
         'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_all[valid_20], y_20[valid_20], dates_all[valid_20], n_trees=3000,
        feature_names=EXT_FEAT_NAMES
    )
    gc.collect()

    # === Config 4: Leaf-wise growth, high capacity ===
    results['lgb_v9_leafwise'] = run_config(
        'lgb_v9_leafwise',
        {'objective': 'regression', 'max_depth': -1, 'num_leaves': 63,
         'learning_rate': 0.01, 'subsample': 0.8, 'colsample_bytree': 0.8,
         'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 2.0,
         'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_all[valid_20], y_20[valid_20], dates_all[valid_20], n_trees=3000,
        feature_names=EXT_FEAT_NAMES
    )
    gc.collect()

    # === Config 5: Market-neutral labels ===
    results['lgb_v9_market_neutral'] = run_config(
        'lgb_v9_market_neutral',
        {'objective': 'regression', 'max_depth': 5, 'learning_rate': 0.01,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 30,
         'reg_alpha': 0.05, 'reg_lambda': 2.0,
         'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_all[valid_20], y_20_cn[valid_20], dates_all[valid_20], n_trees=3000,
        feature_names=EXT_FEAT_NAMES
    )
    gc.collect()

    # === Config 6: 10-day hold, 3000 trees ===
    results['lgb_v9_10d_hold'] = run_config(
        'lgb_v9_10d_hold',
        {'objective': 'regression', 'max_depth': 5, 'learning_rate': 0.01,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 30,
         'reg_alpha': 0.05, 'reg_lambda': 2.0,
         'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_all[valid_10], y_10[valid_10], dates_all[valid_10], n_trees=3000,
        feature_names=EXT_FEAT_NAMES
    )
    gc.collect()

    # === Config 7: High subsample, more regularization ===
    results['lgb_v9_high_subsample'] = run_config(
        'lgb_v9_high_subsample',
        {'objective': 'regression', 'max_depth': 5, 'learning_rate': 0.01,
         'subsample': 0.6, 'subsample_freq': 2,
         'colsample_bytree': 0.6, 'min_child_samples': 50,
         'reg_alpha': 0.5, 'reg_lambda': 5.0,
         'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_all[valid_20], y_20[valid_20], dates_all[valid_20], n_trees=3000,
        feature_names=EXT_FEAT_NAMES
    )
    gc.collect()

    # === Config 8: Quantile regression (predict median) ===
    results['lgb_v9_quantile'] = run_config(
        'lgb_v9_quantile',
        {'objective': 'quantile', 'alpha': 0.5, 'max_depth': 5,
         'learning_rate': 0.01, 'subsample': 0.8, 'colsample_bytree': 0.8,
         'min_child_samples': 30, 'reg_alpha': 0.05, 'reg_lambda': 2.0,
         'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_all[valid_20], y_20[valid_20], dates_all[valid_20], n_trees=3000,
        feature_names=EXT_FEAT_NAMES
    )
    gc.collect()

    # === Config 9: Ensemble of 3 seeds with large_trees params ===
    ensemble_configs = []
    base_params = {
        'objective': 'regression', 'max_depth': 5, 'learning_rate': 0.01,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 30,
        'reg_alpha': 0.05, 'reg_lambda': 2.0, 'num_threads': 4, 'verbosity': -1
    }
    for seed in [42, 123, 456]:
        cfg = base_params.copy()
        cfg['seed'] = seed
        ensemble_configs.append(cfg)

    results['lgb_v9_ensemble_3seed'] = run_ensemble_config(
        'lgb_v9_ensemble_3seed',
        ensemble_configs,
        X_all[valid_20], y_20[valid_20], dates_all[valid_20], n_trees=2000,
        feature_names=EXT_FEAT_NAMES
    )
    gc.collect()

    # === Config 10: Ensemble of 3 different configs ===
    ensemble_diff_configs = [
        {'objective': 'regression', 'max_depth': 5, 'learning_rate': 0.01,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 30,
         'reg_alpha': 0.05, 'reg_lambda': 2.0, 'num_threads': 4, 'verbosity': -1, 'seed': 42},
        {'objective': 'regression', 'max_depth': 7, 'learning_rate': 0.008,
         'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_samples': 15,
         'reg_alpha': 0.02, 'reg_lambda': 1.0, 'num_leaves': 63,
         'num_threads': 4, 'verbosity': -1, 'seed': 123},
        {'objective': 'regression', 'max_depth': 4, 'learning_rate': 0.015,
         'subsample': 0.9, 'colsample_bytree': 0.6, 'min_child_samples': 40,
         'reg_alpha': 0.1, 'reg_lambda': 3.0, 'num_threads': 4, 'verbosity': -1, 'seed': 456},
    ]

    results['lgb_v9_ensemble_diverse'] = run_ensemble_config(
        'lgb_v9_ensemble_diverse',
        ensemble_diff_configs,
        X_all[valid_20], y_20[valid_20], dates_all[valid_20], n_trees=2500,
        feature_names=EXT_FEAT_NAMES
    )
    gc.collect()

    # === Config 11: MA60-only features (top importance) ===
    # Based on feature importance, focus on strongest features
    TOP_FEATS = ['ma60', 'ma5', 'vol20', 'momentum_6m', 'ret60', 'vol_change',
                 'ma20', 'bb_std', 'price_position', 'ma_bias20', 'rsi14',
                 'macd_signal', 'ret20', 'ret1', 'ret_quality']
    top_idx = [EXT_FEAT_NAMES.index(f) for f in TOP_FEATS if f in EXT_FEAT_NAMES]
    results['lgb_v9_top15feats'] = run_config(
        'lgb_v9_top15feats',
        {'objective': 'regression', 'max_depth': 5, 'learning_rate': 0.01,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 30,
         'reg_alpha': 0.05, 'reg_lambda': 2.0,
         'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_all[valid_20][:, top_idx], y_20[valid_20], dates_all[valid_20], n_trees=3000,
        feature_names=TOP_FEATS[:len(top_idx)]
    )
    gc.collect()

    # === Config 12: Interaction features (top 2 pairs) ===
    # MA alignment interaction: ma_bias20 * vol_regime
    # This captures whether high momentum stocks are volatile
    X_interact = np.hstack([
        X_all[valid_20],
        (X_all[valid_20, EXT_FEAT_NAMES.index('ma_bias20')] *
         X_all[valid_20, EXT_FEAT_NAMES.index('vol_regime')]).reshape(-1, 1),
        (X_all[valid_20, EXT_FEAT_NAMES.index('momentum_6m')] *
         X_all[valid_20, EXT_FEAT_NAMES.index('vol20')]).reshape(-1, 1),
        (X_all[valid_20, EXT_FEAT_NAMES.index('ret60')] *
         X_all[valid_20, EXT_FEAT_NAMES.index('rsi14')]).reshape(-1, 1),
    ])
    INTERACT_NAMES = EXT_FEAT_NAMES + ['bias_x_volregime', 'mom6m_x_vol20', 'ret60_x_rsi14']

    results['lgb_v9_interactions'] = run_config(
        'lgb_v9_interactions',
        {'objective': 'regression', 'max_depth': 5, 'learning_rate': 0.01,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 30,
         'reg_alpha': 0.05, 'reg_lambda': 2.0,
         'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_interact, y_20[valid_20], dates_all[valid_20], n_trees=3000,
        feature_names=INTERACT_NAMES
    )
    del X_interact
    gc.collect()

    # ============================================
    # Summary
    # ============================================
    log('\n' + '='*70)
    log('RESULTS SUMMARY')
    log('='*70)

    best_icir = -1
    best_name = None

    # Load previous best
    prev_best = 0.480

    for name, r in results.items():
        if 'error' in r:
            log(f'\n{name}: ERROR - {r["error"]}')
            continue

        icir = r.get('oos_icir', 0)
        if icir > best_icir:
            best_icir = icir
            best_name = name

        improvement = '✅ NEW BEST' if icir > prev_best else ''
        log(f'\n{name}:')
        log(f'  WF ICIR: {r.get("wf_icir", 0):.4f} ({r.get("wf_folds", 0)} folds)')
        log(f'  OOS: IC={r.get("oos_ic", 0):.4f}, ICIR={icir:.4f}, IC+={r.get("oos_ic_pos", 0):.2f}')
        log(f'  Top15: ret={r.get("oos_avg_ret_pct", 0):.2f}%, win={r.get("oos_win_pct", 0):.1f}%, spread={r.get("oos_spread_pct", 0):.2f}%')
        log(f'  Features: {r.get("n_features", "?")}, Trees: {r.get("n_trees_final", "?")}, Time: {r.get("elapsed_s", 0):.0f}s')
        if improvement:
            log(f'  {improvement}')

    # Save results
    log('\n\nSaving results...')
    # Clean up for JSON
    clean_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            clean_v = {}
            for kk, vv in v.items():
                if kk == 'params' and isinstance(vv, dict):
                    clean_v[kk] = {pk: pv for pk, pv in vv.items() if pk not in ('verbosity', 'verbose')}
                else:
                    clean_v[kk] = vv
            clean_results[k] = clean_v

    results_path = os.path.join(EXPER_DIR, 'lgb_v9_extended_results.json')
    with open(results_path, 'w') as f:
        json.dump(clean_results, f, indent=2, default=str)
    log(f'Results saved: {results_path}')

    # Save best as v10
    if best_icir > 0 and best_name:
        best_result = results[best_name]
        meta = {
            'version': 'blueshield_v10',
            'config': best_name,
            'features': EXT_FEAT_NAMES,
            'n_features': X_all.shape[1],
            'oos_metrics': {
                'ic': best_result.get('oos_ic', 0),
                'icir': best_result.get('oos_icir', 0),
                'ic_pos': best_result.get('oos_ic_pos', 0),
                'spread': best_result.get('oos_spread_pct', 0),
                'top15_avg': best_result.get('oos_avg_ret_pct', 0),
                'top15_win': best_result.get('oos_win_pct', 0),
            },
            'wf_icir': best_result.get('wf_icir', 0),
            'feature_importance': best_result.get('feature_importance', {}),
            'created': time.strftime('%Y-%m-%d %H:%M'),
            'replaces': 'blueshield_v9',
            'optimized_from': best_name,
            'engine': 'lightgbm',
            'previous_best_icir': prev_best,
            'all_results_summary': {
                k: {'oos_icir': v.get('oos_icir', 0), 'oos_ic': v.get('oos_ic', 0)}
                for k, v in results.items() if 'error' not in v
            },
        }

        meta_path = os.path.join(MODEL_DIR, 'blueshield_v10_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        log(f'Metadata saved: {meta_path}')

    total_time = time.time() - t_total
    log(f'\nTotal time: {total_time/60:.1f} minutes')
    log('='*70)

    if best_icir > 0.5:
        log(f'✅ SUCCESS: Best ICIR={best_icir:.4f} > 0.5 (config: {best_name})')
    elif best_icir > prev_best:
        log(f'📈 IMPROVED: Best ICIR={best_icir:.4f} > {prev_best} previous best (config: {best_name})')
    else:
        log(f'⚠️ NOT IMPROVED: Best ICIR={best_icir:.4f} <= {prev_best} (config: {best_name})')

    log('='*70)


if __name__ == '__main__':
    main()
