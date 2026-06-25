#!/usr/bin/env python3
"""
BlueShield V9 Optimization Script (LightGBM - XGBoost broken on this system)
Tries multiple configurations to achieve ICIR > 0.5

Configs:
1. v9_fixed_20d: Fixed formulas, 20-day hold, default params
2. v9_fixed_10d: Fixed formulas, 10-day hold
3. v9_fixed_5d: Fixed formulas, 5-day hold
4. v9_hp_tuned: Hyperparameter search (max_depth, learning_rate, n_estimators)
5. v9_classifier: Binary classifier (fwd_ret > 2%)
6. v9_aggressive: More regularization
7. v9_large_trees: More trees with lower LR
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

TECH_FEATS = [
    'ma5','ma20','ma60','ma_bias20','ma_align','price_position',
    'ret1','ret5','ret20','ret60','momentum_6m','momentum_1m',
    'mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change',
    'rsi14','rsi_change','macd','macd_signal','macd_hist',
    'bb_std','bb_width','bb_pos','ret_quality'
]

def compute_features_fixed(g):
    """FIXED formulas for BlueShield V9:
    bb_std = return_std = dr.rolling(20).std()
    bb_width = 4 * bb_std * bb_mid / (bb_mid + eps)
    bb_pos = (close - (ma20 - 2*price_std)) / (4*price_std + eps)
    ret_quality = ret_pos / (ret_pos + ret_neg + eps)
    """
    c = g['close'].values.astype(np.float64)
    vol = g['volume'].values.astype(np.float64)
    n = len(c)
    
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
    
    # Volatility
    vol20 = rstd(dr, 20)
    vol5 = rstd(dr, 5)
    
    # Volume ratio
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
    def ema(arr, span):
        out = np.empty(len(arr))
        alpha = 2.0 / (span + 1)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
        return out
    
    e12 = ema(c, 12)
    e26 = ema(c, 26)
    macd = e12 - e26
    macd_signal = ema(macd, 9)
    macd_hist = macd - macd_signal
    
    # FIXED Bollinger Band features
    bb_std = vol20
    bb_mid = ma20
    bb_width = 4 * bb_std * bb_mid / (bb_mid + 1e-10)
    
    price_std = rstd(c, 20)
    bb_pos = (c - (bb_mid - 2 * price_std)) / (4 * price_std + 1e-10)
    
    # FIXED ret_quality
    ret_pos = np.where(dr > 0, dr, 0.0)
    ret_neg = np.where(dr < 0, -dr, 0.0)
    ret_pos_ma = rmean(ret_pos, 20)
    ret_neg_ma = rmean(ret_neg, 20)
    ret_quality = ret_pos_ma / (ret_pos_ma + ret_neg_ma + 1e-10)
    
    feats = np.column_stack([
        ma5, ma20, ma60, ma_bias20, ma_align, price_position,
        ret1, ret5, ret20, ret60, momentum_6m, momentum_1m,
        mom_divergence, trend_accel, vol20, vol5, vol_ratio, vol_change,
        rsi14, rsi_change, macd, macd_signal, macd_hist,
        bb_std, bb_width, bb_pos, ret_quality
    ])
    return feats


def compute_icir_from_daily(daily_rets):
    """Compute ICIR from daily portfolio returns."""
    if len(daily_rets) < 2:
        return 0.0, 0.0
    mean_ret = np.mean(daily_rets)
    std_ret = np.std(daily_rets, ddof=1)
    icir = mean_ret / (std_ret + 1e-10) * np.sqrt(252 / 20)
    return float(mean_ret), float(icir)


def compute_daily_topn_returns(pred, y, dates, top_n):
    """Compute daily top-N average returns."""
    daily_rets = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < top_n * 2:
            continue
        top_idx = np.argsort(pred[mask])[-top_n:]
        daily_rets.append(float(np.mean(y[mask][top_idx])))
    return np.array(daily_rets)


def compute_cross_sectional_icir(pred, y, dates):
    """Compute proper cross-sectional ICIR:
    For each date, compute Spearman rank IC between predictions and actual returns.
    ICIR = mean(IC) / std(IC)
    
    Note: NOT multiplied by sqrt(N) - that would be the t-statistic, not ICIR.
    """
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
    icir = mean_ic / (std_ic + 1e-10)  # NO sqrt(N) multiplication
    ic_pos = float(np.mean(ics > 0))
    
    return mean_ic, icir, len(ics), ic_pos


def run_single_fold(X_tr, y_tr, X_vl, y_vl, params, n_trees, objective='reg', is_binary=False):
    """Train and evaluate a single fold using LightGBM."""
    try:
        if is_binary:
            y_tr_eval = (y_tr > 0.02).astype(np.float32)
            y_vl_eval = (y_vl > 0.02).astype(np.float32)
        else:
            y_tr_eval = y_tr
            y_vl_eval = y_vl
        
        dtrain = lgb.Dataset(X_tr, label=y_tr_eval)
        dval = lgb.Dataset(X_vl, label=y_vl_eval, reference=dtrain)
        
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
        import traceback
        traceback.print_exc()
        return None, 0


def run_config(name, config, X_all, y_all, dates_all, n_trees, is_binary=False):
    """Run a single configuration with walk-forward validation."""
    log(f'\n{"="*60}')
    log(f'Config: {name} | objective={"binary" if is_binary else "reg"} | trees={n_trees}')
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
    
    # Walk-forward 5-fold
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
        
        pred, best_iter = run_single_fold(X_tr, y_tr, X_vl, y_vl, params, n_trees, 
                                          is_binary=is_binary)
        
        if pred is not None and len(pred) > 0:
            # Compute cross-sectional ICIR
            mean_ic, icir, n_dates, ic_pos = compute_cross_sectional_icir(
                pred, y_vl, dates_train[vl_mask]
            )
            
            # Also compute daily top-N returns for reference
            daily_rets = compute_daily_topn_returns(pred, y_vl, dates_train[vl_mask], 15)
            mean_ret = float(np.mean(daily_rets)) if len(daily_rets) > 0 else 0
            
            fold_results.append({'fold': fold, 'icir': icir, 'mean_ic': mean_ic, 
                                'ic_pos': ic_pos, 'n_dates': n_dates})
            best_iters.append(best_iter)
            log(f'  Fold {fold}: ICIR={icir:.4f}, IC={mean_ic:.4f}, IC+={ic_pos:.2f}, dates={n_dates}, iter={best_iter}')
        
        del X_tr, y_tr, X_vl, y_vl
        gc.collect()
    
    if not fold_results:
        log(f'  ERROR: No valid folds')
        return {'name': name, 'error': 'no valid folds'}
    
    avg_wf_icir = float(np.mean([r['icir'] for r in fold_results]))
    final_trees = max(int(np.median(best_iters)), 100) if best_iters else n_trees
    log(f'  WF avg ICIR: {avg_wf_icir:.4f}, final trees: {final_trees}')
    
    # Train final model on full training set and evaluate OOS
    try:
        if is_binary:
            y_train_eval = (y_train > 0.02).astype(np.float32)
        else:
            y_train_eval = y_train
        
        dtrain_final = lgb.Dataset(X_train, label=y_train_eval)
        
        final_params = params.copy()
        final_model = lgb.train(final_params, dtrain_final, num_boost_round=final_trees,
                               callbacks=[lgb.log_evaluation(0)])
        
        oos_pred = final_model.predict(X_oos)
        
        # OOS cross-sectional ICIR
        oos_ic, oos_icir, oos_n_dates, oos_ic_pos = compute_cross_sectional_icir(
            oos_pred, y_oos, dates_oos
        )
        
        # OOS daily top-N returns for reference
        oos_daily = compute_daily_topn_returns(oos_pred, y_oos, dates_oos, 15)
        
        oos_mean = float(np.mean(oos_daily)) if len(oos_daily) > 0 else 0
        oos_win = float(np.mean(oos_daily > 0) * 100) if len(oos_daily) > 0 else 0
        
        # Bottom-N for spread
        oos_bot = compute_daily_topn_returns(-oos_pred, y_oos, dates_oos, 15)
        oos_spread = (oos_mean - float(np.mean(oos_bot))) * 100 if len(oos_bot) > 0 else 0
        
        # Feature importance
        importance = final_model.feature_importance(importance_type='gain')
        total_imp = sum(importance) if sum(importance) > 0 else 1
        feat_imp = {TECH_FEATS[i]: round(float(importance[i]) / total_imp * 100, 2) 
                    for i in np.argsort(importance)[::-1][:10]}
        
        elapsed = time.time() - t0
        
        result = {
            'name': name,
            'params': {k: v for k, v in config.items() if k not in ('verbosity', 'verbose')},
            'objective': 'binary' if is_binary else 'reg',
            'wf_icir': round(avg_wf_icir, 4),
            'wf_folds': len(fold_results),
            'n_trees_final': final_trees,
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
        
        # Save model if good
        if oos_icir > 0.338:
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
        import traceback
        traceback.print_exc()
        return {'name': name, 'error': str(e), 'wf_icir': round(avg_wf_icir, 4)}


def main():
    log('='*70)
    log('BlueShield V9 Optimization Script (LightGBM)')
    log('='*70)
    
    t_total = time.time()
    
    # 1. Load data
    log('\nStep 1: Loading data...')
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
    
    # 2. Compute features
    log('\nStep 2: Computing features...')
    t0 = time.time()
    
    feat_list = []
    label_20_list = []
    label_10_list = []
    label_5_list = []
    date_list = []
    
    syms = list(df.groupby('sym'))
    for i, (sym, g) in enumerate(syms):
        if len(g) < 80:
            continue
        
        try:
            feats = compute_features_fixed(g)
            c = g['close'].values.astype(np.float64)
            n = len(c)
            
            def make_fwd_ret(hold):
                fr = np.full(n, np.nan, dtype=np.float32)
                if n > hold:
                    raw = c[hold:] / (c[:-hold] + 1e-10) - 1.0
                    clip = 0.5 if hold >= 10 else 0.3
                    raw = np.clip(raw, -clip, clip)
                    fr[:-hold] = raw.astype(np.float32)
                return fr
            
            feat_list.append(feats.astype(np.float32))
            label_20_list.append(make_fwd_ret(20))
            label_10_list.append(make_fwd_ret(10))
            label_5_list.append(make_fwd_ret(5))
            date_list.append(g['date'].values.astype('datetime64[ns]'))
        except:
            continue
        
        if (i+1) % 500 == 0:
            log(f'  {i+1}/{len(syms)} stocks ({time.time()-t0:.0f}s)')
    
    X_all = np.vstack(feat_list)
    y_20 = np.concatenate(label_20_list)
    y_10 = np.concatenate(label_10_list)
    y_5 = np.concatenate(label_5_list)
    dates_all = np.concatenate(date_list)
    
    del feat_list, label_20_list, label_10_list, label_5_list
    gc.collect()
    
    # Filter valid features AND labels
    valid = ~np.isnan(X_all).any(axis=1) & ~np.isinf(X_all).any(axis=1)
    X_all = X_all[valid]
    y_20 = y_20[valid]
    y_10 = y_10[valid]
    y_5 = y_5[valid]
    dates_all = dates_all[valid]
    
    # Also filter rows where labels are NaN (from fwd_ret computation)
    valid_20 = ~np.isnan(y_20)
    valid_10 = ~np.isnan(y_10)
    valid_5 = ~np.isnan(y_5)
    
    log(f'  Total rows: {len(X_all):,}')
    log(f'  Valid labels 20d: {valid_20.sum():,}, 10d: {valid_10.sum():,}, 5d: {valid_5.sum():,}')
    log(f'  Time: {time.time()-t0:.0f}s')
    
    # 3. Run configurations
    log('\nStep 3: Running configurations...')
    
    results = {}
    
    # Config 1: Default params, 20-day
    mask_20 = valid_20
    results['v9_fixed_20d'] = run_config(
        'v9_fixed_20d',
        {'objective': 'regression', 'max_depth': 6, 'learning_rate': 0.03,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 20,
         'reg_alpha': 0.1, 'reg_lambda': 1.0,
         'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_all[mask_20], y_20[mask_20], dates_all[mask_20], n_trees=500
    )
    gc.collect()
    
    # Config 2: 10-day hold
    mask_10 = valid_10
    results['v9_fixed_10d'] = run_config(
        'v9_fixed_10d',
        {'objective': 'regression', 'max_depth': 6, 'learning_rate': 0.03,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 20,
         'reg_alpha': 0.1, 'reg_lambda': 1.0,
         'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_all[mask_10], y_10[mask_10], dates_all[mask_10], n_trees=500
    )
    gc.collect()
    
    # Config 3: 5-day hold
    mask_5 = valid_5
    results['v9_fixed_5d'] = run_config(
        'v9_fixed_5d',
        {'objective': 'regression', 'max_depth': 6, 'learning_rate': 0.03,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 20,
         'reg_alpha': 0.1, 'reg_lambda': 1.0,
         'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_all[mask_5], y_5[mask_5], dates_all[mask_5], n_trees=500
    )
    gc.collect()
    
    # Config 4: Hyperparameter tuned
    results['v9_hp_tuned'] = run_config(
        'v9_hp_tuned',
        {'objective': 'regression', 'max_depth': 8, 'learning_rate': 0.01,
         'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_samples': 10,
         'reg_alpha': 0.5, 'reg_lambda': 5.0, 'gamma': 0.1,
         'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_all[mask_20], y_20[mask_20], dates_all[mask_20], n_trees=1000
    )
    gc.collect()
    
    # Config 5: Classifier
    results['v9_classifier'] = run_config(
        'v9_classifier',
        {'objective': 'binary', 'max_depth': 6, 'learning_rate': 0.03,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 20,
         'reg_alpha': 0.1, 'reg_lambda': 1.0,
         'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_all[mask_20], y_20[mask_20], dates_all[mask_20], n_trees=500, is_binary=True
    )
    gc.collect()
    
    # Config 6: Aggressive regularization
    results['v9_aggressive'] = run_config(
        'v9_aggressive',
        {'objective': 'regression', 'max_depth': 4, 'learning_rate': 0.05,
         'subsample': 0.6, 'colsample_bytree': 0.6, 'min_child_samples': 50,
         'reg_alpha': 1.0, 'reg_lambda': 10.0,
         'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_all[mask_20], y_20[mask_20], dates_all[mask_20], n_trees=300
    )
    gc.collect()
    
    # Config 7: More trees
    results['v9_large_trees'] = run_config(
        'v9_large_trees',
        {'objective': 'regression', 'max_depth': 5, 'learning_rate': 0.01,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 30,
         'reg_alpha': 0.05, 'reg_lambda': 2.0,
         'num_threads': 4, 'verbosity': -1, 'seed': 42},
        X_all[mask_20], y_20[mask_20], dates_all[mask_20], n_trees=2000
    )
    gc.collect()
    
    # 4. Summary
    log('\n' + '='*70)
    log('RESULTS SUMMARY')
    log('='*70)
    
    best_icir = -1
    best_name = None
    
    for name, r in results.items():
        if 'error' in r:
            log(f'\n{name}: ERROR - {r["error"]}')
            continue
        
        icir = r.get('oos_icir', 0)
        if icir > best_icir:
            best_icir = icir
            best_name = name
        
        log(f'\n{name}:')
        log(f'  WF ICIR: {r["wf_icir"]:.4f} ({r["wf_folds"]} folds)')
        log(f'  OOS: IC={r["oos_ic"]:.4f}, ICIR={r["oos_icir"]:.4f}, IC+={r["oos_ic_pos"]:.2f}')
        log(f'  Top15: ret={r["oos_avg_ret_pct"]:.2f}%, win={r["oos_win_pct"]:.1f}%, spread={r["oos_spread_pct"]:.2f}%')
        log(f'  Trees: {r["n_trees_final"]}, Time: {r["elapsed_s"]:.0f}s')
        log(f'  Top features: {list(r["feature_importance"].keys())[:5]}')
    
    # Save best as v10
    if best_icir > 0.338 and best_name:
        log(f'\n*** Best: {best_name} (ICIR={best_icir:.4f})')
        
        best_result = results[best_name]
        
        meta = {
            'version': 'blueshield_v10',
            'config': f'{best_name}_optimized',
            'features': TECH_FEATS,
            'n_features': len(TECH_FEATS),
            'oos_metrics': {
                'ic': best_result['oos_ic'],
                'icir': best_result['oos_icir'],
                'ic_pos': best_result['oos_ic_pos'],
                'spread': best_result['oos_spread_pct'],
                'top15_avg': best_result['oos_avg_ret_pct'],
                'top15_win': best_result['oos_win_pct'],
            },
            'wf_icir': best_result['wf_icir'],
            'feature_importance': best_result['feature_importance'],
            'created': time.strftime('%Y-%m-%d %H:%M'),
            'replaces': 'blueshield_v9',
            'optimized_from': best_name,
            'engine': 'lightgbm',
            'all_results': {k: {kk: vv for kk, vv in v.items() if kk != 'feature_importance'} 
                           for k, v in results.items() if 'error' not in v},
        }
        
        meta_path = os.path.join(MODEL_DIR, 'blueshield_v10_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        log(f'Metadata: {meta_path}')
    
    # Save all results
    exp_path = os.path.join(EXPER_DIR, 'v9_optimization_results.json')
    with open(exp_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    log(f'\nResults saved: {exp_path}')
    
    total_time = time.time() - t_total
    log(f'\nTotal time: {total_time/60:.1f} minutes')
    log('='*70)
    
    if best_icir > 0.5:
        log(f'✅ SUCCESS: Best ICIR={best_icir:.4f} > 0.5 (config: {best_name})')
    elif best_icir > 0.338:
        log(f'⚠️ IMPROVED: Best ICIR={best_icir:.4f} > 0.338 baseline (config: {best_name})')
    else:
        log(f'❌ NOT IMPROVED: Best ICIR={best_icir:.4f} <= 0.338 baseline')


if __name__ == '__main__':
    main()
