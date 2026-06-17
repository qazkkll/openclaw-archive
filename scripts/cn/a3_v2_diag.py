#!/usr/bin/env python3
"""
A3_v2 特征诊断 + 市场状态特征 Walk-Forward对比
借鉴参数扫描脚本架构：先缓存特征 → 逐fold评估
关键改进：
  1. 用1000只流动性好的股票代替全量4580只        
  2. 8个市场状态特征（沪深300）
  3. 对比A3_v1（33特征）vs A3_v2（33+8=41特征）
"""
import sys, io, json, time, os, gc, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
print = lambda *a, **kw: (__import__('builtins').print(*a, **dict({'flush': True}, **kw)))
import xgboost as xgb

BASE = r'/home/hermes/.hermes/openclaw-archive/data'
MODEL_V1 = os.path.join(BASE, 'a1_models', 'a3_v1.json')
HIST_PATH = os.path.join(BASE, 'a_hist_10y.parquet')
# 统一路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import INDEX_300
INDEX_PATH = INDEX_300
CACHE_DIR = os.path.join(BASE, 'a1_models')
REPORT_PATH = os.path.join(CACHE_DIR, 'a3_v2_report.json')

V1_FEATURES = [
    'pct_ma5','pct_ma10','pct_ma20','pct_ma60','pct_ma120',
    'ma20_slope','ma60_slope','ma_align',
    'vol_10d','vol_60d','vol_ratio','atr20_pct',
    'ret_1d','ret_5d','ret_10d','ret_20d','ret_60d','rsi14',
    'vol_ratio_5_20','kdj_k','kdj_d','kdj_j',
    'macd_dif','macd_dea','macd_bar','bb_width','bb_position',
    'obv_ratio_5_20','ret5_max','ret3_vs_ema12','accel_5_10',
    'ma5_ma10_cross','vol_breakout',
]
MKT_FEATURES = ['mkt_ret_5d','mkt_ret_20d','mkt_ma20_trend','mkt_ma60_trend',
                'mkt_vol_20d','mkt_vol_ratio','mkt_rsi14','mkt_regime']

MAX_STOCKS = 1000
HOLD_DAYS = 10
TOP_K = 20
SL_PCT = -15.0

WF_SPLITS = [
    ('fold1', '20160101', '20171231'),
    ('fold2', '20160101', '20181231'),
    ('fold3', '20160101', '20201231'),
    ('fold4', '20160101', '20221231'),
    ('fold5', '20160101', '20240831'),
]

# ─── Feature helpers ───
def _ema_np(arr, period):
    alpha = 2.0 / (period + 1)
    r = np.empty_like(arr, dtype=np.float64)
    r[0] = arr[0]
    for i in range(1, len(arr)):
        r[i] = arr[i] * alpha + r[i-1] * (1 - alpha)
    return r

def compute_v1_features(c, h, l, o, v):
    """33 tech features, returns (n, 33) or None"""
    n = len(c)
    if n < 120:
        return None
    c = c.astype(np.float64); h = h.astype(np.float64)
    l = l.astype(np.float64); v = v.astype(np.float64)
    
    def ma(arr, w):
        cs = np.cumsum(arr)
        cs[w:] = cs[w:] - cs[:-w]
        r = np.full(n, np.nan)
        r[w-1:] = cs[w-1:] / w
        return r
    
    ma5 = ma(c,5); ma10 = ma(c,10); ma20 = ma(c,20); ma60 = ma(c,60); ma120 = ma(c,120)
    ma120 = np.where(np.isnan(ma120), ma60, ma120)
    
    # pct_ma
    pct_ma5 = (c/ma5 - 1)*100; pct_ma10 = (c/ma10 - 1)*100
    pct_ma20 = (c/ma20 - 1)*100; pct_ma60 = (c/ma60 - 1)*100; pct_ma120 = (c/ma120 - 1)*100
    
    # slopes (20-day diff)
    ma20_slope = np.full(n, np.nan); ma60_slope = np.full(n, np.nan)
    ma20_slope[20:] = (ma20[20:]/ma20[:-20] - 1)*100
    ma60_slope[60:] = (ma60[60:]/ma60[:-60] - 1)*100
    
    # ma_align
    bull = (ma5>ma10)&(ma10>ma20)&(ma20>ma60)
    bear = (ma5<ma10)&(ma10<ma20)&(ma20<ma60)
    ma_align = np.where(bull, 1., np.where(bear, -1., 0.))
    
    # volume
    vol_5d = ma(v,5); vol_10d = ma(v,10); vol_20d = ma(v,20); vol_60d = ma(v,60)
    vol_ratio = vol_10d / np.maximum(vol_60d, 1)
    vol_ratio_5_20 = vol_5d / np.maximum(vol_20d, 1)
    
    # ATR
    prev_c = np.roll(c,1); prev_c[0] = c[0]
    tr = np.maximum(h-l, np.maximum(np.abs(h-prev_c), np.abs(l-prev_c)))
    atr20 = ma(tr, 20)
    atr20_pct = atr20 / c * 100
    
    # returns
    ret_1d = np.full(n, np.nan); ret_1d[1:] = (c[1:]/c[:-1]-1)*100
    ret_5d = np.full(n, np.nan); ret_5d[5:] = (c[5:]/c[:-5]-1)*100
    ret_10d = np.full(n, np.nan); ret_10d[10:] = (c[10:]/c[:-10]-1)*100
    ret_20d = np.full(n, np.nan); ret_20d[20:] = (c[20:]/c[:-20]-1)*100
    ret_60d = np.full(n, np.nan); ret_60d[60:] = (c[60:]/c[:-60]-1)*100
    
    # RSI14
    log_ret = np.zeros(n); log_ret[1:] = np.log(c[1:]/c[:-1])
    gains = np.maximum(log_ret, 0); losses = np.maximum(-log_ret, 0)
    rsi14 = np.full(n, np.nan)
    ag = np.mean(gains[1:15]); al = np.mean(losses[1:15])
    rsi14[14] = 100 - 100/(1+ag/max(al,1e-10))
    for i in range(15, n):
        ag = (ag*13 + gains[i])/14; al = (al*13 + losses[i])/14
        rsi14[i] = 100 - 100/(1+ag/max(al,1e-10))
    
    # KDJ
    kdj_k = np.full(n,np.nan); kdj_d = np.full(n,np.nan); kdj_j = np.full(n,np.nan)
    if n >= 9:
        from numpy.lib.stride_tricks import sliding_window_view
        h9 = sliding_window_view(h,9); l9 = sliding_window_view(l,9)
        hh = h9.max(axis=1); ll = l9.min(axis=1)
        denom = hh-ll; rsv = np.full(n,np.nan)
        rsv[8:] = np.where(denom>0, (c[8:]-ll)/denom*100, 50)
        k=d=50.
        for i in range(8,n):
            if not np.isnan(rsv[i]):
                k = 2/3*k + 1/3*rsv[i]
                d = 2/3*d + 1/3*k
            kdj_k[i]=k; kdj_d[i]=d; kdj_j[i]=3*k-2*d
    
    # MACD
    ema12 = _ema_np(c,12); ema26 = _ema_np(c,26)
    macd_dif = (ema12-ema26)/c*100
    macd_dea = _ema_np(macd_dif,9)
    macd_bar = 2*(macd_dif-macd_dea)
    
    # Bollinger
    bb_ma = ma20
    roll_var = np.zeros(n); roll_var[:] = np.nan
    for i in range(19, n):
        roll_var[i] = np.var(c[i-19:i+1], ddof=1)
    bb_std = np.sqrt(roll_var)
    bb_width = bb_std/bb_ma*100
    bb_position = (c-bb_ma)/np.maximum(bb_std*2, 1e-6)
    
    # OBV
    sign = np.sign(c - prev_c); sign[0] = 0
    obv = np.cumsum(sign * v)
    obv_ratio_5_20 = ma(obv,5) / np.maximum(ma(obv,20), 1)
    
    # ret5_max, ret3_vs_ema12, accel_5_10, cross, vol_breakout
    ret5_max = np.full(n, np.nan)
    for i in range(5, n):
        ret5_max[i] = np.max(ret_5d[i-4:i+1])
    ema12_c = _ema_np(c,12)
    ret3_vs_ema12 = np.full(n, np.nan)
    ret3_vs_ema12[3:] = (c[3:]/ema12_c[:-3] - 1)*100
    accel_5_10 = np.full(n, np.nan); accel_5_10[10:] = ret_5d[10:] - ret_10d[10:]
    ma5_ma10_cross = np.where(ma5 > ma10, 1., 0.)
    vol_breakout = v / np.maximum(ma(v, 20), 1)
    
    feat = np.column_stack([
        pct_ma5, pct_ma10, pct_ma20, pct_ma60, pct_ma120,
        ma20_slope, ma60_slope, ma_align,
        vol_10d, vol_60d, vol_ratio, atr20_pct,
        ret_1d, ret_5d, ret_10d, ret_20d, ret_60d, rsi14,
        vol_ratio_5_20, kdj_k, kdj_d, kdj_j,
        macd_dif, macd_dea, macd_bar, bb_width, bb_position,
        obv_ratio_5_20, ret5_max, ret3_vs_ema12, accel_5_10,
        ma5_ma10_cross, vol_breakout,
    ])
    return feat  # (n, 33)


def build_market_features():
    """Compute 8 market state features from CSI 300"""
    print("\n[2] Building market features...", flush=True)
    t0 = time.time()
    idx = json.load(open(INDEX_PATH, 'r', encoding='utf-8'))
    dates = np.array([d['trade_date'] for d in idx])  # sorted 20160104~20260608
    close = np.array([d['close'] for d in idx], dtype=np.float64)
    n = len(dates)
    
    # ret_5d, ret_20d
    ret_5d = np.full(n, np.nan); ret_20d = np.full(n, np.nan)
    ret_5d[5:] = (close[5:]/close[:-5]-1)*100
    ret_20d[20:] = (close[20:]/close[:-20]-1)*100
    
    # MA trends
    def ma(arr,w):
        cs=np.cumsum(arr); cs[w:]=cs[w:]-cs[:-w]
        r=np.full(n,np.nan); r[w-1:]=cs[w-1:]/w; return r
    ma20=ma(close,20); ma60=ma(close,60)
    ma20_trend = np.where(close>ma20, 1., 0.)
    ma60_trend = np.where(close>ma60, 1., 0.)
    
    # volatility
    log_r = np.zeros(n); log_r[1:] = np.log(close[1:]/close[:-1])
    vol_20d = np.full(n, np.nan)
    for i in range(19, n):
        vol_20d[i] = np.std(log_r[i-19:i+1])
    vol_5d = np.full(n, np.nan)
    for i in range(4, n):
        vol_5d[i] = np.std(log_r[i-4:i+1])
    vol_ratio = vol_5d / np.maximum(vol_20d, 1e-6)
    
    # RSI14 on market
    gains=np.maximum(log_r,0); losses=np.maximum(-log_r,0)
    rsi14=np.full(n,np.nan)
    ag=np.mean(gains[1:15]); al=np.mean(losses[1:15])
    rsi14[14]=100-100/(1+ag/max(al,1e-10))
    for i in range(15, n):
        ag=(ag*13+gains[i])/14; al=(al*13+losses[i])/14
        rsi14[i]=100-100/(1+ag/max(al,1e-10))
    
    # Regime: 2=attack (ma20+ma60 above), 1=neutral, 0=defense
    regime = np.where(ma20_trend+ma60_trend>=2, 2., np.where(ma20_trend+ma60_trend==0, 0., 1.))
    
    # Build lookup dict date -> array(8)
    mkt_dict = {}
    for i in range(n):
        mkt_dict[dates[i]] = np.array([
            ret_5d[i], ret_20d[i], ma20_trend[i], ma60_trend[i],
            vol_20d[i], vol_ratio[i], rsi14[i], regime[i]
        ])
    print(f"  Index: {n} days, {time.time()-t0:.1f}s", flush=True)
    return mkt_dict


def build_feature_cache(codes_use, hist, mkt_dict):
    """Compute 33+8=41 features for selected stocks, save as npz cache"""
    mkt_dates = np.array(sorted(mkt_dict.keys()))
    mkt_vals = np.array([mkt_dict[d] for d in mkt_dates])
    
    cache = {}  # code -> {feats, fwd, dates}
    skipped = 0
    t0 = time.time()
    for idx, code in enumerate(codes_use):
        if (idx+1)%200 == 0:
            print(f"    [{idx+1}/{len(codes_use)}] {time.time()-t0:.0f}s", flush=True)
        rec = hist[code]
        c = np.array(rec['c'], dtype=np.float64)
        n = len(c)
        if n < 120:
            skipped += 1; continue
        feats_tech = compute_v1_features(c, np.array(rec['h'],dtype=np.float64),
                                          np.array(rec['l'],dtype=np.float64),
                                          np.array(rec['o'],dtype=np.float64),
                                          np.array(rec['v'],dtype=np.float64))
        if feats_tech is None:
            skipped += 1; continue
        
        # Market features via searchsorted
        dates = np.array(rec['dates'])
        idxs = np.searchsorted(mkt_dates, dates)
        idxs = np.clip(idxs, 0, len(mkt_dates)-1)
        matched = mkt_dates[idxs] == dates
        mkt_arr = np.full((n, len(MKT_FEATURES)), np.nan)
        mkt_arr[matched] = mkt_vals[idxs[matched]]
        
        X = np.hstack([feats_tech, mkt_arr])  # (n, 41)
        
        # Target: 10-day forward return
        fwd = np.full(n, np.nan)
        if HOLD_DAYS < n:
            fwd[:n-HOLD_DAYS] = (c[HOLD_DAYS:] / c[:n-HOLD_DAYS] - 1) * 100
        
        valid = ~np.any(np.isnan(X), axis=1) & ~np.isnan(fwd)
        if np.sum(valid) < 50:
            skipped += 1; continue
        
        cache[code] = {
            'X': X[valid], 'y': fwd[valid], 'dates': dates[valid]
        }
    
    print(f"  Cached: {len(cache)} stocks, skipped {skipped}, {time.time()-t0:.0f}s", flush=True)
    return cache


def walk_forward_eval(model, cache, label="", show_folds=True):
    """Walk-forward evaluation: train_end splits, daily top-k"""
    results = []
    for fold_name, ts, te in WF_SPLITS:
        train_X, train_y = [], []
        test_X_all, test_y_all, test_dates_all = [], [], []
        for code, d in cache.items():
            dates = d['dates']
            mask_train = (dates >= ts) & (dates <= te)
            mask_test = dates > te
            if np.sum(mask_train) > 0:
                train_X.append(d['X'][mask_train])
                train_y.append(d['y'][mask_train])
            if np.sum(mask_test) > 0:
                test_X_all.append(d['X'][mask_test])
                test_y_all.append(d['y'][mask_test])
                test_dates_all.extend(dates[mask_test].tolist())
        
        # Train
        if len(train_X) == 0:
            continue
        X_tr = np.vstack(train_X); y_tr = np.concatenate(train_y)
        n_feats = X_tr.shape[1]
        feat_names = V1_FEATURES[:n_feats] + (MKT_FEATURES if n_feats > len(V1_FEATURES) else [])
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
        params = {'objective':'reg:squarederror', 'eval_metric':'rmse',
                  'max_depth':6, 'eta':0.1, 'subsample':0.8, 'colsample_bytree':0.8,
                  'verbosity':0, 'seed':42}
        if len(X_tr) > 300000:
            idxs = np.random.RandomState(42).permutation(len(X_tr))[:300000]
            X_tr_sub = X_tr[idxs]; y_tr_sub = y_tr[idxs]
            dtrain = xgb.DMatrix(X_tr_sub, label=y_tr_sub, feature_names=feat_names)
        fold_model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
        
        # Evaluate per test date: top-k daily
        if len(test_X_all) == 0:
            continue
        X_test = np.vstack(test_X_all)
        y_test = np.concatenate(test_y_all)
        test_dates = test_dates_all
        udates = sorted(set(test_dates))
        d_rets, d_wins = [], []
        for d in udates:
            mask = np.array(test_dates) == d
            dp = fold_model.predict(xgb.DMatrix(X_test[mask], feature_names=feat_names))
            ty = y_test[mask]
            if len(dp) < TOP_K: continue
            top_i = np.argsort(dp)[-TOP_K:]
            top_y = ty[top_i]
            capped = np.where(top_y < SL_PCT, SL_PCT, top_y)
            d_rets.append(np.mean(capped))
            d_wins.append(np.mean(top_y > 0) * 100)
        
        if d_rets:
            ar, aw = np.mean(d_rets), np.mean(d_wins)
            if show_folds:
                print(f"  {fold_name:>5} | {te:>8} | {len(d_rets):>5}d | {ar:>5.2f}% | {aw:>4.1f}%", flush=True)
            results.append({'fold': fold_name, 'train_end': te, 'n_days': len(d_rets),
                           'avg_ret': round(ar, 2), 'win_rate': round(aw, 1)})
    
    if results and show_folds:
        f1_4 = [r for r in results if r['fold'] in ('fold1','fold2','fold3','fold4')]
        if f1_4:
            print(f"  Fold 1-4 avg: {np.mean([r['avg_ret'] for r in f1_4]):.2f}%  "
                  f"win={np.mean([r['win_rate'] for r in f1_4]):.1f}%", flush=True)
    return results


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("  A3_v2: Feature Diagnosis + Market State Features", flush=True)
    print("=" * 60, flush=True)
    
    # ── [1] Feature Importance ──
    print("\n[1] A3_v1 Feature Importance", flush=True)
    bst = xgb.Booster(); bst.load_model(MODEL_V1)
    scores = bst.get_score(importance_type='gain')
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"  {len(bst.feature_names)} features, {len(scores)} with nonzero gain", flush=True)
    print("  Top 15:", flush=True)
    for i,(f,g) in enumerate(ranked[:15]):
        print(f"    {i+1:2d}. {f:20s}  gain={g:.1f}", flush=True)
    print("  Bottom 5:", flush=True)
    for i,(f,g) in enumerate(ranked[-5:]):
        print(f"    {len(ranked)-4+i:2d}. {f:20s}  gain={g:.1f}", flush=True)
    
    # ── [2] Market features ──
    mkt_dict = build_market_features()
    
    # ── [3] Load stocks (top 1000) ──
    print("\n[3] Loading stock data...", flush=True)
    hist = json.load(open(HIST_PATH, 'r', encoding='utf-8'))
    codes = list(hist.keys())
    # Pick stocks with most data points (likely most liquid)
    codes.sort(key=lambda c: len(hist[c]['c']), reverse=True)
    codes_use = codes[:MAX_STOCKS]
    print(f"  Total: {len(codes)}, Selected: {len(codes_use)}", flush=True)
    
    # ── [4] Build feature cache ──
    print("\n[4] Building feature cache (41 features per stock)...", flush=True)
    cache = build_feature_cache(codes_use, hist, mkt_dict)
    del hist
    gc.collect()
    
    # ── [5] Walk-Forward: A3_v1 (33 tech features only) ──
    print(f"\n[5] Walk-Forward A3_v1 (33 tech features)")
    print(f"  hold={HOLD_DAYS}d, top_k={TOP_K}, SL={SL_PCT}%", flush=True)
    print(f"  {'Fold':>5} | {'End':>8} | {'Days':>5} | {'AvgR':>6} | {'WinR':>5}", flush=True)
    print("  " + "-" * 42, flush=True)
    # Strip market features from cache for V1 baseline
    cache_v1 = {}
    for code, d in cache.items():
        cache_v1[code] = {'X': d['X'][:,:33], 'y': d['y'], 'dates': d['dates']}
    results_v1 = walk_forward_eval(None, cache_v1, label="V1")
    
    # ── [6] Walk-Forward: A3_v2 (41 features: 33 tech + 8 market) ──
    print(f"\n[6] Walk-Forward A3_v2 (41 features)")
    print(f"  {'Fold':>5} | {'End':>8} | {'Days':>5} | {'AvgR':>6} | {'WinR':>5}", flush=True)
    print("  " + "-" * 42, flush=True)
    results_v2 = walk_forward_eval(None, cache, label="A3_v2")
    
    # ── [7] Final comparison ──
    print("\n" + "=" * 60)
    print("  Comparison: A3_v1 vs A3_v2")
    print("=" * 60)
    print(f"  {'Metric':<20} {'A3_v1(33f)':>12} {'A3_v2(41f)':>12} {'Δ':>10}", flush=True)
    print("  " + "-" * 54, flush=True)
    v1_avg = np.mean([r['avg_ret'] for r in results_v1])
    v2_avg = np.mean([r['avg_ret'] for r in results_v2])
    v1_w = np.mean([r['win_rate'] for r in results_v1])
    v2_w = np.mean([r['win_rate'] for r in results_v2])
    v1_nb = np.mean([r['avg_ret'] for r in results_v1 if r['fold']!='fold5'])
    v2_nb = np.mean([r['avg_ret'] for r in results_v2 if r['fold']!='fold5'])
    print(f"  {'Avg Return':<20} {v1_avg:>10.2f}% {v2_avg:>10.2f}% {v2_avg-v1_avg:>+8.2f}%", flush=True)
    print(f"  {'Avg Win Rate':<20} {v1_w:>9.1f}% {v2_w:>9.1f}% {v2_w-v1_w:>+7.1f}%", flush=True)
    print(f"  {'Avg (no boom fold)':<20} {v1_nb:>10.2f}% {v2_nb:>10.2f}% {v2_nb-v1_nb:>+8.2f}%", flush=True)
    
    # Feature importance of V2
    # Train final model on all data for feature importance
    print("\n[7] A3_v2 Feature Importance (final model)", flush=True)
    all_X, all_y = [], []
    for code, d in cache.items():
        all_X.append(d['X']); all_y.append(d['y'])
    X_all = np.vstack(all_X); y_all = np.concatenate(all_y)
    if len(X_all) > 500000:
        idxs = np.random.RandomState(42).permutation(len(X_all))[:500000]
        X_all = X_all[idxs]; y_all = y_all[idxs]
    dtrain = xgb.DMatrix(X_all, label=y_all, feature_names=V1_FEATURES+MKT_FEATURES)
    params = {'objective':'reg:squarederror', 'eval_metric':'rmse',
              'max_depth':6, 'eta':0.1, 'subsample':0.8, 'colsample_bytree':0.8,
              'verbosity':0, 'seed':42}
    final_model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
    scores_v2 = final_model.get_score(importance_type='gain')
    ranked_v2 = sorted(scores_v2.items(), key=lambda x: x[1], reverse=True)
    print("  Top 20:", flush=True)
    for i,(f,g) in enumerate(ranked_v2[:20]):
        tag = " [MKT]" if f in MKT_FEATURES else ""
        print(f"    {i+1:2d}. {f:20s}  gain={g:.1f}{tag}", flush=True)
    
    # Save report
    report = {
        'model': 'A3_v2', 'n_features': 41, 'n_market': 8, 'n_stocks': len(cache),
        'params': {'hold_days': HOLD_DAYS, 'top_k': TOP_K, 'sl_pct': SL_PCT},
        'v1_feature_importance_top15': [(f, round(g,1)) for f,g in ranked[:15]],
        'v2_feature_importance_top20': [(f, round(g,1)) for f,g in ranked_v2[:20]],
        'market_features_in_top20': sum(1 for f,_ in ranked_v2[:20] if f in MKT_FEATURES),
        'walk_forward_v1': results_v1,
        'walk_forward_v2': results_v2,
        'summary': {
            'v1_avg_ret': round(v1_avg, 2),
            'v2_avg_ret': round(v2_avg, 2),
            'v1_win_rate': round(v1_w, 1),
            'v2_win_rate': round(v2_w, 1),
            'v1_no_boom': round(v1_nb, 2),
            'v2_no_boom': round(v2_nb, 2),
        }
    }
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT_PATH}", flush=True)
    print(f"\n  Total: {time.time()-t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
