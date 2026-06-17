"""
A3_v2 特征诊断 + 市场状态特征 + Walk-Forward重训
Optimized: vectorized features, minimal Python loops
"""
import json, time, sys
import numpy as np
from pathlib import Path

MODEL_V1 = Path(r"/home/hermes/.hermes/openclaw-archive/data\a1_models\a3_v1.json")
HIST_FILE = Path(r"/home/hermes/.hermes/openclaw-archive/data\a_hist_10y.parquet")
# 统一路径
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import INDEX_300
INDEX300 = Path(INDEX_300)
REPORT_OUT = Path(r"/home/hermes/.hermes/openclaw-archive/data\a1_models\a3_v2_report.json")

import xgboost as xgb

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

MKT_FEATURES = [
    'mkt_ret_5d', 'mkt_ret_20d', 'mkt_ma20_trend', 'mkt_ma60_trend',
    'mkt_vol_20d', 'mkt_vol_ratio', 'mkt_rsi14', 'mkt_regime'
]
V1_FEATURES = [
    'pct_ma5', 'pct_ma10', 'pct_ma20', 'pct_ma60', 'pct_ma120',
    'ma20_slope', 'ma60_slope', 'ma_align',
    'vol_10d', 'vol_60d', 'vol_ratio', 'atr20_pct',
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d',
    'rsi14', 'vol_ratio_5_20',
    'kdj_k', 'kdj_d', 'kdj_j',
    'macd_dif', 'macd_dea', 'macd_bar',
    'bb_width', 'bb_position',
    'obv_ratio_5_20', 'ret5_max', 'ret3_vs_ema12',
    'accel_5_10', 'ma5_ma10_cross', 'vol_breakout'
]
ALL_FEATURES = V1_FEATURES + MKT_FEATURES


def ma_cumsum(arr, w):
    """Fast MA via cumsum. Returns array same length, NaN for first w-1."""
    cs = np.cumsum(arr)
    out = np.empty_like(arr, dtype=np.float64)
    out[:w-1] = np.nan
    out[w-1] = cs[w-1] / w
    out[w:] = (cs[w:] - cs[:-w]) / w
    return out


def ema_ewma(arr, span):
    """EMA using pandas-style ewma logic, pure numpy."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    # Use iterative approach but with compiled numpy ops
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out


def compute_features_fast(c, h, l, o, v):
    """Compute 33 technical features. All vectorized except KDJ/MACD/RSI which need sequential."""
    n = len(c)
    if n < 120:
        return None

    # MAs
    ma5 = ma_cumsum(c, 5)
    ma10 = ma_cumsum(c, 10)
    ma20 = ma_cumsum(c, 20)
    ma60 = ma_cumsum(c, 60)
    ma120 = ma_cumsum(c, 120)

    # 1-5: pct_ma
    pct_ma5 = (c / ma5 - 1) * 100
    pct_ma10 = (c / ma10 - 1) * 100
    pct_ma20 = (c / ma20 - 1) * 100
    pct_ma60 = (c / ma60 - 1) * 100
    pct_ma120 = (c / ma120 - 1) * 100

    # 6-7: slopes
    ma20_slope = np.full(n, np.nan)
    ma20_slope[20:] = (ma20[20:] / ma20[:-20] - 1) * 100
    ma60_slope = np.full(n, np.nan)
    ma60_slope[60:] = (ma60[60:] / ma60[:-60] - 1) * 100

    # 8: ma_align
    bull = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)
    bear = (ma5 < ma10) & (ma10 < ma20) & (ma20 < ma60)
    ma_align = np.where(bull, 1.0, np.where(bear, -1.0, 0.0))

    # 9-11: volume
    vol_10d = ma_cumsum(v, 10)
    vol_60d = ma_cumsum(v, 60)
    vol_5d = ma_cumsum(v, 5)
    vol_20d = ma_cumsum(v, 20)
    vol_ratio = vol_10d / np.where(vol_60d > 0, vol_60d, 1)
    vol_ratio_5_20 = vol_5d / np.where(vol_20d > 0, vol_20d, 1)

    # 12: atr20_pct
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr20 = ma_cumsum(tr, 20)
    atr20_pct = atr20 / c * 100

    # 13-17: returns
    ret_1d = np.full(n, np.nan); ret_1d[1:] = (c[1:] / c[:-1] - 1) * 100
    ret_5d = np.full(n, np.nan); ret_5d[5:] = (c[5:] / c[:-5] - 1) * 100
    ret_10d = np.full(n, np.nan); ret_10d[10:] = (c[10:] / c[:-10] - 1) * 100
    ret_20d = np.full(n, np.nan); ret_20d[20:] = (c[20:] / c[:-20] - 1) * 100
    ret_60d = np.full(n, np.nan); ret_60d[60:] = (c[60:] / c[:-60] - 1) * 100
    ret_3d = np.full(n, np.nan); ret_3d[3:] = (c[3:] / c[:-3] - 1) * 100

    # 18: rsi14 (sequential but fast)
    log_ret = np.zeros(n); log_ret[1:] = np.log(c[1:] / c[:-1])
    gains = np.maximum(log_ret, 0)
    losses = np.maximum(-log_ret, 0)
    rsi14 = np.full(n, np.nan)
    ag = np.mean(gains[1:15])
    al = np.mean(losses[1:15])
    rsi14[14] = 100 - 100 / (1 + ag / max(al, 1e-10))
    for i in range(15, n):
        ag = (ag * 13 + gains[i]) / 14
        al = (al * 13 + losses[i]) / 14
        rsi14[i] = 100 - 100 / (1 + ag / max(al, 1e-10))

    # 20-22: KDJ (sequential)
    kdj_k = np.full(n, np.nan)
    kdj_d = np.full(n, np.nan)
    kdj_j = np.full(n, np.nan)
    # Rolling min/max for period=9
    from numpy.lib.stride_tricks import sliding_window_view
    if n >= 9:
        h_roll = sliding_window_view(h, 9)
        l_roll = sliding_window_view(l, 9)
        hh = h_roll.max(axis=1)
        ll = l_roll.min(axis=1)
        rsv = np.full(n, np.nan)
        denom = hh - ll
        valid = denom > 0
        rsv[8:8+len(valid)] = np.where(valid, (c[8:] - ll) / np.where(valid, denom, 1) * 100, 50)
        k_val, d_val = 50.0, 50.0
        for i in range(8, n):
            if not np.isnan(rsv[i]):
                k_val = 2/3 * k_val + 1/3 * rsv[i]
                d_val = 2/3 * d_val + 1/3 * k_val
            kdj_k[i] = k_val
            kdj_d[i] = d_val
            kdj_j[i] = 3 * k_val - 2 * d_val

    # 23-25: MACD (sequential)
    ema12 = ema_ewma(c, 12)
    ema26 = ema_ewma(c, 26)
    macd_dif = (ema12 - ema26) / c * 100
    dea = ema_ewma(macd_dif, 9)
    macd_dea = dea / c * 100
    macd_bar = 2 * (macd_dif - macd_dea)

    # 26-27: Bollinger (vectorized with rolling std)
    bb_width = np.full(n, np.nan)
    bb_position = np.full(n, np.nan)
    if n >= 20:
        c_roll = sliding_window_view(c, 20)
        std20 = c_roll.std(axis=1)
        mid20 = c_roll.mean(axis=1)
        valid_std = std20 > 0
        bb_width[19:] = np.where(valid_std, 4 * std20 / mid20 * 100, np.nan)
        lower = mid20 - 2 * std20
        upper = mid20 + 2 * std20
        bb_position[19:] = np.where(valid_std, (c[19:] - lower) / (upper - lower), np.nan)

    # 28: obv_ratio_5_20
    obv = np.zeros(n)
    price_up = c[1:] > c[:-1]
    price_down = c[1:] < c[:-1]
    for i in range(1, n):
        if price_up[i-1]:
            obv[i] = obv[i-1] + v[i]
        elif price_down[i-1]:
            obv[i] = obv[i-1] - v[i]
        else:
            obv[i] = obv[i-1]
    obv_ma5 = ma_cumsum(obv, 5)
    obv_ma20 = ma_cumsum(obv, 20)
    obv_ratio_5_20 = obv_ma5 / np.where(np.abs(obv_ma20) > 0, obv_ma20, 1)

    # 29: ret5_max
    ret5_max = np.full(n, np.nan)
    if n >= 6:
        c_roll6 = sliding_window_view(c, 6)
        daily_rets = c_roll6[:, 1:] / c_roll6[:, :-1] - 1
        ret5_max[5:] = daily_rets.max(axis=1) * 100

    # 30: ret3_vs_ema12
    ema12_ret3 = np.full(n, np.nan)
    ema12_ret3[3:] = (ema12[3:] / ema12[:-3] - 1) * 100
    ret3_vs_ema12 = ret_3d - ema12_ret3

    # 31: accel_5_10
    accel_5_10 = ret_5d - ret_10d / 2

    # 32: ma5_ma10_cross
    ma5_ma10_cross = np.zeros(n)
    if n >= 2:
        diff_prev = ma5[:-1] - ma10[:-1]
        diff_curr = ma5[1:] - ma10[1:]
        golden = (diff_prev < 0) & (diff_curr > 0) & ~np.isnan(diff_prev) & ~np.isnan(diff_curr)
        death = (diff_prev > 0) & (diff_curr < 0) & ~np.isnan(diff_prev) & ~np.isnan(diff_curr)
        ma5_ma10_cross[1:] = np.where(golden, 1.0, np.where(death, -1.0, 0.0))

    # 33: vol_breakout
    vol_breakout = np.where(vol_5d > 2 * vol_60d, 1.0, 0.0)

    return np.column_stack([
        pct_ma5, pct_ma10, pct_ma20, pct_ma60, pct_ma120,
        ma20_slope, ma60_slope, ma_align,
        vol_10d, vol_60d, vol_ratio, atr20_pct,
        ret_1d, ret_5d, ret_10d, ret_20d, ret_60d,
        rsi14, vol_ratio_5_20,
        kdj_k, kdj_d, kdj_j,
        macd_dif, macd_dea, macd_bar,
        bb_width, bb_position,
        obv_ratio_5_20, ret5_max, ret3_vs_ema12,
        accel_5_10, ma5_ma10_cross, vol_breakout
    ])


def build_market_features():
    print("=" * 60)
    print("Part 2: Market features (CSI 300)")
    print("=" * 60, flush=True)

    raw = json.load(open(INDEX300, 'r'))
    raw.sort(key=lambda x: x['trade_date'])
    n = len(raw)
    closes = np.array([r['close'] for r in raw])
    vols = np.array([r['vol'] for r in raw])
    dates = [r['trade_date'] for r in raw]
    print(f"  {n} days: {dates[0]} ~ {dates[-1]}", flush=True)

    log_ret = np.zeros(n); log_ret[1:] = np.log(closes[1:] / closes[:-1])

    mkt_ret_5d = np.full(n, np.nan)
    mkt_ret_5d[5:] = (closes[5:] / closes[:-5] - 1) * 100
    mkt_ret_20d = np.full(n, np.nan)
    mkt_ret_20d[20:] = (closes[20:] / closes[:-20] - 1) * 100

    ma20 = ma_cumsum(closes, 20)
    ma60 = ma_cumsum(closes, 60)
    mkt_ma20 = np.where(closes > ma20, 1.0, 0.0); mkt_ma20[:19] = np.nan
    mkt_ma60 = np.where(closes > ma60, 1.0, 0.0); mkt_ma60[:59] = np.nan

    mkt_vol_20d = np.full(n, np.nan)
    for i in range(20, n):
        mkt_vol_20d[i] = np.std(log_ret[i-19:i+1]) * np.sqrt(252) * 100

    mkt_vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        v5 = np.std(log_ret[i-4:i+1]); v20 = np.std(log_ret[i-19:i+1])
        if v20 > 0: mkt_vol_ratio[i] = v5 / v20

    gains = np.maximum(log_ret, 0); losses = np.maximum(-log_ret, 0)
    mkt_rsi = np.full(n, np.nan)
    ag = np.mean(gains[1:15]); al = np.mean(losses[1:15])
    mkt_rsi[14] = 100 - 100 / (1 + ag / max(al, 1e-10))
    for i in range(15, n):
        ag = (ag * 13 + gains[i]) / 14; al = (al * 13 + losses[i]) / 14
        mkt_rsi[i] = 100 - 100 / (1 + ag / max(al, 1e-10))

    mkt_regime = np.where(np.isnan(mkt_ma20) | np.isnan(mkt_ma60), np.nan, mkt_ma20 + mkt_ma60)

    mkt_dict = {}
    for i in range(n):
        mkt_dict[dates[i]] = np.array([
            mkt_ret_5d[i], mkt_ret_20d[i], mkt_ma20[i], mkt_ma60[i],
            mkt_vol_20d[i], mkt_vol_ratio[i], mkt_rsi[i], mkt_regime[i]
        ])

    valid = sum(1 for v in mkt_dict.values() if not np.isnan(v[7]))
    print(f"  Computed: {valid}/{n} days complete", flush=True)
    return mkt_dict


def part1_diag():
    print("=" * 60)
    print("Part 1: A3_v1 Feature Importance")
    print("=" * 60, flush=True)
    bst = xgb.Booster(); bst.load_model(str(MODEL_V1))
    scores = bst.get_score(importance_type='gain')
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  {len(bst.feature_names)} features, {len(scores)} with nonzero importance", flush=True)
    print(f"\n  Top 15:", flush=True)
    for i, (f, g) in enumerate(ranked[:15]):
        print(f"    {i+1:2d}. {f:20s}  gain={g:.1f}", flush=True)
    zero = [f for f in bst.feature_names if f not in scores]
    if zero:
        print(f"\n  {len(zero)} zero-importance: {zero}", flush=True)
    return ranked


def main():
    t0 = time.time()

    # Part 1
    ranked_v1 = part1_diag()

    # Part 2
    mkt_dict = build_market_features()

    # Part 3
    print("\n" + "=" * 60)
    print("Part 3: A3_v2 Walk-Forward (41 features)")
    print("=" * 60, flush=True)

    print("  Loading stock data...", flush=True)
    t1 = time.time()
    all_stocks = json.load(open(HIST_FILE, 'r'))
    codes = list(all_stocks.keys())
    print(f"  {len(codes)} stocks, loaded in {time.time()-t1:.1f}s", flush=True)

    # Build mkt lookup as float array for speed
    mkt_dates_set = set(mkt_dict.keys())

    # Pre-build sorted mkt arrays for vectorized lookup
    mkt_date_list = sorted(mkt_dict.keys())
    mkt_vals_list = np.array([mkt_dict[d] for d in mkt_date_list])  # (n_dates, 8)
    mkt_date_arr = np.array(mkt_date_list)  # sorted string dates
    print(f"  Market lookup: {len(mkt_date_arr)} dates", flush=True)

    print("  Computing features...", flush=True)
    t1 = time.time()
    chunks_X, chunks_y, chunks_d, chunks_c = [], [], [], []
    skipped = 0

    for idx, code in enumerate(codes):
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t1
            print(f"    [{idx+1}/{len(codes)}] {elapsed:.0f}s elapsed", flush=True)
        rec = all_stocks[code]
        c_arr = np.array(rec['c'], dtype=np.float64)
        n = len(c_arr)
        if n < 120:
            skipped += 1; continue

        feats = compute_features_fast(
            c_arr,
            np.array(rec['h'], dtype=np.float64),
            np.array(rec['l'], dtype=np.float64),
            np.array(rec['o'], dtype=np.float64),
            np.array(rec['v'], dtype=np.float64)
        )
        if feats is None:
            skipped += 1; continue

        dates = np.array(rec['dates'])
        # Vectorized market feature lookup via searchsorted
        indices = np.searchsorted(mkt_date_arr, dates)
        indices = np.clip(indices, 0, len(mkt_date_arr) - 1)
        matched = mkt_date_arr[indices] == dates
        mkt_arr = np.full((n, 8), np.nan)
        mkt_arr[matched] = mkt_vals_list[indices[matched]]

        X = np.hstack([feats, mkt_arr])

        # Target
        fwd = np.full(n, np.nan)
        if HOLD_DAYS < n:
            fwd[:n-HOLD_DAYS] = (c_arr[HOLD_DAYS:] / c_arr[:n-HOLD_DAYS] - 1) * 100

        valid = ~np.any(np.isnan(X), axis=1) & ~np.isnan(fwd)
        nv = np.sum(valid)
        if nv > 0:
            chunks_X.append(X[valid])
            chunks_y.append(fwd[valid])
            chunks_d.append(dates[valid])
            chunks_c.append(np.full(nv, code))

    print(f"  Done: {len(chunks_X)} stocks valid, {skipped} skipped, {time.time()-t1:.1f}s", flush=True)

    X = np.vstack(chunks_X)
    y = np.concatenate(chunks_y)
    dates_all = np.concatenate(chunks_d)
    codes_all = np.concatenate(chunks_c)
    print(f"  Total samples: {X.shape[0]}, features: {X.shape[1]}", flush=True)

    del all_stocks, chunks_X, chunks_y, chunks_d, chunks_c

    # Walk-Forward
    results = []
    print(f"\n  WF: hold={HOLD_DAYS}d, top_k={TOP_K}, SL={SL_PCT}%", flush=True)
    print(f"  {'Fold':>5} | {'End':>8} | {'Days':>5} | {'AvgR':>6} | {'WinR':>5} | {'NoBoom':>6}", flush=True)
    print("  " + "-" * 50, flush=True)

    final_model = None
    for fold_name, ts, te in WF_SPLITS:
        m_tr = (dates_all >= ts) & (dates_all <= te)
        X_tr, y_tr = X[m_tr], y[m_tr]
        X_te, y_te = X[m_te], y[m_te]
        d_te = dates_all[m_te]

        if len(X_tr) < 1000 or len(X_te) < 100:
            print(f"  {fold_name:>5} | {te:>8} | SKIP", flush=True); continue

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dtest = xgb.DMatrix(X_te, label=y_te)

        params = {
            'objective': 'reg:squarederror', 'max_depth': 6,
            'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'min_child_weight': 50, 'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'eval_metric': 'rmse', 'tree_method': 'hist', 'seed': 42
        }
        bst = xgb.train(params, dtrain, num_boost_round=500,
                        evals=[(dtrain, 'tr'), (dtest, 'ev')],
                        early_stopping_rounds=30, verbose_eval=False)
        final_model = bst
        preds = bst.predict(dtest)

        udates = np.unique(d_te)
        d_rets, d_wins, d_nb = [], [], []
        for d in udates:
            mask = d_te == d
            dp = preds[mask]; dy = y_te[mask]
            if len(dp) < TOP_K: continue
            top_i = np.argsort(dp)[-TOP_K:]
            top_y = dy[top_i]
            capped = np.where(top_y < SL_PCT, SL_PCT, top_y)
            d_rets.append(np.mean(capped))
            d_wins.append(np.mean(top_y > 0) * 100)
            d_nb.append(np.mean(top_y < 20) * 100)

        if d_rets:
            ar, aw, an = np.mean(d_rets), np.mean(d_wins), np.mean(d_nb)
            print(f"  {fold_name:>5} | {te:>8} | {len(d_rets):>5} | {ar:>5.2f}% | {aw:>4.1f}% | {an:>5.1f}%", flush=True)
            results.append({'fold': fold_name, 'train_end': te, 'n_days': len(d_rets),
                           'avg_ret': round(ar, 2), 'win_rate': round(aw, 1), 'avg_no_boom': round(an, 1)})

    if len(results) >= 2:
        f4 = results[:4]
        print(f"\n  Fold 1-4 avg: ret={np.mean([r['avg_ret'] for r in f4]):.2f}%, "
              f"win={np.mean([r['win_rate'] for r in f4]):.1f}%", flush=True)

    # Save report
    scores_v2 = final_model.get_score(importance_type='gain') if final_model else {}
    ranked_v2 = sorted(scores_v2.items(), key=lambda x: x[1], reverse=True)

    report = {
        'model': 'A3_v2', 'n_features': len(ALL_FEATURES),
        'v1_top15': [(f, round(g, 1)) for f, g in ranked_v1[:15]],
        'v2_top20': [(f, round(g, 1)) for f, g in ranked_v2[:20]],
        'v2_walk_forward': results,
        'v1_reference': {
            '10d_K20_SL15': {'avg_ret': 8.07, 'win_rate': 62, 'folds': [1.70, 0.40, 4.08, 1.34, 28.10]},
        },
        'params': {'hold_days': HOLD_DAYS, 'top_k': TOP_K, 'stop_loss': SL_PCT}
    }
    with open(REPORT_OUT, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT_OUT}", flush=True)

    print(f"\n  A3_v2 Top 15:", flush=True)
    for i, (f, g) in enumerate(ranked_v2[:15]):
        tag = " [MKT]" if f in MKT_FEATURES else ""
        print(f"    {i+1:2d}. {f:20s}  gain={g:.1f}{tag}", flush=True)

    mkt_count = sum(1 for f, _ in ranked_v2[:20] if f in MKT_FEATURES)
    print(f"\n  Market features in top 20: {mkt_count}/20", flush=True)
    print(f"\n  Total: {time.time()-t0:.1f}s", flush=True)
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
