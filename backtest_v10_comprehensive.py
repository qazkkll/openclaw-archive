#!/usr/bin/env python3
"""
Comprehensive Backtest: V10 BlueShield Quantile Model (43 features)
Optimized: limit to top ~500 stocks by dollar volume, vectorized features.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
import time

# ── Config ──────────────────────────────────────────────────────────────
MODEL_PATH = 'models/us/blueshield_lgb_v9_quantile_lgb.txt'
HIST_PATH = 'data/us/us_hist_full_10y.parquet'
FUND_PATH = 'data/us/fundamentals_latest.parquet'

FEATURES = [
    "ma5", "ma20", "ma60", "ma_bias20", "ma_align", "price_position",
    "ret1", "ret5", "ret20", "ret60", "momentum_6m", "momentum_1m",
    "mom_divergence", "trend_accel", "vol20", "vol5", "vol_ratio", "vol_change",
    "rsi14", "rsi_change", "macd", "macd_signal", "macd_hist",
    "bb_std", "bb_width", "bb_pos", "ret_quality", "range_ratio", "avg_body",
    "vwap_drift", "ret_10d", "ret_30d", "ret_90d", "vol_regime",
    "ma_cross_5_20", "ma_cross_20_60", "rsi_zone", "macd_roc", "dd_60",
    "ud_vol_ratio", "pe_log", "div_yield", "beta"
]

LEVERAGED_ETFS = {
    'AAPU','AAPD','AMZU','AMZD','MSFU','MSFD','GOOX','GOOG',
    'NFLU','METU','PLTU','HOOD','NVDL','NVDX','NVDU',
    'TSLT','TSLL','TSLZ','TSLQ','CONL',
    'SOXL','SOXS','LABU','LABD','FAS','FAZ','TNA','TZA',
    'TECL','TECS','FNGU','FNGD','BULZ','BERZ',
    'DFEN','DUSL','DPST','RETL','CURE',
    'HIBL','HIBS','PILL','MWJ','FLYD','FLYU',
    'SVIX','UVIX','UVXY',
    'UPRO','SPXU','UDOW','SDOW','UMDD','SMDD','URTY','SRTY',
    'SPXS','SPXL','SQQQ','TQQQ','QLD','QID',
    'NUGT','DUST','JNUG','JDST',
    'VXX','VIXY','VIXM',
    'ARTNA','BSMC','SMCX','SMCL','GDXD','RKLX','GUSH','DRIP',
}

LEVERAGED_PATTERNS = ['2X','3X','4X','5X','-1X','-2X','-3X',
                      'BULL','BEAR','ULTRA','SHORT','LONG','DAILY','WEEKLY']

BACKTEST_START = '2025-01-01'
BACKTEST_END = '2026-06-24'
TOP_N = 15
MAX_STOCKS = 800  # limit universe size for speed


def compute_stock_features_vectorized(sdf):
    """
    Vectorized feature computation. sdf has columns: close, high, low, volume.
    Returns DataFrame with FEATURES columns.
    """
    c = sdf['close'].values.astype(np.float64)
    h = sdf['high'].values.astype(np.float64)
    lo = sdf['low'].values.astype(np.float64)
    v = sdf['volume'].values.astype(np.float64)
    n = len(c)

    # Helper functions using numpy for speed
    def rmean(arr, w):
        cs = np.nancumsum(arr)
        out = np.full(n, np.nan)
        out[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
        return out

    def rstd(arr, w):
        out = np.full(n, np.nan)
        for i in range(w-1, n):
            seg = arr[i-w+1:i+1]
            out[i] = np.nanstd(seg, ddof=0)
        return out

    def ema_np(arr, span):
        alpha = 2.0 / (span + 1)
        out = np.full(n, np.nan)
        out[0] = arr[0] if not np.isnan(arr[0]) else 0
        for i in range(1, n):
            if np.isnan(arr[i]):
                out[i] = out[i-1]
            else:
                out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
        return out

    def pct(arr, k):
        out = np.full(n, np.nan)
        denom = arr[:-k].copy()
        denom[denom == 0] = np.nan
        out[k:] = (arr[k:] - arr[:-k]) / denom
        return out

    def diff(arr, k):
        out = np.full(n, np.nan)
        out[k:] = arr[k:] - arr[:-k]
        return out

    # Daily returns
    dret = np.full(n, np.nan)
    dret[1:] = (c[1:] - c[:-1]) / np.where(c[:-1] == 0, np.nan, c[:-1])

    # Moving averages
    ma5 = rmean(c, 5)
    ma20 = rmean(c, 20)
    ma60 = rmean(c, 60)

    # Volatility
    vol20 = rstd(dret, 20)
    vol5 = rstd(dret, 5)
    vol60 = rstd(dret, 60)

    # EMA
    ema12 = ema_np(c, 12)
    ema26 = ema_np(c, 26)
    macd_line = ema12 - ema26
    macd_signal = ema_np(macd_line, 9)
    macd_hist = macd_line - macd_signal

    # RSI
    delta = np.zeros(n)
    delta[1:] = c[1:] - c[:-1]
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = ema_np(gain, 14)
    avg_loss = ema_np(loss, 14)
    rs = avg_gain / (avg_loss + 1e-10)
    rsi14 = 100 - 100 / (1 + rs)

    # Bollinger
    bb_std = rstd(c, 20)
    bb_width = 2 * bb_std / (ma20 + 1e-10)
    bb_pos = (c - ma20) / (2 * bb_std + 1e-10)

    # Price position
    rmin60 = np.full(n, np.nan)
    rmax60 = np.full(n, np.nan)
    rmin60_c = np.full(n, np.nan)
    rmax60_c = np.full(n, np.nan)
    for i in range(59, n):
        rmin60[i] = np.min(c[i-59:i+1])
        rmax60[i] = np.max(c[i-59:i+1])
        rmin60_c[i] = rmin60[i]
        rmax60_c[i] = rmax60[i]
    price_pos = (c - rmin60) / (rmax60 - rmin60 + 1e-10)

    # Volume
    vol_ma20 = rmean(v, 20)
    vol_ratio = v / (vol_ma20 + 1e-10)

    # Positive/negative vol
    pos_r = np.where(dret > 0, dret, 0.0)
    neg_r = np.where(dret < 0, dret, 0.0)
    pos_vol = rstd(pos_r, 20)
    neg_vol = rstd(neg_r, 20)
    ud_vol_ratio = pos_vol / (np.abs(neg_vol) + 1e-10)

    # Drawdown
    dd_60 = c / rmax60_c - 1

    # Assemble features
    feat = np.full((n, 43), np.nan, dtype=np.float64)
    feat[:, 0] = ma5
    feat[:, 1] = ma20
    feat[:, 2] = ma60
    feat[:, 3] = (c - ma20) / (ma20 + 1e-10)
    feat[:, 4] = ((c > ma5).astype(float) + (ma5 > ma20).astype(float))
    feat[:, 5] = price_pos
    feat[:, 6] = pct(c, 1)
    feat[:, 7] = pct(c, 5)
    feat[:, 8] = pct(c, 20)
    feat[:, 9] = pct(c, 60)
    feat[:, 10] = pct(c, 126)
    feat[:, 11] = pct(c, 21)
    feat[:, 12] = feat[:, 11] - feat[:, 8]  # mom_divergence
    feat[:, 13] = np.full(n, np.nan)
    feat[:, 13][5:] = feat[5:, 7] - feat[:-5, 7]  # trend_accel
    feat[:, 14] = vol20
    feat[:, 15] = vol5
    feat[:, 16] = vol_ratio
    vol20_shifted = np.full(n, np.nan)
    vol20_shifted[20:] = vol20[:-20]
    feat[:, 17] = vol20 / (vol20_shifted + 1e-10)  # vol_change
    feat[:, 18] = rsi14
    feat[:, 19] = diff(rsi14, 5)  # rsi_change
    feat[:, 20] = macd_line
    feat[:, 21] = macd_signal
    feat[:, 22] = macd_hist
    feat[:, 23] = bb_std
    feat[:, 24] = bb_width
    feat[:, 25] = bb_pos
    feat[:, 26] = feat[:, 8] / (vol20 + 1e-10)  # ret_quality
    feat[:, 27] = (h - lo) / (c + 1e-10)  # range_ratio
    feat[:, 28] = np.abs(np.nan_to_num(dret, nan=0))  # avg_body
    feat[:, 29] = feat[:, 7] / (vol5 + 1e-10)  # vwap_drift
    feat[:, 30] = pct(c, 10)
    feat[:, 31] = pct(c, 30)
    feat[:, 32] = pct(c, 90)
    feat[:, 33] = vol20 / (vol60 + 1e-10)  # vol_regime
    feat[:, 34] = (ma5 > ma20).astype(float)
    feat[:, 35] = (ma20 > ma60).astype(float)
    feat[:, 36] = (rsi14 // 10)  # rsi_zone
    feat[:, 37] = diff(macd_hist, 5)  # macd_roc
    feat[:, 38] = dd_60
    feat[:, 39] = ud_vol_ratio

    return feat, rsi14, bb_pos


def main():
    print("=" * 90)
    print("V10 BlueShield Quantile Model — Comprehensive Backtest (Optimized)")
    print("=" * 90)
    t_start = time.time()

    # 1. Load model
    print("\n1. Loading model...")
    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"   {model.num_trees()} trees, {len(model.feature_name())} features")

    # 2. Load & filter data
    print("\n2. Loading data...")
    hist_df = pd.read_parquet(HIST_PATH)

    # Filter universe
    df = hist_df[hist_df['close'] >= 10.0].copy()
    sym_upper = df['sym'].str.upper()
    sym_mask = (
        ~df['sym'].str.endswith('W') &
        ~df['sym'].str.endswith('U') &
        ~df['sym'].str.endswith('R') &
        ~df['sym'].isin(LEVERAGED_ETFS) &
        ~sym_upper.str.contains('|'.join(LEVERAGED_PATTERNS), regex=True, na=False)
    )
    df = df[sym_mask].copy()

    # Dollar volume filter
    df['dollar_vol'] = df['volume'] * df['close']
    dv_ma = df.groupby('sym')['dollar_vol'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df = df[dv_ma >= 5_000_000].copy()

    # Keep only top MAX_STOCKS by average dollar volume
    avg_dv = df.groupby('sym')['dollar_vol'].mean().sort_values(ascending=False)
    top_syms = avg_dv.head(MAX_STOCKS).index.tolist()
    df = df[df['sym'].isin(top_syms)].copy()

    print(f"   {df['sym'].nunique()} stocks, {len(df):,} rows")

    # 3. Load fundamentals
    print("\n3. Fundamentals...")
    fund_dict = {}
    try:
        fdf = pd.read_parquet(FUND_PATH)
        fund_dict = fdf.set_index('sym').to_dict('index')
        print(f"   {len(fund_dict)} stocks")
    except:
        print("   Not loaded")

    # 4. SPY benchmark
    print("\n4. SPY benchmark...")
    spy = hist_df[hist_df['sym'] == 'SPY'].sort_values('date').set_index('date')
    spy_close = spy['close']
    spy_bt = spy_close[(spy_close.index >= BACKTEST_START) & (spy_close.index <= BACKTEST_END)]
    spy_return = (spy_bt.iloc[-1] / spy_bt.iloc[0]) - 1
    spy_daily = spy_bt.pct_change().dropna()
    spy_sharpe = (spy_daily.mean() / (spy_daily.std() + 1e-10)) * np.sqrt(252)
    spy_peak = spy_bt.iloc[0]
    spy_max_dd = 0
    for p in spy_bt.values:
        spy_peak = max(spy_peak, p)
        dd = (p - spy_peak) / spy_peak
        spy_max_dd = min(spy_max_dd, dd)
    print(f"   SPY: {spy_return:.1%} ret, {spy_sharpe:.2f} Sharpe, {spy_max_dd:.1%} maxDD")

    # 5. Pre-compute features for all stocks
    print("\n5. Pre-computing features...")
    t_feat = time.time()

    stock_close_dict = {}  # sym -> date-indexed close series
    stock_feat_dict = {}   # sym -> date-indexed feature DataFrame (43 cols)
    stock_rsi_dict = {}
    stock_bbpos_dict = {}

    stocks = df['sym'].unique()
    computed = 0
    for i, sym in enumerate(stocks):
        sdf = df[df['sym'] == sym].sort_values('date').reset_index(drop=True)
        if len(sdf) < 150:
            continue

        dates = sdf['date'].values
        feat, rsi, bbpos = compute_stock_features_vectorized(sdf)

        # Create DataFrame with date index
        feat_df = pd.DataFrame(feat, columns=FEATURES, index=dates)

        # Add fundamentals
        if sym in fund_dict:
            f = fund_dict[sym]
            pe = f.get('pe_trailing', np.nan)
            div = f.get('div_yield', np.nan)
            beta = f.get('beta', 0.73)
            feat_df['pe_log'] = np.log(pe) if pe and pe > 0 else 0
            feat_df['div_yield'] = div if div and not np.isnan(div) else 0
            feat_df['beta'] = np.clip(beta if beta and not np.isnan(beta) else 0.73, -2, 5)
        else:
            feat_df['pe_log'] = 0
            feat_df['div_yield'] = 0
            feat_df['beta'] = 0.73

        stock_feat_dict[sym] = feat_df
        stock_close_dict[sym] = pd.Series(sdf['close'].values, index=dates)
        stock_rsi_dict[sym] = pd.Series(rsi, index=dates)
        stock_bbpos_dict[sym] = pd.Series(bbpos, index=dates)
        computed += 1

        if (i + 1) % 200 == 0:
            print(f"   ... {i+1}/{len(stocks)} processed ({computed} kept)", flush=True)

    print(f"   {computed} stocks with features ({time.time()-t_feat:.1f}s)")

    # 6. Get backtest dates
    all_dates_set = set()
    for s in stock_close_dict.values():
        all_dates_set.update(s.index)
    all_dates = sorted(all_dates_set)
    bt_dates = [d for d in all_dates if str(d)[:10] >= BACKTEST_START and str(d)[:10] <= BACKTEST_END]
    print(f"\n6. Backtest: {len(bt_dates)} trading days ({str(bt_dates[0])[:10]} to {str(bt_dates[-1])[:10]})")

    # 7. Run scenarios
    print("\n7. Running scenarios...")

    scenarios = [
        ("hold5_fixed", 5, 'fixed', {}),
        ("hold10_fixed", 10, 'fixed', {}),
        ("hold15_fixed", 15, 'fixed', {}),
        ("hold20_fixed", 20, 'fixed', {}),
        ("hold10_trailing_stop", 10, 'trailing_stop', {'stop_loss_pct': -0.15}),
        ("hold10_rsi_bb", 10, 'rsi_bb', {'rsi_thresh': 75, 'bb_thresh': 0.9}),
    ]

    all_results = {}

    for sname, hdays, est, params in scenarios:
        print(f"\n  {sname}...", end='', flush=True)
        t0 = time.time()

        pv = 1.0
        peak = 1.0
        maxdd = 0.0
        trades = []
        vals = [1.0]
        rets = []

        # positions: {sym: (entry_date, entry_price, entry_day_idx, highest_price)}
        positions = {}
        last_rot = -hdays

        for di, dt in enumerate(bt_dates):
            # ── Exit checks ──
            exited = []
            for sym, (ed, ep, edi, hp) in positions.items():
                dh = di - edi
                reason = None

                if sym in stock_close_dict and dt in stock_close_dict[sym].index:
                    cp = stock_close_dict[sym].loc[dt]
                else:
                    exited.append(sym)
                    continue

                # Trailing stop update
                if est == 'trailing_stop' and cp > hp:
                    hp = cp
                    positions[sym] = (ed, ep, edi, hp)

                # Exit conditions
                if est == 'trailing_stop' and dh >= 1:
                    r = (cp - ep) / ep if ep > 0 else 0
                    if r <= params.get('stop_loss_pct', -0.15):
                        reason = 'stop_loss'

                if est == 'rsi_bb' and dh >= 1:
                    if sym in stock_rsi_dict and dt in stock_rsi_dict[sym].index:
                        rv = stock_rsi_dict[sym].loc[dt]
                        bv = stock_bbpos_dict[sym].loc[dt]
                        if not np.isnan(rv) and not np.isnan(bv):
                            if rv > params.get('rsi_thresh', 75) and bv > params.get('bb_thresh', 0.9):
                                reason = 'rsi_bb'

                if reason is None and dh >= hdays:
                    reason = 'holding_period'

                if reason:
                    tr = (cp - ep) / ep if ep > 0 else 0
                    yr = pd.Timestamp(dt).year
                    trades.append({'sym': sym, 'entry_date': ed, 'exit_date': dt,
                                   'entry_price': ep, 'exit_price': cp,
                                   'return': tr, 'exit_reason': reason, 'year': yr})
                    exited.append(sym)

            for s in exited:
                positions.pop(s, None)

            # ── Portfolio return ──
            pr = []
            for sym, (ed, ep, edi, hp) in positions.items():
                if sym in stock_close_dict and dt in stock_close_dict[sym].index:
                    pidx = di - 1
                    if 0 <= pidx < len(bt_dates):
                        pdt = bt_dates[pidx]
                        if pdt in stock_close_dict[sym].index:
                            cp = stock_close_dict[sym].loc[dt]
                            pp = stock_close_dict[sym].loc[pdt]
                            if pp > 0:
                                pr.append((cp - pp) / pp)

            ar = np.mean(pr) if pr else 0
            pv *= (1 + ar)
            rets.append(ar)
            vals.append(pv)
            peak = max(peak, pv)
            dd = (pv - peak) / peak
            maxdd = min(maxdd, dd)

            # ── Rotation ──
            if di - last_rot >= hdays:
                scores = []
                for sym in stock_feat_dict:
                    feat = stock_feat_dict[sym]
                    if dt in feat.index:
                        row = feat.loc[dt]
                        fv = row[FEATURES].values.astype(np.float64)
                        if not np.any(np.isnan(fv)):
                            fv2 = fv.reshape(1, -1)
                            fdf = pd.DataFrame(fv2, columns=[f'Column_{i}' for i in range(43)])
                            sc = model.predict(fdf)[0]
                            scores.append((sym, sc))

                if len(scores) >= TOP_N:
                    scores.sort(key=lambda x: x[1], reverse=True)
                    top = [s[0] for s in scores[:TOP_N]]
                    positions = {}
                    for sym in top:
                        if sym in stock_close_dict and dt in stock_close_dict[sym].index:
                            ep = stock_close_dict[sym].loc[dt]
                            positions[sym] = (dt, ep, di, ep)
                    last_rot = di

        elapsed = time.time() - t0
        print(f" {len(trades)} trades, {elapsed:.1f}s")

        # Metrics
        ra = np.array(rets)
        tot_ret = pv - 1.0
        sharpe = (np.mean(ra) / (np.std(ra) + 1e-10)) * np.sqrt(252)
        tr_list = [t['return'] for t in trades]
        wr = np.mean([r > 0 for r in tr_list]) if tr_list else 0
        ap = np.mean(tr_list) if tr_list else 0

        ec = {}
        for t in trades:
            ec[t['exit_reason']] = ec.get(t['exit_reason'], 0) + 1

        yr = {}
        for t in trades:
            y = t['year']
            if y not in yr:
                yr[y] = {'r': [], 'w': 0, 'n': 0}
            yr[y]['r'].append(t['return'])
            yr[y]['n'] += 1
            if t['return'] > 0:
                yr[y]['w'] += 1

        all_results[sname] = {
            'total_return': tot_ret, 'sharpe': sharpe, 'max_drawdown': maxdd,
            'win_rate': wr, 'avg_profit': ap, 'num_trades': len(trades),
            'exit_counts': ec, 'yearly': yr, 'trades': trades,
        }

    # ── Output ─────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    hdr = f"{'Strategy':<28} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8} {'WinRate':>8} {'AvgProfit':>10} {'Trades':>7}"
    print(hdr)
    print("-" * 100)
    print(f"{'SPY Benchmark':<28} {spy_return:>8.1%} {spy_sharpe:>7.2f} {spy_max_dd:>8.1%} {'---':>8} {'---':>10} {'---':>7}")
    print("-" * 100)

    for k in ['hold5_fixed', 'hold10_fixed', 'hold15_fixed', 'hold20_fixed',
              'hold10_trailing_stop', 'hold10_rsi_bb']:
        r = all_results.get(k)
        if not r: continue
        print(f"{k:<28} {r['total_return']:>8.1%} {r['sharpe']:>7.2f} {r['max_drawdown']:>8.1%} "
              f"{r['win_rate']:>7.1%} {r['avg_profit']:>9.2%} {r['num_trades']:>7}")

    # Exit comparison
    print("\n" + "=" * 80)
    print("EXIT STRATEGY COMPARISON (Hold 10d)")
    print("=" * 80)
    for s in ['hold10_fixed', 'hold10_trailing_stop', 'hold10_rsi_bb']:
        r = all_results.get(s)
        if not r: continue
        print(f"\n  {s}:")
        print(f"    Return: {r['total_return']:.1%}  |  Sharpe: {r['sharpe']:.2f}  |  MaxDD: {r['max_drawdown']:.1%}")
        print(f"    WinRate: {r['win_rate']:.1%}  |  AvgProfit: {r['avg_profit']:.2%}  |  Trades: {r['num_trades']}")
        print(f"    Exits: {r['exit_counts']}")

    # Yearly
    print("\n" + "=" * 80)
    print("YEARLY BREAKDOWN (Hold 10d)")
    print("=" * 80)
    for s in ['hold10_fixed', 'hold10_trailing_stop', 'hold10_rsi_bb']:
        r = all_results.get(s)
        if not r or not r.get('yearly'): continue
        print(f"\n  {s}:")
        print(f"  {'Year':<8} {'Trades':>8} {'WinRate':>8} {'AvgRet':>10} {'TotalRet':>10}")
        print("  " + "-" * 50)
        for y in sorted(r['yearly'].keys()):
            d = r['yearly'][y]
            ywr = d['w'] / d['n'] if d['n'] > 0 else 0
            yavg = np.mean(d['r']) if d['r'] else 0
            ytot = np.sum(d['r']) if d['r'] else 0
            print(f"  {y:<8} {d['n']:>8} {ywr:>7.1%} {yavg:>9.2%} {ytot:>9.1%}")

    # Top/Bottom trades
    print("\n" + "=" * 80)
    print("TOP/BOTTOM 10 TRADES (hold10_fixed)")
    print("=" * 80)
    r = all_results.get('hold10_fixed')
    if r and r.get('trades'):
        st = sorted(r['trades'], key=lambda x: x['return'], reverse=True)
        print(f"  {'Sym':<8} {'Entry':>12} {'Exit':>12} {'Return':>8} {'Reason':<15}")
        print("  " + "-" * 65)
        for t in st[:10]:
            print(f"  {t['sym']:<8} {str(t['entry_date'])[:10]:>12} {str(t['exit_date'])[:10]:>12} "
                  f"{t['return']:>7.1%} {t['exit_reason']:<15}")
        print("\n  BOTTOM 10:")
        print("  " + "-" * 65)
        for t in st[-10:]:
            print(f"  {t['sym']:<8} {str(t['entry_date'])[:10]:>12} {str(t['exit_date'])[:10]:>12} "
                  f"{t['return']:>7.1%} {t['exit_reason']:<15}")

    # Holding period comparison detail
    print("\n" + "=" * 80)
    print("HOLDING PERIOD SENSITIVITY")
    print("=" * 80)
    print(f"  {'Period':<15} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8} {'WinRate':>8} {'#Trades':>8} {'Trades/Yr':>10}")
    print("  " + "-" * 65)
    for hp in [5, 10, 15, 20]:
        r = all_results.get(f'hold{hp}_fixed')
        if not r: continue
        # Approximate annualized trade count
        n_years = len(bt_dates) / 252
        tpy = r['num_trades'] / n_years if n_years > 0 else 0
        print(f"  {hp:>3}d{'':<12} {r['total_return']:>8.1%} {r['sharpe']:>7.2f} {r['max_drawdown']:>8.1%} "
              f"{r['win_rate']:>7.1%} {r['num_trades']:>8} {tpy:>10.0f}")

    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")
    print("=" * 90)
    print("BACKTEST COMPLETE")
    print("=" * 90)


if __name__ == '__main__':
    main()
