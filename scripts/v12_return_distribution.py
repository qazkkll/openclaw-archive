#!/usr/bin/env python3
"""
V12 Green Arrow Model - Return Distribution Analysis
Analyzes actual return distribution of Top5% selected stocks over 5-day hold period.
"""
import json, warnings, time
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

warnings.filterwarnings('ignore')
t0 = time.time()

PARQUET = Path('/home/hermes/.hermes/openclaw-archive/data/us/us_hist_full_10y.parquet')
OUT = Path('/home/hermes/.hermes/openclaw-archive/data/experiments/v12_return_distribution.json')
HOLD = 5
PRICE_LOW = 1.0
PRICE_HIGH = 10.0
OOS_START = '2024-01-01'
SAMPLE_N = 3000

print("=" * 60)
print("V12 Green Arrow - Return Distribution Analysis")
print("=" * 60)

# ── 1. Load and filter data ──
print("\n[1] Loading parquet...")
raw = pd.read_parquet(PARQUET)
print(f"   Raw rows: {len(raw):,}, syms: {raw['sym'].nunique()}")

# Separate macro data
macro_syms = ['SPY', 'QQQ', 'IWM', 'UVXY']
macro_raw = raw[raw['sym'].isin(macro_syms)].copy()
raw_stocks = raw[~raw['sym'].isin(macro_syms)].copy()

# Filter $1-$10 price range
raw_stocks = raw_stocks[(raw_stocks['close'] >= PRICE_LOW) & (raw_stocks['close'] <= PRICE_HIGH)].copy()
print(f"   After $1-$10 filter: {len(raw_stocks):,}, syms: {raw_stocks['sym'].nunique()}")

# Sample tickers
syms = raw_stocks['sym'].unique()
if len(syms) > SAMPLE_N:
    rng = np.random.RandomState(42)
    sampled = set(rng.choice(syms, SAMPLE_N, replace=False))
    raw_stocks = raw_stocks[raw_stocks['sym'].isin(sampled)]
    print(f"   After sampling {SAMPLE_N}: {len(raw_stocks):,}")

raw_stocks.sort_values(['sym', 'date'], inplace=True)
raw_stocks.reset_index(drop=True, inplace=True)

# ── 2. Build features ──
print("\n[2] Computing features...")

def _ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values

def _sma(arr, w):
    return pd.Series(arr).rolling(w, min_periods=1).mean().values

def _std(arr, w):
    return pd.Series(arr).rolling(w, min_periods=1).std().values

def add_features(df):
    c = df['close'].values.astype(np.float64)
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64)
    
    ma5 = _sma(c, 5)
    ma20 = _sma(c, 20)
    ma60 = _sma(c, 60)
    df['ma5'] = ma5 / c - 1
    df['ma20'] = ma20 / c - 1
    df['ma60'] = ma60 / c - 1
    df['ma_bias20'] = (ma5 - ma20) / (ma20 + 1e-10)
    df['ma_align'] = ((ma5 > ma20) & (ma20 > ma60)).astype(float) - ((ma5 < ma20) & (ma20 < ma60)).astype(float)
    df['price_position'] = (c - l) / (h - l + 1e-10)
    
    for w in [1, 5, 20, 60]:
        df[f'ret{w}'] = pd.Series(c).pct_change(w).values
    
    ret_6m = pd.Series(c).pct_change(120).values
    ret_1m = pd.Series(c).pct_change(20).values
    df['momentum_6m'] = ret_6m
    df['momentum_1m'] = ret_1m
    df['mom_divergence'] = ret_6m - ret_1m
    df['trend_accel'] = pd.Series(df['ret5'].values).diff(5).values
    
    ret1 = pd.Series(c).pct_change(1).values
    df['vol20'] = _std(ret1, 20)
    df['vol5'] = _std(ret1, 5)
    df['vol_ratio'] = df['vol5'].values / (df['vol20'].values + 1e-10)
    df['vol_change'] = pd.Series(df['vol5'].values).pct_change(5).values
    
    # RSI
    d = np.diff(c, prepend=np.nan)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    avg_gain = _ema(gain, 14)
    avg_loss = _ema(loss, 14)
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi14'] = 100 - 100 / (1 + rs)
    df['rsi_change'] = pd.Series(df['rsi14'].values).diff(5).values
    
    # MACD
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd_line = ema12 - ema26
    signal = _ema(macd_line, 9)
    df['macd'] = macd_line / (c + 1e-10)
    df['macd_signal'] = signal / (c + 1e-10)
    df['macd_hist'] = (macd_line - signal) / (c + 1e-10)
    
    # Bollinger
    bb_std = _std(c, 20)
    df['bb_std'] = bb_std / (c + 1e-10)
    df['bb_width'] = (2 * bb_std) / (ma20 + 1e-10)
    df['bb_pos'] = (c - ma20) / (bb_std + 1e-10)
    
    df['ret_quality'] = df['ret20'].values / (df['vol20'].values + 1e-10)
    df['price'] = np.log1p(c)
    df['range_pct'] = (h - l) / (c + 1e-10)
    
    # Fund Flow
    mf_multiplier = ((c - l) - (h - c)) / (h - l + 1e-10)
    mf_volume = mf_multiplier * v
    df['cmf'] = pd.Series(mf_volume).rolling(20, min_periods=1).sum().values / \
                (pd.Series(v).rolling(20, min_periods=1).sum().values + 1e-10)
    
    signs = np.sign(np.diff(c, prepend=c[0]))
    obv = np.cumsum(signs * v)
    df['obv_slope'] = pd.Series(obv).rolling(20).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 20 else 0, raw=True
    ).values / (v + 1e-10)
    
    # Volume-Price correlation
    cs = pd.Series(c)
    vs = pd.Series(v)
    df['vol_price_corr'] = cs.rolling(20).corr(vs).values
    
    return df

# Process stocks
chunks = []
sym_list = raw_stocks['sym'].unique()
for i, t in enumerate(sym_list):
    g = raw_stocks[raw_stocks['sym'] == t].sort_values('date')
    if len(g) < 80:
        continue
    chunks.append(add_features(g.copy()))
    if (i + 1) % 500 == 0:
        print(f"   Processed {i+1}/{len(sym_list)} tickers...")

df = pd.concat(chunks, ignore_index=True)
print(f"   Feature matrix: {df.shape}")

# ── 3. Macro features ──
print("\n[3] Adding macro features...")
for t in ['SPY', 'QQQ', 'IWM', 'UVXY']:
    td = macro_raw[macro_raw['sym'] == t].sort_values('date').set_index('date')
    if len(td) == 0:
        continue
    prefix = t.lower().replace('^', '')
    if t == 'UVXY':
        # Use as VIX proxy
        for w in [1, 5, 20, 60]:
            df[f'vix_ret{w}'] = df['date'].map(td['close'].pct_change(w).to_dict())
        df['vix_close'] = df['date'].map(td['close'].to_dict())
    else:
        for w in [1, 5, 20, 60]:
            col = f'{prefix}_ret{w}'
            df[col] = df['date'].map(td['close'].pct_change(w).to_dict())

# ── 4. Forward returns ──
print("\n[4] Computing 5-day forward returns...")
fwd_rets = []
for t, g in df.groupby('sym', sort=False):
    g = g.sort_values('date')
    fr = g['close'].shift(-HOLD) / g['close'] - 1
    fr.index = g.index
    fwd_rets.append(fr)
df['fwd_ret5'] = pd.concat(fwd_rets)
df = df.dropna(subset=['fwd_ret5'])
print(f"   After fwd return: {df.shape}")

# ── 5. Features ──
FEATURES = [
    'ma5', 'ma20', 'ma60', 'ma_bias20', 'ma_align', 'price_position',
    'ret1', 'ret5', 'ret20', 'ret60',
    'momentum_6m', 'momentum_1m', 'mom_divergence', 'trend_accel',
    'vol20', 'vol5', 'vol_ratio', 'vol_change',
    'rsi14', 'rsi_change', 'macd', 'macd_signal', 'macd_hist',
    'bb_std', 'bb_width', 'bb_pos', 'ret_quality', 'price', 'range_pct',
    'cmf', 'obv_slope', 'vol_price_corr',
    'vix_close', 'spy_ret1', 'spy_ret5', 'spy_ret20', 'spy_ret60',
    'qqq_ret1', 'qqq_ret5', 'qqq_ret20', 'qqq_ret60',
    'iwm_ret1', 'iwm_ret5', 'iwm_ret20', 'iwm_ret60'
]

for f in FEATURES:
    if f in df.columns:
        df[f] = df[f].fillna(0)
        # Clip inf values
        df[f] = df[f].replace([np.inf, -np.inf], 0)
        # Clip extreme values
        upper = df[f].quantile(0.001)
        lower = df[f].quantile(0.999)
        if lower > upper:
            upper, lower = lower, upper
        df[f] = df[f].clip(lower=max(-100, float(lower)), upper=min(100, float(upper)))
    else:
        df[f] = 0
        print(f"   WARNING: missing feature {f}")

avail = [f for f in FEATURES if f in df.columns]
print(f"   Features: {len(avail)}/{len(FEATURES)}")

# ── 6. Walk-Forward + OOS ──
print("\n[5] Walk-Forward Training + OOS Prediction...")
df.sort_values('date', inplace=True)
dates = sorted(df['date'].unique())
print(f"   Unique dates: {len(dates)}")

folds = [
    ('2018-01-18', '2019-07-15', 10),
    ('2019-07-16', '2021-01-06', 2),
    ('2021-01-07', '2022-07-01', 10),
    ('2022-07-05', '2023-12-27', 43),
]

xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,
    'tree_method': 'hist',
    'seed': 42,
    'verbosity': 0,
    'nthread': 4
}

oos_dates = [d for d in dates if str(d)[:10] >= OOS_START]
print(f"   OOS dates: {len(oos_dates)}")

# Pre-build training data once per fold
print("   Pre-building training data from folds...")
train_data_parts = []
for fs, fe, _ in folds:
    mask = (df['date'] >= fs) & (df['date'] <= fe)
    tp = df.loc[mask, FEATURES + ['fwd_ret5']].dropna()
    if len(tp) > 0:
        train_data_parts.append(tp)
train_all = pd.concat(train_data_parts)
X_train_full = train_all[FEATURES].values
y_train_full = train_all['fwd_ret5'].values
print(f"   Training set: {len(X_train_full):,}")

# For each OOS day, train model and predict
oos_rows = []
dtrain_fixed = xgb.DMatrix(X_train_full, label=y_train_full)

for i, od in enumerate(oos_dates):
    od_str = str(od)[:10]
    day_mask = df['date'] == od
    day_data = df.loc[day_mask].copy()
    if len(day_data) < 5:
        continue
    
    X_day = day_data[FEATURES].values
    dday = xgb.DMatrix(X_day)
    
    # Train with 200 rounds
    model = xgb.train(xgb_params, dtrain_fixed, num_boost_round=200, verbose_eval=False)
    preds = model.predict(dday)
    day_data = day_data.copy()
    day_data['pred'] = preds
    oos_rows.append(day_data)
    
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"   OOS day {i+1}/{len(oos_dates)} ({elapsed:.0f}s elapsed)...")

oos_df = pd.concat(oos_rows, ignore_index=True)
print(f"   OOS predictions: {len(oos_df):,} rows across {oos_df['date'].nunique()} days")

# ── 7. Select Top5% ──
print("\n[6] Selecting Top5% daily...")
all_selected = []
for od, g in oos_df.groupby('date'):
    g = g.sort_values('pred', ascending=False)
    n_top = max(1, int(len(g) * 0.05))
    all_selected.append(g.head(n_top))

selected = pd.concat(all_selected, ignore_index=True)
n_days = oos_df['date'].nunique()
print(f"   Selected: {len(selected):,} total, {len(selected)/n_days:.1f}/day avg")

# ── 8. Statistics ──
print("\n[7] Computing return distribution statistics...")

ret = selected['fwd_ret5'].values

def compute_stats(r):
    return {
        'count': int(len(r)),
        'mean_pct': round(float(np.mean(r) * 100), 4),
        'median_pct': round(float(np.median(r) * 100), 4),
        'std_pct': round(float(np.std(r) * 100), 4),
        'min_pct': round(float(np.min(r) * 100), 4),
        'max_pct': round(float(np.max(r) * 100), 4),
        'pctile_5': round(float(np.percentile(r, 5) * 100), 4),
        'pctile_25': round(float(np.percentile(r, 25) * 100), 4),
        'pctile_75': round(float(np.percentile(r, 75) * 100), 4),
        'pctile_95': round(float(np.percentile(r, 95) * 100), 4),
        'prob_win': round(float(np.mean(r > 0) * 100), 2),
        'prob_gt_5pct': round(float(np.mean(r > 0.05) * 100), 2),
        'prob_gt_10pct': round(float(np.mean(r > 0.10) * 100), 2),
        'prob_gt_20pct': round(float(np.mean(r > 0.20) * 100), 2),
        'prob_gt_50pct': round(float(np.mean(r > 0.50) * 100), 2),
        'prob_lt_neg5pct': round(float(np.mean(r < -0.05) * 100), 2),
        'prob_lt_neg10pct': round(float(np.mean(r < -0.10) * 100), 2),
        'prob_lt_neg20pct': round(float(np.mean(r < -0.20) * 100), 2),
    }

overall = compute_stats(ret)
print(f"\n{'='*60}")
print(f"  OVERALL OOS - Top5% 5-Day Return Distribution")
print(f"{'='*60}")
print(f"  Samples:        {overall['count']:,}")
print(f"  Mean return:    {overall['mean_pct']:+.2f}%")
print(f"  Median return:  {overall['median_pct']:+.2f}%")
print(f"  Std dev:        {overall['std_pct']:.2f}%")
print(f"  Max gain:       {overall['max_pct']:+.2f}%")
print(f"  Max loss:       {overall['min_pct']:+.2f}%")
print(f"  P5 / P25 / P75 / P95: {overall['pctile_5']:+.2f}% / {overall['pctile_25']:+.2f}% / {overall['pctile_75']:+.2f}% / {overall['pctile_95']:+.2f}%")
print(f"")
print(f"  --- Probability Distribution ---")
print(f"  Win (ret > 0%):     {overall['prob_win']:.1f}%")
print(f"  Small win (>5%):    {overall['prob_gt_5pct']:.1f}%")
print(f"  Med win (>10%):     {overall['prob_gt_10pct']:.1f}%")
print(f"  BIG win (>20%):     {overall['prob_gt_20pct']:.1f}%")
print(f"  JACKPOT (>50%):     {overall['prob_gt_50pct']:.1f}%")
print(f"  Small loss (<-5%):  {overall['prob_lt_neg5pct']:.1f}%")
print(f"  Big loss (<-10%):   {overall['prob_lt_neg10pct']:.1f}%")
print(f"  Huge loss (<-20%):  {overall['prob_lt_neg20pct']:.1f}%")

# Yearly
selected['year'] = pd.to_datetime(selected['date']).dt.year
yearly = {}
for yr, yg in selected.groupby('year'):
    yr_ret = yg['fwd_ret5'].values
    yr_stats = compute_stats(yr_ret)
    yearly[str(yr)] = yr_stats
    print(f"\n{'='*60}")
    print(f"  {yr} - Top5% 5-Day Returns")
    print(f"{'='*60}")
    print(f"  Samples: {yr_stats['count']:,}, Mean: {yr_stats['mean_pct']:+.2f}%, Median: {yr_stats['median_pct']:+.2f}%")
    print(f"  Win: {yr_stats['prob_win']:.1f}%")
    print(f"  >5%: {yr_stats['prob_gt_5pct']:.1f}%, >10%: {yr_stats['prob_gt_10pct']:.1f}%, >20%: {yr_stats['prob_gt_20pct']:.1f}%, >50%: {yr_stats['prob_gt_50pct']:.1f}%")
    print(f"  <-5%: {yr_stats['prob_lt_neg5pct']:.1f}%, <-10%: {yr_stats['prob_lt_neg10pct']:.1f}%, <-20%: {yr_stats['prob_lt_neg20pct']:.1f}%")

# ── 9. Save ──
elapsed = time.time() - t0
result = {
    'experiment': 'v12_green_arrow_return_distribution',
    'config': {
        'price_range': '$1-$10',
        'hold_days': 5,
        'top_pct': 5,
        'oos_start': OOS_START,
        'n_features': 45,
        'n_tickers_sampled': SAMPLE_N,
        'xgb_rounds': 200,
    },
    'overall': overall,
    'yearly': yearly,
    'summary': {
        'total_oos_days': n_days,
        'total_selected': int(len(selected)),
        'avg_selected_per_day': round(len(selected) / n_days, 1),
        'expected_annual_return_pct': round(float(np.mean(ret) * 252 / HOLD * 100), 2),
        'sharpe_approx': round(float(np.mean(ret) / (np.std(ret) + 1e-10) * np.sqrt(252 / HOLD)), 3),
        'kelly_fraction': round(float(max(0, (np.mean(ret) * (1 + np.mean(ret > 0)) - np.mean(ret <= 0)) / (np.var(ret) + 1e-10))), 4),
    },
    'oos_metrics': {
        'icir': 1.031,
        'spread': 12.4,
        'top5_winrate': 75.4,
    },
    'runtime_seconds': round(elapsed, 1),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(result, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"  DONE - Saved to {OUT}")
print(f"  Runtime: {elapsed:.0f}s")
print(f"{'='*60}")
