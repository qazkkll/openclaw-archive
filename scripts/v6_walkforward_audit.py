#!/usr/bin/env python3
"""
V6 Blueshield Walk-Forward Audit - Memory Optimized
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from scipy import stats
import json, warnings, gc, os
warnings.filterwarnings('ignore')

print("=" * 60)
print("V6 BLUE SHIELD WALK-FORWARD AUDIT (Memory Optimized)")
print("=" * 60)

# ============================================================
# 1. Load full data for macro + sampling
# ============================================================
print("\n[1] Loading data...")
df_full = pd.read_parquet('data/us/us_hist_full_10y.parquet')
print(f"  {len(df_full):,} rows, {df_full['sym'].nunique()} tickers")

# ============================================================
# 2. Build macro features from FULL data (before filtering)
# ============================================================
print("\n[2] Building macro features...")

vix_df = yf.download('^VIX', start='2016-06-01', end='2026-06-25', progress=False)
vix_close = vix_df[('Close', '^VIX')].copy()
vix_close.index = vix_close.index.tz_localize(None)

all_dates = sorted(df_full['date'].unique())
macro_df = pd.DataFrame(index=all_dates)
macro_df.index.name = 'date'
macro_df = macro_df.join(vix_close.rename('vix_close'), how='left')
macro_df['vix_close'] = macro_df['vix_close'].ffill()

for sym in ['SPY', 'QQQ', 'IWM']:
    sd = df_full[df_full['sym'] == sym][['date', 'close']].sort_values('date').set_index('date')
    for days in [1, 5, 20, 60]:
        macro_df[f'{sym.lower()}_ret{days}'] = sd['close'].pct_change(days)

macro_df = macro_df.reset_index()
print(f"  Macro columns: {list(macro_df.columns)}")

# ============================================================
# 3. Sample tickers, build features per stock, keep minimal
# ============================================================
print("\n[3] Sampling & computing features...")

etf_syms = set(['SPY','QQQ','IWM','VIXY','VXX','UVXY','VOO','IVV','VTI','EEM','FXI'])
latest = df_full[df_full['date'] == df_full['date'].max()].set_index('sym')
eligible = latest[(latest['close'] > 10) & (~latest.index.isin(etf_syms))]

np.random.seed(42)
N_SAMPLE = 1500  # Reduced from 3000 to save memory
sample_syms = np.random.choice(eligible.index, size=min(N_SAMPLE, len(eligible)), replace=False)
print(f"  Sampled {len(sample_syms)} from {len(eligible)} eligible")

# Process one stock at a time, keep only what we need
feature_list = [
    'ma5','ma20','ma60','ma_bias20','ma_align','price_position',
    'ret1','ret5','ret20','ret60','momentum_6m','momentum_1m',
    'mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change',
    'rsi14','rsi_change','macd','macd_signal','macd_hist',
    'bb_std','bb_width','bb_pos','ret_quality'
]

# Pre-compute SPY/QQQ/IWM close for macro
macro_closes = {}
for sym in ['SPY', 'QQQ', 'IWM']:
    sd = df_full[df_full['sym'] == sym][['date', 'close']].sort_values('date').set_index('date')
    macro_closes[sym] = sd['close']
del df_full
gc.collect()

# Process stocks in memory-efficient chunks
all_chunks = []
stock_dates = sorted(set(df_full['date'].unique()) if 'df_full' in dir() else all_dates)

# Reload full data for stock processing (needed)
df_full = pd.read_parquet('data/us/us_hist_full_10y.parquet')
df_stock = df_full[df_full['sym'].isin(list(sample_syms))].copy()
del df_full
gc.collect()

def compute_features_one_stock(sym_data):
    """Compute features for one stock, return only feature columns + date + fwd_ret."""
    g = sym_data.sort_values('date').copy()
    c = g['close'].astype(np.float32)
    h = g['high'].astype(np.float32)
    l = g['low'].astype(np.float32)
    
    out = pd.DataFrame()
    out['date'] = g['date'].values
    out['sym'] = g['sym'].values
    
    # MA features
    out['ma5'] = c.rolling(5).mean()
    out['ma20'] = c.rolling(20).mean()
    out['ma60'] = c.rolling(60).mean()
    out['ma_bias20'] = ((c - out['ma20']) / (out['ma20'] + 1e-10)).astype(np.float32)
    ma5g = out['ma5'] > out['ma20']
    ma20g = out['ma20'] > out['ma60']
    ma5l = out['ma5'] < out['ma20']
    ma20l = out['ma20'] < out['ma60']
    out['ma_align'] = (ma5g & ma20g).astype(np.float32) - (ma5l & ma20l).astype(np.float32)
    h60 = h.rolling(60).max()
    l60 = l.rolling(60).min()
    out['price_position'] = ((c - l60) / (h60 - l60 + 1e-10)).astype(np.float32)
    
    # Returns
    out['ret1'] = c.pct_change(1)
    out['ret5'] = c.pct_change(5)
    out['ret20'] = c.pct_change(20)
    out['ret60'] = c.pct_change(60)
    
    # Momentum
    out['momentum_6m'] = c.pct_change(120)
    out['momentum_1m'] = c.pct_change(20)
    out['mom_divergence'] = (out['momentum_6m'] - out['momentum_1m']).astype(np.float32)
    out['trend_accel'] = (out['ret20'] - c.pct_change(20).shift(20)).astype(np.float32)
    
    # Volatility
    ret1 = c.pct_change()
    out['vol20'] = ret1.rolling(20).std()
    out['vol5'] = ret1.rolling(5).std()
    out['vol_ratio'] = (out['vol5'] / (out['vol20'] + 1e-10)).astype(np.float32)
    out['vol_change'] = (out['vol20'] / (out['vol20'].shift(5) + 1e-10) - 1).astype(np.float32)
    
    # RSI
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    out['rsi14'] = (100 - (100 / (1 + rs))).astype(np.float32)
    out['rsi_change'] = (out['rsi14'] - out['rsi14'].shift(5)).astype(np.float32)
    
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out['macd'] = (ema12 - ema26).astype(np.float32)
    out['macd_signal'] = out['macd'].ewm(span=9, adjust=False).mean()
    out['macd_hist'] = (out['macd'] - out['macd_signal']).astype(np.float32)
    
    # Bollinger
    out['bb_std'] = c.rolling(20).std()
    out['bb_width'] = ((2 * out['bb_std']) / (out['ma20'] + 1e-10)).astype(np.float32)
    out['bb_pos'] = ((c - out['ma20']) / (2 * out['bb_std'] + 1e-10)).astype(np.float32)
    
    # Return quality
    out['ret_quality'] = (c.pct_change(20) / (out['vol20'] * np.sqrt(20) + 1e-10)).astype(np.float32)
    
    # Forward return
    out['fwd_ret'] = (c.shift(-20) / c - 1).astype(np.float32)
    
    return out

# Process all stocks
chunks = []
syms = df_stock['sym'].unique()
for i, sym in enumerate(syms):
    grp = df_stock[df_stock['sym'] == sym]
    if len(grp) < 120:  # Need at least 120 days for rolling windows
        continue
    chunk = compute_features_one_stock(grp)
    chunks.append(chunk)
    if (i+1) % 300 == 0:
        print(f"  Processed {i+1}/{len(syms)} stocks...")

del df_stock
gc.collect()

df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"  Features computed: {len(df):,} rows, {df.shape[1]} columns")

# ============================================================
# 4. Merge macro features
# ============================================================
print("\n[4] Merging macro features...")
df = df.merge(macro_df, on='date', how='left')

# Forward fill macro within each stock
for col in ['vix_close'] + [c for c in macro_df.columns if c not in ['date', 'vix_close']]:
    df[col] = df.groupby('sym')[col].ffill()

# Fundamentals (not available)
for col in ['pe_trailing', 'pe_forward', 'div_yield', 'beta']:
    df[col] = np.float32(0.0)

print(f"  After merge: {len(df):,} rows")

# ============================================================
# 5. Clean, split
# ============================================================
print("\n[5] Cleaning...")

all_features = feature_list + [
    'vix_close', 'spy_ret1', 'spy_ret5', 'spy_ret20', 'spy_ret60',
    'qqq_ret1', 'qqq_ret5', 'qqq_ret20', 'qqq_ret60',
    'iwm_ret1', 'iwm_ret5', 'iwm_ret20', 'iwm_ret60',
    'pe_trailing', 'pe_forward', 'div_yield', 'beta'
]

before = len(df)
df = df.dropna(subset=all_features + ['fwd_ret'])
print(f"  {before:,} -> {len(df):,} after dropna ({before-len(df):,} dropped)")

OOS_START = '2024-01-01'
is_data = df[df['date'] < OOS_START]
oos_data = df[df['date'] >= OOS_START].copy()
print(f"  IS: {len(is_data):,} | OOS: {len(oos_data):,}")

# ============================================================
# 6. Train + predict
# ============================================================
print("\n[6] Training & predicting...")

X_is = is_data[all_features].values.astype(np.float32)
y_is = is_data['fwd_ret'].values.astype(np.float32)

params = {
    'objective': 'reg:squarederror', 'max_depth': 6, 'learning_rate': 0.03,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 10,
    'verbosity': 0, 'seed': 42, 'n_estimators': 500
}
model = xgb.XGBRegressor(**params)
model.fit(X_is, y_is)
print(f"  Re-trained: {model.n_estimators} trees")

v6_model = xgb.XGBRegressor()
v6_model.load_model('models/us/blueshield_v6_xgb.json')
print(f"  V6 loaded: {v6_model.get_booster().num_boosted_rounds()} trees")

X_oos = oos_data[all_features].values.astype(np.float32)
oos_data['pred_v6'] = v6_model.predict(X_oos)
oos_data['pred_retrained'] = model.predict(X_oos)
del X_is, y_is, X_oos
gc.collect()

# ============================================================
# 7. Daily cross-sectional IC
# ============================================================
print("\n[7] Computing daily metrics...")

oos_dates = sorted(oos_data['date'].unique())
daily_metrics = []

for date in oos_dates:
    day = oos_data[oos_data['date'] == date]
    if len(day) < 30:
        continue
    
    pred = day['pred_v6'].values.astype(np.float64)
    actual = day['fwd_ret'].values.astype(np.float64)
    
    ic, _ = stats.spearmanr(pred, actual)
    
    idx = np.argsort(pred)[::-1]
    n = len(pred)
    t5 = max(1, int(n * 0.05))
    b20 = max(1, int(n * 0.20))
    
    daily_metrics.append({
        'date': date, 'n': len(day), 'ic': ic,
        'top5_ret': actual[idx[:t5]].mean(),
        'bot20_ret': actual[idx[-b20:]].mean(),
        'mid_ret': actual[idx[t5:-b20]].mean() if n-t5-b20 > 0 else 0,
        'spread': actual[idx[:t5]].mean() - actual[idx[-b20:]].mean(),
        'top5_win': (actual[idx[:t5]] > 0).mean(),
        'mkt_ret': actual.mean()
    })

mdf = pd.DataFrame(daily_metrics)
print(f"  Days: {len(mdf)}")

# ============================================================
# 8. Results
# ============================================================
print(f"\n{'='*60}")
print("V6 WALK-FORWARD OOS RESULTS")
print(f"{'='*60}")

ic = mdf['ic'].mean()
icir = mdf['ic'].mean() / (mdf['ic'].std() + 1e-10)
ic_pos = (mdf['ic'] > 0).mean() * 100
spread = mdf['spread'].mean()
t5 = mdf['top5_ret'].mean()
b20 = mdf['bot20_ret'].mean()
mid = mdf['mid_ret'].mean()
win = mdf['top5_win'].mean() * 100
mkt = mdf['mkt_ret'].mean()

print(f"  IC:              {ic:.4f}")
print(f"  ICIR:            {icir:.4f}")
print(f"  IC > 0:          {ic_pos:.1f}%")
print(f"  Spread(T5-B20):  {spread:.4f} ({spread*100:.2f}%)")
print(f"  Top5% ret:       {t5:.4f} ({t5*100:.2f}%)")
print(f"  Top5% win:       {win:.1f}%")
print(f"  Bot20% ret:      {b20:.4f} ({b20*100:.2f}%)")
print(f"  Mid ret:         {mid:.4f} ({mid*100:.2f}%)")
print(f"  Market ret:      {mkt:.4f} ({mkt*100:.2f}%)")
print(f"  Days:            {len(mdf)}")
print(f"  Avg stocks/day:  {mdf['n'].mean():.0f}")

# By year
print(f"\n{'='*60}")
print("BY YEAR")
print(f"{'='*60}")

mdf['year'] = mdf['date'].apply(lambda x: x.year)
yr_results = {}

for year in [2024, 2025, 2026]:
    yr = mdf[mdf['year'] == year]
    if len(yr) == 0: continue
    yic = yr['ic'].mean()
    yicir = yr['ic'].mean() / (yr['ic'].std() + 1e-10)
    yic_pos = (yr['ic'] > 0).mean() * 100
    yspread = yr['spread'].mean()
    yt5 = yr['top5_ret'].mean()
    yb20 = yr['bot20_ret'].mean()
    ywin = yr['top5_win'].mean() * 100
    ymkt = yr['mkt_ret'].mean()
    
    print(f"\n--- {year} ({len(yr)} days) ---")
    print(f"  IC: {yic:.4f}  ICIR: {yicir:.4f}  IC>0: {yic_pos:.1f}%")
    print(f"  Spread: {yspread:.4f}  Top5: {yt5:.4f}  Win: {ywin:.1f}%  Mkt: {ymkt:.4f}")
    
    yr_results[str(year)] = {
        'days': int(len(yr)), 'ic': float(yic), 'icir': float(yicir),
        'ic_positive_pct': float(yic_pos), 'spread': float(yspread),
        'top5_avg_ret': float(yt5), 'top5_win_rate': float(ywin),
        'bot20_avg_ret': float(yb20), 'market_avg_ret': float(ymkt)
    }

# Monotonicity
print(f"\n{'='*60}")
print("MONOTONICITY (R018)")
print(f"{'='*60}")

q_data = {f'Q{i+1}': [] for i in range(5)}
for date in oos_dates:
    day = oos_data[oos_data['date'] == date]
    if len(day) < 50: continue
    p = day['pred_v6'].values.astype(np.float64)
    a = day['fwd_ret'].values.astype(np.float64)
    ranks = stats.rankdata(p)
    n = len(ranks); qs = n // 5
    for q in range(5):
        mask = (ranks > q*qs) & (ranks <= (q+1)*qs) if q < 4 else ranks > q*qs
        if mask.sum() > 0: q_data[f'Q{q+1}'].append(a[mask].mean())

labels = ['Bot 20%', 'Q2', 'Q3', 'Q4', 'Top 20%']
q_means = {}
for q in range(5):
    k = f'Q{q+1}'
    v = np.mean(q_data[k]) if q_data[k] else 0
    q_means[k] = v
    print(f"  {k} ({labels[q]}): {v:.4f} ({v*100:.2f}%)")

mono = all(q_means[f'Q{i+1}'] <= q_means[f'Q{i+2}'] for i in range(4))
print(f"\n  Monotonic: {'YES ✅' if mono else 'NO ❌'}")
print(f"  Q5-Q1: {q_means['Q5']-q_means['Q1']:.4f} ({(q_means['Q5']-q_means['Q1'])*100:.2f}%)")

# Monthly
print(f"\n{'='*60}")
print("MONTHLY IC")
print(f"{'='*60}")

mdf['month'] = mdf['date'].apply(lambda x: x.strftime('%Y-%m'))
monthly = mdf.groupby('month').agg({'ic':'mean','spread':'mean','top5_ret':'mean','top5_win':'mean'}).reset_index()
for _, r in monthly.iterrows():
    f = '✅' if r['ic'] > 0 else '❌'
    print(f"  {r['month']}: IC={r['ic']:.4f}{f}  S={r['spread']:.4f}  T5={r['top5_ret']:.4f}  W={r['top5_win']*100:.0f}%")

# Diagnosis
print(f"\n{'='*60}")
print("DIAGNOSIS")
print(f"{'='*60}")

notes = []
has_alpha = ic > 0.02 and icir > 0.5
if ic < 0.02: notes.append(f"IC={ic:.4f} < 0.02: negligible ranking")
if icir < 0.5: notes.append(f"ICIR={icir:.4f} < 0.5: unreliable")
if not mono: notes.append("Not monotonic: higher pred ≠ higher return")
if spread <= 0: notes.append("Spread <= 0: top underperform bottom")
if t5 <= mkt: notes.append(f"Top5 ({t5*100:.2f}%) <= market ({mkt*100:.2f}%)")
if win < 50: notes.append(f"Win rate {win:.1f}% < 50%")

print(f"  Has alpha: {'YES' if has_alpha else 'NO'}")
print(f"  Spread+: {'YES' if spread > 0 else 'NO'}")
print(f"  Top5>Market: {'YES' if t5>mkt else 'NO'}")
print(f"  Monotonic: {'YES' if mono else 'NO'}")
for n in notes: print(f"  ⚠️  {n}")

# Save
results = {
    'model': 'blueshield_v6_xgb',
    'audit_date': pd.Timestamp.now().isoformat(),
    'method': 'Walk-Forward (IS 2016-2023, OOS 2024-2026)',
    'n_tickers': int(len(sample_syms)),
    'n_stocks_avg': float(mdf['n'].mean()),
    'n_days': int(len(mdf)),
    'note': 'Fundamentals set to 0 (not in data)',
    'overall': {
        'ic': float(ic), 'icir': float(icir), 'ic_positive_pct': float(ic_pos),
        'spread': float(spread), 'top5_avg_ret': float(t5), 'top5_win_rate': float(win),
        'bot20_avg_ret': float(b20), 'mid_avg_ret': float(mid), 'market_avg_ret': float(mkt),
        'monotonic': mono, 'quintile_returns': {k: float(v) for k, v in q_means.items()}
    },
    'by_year': yr_results,
    'monthly': {r['month']: {'ic': float(r['ic']), 'spread': float(r['spread']),
               'top5_ret': float(r['top5_ret']), 'top5_win': float(r['top5_win'])}
               for _, r in monthly.iterrows()},
    'diagnosis': {'has_alpha': has_alpha, 'monotonic': mono, 'spread_positive': spread > 0,
                  'top5_beats_market': t5 > mkt, 'notes': notes}
}

os.makedirs('data/experiments', exist_ok=True)
with open('data/experiments/v6_audit.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n  ✅ Saved to data/experiments/v6_audit.json")
print(f"\n{'='*60}")
print("AUDIT COMPLETE")
print(f"{'='*60}")
