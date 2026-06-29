#!/usr/bin/env python3
"""
红杉v1.0 (Redwood v1.0) — A股XGBoost信号生成器
=================================================
Model: XGBoost regression predicting 10-day forward returns
Features: 25 (technical + money flow + market macro)
Universe: A-shares (含科创板688x), price 3-200
Selection: Top 15 by predicted return
Hold period: 10 trading days
Stop loss: -2% (close-based, optimized)

Market filter (optimized):
  - breadth>40% AND mkt_ret20>-5% → FULL
  - breadth 30-40% OR mkt_ret20 -5%~-8% → CAUTIOUS (half)
  - breadth<30% OR mkt_ret20<-8% → BEAR (no new positions)
  NOTE: Market filter is ADVISORY. Top picks always shown.

Signal levels (absolute score):
  - score >= 0.05 → 🟢🟢 精品
  - score >= 0.03 → 🟢 强信号
  - score >= 0.01 → 🟡 观察
"""

import pandas as pd
import numpy as np
import json
import time
import os
import sys
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

# ============================================================
# Configuration
# ============================================================
HOLD_DAYS = 10
TOP_N = 15
SL = -0.02  # Stop loss -2% (close-based, optimized from -3%)
MODEL_PATH = 'models/cn/cn_alpha_v2_xgb.json'
SIGNAL_PATH = 'signals/cn/latest_xgb.json'
NAMES_PATH = 'data/cn/stock_names.json'

FEATURE_COLS = [
    'ret5', 'ret10', 'ret20', 'ma20_bias', 'ma60_bias',
    'vol5', 'vol20', 'rsi_14', 'macd_hist', 'atr_pct', 'vol_ratio',
    'total_net_5d', 'lg_net_5d', 'md_net_5d', 'elg_net_5d',
    'total_net_20d', 'lg_net_20d', 'md_net_20d', 'elg_net_20d',
    'total_net_5d_rk', 'lg_net_5d_rk', 'md_net_5d_rk', 'elg_net_5d_rk',
    'breadth', 'mkt_ret20'
]

# Absolute score thresholds for signal levels (calibrated to score distribution)
SIG_GG = 0.08   # 精品 (top ~3-5)
SIG_G = 0.06    # 强信号 (top ~6-10)
SIG_Y = 0.04    # 观察 (top ~11-15)

def score_to_100(raw):
    """Convert raw model score to 0-100 scale for display."""
    return max(0, min(100, int((raw - 0.02) / 0.10 * 100)))

# ============================================================
# 1. Load Data
# ============================================================
print(f"[Redwood v1.0] A股信号生成 {time.strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)

t0 = time.time()

# Load stock names
name_map, industry_map = {}, {}
if os.path.exists(NAMES_PATH):
    with open(NAMES_PATH) as f:
        nd = json.load(f)
        name_map = nd.get('names', {})
        industry_map = nd.get('industries', {})
    print(f"  Stock names loaded: {len(name_map)}")

# Load OHLCV
df = pd.read_parquet('data/a_hist_10y.parquet')
df = df.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df['date'] = df['date'].astype(int)

# Load moneyflow
mf = pd.read_parquet('data/cn/moneyflow_core.parquet')
mf['sym'] = mf['ts_code'].str[:6]
mf['date'] = mf['trade_date'].astype(int)
for col in ['sm', 'md', 'lg', 'elg']:
    mf[f'{col}_net'] = mf[f'buy_{col}_amount'] - mf[f'sell_{col}_amount']
mf['total_net'] = mf['net_mf_amount']
df = df.merge(mf[['sym', 'date', 'total_net', 'lg_net', 'md_net', 'elg_net']], on=['sym', 'date'], how='left')

# Filter — NO 688 exclusion
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
df = df[df['volume'] > 0].copy()
df = df.sort_values(['sym', 'date']).reset_index(drop=True)

# Filter out stocks with <20 trading days (次新股, features are all zeros)
day_counts = df.groupby('sym')['date'].transform('count')
df = df[day_counts >= 20].copy()
print(f"  Filtered: {df['sym'].nunique()} stocks with ≥20 trading days")

print(f"  Data loaded: {len(df):,} rows, {df['sym'].nunique()} stocks ({time.time()-t0:.0f}s)")

# ============================================================
# 2. Compute Features
# ============================================================
print("  Computing features...")

df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ret10'] = df.groupby('sym')['close'].pct_change(10)
df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma60'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(60, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['ma60_bias'] = (df['close'] - df['ma60']) / df['ma60']
df['vol5'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(5, min_periods=2).std())
df['vol20'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(20, min_periods=2).std())
delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)
ema12 = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=12, min_periods=1).mean())
ema26 = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=26, min_periods=1).mean())
df['macd'] = ema12 - ema26
df['macd_signal'] = df.groupby('sym')['macd'].transform(lambda x: x.ewm(span=9, min_periods=1).mean())
df['macd_hist'] = df['macd'] - df['macd_signal']
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(abs(df['high'] - df.groupby('sym')['close'].shift(1)),
               abs(df['low'] - df.groupby('sym')['close'].shift(1)))
)
df['atr14'] = df.groupby('sym')['tr'].transform(lambda x: x.rolling(14, min_periods=1).mean())
df['atr_pct'] = df['atr14'] / df['close']
df['vol_ratio'] = df.groupby('sym')['volume'].transform(lambda x: x.rolling(5).mean()) / \
                  df.groupby('sym')['volume'].transform(lambda x: x.rolling(20).mean())
for col in ['total_net', 'lg_net', 'md_net', 'elg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())
    df[f'{col}_20d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(20, min_periods=1).sum())
    df[f'{col}_5d_rk'] = df.groupby('date')[f'{col}_5d'].rank(pct=True)
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')

print(f"  Features computed ({time.time()-t0:.0f}s)")

# ============================================================
# 3. Load Model
# ============================================================
import xgboost as xgb

if os.path.exists(MODEL_PATH):
    print(f"  Loading model from {MODEL_PATH}")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
else:
    print("  ERROR: No model file found!")
    sys.exit(1)

# ============================================================
# 4. Generate Signal
# ============================================================
max_date = df['date'].max()
today_data = df[df['date'] == max_date].copy()

if len(today_data) == 0:
    print("  No data for today!")
    sys.exit(1)

# Market state — ADVISORY, not blocking
market_breadth = today_data['breadth'].mean()
market_ret20 = today_data['mkt_ret20'].mean()

# Optimized thresholds: breadth>40% AND ret20>-5% = FULL
if market_breadth > 0.40 and market_ret20 > -0.05:
    regime = 'bull'
    position_pct = 1.0
    regime_advice = '满仓'
elif market_breadth < 0.30 or market_ret20 < -0.08:
    regime = 'bear'
    position_pct = 0.0
    regime_advice = '不建议新买入'
else:
    regime = 'cautious'
    position_pct = 0.5
    regime_advice = '半仓/精选'

# Always predict and rank (never block)
X_today = today_data[FEATURE_COLS].fillna(0)
dtest = xgb.DMatrix(X_today)
today_data['score'] = model.predict(dtest)
today_data = today_data[
    (today_data['close'] >= 3) & (today_data['close'] <= 200) &
    (today_data['volume'] > 0)
]
top = today_data.nlargest(TOP_N, 'score')

# Absolute score signal levels
def signal_label(score):
    if score >= SIG_GG: return 'GG'
    elif score >= SIG_G: return 'G'
    elif score >= SIG_Y: return 'Y'
    else: return '-'

top = top.copy()
top['signal'] = top['score'].apply(signal_label)
top['name'] = top['sym'].map(name_map).fillna('')
top['industry'] = top['sym'].map(industry_map).fillna('')

# Output
print(f"\n{'='*60}")
print(f"Redwood v1.0 Signal {max_date}")
print(f"{'='*60}")
print(f"\nMarket: {regime.upper()} ({regime_advice})")
print(f"  Breadth: {market_breadth:.1%} {'OK' if market_breadth>0.40 else 'Weak' if market_breadth>0.30 else 'CRITICAL'}")
print(f"  20d Momentum: {market_ret20:+.2%} {'Positive' if market_ret20>0 else 'Negative' if market_ret20>-0.05 else 'SEVERE'}")

if regime == 'bear':
    print(f"\n  ⚠️ BEAR模式 — 以下信号仅供参考，不建议新买入")
elif regime == 'cautious':
    print(f"\n  ⚠️ CAUTIOUS模式 — 半仓操作，精选个股")

print(f"\nTop{TOP_N} (Hold {HOLD_DAYS}d, SL{SL*100:.0f}% close-based):")
print(f"{'#':>3} {'信号':>4} {'代码':>8} {'名称':>8} {'价格':>8} {'分数':>8} {'行业':>10} {'RSI':>5} {'20d':>7}")
print("-" * 75)

for i, (_, r) in enumerate(top.iterrows()):
    sig_emoji = {'GG': '🟢🟢', 'G': '🟢', 'Y': '🟡', '-': '⚪'}.get(r['signal'], '⚪')
    ret20_str = f"{r.get('ret20', 0):.1%}" if not pd.isna(r.get('ret20')) else "N/A"
    rsi_str = f"{r.get('rsi_14', 50):.0f}" if not pd.isna(r.get('rsi_14')) else "N/A"
    name_str = r.get('name', '')[:4]
    ind_str = r.get('industry', '')[:6]
    print(f"{i+1:>3} {sig_emoji:>4} {r['sym']:>8} {name_str:>8} {r['close']:>8.2f} {r['score']:>8.4f} {ind_str:>10} {rsi_str:>5} {ret20_str:>7}")

# Stats
g2 = (top['signal'] == 'GG').sum()
g1 = (top['signal'] == 'G').sum()
y = (top['signal'] == 'Y').sum()
print(f"\n  🟢🟢精品: {g2}  🟢强信号: {g1}  🟡观察: {y}")

# ============================================================
# 5. Save Signal
# ============================================================
signal_output = {
    'date': str(max_date),
    'strategy': 'redwood-v1.0',
    'model': 'XGBoost',
    'hold_days': HOLD_DAYS,
    'top_n': TOP_N,
    'stop_loss': SL,
    'regime': regime,
    'position_pct': position_pct,
    'regime_advice': regime_advice,
    'market': {
        'breadth': round(market_breadth, 4),
        'ret20': round(market_ret20, 6)
    },
    'thresholds': {
        'breadth_bull': 0.40,
        'breadth_bear': 0.30,
        'ret20_bull': -0.05,
        'ret20_bear': -0.08,
        'sig_gg': SIG_GG,
        'sig_g': SIG_G,
        'sig_y': SIG_Y
    },
    'top': []
}

for idx, (_, r) in enumerate(top.iterrows()):
    signal_output['top'].append({
        'rank': idx + 1,
        'sym': r['sym'],
        'name': r.get('name', ''),
        'industry': r.get('industry', ''),
        'close': round(float(r['close']), 2),
        'score': round(float(r['score']), 4),
        'score100': score_to_100(float(r['score'])),
        'signal': r['signal'],
        'ret20': round(float(r.get('ret20', 0)), 4),
        'rsi': round(float(r.get('rsi_14', 50)), 1)
    })

os.makedirs('signals/cn', exist_ok=True)
with open(SIGNAL_PATH, 'w') as f:
    json.dump(signal_output, f, indent=2, ensure_ascii=False)

print(f"\nSignal saved to {SIGNAL_PATH}")
print(f"Total time: {time.time()-t0:.0f}s")
