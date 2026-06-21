#!/usr/bin/env python3
"""
CEO Experiment Part 2: XGBoost Feature Importance Analysis
==========================================================
Purpose: Understand what XGBoost is learning that rule-based misses
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

print("=" * 70)
print("CEO Experiment Part 2: XGBoost Feature Importance")
print("=" * 70)
t0 = time.time()

# Load data (same as before)
print("\n[1/3] Loading data...")
df = pd.read_parquet('data/a_hist_10y.parquet')
df = df.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df['date'] = df['date'].astype(int)

mf = pd.read_parquet('data/cn/moneyflow_core.parquet')
mf['sym'] = mf['ts_code'].str[:6]
mf['date'] = mf['trade_date'].astype(int)
for col in ['sm', 'md', 'lg', 'elg']:
    mf[f'{col}_net'] = mf[f'buy_{col}_amount'] - mf[f'sell_{col}_amount']
mf['total_net'] = mf['net_mf_amount']
df = df.merge(mf[['sym', 'date', 'total_net', 'lg_net', 'md_net', 'elg_net']], on=['sym', 'date'], how='left')

df = df[~df['sym'].str.startswith('688')].copy()
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
df = df[df['volume'] > 0].copy()
df = df.sort_values(['sym', 'date']).reset_index(drop=True)

# Compute features (same as before)
print("[2/3] Computing features...")
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ret10'] = df.groupby('sym')['close'].pct_change(10)
df['ret20'] = df.groupby('sym')['close'].pct_change(20)

df['ma5'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
df['ma10'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(10, min_periods=1).mean())
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

df['fwd_10d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-10) / x - 1)

feature_cols = [
    'ret5', 'ret10', 'ret20', 'ma20_bias', 'ma60_bias',
    'vol5', 'vol20', 'rsi_14', 'macd_hist', 'atr_pct', 'vol_ratio',
    'total_net_5d', 'lg_net_5d', 'md_net_5d', 'elg_net_5d',
    'total_net_20d', 'lg_net_20d', 'md_net_20d', 'elg_net_20d',
    'total_net_5d_rk', 'lg_net_5d_rk', 'md_net_5d_rk', 'elg_net_5d_rk',
    'breadth', 'mkt_ret20'
]

df_clean = df.dropna(subset=feature_cols + ['fwd_10d']).copy()
print(f"  Clean: {len(df_clean):,} rows")

# Train on most recent 2 years for feature importance
print("\n[3/3] Training XGBoost for feature importance...")
max_date = df_clean['date'].max()
train_start = max_date - 20000  # ~2 years back
train = df_clean[(df_clean['date'] >= train_start) & (df_clean['date'] <= max_date)].copy()

import xgboost as xgb
X_train = train[feature_cols].fillna(0)
y_train = train['fwd_10d']

model = xgb.XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=4, verbosity=0
)
model.fit(X_train, y_train)

# Get feature importance
importance = model.feature_importances_
feature_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (XGBoost)")
print("=" * 70)
print(f"\n{'Rank':>4} {'Feature':<20} {'Importance':>10} {'Cumulative':>10}")
print("-" * 50)

cumulative = 0
for i, (_, row) in enumerate(feature_imp.iterrows()):
    cumulative += row['importance']
    print(f"{i+1:>4} {row['feature']:<20} {row['importance']:>10.4f} {cumulative:>10.4f}")

# Compare with rule-based weights
print("\n" + "=" * 70)
print("COMPARISON: XGBoost vs Rule-based")
print("=" * 70)

rule_weights = {
    'ret20': 3.0,      # Reversal
    'total_net_5d': 2.0,  # Money flow
    'vol20': 2.0,      # Low vol
    'rsi_14': 1.5,     # RSI oversold
    'lg_net_5d': 1.0,  # Large order flow
    'ma20_bias': 1.0   # MA deviation
}

print(f"\n{'Feature':<20} {'XGB Imp':>10} {'Rule Wt':>10} {'XGB Rank':>10} {'Rule Rank':>10}")
print("-" * 65)

xgb_ranks = {row['feature']: i+1 for i, (_, row) in enumerate(feature_imp.iterrows())}
rule_ranks = {}
sorted_rule = sorted(rule_weights.items(), key=lambda x: x[1], reverse=True)
for i, (feat, _) in enumerate(sorted_rule):
    rule_ranks[feat] = i + 1

for feat in feature_cols:
    xgb_imp = feature_imp[feature_imp['feature'] == feat]['importance'].values[0] if feat in feature_imp['feature'].values else 0
    rule_wt = rule_weights.get(feat, 0)
    xgb_rank = xgb_ranks.get(feat, '-')
    rule_rank = rule_ranks.get(feat, '-')
    
    # Highlight differences
    marker = ''
    if isinstance(xgb_rank, int) and isinstance(rule_rank, int):
        if xgb_rank < rule_rank - 2:
            marker = ' ← XGB higher'
        elif rule_rank < xgb_rank - 2:
            marker = ' ← Rule higher'
    
    print(f"{feat:<20} {xgb_imp:>10.4f} {rule_wt:>10.1f} {str(xgb_rank):>10} {str(rule_rank):>10}{marker}")

# Key insights
print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

top5_xgb = feature_imp.head(5)['feature'].tolist()
top5_rule = [f for f, _ in sorted_rule[:5]]

print(f"\nTop 5 XGBoost features: {', '.join(top5_xgb)}")
print(f"Top 5 Rule-based features: {', '.join(top5_rule)}")

# Features in XGB top 5 but not in rule top 5
xgb_unique = set(top5_xgb) - set(top5_rule)
rule_unique = set(top5_rule) - set(top5_xgb)

if xgb_unique:
    print(f"\nXGBoost values but rule ignores: {', '.join(xgb_unique)}")
if rule_unique:
    print(f"Rule values but XGBoost ignores: {', '.join(rule_unique)}")

# Calculate correlation between XGB and rule rankings
xgb_imp_values = [feature_imp[feature_imp['feature'] == f]['importance'].values[0] if f in feature_imp['feature'].values else 0 for f in feature_cols]
rule_wt_values = [rule_weights.get(f, 0) for f in feature_cols]
correlation = np.corrcoef(xgb_imp_values, rule_wt_values)[0, 1]
print(f"\nCorrelation between XGB importance and rule weights: {correlation:.3f}")

if correlation < 0.3:
    print("→ Low correlation: XGBoost learns very different patterns than rule-based")
elif correlation < 0.6:
    print("→ Moderate correlation: XGBoost partially overlaps with rule-based")
else:
    print("→ High correlation: XGBoost largely agrees with rule-based")

# Save results
output = {
    'timestamp': datetime.now().isoformat(),
    'feature_importance': feature_imp.to_dict('records'),
    'rule_weights': rule_weights,
    'correlation': correlation,
    'top5_xgb': top5_xgb,
    'top5_rule': top5_rule
}

with open('research/xgboost_feature_importance.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Results saved to research/xgboost_feature_importance.json")
print(f"   Total time: {time.time()-t0:.0f}s")
