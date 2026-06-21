#!/usr/bin/env python3
"""
CEO: XGBoost Feature Importance Analysis
Which of the 25 features matter most?
"""

import pandas as pd, numpy as np, json, time, os, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

t0 = time.time()
print(f"[CEO] Feature Importance {time.strftime('%Y-%m-%d %H:%M')}")

# Load
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
print(f"  Data: {len(df):,} ({time.time()-t0:.0f}s)")

# Features
print("  Features...")
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
df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df.groupby('sym')['close'].shift(1)), abs(df['low'] - df.groupby('sym')['close'].shift(1))))
df['atr14'] = df.groupby('sym')['tr'].transform(lambda x: x.rolling(14, min_periods=1).mean())
df['atr_pct'] = df['atr14'] / df['close']
df['vol_ratio'] = df.groupby('sym')['volume'].transform(lambda x: x.rolling(5).mean()) / df.groupby('sym')['volume'].transform(lambda x: x.rolling(20).mean())
for col in ['total_net', 'lg_net', 'md_net', 'elg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())
    df[f'{col}_20d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(20, min_periods=1).sum())
    df[f'{col}_5d_rk'] = df.groupby('date')[f'{col}_5d'].rank(pct=True)
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')
df['fwd_10d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-10) / x - 1)

XGB_FEATURES = [
    'ret5', 'ret10', 'ret20', 'ma20_bias', 'ma60_bias',
    'vol5', 'vol20', 'rsi_14', 'macd_hist', 'atr_pct', 'vol_ratio',
    'total_net_5d', 'lg_net_5d', 'md_net_5d', 'elg_net_5d',
    'total_net_20d', 'lg_net_20d', 'md_net_20d', 'elg_net_20d',
    'total_net_5d_rk', 'lg_net_5d_rk', 'md_net_5d_rk', 'elg_net_5d_rk',
    'breadth', 'mkt_ret20'
]

# Train on full dataset (2016-2024) and test on 2025-2026
train = df[(df['date'] >= 20160101) & (df['date'] <= 20241231)].dropna(subset=XGB_FEATURES + ['fwd_10d'])
test = df[(df['date'] >= 20250101) & (df['date'] <= 20261231)].dropna(subset=XGB_FEATURES + ['fwd_10d'])

print(f"  Train: {len(train):,}, Test: {len(test):,}")

import xgboost as xgb

model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=4, verbosity=0)
model.fit(train[XGB_FEATURES].fillna(0), train['fwd_10d'])

# Feature importance
importance = model.feature_importances_
feat_imp = sorted(zip(XGB_FEATURES, importance), key=lambda x: x[1], reverse=True)

print("\n📊 Feature Importance (Top 25)")
print(f"{'Rank':>4} {'Feature':>20} {'Importance':>10} {'Type':>15}")
print("-" * 55)

feature_types = {
    'ret5': 'momentum', 'ret10': 'momentum', 'ret20': 'momentum',
    'ma20_bias': 'trend', 'ma60_bias': 'trend',
    'vol5': 'volatility', 'vol20': 'volatility',
    'rsi_14': 'technical', 'macd_hist': 'technical',
    'atr_pct': 'volatility', 'vol_ratio': 'volume',
    'total_net_5d': 'flow', 'lg_net_5d': 'flow', 'md_net_5d': 'flow', 'elg_net_5d': 'flow',
    'total_net_20d': 'flow', 'lg_net_20d': 'flow', 'md_net_20d': 'flow', 'elg_net_20d': 'flow',
    'total_net_5d_rk': 'flow_rank', 'lg_net_5d_rk': 'flow_rank', 'md_net_5d_rk': 'flow_rank', 'elg_net_5d_rk': 'flow_rank',
    'breadth': 'market', 'mkt_ret20': 'market'
}

for i, (feat, imp) in enumerate(feat_imp):
    ftype = feature_types.get(feat, 'unknown')
    print(f"{i+1:>4} {feat:>20} {imp:>10.4f} {ftype:>15}")

# Group by type
type_imp = {}
for feat, imp in feat_imp:
    ftype = feature_types.get(feat, 'unknown')
    if ftype not in type_imp:
        type_imp[ftype] = 0
    type_imp[ftype] += imp

print("\n📊 Feature Importance by Type")
print(f"{'Type':>15} {'Importance':>10} {'Pct':>8}")
print("-" * 35)
for ftype, imp in sorted(type_imp.items(), key=lambda x: x[1], reverse=True):
    print(f"{ftype:>15} {imp:>10.4f} {imp/sum(type_imp.values()):>8.1%}")

# Save
output = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'feature_importance': [{'feature': f, 'importance': float(i), 'type': feature_types.get(f, 'unknown')} for f, i in feat_imp],
    'type_importance': {k: float(v) for k, v in type_imp.items()}
}
with open('research/ceo_xgb_feature_importance_v2.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n  Time: {time.time()-t0:.0f}s")
