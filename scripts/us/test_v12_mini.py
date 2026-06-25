#!/usr/bin/env python3
"""Minimal V12 test - just 50 symbols to check if prediction works"""
import json, os, numpy as np, pandas as pd, xgboost as xgb
from datetime import datetime, timedelta

ROOT = '/home/hermes/.hermes/openclaw-archive'

print("1. Loading data...")
df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'))
df = df.dropna(subset=['close', 'volume'])
df = df[(df['close'] > 0.5) & (df['volume'] > 0)]
cutoff = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
df = df[df['date'] >= cutoff]
print("Data: {} rows, {} syms".format(len(df), df['sym'].nunique()))

# Only 50 symbols
syms = df['sym'].unique()[:50]
df = df[df['sym'].isin(syms)].copy()
print("Subset: {} rows, {} syms".format(len(df), len(syms)))

print("2. Computing features for 50 syms...")
def compute_features(group):
    g = group.sort_values('date').copy()
    c = g['close']
    g['ma5'] = c.rolling(5).mean()
    g['ma20'] = c.rolling(20).mean()
    g['ma60'] = c.rolling(60).mean()
    g['ma_bias20'] = (c - g['ma20']) / g['ma20']
    g['ma_align'] = ((c > g['ma5']).astype(int) + (g['ma5'] > g['ma20']).astype(int))
    mn60 = c.rolling(60).min()
    mx60 = c.rolling(60).max()
    g['price_position'] = (c - mn60) / (mx60 - mn60 + 1e-10)
    g['ret1'] = c.pct_change(1)
    g['ret5'] = c.pct_change(5)
    g['ret20'] = c.pct_change(20)
    g['ret60'] = c.pct_change(60)
    g['momentum_6m'] = c.pct_change(126)
    g['momentum_1m'] = c.pct_change(21)
    g['mom_divergence'] = g['momentum_1m'] - g['ret20']
    g['trend_accel'] = g['ret5'] - g['ret5'].shift(5)
    dr = c.pct_change(1)
    g['vol20'] = dr.rolling(20).std()
    g['vol5'] = dr.rolling(5).std()
    g['vol_ratio'] = g['volume'] / g['volume'].rolling(20).mean()
    g['vol_change'] = g['vol20'] / g['vol20'].shift(20)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    g['rsi14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    g['rsi_change'] = g['rsi14'].diff(5)
    e12 = c.ewm(span=12).mean()
    e26 = c.ewm(span=26).mean()
    g['macd'] = e12 - e26
    g['macd_signal'] = g['macd'].ewm(span=9).mean()
    g['macd_hist'] = g['macd'] - g['macd_signal']
    g['bb_std'] = dr.rolling(20).std()
    bb_mid = c.rolling(20).mean()
    g['bb_width'] = 4 * g['bb_std'] * bb_mid / (bb_mid + 1e-10)
    std20 = c.rolling(20).std()
    g['bb_pos'] = (c - (bb_mid - 2 * std20)) / (4 * std20 + 1e-10)
    ret_pos = dr.clip(lower=0).rolling(20).mean()
    ret_neg = (-dr).clip(lower=0).rolling(20).mean()
    g['ret_quality'] = ret_pos / (ret_pos + ret_neg + 1e-10)
    g['price'] = c
    g['range_pct'] = (g['high'] - g['low']) / (c + 1e-10)
    return g

parts = []
for sym, g in df.groupby('sym'):
    f = compute_features(g)
    f['sym'] = sym
    parts.append(f)
df = pd.concat(parts, ignore_index=True)
print("Features done: {} rows".format(len(df)))

df = df.sort_values('date')
latest = df.groupby('sym').tail(1).reset_index(drop=True)

# Macro
try:
    v75 = pd.read_parquet(os.path.join(ROOT, 'data/us/features/us_ml_feats_v75_filtered.parquet'))
    MACRO_COLS = ['vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60',
                  'qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60',
                  'iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60']
    macro_daily = v75[['date'] + MACRO_COLS].drop_duplicates(subset=['date'])
    latest = pd.merge(latest, macro_daily, on='date', how='left')
    for col in MACRO_COLS:
        if col in latest.columns:
            latest[col] = latest[col].ffill().fillna(0)
except Exception as e:
    print("Macro: {}".format(e))

# Filter
latest = latest[(latest['close'] >= 1) & (latest['close'] < 10)].copy()
latest = latest[latest['volume'] > 50000].copy()

TECH_FEATS = ['ma5','ma20','ma60','ma_bias20','ma_align','price_position',
    'ret1','ret5','ret20','ret60','momentum_6m','momentum_1m',
    'mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change',
    'rsi14','rsi_change','macd','macd_signal','macd_hist',
    'bb_std','bb_width','bb_pos','ret_quality','price','range_pct']
MACRO_COLS = ['vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60',
              'qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60',
              'iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60']
ALL_FEATS = TECH_FEATS + MACRO_COLS

latest = latest.dropna(subset=ALL_FEATS)
print("After filter: {} stocks".format(len(latest)))

if len(latest) == 0:
    print("No stocks after filter! Try wider filter.")
    # Check what's available
    df2 = df.groupby('sym').tail(1)
    df2 = df2[(df2['close'] >= 1) & (df2['close'] < 10)]
    print("Close in $1-$10: {}".format(len(df2)))
    df2 = df2[df2['volume'] > 50000]
    print("Volume > 50k: {}".format(len(df2)))
else:
    # Load model
    model = xgb.Booster()
    model.load_model(os.path.join(ROOT, 'models/us/arrow_v12_xgb.json'))
    with open(os.path.join(ROOT, 'models/us/arrow_v12_meta.json')) as f:
        meta = json.load(f)
    feats = meta['features']

    X = latest[feats].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    print("X shape: {}".format(X.shape))

    dtest = xgb.DMatrix(X, feature_names=feats)
    preds = model.predict(dtest)
    latest['pred_rank'] = preds
    latest = latest.sort_values('pred_rank', ascending=False)

    print("\nTop 5:")
    for _, r in latest.head(5).iterrows():
        print("  {} ${:.2f} pred={:.4f}".format(r['sym'], r['close'], r['pred_rank']))

    print("\nScore stats:")
    print("  min={:.4f} max={:.4f} mean={:.4f}".format(preds.min(), preds.max(), preds.mean()))
    print("  P50={:.4f} P90={:.4f} P95={:.4f}".format(
        np.percentile(preds, 50), np.percentile(preds, 90), np.percentile(preds, 95)))

    # Check feature matches
    print("\nMeta features ({})".format(len(feats)))
    print("Computed features ({})".format(len(ALL_FEATS)))
    missing = [f for f in feats if f not in ALL_FEATS]
    print("Missing from compute: {}".format(missing))

    print("\nSUCCESS")
