#!/usr/bin/env python3
"""
持仓评分 + 全市场排名 (V8/V12 2026-06-25升级)
蓝盾V8: 29特征(27技术+vix_close+spy_ret20)
绿箭V12: 42特征(29技术+13宏观)
"""
import json, os
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = '/home/hermes/.hermes/openclaw-archive'

def compute_features(g):
    g = g.sort_values('date').copy()
    c = g['close']
    g['ma5'] = c.rolling(5).mean(); g['ma20'] = c.rolling(20).mean(); g['ma60'] = c.rolling(60).mean()
    g['ma_bias20'] = (c - g['ma20']) / g['ma20']
    g['ma_align'] = ((c > g['ma5']).astype(int) + (g['ma5'] > g['ma20']).astype(int))
    mn60 = c.rolling(60).min(); mx60 = c.rolling(60).max()
    g['price_position'] = (c - mn60) / (mx60 - mn60 + 1e-10)
    g['ret1'] = c.pct_change(1); g['ret5'] = c.pct_change(5)
    g['ret20'] = c.pct_change(20); g['ret60'] = c.pct_change(60)
    g['momentum_6m'] = c.pct_change(126); g['momentum_1m'] = c.pct_change(21)
    g['mom_divergence'] = g['momentum_1m'] - g['ret20']
    g['trend_accel'] = g['ret5'] - g['ret5'].shift(5)
    dr = c.pct_change(1)
    g['vol20'] = dr.rolling(20).std(); g['vol5'] = dr.rolling(5).std()
    g['vol_ratio'] = g['volume'] / g['volume'].rolling(20).mean()
    g['vol_change'] = g['vol20'] / g['vol20'].shift(20)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    g['rsi14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    g['rsi_change'] = g['rsi14'].diff(5)
    e12 = c.ewm(span=12).mean(); e26 = c.ewm(span=26).mean()
    g['macd'] = e12 - e26; g['macd_signal'] = g['macd'].ewm(span=9).mean()
    g['macd_hist'] = g['macd'] - g['macd_signal']
    g['bb_std'] = c.rolling(20).std()
    g['bb_width'] = 2 * g['bb_std'] / g['ma20']
    g['bb_pos'] = (c - g['ma20']) / (2 * g['bb_std'] + 1e-10)
    g['ret_quality'] = g['ret20'] / (g['vol20'] + 1e-10)
    g['price'] = c
    g['range_pct'] = (g['high'] - g['low']) / (c + 1e-10)
    return g

# ========== 加载数据 ==========
print('加载数据...')
df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'))
df = df.dropna(subset=['close', 'volume'])
df = df[(df['close'] > 0.5) & (df['volume'] > 0)]

# 只取最近250天
from datetime import datetime, timedelta
cutoff = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
df = df[df['date'] >= cutoff]
print(f'  {len(df)}行, {df["sym"].nunique()}只')

# ========== 计算技术特征 ==========
print('计算特征...')
parts = []
for i, (sym, g) in enumerate(df.groupby('sym')):
    f = compute_features(g); f['sym'] = sym; parts.append(f)
    if (i+1) % 2000 == 0: print(f'  {i+1}...')
df = pd.concat(parts, ignore_index=True)

# ========== 计算宏观特征（从主数据，不依赖v75） ==========
print('计算宏观特征...')
# VIX
try:
    vix_raw = pd.read_parquet(os.path.join(ROOT, 'data/us/vix_10y.parquet'))
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = [c[0] if isinstance(c, tuple) else c for c in vix_raw.columns]
    vix_raw = vix_raw.reset_index()
    vix_col = [c for c in vix_raw.columns if 'close' in c.lower() or 'Close' in c][0]
    date_col = [c for c in vix_raw.columns if 'date' in c.lower() or 'Date' in c][0]
    vix = pd.DataFrame({'date': pd.to_datetime(vix_raw[date_col]), 'vix_close': vix_raw[vix_col].astype(float)})
except:
    vix = pd.DataFrame({'date': df['date'].unique(), 'vix_close': 19.5})

# SPY/QQQ/IWM returns
macro_syms = {'SPY': 'spy', 'QQQ': 'qqq', 'IWM': 'iwm'}
macro_dfs = {}
for sym, prefix in macro_syms.items():
    s = df[df['sym'] == sym][['date', 'close']].sort_values('date').copy()
    for d in [1, 5, 20, 60]:
        s[f'{prefix}_ret{d}'] = s['close'].pct_change(d)
    macro_dfs[sym] = s.drop(columns=['close'])

MACRO_COLS = ['vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60',
              'qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60',
              'iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60']

# Merge macro by date
df = df.merge(vix, on='date', how='left')
for sym, mdf in macro_dfs.items():
    df = df.merge(mdf, on='date', how='left')
for col in MACRO_COLS:
    if col in df.columns:
        df[col] = df[col].ffill().fillna(0)
    else:
        df[col] = 0

# ========== 蓝盾V8持仓评分 ==========
print('\n蓝盾V8 持仓评分...')
model = xgb.Booster()
model.load_model(os.path.join(ROOT, 'models/us/blueshield_v8_xgb.json'))
meta = json.load(open(os.path.join(ROOT, 'models/us/blueshield_v8_meta.json')))
feats = meta['features']  # 29 features

shield_held = ['ASML', 'ANET', 'COHR', 'CARR', 'NET', 'CSGP']
shield = df[df['sym'].isin(shield_held)]
latest = shield.groupby('sym').tail(1).reset_index(drop=True)
latest = latest.dropna(subset=[f for f in feats if f in latest.columns])

if len(latest) > 0:
    X = latest[feats].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    dtest = xgb.DMatrix(X, feature_names=feats)
    preds = model.predict(dtest)
    latest['score'] = preds

    # 全市场排名
    all_shield = df[df['close'] > 10]
    all_latest = all_shield.groupby('sym').tail(1).reset_index(drop=True)
    all_latest = all_latest.dropna(subset=[f for f in feats if f in all_latest.columns])
    X_all = all_latest[feats].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)
    dtest_all = xgb.DMatrix(X_all, feature_names=feats)
    preds_all = model.predict(dtest_all)
    all_latest['score'] = preds_all
    all_latest = all_latest.sort_values('score', ascending=False).reset_index(drop=True)

    print('\n' + '='*60)
    print('🛡️ 蓝盾V8 持仓评分 + 全市场排名')
    print('='*60)
    for _, r in latest.sort_values('score', ascending=False).iterrows():
        s = r['score']
        rank = int(all_latest[all_latest['sym'] == r['sym']].index[0]) + 1
        total = len(all_latest)
        pct = rank / total * 100
        sig = '🟢🟢' if pct <= 5 else '🟢' if pct <= 10 else '🟡' if pct <= 20 else '🔴'
        print(f'  {r["sym"]:6} ${r["close"]:>8.2f}  score={s:.4f} {sig}  排名{rank}/{total} (前{pct:.0f}%)  RSI={r.get("rsi14",0):.0f}  5d={r.get("ret5",0)*100:+.1f}%  20d={r.get("ret20",0)*100:+.1f}%')
else:
    print('  ⚠️ 无蓝盾持仓数据')

# ========== 绿箭V12持仓评分 ==========
print('\n绿箭V12 持仓评分...')
model_a = xgb.Booster()
model_a.load_model(os.path.join(ROOT, 'models/us/arrow_v12_xgb.json'))
meta_a = json.load(open(os.path.join(ROOT, 'models/us/arrow_v12_meta.json')))
feats_a = meta_a['features']  # 42 features

arrow_held = ['FATE', 'NYXH', 'ZEPP']
arrow = df[df['sym'].isin(arrow_held)]
latest_a = arrow.groupby('sym').tail(1).reset_index(drop=True)
latest_a = latest_a.dropna(subset=[f for f in feats_a if f in latest_a.columns])

if len(latest_a) > 0:
    X_a = latest_a[feats_a].values.astype(np.float32)
    X_a = np.nan_to_num(X_a, nan=0, posinf=0, neginf=0)
    dtest_a = xgb.DMatrix(X_a, feature_names=feats_a)
    preds_a = model_a.predict(dtest_a)
    latest_a['score'] = preds_a

    # 全市场排名
    all_arrow = df[df['close'].between(0.5, 10)]
    all_latest_a = all_arrow.groupby('sym').tail(1).reset_index(drop=True)
    all_latest_a = all_latest_a.dropna(subset=[f for f in feats_a if f in all_latest_a.columns])
    X_aa = all_latest_a[feats_a].values.astype(np.float32)
    X_aa = np.nan_to_num(X_aa, nan=0, posinf=0, neginf=0)
    dtest_aa = xgb.DMatrix(X_aa, feature_names=feats_a)
    preds_aa = model_a.predict(dtest_aa)
    all_latest_a['score'] = preds_aa
    all_latest_a = all_latest_a.sort_values('score', ascending=False).reset_index(drop=True)

    print('\n' + '='*60)
    print('🎯 绿箭V12 持仓评分 + 全市场排名')
    print('='*60)
    for _, r in latest_a.sort_values('score', ascending=False).iterrows():
        s = r['score']
        rank = int(all_latest_a[all_latest_a['sym'] == r['sym']].index[0]) + 1
        total = len(all_latest_a)
        pct = rank / total * 100
        sig = '🟢🟢' if pct <= 5 else '🟢' if pct <= 10 else '🟡' if pct <= 20 else '🔴'
        print(f'  {r["sym"]:6} ${r["close"]:>8.2f}  score={s:.4f} {sig}  排名{rank}/{total} (前{pct:.0f}%)  RSI={r.get("rsi14",0):.0f}  5d={r.get("ret5",0)*100:+.1f}%  20d={r.get("ret20",0)*100:+.1f}%')
else:
    print('  ⚠️ 无绿箭持仓数据')

print('\n✅ 持仓评分完成')
