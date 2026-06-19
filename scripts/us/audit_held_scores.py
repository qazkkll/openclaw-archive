#!/usr/bin/env python3
"""审计：持仓在模型中的评分"""
import json, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ROOT = '/home/hermes/.hermes/openclaw-archive'

df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_yf_10y.parquet'))
df = df.rename(columns={'ticker': 'sym'})
df = df[(df['close'] > 0.5) & (df['volume'] > 0)]

def compute_features(group):
    g = group.sort_values('date').copy()
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

print('计算特征...')
parts = []
for i, (sym, g) in enumerate(df.groupby('sym')):
    f = compute_features(g); f['sym'] = sym; parts.append(f)
    if (i+1) % 500 == 0: print(f'  {i+1}...')
df = pd.concat(parts, ignore_index=True)

MACRO = ['vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60','qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60','iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60']
try:
    v75 = pd.read_parquet(os.path.join(ROOT, 'data/us/features/us_ml_feats_v75_filtered.parquet'))
    macro_daily = v75[['date']+MACRO].drop_duplicates(subset=['date'])
    df = pd.merge(df, macro_daily, on='date', how='left')
    for col in MACRO:
        if col in df.columns: df[col] = df[col].ffill().fillna(0)
except:
    for col in MACRO: df[col] = 0

FUND = ['pe_trailing','pe_forward','div_yield','beta']
try:
    fund_daily = v75[['sym','date']+FUND]
    df = pd.merge(df, fund_daily, on=['sym','date'], how='left')
    for col in FUND:
        if col in df.columns: df[col] = df[col].fillna(df[col].median())
except:
    for col in FUND: df[col] = 0

ALL = ['ma5','ma20','ma60','ma_bias20','ma_align','price_position','ret1','ret5','ret20','ret60','momentum_6m','momentum_1m','mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change','rsi14','rsi_change','macd','macd_signal','macd_hist','bb_std','bb_width','bb_pos','ret_quality'] + MACRO + FUND

import xgboost as xgb

# 蓝盾V6持仓评分
shield_held = ['ASML','ANET','COHR','CARR','NET']
shield = df[df['sym'].isin(shield_held)]
latest = shield.groupby('sym').last().reset_index()
latest = latest.dropna(subset=ALL)

model = xgb.Booster()
model.load_model(os.path.join(ROOT, 'models/us/blueshield_v6_xgb.json'))
meta = json.load(open(os.path.join(ROOT, 'models/us/blueshield_v6_meta.json')))
feats = meta['features']

X = latest[feats].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
dtest = xgb.DMatrix(X, feature_names=feats)
preds = model.predict(dtest)
latest['score'] = preds

# 全市场排名
all_shield = df[df['close'] > 10]
all_latest = all_shield.groupby('sym').last().reset_index()
all_latest = all_latest.dropna(subset=ALL)
X_all = all_latest[feats].values.astype(np.float32)
X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)
dtest_all = xgb.DMatrix(X_all, feature_names=feats)
preds_all = model.predict(dtest_all)
all_latest['score'] = preds_all
all_latest = all_latest.sort_values('score', ascending=False).reset_index(drop=True)

print('\n' + '='*60)
print('🛡️ 蓝盾V6 持仓评分 + 全市场排名')
print('='*60)
for _, r in latest.sort_values('score', ascending=False).iterrows():
    s = r['score']
    rank = int(all_latest[all_latest['sym'] == r['sym']].index[0]) + 1
    total = len(all_latest)
    pct = rank / total * 100
    sig = '🟢🟢' if pct <= 5 else '🟢' if pct <= 10 else '🟡' if pct <= 20 else '🔴'
    print(f'  {r["sym"]:6} ${r["close"]:>8.2f}  score={s:.4f} {sig}  排名{rank}/{total} (前{pct:.0f}%)  RSI={r.get("rsi14",0):.0f}  5d={r.get("ret5",0)*100:+.1f}%  20d={r.get("ret20",0)*100:+.1f}%')

# 绿箭V11持仓评分
arrow_held = ['NGEN','PPBT','NYXH','NXTC']
arrow = df[df['sym'].isin(arrow_held)]
latest_a = arrow.groupby('sym').last().reset_index()
latest_a = latest_a.dropna(subset=ALL)

model_a = xgb.Booster()
model_a.load_model(os.path.join(ROOT, 'models/us/arrow_v11_xgb.json'))
meta_a = json.load(open(os.path.join(ROOT, 'models/us/arrow_v11_meta.json')))
feats_a = meta_a['features']

X_a = latest_a[feats_a].values.astype(np.float32)
X_a = np.nan_to_num(X_a, nan=0, posinf=0, neginf=0)
dtest_a = xgb.DMatrix(X_a, feature_names=feats_a)
preds_a = model_a.predict(dtest_a)
latest_a['score'] = preds_a

# 全市场排名
all_arrow = df[df['close'].between(0.5, 10)]
all_latest_a = all_arrow.groupby('sym').last().reset_index()
all_latest_a = all_latest_a.dropna(subset=ALL)
X_aa = all_latest_a[feats_a].values.astype(np.float32)
X_aa = np.nan_to_num(X_aa, nan=0, posinf=0, neginf=0)
dtest_aa = xgb.DMatrix(X_aa, feature_names=feats_a)
preds_aa = model_a.predict(dtest_aa)
all_latest_a['score'] = preds_aa
all_latest_a = all_latest_a.sort_values('score', ascending=False).reset_index(drop=True)

print('\n' + '='*60)
print('🎯 绿箭V11 持仓评分 + 全市场排名')
print('='*60)
for _, r in latest_a.sort_values('score', ascending=False).iterrows():
    s = r['score']
    rank = int(all_latest_a[all_latest_a['sym'] == r['sym']].index[0]) + 1
    total = len(all_latest_a)
    pct = rank / total * 100
    sig = '🟢🟢' if pct <= 5 else '🟢' if pct <= 10 else '🟡' if pct <= 20 else '🔴'
    print(f'  {r["sym"]:6} ${r["close"]:>8.2f}  score={s:.4f} {sig}  排名{rank}/{total} (前{pct:.0f}%)  RSI={r.get("rsi14",0):.0f}  5d={r.get("ret5",0)*100:+.1f}%')

# 保存结果
result = {
    'shield': [{'sym': r['sym'], 'price': round(r['close'],2), 'score': round(r['score'],4), 
                'rank': int(all_latest[all_latest['sym']==r['sym']].index[0])+1,
                'total': len(all_latest), 'rsi': round(r.get('rsi14',0),1),
                'ret5d': round(r.get('ret5',0)*100,1), 'ret20d': round(r.get('ret20',0)*100,1)}
               for _, r in latest.iterrows()],
    'arrow': [{'sym': r['sym'], 'price': round(r['close'],2), 'score': round(r['score'],4),
               'rank': int(all_latest_a[all_latest_a['sym']==r['sym']].index[0])+1,
               'total': len(all_latest_a), 'rsi': round(r.get('rsi14',0),1),
               'ret5d': round(r.get('ret5',0)*100,1)}
              for _, r in latest_a.iterrows()]
}
json.dump(result, open(os.path.join(ROOT, 'output/held_scores.json'), 'w'), indent=2)
print('\n✅ 保存: output/held_scores.json')
