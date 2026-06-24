#!/usr/bin/env python3
"""
绿箭V11 两周回测
每天评分 → 记录🟢🟢/🟢信号 → 5天后检查盈亏
"""
import json, os, warnings, numpy as np, pandas as pd, xgboost as xgb
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

ROOT = '/home/hermes/.hermes/openclaw-archive'

# 加载数据
print('加载数据...')
df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_yf_10y.parquet'))
df = df.rename(columns={'ticker': 'sym'})
df = df[(df['close'] > 0.5) & (df['close'] < 10) & (df['volume'] > 0)]

# 特征计算
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
df = pd.concat(parts, ignore_index=True)

# 宏观特征
MACRO = ['vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60','qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60','iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60']
try:
    v75 = pd.read_parquet(os.path.join(ROOT, 'data/us/features/us_ml_feats_v75_filtered.parquet'))
    macro_daily = v75[['date']+MACRO].drop_duplicates(subset=['date'])
    df = pd.merge(df, macro_daily, on='date', how='left')
    for col in MACRO:
        if col in df.columns: df[col] = df[col].ffill().fillna(0)
except:
    for col in MACRO: df[col] = 0

# 加载模型
model = xgb.Booster()
model.load_model(os.path.join(ROOT, 'models/us/arrow_v12_xgb.json'))
meta = json.load(open(os.path.join(ROOT, 'models/us/arrow_v12_meta.json')))
feats = meta['features']

# 获取所有交易日
all_dates = sorted(df['date'].unique())
# 最近10个交易日（覆盖2周+5天持有期）
test_dates = [str(d)[:10] for d in all_dates[-15:]]

print(f'\n回测日期: {test_dates[0]} ~ {test_dates[-1]}')
print('='*70)

results = []

for date_str in test_dates:
    day_data = df[df['date'] == date_str]
    if len(day_data) < 100:
        continue
    
    # 评分
    day_clean = day_data.dropna(subset=feats)
    if len(day_clean) < 50:
        continue
    
    X = day_clean[feats].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    dtest = xgb.DMatrix(X, feature_names=feats)
    preds = model.predict(dtest)
    day_clean = day_clean.copy()
    day_clean['score'] = preds
    
    # 取Top-10
    top10 = day_clean.sort_values('score', ascending=False).head(10)
    
    # 找5天后的价格
    date_idx = all_dates.index(pd.Timestamp(date_str))
    if date_idx + 5 >= len(all_dates):
        future_date = all_dates[-1]
    else:
        future_date = all_dates[date_idx + 5]
    
    # 筛选信号
    signals = []
    for _, row in top10.iterrows():
        score = row['score']
        if score < 0.70:
            continue
        
        sym = row['sym']
        entry_price = row['close']
        
        # 5天后价格
        future_data = df[(df['sym'] == sym) & (df['date'] == future_date)]
        if len(future_data) == 0:
            continue
        
        exit_price = future_data['close'].iloc[0]
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        
        sig = '🟢🟢' if score >= 0.90 else '🟢' if score >= 0.80 else '🟡'
        
        signals.append({
            'date': date_str,
            'ticker': sym,
            'score': round(score, 4),
            'signal': sig,
            'entry': round(entry_price, 2),
            'exit': round(exit_price, 2),
            'pnl': round(pnl_pct, 1),
            'hit': pnl_pct > 0
        })
    
    results.extend(signals)
    
    # 打印当日
    if signals:
        print(f'\n📅 {date_str} ({len(signals)}个信号)')
        for s in signals:
            hit = '✅' if s['hit'] else '❌'
            print(f'  {s["signal"]} {s["ticker"]:6} score={s["score"]:.3f} ${s["entry"]:.2f}→${s["exit"]:.2f} {s["pnl"]:+.1f}% {hit}')
    else:
        print(f'\n📅 {date_str} (无≥0.70信号)')

# 汇总
print('\n' + '='*70)
print('📊 两周回测汇总')
print('='*70)

if not results:
    print('无有效信号')
else:
    total = len(results)
    winners = sum(1 for r in results if r['hit'])
    losers = total - winners
    wr = winners / total * 100
    avg_pnl = np.mean([r['pnl'] for r in results])
    
    print(f'总信号: {total}个')
    print(f'胜率: {wr:.0f}% ({winners}赢/{losers}亏)')
    print(f'平均盈亏: {avg_pnl:+.1f}%')
    
    # 按信号级别
    for sig in ['🟢🟢', '🟢', '🟡']:
        sig_results = [r for r in results if r['signal'] == sig]
        if sig_results:
            sig_wr = sum(1 for r in sig_results if r['hit']) / len(sig_results) * 100
            sig_avg = np.mean([r['pnl'] for r in sig_results])
            print(f'\n{sig} ({len(sig_results)}个):')
            print(f'  胜率: {sig_wr:.0f}% | 平均: {sig_avg:+.1f}%')
            for r in sig_results:
                hit = '✅' if r['hit'] else '❌'
                print(f'    {r["ticker"]:6} {r["entry"]:.2f}→{r["exit"]:.2f} {r["pnl"]:+.1f}% {hit}')
    
    # Top赢家/输家
    sorted_results = sorted(results, key=lambda x: -x['pnl'])
    print(f'\n🏆 Top赢家:')
    for r in sorted_results[:3]:
        print(f'  {r["date"]} {r["signal"]} {r["ticker"]} {r["pnl"]:+.1f}%')
    print(f'\n💀 Top输家:')
    for r in sorted_results[-3:]:
        print(f'  {r["date"]} {r["signal"]} {r["ticker"]} {r["pnl"]:+.1f}%')

# 保存
json.dump(results, open(os.path.join(ROOT, 'output/arrow_2week_backtest.json'), 'w'), indent=2)
print(f'\n✅ 保存: output/arrow_2week_backtest.json')
