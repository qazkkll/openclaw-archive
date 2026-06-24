#!/usr/bin/env python3
"""
蓝盾+绿箭 两年信号频率回测
每周采样一次，按月统计🟢🟢/🟢/🟡信号数量
"""
import json, os, warnings, numpy as np, pandas as pd, xgboost as xgb
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings('ignore')

ROOT = '/home/hermes/.hermes/openclaw-archive'

print('📊 加载数据...')
df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_yf_10y.parquet'))
df = df.rename(columns={'ticker': 'sym'})
df = df[(df['close'] > 0.5) & (df['volume'] > 0)]

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

print('⚙️ 计算特征...')
parts = []
for i, (sym, g) in enumerate(df.groupby('sym')):
    f = compute_features(g); f['sym'] = sym; parts.append(f)
    if (i+1) % 500 == 0: print(f'  {i+1}/...')
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

# 基本面
FUND = ['pe_trailing','pe_forward','div_yield','beta']
try:
    fund_daily = v75[['sym','date']+FUND]
    df = pd.merge(df, fund_daily, on=['sym','date'], how='left')
    for col in FUND:
        if col in df.columns: df[col] = df[col].fillna(df[col].median())
except:
    for col in FUND: df[col] = 0

# 加载模型
shield_model = xgb.Booster()
shield_model.load_model(os.path.join(ROOT, 'models/us/blueshield_v8_xgb.json'))
shield_meta = json.load(open(os.path.join(ROOT, 'models/us/blueshield_v8_meta.json')))
shield_feats = shield_meta['features']

arrow_model = xgb.Booster()
arrow_model.load_model(os.path.join(ROOT, 'models/us/arrow_v12_xgb.json'))
arrow_meta = json.load(open(os.path.join(ROOT, 'models/us/arrow_v12_meta.json')))
arrow_feats = arrow_meta['features']

# 获取最近2年的交易日（每周采样1天）
all_dates = sorted(df['date'].unique())
two_years_ago = all_dates[-1] - pd.Timedelta(days=730)
recent_dates = [d for d in all_dates if d >= two_years_ago]

# 每周采样（取每周第一个交易日）
weekly_dates = []
last_week = None
for d in recent_dates:
    week = d.isocalendar()[1]
    year = d.year
    if (year, week) != last_week:
        weekly_dates.append(d)
        last_week = (year, week)

print(f'\n📅 回测区间: {str(recent_dates[0])[:10]} ~ {str(recent_dates[-1])[:10]}')
print(f'📊 采样: {len(weekly_dates)}周 (每周1天)')
print('='*70)

# 按月统计
shield_monthly = defaultdict(lambda: {'🟢🟢':0, '🟢':0, '🟡':0, 'total':0, 'dates':0})
arrow_monthly = defaultdict(lambda: {'🟢🟢':0, '🟢':0, '🟡':0, 'total':0, 'dates':0})

for i, date in enumerate(weekly_dates):
    date_str = str(date)[:10]
    month_key = date_str[:7]  # YYYY-MM
    
    day_data = df[df['date'] == date]
    if len(day_data) < 100:
        continue
    
    # 蓝盾评分（>$10）
    shield_day = day_data[day_data['close'] > 10].dropna(subset=shield_feats)
    if len(shield_day) > 50:
        X = shield_day[shield_feats].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        dtest = xgb.DMatrix(X, feature_names=shield_feats)
        preds = shield_model.predict(dtest)
        
        g2 = sum(1 for p in preds if p >= 0.90)
        g1 = sum(1 for p in preds if 0.80 <= p < 0.90)
        y = sum(1 for p in preds if 0.70 <= p < 0.80)
        
        shield_monthly[month_key]['🟢🟢'] += g2
        shield_monthly[month_key]['🟢'] += g1
        shield_monthly[month_key]['🟡'] += y
        shield_monthly[month_key]['total'] += g2 + g1 + y
        shield_monthly[month_key]['dates'] += 1
    
    # 绿箭评分（$1-$10）
    arrow_day = day_data[day_data['close'].between(0.5, 10)].dropna(subset=arrow_feats)
    if len(arrow_day) > 50:
        X = arrow_day[arrow_feats].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        dtest = xgb.DMatrix(X, feature_names=arrow_feats)
        preds = arrow_model.predict(dtest)
        
        g2 = sum(1 for p in preds if p >= 0.90)
        g1 = sum(1 for p in preds if 0.80 <= p < 0.90)
        y = sum(1 for p in preds if 0.70 <= p < 0.80)
        
        arrow_monthly[month_key]['🟢🟢'] += g2
        arrow_monthly[month_key]['🟢'] += g1
        arrow_monthly[month_key]['🟡'] += y
        arrow_monthly[month_key]['total'] += g2 + g1 + y
        arrow_monthly[month_key]['dates'] += 1
    
    if (i+1) % 10 == 0:
        print(f'  进度: {i+1}/{len(weekly_dates)} ({date_str})', flush=True)

# 输出结果
print('\n' + '='*70)
print('🛡️ 蓝盾V6 月度信号频率（>$10，每周采样）')
print('='*70)
print(f'{"月份":<10} {"🟢🟢":>5} {"🟢":>5} {"🟡":>5} {"合计":>6} {"周数":>4} {"周均":>6}')
print('-'*50)

total_g2 = total_g1 = total_y = total_all = 0
for month in sorted(shield_monthly.keys()):
    d = shield_monthly[month]
    wk_avg = d['total'] / d['dates'] if d['dates'] > 0 else 0
    print(f'{month:<10} {d["🟢🟢"]:>5} {d["🟢"]:>5} {d["🟡"]:>5} {d["total"]:>6} {d["dates"]:>4} {wk_avg:>6.1f}')
    total_g2 += d['🟢🟢']
    total_g1 += d['🟢']
    total_y += d['🟡']
    total_all += d['total']

n_months = len(shield_monthly)
print('-'*50)
print(f'{"合计":<10} {total_g2:>5} {total_g1:>5} {total_y:>5} {total_all:>6}')
print(f'{"月均":<10} {total_g2/n_months:>5.1f} {total_g1/n_months:>5.1f} {total_y/n_months:>5.1f} {total_all/n_months:>6.1f}')

print('\n' + '='*70)
print('🎯 绿箭V11 月度信号频率（$1-$10，每周采样）')
print('='*70)
print(f'{"月份":<10} {"🟢🟢":>5} {"🟢":>5} {"🟡":>5} {"合计":>6} {"周数":>4} {"周均":>6}')
print('-'*50)

total_g2 = total_g1 = total_y = total_all = 0
for month in sorted(arrow_monthly.keys()):
    d = arrow_monthly[month]
    wk_avg = d['total'] / d['dates'] if d['dates'] > 0 else 0
    print(f'{month:<10} {d["🟢🟢"]:>5} {d["🟢"]:>5} {d["🟡"]:>5} {d["total"]:>6} {d["dates"]:>4} {wk_avg:>6.1f}')
    total_g2 += d['🟢🟢']
    total_g1 += d['🟢']
    total_y += d['🟡']
    total_all += d['total']

n_months = len(arrow_monthly)
print('-'*50)
print(f'{"合计":<10} {total_g2:>5} {total_g1:>5} {total_y:>5} {total_all:>6}')
print(f'{"月均":<10} {total_g2/n_months:>5.1f} {total_g1/n_months:>5.1f} {total_y/n_months:>5.1f} {total_all/n_months:>6.1f}')

# 保存
result = {
    'shield': dict(shield_monthly),
    'arrow': dict(arrow_monthly),
    'period': f'{str(recent_dates[0])[:10]} ~ {str(recent_dates[-1])[:10]}',
    'sampled_weeks': len(weekly_dates)
}
# Convert defaultdict to dict for JSON
result['shield'] = {k: dict(v) for k, v in shield_monthly.items()}
result['arrow'] = {k: dict(v) for k, v in arrow_monthly.items()}
json.dump(result, open(os.path.join(ROOT, 'output/signal_frequency_2yr.json'), 'w'), indent=2)
print(f'\n✅ 保存: output/signal_frequency_2yr.json')
