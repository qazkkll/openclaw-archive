#!/usr/bin/env python3
"""
三层过滤信号系统回测
L1: VIX市场状态（>30关闭信号）
L2: 绝对门槛（>中位数）
L3: 百分位（Top5%=🟢🟢, Top10%=🟢, Top20%=🟡）

对比原版绝对阈值（≥0.90/0.80/0.70）的效果
"""
import json, os, warnings, numpy as np, pandas as pd, xgboost as xgb
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings('ignore')

ROOT = '/home/hermes/.hermes/openclaw-archive'
OUT_DIR = os.path.join(ROOT, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

print('📊 加载数据...')
df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_yf_10y.parquet'))
df = df.rename(columns={'ticker': 'sym'})
df = df[(df['close'] > 0.5) & (df['volume'] > 0)]

# ========== 特征计算 ==========
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

print('⚙️ 计算特征（约3-5分钟）...')
parts = []
for i, (sym, g) in enumerate(df.groupby('sym')):
    f = compute_features(g); f['sym'] = sym; parts.append(f)
    if (i+1) % 500 == 0: print(f'  {i+1}/{df["sym"].nunique()}', flush=True)
df = pd.concat(parts, ignore_index=True)

# 宏观特征
MACRO = ['vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60',
         'qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60',
         'iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60']
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

# ========== 加载模型 ==========
print('🤖 加载模型...')
shield_model = xgb.Booster()
shield_model.load_model(os.path.join(ROOT, 'models/us/blueshield_v6_xgb.json'))
shield_meta = json.load(open(os.path.join(ROOT, 'models/us/blueshield_v6_meta.json')))
shield_feats = shield_meta['features']

arrow_model = xgb.Booster()
arrow_model.load_model(os.path.join(ROOT, 'models/us/arrow_v11_xgb.json'))
arrow_meta = json.load(open(os.path.join(ROOT, 'models/us/arrow_v11_meta.json')))
arrow_feats = arrow_meta['features']

# ========== 计算前向收益 ==========
print('📈 计算前向收益...')
# 为每个股票计算未来5天和20天收益
for hold in [5, 20]:
    df[f'fwd_{hold}d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-hold) / x - 1)

# ========== 回测 ==========
all_dates = sorted(df['date'].unique())
# 最近2年
two_years_ago = all_dates[-1] - pd.Timedelta(days=730)
recent_dates = [d for d in all_dates if d >= two_years_ago]

# 每周采样
weekly_dates = []
last_week = None
for d in recent_dates:
    week = d.isocalendar()[1]
    year = d.year
    if (year, week) != last_week:
        weekly_dates.append(d)
        last_week = (year, week)

print(f'\n📅 回测区间: {str(recent_dates[0])[:10]} ~ {str(recent_dates[-1])[:10]}')
print(f'📊 采样: {len(weekly_dates)}周')
print('='*70)

# ========== 方案对比 ==========
# 方案A: 原版绝对阈值 (≥0.90/0.80/0.70)
# 方案B: 三层过滤 (VIX>30关闭 + >中位数 + 百分位Top5/10/20)
# 方案C: 纯百分位 (无VIX, 无中位数门槛)

results = {
    'A_absolute': {'shield': defaultdict(list), 'arrow': defaultdict(list)},
    'B_three_layer': {'shield': defaultdict(list), 'arrow': defaultdict(list)},
    'C_percentile_only': {'shield': defaultdict(list), 'arrow': defaultdict(list)},
}

for i, date in enumerate(weekly_dates):
    date_str = str(date)[:10]
    month_key = date_str[:7]
    day_data = df[df['date'] == date]
    if len(day_data) < 100: continue
    
    # 获取当日VIX
    vix_val = day_data['vix_close'].iloc[0] if 'vix_close' in day_data.columns else 20
    if pd.isna(vix_val): vix_val = 20
    
    # ===== 蓝盾 (>$10) =====
    shield_day = day_data[day_data['close'] > 10].dropna(subset=shield_feats)
    if len(shield_day) > 50:
        X = shield_day[shield_feats].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        dtest = xgb.DMatrix(X, feature_names=shield_feats)
        preds = shield_model.predict(dtest)
        
        scores_df = pd.DataFrame({
            'sym': shield_day['sym'].values,
            'score': preds,
            'fwd_5d': shield_day['fwd_5d'].values if 'fwd_5d' in shield_day else np.nan,
            'fwd_20d': shield_day['fwd_20d'].values if 'fwd_20d' in shield_day else np.nan,
            'price': shield_day['close'].values,
        })
        
        median_score = np.median(preds)
        p95 = np.percentile(preds, 95)
        p90 = np.percentile(preds, 90)
        p80 = np.percentile(preds, 80)
        
        # 方案A: 原版绝对阈值
        for threshold, level in [(0.90, '🟢🟢'), (0.80, '🟢'), (0.70, '🟡')]:
            picks = scores_df[scores_df['score'] >= threshold]
            results['A_absolute']['shield'][month_key].append({
                'level': level, 'n': len(picks),
                'avg_fwd5d': picks['fwd_5d'].mean() if len(picks) > 0 else np.nan,
                'avg_fwd20d': picks['fwd_20d'].mean() if len(picks) > 0 else np.nan,
            })
        
        # 方案B: 三层过滤
        l1_pass = vix_val <= 30  # L1: VIX <= 30
        l2_pool = scores_df[scores_df['score'] > median_score] if l1_pass else pd.DataFrame()
        
        for pct, level, label in [(95, 0.95, '🟢🟢'), (90, 0.90, '🟢'), (80, 0.80, '🟡')]:
            if l1_pass and len(l2_pool) > 0:
                cutoff = np.percentile(l2_pool['score'], pct)
                picks = l2_pool[l2_pool['score'] >= cutoff]
            else:
                picks = pd.DataFrame()
            results['B_three_layer']['shield'][month_key].append({
                'level': level, 'n': len(picks),
                'avg_fwd5d': picks['fwd_5d'].mean() if len(picks) > 0 else np.nan,
                'avg_fwd20d': picks['fwd_20d'].mean() if len(picks) > 0 else np.nan,
                'l1_pass': l1_pass, 'vix': vix_val,
            })
        
        # 方案C: 纯百分位（无VIX, 无中位数）
        for pct, level in [(95, '🟢🟢'), (90, '🟢'), (80, '🟡')]:
            cutoff = np.percentile(preds, pct)
            picks = scores_df[scores_df['score'] >= cutoff]
            results['C_percentile_only']['shield'][month_key].append({
                'level': level, 'n': len(picks),
                'avg_fwd5d': picks['fwd_5d'].mean() if len(picks) > 0 else np.nan,
                'avg_fwd20d': picks['fwd_20d'].mean() if len(picks) > 0 else np.nan,
            })
    
    # ===== 绿箭 ($1-$10) =====
    arrow_day = day_data[day_data['close'].between(0.5, 10)].dropna(subset=arrow_feats)
    if len(arrow_day) > 50:
        X = arrow_day[arrow_feats].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        dtest = xgb.DMatrix(X, feature_names=arrow_feats)
        preds = arrow_model.predict(dtest)
        
        scores_df = pd.DataFrame({
            'sym': arrow_day['sym'].values,
            'score': preds,
            'fwd_5d': arrow_day['fwd_5d'].values if 'fwd_5d' in arrow_day else np.nan,
            'fwd_20d': arrow_day['fwd_20d'].values if 'fwd_20d' in arrow_day else np.nan,
            'price': arrow_day['close'].values,
        })
        
        median_score = np.median(preds)
        
        # 方案A
        for threshold, level in [(0.90, '🟢🟢'), (0.80, '🟢'), (0.70, '🟡')]:
            picks = scores_df[scores_df['score'] >= threshold]
            results['A_absolute']['arrow'][month_key].append({
                'level': level, 'n': len(picks),
                'avg_fwd5d': picks['fwd_5d'].mean() if len(picks) > 0 else np.nan,
                'avg_fwd20d': picks['fwd_20d'].mean() if len(picks) > 0 else np.nan,
            })
        
        # 方案B: 三层过滤
        l1_pass = vix_val <= 30
        l2_pool = scores_df[scores_df['score'] > median_score] if l1_pass else pd.DataFrame()
        
        for pct, level in [(95, 0.95), (90, 0.90), (80, 0.80)]:
            if l1_pass and len(l2_pool) > 0:
                cutoff = np.percentile(l2_pool['score'], pct)
                picks = l2_pool[l2_pool['score'] >= cutoff]
            else:
                picks = pd.DataFrame()
            label = {0.95: '🟢🟢', 0.90: '🟢', 0.80: '🟡'}[level]
            results['B_three_layer']['arrow'][month_key].append({
                'level': level, 'n': len(picks),
                'avg_fwd5d': picks['fwd_5d'].mean() if len(picks) > 0 else np.nan,
                'avg_fwd20d': picks['fwd_20d'].mean() if len(picks) > 0 else np.nan,
                'l1_pass': l1_pass, 'vix': vix_val,
            })
        
        # 方案C
        for pct, level in [(95, '🟢🟢'), (90, '🟢'), (80, '🟡')]:
            cutoff = np.percentile(preds, pct)
            picks = scores_df[scores_df['score'] >= cutoff]
            results['C_percentile_only']['arrow'][month_key].append({
                'level': level, 'n': len(picks),
                'avg_fwd5d': picks['fwd_5d'].mean() if len(picks) > 0 else np.nan,
                'avg_fwd20d': picks['fwd_20d'].mean() if len(picks) > 0 else np.nan,
            })
    
    if (i+1) % 10 == 0:
        print(f'  进度: {i+1}/{len(weekly_dates)} ({date_str})', flush=True)

# ========== 汇总统计 ==========
print('\n' + '='*70)
print('📊 三层过滤回测结果')
print('='*70)

for model_name, model_key in [('🛡️ 蓝盾V6', 'shield'), ('🎯 绿箭V11', 'arrow')]:
    print(f'\n{model_name}')
    print('-'*70)
    
    for scheme_name, scheme_key in [
        ('A. 原版绝对阈值 (≥0.90/0.80/0.70)', 'A_absolute'),
        ('B. 三层过滤 (VIX>30关闭 + >中位数 + 百分位)', 'B_three_layer'),
        ('C. 纯百分位 (Top5/10/20%)', 'C_percentile_only'),
    ]:
        print(f'\n  {scheme_name}')
        print(f'  {"信号":<8} {"月均数":>8} {"月均5d%":>10} {"月均20d%":>10} {"5d胜率":>8} {"20d胜率":>8}')
        print(f'  {"-"*55}')
        
        for level_name, level_idx in [('🟢🟢', 0), ('🟢', 1), ('🟡', 2)]:
            all_n = []
            all_f5 = []
            all_f20 = []
            for month in results[scheme_key][model_key]:
                entries = results[scheme_key][model_key][month]
                # 每个月有3个entry (🟢🟢, 🟢, 🟡)
                if level_idx < len(entries):
                    e = entries[level_idx]
                    all_n.append(e['n'])
                    if not np.isnan(e.get('avg_fwd5d', np.nan)):
                        all_f5.append(e['avg_fwd5d'])
                    if not np.isnan(e.get('avg_fwd20d', np.nan)):
                        all_f20.append(e['avg_fwd20d'])
            
            n_months = len(results[scheme_key][model_key])
            avg_n = np.mean(all_n) if all_n else 0
            avg_f5 = np.mean(all_f5) * 100 if all_f5 else 0
            avg_f20 = np.mean(all_f20) * 100 if all_f20 else 0
            wr5 = np.mean([1 for x in all_f5 if x > 0]) * 100 if all_f5 else 0
            wr20 = np.mean([1 for x in all_f20 if x > 0]) * 100 if all_f20 else 0
            
            print(f'  {level_name:<8} {avg_n:>7.1f} {avg_f5:>9.2f}% {avg_f20:>9.2f}% {wr5:>7.1f}% {wr20:>7.1f}%')

# VIX统计
vix_vals = df[df['date'].isin(weekly_dates)]['vix_close'].dropna()
print(f'\n📈 VIX统计（回测期间）')
print(f'  均值: {vix_vals.mean():.1f}')
print(f'  >20占比: {(vix_vals > 20).mean()*100:.1f}%')
print(f'  >25占比: {(vix_vals > 25).mean()*100:.1f}%')
print(f'  >30占比: {(vix_vals > 30).mean()*100:.1f}%')
print(f'  >35占比: {(vix_vals > 35).mean()*100:.1f}%')

# 保存
output = {
    'period': f'{str(weekly_dates[0])[:10]} ~ {str(weekly_dates[-1])[:10]}',
    'weeks': len(weekly_dates),
    'vix_stats': {
        'mean': float(vix_vals.mean()),
        'gt20_pct': float((vix_vals > 20).mean() * 100),
        'gt30_pct': float((vix_vals > 30).mean() * 100),
    },
}
json.dump(output, open(os.path.join(OUT_DIR, 'three_layer_backtest.json'), 'w'), indent=2)
print(f'\n✅ 保存: output/three_layer_backtest.json')
