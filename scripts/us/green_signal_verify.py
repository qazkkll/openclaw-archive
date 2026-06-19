#!/usr/bin/env python3
"""
🟢🟢信号抽样验证
随机取5个时间窗口，验证双绿灯信号的真实表现
"""
import json, os, warnings, numpy as np, pandas as pd, xgboost as xgb
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

ROOT = '/home/hermes/.hermes/openclaw-archive'
N_SAMPLES = 5  # 抽样组数

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

FUND = ['pe_trailing','pe_forward','div_yield','beta']
try:
    fund_daily = v75[['sym','date']+FUND]
    df = pd.merge(df, fund_daily, on=['sym','date'], how='left')
    for col in FUND:
        if col in df.columns: df[col] = df[col].fillna(df[col].median())
except:
    for col in FUND: df[col] = 0

# 前向收益
for hold in [5, 20]:
    df[f'fwd_{hold}d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-hold) / x - 1)

# 加载模型
print('🤖 加载模型...')
shield_model = xgb.Booster()
shield_model.load_model(os.path.join(ROOT, 'models/us/blueshield_v6_xgb.json'))
shield_meta = json.load(open(os.path.join(ROOT, 'models/us/blueshield_v6_meta.json')))
shield_feats = shield_meta['features']

arrow_model = xgb.Booster()
arrow_model.load_model(os.path.join(ROOT, 'models/us/arrow_v11_xgb.json'))
arrow_meta = json.load(open(os.path.join(ROOT, 'models/us/arrow_v11_meta.json')))
arrow_feats = arrow_meta['features']

# 随机抽样日期
all_dates = sorted(df['date'].unique())
np.random.seed(42)
sample_dates = np.random.choice(all_dates[60:-30], size=N_SAMPLES, replace=False)
sample_dates = sorted(sample_dates)

print(f'\n📅 抽样日期: {[str(d)[:10] for d in sample_dates]}')
print('='*80)

results = {'shield': [], 'arrow': []}

for date in sample_dates:
    date_str = str(date)[:10]
    day_data = df[df['date'] == date]
    if len(day_data) < 100: continue
    
    vix_val = day_data['vix_close'].iloc[0] if 'vix_close' in day_data.columns else 20
    if pd.isna(vix_val): vix_val = 20
    
    print(f'\n📆 {date_str} | VIX={vix_val:.1f}')
    print('-'*80)
    
    # ===== 蓝盾 =====
    shield_day = day_data[day_data['close'] > 10].dropna(subset=shield_feats)
    if len(shield_day) > 50:
        X = shield_day[shield_feats].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        dtest = xgb.DMatrix(X, feature_names=shield_feats)
        preds = shield_model.predict(dtest)
        
        median = np.median(preds)
        p95 = np.percentile(preds, 95)
        
        # Top5% 🟢🟢
        top5_mask = preds >= p95
        top5_stocks = shield_day[top5_mask]
        top5_returns_5d = top5_stocks['fwd_5d'].dropna()
        top5_returns_20d = top5_stocks['fwd_20d'].dropna()
        
        # 随机对照组（同样数量）
        n_top5 = top5_mask.sum()
        random_mask = np.random.choice(len(shield_day), size=min(n_top5, len(shield_day)), replace=False)
        random_stocks = shield_day.iloc[random_mask]
        random_returns_5d = random_stocks['fwd_5d'].dropna()
        random_returns_20d = random_stocks['fwd_20d'].dropna()
        
        print(f'  🛡️ 蓝盾: {len(shield_day)}只股票 | 🟢🟢 Top5%={n_top5}只 (阈值>{p95:.3f})')
        print(f'  🟢🟢 5d收益: 均值{top5_returns_5d.mean()*100:+.2f}% | 中位数{top5_returns_5d.median()*100:+.2f}% | 胜率{(top5_returns_5d>0).mean()*100:.0f}%')
        print(f'  🟢🟢 20d收益: 均值{top5_returns_20d.mean()*100:+.2f}% | 中位数{top5_returns_20d.median()*100:+.2f}% | 胜率{(top5_returns_20d>0).mean()*100:.0f}%')
        print(f'  随机对照 5d收益: 均值{random_returns_5d.mean()*100:+.2f}% | 中位数{random_returns_5d.median()*100:+.2f}% | 胜率{(random_returns_5d>0).mean()*100:.0f}%')
        print(f'  随机对照 20d收益: 均值{random_returns_20d.mean()*100:+.2f}% | 中位数{random_returns_20d.median()*100:+.2f}% | 胜率{(random_returns_20d>0).mean()*100:.0f}%')
        
        alpha_5d = top5_returns_5d.mean() - random_returns_5d.mean()
        alpha_20d = top5_returns_20d.mean() - random_returns_20d.mean()
        print(f'  Alpha: 5d={alpha_5d*100:+.2f}% | 20d={alpha_20d*100:+.2f}%')
        
        results['shield'].append({
            'date': date_str, 'vix': vix_val,
            'n_stocks': len(shield_day), 'n_top5': n_top5,
            'top5_5d_mean': float(top5_returns_5d.mean()),
            'top5_5d_median': float(top5_returns_5d.median()),
            'top5_5d_wr': float((top5_returns_5d>0).mean()),
            'top5_20d_mean': float(top5_returns_20d.mean()),
            'random_5d_mean': float(random_returns_5d.mean()),
            'alpha_5d': float(alpha_5d),
            'alpha_20d': float(alpha_20d),
        })
    
    # ===== 绿箭 =====
    arrow_day = day_data[day_data['close'].between(0.5, 10)].dropna(subset=arrow_feats)
    if len(arrow_day) > 50:
        X = arrow_day[arrow_feats].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        dtest = xgb.DMatrix(X, feature_names=arrow_feats)
        preds = arrow_model.predict(dtest)
        
        p95 = np.percentile(preds, 95)
        top5_mask = preds >= p95
        top5_stocks = arrow_day[top5_mask]
        top5_returns_5d = top5_stocks['fwd_5d'].dropna()
        
        n_top5 = top5_mask.sum()
        random_mask = np.random.choice(len(arrow_day), size=min(n_top5, len(arrow_day)), replace=False)
        random_stocks = arrow_day.iloc[random_mask]
        random_returns_5d = random_stocks['fwd_5d'].dropna()
        
        print(f'  🎯 绿箭: {len(arrow_day)}只股票 | 🟢🟢 Top5%={n_top5}只 (阈值>{p95:.3f})')
        print(f'  🟢🟢 5d收益: 均值{top5_returns_5d.mean()*100:+.2f}% | 中位数{top5_returns_5d.median()*100:+.2f}% | 胜率{(top5_returns_5d>0).mean()*100:.0f}%')
        print(f'  随机对照 5d收益: 均值{random_returns_5d.mean()*100:+.2f}% | 中位数{random_returns_5d.median()*100:+.2f}% | 胜率{(random_returns_5d>0).mean()*100:.0f}%')
        
        alpha_5d = top5_returns_5d.mean() - random_returns_5d.mean()
        print(f'  Alpha: 5d={alpha_5d*100:+.2f}%')
        
        results['arrow'].append({
            'date': date_str, 'vix': vix_val,
            'n_stocks': len(arrow_day), 'n_top5': n_top5,
            'top5_5d_mean': float(top5_returns_5d.mean()),
            'top5_5d_median': float(top5_returns_5d.median()),
            'top5_5d_wr': float((top5_returns_5d>0).mean()),
            'random_5d_mean': float(random_returns_5d.mean()),
            'alpha_5d': float(alpha_5d),
        })

# 汇总
print('\n' + '='*80)
print('📊 抽样汇总')
print('='*80)

for model_name, key in [('🛡️ 蓝盾', 'shield'), ('🎯 绿箭', 'arrow')]:
    if not results[key]: continue
    r = results[key]
    print(f'\n{model_name} ({len(r)}组抽样):')
    
    top5_5d = [x['top5_5d_mean'] for x in r]
    random_5d = [x['random_5d_mean'] for x in r]
    alpha = [x['alpha_5d'] for x in r]
    
    print(f'  🟢🟢 5d均值收益: {np.mean(top5_5d)*100:+.2f}% (范围 {np.min(top5_5d)*100:+.2f}% ~ {np.max(top5_5d)*100:+.2f}%)')
    print(f'  随机 5d均值收益: {np.mean(random_5d)*100:+.2f}% (范围 {np.min(random_5d)*100:+.2f}% ~ {np.max(random_5d)*100:+.2f}%)')
    print(f'  Alpha (🟢🟢-随机): {np.mean(alpha)*100:+.2f}% (范围 {np.min(alpha)*100:+.2f}% ~ {np.max(alpha)*100:+.2f}%)')
    
    if key == 'shield':
        top5_20d = [x['top5_20d_mean'] for x in r]
        alpha_20d = [x['alpha_20d'] for x in r]
        print(f'  🟢🟢 20d均值收益: {np.mean(top5_20d)*100:+.2f}%')
        print(f'  Alpha 20d: {np.mean(alpha_20d)*100:+.2f}%')
    
    # 统计显著性
    alpha_arr = np.array(alpha)
    positive_alpha = (alpha_arr > 0).sum()
    print(f'  Alpha为正的比例: {positive_alpha}/{len(alpha_arr)} ({positive_alpha/len(alpha_arr)*100:.0f}%)')
    
    # 胜率
    wr = [x['top5_5d_wr'] for x in r]
    print(f'  🟢🟢平均胜率: {np.mean(wr)*100:.0f}%')

# 保存
json.dump(results, open(os.path.join(ROOT, 'output/green_signal_sampling.json'), 'w'), indent=2)
print(f'\n✅ 保存: output/green_signal_sampling.json')
