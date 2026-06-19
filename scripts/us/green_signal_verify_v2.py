#!/usr/bin/env python3
"""
🟢🟢信号验证 — 修正版：蓝盾用20天，绿箭用5天
"""
import json, os, warnings, numpy as np, pandas as pd, xgboost as xgb
warnings.filterwarnings('ignore')

ROOT = '/home/hermes/.hermes/openclaw-archive'
N_SAMPLES = 8  # 增加到8组

print('📊 加载数据...')
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

print('⚙️ 计算特征...')
parts = []
for i, (sym, g) in enumerate(df.groupby('sym')):
    f = compute_features(g); f['sym'] = sym; parts.append(f)
df = pd.concat(parts, ignore_index=True)

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
df['fwd_5d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-5) / x - 1)
df['fwd_20d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-20) / x - 1)

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

# 随机抽样
all_dates = sorted(df['date'].unique())
np.random.seed(123)  # 不同seed
sample_dates = np.random.choice(all_dates[60:-30], size=N_SAMPLES, replace=False)
sample_dates = sorted(sample_dates)

print(f'\n📅 抽样日期: {[str(d)[:10] for d in sample_dates]}')
print('='*80)

shield_results = []
arrow_results = []

for date in sample_dates:
    date_str = str(date)[:10]
    day_data = df[df['date'] == date]
    if len(day_data) < 100: continue
    
    vix_val = day_data['vix_close'].iloc[0] if 'vix_close' in day_data.columns else 20
    if pd.isna(vix_val): vix_val = 20
    
    print(f'\n📆 {date_str} | VIX={vix_val:.1f}')
    print('-'*80)
    
    # ===== 蓝盾（20天持有期）=====
    shield_day = day_data[day_data['close'] > 10].dropna(subset=shield_feats)
    if len(shield_day) > 50:
        X = shield_day[shield_feats].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        dtest = xgb.DMatrix(X, feature_names=shield_feats)
        preds = shield_model.predict(dtest)
        
        p95 = np.percentile(preds, 95)
        p90 = np.percentile(preds, 90)
        p80 = np.percentile(preds, 80)
        median = np.median(preds)
        
        # Top5% 🟢🟢
        top5_mask = preds >= p95
        top5_stocks = shield_day[top5_mask]
        top5_20d = top5_stocks['fwd_20d'].dropna()
        top5_5d = top5_stocks['fwd_5d'].dropna()
        
        # 随机对照
        n_top5 = top5_mask.sum()
        if n_top5 > 0 and n_top5 <= len(shield_day):
            random_mask = np.random.choice(len(shield_day), size=n_top5, replace=False)
            random_stocks = shield_day.iloc[random_mask]
            random_20d = random_stocks['fwd_20d'].dropna()
            random_5d = random_stocks['fwd_5d'].dropna()
        else:
            random_20d = pd.Series([])
            random_5d = pd.Series([])
        
        # Top10% 🟢
        top10_mask = preds >= p90
        top10_20d = shield_day[top10_mask]['fwd_20d'].dropna()
        
        # Top20% 🟡
        top20_mask = preds >= p80
        top20_20d = shield_day[top20_mask]['fwd_20d'].dropna()
        
        # 低于中位数 🔴
        below_median = shield_day[preds <= median]
        below_20d = below_median['fwd_20d'].dropna()
        
        print(f'  🛡️ 蓝盾: {len(shield_day)}只 | 🟢🟢Top5%={n_top5}只')
        if len(top5_20d) > 0:
            print(f'  🟢🟢 20d: 均值{top5_20d.mean()*100:+.2f}% | 中位{top5_20d.median()*100:+.2f}% | 胜率{(top5_20d>0).mean()*100:.0f}%')
        if len(top10_20d) > 0:
            print(f'  🟢  Top10% 20d: 均值{top10_20d.mean()*100:+.2f}%')
        if len(top20_20d) > 0:
            print(f'  🟡  Top20% 20d: 均值{top20_20d.mean()*100:+.2f}%')
        if len(below_20d) > 0:
            print(f'  🔴  低于中位数 20d: 均值{below_20d.mean()*100:+.2f}%')
        if len(random_20d) > 0:
            print(f'  随机对照 20d: 均值{random_20d.mean()*100:+.2f}%')
        
        alpha_20d = top5_20d.mean() - random_20d.mean() if len(random_20d) > 0 else 0
        alpha_5d = top5_5d.mean() - random_5d.mean() if len(random_5d) > 0 else 0
        print(f'  Alpha: 20d={alpha_20d*100:+.2f}% | 5d={alpha_5d*100:+.2f}%')
        
        shield_results.append({
            'date': date_str, 'vix': float(vix_val),
            'n_stocks': int(len(shield_day)),
            'n_top5': int(n_top5),
            'top5_20d_mean': float(top5_20d.mean()) if len(top5_20d) > 0 else 0,
            'top5_20d_median': float(top5_20d.median()) if len(top5_20d) > 0 else 0,
            'top5_20d_wr': float((top5_20d > 0).mean()) if len(top5_20d) > 0 else 0,
            'top10_20d_mean': float(top10_20d.mean()) if len(top10_20d) > 0 else 0,
            'top20_20d_mean': float(top20_20d.mean()) if len(top20_20d) > 0 else 0,
            'below_median_20d_mean': float(below_20d.mean()) if len(below_20d) > 0 else 0,
            'random_20d_mean': float(random_20d.mean()) if len(random_20d) > 0 else 0,
            'alpha_20d': float(alpha_20d),
            'alpha_5d': float(alpha_5d),
        })
    
    # ===== 绿箭（5天持有期）=====
    arrow_day = day_data[day_data['close'].between(0.5, 10)].dropna(subset=arrow_feats)
    if len(arrow_day) > 50:
        X = arrow_day[arrow_feats].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        dtest = xgb.DMatrix(X, feature_names=arrow_feats)
        preds = arrow_model.predict(dtest)
        
        p95 = np.percentile(preds, 95)
        median = np.median(preds)
        
        top5_mask = preds >= p95
        top5_stocks = arrow_day[top5_mask]
        top5_5d = top5_stocks['fwd_5d'].dropna()
        
        n_top5 = top5_mask.sum()
        if n_top5 > 0 and n_top5 <= len(arrow_day):
            random_mask = np.random.choice(len(arrow_day), size=n_top5, replace=False)
            random_stocks = arrow_day.iloc[random_mask]
            random_5d = random_stocks['fwd_5d'].dropna()
        else:
            random_5d = pd.Series([])
        
        below_median = arrow_day[preds <= median]
        below_5d = below_median['fwd_5d'].dropna()
        
        print(f'  🎯 绿箭: {len(arrow_day)}只 | 🟢🟢Top5%={n_top5}只')
        if len(top5_5d) > 0:
            print(f'  🟢🟢 5d: 均值{top5_5d.mean()*100:+.2f}% | 中位{top5_5d.median()*100:+.2f}% | 胜率{(top5_5d>0).mean()*100:.0f}%')
        if len(below_5d) > 0:
            print(f'  🔴  低于中位数 5d: 均值{below_5d.mean()*100:+.2f}%')
        if len(random_5d) > 0:
            print(f'  随机对照 5d: 均值{random_5d.mean()*100:+.2f}%')
        
        alpha_5d = top5_5d.mean() - random_5d.mean() if len(random_5d) > 0 else 0
        print(f'  Alpha 5d: {alpha_5d*100:+.2f}%')
        
        arrow_results.append({
            'date': date_str, 'vix': float(vix_val),
            'n_stocks': int(len(arrow_day)),
            'n_top5': int(n_top5),
            'top5_5d_mean': float(top5_5d.mean()) if len(top5_5d) > 0 else 0,
            'top5_5d_median': float(top5_5d.median()) if len(top5_5d) > 0 else 0,
            'top5_5d_wr': float((top5_5d > 0).mean()) if len(top5_5d) > 0 else 0,
            'below_median_5d_mean': float(below_5d.mean()) if len(below_5d) > 0 else 0,
            'random_5d_mean': float(random_5d.mean()) if len(random_5d) > 0 else 0,
            'alpha_5d': float(alpha_5d),
        })

# ===== 汇总 =====
print('\n' + '='*80)
print('📊 抽样汇总（修正版：蓝盾20天/绿箭5天）')
print('='*80)

if shield_results:
    print(f'\n🛡️ 蓝盾 ({len(shield_results)}组):')
    top5_20d = [r['top5_20d_mean'] for r in shield_results]
    top10_20d = [r['top10_20d_mean'] for r in shield_results]
    top20_20d = [r['top20_20d_mean'] for r in shield_results]
    below_20d = [r['below_median_20d_mean'] for r in shield_results]
    random_20d = [r['random_20d_mean'] for r in shield_results]
    alpha_20d = [r['alpha_20d'] for r in shield_results]
    
    print(f'  🟢🟢 Top5% 20d: 均值{np.mean(top5_20d)*100:+.2f}% (范围{np.min(top5_20d)*100:+.2f}%~{np.max(top5_20d)*100:+.2f}%)')
    print(f'  🟢  Top10% 20d: 均值{np.mean(top10_20d)*100:+.2f}%')
    print(f'  🟡  Top20% 20d: 均值{np.mean(top20_20d)*100:+.2f}%')
    print(f'  🔴  低于中位数 20d: 均值{np.mean(below_20d)*100:+.2f}%')
    print(f'  随机对照 20d: 均值{np.mean(random_20d)*100:+.2f}%')
    print(f'  Alpha 20d: {np.mean(alpha_20d)*100:+.2f}% (范围{np.min(alpha_20d)*100:+.2f}%~{np.max(alpha_20d)*100:+.2f}%)')
    print(f'  Alpha为正: {sum(1 for a in alpha_20d if a > 0)}/{len(alpha_20d)} ({sum(1 for a in alpha_20d if a > 0)/len(alpha_20d)*100:.0f}%)')
    
    wr = [r['top5_20d_wr'] for r in shield_results]
    print(f'  🟢🟢 20d胜率: {np.mean(wr)*100:.0f}%')
    
    # 分层验证：分数越高收益越高？
    print(f'\n  分层验证（20d收益递减？）:')
    print(f'    🟢🟢(Top5%): {np.mean(top5_20d)*100:+.2f}%')
    print(f'    🟢 (Top10%): {np.mean(top10_20d)*100:+.2f}%')
    print(f'    🟡 (Top20%): {np.mean(top20_20d)*100:+.2f}%')
    print(f'    🔴 (<中位数): {np.mean(below_20d)*100:+.2f}%')
    
    # 年化
    annual_alpha = np.mean(alpha_20d) * (252/20)
    print(f'\n  年化Alpha: {annual_alpha*100:+.1f}%')

if arrow_results:
    print(f'\n🎯 绿箭 ({len(arrow_results)}组):')
    top5_5d = [r['top5_5d_mean'] for r in arrow_results]
    below_5d = [r['below_median_5d_mean'] for r in arrow_results]
    random_5d = [r['random_5d_mean'] for r in arrow_results]
    alpha_5d = [r['alpha_5d'] for r in arrow_results]
    
    print(f'  🟢🟢 Top5% 5d: 均值{np.mean(top5_5d)*100:+.2f}% (范围{np.min(top5_5d)*100:+.2f}%~{np.max(top5_5d)*100:+.2f}%)')
    print(f'  🔴  低于中位数 5d: 均值{np.mean(below_5d)*100:+.2f}%')
    print(f'  随机对照 5d: 均值{np.mean(random_5d)*100:+.2f}%')
    print(f'  Alpha 5d: {np.mean(alpha_5d)*100:+.2f}% (范围{np.min(alpha_5d)*100:+.2f}%~{np.max(alpha_5d)*100:+.2f}%)')
    print(f'  Alpha为正: {sum(1 for a in alpha_5d if a > 0)}/{len(alpha_5d)} ({sum(1 for a in alpha_5d if a > 0)/len(alpha_5d)*100:.0f}%)')
    
    wr = [r['top5_5d_wr'] for r in arrow_results]
    print(f'  🟢🟢 5d胜率: {np.mean(wr)*100:.0f}%')
    
    annual_alpha = np.mean(alpha_5d) * (252/5)
    print(f'\n  年化Alpha: {annual_alpha*100:+.1f}%')

print('\n✅ 完成')
