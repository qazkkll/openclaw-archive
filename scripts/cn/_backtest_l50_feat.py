"""
L50 + 6Feat 前瞻验证 — 在7个历史时间点评分，追踪后续5天爆涨
"""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import xgboost as xgb

MD = '/home/hermes/.hermes/openclaw-project/data/models'
ML = '/home/hermes/.hermes/openclaw-archive/scripts/system'

print('L50-Feat 前瞻验证 — 7个时间点 top5 爆涨命中率')
print('='*55)

# ===== 加载模型 =====
model = xgb.Booster()
model.load_model(f'{MD}/us_v7_5_l50_feat.json')
with open(f'{MD}/us_v7_5_l50_feat_report.json') as f:
    report = json.load(f)
FEATS = report['features']

# 同时加载旧L50作对比
model_old = xgb.Booster()
model_old.load_model(f'{MD}/us_v7_5_l50.json')
with open(f'{MD}/us_v7_5_l50_report.json') as f:
    report_old = json.load(f)
FEATS_OLD = report_old['features']
print(f'\n新模型: {len(FEATS)} 特征 (含6个新特征)')
print(f'旧模型: {len(FEATS_OLD)} 特征')

def gen_lottery_feats(df, extra=True):
    d = df.copy()
    d['close_log'] = np.log1p(d['ma5'].clip(lower=0.01))
    d['close_x_vol'] = d['ma5'] * d['vol_ratio']
    d['plus_di_x_low_vol'] = d['plus_di'] * (1 / (1 + d['vol_ratio']))
    d['adx_x_rsi'] = d['adx'] * d['rsi14']
    d['bb_x_vol'] = d['bb_width'] * d['vol_ratio']
    d['rsi_x_kdj'] = d['rsi14'] * (d['k'] + d['d']) / 100
    d['low_price'] = (d['ma5'] < 3.0).astype(float)
    if extra:
        d['ma5_prev'] = d.groupby('sym')['ma5'].shift(1)
        d['pct_chg_1d'] = (d['ma5'] / d['ma5_prev'] - 1).fillna(0).clip(-0.3, 0.3)
        d['ma5_5d_ago'] = d.groupby('sym')['ma5'].shift(5)
        d['pct_chg_5d'] = (d['ma5'] / d['ma5_5d_ago'] - 1).fillna(0).clip(-0.5, 0.5)
        d['rsi_plus_di_cross'] = ((d['rsi14'] < 50) & (d['plus_di'] > d['minus_di']) & (d['plus_di'] > 15)).astype(float)
        d['vol_surge_signal'] = ((d['vol_ratio'] > 0.8) & (d['vol_ratio'] < 1.5)).astype(float)
        d['bb_width_ma20'] = d.groupby('sym')['bb_width'].transform(lambda x: x.rolling(20, min_periods=5).mean())
        d['bb_squeeze'] = (d['bb_width'] < d['bb_width_ma20'] * 0.8).astype(float)
        d['price_reversal'] = ((d['price_position'] < 0.3) & (d['rsi14'] > 35) & (d['rsi14'] < 55)).astype(float)
    return d

# ===== 加载数据 =====
df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str'] = df['date'].astype(str).str[:10]
all_dates = sorted(df['date_str'].unique())

# 7个验证时间点（同 __backtest_prospective.py）
test_dates = [
    '2024-04-01', '2024-06-03', '2024-09-03', '2024-12-02',
    '2025-01-02', '2025-03-03', '2026-04-01'
]

total_t5_new = 0
total_t5_old = 0
total_t10_new = 0
total_t10_old = 0
total_days = 0

for test_date in test_dates:
    print(f'\n{"-"*55}')
    
    # 找最接近的有效日期
    candidate_date = None
    for d in reversed(sorted([d for d in all_dates if d <= test_date])):
        day = df[df['date_str'] == d].copy()
        day_feat = gen_lottery_feats(day, extra=True)
        pool = day_feat[(day_feat['ma5'] >= 1.0) & (day_feat['ma5'] <= 10.0)].dropna(subset=FEATS)
        if len(pool) >= 20:
            candidate_date = d
            break
    
    if candidate_date is None:
        print(f'{test_date}: no data')
        continue
    
    total_days += 1
    print(f'日期: {candidate_date}')
    
    # ---- 新模型评分 ----
    day = df[df['date_str'] == candidate_date].copy()
    day_feat = gen_lottery_feats(day, extra=True)
    pool_new = day_feat[(day_feat['ma5'] >= 1.0) & (day_feat['ma5'] <= 10.0)].dropna(subset=FEATS)
    X_new = pool_new[FEATS].values.astype(np.float32)
    prob_new = model.predict(xgb.DMatrix(X_new, feature_names=FEATS))
    
    r_new = [{'sym': pool_new.iloc[i]['sym'], 'price': float(pool_new.iloc[i]['ma5']),
              'ret': float(pool_new.iloc[i].get('fwd_5d_ret', 0)), 'prob': prob_new[i]}
             for i in range(len(pool_new))]
    r_new.sort(key=lambda x: -x['prob'])
    top5_new = r_new[:5]
    top10_new = r_new[:10]
    t5_new_hit = sum(1 for r in top5_new if r['ret'] > 0.50)
    t10_new_hit = sum(1 for r in top10_new if r['ret'] > 0.50)
    total_t5_new += t5_new_hit
    total_t10_new += t10_new_hit
    
    # ---- 旧L50模型评分 ----
    day_feat_old = gen_lottery_feats(day, extra=False)
    pool_old = day_feat_old[(day_feat_old['ma5'] >= 1.0) & (day_feat_old['ma5'] <= 10.0)].dropna(subset=FEATS_OLD)
    X_old = pool_old[FEATS_OLD].values.astype(np.float32)
    prob_old = model_old.predict(xgb.DMatrix(X_old, feature_names=FEATS_OLD))
    
    r_old = [{'sym': pool_old.iloc[i]['sym'], 'price': float(pool_old.iloc[i]['ma5']),
              'ret': float(pool_old.iloc[i].get('fwd_5d_ret', 0)), 'prob': prob_old[i]}
             for i in range(len(pool_old))]
    r_old.sort(key=lambda x: -x['prob'])
    top5_old = r_old[:5]
    top10_old = r_old[:10]
    t5_old_hit = sum(1 for r in top5_old if r['ret'] > 0.50)
    t10_old_hit = sum(1 for r in top10_old if r['ret'] > 0.50)
    total_t5_old += t5_old_hit
    total_t10_old += t10_old_hit
    
    print(f'  L50-新 top5 涨>50%: {t5_new_hit}/5  ({t5_new_hit*20:.0f}%)')
    print(f'  L50-旧 top5 涨>50%: {t5_old_hit}/5  ({t5_old_hit*20:.0f}%)')
    print(f'  L50-新 top10 涨>50%: {t10_new_hit}/10')
    print(f'  L50-旧 top10 涨>50%: {t10_old_hit}/10')
    
    # 新模型top5明细
    print(f'  L50-新 Top5:')
    for r in top5_new:
        hit = ' <<<' if r['ret'] > 0.50 else (' <<' if r['ret'] > 0.30 else '')
        print(f'    {r["sym"]:6s}  prob={r["prob"]:.3f}  ${r["price"]:.2f}  -> {r["ret"]*100:+.0f}%{hit}')
    
    print(f'  L50-旧 Top5:')
    for r in top5_old:
        hit = ' <<<' if r['ret'] > 0.50 else ''
        print(f'    {r["sym"]:6s}  prob={r["prob"]:.3f}  ${r["price"]:.2f}  -> {r["ret"]*100:+.0f}%{hit}')

# ===== 汇总 =====
print('\n\n' + '='*55)
print('综合统计：7个时间点 Top5 前瞻命中率')
print('='*55)
print(f'验证天数: {total_days}')

n_picks = total_days * 5
rate_new = total_t5_new / n_picks * 100
rate_old = total_t5_old / n_picks * 100
change = rate_new - rate_old

print(f'\n{"":30s}   L50(旧)   L50+6Feat(新)   变化')
print(f'  {"-"*52}')
print(f'  {"Top5 涨>50%(笔)":>25s}:   {total_t5_old:>7d}       {total_t5_new:>7d}       {total_t5_new-total_t5_old:>+7d}')
print(f'  {"Top5 涨>50%(率)":>25s}:   {rate_old:>5.1f}%       {rate_new:>5.1f}%       {change:>+5.1f}%')
print(f'  {"Top10 涨>50%(笔)":>25s}:   {total_t10_old:>7d}       {total_t10_new:>7d}       {total_t10_new-total_t10_old:>+7d}')

verdict = '✅ 提升!' if change > 0 else ('❌ 下降' if change < 0 else '➡️ 持平')
print(f'\n结论: {verdict} 命中率从 {rate_old:.1f}% -> {rate_new:.1f}% (变化 {change:+.1f}pp)')

print(f'\n结束')
