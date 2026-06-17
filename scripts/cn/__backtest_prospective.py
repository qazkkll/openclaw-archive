"""
真实前瞻验证：在历史时间点用L50评分，追踪后续3-5天爆发
只用那天的特征数据，看后面真的涨超50%的有多少
"""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import xgboost as xgb

MD = '/home/hermes/.hermes/openclaw-project/data/models'
ML = '/home/hermes/.hermes/openclaw-archive/scripts/system'

print('真实前瞻验证 — 在历史时间点评分，然后看后面真的爆了没有')
print('='*55)

# 加载模型
model = xgb.Booster()
model.load_model(f'{MD}/us_v7_5_l50.json')
with open(f'{MD}/us_v7_5_l50_report.json') as f:
    report = json.load(f)
FEATS = report['features']

def gen_lottery_feats(df):
    d = df.copy()
    d['close_log'] = np.log1p(d['ma5'].clip(lower=0.01))
    d['close_x_vol'] = d['ma5'] * d['vol_ratio']
    d['plus_di_x_low_vol'] = d['plus_di'] * (1 / (1 + d['vol_ratio']))
    d['adx_x_rsi'] = d['adx'] * d['rsi14']
    d['bb_x_vol'] = d['bb_width'] * d['vol_ratio']
    d['rsi_x_kdj'] = d['rsi14'] * (d['k'] + d['d']) / 100
    d['low_price'] = (d['ma5'] < 3.0).astype(float)
    return d

# 加载数据
df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str'] = df['date'].astype(str).str[:10]
all_dates = sorted(df['date_str'].unique())

# 验证7个时间点，跨各市场周期
test_dates = [
    '2024-04-01', '2024-06-03', '2024-09-03', '2024-12-02',
    '2025-01-02', '2025-03-03', '2026-04-01'
]

# fwd_5d_ret 直接可用来检查
# 找到每个测试点，用模型评分，然后看该date的 fwd_5d_ret

for test_date in test_dates:
    print(f'\n{"─"*55}')
    
    # 找最接近的有效日期
    candidate_date = None
    for d in reversed(sorted([d for d in all_dates if d <= test_date])):
        day = df[df['date_str'] == d]
        day = gen_lottery_feats(day)
        pool = day[(day['ma5'] >= 1.0) & (day['ma5'] <= 10.0)].dropna(subset=FEATS)
        if len(pool) >= 20:
            candidate_date = d
            break
    
    if candidate_date is None:
        print(f'{test_date}: ❌ 无有效数据')
        continue
    
    day = df[df['date_str'] == candidate_date].copy()
    day = gen_lottery_feats(day)
    pool = day[(day['ma5'] >= 1.0) & (day['ma5'] <= 10.0)].dropna(subset=FEATS)
    
    X = pool[FEATS].values.astype(np.float32)
    prob = model.predict(xgb.DMatrix(X, feature_names=FEATS))
    
    results = []
    for i in range(len(pool)):
        results.append({
            'sym': pool.iloc[i]['sym'],
            'score': round(float(prob[i] * 100), 1),
            'prob': round(float(prob[i]), 4),
            'price': float(pool.iloc[i]['ma5']),
            'fwd_5d_ret': float(pool.iloc[i].get('fwd_5d_ret', 0)),
        })
    results.sort(key=lambda x: -x['score'])
    
    # top5 里有多少真的涨了>50%
    top5 = results[:5]
    top10 = results[:10]
    top5_hit = sum(1 for r in top5 if r['fwd_5d_ret'] > 0.50)
    top10_hit = sum(1 for r in top10 if r['fwd_5d_ret'] > 0.50)
    
    # &gt;30%
    top5_hit30 = sum(1 for r in top5 if r['fwd_5d_ret'] > 0.30)
    top10_hit30 = sum(1 for r in top10 if r['fwd_5d_ret'] > 0.30)
    
    print(f'{candidate_date} 评分 → 未来5天')
    print(f'  Top5 涨>50%: {top5_hit}/5')
    print(f'  Top10 涨>50%: {top10_hit}/10')
    print(f'  Top5 涨>30%: {top5_hit30}/5')
    print(f'  Top10 涨>30%: {top10_hit30}/10')
    
    # 展示top5明细
    if top5_hit > 0:
        print(f'  爆涨明细:')
        for r in top5:
            if r['fwd_5d_ret'] > 0.50:
                sym = r['sym']
                bar = '█' * int(min(r['fwd_5d_ret'], 2)*20)
                print(f'    {sym} ${r["price"]:.2f} → +{r["fwd_5d_ret"]*100:.0f}% {bar}')
    
    print(f'  Top5列表:')
    for r in top5:
        hit = '⬆️' if r['fwd_5d_ret'] > 0.50 else ('⬆' if r['fwd_5d_ret'] > 0.30 else '')
        print(f'    {r["sym"]:6s} 评分{r["score"]:5.1f}  ${r["price"]:>5.2f}  fwd5d:{r["fwd_5d_ret"]*100:+.0f}% {hit}')

# ===== 可选的综合统计 =====
print('\n\n' + '='*55)
print('综合统计：对比旧模型vs L50 在7个日期的top5前瞻')
print('='*55)

# 加载旧模型作对比
old_model = xgb.Booster()
old_model.load_model(f'{MD}/us_v7_5.json')
old_report = json.load(open(f'{MD}/us_v7_5_report.json'))
OLD_FEATS = old_report['features']

total_t5_hit_l50 = 0
total_t5_hit_old = 0
total_t10_hit_l50 = 0
total_t10_hit_old = 0
total_days = 0

for test_date in test_dates:
    candidate_date = None
    for d in reversed(sorted([d for d in all_dates if d <= test_date])):
        day = df[df['date_str'] == d].copy()
        pool = day[(day['ma5'] >= 1.0) & (day['ma5'] <= 10.0)].dropna(subset=OLD_FEATS)
        if len(pool) >= 20:
            candidate_date = d
            break
    if candidate_date is None:
        continue
    
    total_days += 1
    
    # L50
    day = df[df['date_str'] == candidate_date].copy()
    day = gen_lottery_feats(day)
    pool50 = day[(day['ma5'] >= 1.0) & (day['ma5'] <= 10.0)].dropna(subset=FEATS)
    X50 = pool50[FEATS].values.astype(np.float32)
    prob50 = model.predict(xgb.DMatrix(X50, feature_names=FEATS))
    r50 = sorted([{'sym': pool50.iloc[i]['sym'], 'ret': float(pool50.iloc[i].get('fwd_5d_ret',0))} for i in range(len(pool50))], key=lambda x: -prob50[i if 'i' in dir() else 0])
    # proper sort
    r50 = []
    for i in range(len(pool50)):
        r50.append({'sym': pool50.iloc[i]['sym'], 'ret': float(pool50.iloc[i].get('fwd_5d_ret',0)), 'prob': prob50[i]})
    r50.sort(key=lambda x: -x['prob'])
    top5_l50 = r50[:5]
    total_t5_hit_l50 += sum(1 for r in top5_l50 if r['ret'] > 0.50)
    top10_l50 = r50[:10]
    total_t10_hit_l50 += sum(1 for r in top10_l50 if r['ret'] > 0.50)
    
    # Old model
    day = df[df['date_str'] == candidate_date].copy()
    pool_old = day[(day['ma5'] >= 1.0) & (day['ma5'] <= 10.0)].dropna(subset=OLD_FEATS)
    X_old = pool_old[OLD_FEATS].values.astype(np.float32)
    mo = old_model.predict(xgb.DMatrix(X_old, feature_names=OLD_FEATS), output_margin=True)
    zo = (mo - np.mean(mo)) / max(np.std(mo), 0.001)
    sc_old = 100/(1+np.exp(-zo*1.5))
    r_old = []
    for i in range(len(pool_old)):
        r_old.append({'sym': pool_old.iloc[i]['sym'], 'ret': float(pool_old.iloc[i].get('fwd_5d_ret',0)), 'score': sc_old[i]})
    r_old.sort(key=lambda x: -x['score'])
    top5_old = r_old[:5]
    total_t5_hit_old += sum(1 for r in top5_old if r['ret'] > 0.50)
    top10_old = r_old[:10]
    total_t10_hit_old += sum(1 for r in top10_old if r['ret'] > 0.50)

if total_days > 0:
    print(f'验证天数: {total_days}')
    print(f'  L50 top5 爆涨: {total_t5_hit_l50}/{total_days*5} ({total_t5_hit_l50/(total_days*5)*100:.1f}%)')
    print(f'  Old top5 爆涨: {total_t5_hit_old}/{total_days*5} ({total_t5_hit_old/(total_days*5)*100:.1f}%)')
    print(f'  L50 top10 爆涨: {total_t10_hit_l50}/{total_days*10} ({total_t10_hit_l50/(total_days*10)*100:.1f}%)')
    print(f'  Old top10 爆涨: {total_t10_hit_old}/{total_days*10} ({total_t10_hit_old/(total_days*10)*100:.1f}%)')

print(f'\n⏱️ 结束')
