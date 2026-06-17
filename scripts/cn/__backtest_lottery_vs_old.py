"""
彩票模型 vs 旧模型 — 5月逐日彩票捕捉率对比
"""
import sys, json, os, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import xgboost as xgb
from collections import defaultdict

t0 = time.time()
print('彩票模型 vs 旧模型 对比')
print('='*50)

MD = '/home/hermes/.hermes/openclaw-project/data/models'
ML = '/home/hermes/.hermes/openclaw-archive/scripts/system'

# ===== 加载两个模型 =====
old_model = xgb.Booster()
old_model.load_model(f'{MD}/us_v7_5.json')
old_report = json.load(open(f'{MD}/us_v7_5_report.json'))
OLD_FEATS = old_report['features']

lot_model = xgb.Booster()
lot_model.load_model(f'{MD}/us_v7_5_lottery.json')
lot_report = json.load(open(f'{MD}/us_v7_5_lottery_report.json'))
LOT_FEATS = lot_report['features']

print(f'旧模型: {len(OLD_FEATS)} 特征')
print(f'彩票模型: {len(LOT_FEATS)} 特征')

# ===== 加载数据 =====
df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str'] = df['date'].astype(str).str[:10]

with open('_deep_analysis_cache.json') as f:
    cache = json.load(f)
surge_all = cache['surge_bought'] + cache['surge_missed']
surge_by_date = defaultdict(list)
for c in surge_all:
    surge_by_date[c['date']].append(c)
may_dates = cache['may_dates']

# ===== 彩票模型特征生成函数 =====
def gen_lottery_feats(day_df):
    d = day_df.copy()
    d['close_log'] = np.log1p(d['ma5'].clip(lower=0.01))
    d['close_x_vol'] = d['ma5'] * d['vol_ratio']
    d['plus_di_x_low_vol'] = d['plus_di'] * (1 / (1 + d['vol_ratio']))
    d['adx_x_rsi'] = d['adx'] * d['rsi14']
    d['bb_x_vol'] = d['bb_width'] * d['vol_ratio']
    d['rsi_x_kdj'] = d['rsi14'] * (d['k'] + d['d']) / 100
    d['low_price'] = (d['ma5'] < 3.0).astype(float)
    return d

# ===== 逐日评分 =====
print('\n逐日评分...')
day_results = []

for di, d in enumerate(may_dates):
    day = d[:10]
    today_data = df[df['date_str'] == day].copy()
    if len(today_data) == 0:
        continue
    
    # ---- 旧模型 ----
    clean = today_data.dropna(subset=OLD_FEATS).reset_index(drop=True)
    if len(clean) == 0: continue
    X_old = clean[OLD_FEATS].values.astype(np.float32)
    mo = old_model.predict(xgb.DMatrix(X_old, feature_names=OLD_FEATS), output_margin=True)
    zo = (mo - np.mean(mo)) / max(np.std(mo), 0.001)
    so = 100 / (1 + np.exp(-zo * 1.5))
    old_t5 = set(clean.iloc[i]['sym'] for i in np.argsort(-so)[:5])
    old_t10 = set(clean.iloc[i]['sym'] for i in np.argsort(-so)[:10])
    
    # ---- 彩票模型 ----
    # 只跑$1-10的票 (彩票模型训练过滤)
    lot_data = gen_lottery_feats(today_data)
    lot_clean = lot_data[(lot_data['ma5'] >= 1.0) & (lot_data['ma5'] <= 10.0)].dropna(subset=LOT_FEATS).reset_index(drop=True)
    if len(lot_clean) == 0: continue
    
    # 彩票模型score
    X_lot = lot_clean[LOT_FEATS].values.astype(np.float32)
    ml = lot_model.predict(xgb.DMatrix(X_lot, feature_names=LOT_FEATS), output_margin=True)
    # 用彩票模型自己的概率输出 (已经是logistic概率)
    # 彩票模型的margin经过logistic后是概率, 直接排序
    prob = lot_model.predict(xgb.DMatrix(X_lot, feature_names=LOT_FEATS))
    lot_t5 = set(lot_clean.iloc[i]['sym'] for i in np.argsort(-prob)[:5])
    lot_t10 = set(lot_clean.iloc[i]['sym'] for i in np.argsort(-prob)[:10])
    
    # ---- 当天爆涨票 ----
    day_surges = surge_by_date.get(d, [])
    # 只统计 $1-10的爆涨
    surge_syms = set(c['sym'] for c in day_surges if c['buy_price'] < 10)
    
    # ---- 捕捉统计 ----
    # 旧模型: 只看这些爆涨票是否在它的top里
    old_captured_5 = surge_syms & old_t5
    old_captured_10 = surge_syms & old_t10
    lot_captured_5 = surge_syms & lot_t5
    lot_captured_10 = surge_syms & lot_t10
    
    # 遗漏排名：没抓住的爆涨票在候选里的位置
    day_clean = old_report['features'][:0]
    old_ranked = sorted(
        [{'sym': clean.iloc[i]['sym'], 'score': float(so[i])} for i in range(len(clean))],
        key=lambda x: -x['score']
    )
    lot_ranked = sorted(
        [{'sym': lot_clean.iloc[i]['sym'], 'score': float(prob[i])} for i in range(len(lot_clean))],
        key=lambda x: -x['score']
    )
    
    day_results.append({
        'date': day,
        'n_surge_lt10': len(surge_syms),
        'old_top5': len(old_captured_5),
        'old_top10': len(old_captured_10),
        'lot_top5': len(lot_captured_5),
        'lot_top10': len(lot_captured_10),
        'surge_syms': list(surge_syms),
        'old_captured': list(old_captured_5),
        'lot_captured': list(lot_captured_5),
    })
    
    if (di+1) % 5 == 0 or di == len(may_dates)-1:
        print(f'  {day}: 爆涨{len(surge_syms)} | 旧top5={len(old_captured_5)} 彩票top5={len(lot_captured_5)}')

print(f'\n评分完成: {len(day_results)}天')

# ===== 汇总 =====
print('\n' + '='*50)
print('汇总对比')
print('='*50)

total_surge = sum(r['n_surge_lt10'] for r in day_results)
total_old5 = sum(r['old_top5'] for r in day_results)
total_old10 = sum(r['old_top10'] for r in day_results)
total_lot5 = sum(r['lot_top5'] for r in day_results)
total_lot10 = sum(r['lot_top10'] for r in day_results)

print(f'\n5月 $1-10爆涨事件: {total_surge}笔')
print(f'{"捕获":>20s}   {"旧模型":>12s}   {"彩票模型":>10s}   {"变化":>8s}')
print(f'  {"-"*52}')
print(f'  {"top5捕获(笔)":>20s}   {total_old5:>12d}   {total_lot5:>10d}   {total_lot5-total_old5:>+8d}')
print(f'  {"top5捕获率":>20s}   {total_old5/total_surge*100:>11.1f}%   {total_lot5/total_surge*100:>9.1f}%   {(total_lot5-total_old5)/total_surge*100:>+7.1f}%')
print(f'  {"top10捕获(笔)":>20s}   {total_old10:>12d}   {total_lot10:>10d}   {total_lot10-total_old10:>+8d}')
print(f'  {"top10捕获率":>20s}   {total_old10/total_surge*100:>11.1f}%   {total_lot10/total_surge*100:>9.1f}%   {(total_lot10-total_old10)/total_surge*100:>+7.1f}%')

print(f'\n具体变化:')
wins, losses = 0, 0
for r in day_results:
    old_only = set(r['old_captured']) - set(r['lot_captured'])
    lot_only = set(r['lot_captured']) - set(r['old_captured'])
    for sym in lot_only:
        wins += 1
        print(f'  +{r["date"]} {sym}: 彩票新捕获')
    for sym in old_only:
        losses += 1
        print(f'  -{r["date"]} {sym}: 旧模型捕获但彩票遗漏')
print(f'\n净改善: +{wins}个新捕获 - {losses}个丢失 = {wins-losses}')

# ===== 6月对比 =====
print('\n' + '='*50)
print('6月 Top5 对比')
print('='*50)

jun_dates = sorted(df[df['date_str'].str[:7] == '2026-06']['date_str'].unique())
for jd in jun_dates:
    jun_day = df[df['date_str'] == jd].copy()
    if len(jun_day) < 50: continue
    
    # 旧模型
    clean = jun_day.dropna(subset=OLD_FEATS).reset_index(drop=True)
    if len(clean) < 50: continue
    X_old = clean[OLD_FEATS].values.astype(np.float32)
    so = 100/(1+np.exp(-(old_model.predict(xgb.DMatrix(X_old, feature_names=OLD_FEATS), output_margin=True) - np.mean(so:=old_model.predict(xgb.DMatrix(X_old, feature_names=OLD_FEATS), output_margin=True)))/max(np.std(so),0.001)*1.5))
    old_t5 = set(clean.iloc[i]['sym'] for i in np.argsort(-so)[:5])
    old_prices = [float(clean[clean['sym']==sym].iloc[0]['ma5']) for sym in sorted(old_t5)]

    # 彩票模型
    lot_day = gen_lottery_feats(jun_day)
    lot_clean = lot_day[(lot_day['ma5']>=1)&(lot_day['ma5']<=10)].dropna(subset=LOT_FEATS).reset_index(drop=True)
    if len(lot_clean) < 20: continue
    X_lot = lot_clean[LOT_FEATS].values.astype(np.float32)
    prob = lot_model.predict(xgb.DMatrix(X_lot, feature_names=LOT_FEATS))
    lot_t5 = set(lot_clean.iloc[i]['sym'] for i in np.argsort(-prob)[:5])
    lot_prices = [float(lot_clean[lot_clean['sym']==sym].iloc[0]['ma5']) for sym in sorted(lot_t5)]
    
    print(f'\n{jd}:')
    print(f'  旧模型:  均价${np.mean(old_prices):.1f} | {", ".join(sorted(old_t5))}')
    print(f'  彩票模型: 均价${np.mean(lot_prices):.1f} | {", ".join(sorted(lot_t5))}')
    print(f'  重叠: {len(old_t5 & lot_t5)}只 | 彩票独有<$5: {sum(1 for p in lot_prices if p<5)}只')

print(f'\n⏱️ 耗时: {time.time()-t0:.1f}s')
