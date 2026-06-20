#!/usr/bin/env python3
"""
V1.1 + 市场过滤器 overlay 测试
不改模型，只在paper trade层加市场状态判断
"""
import pandas as pd, numpy as np, xgboost as xgb, json, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

HOLD_DAYS = 10
TOP_K = 15

print("V1.1 + 市场过滤器 overlay 测试\n")

# 加载数据
df = pd.read_parquet('data/cn/features_v2.parquet')
df['date'] = pd.to_datetime(df['date'])
df['date_int'] = df['date'].dt.strftime('%Y%m%d').astype(int)
all_dates = sorted(df['date_int'].unique())

# 计算市场状态（每日）
print("计算市场状态...")
mkt = df.groupby('date_int').agg(
    avg_close=('close', 'mean'),
    avg_r5=('r5', 'mean'),
    avg_r20=('r20', 'mean'),
    avg_vol20=('vol20', 'mean'),
).reset_index()

for w in [20, 60, 120]:
    mkt[f'ma{w}'] = mkt['avg_close'].rolling(w).mean()

mkt['trend'] = (mkt['ma60'] > mkt['ma120']).astype(int)
mkt['momentum'] = (mkt['avg_r20'] > 0).astype(int)

# 涨跌家数比
adv_dec = df.groupby('date_int').apply(
    lambda x: (x['r5'] > 0).sum() / max(len(x), 1)
).reset_index()
adv_dec.columns = ['date_int', 'adv_ratio']
mkt = mkt.merge(adv_dec, on='date_int', how='left')
mkt['breadth'] = (mkt['adv_ratio'] > 0.5).astype(int)

mkt['score'] = mkt['trend'] + mkt['momentum'] + mkt['breadth']
mkt['regime'] = mkt['score'].map({0: 'bear', 1: 'cautious', 2: 'cautious', 3: 'bull'})

# 加载v1.1模型
model = xgb.Booster()
model.load_model('models/cn/cn_alpha_v1.1.json')
feats = model.feature_names

# 确保特征存在
for f in feats:
    if f not in df.columns:
        df[f] = 0

# 计算标签
df = df.sort_values(['sym', 'date_int'])
df['fwd_ret'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-HOLD_DAYS) / x - 1)
df_valid = df.dropna(subset=['fwd_ret'])

# Paper Trade时点
quarter_starts = []
for year in range(2021, 2027):
    for month in [1, 4, 7, 10]:
        qdate = int(f"{year}{month:02d}01")
        candidates = [d for d in all_dates if d >= 20210101 and abs(d - qdate) < 2000]
        if candidates:
            quarter_starts.append(min(candidates, key=lambda x: abs(x - qdate)))
quarter_starts = sorted(set(quarter_starts))

# 三种策略对比
results = {'no_filter': [], 'with_filter': [], 'adaptive': []}

for signal_date in quarter_starts:
    day = df_valid[df_valid['date_int'] == signal_date].copy()
    if len(day) < 50:
        continue
    day = day[day['close'] > 3]
    
    # 模型预测
    X = day[feats].fillna(0)
    day['pred'] = model.predict(xgb.DMatrix(X))
    
    # 市场状态
    mkt_row = mkt[mkt['date_int'] == signal_date]
    if len(mkt_row) == 0:
        continue
    regime = mkt_row['regime'].iloc[0]
    mkt_score = mkt_row['score'].iloc[0]
    
    # Top K
    top = day.nlargest(TOP_K, 'pred')
    bench_ret = day['fwd_ret'].dropna().mean()
    
    # 策略1: 无过滤（V1.1原始）
    rets_raw = top['fwd_ret'].fillna(0).tolist()
    port_raw = np.mean(rets_raw)
    results['no_filter'].append({
        'date': signal_date, 'port': port_raw, 'bench': bench_ret,
        'alpha': port_raw - bench_ret, 'regime': regime
    })
    
    # 策略2: 市场过滤（bear=空仓, cautious=半仓）
    if regime == 'bear':
        port_filtered = 0
    elif regime == 'cautious':
        port_filtered = port_raw * 0.5
    else:
        port_filtered = port_raw
    results['with_filter'].append({
        'date': signal_date, 'port': port_filtered, 'bench': bench_ret,
        'alpha': port_filtered - bench_ret, 'regime': regime
    })
    
    # 策略3: 自适应（bear空仓, cautious半仓+选低波动top, bull满仓）
    if regime == 'bear':
        port_adaptive = 0
    elif regime == 'cautious':
        # 选vol最低的TOP_K/2只
        safe_top = day.nsmallest(TOP_K // 2, 'vol20').nlargest(TOP_K // 2, 'pred')
        if len(safe_top) > 0:
            safe_rets = safe_top['fwd_ret'].fillna(0).tolist()
            port_adaptive = np.mean(safe_rets) * 0.5
        else:
            port_adaptive = 0
    else:
        port_adaptive = port_raw
    results['adaptive'].append({
        'date': signal_date, 'port': port_adaptive, 'bench': bench_ret,
        'alpha': port_adaptive - bench_ret, 'regime': regime
    })

# 输出
print("=" * 70)
print("三种策略对比（同一模型、同一时点）")
print("=" * 70)

for name, label in [('no_filter', 'V1.1原始（无过滤）'), 
                     ('with_filter', 'V1.1+市场过滤'),
                     ('adaptive', 'V1.1+自适应')]:
    rdf = pd.DataFrame(results[name])
    active = rdf[rdf['port'] != 0] if name != 'no_filter' else rdf
    
    if len(active) == 0:
        print(f"\n{label}: 无活跃期")
        continue
    
    alpha_pos = (active['alpha'] > 0).sum()
    alpha_pct = alpha_pos / len(active) * 100
    cum = (1 + active['port']).prod() - 1
    n_yrs = len(active) * HOLD_DAYS / 365
    ann = (1 + cum) ** (1 / max(n_yrs, 0.5)) - 1
    sharpe = active['port'].mean() / active['port'].std() * np.sqrt(365 / HOLD_DAYS) if active['port'].std() > 0 else 0
    dd_series = (1 + active['port']).cumprod()
    max_dd = ((dd_series - dd_series.expanding().max()) / dd_series.expanding().max()).min()
    
    print(f"\n{label}:")
    print(f"  活跃期: {len(active)}/{len(rdf)}")
    print(f"  Alpha正: {alpha_pos}/{len(active)} = {alpha_pct:.1f}%")
    print(f"  年化: {ann*100:+.1f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  MaxDD: {max_dd*100:.1f}%")
    
    # 分年
    active_copy = active.copy()
    active_copy['year'] = active_copy['date'] // 10000
    for year, grp in active_copy.groupby('year'):
        print(f"    {year}: 收益={grp['port'].mean()*100:+.2f}%, Alpha={grp['alpha'].mean()*100:+.2f}%")

# 逐期详情
print(f"\n{'='*70}")
print("逐期详情:")
print(f"{'日期':>10} {'市场':>8} {'原始':>8} {'过滤':>8} {'自适应':>8}")
for i in range(len(results['no_filter'])):
    d = results['no_filter'][i]
    nf = results['no_filter'][i]['port']
    wf = results['with_filter'][i]['port']
    ad = results['adaptive'][i]['port']
    regime = d['regime']
    print(f"{int(d['date']):>10} {regime:>8} {nf*100:>+7.2f}% {wf*100:>+7.2f}% {ad*100:>+7.2f}%")
