#!/usr/bin/env python3
"""Phase 3: 资金流核心模型 Paper Trade验证
9特征组合 + 20天持有期"""
import pandas as pd, numpy as np, xgboost as xgb, json, time, sys
def log(msg):
    print(msg, flush=True)
    with open('research/phase3_log.txt','a') as f:
        f.write(msg+'\n')

open('research/phase3_log.txt','w').close()
log('Phase 3: 资金流核心模型 Paper Trade')

# 加载全量数据
df = pd.read_parquet('data/cn/features_v2.parquet')
df['date_int'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d').astype(int)
df = df[df['date_int'] >= 20190101].copy()
log(f'Data: {len(df):,} rows')

# 计算特征
df = df.sort_values(['sym','date_int'])
for w in [5]:
    for col in ['sm_net','md_net','lg_net','elg_net','total_net']:
        df[f'{col}_{w}d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(w, min_periods=1).sum())

df['lg_mom'] = df['lg_net_5d'] - df.groupby('sym')['lg_net_5d'].shift(20)/4
df['lg_accel'] = df['lg_net_5d'] - df.groupby('sym')['lg_net_5d'].shift(5)

for col in ['lg_net_5d']:
    df[f'{col}_rk'] = df.groupby('date_int')[col].rank(pct=True)

df['rev_20d'] = -df['r20']
df['rsi_signal'] = 100 - df['rsi14']

num_cols = df.select_dtypes(include=[np.number]).columns
for c in num_cols: df[c] = df[c].fillna(0).replace([np.inf,-np.inf],0)

# 特征和标签
features = ['lg_net_5d','elg_net_5d','lg_mom','lg_accel','lg_net_5d_rk',
            'rev_20d','rsi_signal','macd_hist','log_circ_mv']

HOLD_DAYS = 20
TOP_K = 15
df['fwd_20d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-HOLD_DAYS)/x-1)
df_valid = df.dropna(subset=['fwd_20d'])
log(f'Valid: {len(df_valid):,}')

# Walk-Forward 5折
all_dates = sorted(df_valid['date_int'].unique())
fold_size = len(all_dates) // 6  # 约6折

log('\nWalk-Forward:')
wf = []
for i in range(5):
    tr_end = all_dates[(i+1)*fold_size - 1]
    te_start = all_dates[(i+1)*fold_size]
    te_end_idx = min((i+2)*fold_size, len(all_dates)-1)
    te_end = all_dates[te_end_idx]
    
    tr = df_valid[df_valid['date_int'] <= tr_end]
    te = df_valid[(df_valid['date_int'] >= te_start) & (df_valid['date_int'] <= te_end)]
    if len(tr) < 1000 or len(te) < 500: continue
    
    params = {'max_depth':5,'eta':0.05,'subsample':0.8,'colsample_bytree':0.8,
              'min_child_weight':100,'objective':'reg:squarederror','tree_method':'hist'}
    m = xgb.train(params, xgb.DMatrix(tr[features], label=tr['fwd_20d']), num_boost_round=300, verbose_eval=False)
    te = te.copy()
    te['pred'] = m.predict(xgb.DMatrix(te[features]))
    
    ic = te.groupby('date_int').apply(lambda x: x['pred'].corr(x['fwd_20d'])).mean()
    ric = te.groupby('date_int').apply(lambda x: x['pred'].corr(x['fwd_20d'], method='spearman')).mean()
    top = te.groupby('date_int').apply(lambda x: x.nlargest(TOP_K,'pred')['fwd_20d'].mean()).mean()
    bot = te.groupby('date_int').apply(lambda x: x.nsmallest(TOP_K,'pred')['fwd_20d'].mean()).mean()
    wf.append({'ic':ic,'ric':ric,'ls':top-bot,'top':top})
    log(f'  Fold{i+1} ({te_start}-{te_end}): IC={ic:.4f} LS={top-bot:.4f}')

avg_ic = np.mean([r['ic'] for r in wf])
avg_ls = np.mean([r['ls'] for r in wf])
log(f'  Avg: IC={avg_ic:.4f} LS={avg_ls:.4f}')

# Paper Trade 2021-2026
log('\nPaper Trade (2021-2026, 20天持有):')
train_final = df_valid[df_valid['date_int'] <= 20210101]
final_m = xgb.train(params, xgb.DMatrix(train_final[features], label=train_final['fwd_20d']), num_boost_round=300, verbose_eval=False)

quarter_starts = []
for year in range(2021, 2027):
    for month in [1, 3, 5, 7, 9, 11]:  # 每2个月
        qdate = int(f"{year}{month:02d}01")
        cands = [d for d in all_dates if abs(d-qdate) < 2000]
        if cands: quarter_starts.append(min(cands, key=lambda x: abs(x-qdate)))
quarter_starts = sorted(set(quarter_starts))

paper = []
for sd in quarter_starts:
    day = df_valid[df_valid['date_int'] == sd].copy()
    if len(day) < 50: continue
    day = day[day['close'] > 3]
    
    # 简单市场过滤
    mkt_avg_r20 = day['r20'].mean()
    position = 1.0 if mkt_avg_r20 > -0.05 else 0.5
    
    day['pred'] = final_m.predict(xgb.DMatrix(day[features]))
    top = day.nlargest(TOP_K, 'pred')
    rets = top['fwd_20d'].fillna(0).tolist()
    port = np.mean(rets) * position
    bench = day['fwd_20d'].dropna().mean()
    wr = sum(1 for r in rets if r > 0) / len(rets) * 100
    
    paper.append({'date': sd, 'port': port, 'bench': bench, 'alpha': port - bench, 'wr': wr, 'pos': position})

# 汇总
rdf = pd.DataFrame(paper)
active = rdf[rdf['pos'] > 0]
alpha_pct = (active['alpha'] > 0).sum() / len(active) * 100
cum = (1 + active['port']).prod() - 1
n_yrs = len(active) * HOLD_DAYS / 365
ann = (1 + cum) ** (1 / max(n_yrs, 0.5)) - 1
sharpe = active['port'].mean() / active['port'].std() * np.sqrt(365 / HOLD_DAYS) if active['port'].std() > 0 else 0
dd = (1 + active['port']).cumprod()
max_dd = ((dd - dd.expanding().max()) / dd.expanding().max()).min()

log(f'\n{"="*60}')
log(f'📊 资金流核心模型 Paper Trade 结果')
log(f'{"="*60}')
log(f'  特征: {len(features)}个, 持有: {HOLD_DAYS}天, Top{TOP_K}')
log(f'  活跃期: {len(active)}/{len(rdf)}')
log(f'  Alpha正: {(active["alpha"]>0).sum()}/{len(active)} = {alpha_pct:.1f}%')
log(f'  年化: {ann*100:+.1f}%')
log(f'  Sharpe: {sharpe:.2f}')
log(f'  MaxDD: {max_dd*100:.1f}%')

log(f'\n  分年:')
ac = active.copy()
ac['year'] = ac['date'] // 10000
for y, g in ac.groupby('year'):
    log(f'    {y}: 收益={g["port"].mean()*100:+.2f}% Alpha={g["alpha"].mean()*100:+.2f}% WR={g["wr"].mean():.0f}%')

log(f'\n  逐期:')
for _, r in rdf.iterrows():
    log(f'    {int(r["date"])} {r["port"]*100:>+6.2f}% Alpha={r["alpha"]*100:>+6.2f}% WR={r["wr"]:.0f}%')

# 特征重要性
imp = final_m.get_score(importance_type='gain')
total = sum(imp.values())
log(f'\n  特征重要性:')
for f, g in sorted(imp.items(), key=lambda x: x[1], reverse=True):
    log(f'    {f:<25} {g/total*100:.1f}%')

# 保存
final_m.save_model('models/cn/cn_alpha_v1.4_flow.json')
log(f'\n✅ 模型已保存: models/cn/cn_alpha_v1.4_flow.json')
