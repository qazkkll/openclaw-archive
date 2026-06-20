#!/usr/bin/env python3
"""
cn-alpha-v1.2 完整训练+验证
从features_v2.parquet基础数据出发：
1. 计算v1.1的36个特征（反转+资金流+基本面）
2. 新增截面排名+市场状态特征
3. Walk-Forward验证
4. Paper Trade验证
"""
import pandas as pd, numpy as np, xgboost as xgb, json, os, time, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

HOLD_DAYS = 10
TOP_K = 15

print("=" * 70)
print("cn-alpha-v1.2 完整训练")
print("=" * 70)

# ============================================================
# 1. 加载基础数据
# ============================================================
print("\n[1/7] 加载数据...")
df = pd.read_parquet('data/cn/features_v2.parquet')
df['date'] = pd.to_datetime(df['date'])
df['date_int'] = df['date'].dt.strftime('%Y%m%d').astype(int)
all_dates = sorted(df['date_int'].unique())
print(f"  {len(df):,}行, {df['sym'].nunique()}只, {len(all_dates)}天")

# ============================================================
# 2. 计算v1.1的36个特征
# ============================================================
print("\n[2/7] 计算v1.1特征...")

# 反转（用负收益 = 反转）
df['rev_5d'] = -df['r5']
df['rev_10d'] = -df['r10']
df['rev_20d'] = -df['r20']

# RSI反转（RSI偏离50的负值）
df['rsi_reversal'] = -(df['rsi14'] - 50)

# MACD反转
df['macd_reversal'] = -df['macd']

# 低波动
df['low_vol_5d'] = -df['vol5']
df['low_vol_20d'] = -df['vol20']
df['low_atr'] = -df['atr_pct']

# 小盘
df['small_cap'] = -df['log_circ_mv']

# 残差动量（截面去均值）
for col, src in [('residual_mom_5d', 'r5'), ('residual_mom_20d', 'r20')]:
    grp = df.groupby('date_int')[src]
    df[col] = df[src] - grp.transform('mean')

# 资金流动量
df['lg_flow_momentum'] = df['lg_net_5'] - df['lg_net_20'] / 4
df['total_flow_momentum'] = df['total_net_5'] - df['total_net_20'] / 4

# 截面排名（v1.1用的rank）
for col in ['lg_net_20', 'md_net_20', 'total_net_20']:
    df[f'{col}_rank'] = df.groupby('date_int')[col].rank(pct=True)

# 反转×资金流交互
df['rev_flow_interaction'] = df['rev_20d'] * df['lg_net_20_rank']

# 换手率排名
df['turnover_rank'] = df.groupby('date_int')['vol_r'].rank(pct=True)

# 基本面（features_v2没有PE/PB/PS/dividend，用0填充）
for f in ['pe_rank', 'pe_inverse', 'pb_rank', 'pb_inverse', 'div_rank', 'ps_rank']:
    df[f] = 0

# V1.1完整特征列表
v11_features = [
    'rev_5d','rev_10d','rev_20d','rsi_reversal','macd_reversal','macd_hist',
    'low_vol_5d','low_vol_20d','low_atr',
    'md_net_5','md_net_20','lg_net_5','lg_net_20','total_net_5','total_net_20',
    'small_cap','residual_mom_5d','residual_mom_20d',
    'lg_flow_momentum','total_flow_momentum',
    'lg_net_20_rank','md_net_20_rank','total_net_20_rank',
    'rev_flow_interaction','turnover_rank',
    'pe_rank','pe_inverse','pb_rank','pb_inverse','div_rank','ps_rank',
    'vol_r','sm_net_5','sm_net_20','elg_net_5','elg_net_20',
]

# 验证所有特征存在
missing = [f for f in v11_features if f not in df.columns]
if missing:
    print(f"  ⚠️ 缺失特征: {missing}")
    for f in missing:
        df[f] = 0

print(f"  V1.1特征: {len(v11_features)}个")

# ============================================================
# 3. 新增v1.2特征（截面排名+市场状态+交互）
# ============================================================
print("\n[3/7] 计算v1.2新增特征...")

# 截面zscore
for src in ['rev_20d', 'lg_net_20', 'vol_r', 'total_net_20']:
    grp = df.groupby('date_int')[src]
    df[f'{src}_zscore'] = (df[src] - grp.transform('mean')) / grp.transform('std').clip(lower=1e-8)

# 市场状态特征（每日全市场统计）
mkt = df.groupby('date_int').agg(
    mkt_avg_close=('close', 'mean'),
    mkt_ret_5d=('r5', 'mean'),
    mkt_ret_20d=('r20', 'mean'),
    mkt_vol_20d=('vol20', 'mean'),
    mkt_adv=('r5', lambda x: (x > 0).sum() / max(len(x), 1)),
).reset_index()

for w in [20, 60, 120]:
    mkt[f'mkt_ma{w}'] = mkt['mkt_avg_close'].rolling(w).mean()

mkt['mkt_ma60_above_120'] = (mkt['mkt_ma60'] > mkt['mkt_ma120']).astype(float)
mkt['mkt_momentum'] = (mkt['mkt_ret_20d'] > 0).astype(float)
mkt['mkt_breadth'] = (mkt['mkt_adv'] > 0.5).astype(float)
mkt['mkt_trend_score'] = mkt['mkt_ma60_above_120'] + mkt['mkt_momentum'] + mkt['mkt_breadth']

df = df.merge(mkt[['date_int', 'mkt_ma60_above_120', 'mkt_momentum', 'mkt_breadth', 'mkt_trend_score']],
              on='date_int', how='left')

# 交互特征
df['low_vol_x_rev'] = df['low_vol_20d'] * df['rev_20d']
df['rank_x_flow'] = df['turnover_rank'] * df['lg_net_20_rank']

# V1.2新增特征
v12_new = [
    'rev_20d_zscore', 'lg_net_20_zscore', 'vol_r_zscore', 'total_net_20_zscore',
    'mkt_ma60_above_120', 'mkt_momentum', 'mkt_breadth', 'mkt_trend_score',
    'low_vol_x_rev', 'rank_x_flow'
]

v12_features = v11_features + v12_new
print(f"  V1.2新增: {len(v12_new)}个")
print(f"  V1.2总计: {len(v12_features)}个")

# ============================================================
# 4. 计算标签
# ============================================================
print("\n[4/7] 计算标签(fwd_10d)...")
df = df.sort_values(['sym', 'date_int'])
df['fwd_ret'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-HOLD_DAYS) / x - 1)
df_valid = df.dropna(subset=['fwd_ret'])
print(f"  有效样本: {len(df_valid):,}行")

# ============================================================
# 5. Walk-Forward验证
# ============================================================
print("\n[5/7] Walk-Forward验证...")

folds = [
    (20160101, 20201231, 20210101, 20220630),
    (20170101, 20210630, 20210701, 20221231),
    (20180101, 20220630, 20220701, 20231231),
    (20190101, 20230630, 20230701, 20241231),
    (20200101, 20240630, 20240701, 20260630),
]

params = {
    'max_depth': 6, 'eta': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 100, 'objective': 'reg:squarederror',
    'eval_metric': 'rmse', 'tree_method': 'hist',
}

# 对v1.1和v1.2都做WF
for label, feats in [("V1.1", v11_features), ("V1.2", v12_features)]:
    print(f"\n  === {label} ({len(feats)}特征) ===")
    wf_results = []
    
    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        t0 = time.time()
        tr = df_valid[(df_valid['date_int'] >= tr_s) & (df_valid['date_int'] <= tr_e)]
        te = df_valid[(df_valid['date_int'] >= te_s) & (df_valid['date_int'] <= te_e)]
        
        if len(tr) < 10000 or len(te) < 1000:
            continue
        
        X_tr = tr[feats].fillna(0)
        X_te = te[feats].fillna(0)
        dtrain = xgb.DMatrix(X_tr, label=tr['fwd_ret'])
        dtest = xgb.DMatrix(X_te, label=te['fwd_ret'])
        
        model = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
        te = te.copy()
        te['pred'] = model.predict(dtest)
        
        ic = te.groupby('date_int').apply(lambda x: x['pred'].corr(x['fwd_ret'])).mean()
        ric = te.groupby('date_int').apply(lambda x: x['pred'].corr(x['fwd_ret'], method='spearman')).mean()
        
        top_ret = te.groupby('date_int').apply(lambda x: x.nlargest(TOP_K, 'pred')['fwd_ret'].mean()).mean()
        bot_ret = te.groupby('date_int').apply(lambda x: x.nsmallest(TOP_K, 'pred')['fwd_ret'].mean()).mean()
        ls = top_ret - bot_ret
        
        elapsed = time.time() - t0
        wf_results.append({'fold': fi+1, 'ic': ic, 'rank_ic': ric, 'ls': ls, 'top_ret': top_ret})
        print(f"    Fold {fi+1}: IC={ic:.4f}, RankIC={ric:.4f}, LS={ls:.4f} ({elapsed:.0f}s)")
    
    avg_ic = np.mean([r['ic'] for r in wf_results])
    avg_ric = np.mean([r['rank_ic'] for r in wf_results])
    avg_ls = np.mean([r['ls'] for r in wf_results])
    print(f"    汇总: IC={avg_ic:.4f}, RankIC={avg_ric:.4f}, LS={avg_ls:.4f}")

# ============================================================
# 6. 最终模型训练（用全量数据到2024）+ Paper Trade
# ============================================================
print("\n[6/7] 最终模型训练 + Paper Trade...")

# 用V1.2特征训练最终模型
train_final = df_valid[df_valid['date_int'] <= 20240630]
X_final = train_final[v12_features].fillna(0)
dtrain_final = xgb.DMatrix(X_final, label=train_final['fwd_ret'])
final_model = xgb.train(params, dtrain_final, num_boost_round=500, verbose_eval=False)

# Paper Trade: 2021-2026每季度
quarter_starts = []
for year in range(2021, 2027):
    for month in [1, 4, 7, 10]:
        qdate = int(f"{year}{month:02d}01")
        candidates = [d for d in all_dates if abs(d - qdate) < 2000]
        if candidates:
            quarter_starts.append(min(candidates, key=lambda x: abs(x - qdate)))
quarter_starts = sorted(set(quarter_starts))

paper_results = []
for signal_date in quarter_starts:
    day = df_valid[df_valid['date_int'] == signal_date].copy()
    if len(day) < 50:
        continue
    day = day[day['close'] > 3]
    
    # 市场过滤
    mkt_score = day['mkt_trend_score'].iloc[0] if 'mkt_trend_score' in day.columns else 3
    if mkt_score <= 1:
        position = 0  # bear
        regime = 'bear'
    elif mkt_score <= 2:
        position = 0.5  # cautious
        regime = 'cautious'
    else:
        position = 1.0  # bull
        regime = 'bull'
    
    if position == 0:
        paper_results.append({
            'signal_date': signal_date, 'port_return': 0, 'bench_return': 0,
            'alpha': 0, 'win_rate': 0, 'position': 0, 'regime': regime
        })
        continue
    
    # 预测
    X_day = day[v12_features].fillna(0)
    day['pred'] = final_model.predict(xgb.DMatrix(X_day))
    
    top = day.nlargest(TOP_K, 'pred')
    rets = top['fwd_ret'].fillna(0).tolist()
    port_ret = np.mean(rets) * position
    bench_ret = day['fwd_ret'].dropna().mean()
    winners = sum(1 for r in rets if r > 0)
    
    paper_results.append({
        'signal_date': signal_date,
        'port_return': port_ret,
        'bench_return': bench_ret,
        'alpha': port_ret - bench_ret,
        'win_rate': winners / len(rets) * 100,
        'position': position,
        'regime': regime
    })

# Paper Trade汇总
rdf = pd.DataFrame(paper_results)
active = rdf[rdf['position'] > 0]

alpha_pos_pct = (active['alpha'] > 0).sum() / len(active) * 100 if len(active) > 0 else 0
cum_port = (1 + active['port_return']).prod() - 1
n_years = len(active) * HOLD_DAYS / 365
ann_port = (1 + cum_port) ** (1 / max(n_years, 0.5)) - 1
sharpe = active['port_return'].mean() / active['port_return'].std() * np.sqrt(365 / HOLD_DAYS) if active['port_return'].std() > 0 else 0
downside = active['port_return'][active['port_return'] < 0]
sortino = active['port_return'].mean() / downside.std() * np.sqrt(365 / HOLD_DAYS) if len(downside) > 0 and downside.std() > 0 else 0
cum = (1 + active['port_return']).cumprod()
max_dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()

# ============================================================
# 7. 输出
# ============================================================
print("\n[7/7] 结果\n")
print("=" * 70)
print("📊 cn-alpha-v1.2 Paper Trade结果")
print("=" * 70)
print(f"  验证期数: {len(active)}活跃 / {len(rdf)}总")
print(f"  Alpha正占比: {(active['alpha']>0).sum()}/{len(active)} = {alpha_pos_pct:.1f}%")
print(f"  年化收益: {ann_port*100:+.2f}%")
print(f"  Sharpe: {sharpe:.3f}")
print(f"  Sortino: {sortino:.3f}")
print(f"  最大回撤: {max_dd*100:.2f}%")

print(f"\n  {'日期':>10} {'市场':>8} {'收益':>8} {'Alpha':>8} {'胜率':>6}")
for _, r in rdf.iterrows():
    if r['position'] == 0:
        print(f"  {int(r['signal_date']):>10} {r['regime']:>8} {'CASH':>8} {'-':>8} {'-':>6}")
    else:
        print(f"  {int(r['signal_date']):>10} {r['regime']:>8} {r['port_return']*100:>+7.2f}% {r['alpha']*100:>+7.2f}% {r['win_rate']:>5.0f}%")

print(f"\n  分年:")
active_copy = active.copy()
active_copy['year'] = active_copy['signal_date'] // 10000
for year, grp in active_copy.groupby('year'):
    print(f"    {year}: 收益={grp['port_return'].mean()*100:+.2f}%, Alpha={grp['alpha'].mean()*100:+.2f}%, WR={grp['win_rate'].mean():.0f}%")

# 特征重要性
imp = final_model.get_score(importance_type='gain')
total = sum(imp.values())
new_feats_gain = sum(imp.get(f, 0) for f in v12_new)
print(f"\n  新增特征贡献: {new_feats_gain/total*100:.1f}%")
for f in sorted(v12_new, key=lambda x: imp.get(x, 0), reverse=True):
    g = imp.get(f, 0)
    if g > 0:
        print(f"    {f:<30} {g/total*100:.1f}%")

# 保存
final_model.save_model('models/cn/cn_alpha_v1.2.json')
summary = {
    'version': 'cn-alpha-v1.2',
    'date': time.strftime('%Y-%m-%d'),
    'features': len(v12_features),
    'feature_list': v12_features,
    'hold_days': HOLD_DAYS,
    'paper_trade': {
        'total_periods': len(rdf),
        'active_periods': len(active),
        'alpha_positive_pct': round(alpha_pos_pct, 1),
        'ann_return': round(ann_port*100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd': round(max_dd*100, 2),
    }
}
with open('models/cn/cn_alpha_v1.2_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n✅ 模型已保存: models/cn/cn_alpha_v1.2.json")
