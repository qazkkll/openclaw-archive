#!/usr/bin/env python3
"""
cn-alpha-v1.2 训练 + 验证
CEO三层优化：
  L1: 市场过滤器（MA60/120 + 动量 + 涨跌比）
  L2: 截面排名特征（rank/zscore，美股V4突破关键）
  L3: 降低small_cap依赖，增加信号稳定性
"""
import pandas as pd, numpy as np, xgboost as xgb, json, os, sys, warnings, time
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

HOLD_DAYS = 10
TOP_K = 15
VERSION = "cn-alpha-v1.2"

print("=" * 70)
print(f"{VERSION} 训练 + 验证")
print("=" * 70)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[1/6] 加载数据...")
hist = pd.read_parquet('data/cn/features_v2.parquet')
hist['date'] = pd.to_datetime(hist['date'])
hist['date_int'] = hist['date'].dt.strftime('%Y%m%d').astype(int)

all_dates = sorted(hist['date_int'].unique())
print(f"  数据: {len(hist):,}行, {hist['sym'].nunique()}只, {len(all_dates)}天")
print(f"  范围: {all_dates[0]} ~ {all_dates[-1]}")

# ============================================================
# 2. 构建增强特征集
# ============================================================
print("\n[2/6] 构建增强特征集...")

# V1.1的36个特征（从模型获取）
m11 = xgb.Booster()
m11.load_model('models/cn/cn_alpha_v1.1.json')
v11_features = m11.feature_names

# 确保v1.1特征存在
for f in v11_features:
    if f not in hist.columns:
        hist[f] = 0

# === 新增：截面排名特征（美股V4突破关键） ===
print("  计算截面排名特征...")

# 每日截面排名（rank百分位，0-1）
rank_features = {
    'rev_20d': 'rev_20d_rank',       # 反转强度排名
    'lg_net_20': 'lg_net_20_csrank',  # 大单净流入排名
    'total_net_20': 'total_net_20_csrank',  # 总资金流排名
    'vol_r': 'vol_r_csrank',          # 换手率排名
}

for src, dst in rank_features.items():
    if src in hist.columns:
        hist[dst] = hist.groupby('date_int')[src].rank(pct=True)
    else:
        hist[dst] = 0.5

# zscore特征（标准化截面偏离）
for src in ['rev_20d', 'lg_net_20', 'vol_r']:
    if src in hist.columns:
        zname = f'{src}_zscore'
        grp = hist.groupby('date_int')[src]
        hist[zname] = (hist[src] - grp.transform('mean')) / grp.transform('std').clip(lower=1e-8)

# === 新增：市场状态特征（L1过滤器内嵌） ===
print("  计算市场状态特征...")

# 用全市场平均收益计算市场状态
market_daily = hist.groupby('date_int').agg({
    'close': 'mean',  # 平均价格作为市场指标
}).reset_index()
market_daily.columns = ['date_int', 'mkt_avg_close']

# 计算市场MA
for w in [20, 60, 120]:
    market_daily[f'mkt_ma{w}'] = market_daily['mkt_avg_close'].rolling(w).mean()

market_daily['mkt_ma60_above_120'] = (market_daily['mkt_ma60'] > market_daily['mkt_ma120']).astype(float)
market_daily['mkt_ret_20d'] = market_daily['mkt_avg_close'].pct_change(20)
market_daily['mkt_momentum'] = (market_daily['mkt_ret_20d'] > 0).astype(float)

# 涨跌家数比
adv_dec = hist.groupby('date_int').apply(
    lambda x: (x['close'] > x['close'].shift(1) if 'close' in x.columns else pd.Series(dtype=float)).sum() / max(len(x), 1)
).reset_index()
adv_dec.columns = ['date_int', 'adv_ratio']

# 合并市场特征
market_feats = market_daily[['date_int', 'mkt_ma60_above_120', 'mkt_momentum']].merge(
    adv_dec[['date_int', 'adv_ratio']], on='date_int', how='left'
)
hist = hist.merge(market_feats, on='date_int', how='left')
hist['mkt_breadth'] = (hist['adv_ratio'] > 0.5).astype(float)

# === 新增：低波动交互特征（降低small_cap依赖） ===
print("  计算交互特征...")
hist['low_vol_x_rev'] = hist.get('low_vol_20d', 0) * hist.get('rev_20d', 0)
hist['small_cap_x_flow'] = hist.get('small_cap', 0) * hist.get('lg_net_20_csrank', 0.5)
hist['rank_x_flow'] = hist.get('rev_20d_rank', 0.5) * hist.get('lg_net_20_csrank', 0.5)

# 最终特征列表
v12_features = v11_features + [
    'rev_20d_rank', 'lg_net_20_csrank', 'total_net_20_csrank', 'vol_r_csrank',
    'rev_20d_zscore', 'lg_net_20_zscore', 'vol_r_zscore',
    'mkt_ma60_above_120', 'mkt_momentum', 'mkt_breadth',
    'low_vol_x_rev', 'small_cap_x_flow', 'rank_x_flow'
]

print(f"  V1.1特征: {len(v11_features)}")
print(f"  V1.2特征: {len(v12_features)} (+{len(v12_features)-len(v11_features)}个截面/市场)")

# ============================================================
# 3. Walk-Forward验证
# ============================================================
print("\n[3/6] Walk-Forward验证...")

# 5折WF，每折训练2-3年，测试半年
train_start = 20160101
folds = [
    (20160101, 20201231, 20210101, 20220630),  # WF1: train到2020, test 2021H1
    (20170101, 20210630, 20210701, 20221231),  # WF2
    (20180101, 20220630, 20220701, 20231231),  # WF3
    (20190101, 20230630, 20230701, 20241231),  # WF4
    (20200101, 20240630, 20240701, 20260630),  # WF5
]

wf_results = []
all_predictions = []

for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
    t0 = time.time()
    
    train_data = hist[(hist['date_int'] >= tr_start) & (hist['date_int'] <= tr_end)].copy()
    test_data = hist[(hist['date_int'] >= te_start) & (hist['date_int'] <= te_end)].copy()
    
    if len(train_data) < 10000 or len(test_data) < 1000:
        print(f"  Fold {fold_idx+1}: 数据不足，跳过")
        continue
    
    # 计算标签：未来HOLD_DAYS天收益
    # 需要在合并数据上计算，避免数据泄露
    combined = pd.concat([train_data, test_data]).sort_values(['sym', 'date_int'])
    
    # 计算forward returns
    combined['fwd_ret'] = combined.groupby('sym')['close'].transform(
        lambda x: x.shift(-HOLD_DAYS) / x - 1
    )
    
    train_data = combined[combined['date_int'].between(tr_start, tr_end)]
    test_data = combined[combined['date_int'].between(te_start, te_end)]
    
    # 去掉标签缺失
    train_data = train_data.dropna(subset=['fwd_ret'])
    test_data = test_data.dropna(subset=['fwd_ret'])
    
    # 训练
    X_train = train_data[v12_features].fillna(0)
    y_train = train_data['fwd_ret']
    X_test = test_data[v12_features].fillna(0)
    y_test = test_data['fwd_ret']
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'max_depth': 6,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 100,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'device': 'cuda',
    }
    
    model = xgb.train(params, dtrain, num_boost_round=500,
                      evals=[(dtrain, 'train')], verbose_eval=False)
    
    # 预测
    test_data = test_data.copy()
    test_data['pred'] = model.predict(dtest)
    
    # IC
    ic_by_date = test_data.groupby('date_int').apply(
        lambda x: x['pred'].corr(x['fwd_ret'])
    )
    rank_ic_by_date = test_data.groupby('date_int').apply(
        lambda x: x['pred'].corr(x['fwd_ret'], method='spearman')
    )
    
    ic = ic_by_date.mean()
    rank_ic = rank_ic_by_date.mean()
    icir = ic / ic_by_date.std() if ic_by_date.std() > 0 else 0
    
    # Top K收益
    def top_k_return(group):
        top = group.nlargest(TOP_K, 'pred')
        return top['fwd_ret'].mean()
    
    def bottom_k_return(group):
        bot = group.nsmallest(TOP_K, 'pred')
        return bot['fwd_ret'].mean()
    
    top_ret = test_data.groupby('date_int').apply(top_k_return).mean()
    bot_ret = test_data.groupby('date_int').apply(bottom_k_return).mean()
    ls = top_ret - bot_ret
    
    elapsed = time.time() - t0
    wf_results.append({
        'fold': fold_idx + 1,
        'train': f"{tr_start}-{tr_end}",
        'test': f"{te_start}-{te_end}",
        'ic': ic,
        'rank_ic': rank_ic,
        'icir': icir,
        'top_ret': top_ret,
        'bot_ret': bot_ret,
        'ls': ls,
        'train_n': len(train_data),
        'test_n': len(test_data),
        'time': elapsed
    })
    
    print(f"  Fold {fold_idx+1}: IC={ic:.4f}, RankIC={rank_ic:.4f}, "
          f"ICIR={icir:.2f}, LS={ls:.4f}, Top{TOP_K}={top_ret:.4f} ({elapsed:.0f}s)")
    
    # 保存预测用于后续分析
    all_predictions.append(test_data[['date_int', 'sym', 'pred', 'fwd_ret', 'close']].copy())

# 汇总
avg_ic = np.mean([r['ic'] for r in wf_results])
avg_rank_ic = np.mean([r['rank_ic'] for r in wf_results])
avg_icir = np.mean([r['icir'] for r in wf_results])
avg_ls = np.mean([r['ls'] for r in wf_results])

print(f"\n  WF汇总: IC={avg_ic:.4f}, RankIC={avg_rank_ic:.4f}, ICIR={avg_icir:.2f}, LS={avg_ls:.4f}")

# ============================================================
# 4. Paper Trade验证（带市场过滤器）
# ============================================================
print("\n[4/6] Paper Trade验证（含市场过滤器）...")

# 合并所有预测
pred_df = pd.concat(all_predictions)
pred_df['date_int'] = pred_df['date_int'].astype(int)

# 选时点
quarter_starts = []
for year in range(2021, 2027):
    for month in [1, 4, 7, 10]:
        qdate = int(f"{year}{month:02d}01")
        candidates = [d for d in sorted(pred_df['date_int'].unique()) if abs(d - qdate) < 2000]
        if candidates:
            quarter_starts.append(min(candidates, key=lambda x: abs(x - qdate)))
quarter_starts = sorted(set(quarter_starts))

paper_results = []
for signal_date in quarter_starts:
    day_pred = pred_df[pred_df['date_int'] == signal_date].copy()
    if len(day_pred) < 50:
        continue
    
    # 过滤
    day_pred = day_pred[day_pred['close'] > 3]
    
    # 市场过滤器
    mkt = hist[hist['date_int'] == signal_date]
    if len(mkt) > 0:
        ma60_above = mkt['mkt_ma60_above_120'].iloc[0] if 'mkt_ma60_above_120' in mkt.columns else 1
        mkt_mom = mkt['mkt_momentum'].iloc[0] if 'mkt_momentum' in mkt.columns else 1
    else:
        ma60_above = 1
        mkt_mom = 1
    
    # 仓位决策
    if ma60_above == 0 and mkt_mom == 0:
        position = 0  # bear: 空仓
    elif ma60_above == 0 or mkt_mom == 0:
        position = 0.5  # cautious: 半仓
    else:
        position = 1.0  # bull: 满仓
    
    if position == 0:
        paper_results.append({
            'signal_date': signal_date, 'port_return': 0, 'bench_return': 0,
            'alpha': 0, 'win_rate': 0, 'position': 0, 'regime': 'bear'
        })
        continue
    
    # Top K
    top_k = day_pred.nlargest(TOP_K, 'pred')
    
    # 计算收益
    rets = []
    for _, row in top_k.iterrows():
        ret = row['fwd_ret'] if not pd.isna(row['fwd_ret']) else 0
        rets.append(ret * position)
    
    # 基准
    bench_ret = day_pred['fwd_ret'].dropna().mean()
    
    port_ret = np.mean(rets)
    winners = sum(1 for r in rets if r > 0)
    
    regime = 'bull' if position == 1 else 'cautious'
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

alpha_pos = (active['alpha'] > 0).sum()
alpha_pos_pct = alpha_pos / len(active) * 100 if len(active) > 0 else 0
cum_port = (1 + active['port_return']).prod() - 1
cum_bench = (1 + active['bench_return']).prod() - 1
n_years = len(active) * HOLD_DAYS / 365
ann_port = (1 + cum_port) ** (1 / max(n_years, 0.5)) - 1
ann_bench = (1 + cum_bench) ** (1 / max(n_years, 0.5)) - 1
sharpe = active['port_return'].mean() / active['port_return'].std() * np.sqrt(365 / HOLD_DAYS) if active['port_return'].std() > 0 else 0

downside = active['port_return'][active['port_return'] < 0]
sortino = active['port_return'].mean() / downside.std() * np.sqrt(365 / HOLD_DAYS) if len(downside) > 0 and downside.std() > 0 else 0

cum = (1 + active['port_return']).cumprod()
max_dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()

# 分年
active_copy = active.copy()
active_copy['year'] = active_copy['signal_date'] // 10000
yearly = active_copy.groupby('year').agg({
    'port_return': ['mean', 'count'],
    'alpha': 'mean',
    'win_rate': 'mean'
})

# ============================================================
# 5. 与V1.0/V1.1对比
# ============================================================
print("\n[5/6] 与V1.0/V1.1对比...")

# 加载v1.1 paper trade
with open('models/cn/cn_alpha_v1.1_paper_trade.json') as f:
    v11_pt = json.load(f)

# ============================================================
# 6. 输出报告
# ============================================================
print("\n[6/6] 输出报告...\n")

print("=" * 70)
print(f"📊 {VERSION} 验证报告")
print("=" * 70)

print(f"\n📈 Walk-Forward ({len(wf_results)}折):")
print(f"  IC:     {avg_ic:.4f}")
print(f"  RankIC: {avg_rank_ic:.4f}")
print(f"  ICIR:   {avg_icir:.2f}")
print(f"  L/S:    {avg_ls:.4f}")

print(f"\n📈 Paper Trade ({len(active)}活跃期/{len(rdf)}总期):")
print(f"  Alpha正占比: {alpha_pos}/{len(active)} = {alpha_pos_pct:.1f}%")
print(f"  年化收益: {ann_port*100:+.2f}%")
print(f"  年化基准: {ann_bench*100:+.2f}%")
print(f"  年化Alpha: {(ann_port-ann_bench)*100:+.2f}%")
print(f"  Sharpe: {sharpe:.3f}")
print(f"  Sortino: {sortino:.3f}")
print(f"  最大回撤: {max_dd*100:.2f}%")

print(f"\n📅 分年:")
print(f"  {'年份':>6} {'期数':>4} {'收益':>8} {'Alpha':>8} {'胜率':>6}")
for year, row in yearly.iterrows():
    n = int(row[('port_return', 'count')])
    ret = row[('port_return', 'mean')] * 100
    alpha = row[('alpha', 'mean')] * 100
    wr = row[('win_rate', 'mean')]
    print(f"  {year:>6} {n:>4} {ret:>+7.2f}% {alpha:>+7.2f}% {wr:>5.1f}%")

print(f"\n📊 版本对比:")
v11_ann = v11_pt["ann_return"]
v11_sharpe = v11_pt["sharpe"]
v11_dd = v11_pt["max_dd"]
v11_ap = v11_pt["alpha_positive_pct"]
print(f"  {'年化收益':<15} {'+13%':>10} {'+' + str(v11_ann) + '%':>10} {ann_port*100:>+9.1f}%")
print(f"  {'Sharpe':<15} {'0.72':>10} {str(v11_sharpe):>10} {sharpe:>10.2f}")
print(f"  {'最大回撤':<15} {'-26.9%':>10} {str(v11_dd) + '%':>10} {max_dd*100:>9.1f}%")
print(f"  {'Alpha正占比':<15} {'67%':>10} {str(v11_ap) + '%':>10} {alpha_pos_pct:>9.0f}%")
print(f"  {'特征数':<15} {'23':>10} {'36':>10} {len(v12_features):>10}")

# CEO评估
print(f"\n{'='*70}")
print("🔍 CEO评估:")
if alpha_pos_pct >= 60 and sharpe >= 0.8:
    print("  ✅ V1.2通过验证标准，建议部署")
elif alpha_pos_pct >= 55 and sharpe >= 0.5:
    print("  ⚠️ V1.2有改善但未达标准，需继续优化")
else:
    print("  ❌ V1.2未通过，需重新审视优化方向")

# 特征重要性
print(f"\n📊 V1.2新增特征贡献:")
# 用最后一折的模型
if 'model' in dir():
    imp = model.get_score(importance_type='gain')
    total = sum(imp.values())
    new_feats = [f for f in v12_features if f not in v11_features]
    new_gain = sum(imp.get(f, 0) for f in new_feats)
    print(f"  新增特征占总gain: {new_gain/total*100:.1f}%")
    for f in sorted(new_feats, key=lambda x: imp.get(x, 0), reverse=True):
        g = imp.get(f, 0)
        print(f"    {f:<30} gain={g:.1f} ({g/total*100:.1f}%)")

# 保存
output = {
    'version': VERSION,
    'date': time.strftime('%Y-%m-%d'),
    'features': len(v12_features),
    'feature_list': v12_features,
    'hold_days': HOLD_DAYS,
    'top_k': TOP_K,
    'wf': {
        'folds': len(wf_results),
        'ic': round(avg_ic, 4),
        'rank_ic': round(avg_rank_ic, 4),
        'icir': round(avg_icir, 3),
        'ls': round(avg_ls, 4),
        'details': wf_results
    },
    'paper_trade': {
        'total_periods': len(rdf),
        'active_periods': len(active),
        'alpha_positive_pct': round(alpha_pos_pct, 1),
        'ann_return': round(ann_port * 100, 2),
        'ann_bench': round(ann_bench * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd': round(max_dd * 100, 2),
        'periods': paper_results
    }
}

os.makedirs('models/cn', exist_ok=True)
with open(f'models/cn/cn_alpha_v1.2_summary.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# 保存模型
model.save_model(f'models/cn/cn_alpha_v1.2.json')

print(f"\n✅ 模型已保存: models/cn/cn_alpha_v1.2.json")
print(f"✅ 报告已保存: models/cn/cn_alpha_v1.2_summary.json")
