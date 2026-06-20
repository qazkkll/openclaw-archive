#!/usr/bin/env python3
"""
cn-alpha-v1.0: A股多因子反转选股模型
核心逻辑：A股短期是反转市场，用反转信号+资金流构建alpha

特征设计（基于诊断发现）：
  - 反转信号（负IC特征取反）：低动量、低波动、低RSI → 未来收益更高
  - 资金流信号（正IC特征）：大资金净流入 → 未来收益更高
  - 市值信号：小盘溢价（但控制极端值）

版本: cn-alpha-v1.0
"""
import pandas as pd, numpy as np, xgboost as xgb, json, time, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("=" * 60)
print("cn-alpha-v1.0 — A股反转+资金流选股模型")
print("=" * 60)
t0 = time.time()

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[1/5] 加载数据...")
df = pd.read_parquet('data/cn/features_v2.parquet')
df['date'] = pd.to_datetime(df['date'])
df['date_int'] = df['date'].dt.strftime('%Y%m%d').astype(int)
all_dates = sorted(df['date_int'].unique())

print(f"  原始: {len(df):,}行, {df['sym'].nunique()}只, {all_dates[0]}→{all_dates[-1]}")

# ============================================================
# 2. 数据清洗
# ============================================================
print("\n[2/5] 数据清洗...")
n_before = len(df)

# 过滤价格异常（涨跌停、极端值）
df = df[df['close'] > 1]  # 过滤低价股
df = df[df['close'] < 10000]  # 过滤异常高价

# 计算日收益率并过滤极端值（>11%涨跌停，考虑ST的5%）
df['daily_ret'] = df.groupby('sym')['close'].pct_change()
df = df[df['daily_ret'].abs() < 0.12]  # 过滤涨跌停和异常
df = df.drop(columns=['daily_ret'])

# 过滤成交量为0（停牌）
df = df[df['volume'] > 0]

print(f"  清洗: {n_before:,} → {len(df):,} ({(n_before-len(df))/n_before*100:.1f}%移除)")

# ============================================================
# 3. 特征工程（反转+资金流）
# ============================================================
print("\n[3/5] 构建特征（反转+资金流）...")

# === 反转特征（取反原始动量：低动量=高信号）===
# 原始r1/r5/r10/r20是正向动量，IC为负
# 取反后：-r5意味着"近期跌得多" → 预期反弹
df['rev_5d'] = -df['r5']       # 5日反转
df['rev_10d'] = -df['r10']     # 10日反转
df['rev_20d'] = -df['r20']     # 20日反转

# === 波动率特征（取反：低波动=高信号）===
# vol5/vol20的IC为负，说明低波动股表现更好
df['low_vol_5d'] = -df['vol5']    # 低波动信号
df['low_vol_20d'] = -df['vol20']  # 低波动信号
df['low_atr'] = -df['atr_pct']    # 低ATR

# === RSI/动量指标（取反：低RSI=超卖=反弹机会）===
df['rsi_reversal'] = -df['rsi14']  # RSI反转
df['macd_reversal'] = -df['macd']  # MACD反转

# === 资金流特征（保持原方向，IC为正）===
# md_net_5/20, lg_net_5/20, total_net_5/20 都是正IC
# 这些特征保持不变

# === 市值特征（负IC，小盘溢价）===
# log_circ_mv的IC=-0.112，是最强信号
# 但小盘有流动性风险，用但不做极端暴露
df['small_cap'] = -df['log_circ_mv']  # 小盘信号

# === 残差动量（核心创新：剥离市场+行业后的个股特异性趋势）===
# 计算方法：个股收益 - 全市场平均收益
market_ret = df.groupby('date')['r1'].transform('mean')
df['residual_mom_5d'] = df['r5'] - df.groupby('date')['r5'].transform('mean')
df['residual_mom_20d'] = df['r20'] - df.groupby('date')['r20'].transform('mean')

# === 换手率变化（异常换手率=信号）===
# turnover_20的IC为nan，可能是数据问题，先用已有列
# df['turnover_ratio'] 已有 vol_r

# 最终特征列表
features = [
    # 反转信号（6个）
    'rev_5d', 'rev_10d', 'rev_20d',
    'rsi_reversal', 'macd_reversal', 'macd_hist',
    # 波动率（3个）
    'low_vol_5d', 'low_vol_20d', 'low_atr',
    # 资金流（6个）— 核心alpha来源
    'md_net_5', 'md_net_20', 'lg_net_5', 'lg_net_20',
    'total_net_5', 'total_net_20',
    # 市值（1个）
    'small_cap',
    # 残差动量（2个）
    'residual_mom_5d', 'residual_mom_20d',
    # 辅助
    'vol_r',  # 量比
    'sm_net_5', 'sm_net_20',  # 小资金流（负IC，可能有反转价值）
    'elg_net_5', 'elg_net_20',  # 超大资金流
]

print(f"  特征数: {len(features)}")
print(f"  反转信号: rev_5d/10d/20d, rsi_reversal, macd_reversal, low_vol/atr")
print(f"  资金流: md/lg/total_net_5/20 (核心alpha)")
print(f"  残差动量: residual_mom_5/20d (剥离市场后的个股趋势)")

# 检查特征可用性
for f in features:
    if f not in df.columns:
        print(f"  ⚠️ 缺失特征: {f}")
    else:
        pct = df[f].notna().mean()
        if pct < 0.5:
            print(f"  ⚠️ 特征 {f} 缺失率 {(1-pct)*100:.0f}%")

# ============================================================
# 4. 市场状态计算
# ============================================================
print("\n[4/5] 计算市场状态...")

# 计算全市场月度收益
market_daily = df.groupby('date_int')['r1'].mean()
market_monthly = market_daily.rolling(20).sum()  # 20日滚动收益

# 大盘均线
market_ma60 = market_daily.rolling(60).mean()
market_ma120 = market_daily.rolling(120).mean()

# 涨跌家数比
adv_dec = df.groupby('date_int').apply(
    lambda x: (x['r1'] > 0).sum() / max((x['r1'] < 0).sum(), 1)
).rename('adv_dec_ratio')

# 合并市场状态
market_state = pd.DataFrame({
    'market_ret_20d': market_monthly,
    'market_ma60': market_ma60,
    'market_ma120': market_ma120,
    'adv_dec_ratio': adv_dec
})

# 市场状态分类
def classify_market(row):
    """三层过滤器"""
    if pd.isna(row['market_ma60']) or pd.isna(row['market_ma120']):
        return 'unknown'
    
    # L1: 均线趋势
    ma_bull = row['market_ma60'] > row['market_ma120']
    
    # L2: 近期动量
    momentum_positive = row['market_ret_20d'] > 0 if not pd.isna(row['market_ret_20d']) else True
    
    # L3: 涨跌家数
    breadth_ok = row['adv_dec_ratio'] > 0.5 if not pd.isna(row['adv_dec_ratio']) else True
    
    if not ma_bull:
        return 'bear'
    elif not momentum_positive:
        return 'cautious'
    elif not breadth_ok:
        return 'weak'
    else:
        return 'bull'

market_state['regime'] = market_state.apply(classify_market, axis=1)
print(f"  市场状态分布:")
print(f"  {market_state['regime'].value_counts().to_dict()}")

# ============================================================
# 5. 训练 + Walk-Forward验证
# ============================================================
print("\n[5/5] Walk-Forward训练+验证...")

# Walk-Forward参数
train_window = 504  # 2年训练
test_window = 21    # 1个月测试
step = 21           # 每月滚动

# 准备数据
df_model = df.dropna(subset=features + ['fwd20'])
print(f"  有效样本: {len(df_model):,}")

# Walk-Forward
results = []
all_predictions = []
start_idx = train_window

while start_idx + test_window <= len(all_dates):
    train_dates = all_dates[start_idx - train_window:start_idx]
    test_dates = all_dates[start_idx:start_idx + test_window]
    
    train = df_model[df_model['date_int'].isin(train_dates)]
    test = df_model[df_model['date_int'].isin(test_dates)]
    
    if len(train) < 1000 or len(test) < 100:
        start_idx += step
        continue
    
    X_train = train[features].fillna(0)
    y_train = train['fwd20']
    X_test = test[features].fillna(0)
    y_test = test['fwd20']
    
    # 排名目标（截面排名比绝对收益更适合A股）
    y_train_rank = y_train.groupby(train['date_int']).rank(pct=True)
    
    # 训练
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', device='cuda',
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train_rank)
    
    # 预测
    test = test.copy()
    test['pred'] = model.predict(X_test)
    
    # 计算IC
    for td in test['date_int'].unique():
        t = test[test['date_int'] == td]
        if len(t) > 50:
            ic = t['pred'].corr(t['fwd20'], method='spearman')
            results.append({
                'date': td,
                'ic': ic,
                'n_stocks': len(t)
            })
    
    all_predictions.append(test[['date_int', 'sym', 'close', 'pred', 'fwd20']])
    
    if len(results) % 50 == 0 and len(results) > 0:
        avg_ic = np.mean([r['ic'] for r in results[-50:]])
        print(f"  {len(results)}个测试日, 近50日IC: {avg_ic:.4f}")
    
    start_idx += step

# 汇总
rdf = pd.DataFrame(results)
avg_ic = rdf['ic'].mean()
icir = avg_ic / rdf['ic'].std() if rdf['ic'].std() > 0 else 0
pos_ic_pct = (rdf['ic'] > 0).mean() * 100

print(f"\n  === Walk-Forward结果 ===")
print(f"  测试日数: {len(rdf)}")
print(f"  平均IC: {avg_ic:.4f}")
print(f"  ICIR: {icir:.3f}")
print(f"  IC>0占比: {pos_ic_pct:.1f}%")

# 分年度IC
rdf['year'] = rdf['date'] // 10000
yearly = rdf.groupby('year')['ic'].agg(['mean','std','count'])
yearly['icir'] = yearly['mean'] / yearly['std']
print(f"\n  分年度IC:")
for y, r in yearly.iterrows():
    print(f"    {y}: IC={r['mean']:.4f} ICIR={r['icir']:.3f} ({int(r['count'])}天)")

# ============================================================
# 6. 模拟验证（Paper Trading with Market Filter）
# ============================================================
print("\n[6/6] Paper Trading（含市场过滤器）...")

# 用预测结果做模拟
all_pred = pd.concat(all_predictions)
test_periods = sorted(all_pred['date_int'].unique())

# 每20天调仓
rebalance_dates = test_periods[::20]
n_select = 15

pt_results = []
for rd in rebalance_dates:
    # 检查市场状态
    if rd in market_state.index:
        regime = market_state.loc[rd, 'regime']
    else:
        regime = 'unknown'
    
    # 市场过滤：熊市不开仓
    if regime == 'bear':
        pt_results.append({
            'signal_date': rd, 'port_ret': 0, 'bench_ret': 0,
            'alpha': 0, 'win_rate': 0, 'regime': regime, 'action': 'HOLD_CASH'
        })
        continue
    
    # 当日信号
    signal = all_pred[all_pred['date_int'] == rd].copy()
    signal = signal[signal['close'] > 3]  # 过滤低价
    
    if len(signal) < n_select:
        continue
    
    # Top N
    top = signal.nlargest(n_select, 'pred')
    entry_prices = dict(zip(top['sym'], top['close'].astype(float)))
    
    # 找到期日
    rd_idx = test_periods.index(rd) if rd in test_periods else -1
    if rd_idx + 20 >= len(test_periods):
        continue
    exit_date = test_periods[rd_idx + 20]
    
    # 计算收益
    exit_df = all_pred[all_pred['date_int'] == exit_date]
    rets = []
    for sym in top['sym']:
        ep = entry_prices[sym]
        exit_stock = exit_df[exit_df['sym'] == sym]
        if len(exit_stock) > 0:
            ret = (exit_stock.iloc[0]['close'] - ep) / ep
            rets.append(ret)
        else:
            rets.append(0)
    
    # 基准
    bench = exit_df.copy()
    bench_entry = all_pred[all_pred['date_int'] == rd]
    merged_bench = bench_entry[['sym','close']].merge(
        bench[['sym','close']], on='sym', suffixes=('_s','_e'))
    bench_ret = ((merged_bench['close_e'] - merged_bench['close_s'])/merged_bench['close_s']).mean() if len(merged_bench) > 0 else 0
    
    port_ret = np.mean(rets)
    
    # 调整仓位（cautious/weak减仓50%）
    if regime in ['cautious', 'weak']:
        port_ret *= 0.5
        bench_ret *= 0.5
        action = f'{regime}_HALF'
    else:
        action = 'FULL'
    
    alpha = port_ret - bench_ret
    wr = len([r for r in rets if r > 0]) / len(rets) * 100 if rets else 0
    
    pt_results.append({
        'signal_date': rd, 'port_ret': port_ret, 'bench_ret': bench_ret,
        'alpha': alpha, 'win_rate': wr, 'regime': regime, 'action': action
    })

# 汇总PT结果
pt_df = pd.DataFrame(pt_results)
active = pt_df[pt_df['action'] != 'HOLD_CASH']
hold_cash = pt_df[pt_df['action'] == 'HOLD_CASH']

print(f"\n  === Paper Trading结果 ===")
print(f"  总期数: {len(pt_df)}")
print(f"  持仓期: {len(active)} (市场过滤: {len(hold_cash)}期空仓)")

if len(active) > 0:
    ann_factor = 252 / 20
    avg_port = active['port_ret'].mean()
    avg_bench = active['bench_ret'].mean()
    avg_alpha = active['alpha'].mean()
    alpha_pos = (active['alpha'] > 0).mean() * 100
    
    print(f"  平均收益: {avg_port*100:+.2f}%")
    print(f"  基准收益: {avg_bench*100:+.2f}%")
    print(f"  Alpha: {avg_alpha*100:+.2f}% (正占比: {alpha_pos:.0f}%)")
    print(f"  年化收益: {avg_port*ann_factor*100:+.1f}%")
    print(f"  年化Alpha: {avg_alpha*ann_factor*100:+.1f}%")
    
    sharpe = avg_port / active['port_ret'].std() * np.sqrt(ann_factor) if active['port_ret'].std() > 0 else 0
    cum = np.cumprod(1 + active['port_ret'].values)
    peak = np.maximum.accumulate(cum)
    max_dd = ((cum - peak) / peak).min()
    
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  最大回撤: {max_dd*100:.1f}%")
    print(f"  胜率均值: {active['win_rate'].mean():.0f}%")
    
    # 分市场状态
    print(f"\n  分市场状态:")
    for regime in ['bull', 'cautious', 'weak']:
        sub = active[active['regime'] == regime]
        if len(sub) > 0:
            print(f"    {regime}: {len(sub)}期, 均收{sub['port_ret'].mean()*100:+.2f}%, Alpha{sub['alpha'].mean()*100:+.2f}%")

# 保存
model.save_model('models/cn/cn_alpha_v1.0.json')
print(f"\n  模型已保存: models/cn/cn_alpha_v1.0.json")

summary = {
    'version': 'cn-alpha-v1.0',
    'date': '2026-06-20',
    'logic': '反转+资金流+市场过滤器',
    'features': len(features),
    'feature_list': features,
    'wf': {
        'ic': round(avg_ic, 4),
        'icir': round(icir, 3),
        'ic_positive_pct': round(pos_ic_pct, 1),
        'n_days': len(rdf)
    },
    'paper_trade': {
        'total_periods': len(pt_df),
        'active_periods': len(active),
        'cash_periods': len(hold_cash),
        'avg_return': round(avg_port*100, 2) if len(active) > 0 else 0,
        'avg_alpha': round(avg_alpha*100, 2) if len(active) > 0 else 0,
        'ann_return': round(avg_port*ann_factor*100, 1) if len(active) > 0 else 0,
        'sharpe': round(sharpe, 2) if len(active) > 0 else 0,
        'max_dd': round(max_dd*100, 1) if len(active) > 0 else 0,
        'win_rate': round(active['win_rate'].mean(), 0) if len(active) > 0 else 0,
        'alpha_positive_pct': round(alpha_pos, 0) if len(active) > 0 else 0
    },
    'market_filter': {
        'L1': 'MA60>MA120 (趋势)',
        'L2': '20日动量>0 (动量)',
        'L3': '涨跌家数比>0.5 (宽度)',
        'bear': '关闭',
        'cautious/weak': '半仓',
        'bull': '全仓'
    }
}
with open('models/cn/cn_alpha_v1.0_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n  总结已保存: models/cn/cn_alpha_v1.0_summary.json")
print(f"\n  总耗时: {time.time()-t0:.0f}秒")
print("=" * 60)
