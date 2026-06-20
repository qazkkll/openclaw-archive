#!/usr/bin/env python3
"""
cn-alpha-v1.1 Paper Trade验证
方法：选多个历史时点，用模型生成Top15信号，持有10天（fwd_10d目标），计算收益
"""
import pandas as pd, numpy as np, xgboost as xgb, json, os, sys, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("=" * 70)
print("cn-alpha-v1.1 Paper Trade 验证")
print("=" * 70)

# ============================================================
# 1. 加载数据和模型
# ============================================================
print("\n[1/5] 加载数据和模型...")

# 加载特征数据
hist = pd.read_parquet('data/cn/features_v2.parquet')
hist['date'] = pd.to_datetime(hist['date'])
hist['date_int'] = hist['date'].dt.strftime('%Y%m%d').astype(int)

print(f"  特征数据: {len(hist):,}行, {hist['sym'].nunique()}只股票")
print(f"  日期范围: {hist['date'].min().strftime('%Y-%m-%d')} ~ {hist['date'].max().strftime('%Y-%m-%d')}")

# 加载模型
model = xgb.Booster()
model.load_model('models/cn/cn_alpha_v1.1.json')
model_features = model.feature_names
print(f"  模型特征: {len(model_features)}个")

# 检查特征是否在数据中
missing_feats = [f for f in model_features if f not in hist.columns]
if missing_feats:
    print(f"  ⚠️ 缺失特征: {missing_feats}")
    # 对于缺失特征，用0填充
    for f in missing_feats:
        hist[f] = 0

# ============================================================
# 2. 构建完整的模型特征（使用历史数据计算截面排名等）
# ============================================================
print("\n[2/5] 构建特征...")

# 获取所有交易日
all_dates = sorted(hist['date_int'].unique())
# 使用2020年之后的数据做验证（确保有足够训练数据）
eval_dates = [d for d in all_dates if d >= 20200101]
print(f"  验证日期: {len(eval_dates)}天 ({eval_dates[0]} ~ {eval_dates[-1]})")

# ============================================================
# 3. 选时点做Paper Trade
# ============================================================
print("\n[3/5] 选时点做Paper Trade...")

# 每季度选1个时点，共约20+个时点
quarter_starts = []
for year in range(2020, 2027):
    for month in [1, 4, 7, 10]:
        qdate = int(f"{year}{month:02d}01")
        # 找最近的交易日
        candidates = [d for d in eval_dates if abs(d - qdate) < 2000]  # 20天内
        if candidates:
            quarter_starts.append(min(candidates, key=lambda x: abs(x - qdate)))

# 去重
quarter_starts = sorted(set(quarter_starts))
print(f"  选中时点: {len(quarter_starts)}个")

# 对每个时点做回测
results = []
HOLD_DAYS = 10  # fwd_10d目标
TOP_K = 15

for i, signal_date in enumerate(quarter_starts):
    # 获取信号日数据
    signal_day = hist[hist['date_int'] == signal_date].copy()
    if len(signal_day) < 100:
        continue
    
    # 过滤：价格>3，排除ST
    signal_day = signal_day[signal_day['close'] > 3]
    
    # 确保所有特征存在
    for f in model_features:
        if f not in signal_day.columns:
            signal_day[f] = 0
    
    # 模型预测
    X = signal_day[model_features].fillna(0)
    signal_day['score'] = model.predict(xgb.DMatrix(X))
    
    # Top K
    top_k = signal_day.nlargest(TOP_K, 'score')
    top_codes = set(top_k['sym'].tolist())
    
    # 找退出日（HOLD_DAYS后）
    signal_idx = all_dates.index(signal_date)
    if signal_idx + HOLD_DAYS >= len(all_dates):
        continue
    exit_date = all_dates[signal_idx + HOLD_DAYS]
    
    # 获取退出日价格
    exit_day = hist[hist['date_int'] == exit_date].set_index('sym')
    
    # 计算每只股票收益
    stock_returns = []
    for _, row in top_k.iterrows():
        sym = row['sym']
        entry_price = row['close']
        if sym in exit_day.index:
            exit_price = exit_day.loc[sym, 'close']
            if isinstance(exit_price, pd.Series):
                exit_price = exit_price.iloc[0]
            ret = (exit_price - entry_price) / entry_price
            stock_returns.append(ret)
        else:
            stock_returns.append(0)  # 停牌
    
    # 基准收益（全市场平均）
    bench_entry = signal_day[['sym', 'close']].set_index('sym')
    bench_exit = exit_day[['close']]
    bench_merged = bench_entry.join(bench_exit, lsuffix='_entry', rsuffix='_exit')
    bench_merged = bench_merged.dropna()
    bench_ret = ((bench_merged['close_exit'] - bench_merged['close_entry']) / bench_merged['close_entry']).mean()
    
    # 记录
    port_ret = np.mean(stock_returns)
    winners = sum(1 for r in stock_returns if r > 0)
    
    results.append({
        'signal_date': signal_date,
        'exit_date': exit_date,
        'port_return': port_ret,
        'bench_return': bench_ret,
        'alpha': port_ret - bench_ret,
        'win_rate': winners / len(stock_returns) * 100,
        'winners': winners,
        'total': len(stock_returns),
        'best': max(stock_returns),
        'worst': min(stock_returns),
        'n_stocks': len(signal_day)
    })
    
    status = "OK" if port_ret > bench_ret else "UNDER"
    print(f"  [{i+1}/{len(quarter_starts)}] {signal_date} -> {exit_date}: "
          f"组合={port_ret*100:+.2f}%, 基准={bench_ret*100:+.2f}%, "
          f"Alpha={port_ret*100-bench_ret*100:+.2f}%, WR={winners}/{len(stock_returns)} [{status}]")

# ============================================================
# 4. 汇总统计
# ============================================================
print("\n[4/5] 汇总统计...")

if not results:
    print("  ❌ 无有效结果")
    sys.exit(1)

rdf = pd.DataFrame(results)

# 基本统计
total_periods = len(rdf)
alpha_positive = (rdf['alpha'] > 0).sum()
alpha_positive_pct = alpha_positive / total_periods * 100

# 累计收益
cum_port = (1 + rdf['port_return']).prod() - 1
cum_bench = (1 + rdf['bench_return']).prod() - 1

# 年化收益
n_years = total_periods * HOLD_DAYS / 365
ann_port = (1 + cum_port) ** (1 / n_years) - 1 if n_years > 0 else 0
ann_bench = (1 + cum_bench) ** (1 / n_years) - 1 if n_years > 0 else 0

# 夏普比率（用每期收益）
if rdf['port_return'].std() > 0:
    sharpe = rdf['port_return'].mean() / rdf['port_return'].std() * np.sqrt(365 / HOLD_DAYS)
else:
    sharpe = 0

# Sortino
downside = rdf['port_return'][rdf['port_return'] < 0]
if len(downside) > 0 and downside.std() > 0:
    sortino = rdf['port_return'].mean() / downside.std() * np.sqrt(365 / HOLD_DAYS)
else:
    sortino = 0

# 最大回撤
cum_returns = (1 + rdf['port_return']).cumprod()
rolling_max = cum_returns.expanding().max()
drawdowns = (cum_returns - rolling_max) / rolling_max
max_dd = drawdowns.min()

# 按年份统计
rdf['year'] = rdf['signal_date'] // 10000
yearly = rdf.groupby('year').agg({
    'port_return': ['mean', 'count'],
    'alpha': 'mean',
    'win_rate': 'mean'
}).round(4)

# ============================================================
# 5. 输出报告
# ============================================================
print("\n" + "=" * 70)
print("📊 cn-alpha-v1.1 Paper Trade 验证报告")
print("=" * 70)

print(f"\n📈 总体表现:")
print(f"  验证期数: {total_periods}")
print(f"  每期持有: {HOLD_DAYS}天, Top{TOP_K}等权")
print(f"  Alpha正占比: {alpha_positive}/{total_periods} = {alpha_positive_pct:.1f}%")
print(f"  累计收益(模型): {cum_port*100:+.2f}%")
print(f"  累计收益(基准): {cum_bench*100:+.2f}%")
print(f"  年化收益(模型): {ann_port*100:+.2f}%")
print(f"  年化收益(基准): {ann_bench*100:+.2f}%")
print(f"  年化Alpha: {(ann_port-ann_bench)*100:+.2f}%")
print(f"  Sharpe: {sharpe:.3f}")
print(f"  Sortino: {sortino:.3f}")
print(f"  最大回撤: {max_dd*100:.2f}%")

print(f"\n📅 分年统计:")
print(f"  {'年份':>6} {'期数':>4} {'平均收益':>8} {'平均Alpha':>10} {'平均胜率':>8}")
for year, row in yearly.iterrows():
    print(f"  {year:>6} {int(row[('port_return','count')]):>4} "
          f"{row[('port_return','mean')]*100:>+7.2f}% "
          f"{row[('alpha','mean')]*100:>+9.2f}% "
          f"{row[('win_rate','mean')]:>7.1f}%")

print(f"\n📊 收益分布:")
print(f"  最佳期: {rdf['port_return'].max()*100:+.2f}% ({rdf.loc[rdf['port_return'].idxmax(), 'signal_date']})")
print(f"  最差期: {rdf['port_return'].min()*100:+.2f}% ({rdf.loc[rdf['port_return'].idxmin(), 'signal_date']})")
print(f"  中位数: {rdf['port_return'].median()*100:+.2f}%")
print(f"  标准差: {rdf['port_return'].std()*100:.2f}%")

# CEO评估
print(f"\n{'='*70}")
print("🔍 CEO评估:")
print(f"{'='*70}")

if alpha_positive_pct >= 60 and sharpe > 1.0:
    print("  ✅ 模型有效：Alpha正占比>60%, Sharpe>1.0")
elif alpha_positive_pct >= 55 and sharpe > 0.5:
    print("  ⚠️ 模型可用但需改进：Alpha正占比略低或Sharpe偏弱")
else:
    print("  ❌ 模型不可信：Alpha正占比过低或Sharpe太弱")

# 与v1.0对比
print(f"\n  vs cn-alpha-v1.0 (Paper Trade):")
print(f"  v1.0: 年化13%, Sharpe0.72, DD-26.9%")
print(f"  v1.1: 年化{ann_port*100:+.1f}%, Sharpe{sharpe:.2f}, DD{max_dd*100:.1f}%")

if ann_port > 0.13 and sharpe > 0.72:
    print(f"  → v1.1优于v1.0 ✅")
elif ann_port > 0.10:
    print(f"  → v1.1与v1.0相当 ⚠️")
else:
    print(f"  → v1.1不如v1.0 ❌")

# 保存结果
output = {
    'model': 'cn-alpha-v1.1',
    'hold_days': HOLD_DAYS,
    'top_k': TOP_K,
    'total_periods': total_periods,
    'alpha_positive_pct': round(alpha_positive_pct, 1),
    'cum_return': round(cum_port * 100, 2),
    'cum_bench': round(cum_bench * 100, 2),
    'ann_return': round(ann_port * 100, 2),
    'ann_bench': round(ann_bench * 100, 2),
    'sharpe': round(sharpe, 3),
    'sortino': round(sortino, 3),
    'max_dd': round(max_dd * 100, 2),
    'periods': [{
        'signal_date': int(r['signal_date']),
        'exit_date': int(r['exit_date']),
        'port_return': round(r['port_return'] * 100, 2),
        'bench_return': round(r['bench_return'] * 100, 2),
        'alpha': round(r['alpha'] * 100, 2),
        'win_rate': round(r['win_rate'], 1)
    } for r in results]
}

with open('models/cn/cn_alpha_v1.1_paper_trade.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ 详细结果已保存: models/cn/cn_alpha_v1.1_paper_trade.json")
