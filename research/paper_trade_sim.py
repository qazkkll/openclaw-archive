#!/usr/bin/env python3
"""A股模型V2 回溯模拟验证：用多个历史时点的信号做Paper Trading"""
import json, pandas as pd, numpy as np, xgboost as xgb
import os, warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("=== A股模型V2 回溯模拟验证 ===")
print("方法: Walk-Forward × 多时点Paper Trading\n")

# 加载数据和模型
df = pd.read_parquet('data/cn/features_v2.parquet')
model = xgb.Booster()
model.load_model('models/cn/a_stock_xgb_v2.json')

# 处理日期
df['date'] = pd.to_datetime(df['date'])
df['date_int'] = df['date'].dt.strftime('%Y%m%d').astype(int)
all_dates = sorted(df['date_int'].unique())

print(f"数据: {len(df):,}行, {df['sym'].nunique()}只股票")
print(f"日期: {all_dates[0]} → {all_dates[-1]}")
print(f"交易日: {len(all_dates)}天\n")

features = ['r1','r5','r10','r20','d5','d10','d20','vol5','vol20','atr_pct',
    'vol_r','rsi14','macd','macd_sig','macd_hist','log_circ_mv',
    'turnover_20','sm_net_5','sm_net_20','md_net_5','md_net_20',
    'lg_net_5','lg_net_20','elg_net_5','elg_net_20','total_net_5','total_net_20']

# 验证策略: 
# 1. 选10个历史时点（每3个月一个）
# 2. 每个时点用模型选出Top15
# 3. 持有20个交易日后计算收益
# 4. 对比基准

hold_days = 20
n_select = 15
# 从2018年开始（跳过2016-2017训练期），每季度一个测试点
test_dates = [d for d in all_dates if d >= 20180101]
# 每60个交易日（约3个月）选一个
test_indices = list(range(0, len(test_dates), 60))
signal_dates = [test_dates[i] for i in test_indices]

print(f"验证时点: {len(signal_dates)}个 (从{signal_dates[0]}到{signal_dates[-1]})")
print(f"持有期: {hold_days}个交易日\n")

all_results = []

for si, sig_date in enumerate(signal_dates):
    # 建仓日数据
    sig_df = df[(df['date_int'] == sig_date) & (df['close'] > 3)].copy()
    if len(sig_df) < 100:
        continue
    
    # 预测
    X = sig_df[features].fillna(0)
    sig_df['score'] = model.predict(xgb.DMatrix(X))
    
    # Top N
    top = sig_df.nlargest(n_select, 'score')[['sym','close','score']].copy()
    top_codes = set(top['sym'].tolist())
    entry_prices = dict(zip(top['sym'], top['close'].astype(float)))
    
    # 找到期日
    sig_idx = all_dates.index(sig_date)
    if sig_idx + hold_days >= len(all_dates):
        continue
    exit_date = all_dates[sig_idx + hold_days]
    
    # 计算组合收益
    exit_df = df[(df['date_int'] == exit_date) & (df['sym'].isin(top_codes))]
    
    rets = []
    for code in top_codes:
        ep = entry_prices[code]
        stock = exit_df[exit_df['sym'] == code]
        if len(stock) > 0:
            ret = (stock.iloc[0]['close'] - ep) / ep
            rets.append(ret)
        else:
            rets.append(0)  # 停牌视为0收益
    
    # 基准（全市场等权）
    bench_start = df[(df['date_int'] == sig_date) & (df['close'] > 3)]
    bench_end = df[(df['date_int'] == exit_date)]
    bench_merged = bench_start[['sym','close']].merge(
        bench_end[['sym','close']], on='sym', suffixes=('_s','_e'))
    bench_ret = ((bench_merged['close_e'] - bench_merged['close_s'])/bench_merged['close_s']).mean() if len(bench_merged) > 0 else 0
    
    port_ret = np.mean(rets)
    alpha = port_ret - bench_ret
    wr = len([r for r in rets if r > 0]) / len(rets) * 100
    
    all_results.append({
        'signal_date': sig_date,
        'exit_date': exit_date,
        'port_ret': port_ret,
        'bench_ret': bench_ret,
        'alpha': alpha,
        'win_rate': wr,
        'n_stocks': len(rets),
        'best': max(rets),
        'worst': min(rets)
    })
    
    emoji = "🟢" if alpha > 0 else "🔴"
    print(f"  {emoji} {sig_date} → {exit_date}: 模型{port_ret*100:+6.1f}% 基准{bench_ret*100:+6.1f}% Alpha{alpha*100:+6.1f}% WR{wr:.0f}%")

# 汇总统计
rdf = pd.DataFrame(all_results)
print(f"\n{'='*70}")
print(f"📊 模拟验证汇总 ({len(rdf)}个时点)")
print(f"{'='*70}")

alpha_mean = rdf['alpha'].mean()
alpha_median = rdf['alpha'].median()
alpha_positive = (rdf['alpha'] > 0).sum() / len(rdf) * 100
port_mean = rdf['port_ret'].mean()
bench_mean = rdf['bench_ret'].mean()
port_win = (rdf['port_ret'] > 0).sum() / len(rdf) * 100

print(f"\n  模型组合:")
print(f"    平均收益:    {port_mean*100:+.2f}%")
print(f"    正收益占比:  {port_win:.0f}%")
print(f"    最佳:        {rdf['port_ret'].max()*100:+.1f}%")
print(f"    最差:        {rdf['port_ret'].min()*100:+.1f}%")

print(f"\n  全市场基准:")
print(f"    平均收益:    {bench_mean*100:+.2f}%")

print(f"\n  Alpha:")
print(f"    平均:        {alpha_mean*100:+.2f}%")
print(f"    中位数:      {alpha_median*100:+.2f}%")
print(f"    正Alpha占比: {alpha_positive:.0f}%")

# 年化指标
# 20天持有 → 约12.5次/年
annual_factor = 252 / hold_days
ann_alpha = alpha_mean * annual_factor
ann_port = port_mean * annual_factor

print(f"\n  年化估算 (20天→年):")
print(f"    模型年化:    {ann_port*100:+.1f}%")
print(f"    Alpha年化:   {ann_alpha*100:+.1f}%")

# 风险指标
port_rets = rdf['port_ret'].values
sharpe_ann = np.mean(port_rets) / np.std(port_rets) * np.sqrt(annual_factor) if np.std(port_rets) > 0 else 0
max_dd = 0
cum = np.cumprod(1 + port_rets)
peak = np.maximum.accumulate(cum)
dd = (cum - peak) / peak
max_dd = dd.min()

print(f"\n  风险指标 (每期):")
print(f"    Sharpe:      {sharpe_ann:.2f}")
print(f"    最大回撤:    {max_dd*100:.1f}%")
print(f"    胜率均值:    {rdf['win_rate'].mean():.0f}%")

# 保存
summary = {
    'n_periods': len(rdf),
    'hold_days': hold_days,
    'avg_port_return': round(port_mean*100, 2),
    'avg_bench_return': round(bench_mean*100, 2),
    'avg_alpha': round(alpha_mean*100, 2),
    'alpha_positive_pct': round(alpha_positive, 0),
    'ann_port': round(ann_port*100, 1),
    'ann_alpha': round(ann_alpha*100, 1),
    'sharpe': round(sharpe_ann, 2),
    'max_dd': round(max_dd*100, 1),
    'avg_win_rate': round(rdf['win_rate'].mean(), 0),
    'periods': all_results
}
with open('research/paper_trade_sim.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\n✅ 结果已保存 research/paper_trade_sim.json")
