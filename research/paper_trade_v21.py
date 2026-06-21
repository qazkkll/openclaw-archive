#!/usr/bin/env python3
"""Paper Trade验证: cn-alpha-v2.1 (XGBoost + SL-3%)
- 34+历史时点(每季度), 2年训练窗口
- Top15持有10天, SL-3%
- 信号分层: Top5(🟢🟢) > Top10(🟢) > Top15(🟡)
- 成本敏感性: 0.1%/0.15%/0.3%
- Alpha vs 市场基准(等权全市场)
"""
import pandas as pd, numpy as np, json, time, os, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))
t0 = time.time()
print(f"[Paper Trade] cn-alpha-v2.1 {time.strftime('%Y-%m-%d %H:%M')}")

# ========== 加载数据 ==========
print("[1/6] 加载数据...")
df = pd.read_parquet('data/a_hist_10y.parquet')
df = df.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df['date'] = df['date'].astype(int)
mf = pd.read_parquet('data/cn/moneyflow_core.parquet')
mf['sym'] = mf['ts_code'].str[:6]
mf['date'] = mf['trade_date'].astype(int)
for col in ['sm', 'md', 'lg', 'elg']:
    mf[f'{col}_net'] = mf[f'buy_{col}_amount'] - mf[f'sell_{col}_amount']
mf['total_net'] = mf['net_mf_amount']
df = df.merge(mf[['sym', 'date', 'total_net', 'lg_net', 'md_net', 'elg_net']], on=['sym', 'date'], how='left')
df = df[~df['sym'].str.startswith('688')].copy()
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
df = df[df['volume'] > 0].copy()
df = df.sort_values(['sym', 'date']).reset_index(drop=True)
print(f"  Data: {len(df):,} rows, {df['sym'].nunique()} stocks")

# ========== Price lookup ==========
print("[2/6] Price lookup + 特征...")
price_lookup = dict(zip(zip(df['sym'], df['date']), df['close']))

# ========== 特征 ==========
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ret10'] = df.groupby('sym')['close'].pct_change(10)
df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma60'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(60, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['ma60_bias'] = (df['close'] - df['ma60']) / df['ma60']
df['vol5'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(5, min_periods=2).std())
df['vol20'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(20, min_periods=2).std())
delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)
ema12 = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=12, min_periods=1).mean())
ema26 = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=26, min_periods=1).mean())
df['macd'] = ema12 - ema26
df['macd_signal'] = df.groupby('sym')['macd'].transform(lambda x: x.ewm(span=9, min_periods=1).mean())
df['macd_hist'] = df['macd'] - df['macd_signal']
df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df.groupby('sym')['close'].shift(1)), abs(df['low'] - df.groupby('sym')['close'].shift(1))))
df['atr14'] = df.groupby('sym')['tr'].transform(lambda x: x.rolling(14, min_periods=1).mean())
df['atr_pct'] = df['atr14'] / df['close']
df['vol_ratio'] = df.groupby('sym')['volume'].transform(lambda x: x.rolling(5).mean()) / df.groupby('sym')['volume'].transform(lambda x: x.rolling(20).mean())
for col in ['total_net', 'lg_net', 'md_net', 'elg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())
    df[f'{col}_20d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(20, min_periods=1).sum())
    df[f'{col}_5d_rk'] = df.groupby('date')[f'{col}_5d'].rank(pct=True)
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')
df['fwd_10d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-10) / x - 1)

XGB_FEATURES = [
    'ret5', 'ret10', 'ret20', 'ma20_bias', 'ma60_bias',
    'vol5', 'vol20', 'rsi_14', 'macd_hist', 'atr_pct', 'vol_ratio',
    'total_net_5d', 'lg_net_5d', 'md_net_5d', 'elg_net_5d',
    'total_net_20d', 'lg_net_20d', 'md_net_20d', 'elg_net_20d',
    'total_net_5d_rk', 'lg_net_5d_rk', 'md_net_5d_rk', 'elg_net_5d_rk',
    'breadth', 'mkt_ret20'
]

# ========== 构建季度调仓时点 ==========
print("[3/6] 构建季度调仓时点...")
all_dates = sorted(df['date'].unique())
def int_to_dt(d): return datetime(int(str(d)[:4]), int(str(d)[4:6]), int(str(d)[6:8]))
def dt_to_int(d): return int(d.strftime('%Y%m%d'))

# 每季度一个调仓点(1/4/7/10月第一个交易日)
quarter_dates = []
for d in all_dates:
    dt = int_to_dt(d)
    if dt.month in [1, 4, 7, 10] and dt.day <= 10:
        quarter_dates.append(d)

# 从2018年开始(需要2年训练窗口到2016)
rebal_dates = [d for d in quarter_dates if d >= 20180101]
print(f"  {len(rebal_dates)} 个季度调仓点")

# ========== Paper Trade模拟 ==========
print("[4/6] Paper Trade模拟...")
import xgboost as xgb

HOLD = 10
SL = -0.03
TOP_N = 15
TRAIN_WINDOW = 365 * 2  # 2年

# 结果存储
results = {
    'periods': [],       # 每个调仓期的详细结果
    'equity_curve': [],  # 权益曲线
    'signal_levels': {   # 信号分层
        'top5': {'rets': [], 'n': 0},
        'top10': {'rets': [], 'n': 0},
        'top15': {'rets': [], 'n': 0},
    }
}

equity = 1.0
equity_curve = [(rebal_dates[0], equity)]

for i, rebal_date in enumerate(rebal_dates):
    # 找退出日期
    rebal_idx = all_dates.index(rebal_date)
    exit_idx = min(rebal_idx + HOLD, len(all_dates) - 1)
    exit_date = all_dates[exit_idx]
    if exit_date == rebal_date:
        continue
    
    # 训练窗口
    train_end = rebal_date
    train_dates = [d for d in all_dates if d < train_end and d >= dt_to_int(int_to_dt(train_end) - timedelta(days=TRAIN_WINDOW))]
    
    train = df[df['date'].isin(train_dates)].dropna(subset=XGB_FEATURES + ['fwd_10d'])
    if len(train) < 1000:
        continue
    
    # 训练XGBoost
    model = xgb.XGBRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=4, verbosity=0
    )
    model.fit(train[XGB_FEATURES].fillna(0), train['fwd_10d'])
    
    # 预测
    day = df[df['date'] == rebal_date].copy()
    day = day[(day['close'] >= 3) & (day['close'] <= 200) & (~day['sym'].str.contains('ST|退市', na=False)) & (day['volume'] > 0)]
    if len(day) < 50:
        continue
    
    day['xgb_score'] = model.predict(day[XGB_FEATURES].fillna(0))
    day = day.nlargest(TOP_N, 'xgb_score')
    
    # 市场基准(等权全市场)
    all_stocks = df[df['date'] == rebal_date]
    all_stocks_exit = df[df['date'] == exit_date]
    if len(all_stocks) > 0 and len(all_stocks_exit) > 0:
        mkt_rets = []
        for _, row in all_stocks.head(500).iterrows():  # 取前500只代表市场
            ep = price_lookup.get((row['sym'], exit_date))
            if ep is not None:
                mkt_rets.append(ep / row['close'] - 1)
        mkt_ret = np.mean(mkt_rets) if mkt_rets else 0
    else:
        mkt_ret = 0
    
    # 计算各层信号收益
    signal_groups = {
        'top5': day.head(5),
        'top10': day.head(10),
        'top15': day,
    }
    
    period_detail = {
        'rebal_date': rebal_date,
        'exit_date': exit_date,
        'train_size': len(train),
        'market_ret': mkt_ret,
    }
    
    for level, group in signal_groups.items():
        rets = []
        for _, row in group.iterrows():
            exit_price = price_lookup.get((row['sym'], exit_date))
            if exit_price is None:
                continue
            ret = exit_price / row['close'] - 1
            # Stop loss
            for j in range(rebal_idx + 1, exit_idx + 1):
                ip = price_lookup.get((row['sym'], all_dates[j]))
                if ip is not None and ip / row['close'] - 1 <= SL:
                    ret = SL
                    break
            rets.append(ret)
        
        if rets:
            avg_ret = np.mean(rets)
            period_detail[f'{level}_ret'] = avg_ret
            period_detail[f'{level}_alpha'] = avg_ret - mkt_ret
            results['signal_levels'][level]['rets'].append(avg_ret)
            results['signal_levels'][level]['n'] += len(rets)
    
    # 等权Top15收益(无成本)
    top15_ret = period_detail.get('top15_ret', 0)
    period_detail['top15_raw'] = top15_ret
    
    results['periods'].append(period_detail)
    equity *= (1 + top15_ret)
    equity_curve.append((exit_date, equity))
    
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(rebal_dates)} 完成, equity={equity:.4f}")

print(f"  全部完成: {len(results['periods'])}个调仓期")

# ========== 成本敏感性分析 ==========
print("[5/6] 成本敏感性分析...")
costs = [0.001, 0.0015, 0.003]  # 0.1%, 0.15%, 0.3% 单边
cost_results = {}

for cost in costs:
    eq = 1.0
    rets_with_cost = []
    for p in results['periods']:
        ret = p.get('top15_raw', 0) - cost * 2  # 双边
        rets_with_cost.append(ret)
        eq *= (1 + ret)
    
    if len(rets_with_cost) > 2:
        avg = np.mean(rets_with_cost)
        std = np.std(rets_with_cost)
        ann_ret = avg * (252 / HOLD)
        ann_std = std * np.sqrt(252 / HOLD)
        sharpe = ann_ret / ann_std if ann_std > 0 else 0
        wr = np.mean([r > 0 for r in rets_with_cost])
        
        # 计算最大回撤
        eqs = [1.0]
        for r in rets_with_cost:
            eqs.append(eqs[-1] * (1 + r))
        peak = eqs[0]
        max_dd = 0
        for e in eqs:
            if e > peak:
                peak = e
            dd = (e - peak) / peak
            if dd < max_dd:
                max_dd = dd
        
        cost_results[f'{cost*100:.1f}%'] = {
            'sharpe': sharpe,
            'ann_ret': ann_ret,
            'win_rate': wr,
            'max_dd': max_dd,
            'final_equity': eqs[-1],
        }

# ========== 输出 ==========
print("[6/6] 输出结果...")
print("\n" + "=" * 100)
print("📊 Paper Trade验证: cn-alpha-v2.1 (XGBoost + SL-3%)")
print("=" * 100)

# 基础指标
periods = results['periods']
top15_rets = [p.get('top15_ret', 0) for p in periods]
mkt_rets = [p['market_ret'] for p in periods]
alphas = [p.get('top15_alpha', 0) for p in periods]

avg_ret = np.mean(top15_rets)
std_ret = np.std(top15_rets)
ann_ret = avg_ret * (252 / HOLD)
ann_std = std_ret * np.sqrt(252 / HOLD)
sharpe = ann_ret / ann_std if ann_std > 0 else 0
wr = np.mean([r > 0 for r in top15_rets])
alpha_positive = np.mean([a > 0 for a in alphas])

# 权益曲线DD
eqs = [e for _, e in equity_curve]
peak = eqs[0]
max_dd = 0
for e in eqs:
    if e > peak: peak = e
    dd = (e - peak) / peak
    if dd < max_dd: max_dd = dd

print(f"\n📈 总体指标")
print(f"  调仓期数:     {len(periods)}")
print(f"  平均收益:     {avg_ret*100:.2f}% (每期)")
print(f"  年化收益:     {ann_ret*100:.1f}%")
print(f"  年化波动:     {ann_std*100:.1f}%")
print(f"  Sharpe:       {sharpe:.3f} {'✅' if sharpe > 1.0 else '⚠️'}")
print(f"  胜率:         {wr:.1%} {'✅' if wr > 50 else '⚠️'}")
print(f"  Alpha正占比:  {alpha_positive:.1%} {'✅' if alpha_positive > 0.6 else '⚠️'}")
print(f"  最大回撤:     {max_dd*100:.1f}% {'✅' if max_dd > -0.20 else '⚠️'}")
print(f"  最终权益:     {equity:.4f}")

# 信号分层
print(f"\n📊 信号分层验证")
print(f"  {'层级':<8} {'平均收益':>10} {'Alpha':>10} {'胜率':>8} {'交易数':>8}")
print(f"  {'-'*50}")
for level in ['top5', 'top10', 'top15']:
    sl = results['signal_levels'][level]
    if sl['rets']:
        avg = np.mean(sl['rets'])
        avg_alpha = avg - np.mean(mkt_rets)
        wr_l = np.mean([r > 0 for r in sl['rets']])
        label = {'top5': '🟢🟢精品', 'top10': '🟢强信号', 'top15': '🟡观察'}[level]
        print(f"  {label:<8} {avg*100:>9.2f}% {avg_alpha*100:>9.2f}% {wr_l:>7.1%} {sl['n']:>8}")

# 分层是否递减
t5 = np.mean(results['signal_levels']['top5']['rets']) if results['signal_levels']['top5']['rets'] else 0
t10 = np.mean(results['signal_levels']['top10']['rets']) if results['signal_levels']['top10']['rets'] else 0
t15 = np.mean(results['signal_levels']['top15']['rets']) if results['signal_levels']['top15']['rets'] else 0
tier_valid = t5 > t10 > t15 or abs(t5 - t15) < 0.005  # 允许小差异
print(f"\n  分层递减: {'✅ 🟢🟢>🟢>🟡' if t5 > t10 > t15 else '⚠️ 不完全递减'}")

# 成本敏感性
print(f"\n💰 成本敏感性")
print(f"  {'单边成本':>10} {'Sharpe':>8} {'年化':>8} {'胜率':>8} {'MaxDD':>8} {'终值':>10}")
print(f"  {'-'*58}")
for cost_label, cr in cost_results.items():
    print(f"  {cost_label:>10} {cr['sharpe']:>8.3f} {cr['ann_ret']*100:>7.1f}% {cr['win_rate']:>7.1%} {cr['max_dd']*100:>7.1f}% {cr['final_equity']:>10.4f}")

# 门限判定
print(f"\n🏁 门限判定")
thresholds = [
    ('Sharpe > 1.0', sharpe > 1.0, f'{sharpe:.3f}'),
    ('Alpha正占比 > 60%', alpha_positive > 0.6, f'{alpha_positive:.1%}'),
    ('Max DD < -20%', max_dd > -0.20, f'{max_dd*100:.1f}%'),
    ('胜率 > 50%', wr > 0.5, f'{wr:.1%}'),
    ('调仓期 > 30', len(periods) >= 30, f'{len(periods)}'),
]
all_pass = True
for name, passed, value in thresholds:
    status = '✅' if passed else '❌'
    print(f"  {status} {name}: {value}")
    if not passed:
        all_pass = False

print(f"\n{'✅ Paper Trade验证通过!' if all_pass else '⚠️ 部分门限未通过'}")
print(f"⏱️ 耗时: {time.time()-t0:.0f}s")

# 保存结果
output = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'model': 'cn-alpha-v2.1',
    'config': {'hold': HOLD, 'sl': SL, 'top_n': TOP_N, 'train_window_days': TRAIN_WINDOW},
    'metrics': {
        'sharpe': round(sharpe, 4),
        'ann_ret': round(ann_ret, 4),
        'win_rate': round(wr, 4),
        'alpha_positive': round(alpha_positive, 4),
        'max_dd': round(max_dd, 4),
        'n_periods': len(periods),
        'final_equity': round(equity, 4),
    },
    'signal_levels': {k: {'avg_ret': round(np.mean(v['rets']), 6) if v['rets'] else 0, 'n': v['n']} for k, v in results['signal_levels'].items()},
    'cost_sensitivity': cost_results,
    'thresholds_passed': all_pass,
}
with open('research/paper_trade_v21_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n结果已保存: research/paper_trade_v21_results.json")
