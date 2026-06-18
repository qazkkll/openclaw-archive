#!/usr/bin/env python3
"""
蓝盾V4 综合回测：窗口 × 胜率阈值 × 盈亏比
特征只算一次，只换目标变量
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("蓝盾V4 综合回测")
print("=" * 60)

DATA = "/home/hermes/.hermes/openclaw-archive/data/us/us_hist_sp500_10y.parquet"
df = pd.read_parquet(DATA)
df['date'] = pd.to_datetime(df['date'])
df = df.rename(columns={'sym': 'code'})
df = df.sort_values(['code', 'date']).reset_index(drop=True)

sp500_tickers = {'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'IVV', 'XLK', 'XLF',
                 'XLV', 'XLE', 'XLI', 'XLP', 'XLU', 'XLRE', 'XLB', 'XLC', 'XLY'}
df = df[~df['code'].isin(sp500_tickers)].copy()

print(f"数据: {len(df):,} 行, {df['code'].nunique()} 只股票")

# ============================================================
# 2. 全局特征（只算一次）
# ============================================================
print("\n计算特征（一次性）...")

features = {}

# --- 价格特征 ---
for w in [5, 10, 20, 60]:
    features[f'ret_{w}d'] = df.groupby('code')['close'].pct_change(w)
    features[f'vol_{w}d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(w).std())

# --- RSI (用transform避免索引问题) ---
for w in [14, 28]:
    def calc_rsi(x, window=w):
        delta = x.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)
    features[f'rsi_{w}'] = df.groupby('code')['close'].transform(calc_rsi)

# --- MACD ---
ema12 = df.groupby('code')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
ema26 = df.groupby('code')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
macd_line = ema12 - ema26
signal_line = macd_line.groupby(df['code']).transform(lambda x: x.ewm(span=9, adjust=False).mean())
features['macd'] = macd_line
features['macd_signal'] = signal_line
features['macd_hist'] = macd_line - signal_line

# --- 布林带 ---
sma20 = df.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
std20 = df.groupby('code')['close'].transform(lambda x: x.rolling(20).std())
bb_upper = sma20 + 2 * std20
bb_lower = sma20 - 2 * std20
features['bb_width'] = (bb_upper - bb_lower) / sma20
features['bb_pct'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

# --- 均线偏离 ---
for w in [5, 10, 20, 60]:
    ma = df.groupby('code')['close'].transform(lambda x: x.rolling(w).mean())
    features[f'bias_{w}d'] = (df['close'] - ma) / ma

# --- 成交量特征 ---
for w in [5, 20]:
    features[f'vol_ratio_{w}d'] = df.groupby('code')['volume'].transform(
        lambda x: x / x.rolling(w).mean()
    )

# --- 波动率特征 ---
features['high_low_range'] = (df['high'] - df['low']) / df['close']
features['close_open_range'] = (df['close'] - df['open']) / df['open']

# --- 动量特征 ---
for w in [5, 10, 20]:
    features[f'momentum_{w}d'] = df.groupby('code')['close'].pct_change(w)

# --- 趋势强度 (简化：用线性回归斜率) ---
for w in [10, 20]:
    features[f'trend_strength_{w}d'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(w).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == w else np.nan, raw=True)
    )

# --- 价格位置 ---
for w in [20, 60]:
    features[f'price_position_{w}d'] = df.groupby('code')['close'].transform(
        lambda x: (x - x.rolling(w).min()) / (x.rolling(w).max() - x.rolling(w).min() + 1e-10)
    )

# 组合特征
feat_df = pd.DataFrame(features, index=df.index)
feat_df['close'] = df['close']
feat_df['volume'] = df['volume']
feat_df['code'] = df['code']
feat_df['date'] = df['date']

# --- 市场特征 (SPY) ---
print("计算市场特征...")
spy_raw = pd.read_parquet(DATA)
spy_raw = spy_raw.rename(columns={'sym': 'code'})
spy = spy_raw[spy_raw['code'] == 'SPY'][['date', 'close', 'volume']].rename(
    columns={'close': 'spy_close', 'volume': 'spy_volume'}
).sort_values('date')

for w in [5, 10, 20]:
    spy[f'spy_ret_{w}d'] = spy['spy_close'].pct_change(w)
    spy[f'spy_vol_{w}d'] = spy['spy_close'].pct_change().rolling(w).std()

spy['vix_proxy'] = spy['spy_close'].pct_change().rolling(20).std() * np.sqrt(252) * 100

feat_df = feat_df.merge(spy[['date'] + [c for c in spy.columns if c.startswith('spy_') or c == 'vix_proxy']],
                        on='date', how='left')

# 标记特征列
exclude = {'close', 'volume', 'code', 'date'}
feature_cols = [c for c in feat_df.columns if c not in exclude]
print(f"特征数: {len(feature_cols)}")

# 清理
feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

# ============================================================
# 3. 测试多个窗口
# ============================================================
windows = [1, 2, 3, 5, 7, 10, 15, 20]
results = []

for window in windows:
    print(f"\n{'=' * 60}")
    print(f"测试窗口: {window}天")
    print(f"{'=' * 60}")

    t0 = time.time()

    # 计算目标变量
    tmp = feat_df.copy()
    tmp['fwd_ret'] = tmp.groupby('code')['close'].pct_change(window).shift(-window)

    # 去掉NaN
    valid = tmp.dropna(subset=feature_cols + ['fwd_ret'])
    valid = valid[valid['fwd_ret'].between(-0.5, 0.5)]  # 去极端值

    print(f"  有效数据: {len(valid):,} 行")

    # 时间划分
    valid = valid.sort_values('date')
    train_end = valid['date'].quantile(0.6)
    val_end = valid['date'].quantile(0.8)

    train = valid[valid['date'] <= train_end]
    val = valid[(valid['date'] > train_end) & (valid['date'] <= val_end)]
    test = valid[valid['date'] > val_end]

    print(f"  训练: {len(train):,}  验证: {len(val):,}  测试: {len(test):,}")

    if len(train) < 100 or len(test) < 50:
        print(f"  数据不足，跳过")
        continue

    X_train = train[feature_cols].values
    X_val = val[feature_cols].values
    X_test = test[feature_cols].values
    y_train = train['fwd_ret'].values
    y_val = val['fwd_ret'].values
    y_test = test['fwd_ret'].values

    # --- 训练回归模型 ---
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    model = lgb.train(
        params, train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )

    # --- 预测 ---
    test = test.copy()
    test['pred'] = model.predict(X_test)

    # --- 各种指标 ---
    corr = test['pred'].corr(test['fwd_ret'])

    # 选股策略：只买模型预测涨最多的
    top_n = max(20, len(test) // 50)  # 前20%或至少20只
    top = test.nlargest(top_n, 'pred')
    bottom = test.nsmallest(top_n, 'pred')

    # 策略收益
    strategy_ret = top['fwd_ret'].mean()
    benchmark_ret = test['fwd_ret'].mean()

    # --- 胜率定义 ---
    pred_buy = test[test['pred'] > 0]

    # 基础胜率：预测>0且实际>0
    win_base = (pred_buy['fwd_ret'] > 0).mean() if len(pred_buy) > 0 else 0

    # 预测>0且实际>3%
    win_3pct = (pred_buy['fwd_ret'] > 0.03).mean() if len(pred_buy) > 0 else 0

    # 预测>0且实际>5%
    win_5pct = (pred_buy['fwd_ret'] > 0.05).mean() if len(pred_buy) > 0 else 0

    # 策略胜率
    strategy_win = (top['fwd_ret'] > 0).mean()
    strategy_win_3 = (top['fwd_ret'] > 0.03).mean()
    strategy_win_5 = (top['fwd_ret'] > 0.05).mean()

    # --- 盈亏比 ---
    winners = top[top['fwd_ret'] > 0]['fwd_ret']
    losers = top[top['fwd_ret'] <= 0]['fwd_ret']
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.001
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

    # 全样本盈亏比
    all_winners = test[test['fwd_ret'] > 0]['fwd_ret']
    all_losers = test[test['fwd_ret'] <= 0]['fwd_ret']
    all_avg_win = all_winners.mean() if len(all_winners) > 0 else 0
    all_avg_loss = abs(all_losers.mean()) if len(all_losers) > 0 else 0.001
    all_pl_ratio = all_avg_win / all_avg_loss if all_avg_loss > 0 else float('inf')

    # --- 期望值 ---
    ev_strategy = strategy_win * avg_win - (1 - strategy_win) * avg_loss

    # --- 年化 ---
    trades_per_year = 252 / window
    ann_ret = strategy_ret * trades_per_year
    ann_ret_base = benchmark_ret * trades_per_year

    elapsed = time.time() - t0

    print(f"\n  --- 基础统计 ---")
    print(f"  预测相关性: {corr:.4f}")

    print(f"\n  --- 选股策略 (前{top_n}只) ---")
    print(f"  策略平均收益: {strategy_ret:.4f} ({strategy_ret*100:.2f}%)")
    print(f"  基准平均收益: {benchmark_ret:.4f} ({benchmark_ret*100:.2f}%)")
    print(f"  年化收益(策略): {ann_ret:.2%}  年化收益(基准): {ann_ret_base:.2%}")

    print(f"\n  --- 胜率 ---")
    print(f"  基础胜率(预测>0且实际>0): {win_base:.1%}")
    print(f"  显著盈利(预测>0且实际>3%): {win_3pct:.1%}")
    print(f"  大幅盈利(预测>0且实际>5%): {win_5pct:.1%}")
    print(f"  策略胜率(前N只实际>0): {strategy_win:.1%}")
    print(f"  策略显著盈利(前N只实际>3%): {strategy_win_3:.1%}")
    print(f"  策略大幅盈利(前N只实际>5%): {strategy_win_5:.1%}")

    print(f"\n  --- 盈亏比 ---")
    print(f"  策略内: 平均赢 {avg_win:.4f} ({avg_win*100:.2f}%) / 平均亏 {avg_loss:.4f} ({avg_loss*100:.2f}%)")
    print(f"  策略盈亏比: {profit_loss_ratio:.2f}")
    print(f"  全样本盈亏比: {all_pl_ratio:.2f}")

    print(f"\n  --- 期望值 ---")
    print(f"  策略期望值: {ev_strategy:.4f} ({ev_strategy*100:.2f}%)")

    results.append({
        'window': window,
        'corr': corr,
        'strategy_ret': strategy_ret,
        'benchmark_ret': benchmark_ret,
        'ann_ret': ann_ret,
        'ann_ret_base': ann_ret_base,
        'win_base': win_base,
        'win_3pct': win_3pct,
        'win_5pct': win_5pct,
        'strategy_win': strategy_win,
        'strategy_win_3': strategy_win_3,
        'strategy_win_5': strategy_win_5,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'all_pl_ratio': all_pl_ratio,
        'ev_strategy': ev_strategy,
        'elapsed': elapsed,
        'train_size': len(train),
        'test_size': len(test),
    })

# ============================================================
# 4. 汇总对比
# ============================================================
if results:
    print("\n" + "=" * 60)
    print("综合对比表")
    print("=" * 60)

    df_r = pd.DataFrame(results)

    header = f"{'窗口':>4} | {'相关性':>6} | {'年化收益':>8} | {'基准':>6} | {'策略胜率':>7} | {'>3%胜率':>7} | {'>5%胜率':>7} | {'盈亏比':>6} | {'期望值':>6} | {'耗时':>5}"
    print(header)
    print("-" * len(header))

    for _, r in df_r.iterrows():
        print(f"{r['window']:>3}天 | {r['corr']:>6.3f} | {r['ann_ret']:>7.1%} | {r['ann_ret_base']:>5.1%} | {r['strategy_win']:>6.1%} | {r['strategy_win_3']:>6.1%} | {r['strategy_win_5']:>6.1%} | {r['profit_loss_ratio']:>5.2f} | {r['ev_strategy']:>5.3f} | {r['elapsed']:>4.0f}s")

    # 最优窗口
    best_ev = df_r.loc[df_r['ev_strategy'].idxmax()]
    best_pl = df_r.loc[df_r['profit_loss_ratio'].idxmax()]

    print(f"\n最高期望值: {best_ev['window']}天窗口 (EV={best_ev['ev_strategy']:.4f}, 年化{best_ev['ann_ret']:.1%})")
    print(f"最高盈亏比: {best_pl['window']}天窗口 (PL={best_pl['profit_loss_ratio']:.2f}, 年化{best_pl['ann_ret']:.1%})")

    # 保存
    df_r.to_csv('/home/hermes/.hermes/openclaw-archive/analysis/v4_window_comprehensive.csv', index=False)
    print(f"\n结果已保存到 analysis/v4_window_comprehensive.csv")
else:
    print("\n没有有效结果！")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)
