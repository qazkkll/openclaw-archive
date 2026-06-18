#!/usr/bin/env python3
"""
蓝盾V4 分类模型深度分析
信号分级 + 时间分布 + 月度收益 + 连续亏损
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据 & 特征（复用之前逻辑）
# ============================================================
print("=" * 70)
print("蓝盾V4 分类模型深度分析")
print("=" * 70)

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
# 2. 特征计算
# ============================================================
print("\n计算特征...")
t_start = time.time()

features = {}

for w in [5, 10, 20, 60]:
    features[f'ret_{w}d'] = df.groupby('code')['close'].pct_change(w)
    features[f'vol_{w}d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(w).std())

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

ema12 = df.groupby('code')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
ema26 = df.groupby('code')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
macd_line = ema12 - ema26
signal_line = macd_line.groupby(df['code']).transform(lambda x: x.ewm(span=9, adjust=False).mean())
features['macd'] = macd_line
features['macd_signal'] = signal_line
features['macd_hist'] = macd_line - signal_line

sma20 = df.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
std20 = df.groupby('code')['close'].transform(lambda x: x.rolling(20).std())
bb_upper = sma20 + 2 * std20
bb_lower = sma20 - 2 * std20
features['bb_width'] = (bb_upper - bb_lower) / sma20
features['bb_pct'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

for w in [5, 10, 20, 60]:
    ma = df.groupby('code')['close'].transform(lambda x: x.rolling(w).mean())
    features[f'bias_{w}d'] = (df['close'] - ma) / ma

for w in [5, 20]:
    features[f'vol_ratio_{w}d'] = df.groupby('code')['volume'].transform(
        lambda x: x / x.rolling(w).mean()
    )

features['high_low_range'] = (df['high'] - df['low']) / df['close']
features['close_open_range'] = (df['close'] - df['open']) / df['open']

for w in [5, 10, 20]:
    features[f'momentum_{w}d'] = df.groupby('code')['close'].pct_change(w)

for w in [10, 20]:
    features[f'trend_strength_{w}d'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(w).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == w else np.nan, raw=True)
    )

for w in [20, 60]:
    features[f'price_position_{w}d'] = df.groupby('code')['close'].transform(
        lambda x: (x - x.rolling(w).min()) / (x.rolling(w).max() - x.rolling(w).min() + 1e-10)
    )

feat_df = pd.DataFrame(features, index=df.index)
feat_df['close'] = df['close']
feat_df['volume'] = df['volume']
feat_df['code'] = df['code']
feat_df['date'] = df['date']

exclude = {'close', 'volume', 'code', 'date'}
feature_cols = [c for c in feat_df.columns if c not in exclude]
feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

print(f"特征数: {len(feature_cols)}, 耗时: {time.time()-t_start:.1f}s")

# ============================================================
# 3. 训练分类模型（正类=5%，最实用的阈值）
# ============================================================
WINDOW = 5
POS_THRESH = 0.05  # 正类：5天后涨超5%

feat_df['fwd_ret'] = feat_df.groupby('code')['close'].pct_change(WINDOW).shift(-WINDOW)
valid = feat_df.dropna(subset=feature_cols + ['fwd_ret']).copy()
valid = valid[valid['fwd_ret'].between(-0.5, 0.5)]

valid = valid.sort_values('date')
train_end = valid['date'].quantile(0.6)
val_end = valid['date'].quantile(0.8)

train = valid[valid['date'] <= train_end].copy()
val = valid[(valid['date'] > train_end) & (valid['date'] <= val_end)].copy()
test = valid[valid['date'] > val_end].copy()

print(f"\n训练: {len(train):,}  验证: {len(val):,}  测试: {len(test):,}")

X_train = train[feature_cols].values
X_val = val[feature_cols].values
X_test = test[feature_cols].values

y_train = (train['fwd_ret'].values >= POS_THRESH).astype(int)
y_val = (val['fwd_ret'].values >= POS_THRESH).astype(int)

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale = n_neg / n_pos

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': scale,
    'verbose': -1,
    'seed': 42
}

model = lgb.train(
    params, train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

test = test.copy()
test['prob'] = model.predict(X_test)
test['actual'] = test['fwd_ret'].values

print(f"模型训练完成, 最佳轮数: {model.best_iteration}")

# ============================================================
# 4. 信号分级分析
# ============================================================
print(f"\n{'=' * 70}")
print("信号分级分析")
print(f"{'=' * 70}")

levels = [
    ('🟢🟢 强烈推荐', 0.80),
    ('🟢 重点关注', 0.70),
    ('🟡 观察', 0.60),
    ('⚪ 一般', 0.50),
]

for name, threshold in levels:
    signals = test[test['prob'] >= threshold]
    n = len(signals)
    if n == 0:
        print(f"\n{name} (≥{threshold:.0%}): 无信号")
        continue
    
    actual = signals['actual']
    win_0 = (actual > 0).mean()
    win_3 = (actual > 0.03).mean()
    win_5 = (actual >= 0.05).mean()
    
    winners = actual[actual > 0]
    losers = actual[actual <= 0]
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.001
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    ev = win_0 * avg_win - (1 - win_0) * avg_loss
    
    # 年化收益（简化）
    # 每笔交易持有5天，一年约50次机会
    trades_per_year = min(n / 3, 252 / WINDOW)  # 3年测试期
    ann_ret = ev * trades_per_year
    
    print(f"\n{name} (≥{threshold:.0%}):")
    print(f"  信号数: {n:,} (约{trades_per_year:.0f}笔/年)")
    print(f"  实际>0%: {win_0:.1%}  实际>3%: {win_3:.1%}  实际≥5%: {win_5:.1%}")
    print(f"  平均赢: {avg_win:.2%}  平均亏: {avg_loss:.2%}  盈亏比: {pl_ratio:.2f}")
    print(f"  期望值/笔: {ev:.4f} ({ev*100:.2f}%)")
    print(f"  估算年化: {ann_ret:.1%}")

# ============================================================
# 5. 时间分布分析
# ============================================================
print(f"\n{'=' * 70}")
print("时间分布分析 (🟢🟢 强烈推荐)")
print(f"{'=' * 70}")

strong = test[test['prob'] >= 0.80].copy()
if len(strong) > 0:
    strong['month'] = strong['date'].dt.to_period('M')
    strong['year'] = strong['date'].dt.year
    
    # 月度信号数
    monthly = strong.groupby('month').agg(
        signals=('prob', 'count'),
        avg_prob=('prob', 'mean'),
        win_rate=('actual', lambda x: (x > 0).mean()),
        avg_ret=('actual', 'mean')
    )
    
    print(f"\n月度信号分布:")
    print(f"{'月份':>10} | {'信号数':>6} | {'平均概率':>8} | {'胜率':>6} | {'平均收益':>8}")
    print("-" * 55)
    for month, row in monthly.iterrows():
        print(f"{str(month):>10} | {row['signals']:>6.0f} | {row['avg_prob']:>7.1%} | {row['win_rate']:>5.1%} | {row['avg_ret']:>7.2%}")
    
    # 年度统计
    yearly = strong.groupby('year').agg(
        signals=('prob', 'count'),
        win_rate=('actual', lambda x: (x > 0).mean()),
        avg_ret=('actual', 'mean')
    )
    
    print(f"\n年度统计:")
    for year, row in yearly.iterrows():
        print(f"  {year}: {row['signals']:.0f}个信号, 胜率{row['win_rate']:.1%}, 平均收益{row['avg_ret']:.2%}")

# ============================================================
# 6. 连续亏损分析
# ============================================================
print(f"\n{'=' * 70}")
print("连续亏损分析 (🟢🟢 强烈推荐)")
print(f"{'=' * 70}")

if len(strong) > 0:
    # 按日期排序，计算连续亏损
    strong_sorted = strong.sort_values('date')
    strong_sorted['win'] = (strong_sorted['actual'] > 0).astype(int)
    
    # 计算连续亏损次数
    strong_sorted['loss_streak'] = 0
    streak = 0
    for i in range(len(strong_sorted)):
        if strong_sorted.iloc[i]['win'] == 0:
            streak += 1
        else:
            streak = 0
        strong_sorted.iloc[i, strong_sorted.columns.get_loc('loss_streak')] = streak
    
    max_streak = strong_sorted['loss_streak'].max()
    avg_streak = strong_sorted['loss_streak'][strong_sorted['loss_streak'] > 0].mean() if (strong_sorted['loss_streak'] > 0).any() else 0
    
    print(f"  最大连续亏损: {max_streak:.0f}笔")
    print(f"  平均连续亏损: {avg_streak:.1f}笔")
    
    # 连续亏损分布
    streak_dist = strong_sorted['loss_streak'][strong_sorted['loss_streak'] > 0].value_counts().sort_index()
    if len(streak_dist) > 0:
        print(f"\n  连续亏损分布:")
        for streak_len, count in streak_dist.items():
            print(f"    连续{streak_len}笔亏损: {count:.0f}次")

# ============================================================
# 7. 月度收益分布
# ============================================================
print(f"\n{'=' * 70}")
print("月度收益分布 (🟢🟢 强烈推荐)")
print(f"{'=' * 70}")

if len(strong) > 0:
    monthly_ret = strong.groupby('month').agg(
        total_ret=('actual', 'sum'),
        avg_ret=('actual', 'mean'),
        n_trades=('actual', 'count'),
        win_rate=('actual', lambda x: (x > 0).mean())
    )
    
    print(f"\n{'月份':>10} | {'总收益':>8} | {'平均收益':>8} | {'交易数':>6} | {'胜率':>6}")
    print("-" * 55)
    for month, row in monthly_ret.iterrows():
        print(f"{str(month):>10} | {row['total_ret']:>7.2%} | {row['avg_ret']:>7.2%} | {row['n_trades']:>6.0f} | {row['win_rate']:>5.1%}")
    
    # 月度收益统计
    pos_months = (monthly_ret['total_ret'] > 0).sum()
    neg_months = (monthly_ret['total_ret'] <= 0).sum()
    print(f"\n  盈利月: {pos_months}  亏损月: {neg_months}  盈亏比: {pos_months/max(neg_months,1):.1f}")

# ============================================================
# 8. 与V3公式对比
# ============================================================
print(f"\n{'=' * 70}")
print("与V3公式评分对比")
print(f"{'=' * 70}")

print(f"""
V3公式评分（生产中）:
  - 110分制，6维度
  - ≥80分买入，<70分退出
  - 回撤: ~4%
  - 年化: ~15-20%
  - 胜率: 未统计（趋势确认，非预测）

V4分类模型（实验中）:
  - 正类5%，入场80%
  - 4,746笔交易/3年
  - 胜率: 56.8%（实际>0%）
  - 盈亏比: 1.27
  - 期望值: 0.018/笔

  - 正类2%，入场80%
  - 176笔交易/3年
  - 胜率: 69.3%（实际>0%）
  - 盈亏比: 1.43
  - 期望值: 0.054/笔

关键差异:
  V3: 趋势确认，不预测涨幅，回撤小
  V4: 预测涨幅概率，信号更精确，但需要止损
""")

print("=" * 70)
print("分析完成！")
print("=" * 70)
