#!/usr/bin/env python3
"""
蓝盾V4 分类模型实验矩阵
正类阈值 × 入场概率阈值 × 多维评估
无市场特征（数据中无SPY），纯个股39特征
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
print("=" * 70)
print("蓝盾V4 分类模型实验矩阵")
print("=" * 70)

DATA = "/home/hermes/.hermes/openclaw-archive/data/us/us_hist_sp500_10y.parquet"
df = pd.read_parquet(DATA)
df['date'] = pd.to_datetime(df['date'])
df = df.rename(columns={'sym': 'code'})
df = df.sort_values(['code', 'date']).reset_index(drop=True)

# 去掉ETF/index（数据中没有，但留着保险）
sp500_tickers = {'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'IVV', 'XLK', 'XLF',
                 'XLV', 'XLE', 'XLI', 'XLP', 'XLU', 'XLRE', 'XLB', 'XLC', 'XLY'}
df = df[~df['code'].isin(sp500_tickers)].copy()

print(f"数据: {len(df):,} 行, {df['code'].nunique()} 只股票")

# ============================================================
# 2. 全局特征（只算一次，无市场特征）
# ============================================================
print("\n计算特征（一次性）...")
t_start = time.time()

features = {}

# --- 价格特征 ---
for w in [5, 10, 20, 60]:
    features[f'ret_{w}d'] = df.groupby('code')['close'].pct_change(w)
    features[f'vol_{w}d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(w).std())

# --- RSI ---
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

# --- 趋势强度 ---
for w in [10, 20]:
    features[f'trend_strength_{w}d'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(w).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == w else np.nan, raw=True)
    )

# --- 价格位置 ---
for w in [20, 60]:
    features[f'price_position_{w}d'] = df.groupby('code')['close'].transform(
        lambda x: (x - x.rolling(w).min()) / (x.rolling(w).max() - x.rolling(w).min() + 1e-10)
    )

# 组合
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
# 3. 固定5天窗口，测试正类阈值 × 入场概率
# ============================================================
WINDOW = 5
print(f"\n预测窗口: {WINDOW}天")

# 预计算 forward return
feat_df['fwd_ret'] = feat_df.groupby('code')['close'].pct_change(WINDOW).shift(-WINDOW)
valid = feat_df.dropna(subset=feature_cols + ['fwd_ret']).copy()
valid = valid[valid['fwd_ret'].between(-0.5, 0.5)]

# 时间划分
valid = valid.sort_values('date')
train_end = valid['date'].quantile(0.6)
val_end = valid['date'].quantile(0.8)

train = valid[valid['date'] <= train_end].copy()
val = valid[(valid['date'] > train_end) & (valid['date'] <= val_end)].copy()
test = valid[valid['date'] > val_end].copy()

print(f"训练: {len(train):,}  验证: {len(val):,}  测试: {len(test):,}")

X_train = train[feature_cols].values
X_val = val[feature_cols].values
X_test = test[feature_cols].values
y_train_ret = train['fwd_ret'].values
y_val_ret = val['fwd_ret'].values
y_test_ret = test['fwd_ret'].values

# ============================================================
# 4. 测试不同正类阈值
# ============================================================
pos_thresholds = [0.02, 0.03, 0.05, 0.08, 0.10]
entry_probs = [0.50, 0.60, 0.70, 0.80]

all_results = []

for pos_thresh in pos_thresholds:
    print(f"\n{'=' * 70}")
    print(f"正类阈值: {pos_thresh:.0%} (5天后涨超{pos_thresh:.0%}为正类)")
    print(f"{'=' * 70}")

    # 转换为目标变量
    y_train = (y_train_ret >= pos_thresh).astype(int)
    y_val = (y_val_ret >= pos_thresh).astype(int)
    y_test = (y_test_ret >= pos_thresh).astype(int)

    pos_rate_train = y_train.mean()
    pos_rate_val = y_val.mean()
    pos_rate_test = y_test.mean()

    print(f"  正类比例: 训练{pos_rate_train:.1%}  验证{pos_rate_val:.1%}  测试{pos_rate_test:.1%}")

    # 训练分类模型
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 处理类别不平衡
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale = n_neg / n_pos if n_pos > 0 else 1

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

    # 预测概率
    test = test.copy()
    test['prob'] = model.predict(X_test)

    avg_prob = test['prob'].mean()
    print(f"  平均预测概率: {avg_prob:.3f}")
    print(f"  模型最佳轮数: {model.best_iteration}")

    # 测试不同入场概率阈值
    print(f"\n  {'入场阈值':>8} | {'交易数':>6} | {'实际>0%':>7} | {'实际>3%':>7} | {'达标胜率':>8} | {'平均赢':>7} | {'平均亏':>7} | {'盈亏比':>6} | {'期望值':>7}")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")

    for entry_prob in entry_probs:
        signals = test[test['prob'] >= entry_prob].copy()
        n_trades = len(signals)

        if n_trades == 0:
            print(f"  {entry_prob:>7.0%} | {'0':>6} | {'-':>7} | {'-':>7} | {'-':>8} | {'-':>7} | {'-':>7} | {'-':>6} | {'-':>7}")
            all_results.append({
                'pos_thresh': pos_thresh,
                'entry_prob': entry_prob,
                'n_trades': 0,
                'win_0': np.nan,
                'win_3': np.nan,
                'win_target': np.nan,
                'avg_win': np.nan,
                'avg_loss': np.nan,
                'pl_ratio': np.nan,
                'ev': np.nan,
                'pos_rate': pos_rate_test,
            })
            continue

        actual = signals['fwd_ret']
        win_0 = (actual > 0).mean()
        win_3 = (actual > 0.03).mean()
        win_target = (actual >= pos_thresh).mean()

        winners = actual[actual > 0]
        losers = actual[actual <= 0]
        avg_win = winners.mean() if len(winners) > 0 else 0
        avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.001
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        ev = win_0 * avg_win - (1 - win_0) * avg_loss

        print(f"  {entry_prob:>7.0%} | {n_trades:>6,} | {win_0:>6.1%} | {win_3:>6.1%} | {win_target:>7.1%} | {avg_win:>6.2%} | {avg_loss:>6.2%} | {pl_ratio:>5.2f} | {ev:>6.3f}")

        all_results.append({
            'pos_thresh': pos_thresh,
            'entry_prob': entry_prob,
            'n_trades': n_trades,
            'win_0': win_0,
            'win_3': win_3,
            'win_target': win_target,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'pl_ratio': pl_ratio,
            'ev': ev,
            'pos_rate': pos_rate_test,
        })

# ============================================================
# 5. 综合对比
# ============================================================
print("\n" + "=" * 70)
print("综合对比矩阵")
print("=" * 70)

df_r = pd.DataFrame(all_results)

header = f"{'正类阈值':>8} | {'入场概率':>8} | {'交易数':>6} | {'实际>0%':>7} | {'实际>3%':>7} | {'达标胜率':>8} | {'盈亏比':>6} | {'期望值':>7}"
print(header)
print("-" * len(header))

for _, r in df_r.iterrows():
    if r['n_trades'] == 0:
        print(f"{r['pos_thresh']:>7.0%} | {r['entry_prob']:>7.0%} | {'0':>6} | {'-':>7} | {'-':>7} | {'-':>8} | {'-':>6} | {'-':>7}")
    else:
        print(f"{r['pos_thresh']:>7.0%} | {r['entry_prob']:>7.0%} | {r['n_trades']:>6,} | {r['win_0']:>6.1%} | {r['win_3']:>6.1%} | {r['win_target']:>7.1%} | {r['pl_ratio']:>5.2f} | {r['ev']:>6.3f}")

# 找最优组合
valid_r = df_r[df_r['n_trades'] > 100].copy()
if len(valid_r) > 0:
    best_ev = valid_r.loc[valid_r['ev'].idxmax()]
    best_pl = valid_r.loc[valid_r['pl_ratio'].idxmax()]
    best_win = valid_r.loc[valid_r['win_target'].idxmax()]

    print(f"\n--- 最优组合（交易数>100）---")
    print(f"最高期望值: 正类{best_ev['pos_thresh']:.0%} + 入场{best_ev['entry_prob']:.0%} (EV={best_ev['ev']:.4f}, 胜率{best_ev['win_0']:.1%}, 盈亏比{best_ev['pl_ratio']:.2f})")
    print(f"最高盈亏比: 正类{best_pl['pos_thresh']:.0%} + 入场{best_pl['entry_prob']:.0%} (PL={best_pl['pl_ratio']:.2f}, EV={best_pl['ev']:.4f}, 胜率{best_pl['win_0']:.1%})")
    print(f"最高达标率: 正类{best_win['pos_thresh']:.0%} + 入场{best_win['entry_prob']:.0%} (达标{best_win['win_target']:.1%}, EV={best_win['ev']:.4f}, 盈亏比{best_win['pl_ratio']:.2f})")
else:
    print("\n没有交易数>100的有效组合")

# 保存
df_r.to_csv('/home/hermes/.hermes/openclaw-archive/analysis/v4_classification_matrix.csv', index=False)
print(f"\n结果已保存到 analysis/v4_classification_matrix.csv")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
