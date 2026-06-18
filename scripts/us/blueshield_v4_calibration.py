#!/usr/bin/env python3
"""
蓝盾V4 概率校准实验
Platt Scaling + Isotonic Regression
让模型输出更分散，真正有区分度
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据 & 特征（简化版，复用之前逻辑）
# ============================================================
print("=" * 70)
print("蓝盾V4 概率校准实验")
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
# 2. 特征计算（Top30）
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

df['ret_5d'] = df.groupby('code')['close'].pct_change(5)
df['ret_20d'] = df.groupby('code')['close'].pct_change(20)
df['vol_ratio'] = df.groupby('code')['volume'].transform(lambda x: x / x.rolling(20).mean())
df['bias_20d'] = df.groupby('code')['close'].transform(lambda x: (x - x.rolling(20).mean()) / x.rolling(20).mean())

for col in ['ret_5d', 'ret_20d', 'vol_ratio', 'bias_20d']:
    features[f'rank_{col}'] = df.groupby('date')[col].rank(pct=True)

df['daily_ret'] = df.groupby('code')['close'].pct_change()
for w in [5, 20]:
    df[f'ret_{w}d_raw'] = df.groupby('code')['close'].pct_change(w)
    market_avg = df.groupby('date')[f'ret_{w}d_raw'].transform('mean')
    market_std = df.groupby('date')[f'ret_{w}d_raw'].transform('std')
    features[f'zscore_ret_{w}d'] = (df[f'ret_{w}d_raw'] - market_avg) / (market_std + 1e-10)

for w in [5, 20, 60]:
    features[f'volatility_{w}d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(w).std())

vol20 = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(20).std())
vol60 = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(60).std())
features['vol_change'] = vol20 / (vol60 + 1e-10)

df['vol_20d'] = vol20
features['rank_volatility'] = df.groupby('date')['vol_20d'].rank(pct=True)

price_up = df.groupby('code')['close'].pct_change(5) > 0
vol_down = df.groupby('code')['volume'].transform(lambda x: x.rolling(5).mean()) < df.groupby('code')['volume'].transform(lambda x: x.rolling(20).mean())
features['price_vol_diverge'] = (price_up & vol_down).astype(float)

price_down = df.groupby('code')['close'].pct_change(5) < 0
vol_up = df.groupby('code')['volume'].transform(lambda x: x.rolling(5).mean()) > df.groupby('code')['volume'].transform(lambda x: x.rolling(20).mean())
features['panic_signal'] = (price_down & vol_up).astype(float)

obv = df.groupby('code').apply(lambda x: (np.sign(x['close'].diff()) * x['volume']).cumsum()).reset_index(level=0, drop=True)
features['obv_slope'] = obv.groupby(df['code']).transform(lambda x: x.rolling(20).apply(
    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == 20 else np.nan, raw=True
))

features['price_vol_corr'] = df.groupby('code').apply(
    lambda x: x['close'].pct_change().rolling(20).corr(x['volume'].pct_change())
).reset_index(level=0, drop=True)

feat_df = pd.DataFrame(features, index=df.index)
feat_df['close'] = df['close']
feat_df['volume'] = df['volume']
feat_df['code'] = df['code']
feat_df['date'] = df['date']

exclude = {'close', 'volume', 'code', 'date'}
all_feature_cols = [c for c in feat_df.columns if c not in exclude]
feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

print(f"特征数: {len(all_feature_cols)}, 耗时: {time.time()-t_start:.1f}s")

# ============================================================
# 3. 数据划分
# ============================================================
WINDOW = 5
POS_THRESH = 0.03

feat_df['fwd_ret'] = feat_df.groupby('code')['close'].pct_change(WINDOW).shift(-WINDOW)
valid = feat_df.dropna(subset=all_feature_cols + ['fwd_ret']).copy()
valid = valid[valid['fwd_ret'].between(-0.5, 0.5)]

valid = valid.sort_values('date')
train_end = valid['date'].quantile(0.6)
val_end = valid['date'].quantile(0.8)

train = valid[valid['date'] <= train_end].copy()
val = valid[(valid['date'] > train_end) & (valid['date'] <= val_end)].copy()
test = valid[valid['date'] > val_end].copy()

print(f"\n训练: {len(train):,}  验证: {len(val):,}  测试: {len(test):,}")

# Top30特征
top30_features = ['vol_60d', 'volatility_60d', 'vol_20d', 'high_low_range', 'vol_5d',
                  'volatility_20d', 'rsi_28', 'rank_volatility', 'vol_change', 'ret_60d',
                  'ret_5d', 'vol_10d', 'price_position_60d', 'rank_bias_20d', 'bias_5d',
                  'bias_10d', 'zscore_ret_20d', 'bias_60d', 'ret_20d', 'vol_ratio_5d',
                  'vol_ratio_20d', 'ret_10d', 'trend_strength_10d', 'trend_strength_20d',
                  'zscore_ret_5d', 'rank_ret_20d', 'rsi_14', 'macd_hist', 'bias_20d',
                  'close_open_range']

X_train = train[top30_features].values
X_val = val[top30_features].values
X_test = test[top30_features].values

y_train = (train['fwd_ret'].values >= POS_THRESH).astype(int)
y_val = (val['fwd_ret'].values >= POS_THRESH).astype(int)

n_pos = y_train.sum()
scale = (len(y_train) - n_pos) / n_pos

# ============================================================
# 4. 训练模型 + 校准
# ============================================================
print(f"\n{'=' * 70}")
print("训练模型 + 概率校准")
print(f"{'=' * 70}")

# 训练基础LGB模型
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

base_model = lgb.train(
    {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
     'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
     'bagging_fraction': 0.8, 'bagging_freq': 5, 'scale_pos_weight': scale,
     'verbose': -1, 'seed': 42},
    train_data, num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

print(f"基础模型最佳轮数: {base_model.best_iteration}")

# 获取验证集预测
val_pred = base_model.predict(X_val)
test_pred_raw = base_model.predict(X_test)

# 方法1: Platt Scaling (Sigmoid校准)
print("\n--- Platt Scaling ---")
from sklearn.linear_model import LogisticRegression

# 用验证集拟合Platt Scaling
lr = LogisticRegression(random_state=42)
lr.fit(val_pred.reshape(-1, 1), y_val)

# 校准测试集
test_pred_platt = lr.predict_proba(test_pred_raw.reshape(-1, 1))[:, 1]

# 方法2: Isotonic Regression
print("--- Isotonic Regression ---")
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(val_pred, y_val)
test_pred_iso = iso.transform(test_pred_raw)

# 方法3: 分位数校准（自定义）
print("--- 分位数校准 ---")
# 用验证集的分位数来校准
val_quantiles = np.percentile(val_pred, np.arange(0, 101, 5))
test_pred_quantile = np.searchsorted(val_quantiles, test_pred_raw) / 20.0  # 归一化到0-1

# ============================================================
# 5. 校准效果对比
# ============================================================
print(f"\n{'=' * 70}")
print("校准效果对比")
print(f"{'=' * 70}")

test = test.copy()
test['actual'] = test['fwd_ret'].values
test['pred_raw'] = test_pred_raw
test['pred_platt'] = test_pred_platt
test['pred_iso'] = test_pred_iso
test['pred_quantile'] = test_pred_quantile

methods = [
    ("原始输出", test['pred_raw']),
    ("Platt Scaling", test['pred_platt']),
    ("Isotonic", test['pred_iso']),
    ("分位数校准", test['pred_quantile']),
]

print(f"\n概率分布统计:")
print(f"{'方法':<20} | {'均值':>6} | {'标准差':>6} | {'最小':>6} | {'25%':>6} | {'50%':>6} | {'75%':>6} | {'最大':>6}")
print("-" * 100)

for name, pred in methods:
    print(f"{name:<20} | {pred.mean():>5.2f} | {pred.std():>5.2f} | {pred.min():>5.2f} | {pred.quantile(0.25):>5.2f} | {pred.quantile(0.50):>5.2f} | {pred.quantile(0.75):>5.2f} | {pred.max():>5.2f}")

# ============================================================
# 6. 校准后各阈值表现
# ============================================================
print(f"\n{'=' * 70}")
print("校准后各阈值表现")
print(f"{'=' * 70}")

entry_probs = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

for name, pred in methods:
    print(f"\n--- {name} ---")
    print(f"  {'阈值':>6} | {'交易数':>6} | {'实际>0%':>7} | {'实际>3%':>7} | {'盈亏比':>6} | {'期望值':>7}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")
    
    for ep in entry_probs:
        signals = test[pred >= ep]
        n = len(signals)
        
        if n < 10:
            print(f"  {ep:>5.0%} | {n:>6} | {'-':>7} | {'-':>7} | {'-':>6} | {'-':>7}")
            continue
        
        actual = signals['actual']
        win_0 = (actual > 0).mean()
        win_3 = (actual > 0.03).mean()
        
        winners = actual[actual > 0]
        losers = actual[actual <= 0]
        avg_win = winners.mean() if len(winners) > 0 else 0
        avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.001
        pl = avg_win / avg_loss if avg_loss > 0 else float('inf')
        ev = win_0 * avg_win - (1 - win_0) * avg_loss
        
        print(f"  {ep:>5.0%} | {n:>6,} | {win_0:>6.1%} | {win_3:>6.1%} | {pl:>5.2f} | {ev:>6.4f}")

# ============================================================
# 7. 找最优校准方案
# ============================================================
print(f"\n{'=' * 70}")
print("最优校准方案搜索")
print(f"{'=' * 70}")

best_combos = []

for name, pred in methods:
    for ep in entry_probs:
        signals = test[pred >= ep]
        n = len(signals)
        
        if n < 50:
            continue
        
        actual = signals['actual']
        win_0 = (actual > 0).mean()
        win_3 = (actual > 0.03).mean()
        
        winners = actual[actual > 0]
        losers = actual[actual <= 0]
        avg_win = winners.mean() if len(winners) > 0 else 0
        avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.001
        pl = avg_win / avg_loss if avg_loss > 0 else float('inf')
        ev = win_0 * avg_win - (1 - win_0) * avg_loss
        
        best_combos.append({
            'method': name,
            'threshold': ep,
            'n_trades': n,
            'win_0': win_0,
            'win_3': win_3,
            'pl_ratio': pl,
            'ev': ev,
        })

if best_combos:
    df_combos = pd.DataFrame(best_combos)
    df_combos = df_combos.sort_values('win_0', ascending=False)
    
    print(f"\nTop 10（按胜率排序，交易数≥50）:")
    print(f"{'方法':<20} | {'阈值':>5} | {'交易数':>6} | {'胜率':>6} | {'>3%胜率':>7} | {'盈亏比':>6} | {'期望值':>6}")
    print("-" * 85)
    
    for _, row in df_combos.head(10).iterrows():
        print(f"{row['method']:<20} | {row['threshold']:>4.0%} | {row['n_trades']:>6,} | {row['win_0']:>5.1%} | {row['win_3']:>6.1%} | {row['pl_ratio']:>5.2f} | {row['ev']:>5.4f}")
    
    # 70%目标
    target_70 = df_combos[df_combos['win_0'] >= 0.70]
    if len(target_70) > 0:
        print(f"\n🎯 达到70%胜率:")
        for _, row in target_70.head(5).iterrows():
            print(f"  {row['method']} + 阈值{row['threshold']:.0%}: 胜率{row['win_0']:.1%}, {row['n_trades']:.0f}笔, PL{row['pl_ratio']:.2f}")
    else:
        print(f"\n❌ 没有达到70%胜率")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
