#!/usr/bin/env python3
"""
蓝盾V4 分类模型 + 市场环境过滤器（简化版）
按日期过滤，不merge市场特征到每行
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
print("蓝盾V4 分类模型 + 市场环境过滤器")
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
# 2. 合成市场指数
# ============================================================
print("\n计算合成市场指数...")
df['daily_ret'] = df.groupby('code')['close'].pct_change()
ret_pivot = df.pivot_table(index='date', columns='code', values='daily_ret')
market_daily = ret_pivot.mean(axis=1, skipna=True)

# 市场regime（按日期）
mkt = pd.DataFrame({'date': ret_pivot.index, 'mkt_ret': market_daily.values})
mkt['mkt_ret_5d'] = mkt['mkt_ret'].rolling(5).sum()
mkt['mkt_ret_10d'] = mkt['mkt_ret'].rolling(10).sum()
mkt['mkt_ret_20d'] = mkt['mkt_ret'].rolling(20).sum()
mkt['vix_proxy'] = mkt['mkt_ret'].rolling(20).std() * np.sqrt(252) * 100

# 定义regime
def get_regime(row):
    vix = row['vix_proxy']
    ret20 = row['mkt_ret_20d']
    if pd.isna(vix) or pd.isna(ret20):
        return 'unknown'
    if vix > 30 and ret20 < -0.02:
        return 'panic'
    if vix > 25 or ret20 < -0.03:
        return 'bear'
    if vix < 20 and ret20 > 0.01:
        return 'bull'
    return 'neutral'

mkt['regime'] = mkt.apply(get_regime, axis=1)

print(f"Regime分布:")
print(mkt['regime'].value_counts())
print(f"\n各regime平均VIX:")
print(mkt.groupby('regime')['vix_proxy'].agg(['mean', 'min', 'max']))

# ============================================================
# 3. 个股特征（不含市场特征）
# ============================================================
print("\n计算个股特征...")
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
# 4. 训练分类模型
# ============================================================
WINDOW = 5
POS_THRESH = 0.05

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

# 合并regime到test
test = test.merge(mkt[['date', 'regime', 'vix_proxy', 'mkt_ret_20d']], on='date', how='left')
test['regime'] = test['regime'].fillna('unknown')

print(f"模型训练完成, 最佳轮数: {model.best_iteration}")
print(f"\nTest regime分布:")
print(test['regime'].value_counts())

# ============================================================
# 5. 过滤策略对比
# ============================================================
print(f"\n{'=' * 70}")
print("过滤策略对比")
print(f"{'=' * 70}")

entry_probs = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90]

strategies = [
    ("无过滤", lambda x: True),
    ("排除panic", lambda x: x['regime'] != 'panic'),
    ("排除panic+bear", lambda x: x['regime'] not in ['panic', 'bear']),
    ("只保留bull+neutral", lambda x: x['regime'] in ['bull', 'neutral']),
    ("VIX<30", lambda x: pd.isna(x.get('vix_proxy')) or x['vix_proxy'] < 30),
    ("VIX<25", lambda x: pd.isna(x.get('vix_proxy')) or x['vix_proxy'] < 25),
    ("VIX<20", lambda x: pd.isna(x.get('vix_proxy')) or x['vix_proxy'] < 20),
    ("排除panic+bear且VIX<30", lambda x: x['regime'] not in ['panic', 'bear'] and (pd.isna(x.get('vix_proxy')) or x['vix_proxy'] < 30)),
]

all_combos = []

for strat_name, filter_fn in strategies:
    print(f"\n--- {strat_name} ---")
    print(f"  {'入场阈值':>8} | {'交易数':>6} | {'实际>0%':>7} | {'实际>3%':>7} | {'达标胜率':>8} | {'盈亏比':>6} | {'期望值':>7}")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}")
    
    for entry_prob in entry_probs:
        filtered = test[test.apply(filter_fn, axis=1)].copy()
        signals = filtered[filtered['prob'] >= entry_prob]
        n_trades = len(signals)
        
        if n_trades < 10:
            print(f"  {entry_prob:>7.0%} | {n_trades:>6} | {'-':>7} | {'-':>7} | {'-':>8} | {'-':>6} | {'-':>7}")
            continue
        
        actual = signals['actual']
        win_0 = (actual > 0).mean()
        win_3 = (actual > 0.03).mean()
        win_target = (actual >= POS_THRESH).mean()
        
        winners = actual[actual > 0]
        losers = actual[actual <= 0]
        avg_win = winners.mean() if len(winners) > 0 else 0
        avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.001
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        ev = win_0 * avg_win - (1 - win_0) * avg_loss
        
        print(f"  {entry_prob:>7.0%} | {n_trades:>6,} | {win_0:>6.1%} | {win_3:>6.1%} | {win_target:>7.1%} | {pl_ratio:>5.2f} | {ev:>6.3f}")
        
        all_combos.append({
            'strategy': strat_name,
            'entry_prob': entry_prob,
            'n_trades': n_trades,
            'win_0': win_0,
            'win_3': win_3,
            'pl_ratio': pl_ratio,
            'ev': ev,
        })

# ============================================================
# 6. 找最优组合
# ============================================================
print(f"\n{'=' * 70}")
print("最优组合搜索（交易数≥50，按胜率排序）")
print(f"{'=' * 70}")

if all_combos:
    df_combos = pd.DataFrame(all_combos)
    df_combos = df_combos[df_combos['n_trades'] >= 50]
    df_combos = df_combos.sort_values('win_0', ascending=False)
    
    print(f"\nTop 15:")
    print(f"{'策略':<25} | {'入场':>5} | {'交易数':>6} | {'胜率':>6} | {'>3%胜率':>7} | {'盈亏比':>6} | {'期望值':>6}")
    print("-" * 85)
    
    for _, row in df_combos.head(15).iterrows():
        print(f"{row['strategy']:<25} | {row['entry_prob']:>4.0%} | {row['n_trades']:>6,} | {row['win_0']:>5.1%} | {row['win_3']:>6.1%} | {row['pl_ratio']:>5.2f} | {row['ev']:>5.3f}")
    
    # 70%胜率目标
    target_70 = df_combos[df_combos['win_0'] >= 0.70]
    if len(target_70) > 0:
        print(f"\n🎯 达到70%胜率的组合:")
        for _, row in target_70.iterrows():
            print(f"  {row['strategy']} + 入场{row['entry_prob']:.0%}: 胜率{row['win_0']:.1%}, {row['n_trades']:.0f}笔, 盈亏比{row['pl_ratio']:.2f}")
    else:
        print(f"\n❌ 没有组合达到70%胜率")
        if len(df_combos) > 0:
            best = df_combos.iloc[0]
            print(f"最接近: {best['strategy']} + 入场{best['entry_prob']:.0%}: 胜率{best['win_0']:.1%}")
    
    df_combos.to_csv('/home/hermes/.hermes/openclaw-archive/analysis/v4_regime_filter_results.csv', index=False)
    print(f"\n结果已保存")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
