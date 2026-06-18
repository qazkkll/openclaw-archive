#!/usr/bin/env python3
"""
蓝盾V4 新特征方向实验
截面特征 + 波动率regime + 量价背离
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 70)
print("蓝盾V4 新特征方向实验")
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
# 2. 基础特征（复用）
# ============================================================
print("\n计算基础特征...")
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

print(f"基础特征: {len(features)} 个")

# ============================================================
# 3. 新特征：截面特征（最有希望）
# ============================================================
print("\n计算截面特征...")

# 每日截面排名（归一化到0-1）
ret5 = df.groupby('code')['close'].pct_change(5)
ret20 = df.groupby('code')['close'].pct_change(20)

# 每日排名：这只股票在当日所有股票中的排名
df['ret_5d'] = ret5
df['ret_20d'] = ret20
df['vol_ratio'] = df.groupby('code')['volume'].transform(lambda x: x / x.rolling(20).mean())
df['bias_20d'] = df.groupby('code')['close'].transform(lambda x: (x - x.rolling(20).mean()) / x.rolling(20).mean())

# 截面排名（百分位）
for col in ['ret_5d', 'ret_20d', 'vol_ratio', 'bias_20d']:
    features[f'rank_{col}'] = df.groupby('date')[col].rank(pct=True)

# 截面均值和标准差（市场平均）
df['daily_ret'] = df.groupby('code')['close'].pct_change()
for w in [5, 20]:
    df[f'ret_{w}d_raw'] = df.groupby('code')['close'].pct_change(w)
    market_avg = df.groupby('date')[f'ret_{w}d_raw'].transform('mean')
    market_std = df.groupby('date')[f'ret_{w}d_raw'].transform('std')
    features[f'zscore_ret_{w}d'] = (df[f'ret_{w}d_raw'] - market_avg) / (market_std + 1e-10)

# 行业相对强度（用GICS行业，但数据中没有，用代码前缀近似）
# 简化：用股票自身的排名变化作为相对强度
features['rank_change_5d'] = df.groupby('code')['rank_ret_5d'].diff(5) if 'rank_ret_5d' in df.columns else 0

print(f"截面特征: 8 个")

# ============================================================
# 4. 新特征：波动率regime
# ============================================================
print("\n计算波动率regime特征...")

# 个人波动率变化
for w in [5, 20, 60]:
    vol = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(w).std())
    features[f'volatility_{w}d'] = vol

# 波动率变化率
vol20 = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(20).std())
vol60 = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(60).std())
features['vol_change'] = vol20 / (vol60 + 1e-10)  # 短期/长期波动率比

# 波动率截面排名
df['vol_20d'] = vol20
features['rank_volatility'] = df.groupby('date')['vol_20d'].rank(pct=True)

# 市场波动率（等权平均）
market_vol = df.groupby('date')['daily_ret'].std()
df['date_idx'] = df['date']
market_vol_df = market_vol.reset_index()
market_vol_df.columns = ['date', 'market_vol']
df = df.merge(market_vol_df, on='date', how='left')
features['market_vol_zscore'] = (df['market_vol'] - df.groupby('code')['market_vol'].transform(lambda x: x.rolling(60).mean())) / (df.groupby('code')['market_vol'].transform(lambda x: x.rolling(60).std()) + 1e-10)

print(f"波动率特征: 6 个")

# ============================================================
# 5. 新特征：量价背离
# ============================================================
print("\n计算量价背离特征...")

# 价涨量缩（看跌信号）
price_up = df.groupby('code')['close'].pct_change(5) > 0
vol_down = df.groupby('code')['volume'].transform(lambda x: x.rolling(5).mean()) < df.groupby('code')['volume'].transform(lambda x: x.rolling(20).mean())
features['price_vol_diverge'] = (price_up & vol_down).astype(float)

# 价跌量增（恐慌信号）
price_down = df.groupby('code')['close'].pct_change(5) < 0
vol_up = df.groupby('code')['volume'].transform(lambda x: x.rolling(5).mean()) > df.groupby('code')['volume'].transform(lambda x: x.rolling(20).mean())
features['panic_signal'] = (price_down & vol_up).astype(float)

# OBV（能量潮）
obv = df.groupby('code').apply(lambda x: (np.sign(x['close'].diff()) * x['volume']).cumsum()).reset_index(level=0, drop=True)
features['obv_slope'] = obv.groupby(df['code']).transform(lambda x: x.rolling(20).apply(
    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == 20 else np.nan, raw=True
))

# 量价相关性
features['price_vol_corr'] = df.groupby('code').apply(
    lambda x: x['close'].pct_change().rolling(20).corr(x['volume'].pct_change())
).reset_index(level=0, drop=True)

print(f"量价背离特征: 4 个")

# ============================================================
# 6. 组合所有特征
# ============================================================
feat_df = pd.DataFrame(features, index=df.index)
feat_df['close'] = df['close']
feat_df['volume'] = df['volume']
feat_df['code'] = df['code']
feat_df['date'] = df['date']

exclude = {'close', 'volume', 'code', 'date'}
feature_cols = [c for c in feat_df.columns if c not in exclude]
feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

print(f"\n总特征数: {len(feature_cols)}, 耗时: {time.time()-t_start:.1f}s")

# ============================================================
# 7. 实验矩阵：新特征 vs 旧特征
# ============================================================
print(f"\n{'=' * 70}")
print("实验矩阵：新特征 vs 旧特征")
print(f"{'=' * 70}")

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

print(f"训练: {len(train):,}  验证: {len(val):,}  测试: {len(test):,}")

# 特征组
old_features = [f for f in feature_cols if not any(x in f for x in ['rank_', 'zscore_', 'volatility_', 'vol_change', 'market_vol', 'price_vol_diverge', 'panic_signal', 'obv_', 'price_vol_corr'])]
new_features = [f for f in feature_cols if any(x in f for x in ['rank_', 'zscore_', 'volatility_', 'vol_change', 'market_vol', 'price_vol_diverge', 'panic_signal', 'obv_', 'price_vol_corr'])]

print(f"\n旧特征: {len(old_features)} 个")
print(f"新特征: {len(new_features)} 个")
print(f"新特征列表: {new_features}")

# 测试不同特征组合
experiments = [
    ("旧特征40个", old_features),
    ("新特征18个", new_features),
    ("全部特征58个", feature_cols),
]

entry_probs = [0.70, 0.80, 0.85, 0.90]

all_results = []

for exp_name, feat_list in experiments:
    print(f"\n--- {exp_name} ---")
    
    X_train = train[feat_list].values
    X_val = val[feat_list].values
    X_test = test[feat_list].values
    
    y_train = (train['fwd_ret'].values >= POS_THRESH).astype(int)
    y_val = (val['fwd_ret'].values >= POS_THRESH).astype(int)
    
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale = n_neg / n_pos
    
    # LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    lgb_params = {
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
    
    lgb_model = lgb.train(
        lgb_params, train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    # XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest_xgb = xgb.DMatrix(X_test)
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale,
        'seed': 42,
        'verbosity': 0
    }
    
    xgb_model = xgb.train(
        xgb_params, dtrain,
        num_boost_round=1000,
        evals=[(dval, 'eval')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    
    # 预测
    prob_lgb = lgb_model.predict(X_test)
    prob_xgb = xgb_model.predict(dtest_xgb)
    prob_avg = (prob_lgb + prob_xgb) / 2
    
    test_exp = test.copy()
    test_exp['actual'] = test_exp['fwd_ret'].values
    test_exp['prob_avg'] = prob_avg
    
    print(f"  LGB最佳轮数: {lgb_model.best_iteration}, XGB最佳轮数: {xgb_model.best_iteration}")
    
    for entry_prob in entry_probs:
        signals = test_exp[test_exp['prob_avg'] >= entry_prob]
        n_trades = len(signals)
        
        if n_trades < 10:
            print(f"  入场{entry_prob:.0%}: {n_trades}笔 (不足)")
            continue
        
        actual = signals['actual']
        win_0 = (actual > 0).mean()
        win_3 = (actual > 0.03).mean()
        
        winners = actual[actual > 0]
        losers = actual[actual <= 0]
        avg_win = winners.mean() if len(winners) > 0 else 0
        avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.001
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        ev = win_0 * avg_win - (1 - win_0) * avg_loss
        
        print(f"  入场{entry_prob:.0%}: {n_trades:,}笔, 胜率{win_0:.1%}, >3%{win_3:.1%}, PL{pl_ratio:.2f}, EV{ev:.4f}")
        
        all_results.append({
            'experiment': exp_name,
            'entry_prob': entry_prob,
            'n_trades': n_trades,
            'win_0': win_0,
            'win_3': win_3,
            'pl_ratio': pl_ratio,
            'ev': ev,
        })

# ============================================================
# 8. 汇总对比
# ============================================================
print(f"\n{'=' * 70}")
print("汇总对比")
print(f"{'=' * 70}")

df_r = pd.DataFrame(all_results)
print(f"\n{'实验':<15} | {'入场':>5} | {'交易数':>6} | {'胜率':>6} | {'>3%胜率':>7} | {'盈亏比':>6} | {'期望值':>6}")
print("-" * 75)

for _, row in df_r.iterrows():
    print(f"{row['experiment']:<15} | {row['entry_prob']:>4.0%} | {row['n_trades']:>6,} | {row['win_0']:>5.1%} | {row['win_3']:>6.1%} | {row['pl_ratio']:>5.2f} | {row['ev']:>5.3f}")

# 找最优
df_r_valid = df_r[df_r['n_trades'] >= 50]
if len(df_r_valid) > 0:
    best = df_r_valid.loc[df_r_valid['win_0'].idxmax()]
    print(f"\n🏆 最优: {best['experiment']} + 入场{best['entry_prob']:.0%}")
    print(f"   胜率: {best['win_0']:.1%}, 交易数: {best['n_trades']:.0f}, 盈亏比: {best['pl_ratio']:.2f}")

# 保存
df_r.to_csv('/home/hermes/.hermes/openclaw-archive/analysis/v4_new_features_results.csv', index=False)
print(f"\n结果已保存")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
