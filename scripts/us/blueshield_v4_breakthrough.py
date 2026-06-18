#!/usr/bin/env python3
"""
蓝盾V4 三方向突破实验
1. 特征选择（去噪）
2. 不同正类阈值（3%/8%）
3. Stacking（第二层模型）
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 70)
print("蓝盾V4 三方向突破实验")
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
# 2. 全部特征计算
# ============================================================
print("\n计算全部特征...")
t_start = time.time()

features = {}

# 基础特征
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

# 截面特征
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

# 波动率特征
for w in [5, 20, 60]:
    vol = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(w).std())
    features[f'volatility_{w}d'] = vol

vol20 = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(20).std())
vol60 = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(60).std())
features['vol_change'] = vol20 / (vol60 + 1e-10)

df['vol_20d'] = vol20
features['rank_volatility'] = df.groupby('date')['vol_20d'].rank(pct=True)

# 量价背离
price_up = df.groupby('code')['close'].pct_change(5) > 0
vol_down = df.groupby('code')['volume'].transform(lambda x: x.rolling(5).mean()) < df.groupby('code')['volume'].transform(lambda x: x.rolling(20).mean())
features['price_vol_diverge'] = (price_up & vol_down).astype(float)

price_down = df.groupby('code')['close'].pct_change(5) < 0
vol_up = df.groupby('code')['volume'].transform(lambda x: x.rolling(5).mean()) > df.groupby('code')['volume'].transform(lambda x: x.rolling(20).mean())
features['panic_signal'] = (price_down & vol_up).astype(float)

# OBV斜率
obv = df.groupby('code').apply(lambda x: (np.sign(x['close'].diff()) * x['volume']).cumsum()).reset_index(level=0, drop=True)
features['obv_slope'] = obv.groupby(df['code']).transform(lambda x: x.rolling(20).apply(
    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == 20 else np.nan, raw=True
))

features['price_vol_corr'] = df.groupby('code').apply(
    lambda x: x['close'].pct_change().rolling(20).corr(x['volume'].pct_change())
).reset_index(level=0, drop=True)

# 组合
feat_df = pd.DataFrame(features, index=df.index)
feat_df['close'] = df['close']
feat_df['volume'] = df['volume']
feat_df['code'] = df['code']
feat_df['date'] = df['date']

exclude = {'close', 'volume', 'code', 'date'}
all_feature_cols = [c for c in feat_df.columns if c not in exclude]
feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

print(f"总特征数: {len(all_feature_cols)}, 耗时: {time.time()-t_start:.1f}s")

# ============================================================
# 3. 数据划分
# ============================================================
WINDOW = 5

feat_df['fwd_ret'] = feat_df.groupby('code')['close'].pct_change(WINDOW).shift(-WINDOW)
valid = feat_df.dropna(subset=all_feature_cols + ['fwd_ret']).copy()
valid = valid[valid['fwd_ret'].between(-0.5, 0.5)]

valid = valid.sort_values('date')
train_end = valid['date'].quantile(0.6)
val_end = valid['date'].quantile(0.8)

train = valid[valid['date'] <= train_end].copy()
val = valid[(valid['date'] > train_end) & (valid['date'] <= val_end)].copy()
test = valid[valid['date'] > val_end].copy()

print(f"训练: {len(train):,}  验证: {len(val):,}  测试: {len(test):,}")

# ============================================================
# 4. 方向一：特征选择（用LGB特征重要性）
# ============================================================
print(f"\n{'=' * 70}")
print("方向一：特征选择")
print(f"{'=' * 70}")

# 先用5%正类训练，获取特征重要性
POS_THRESH = 0.05
y_train_5 = (train['fwd_ret'].values >= POS_THRESH).astype(int)
y_val_5 = (val['fwd_ret'].values >= POS_THRESH).astype(int)
X_train_all = train[all_feature_cols].values
X_val_all = val[all_feature_cols].values
X_test_all = test[all_feature_cols].values

n_pos = y_train_5.sum()
scale = (len(y_train_5) - n_pos) / n_pos

train_data = lgb.Dataset(X_train_all, label=y_train_5)
val_data = lgb.Dataset(X_val_all, label=y_val_5, reference=train_data)

lgb_model = lgb.train(
    {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
     'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
     'bagging_fraction': 0.8, 'bagging_freq': 5, 'scale_pos_weight': scale,
     'verbose': -1, 'seed': 42},
    train_data, num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

# 特征重要性
importance = pd.DataFrame({
    'feature': all_feature_cols,
    'importance': lgb_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("\nTop 20 特征重要性:")
for _, row in importance.head(20).iterrows():
    print(f"  {row['feature']:<25} {row['importance']:>10.0f}")

# 选Top N特征
top_20 = importance.head(20)['feature'].tolist()
top_30 = importance.head(30)['feature'].tolist()
top_40 = importance.head(40)['feature'].tolist()

feat_sets = {
    'Top20': top_20,
    'Top30': top_30,
    'Top40': top_40,
    '全部58': all_feature_cols,
}

# ============================================================
# 5. 方向二：不同正类阈值
# ============================================================
print(f"\n{'=' * 70}")
print("方向二：不同正类阈值")
print(f"{'=' * 70}")

pos_thresholds = [0.03, 0.05, 0.08]

# ============================================================
# 6. 方向三：Stacking
# ============================================================
print(f"\n{'=' * 70}")
print("方向三：Stacking")
print(f"{'=' * 70}")

# Level 1: LGB + XGB
# Level 2: Logistic Regression

# 用5%正类，全部特征训练Level 1
dtrain_xgb = xgb.DMatrix(X_train_all, label=y_train_5)
dval_xgb = xgb.DMatrix(X_val_all, label=y_val_5)
dtest_xgb = xgb.DMatrix(X_test_all)

xgb_model = xgb.train(
    {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 6,
     'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
     'scale_pos_weight': scale, 'seed': 42, 'verbosity': 0},
    dtrain_xgb, num_boost_round=1000,
    evals=[(dval_xgb, 'eval')],
    early_stopping_rounds=100, verbose_eval=False
)

# Level 1 预测
lgb_val_pred = lgb_model.predict(X_val_all)
xgb_val_pred = xgb_model.predict(dval_xgb)
lgb_test_pred = lgb_model.predict(X_test_all)
xgb_test_pred = xgb_model.predict(dtest_xgb)

# Level 2: Logistic Regression
X_val_stack = np.column_stack([lgb_val_pred, xgb_val_pred])
X_test_stack = np.column_stack([lgb_test_pred, xgb_test_pred])

lr = LogisticRegression(random_state=42)
lr.fit(X_val_stack, y_val_5)
stack_test_pred = lr.predict_proba(X_test_stack)[:, 1]

print(f"Stacking权重: LGB={lr.coef_[0][0]:.3f}, XGB={lr.coef_[0][1]:.3f}")

# ============================================================
# 7. 综合实验矩阵
# ============================================================
print(f"\n{'=' * 70}")
print("综合实验矩阵")
print(f"{'=' * 70}")

entry_probs = [0.70, 0.75, 0.80, 0.85, 0.90]

all_results = []

# --- 实验A：特征选择 × 正类阈值 ---
print("\n--- 实验A：特征选择 × 正类阈值 ---")

for pos_thresh in pos_thresholds:
    y_train_pt = (train['fwd_ret'].values >= pos_thresh).astype(int)
    y_val_pt = (val['fwd_ret'].values >= pos_thresh).astype(int)
    
    n_pos_pt = y_train_pt.sum()
    scale_pt = (len(y_train_pt) - n_pos_pt) / n_pos_pt if n_pos_pt > 0 else 1
    
    for feat_name, feat_list in feat_sets.items():
        X_tr = train[feat_list].values
        X_v = val[feat_list].values
        X_te = test[feat_list].values
        
        # LGB
        td = lgb.Dataset(X_tr, label=y_train_pt)
        vd = lgb.Dataset(X_v, label=y_val_pt, reference=td)
        
        m = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
             'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
             'bagging_fraction': 0.8, 'bagging_freq': 5, 'scale_pos_weight': scale_pt,
             'verbose': -1, 'seed': 42},
            td, num_boost_round=1000,
            valid_sets=[vd],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        prob = m.predict(X_te)
        test_exp = test.copy()
        test_exp['actual'] = test_exp['fwd_ret'].values
        test_exp['prob'] = prob
        
        for ep in entry_probs:
            sig = test_exp[test_exp['prob'] >= ep]
            n = len(sig)
            if n < 10:
                continue
            
            actual = sig['actual']
            win_0 = (actual > 0).mean()
            win_3 = (actual > 0.03).mean()
            winners = actual[actual > 0]
            losers = actual[actual <= 0]
            avg_win = winners.mean() if len(winners) > 0 else 0
            avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.001
            pl = avg_win / avg_loss if avg_loss > 0 else float('inf')
            ev = win_0 * avg_win - (1 - win_0) * avg_loss
            
            all_results.append({
                'experiment': f"正类{pos_thresh:.0%}+{feat_name}",
                'entry_prob': ep,
                'n_trades': n,
                'win_0': win_0,
                'win_3': win_3,
                'pl_ratio': pl,
                'ev': ev,
            })

# --- 实验B：Stacking ---
print("\n--- 实验B：Stacking ---")

test_stack = test.copy()
test_stack['actual'] = test_stack['fwd_ret'].values
test_stack['prob'] = stack_test_pred

for ep in entry_probs:
    sig = test_stack[test_stack['prob'] >= ep]
    n = len(sig)
    if n < 10:
        print(f"  Stacking 入场{ep:.0%}: {n}笔 (不足)")
        continue
    
    actual = sig['actual']
    win_0 = (actual > 0).mean()
    win_3 = (actual > 0.03).mean()
    winners = actual[actual > 0]
    losers = actual[actual <= 0]
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.001
    pl = avg_win / avg_loss if avg_loss > 0 else float('inf')
    ev = win_0 * avg_win - (1 - win_0) * avg_loss
    
    print(f"  Stacking 入场{ep:.0%}: {n:,}笔, 胜率{win_0:.1%}, >3%{win_3:.1%}, PL{pl:.2f}, EV{ev:.4f}")
    
    all_results.append({
        'experiment': 'Stacking',
        'entry_prob': ep,
        'n_trades': n,
        'win_0': win_0,
        'win_3': win_3,
        'pl_ratio': pl,
        'ev': ev,
    })

# ============================================================
# 8. 汇总
# ============================================================
print(f"\n{'=' * 70}")
print("汇总（交易数≥50，按胜率排序）")
print(f"{'=' * 70}")

df_r = pd.DataFrame(all_results)
df_r = df_r[df_r['n_trades'] >= 50]
df_r = df_r.sort_values('win_0', ascending=False)

print(f"\n{'实验':<25} | {'入场':>5} | {'交易数':>6} | {'胜率':>6} | {'>3%胜率':>7} | {'盈亏比':>6} | {'期望值':>6}")
print("-" * 90)

for _, row in df_r.head(20).iterrows():
    print(f"{row['experiment']:<25} | {row['entry_prob']:>4.0%} | {row['n_trades']:>6,} | {row['win_0']:>5.1%} | {row['win_3']:>6.1%} | {row['pl_ratio']:>5.2f} | {row['ev']:>5.3f}")

# 70%目标
target_70 = df_r[df_r['win_0'] >= 0.70]
if len(target_70) > 0:
    print(f"\n🎯 达到70%胜率的组合:")
    for _, row in target_70.head(10).iterrows():
        print(f"  {row['experiment']} + 入场{row['entry_prob']:.0%}: 胜率{row['win_0']:.1%}, {row['n_trades']:.0f}笔, PL{row['pl_ratio']:.2f}")
else:
    print(f"\n❌ 没有组合达到70%胜率")
    if len(df_r) > 0:
        best = df_r.iloc[0]
        print(f"最接近: {best['experiment']} + 入场{best['entry_prob']:.0%}: 胜率{best['win_0']:.1%}")

df_r.to_csv('/home/hermes/.hermes/openclaw-archive/analysis/v4_breakthrough_results.csv', index=False)
print(f"\n结果已保存")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
