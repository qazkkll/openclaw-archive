# -*- coding: utf-8 -*-
"""
蓝盾V4 回测周期实验
测试不同fwd_ret窗口：1d, 3d, 5d, 10d, 20d
用V4.14配置（去市场+Adj-MSE），只比较最优窗口
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import os
import json

BASE_DIR = os.path.expanduser("~/.hermes/openclaw-archive")
DATA_PATH = os.path.join(BASE_DIR, "data/us/us_hist_sp500_10y.parquet")

print("=" * 60)
print("蓝盾V4 回测周期实验")
print("=" * 60)

# 读数据
df = pd.read_parquet(DATA_PATH)
print(f"原始数据: {len(df):,} 行, {df['sym'].nunique()} 只股票")

# 514只S&P500成分股
sp500 = json.load(open(os.path.join(BASE_DIR, "data/sp500_symbols.json")))
valid_tickers = set(sp500[:514])
df = df[df['sym'].isin(valid_tickers)].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['sym', 'date'])

# 特征列（去市场依赖）
feat_cols = [
    'ma5_ratio', 'ma20_ratio', 'ma60_ratio', 'ma120_ratio',
    'vol_20d', 'vol_5d', 'vol_ratio',
    'rsi_14', 'rsi_7',
    'macd_hist', 'macd_signal_diff',
    'bb_position', 'bb_width',
    'momentum_5d', 'momentum_10d', 'momentum_20d',
    'momentum_60d',
    'volume_ratio_20d', 'volume_ratio_5d',
    'price_range_20d', 'price_range_5d',
    'atr_14', 'adx_14',
    'obv_slope_20d', 'obv_slope_5d',
    'stoch_k', 'stoch_d',
    'williams_r',
    'cci_20',
    'mfi_14',
    'force_index_13',
    'tsi',
    'high_low_range',
    'close_position',
    'gap',
    'consecutive_up', 'consecutive_down',
    'distance_from_52w_high', 'distance_from_52w_low',
    'rolling_skew_20d', 'rolling_kurt_20d',
    'down_vol_20d', 'up_vol_20d',
    'volume_trend_20d',
    'high_vol_return_20d',
    'low_vol_return_20d',
    'max_drawdown_20d',
    'realized_vol_ratio',
]

# 测试不同fwd_ret窗口
windows = [1, 3, 5, 10, 20]
results = []

for window in windows:
    print(f"\n{'='*60}")
    print(f"测试窗口: {window}天")
    print(f"{'='*60}")
    
    # 计算fwd_ret
    df['fwd_ret'] = df.groupby('sym')['close'].transform(
        lambda x: x.shift(-window) / x - 1
    )
    
    # 计算特征
    for g, gdf in df.groupby('sym'):
        idx = gdf.index
        close = gdf['close'].values
        high = gdf['high'].values
        low = gdf['low'].values
        volume = gdf['volume'].values
        
        # MA ratios
        for n in [5, 20, 60, 120]:
            ma = pd.Series(close).rolling(n).mean().values
            df.loc[idx, f'ma{n}_ratio'] = close / np.where(ma > 0, ma, np.nan)
        
        # Volatility
        ret = pd.Series(close).pct_change()
        df.loc[idx, 'vol_20d'] = ret.rolling(20).std().values
        df.loc[idx, 'vol_5d'] = ret.rolling(5).std().values
        df.loc[idx, 'vol_ratio'] = df.loc[idx, 'vol_5d'] / df.loc[idx, 'vol_20d'].replace(0, np.nan)
        
        # RSI
        delta = ret.copy()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        for n, name in [(14, 'rsi_14'), (7, 'rsi_7')]:
            avg_gain = gain.rolling(n).mean()
            avg_loss = loss.rolling(n).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df.loc[idx, name] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = pd.Series(close).ewm(span=12).mean()
        ema26 = pd.Series(close).ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        df.loc[idx, 'macd_hist'] = (macd_line - signal_line).values
        df.loc[idx, 'macd_signal_diff'] = (macd_line / pd.Series(close)).values
        
        # Bollinger Bands
        sma20 = pd.Series(close).rolling(20).mean()
        std20 = pd.Series(close).rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        df.loc[idx, 'bb_position'] = ((pd.Series(close) - lower) / (upper - lower).replace(0, np.nan)).values
        df.loc[idx, 'bb_width'] = ((upper - lower) / sma20).values
        
        # Momentum
        for n in [5, 10, 20, 60]:
            df.loc[idx, f'momentum_{n}d'] = (pd.Series(close).pct_change(n)).values
        
        # Volume ratios
        vol_series = pd.Series(volume)
        df.loc[idx, 'volume_ratio_20d'] = (vol_series / vol_series.rolling(20).mean().replace(0, np.nan)).values
        df.loc[idx, 'volume_ratio_5d'] = (vol_series / vol_series.rolling(5).mean().replace(0, np.nan)).values
        
        # Price range
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        df.loc[idx, 'price_range_20d'] = ((high_s.rolling(20).max() - low_s.rolling(20).min()) / pd.Series(close)).values
        df.loc[idx, 'price_range_5d'] = ((high_s.rolling(5).max() - low_s.rolling(5).min()) / pd.Series(close)).values
        
        # ATR
        tr = pd.concat([high_s - low_s, (high_s - pd.Series(close).shift(1)).abs(), (low_s - pd.Series(close).shift(1)).abs()], axis=1).max(axis=1)
        df.loc[idx, 'atr_14'] = tr.rolling(14).mean().values / pd.Series(close).values
        
        # ADX (简化)
        plus_dm = (high_s - high_s.shift(1)).clip(lower=0)
        minus_dm = (low_s.shift(1) - low_s).clip(lower=0)
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0
        atr14 = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df.loc[idx, 'adx_14'] = dx.rolling(14).mean().values
        
        # OBV slope
        obv = (vol_series * np.sign(pd.Series(close).diff())).cumsum()
        df.loc[idx, 'obv_slope_20d'] = (obv.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan, raw=True)).values
        df.loc[idx, 'obv_slope_5d'] = (obv.rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan, raw=True)).values
        
        # Stochastic
        low14 = low_s.rolling(14).min()
        high14 = high_s.rolling(14).max()
        stoch_k = 100 * (pd.Series(close) - low14) / (high14 - low14).replace(0, np.nan)
        stoch_d = stoch_k.rolling(3).mean()
        df.loc[idx, 'stoch_k'] = stoch_k.values
        df.loc[idx, 'stoch_d'] = stoch_d.values
        
        # Williams %R
        df.loc[idx, 'williams_r'] = (-100 * (high14 - pd.Series(close)) / (high14 - low14).replace(0, np.nan)).values
        
        # CCI
        tp = (high_s + low_s + pd.Series(close)) / 3
        tp_sma = tp.rolling(20).mean()
        tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df.loc[idx, 'cci_20'] = ((tp - tp_sma) / (0.015 * tp_mad).replace(0, np.nan)).values
        
        # MFI
        mf = tp * vol_series
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        mfi = 100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))
        df.loc[idx, 'mfi_14'] = mfi.values
        
        # Force Index
        fi = pd.Series(close).diff() * vol_series
        df.loc[idx, 'force_index_13'] = fi.ewm(span=13).mean().values
        
        # TSI
        pc = pd.Series(close).diff()
        pc_ema1 = pc.ewm(span=25).mean()
        pc_ema2 = pc_ema1.ewm(span=13).mean()
        abs_pc_ema1 = pc.abs().ewm(span=25).mean()
        abs_pc_ema2 = abs_pc_ema1.ewm(span=13).mean()
        df.loc[idx, 'tsi'] = (100 * pc_ema2 / abs_pc_ema2.replace(0, np.nan)).values
        
        # High-Low range
        df.loc[idx, 'high_low_range'] = ((high_s - low_s) / pd.Series(close)).values
        
        # Close position
        df.loc[idx, 'close_position'] = ((pd.Series(close) - low_s) / (high_s - low_s).replace(0, np.nan)).values
        
        # Gap
        df.loc[idx, 'gap'] = (pd.Series(close) / pd.Series(close).shift(1) - 1).values
        
        # Consecutive up/down
        up = (pd.Series(close).diff() > 0).astype(int)
        down = (pd.Series(close).diff() < 0).astype(int)
        df.loc[idx, 'consecutive_up'] = up.rolling(5).sum().values
        df.loc[idx, 'consecutive_down'] = down.rolling(5).sum().values
        
        # 52-week
        high_52w = high_s.rolling(252).max()
        low_52w = low_s.rolling(252).min()
        df.loc[idx, 'distance_from_52w_high'] = ((pd.Series(close) - high_52w) / high_52w.replace(0, np.nan)).values
        df.loc[idx, 'distance_from_52w_low'] = ((pd.Series(close) - low_52w) / low_52w.replace(0, np.nan)).values
        
        # Statistical
        df.loc[idx, 'rolling_skew_20d'] = ret.rolling(20).skew().values
        df.loc[idx, 'rolling_kurt_20d'] = ret.rolling(20).kurt().values
        
        # Vol decomposition
        df.loc[idx, 'down_vol_20d'] = ret.where(ret < 0, 0).rolling(20).std().values
        df.loc[idx, 'up_vol_20d'] = ret.where(ret > 0, 0).rolling(20).std().values
        
        # Volume trend
        df.loc[idx, 'volume_trend_20d'] = (vol_series.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan, raw=True)).values / vol_series.rolling(20).mean().replace(0, np.nan)
        
        # High/Low vol returns
        high_vol = ret.rolling(20).std() > ret.rolling(60).std()
        df.loc[idx, 'high_vol_return_20d'] = (ret.where(high_vol, 0).rolling(20).mean()).values
        df.loc[idx, 'low_vol_return_20d'] = (ret.where(~high_vol, 0).rolling(20).mean()).values
        
        # Max drawdown
        rolling_max = pd.Series(close).rolling(20).max()
        df.loc[idx, 'max_drawdown_20d'] = ((pd.Series(close) - rolling_max) / rolling_max).values
        
        # Realized vol ratio
        vol_short = ret.rolling(10).std()
        vol_long = ret.rolling(60).std()
        df.loc[idx, 'realized_vol_ratio'] = (vol_short / vol_long.replace(0, np.nan)).values
    
    # Drop NaN
    df_clean = df.dropna(subset=feat_cols + ['fwd_ret']).copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols)
    
    print(f"  有效数据: {len(df_clean):,} 行")
    
    # 数据划分
    train = df_clean[df_clean['date'] < '2022-01-01']
    val = df_clean[(df_clean['date'] >= '2022-01-01') & (df_clean['date'] < '2024-01-01')]
    test = df_clean[df_clean['date'] >= '2024-01-01']
    
    print(f"  训练: {len(train):,}  验证: {len(val):,}  测试: {len(test):,}")
    
    X_train = train[feat_cols].values
    y_train = train['fwd_ret'].values
    X_val = val[feat_cols].values
    y_val = val['fwd_ret'].values
    X_test = test[feat_cols].values
    y_test = test['fwd_ret'].values
    test_dates = test['date'].values
    test_codes = test['sym'].values
    
    # Adjusted MSE
    def adj_mse_obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        residual = y_pred - y_true
        sign_match = np.sign(y_pred) == np.sign(y_true)
        weights = np.where(sign_match, 0.1, 1.1)
        grad = 2 * weights * residual
        hess = 2 * weights * np.ones_like(residual)
        return grad, hess
    
    # Train
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
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
        'seed': 42,
    }
    
    callbacks = [lgb.early_stopping(100), lgb.log_evaluation(0)]
    model = lgb.train(params, dtrain, num_boost_round=800, valid_sets=[dval], callbacks=callbacks)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    pred_df = pd.DataFrame({
        'date': test_dates,
        'sym': test_codes,
        'pred': y_pred,
        'actual': y_test
    })
    
    # 每天Top10
    daily_returns = []
    for dt, group in pred_df.groupby('date'):
        top10 = group.nlargest(10, 'pred')
        daily_ret = top10['actual'].mean()
        daily_returns.append({'date': dt, 'ret': daily_ret})
    
    daily_df = pd.DataFrame(daily_returns)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # 基准
    spy = df_clean[df_clean['sym'] == 'SPY'].set_index('date')['fwd_ret']
    
    # 统计
    total_days = len(daily_df)
    years = total_days / 252
    
    cum_ret = (1 + daily_df['ret']).cumprod().iloc[-1] - 1
    ann_ret = (1 + cum_ret) ** (1/years) - 1
    
    avg_daily = daily_df['ret'].mean()
    std_daily = daily_df['ret'].std()
    sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
    
    cum_max = (1 + daily_df['ret']).cumprod().cummax()
    drawdown = (1 + daily_df['ret']).cumprod() / cum_max - 1
    max_dd = drawdown.min()
    
    win_rate = (daily_df['ret'] > 0).mean()
    
    # SPY基准
    if len(spy) > 0:
        spy_cum = (1 + spy).cumprod().iloc[-1] - 1
        spy_ann = (1 + spy_cum) ** (1/years) - 1
        spy_sharpe = (spy.mean() / spy.std()) * np.sqrt(252) if spy.std() > 0 else 0
    else:
        spy_ann = spy_sharpe = 0
    
    results.append({
        'window': window,
        'ann_ret': ann_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'spy_ann': spy_ann,
        'spy_sharpe': spy_sharpe,
        'total_days': total_days,
    })
    
    print(f"  年化: {ann_ret*100:.1f}%  夏普: {sharpe:.2f}  回撤: {max_dd*100:.1f}%  胜率: {win_rate*100:.1f}%")
    print(f"  SPY:  年化: {spy_ann*100:.1f}%  夏普: {spy_sharpe:.2f}")

# 汇总
print(f"\n{'='*60}")
print("汇总对比")
print(f"{'='*60}")
print(f"{'窗口':<8} {'年化':<10} {'夏普':<8} {'回撤':<10} {'胜率':<8} {'SPY年化':<10} {'SPY夏普':<8}")
print("-" * 60)
for r in results:
    print(f"{r['window']}天    {r['ann_ret']*100:.1f}%    {r['sharpe']:.2f}   {r['max_dd']*100:.1f}%    {r['win_rate']*100:.1f}%    {r['spy_ann']*100:.1f}%     {r['spy_sharpe']:.2f}")

print(f"\n结论：看哪个窗口在风险调整后（夏普）表现最好")
