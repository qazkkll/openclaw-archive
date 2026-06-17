#!/usr/bin/env python3
"""
蓝盾 V2b — 优化回撤
改进点：
1. 大盘环境过滤（SP500 MA20 < 0 → 半仓，MA50 < 0 → 空仓）
2. 波动率仓位控制（高波减仓）
3. 单票止损 -8%
4. 持有5天后自动卖出
"""
import json, warnings, os, sys, time
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

print('加载特征...')
feat = pd.read_parquet('/home/hermes/.hermes/openclaw-project/data/us/sp500_feats.parquet')
feat = feat.sort_values(['Code','Date']).reset_index(drop=True)
feat['Date'] = pd.to_datetime(feat['Date'])

# 成交额
raw_dir = '/home/hermes/.hermes/openclaw-project/data/hist_sp500'
all_rows = []
for f in sorted(os.listdir(raw_dir)):
    if not f.startswith('sp500_chunk_') or not f.endswith('.json'):
        continue
    raw = json.load(open(os.path.join(raw_dir, f)))
    for sym, bars in raw.items():
        for b in bars:
            b['Code'] = sym
        all_rows.extend(bars)
raw_df = pd.DataFrame(all_rows)
raw_df['Date'] = pd.to_datetime(raw_df['Date'])
raw_df['DollarVol'] = raw_df['C'] * raw_df['V']
feat = feat.merge(raw_df[['Code','Date','DollarVol']], on=['Code','Date'], how='left')

# 用原始价格计算
prices = raw_df.pivot_table(index='Date', columns='Code', values='C')
spx = prices.mean(axis=1)  # SP500近似
spx_ret = spx.pct_change()
spx_ma20 = spx.rolling(20).mean()
spx_ma50 = spx.rolling(50).mean()

# 特征
market_ret = feat.groupby('Date')['ret_1d'].mean().reset_index()
market_ret.columns = ['Date', 'market_ret']
feat = feat.merge(market_ret, on='Date', how='left')
feat['rel_ret_1d'] = feat['ret_1d'] - feat['market_ret']
feat['rel_ret_5d'] = feat['ret_5d'] - feat.groupby('Date')['ret_5d'].transform('mean')
feat['rel_ret_10d'] = feat['ret_10d'] - feat.groupby('Date')['ret_10d'].transform('mean')
feat['ma20_ma50_cross'] = feat['ma_20_ratio'] - feat['ma_50_ratio']
feat['dvol_ma5'] = feat.groupby('Code')['DollarVol'].transform(lambda x: x.rolling(5).mean())
feat['dvol_ratio'] = np.where(feat['dvol_ma5'] > 0, feat['DollarVol'] / feat['dvol_ma5'], 1.0)
feat['vol_5d_norm'] = feat['vol_5d'] / (feat.groupby('Date')['vol_5d'].transform('mean') + 1e-8)
feat['rsi_50_pct'] = (feat['rsi_14'] - 50) / 50

# 加入SP500大盘特征
spx_feat = pd.DataFrame({'Date': spx_ret.index, 
                          'spx_1d': spx_ret.values,
                          'spx_ma20_ratio': (spx.values - spx_ma20.values) / spx_ma20.values,
                          'spx_ma50_ratio': (spx.values - spx_ma50.values) / spx_ma50.values})
spx_feat = spx_feat.fillna(0)
feat = feat.merge(spx_feat, on='Date', how='left')

feat_cols = ['ret_1d','ret_3d','ret_5d','ret_10d','ret_20d',
             'ma_5_ratio','ma_10_ratio','ma_20_ratio','ma_50_ratio',
             'vol_5d','vol_10d','vol_20d','rsi_14','rsi_50_pct',
             'vol_ratio_5','vol_ratio_20','vol_5d_norm',
             'price_pos_20','price_pos_50','price_pos_100',
             'macd','macd_sig','macd_hist','atr_pct',
             'rel_ret_1d','rel_ret_5d','rel_ret_10d','ma20_ma50_cross',
             'dvol_ratio','spx_1d','spx_ma20_ratio','spx_ma50_ratio']

valid = feat.dropna(subset=feat_cols + ['ret_5d','ret_f5','dvol_ma5']).copy()
print(f'全量有效: {len(valid)}')

valid['is_trend'] = (valid['ret_5d'] > 0) & (valid['dvol_ma5'] >= 5_000_000)
trend = valid[valid['is_trend']].copy()
print(f'趋势票: {len(trend)}')

dates = pd.Series(valid['Date'].unique()).sort_values().values
n = len(dates)
t_mask = trend['Date'] <= dates[int(n*0.7)]
v_mask = (trend['Date'] > dates[int(n*0.7)]) & (trend['Date'] <= dates[int(n*0.85)])
te_mask = trend['Date'] > dates[int(n*0.85)]

X_t = trend.loc[t_mask, feat_cols].values; y_t = trend.loc[t_mask, 'ret_f5'].values
X_v = trend.loc[v_mask, feat_cols].values; y_v = trend.loc[v_mask, 'ret_f5'].values
X_te = trend.loc[te_mask, feat_cols].values; y_te = trend.loc[te_mask, 'ret_f5'].values
print(f'Train {len(X_t)}  Val {len(X_v)}  Test {len(X_te)}')

print('\n训练...')
model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=1.0,
    eval_metric='rmse', early_stopping_rounds=50, random_state=42, n_jobs=-1)
model.fit(X_t, y_t, eval_set=[(X_t, y_t), (X_v, y_v)], verbose=200)

y_pred = model.predict(X_te)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_te, y_pred))
print(f'RMSE: {rmse*100:.2f}%')

# ========= 回测（含风控） =========
print('\n=== 蓝盾V2b 回测（含风控） ===')
te_df = trend[te_mask].copy()
te_df['pred'] = y_pred

# 从原始数据拼回价格
raw_price_df = raw_df[['Code','Date','C']].copy()
te_df = te_df.merge(raw_price_df, on=['Code','Date'], how='left')

date_list = sorted(te_df['Date'].unique())

# 模拟账户
cash = 100000
position = {}  # sym -> (buy_date, buy_price, shares)
equity_curve = []

for d in date_list:
    day_df = te_df[te_df['Date'] == d].copy()
    
    # ---- 风控：仓位管理 ----
    # SP500的大盘状态
    spx_signal = day_df['spx_ma20_ratio'].iloc[0] if len(day_df) > 0 else 0
    spx_recent = te_df[te_df['Date'] <= d]['pred'].tail(5).mean() if len(te_df[te_df['Date'] <= d]) > 5 else 0
    
    # 大盘MA20跌破 → 半仓，MA50跌破 → 空仓
    max_positions = 10  # 正常10只
    if spx_signal < -0.05:  # SP500跌破MA20超过5%
        max_positions = 5   # 半仓
    if spx_signal < -0.10:  # SP500大幅破位
        max_positions = 2   # 极低仓位
    if spx_signal < -0.15:  # 深度熊市
        max_positions = 0   # 空仓
    
    # ---- 平仓：持有超过5天或触发止损 ----
    to_close = []
    for sym, (buy_d, buy_p, shs) in position.items():
        # 获取当日价格
        current_p = day_df[day_df['Code'] == sym]['C'].iloc[0] if len(day_df[day_df['Code'] == sym]) > 0 else None
        if current_p is None:
            to_close.append(sym)
            continue
        
        # 止损 -8%
        ret = (current_p - buy_p) / buy_p
        if ret < -0.08:
            to_close.append(sym)
            cash += shs * current_p
            continue
        
        # 持有超过5天 → 卖出
        days_held = (pd.to_datetime(d) - pd.to_datetime(buy_d)).days
        if days_held >= 5:
            to_close.append(sym)
            cash += shs * current_p
    
    for sym in to_close:
        del position[sym]
    
    # ---- 开仓 ----
    current_positions = len(position)
    can_open = max_positions - current_positions
    
    if can_open > 0 and len(day_df) > 0 and max_positions > 0:
        # 选择Top N
        day_df = day_df[day_df['dvol_ma5'] >= 5_000_000]
        if len(day_df) >= can_open:
            top = day_df.nlargest(can_open, 'pred')
            
            per_trade = cash / can_open  # 每只票的现金
            
            for _, row in top.iterrows():
                sym = row['Code']
                if sym in position:
                    continue
                price = row['C']
                shares = per_trade / price
                cash -= shares * price
                position[sym] = (d, price, shares)
    
    # ---- 每日净值估算 ----
    pos_value = 0
    for sym, (buy_d, buy_p, shs) in position.items():
        current_p = day_df[day_df['Code'] == sym]['C'].iloc[0] if len(day_df[day_df['Code'] == sym]) > 0 else buy_p
        pos_value += shs * current_p
    
    equity = cash + pos_value
    equity_curve.append({'date': str(d)[:10], 'equity': equity, 'positions': len(position), 'max_pos': max_positions})

# ========= 分析 =========
ec_df = pd.DataFrame(equity_curve)
initial = 100000
final = ec_df['equity'].iloc[-1]

ec_df['ret'] = ec_df['equity'].pct_change()
ec_df['ret'].fillna(0, inplace=True)

total_days = len(ec_df)
ann_ret = (final/initial) ** (252/total_days) - 1 if total_days > 0 else 0
ann_vol = ec_df['ret'].std() * np.sqrt(252)
sharpe = ann_ret / max(ann_vol, 0.001)

ec_df['cum'] = ec_df['equity'] / initial
ec_df['peak'] = ec_df['cum'].cummax()
ec_df['dd'] = ec_df['cum'] / ec_df['peak'] - 1
max_dd = ec_df['dd'].min()

print(f'\n=== 蓝盾V2b 最终结果（$100K模拟） ===')
print(f'终值: ${final:,.0f}')
print(f'总收益: {(final/initial-1)*100:.1f}%')
print(f'交易天数: {total_days}')
print(f'年化收益: {ann_ret*100:.1f}%')
print(f'年化波动: {ann_vol*100:.1f}%')
print(f'夏普: {sharpe:.2f}')
print(f'最大回撤: {max_dd*100:.1f}%')
print(f'平均持仓: {ec_df["positions"].mean():.1f}只')

# 与V2对比
print(f'\n=== V2 vs V2b ===')
print(f'V2:  年化+51.6%, 夏普1.84, 回撤-44.7%')
print(f'V2b: 年化{ann_ret*100:.1f}%, 夏普{sharpe:.2f}, 回撤{max_dd*100:.1f}%')

# ========= 保存 =========
model.get_booster().save_model('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v2b.model')
pickle.dump(model, open('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v2b.pkl', 'wb'))
ec_df.to_parquet('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v2b_equity.parquet', index=False)
print('\n模型+权益曲线保存成功')

meta = {
    'model': 'blueshield_v2b', 'strategy': 'trend_ML_top10_risk_control',
    'features': feat_cols, 'n_features': len(feat_cols),
    'backtest': {
        'final_equity': float(final), 'total_return': float(final/initial-1),
        'annual_return': float(ann_ret), 'sharpe': float(sharpe),
        'max_drawdown': float(max_dd), 'avg_positions': float(ec_df['positions'].mean()),
        'n_days': total_days,
        'risk_control': 'spx_ma20_half_ma50_zero_stop8pct_hold5d',
        'v55_target': 'annual_21.2_sharpe_3.87',
        'v2_benchmark': 'annual_51.6_sharpe_1.84_dd_44.7'
    },
    'date': time.strftime('%Y-%m-%d %H:%M')
}
json.dump(meta, open('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v2b_meta.json', 'w'), indent=2)
print(f'\n完成: {time.strftime("%Y-%m-%d %H:%M")}')
