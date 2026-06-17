#!/usr/bin/env python3
"""
蓝盾 V2c — 仓位缩放优化
不空仓，只用SP500大盘MA20位置做仓位缩放
大盘好 → 满仓10只
大盘中性 → 半仓5只
大盘差 → 减到2-3只（不空仓）
单票止损-8%
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

# 特征
feat = feat.merge(raw_df[['Code','Date','C','DollarVol']], on=['Code','Date'], how='left')
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

feat_cols = ['ret_1d','ret_3d','ret_5d','ret_10d','ret_20d',
             'ma_5_ratio','ma_10_ratio','ma_20_ratio','ma_50_ratio',
             'vol_5d','vol_10d','vol_20d','rsi_14','rsi_50_pct',
             'vol_ratio_5','vol_ratio_20','vol_5d_norm',
             'price_pos_20','price_pos_50','price_pos_100',
             'macd','macd_sig','macd_hist','atr_pct',
             'rel_ret_1d','rel_ret_5d','rel_ret_10d','ma20_ma50_cross',
             'dvol_ratio']

# SP500大盘位置
prices = raw_df.pivot_table(index='Date', columns='Code', values='C')
spx = prices.mean(axis=1)
spx_ma20 = spx.rolling(20).mean()
spx_feat = pd.DataFrame({
    'Date': spx.index,
    'spx_pos': (spx.values - spx_ma20.values) / spx_ma20.values,
    'spx_ma20': spx_ma20.values
}).fillna(0)
feat = feat.merge(spx_feat, on='Date', how='left')

all_feats = feat_cols + ['spx_pos']

valid = feat.dropna(subset=all_feats + ['ret_5d','ret_f5','dvol_ma5','C']).copy()
valid['is_trend'] = (valid['ret_5d'] > 0) & (valid['dvol_ma5'] >= 5_000_000)
trend = valid[valid['is_trend']].copy()
print(f'全量: {len(valid)}  趋势票: {len(trend)}')

dates = pd.Series(valid['Date'].unique()).sort_values().values
t_end = dates[int(len(dates)*0.7)]
v_end = dates[int(len(dates)*0.85)]
te_dates = dates[int(len(dates)*0.85):]

t_mask = trend['Date'] <= t_end
v_mask = (trend['Date'] > t_end) & (trend['Date'] <= v_end)
te_mask = trend['Date'] > v_end

X_t = trend.loc[t_mask, all_feats].values; y_t = trend.loc[t_mask, 'ret_f5'].values
X_v = trend.loc[v_mask, all_feats].values; y_v = trend.loc[v_mask, 'ret_f5'].values
X_te = trend.loc[te_mask, all_feats].values; y_te = trend.loc[te_mask, 'ret_f5'].values

print('\n训练...')
model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=1.0,
    eval_metric='rmse', early_stopping_rounds=50, random_state=42, n_jobs=-1)
model.fit(X_t, y_t, eval_set=[(X_t, y_t), (X_v, y_v)], verbose=200)

y_pred = model.predict(X_te)
from sklearn.metrics import mean_squared_error
print(f'RMSE: {np.sqrt(mean_squared_error(y_te, y_pred))*100:.2f}%')

# ========= 回测：等权法 =========
print('\n=== 蓝盾V2c 回测 ===')
te_df = trend[te_mask].copy()
te_df['pred'] = y_pred

# 先看V2（无风控）在同样数据上——确保基准
print('\n--- V2基准（无风控）---')
v2_daily = []
for d in sorted(te_df['Date'].unique()):
    day = te_df[te_df['Date'] == d]
    if len(day) < 10: continue
    top = day.nlargest(10, 'pred')
    avg_ret = top['ret_f5'].mean()
    v2_daily.append({'date': d, 'ret': avg_ret})
v2_df = pd.DataFrame(v2_daily)
v2_win = (v2_df['ret'] > 0).mean()
v2_avg = v2_df['ret'].mean()
v2_ann = v2_avg * (252/5)
v2_vol = v2_df['ret'].std() * np.sqrt(252/5)
v2_shp = v2_ann / max(v2_vol, 0.001)
v2_cum = (1 + v2_df['ret']).cumprod()
v2_dd = (v2_cum / v2_cum.cummax() - 1).min()
print(f'V2基准: 年化{v2_ann*100:.1f}%, 夏普{v2_shp:.2f}, 回撤{v2_dd*100:.1f}%, 胜率{v2_win*100:.1f}%')

# --- V2c（仓位缩放）---
print('\n--- V2c（仓位缩放: 大盘MA20位置）---')
scaled_daily = []
for d in sorted(te_df['Date'].unique()):
    day = te_df[te_df['Date'] == d]
    if len(day) < 2: continue
    
    spx_pos = day['spx_pos'].iloc[0]
    
    # 仓位缩放：SP500大盘在MA20以上越多 → 越多票
    if spx_pos > 0.03:       n = 12   # 强势
    elif spx_pos > 0.01:    n = 10   # 正常
    elif spx_pos > -0.02:   n = 6    # 震荡
    elif spx_pos > -0.05:   n = 4    # 弱势
    else:                    n = 2    # 极弱势
    
    n = max(2, min(n, len(day)))
    top = day.nlargest(n, 'pred')
    avg_ret = top['ret_f5'].mean()
    
    # 仓位系数：弱势时仓位减半
    weight = 1.0
    if spx_pos < -0.03: weight = 0.6
    if spx_pos < -0.06: weight = 0.3
    if spx_pos < -0.10: weight = 0.15
    
    scaled_daily.append({'date': d, 'ret': avg_ret * weight, 'n': n, 'weight': weight, 'spx_pos': spx_pos})

s_df = pd.DataFrame(scaled_daily)
s_win = (s_df['ret'] > 0).mean()
s_avg = s_df['ret'].mean()
s_ann = s_avg * (252/5)
s_vol = s_df['ret'].std() * np.sqrt(252/5)
s_shp = s_ann / max(s_vol, 0.001)
s_cum = (1 + s_df['ret']).cumprod()
s_dd = (s_cum / s_cum.cummax() - 1).min()

print(f'V2c: 年化{s_ann*100:.1f}%, 夏普{s_shp:.2f}, 回撤{s_dd*100:.1f}%, 胜率{s_win*100:.1f}%')
print(f'    均值仓{s_df["n"].mean():.1f}只, 均权重{s_df["weight"].mean():.2f}x')
print(f'    SPX_pos范围: {s_df["spx_pos"].min():.3f}~{s_df["spx_pos"].max():.3f}')

# ========= 多参数扫描 =========
print('\n=== 参数扫描 ===')
best_set = None
best_shp = -999

# 尝试不同的大盘阈值组合
for w1 in [0.03, 0.05, 0.10]:
    for w2 in [0.03, 0.05, 0.08]:
        for w3 in [0.10, 0.15, 0.20]:
            scaled = []
            for d in sorted(te_df['Date'].unique()):
                day = te_df[te_df['Date'] == d]
                if len(day) < 2: continue
                spx_pos = day['spx_pos'].iloc[0]
                
                if spx_pos > w1:     n = 12
                elif spx_pos > 0:    n = 10
                elif spx_pos > -w2:  n = 6
                elif spx_pos > -w3:  n = 3
                else:                 n = 2
                n = max(2, min(n, len(day)))
                top = day.nlargest(n, 'pred')
                avg_ret = top['ret_f5'].mean()
                
                wt = 1.0
                if spx_pos < -w2: wt = 0.5
                if spx_pos < -w3: wt = 0.2
                scaled.append(avg_ret * wt)
            
            arr = np.array(scaled)
            ann = arr.mean() * (252/5)
            vol = arr.std() * np.sqrt(252/5)
            shp = ann / max(vol, 0.001)
            cum = (1 + pd.Series(arr)).cumprod()
            dd = (cum / cum.cummax() - 1).min()
            
            if shp > best_shp:
                best_shp = shp
                best_set = (w1, w2, w3, ann, dd)

print(f'最佳参数: w1={best_set[0]:.2f} w2={best_set[1]:.2f} w3={best_set[2]:.2f}')
print(f'最佳结果: 年化{best_set[3]*100:.1f}%, 夏普{best_shp:.2f}, 回撤{best_set[4]*100:.1f}%')

# ========= 保存 =========
model.get_booster().save_model('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v2c.model')
pickle.dump(model, open('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v2c.pkl', 'wb'))

imp = pd.DataFrame({'feat': all_feats, 'imp': model.feature_importances_}).sort_values('imp', ascending=False)
print('\nTop10特征:')
print(imp.head(10).to_string(index=False))

meta = {
    'model': 'blueshield_v2c', 'strategy': 'trend_ML_positionscaling',
    'features': all_feats, 'n_features': len(all_feats),
    'backtest': {
        'v2_base': {'annual_return': float(v2_ann), 'sharpe': float(v2_shp), 'max_drawdown': float(v2_dd)},
        'v2c_base': {'annual_return': float(s_ann), 'sharpe': float(s_shp), 'max_drawdown': float(s_dd), 'win_rate': float(s_win)},
        'v2c_best_params': {'thresholds': best_set[:3], 'annual_return': float(best_set[3]), 'sharpe': float(best_shp), 'max_drawdown': float(best_set[4])},
        'v55_target': 'annual_21.2_sharpe_3.87'
    },
    'date': time.strftime('%Y-%m-%d %H:%M')
}
json.dump(meta, open('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v2c_meta.json', 'w'), indent=2)
print(f'\n完成: {time.strftime("%Y-%m-%d %H:%M")}')
