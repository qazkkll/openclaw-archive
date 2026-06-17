#!/usr/bin/env python3
"""
蓝盾 V1b（Blue Shield V1b）
改进：
- 减少规则过滤（只保留趋势向上）
- 放宽ML权重（全ML驱动）
- 增加持仓（Top 20）
- 每日全仓进出
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
    data = json.load(open(os.path.join(raw_dir, f)))
    for sym, bars in data.items():
        for b in bars:
            b['Code'] = sym
        all_rows.extend(bars)
raw_df = pd.DataFrame(all_rows)
raw_df['Date'] = pd.to_datetime(raw_df['Date'])
raw_df['DollarVol'] = raw_df['C'] * raw_df['V']
feat = feat.merge(raw_df[['Code','Date','DollarVol']], on=['Code','Date'], how='left')

# 特征
feat_cols = ['ret_1d','ret_3d','ret_5d','ret_10d','ret_20d',
             'ma_5_ratio','ma_10_ratio','ma_20_ratio','ma_50_ratio',
             'vol_5d','vol_10d','vol_20d','rsi_14',
             'vol_ratio_5','vol_ratio_20',
             'price_pos_20','price_pos_50','price_pos_100',
             'macd','macd_sig','macd_hist','atr_pct']

# 更多特征
market_ret = feat.groupby('Date')['ret_1d'].mean().reset_index()
market_ret.columns = ['Date', 'market_ret']
feat = feat.merge(market_ret, on='Date', how='left')
feat['rel_ret_5d'] = feat['ret_5d'] - feat.groupby('Date')['ret_5d'].transform('mean')
feat['rel_ret_10d'] = feat['ret_10d'] - feat.groupby('Date')['ret_10d'].transform('mean')
feat['ma20_ma50_cross'] = feat['ma_20_ratio'] - feat['ma_50_ratio']
feat['dvol_ma5'] = feat.groupby('Code')['DollarVol'].transform(lambda x: x.rolling(5).mean())
feat['dvol_ratio'] = np.where(feat['dvol_ma5'] > 0, feat['DollarVol'] / feat['dvol_ma5'], 1.0)
feat['ret_vol_ratio'] = feat['ret_5d'] / (feat['vol_5d'] + 1e-8)
feat['stoch_k'] = feat['price_pos_20']  # 已经算过了
# 均线斜率
feat['ma10_slope'] = feat['ma_10_ratio'] - feat['ret_1d']
feat['ma50_slope'] = feat.groupby('Code')['ma_10_ratio'].diff(5)

all_feats = feat_cols + ['rel_ret_5d','rel_ret_10d','ma20_ma50_cross',
                         'dvol_ratio','ret_vol_ratio','ma10_slope','ma50_slope']

valid = feat.dropna(subset=all_feats + ['ret_f5']).copy()
print(f'有效样本: {len(valid)}')

dates = pd.Series(valid['Date'].unique()).sort_values().values
n = len(dates)
train_mask = valid['Date'] <= dates[int(n*0.7)]
val_mask = (valid['Date'] > dates[int(n*0.7)]) & (valid['Date'] <= dates[int(n*0.85)])
test_mask = valid['Date'] > dates[int(n*0.85)]

print(f'Train: {train_mask.sum()}  Val: {val_mask.sum()}  Test: {test_mask.sum()}')

X_train = valid.loc[train_mask, all_feats].values
y_train = valid.loc[train_mask, 'ret_f5'].values
X_val = valid.loc[val_mask, all_feats].values
y_val = valid.loc[val_mask, 'ret_f5'].values
X_test = valid.loc[test_mask, all_feats].values
y_test = valid.loc[test_mask, 'ret_f5'].values

# 训练
print('\n训练XGBoost回归...')
model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.5,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric='rmse', early_stopping_rounds=50,
    random_state=42, n_jobs=-1
)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_val, y_val)],
          verbose=100)

y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'\nRMSE: {rmse*100:.2f}%  |  实际均值: {y_test.mean()*100:.2f}%')

# ========= 回测 =========
print('\n=== 蓝盾V1b 回测 ===')
test_df = valid[test_mask].copy()
test_df['pred'] = y_pred

dates_list = sorted(test_df['Date'].unique())
results = []

for d in dates_list:
    day_df = test_df[test_df['Date'] == d]
    
    # 只过滤成交额
    day_df = day_df[day_df['dvol_ma5'] >= 5_000_000]
    if len(day_df) < 20:
        continue
    
    # 全ML驱动（不做规则二次过滤）
    # 取ML预测值最高的前20名
    top = day_df.nlargest(20, 'pred')
    
    for _, row in top.iterrows():
        idx = row.name
        future = valid.iloc[idx:idx+6]
        if len(future) < 6:
            continue
        ret_5d = (future.iloc[-1]['ret_f5'] if 'ret_f5' in future.columns else 0) if len(future) > 5 else 0
        
        results.append({
            'sym': row['Code'],
            'date': str(d)[:10],
            'pred': round(row['pred'], 4),
            'ret_5d': round(future.iloc[-1]['ret_f5'], 4) if len(future) > 5 and 'ret_f5' in future.columns else 0
        })

res_df = pd.DataFrame(results)
print(f'交易总笔: {len(res_df)}')

if len(res_df) > 0:
    daily = res_df.groupby('date')['ret_5d'].mean()
    win = (daily > 0).mean()
    avg = daily.mean()
    vol = daily.std()
    
    ann_ret = avg * (252/5)
    ann_vol = vol * np.sqrt(252/5)
    sharpe = ann_ret / max(ann_vol, 0.001)
    
    cum = (1 + daily).cumprod()
    dd = (cum / cum.cummax() - 1)
    max_dd = dd.min()
    
    print(f'\n=== 蓝盾V1b 最终结果 ===')
    print(f'交易天数: {len(daily)}')
    print(f'胜率(日): {win*100:.1f}%')
    print(f'年化收益: {ann_ret*100:.1f}%')
    print(f'年化波动: {ann_vol*100:.1f}%')
    print(f'夏普: {sharpe:.2f}')
    print(f'最大回撤: {max_dd*100:.1f}%')
    
    # 与V5.5对比
    print(f'\n=== 对比基准 (V5.5) ===')
    print(f'目标: 年化+21.2%, 夏普3.87')
    print(f'蓝盾V1b: 年化{ann_ret*100:.1f}%, 夏普{sharpe:.2f}')
    print(f'差距: 年化{ann_ret*100-21.2:+.1f}%, 夏普{sharpe-3.87:+.2f}')
    
    top_syms = res_df.groupby('sym')['ret_5d'].agg(['count','mean']).sort_values('count', ascending=False).head(15)
    print(f'\n最常买入Top15:')
    for s, row in top_syms.iterrows():
        print(f'  {s}: {int(row["count"])}次, 平均5天{row["mean"]*100:.2f}%')

# ========= 保存 =========
model_path = '/home/hermes/.hermes/openclaw-project/data/models/blueshield_v1b.model'
model.save_model(model_path)
meta_path = '/home/hermes/.hermes/openclaw-project/data/models/blueshield_v1b.pkl'
pickle.dump(model, open(meta_path, 'wb'))
print(f'\n模型保存: blueshield_v1b.model/.pkl')

imp = pd.DataFrame({'feat': all_feats, 'imp': model.feature_importances_}).sort_values('imp', ascending=False)
print('\nTop 10特征:')
print(imp.head(10).to_string(index=False))

meta = {
    'model': 'blueshield_v1b', 'type': 'xgboost_regression',
    'features': all_feats, 'n_features': len(all_feats),
    'rmse': float(rmse),
    'backtest': {
        'annual_return': float(ann_ret) if len(daily) > 0 else 0,
        'sharpe': float(sharpe) if len(daily) > 0 else 0,
        'max_drawdown': float(max_dd) if len(daily) > 0 else 0,
        'win_rate': float(win) if len(daily) > 0 else 0,
        'n_trades': len(res_df),
        'strategy': 'ML_only_top20_hold5d',
        'v55_target': 'annual_21.2_sharpe_3.87'
    },
    'date': time.strftime('%Y-%m-%d %H:%M')
}
json.dump(meta, open('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v1b_meta.json', 'w'), indent=2)
print(f'\n完成: {time.strftime("%Y-%m-%d %H:%M")}')
