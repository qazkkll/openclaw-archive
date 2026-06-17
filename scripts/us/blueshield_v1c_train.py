#!/usr/bin/env python3
"""
蓝盾 V1c — 核心理念切换
预测目标：不是"未来5天赚多少"，而是"未来5天赢SP500多少"

标签 = ret_f5 - 同期SP500平均ret_f5
预测相对Alpha，不是绝对收益
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

# ========= 新标签：相对Alpha =========
# 每只票的未来5天收益减去同期市场平均未来5天收益
market_f5 = feat.groupby('Date')['ret_f5'].mean().reset_index()
market_f5.columns = ['Date', 'market_f5']
feat = feat.merge(market_f5, on='Date', how='left')
feat['alpha_5d'] = feat['ret_f5'] - feat['market_f5']

print(f'Alpha分布: min={feat["alpha_5d"].min()*100:.1f}% max={feat["alpha_5d"].max()*100:.1f}% mean={feat["alpha_5d"].mean()*100:.2f}%')

# 特征
feat_cols = ['ret_1d','ret_3d','ret_5d','ret_10d','ret_20d',
             'ma_5_ratio','ma_10_ratio','ma_20_ratio','ma_50_ratio',
             'vol_5d','vol_10d','vol_20d','rsi_14',
             'vol_ratio_5','vol_ratio_20',
             'price_pos_20','price_pos_50','price_pos_100',
             'macd','macd_sig','macd_hist','atr_pct']

# 相对强度特征
market_ret = feat.groupby('Date')['ret_1d'].mean().reset_index()
market_ret.columns = ['Date', 'market_ret']
feat = feat.merge(market_ret, on='Date', how='left')
feat['rel_ret_1d'] = feat['ret_1d'] - feat['market_ret']
feat['rel_ret_5d'] = feat['ret_5d'] - feat.groupby('Date')['ret_5d'].transform('mean')
feat['rel_ret_10d'] = feat['ret_10d'] - feat.groupby('Date')['ret_10d'].transform('mean')

feat['ma20_ma50_cross'] = feat['ma_20_ratio'] - feat['ma_50_ratio']
feat['dvol_ma5'] = feat.groupby('Code')['DollarVol'].transform(lambda x: x.rolling(5).mean())
feat['dvol_ratio'] = np.where(feat['dvol_ma5'] > 0, feat['DollarVol'] / feat['dvol_ma5'], 1.0)
feat['vol_5d_ratio'] = feat['vol_5d'] / (feat.groupby('Date')['vol_5d'].transform('mean') + 1e-8)

all_feats = feat_cols + ['rel_ret_1d','rel_ret_5d','rel_ret_10d','ma20_ma50_cross',
                         'dvol_ratio','vol_5d_ratio']

valid = feat.dropna(subset=all_feats + ['alpha_5d']).copy()
print(f'有效样本: {len(valid)}')

dates = pd.Series(valid['Date'].unique()).sort_values().values
n = len(dates)
train_mask = valid['Date'] <= dates[int(n*0.7)]
val_mask = (valid['Date'] > dates[int(n*0.7)]) & (valid['Date'] <= dates[int(n*0.85)])
test_mask = valid['Date'] > dates[int(n*0.85)]

X_train = valid.loc[train_mask, all_feats].values
y_train = valid.loc[train_mask, 'alpha_5d'].values
X_val = valid.loc[val_mask, all_feats].values
y_val = valid.loc[val_mask, 'alpha_5d'].values
X_test = valid.loc[test_mask, all_feats].values
y_test = valid.loc[test_mask, 'alpha_5d'].values

print(f'Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}')

print('\n训练XGBoost回归（预测Alpha）...')
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
print(f'\nAlpha预测 RMSE: {rmse*100:.2f}%')

# 看模型是否有区分度：按预测Alpha高低分桶，看实际Alpha
test_df = valid[test_mask].copy()
test_df['pred_alpha'] = y_pred
test_df['alpha_bucket'] = pd.qcut(test_df['pred_alpha'], 5, labels=['Q1最低','Q2','Q3','Q4','Q5最高'])

bucket_perf = test_df.groupby('alpha_bucket')['alpha_5d'].agg(['mean','std','count'])
print('\n按预测Alpha分桶的实际表现:')
print(bucket_perf.to_string())

# ========= 回测 =========
print('\n=== 回测：选Alpha最高的Top15 ===')
dates_list = sorted(test_df['Date'].unique())
results = []

for d in dates_list:
    day_df = test_df[test_df['Date'] == d].copy()
    day_df = day_df[day_df['dvol_ma5'] >= 5_000_000]
    if len(day_df) < 20:
        continue
    
    top = day_df.nlargest(15, 'pred_alpha')
    
    for _, row in top.iterrows():
        results.append({
            'sym': row['Code'],
            'date': str(d)[:10],
            'pred_alpha': round(row['pred_alpha'], 4),
            'alpha_5d': round(row['alpha_5d'], 4),
            'ret_f5': round(row['ret_f5'], 4),
        })

res_df = pd.DataFrame(results)
print(f'交易总笔: {len(res_df)}')

if len(res_df) > 0:
    # 组合Alpha = 每个持仓的alpha平均
    daily_alpha = res_df.groupby('date')['alpha_5d'].mean()
    # 组合总收益 = 市场收益 + 组合alpha
    market_daily = test_df.groupby('Date')['ret_f5'].mean()
    daily_ret = market_daily + daily_alpha
    
    win = (daily_ret > 0).mean()
    avg = daily_ret.mean()
    vol = daily_ret.std()
    
    ann_ret = avg * (252/5)
    ann_vol = vol * np.sqrt(252/5)
    sharpe = ann_ret / max(ann_vol, 0.001)
    
    cum = (1 + daily_ret).cumprod()
    dd = (cum / cum.cummax() - 1)
    max_dd = dd.min()
    
    print(f'\n=== 蓝盾V1c 最终结果 ===')
    print(f'交易天数: {len(daily_ret)}')
    print(f'组合Alpha均值(5天): {daily_alpha.mean()*100:.2f}%')
    print(f'胜率(日): {win*100:.1f}%')
    print(f'年化收益: {ann_ret*100:.1f}%')
    print(f'年化波动: {ann_vol*100:.1f}%')
    print(f'夏普: {sharpe:.2f}')
    print(f'最大回撤: {max_dd*100:.1f}%')
    
    print(f'\n=== 对比基准 (V5.5: 年化+21.2%, 夏普3.87) ===')
    
    top_syms = res_df.groupby('sym')['alpha_5d'].agg(['count','mean']).sort_values('count', ascending=False).head(10)
    print(f'\n最常买入Top10:')
    for s, row in top_syms.iterrows():
        print(f'  {s}: {int(row["count"])}次, 平均Alpha {row["mean"]*100:.2f}%')

model_path = '/home/hermes/.hermes/openclaw-project/data/models/blueshield_v1c.model'
model.save_model(model_path)
pickle.dump(model, open('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v1c.pkl', 'wb'))

imp = pd.DataFrame({'feat': all_feats, 'imp': model.feature_importances_}).sort_values('imp', ascending=False)
print('\nTop 10特征:')
print(imp.head(10).to_string(index=False))

meta = {
    'model': 'blueshield_v1c',
    'label': 'alpha_5d_prediction',
    'features': all_feats, 'n_features': len(all_feats),
    'backtest': {
        'annual_return': float(ann_ret),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'win_rate': float(win),
        'n_trades': len(res_df),
        'avg_alpha_5d': float(daily_alpha.mean()),
        'strategy': 'predicted_alpha_top15_hold5d',
    },
    'date': time.strftime('%Y-%m-%d %H:%M')
}
json.dump(meta, open('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v1c_meta.json', 'w'), indent=2)
print(f'\n完成: {time.strftime("%Y-%m-%d %H:%M")}')
