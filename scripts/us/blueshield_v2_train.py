#!/usr/bin/env python3
"""
蓝盾 V2（Blue Shield V2）
核心思路：不预测全市场，只预测趋势票中的赢家

流程：
1. 趋势筛 → 过去5天涨的票（趋势票候选池）
2. ML回归 → 对这些趋势票预测未来5天收益
3. 综合排名 → Top 10建仓
4. 持有5天，止损-8%
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

# 成交额数据
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

# ========= 特征工程 =========
# 市场均值和相对强度
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
feat['rsi_50_pct'] = (feat['rsi_14'] - 50) / 50  # RSI偏离50的程度

feat_cols = ['ret_1d','ret_3d','ret_5d','ret_10d','ret_20d',
             'ma_5_ratio','ma_10_ratio','ma_20_ratio','ma_50_ratio',
             'vol_5d','vol_10d','vol_20d','rsi_14','rsi_50_pct',
             'vol_ratio_5','vol_ratio_20','vol_5d_norm',
             'price_pos_20','price_pos_50','price_pos_100',
             'macd','macd_sig','macd_hist','atr_pct',
             'rel_ret_1d','rel_ret_5d','rel_ret_10d','ma20_ma50_cross',
             'dvol_ratio','dvol_ma5']

# 市场episode特征（这个是大盘大势）
feat['market_vol_20d'] = feat.groupby('Date')['vol_5d'].transform('mean').rolling(20, min_periods=5).mean()

print(f'特征数: {len(feat_cols)+1}')

# ========= 标签：未来5天收益 =========
valid = feat.dropna(subset=feat_cols + ['ret_5d','ret_f5','dvol_ma5']).copy()
print(f'全量有效样本: {len(valid)}')

# ========= 趋势票筛选：只训练买入\有趋势的票 =========
# 这里"趋势"定义为：过去5天涨 > 0（相对中性）
valid['is_trend'] = (valid['ret_5d'] > 0) & (valid['dvol_ma5'] >= 5_000_000)
trend = valid[valid['is_trend']].copy()
print(f'趋势票样本: {len(trend)} ({len(trend)/len(valid)*100:.1f}%)')

# 时间分割
dates = pd.Series(valid['Date'].unique()).sort_values().values
n = len(dates)
train_end = dates[int(n*0.7)]
val_end = dates[int(n*0.85)]

t_mask = trend['Date'] <= train_end
v_mask = (trend['Date'] > train_end) & (trend['Date'] <= val_end)
te_mask = trend['Date'] > val_end

X_t = trend.loc[t_mask, feat_cols].values
y_t = trend.loc[t_mask, 'ret_f5'].values
X_v = trend.loc[v_mask, feat_cols].values
y_v = trend.loc[v_mask, 'ret_f5'].values
X_te = trend.loc[te_mask, feat_cols].values
y_te = trend.loc[te_mask, 'ret_f5'].values

print(f'趋势票: Train {len(X_t)}  Val {len(X_v)}  Test {len(X_te)}')
print(f'训练集均值alpha: {y_t.mean()*100:.2f}%')

# ========= 训练 =========
print('\n训练XGBoost（趋势票收益预测）...')
model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.5,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric='rmse', early_stopping_rounds=50,
    random_state=42, n_jobs=-1
)
model.fit(X_t, y_t, eval_set=[(X_t, y_t), (X_v, y_v)], verbose=100)

y_pred = model.predict(X_te)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_te, y_pred))
print(f'\nRMSE: {rmse*100:.2f}%  |  趋势票均值: {y_te.mean()*100:.2f}%')

# ========= 按分桶验证区分度 =========
te_df = trend[te_mask].copy()
te_df['pred'] = y_pred
try:
    te_df['bucket'] = pd.qcut(te_df['pred'], 5, labels=['Q1','Q2','Q3','Q4','Q5'])
    bucket_perf = te_df.groupby('bucket')['ret_f5'].agg(['mean','std','count'])
    print('\n分桶表现:')
    print(bucket_perf.to_string())
except:
    print('分桶失败（可能预测值重复太多）')

# ========= 回测 =========
print('\n=== 蓝盾V2 回测 ===')
# 测试集上每天：对趋势票做ML评分，选Top 10
test_dates = sorted(te_df['Date'].unique())
results = []

for d in test_dates:
    day_df = te_df[te_df['Date'] == d].copy()
    if len(day_df) < 10:
        continue
    
    top = day_df.nlargest(10, 'pred')
    
    for _, row in top.iterrows():
        results.append({
            'sym': row['Code'], 'date': str(d)[:10],
            'pred': round(row['pred'], 4),
            'ret_f5': round(row['ret_f5'], 4),
            'alpha': round(row['ret_f5'] - row['market_ret'], 4),
        })

res_df = pd.DataFrame(results)
print(f'交易总笔: {len(res_df)}')

if len(res_df) > 0:
    daily = res_df.groupby('date')['ret_f5'].mean()
    win = (daily > 0).mean()
    avg = daily.mean()
    vol = daily.std()
    
    ann_ret = avg * (252/5)
    ann_vol = vol * np.sqrt(252/5)
    sharpe = ann_ret / max(ann_vol, 0.001)
    
    cum = (1 + daily).cumprod()
    dd = (cum / cum.cummax() - 1)
    max_dd = dd.min()
    
    # 同期纯趋势买入（买入所有趋势票的等权）作为基准
    # 只买趋势票：简化为每天选全部趋势票等权
    trend_daily = te_df.groupby('Date')['ret_f5'].mean()
    trend_win = (trend_daily > 0).mean()
    trend_avg = trend_daily.mean()
    trend_ann = trend_avg * (252/5)
    
    print(f'\n=== 蓝盾V2 结果 ===')
    print(f'交易天数: {len(daily)}')
    print(f'胜率(日): {win*100:.1f}%')
    print(f'平均5天收益: {avg*100:.2f}%')
    print(f'年化收益: {ann_ret*100:.1f}%')
    print(f'年化波动: {ann_vol*100:.1f}%')
    print(f'夏普: {sharpe:.2f}')
    print(f'最大回撤: {max_dd*100:.1f}%')
    
    print(f'\n=== 对比 ===')
    print(f'买所有趋势票(基准): 年化{trend_ann*100:.1f}%')
    print(f'蓝盾V2(ML精选): 年化{ann_ret*100:.1f}%')
    print(f'ML增益: {ann_ret - trend_ann:+.1f}%')
    print(f'目标V5.5: 年化+21.2%, 夏普3.87')
    
    top_syms = res_df.groupby('sym')['ret_f5'].agg(['count','mean']).sort_values('count', ascending=False).head(10)
    print(f'\n最常买入Top10:')
    for s, row in top_syms.iterrows():
        print(f'  {s}: {int(row["count"])}次, 平均{row["mean"]*100:.2f}%')

# ========= 保存 =========
# xgboost >= 2.1 保存方法
try:
    model.get_booster().save_model('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v2.model')
    pickle.dump(model, open('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v2.pkl', 'wb'))
    print('\n模型保存成功')
except Exception as e:
    print(f'模型保存失败: {e}')
    # fallback: 用booster
    try:
        model.get_booster().save_model('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v2.model')
        print('    但booster保存成功')
    except:
        pass

imp = pd.DataFrame({'feat': feat_cols, 'imp': model.feature_importances_}).sort_values('imp', ascending=False)
print('\nTop 10特征:')
print(imp.head(10).to_string(index=False))

meta = {
    'model': 'blueshield_v2',
    'strategy': 'trend_filter_ML_top10_hold5d',
    'features': feat_cols,
    'n_features': len(feat_cols),
    'rmse': float(rmse),
    'backtest': {
        'annual_return': float(ann_ret),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'win_rate': float(win),
        'n_trades': len(res_df),
        'n_days': len(daily),
        'trend_baseline_ann': float(trend_ann),
        'ml_gain_over_trend': float(ann_ret - trend_ann),
        'v55_target': 'annual_21.2_sharpe_3.87'
    },
    'date': time.strftime('%Y-%m-%d %H:%M')
}
json.dump(meta, open('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v2_meta.json', 'w'), indent=2)
print(f'\n完成: {time.strftime("%Y-%m-%d %H:%M")}')
