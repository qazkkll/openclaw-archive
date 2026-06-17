#!/usr/bin/env python3
"""
蓝盾 V1（Blue Shield）
SP500大盘ML引擎 — 回归预测+规则混合
目标：超越V5.5基准（年化+21.2%, 夏普3.87）

流程：
1. 规则初筛（成交额+趋势）
2. ML回归—预测未来5天绝对收益%
3. 综合排名（ML×70% + 趋势×30%）
4. Top 15建仓，持有5天，每日轮换
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

# 行情原始数据（用于成交额）
import glob
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

# 合并成交额
feat = feat.merge(raw_df[['Code','Date','DollarVol']], on=['Code','Date'], how='left')

# ========= 特征列 =========
feat_cols = ['ret_1d','ret_3d','ret_5d','ret_10d','ret_20d',
             'ma_5_ratio','ma_10_ratio','ma_20_ratio','ma_50_ratio',
             'vol_5d','vol_10d','vol_20d','rsi_14',
             'vol_ratio_5','vol_ratio_20',
             'price_pos_20','price_pos_50','price_pos_100',
             'macd','macd_sig','macd_hist','atr_pct']

# 加入相对强度
feat['ma20_ma50_cross'] = feat['ma_20_ratio'] - feat['ma_50_ratio']
feat['ret_5d_vol'] = feat['ret_5d'] / (feat['vol_5d'] + 1e-8)

# SP500均值
market_ret = feat.groupby('Date')['ret_1d'].mean().reset_index()
market_ret.columns = ['Date', 'market_ret']
feat = feat.merge(market_ret, on='Date', how='left')
feat['rel_ret_5d'] = feat['ret_5d'] - feat.groupby('Date')['ret_5d'].transform('mean')
feat['rel_ret_10d'] = feat['ret_10d'] - feat.groupby('Date')['ret_10d'].transform('mean')

# 成交额特征
feat['dvol_ma5'] = feat.groupby('Code')['DollarVol'].transform(lambda x: x.rolling(5).mean())
feat['dvol_ratio'] = np.where(feat['dvol_ma5'] > 0, feat['DollarVol'] / feat['dvol_ma5'], 1.0)

all_feats = feat_cols + ['ma20_ma50_cross','ret_5d_vol','rel_ret_5d','rel_ret_10d','dvol_ratio']

print(f'特征列数: {len(all_feats)}')
valid = feat.dropna(subset=all_feats + ['ret_f5']).copy()
print(f'有效样本: {len(valid)}')

# ========= 按时间分集 =========
dates = pd.Series(valid['Date'].unique()).sort_values().values
# 5年数据: 80%训练/10%验证/10%测试
n = len(dates)
train_end = dates[int(n*0.7)]  # 70%训练
val_end = dates[int(n*0.85)]   # 15%验证

train_mask = valid['Date'] <= train_end
val_mask = (valid['Date'] > train_end) & (valid['Date'] <= val_end)
test_mask = valid['Date'] > val_end

print(f'训练: {train_mask.sum()}  验证: {val_mask.sum()}  测试: {test_mask.sum()}')
print(f'训练截止: {str(train_end)[:10]}  验证截止: {str(val_end)[:10]}')

X_train = valid.loc[train_mask, all_feats].values
y_train = valid.loc[train_mask, 'ret_f5'].values
X_val = valid.loc[val_mask, all_feats].values
y_val = valid.loc[val_mask, 'ret_f5'].values
X_test = valid.loc[test_mask, all_feats].values
y_test = valid.loc[test_mask, 'ret_f5'].values

# ========= 训练回归XGBoost =========
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

# ========= 测试表现 =========
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f'\n=== 回归表现 ===')
print(f'RMSE: {rmse:.4f} ({rmse*100:.2f}%)')
print(f'MAE: {mae:.4f} ({mae*100:.2f}%)')
print(f'实际收益范围: {y_test.min()*100:.1f}%~{y_test.max()*100:.1f}%')
print(f'实际收益均值: {y_test.mean()*100:.2f}%')

# ========= 回测：选择Top N =========
print('\n=== 回测：每日选Top买入 ===')
test_df = valid[test_mask].copy()
test_df['pred'] = y_pred

# 规则过滤：趋势向上
def trend_filter(g):
    g = g.sort_values('Date')
    g['ma20'] = g['ret_f5'].rolling(20).mean()  # 近似
    return g

# 模拟：每日选Top 15
dates_list = sorted(test_df['Date'].unique())
results = []

for di, d in enumerate(dates_list):
    day_df = test_df[test_df['Date'] == d]
    
    # 规则过滤1: 成交额≥10M
    day_df = day_df[day_df['dvol_ma5'] >= 10_000_000]
    if len(day_df) < 5:
        continue
    
    # 规则过滤2: 趋势向上 (ma20 > ma50 ratio)
    day_df = day_df[day_df['ma20_ma50_cross'] > -0.02]
    if len(day_df) < 5:
        continue
    
    # 规则过滤3: 相对强度前一半
    median_rel = day_df['rel_ret_10d'].median()
    day_df = day_df[day_df['rel_ret_10d'] >= median_rel]
    if len(day_df) < 5:
        continue
    
    # 综合排名（ML得分×70% + 趋势得分×30%）
    # 归一化ML得分
    p_min, p_max = day_df['pred'].min(), day_df['pred'].max()
    if p_max > p_min:
        day_df['ml_score'] = (day_df['pred'] - p_min) / (p_max - p_min)
    else:
        day_df['ml_score'] = 0.5
    
    # 趋势得分 = 最近的5日收益（越高越好）
    t_min, t_max = day_df['ret_5d'].min(), day_df['ret_5d'].max()
    if t_max > t_min:
        day_df['trend_score'] = (day_df['ret_5d'] - t_min) / (t_max - t_min)
    else:
        day_df['trend_score'] = 0.5
    
    day_df['composite'] = day_df['ml_score'] * 0.7 + day_df['trend_score'] * 0.3
    
    # Top 15
    top = day_df.nlargest(15, 'composite')
    
    # 持有5天：用已有的标签
    for _, row in top.iterrows():
        idx = row.name
        if idx + 5 < len(valid):
            ret_5d_actual = valid.iloc[idx + 5]['ret_f5'] if idx + 5 < len(valid) else 0
        else:
            ret_5d_actual = 0
        results.append({
            'sym': row['Code'],
            'date': str(d)[:10],
            'pred': round(row['pred'], 4),
            'composite': round(row['composite'], 4),
            'ml_score': round(row['ml_score'], 4),
            'ret_5d_actual': round(ret_5d_actual, 4),
        })

res_df = pd.DataFrame(results)
print(f'交易: {len(res_df)}次')

if len(res_df) > 0:
    # 等权：每天买入15只，5天后卖出
    # 每天总收益 = 15只收益均值
    daily_returns = res_df.groupby('date')['ret_5d_actual'].mean().reset_index()
    # 但ret_f5是5日收益，要年化
    daily_ret = daily_returns['ret_5d_actual']
    
    total_days = len(daily_ret)
    n_trades = len(res_df)
    win_pct = (daily_ret > 0).mean()
    avg_ret = daily_ret.mean()
    vol = daily_ret.std()
    
    # 年化
    ann_ret = avg_ret * (252/5)  # 5天持仓换算成年
    ann_vol = vol * np.sqrt(252/5)
    sharpe = ann_ret / max(ann_vol, 0.001)
    
    # 回撤
    cum = (1 + daily_ret).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum / rolling_max - 1)
    max_dd = drawdown.min()
    
    print(f'\n=== 蓝盾V1 回测结果（测试集）===')
    print(f'交易天数: {total_days}')
    print(f'交易总笔: {n_trades}')
    print(f'胜率(日): {win_pct*100:.1f}%')
    print(f'平均5天收益/笔: {avg_ret*100:.2f}%')
    print(f'年化收益: {ann_ret*100:.1f}%')
    print(f'年化波动: {ann_vol*100:.1f}%')
    print(f'夏普: {sharpe:.2f}')
    print(f'最大回撤: {max_dd*100:.1f}%')
    
    # Top10票分布
    top_syms = res_df.groupby('sym').size().sort_values(ascending=False).head(10)
    print(f'\n最常买入Top10:')
    for s, c in top_syms.items():
        avg_r = res_df[res_df['sym']==s]['ret_5d_actual'].mean()
        print(f'  {s}: {c}次, 平均5天收益{avg_r*100:.2f}%')

# ========= 保存 =========
print('\n保存模型...')
model.save_model('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v1.model')
pickle.dump(model, open('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v1.pkl', 'wb'))

# 特征重要性
imp = pd.DataFrame({'feat': all_feats, 'imp': model.feature_importances_})
imp = imp.sort_values('imp', ascending=False)
print('\nTop 10特征:')
print(imp.head(10).to_string(index=False))

# 元数据
meta = {
    'model': 'blueshield_v1', 'type': 'regression',
    'features': all_feats, 'n_features': len(all_feats),
    'n_train': len(X_train), 'n_val': len(X_val), 'n_test': len(X_test),
    'rmse': float(rmse), 'mae': float(mae),
    'backtest': {
        'annual_return': float(ann_ret) if len(res_df) > 0 else 0,
        'sharpe': float(sharpe) if len(res_df) > 0 else 0,
        'max_drawdown': float(max_dd) if len(res_df) > 0 else 0,
        'win_rate': float(win_pct) if len(res_df) > 0 else 0,
        'n_trades': n_trades,
        'strategy': 'ML70_trend30_top15_hold5d',
        'target': 'V5.5: +21.2%ann, 3.87sh, 28.2%dd'
    },
    'date': time.strftime('%Y-%m-%d %H:%M')
}
json.dump(meta, open('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v1_meta.json', 'w'), indent=2)
print(f'\n保存: blueshield_v1_meta.json')
print(f'完成: {time.strftime("%Y-%m-%d %H:%M")}')
