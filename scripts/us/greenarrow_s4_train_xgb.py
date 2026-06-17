#!/usr/bin/env python3
"""
绿箭 S4：XGBoost训练 v2 — 修复保存 + 加入SP500相对强度特征
"""
import json, warnings, os, sys, time
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print('加载特征...')
feat = pd.read_parquet('/home/hermes/.hermes/openclaw-project/data/us/sp500_feats.parquet')
feat = feat.sort_values(['Code', 'Date']).reset_index(drop=True)

# ========= 加入相对强度特征 =========
print('计算相对强度特征...')
# 计算SP500指数(等权或市值权)的每日涨跌
# 按日期取所有股票收盘价均值作为"市场基准"
market_avg = feat.groupby('Date')['ret_1d'].mean().reset_index()
market_avg.columns = ['Date', 'market_ret_1d']
feat = feat.merge(market_avg, on='Date', how='left')

# 相对强度
feat['rel_strength_1d'] = feat['ret_1d'] - feat['market_ret_1d']
feat['rel_strength_5d'] = feat['ret_5d'] - feat.groupby('Date')['ret_5d'].transform('mean')
feat['rel_strength_10d'] = feat['ret_10d'] - feat.groupby('Date')['ret_10d'].transform('mean')
feat['rel_strength_20d'] = feat['ret_20d'] - feat.groupby('Date')['ret_20d'].transform('mean')

# 加入动量加速度
feat['momentum_accel'] = feat['ret_5d'] - feat['ret_10d']
feat['momentum_accel_10'] = feat['ret_10d'] - feat['ret_20d']

# 特征列
feat_cols = ['ret_1d','ret_3d','ret_5d','ret_10d','ret_20d',
             'ma_5_ratio','ma_10_ratio','ma_20_ratio','ma_50_ratio',
             'vol_5d','vol_10d','vol_20d','rsi_14',
             'vol_ratio_5','vol_ratio_20',
             'price_pos_20','price_pos_50','price_pos_100',
             'macd','macd_sig','macd_hist','atr_pct',
             'rel_strength_1d','rel_strength_5d','rel_strength_10d',
             'rel_strength_20d',
             'momentum_accel','momentum_accel_10']

valid = feat.dropna(subset=feat_cols).copy()
print(f'有效样本: {len(valid)}')

# 分训练/测试
valid['Date'] = pd.to_datetime(valid['Date'])
dates = pd.Series(valid['Date'].unique()).sort_values().values
split_idx = int(len(dates) * 0.8)
train_dates = dates[:split_idx]
test_dates = dates[split_idx:]

train_mask = valid['Date'].isin(train_dates)
test_mask = valid['Date'].isin(test_dates)

X_train = valid.loc[train_mask, feat_cols].values
y_train = valid.loc[train_mask, 'label_buy'].values
X_test = valid.loc[test_mask, feat_cols].values
y_test = valid.loc[test_mask, 'label_buy'].values

print(f'Train: {len(X_train)}  Test: {len(X_test)}')

# ========= 训练 =========
print('\n训练XGBoost...')
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6,
    scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
    eval_metric='auc', early_stopping_rounds=30,
    random_state=42, n_jobs=-1
)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          verbose=50)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.35).astype(int)

auc = roc_auc_score(y_test, y_prob)
print(f'\n=== AUC: {auc:.4f} ===')
print(classification_report(y_test, y_pred, target_names=['Hold','Buy']))

# 概率门槛分析
print('\n概率门槛分析:')
for th in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    p = (y_prob > th).astype(int)
    n_sig = int(p.sum())
    prec = ((p == 1) & (y_test == 1)).sum() / max(n_sig, 1)
    rec = ((p == 1) & (y_test == 1)).sum() / max(int((y_test == 1).sum()), 1)
    print(f'  >{th:.2f}: {n_sig}次 ({prec*100:.1f}%精确, {rec*100:.1f}%召回)')

# ========= 回测 =========
print('\n=== 回测: 概率>0.35买入, 持有10天 ===')
test_df = valid[test_mask].copy()
test_df['prob'] = y_prob
test_df['pred'] = y_pred

trades = []
for sym in test_df['Code'].unique():
    sdf = test_df[test_df['Code'] == sym].sort_values('Date').reset_index(drop=True)
    cols = sdf.columns.tolist()
    for i, row in sdf.iterrows():
        if row['prob'] <= 0.35:
            continue
        # 用未来5天的收益作为替代度量
        ret_f5 = sdf.iloc[i]['ret_f5'] if 'ret_f5' in cols else 0
        trades.append({
            'sym': sym, 'date': str(sdf.iloc[i]['Date'])[:10],
            'prob': round(row['prob'], 4), 'ret_f5': round(ret_f5, 4)
        })
        trades.append({
            'sym': sym, 'date': str(sdf.iloc[i]['Date'])[:10],
            'prob': round(row['prob'], 4), 'ret_10': round(ret_10, 4),
            'ret_f5': round(row['ret_f5'], 4)
        })

trade_df = pd.DataFrame(trades)
print(f'交易: {len(trade_df)}次')
if len(trade_df) > 0:
    win = (trade_df['ret_10'] > 0).mean()
    avg_ret = trade_df['ret_10'].mean()
    print(f'  胜率: {win*100:.1f}%')
    print(f'  平均10天收益: {avg_ret*100:.2f}%')
    print(f'  中位10天收益: {trade_df["ret_10"].median()*100:.2f}%')
    print(f'  最大盈利: {trade_df["ret_10"].max()*100:.2f}%')
    print(f'  最大亏损: {trade_df["ret_10"].min()*100:.2f}%')

# 特征重要性
imp_df = pd.DataFrame({'feat': feat_cols, 'imp': model.feature_importances_})
imp_df = imp_df.sort_values('imp', ascending=False)
print('\nTop15特征:')
print(imp_df.head(15).to_string(index=False))

# 保存模型 — 修复
import pickle
try:
    # xgboost的Booster可以保存
    model.get_booster().save_model('/home/hermes/.hermes/openclaw-project/data/models/greenarrow_v1.model')
    # 也保存整个scikit-learn对象
    pickle.dump(model, open('/home/hermes/.hermes/openclaw-project/data/models/greenarrow_v1.pkl', 'wb'))
    print(f'模型保存: greenarrow_v1.pkl + .model')
except Exception as e:
    print(f'模型保存部分失败: {e}')
print(f'\n模型保存: greenarrow_v1.pkl')

# 保存元数据
meta = {
    'model': 'greenarrow_v1', 'features': feat_cols,
    'n_train': len(X_train), 'n_test': len(X_test),
    'auc': float(auc), 'date': time.strftime('%Y-%m-%d %H:%M'),
    'train_buy_pct': float(y_train.mean()*100),
    'test_buy_pct': float(y_test.mean()*100),
    'n_trades': len(trade_df),
    'win_rate': float(win) if len(trade_df) > 0 else 0,
    'avg_10d_ret': float(avg_ret) if len(trade_df) > 0 else 0,
}
json.dump(meta, open('/home/hermes/.hermes/openclaw-project/data/models/greenarrow_v1_meta.json', 'w'), indent=2)
print(f'\n完成: {time.strftime("%Y-%m-%d %H:%M")}')
