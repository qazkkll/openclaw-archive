"""
A3_v3 全量验证脚本（优化版）
优化策略：不做merge，在calc_tech里直接查mf数据，一次性生成完整行
"""
import sys, os, json, gc, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE = r'/home/hermes/.hermes/openclaw-archive/data'

print("=" * 60)
print("A3_v3 全量验证（优化版：无merge）")
print("=" * 60)

# 1. 加载数据
print("\n[1/5] 加载数据...")
hist = pd.read_parquet(BASE + '/a_hist_10y.parquet')
mf = pd.read_parquet(BASE + '/a3_moneyflow_factors.parquet')
print(f"  hist: {hist.shape[0]} 只股票")
print(f"  mf: {mf.shape[0]:,} 行")

# 2. 构建mf快速查找表（向量化方式）
print("\n[2/5] 构建mf查找表...")
mf_cols = [c for c in mf.columns if c not in ['ts_code', 'trade_date']]
# 清理ticker后缀（000001.SZ → 000001）
mf['ticker_clean'] = mf['ts_code'].str.split('.').str[0]
# 向量化转dict: {ticker: {date: [features]}}
mf_dict = {}
for ticker, group in mf.groupby('ticker_clean'):
    mf_dict[ticker] = dict(zip(group['trade_date'], group[mf_cols].values.tolist()))
print(f"  已构建 {len(mf_dict):,} 只股票的查找表")

# 3. 全量股票池
pool_codes = hist['ticker'].unique().tolist()
print(f"\n[3/5] 全量股票池: {len(pool_codes)} 只")

# 特征列定义
tech_cols = ['pct_ma5','pct_ma10','pct_ma20','pct_ma60','pct_ma120',
    'ma20_slope','ma60_slope','vol_atr20','vol_ratio','ret_1d','ret_5d',
    'ret_10d','ret_20d','ret_60d','rsi14','macd_dif','macd_dea','macd_bar',
    'bb_width','bb_position','vol_ratio_5_20','obv_ratio_5_20',
    'kdj_k','kdj_d','kdj_j','ma5_ma10_cross','accel_5_10',
    'vol_breakout','ma_align','ret5_max','ret3_vs_ema12']
feat_cols = tech_cols + mf_cols

def calc_features(ticker, row):
    """一次性计算技术特征+资金流特征，不做merge"""
    dates_arr = row['dates']
    c = np.array(row['c'], dtype=np.float64)
    h = np.array(row['h'], dtype=np.float64)
    l = np.array(row['l'], dtype=np.float64)
    o = np.array(row['o'], dtype=np.float64)
    v = np.array(row['v'], dtype=np.float64)
    n = len(c)
    if n < 140:
        return None
    
    # 技术指标计算（和原版一样）
    def _ma(arr, w):
        cs = np.cumsum(arr)
        res = np.full(n, np.nan)
        res[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
        return res
    
    ma5 = _ma(c, 5); ma10 = _ma(c, 10); ma20 = _ma(c, 20)
    ma60 = _ma(c, 60); ma120 = _ma(c, 120)
    
    pct_ma5 = c / ma5 - 1
    pct_ma10 = c / ma10 - 1
    pct_ma20 = c / ma20 - 1
    pct_ma60 = c / ma60 - 1
    pct_ma120 = c / ma120 - 1
    
    ma20_slope = ma20 / np.roll(ma20, 5) - 1
    ma60_slope = ma60 / np.roll(ma60, 10) - 1
    
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr20 = _ma(tr, 20)
    vol_atr20 = v / (atr20 * ma20 + 1e-10)
    
    vol_ma5 = _ma(v, 5); vol_ma20 = _ma(v, 20)
    vol_ratio = v / (vol_ma20 + 1e-10)
    
    ret_1d = c / np.roll(c, 1) - 1; ret_1d[0] = 0
    ret_5d = c / np.roll(c, 5) - 1; ret_5d[:5] = 0
    ret_10d = c / np.roll(c, 10) - 1; ret_10d[:10] = 0
    ret_20d = c / np.roll(c, 20) - 1; ret_20d[:20] = 0
    ret_60d = c / np.roll(c, 60) - 1; ret_60d[:60] = 0
    
    delta = c - np.roll(c, 1); delta[0] = 0
    gain = np.where(delta > 0, delta, 0); loss = np.where(delta < 0, -delta, 0)
    avg_gain = _ma(gain, 14); avg_loss = _ma(loss, 14)
    rs = avg_gain / (avg_loss + 1e-10)
    rsi14 = 100 - 100 / (1 + rs)
    
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(c).ewm(span=26, adjust=False).mean().values
    macd_dif = ema12 - ema26
    macd_dea = pd.Series(macd_dif).ewm(span=9, adjust=False).mean().values
    macd_bar = 2 * (macd_dif - macd_dea)
    
    bb_ma = ma20; bb_std = pd.Series(c).rolling(20, min_periods=20).std().values
    bb_up = bb_ma + 2 * bb_std; bb_dn = bb_ma - 2 * bb_std
    bb_width = (bb_up - bb_dn) / (bb_ma + 1e-10)
    bb_position = (c - bb_dn) / (bb_up - bb_dn + 1e-10)
    
    vol_ratio_5_20 = vol_ma5 / (vol_ma20 + 1e-10)
    obv = np.cumsum(np.where(c > np.roll(c, 1), v, -v) * (c != np.roll(c, 1)))
    obv_ma5 = _ma(obv, 5); obv_ma20 = _ma(obv, 20)
    obv_ratio_5_20 = obv_ma5 / (obv_ma20 + 1e-10)
    
    low9 = np.minimum.reduce([np.roll(l, i) for i in range(10)])
    low9[:10] = np.minimum.reduce([l[:10]])
    high9 = np.maximum.reduce([np.roll(h, i) for i in range(10)])
    high9[:10] = np.maximum.reduce([h[:10]])
    rsv = (c - low9) / (high9 - low9 + 1e-10) * 100
    kdj_k = np.full(n, np.nan); kdj_k[8] = 50
    for i in range(9, n): kdj_k[i] = 2/3 * kdj_k[i-1] + 1/3 * rsv[i]
    kdj_d = np.full(n, np.nan); kdj_d[8] = 50
    for i in range(9, n): kdj_d[i] = 2/3 * kdj_d[i-1] + 1/3 * kdj_k[i]
    kdj_j = 3 * kdj_k - 2 * kdj_d
    
    ma5_ma10_cross = np.where((ma5 > ma10) & (np.roll(ma5, 1) <= np.roll(ma10, 1)), 1,
                    np.where((ma5 < ma10) & (np.roll(ma5, 1) >= np.roll(ma10, 1)), -1, 0))
    accel_5_10 = (ma5 - ma10) / (np.abs(ma10) + 1e-10)
    vol_breakout = np.where(v > np.roll(pd.Series(v).rolling(20, min_periods=20).max().values, 1), 1, 0)
    ma_align = np.where((ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60), 1,
                np.where((ma5 < ma10) & (ma10 < ma20) & (ma20 < ma60), -1, 0))
    ret5_max = pd.Series(ret_5d).rolling(5, min_periods=1).max().values
    ret3_vs_ema12 = (c - ema12) / (np.abs(ema12) + 1e-10)
    
    # 组装技术特征
    tech_data = np.column_stack([
        pct_ma5, pct_ma10, pct_ma20, pct_ma60, pct_ma120,
        ma20_slope, ma60_slope, vol_atr20, vol_ratio, ret_1d, ret_5d,
        ret_10d, ret_20d, ret_60d, rsi14, macd_dif, macd_dea, macd_bar,
        bb_width, bb_position, vol_ratio_5_20, obv_ratio_5_20,
        kdj_k, kdj_d, kdj_j, ma5_ma10_cross, accel_5_10,
        vol_breakout, ma_align, ret5_max, ret3_vs_ema12
    ])
    
    # 查mf数据，组装资金流特征
    mf_lookup = mf_dict.get(ticker, {})
    mf_data = np.zeros((n, len(mf_cols)))
    for i, date in enumerate(dates_arr):
        if date in mf_lookup:
            mf_data[i] = mf_lookup[date]
    
    # 合并技术+资金流
    features = np.hstack([tech_data, mf_data])
    
    # 构建结果DataFrame
    result = pd.DataFrame(features, columns=feat_cols)
    result['ticker'] = ticker
    result['trade_date'] = dates_arr
    result['close'] = c
    result['label'] = (np.roll(c, -5) / c - 1 > 0.05).astype(int)
    result.loc[n-5:, 'label'] = 0  # 最后5天不算
    
    return result

# 4. 处理所有股票
print("\n[4/5] 处理股票（无merge，直接查mf）...")
all_dfs = []
processed = 0
skipped = 0

for idx, row in hist.iterrows():
    ticker = row['ticker']
    try:
        df = calc_features(ticker, row)
        if df is not None:
            all_dfs.append(df)
            processed += 1
            if processed % 500 == 0:
                print(f"  已处理 {processed}/{len(pool_codes)} 只股票...")
        else:
            skipped += 1
    except Exception as e:
        skipped += 1
        if skipped <= 5:
            print(f"  跳过 {ticker}: {e}")

print(f"  完成: 成功 {processed} 只, 跳过 {skipped} 只")

# 5. 合并并验证
print("\n[5/5] Walk-Forward验证...")
train = pd.concat(all_dfs, ignore_index=True)
del all_dfs; gc.collect()

n_pos = (train['label'] == 1).sum()
n_neg = (train['label'] == 0).sum()
print(f"  总样本: {len(train):,} (正样本: {n_pos:,}, 负样本: {n_neg:,})")

# Walk-Forward
import lightgbm as lgb
all_dates = sorted(train['trade_date'].unique())
n_dates = len(all_dates)
fold_size = n_dates // 5
reports = []

for fold in range(5):
    tst_start = fold * fold_size
    tst_end = (fold + 1) * fold_size if fold < 4 else n_dates
    tst_dates = set(all_dates[tst_start:tst_end])
    trn_dates = set(all_dates[:tst_start])
    
    tr = train[train['trade_date'].isin(trn_dates)]
    te = train[train['trade_date'].isin(tst_dates)]
    
    if len(tr) < 500 or len(te) < 100:
        print(f"  Fold {fold+1}: 数据不足，跳过")
        continue
    
    scale_pos = sum(tr['label']==0) / max(1, sum(tr['label']==1))
    X_tr = tr[feat_cols].fillna(0).values
    y_tr = tr['label'].values
    X_te = te[feat_cols].fillna(0).values
    y_te = te['label'].values
    
    model = lgb.LGBMClassifier(
        num_leaves=31, learning_rate=0.05, n_estimators=200,
        scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, random_state=42
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], eval_metric='auc')
    
    y_p = (model.predict(X_te) > 0.5).astype(int)
    acc = np.mean(y_p == y_te)
    prec = np.sum((y_p==1)&(y_te==1)) / max(1, np.sum(y_p==1))
    rec = np.sum((y_p==1)&(y_te==1)) / max(1, np.sum(y_te==1))
    
    print(f"  Fold {fold+1}: train={len(tr):,} test={len(te):,} | acc={acc:.4f} prec={prec:.4f} rec={rec:.4f}")
    reports.append({'fold':fold+1, 'accuracy':round(acc,4), 'precision':round(prec,4), 'recall':round(rec,4)})

# 保存模型
model.booster_.save_model(BASE+'/models/a3_v3_lightgbm_full.txt')

# 特征重要性
imp = pd.DataFrame({'feature': feat_cols, 'importance': model.feature_importances_})
imp = imp.sort_values('importance', ascending=False).head(20)
print(f"\nTop 20 features:")
print(imp.to_string(index=False))

avg_acc = np.mean([r['accuracy'] for r in reports])
avg_prec = np.mean([r['precision'] for r in reports])
print(f"\n=== A3_v3 全量 WF Average ===")
print(f"Accuracy: {avg_acc:.4f} (V1 baseline: 0.49, improvement: {avg_acc-0.49:+.4f})")
print(f"Precision: {avg_prec:.4f}")

report = {
    'model': 'a3_v3_lgb_full',
    'n_stocks': processed,
    'n_skipped': skipped,
    'n_samples': len(train),
    'n_pos': int(n_pos),
    'n_neg': int(n_neg),
    'n_features': len(feat_cols),
    'tech_features': len(tech_cols),
    'mf_features': len(mf_cols),
    'avg_accuracy': round(avg_acc, 4),
    'avg_precision': round(avg_prec, 4),
    'improvement_v1': round(avg_acc - 0.49, 4),
    'fold_reports': reports,
    'top_features': imp['feature'].tolist()[:20],
    'v1_baseline': 0.49,
    'optimization': 'no_merge_direct_lookup'
}

with open(BASE+'/a3_v3_full_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\nReport saved: {BASE+'/a3_v3_full_report.json'}")
print("Done!")
