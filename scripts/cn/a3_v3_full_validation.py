"""
A3_v3 全量验证 v6（一体化版）
- 在calc_tech循环里直接查mf数据，不做后续merge
- 避免pandas pyarrow string bug
"""
import sys, os, json, gc, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE = r'/home/hermes/.hermes/openclaw-archive/data'
LOG_FILE = BASE + '/a3_v3_validation.log'

def log(msg):
    """写日志到文件，不print"""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")

# 清空旧日志
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write(f"A3_v3 全量验证 v6 开始 - {datetime.now()}\n{'='*60}\n")

# 1. 加载数据
log("[1/5] 加载数据...")
hist = pd.read_parquet(BASE + '/a_hist_10y.parquet')
mf = pd.read_parquet(BASE + '/a3_moneyflow_factors.parquet')

log(f"  K线: {len(hist)} 只股票")
log(f"  资金流: {len(mf):,} 行")

# 匹配代码格式: mf是000001.SZ, hist是000001
mf['ts_code_base'] = mf['ts_code'].str.replace(r'\.(SZ|SH|BJ)$', '', regex=True)
mf_codes = set(mf['ts_code_base'].unique())
valid_codes = [c for c in hist['ticker'].tolist() if c in mf_codes]
log(f"  有效股票: {len(valid_codes)} 只")

# 抽样1000只
np.random.seed(42)
sample_codes = list(np.random.choice(valid_codes, size=min(1000, len(valid_codes)), replace=False))
sample_set = set(sample_codes)
log(f"  抽样: {len(sample_codes)} 只")

# 过滤到抽样股票
hist_s = hist[hist['ticker'].isin(sample_set)].reset_index(drop=True)
mf_s = mf[mf['ts_code_base'].isin(sample_set)].copy()
del hist, mf; gc.collect()

# 2. 特征列
mf_cols = [c for c in mf_s.columns if c not in ['ts_code','ts_code_base','trade_date']]
tech_cols = ['pct_ma5','pct_ma10','pct_ma20','pct_ma60','pct_ma120',
    'ma20_slope','ma60_slope','vol_atr20','vol_ratio','ret_1d','ret_5d',
    'ret_10d','ret_20d','ret_60d','rsi14','macd_dif','macd_dea','macd_bar',
    'bb_width','bb_position','vol_ratio_5_20','obv_ratio_5_20',
    'kdj_k','kdj_d','kdj_j','ma5_ma10_cross','accel_5_10',
    'vol_breakout','ma_align','ret5_max','ret3_vs_ema12']
feat_cols = tech_cols + mf_cols
log(f"  特征: {len(feat_cols)} (技术{len(tech_cols)} + 资金流{len(mf_cols)})")

# 3. 预先构建mf_by_ticker dict（key=ticker, value=DataFrame with trade_date index）
log("[2/5] 构建mf lookup...")
mf_s_rename = mf_s.rename(columns={'ts_code_base': 'ticker'})
mf_s_rename['trade_date'] = mf_s_rename['trade_date'].astype(str)
mf_by_ticker = {}
for ticker, group in mf_s_rename.groupby('ticker'):
    mf_by_ticker[ticker] = group.set_index('trade_date')[mf_cols]
del mf_s, mf_s_rename; gc.collect()
log(f"  mf_by_ticker: {len(mf_by_ticker)} tickers")

# 4. 一体化计算：技术指标 + 资金流特征 + label
log("[3/5] 一体化计算...")

def calc_all(row, mf_lookup):
    """一次性计算技术+资金流+label"""
    ticker = row['ticker']
    dates_arr = row['dates']
    c = np.array(row['c'], dtype=np.float64)
    h = np.array(row['h'], dtype=np.float64)
    l = np.array(row['l'], dtype=np.float64)
    v = np.array(row['v'], dtype=np.float64)
    n = len(c)
    if n < 120:
        return None
    
    # 技术指标
    ma5 = pd.Series(c).rolling(5, min_periods=1).mean().values
    ma10 = pd.Series(c).rolling(10, min_periods=1).mean().values
    ma20 = pd.Series(c).rolling(20, min_periods=1).mean().values
    ma60 = pd.Series(c).rolling(60, min_periods=1).mean().values
    ma120 = pd.Series(c).rolling(120, min_periods=1).mean().values
    
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr20 = pd.Series(tr).rolling(20, min_periods=1).mean().values
    
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(c).ewm(span=26, adjust=False).mean().values
    macd_dif = ema12 - ema26
    macd_dea = pd.Series(macd_dif).ewm(span=9, adjust=False).mean().values
    
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean().values
    avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean().values
    rsi14 = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-8))
    
    bb_std = pd.Series(c).rolling(20, min_periods=1).std().values
    low9 = pd.Series(l).rolling(9, min_periods=1).min().values
    high9 = pd.Series(h).rolling(9, min_periods=1).max().values
    rsv = (c - low9) / (high9 - low9 + 1e-8) * 100
    kdj_k = pd.Series(rsv).ewm(com=2, adjust=False).mean().values
    kdj_d = pd.Series(kdj_k).ewm(com=2, adjust=False).mean().values
    
    obv = pd.Series(np.where(c > np.roll(c, 1), v, -v)).cumsum().values
    vol20 = pd.Series(v).rolling(20, min_periods=1).mean().values
    
    # 标签: 未来5日最大涨幅>8%
    fut_max = np.maximum.reduce([np.roll(c,-i) for i in range(1,6)])
    label = (fut_max/c - 1 > 0.08).astype(int)
    label[-5:] = -1  # 最后5天无未来数据
    
    # 构建基础DataFrame
    result = pd.DataFrame({
        'ticker': ticker,
        'trade_date': dates_arr.astype(str),
        'label': label,
        'pct_ma5': c/ma5-1, 'pct_ma10': c/ma10-1, 'pct_ma20': c/ma20-1,
        'pct_ma60': c/ma60-1, 'pct_ma120': c/ma120-1,
        'ma20_slope': ma20/np.roll(ma20,5)-1, 'ma60_slope': ma60/np.roll(ma60,10)-1,
        'vol_atr20': v/(atr20+1e-8), 'vol_ratio': v/(vol20+1e-8),
        'ret_1d': c/np.roll(c,1)-1, 'ret_5d': c/np.roll(c,5)-1,
        'ret_10d': c/np.roll(c,10)-1, 'ret_20d': c/np.roll(c,20)-1,
        'ret_60d': c/np.roll(c,60)-1,
        'rsi14': rsi14, 'macd_dif': macd_dif, 'macd_dea': macd_dea,
        'macd_bar': macd_dif - macd_dea,
        'bb_width': bb_std*4/(ma20+1e-8),
        'bb_position': (c-(ma20-bb_std*2))/(bb_std*4+1e-8),
        'vol_ratio_5_20': pd.Series(v).rolling(5,min_periods=1).mean().values/(vol20+1e-8),
        'obv_ratio_5_20': pd.Series(obv).rolling(5,min_periods=1).mean().values/(pd.Series(obv).rolling(20,min_periods=1).mean().values+1e-8),
        'kdj_k': kdj_k, 'kdj_d': kdj_d, 'kdj_j': 3*kdj_k-2*kdj_d,
        'ma5_ma10_cross': (ma5>ma10).astype(int),
        'accel_5_10': (c/ma5-1)-(c/ma10-1),
        'vol_breakout': (v > pd.Series(v).rolling(20,min_periods=1).max().values*0.8).astype(int),
        'ma_align': ((ma5>ma10)&(ma10>ma20)&(ma20>ma60)).astype(int),
        'ret5_max': np.maximum.reduce([c/np.roll(c,i)-1 for i in range(1,6)]),
        'ret3_vs_ema12': c/ema12-1,
    })
    
    # 直接查mf数据，按trade_date对齐
    if ticker in mf_lookup:
        mf_t = mf_lookup[ticker]
        # 用trade_date作为index查mf
        mf_aligned = mf_t.reindex(result['trade_date'].values)
        mf_aligned.index = result.index  # 对齐index
        for col in mf_cols:
            result[col] = mf_aligned[col].values
    
    return result

all_data = []
for i, (_, row) in enumerate(hist_s.iterrows()):
    if (i+1) % 100 == 0:
        log(f"  进度: {i+1}/{len(hist_s)}")
    data = calc_all(row, mf_by_ticker)
    if data is not None:
        all_data.append(data)

merged = pd.concat(all_data, ignore_index=True)
merged = merged.dropna(subset=feat_cols+['label'])
log(f"  合并后: {len(merged):,} 行")
del all_data, hist_s, mf_by_ticker; gc.collect()

n_pos = (merged['label']==1).sum()
n_neg = (merged['label']==0).sum()
log(f"  正样本: {n_pos:,} ({n_pos/len(merged)*100:.1f}%)")
log(f"  负样本: {n_neg:,} ({n_neg/len(merged)*100:.1f}%)")

# 5. Walk-forward验证
log("[4/5] Walk-forward 5折验证...")
reports = []
unique_dates = sorted(merged['trade_date'].unique())
n_dates = len(unique_dates)
fold_size = n_dates // 5

for fold in range(5):
    tst_start = fold * fold_size
    tst_end = min((fold+1) * fold_size, n_dates) if fold < 4 else n_dates
    dates = unique_dates
    tr_end_idx = max(0, tst_start - 1)
    tr_start_idx = max(0, tr_end_idx - fold_size)
    tr_dates = dates[tr_start_idx:tr_end_idx+1]
    te_dates = dates[tst_start:tst_end]
    tr = merged[merged['trade_date'].isin(tr_dates)]
    te = merged[merged['trade_date'].isin(te_dates)]
    if len(tr) < 1000 or len(te) < 200:
        log(f"  Fold {fold+1}: 跳过 (train={len(tr)}, test={len(te)})")
        continue
    sp = sum(tr['label']==0) / max(1, sum(tr['label']==1))
    X_tr, y_tr = tr[feat_cols].values, tr['label'].values
    X_te, y_te = te[feat_cols].values, te['label'].values
    model = lgb.LGBMClassifier(
        num_leaves=31, learning_rate=0.05, n_estimators=200,
        scale_pos_weight=sp, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, random_state=42)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], eval_metric='auc')
    y_p = (model.predict(X_te) > 0.5).astype(int)
    acc = np.mean(y_p == y_te)
    prec = np.sum((y_p==1)&(y_te==1)) / max(1, np.sum(y_p==1))
    rec = np.sum((y_p==1)&(y_te==1)) / max(1, np.sum(y_te==1))
    log(f"  Fold {fold+1}: train={len(tr):,} test={len(te):,} | acc={acc:.4f} prec={prec:.4f} rec={rec:.4f}")
    reports.append({'fold':fold+1, 'accuracy':round(acc,4), 'precision':round(prec,4), 'recall':round(rec,4)})

# 6. 保存
log("[5/5] 保存...")
model.booster_.save_model(BASE+'/models/a3_v3_full_lightgbm.txt')
imp = pd.DataFrame({'feature': feat_cols, 'importance': model.feature_importances_})
imp = imp.sort_values('importance', ascending=False).head(20)

avg_acc = np.mean([r['accuracy'] for r in reports])
avg_prec = np.mean([r['precision'] for r in reports])

# 最终输出（只print这个，给exec返回）
print(f"A3_v3 全量验证完成")
print(f"股票数: {len(sample_codes)}")
print(f"样本数: {len(merged):,}")
print(f"准确率: {avg_acc:.4f} (vs 300只: {avg_acc-0.548:+.4f}, vs V1: {avg_acc-0.49:+.4f})")
print(f"精确率: {avg_prec:.4f}")
print(f"报告: {BASE}/a3_v3_full_report.json")
print(f"日志: {LOG_FILE}")

report = {
    'model': 'a3_v3_full_lgb', 'validation': 'full_sample_1000',
    'n_stocks': len(sample_codes), 'n_samples': len(merged),
    'n_pos': n_pos, 'n_neg': n_neg, 'n_features': len(feat_cols),
    'avg_accuracy': round(avg_acc, 4), 'avg_precision': round(avg_prec, 4),
    'improvement_v1': round(avg_acc - 0.49, 4),
    'improvement_300': round(avg_acc - 0.548, 4),
    'fold_reports': reports,
    'top_features': imp['feature'].tolist()[:20],
}
with open(BASE+'/a3_v3_full_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

log(f"[完成] 准确率={avg_acc:.4f}")
print("Done!")
