#!/usr/bin/env python3
"""
A股截面排名模型 Walk-Forward验证
Plan A: XGBoost回归 (预测收益率)
Plan B: LightGBM lambdarank (预测分位数排名)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import spearmanr
import json
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/home/hermes/.hermes/openclaw-archive/data'
OUTPUT_DIR = '/home/hermes/.hermes/openclaw-archive/research'

print("=" * 60)
print("A股截面排名模型 Walk-Forward验证")
print("=" * 60)

# ===== 1. 加载数据 =====
print("\n[1] 加载数据...")
t0 = time.time()

df_hist = pd.read_parquet(f'{DATA_DIR}/a_hist_10y.parquet')
df_hist = df_hist.rename(columns={
    'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high',
    'L': 'low', 'C': 'close', 'V': 'volume'
})
df_hist['date'] = pd.to_datetime(df_hist['date'].astype(str), format='%Y%m%d')
print(f"  历史: {len(df_hist):,} 行, {df_hist['sym'].nunique()} 股, {df_hist['date'].min().date()}~{df_hist['date'].max().date()}")

df_mf = pd.read_parquet(f'{DATA_DIR}/moneyflow_core.parquet')
df_mf['sym'] = df_mf['ts_code'].str.replace(r'\.\w+$', '', regex=True)
df_mf['date'] = pd.to_datetime(df_mf['trade_date'].astype(str), format='%Y%m%d')
df_mf['sm_net'] = df_mf['buy_sm_amount'] - df_mf['sell_sm_amount']
df_mf['md_net'] = df_mf['buy_md_amount'] - df_mf['sell_md_amount']
df_mf['lg_net'] = df_mf['buy_lg_amount'] - df_mf['sell_lg_amount']
df_mf['elg_net'] = df_mf['buy_elg_amount'] - df_mf['sell_elg_amount']
df_mf['total_net'] = df_mf['net_mf_amount']
mf_cols = ['sym', 'date', 'sm_net', 'md_net', 'lg_net', 'elg_net', 'total_net']
df_mf = df_mf[mf_cols].drop_duplicates(subset=['sym', 'date'])
print(f"  资金流: {len(df_mf):,} 行, {df_mf['sym'].nunique()} 股")

# ===== 2. 合并 =====
print("\n[2] 合并数据...")
df = pd.merge(df_hist, df_mf, on=['sym', 'date'], how='inner')
df = df.sort_values(['sym', 'date']).reset_index(drop=True)
df = df[df['close'] > 0]
print(f"  合并后: {len(df):,} 行, {df['sym'].nunique()} 股")

# ===== 3. 特征工程（向量化，不做groupby.apply）=====
print("\n[3] 特征工程...")

# 按sym, date排序（已排好）
# 用shift + rolling向量化计算

# 价格变化率
df['r1'] = df.groupby('sym')['close'].pct_change(1)
df['r5'] = df.groupby('sym')['close'].pct_change(5)
df['r10'] = df.groupby('sym')['close'].pct_change(10)
df['r20'] = df.groupby('sym')['close'].pct_change(20)

# 均线偏离
df['ma5'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(5).mean())
df['ma10'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(10).mean())
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20).mean())
df['d5'] = (df['close'] - df['ma5']) / df['ma5']
df['d10'] = (df['close'] - df['ma10']) / df['ma10']
df['d20'] = (df['close'] - df['ma20']) / df['ma20']

# 波动率
df['vol5'] = df.groupby('sym')['r1'].transform(lambda x: x.rolling(5).std())
df['vol20'] = df.groupby('sym')['r1'].transform(lambda x: x.rolling(20).std())
df['high_low'] = df['high'] - df['low']
df['atr'] = df.groupby('sym')['high_low'].transform(lambda x: x.rolling(14).mean())
df['atr_pct'] = df['atr'] / df['close']

# 成交量比
df['vol_ma5'] = df.groupby('sym')['volume'].transform(lambda x: x.rolling(5).mean())
df['vol_ma20'] = df.groupby('sym')['volume'].transform(lambda x: x.rolling(20).mean())
df['vol_ratio'] = df['vol_ma5'] / (df['vol_ma20'] + 1)

# RSI
delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
# 需要按sym分组rolling
df['_gain'] = gain
df['_loss'] = loss
df['_avg_gain'] = df.groupby('sym')['_gain'].transform(lambda x: x.rolling(14).mean())
df['_avg_loss'] = df.groupby('sym')['_loss'].transform(lambda x: x.rolling(14).mean())
df['rsi14'] = 100 - (100 / (1 + df['_avg_gain'] / (df['_avg_loss'] + 1e-10)))

# MACD
df['ema12'] = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=12).mean())
df['ema26'] = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=26).mean())
df['macd'] = df['ema12'] - df['ema26']
df['macd_signal'] = df.groupby('sym')['macd'].transform(lambda x: x.ewm(span=9).mean())
df['macd_hist'] = df['macd'] - df['macd_signal']

# 资金流rolling
for col in ['sm_net', 'md_net', 'lg_net', 'elg_net', 'total_net']:
    df[f'{col}_5'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5).sum())
    df[f'{col}_20'] = df.groupby('sym')[col].transform(lambda x: x.rolling(20).sum())

# 标签
df['fwd_ret_20'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-20) / x - 1)

print(f"  特征计算完成: {len(df):,} 行")

# 特征列表
feature_cols = [
    'r1', 'r5', 'r10', 'r20',
    'd5', 'd10', 'd20',
    'vol5', 'vol20', 'atr_pct', 'vol_ratio',
    'rsi14', 'macd', 'macd_signal', 'macd_hist',
    'sm_net_5', 'sm_net_20',
    'md_net_5', 'md_net_20',
    'lg_net_5', 'lg_net_20',
    'elg_net_5', 'elg_net_20',
    'total_net_5', 'total_net_20',
]

available_features = [f for f in feature_cols if f in df.columns]
print(f"  可用特征: {len(available_features)}")

df = df.dropna(subset=available_features + ['fwd_ret_20'])
print(f"  有效样本: {len(df):,} 行, {df['sym'].nunique()} 股")

# ===== 4. Walk-Forward =====
print("\n[4] Walk-Forward验证...")

min_date, max_date = df['date'].min(), df['date'].max()
time_splits = []
cursor = min_date
while True:
    ts = cursor
    te = ts + pd.DateOffset(years=1, months=6)
    vs = te + pd.Timedelta(days=1)
    ve = vs + pd.DateOffset(months=6)
    if ve > max_date:
        ve = max_date
        if vs < max_date:
            time_splits.append((ts, te, vs, ve))
        break
    time_splits.append((ts, te, vs, ve))
    cursor += pd.DateOffset(years=2)

print(f"  {len(time_splits)} 个时间段")
for i, (ts, te, vs, ve) in enumerate(time_splits):
    print(f"    {i+1}. 训练 {ts.date()}~{te.date()} | 测试 {vs.date()}~{ve.date()}")

results_a, results_b = [], []

for idx, (tr_s, tr_e, te_s, te_e) in enumerate(time_splits):
    print(f"\n  --- Fold {idx+1}/{len(time_splits)} ---")
    
    train = df[(df['date'] >= tr_s) & (df['date'] <= tr_e)].copy()
    test = df[(df['date'] >= te_s) & (df['date'] <= te_e)].copy()
    print(f"  训练: {len(train):,} | 测试: {len(test):,}")
    
    if len(train) < 50000 or len(test) < 10000:
        print("  跳过")
        continue
    
    X_tr = train[available_features].values
    y_tr = train['fwd_ret_20'].values
    X_te = test[available_features].values
    y_te = test['fwd_ret_20'].values
    
    # Plan A: XGBoost
    print("  [A] XGBoost...", end='', flush=True)
    ta = time.time()
    ma = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=4, random_state=42, verbosity=0
    )
    ma.fit(X_tr, y_tr)
    pa = ma.predict(X_te)
    print(f" {time.time()-ta:.0f}s")
    
    # Plan B: LightGBM lambdarank
    print("  [B] LightGBM...", end='', flush=True)
    tb = time.time()
    
    # 截面排名标签
    def rank_labels(sub_df):
        labels = np.zeros(len(sub_df), dtype=np.int32)
        dates = sub_df['date'].values
        unique_dates = np.unique(dates)
        for d in unique_dates:
            mask = dates == d
            vals = sub_df.loc[mask, 'fwd_ret_20'].values
            if len(vals) < 5:
                labels[mask] = 2
                continue
            qs = np.percentile(vals, [20, 40, 60, 80])
            labels[mask] = np.digitize(vals, qs)
        return labels
    
    y_tr_r = rank_labels(train)
    y_te_r = rank_labels(test)
    
    tr_grp = train.groupby('date').size().values
    te_grp = test.groupby('date').size().values
    
    ds_tr = lgb.Dataset(X_tr, label=y_tr_r, group=tr_grp)
    ds_te = lgb.Dataset(X_te, label=y_te_r, group=te_grp, reference=ds_tr)
    
    params = {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [10], 'device': 'cpu',
        'num_leaves': 31, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 5, 'verbose': -1, 'n_jobs': 4,
    }
    mb = lgb.train(params, ds_tr, 300, valid_sets=[ds_te], callbacks=[lgb.log_evaluation(0)])
    pb = mb.predict(X_te)
    print(f" {time.time()-tb:.0f}s")
    
    # 评估
    def evaluate(sub_df, pred, name):
        sub_df = sub_df.copy()
        sub_df['pred'] = pred
        ics, rics = [], []
        for d in sub_df['date'].unique():
            dd = sub_df[sub_df['date'] == d]
            if len(dd) < 30:
                continue
            ics.append(np.corrcoef(dd['fwd_ret_20'], dd['pred'])[0, 1])
            rics.append(spearmanr(dd['fwd_ret_20'], dd['pred'])[0])
        
        ic, ric = np.nanmean(ics), np.nanmean(rics)
        ic_std, ric_std = np.nanstd(ics), np.nanstd(rics)
        
        sub_df['pct'] = sub_df.groupby('date')['pred'].rank(pct=True)
        top = sub_df[sub_df['pct'] >= 0.9]['fwd_ret_20'].mean()
        bot = sub_df[sub_df['pct'] <= 0.1]['fwd_ret_20'].mean()
        ls = top - bot
        
        top_d = sub_df[sub_df['pct'] >= 0.9].groupby('date')['fwd_ret_20'].mean()
        if len(top_d) > 5:
            cum = (1 + top_d).cumprod()
            dd_val = (cum / cum.cummax() - 1).min()
            ann = cum.iloc[-1] ** (252 / max(len(cum), 1)) - 1
            calmar = ann / abs(dd_val) if dd_val != 0 else 0
        else:
            ann, dd_val, calmar = 0, 0, 0
        
        return {
            'model': name,
            'ic': ic, 'ic_std': ic_std, 'icir': ic / (ic_std + 1e-10),
            'rank_ic': ric, 'rank_ic_std': ric_std, 'rank_icir': ric / (ric_std + 1e-10),
            'top10_ret': top, 'bottom10_ret': bot, 'long_short': ls,
            'annual_ret': ann, 'max_dd': dd_val, 'calmar': calmar,
        }
    
    ra = evaluate(test, pa, 'Plan A')
    rb = evaluate(test, pb, 'Plan B')
    results_a.append(ra)
    results_b.append(rb)
    
    print(f"  A: IC={ra['ic']:.4f} RIC={ra['rank_ic']:.4f} LS={ra['long_short']*100:.2f}%")
    print(f"  B: IC={rb['ic']:.4f} RIC={rb['rank_ic']:.4f} LS={rb['long_short']*100:.2f}%")

# ===== 5. 汇总 =====
print("\n" + "=" * 60)
print("Plan A vs Plan B Walk-Forward 结果")
print("=" * 60)

def summarize(results):
    s = {}
    for k in ['ic', 'rank_ic', 'icir', 'rank_icir', 'top10_ret', 'bottom10_ret', 'long_short', 'annual_ret', 'max_dd', 'calmar']:
        vals = [r[k] for r in results if not (np.isnan(r[k]) or np.isinf(r[k]))]
        if vals:
            s[k] = float(np.mean(vals))
            s[k + '_std'] = float(np.std(vals))
    return s

sa, sb = summarize(results_a), summarize(results_b)

print(f"\n{'指标':<18} {'Plan A (XGB回归)':<22} {'Plan B (LGB排名)':<22} {'胜'}")
print("-" * 68)
for label, k in [
    ('IC', 'ic'), ('Rank IC', 'rank_ic'), ('ICIR', 'icir'), ('Rank ICIR', 'rank_icir'),
    ('Top10%多头', 'top10_ret'), ('Bot10%空头', 'bottom10_ret'), ('多空利差', 'long_short'),
    ('年化收益', 'annual_ret'), ('最大回撤', 'max_dd'), ('Calmar', 'calmar'),
]:
    am, bm = sa.get(k, 0), sb.get(k, 0)
    astd, bstd = sa.get(k+'_std', 0), sb.get(k+'_std', 0)
    w = 'A' if (abs(am) > abs(bm) if 'dd' not in k else am > bm) else 'B'
    print(f"{label:<18} {am*100:>8.2f}%±{astd*100:.1f}%  {bm*100:>8.2f}%±{bstd*100:.1f}%  {w}")

# 特征重要性
print(f"\nPlan A 特征重要性 TOP 10:")
imp = ma.feature_importances_
fi = sorted(zip(available_features, imp), key=lambda x: -x[1])
for fn, fv in fi[:10]:
    bar = '█' * int(fv / max(imp) * 30)
    print(f"  {fn:<18} {fv:.4f} {bar}")

# CEO决策
ls_a = sa.get('long_short', 0)
ls_b = sb.get('long_short', 0)
diff_pct = abs(ls_a - ls_b) / max(abs(ls_a), abs(ls_b), 1e-10) * 100

print(f"\n{'='*60}")
print("CEO决策")
print(f"{'='*60}")
if diff_pct < 10:
    decision = "BOTH"
    print(f"多空利差差距 {diff_pct:.1f}% < 10% → 两个方案都保留")
    print("  Plan A: 绝对收益预测 + 置信度")
    print("  Plan B: 截面排名 + 行业轮动")
elif ls_a > ls_b:
    decision = "A"
    print(f"Plan A更优 ({ls_a*100:.2f}% vs {ls_b*100:.2f}%)")
else:
    decision = "B"
    print(f"Plan B更优 ({ls_b*100:.2f}% vs {ls_a*100:.2f}%)")

# 保存
output = {
    'plan_a': sa, 'plan_b': sb,
    'plan_a_details': results_a, 'plan_b_details': results_b,
    'feature_importance': [{'feature': f, 'importance': float(v)} for f, v in fi],
    'decision': decision,
    'data_info': {
        'samples': len(df), 'stocks': int(df['sym'].nunique()),
        'features': len(available_features),
        'date_range': f"{df['date'].min().date()} ~ {df['date'].max().date()}",
        'folds': len(time_splits),
    }
}
with open(f'{OUTPUT_DIR}/plan_ab_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n保存: {OUTPUT_DIR}/plan_ab_results.json")
print(f"耗时: {time.time()-t0:.0f}s")
