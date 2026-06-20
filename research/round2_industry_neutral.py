#!/usr/bin/env python3
"""
第二轮实验：
1. 行业中性化（加入申万行业哑变量）
2. 对数市值控制
3. 资金流去噪（去掉资金流特征对比）

CEO决策：Plan A赢了，现在优化Plan A
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
import json, time, os, warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/home/hermes/.hermes/openclaw-archive/data'
OUTPUT_DIR = '/home/hermes/.hermes/openclaw-archive/research'

print("=" * 60)
print("第二轮实验：行业中性化 + 资金流去噪")
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

df = pd.merge(df_hist, df_mf, on=['sym', 'date'], how='inner')
df = df.sort_values(['sym', 'date']).reset_index(drop=True)
df = df[df['close'] > 0]
print(f"  {len(df):,} 行, {df['sym'].nunique()} 股")

# ===== 2. 加载行业信息 =====
print("\n[2] 行业信息...")
stock_info = pd.read_json(f'{DATA_DIR}/stock_info.json')
# stock_info: 行=index(属性), 列=股票代码, 值=属性值
# 提取industry行
if 'industry' in stock_info.index:
    industry_series = stock_info.loc['industry']
    # 转为dict: 股票代码 -> 行业
    industry_map = {}
    for col in industry_series.index:
        industry_map[str(col)] = industry_series[col]
    df['industry'] = df['sym'].map(industry_map)
    print(f"  行业映射: {df['industry'].notna().sum()}/{len(df)} 成功")
else:
    print(f"  stock_info index: {list(stock_info.index)}")
    df['industry'] = df['sym'].str[:3]

if df['industry'].isna().all():
    df['industry'] = df['sym'].str[:3]

print(f"  行业数: {df['industry'].nunique()}")

# ===== 3. 特征工程 =====
print("\n[3] 特征工程...")

df['r1'] = df.groupby('sym')['close'].pct_change(1)
df['r5'] = df.groupby('sym')['close'].pct_change(5)
df['r10'] = df.groupby('sym')['close'].pct_change(10)
df['r20'] = df.groupby('sym')['close'].pct_change(20)

df['ma5'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(5).mean())
df['ma10'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(10).mean())
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20).mean())
df['d5'] = (df['close'] - df['ma5']) / df['ma5']
df['d10'] = (df['close'] - df['ma10']) / df['ma10']
df['d20'] = (df['close'] - df['ma20']) / df['ma20']

df['vol5'] = df.groupby('sym')['r1'].transform(lambda x: x.rolling(5).std())
df['vol20'] = df.groupby('sym')['r1'].transform(lambda x: x.rolling(20).std())
df['atr'] = df.groupby('sym').apply(lambda g: (g['high']-g['low']).rolling(14).mean(), include_groups=False).reset_index(level=0, drop=True)
df['atr_pct'] = df['atr'] / df['close']
df['vol_ma5'] = df.groupby('sym')['volume'].transform(lambda x: x.rolling(5).mean())
df['vol_ma20'] = df.groupby('sym')['volume'].transform(lambda x: x.rolling(20).mean())
df['vol_ratio'] = df['vol_ma5'] / (df['vol_ma20'] + 1)

delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
df['_gain'] = gain
df['_loss'] = loss
df['_avg_gain'] = df.groupby('sym')['_gain'].transform(lambda x: x.rolling(14).mean())
df['_avg_loss'] = df.groupby('sym')['_loss'].transform(lambda x: x.rolling(14).mean())
df['rsi14'] = 100 - (100 / (1 + df['_avg_gain'] / (df['_avg_loss'] + 1e-10)))

df['ema12'] = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=12).mean())
df['ema26'] = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=26).mean())
df['macd'] = df['ema12'] - df['ema26']
df['macd_signal'] = df.groupby('sym')['macd'].transform(lambda x: x.ewm(span=9).mean())
df['macd_hist'] = df['macd'] - df['macd_signal']

for col in ['sm_net', 'md_net', 'lg_net', 'elg_net', 'total_net']:
    df[f'{col}_5'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5).sum())
    df[f'{col}_20'] = df.groupby('sym')[col].transform(lambda x: x.rolling(20).sum())

# 对数市值代理（用volume * close作为成交额，作为市值的粗代理）
df['log_amount'] = np.log1p(df['volume'] * df['close'])
df['log_amount_ma20'] = df.groupby('sym')['log_amount'].transform(lambda x: x.rolling(20).mean())

df['fwd_ret_20'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-20) / x - 1)

# 行业哑变量 - 只用top50行业（减少维度）
top_industries = df['industry'].value_counts().head(50).index.tolist()
df['industry_group'] = df['industry'].where(df['industry'].isin(top_industries), 'OTHER')
industry_dummies = pd.get_dummies(df['industry_group'], prefix='ind', dtype=np.float32)
# 降采样：只在每月第一天计算行业哑变量（减少内存）
# 不，直接合并
df = pd.concat([df, industry_dummies], axis=1)
del industry_dummies

print(f"  特征完成: {len(df):,} 行")

# 特征集
base_features = [
    'r1', 'r5', 'r10', 'r20', 'd5', 'd10', 'd20',
    'vol5', 'vol20', 'atr_pct', 'vol_ratio',
    'rsi14', 'macd', 'macd_signal', 'macd_hist',
]
mf_features = [
    'sm_net_5', 'sm_net_20', 'md_net_5', 'md_net_20',
    'lg_net_5', 'lg_net_20', 'elg_net_5', 'elg_net_20',
    'total_net_5', 'total_net_20',
]
size_features = ['log_amount_ma20']
industry_features = [c for c in df.columns if c.startswith('ind_')]

# 4种特征组合
experiments = {
    'A_baseline': base_features + mf_features,
    'B_no_moneyflow': base_features,
    'C_with_size': base_features + mf_features + size_features,
    'D_industry_neutral': base_features + mf_features + size_features + industry_features,
}

df = df.dropna(subset=base_features + mf_features + ['fwd_ret_20'])
print(f"  有效样本: {len(df):,} 行")

# ===== 4. Walk-Forward =====
print("\n[4] Walk-Forward (4种特征组合)...")

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

all_results = {}

for exp_name, feat_list in experiments.items():
    avail = [f for f in feat_list if f in df.columns]
    print(f"\n  === {exp_name} ({len(avail)} 特征) ===")
    
    exp_results = []
    for idx, (tr_s, tr_e, te_s, te_e) in enumerate(time_splits):
        train = df[(df['date'] >= tr_s) & (df['date'] <= tr_e)]
        test = df[(df['date'] >= te_s) & (df['date'] <= te_e)]
        
        if len(train) < 50000 or len(test) < 10000:
            continue
        
        X_tr = train[avail].values
        y_tr = train['fwd_ret_20'].values
        X_te = test[avail].values
        
        ta = time.time()
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=4, random_state=42, verbosity=0
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        
        # 评估
        test_copy = test.copy()
        test_copy['pred'] = pred
        ics, rics = [], []
        for d in test_copy['date'].unique():
            dd = test_copy[test_copy['date'] == d]
            if len(dd) < 30:
                continue
            ics.append(np.corrcoef(dd['fwd_ret_20'], dd['pred'])[0, 1])
            rics.append(spearmanr(dd['fwd_ret_20'], dd['pred'])[0])
        
        ic, ric = np.nanmean(ics), np.nanmean(rics)
        ic_std, ric_std = np.nanstd(ics), np.nanstd(rics)
        
        test_copy['pct'] = test_copy.groupby('date')['pred'].rank(pct=True)
        top = test_copy[test_copy['pct'] >= 0.9]['fwd_ret_20'].mean()
        bot = test_copy[test_copy['pct'] <= 0.1]['fwd_ret_20'].mean()
        ls = top - bot
        
        top_d = test_copy[test_copy['pct'] >= 0.9].groupby('date')['fwd_ret_20'].mean()
        if len(top_d) > 5:
            cum = (1 + top_d).cumprod()
            dd_val = (cum / cum.cummax() - 1).min()
            ann = cum.iloc[-1] ** (252 / max(len(cum), 1)) - 1
            calmar = ann / abs(dd_val) if dd_val != 0 else 0
        else:
            ann, dd_val, calmar = 0, 0, 0
        
        exp_results.append({
            'ic': ic, 'ic_std': ic_std, 'icir': ic/(ic_std+1e-10),
            'rank_ic': ric, 'rank_ic_std': ric_std, 'rank_icir': ric/(ric_std+1e-10),
            'top10_ret': top, 'bottom10_ret': bot, 'long_short': ls,
            'annual_ret': ann, 'max_dd': dd_val, 'calmar': calmar,
            'time': time.time()-ta,
        })
        
        print(f"    Fold {idx+1}: IC={ic:.4f} RIC={ric:.4f} LS={ls*100:.2f}% ({time.time()-ta:.0f}s)")
    
    # 汇总
    s = {}
    for k in ['ic', 'rank_ic', 'icir', 'rank_icir', 'top10_ret', 'bottom10_ret', 'long_short', 'annual_ret', 'max_dd', 'calmar', 'time']:
        vals = [r[k] for r in exp_results if not (np.isnan(r[k]) or np.isinf(r[k]))]
        if vals:
            s[k] = float(np.mean(vals))
            s[k+'_std'] = float(np.std(vals))
    
    all_results[exp_name] = {'summary': s, 'details': exp_results, 'features': len(avail)}
    print(f"  → 平均 IC={s.get('ic',0)*100:.2f}% LS={s.get('long_short',0)*100:.2f}%")

# ===== 5. 对比 =====
print("\n" + "=" * 70)
print("4种特征组合对比")
print("=" * 70)

header = f"{'实验':<22} {'IC':>8} {'RankIC':>8} {'ICIR':>8} {'多空':>8} {'回撤':>8} {'特征数'}"
print(header)
print("-" * 70)

for name, data in all_results.items():
    s = data['summary']
    print(f"{name:<22} {s.get('ic',0)*100:>7.2f}% {s.get('rank_ic',0)*100:>7.2f}% "
          f"{s.get('icir',0)*100:>7.1f}% {s.get('long_short',0)*100:>7.2f}% "
          f"{s.get('max_dd',0)*100:>7.1f}% {data['features']:>5}")

# 最优方案
best = max(all_results.items(), key=lambda x: x[1]['summary'].get('long_short', 0))
print(f"\n最优方案: {best[0]} (多空利差 {best[1]['summary']['long_short']*100:.2f}%)")

# 保存
with open(f'{OUTPUT_DIR}/round2_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n保存: {OUTPUT_DIR}/round2_results.json")
print(f"耗时: {time.time()-t0:.0f}s")
