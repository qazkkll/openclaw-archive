"""
绿箭v19 诊断核心问题
审计2只做了一天，样本太少。做20天多日平均对比。
"""
import sys, os, json, math, time, gc
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import xgboost as xgb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _paths

T0 = time.time()
print("=" * 60)
print("绿箭v19 核心诊断")
print("=" * 60)

df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v3_dated.parquet")
df = df[(df['label_5d_pct'] >= -50) & (df['label_5d_pct'] <= 50)].copy()

with open(_paths.ML_DIR + "/us_sector_etf.json") as f:
    etf_data = json.load(f)
s2e = {'Technology':'XLK','Financial Services':'XLF','Financial':'XLF','Energy':'XLE',
       'Healthcare':'XLV','Industrials':'XLI','Consumer Defensive':'XLP',
       'Consumer Cyclical':'XLY','Utilities':'XLU','Basic Materials':'XLB',
       'Materials':'XLB','Real Estate':'XLRE','Communication Services':'XLC','Semiconductor':'SMH'}
def get_er(s):
    e = s2e.get(s)
    return etf_data[e]['ret5'] if e and e in etf_data else etf_data['SPY']['ret5']
df['sector_etf_ret5'] = df['sector'].apply(get_er)
for k in ['SPY','QQQ','IWM']:
    df[f'{k.lower()}_ret5'] = etf_data[k]['ret5']
df['sc'] = df['sector'].astype('category').cat.codes.astype(int)

feats = ['price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
         'ret1','ret5','ret20','ret60','macd','macd_signal','macd_hist',
         'vol_ratio','ma_bias20','vol5','trend_accel',
         'short_ratio','short_pct','market_cap','sector_etf_ret5',
         'spy_ret5','qqq_ret5','iwm_ret5','sc']

df = df.dropna(subset=feats + ['label_5d_pct', 'label_5d_5class']).copy()
df = df.sort_values(['date','sym']).reset_index(drop=True)
dates = sorted(df['date'].unique())

print(f"数据: {len(df):,}行, {df['sym'].nunique()}只, {len(dates)}天")

# ============================================================
# 诊断1: 多日随机标签对比（至少20天）
# ============================================================
print("\n【诊断1】多日随机标签 vs 绿箭模型")
split_idx = int(len(dates) * 0.7)
test_dates = dates[split_idx:]

# 用每5天调仓（同v5）
rebal_dates = test_dates[::5]

normal_results = []  # (date, ret)
random_results = []

model_norm = None
model_rand = None
rand_seeds = [42, 99, 123, 777, 2026]

for ri, rebal_date in enumerate(rebal_dates):
    rebal_idx = test_dates.index(rebal_date)
    if rebal_idx + 5 >= len(test_dates):
        continue
    
    if ri % 10 == 0:
        print(f"  {rebal_date} ({ri}/{len(rebal_dates)})", flush=True)
    
    day_df = df[df['date'] == rebal_date]
    if len(day_df) < 100:
        continue
    
    # 正常模型
    if model_norm is None or ri % 4 == 0:
        train = df[df['date'] < rebal_date]
        if len(train) >= 10000:
            model_norm = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                objective='multi:softprob', num_class=5,
                eval_metric='mlogloss', verbosity=0, device='cuda'
            )
            model_norm.fit(train[feats].values, train['label_5d_5class'].values)
    
    if model_norm is not None:
        X_day = day_df[feats].values
        pu5 = model_norm.predict_proba(X_day)[:, 4]
        idx = np.argsort(-pu5)[:10]
        normal_results.append(float(np.mean(day_df['label_5d_pct'].iloc[idx])) - 0.2)
    
    # 随机模型（每次new一个，打乱label）
    seed = rand_seeds[ri % len(rand_seeds)]
    train = df[df['date'] < rebal_date]
    if len(train) >= 10000:
        np.random.seed(seed)
        shuffled = train['label_5d_5class'].sample(frac=1).values
        
        model_rand = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective='multi:softprob', num_class=5,
            eval_metric='mlogloss', verbosity=0, device='cuda'
        )
        model_rand.fit(train[feats].values, shuffled)
        
        X_day = day_df[feats].values
        pu5r = model_rand.predict_proba(X_day)[:, 4]
        idx_r = np.argsort(-pu5r)[:10]
        random_results.append(float(np.mean(day_df['label_5d_pct'].iloc[idx_r])) - 0.2)
    
    # 释放
    if ri % 4 == 0 and ri > 0:
        del model_norm
        model_norm = None
    if model_rand is not None:
        del model_rand
        model_rand = None
    gc.collect()

print(f"\n  总共完成了 {len(normal_results)} 笔正常 + {len(random_results)} 笔随机")

norm = np.array(normal_results)
rand = np.array(random_results[:len(normal_results)])

avg_norm = float(np.mean(norm))
avg_rand = float(np.mean(rand))
cum_norm = (np.prod(1 + norm / 100) - 1) * 100
cum_rand = (np.prod(1 + rand / 100) - 1) * 100
win_norm = float((norm > 0).mean())
win_rand = float((rand > 0).mean())

print(f"\n  绿箭Top10: 累积={cum_norm:+.2f}%  均收益={avg_norm:+.4f}%  胜率={win_norm:.0%}")
print(f"  随机Top10: 累积={cum_rand:+.2f}%  均收益={avg_rand:+.4f}%  胜率={win_rand:.0%}")
if avg_norm > avg_rand:
    print(f"  ✅ 模型显著优于随机! 超额={avg_norm - avg_rand:+.4f}%/笔")
else:
    print(f"  ❌ 模型不如随机! 差距={avg_norm - avg_rand:+.4f}%/笔")

# ============================================================
# 诊断2: 特征重要性分析
# ============================================================
print("\n\n【诊断2】特征重要性分析")
print("用全量训练集训练一个模型，看哪些特征最重要...")

train_full = df[df['date'] < test_dates[0]]
print(f"  训练: {len(train_full):,}行")

m = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=5,
    eval_metric='mlogloss', verbosity=0, device='cuda'
)
m.fit(train_full[feats].values, train_full['label_5d_5class'].values)

importances = sorted(zip(feats, m.feature_importances_), key=lambda x: -x[1])
print(f"\n  {'特征':>25} {'重要性':>10}")
print(f"  {'-'*35}")
for f, imp in importances:
    print(f"  {f:>25} {imp:>10.4f}")
del m
gc.collect()

# ============================================================
# 诊断3: 特征反相关检查
# ============================================================
print("\n\n【诊断3】label_5d_pct与各特征的相关性")
# 选测试期第1天
day1 = test_dates[0]
day_df = df[df['date'] == day1].copy()
corrs = []
for f in feats:
    c = day_df[f].corr(day_df['label_5d_pct'])
    corrs.append((f, c))
corrs.sort(key=lambda x: -abs(x[1]))
print(f"\n  {test_dates[0]}, {len(day_df)}只股票:")
print(f"  {'特征':>25} {'相关性':>10}")
print(f"  {'-'*35}")
for f, c in corrs:
    print(f"  {f:>25} {c:>+10.4f}")

# 检查price的分布——看模型是否主要靠小市值做多
print(f"\n  price分位数:")
for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"    P{int(q*100)}%: ${day_df['price'].quantile(q):.2f}")

print(f"\n总耗时: {time.time() - T0:.0f}s")
