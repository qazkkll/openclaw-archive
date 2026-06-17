"""
绿箭v19 快速诊断（轻量版）
用少量训练数据快速对比绿箭vs随机
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
print("绿箭v19 快速诊断")
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

print(f"数据: {len(df):,}行, {df['sym'].nunique()}只, {len(dates)}天\n")

# 只用最后30%数据的最近15个调仓日
split_idx = int(len(dates) * 0.7)
test_dates = dates[split_idx:]

# 取最后15个调仓日（更快，但也足够有统计意义）
rebal_dates = test_dates[::5]
rebal_dates = rebal_dates[-15:]  # 只用最后15轮

normal_rets = []
random_rets = []
all_rets = []

model = None

for ri, rebal_date in enumerate(rebal_dates):
    rebal_idx = test_dates.index(rebal_date)
    if rebal_idx + 5 >= len(test_dates):
        continue
    
    print(f"  {ri+1}/{len(rebal_dates)} {rebal_date}", flush=True)
    
    day_df = df[df['date'] == rebal_date]
    if len(day_df) < 100:
        continue
    
    # --- 绿箭模型 ---
    if model is None:
        train = df[df['date'] < rebal_date]
        print(f"    训练: {len(train):,}行...", end=' ', flush=True)
        model = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective='multi:softprob', num_class=5,
            eval_metric='mlogloss', verbosity=0, device='cuda'
        )
        model.fit(train[feats].values, train['label_5d_5class'].values)
        print(f"done", flush=True)
    
    X_day = day_df[feats].values
    pct_day = day_df['label_5d_pct'].values
    pu5 = model.predict_proba(X_day)[:, 4]
    
    idx = np.argsort(-pu5)[:10]
    n_ret = float(np.mean(pct_day[idx]))
    normal_rets.append(n_ret - 0.2)
    
    # --- 随机对比（不做训练，直接用随机选取） ---
    np.random.seed(int(pd.Timestamp(rebal_date).timestamp()) % 100000)
    rand_idx = np.random.choice(len(day_df), 10, replace=False)
    r_ret = float(np.mean(pct_day[rand_idx]))
    random_rets.append(r_ret - 0.2)
    
    avg_all = float(np.mean(pct_day))
    all_rets.append(avg_all)
    
    # 报告当前进度
    if len(normal_rets) >= 3:
        n3 = np.array(normal_rets[-3:])
        r3 = np.array(random_rets[-3:])
        a3 = np.array(all_rets[-3:])
        print(f"    近3笔: 绿箭={np.mean(n3):+.3f}% 随机={np.mean(r3):+.3f}% 平均={np.mean(a3):+.3f}%", flush=True)

# 统计
norm = np.array(normal_rets)
rand = np.array(random_rets)
all_ret = np.array(all_rets)

print(f"\n{'='*60}")
print(f"共 {len(norm)} 笔交易对比")
print(f"{'='*60}")
print(f"\n  绿箭Top10:   cum={(np.prod(1+norm/100)-1)*100:+7.2f}%  "
      f"avg={np.mean(norm):+6.3f}%  win={(norm>0).mean():.0%}  "
      f"std={np.std(norm):.3f}")
print(f"  随机Top10:   cum={(np.prod(1+rand/100)-1)*100:+7.2f}%  "
      f"avg={np.mean(rand):+6.3f}%  win={(rand>0).mean():.0%}  "
      f"std={np.std(rand):.3f}")
print(f"  市场平均:    cum={(np.prod(1+all_ret/100)-1)*100:+7.2f}%  "
      f"avg={np.mean(all_ret):+6.3f}%  win={(all_ret>0).mean():.0%}")

# 超额
excess = norm - rand
print(f"\n  绿箭超额随机: {np.mean(excess):+.3f}%/笔")
print(f"  绿箭胜出次数: {(norm > rand).sum()}/{len(norm)} ({(norm > rand).mean():.0%})")

# 逐笔输出
print(f"\n  逐笔:")
print(f"  {'#':>3} {'日期':>12} {'绿箭':>9} {'随机':>9} {'平均':>9} {'赢':>4}")
for i in range(len(norm)):
    w = '✅' if norm[i] > rand[i] else '❌'
    print(f"  {i:>3} {rebal_dates[i]} {norm[i]:>+8.3f}% {rand[i]:>+8.3f}% {all_ret[i]:>+8.3f}% {w:>4}")

# 特征重要性
print(f"\n\n【特征重要性】")
m2 = xgb.XGBClassifier(
    n_estimators=150, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=5,
    eval_metric='mlogloss', verbosity=0, device='cuda'
)
train_full = df[df['date'] < test_dates[0]]
m2.fit(train_full[feats].values, train_full['label_5d_5class'].values)
imps = sorted(zip(feats, m2.feature_importances_), key=lambda x: -x[1])
for f, i in imps:
    print(f"  {f:>25} {i:.4f}")
del m2

print(f"\n总耗时: {time.time() - T0:.0f}s")
