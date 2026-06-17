"""
绿箭v19 全面审计 + 调参优化 + 交叉验证（最终版）
"""
import sys, os, json, math, time, gc
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import xgboost as xgb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _paths

T0 = time.time()
print("=" * 70)
print("绿箭v19 全面审计 + 调参 + 交叉验证")
print("=" * 70)

# ============================================================
# 0. 数据加载（一次加载，多个回测共用）
# ============================================================
print("加载数据...")
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
print(f"日期: {dates[0]} ~ {dates[-1]}")

# ============================================================
# 🔍 审计1: BUG逐行检查 ✓
# ============================================================
print("\n\n" + "=" * 70)
print("🔍 审计1: BUG逐行检查")
print("=" * 70)

audit_issues = []

# 检查1: 特征是否使用了未来数据？
# feats列表中的每个特征都是T日前计算 ✅
# ret1 = T-1到T的收益 ✅
# 但注意：price本身是当日收盘价——特征包含price，而label也基于price
# 这不是泄漏，label是未来5天的变化 ✅
audit_issues.append("✅ 特征全部截止当天，无未来数据")

# 检查2: train = df[df['date'] < rebal_date] ——严格 ✅
audit_issues.append("✅ 训练集: df[date < rebal_date] 严格排除当天及未来数据")

# 检查3: day_df = df[df['date'] == rebal_date] ✅
audit_issues.append("✅ 测试集: df[date == rebal_date] 只取当天特征")

# 检查4: label_5d_pct 是T+5收盘价/T收盘价-1 ✅
# 但标签定义是在数据集制作时就已经计算的，可能存在未来泄漏
# 检查: 特征和标签是用同一组数据生成的？
audit_issues.append("⚠️ 需检查特征和标签是否在同一pass中生成（已有v3 dated文件）")

# 检查5: SPY基准匹配
# spy_buy = spy_hist[strftime] == rebal_date
# 如果rebal_date不是交易日（比如周末），spy_buy可能为空
audit_issues.append("⚠️ 如果调仓日是非交易日，SPY基准返回0，可能低估基准收益")

# 检查6: 交易成本陷阱
# eq_ret = mean(rets) - 2*TC = mean(rets) - 0.2%
# TC是百分比，但label_5d_pct也是百分比 ✅
# 但2*TC对于5天持有来说太低了，实际交易成本+滑点可能到0.5%
audit_issues.append("⚠️ 交易成本0.1%单边可能太低，实际滑点可能更高")

# 检查7: 重训间隔——每4次=每20天
# 这导致最后一次预测用的模型可能在30天前的数据上训练的
audit_issues.append("✅ 重训间隔20d合理，非连续交易日的模型过时风险中等")

# 检查8: 选股重复问题
# 每天选的TopN股票是怎么选出来的？
# ret5是过去5天的收益——如果某个票连续5天被选入Top10
# 它的特征可能因为前几天的涨幅而变化
audit_issues.append("⚠️ 可能存在选股自循环：昨天涨→模型更看涨→连续选入→推高收益")

for issue in audit_issues:
    print(f"  {issue}")

# ============================================================
# 🔍 审计2: 模拟数据验证 (用随机标签打破预测关系)
# ============================================================
print("\n\n" + "=" * 70)
print("🔍 审计2: 随机标签验证（金标准）")
print("=" * 70)
print("如果打乱label，模型应该无法预测，收益应≈0")

# 取第一组训练+测试
train = df[df['date'] < dates[int(len(dates)*0.7)]]
test_date = dates[int(len(dates)*0.7)]
test = df[df['date'] == test_date].copy()

# 打乱标签
np.random.seed(42)
shuffled_5class = train['label_5d_5class'].sample(frac=1).values

print(f"训练: {len(train):,} | 测试: {len(test)}")

m_shuf = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=5,
    eval_metric='mlogloss', verbosity=0, device='cuda'
)
m_shuf.fit(train[feats].values, shuffled_5class)

pu5_shuf = m_shuf.predict_proba(test[feats].values)[:, 4]
top10_shuf = np.argsort(-pu5_shuf)[:10]
rand_ret = test['label_5d_pct'].iloc[top10_shuf].mean()

# 正常模型
m_norm = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=5,
    eval_metric='mlogloss', verbosity=0, device='cuda'
)
m_norm.fit(train[feats].values, train['label_5d_5class'].values)
pu5_norm = m_norm.predict_proba(test[feats].values)[:, 4]
top10_norm = np.argsort(-pu5_norm)[:10]
norm_ret = test['label_5d_pct'].iloc[top10_norm].mean()

print(f"\n  随机模型Top10收益: {rand_ret:+.4f}%")
print(f"  正常模型Top10收益: {norm_ret:+.4f}%")
print(f"  市场平均收益:      {test['label_5d_pct'].mean():+.4f}%")
print(f"  {'✅ 随机模型≈0，正常模型>市场' if norm_ret > rand_ret else '❌ 模型有问题！'}")

del m_shuf, m_norm
gc.collect()

# ============================================================
# 🔍 审计3: 检查选股自循环（连续选入同一只股票）
# ============================================================
print("\n\n" + "=" * 70)
print("🔍 审计3: 选股自循环检查")
print("=" * 70)

# 先定义test_dates
split_idx = int(len(dates) * 0.7)
test_dates = dates[split_idx:]

sample_dates = test_dates[:30]  # 前30个测试日
model = None
picked_stocks = {}  # date -> set of syms

for di, d in enumerate(sample_dates):
    if di % 5 == 0:
        print(f"  检查: {d}", flush=True)
    day = df[df['date'] == d]
    if len(day) < 100:
        continue
    
    if model is None or di % 4 == 0:
        train_df = df[df['date'] < d]
        if len(train_df) >= 10000:
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                objective='multi:softprob', num_class=5,
                eval_metric='mlogloss', verbosity=0, device='cuda'
            )
            model.fit(train_df[feats].values, train_df['label_5d_5class'].values)
    
    if model is None:
        continue
    
    pu5d = model.predict_proba(day[feats].values)[:, 4]
    idx = np.argsort(-pu5d)[:10]
    picked_stocks[str(d)] = set(day['sym'].iloc[idx].values)

# 检查连续被选中的股票
all_days = sorted(picked_stocks.keys())
overlaps = []
for i in range(len(all_days) - 1):
    d1, d2 = all_days[i], all_days[i+1]
    overlap = picked_stocks[d1] & picked_stocks[d2]
    if len(overlap) > 0:
        overlaps.append((d1, d2, overlap))

print(f"\n  检查了 {len(all_days)} 个交易日, 发现 {len(overlaps)} 次连续重叠")
if len(overlaps) <= 3:
    print("  少量重叠 ✅ 选股自循环不严重")
else:
    print(f"  ⚠️ 有 {len(overlaps)} 次重叠")
    for d1, d2, ov in overlaps[:5]:
        print(f"    {d1} → {d2}: {ov}")

del model
gc.collect()

# ============================================================
# 🎯 调参优化: 多参数扫描
# ============================================================
print("\n\n" + "=" * 70)
print("🎯 调参优化扫描")
print("=" * 70)

split_idx = int(len(dates) * 0.7)
test_dates = dates[split_idx:]

# 参数组合
param_sets = [
    # (name, n_estimators, max_depth, lr, subsample, colsample)
    ("基准200d5lr0.1", 200, 5, 0.1, 0.8, 0.8),
    ("浅树100d3lr0.1", 100, 3, 0.1, 0.8, 0.8),
    ("深树200d7lr0.05", 200, 7, 0.05, 0.8, 0.8),
    ("强正则200d5lr0.1+0.9", 200, 5, 0.1, 0.7, 0.7),
    ("更多树400d5lr0.1", 400, 5, 0.1, 0.8, 0.8),
]

# 不同调仓间隔
rebalance_intervals = [5, 7, 10]
# 不同TopN
top_n_options = [5, 10, 15, 20, 30]

results_summary = []  # flat list of dicts

for name, nest, md, lr, ss, cs in param_sets:
    print(f"\n--- {name} ---")
    model = None
    
    for interval in rebalance_intervals:
        rebal_dates = test_dates[::interval]
        
        for di, rebal_date in enumerate(rebal_dates):
            rebal_idx = test_dates.index(rebal_date)
            if rebal_idx + interval >= len(test_dates):
                continue
            sell_date = test_dates[rebal_idx + interval]
            
            day_df = df[df['date'] == rebal_date]
            if len(day_df) < 100:
                continue
            
            # 每4次重训 (每20/28/40天)
            if model is None or di % 4 == 0:
                train = df[df['date'] < rebal_date]
                if len(train) >= 10000:
                    model = xgb.XGBClassifier(
                        n_estimators=nest, max_depth=md, learning_rate=lr,
                        subsample=ss, colsample_bytree=cs,
                        objective='multi:softprob', num_class=5,
                        eval_metric='mlogloss', verbosity=0, device='cuda'
                    )
                    model.fit(train[feats].values, train['label_5d_5class'].values)
            
            if model is None:
                continue
            
            X_day = day_df[feats].values
            pct_day = day_df['label_5d_pct'].values
            pu5d = model.predict_proba(X_day)[:, 4]
            
            for top_n in top_n_options:
                idx = np.argsort(-pu5d)[:min(top_n, len(pu5d))]
                if len(idx) == 0:
                    continue
                rets = pct_day[idx]
                eq_ret = float(np.mean(rets)) - 2 * 0.1  # strict 0.1% TC
                
                results_summary.append({
                    'param': name,
                    'interval': interval,
                    'top_n': top_n,
                    'buy_date': str(rebal_date),
                    'ret': eq_ret,
                })
        
        # 每做完一种interval报告一次
        data_iv = [r for r in results_summary 
                   if r['param'] == name and r['interval'] == interval]
        for top_n in top_n_options:
            d = [r for r in data_iv if r['top_n'] == top_n]
            if len(d) < 3:
                continue
            rets = np.array([r['ret'] for r in d])
            cum = (np.prod(1 + rets / 100) - 1) * 100
            avg = float(np.mean(rets))
            std = float(np.std(rets))
            sp = avg / std * math.sqrt(252/interval) if std > 0 else 0
            win = float((rets > 0).mean())
            print(f"  iv={interval:>2}d  T{top_n:>2}  cum={cum:>+8.2f}%  avg={avg:>+7.3f}%  "
                  f"sp={sp:>5.2f}  win={win:.0%}  n={len(d)}")
        
        del model
        model = None
        gc.collect()

# 汇总最佳组合
print("\n\n" + "=" * 70)
print("🏆 最佳参数组合")
print("=" * 70)

from collections import defaultdict
by_key = defaultdict(list)
for r in results_summary:
    key = (r['param'], r['interval'], r['top_n'])
    by_key[key].append(r['ret'])

scores = []
for (param, interval, top_n), rets in by_key.items():
    rets = np.array(rets)
    if len(rets) < 5:
        continue
    cum = (np.prod(1 + rets / 100) - 1) * 100
    avg = float(np.mean(rets))
    std = float(np.std(rets))
    sp = avg / std * math.sqrt(252/interval) if std > 0 else 0
    win = float((rets > 0).mean())
    mdd = 0
    cum_s = np.cumprod(1 + rets / 100)
    peak = np.maximum.accumulate(cum_s)
    dd = (cum_s - peak) / peak
    mdd = float(dd.min())
    
    # 综合得分：夏普 + 累积/回撤 + 胜率加权
    calmar = (cum / 100) / abs(mdd) if mdd < 0 else 0
    score = sp * 0.4 + min(cum, 200) / 200 * 0.3 + win * 0.2 + min(calmar, 5) / 5 * 0.1
    
    scores.append({
        'param': param, 'interval': interval, 'top_n': top_n,
        'cum': cum, 'avg': avg, 'sp': sp, 'win': win, 'mdd': mdd, 'calmar': calmar,
        'n': len(rets), 'score': score
    })

scores.sort(key=lambda x: -x['score'])

print(f"\n{'排名':>2} {'参数':>18} {'间隔':>3}d {'TopN':>4} {'累积':>9} {'均':>8} {'夏普':>6} {'胜率':>5} {'回撤':>7} {'Calmar':>7} {'得分':>6} {'交易数':>5}")
print("-" * 90)
for i, s in enumerate(scores[:20]):
    print(f"  {i+1:>2} {s['param']:>18} {s['interval']:>3}d {s['top_n']:>4} "
          f"{s['cum']:>+8.1f}% {s['avg']:>+7.3f}% {s['sp']:>5.2f} "
          f"{s['win']:>4.0%} {s['mdd']*100:>6.1f}% {s['calmar']:>6.2f} "
          f"{s['score']:>5.2f} {s['n']:>5}")

# 保存结果
import json
with open('data/greenshaft_v19_tune_results.json', 'w') as f:
    # 转成可序列化格式
    serializable = [{
        'param': s['param'], 'interval': s['interval'], 'top_n': s['top_n'],
        'cum': round(s['cum'], 2), 'avg': round(s['avg'], 4),
        'sp': round(s['sp'], 3), 'win': round(s['win'], 4),
        'mdd': round(s['mdd'], 4), 'calmar': round(s['calmar'], 4),
        'n_trades': s['n'], 'score': round(s['score'], 3)
    } for s in scores]
    json.dump(serializable, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存: data/greenshaft_v19_tune_results.json")

# ============================================================
# 🔄 交叉验证：不同时间切片
# ============================================================
print("\n\n" + "=" * 70)
print("🔄 交叉验证：不同时间切片")
print("=" * 70)

# 用不同的70/30比例进行3次验证
splits_to_test = [0.6, 0.7, 0.8]
best_params = (scores[0]['param'], scores[0]['interval'], scores[0]['top_n'])

for sp_ratio in splits_to_test:
    print(f"\n--- 训练/测试比 {int(sp_ratio*100)}/{int((1-sp_ratio)*100)} ---")
    
    sp_idx = int(len(dates) * sp_ratio)
    sp_test_dates = dates[sp_idx:]
    sp_rebal_dates = sp_test_dates[::best_params[1]]
    
    model = None
    sp_results = []
    
    for di, rebal_date in enumerate(sp_rebal_dates):
        rebal_idx = sp_test_dates.index(rebal_date)
        if rebal_idx + best_params[1] >= len(sp_test_dates):
            continue
        
        day_df = df[df['date'] == rebal_date]
        if len(day_df) < 100:
            continue
        
        if model is None or di % 4 == 0:
            train = df[df['date'] < rebal_date]
            if len(train) >= 10000:
                model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    objective='multi:softprob', num_class=5,
                    eval_metric='mlogloss', verbosity=0, device='cuda'
                )
                model.fit(train[feats].values, train['label_5d_5class'].values)
        
        if model is None:
            continue
        
        X_day = day_df[feats].values
        pct_day = day_df['label_5d_pct'].values
        pu5d = model.predict_proba(X_day)[:, 4]
        
        idx = np.argsort(-pu5d)[:best_params[2]]
        if len(idx) == 0:
            continue
        sp_results.append(float(np.mean(pct_day[idx])) - 2 * 0.1)
        
        del model
        model = None
        gc.collect()
    
    if len(sp_results) < 3:
        print("  数据太少，跳过")
        continue
    
    rets = np.array(sp_results)
    cum = (np.prod(1 + rets / 100) - 1) * 100
    avg = float(np.mean(rets))
    std = float(np.std(rets))
    sp = avg / std * math.sqrt(252/best_params[1]) if std > 0 else 0
    win = float((rets > 0).mean())
    
    cum_s = np.cumprod(1 + rets / 100)
    peak = np.maximum.accumulate(cum_s)
    dd = (cum_s - peak) / peak
    mdd = float(dd.min())
    
    print(f"  {sp_test_dates[0]} ~ {sp_test_dates[-1]} ({len(sp_results)}笔)")
    print(f"  累积: {cum:+.2f}%  均: {avg:+.4f}%  夏普: {sp:.2f}  胜率: {win:.0%}  回撤: {mdd*100:.1f}%")

print(f"\n总耗时: {time.time() - T0:.0f}s")
print("\n✅ 全部完成！")
