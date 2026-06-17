"""
绿箭v19 真实回测 v4 — 每5天调仓版本

真正的问题在v3：
每天选Top5/10，持有5天，每天换仓 — 这在实际操作中不可行，
因为每天都有新的Top5/10，旧的5天未到期就被替换了。

真实做法：每5天选一次股，持有5天不动。
"""
import sys, os, json, math, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import xgboost as xgb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _paths

T0 = time.time()
print("=" * 60)
print("绿箭v19 真实回测 v4 — 每5天调仓")
print("=" * 60)

df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v3_dated.parquet")
df = df[(df['label_5d_pct'] >= -50) & (df['label_5d_pct'] <= 50)].copy()

# 补特征
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
split_idx = int(len(dates) * 0.7)
test_dates = dates[split_idx:]
print(f"训练截止: {dates[split_idx-1]}")
print(f"回测范围: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}天)")
print(f"调仓频率: 每5天")

# 每5天选一次日期（从第一调仓日算起）
rebalance_dates = test_dates[::5]  # 每5天一次调仓
print(f"调仓次数: {len(rebalance_dates)}")

TC = 0.1  # 单边 0.1%
top_ns = [5, 10, 20]
results = {n: [] for n in top_ns}
model = None
prev_rebalance = None

for ri, rebal_date in enumerate(rebalance_dates):
    # 找到5天后的对应日期
    rebal_idx = test_dates.index(rebal_date)
    sell_date = test_dates[rebal_idx + 5] if rebal_idx + 5 < len(test_dates) else None
    if sell_date is None:
        continue

    if ri % 5 == 0:
        print(f"  调仓 {ri+1}/{len(rebalance_dates)}: {rebal_date} -> {sell_date}", flush=True)

    # 调仓日数据
    day_df = df[df['date'] == rebal_date]
    if len(day_df) < 100:
        continue

    # 每4次调仓重训一次
    if model is None or ri % 4 == 0:
        train = df[df['date'] < rebal_date]
        if len(train) >= 10000:
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                objective='multi:softprob', num_class=5,
                eval_metric='mlogloss', verbosity=0, device='cuda'
            )
            model.fit(train[feats].values, train['label_5d_5class'].values)
            # print(f"    重训模型: {len(train):,}行")

    if model is None:
        continue

    # 预测
    X_day = day_df[feats].values
    pct_day = day_df['label_5d_pct'].values
    sym_day = day_df['sym'].values

    pu5 = model.predict_proba(X_day)[:, 4]

    for top_n in top_ns:
        idx = np.argsort(-pu5)[:min(top_n, len(pu5))]
        if len(idx) == 0:
            continue
        rets = pct_day[idx]
        eq_ret = float(np.mean(rets)) - 2 * TC  # 1次买入+1次卖出

        results[top_n].append({
            'buy': str(rebal_date),
            'sell': str(sell_date),
            'eq_ret': eq_ret,
            'hit_up5': float((rets > 5).mean()),
            'hit_up0': float((rets > 0).mean()),
            'n': len(idx),
            'max_ret': float(rets.max()),
            'min_ret': float(rets.min()),
        })

print("\n" + "=" * 60)
print("回测结果（每5天调仓，每次持有5天）")
print("=" * 60)

for top_n in top_ns:
    data = results[top_n]
    if not data:
        continue
    rets = np.array([r['eq_ret'] for r in data])
    n = len(rets)
    if n == 0:
        continue

    cum = (np.prod(1 + rets / 100) - 1) * 100
    avg = float(np.mean(rets))
    med = float(np.median(rets))
    std = float(np.std(rets))
    sp = avg / std * math.sqrt(252/5) if std > 0 else 0  # 5天周期年化
    win = float((rets > 0).mean())
    best = float(rets.max())
    worst = float(rets.min())

    cum_series = np.cumprod(1 + rets / 100)
    peak = np.maximum.accumulate(cum_series)
    dd = (cum_series - peak) / peak
    mdd = float(dd.min())

    avg_hit5 = float(np.mean([r['hit_up5'] for r in data]))
    avg_hit0 = float(np.mean([r['hit_up0'] for r in data]))

    print(f"\n【每5天选Top{top_n}, 持有5天】 {n}笔交易")
    print(f"  累积收益: {cum:+.2f}%")
    print(f"  单笔均收益: {avg:+.4f}% | 中位: {med:+.4f}%")
    print(f"  胜率(涨>0): {win:.1%}")
    print(f"  夏普(年化): {sp:.3f}")
    print(f"  最大回撤: {mdd*100:.1f}%")
    print(f"  多笔最佳: {best:+.2f}% | 多笔最差: {worst:+.2f}%")
    print(f"  涨>5%命中: {avg_hit5:.1%}")
    print(f"  涨>0胜率: {avg_hit0:.1%}")

print(f"\n总耗时: {time.time() - T0:.0f}s")
