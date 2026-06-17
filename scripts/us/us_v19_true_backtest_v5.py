"""
绿箭v19 真实回测 v5 — 每5天调仓 + SPY对比 + 逐年分解
加上最关键的：每个调仓日也买入SPY作为基准
"""
import sys, os, json, math, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import xgboost as xgb
import yfinance as yf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _paths

T0 = time.time()
print("=" * 60)
print("绿箭v19 真实回测 v5 — 每5天调仓 + SPY对比")
print("=" * 60)

# 1. SPY基准
print("获取SPY数据...")
spy = yf.Ticker("SPY")
spy_hist = spy.history(period='2y')['Close']
print(f"  SPY数据: {len(spy_hist)}天")

# 2. 加载数据
df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v3_dated.parquet")
df = df[(df['label_5d_pct'] >= -50) & (df['label_5d_pct'] <= 50)].copy()

# 3. 特征
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
rebalance_dates = test_dates[::5]

print(f"训练截止: {dates[split_idx-1]}")
print(f"回测范围: {test_dates[0]} ~ {test_dates[-1]}")
print(f"调仓次数: {len(rebalance_dates)}")

TC = 0.1
top_ns = [5, 10, 20, 50]
results = {n: [] for n in top_ns}
model = None

# 统计原始命中
all_probs = []
all_actuals = []

for ri, rebal_date in enumerate(rebalance_dates):
    rebal_idx = test_dates.index(rebal_date)
    if rebal_idx + 5 >= len(test_dates):
        continue
    sell_date = test_dates[rebal_idx + 5]

    if ri % 5 == 0:
        print(f"  调仓 {ri+1}/{len(rebalance_dates)}: {rebal_date}", flush=True)

    day_df = df[df['date'] == rebal_date]
    if len(day_df) < 100:
        continue

    # 重训
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

    if model is None:
        continue

    X_day = day_df[feats].values
    pct_day = day_df['label_5d_pct'].values
    pu5 = model.predict_proba(X_day)[:, 4]

    # SPY同期收益
    spy_buy = spy_hist[spy_hist.index.strftime('%Y-%m-%d') == rebal_date]
    spy_sell = spy_hist[spy_hist.index.strftime('%Y-%m-%d') == sell_date]
    spy_5d = 0
    if len(spy_buy) > 0 and len(spy_sell) > 0:
        spy_5d = (float(spy_sell.iloc[0]) / float(spy_buy.iloc[0]) - 1) * 100

    # ✅ 记录所有概率（用于后验校准分析）
    all_probs.extend(pu5.tolist())
    all_actuals.extend(pct_day.tolist())

    for top_n in top_ns:
        idx = np.argsort(-pu5)[:min(top_n, len(pu5))]
        if len(idx) == 0:
            continue

        rets = pct_day[idx]
        eq_ret = float(np.mean(rets)) - 2 * TC

        results[top_n].append({
            'buy': rebal_date, 'sell': sell_date,
            'eq_ret': eq_ret,
            'spy_ret': spy_5d,
            'excess': eq_ret - spy_5d,
            'hit_up5': float((rets > 5).mean()),
            'hit_up0': float((rets > 0).mean()),
            'max_ret': float(rets.max()),
            'min_ret': float(rets.min()),
            'n': len(idx),
        })

# 4. 输出
print("\n" + "=" * 60)
print("回测结果（每5天调仓 · SPY对比）")
print("=" * 60)

# 先计算SPY全程收益
spy_start = spy_hist[spy_hist.index.strftime('%Y-%m-%d') == test_dates[0]]
spy_end = spy_hist[spy_hist.index.strftime('%Y-%m-%d') == test_dates[-1]]
spy_total = 0
if len(spy_start) > 0 and len(spy_end) > 0:
    spy_total = (float(spy_end.iloc[0]) / float(spy_start.iloc[0]) - 1) * 100
print(f"\nSPY全程持有 {test_dates[0]} ~ {test_dates[-1]}: {spy_total:+.2f}%")

for top_n in top_ns:
    data = results[top_n]
    if not data:
        continue

    rets = np.array([r['eq_ret'] for r in data])
    spys = np.array([r['spy_ret'] for r in data])
    exs = np.array([r['excess'] for r in data])
    n = len(rets)
    if n == 0:
        continue

    cum = (np.prod(1 + rets / 100) - 1) * 100
    cum_spy = (np.prod(1 + spys / 100) - 1) * 100
    cum_ex = (np.prod(1 + exs / 100) - 1) * 100

    avg = float(np.mean(rets))
    std = float(np.std(rets))
    sp = avg / std * math.sqrt(252/5) if std > 0 else 0
    win = float((rets > 0).mean())

    cum_series = np.cumprod(1 + rets / 100)
    peak = np.maximum.accumulate(cum_series)
    dd = (cum_series - peak) / peak
    mdd = float(dd.min())

    avg_hit5 = float(np.mean([r['hit_up5'] for r in data]))
    avg_hit0 = float(np.mean([r['hit_up0'] for r in data]))

    print(f"\n【每5天Top{top_n}】 {n}笔交易")
    print(f"  ┌─────────────────────────────────┐")
    print(f"  │ 策略累积: {cum:>+10.2f}%   │")
    print(f"  │ SPY同期:  {cum_spy:>+10.2f}%   │")
    print(f"  │ 超额累积: {cum_ex:>+10.2f}%   │")
    print(f"  ├─────────────────────────────────┤")
    print(f"  │ 单笔均:   {avg:>+9.4f}%      │")
    print(f"  │ 胜率:     {win:>8.1%}         │")
    print(f"  │ 夏普:     {sp:>8.3f}         │")
    print(f"  │ 回撤:     {mdd*100:>8.1f}%         │")
    print(f"  │ 涨>5%:    {avg_hit5:>8.1%}         │")
    print(f"  │ 涨>0:     {avg_hit0:>8.1%}         │")
    print(f"  └─────────────────────────────────┘")

    # 逐年
    from collections import defaultdict
    yearly = defaultdict(list)
    for r in data:
        y = r['buy'][:4]
        yearly[y].append(r)

    if len(yearly) > 1:
        print(f"\n  {'年':>3} {'笔':>3} {'策略均':>9} {'SPY均':>9} {'超额均':>9} {'胜率':>6}")
        for yr in sorted(yearly.keys()):
            yr_data = yearly[yr]
            y_ret = np.array([x['eq_ret'] for x in yr_data])
            y_spy = np.array([x['spy_ret'] for x in yr_data])
            y_ex = np.array([x['excess'] for x in yr_data])
            print(f"  {yr:>3} {len(yr_data):>3} {float(np.mean(y_ret)):+8.3f}% {float(np.mean(y_spy)):+8.3f}% {float(np.mean(y_ex)):+8.3f}% {float((y_ret>0).mean()):>5.1%}")

# 5. 后验校准分析
print("\n" + "=" * 60)
print("后验校准分析 (全部2435只股票预测)")
print("=" * 60)
all_probs = np.array(all_probs)
all_actuals = np.array(all_actuals)

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print(f"\n{'概率区间':>8} {'样本数':>8} {'平均预测概率':>14} {'涨>5%实际':>12} {'涨>0%实际':>12}")
print("-" * 56)
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    mask = (all_probs * 100 >= lo) & (all_probs * 100 < hi)
    cnt = mask.sum()
    if cnt < 10:
        continue
    avg_prob = float(np.mean(all_probs[mask])) * 100
    act_up5 = float((all_actuals[mask] > 5).mean()) * 100
    act_up0 = float((all_actuals[mask] > 0).mean()) * 100
    print(f"  {lo:>3}-{hi:<3}%  {cnt:>8}  {avg_prob:>8.1f}%     {act_up5:>8.1f}%     {act_up0:>8.1f}%")

print(f"\n总耗时: {time.time() - T0:.0f}s")
