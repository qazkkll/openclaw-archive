"""
绿箭v19 真·干净回测 v3（最终版）
使用 us_ml_feats_v3_dated.parquet

关键修正：
1. 剔除极端值 (label_5d_pct > 50% or < -50%)
2. 扣交易成本 0.1%单边
3. 对比SPY同期的5天持有收益 — 这才是真正的超额收益
4. 加最大回撤、逐年分解、Calmar比率
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
print("绿箭v19 真·干净回测 v3（超额收益版）")
print("=" * 60)

# 1. 加载数据
print("加载 us_ml_feats_v3_dated.parquet ...")
df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v3_dated.parquet")

# 2. 剔除极端值 (>50% / <-50%)
before = len(df)
df = df[(df['label_5d_pct'] >= -50) & (df['label_5d_pct'] <= 50)].copy()
print(f"  剔除极端值: {before:,} → {len(df):,}")

# 3. 补特征
print("  补sector_etf_ret5等特征...")
with open(_paths.ML_DIR + "/us_sector_etf.json") as f:
    etf_data = json.load(f)

s2e = {
    'Technology': 'XLK', 'Financial Services': 'XLF', 'Financial': 'XLF',
    'Energy': 'XLE', 'Healthcare': 'XLV', 'Industrials': 'XLI',
    'Consumer Defensive': 'XLP', 'Consumer Cyclical': 'XLY', 'Utilities': 'XLU',
    'Basic Materials': 'XLB', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Communication Services': 'XLC', 'Semiconductor': 'SMH'
}

def get_er(s):
    e = s2e.get(s)
    return etf_data[e]['ret5'] if e and e in etf_data else etf_data['SPY']['ret5']

df['sector_etf_ret5'] = df['sector'].apply(get_er)
for k in ['SPY', 'QQQ', 'IWM']:
    df[f'{k.lower()}_ret5'] = etf_data[k]['ret5']

df['sc'] = df['sector'].astype('category').cat.codes.astype(int)

feats = [
    'price', 'volume', 'ma5', 'ma10', 'ma20', 'ma60', 'rsi14', 'vol20', 'p52',
    'ret1', 'ret5', 'ret20', 'ret60', 'macd', 'macd_signal', 'macd_hist',
    'vol_ratio', 'ma_bias20', 'vol5', 'trend_accel',
    'short_ratio', 'short_pct', 'market_cap', 'sector_etf_ret5',
    'spy_ret5', 'qqq_ret5', 'iwm_ret5', 'sc'
]

df = df.dropna(subset=feats + ['label_5d_pct', 'label_5d_5class']).copy()
df = df.sort_values(['date', 'sym']).reset_index(drop=True)

dates = sorted(df['date'].unique())
print(f"  日期范围: {dates[0]} ~ {dates[-1]} ({len(dates)}天)")
print(f"  股票数: {df['sym'].nunique()}")

# 4. 时间切分
split_idx = int(len(dates) * 0.7)
test_dates = dates[split_idx:]
print(f"  训练截止: {dates[split_idx - 1]}")
print(f"  回测开始: {test_dates[0]}")
print(f"  回测天数: {len(test_dates)}")

# 5. 获取SPY同期5天收益作为基准
print("\n获取SPY基准收益...")
spy = yf.Ticker("SPY")
spy_hist = spy.history(period='2y')
spy_close = spy_hist['Close']
spy_rets = {}
for i in range(len(spy_close) - 5):
    d = spy_close.index[i].strftime('%Y-%m-%d')
    ret5 = (spy_close.iloc[i+5] / spy_close.iloc[i] - 1) * 100
    spy_rets[d] = float(ret5)
print(f"  SPY数据: {len(spy_rets)}组5日收益")

# 6. 交易成本
TC = 0.1  # 单边 0.1%

# 7. 回测
RETRAIN_INTERVAL = 20
top_ns = [5, 10, 20, 50]
daily_results = {n: [] for n in top_ns}
model = None

print("\n逐日回测...")

for di, test_date in enumerate(test_dates):
    if di % 20 == 0:
        print(f"  {test_date} ({di}/{len(test_dates)})", flush=True)

    day_df = df[df['date'] == test_date]
    if len(day_df) < 100:
        continue

    if model is None or di % RETRAIN_INTERVAL == 0:
        train = df[df['date'] < test_date]
        if len(train) < 10000:
            if model is None:
                continue
        else:
            X_tr = train[feats].values
            y_tr = train['label_5d_5class'].values

            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                objective='multi:softprob', num_class=5,
                eval_metric='mlogloss', verbosity=0, device='cuda'
            )
            model.fit(X_tr, y_tr)

    if model is None:
        continue

    X_day = day_df[feats].values
    pct_day = day_df['label_5d_pct'].values
    sym_day = day_df['sym'].values

    # SPY基准
    spy_ret = spy_rets.get(test_date, 0)

    pu5 = model.predict_proba(X_day)[:, 4]

    for top_n in top_ns:
        idx = np.argsort(-pu5)[:min(top_n, len(pu5))]
        if len(idx) == 0:
            continue

        rets = pct_day[idx]
        eq_ret = float(np.mean(rets)) - 2 * TC
        excess = eq_ret - spy_ret  # 超额收益

        daily_results[top_n].append({
            'date': test_date,
            'eq_ret': eq_ret,
            'excess': excess,
            'spy_ret': spy_ret,
            'hit_up5': float((rets > 5).mean()),
            'hit_up0': float((rets > 0).mean()),
            'n': len(idx),
            'max_ret': float(rets.max()),
            'min_ret': float(rets.min()),
            'syms': ','.join(sym_day[idx][:5]),
        })

# 8. 输出
print("\n" + "=" * 60)
print("回测结果（超额收益版）")
print("=" * 60)

for top_n in top_ns:
    results = daily_results[top_n]
    if not results:
        continue

    eq_rets = np.array([r['eq_ret'] for r in results])
    ex_rets = np.array([r['excess'] for r in results])
    spy_series = np.array([r['spy_ret'] for r in results])

    n_days = len(eq_rets)

    # 累积策略
    cum_eq = (np.prod(1 + eq_rets / 100) - 1) * 100
    # 累积超额
    cum_ex = (np.prod(1 + ex_rets / 100) - 1) * 100
    # 累积SPY
    cum_spy = (np.prod(1 + spy_series / 100) - 1) * 100

    avg_eq = float(np.mean(eq_rets))
    std_eq = float(np.std(eq_rets))
    sp_eq = avg_eq / std_eq * math.sqrt(252) if std_eq > 0 else 0
    win_eq = float((eq_rets > 0).mean())

    avg_ex = float(np.mean(ex_rets))
    std_ex = float(np.std(ex_rets))

    best = float(eq_rets.max())
    worst = float(eq_rets.min())

    # 回撤（策略收益）
    cum_series = np.cumprod(1 + eq_rets / 100)
    peak = np.maximum.accumulate(cum_series)
    dd = (cum_series - peak) / peak
    mdd = float(dd.min())

    # Calmar
    calmar = (cum_eq / 100) / abs(mdd) if mdd < 0 else 0

    avg_hit5 = float(np.mean([r['hit_up5'] for r in results]))
    avg_hit0 = float(np.mean([r['hit_up0'] for r in results]))

    print(f"\n【每日Top{top_n}】 {n_days}个交易日")
    print(f"  ┌──────────────────────────────────┐")
    print(f"  │ 策略累积收益: {cum_eq:>+10.2f}%     │")
    print(f"  │ SPY基准累积:  {cum_spy:>+10.2f}%     │")
    print(f"  │ 超额累积:     {cum_ex:>+10.2f}%     │")
    print(f"  ├──────────────────────────────────┤")
    print(f"  │ 单笔均收益:   {avg_eq:>+9.4f}%     │")
    print(f"  │ 胜率(涨>0):   {win_eq:>8.1%}        │")
    print(f"  │ 夏普(年化):   {sp_eq:>8.3f}        │")
    print(f"  │ 最大回撤:     {mdd*100:>8.1f}%        │")
    print(f"  │ Calmar比率:   {calmar:>8.3f}        │")
    print(f"  │ 涨>5%命中:    {avg_hit5:>8.1%}        │")
    print(f"  │ 涨>0胜率:     {avg_hit0:>8.1%}        │")
    print(f"  │ 单日最佳:     {best:>+9.2f}%     │")
    print(f"  │ 单日最差:     {worst:>+9.2f}%     │")
    print(f"  └──────────────────────────────────┘")

    # 逐年
    from collections import defaultdict
    yearly = defaultdict(list)
    for r in results:
        y = r['date'][:4]
        yearly[y].append(r)

    if len(yearly) > 1:
        print(f"\n  {'年份':>6} {'天数':>5} {'策略均':>9} {'超额均':>9} {'SPY':>8} {'胜率':>7} {'夏普':>8}")
        for yr in sorted(yearly.keys()):
            yr_data = yearly[yr]
            y_eq = np.array([x['eq_ret'] for x in yr_data])
            y_ex = np.array([x['excess'] for x in yr_data])
            y_sp = np.array([x['spy_ret'] for x in yr_data])

            y_cum = (np.prod(1 + y_eq / 100) - 1) * 100
            y_cum_ex = (np.prod(1 + y_ex / 100) - 1) * 100
            y_cum_spy = (np.prod(1 + y_sp / 100) - 1) * 100
            y_avg = float(np.mean(y_eq))
            y_ex_avg = float(np.mean(y_ex))
            y_sp_avg = float(np.mean(y_sp))
            y_std = float(np.std(y_eq))
            y_sp_sharpe = y_avg / y_std * math.sqrt(252) if y_std > 0 else 0
            y_win = sum(1 for r in y_eq if r > 0) / len(y_eq)
            print(f"  {yr:>6} {len(yr_data):>5} {y_avg:>+8.3f}% {y_ex_avg:>+8.3f}% {y_sp_avg:>+7.3f}% {y_win:>6.1%} {y_sp_sharpe:>7.3f}")

print(f"\n总耗时: {time.time() - T0:.0f}s")
