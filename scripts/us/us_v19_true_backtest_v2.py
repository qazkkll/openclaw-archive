"""
绿箭v19 真·干净回测 v2
使用 us_ml_feats_v3_dated.parquet

修正v1的问题：
1. 剔除 label_5d_pct 极端值（>50%和<-50%），防止小盘垃圾股扭曲结果
2. 加交易成本（0.1%单边）
3. 每天最多买x只，结果对比5/10/20/50
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
print("绿箭v19 真·干净回测 v2")
print("=" * 60)

# 1. 加载数据
print("加载 us_ml_feats_v3_dated.parquet ...")
df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v3_dated.parquet")

# 2. 剔除极端值
print("剔除极端值 (>50%或<-50%) ...")
before = len(df)
df = df[(df['label_5d_pct'] >= -50) & (df['label_5d_pct'] <= 50)].copy()
print(f"  剔除前: {before:,} → 剔除后: {len(df):,} (剔除 {before - len(df):,}行)")

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
print(f"  日期范围: {dates[0]} ~ {dates[-1]} ({len(dates)}个交易日)")
print(f"  股票数: {df['sym'].nunique()}")

# 4. 切分
split_idx = int(len(dates) * 0.7)
test_dates = dates[split_idx:]
print(f"  训练截止: {dates[split_idx - 1]}")
print(f"  回测开始: {test_dates[0]}")
print(f"  回测天数: {len(test_dates)}")

# 5. 交易成本
TC = 0.1  # 单边 0.1%

# 6. 逐日回测（每20天重训）
RETRAIN_INTERVAL = 20
top_ns = [5, 10, 20, 50]
daily_results = {n: [] for n in top_ns}
model = None

print("\n逐日回测...")

for di, test_date in enumerate(test_dates):
    if di % 10 == 0:
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
    price_day = day_df['price'].values

    pu5 = model.predict_proba(X_day)[:, 4]

    for top_n in top_ns:
        idx = np.argsort(-pu5)[:min(top_n, len(pu5))]
        if len(idx) == 0:
            continue

        rets = pct_day[idx]
        # 扣交易成本（买入+卖出 = 2倍TC）
        eq_ret = float(np.mean(rets)) - 2 * TC

        daily_results[top_n].append({
            'date': str(test_date),
            'eq_ret': eq_ret,
            'hit_up5': float((rets > 5).mean()),
            'hit_up0': float((rets > 0).mean()),
            'n': len(idx),
            'max_ret': float(rets.max()),
            'min_ret': float(rets.min()),
            'avg_price': float(np.mean(price_day[idx])),
            'syms': ','.join(sym_day[idx][:5]),
        })

# 7. 输出
print("\n" + "=" * 60)
print("回测结果（已剔除极端值+已扣交易成本）")
print("=" * 60)

for top_n in top_ns:
    results = daily_results[top_n]
    if not results:
        continue

    eq_rets = np.array([r['eq_ret'] for r in results])
    eq_rets_clip = np.clip(eq_rets, -50, 50)

    n_days = len(eq_rets_clip)
    cum_eq = (np.prod(1 + eq_rets_clip / 100) - 1) * 100
    avg_eq = float(np.mean(eq_rets_clip))
    med_eq = float(np.median(eq_rets_clip))
    std_eq = float(np.std(eq_rets_clip))
    sp_eq = avg_eq / std_eq * math.sqrt(252) if std_eq > 0 else 0
    win_eq = float((eq_rets_clip > 0).mean())
    best = float(eq_rets_clip.max())
    worst = float(eq_rets_clip.min())

    cum_series = np.cumprod(1 + eq_rets_clip / 100)
    peak = np.maximum.accumulate(cum_series)
    dd = (cum_series - peak) / peak
    mdd = float(dd.min())

    avg_hit5 = float(np.mean([r['hit_up5'] for r in results]))
    avg_hit0 = float(np.mean([r['hit_up0'] for r in results]))

    print(f"\n【每日Top{top_n}】 {n_days}个交易日")
    print(f"  累积收益: {cum_eq:+.2f}%")
    print(f"  单笔均收益: {avg_eq:+.4f}% | 中位: {med_eq:+.4f}%")
    print(f"  胜率(涨>0): {win_eq:.1%}")
    print(f"  夏普(年化): {sp_eq:.3f}")
    print(f"  最大回撤: {mdd * 100:.1f}%")
    print(f"  单日最佳: {best:+.2f}% | 单日最差: {worst:+.2f}%")
    print(f"  涨>5%平均命中: {avg_hit5:.1%}")
    print(f"  涨>0平均: {avg_hit0:.1%}")

    # 年度
    from collections import defaultdict
    yearly = defaultdict(list)
    for r in results:
        y = r['date'][:4]
        if y >= '2024':
            yearly[y].append(r['eq_ret'])

    if len(yearly) > 1:
        print(f"\n  {'年份':>6} {'天数':>5} {'均收益':>9} {'胜率':>7} {'夏普':>8} {'累积':>9}")
        for yr in sorted(yearly.keys()):
            yr_rets = yearly[yr]
            y_avg = float(np.mean(yr_rets))
            y_std = float(np.std(yr_rets))
            y_sp = y_avg / y_std * math.sqrt(252) if y_std > 0 else 0
            y_cum = (np.prod(1 + np.array(yr_rets) / 100) - 1) * 100
            y_win = sum(1 for r in yr_rets if r > 0) / len(yr_rets)
            print(f"  {yr:>6} {len(yr_rets):>5} {y_avg:>+8.3f}% {y_win:>6.1%} {y_sp:>7.3f} {y_cum:>+8.2f}%")

print(f"\n总耗时: {time.time() - T0:.0f}s")
