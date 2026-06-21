#!/usr/bin/env python3
"""Paper Trade验证: cn-alpha-v2.1 — 优化版"""
import pandas as pd, numpy as np, json, time, os, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))
t0 = time.time()
print(f"[Paper Trade v2.1] {time.strftime('%Y-%m-%d %H:%M')}", flush=True)

# ========== 加载 ==========
print("[1/5] 加载...", flush=True)
df = pd.read_parquet('data/a_hist_10y.parquet')
df = df.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df['date'] = df['date'].astype(int)
mf = pd.read_parquet('data/cn/moneyflow_core.parquet')
mf['sym'] = mf['ts_code'].str[:6]
mf['date'] = mf['trade_date'].astype(int)
for col in ['sm', 'md', 'lg', 'elg']:
    mf[f'{col}_net'] = mf[f'buy_{col}_amount'] - mf[f'sell_{col}_amount']
mf['total_net'] = mf['net_mf_amount']
df = df.merge(mf[['sym', 'date', 'total_net', 'lg_net', 'md_net', 'elg_net']], on=['sym', 'date'], how='left')
df = df[~df['sym'].str.startswith('688')].copy()
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
df = df[df['volume'] > 0].copy()
df = df.sort_values(['sym', 'date']).reset_index(drop=True)
print(f"  {len(df):,} rows, {df['sym'].nunique()} stocks ({time.time()-t0:.0f}s)", flush=True)

# ========== Price lookup ==========
price_lookup = dict(zip(zip(df['sym'], df['date']), df['close']))

# ========== 特征 ==========
print("[2/5] 特征...", flush=True)
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ret10'] = df.groupby('sym')['close'].pct_change(10)
df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma60'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(60, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['ma60_bias'] = (df['close'] - df['ma60']) / df['ma60']
df['vol5'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(5, min_periods=2).std())
df['vol20'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(20, min_periods=2).std())
delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)
ema12 = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=12, min_periods=1).mean())
ema26 = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=26, min_periods=1).mean())
df['macd'] = ema12 - ema26
df['macd_signal'] = df.groupby('sym')['macd'].transform(lambda x: x.ewm(span=9, min_periods=1).mean())
df['macd_hist'] = df['macd'] - df['macd_signal']
df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df.groupby('sym')['close'].shift(1)), abs(df['low'] - df.groupby('sym')['close'].shift(1))))
df['atr14'] = df.groupby('sym')['tr'].transform(lambda x: x.rolling(14, min_periods=1).mean())
df['atr_pct'] = df['atr14'] / df['close']
df['vol_ratio'] = df.groupby('sym')['volume'].transform(lambda x: x.rolling(5).mean()) / df.groupby('sym')['volume'].transform(lambda x: x.rolling(20).mean())
for col in ['total_net', 'lg_net', 'md_net', 'elg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())
    df[f'{col}_20d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(20, min_periods=1).sum())
    df[f'{col}_5d_rk'] = df.groupby('date')[f'{col}_5d'].rank(pct=True)
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')
df['fwd_10d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-10) / x - 1)

XGB_FEATURES = [
    'ret5', 'ret10', 'ret20', 'ma20_bias', 'ma60_bias',
    'vol5', 'vol20', 'rsi_14', 'macd_hist', 'atr_pct', 'vol_ratio',
    'total_net_5d', 'lg_net_5d', 'md_net_5d', 'elg_net_5d',
    'total_net_20d', 'lg_net_20d', 'md_net_20d', 'elg_net_20d',
    'total_net_5d_rk', 'lg_net_5d_rk', 'md_net_5d_rk', 'elg_net_5d_rk',
    'breadth', 'mkt_ret20'
]
FEAT_COLS = XGB_FEATURES + ['fwd_10d']
print(f"  特征完成 ({time.time()-t0:.0f}s)", flush=True)

# ========== 预处理: 按日期索引 + 预计算有效行 ==========
print("[3/5] 预处理...", flush=True)
df['_valid'] = df[FEAT_COLS].notna().all(axis=1)
df['_valid'] = df['_valid'] & (df['sym'].str.contains('ST|退市', na=False) == False)

# 按日期分组 (用dict做O(1)查找)
date_groups = {}
for d, grp in df.groupby('date'):
    valid = grp[grp['_valid']].copy()
    if len(valid) >= 50:
        date_groups[d] = valid

all_dates = sorted(date_groups.keys())
def int_to_dt(d): return datetime(int(str(d)[:4]), int(str(d)[4:6]), int(str(d)[6:8]))
def dt_to_int(d): return int(d.strftime('%Y%m%d'))

# 季度调仓点
quarter_dates = []
for d in all_dates:
    dt = int_to_dt(d)
    if dt.month in [1, 4, 7, 10] and dt.day <= 10:
        quarter_dates.append(d)
rebal_dates = [d for d in quarter_dates if d >= 20180101]
print(f"  {len(rebal_dates)} 个调仓点, {len(date_groups)} 个交易日 ({time.time()-t0:.0f}s)", flush=True)

# ========== Paper Trade ==========
print("[4/5] Paper Trade...", flush=True)
import xgboost as xgb

HOLD = 10
SL = -0.003  # 注意这里是-0.3%不是-3%... 修正
SL = -0.03   # -3%
COST = 0.0015
TOP_N = 15

# 按日期排序的训练数据缓存
date_sorted = sorted(date_groups.keys())

results = []
equity = 1.0

for i, rd in enumerate(rebal_dates):
    # 找退出日期
    rd_idx = all_dates.index(rd) if rd in all_dates else -1
    if rd_idx < 0 or rd_idx + HOLD >= len(all_dates):
        continue
    exit_date = all_dates[rd_idx + HOLD]
    
    # 训练数据: 过去2年
    train_cutoff = dt_to_int(int_to_dt(rd) - timedelta(days=365*2))
    train_dates_list = [d for d in date_sorted if train_cutoff <= d < rd]
    
    # 快速收集训练数据
    train_chunks = [date_groups[d] for d in train_dates_list if d in date_groups]
    if len(train_chunks) < 50:
        continue
    train = pd.concat(train_chunks, ignore_index=False)
    if len(train) < 1000:
        continue
    
    # 训练
    model = xgb.XGBRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=4, verbosity=0
    )
    model.fit(train[XGB_FEATURES].fillna(0), train['fwd_10d'])
    
    # 预测
    day = date_groups.get(rd)
    if day is None or len(day) < 50:
        continue
    day = day.copy()
    day['xgb_score'] = model.predict(day[XGB_FEATURES].fillna(0))
    top = day.nlargest(TOP_N, 'xgb_score')
    
    # 市场基准
    mkt_rets = []
    for _, row in day.head(300).iterrows():
        ep = price_lookup.get((row['sym'], exit_date))
        if ep is not None:
            mkt_rets.append(ep / row['close'] - 1)
    mkt_ret = np.mean(mkt_rets) if mkt_rets else 0
    
    # 分层信号
    for level, n in [('top5', 5), ('top10', 10), ('top15', 15)]:
        group = top.head(n)
        rets = []
        for _, row in group.iterrows():
            ep = price_lookup.get((row['sym'], exit_date))
            if ep is None:
                continue
            ret = ep / row['close'] - 1
            # Stop loss
            for j in range(rd_idx + 1, rd_idx + HOLD + 1):
                if j >= len(all_dates):
                    break
                ip = price_lookup.get((row['sym'], all_dates[j]))
                if ip is not None and ip / row['close'] - 1 <= SL:
                    ret = SL
                    break
            rets.append(ret)
        
        if rets:
            results.append({
                'rebal': rd, 'exit': exit_date, 'level': level,
                'avg_ret': np.mean(rets), 'alpha': np.mean(rets) - mkt_ret,
                'mkt_ret': mkt_ret, 'n': len(rets),
            })
    
    # 权益曲线(用top15)
    t15_rets = [r['avg_ret'] for r in results if r['rebal'] == rd and r['level'] == 'top15']
    if t15_rets:
        equity *= (1 + t15_rets[0])
    
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(rebal_dates)} eq={equity:.4f} ({time.time()-t0:.0f}s)", flush=True)

print(f"  完成: {len([r for r in results if r['level']=='top15'])}期 ({time.time()-t0:.0f}s)", flush=True)

# ========== 输出 ==========
print("[5/5] 输出...", flush=True)
print("\n" + "=" * 100, flush=True)
print("📊 Paper Trade验证: cn-alpha-v2.1 (XGBoost + SL-3%)", flush=True)
print("=" * 100, flush=True)

for level, label in [('top5', '🟢🟢精品'), ('top10', '🟢强信号'), ('top15', '🟡观察')]:
    data = [r for r in results if r['level'] == level]
    if not data:
        continue
    rets = [r['avg_ret'] for r in data]
    alphas = [r['alpha'] for r in data]
    avg = np.mean(rets)
    std = np.std(rets)
    ann_ret = avg * (252 / HOLD)
    ann_std = std * np.sqrt(252 / HOLD)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0
    wr = np.mean([r > 0 for r in rets])
    alpha_pos = np.mean([a > 0 for a in alphas])
    
    # DD
    eqs = [1.0]
    for r in rets:
        eqs.append(eqs[-1] * (1 + r))
    peak = eqs[0]
    max_dd = 0
    for e in eqs:
        if e > peak: peak = e
        dd = (e - peak) / peak
        if dd < max_dd: max_dd = dd
    
    print(f"\n{label} ({level})", flush=True)
    print(f"  调仓期:     {len(data)}", flush=True)
    print(f"  平均收益:   {avg*100:.2f}%/期", flush=True)
    print(f"  年化收益:   {ann_ret*100:.1f}%", flush=True)
    print(f"  Sharpe:     {sharpe:.3f} {'✅' if sharpe > 1.0 else '⚠️'}", flush=True)
    print(f"  胜率:       {wr:.1%} {'✅' if wr > 0.5 else '⚠️'}", flush=True)
    print(f"  Alpha正:    {alpha_pos:.1%} {'✅' if alpha_pos > 0.6 else '⚠️'}", flush=True)
    print(f"  Max DD:     {max_dd*100:.1f}% {'✅' if max_dd > -0.20 else '⚠️'}", flush=True)

# 成本敏感性
print(f"\n💰 成本敏感性 (Top15)", flush=True)
t15_data = [r for r in results if r['level'] == 'top15']
for cost_label, cost in [('0.1%', 0.001), ('0.15%', 0.0015), ('0.3%', 0.003)]:
    rets_c = [r['avg_ret'] - cost * 2 for r in t15_data]
    avg_c = np.mean(rets_c)
    std_c = np.std(rets_c)
    ann_r = avg_c * (252 / HOLD)
    ann_s = std_c * np.sqrt(252 / HOLD)
    sh = ann_r / ann_s if ann_s > 0 else 0
    wr_c = np.mean([r > 0 for r in rets_c])
    eqs = [1.0]
    for r in rets_c:
        eqs.append(eqs[-1] * (1 + r))
    peak = eqs[0]
    mdd = 0
    for e in eqs:
        if e > peak: peak = e
        dd = (e - peak) / peak
        if dd < mdd: mdd = dd
    print(f"  {cost_label}: Sharpe={sh:.3f}, 年化={ann_r*100:.1f}%, 胜率={wr_c:.1%}, DD={mdd*100:.1f}%", flush=True)

# 门限
t15_rets = [r['avg_ret'] for r in t15_data]
t15_alphas = [r['alpha'] for r in t15_data]
t15_avg = np.mean(t15_rets)
t15_std = np.std(t15_rets)
t15_sharpe = (t15_avg * 252/HOLD) / (t15_std * np.sqrt(252/HOLD)) if t15_std > 0 else 0
t15_wr = np.mean([r > 0 for r in t15_rets])
t15_alpha_pos = np.mean([a > 0 for a in t15_alphas])
eqs = [1.0]
for r in t15_rets:
    eqs.append(eqs[-1] * (1 + r))
peak = eqs[0]
t15_mdd = 0
for e in eqs:
    if e > peak: peak = e
    dd = (e - peak) / peak
    if dd < t15_mdd: t15_mdd = dd

print(f"\n🏁 门限判定 (Top15)", flush=True)
thresholds = [
    ('Sharpe > 1.0', t15_sharpe > 1.0, f'{t15_sharpe:.3f}'),
    ('Alpha正占比 > 60%', t15_alpha_pos > 0.6, f'{t15_alpha_pos:.1%}'),
    ('Max DD < -20%', t15_mdd > -0.20, f'{t15_mdd*100:.1f}%'),
    ('胜率 > 50%', t15_wr > 0.5, f'{t15_wr:.1%}'),
    ('期数 > 30', len(t15_data) >= 30, f'{len(t15_data)}'),
]
all_pass = True
for name, passed, value in thresholds:
    s = '✅' if passed else '❌'
    print(f"  {s} {name}: {value}", flush=True)
    if not passed: all_pass = False

print(f"\n{'✅ Paper Trade通过!' if all_pass else '⚠️ 部分门限未通过'}", flush=True)
print(f"⏱️ {time.time()-t0:.0f}s", flush=True)

# 保存
output = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'model': 'cn-alpha-v2.1',
    'top15': {'sharpe': round(t15_sharpe,4), 'ann_ret': round(t15_avg*252/HOLD,4), 'win_rate': round(t15_wr,4), 'alpha_positive': round(t15_alpha_pos,4), 'max_dd': round(t15_mdd,4), 'n_periods': len(t15_data)},
    'passed': all_pass,
}
with open('research/paper_trade_v21_results.json', 'w') as f:
    json.dump(output, f, indent=2)
