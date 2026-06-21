#!/usr/bin/env python3
"""Paper Trade v2.1 — 基于CEO param_sweep框架，加alpha/信号分层/成本"""
import pandas as pd, numpy as np, json, time, os, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))
t0 = time.time()
print(f"[PT] {time.strftime('%H:%M')}", flush=True)

# 加载 (和CEO脚本完全一样)
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
print(f"  Data: {len(df):,} ({time.time()-t0:.0f}s)", flush=True)

price_lookup = dict(zip(zip(df['sym'], df['date']), df['close']))

# 特征 (和CEO完全一样)
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
print(f"  Features: {time.time()-t0:.0f}s", flush=True)

XGB_FEATURES = [
    'ret5', 'ret10', 'ret20', 'ma20_bias', 'ma60_bias',
    'vol5', 'vol20', 'rsi_14', 'macd_hist', 'atr_pct', 'vol_ratio',
    'total_net_5d', 'lg_net_5d', 'md_net_5d', 'elg_net_5d',
    'total_net_20d', 'lg_net_20d', 'md_net_20d', 'elg_net_20d',
    'total_net_5d_rk', 'lg_net_5d_rk', 'md_net_5d_rk', 'elg_net_5d_rk',
    'breadth', 'mkt_ret20'
]
HOLD = 10
SL = -0.03
TOP_N = 15

# WF folds (和CEO一样)
all_dates = sorted(df['date'].unique())
def int_to_dt(d): return datetime(int(str(d)[:4]), int(str(d)[4:6]), int(str(d)[6:8]))
def dt_to_int(d): return int(d.strftime('%Y%m%d'))

folds = []
train_start_dt = datetime(2016, 1, 1)
while True:
    train_end_dt = train_start_dt + timedelta(days=365*2)
    test_start_dt = train_end_dt
    test_end_dt = test_start_dt + timedelta(days=182)
    if test_end_dt > int_to_dt(all_dates[-1]):
        break
    td = [d for d in all_dates if dt_to_int(train_start_dt) <= d <= dt_to_int(train_end_dt)]
    ted = [d for d in all_dates if dt_to_int(test_start_dt) <= d <= dt_to_int(test_end_dt)]
    if len(td) >= 200 and len(ted) >= 20:
        folds.append({'train_dates': td, 'test_dates': ted})
    train_start_dt += timedelta(days=182)

print(f"  {len(folds)} folds ({time.time()-t0:.0f}s)", flush=True)

# Paper Trade模拟
print("[PT] 模拟...", flush=True)
import xgboost as xgb

all_period_results = []  # 每个rebalancing的详细结果

for fold_idx, fold in enumerate(folds):
    t1 = time.time()
    train = df[df['date'].isin(fold['train_dates'])].dropna(subset=XGB_FEATURES + ['fwd_10d'])
    if len(train) < 1000:
        continue
    
    model = xgb.XGBRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=4, verbosity=0
    )
    model.fit(train[XGB_FEATURES].fillna(0), train['fwd_10d'])
    
    test_data = df[df['date'].isin(fold['test_dates'])].copy()
    test_data['xgb_score'] = model.predict(test_data[XGB_FEATURES].fillna(0))
    test_data = test_data[
        (test_data['close'] >= 3) & (test_data['close'] <= 200) &
        (~test_data['sym'].str.contains('ST|退市', na=False)) & (test_data['volume'] > 0)
    ]
    
    test_dates = fold['test_dates']
    rebal_dates = test_dates[::HOLD]
    
    for rd in rebal_dates:
        day = test_data[test_data['date'] == rd]
        if len(day) < 50:
            continue
        top = day.nlargest(TOP_N, 'xgb_score')
        
        rebal_idx = test_dates.index(rd)
        exit_idx = min(rebal_idx + HOLD, len(test_dates) - 1)
        exit_date = test_dates[exit_idx]
        if exit_date == rd:
            continue
        
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
                for j in range(rebal_idx + 1, exit_idx + 1):
                    ip = price_lookup.get((row['sym'], test_dates[j]))
                    if ip is not None and ip / row['close'] - 1 <= SL:
                        ret = SL
                        break
                rets.append(ret)
            
            if rets:
                all_period_results.append({
                    'rebal': rd, 'exit': exit_date, 'level': level,
                    'avg_ret': np.mean(rets), 'alpha': np.mean(rets) - mkt_ret,
                    'mkt_ret': mkt_ret,
                })
    
    print(f"  F{fold_idx+1}/{len(folds)} ({time.time()-t1:.0f}s)", flush=True)

# 输出
print(f"\n{'='*100}", flush=True)
print("📊 Paper Trade: cn-alpha-v2.1 (XGBoost + SL-3%)", flush=True)
print(f"{'='*100}", flush=True)

for level, label in [('top5', '🟢🟢精品'), ('top10', '🟢强信号'), ('top15', '🟡观察')]:
    data = [r for r in all_period_results if r['level'] == level]
    if not data:
        continue
    rets = [r['avg_ret'] for r in data]
    alphas = [r['alpha'] for r in data]
    avg = np.mean(rets); std = np.std(rets)
    ann_ret = avg * (252/HOLD); ann_std = std * np.sqrt(252/HOLD)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0
    wr = np.mean([r > 0 for r in rets])
    alpha_pos = np.mean([a > 0 for a in alphas])
    eqs = [1.0]
    for r in rets: eqs.append(eqs[-1]*(1+r))
    pk = eqs[0]; mdd = 0
    for e in eqs:
        if e > pk: pk = e
        dd = (e-pk)/pk
        if dd < mdd: mdd = dd
    
    print(f"\n{label} ({level})", flush=True)
    print(f"  期数: {len(data)} | 收益: {avg*100:.2f}%/期 | 年化: {ann_ret*100:.1f}%", flush=True)
    print(f"  Sharpe: {sharpe:.3f} {'✅' if sharpe>1 else '⚠️'} | 胜率: {wr:.1%} | Alpha正: {alpha_pos:.1%} {'✅' if alpha_pos>.6 else '⚠️'} | DD: {mdd*100:.1f}% {'✅' if mdd>-0.20 else '⚠️'}", flush=True)

# 成本敏感性
t15 = [r for r in all_period_results if r['level'] == 'top15']
print(f"\n💰 成本敏感性 (Top15)", flush=True)
for cl, c in [('0.1%', 0.001), ('0.15%', 0.0015), ('0.3%', 0.003)]:
    rc = [r['avg_ret']-c*2 for r in t15]
    a = np.mean(rc); s = np.std(rc)
    sh = (a*252/HOLD)/(s*np.sqrt(252/HOLD)) if s>0 else 0
    w = np.mean([r>0 for r in rc])
    eq = [1.0]
    for r in rc: eq.append(eq[-1]*(1+r))
    pk=eq[0]; md=0
    for e in eq:
        if e>pk: pk=e
        d=(e-pk)/pk
        if d<md: md=d
    print(f"  {cl}: Sharpe={sh:.3f} 年化={a*252/HOLD*100:.1f}% 胜率={w:.1%} DD={md*100:.1f}%", flush=True)

# 门限
tr = [r['avg_ret'] for r in t15]
ta = [r['alpha'] for r in t15]
ts = (np.mean(tr)*252/HOLD)/(np.std(tr)*np.sqrt(252/HOLD)) if np.std(tr)>0 else 0
tw = np.mean([r>0 for r in tr])
tap = np.mean([a>0 for a in ta])
eq=[1.0]
for r in tr: eq.append(eq[-1]*(1+r))
pk=eq[0]; tm=0
for e in eq:
    if e>pk: pk=e
    d=(e-pk)/pk
    if d<tm: tm=d

print(f"\n🏁 门限判定", flush=True)
for n,p,v in [('Sharpe>1.0',ts>1,f'{ts:.3f}'),('Alpha正>60%',tap>.6,f'{tap:.1%}'),('DD<-20%',tm>-.2,f'{tm*100:.1f}%'),('胜率>50%',tw>.5,f'{tw:.1%}')]:
    print(f"  {'✅' if p else '❌'} {n}: {v}", flush=True)

print(f"\n⏱️ {time.time()-t0:.0f}s", flush=True)
