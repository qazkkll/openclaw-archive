#!/usr/bin/env python3
"""
CEO: XGBoost Parameter Sweep (hold period + stop loss)
Find optimal configuration for cn-alpha-v2.0.
"""

import pandas as pd, numpy as np, json, time, os, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

t0 = time.time()
print(f"[CEO] XGB Param Sweep {time.strftime('%Y-%m-%d %H:%M')}")

# Load
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
print(f"  Data: {len(df):,} ({time.time()-t0:.0f}s)")

# Price lookup
print("  Price lookup...")
price_lookup = dict(zip(zip(df['sym'], df['date']), df['close']))

# Features
print("  Features...")
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

# Market regime per date
date_info = df.groupby('date').agg(breadth=('breadth', 'mean'), mkt_ret20=('mkt_ret20', 'mean')).reset_index()
date_regime = {}
for _, row in date_info.iterrows():
    if row['breadth'] > 0.5 and row['mkt_ret20'] > 0:
        date_regime[row['date']] = 'bull'
    elif row['breadth'] < 0.3 or row['mkt_ret20'] < -0.05:
        date_regime[row['date']] = 'bear'
    else:
        date_regime[row['date']] = 'cautious'

XGB_FEATURES = [
    'ret5', 'ret10', 'ret20', 'ma20_bias', 'ma60_bias',
    'vol5', 'vol20', 'rsi_14', 'macd_hist', 'atr_pct', 'vol_ratio',
    'total_net_5d', 'lg_net_5d', 'md_net_5d', 'elg_net_5d',
    'total_net_20d', 'lg_net_20d', 'md_net_20d', 'elg_net_20d',
    'total_net_5d_rk', 'lg_net_5d_rk', 'md_net_5d_rk', 'elg_net_5d_rk',
    'breadth', 'mkt_ret20'
]

TOP_N = 15
COST = 0.0015

all_dates = sorted(df['date'].unique())
def int_to_dt(d): return datetime(int(str(d)[:4]), int(str(d)[4:6]), int(str(d)[6:8]))
def dt_to_int(d): return int(d.strftime('%Y%m%d'))

# Walk-Forward folds (use fewer folds for speed)
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

print(f"  {len(folds)} folds")

import xgboost as xgb

# Parameter sweep
configs = [
    {'hold': 5, 'sl': -0.01, 'label': 'H5_SL1'},
    {'hold': 5, 'sl': -0.03, 'label': 'H5_SL3'},
    {'hold': 5, 'sl': -0.05, 'label': 'H5_SL5'},
    {'hold': 10, 'sl': -0.01, 'label': 'H10_SL1'},
    {'hold': 10, 'sl': -0.03, 'label': 'H10_SL3'},
    {'hold': 10, 'sl': -0.05, 'label': 'H10_SL5'},
    {'hold': 10, 'sl': -0.08, 'label': 'H10_SL8'},
    {'hold': 15, 'sl': -0.03, 'label': 'H15_SL3'},
    {'hold': 15, 'sl': -0.05, 'label': 'H15_SL5'},
    {'hold': 20, 'sl': -0.03, 'label': 'H20_SL3'},
    {'hold': 20, 'sl': -0.05, 'label': 'H20_SL5'},
]

results = {c['label']: {'port_rets': [], 'fold_sharpes': [], 'regime': {'bull': [], 'cautious': [], 'bear': []}} for c in configs}

for fold_idx, fold in enumerate(folds):
    t1 = time.time()
    print(f"  Fold {fold_idx+1}/{len(folds)}", end="")
    
    # Train XGBoost once per fold
    train = df[df['date'].isin(fold['train_dates'])].dropna(subset=XGB_FEATURES + ['fwd_10d'])
    if len(train) < 1000:
        print(" SKIP")
        continue
    
    model = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=4, verbosity=0)
    model.fit(train[XGB_FEATURES].fillna(0), train['fwd_10d'])
    
    # Predict on all test dates
    test_data = df[df['date'].isin(fold['test_dates'])].copy()
    test_data['xgb_score'] = model.predict(test_data[XGB_FEATURES].fillna(0))
    test_data['xgb_rank'] = test_data.groupby('date')['xgb_score'].rank(pct=True)
    
    # Filter
    test_data = test_data[
        (test_data['close'] >= 3) & (test_data['close'] <= 200) &
        (~test_data['sym'].str.contains('ST|退市', na=False)) & (test_data['volume'] > 0)
    ]
    
    test_dates = fold['test_dates']
    
    # For each config
    for cfg in configs:
        hold = cfg['hold']
        sl = cfg['sl']
        label = cfg['label']
        
        rebal_dates = test_dates[::hold]
        strat_rets = []
        
        for rd in rebal_dates:
            day = test_data[test_data['date'] == rd]
            if len(day) < 50:
                continue
            
            top = day.nlargest(TOP_N, 'xgb_score')
            
            rebal_idx = test_dates.index(rd)
            exit_idx = min(rebal_idx + hold, len(test_dates) - 1)
            exit_date = test_dates[exit_idx]
            if exit_date == rd:
                continue
            
            rets = []
            for _, row in top.iterrows():
                exit_price = price_lookup.get((row['sym'], exit_date))
                if exit_price is None:
                    continue
                ret = exit_price / row['close'] - 1
                
                # Stop loss
                for i in range(rebal_idx + 1, exit_idx + 1):
                    ip = price_lookup.get((row['sym'], test_dates[i]))
                    if ip is not None and ip / row['close'] - 1 <= sl:
                        ret = sl
                        break
                
                ret -= COST
                rets.append(ret)
            
            if rets:
                port_ret = np.mean(rets)
                strat_rets.append(port_ret)
                regime = date_regime.get(rd, 'cautious')
                results[label]['regime'][regime].append(port_ret)
        
        if len(strat_rets) > 2:
            avg = np.mean(strat_rets)
            std = np.std(strat_rets)
            rebal_per_year = 252 / hold
            fs = (avg * rebal_per_year) / (std * np.sqrt(rebal_per_year)) if std > 0 else 0
            results[label]['fold_sharpes'].append(fs)
            results[label]['port_rets'].extend(strat_rets)
    
    print(f" done ({time.time()-t1:.0f}s)")

# Results
print("\n" + "=" * 100)
print("📊 XGBoost Parameter Sweep")
print("=" * 100)

summary = []
for cfg in configs:
    label = cfg['label']
    data = results[label]
    pr = data['port_rets']
    fs = data['fold_sharpes']
    if not pr or not fs:
        continue
    avg = np.mean(pr)
    std = np.std(pr)
    wr = np.mean([r > 0 for r in pr])
    rebal_per_year = 252 / cfg['hold']
    ann_ret = avg * rebal_per_year
    ann_std = std * np.sqrt(rebal_per_year)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0
    regime_avg = {r: np.mean(v) if v else 0 for r, v in data['regime'].items()}
    summary.append({
        'label': label, 'hold': cfg['hold'], 'sl': cfg['sl'],
        'n_rebal': len(pr), 'avg_ret': avg, 'win_rate': wr,
        'ann_ret': ann_ret, 'sharpe': sharpe,
        'wf_sharpe_mean': np.mean(fs), 'wf_sharpe_std': np.std(fs),
        'regime': regime_avg
    })

summary.sort(key=lambda x: x['wf_sharpe_mean'], reverse=True)

print(f"\n{'Config':>12} {'Hold':>5} {'SL':>5} {'Rebal':>6} {'AvgRet':>8} {'WinRate':>8} {'AnnRet':>8} {'Sharpe':>8} {'WF Sharpe':>14} {'Bull':>7} {'Caut':>7} {'Bear':>7}")
print("-" * 110)
for s in summary:
    print(f"{s['label']:>12} {s['hold']:>5} {s['sl']*100:>4.0f}% {s['n_rebal']:>6} {s['avg_ret']:>8.4f} {s['win_rate']:>8.1%} {s['ann_ret']:>8.1%} {s['sharpe']:>8.2f} {s['wf_sharpe_mean']:>6.2f}±{s['wf_sharpe_std']:.2f} {s['regime']['bull']:>7.3f} {s['regime']['cautious']:>7.3f} {s['regime']['bear']:>7.3f}")

output = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'experiment': 'xgb_param_sweep',
    'summary': summary,
    'best': summary[0]['label'] if summary else None
}
with open('research/ceo_xgb_param_sweep.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n✅ Best: {summary[0]['label']} (WF Sharpe: {summary[0]['wf_sharpe_mean']:.2f})")
print(f"   Time: {time.time()-t0:.0f}s")
