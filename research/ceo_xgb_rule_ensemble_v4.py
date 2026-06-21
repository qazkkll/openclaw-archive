#!/usr/bin/env python3
"""
CEO: XGBoost + Rule Ensemble (v4 - fast, pre-compute everything)
"""

import pandas as pd, numpy as np, json, time, os, sys, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

t0 = time.time()
print(f"[CEO] Ensemble v4 {time.strftime('%Y-%m-%d %H:%M')}")

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

# Pre-compute rule scores for ALL dates at once
print("  Pre-computing rule scores...")
df['rule_score'] = 0.0
df['rule_score'] += (-df['ret20'].fillna(0)).clip(-0.3, 0.3) * 3
df['rule_score'] += df.groupby('date')['total_net_5d'].transform(lambda x: x.fillna(0).rank(pct=True)) * 2
df['rule_score'] += (1 - df.groupby('date')['vol20'].transform(lambda x: x.fillna(x.median()).rank(pct=True))) * 2
df['rule_score'] += (df['rsi_14'].fillna(50) < 35).astype(float) * 1.5
df['rule_score'] += df.groupby('date')['lg_net_5d'].transform(lambda x: x.fillna(0).rank(pct=True)) * 1
df['rule_score'] += (-df['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1
df['rule_rank'] = df.groupby('date')['rule_score'].rank(pct=True)
print(f"  Rule scores done ({time.time()-t0:.0f}s)")

XGB_FEATURES = [
    'ret5', 'ret10', 'ret20', 'ma20_bias', 'ma60_bias',
    'vol5', 'vol20', 'rsi_14', 'macd_hist', 'atr_pct', 'vol_ratio',
    'total_net_5d', 'lg_net_5d', 'md_net_5d', 'elg_net_5d',
    'total_net_20d', 'lg_net_20d', 'md_net_20d', 'elg_net_20d',
    'total_net_5d_rk', 'lg_net_5d_rk', 'md_net_5d_rk', 'elg_net_5d_rk',
    'breadth', 'mkt_ret20'
]

TOP_N = 15
HOLD_DAYS = 10
SL = -0.03
COST = 0.0015

# Walk-Forward folds
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

print(f"  {len(folds)} folds")

import xgboost as xgb

ensemble_configs = {
    'xgb_only': (0.0, 1.0),
    'rule_only': (1.0, 0.0),
    'ens_50_50': (0.5, 0.5),
    'ens_30_70': (0.3, 0.7),
    'ens_70_30': (0.7, 0.3),
    'ens_20_80': (0.2, 0.8),
}

results = {n: {'port_rets': [], 'fold_sharpes': [], 'regime': {'bull': [], 'cautious': [], 'bear': []}} for n in ensemble_configs}

for fold_idx, fold in enumerate(folds):
    print(f"  Fold {fold_idx+1}/{len(folds)}", end="")
    
    # Train XGBoost
    train = df[df['date'].isin(fold['train_dates'])].dropna(subset=XGB_FEATURES + ['fwd_10d'])
    if len(train) < 1000:
        print(" SKIP")
        continue
    
    model = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=4, verbosity=0)
    model.fit(train[XGB_FEATURES].fillna(0), train['fwd_10d'])
    
    # Predict XGB on ALL test dates at once
    test_data = df[df['date'].isin(fold['test_dates'])].copy()
    test_data['xgb_score'] = model.predict(test_data[XGB_FEATURES].fillna(0))
    test_data['xgb_rank'] = test_data.groupby('date')['xgb_score'].rank(pct=True)
    
    # Market regime per date
    date_regime = {}
    for d in fold['test_dates']:
        dd = test_data[test_data['date'] == d]
        if len(dd) > 0:
            mb = dd['breadth'].mean()
            mr = dd['mkt_ret20'].mean()
            if mb > 0.5 and mr > 0: date_regime[d] = 'bull'
            elif mb < 0.3 or mr < -0.05: date_regime[d] = 'bear'
            else: date_regime[d] = 'cautious'
    
    # Filter
    test_data = test_data[
        (test_data['close'] >= 3) & (test_data['close'] <= 200) &
        (~test_data['sym'].str.contains('ST|退市', na=False)) & (test_data['volume'] > 0)
    ]
    
    # Pre-compute top-N for each ensemble weight
    rebal_dates = fold['test_dates'][::HOLD_DAYS]
    
    # For each strategy
    for strat_name, (rw, xw) in ensemble_configs.items():
        strat_rets = []
        
        for rebal_date in rebal_dates:
            day = test_data[test_data['date'] == rebal_date].copy()
            if len(day) < 50:
                continue
            
            day['ens_score'] = day['rule_rank'] * rw + day['xgb_rank'] * xw
            top = day.nlargest(TOP_N, 'ens_score')
            
            # Exit date
            rebal_idx = fold['test_dates'].index(rebal_date)
            exit_idx = min(rebal_idx + HOLD_DAYS, len(fold['test_dates']) - 1)
            exit_date = fold['test_dates'][exit_idx]
            if exit_date == rebal_date:
                continue
            
            # Portfolio return
            rets = []
            for _, row in top.iterrows():
                exit_rows = df[(df['sym'] == row['sym']) & (df['date'] == exit_date)]
                if len(exit_rows) == 0:
                    continue
                ret = exit_rows.iloc[0]['close'] / row['close'] - 1
                
                # Stop loss
                hold = df[(df['sym'] == row['sym']) & (df['date'] > rebal_date) & (df['date'] <= exit_date)]
                if len(hold) > 0 and hold['close'].min() / row['close'] - 1 <= SL:
                    ret = SL
                
                ret -= COST
                rets.append(ret)
            
            if rets:
                port_ret = np.mean(rets)
                strat_rets.append(port_ret)
                regime = date_regime.get(rebal_date, 'cautious')
                results[strat_name]['regime'][regime].append(port_ret)
        
        if len(strat_rets) > 2:
            avg = np.mean(strat_rets)
            std = np.std(strat_rets)
            fs = (avg * 24) / (std * np.sqrt(24)) if std > 0 else 0
            results[strat_name]['fold_sharpes'].append(fs)
            results[strat_name]['port_rets'].extend(strat_rets)
    
    print(f" done ({time.time()-t0:.0f}s)")

# Results
print("\n" + "=" * 100)
print("📊 ENSEMBLE RESULTS (Portfolio-level)")
print("=" * 100)

summary = []
for name, data in results.items():
    pr = data['port_rets']
    fs = data['fold_sharpes']
    if not pr or not fs:
        continue
    avg = np.mean(pr)
    std = np.std(pr)
    wr = np.mean([r > 0 for r in pr])
    ann_ret = avg * 24
    ann_std = std * np.sqrt(24)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0
    regime_avg = {r: np.mean(v) if v else 0 for r, v in data['regime'].items()}
    summary.append({
        'strategy': name, 'n_rebal': len(pr), 'avg_ret': avg, 'win_rate': wr,
        'ann_ret': ann_ret, 'sharpe': sharpe,
        'wf_sharpe_mean': np.mean(fs), 'wf_sharpe_std': np.std(fs),
        'regime': regime_avg
    })

summary.sort(key=lambda x: x['wf_sharpe_mean'], reverse=True)

print(f"\n{'Strategy':>15} {'Rebal':>6} {'AvgRet':>8} {'WinRate':>8} {'AnnRet':>8} {'Sharpe':>8} {'WF Sharpe':>14} {'Bull':>7} {'Caut':>7} {'Bear':>7}")
print("-" * 100)
for s in summary:
    print(f"{s['strategy']:>15} {s['n_rebal']:>6} {s['avg_ret']:>8.4f} {s['win_rate']:>8.1%} {s['ann_ret']:>8.1%} {s['sharpe']:>8.2f} {s['wf_sharpe_mean']:>6.2f}±{s['wf_sharpe_std']:.2f} {s['regime']['bull']:>7.3f} {s['regime']['cautious']:>7.3f} {s['regime']['bear']:>7.3f}")

output = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'experiment': 'xgb_rule_ensemble_v4',
    'config': {'top_n': TOP_N, 'hold_days': HOLD_DAYS, 'stop_loss': SL, 'cost': COST, 'n_folds': len(folds)},
    'summary': summary,
    'best': summary[0]['strategy'] if summary else None
}
with open('research/ceo_ensemble_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n✅ Best: {summary[0]['strategy']} (WF Sharpe: {summary[0]['wf_sharpe_mean']:.2f})")
print(f"   Time: {time.time()-t0:.0f}s")
