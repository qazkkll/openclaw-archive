#!/usr/bin/env python3
"""
CEO Experiment: XGBoost + Rule-based Ensemble (v2 - fixed dates)
================================================================
Walk-Forward with proper datetime handling.
Compute portfolio-level Sharpe (per rebalance, not per trade).
"""

import pandas as pd
import numpy as np
import json
import time
import os
import sys
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print(f"[CEO] Ensemble v2 {time.strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)
t0 = time.time()

# ============================================================
# 1. Load Data
# ============================================================
print("  Loading data...")
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

print(f"  Data: {len(df):,} rows, {df['sym'].nunique()} stocks ({time.time()-t0:.0f}s)")

# ============================================================
# 2. Compute Features
# ============================================================
print("  Computing features...")

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

df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(abs(df['high'] - df.groupby('sym')['close'].shift(1)),
               abs(df['low'] - df.groupby('sym')['close'].shift(1)))
)
df['atr14'] = df.groupby('sym')['tr'].transform(lambda x: x.rolling(14, min_periods=1).mean())
df['atr_pct'] = df['atr14'] / df['close']

df['vol_ratio'] = df.groupby('sym')['volume'].transform(lambda x: x.rolling(5).mean()) / \
                  df.groupby('sym')['volume'].transform(lambda x: x.rolling(20).mean())

for col in ['total_net', 'lg_net', 'md_net', 'elg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())
    df[f'{col}_20d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(20, min_periods=1).sum())
    df[f'{col}_5d_rk'] = df.groupby('date')[f'{col}_5d'].rank(pct=True)

df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')

# Forward return for training
df['fwd_10d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-10) / x - 1)

print(f"  Features done ({time.time()-t0:.0f}s)")

# ============================================================
# 3. Rule-based Scoring
# ============================================================
def score_rule(day):
    s = day.copy()
    s['rule_score'] = 0.0
    s['rule_score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 3
    s['rule_score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 2
    s['rule_score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 2
    s['rule_score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1.5
    s['rule_score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 1
    s['rule_score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1
    return s

# ============================================================
# 4. Walk-Forward with proper dates
# ============================================================
print("\n  Running Walk-Forward...")

import xgboost as xgb

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

all_dates = sorted(df['date'].unique())
date_to_idx = {d: i for i, d in enumerate(all_dates)}

# Convert YYYYMMDD to datetime for proper arithmetic
def int_to_dt(d):
    return datetime(int(str(d)[:4]), int(str(d)[4:6]), int(str(d)[6:8]))

def dt_to_int(d):
    return int(d.strftime('%Y%m%d'))

# Walk-Forward: 2yr train + 6mo test, slide 6mo
folds = []
train_start_dt = datetime(2016, 1, 1)

while True:
    train_end_dt = train_start_dt + timedelta(days=365*2)  # 2 years
    test_start_dt = train_end_dt
    test_end_dt = test_start_dt + timedelta(days=182)  # ~6 months
    
    if test_end_dt > int_to_dt(all_dates[-1]):
        break
    
    train_start_int = dt_to_int(train_start_dt)
    train_end_int = dt_to_int(train_end_dt)
    test_start_int = dt_to_int(test_start_dt)
    test_end_int = dt_to_int(test_end_dt)
    
    # Find actual dates in data
    train_dates = [d for d in all_dates if train_start_int <= d <= train_end_int]
    test_dates = [d for d in all_dates if test_start_int <= d <= test_end_int]
    
    if len(train_dates) < 200 or len(test_dates) < 20:
        train_start_dt += timedelta(days=182)
        continue
    
    folds.append({
        'train_dates': train_dates,
        'test_dates': test_dates,
        'train_start': train_start_int,
        'train_end': train_end_int,
        'test_start': test_start_int,
        'test_end': test_end_int
    })
    
    train_start_dt += timedelta(days=182)  # slide 6 months

print(f"  {len(folds)} folds")

strategies = {
    'xgb_only': {'rule_w': 0.0, 'xgb_w': 1.0},
    'rule_only': {'rule_w': 1.0, 'xgb_w': 0.0},
    'ens_50_50': {'rule_w': 0.5, 'xgb_w': 0.5},
    'ens_30_70': {'rule_w': 0.3, 'xgb_w': 0.7},
    'ens_70_30': {'rule_w': 0.7, 'xgb_w': 0.3},
    'ens_20_80': {'rule_w': 0.2, 'xgb_w': 0.8},
}

def dynamic_weight(regime):
    if regime == 'bear':
        return 0.2, 0.8
    elif regime == 'cautious':
        return 0.3, 0.7
    else:
        return 0.6, 0.4

strategies['ens_dynamic'] = {'rule_w': 'dynamic', 'xgb_w': 'dynamic'}

# Results: portfolio-level returns (one per rebalance period per fold)
results = {name: {'port_returns': [], 'fold_sharpes': [], 'regime_returns': {'bull': [], 'cautious': [], 'bear': []}} for name in strategies}

for fold_idx, fold in enumerate(folds):
    print(f"  Fold {fold_idx+1}/{len(folds)}: test {fold['test_start']}-{fold['test_end']}", end="")
    
    # Train XGBoost
    train_data = df[df['date'].isin(fold['train_dates'])].dropna(subset=XGB_FEATURES + ['fwd_10d'])
    
    if len(train_data) < 1000:
        print(" SKIP")
        continue
    
    X_train = train_data[XGB_FEATURES].fillna(0)
    y_train = train_data['fwd_10d']
    
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=4, verbosity=0
    )
    model.fit(X_train, y_train)
    
    # Test period: rebalance every HOLD_DAYS
    test_dates = fold['test_dates']
    rebal_dates = test_dates[::HOLD_DAYS]
    
    for strat_name, weights in strategies.items():
        strat_port_returns = []
        
        for rebal_date in rebal_dates:
            day_data = df[df['date'] == rebal_date].copy()
            if len(day_data) < 50:
                continue
            
            # Compute scores
            X_day = day_data[XGB_FEATURES].fillna(0)
            day_data['xgb_score'] = model.predict(X_day)
            day_data['xgb_rank'] = day_data['xgb_score'].rank(pct=True)
            
            day_data = score_rule(day_data)
            day_data['rule_rank'] = day_data['rule_score'].rank(pct=True)
            
            # Market regime
            market_breadth = day_data['breadth'].mean()
            market_ret20 = day_data['mkt_ret20'].mean()
            
            if market_breadth > 0.5 and market_ret20 > 0:
                regime = 'bull'
            elif market_breadth < 0.3 or market_ret20 < -0.05:
                regime = 'bear'
            else:
                regime = 'cautious'
            
            # Ensemble weights
            if weights['rule_w'] == 'dynamic':
                rw, xw = dynamic_weight(regime)
            else:
                rw, xw = weights['rule_w'], weights['xgb_w']
            
            day_data['ensemble_score'] = day_data['rule_rank'] * rw + day_data['xgb_rank'] * xw
            
            # Filter
            day_data = day_data[
                (day_data['close'] >= 3) &
                (day_data['close'] <= 200) &
                (~day_data['sym'].str.contains('ST|退市', na=False)) &
                (day_data['volume'] > 0)
            ]
            
            # Top N
            top = day_data.nlargest(TOP_N, 'ensemble_score')
            
            # Compute portfolio return over holding period
            # Find the date HOLD_DAYS later
            rebal_idx = test_dates.index(rebal_date) if rebal_date in test_dates else -1
            if rebal_idx < 0:
                continue
            
            exit_idx = min(rebal_idx + HOLD_DAYS, len(test_dates) - 1)
            exit_date = test_dates[exit_idx]
            
            if exit_date == rebal_date:
                continue
            
            # Compute equal-weighted portfolio return
            stock_returns = []
            for _, row in top.iterrows():
                sym = row['sym']
                entry_price = row['close']
                
                # Get exit price
                exit_rows = df[(df['sym'] == sym) & (df['date'] == exit_date)]
                if len(exit_rows) == 0:
                    continue
                
                exit_price = exit_rows.iloc[0]['close']
                ret = exit_price / entry_price - 1
                
                # Check stop loss
                hold_data = df[
                    (df['sym'] == sym) &
                    (df['date'] > rebal_date) &
                    (df['date'] <= exit_date)
                ]
                if len(hold_data) > 0:
                    min_price = hold_data['close'].min()
                    min_ret = min_price / entry_price - 1
                    if min_ret <= SL:
                        ret = SL
                
                ret -= COST
                stock_returns.append(ret)
            
            if stock_returns:
                port_ret = np.mean(stock_returns)
                strat_port_returns.append(port_ret)
                results[strat_name]['regime_returns'][regime].append(port_ret)
        
        # Compute fold Sharpe (portfolio-level)
        if len(strat_port_returns) > 2:
            avg_ret = np.mean(strat_port_returns)
            std_ret = np.std(strat_port_returns)
            # Annualize: ~24 rebalances per year
            ann_ret = avg_ret * 24
            ann_std = std_ret * np.sqrt(24)
            fold_sharpe = ann_ret / ann_std if ann_std > 0 else 0
            results[strat_name]['fold_sharpes'].append(fold_sharpe)
            results[strat_name]['port_returns'].extend(strat_port_returns)
    
    print(f" done ({len(rebal_dates)} rebal)")

# ============================================================
# 5. Results
# ============================================================
print("\n" + "=" * 90)
print("📊 ENSEMBLE RESULTS (Portfolio-level Sharpe)")
print("=" * 90)

summary = []
for strat_name, data in results.items():
    port_rets = data['port_returns']
    fold_sharpes = data['fold_sharpes']
    
    if not port_rets or not fold_sharpes:
        continue
    
    avg_ret_per_rebal = np.mean(port_rets)
    std_ret_per_rebal = np.std(port_rets)
    win_rate = np.mean([r > 0 for r in port_rets])
    
    # Annualize
    ann_ret = avg_ret_per_rebal * 24
    ann_std = std_ret_per_rebal * np.sqrt(24)
    overall_sharpe = ann_ret / ann_std if ann_std > 0 else 0
    
    wf_sharpe_mean = np.mean(fold_sharpes)
    wf_sharpe_std = np.std(fold_sharpes)
    
    # Regime analysis
    regime_avg = {}
    for regime in ['bull', 'cautious', 'bear']:
        rr = data['regime_returns'][regime]
        if rr:
            regime_avg[regime] = np.mean(rr)
        else:
            regime_avg[regime] = 0
    
    summary.append({
        'strategy': strat_name,
        'n_rebal': len(port_rets),
        'avg_ret_per_rebal': avg_ret_per_rebal,
        'win_rate': win_rate,
        'overall_sharpe': overall_sharpe,
        'wf_sharpe_mean': wf_sharpe_mean,
        'wf_sharpe_std': wf_sharpe_std,
        'ann_ret': ann_ret,
        'regime': regime_avg
    })

summary.sort(key=lambda x: x['wf_sharpe_mean'], reverse=True)

print(f"\n{'Strategy':>15} {'Rebal':>6} {'AvgRet':>8} {'WinRate':>8} {'AnnRet':>8} {'Sharpe':>8} {'WF Sharpe':>14} {'Bull':>7} {'Caut':>7} {'Bear':>7}")
print("-" * 100)

for s in summary:
    print(f"{s['strategy']:>15} {s['n_rebal']:>6} {s['avg_ret_per_rebal']:>8.4f} {s['win_rate']:>8.1%} {s['ann_ret']:>8.1%} {s['overall_sharpe']:>8.2f} {s['wf_sharpe_mean']:>6.2f}±{s['wf_sharpe_std']:.2f} {s['regime']['bull']:>7.3f} {s['regime']['cautious']:>7.3f} {s['regime']['bear']:>7.3f}")

# ============================================================
# 6. Save
# ============================================================
output = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'experiment': 'xgb_rule_ensemble_v2',
    'config': {
        'top_n': TOP_N,
        'hold_days': HOLD_DAYS,
        'stop_loss': SL,
        'cost': COST,
        'n_folds': len(folds)
    },
    'summary': summary,
    'best_strategy': summary[0]['strategy'] if summary else None,
    'best_wf_sharpe': summary[0]['wf_sharpe_mean'] if summary else 0
}

with open('research/ceo_xgb_rule_ensemble_v2.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n✅ Best: {summary[0]['strategy']} (WF Sharpe: {summary[0]['wf_sharpe_mean']:.2f})")
print(f"   Time: {time.time()-t0:.0f}s")
