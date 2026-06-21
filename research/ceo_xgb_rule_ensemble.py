#!/usr/bin/env python3
"""
CEO Experiment: XGBoost + Rule-based Ensemble
==============================================
Key insight from last session:
- XGBoost alpha = market timing (breadth + mkt_ret20 = 31.7%)
- Rule-based alpha = stock selection (reversal)
- Correlation = -0.018 (nearly zero!)

Hypothesis: Ensemble of uncorrelated strategies should improve Sharpe.

Tests:
1. Pure XGBoost (baseline)
2. Pure Rule-based (baseline)
3. Ensemble: 50/50 weight
4. Ensemble: 30/70 (rule/xgb)
5. Ensemble: 70/30 (rule/xgb)
6. Ensemble with dynamic weight based on market regime

Walk-Forward: 2yr train + 6mo test, 17 folds
"""

import pandas as pd
import numpy as np
import json
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print(f"[CEO] XGB + Rule Ensemble Experiment {time.strftime('%Y-%m-%d %H:%M')}")
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

# Filter
df = df[~df['sym'].str.startswith('688')].copy()
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
df = df[df['volume'] > 0].copy()
df = df.sort_values(['sym', 'date']).reset_index(drop=True)

print(f"  Data: {len(df):,} rows, {df['sym'].nunique()} stocks ({time.time()-t0:.0f}s)")

# ============================================================
# 2. Compute Features
# ============================================================
print("  Computing features...")

# Returns
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ret10'] = df.groupby('sym')['close'].pct_change(10)
df['ret20'] = df.groupby('sym')['close'].pct_change(20)

# MA deviation
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma60'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(60, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['ma60_bias'] = (df['close'] - df['ma60']) / df['ma60']

# Volatility
df['vol5'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(5, min_periods=2).std())
df['vol20'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(20, min_periods=2).std())

# RSI
delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)

# MACD
ema12 = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=12, min_periods=1).mean())
ema26 = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=26, min_periods=1).mean())
df['macd'] = ema12 - ema26
df['macd_signal'] = df.groupby('sym')['macd'].transform(lambda x: x.ewm(span=9, min_periods=1).mean())
df['macd_hist'] = df['macd'] - df['macd_signal']

# ATR
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(abs(df['high'] - df.groupby('sym')['close'].shift(1)),
               abs(df['low'] - df.groupby('sym')['close'].shift(1)))
)
df['atr14'] = df.groupby('sym')['tr'].transform(lambda x: x.rolling(14, min_periods=1).mean())
df['atr_pct'] = df['atr14'] / df['close']

# Volume ratio
df['vol_ratio'] = df.groupby('sym')['volume'].transform(lambda x: x.rolling(5).mean()) / \
                  df.groupby('sym')['volume'].transform(lambda x: x.rolling(20).mean())

# Money flow features
for col in ['total_net', 'lg_net', 'md_net', 'elg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())
    df[f'{col}_20d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(20, min_periods=1).sum())
    df[f'{col}_5d_rk'] = df.groupby('date')[f'{col}_5d'].rank(pct=True)

# Market features
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')

# Forward return
df['fwd_10d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-10) / x - 1)

print(f"  Features done ({time.time()-t0:.0f}s)")

# ============================================================
# 3. Rule-based Scoring Function
# ============================================================
def score_rule(day):
    """rule-alpha-v1.0 scoring function"""
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
# 4. Walk-Forward Backtest
# ============================================================
print("\n  Running Walk-Forward backtest...")

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
COST = 0.0015  # 0.15% round-trip

all_dates = sorted(df['date'].unique())
min_date = all_dates[0]
max_date = all_dates[-1]

# Walk-Forward: 2yr train + 6mo test
train_start = 20160101
folds = []

while True:
    train_end = train_start + 10000  # ~1 year
    test_start = train_end
    test_end = test_start + 5000  # ~6 months
    
    if test_end > max_date:
        break
    
    # Need 2 years of training data
    actual_train_start = train_start - 10000  # go back 1 more year
    
    folds.append({
        'train_start': actual_train_start,
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end
    })
    
    train_start += 5000  # slide 6 months

print(f"  {len(folds)} folds created")

# Strategies to test
strategies = {
    'xgb_only': {'rule_w': 0.0, 'xgb_w': 1.0},
    'rule_only': {'rule_w': 1.0, 'xgb_w': 0.0},
    'ens_50_50': {'rule_w': 0.5, 'xgb_w': 0.5},
    'ens_30_70': {'rule_w': 0.3, 'xgb_w': 0.7},
    'ens_70_30': {'rule_w': 0.7, 'xgb_w': 0.3},
    'ens_20_80': {'rule_w': 0.2, 'xgb_w': 0.8},
    'ens_80_20': {'rule_w': 0.8, 'xgb_w': 0.2},
}

# Dynamic weight: more XGB in bear (market timing), more rule in bull (stock selection)
def dynamic_weight(regime):
    if regime == 'bear':
        return 0.2, 0.8  # rule_w, xgb_w (XGB handles bear better)
    elif regime == 'cautious':
        return 0.3, 0.7
    else:  # bull
        return 0.6, 0.4  # rule_w, xgb_w (rule handles stock selection)

strategies['ens_dynamic'] = {'rule_w': 'dynamic', 'xgb_w': 'dynamic'}

results = {name: {'folds': [], 'all_trades': []} for name in strategies}

for fold_idx, fold in enumerate(folds):
    print(f"\n  Fold {fold_idx+1}/{len(folds)}: test {fold['test_start']}-{fold['test_end']}")
    
    # Train XGBoost
    train_mask = (df['date'] >= fold['train_start']) & (df['date'] <= fold['train_end'])
    train_data = df[train_mask].dropna(subset=XGB_FEATURES + ['fwd_10d'])
    
    if len(train_data) < 1000:
        print(f"    Skipping: insufficient training data ({len(train_data)})")
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
    
    # Test period
    test_mask = (df['date'] >= fold['test_start']) & (df['date'] <= fold['test_end'])
    test_data = df[test_mask].copy()
    
    if len(test_data) == 0:
        continue
    
    # Compute scores for test period
    test_dates = sorted(test_data['date'].unique())
    
    # Rebalance every HOLD_DAYS
    rebal_dates = test_dates[::HOLD_DAYS]
    
    for strat_name, weights in strategies.items():
        fold_trades = []
        
        for rebal_date in rebal_dates:
            day_data = test_data[test_data['date'] == rebal_date].copy()
            if len(day_data) < 50:
                continue
            
            # XGBoost score
            X_day = day_data[XGB_FEATURES].fillna(0)
            day_data['xgb_score'] = model.predict(X_day)
            day_data['xgb_rank'] = day_data['xgb_score'].rank(pct=True)
            
            # Rule-based score
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
            
            # Ensemble score
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
            
            # Simulate holding period
            for _, row in top.iterrows():
                sym = row['sym']
                entry_date = rebal_date
                entry_price = row['close']
                
                # Find exit date (HOLD_DAYS later)
                exit_idx = test_dates.index(rebal_date) + HOLD_DAYS if rebal_date in test_dates else -1
                if exit_idx >= len(test_dates):
                    exit_date = test_dates[-1]
                else:
                    exit_date = test_dates[exit_idx]
                
                # Get exit price
                exit_data = test_data[(test_data['sym'] == sym) & (test_data['date'] == exit_date)]
                if len(exit_data) == 0:
                    # Try to find the closest available date
                    available_dates = test_data[test_data['sym'] == sym]['date'].unique()
                    valid_dates = [d for d in available_dates if d > entry_date]
                    if valid_dates:
                        exit_date = valid_dates[min(HOLD_DAYS-1, len(valid_dates)-1)]
                        exit_data = test_data[(test_data['sym'] == sym) & (test_data['date'] == exit_date)]
                
                if len(exit_data) > 0:
                    exit_price = exit_data.iloc[0]['close']
                    ret = (exit_price / entry_price - 1)
                    
                    # Check stop loss during hold period
                    hold_data = test_data[
                        (test_data['sym'] == sym) & 
                        (test_data['date'] > entry_date) & 
                        (test_data['date'] <= exit_date)
                    ]
                    
                    if len(hold_data) > 0:
                        min_ret = (hold_data['close'].min() / entry_price - 1)
                        if min_ret <= SL:
                            ret = SL  # Stop loss triggered
                    
                    # Apply cost
                    ret -= COST
                    
                    fold_trades.append({
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'sym': sym,
                        'ret': ret,
                        'regime': regime
                    })
        
        results[strat_name]['folds'].append({
            'fold': fold_idx,
            'test_period': f"{fold['test_start']}-{fold['test_end']}",
            'trades': fold_trades
        })
        results[strat_name]['all_trades'].extend(fold_trades)

# ============================================================
# 5. Compute Results
# ============================================================
print("\n" + "=" * 80)
print("📊 RESULTS")
print("=" * 80)

summary = []
for strat_name, data in results.items():
    trades = data['all_trades']
    if not trades:
        continue
    
    rets = [t['ret'] for t in trades]
    n_trades = len(rets)
    avg_ret = np.mean(rets)
    std_ret = np.std(rets)
    win_rate = np.mean([r > 0 for r in rets])
    
    # Annualize (assume ~24 rebalances per year, 15 trades each)
    trades_per_year = 24 * TOP_N
    ann_ret = avg_ret * trades_per_year
    ann_std = std_ret * np.sqrt(trades_per_year)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0
    
    # Fold-level Sharpe
    fold_sharpes = []
    for fold_data in data['folds']:
        fold_rets = [t['ret'] for t in fold_data['trades']]
        if len(fold_rets) > 5:
            fold_avg = np.mean(fold_rets)
            fold_std = np.std(fold_rets)
            fold_sharpe = (fold_avg * trades_per_year) / (fold_std * np.sqrt(trades_per_year)) if fold_std > 0 else 0
            fold_sharpes.append(fold_sharpe)
    
    avg_fold_sharpe = np.mean(fold_sharpes) if fold_sharpes else 0
    std_fold_sharpe = np.std(fold_sharpes) if fold_sharpes else 0
    
    # By regime
    regime_stats = {}
    for regime in ['bull', 'cautious', 'bear']:
        regime_trades = [t['ret'] for t in trades if t.get('regime') == regime]
        if regime_trades:
            regime_stats[regime] = {
                'count': len(regime_trades),
                'avg_ret': np.mean(regime_trades),
                'win_rate': np.mean([r > 0 for r in regime_trades])
            }
    
    summary.append({
        'strategy': strat_name,
        'n_trades': n_trades,
        'avg_ret': avg_ret,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'wf_sharpe_mean': avg_fold_sharpe,
        'wf_sharpe_std': std_fold_sharpe,
        'regime_stats': regime_stats
    })

# Sort by WF Sharpe
summary.sort(key=lambda x: x['wf_sharpe_mean'], reverse=True)

print(f"\n{'Strategy':>15} {'Trades':>7} {'AvgRet':>8} {'WinRate':>8} {'Sharpe':>8} {'WF Sharpe':>12} {'Regime (B/C/A)':>25}")
print("-" * 90)

for s in summary:
    regime_str = ""
    for regime in ['bull', 'cautious', 'bear']:
        if regime in s['regime_stats']:
            rs = s['regime_stats'][regime]
            regime_str += f"{rs['avg_ret']:.3f}/"
        else:
            regime_str += "N/A/"
    
    print(f"{s['strategy']:>15} {s['n_trades']:>7} {s['avg_ret']:>8.4f} {s['win_rate']:>8.1%} {s['sharpe']:>8.2f} {s['wf_sharpe_mean']:>6.2f}±{s['wf_sharpe_std']:.2f} {regime_str:>25}")

# ============================================================
# 6. Save Results
# ============================================================
output = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'experiment': 'xgb_rule_ensemble',
    'config': {
        'top_n': TOP_N,
        'hold_days': HOLD_DAYS,
        'stop_loss': SL,
        'cost': COST,
        'n_folds': len(folds),
        'xgb_features': len(XGB_FEATURES)
    },
    'summary': summary,
    'best_strategy': summary[0]['strategy'] if summary else None,
    'best_wf_sharpe': summary[0]['wf_sharpe_mean'] if summary else 0
}

os.makedirs('research', exist_ok=True)
with open('research/ceo_xgb_rule_ensemble.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n✅ Results saved to research/ceo_xgb_rule_ensemble.json")
print(f"   Best strategy: {summary[0]['strategy']} (WF Sharpe: {summary[0]['wf_sharpe_mean']:.2f})")
print(f"   Total time: {time.time()-t0:.0f}s")
