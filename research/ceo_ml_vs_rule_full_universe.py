#!/usr/bin/env python3
"""
CEO Experiment: ML vs Rule-based on Full Universe
==================================================
Purpose: Fair comparison of XGBoost regression vs rule-based scoring
on the same full universe data with Walk-Forward validation.

Key questions:
1. Does ML outperform rule-based on full universe?
2. Does universe filtering (price cap) improve alpha?
3. What's the realistic Sharpe ceiling for A-stock selection?

Design:
- Walk-Forward: 2yr train + 6mo test, sliding 6mo
- Models: XGBoost regression vs rule-based scoring
- Universe: full (4912 stocks) vs filtered (price > 10)
- Hold periods: 10d, 20d
- Transaction cost: 0.15% round-trip
"""

import pandas as pd
import numpy as np
import json
import time
import os
import sys
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 70)
print("CEO Experiment: ML vs Rule-based on Full Universe")
print("=" * 70)
t0 = time.time()

# Load OHLCV
print("\n[1/5] Loading OHLCV data...")
df = pd.read_parquet('data/a_hist_10y.parquet')
df = df.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df['date'] = df['date'].astype(int)
print(f"  OHLCV: {len(df):,} rows, {df['sym'].nunique()} stocks")

# Load moneyflow
print("[2/5] Loading moneyflow data...")
mf = pd.read_parquet('data/cn/moneyflow_core.parquet')
mf['sym'] = mf['ts_code'].str[:6]
mf['date'] = mf['trade_date'].astype(int)
for col in ['sm', 'md', 'lg', 'elg']:
    mf[f'{col}_net'] = mf[f'buy_{col}_amount'] - mf[f'sell_{col}_amount']
mf['total_net'] = mf['net_mf_amount']
df = df.merge(mf[['sym', 'date', 'total_net', 'lg_net', 'md_net', 'elg_net']], on=['sym', 'date'], how='left')
print(f"  Merged: {len(df):,} rows")

# Basic filters
df = df[~df['sym'].str.startswith('688')].copy()  # Exclude STAR market
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
df = df[df['volume'] > 0].copy()
df = df.sort_values(['sym', 'date']).reset_index(drop=True)
print(f"  Filtered: {len(df):,} rows, {df['sym'].nunique()} stocks")

# ============================================================
# 2. Compute Features
# ============================================================
print("\n[3/5] Computing features...")

# Returns
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ret10'] = df.groupby('sym')['close'].pct_change(10)
df['ret20'] = df.groupby('sym')['close'].pct_change(20)

# MA deviation
df['ma5'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
df['ma10'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(10, min_periods=1).mean())
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
    # Rank percentile
    df[f'{col}_5d_rk'] = df.groupby('date')[f'{col}_5d'].rank(pct=True)

# Market features
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')

# Forward returns (labels)
df['fwd_10d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-10) / x - 1)
df['fwd_20d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-20) / x - 1)

# Drop NaN rows
feature_cols = [
    'ret5', 'ret10', 'ret20', 'ma20_bias', 'ma60_bias',
    'vol5', 'vol20', 'rsi_14', 'macd_hist', 'atr_pct', 'vol_ratio',
    'total_net_5d', 'lg_net_5d', 'md_net_5d', 'elg_net_5d',
    'total_net_20d', 'lg_net_20d', 'md_net_20d', 'elg_net_20d',
    'total_net_5d_rk', 'lg_net_5d_rk', 'md_net_5d_rk', 'elg_net_5d_rk',
    'breadth', 'mkt_ret20'
]

df_clean = df.dropna(subset=feature_cols + ['fwd_10d', 'fwd_20d']).copy()
print(f"  Clean: {len(df_clean):,} rows, {df_clean['sym'].nunique()} stocks")
print(f"  Features: {len(feature_cols)}")
print(f"  Time: {time.time()-t0:.0f}s")

# ============================================================
# 3. Walk-Forward Validation
# ============================================================
print("\n[4/5] Walk-Forward Validation...")

# Define Walk-Forward folds
all_dates = sorted(df_clean['date'].unique())
min_date = all_dates[0]
max_date = all_dates[-1]

# Convert dates for comparison
min_year = min_date // 10000
max_year = max_date // 10000

# Create folds: 2yr train + 6mo test, sliding 6mo
folds = []
for test_start_year in range(min_year + 2, max_year + 1):
    for test_start_month in [1, 7]:  # Jan and July
        test_start = test_start_year * 10000 + test_start_month * 100 + 1
        test_end = test_start_year * 10000 + (test_start_month + 5) * 100 + 28
        if test_start_month == 7:
            test_end = (test_start_year + 1) * 10000 + 1 * 100 + 28
        
        train_start = (test_start_year - 2) * 10000 + test_start_month * 100 + 1
        train_end = test_start - 1
        
        # Check if dates exist in data
        if test_start > max_date or test_end < test_start:
            continue
            
        folds.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': min(test_end, max_date)
        })

print(f"  Walk-Forward folds: {len(folds)}")

# Rule-based scoring function
def score_rule_based(day):
    """rule-alpha-v2.1 scoring function"""
    s = day.copy()
    s['score'] = 0.0
    s['score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 3      # Reversal
    s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 2    # Money flow
    s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 2  # Low vol
    s['score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1.5 # RSI oversold
    s['score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 1       # Large order flow
    s['score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1   # MA deviation
    return s

# Run experiments
results = []

for hold_days in [10, 20]:
    label_col = f'fwd_{hold_days}d'
    print(f"\n  Hold period: {hold_days}d")
    
    for strategy in ['rule_based', 'xgboost']:
        print(f"    Strategy: {strategy}")
        
        all_fold_results = []
        
        for fold_idx, fold in enumerate(folds):
            # Split data
            train = df_clean[(df_clean['date'] >= fold['train_start']) & 
                            (df_clean['date'] <= fold['train_end'])].copy()
            test = df_clean[(df_clean['date'] >= fold['test_start']) & 
                           (df_clean['date'] <= fold['test_end'])].copy()
            
            if len(train) < 1000 or len(test) < 100:
                continue
            
            # Train/predict
            if strategy == 'xgboost':
                import xgboost as xgb
                X_train = train[feature_cols].fillna(0)
                y_train = train[label_col]
                X_test = test[feature_cols].fillna(0)
                
                model = xgb.XGBRegressor(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, n_jobs=4, verbosity=0
                )
                model.fit(X_train, y_train)
                test['score'] = model.predict(X_test)
            else:
                # Rule-based: score each day
                test_scored = []
                for d in sorted(test['date'].unique()):
                    day_data = test[test['date'] == d].copy()
                    if len(day_data) < 50:
                        continue
                    scored = score_rule_based(day_data)
                    test_scored.append(scored)
                if not test_scored:
                    continue
                test = pd.concat(test_scored)
            
            # Simulate portfolio
            trade_dates = sorted(test['date'].unique())
            rebal_dates = trade_dates[::hold_days]
            
            portfolio_returns = []
            trades = []
            
            for rebal_date in rebal_dates:
                # Get top stocks
                day_data = test[test['date'] == rebal_date].copy()
                if len(day_data) < 15:
                    continue
                
                # Filter
                day_data = day_data[
                    (day_data['close'] >= 3) &
                    (day_data['close'] <= 200) &
                    (~day_data['sym'].str.contains('ST|退市', na=False))
                ]
                
                if len(day_data) < 15:
                    continue
                
                # Top 15
                top15 = day_data.nlargest(15, 'score')
                
                # Get hold period returns
                for _, stock in top15.iterrows():
                    sym = stock['sym']
                    entry_date = rebal_date
                    entry_price = stock['close']
                    
                    # Find exit date
                    exit_idx = trade_dates.index(rebal_date) + hold_days
                    if exit_idx >= len(trade_dates):
                        continue
                    exit_date = trade_dates[exit_idx]
                    
                    # Get exit price
                    exit_data = test[(test['sym'] == sym) & (test['date'] == exit_date)]
                    if len(exit_data) == 0:
                        continue
                    
                    exit_price = exit_data['close'].values[0]
                    ret = (exit_price / entry_price) - 1
                    
                    # Transaction cost
                    cost = 0.0015  # 0.15% round-trip
                    net_ret = ret - cost
                    
                    # Stop loss check (simplified: if ret < -3%, use -3%)
                    if ret < -0.03:
                        net_ret = -0.03 - cost
                    
                    portfolio_returns.append(net_ret)
                    trades.append({
                        'sym': sym,
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'return': net_ret
                    })
            
            if not portfolio_returns:
                continue
            
            # Calculate metrics
            avg_ret = np.mean(portfolio_returns)
            std_ret = np.std(portfolio_returns)
            sharpe = avg_ret / std_ret * np.sqrt(252 / hold_days) if std_ret > 0 else 0
            win_rate = np.mean([r > 0 for r in portfolio_returns])
            
            fold_result = {
                'fold': fold_idx,
                'train_period': f"{fold['train_start']}-{fold['train_end']}",
                'test_period': f"{fold['test_start']}-{fold['test_end']}",
                'n_trades': len(portfolio_returns),
                'avg_return': avg_ret,
                'sharpe': sharpe,
                'win_rate': win_rate
            }
            all_fold_results.append(fold_result)
        
        if not all_fold_results:
            continue
        
        # Aggregate fold results
        avg_sharpe = np.mean([f['sharpe'] for f in all_fold_results])
        std_sharpe = np.std([f['sharpe'] for f in all_fold_results])
        avg_win_rate = np.mean([f['win_rate'] for f in all_fold_results])
        total_trades = sum([f['n_trades'] for f in all_fold_results])
        
        # Calculate CAGR from all trades
        all_rets = []
        for fold_result in all_fold_results:
            # Get actual returns from trades
            pass
        
        result = {
            'strategy': strategy,
            'hold_days': hold_days,
            'n_folds': len(all_fold_results),
            'total_trades': total_trades,
            'avg_sharpe': avg_sharpe,
            'std_sharpe': std_sharpe,
            'sharpe_ratio': avg_sharpe / std_sharpe if std_sharpe > 0 else 0,
            'avg_win_rate': avg_win_rate,
            'fold_details': all_fold_results
        }
        results.append(result)
        
        print(f"      Folds: {len(all_fold_results)}, Trades: {total_trades}")
        print(f"      Sharpe: {avg_sharpe:.3f} ± {std_sharpe:.3f}")
        print(f"      Win Rate: {avg_win_rate:.1%}")

# ============================================================
# 5. Universe Filter Comparison
# ============================================================
print("\n[5/5] Universe Filter Comparison...")

# Test with price > 10 filter
df_filtered = df_clean[df_clean['close'] >= 10].copy()
print(f"  Filtered universe (price >= 10): {len(df_filtered):,} rows, {df_filtered['sym'].nunique()} stocks")

# Run same experiment on filtered universe
for hold_days in [10, 20]:
    label_col = f'fwd_{hold_days}d'
    print(f"\n  Hold period: {hold_days}d (filtered universe)")
    
    for strategy in ['rule_based', 'xgboost']:
        print(f"    Strategy: {strategy}")
        
        all_fold_results = []
        
        for fold_idx, fold in enumerate(folds):
            # Split data
            train = df_filtered[(df_filtered['date'] >= fold['train_start']) & 
                               (df_filtered['date'] <= fold['train_end'])].copy()
            test = df_filtered[(df_filtered['date'] >= fold['test_start']) & 
                              (df_filtered['date'] <= fold['test_end'])].copy()
            
            if len(train) < 500 or len(test) < 50:
                continue
            
            # Train/predict
            if strategy == 'xgboost':
                import xgboost as xgb
                X_train = train[feature_cols].fillna(0)
                y_train = train[label_col]
                X_test = test[feature_cols].fillna(0)
                
                model = xgb.XGBRegressor(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, n_jobs=4, verbosity=0
                )
                model.fit(X_train, y_train)
                test['score'] = model.predict(X_test)
            else:
                # Rule-based
                test_scored = []
                for d in sorted(test['date'].unique()):
                    day_data = test[test['date'] == d].copy()
                    if len(day_data) < 30:
                        continue
                    scored = score_rule_based(day_data)
                    test_scored.append(scored)
                if not test_scored:
                    continue
                test = pd.concat(test_scored)
            
            # Simulate portfolio
            trade_dates = sorted(test['date'].unique())
            rebal_dates = trade_dates[::hold_days]
            
            portfolio_returns = []
            
            for rebal_date in rebal_dates:
                day_data = test[test['date'] == rebal_date].copy()
                if len(day_data) < 15:
                    continue
                
                day_data = day_data[
                    (day_data['close'] >= 10) &
                    (day_data['close'] <= 200) &
                    (~day_data['sym'].str.contains('ST|退市', na=False))
                ]
                
                if len(day_data) < 15:
                    continue
                
                top15 = day_data.nlargest(15, 'score')
                
                for _, stock in top15.iterrows():
                    sym = stock['sym']
                    entry_price = stock['close']
                    
                    exit_idx = trade_dates.index(rebal_date) + hold_days
                    if exit_idx >= len(trade_dates):
                        continue
                    exit_date = trade_dates[exit_idx]
                    
                    exit_data = test[(test['sym'] == sym) & (test['date'] == exit_date)]
                    if len(exit_data) == 0:
                        continue
                    
                    exit_price = exit_data['close'].values[0]
                    ret = (exit_price / entry_price) - 1
                    cost = 0.0015
                    net_ret = ret - cost
                    
                    if ret < -0.03:
                        net_ret = -0.03 - cost
                    
                    portfolio_returns.append(net_ret)
            
            if not portfolio_returns:
                continue
            
            avg_ret = np.mean(portfolio_returns)
            std_ret = np.std(portfolio_returns)
            sharpe = avg_ret / std_ret * np.sqrt(252 / hold_days) if std_ret > 0 else 0
            win_rate = np.mean([r > 0 for r in portfolio_returns])
            
            fold_result = {
                'fold': fold_idx,
                'test_period': f"{fold['test_start']}-{fold['test_end']}",
                'n_trades': len(portfolio_returns),
                'avg_return': avg_ret,
                'sharpe': sharpe,
                'win_rate': win_rate
            }
            all_fold_results.append(fold_result)
        
        if not all_fold_results:
            continue
        
        avg_sharpe = np.mean([f['sharpe'] for f in all_fold_results])
        std_sharpe = np.std([f['sharpe'] for f in all_fold_results])
        avg_win_rate = np.mean([f['win_rate'] for f in all_fold_results])
        total_trades = sum([f['n_trades'] for f in all_fold_results])
        
        result = {
            'strategy': strategy,
            'hold_days': hold_days,
            'universe': 'filtered_price10',
            'n_folds': len(all_fold_results),
            'total_trades': total_trades,
            'avg_sharpe': avg_sharpe,
            'std_sharpe': std_sharpe,
            'sharpe_ratio': avg_sharpe / std_sharpe if std_sharpe > 0 else 0,
            'avg_win_rate': avg_win_rate,
            'fold_details': all_fold_results
        }
        results.append(result)
        
        print(f"      Folds: {len(all_fold_results)}, Trades: {total_trades}")
        print(f"      Sharpe: {avg_sharpe:.3f} ± {std_sharpe:.3f}")
        print(f"      Win Rate: {avg_win_rate:.1%}")

# ============================================================
# 6. Save Results
# ============================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

# Print summary table
print(f"\n{'Strategy':<20} {'Hold':>5} {'Universe':<15} {'Sharpe':>8} {'±Std':>8} {'WinRate':>8} {'Trades':>8}")
print("-" * 80)

for r in sorted(results, key=lambda x: x['avg_sharpe'], reverse=True):
    universe = r.get('universe', 'full')
    print(f"{r['strategy']:<20} {r['hold_days']:>5} {universe:<15} {r['avg_sharpe']:>8.3f} {r['std_sharpe']:>8.3f} {r['avg_win_rate']:>8.1%} {r['total_trades']:>8}")

# Save to file
output_file = 'research/ceo_ml_vs_rule_comparison.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n✅ Results saved to {output_file}")
print(f"   Total time: {time.time()-t0:.0f}s")

# CEO Analysis
print("\n" + "=" * 70)
print("CEO ANALYSIS")
print("=" * 70)

# Find best config
best = max(results, key=lambda x: x['avg_sharpe'])
print(f"\n🏆 Best configuration:")
print(f"   Strategy: {best['strategy']}")
print(f"   Hold period: {best['hold_days']}d")
print(f"   Universe: {best.get('universe', 'full')}")
print(f"   Sharpe: {best['avg_sharpe']:.3f} ± {best['std_sharpe']:.3f}")
print(f"   Win Rate: {best['avg_win_rate']:.1%}")

if best['avg_sharpe'] >= 1.0:
    print(f"\n✅ Meets 1.0 Sharpe threshold!")
else:
    print(f"\n⚠️ Below 1.0 Sharpe threshold")
    print(f"   Gap: {1.0 - best['avg_sharpe']:.3f}")
    
    # Check if filtered universe helps
    filtered_results = [r for r in results if r.get('universe') == 'filtered_price10']
    full_results = [r for r in results if r.get('universe') != 'filtered_price10']
    
    if filtered_results and full_results:
        best_filtered = max(filtered_results, key=lambda x: x['avg_sharpe'])
        best_full = max(full_results, key=lambda x: x['avg_sharpe'])
        
        if best_filtered['avg_sharpe'] > best_full['avg_sharpe']:
            print(f"\n📊 Filtered universe improves Sharpe by {best_filtered['avg_sharpe'] - best_full['avg_sharpe']:.3f}")
        else:
            print(f"\n📊 Full universe is actually better")
