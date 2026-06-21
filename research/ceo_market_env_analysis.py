#!/usr/bin/env python3
"""
CEO Experiment Part 3: Market Environment Analysis
===================================================
Purpose: Understand how XGBoost performs in different market conditions
"""

import pandas as pd
import numpy as np
import json
import time
import os
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("=" * 70)
print("CEO Experiment Part 3: Market Environment Analysis")
print("=" * 70)
t0 = time.time()

# Load the comparison results
with open('research/ceo_ml_vs_rule_comparison.json') as f:
    results = json.load(f)

# Find XGBoost 10d full universe results
xgb_10d = [r for r in results if r['strategy'] == 'xgboost' and r['hold_days'] == 10 and 'universe' not in r][0]
rule_10d = [r for r in results if r['strategy'] == 'rule_based' and r['hold_days'] == 10 and 'universe' not in r][0]

print("\n[1/3] Analyzing fold performance by market environment...")

# Load market data to determine market state for each fold
df = pd.read_parquet('data/a_hist_10y.parquet')
df = df.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df['date'] = df['date'].astype(int)

# Calculate market returns
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
market_ret = df.groupby('date')['ret5'].mean()

# Define market states
def get_market_state(date, market_ret, lookback=60):
    """Determine market state based on recent returns"""
    recent_dates = sorted(market_ret.index)
    idx = recent_dates.index(date) if date in recent_dates else -1
    if idx < lookback:
        return 'unknown'
    
    recent_rets = market_ret.iloc[idx-lookback:idx]
    avg_ret = recent_rets.mean()
    volatility = recent_rets.std()
    
    if avg_ret > 0.01:  # Strong uptrend
        return 'bull'
    elif avg_ret < -0.01:  # Strong downtrend
        return 'bear'
    else:
        return 'neutral'

# Analyze each fold
print("\n[2/3] Classifying folds by market environment...")

fold_analysis = []
for fold in xgb_10d['fold_details']:
    test_start = int(fold['test_period'].split('-')[0])
    test_end = int(fold['test_period'].split('-')[1])
    
    # Get market returns during test period
    test_dates = [d for d in market_ret.index if test_start <= d <= test_end]
    if not test_dates:
        continue
    
    test_market_ret = market_ret.loc[test_dates]
    avg_market_ret = test_market_ret.mean()
    market_vol = test_market_ret.std()
    
    # Classify market environment
    if avg_market_ret > 0.005:
        env = 'bull'
    elif avg_market_ret < -0.005:
        env = 'bear'
    else:
        env = 'neutral'
    
    # Get corresponding rule-based fold
    rule_fold = [f for f in rule_10d['fold_details'] if f['fold'] == fold['fold']][0]
    
    fold_analysis.append({
        'fold': fold['fold'],
        'test_period': fold['test_period'],
        'market_env': env,
        'avg_market_ret': avg_market_ret,
        'market_vol': market_vol,
        'xgb_sharpe': fold['sharpe'],
        'xgb_avg_ret': fold['avg_return'],
        'xgb_win_rate': fold['win_rate'],
        'rule_sharpe': rule_fold['sharpe'],
        'rule_avg_ret': rule_fold['avg_return'],
        'rule_win_rate': rule_fold['win_rate'],
        'xgb_advantage': fold['sharpe'] - rule_fold['sharpe']
    })

# Convert to DataFrame
fold_df = pd.DataFrame(fold_analysis)

print("\n[3/3] Analyzing performance by market environment...")

# Group by market environment
env_analysis = fold_df.groupby('market_env').agg({
    'xgb_sharpe': ['mean', 'std', 'count'],
    'rule_sharpe': ['mean', 'std'],
    'xgb_avg_ret': 'mean',
    'rule_avg_ret': 'mean',
    'xgb_win_rate': 'mean',
    'rule_win_rate': 'mean',
    'xgb_advantage': 'mean',
    'avg_market_ret': 'mean'
}).round(3)

print("\n" + "=" * 70)
print("PERFORMANCE BY MARKET ENVIRONMENT")
print("=" * 70)

for env in ['bull', 'neutral', 'bear']:
    env_data = fold_df[fold_df['market_env'] == env]
    if len(env_data) == 0:
        continue
    
    print(f"\n{'='*70}")
    print(f"Market Environment: {env.upper()} ({len(env_data)} folds)")
    print(f"{'='*70}")
    
    print(f"\n  Average Market Return: {env_data['avg_market_ret'].mean():.4f}")
    print(f"  Market Volatility: {env_data['market_vol'].mean():.4f}")
    
    print(f"\n  {'Metric':<25} {'XGBoost':>10} {'Rule-based':>10} {'XGB Advantage':>15}")
    print(f"  {'-'*60}")
    
    print(f"  {'Sharpe':<25} {env_data['xgb_sharpe'].mean():>10.3f} {env_data['rule_sharpe'].mean():>10.3f} {env_data['xgb_advantage'].mean():>15.3f}")
    print(f"  {'Avg Return per Trade':<25} {env_data['xgb_avg_ret'].mean():>10.4f} {env_data['rule_avg_ret'].mean():>10.4f} {(env_data['xgb_avg_ret'].mean() - env_data['rule_avg_ret'].mean()):>15.4f}")
    print(f"  {'Win Rate':<25} {env_data['xgb_win_rate'].mean():>10.1%} {env_data['rule_win_rate'].mean():>10.1%} {(env_data['xgb_win_rate'].mean() - env_data['rule_win_rate'].mean()):>15.1%}")
    
    # List individual folds
    print(f"\n  Individual Folds:")
    for _, row in env_data.iterrows():
        print(f"    {row['test_period']}: XGB={row['xgb_sharpe']:.3f}, Rule={row['rule_sharpe']:.3f}, Diff={row['xgb_advantage']:.3f}")

# Overall analysis
print("\n" + "=" * 70)
print("OVERALL ANALYSIS")
print("=" * 70)

print(f"\nTotal Folds: {len(fold_df)}")
print(f"  Bull: {len(fold_df[fold_df['market_env'] == 'bull'])}")
print(f"  Neutral: {len(fold_df[fold_df['market_env'] == 'neutral'])}")
print(f"  Bear: {len(fold_df[fold_df['market_env'] == 'bear'])}")

print(f"\nOverall XGBoost Advantage: {fold_df['xgb_advantage'].mean():.3f}")

# Check if XGBoost advantage is consistent
xgb_adv_positive = (fold_df['xgb_advantage'] > 0).sum()
print(f"Folds where XGBoost outperforms: {xgb_adv_positive}/{len(fold_df)} ({xgb_adv_positive/len(fold_df):.1%})")

# Correlation between market return and XGBoost advantage
corr = fold_df['avg_market_ret'].corr(fold_df['xgb_advantage'])
print(f"\nCorrelation between market return and XGBoost advantage: {corr:.3f}")

if corr > 0.3:
    print("→ XGBoost advantage increases in bull markets")
elif corr < -0.3:
    print("→ XGBoost advantage increases in bear markets")
else:
    print("→ XGBoost advantage is relatively consistent across market conditions")

# Key insight
print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)

# Check if XGBoost's advantage comes from avoiding bear markets
bear_folds = fold_df[fold_df['market_env'] == 'bear']
bull_folds = fold_df[fold_df['market_env'] == 'bull']

if len(bear_folds) > 0 and len(bull_folds) > 0:
    bear_advantage = bear_folds['xgb_advantage'].mean()
    bull_advantage = bull_folds['xgb_advantage'].mean()
    
    print(f"\nXGBoost advantage in bear markets: {bear_advantage:.3f}")
    print(f"XGBoost advantage in bull markets: {bull_advantage:.3f}")
    
    if bear_advantage > bull_advantage:
        print("\n→ XGBoost's main advantage is in bear markets (avoiding losses)")
    else:
        print("\n→ XGBoost's main advantage is in bull markets (capturing gains)")

# Save analysis
analysis_output = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M'),
    'total_folds': len(fold_df),
    'market_env_distribution': {
        'bull': len(fold_df[fold_df['market_env'] == 'bull']),
        'neutral': len(fold_df[fold_df['market_env'] == 'neutral']),
        'bear': len(fold_df[fold_df['market_env'] == 'bear'])
    },
    'overall_xgb_advantage': fold_df['xgb_advantage'].mean(),
    'xgb_wins_pct': xgb_adv_positive / len(fold_df),
    'fold_details': fold_df.to_dict('records')
}

with open('research/ceo_market_env_analysis.json', 'w') as f:
    json.dump(analysis_output, f, indent=2, default=str)

print(f"\n✅ Analysis saved to research/ceo_market_env_analysis.json")
print(f"   Total time: {time.time()-t0:.0f}s")
