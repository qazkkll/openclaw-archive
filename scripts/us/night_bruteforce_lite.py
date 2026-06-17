#!/usr/bin/env python3
"""
轻量级Bruteforce - 在云端运行，使用3个月数据
作为Windows全量回测的备份
"""
import tushare as ts, pandas as pd, numpy as np, json, itertools, time
from datetime import datetime

TS_TOKEN = 'ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db'
pro = ts.pro_api(TS_TOKEN)

def log(m): print(f'[{datetime.now():%H:%M:%S}] {m}', flush=True)

log('=== 轻量级Bruteforce ===')

# Pull 6 months of data (Jan-Jun 2026)
cal = pro.trade_call(start_date='20260101', end_date='20260601')
days = sorted(cal[cal['is_open']==1]['cal_date'].tolist())
log(f'Trading days available: {len(days)}')

# Use last 120 days for testing, first 60 days for warmup
test_days = days[-120:]
warmup_days = days[:60]
log(f'Warmup: {len(warmup_days)} days, Test: {len(test_days)} days')

# Pull OHLCV for all days
log('Pulling OHLCV...')
all_d = []
for d in days:
    df = pro.daily(trade_date=d, fields='ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount')
    if df is not None and len(df) > 0:
        all_d.append(df)

df = pd.concat(all_d, ignore_index=True)
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.sort_values(['ts_code','trade_date'])
log(f'Total: {len(df)} rows, {df["ts_code"].nunique()} stocks')

# Compute factors
df['mom20'] = df.groupby('ts_code')['close'].transform(lambda x: x / x.shift(20) - 1)
df['mom10'] = df.groupby('ts_code')['close'].transform(lambda x: x / x.shift(10) - 1)
df['mom5'] = df.groupby('ts_code')['close'].transform(lambda x: x / x.shift(5) - 1)
df['fwd_5d'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(-5) / x - 1)
df['fwd_10d'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(-10) / x - 1)
df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5, min_periods=3).mean())
df['ma10'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(10, min_periods=5).mean())
df['vol_ma5'] = df.groupby('ts_code')['vol'].transform(lambda x: x.rolling(5, min_periods=3).mean())
df['ma_signal'] = ((df['ma5'] > df['ma10']) & (df['close'] > df['ma5'])).astype(float)
df['double_vol'] = (df['vol'] / (df['vol_ma5']+1) > 2.0).astype(float)

# Split into test set only (last 60 days)
test_start = pd.to_datetime(test_days[-60] if len(test_days) >= 60 else test_days[0])
test_set = df[df['trade_date'] >= test_start].copy()
test_set = test_set.dropna(subset=['fwd_5d','mom20','mom10','mom5']).copy()
log(f'Test set: {len(test_set)} rows')

# ===== STRATEGY A: Pure Momentum (vary weights) =====
log('\n=== Strategy A: Pure Momentum Parameter Search ===')
results_a = []

for w20 in [i/10 for i in range(3,9)]:  # 0.3-0.8
    for w10 in [i/10 for i in range(1,5)]:  # 0.1-0.4
        for w5 in [i/10 for i in range(0,4)]:  # 0.0-0.3
            total = round(w20+w10+w5, 2)
            if abs(total-1.0) > 0.01:
                continue
            
            # Score
            test_set['score'] = test_set['mom20']*w20 + test_set['mom10']*w10 + test_set['mom5']*w5
            test_set['rank'] = test_set.groupby('trade_date')['score'].rank(pct=True)
            
            # Test multiple percentiles
            for pct in [0.9, 0.8, 0.7]:
                top = test_set[test_set['rank'] > pct]
                if len(top) == 0: continue
                ret = top['fwd_5d'].mean()
                win = (top['fwd_5d'] > 0).mean()
                
                results_a.append({
                    'weights': f'{w20:.1f}/{w10:.1f}/{w5:.1f}',
                    'percentile': pct,
                    'ret_5d': round(float(ret), 4),
                    'win_rate': round(float(win), 3),
                    'trades': len(top)
                })

# Rank
results_a.sort(key=lambda x: x['ret_5d'], reverse=True)
log(f'Total A combinations tested: {len(results_a)}')
print('\nStrategy A Top 10:')
for r in results_a[:10]:
    print(f'  w={r[\"weights\"]} p{int(r[\"percentile\"]*100)}%: ret={r[\"ret_5d\"]:+.2%} win={r[\"win_rate\"]:.0%} n={r[\"trades\"]}')

# ===== STRATEGY B: Momentum + Gate =====
log('\n=== Strategy B: Momentum + Gate Search ===')
results_b = []

for w20 in [0.4, 0.5, 0.6]:
    for w_gate in [0.1, 0.2, 0.3, 0.4]:
        w_mom10 = round(1 - w20 - w_gate, 2)
        if w_mom10 < 0.1: continue
        
        for gate_feature in ['ma_signal', 'double_vol']:
            test_set['score_b'] = (test_set['mom20']*w20 + 
                                   test_set[gate_feature]*w_gate + 
                                   test_set['mom10'].fillna(0)*(1-w20-w_gate))
            test_set['rank_b'] = test_set.groupby('trade_date')['score_b'].rank(pct=True)
            
            for pct in [0.9, 0.8, 0.7]:
                top = test_set[test_set['rank_b'] > pct]
                if len(top) == 0: continue
                results_b.append({
                    'formula': f'mom20*{w20:.1f}+{gate_feature}*{w_gate:.1f}+mom10*{w_mom10:.1f}',
                    'percentile': pct,
                    'ret_5d': round(float(top['fwd_5d'].mean()), 4),
                    'win_rate': round(float((top['fwd_5d'] > 0).mean()), 3),
                    'trades': len(top)
                })

results_b.sort(key=lambda x: x['ret_5d'], reverse=True)
log(f'Total B combinations: {len(results_b)}')
print('\nStrategy B Top 10:')
for r in results_b[:10]:
    print(f'  {r[\"formula\"]} p{int(r[\"percentile\"]*100)}%: ret={r[\"ret_5d\"]:+.2%} win={r[\"win_rate\"]:.0%}')

# ===== FINAL COMPARISON =====
print(f'\n{"="*60}')
print('FINAL VERDICT: 6-month backtest')
print(f'{"="*60}')
best_a = results_a[0] if results_a else None
best_b = results_b[0] if results_b else None

if best_a and best_b:
    print(f'\nBest Strategy A (pure momentum):')
    print(f'  Weights: {best_a["weights"]}')
    print(f'  Top {int(best_a["percentile"]*100)}%: {best_a["ret_5d"]:+.2%} 5d return, {best_a["win_rate"]:.0%} win rate')
    
    print(f'\nBest Strategy B (momentum + gate):')
    print(f'  Formula: {best_b["formula"]}')
    print(f'  Top {int(best_b["percentile"]*100)}%: {best_b["ret_5d"]:+.2%} 5d return, {best_b["win_rate"]:.0%} win rate')

# Save
output = {
    'timestamp': datetime.now().isoformat(), 
    'strategy_a_top20': results_a[:20],
    'strategy_b_top20': results_b[:20]
}
with open('/tmp/final_bruteforce_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
log('\n✅ Results saved to /tmp/final_bruteforce_results.json')
