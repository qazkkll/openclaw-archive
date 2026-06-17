#!/usr/bin/env python3
"""快速因子扫描 v2 - 修正字段名"""
import tushare as ts, pandas as pd, numpy as np, time, json
from datetime import datetime

TS_TOKEN = 'ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db'
pro = ts.pro_api(TS_TOKEN)

def log(m): print(f'[{datetime.now():%H:%M:%S}] {m}', flush=True)

log('=== 快速因子扫描 v2 ===')

# 1. Get calendar
cal = pro.trade_cal(start_date='20260301', end_date='20260531')
days = sorted(cal[cal['is_open']==1]['cal_date'].tolist()[-60:])
log(f'Using {len(days)} trading days: {days[0]} to {days[-1]}')

# 2. Pull daily_basic
log('Pulling daily_basic...')
all_db = []
for d in days:
    df = pro.daily_basic(trade_date=d, fields='ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pe_ttm,pb,ps,total_mv,circ_mv')
    if df is not None and len(df) > 0:
        all_db.append(df)
    time.sleep(0.1)
df_db = pd.concat(all_db, ignore_index=True)
log(f'daily_basic: {len(df_db)} rows, cols: {df_db.columns.tolist()}')

# 3. Pull OHLCV
log('Pulling OHLCV...')
all_daily = []
for d in days:
    df = pro.daily(trade_date=d, fields='ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount')
    if df is not None and len(df) > 0:
        all_daily.append(df)
    time.sleep(0.1)
df_ohlcv = pd.concat(all_daily, ignore_index=True)
log(f'OHLCV: {len(df_ohlcv)} rows')

# 4. Merge
log('Merging...')
df_merged = df_ohlcv.merge(
    df_db[['ts_code','trade_date','turnover_rate','volume_ratio','pe','pe_ttm','pb','circ_mv']],
    on=['ts_code','trade_date'], how='left'
)
log(f'Merged: {len(df_merged)} rows')

# 5. Compute factors
log('Computing factors...')
df_merged['trade_date'] = pd.to_datetime(df_merged['trade_date'])
df_merged = df_merged.sort_values(['ts_code', 'trade_date'])

df_merged['mom5'] = df_merged.groupby('ts_code')['close'].transform(lambda x: x / x.shift(5) - 1)
df_merged['mom10'] = df_merged.groupby('ts_code')['close'].transform(lambda x: x / x.shift(10) - 1)
df_merged['mom20'] = df_merged.groupby('ts_code')['close'].transform(lambda x: x / x.shift(20) - 1)

df_merged['ma5'] = df_merged.groupby('ts_code')['close'].transform(lambda x: x.rolling(5, min_periods=3).mean())
df_merged['ma10'] = df_merged.groupby('ts_code')['close'].transform(lambda x: x.rolling(10, min_periods=5).mean())
df_merged['ma20'] = df_merged.groupby('ts_code')['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())

df_merged['vol_ma5'] = df_merged.groupby('ts_code')['vol'].transform(lambda x: x.rolling(5, min_periods=3).mean())
df_merged['vol_ratio_calc'] = df_merged['vol'] / (df_merged['vol_ma5'] + 1)
df_merged['double_vol'] = (df_merged['vol_ratio_calc'] > 2.0).astype(float)

# 6. Forward returns
log('Computing forward returns...')
df_merged['fwd_1d'] = df_merged.groupby('ts_code')['close'].transform(lambda x: x.shift(-1) / x - 1)
df_merged['fwd_5d'] = df_merged.groupby('ts_code')['close'].transform(lambda x: x.shift(-5) / x - 1)

clean = df_merged.dropna(subset=['fwd_5d']).copy()
log(f'Clean samples: {len(clean)}')

# 7. Factor correlation analysis
factors = ['mom5', 'mom10', 'mom20', 'volume_ratio', 'vol_ratio_calc', 'double_vol',
           'pe', 'pe_ttm', 'pb', 'turnover_rate', 'pct_chg']

log('\n=== 因子预测力排名 (与5日收益的相关性) ===')
results = []
for f in factors:
    if f not in clean.columns:
        continue
    sub = clean[clean[f].notna() & (~np.isinf(clean[f]))]
    if len(sub) < 100:
        continue
    corr = sub[f].corr(sub['fwd_5d'])
    sub = sub.copy()
    sub['factor_rank'] = sub[f].rank(pct=True)
    top = sub[sub['factor_rank'] > 0.8]['fwd_5d'].mean()
    bot = sub[sub['factor_rank'] < 0.2]['fwd_5d'].mean()
    spread = top - bot
    top_win = (sub[sub['factor_rank'] > 0.8]['fwd_5d'] > 0).mean()
    results.append({'factor': f, 'corr': round(corr,4), 'spread': round(spread,4),
                    'top_win': round(top_win,3), 'samples': len(sub)})

results.sort(key=lambda x: abs(x['corr']), reverse=True)
for r in results:
    print(f"  {r['factor']:>15s}: corr={r['corr']:+.4f} spread={r['spread']:+.2%} top_win={r['top_win']:.0%} n={r['samples']}")

# 8. PE/PB bucketed returns
log('\n=== PE分段收益率 ===')
clean['pe_bucket'] = pd.qcut(clean['pe'].clip(0,100), 5, labels=['极低','低','中','高','极高'], duplicates='drop')
print(clean.groupby('pe_bucket')['fwd_5d'].agg(['mean','count']).to_string())

log('\n=== PB分段 ===')
clean['pb_bucket'] = pd.qcut(clean['pb'].clip(0,20), 5, labels=['极低','低','中','高','极高'], duplicates='drop')
print(clean.groupby('pb_bucket')['fwd_5d'].agg(['mean','count']).to_string())

log('\n=== 换手率分段 ===')
clean['turn_bucket'] = pd.qcut(clean['turnover_rate'].clip(0,20), 5, labels=['极低','低','中','高','极高'], duplicates='drop')
print(clean.groupby('turn_bucket')['fwd_5d'].agg(['mean','count']).to_string())

log('\n=== 量比(volume_ratio)分段 ===')
clean['vr_bucket'] = pd.qcut(clean['volume_ratio'].clip(0,5), 5, labels=['极低','低','中','高','极高'], duplicates='drop')
print(clean.groupby('vr_bucket')['fwd_5d'].agg(['mean','count']).to_string())

# 9. Combo tests
log('\n=== 多因子组合测试 ===')

clean['score_mom'] = clean['mom20'].fillna(0) * 0.6 + clean['mom10'].fillna(0) * 0.4
clean['rank_mom'] = clean.groupby('trade_date')['score_mom'].rank(pct=True)
c1 = clean[clean['rank_mom'] > 0.8]
log(f'C1纯动量 Top20%: 收益={c1["fwd_5d"].mean():+.4f} 胜率={(c1["fwd_5d"]>0).mean():.1%}')

clean['ma_signal'] = ((clean['ma5'] > clean['ma10']) & (clean['close'] > clean['ma5'])).astype(float)
clean['score_trend'] = clean['mom20'].fillna(0) * 0.5 + clean['ma_signal'] * 0.5
clean['rank_trend'] = clean.groupby('trade_date')['score_trend'].rank(pct=True)
c2 = clean[clean['rank_trend'] > 0.8]
log(f'C2动量+MA趋势 Top20%: 收益={c2["fwd_5d"].mean():+.4f} 胜率={(c2["fwd_5d"]>0).mean():.1%}')

clean['pe_inv'] = 1 / (clean['pe'].clip(1, 100))
clean['score_value'] = clean['mom20'].fillna(0) * 0.4 + clean['mom10'].fillna(0) * 0.3 + clean['pe_inv'].fillna(0) * 0.3
clean['rank_value'] = clean.groupby('trade_date')['score_value'].rank(pct=True)
c4 = clean[clean['rank_value'] > 0.8]
log(f'C4动量+低PE Top20%: 收益={c4["fwd_5d"].mean():+.4f} 胜率={(c4["fwd_5d"]>0).mean():.1%}')

clean['score_short'] = clean['ma_signal'] * 0.5 + clean['double_vol'] * 0.3 + clean['mom5'].fillna(0) * 0.2
clean['rank_short'] = clean.groupby('trade_date')['score_short'].rank(pct=True)
c5 = clean[clean['rank_short'] > 0.8]
log(f'C5倍量+MA趋势(短线) Top20%: 收益={c5["fwd_5d"].mean():+.4f} 胜率={(c5["fwd_5d"]>0).mean():.1%}')

clean['score_all'] = (clean['mom20'].fillna(0) * 0.25 + clean['mom10'].fillna(0) * 0.15 +
                      clean['ma_signal'] * 0.2 + clean['volume_ratio'].fillna(1) * 0.05 +
                      clean['double_vol'] * 0.1 + clean['pe_inv'].fillna(0) * 0.05 +
                      clean['pct_chg'].fillna(0) * 0.01)
clean['rank_all'] = clean.groupby('trade_date')['score_all'].rank(pct=True)
c6 = clean[clean['rank_all'] > 0.8]
log(f'C6全因子综合 Top20%: 收益={c6["fwd_5d"].mean():+.4f} 胜率={(c6["fwd_5d"]>0).mean():.1%}')

# Save
summary = {
    'timestamp': datetime.now().isoformat(),
    'factor_correlations': results,
    'combos': {
        'C1_pure_momentum': {'return': float(c1['fwd_5d'].mean()), 'win_rate': float((c1['fwd_5d']>0).mean()), 'trades': len(c1)},
        'C2_momentum_ma': {'return': float(c2['fwd_5d'].mean()), 'win_rate': float((c2['fwd_5d']>0).mean()), 'trades': len(c2)},
        'C4_momentum_value': {'return': float(c4['fwd_5d'].mean()), 'win_rate': float((c4['fwd_5d']>0).mean()), 'trades': len(c4)},
        'C5_short_term': {'return': float(c5['fwd_5d'].mean()), 'win_rate': float((c5['fwd_5d']>0).mean()), 'trades': len(c5)},
        'C6_full_factor': {'return': float(c6['fwd_5d'].mean()), 'win_rate': float((c6['fwd_5d']>0).mean()), 'trades': len(c6)},
    }
}
with open('/tmp/factor_v2_results.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

log('\n✅ 快速因子扫描v2完成！')
