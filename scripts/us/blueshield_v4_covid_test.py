#!/usr/bin/env python3
"""蓝盾V4 — 排除疫情训练 + 极端环境止损测试"""
import pandas as pd
import numpy as np
import time, warnings
warnings.filterwarnings('ignore')

DATA = '/home/hermes/.hermes/openclaw-archive/data/us/us_hist_sp500_10y.parquet'

t0 = time.time()
print("=" * 90)
print("蓝盾V4 — 排除COVID训练 + 极端环境回测")
print("=" * 90)

# 数据+特征
df = pd.read_parquet(DATA).rename(columns={'sym': 'code'})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['code', 'date']).reset_index(drop=True)
mkt = df.groupby('date').agg(mkt_breadth=('close', lambda x: (x.pct_change(5) > 0).mean())).reset_index()
df = df.merge(mkt, on='date', how='left')

def feats(g):
    c, v, h, l, o = g['close'], g['volume'], g['high'], g['low'], g['open']
    for n in [1,2,3,5,10,20,60]: g[f'ret_{n}'] = c.pct_change(n)
    for n in [5,10,20,60]: g[f'vol_{n}'] = c.pct_change().rolling(n).std() * np.sqrt(252)
    d = c.diff(); up = d.clip(lower=0); dn = (-d).clip(lower=0)
    rs = up.rolling(14).mean() / dn.rolling(14).mean().replace(0, np.nan)
    g['rsi_14'] = 100 - 100/(1+rs)
    ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
    g['macd_hist'] = (ema12-ema26) - (ema12-ema26).ewm(span=9).mean()
    sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
    g['bb_pos'] = (c - (sma20-2*std20)) / (4*std20).replace(0, np.nan)
    vs20 = v.rolling(20).mean(); vs60 = v.rolling(60).mean()
    g['vol_ratio'] = v / vs20.replace(0, np.nan)
    g['vol_trend'] = vs20 / vs60.replace(0, np.nan)
    g['hl_range'] = (h-l)/c
    g['body_ratio'] = abs(c-o)/(h-l).replace(0, np.nan)
    for n in [5,10,20,50]: g[f'bias_{n}'] = (c-c.rolling(n).mean())/c.rolling(n).mean().replace(0, np.nan)
    g['high_52w'] = h.rolling(250).max(); g['low_52w'] = l.rolling(250).min()
    g['dist_52w_high'] = c/g['high_52w']-1; g['dist_52w_low'] = c/g['low_52w']-1
    for n in [5,10,20,60]: g[f'mom_{n}'] = c/c.shift(n)-1
    tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    g['atr_pct'] = tr.rolling(14).mean()/c
    g['ret_skew'] = c.pct_change().rolling(20).skew()
    g['ret_ratio_5_20'] = c.pct_change(5)/c.pct_change(20).replace(0, np.nan)
    g['rank_ret_5'] = g['ret_5'].rank(pct=True)
    g['rank_vol_20'] = g['vol_20'].rank(pct=True)
    g['rank_rsi'] = g['rsi_14'].rank(pct=True)
    g['rank_bias_20'] = g['bias_20'].rank(pct=True)
    return g

groups = []
for code, grp in df.groupby('code'):
    groups.append(feats(grp))
df = pd.concat(groups, ignore_index=True)
df['target_5d'] = df.groupby('code')['close'].transform(lambda x: x.shift(-5)/x-1)
df['daily_ret'] = df.groupby('code')['close'].pct_change()

skip = {'date','code','open','high','low','close','volume','target_5d','daily_ret',
        'high_52w','low_52w','mkt_breadth'}
feat_cols = [c for c in df.columns if c not in skip]
df = df.replace([np.inf,-np.inf], np.nan)
core = [c for c in feat_cols if not c.startswith('dist_52w') and not c.startswith('ret_skew')
        and not c.startswith('ret_ratio')]
df = df.dropna(subset=core + ['target_5d','daily_ret']).sort_values('date').reset_index(drop=True)

# COVID期间定义
covid_start = pd.Timestamp('2020-02-15')
covid_end = pd.Timestamp('2020-08-31')

# ============================================================
# 实验1：正常训练（包含COVID）
# ============================================================
print("\n[1/3] 正常训练（包含COVID）...")
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

train_end = pd.Timestamp('2021-12-31')
val_end = pd.Timestamp('2023-12-31')

Xt_all = df.loc[df['date']<=train_end, feat_cols].values
yt_all = df.loc[df['date']<=train_end, 'target_5d'].values
Xv = df.loc[(df['date']>train_end)&(df['date']<=val_end), feat_cols].values
yv = df.loc[(df['date']>train_end)&(df['date']<=val_end), 'target_5d'].values
test_mask = df['date'] > pd.Timestamp('2017-12-31')
Xs = df.loc[test_mask, feat_cols].values

models_all = {}
for name, Cls, kw in [
    ('LGB', LGBMRegressor, dict(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=0.1, min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1)),
    ('Cat', CatBoostRegressor, dict(iterations=600, depth=6, learning_rate=0.03, l2_leaf_reg=3,
        random_seed=42, verbose=0)),
]:
    print(f"  {name}...", end=' ', flush=True)
    m = Cls(**kw)
    if name == 'Cat':
        m.fit(Xt_all, yt_all, eval_set=(Xv, yv))
    else:
        m.fit(Xt_all, yt_all, eval_set=[(Xv, yv)])
    models_all[name] = m.predict(Xs)
    print(f"done ({time.time()-t0:.0f}s)")

# ============================================================
# 实验2：排除COVID训练
# ============================================================
print("\n[2/3] 排除COVID训练（2020-02~2020-08移除）...")

no_covid_mask = (df['date'] <= train_end) & ~((df['date'] >= covid_start) & (df['date'] <= covid_end))
Xt_nocovid = df.loc[no_covid_mask, feat_cols].values
yt_nocovid = df.loc[no_covid_mask, 'target_5d'].values

print(f"  正常训练样本: {len(Xt_all):,} | 排除COVID后: {len(Xt_nocovid):,} (减少{len(Xt_all)-len(Xt_nocovid):,})")

models_nocovid = {}
for name, Cls, kw in [
    ('LGB', LGBMRegressor, dict(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=0.1, min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1)),
    ('Cat', CatBoostRegressor, dict(iterations=600, depth=6, learning_rate=0.03, l2_leaf_reg=3,
        random_seed=42, verbose=0)),
]:
    print(f"  {name}(无COVID)...", end=' ', flush=True)
    m = Cls(**kw)
    if name == 'Cat':
        m.fit(Xt_nocovid, yt_nocovid, eval_set=(Xv, yv))
    else:
        m.fit(Xt_nocovid, yt_nocovid, eval_set=[(Xv, yv)])
    models_nocovid[name] = m.predict(Xs)
    print(f"done ({time.time()-t0:.0f}s)")

# 写入分数
test_idx = df[test_mask].index
df.loc[test_idx, 'score_LGB_all'] = models_all['LGB']
df.loc[test_idx, 'score_Cat_all'] = models_all['Cat']
df.loc[test_idx, 'score_LGB_nc'] = models_nocovid['LGB']
df.loc[test_idx, 'score_Cat_nc'] = models_nocovid['Cat']

test_df = df.dropna(subset=['score_LGB_all','score_Cat_all','score_LGB_nc','score_Cat_nc']).copy()
print(f"  测试数据: {len(test_df):,}行, {test_df['date'].min().date()}~{test_df['date'].max().date()}")

# ============================================================
# 回测 + 止损分析
# ============================================================
print(f"\n[3/3] 回测 + 止损分析...")

def backtest(test_df, score_col, top_n=15, hold_days=5, stop_loss=None):
    dates = sorted(test_df['date'].unique())
    cash = 1.0; positions = {}; equity_curve = []; trades = []; peak = 1.0
    for i, date in enumerate(dates):
        day = test_df[test_df['date'] == date]
        if len(day) == 0:
            equity_curve.append({'date': date, 'equity': cash, 'dd': cash/peak-1})
            continue
        for code in list(positions.keys()):
            row = day[day['code'] == code]
            if len(row) == 0: continue
            pos = positions[code]
            cur = row['close'].values[0]
            ret = cur / pos['entry_price'] - 1
            pos['peak_price'] = max(pos.get('peak_price', pos['entry_price']), cur)
            if stop_loss and ret <= stop_loss:
                cash += pos['shares'] * cur
                trades.append({'code': code, 'entry_date': pos['entry_date'], 'exit_date': date,
                    'entry_price': pos['entry_price'], 'exit_price': cur,
                    'pnl': ret, 'reason': 'stop_loss', 'days_held': (date - pos['entry_date']).days})
                del positions[code]
        if i % hold_days == 0 and i > 0:
            for code in list(positions.keys()):
                row = day[day['code'] == code]
                if len(row) == 0: continue
                pos = positions.pop(code)
                price = row['close'].values[0]
                cash += pos['shares'] * price
                trades.append({'code': code, 'entry_date': pos['entry_date'], 'exit_date': date,
                    'entry_price': pos['entry_price'], 'exit_price': price,
                    'pnl': price/pos['entry_price']-1, 'reason': 'expire', 'days_held': (date - pos['entry_date']).days})
            avail = day[~day['code'].isin(positions.keys())].copy()
            if len(avail) > 0:
                top = avail.nlargest(top_n, score_col)
                size_per = cash / len(top)
                for _, row in top.iterrows():
                    if row['close'] > 0:
                        positions[row['code']] = {'entry_price': row['close'], 'shares': size_per/row['close'],
                            'entry_date': date, 'peak_price': row['close']}
                        cash -= size_per
        pos_val = sum(pos['shares'] * day[day['code']==c]['close'].values[0]
                      for c, pos in positions.items() if len(day[day['code']==c])>0)
        eq = cash + pos_val; peak = max(peak, eq)
        equity_curve.append({'date': date, 'equity': eq, 'dd': eq/peak-1})
    return equity_curve, trades

def metrics(eq_list, trades, name):
    eq = pd.DataFrame(eq_list); eq['ret'] = eq['equity'].pct_change()
    days = (eq['date'].max()-eq['date'].min()).days
    total = eq['equity'].iloc[-1]/eq['equity'].iloc[0]-1
    annual = (1+total)**(365/max(days,1))-1
    max_dd = eq['dd'].min()
    in_dd = False; dd_list = []; s = 0
    for j in range(len(eq)):
        if eq['dd'].iloc[j] < -0.001 and not in_dd: in_dd = True; s = j
        elif eq['dd'].iloc[j] >= -0.001 and in_dd: in_dd = False; dd_list.append(eq['dd'].iloc[s:j].min())
    if in_dd: dd_list.append(eq['dd'].iloc[s:].min())
    avg_dd = np.mean(dd_list) if dd_list else 0
    dr = eq['ret'].dropna()
    sharpe = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    sortino = dr.mean()/dr[dr<0].std()*np.sqrt(252) if (dr<0).sum()>0 else 0
    if trades:
        tdf = pd.DataFrame(trades)
        win_rate = len(tdf[tdf['pnl']>0])/len(tdf) if len(tdf)>0 else 0
        n_sl = len(tdf[tdf['reason']=='stop_loss']) if 'reason' in tdf.columns else 0
    else: win_rate = n_sl = 0
    return {'name': name, 'annual_return': annual, 'max_drawdown': max_dd, 'avg_drawdown': avg_dd,
            'sharpe': sharpe, 'sortino': sortino, 'win_rate': win_rate, 'n_trades': len(trades), 'n_sl': n_sl}

# 实验矩阵
configs = [
    # 正常训练
    ('LGB(含COVID) Top-15 5d', 'score_LGB_all', 15, 5, None),
    ('LGB(含COVID) Top-15 5d SL-8%', 'score_LGB_all', 15, 5, -0.08),
    ('LGB(含COVID) Top-15 5d SL-10%', 'score_LGB_all', 15, 5, -0.10),
    ('Cat(含COVID) Top-15 7d', 'score_Cat_all', 15, 7, None),
    ('Cat(含COVID) Top-15 7d SL-10%', 'score_Cat_all', 15, 7, -0.10),
    # 排除COVID训练
    ('LGB(无COVID) Top-15 5d', 'score_LGB_nc', 15, 5, None),
    ('LGB(无COVID) Top-15 5d SL-8%', 'score_LGB_nc', 15, 5, -0.08),
    ('LGB(无COVID) Top-15 5d SL-10%', 'score_LGB_nc', 15, 5, -0.10),
    ('Cat(无COVID) Top-15 7d', 'score_Cat_nc', 15, 7, None),
    ('Cat(无COVID) Top-15 7d SL-10%', 'score_Cat_nc', 15, 7, -0.10),
]

results = []
for name, col, tn, hd, sl in configs:
    eq, trades = backtest(test_df, col, tn, hd, sl)
    m = metrics(eq, trades, name)
    m['equity'] = eq
    results.append(m)
    q = "🟢" if m['avg_drawdown'] > -0.03 else "🟡" if m['avg_drawdown'] > -0.05 else "🔴"
    sl_str = f" SL{sl:.0%}" if sl else ""
    print(f"  {name:<40} 夏普{m['sharpe']:.2f} 年化{m['annual_return']:+.1%} 最大DD{m['max_drawdown']:.1%} 平均DD{m['avg_drawdown']:.2%} 止损{m['n_sl']}笔 {q}")

# ============================================================
# COVID期间专项分析
# ============================================================
print(f"\n{'=' * 90}")
print("📊 COVID期间专项分析（2020-02-15 ~ 2020-08-31）")
print("=" * 90)

covid_results = []
for r in results:
    eq_df = pd.DataFrame(r['equity'])
    covid_eq = eq_df[(eq_df['date'] >= covid_start) & (eq_df['date'] <= covid_end)]
    if len(covid_eq) < 10: continue
    covid_ret = covid_eq['equity'].iloc[-1] / covid_eq['equity'].iloc[0] - 1
    covid_dd = covid_eq['dd'].min()
    covid_results.append({
        'name': r['name'], 'covid_return': covid_ret, 'covid_dd': covid_dd,
        'n_sl': r['n_sl']
    })

print(f"\n  {'策略':<40} {'COVID收益':>10} {'COVID最大DD':>12} {'止损笔数':>8}")
print("  " + "-" * 75)
for cr in sorted(covid_results, key=lambda x: x['covid_return'], reverse=True):
    q = "🟢" if cr['covid_return'] > 0 else "🔴"
    print(f"  {cr['name']:<40} {cr['covid_return']:>+9.1%} {cr['covid_dd']:>11.1%} {cr['n_sl']:>8} {q}")

# ============================================================
# 2022熊市分析
# ============================================================
print(f"\n{'=' * 90}")
print("📊 2022熊市分析")
print("=" * 90)

bear_2022_start = pd.Timestamp('2022-01-01')
bear_2022_end = pd.Timestamp('2022-12-31')

bear_results = []
for r in results:
    eq_df = pd.DataFrame(r['equity'])
    bear_eq = eq_df[(eq_df['date'] >= bear_2022_start) & (eq_df['date'] <= bear_2022_end)]
    if len(bear_eq) < 10: continue
    bear_ret = bear_eq['equity'].iloc[-1] / bear_eq['equity'].iloc[0] - 1
    bear_dd = bear_eq['dd'].min()
    bear_results.append({
        'name': r['name'], 'bear_return': bear_ret, 'bear_dd': bear_dd
    })

print(f"\n  {'策略':<40} {'2022收益':>10} {'2022最大DD':>12}")
print("  " + "-" * 65)
for br in sorted(bear_results, key=lambda x: x['bear_return'], reverse=True):
    q = "🟢" if br['bear_return'] > -0.15 else "🟡" if br['bear_return'] > -0.30 else "🔴"
    print(f"  {br['name']:<40} {br['bear_return']:>+9.1%} {br['bear_dd']:>11.1%} {q}")

# ============================================================
# 总结
# ============================================================
print(f"\n{'=' * 90}")
print("📊 核心对比：含COVID vs 无COVID训练")
print("=" * 90)

print(f"\n  {'指标':<20} {'LGB含COVID':>12} {'LGB无COVID':>12} {'Cat含COVID':>12} {'Cat无COVID':>12}")
print("  " + "-" * 70)

for metric in ['sharpe', 'annual_return', 'max_drawdown', 'avg_drawdown', 'win_rate', 'n_sl']:
    vals = []
    for r in results:
        vals.append(r[metric])
    # 按顺序：LGB_all, LGB_nc, Cat_all, Cat_nc（无SL的版本）
    if metric == 'n_sl':
        row_vals = [vals[0], vals[5], vals[3], vals[8]]
    else:
        row_vals = [vals[0], vals[5], vals[3], vals[8]]
    
    if metric in ['annual_return', 'win_rate']:
        print(f"  {metric:<20} {row_vals[0]:>+11.1%} {row_vals[1]:>+11.1%} {row_vals[2]:>+11.1%} {row_vals[3]:>+11.1%}")
    elif metric == 'n_sl':
        print(f"  {metric:<20} {row_vals[0]:>12} {row_vals[1]:>12} {row_vals[2]:>12} {row_vals[3]:>12}")
    else:
        print(f"  {metric:<20} {row_vals[0]:>11.2%} {row_vals[1]:>11.2%} {row_vals[2]:>11.2%} {row_vals[3]:>11.2%}")

print(f"\n  COVID对LGB的影响: 夏普从{results[0]['sharpe']:.2f}→{results[5]['sharpe']:.2f} ({(results[5]['sharpe']-results[0]['sharpe'])/results[0]['sharpe']*100:+.0f}%)")
print(f"  COVID对Cat的影响: 夏普从{results[3]['sharpe']:.2f}→{results[8]['sharpe']:.2f} ({(results[8]['sharpe']-results[3]['sharpe'])/results[3]['sharpe']*100:+.0f}%)")

print(f"\n总耗时: {time.time()-t0:.1f}s")
print("=" * 90)
