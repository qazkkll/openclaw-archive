#!/usr/bin/env python3
"""蓝盾V4 — 单次训练 vs Walk-Forward 10年对比"""
import pandas as pd
import numpy as np
import time, warnings
warnings.filterwarnings('ignore')

DATA = '/home/hermes/.hermes/openclaw-archive/data/us/us_hist_sp500_10y.parquet'

t0 = time.time()
print("=" * 90)
print("蓝盾V4 — 单次训练10年回测（不做Walk-Forward）")
print("=" * 90)

# 数据+特征（同10年版本）
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
print(f"  特征: {len(feat_cols)}维, 数据: {len(df):,}行 ({time.time()-t0:.1f}s)")

# ============================================================
# 单次训练：用2016-2021训练，测2018-2026全部
# ============================================================
print("\n[1/2] 单次训练（2016-2021训练，2018-2026测试）...")

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

train_end = pd.Timestamp('2021-12-31')
val_end = pd.Timestamp('2023-12-31')

Xt = df.loc[df['date']<=train_end, feat_cols].values
yt = df.loc[df['date']<=train_end, 'target_5d'].values
Xv = df.loc[(df['date']>train_end)&(df['date']<=val_end), feat_cols].values
yv = df.loc[(df['date']>train_end)&(df['date']<=val_end), 'target_5d'].values

# 测试期：2018-2026（全部可用数据）
test_mask = df['date'] > pd.Timestamp('2017-12-31')
Xs = df.loc[test_mask, feat_cols].values

print(f"  训练: {len(Xt):,} | 验证: {len(Xv):,} | 测试: {len(Xs):,}")

preds = {}
print("  XGB...", end=' ', flush=True)
m = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.7, reg_alpha=0.1, min_child_weight=10, random_state=42, n_jobs=-1, verbosity=0)
m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
preds['XGB'] = m.predict(Xs)
print(f"done ({time.time()-t0:.0f}s)")

print("  LGB...", end=' ', flush=True)
m = LGBMRegressor(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.7, reg_alpha=0.1, min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1)
m.fit(Xt, yt, eval_set=[(Xv, yv)])
preds['LGB'] = m.predict(Xs)
print(f"done ({time.time()-t0:.0f}s)")

print("  Cat...", end=' ', flush=True)
m = CatBoostRegressor(iterations=600, depth=6, learning_rate=0.03, l2_leaf_reg=3,
    random_seed=42, verbose=0)
m.fit(Xt, yt, eval_set=(Xv, yv))
preds['Cat'] = m.predict(Xs)
print(f"done ({time.time()-t0:.0f}s)")

preds['ENS'] = preds['XGB']*0.4 + preds['LGB']*0.3 + preds['Cat']*0.3

# 写入测试数据
test_idx = df[test_mask].index
for n, p in preds.items():
    df.loc[test_idx, f'score_{n}'] = p

test_df = df.dropna(subset=['score_XGB','score_LGB','score_Cat','score_ENS']).copy()
print(f"  测试数据: {len(test_df):,}行, {test_df['date'].min().date()}~{test_df['date'].max().date()}")

# 回测函数
def backtest(test_df, score_col, top_n=15, hold_days=7, stop_loss=None):
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
                trades.append({'pnl': ret, 'reason': 'stop_loss'})
                del positions[code]
        if i % hold_days == 0 and i > 0:
            for code in list(positions.keys()):
                row = day[day['code'] == code]
                if len(row) == 0: continue
                pos = positions.pop(code)
                price = row['close'].values[0]
                cash += pos['shares'] * price
                trades.append({'pnl': price/pos['entry_price']-1, 'reason': 'expire'})
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
        win_rate = len(tdf[tdf['pnl']>0])/len(tdf)
    else: win_rate = 0
    return {'name': name, 'annual_return': annual, 'max_drawdown': max_dd, 'avg_drawdown': avg_dd,
            'sharpe': sharpe, 'sortino': sortino, 'win_rate': win_rate, 'n_trades': len(trades)}

# ============================================================
# 跑实验
# ============================================================
print(f"\n[2/2] 回测...")

exps = [
    ('Cat Top-15 10d', 'score_Cat', 15, 10, None),
    ('Cat Top-20 10d', 'score_Cat', 20, 10, None),
    ('Cat Top-15 7d', 'score_Cat', 15, 7, None),
    ('LGB Top-10 10d', 'score_LGB', 10, 10, None),
    ('LGB Top-15 5d', 'score_LGB', 15, 5, None),
    ('LGB Top-15 10d', 'score_LGB', 15, 10, None),
    ('XGB Top-15 10d', 'score_XGB', 15, 10, None),
    ('集成 Top-15 10d', 'score_ENS', 15, 10, None),
    ('集成 Top-15 7d', 'score_ENS', 15, 7, None),
    ('集成 Top-15 7d SL-8%', 'score_ENS', 15, 7, -0.08),
    ('集成 Top-15 7d SL-10%', 'score_ENS', 15, 7, -0.10),
]

print(f"\n  === 单次训练（2016-2021训练，2018-2026测试，~9年）===")
print(f"  {'策略':<35} {'年化':>7} {'最大DD':>8} {'平均DD':>8} {'夏普':>6} {'Sortino':>7}")
print("  " + "-" * 80)
single_results = []
for name, col, tn, hd, sl in exps:
    eq, trades = backtest(test_df, col, tn, hd, sl)
    m = metrics(eq, trades, name)
    m['equity'] = eq
    single_results.append(m)
    q = "🟢" if m['avg_drawdown'] > -0.03 else "🟡" if m['avg_drawdown'] > -0.05 else "🔴"
    print(f"  {name:<35} {m['annual_return']:>+6.1%} {m['max_drawdown']:>7.1%} {m['avg_drawdown']:>7.2%} {m['sharpe']:>6.2f} {m['sortino']:>7.2f} {q}")

# ============================================================
# 分年对比
# ============================================================
print(f"\n{'=' * 90}")
print("📊 分年表现（单次训练）")
print("=" * 90)

# 用最佳配置做分年
best_col = 'score_LGB'; best_tn = 15; best_hd = 5
eq, trades = backtest(test_df, best_col, best_tn, best_hd)
eq_df = pd.DataFrame(eq)
eq_df['year'] = eq_df['date'].dt.year

print(f"\n  LGB Top-15 5d 分年表现:")
print(f"  {'年份':>6} {'年化':>8} {'最大DD':>8} {'平均DD':>8}")
print("  " + "-" * 40)
for year in sorted(eq_df['year'].unique()):
    ye = eq_df[eq_df['year'] == year]
    if len(ye) < 10: continue
    yr = ye['equity'].iloc[-1] / ye['equity'].iloc[0] - 1
    rm = ye['dd'].min()
    # 平均DD
    in_dd = False; ddl = []; s = 0
    for j in range(len(ye)):
        if ye['dd'].iloc[j] < -0.001 and not in_dd: in_dd = True; s = j
        elif ye['dd'].iloc[j] >= -0.001 and in_dd: in_dd = False; ddl.append(ye['dd'].iloc[s:j].min())
    if in_dd: ddl.append(ye['dd'].iloc[s:].min())
    ad = np.mean(ddl) if ddl else 0
    print(f"  {year:>6} {yr:>+7.1%} {rm:>7.1%} {ad:>7.2%}")

# ============================================================
# 总结对比
# ============================================================
print(f"\n{'=' * 90}")
print("📊 关键对比：单次训练 vs Walk-Forward")
print("=" * 90)

wf_results = {
    'Cat Top-15 10d': {'sharpe': 0.62, 'annual': 0.187, 'max_dd': -0.478, 'avg_dd': -0.0583},
    'Cat Top-20 10d': {'sharpe': 0.67, 'annual': 0.201, 'max_dd': -0.462, 'avg_dd': -0.0508},
    'Cat Top-15 7d': {'sharpe': 0.76, 'annual': 0.248, 'max_dd': -0.496, 'avg_dd': -0.0482},
    'LGB Top-10 10d': {'sharpe': 1.02, 'annual': 0.479, 'max_dd': -0.568, 'avg_dd': -0.0565},
    'LGB Top-15 5d': {'sharpe': 1.13, 'annual': 0.484, 'max_dd': -0.568, 'avg_dd': -0.0477},
    'XGB Top-15 10d': {'sharpe': 1.01, 'annual': 0.430, 'max_dd': -0.598, 'avg_dd': -0.0466},
    '集成 Top-15 10d': {'sharpe': 0.98, 'annual': 0.402, 'max_dd': -0.586, 'avg_dd': -0.0471},
}

print(f"\n  {'策略':<30} {'单次训练夏普':>12} {'WF夏普':>10} {'衰减':>8}")
print("  " + "-" * 65)
for s in single_results:
    name = s['name']
    if name in wf_results:
        wf = wf_results[name]
        decay = (s['sharpe'] - wf['sharpe']) / wf['sharpe'] * 100
        print(f"  {name:<30} {s['sharpe']:>12.2f} {wf['sharpe']:>10.2f} {decay:>+7.1f}%")

print(f"\n{'=' * 90}")
print(f"总耗时: {time.time()-t0:.1f}s")
print("=" * 90)
