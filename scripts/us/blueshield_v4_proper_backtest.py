#!/usr/bin/env python3
"""
蓝盾V4 — 正确的逐日持仓模拟
修复：年化计算、真实持仓跟踪、不重叠入场
"""
import pandas as pd
import numpy as np
import json, time, warnings
warnings.filterwarnings('ignore')

DATA = '/home/hermes/.hermes/openclaw-archive/data/us/us_hist_sp500_10y.parquet'
OUT = '/home/hermes/.hermes/openclaw-archive/analysis'

print("=" * 90)
print("蓝盾V4 — 正确持仓模拟")
print("=" * 90)

# ============================================================
# 1. 数据+特征（精简版，加速）
# ============================================================
t0 = time.time()
print("\n[1/4] 数据...")
df = pd.read_parquet(DATA).rename(columns={'sym': 'code'})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['code', 'date']).reset_index(drop=True)

# 市场特征
mkt = df.groupby('date').agg(
    mkt_breadth=('close', lambda x: (x.pct_change(5) > 0).mean()),
    mkt_ret_5d=('close', lambda x: x.pct_change(5).median()),
    mkt_vol=('close', lambda x: x.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) if len(x) > 20 else np.nan),
).reset_index()
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
        'high_52w','low_52w','mkt_vol'}
feat_cols = [c for c in df.columns if c not in skip]
df = df.replace([np.inf,-np.inf], np.nan)
core = [c for c in feat_cols if not c.startswith('dist_52w') and not c.startswith('ret_skew') 
        and not c.startswith('ret_ratio')]
df = df.dropna(subset=core + ['target_5d','daily_ret']).sort_values('date').reset_index(drop=True)
print(f"  特征: {len(feat_cols)}维, 数据: {len(df):,}行 ({time.time()-t0:.1f}s)")

# ============================================================
# 2. 训练
# ============================================================
print("\n[2/4] 训练...")
t0 = time.time()
train_end = pd.Timestamp('2021-12-31')
val_end = pd.Timestamp('2023-12-31')

Xt = df.loc[df['date']<=train_end, feat_cols].values
yt = df.loc[df['date']<=train_end, 'target_5d'].values
Xv = df.loc[(df['date']>train_end)&(df['date']<=val_end), feat_cols].values
yv = df.loc[(df['date']>train_end)&(df['date']<=val_end), 'target_5d'].values
Xs = df.loc[df['date']>val_end, feat_cols].values

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

models = {}
for name, Cls, kw in [
    ('XGB', XGBRegressor, dict(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, min_child_weight=10, random_state=42, n_jobs=-1, verbosity=0)),
    ('LGB', LGBMRegressor, dict(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1)),
    ('Cat', CatBoostRegressor, dict(iterations=600, depth=6, learning_rate=0.03, l2_leaf_reg=3, random_seed=42, verbose=0)),
]:
    print(f"  {name}...", end=' ', flush=True)
    m = Cls(**kw)
    if name == 'Cat':
        m.fit(Xt, yt, eval_set=(Xv, yv))
    elif name == 'LGB':
        m.fit(Xt, yt, eval_set=[(Xv, yv)])
    else:
        m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
    models[name] = m
    print(f"done ({time.time()-t0:.0f}s)")

# 预测
preds = {n: m.predict(Xs) for n, m in models.items()}
preds['ENS'] = (preds['XGB']*0.4 + preds['LGB']*0.3 + preds['Cat']*0.3)

# ============================================================
# 3. 正确的逐日持仓模拟
# ============================================================
print(f"\n[3/4] 逐日持仓模拟...")

test_df = df.loc[df['date']>val_end, ['date','code','close','daily_ret']].copy()
for n, p in preds.items():
    test_df[f'score_{n}'] = p

# 市场择时
mkt_sig = test_df.groupby('date').agg(
    breadth=('daily_ret', lambda x: (x>0).mean())
).reset_index()
mkt_sig['pos'] = 1.0
mkt_sig.loc[mkt_sig['breadth'] < 0.45, 'pos'] = 0.6
mkt_sig.loc[mkt_sig['breadth'] < 0.35, 'pos'] = 0.3
mkt_timing = mkt_sig.set_index('date')['pos']

def proper_backtest(test_df, score_col, top_n=15, hold_days=7, 
                    stop_loss=None, market_timing=None):
    """
    正确的逐日模拟：
    - 每hold_days天轮换一次（不重叠入场）
    - 持仓期间每日计算真实PnL
    - 支持止损和市场择时
    """
    dates = sorted(test_df['date'].unique())
    
    cash = 1.0
    positions = {}  # code -> {entry_price, shares, entry_date, peak_price}
    equity_curve = []
    trades = []
    
    for i, date in enumerate(dates):
        day = test_df[test_df['date'] == date]
        if len(day) == 0:
            equity_curve.append({'date': date, 'equity': cash})
            continue
        
        # 1. 按日计算持仓PnL
        pos_value = 0
        closed_today = []
        for code, pos in list(positions.items()):
            row = day[day['code'] == code]
            if len(row) == 0:
                continue
            cur_price = row['close'].values[0]
            ret = cur_price / pos['entry_price'] - 1
            pos['peak_price'] = max(pos.get('peak_price', pos['entry_price']), cur_price)
            
            # 止损检查
            if stop_loss and ret <= stop_loss:
                closed_today.append((code, cur_price, 'stop_loss'))
            # Trailing stop: 从最高点回落
            elif stop_loss and pos['peak_price'] > pos['entry_price'] * 1.05:
                trail_ret = (pos['peak_price'] - cur_price) / pos['peak_price']
                if trail_ret >= 0.05:
                    closed_today.append((code, cur_price, 'trailing'))
            
            pos_value += pos['shares'] * cur_price
        
        # 执行止损
        for code, price, reason in closed_today:
            pos = positions.pop(code)
            pnl = pos['shares'] * (price - pos['entry_price'])
            cash += pos['shares'] * price
            trades.append({
                'code': code, 'entry_date': pos['entry_date'], 'exit_date': date,
                'entry_price': pos['entry_price'], 'exit_price': price,
                'pnl': pnl / pos['shares'] / pos['entry_price'],  # return %
                'reason': reason,
                'days_held': (date - pos['entry_date']).days,
            })
        
        # 2. 检查是否轮换
        if i % hold_days == 0 and i > 0:
            # 关闭所有剩余持仓
            for code in list(positions.keys()):
                row = day[day['code'] == code]
                if len(row) == 0:
                    continue
                price = row['close'].values[0]
                pos = positions.pop(code)
                pnl = pos['shares'] * (price - pos['entry_price'])
                cash += pos['shares'] * price
                trades.append({
                    'code': code, 'entry_date': pos['entry_date'], 'exit_date': date,
                    'entry_price': pos['entry_price'], 'exit_price': price,
                    'pnl': pnl / pos['shares'] / pos['entry_price'],
                    'reason': 'expire',
                    'days_held': (date - pos['entry_date']).days,
                })
            
            # 3. 开新仓
            avail = day[~day['code'].isin(positions.keys())].copy()
            if len(avail) > 0:
                avail['rank'] = avail[score_col].rank(ascending=False)
                top = avail.nsmallest(top_n, 'rank')
                
                # 市场择时
                pos_factor = 1.0
                if market_timing is not None and date in market_timing.index:
                    pos_factor = float(market_timing.loc[date])
                
                alloc = cash * pos_factor
                n_new = len(top)
                if n_new > 0:
                    size_per = alloc / n_new
                    for _, row in top.iterrows():
                        if row['close'] > 0:
                            shares = size_per / row['close']
                            positions[row['code']] = {
                                'entry_price': row['close'],
                                'shares': shares,
                                'entry_date': date,
                                'peak_price': row['close'],
                            }
                            cash -= size_per
        
        # 4. 计算总权益
        pos_value = sum(
            pos['shares'] * day[day['code']==code]['close'].values[0]
            for code, pos in positions.items()
            if len(day[day['code']==code]) > 0
        )
        equity = cash + pos_value
        equity_curve.append({'date': date, 'equity': equity})
    
    return equity_curve, trades

def metrics(eq_list, trades, name):
    eq = pd.DataFrame(eq_list)
    eq['ret'] = eq['equity'].pct_change()
    
    days = (eq['date'].max()-eq['date'].min()).days
    total = eq['equity'].iloc[-1]/eq['equity'].iloc[0]-1
    annual = (1+total)**(365/max(days,1))-1
    
    rolling_max = eq['equity'].cummax()
    dd = eq['equity']/rolling_max - 1
    max_dd = dd.min()
    
    in_dd = False; dd_list = []; s = 0
    for j in range(len(dd)):
        if dd.iloc[j] < -0.001 and not in_dd:
            in_dd = True; s = j
        elif dd.iloc[j] >= -0.001 and in_dd:
            in_dd = False; dd_list.append(dd.iloc[s:j].min())
    if in_dd: dd_list.append(dd.iloc[s:].min())
    avg_dd = np.mean(dd_list) if dd_list else 0
    
    dr = eq['ret'].dropna()
    sharpe = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    sortino = dr.mean()/dr[dr<0].std()*np.sqrt(252) if (dr<0).sum()>0 else 0
    calmar = annual/abs(max_dd) if max_dd != 0 else 0
    
    if trades:
        tdf = pd.DataFrame(trades)
        wins = tdf[tdf['pnl']>0]
        losses = tdf[tdf['pnl']<=0]
        win_rate = len(wins)/len(tdf)
        avg_win = wins['pnl'].mean() if len(wins)>0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses)>0 else 0
        reasons = tdf['reason'].value_counts().to_dict() if 'reason' in tdf.columns else {}
        avg_hold = tdf['days_held'].mean() if 'days_held' in tdf.columns else 0
    else:
        win_rate = avg_win = avg_loss = avg_hold = 0
        reasons = {}
    
    return {
        'name': name, 'annual_return': annual, 'max_drawdown': max_dd,
        'avg_drawdown': avg_dd, 'n_dd_events': len(dd_list),
        'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar,
        'win_rate': win_rate, 'n_trades': len(trades),
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'avg_hold_days': avg_hold, 'exit_reasons': reasons,
        'total_return': total,
    }

# ============================================================
# 4. 实验矩阵
# ============================================================
print(f"\n[4/4] 实验矩阵...")

score_cols = {'XGB': 'score_XGB', 'LGB': 'score_LGB', 'Cat': 'score_Cat', '加权集成': 'score_ENS'}
exps = []

# 单模型 × Top-N × hold
for name, col in score_cols.items():
    for tn in [10, 15, 20]:
        for hd in [5, 7, 10]:
            exps.append({'name': f'{name} Top-{tn} {hd}d', 'score': col, 'tn': tn, 'hd': hd, 'sl': None, 'timing': None})

# 止损
for sl in [-0.05, -0.08, -0.10]:
    exps.append({'name': f'加权集成 Top-15 7d SL{int(sl*100)}%', 'score': 'score_ENS', 'tn': 15, 'hd': 7, 'sl': sl, 'timing': None})

# 择时
exps.append({'name': '加权集成 Top-15 7d + 择时', 'score': 'score_ENS', 'tn': 15, 'hd': 7, 'sl': None, 'timing': mkt_timing})
exps.append({'name': '加权集成 Top-15 7d + 择时+SL-8%', 'score': 'score_ENS', 'tn': 15, 'hd': 7, 'sl': -0.08, 'timing': mkt_timing})

results = []
for i, e in enumerate(exps):
    t1 = time.time()
    eq, trades = proper_backtest(test_df, e['score'], e['tn'], e['hd'], e['sl'], e['timing'])
    m = metrics(eq, trades, e['name'])
    m['config'] = e
    results.append(m)
    dt = time.time()-t1
    q = "🟢" if m['avg_drawdown'] > -0.03 else "🟡" if m['avg_drawdown'] > -0.05 else "🔴"
    print(f"  [{i+1}/{len(exps)}] {e['name']:<45} 夏普{m['sharpe']:.2f} 年化{m['annual_return']:+.1%} 最大DD{m['max_drawdown']:.1%} 平均DD{m['avg_drawdown']:.2%} 胜率{m['win_rate']:.1%} {q} ({dt:.1f}s)")

# ============================================================
# 输出
# ============================================================
print("\n" + "=" * 120)
print("📊 全部结果（按效率排序）")
print("=" * 120)

for r in results:
    r['efficiency'] = r['sharpe'] * max(0.01, 1 + r['avg_drawdown'])

results.sort(key=lambda x: x['efficiency'], reverse=True)

print(f"\n{'#':>3} {'策略':<48} {'年化':>7} {'最大DD':>8} {'平均DD':>8} {'夏普':>6} {'Sortino':>7} {'胜率':>6} {'交易':>6}")
print("-" * 110)
for i, r in enumerate(results):
    q = "🟢" if r['avg_drawdown'] > -0.03 else "🟡" if r['avg_drawdown'] > -0.05 else "🔴"
    print(f"{i+1:>3} {r['name']:<48} {r['annual_return']:>+6.1%} {r['max_drawdown']:>7.1%} {r['avg_drawdown']:>7.2%} {r['sharpe']:>6.2f} {r['sortino']:>7.2f} {r['win_rate']:>5.1%} {r['n_trades']:>6} {q}")

# 深度分析
print("\n" + "=" * 120)
print("🔍 分层分析")
print("=" * 120)

print("\n📊 模型层 — 平均DD对比（Top-15 7d基线）")
for r in [r for r in results if 'Top-15 7d' in r['name'] and '择时' not in r['name'] and 'SL' not in r['name']]:
    print(f"  {r['name']:<35} 平均DD: {r['avg_drawdown']:.3%} | 胜率: {r['win_rate']:.1%} | 夏普: {r['sharpe']:.2f}")

print("\n📊 决策层 — 止损/择时效果")
for r in [r for r in results if 'SL' in r['name'] or '择时' in r['name']]:
    print(f"  {r['name']:<48} 平均DD: {r['avg_drawdown']:.3%} | 最大DD: {r['max_drawdown']:.1%} | 夏普: {r['sharpe']:.2f}")

print("\n📊 退出原因分布（最优配置）")
for r in results[:3]:
    print(f"  {r['name']}: {r['exit_reasons']}")

print("\n🏆 Top-5 推荐")
for i, r in enumerate(results[:5]):
    print(f"\n  #{i+1} {r['name']}")
    print(f"      年化: {r['annual_return']:+.1%} | 夏普: {r['sharpe']:.2f} | Sortino: {r['sortino']:.2f}")
    print(f"      最大DD: {r['max_drawdown']:.1%} | 平均DD: {r['avg_drawdown']:.3%} ({r['n_dd_events']}次)")
    print(f"      胜率: {r['win_rate']:.1%} | 盈亏比: {r['avg_win']/max(r['avg_loss'],0.001):.2f} | 交易数: {r['n_trades']}")
    print(f"      退出: {r['exit_reasons']}")

# 保存
out = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'n_experiments': len(results),
    'results': [{k:v for k,v in r.items() if k!='config'} for r in results],
}
with open(f'{OUT}/v4_proper_backtest.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)

print(f"\n\n保存 → analysis/v4_proper_backtest.json")
print(f"总耗时: {time.time()-t0:.1f}s")
print("=" * 90)
