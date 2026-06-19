#!/usr/bin/env python3
"""
蓝盾V3 vs V4 — 同条件公平对比
V3: 110分公式（趋势30+动量25+MACD25+偏离10+RSI10+52周）
V4: LGB ML模型
使用相同的回测框架、数据、参数矩阵
"""
import pandas as pd
import numpy as np
import json, time, warnings
warnings.filterwarnings('ignore')

DATA = '/home/hermes/.hermes/openclaw-archive/data/us/us_hist_sp500_10y.parquet'
OUT = '/home/hermes/.hermes/openclaw-archive/analysis'

print("=" * 90)
print("蓝盾V3 vs V4 — 同条件公平对比")
print("=" * 90)

# ============================================================
# 1. 数据
# ============================================================
t0 = time.time()
print("\n[1/5] 数据...")
df = pd.read_parquet(DATA).rename(columns={'sym': 'code'})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['code', 'date']).reset_index(drop=True)
print(f"  数据: {len(df):,} 行, {df['code'].nunique()} 只股票 ({time.time()-t0:.1f}s)")

# ============================================================
# 2. V3公式评分（向量化，逐股票）
# ============================================================
print("\n[2/5] V3 110分公式评分...")
t1 = time.time()

def calc_v3_score_vectorized(g):
    """V3公式：110分制，6维度，完全向量化"""
    c = g['close']
    h = g['high']
    l = g['low']
    
    # MA
    ma5 = c.rolling(5).mean()
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    
    # 1. 趋势分（30分）：价格在均线上方 + 均线多头排列
    trend = ((c > ma5).astype(float) * 10 + 
             (ma5 > ma20).astype(float) * 10 + 
             (ma20 > ma60).astype(float) * 10)
    
    # 2. 动量分（25分）：5日和20日收益率为正
    ret5 = c.pct_change(5)
    ret20 = c.pct_change(20)
    momentum = ((ret5 > 0).astype(float) * 12.5 + 
                (ret20 > 0).astype(float) * 12.5)
    
    # 3. MACD分（25分）：MACD柱状图为正 + 金叉
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_score = ((macd > 0).astype(float) * 12.5 + 
                  (macd > signal).astype(float) * 12.5)
    
    # 4. 均线偏离分（10分）：价格在合理偏离范围内
    bias = (c - ma20) / ma20
    bias_score = ((bias > -0.05) & (bias < 0.10)).astype(float) * 10
    
    # 5. RSI分（10分）：RSI不超买不超卖
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi_score = ((rsi > 30) & (rsi < 70)).astype(float) * 10
    
    # 6. 52周位置分（10分）：价格在52周高位附近
    high_52w = h.rolling(252).max()
    low_52w = l.rolling(252).min()
    pos_52w = (c - low_52w) / (high_52w - low_52w + 1e-10)
    pos_score = (pos_52w > 0.7).astype(float) * 10
    
    scores = trend + momentum + macd_score + bias_score + rsi_score + pos_score
    return scores

df['v3_score'] = df.groupby('code').apply(
    calc_v3_score_vectorized
).reset_index(level=0, drop=True)

print(f"  V3评分完成 ({time.time()-t1:.1f}s)")
print(f"  V3评分分布:")
print(f"    均值: {df['v3_score'].mean():.1f}, 中位: {df['v3_score'].median():.1f}")
print(f"    标准差: {df['v3_score'].std():.1f}")
print(f"    范围: [{df['v3_score'].min():.0f}, {df['v3_score'].max():.0f}]")
print(f"    >60分占比: {(df['v3_score']>60).mean()*100:.1f}%")
print(f"    >80分占比: {(df['v3_score']>80).mean()*100:.1f}%")

# ============================================================
# 3. V4 LGB特征+模型（用于对比）
# ============================================================
print("\n[3/5] V4 LGB模型...")
t2 = time.time()

# V4特征
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
        'high_52w','low_52w','mkt_vol','v3_score'}
feat_cols = [c for c in df.columns if c not in skip]
df = df.replace([np.inf,-np.inf], np.nan)
core = [c for c in feat_cols if not c.startswith('dist_52w') and not c.startswith('ret_skew') 
        and not c.startswith('ret_ratio')]
df = df.dropna(subset=core + ['target_5d','daily_ret']).sort_values('date').reset_index(drop=True)
print(f"  特征: {len(feat_cols)}维, 数据: {len(df):,}行 ({time.time()-t2:.1f}s)")

# 训练LGB
from lightgbm import LGBMRegressor

train_end = pd.Timestamp('2021-12-31')
val_end = pd.Timestamp('2023-12-31')

Xt = df.loc[df['date']<=train_end, feat_cols].values
yt = df.loc[df['date']<=train_end, 'target_5d'].values
Xv = df.loc[(df['date']>train_end)&(df['date']<=val_end), feat_cols].values
yv = df.loc[(df['date']>train_end)&(df['date']<=val_end), 'target_5d'].values

print("  训练LGB...", end=' ', flush=True)
t3 = time.time()
lgb = LGBMRegressor(
    n_estimators=600, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
    min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1
)
lgb.fit(Xt, yt, eval_set=[(Xv, yv)])
print(f"done ({time.time()-t3:.0f}s)")

# 测试集预测
test_mask = df['date'] > val_end
Xs = df.loc[test_mask, feat_cols].values
df.loc[test_mask, 'lgb_score'] = lgb.predict(Xs)

# ============================================================
# 4. 回测框架（与V4完全相同）
# ============================================================
print("\n[4/5] 回测...")

mkt = df.groupby('date').agg(
    mkt_breadth=('daily_ret', lambda x: (x>0).mean()),
).reset_index()
mkt['pos'] = 1.0
mkt.loc[mkt['mkt_breadth'] < 0.45, 'pos'] = 0.6
mkt.loc[mkt['mkt_breadth'] < 0.35, 'pos'] = 0.3
mkt_timing = mkt.set_index('date')['pos']

def proper_backtest(test_df, score_col, top_n=15, hold_days=7, 
                    stop_loss=None, market_timing=None):
    dates = sorted(test_df['date'].unique())
    cash = 1.0
    positions = {}
    equity_curve = []
    trades = []
    
    for i, date in enumerate(dates):
        day = test_df[test_df['date'] == date]
        if len(day) == 0:
            equity_curve.append({'date': date, 'equity': cash})
            continue
        
        # 持仓PnL
        for code, pos in list(positions.items()):
            row = day[day['code'] == code]
            if len(row) == 0: continue
            cur_price = row['close'].values[0]
            ret = cur_price / pos['entry_price'] - 1
            pos['peak_price'] = max(pos.get('peak_price', pos['entry_price']), cur_price)
            
            if stop_loss and ret <= stop_loss:
                pnl = pos['shares'] * (cur_price - pos['entry_price'])
                cash += pos['shares'] * cur_price
                trades.append({'code': code, 'entry_date': pos['entry_date'], 'exit_date': date,
                    'entry_price': pos['entry_price'], 'exit_price': cur_price,
                    'pnl': pnl / pos['shares'] / pos['entry_price'], 'reason': 'stop_loss',
                    'days_held': (date - pos['entry_date']).days})
                del positions[code]
            elif stop_loss and pos['peak_price'] > pos['entry_price'] * 1.05:
                trail_ret = (pos['peak_price'] - cur_price) / pos['peak_price']
                if trail_ret >= 0.05:
                    pnl = pos['shares'] * (cur_price - pos['entry_price'])
                    cash += pos['shares'] * cur_price
                    trades.append({'code': code, 'entry_date': pos['entry_date'], 'exit_date': date,
                        'entry_price': pos['entry_price'], 'exit_price': cur_price,
                        'pnl': pnl / pos['shares'] / pos['entry_price'], 'reason': 'trailing',
                        'days_held': (date - pos['entry_date']).days})
                    del positions[code]
        
        # 轮换
        if i % hold_days == 0 and i > 0:
            for code in list(positions.keys()):
                row = day[day['code'] == code]
                if len(row) == 0: continue
                price = row['close'].values[0]
                pos = positions.pop(code)
                pnl = pos['shares'] * (price - pos['entry_price'])
                cash += pos['shares'] * price
                trades.append({'code': code, 'entry_date': pos['entry_date'], 'exit_date': date,
                    'entry_price': pos['entry_price'], 'exit_price': price,
                    'pnl': pnl / pos['shares'] / pos['entry_price'], 'reason': 'expire',
                    'days_held': (date - pos['entry_date']).days})
            
            avail = day[~day['code'].isin(positions.keys())].copy()
            if len(avail) > 0:
                avail['rank'] = avail[score_col].rank(ascending=False)
                top = avail.nsmallest(top_n, 'rank')
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
                                'entry_price': row['close'], 'shares': shares,
                                'entry_date': date, 'peak_price': row['close']}
                            cash -= size_per
        
        pos_value = sum(
            pos['shares'] * day[day['code']==code]['close'].values[0]
            for code, pos in positions.items()
            if len(day[day['code']==code]) > 0)
        equity_curve.append({'date': date, 'equity': cash + pos_value})
    
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
        wins = tdf[tdf['pnl']>0]; losses = tdf[tdf['pnl']<=0]
        win_rate = len(wins)/len(tdf)
        avg_win = wins['pnl'].mean() if len(wins)>0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses)>0 else 0
        reasons = tdf['reason'].value_counts().to_dict() if 'reason' in tdf.columns else {}
    else:
        win_rate = avg_win = avg_loss = 0; reasons = {}
    return {'name': name, 'annual_return': annual, 'max_drawdown': max_dd,
        'avg_drawdown': avg_dd, 'n_dd_events': len(dd_list),
        'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar,
        'win_rate': win_rate, 'n_trades': len(trades),
        'avg_win': avg_win, 'avg_loss': avg_loss, 'exit_reasons': reasons,
        'total_return': total}

# 测试集数据
test_df = df.loc[df['date'] > val_end, ['date','code','close','daily_ret','v3_score','lgb_score']].copy()

# ============================================================
# 5. 实验矩阵
# ============================================================
print("\n[5/5] 实验矩阵...")

exps = []

# V3公式 — 各种Top-N和持有期
for tn in [10, 15, 20]:
    for hd in [5, 7, 10]:
        exps.append({'name': f'V3公式 Top-{tn} {hd}d', 'score': 'v3_score', 'tn': tn, 'hd': hd, 'sl': None, 'timing': None})

# V3 + 止损
for sl in [-0.05, -0.08, -0.10]:
    exps.append({'name': f'V3公式 Top-15 7d SL{int(sl*100)}%', 'score': 'v3_score', 'tn': 15, 'hd': 7, 'sl': sl, 'timing': None})

# V3 + 择时
exps.append({'name': 'V3公式 Top-15 7d + 择时', 'score': 'v3_score', 'tn': 15, 'hd': 7, 'sl': None, 'timing': mkt_timing})
exps.append({'name': 'V3公式 Top-15 7d + 择时+SL-8%', 'score': 'v3_score', 'tn': 15, 'hd': 7, 'sl': -0.08, 'timing': mkt_timing})

# V4 LGB — 同样的配置
for tn in [10, 15, 20]:
    for hd in [5, 7, 10]:
        exps.append({'name': f'V4-LGB Top-{tn} {hd}d', 'score': 'lgb_score', 'tn': tn, 'hd': hd, 'sl': None, 'timing': None})

# V4 LGB + 止损
for sl in [-0.05, -0.08, -0.10]:
    exps.append({'name': f'V4-LGB Top-15 7d SL{int(sl*100)}%', 'score': 'lgb_score', 'tn': 15, 'hd': 7, 'sl': sl, 'timing': None})

# V4 LGB + 择时
exps.append({'name': 'V4-LGB Top-15 7d + 择时', 'score': 'lgb_score', 'tn': 15, 'hd': 7, 'sl': None, 'timing': mkt_timing})
exps.append({'name': 'V4-LGB Top-15 7d + 择时+SL-8%', 'score': 'lgb_score', 'tn': 15, 'hd': 7, 'sl': -0.08, 'timing': mkt_timing})

results = []
for i, e in enumerate(exps):
    t_start = time.time()
    eq, trades = proper_backtest(test_df, e['score'], e['tn'], e['hd'], e['sl'], e['timing'])
    m = metrics(eq, trades, e['name'])
    m['config'] = e
    results.append(m)
    dt = time.time()-t_start
    q = "🟢" if m['avg_drawdown'] > -0.03 else "🟡" if m['avg_drawdown'] > -0.05 else "🔴"
    tag = "V3" if "V3" in e['name'] else "V4"
    print(f"  [{i+1}/{len(exps)}] {e['name']:<48} 夏普{m['sharpe']:.2f} 年化{m['annual_return']:+.1%} 最大DD{m['max_drawdown']:.1%} 平均DD{m['avg_drawdown']:.2%} 胜率{m['win_rate']:.1%} {q} ({dt:.1f}s)")

# ============================================================
# 输出
# ============================================================
print("\n" + "=" * 130)
print("📊 V3 vs V4 完整对比")
print("=" * 130)

for r in results:
    r['efficiency'] = r['sharpe'] * max(0.01, 1 + r['avg_drawdown'])

results.sort(key=lambda x: x['efficiency'], reverse=True)

print(f"\n{'#':>3} {'策略':<50} {'年化':>7} {'最大DD':>8} {'平均DD':>8} {'夏普':>6} {'Sortino':>7} {'胜率':>6} {'交易':>6}")
print("-" * 115)
for i, r in enumerate(results):
    q = "🟢" if r['avg_drawdown'] > -0.03 else "🟡" if r['avg_drawdown'] > -0.05 else "🔴"
    tag = "🔵" if "V3" in r['name'] else "🟡"
    print(f"{i+1:>3} {tag} {r['name']:<48} {r['annual_return']:>+6.1%} {r['max_drawdown']:>7.1%} {r['avg_drawdown']:>7.2%} {r['sharpe']:>6.2f} {r['sortino']:>7.2f} {r['win_rate']:>5.1%} {r['n_trades']:>6} {q}")

# 核心对比：同Top-N同持有期
print("\n" + "=" * 130)
print("🔍 核心对比：V3公式 vs V4-LGB（同Top-N同持有期）")
print("=" * 130)

pairs = [
    ('Top-10 5d', 'Top-10 5d'),
    ('Top-15 5d', 'Top-15 5d'),
    ('Top-15 7d', 'Top-15 7d'),
    ('Top-15 10d', 'Top-15 10d'),
    ('Top-20 7d', 'Top-20 7d'),
]

v3_res = {r['name'].replace('V3公式 ',''): r for r in results if 'V3' in r['name'] and 'SL' not in r['name'] and '择时' not in r['name']}
v4_res = {r['name'].replace('V4-LGB ',''): r for r in results if 'V4' in r['name'] and 'SL' not in r['name'] and '择时' not in r['name']}

print(f"\n{'配置':<15} {'V3夏普':>8} {'V4夏普':>8} {'差值':>8} {'V3年化':>8} {'V4年化':>8} {'V3平均DD':>9} {'V4平均DD':>9}")
print("-" * 90)
for v3k, v4k in pairs:
    if v3k in v3_res and v4k in v4_res:
        v3 = v3_res[v3k]; v4 = v4_res[v4k]
        diff = v4['sharpe'] - v3['sharpe']
        print(f"{v3k:<15} {v3['sharpe']:>8.2f} {v4['sharpe']:>8.2f} {diff:>+8.2f} {v3['annual_return']:>+7.1%} {v4['annual_return']:>+7.1%} {v3['avg_drawdown']:>8.2%} {v4['avg_drawdown']:>8.2%}")

# 止损效果对比
print("\n" + "=" * 130)
print("🛡️ 止损效果对比（Top-15 7d）")
print("=" * 130)
sl_results = [r for r in results if 'Top-15 7d' in r['name']]
sl_results.sort(key=lambda x: (-('V3' in x['name']), x['name']))

print(f"\n{'策略':<50} {'夏普':>6} {'年化':>8} {'最大DD':>8} {'平均DD':>8} {'胜率':>6}")
print("-" * 90)
for r in sl_results:
    tag = "🔵" if "V3" in r['name'] else "🟡"
    print(f"{tag} {r['name']:<48} {r['sharpe']:>6.2f} {r['annual_return']:>+7.1%} {r['max_drawdown']:>7.1%} {r['avg_drawdown']:>7.2%} {r['win_rate']:>5.1%}")

# 退出原因
print("\n" + "=" * 130)
print("📋 退出原因分布（Top-15 7d 基线）")
print("=" * 130)
for r in sl_results:
    if 'SL' not in r['name'] and '择时' not in r['name']:
        tag = "🔵 V3" if "V3" in r['name'] else "🟡 V4"
        print(f"\n  {tag}: {r['name']}")
        print(f"    {r['exit_reasons']}")

# 保存
out = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'comparison': 'V3 formula vs V4-LGB',
    'n_experiments': len(results),
    'results': [{k:v for k,v in r.items() if k!='config'} for r in results],
}
with open(f'{OUT}/v3_vs_v4_comparison.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)

print(f"\n\n保存 → analysis/v3_vs_v4_comparison.json")
print(f"总耗时: {time.time()-t0:.1f}s")
print("=" * 90)
