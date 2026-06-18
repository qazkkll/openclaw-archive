#!/usr/bin/env python3
"""
蓝盾V3+V4 综合方案探索
方向：
1. V3过滤 + V4排序（V3作为安全网）
2. V3+V4加权融合（不同权重组合）
3. V3评分作为V4的额外特征
4. 动态切换（V3信号质量差时退回V3纯规则）
5. V3引导仓位管理
"""
import pandas as pd
import numpy as np
import json, time, warnings
warnings.filterwarnings('ignore')

DATA = '/home/hermes/.hermes/openclaw-archive/data/us/us_hist_sp500_10y.parquet'
OUT = '/home/hermes/.hermes/openclaw-archive/analysis'

print("=" * 90)
print("蓝盾V3+V4 综合方案探索")
print("=" * 90)

# ============================================================
# 1. 数据 + V3评分 + V4特征
# ============================================================
t0 = time.time()
print("\n[1/4] 数据+特征...")
df = pd.read_parquet(DATA).rename(columns={'sym': 'code'})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['code', 'date']).reset_index(drop=True)

# V3公式（向量化）
def calc_v3_score(g):
    c, h, l = g['close'], g['high'], g['low']
    ma5 = c.rolling(5).mean()
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    trend = ((c > ma5).astype(float)*10 + (ma5 > ma20).astype(float)*10 + (ma20 > ma60).astype(float)*10)
    ret5 = c.pct_change(5); ret20 = c.pct_change(20)
    momentum = ((ret5 > 0).astype(float)*12.5 + (ret20 > 0).astype(float)*12.5)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26; signal = macd.ewm(span=9, adjust=False).mean()
    macd_s = ((macd > 0).astype(float)*12.5 + (macd > signal).astype(float)*12.5)
    bias = (c - ma20) / ma20
    bias_s = ((bias > -0.05) & (bias < 0.10)).astype(float) * 10
    delta = c.diff(); gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan); rsi = 100 - 100/(1+rs)
    rsi_s = ((rsi > 30) & (rsi < 70)).astype(float) * 10
    high_52w = h.rolling(252).max(); low_52w = l.rolling(252).min()
    pos_52w = (c - low_52w) / (high_52w - low_52w + 1e-10)
    pos_s = (pos_52w > 0.7).astype(float) * 10
    return trend + momentum + macd_s + bias_s + rsi_s + pos_s

df['v3_score'] = df.groupby('code').apply(calc_v3_score).reset_index(level=0, drop=True)
print(f"  V3评分: 均值{df['v3_score'].mean():.1f}, 中位{df['v3_score'].median():.1f}")

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
print(f"  V4特征: {len(feat_cols)}维, 数据: {len(df):,}行 ({time.time()-t0:.1f}s)")

# ============================================================
# 2. 训练模型
# ============================================================
print("\n[2/4] 训练模型...")
t1 = time.time()

from lightgbm import LGBMRegressor

train_end = pd.Timestamp('2021-12-31')
val_end = pd.Timestamp('2023-12-31')

Xt = df.loc[df['date']<=train_end, feat_cols].values
yt = df.loc[df['date']<=train_end, 'target_5d'].values
Xv = df.loc[(df['date']>train_end)&(df['date']<=val_end), feat_cols].values
yv = df.loc[(df['date']>train_end)&(df['date']<=val_end), 'target_5d'].values

# 模型1：纯V4 LGB（基线）
print("  V4-LGB基线...", end=' ', flush=True)
lgb_base = LGBMRegressor(n_estimators=600, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, min_child_samples=20,
    random_state=42, n_jobs=-1, verbose=-1)
lgb_base.fit(Xt, yt, eval_set=[(Xv, yv)])
print("done")

# 模型2：V4 LGB + V3评分作为额外特征
print("  V4-LGB+V3特征...", end=' ', flush=True)
feat_cols_v3 = feat_cols + ['v3_score']
Xt_v3 = df.loc[df['date']<=train_end, feat_cols_v3].values
Xv_v3 = df.loc[(df['date']>train_end)&(df['date']<=val_end), feat_cols_v3].values
lgb_v3feat = LGBMRegressor(n_estimators=600, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, min_child_samples=20,
    random_state=42, n_jobs=-1, verbose=-1)
lgb_v3feat.fit(Xt_v3, yt, eval_set=[(Xv_v3, yv)])
print("done")

# 模型3：V3过滤后的V4 LGB（只在V3>=60的样本上训练）
print("  V4-LGB(V3过滤训练)...", end=' ', flush=True)
v3_filter_train = df['date']<=train_end
v3_mask_train = df.loc[v3_filter_train, 'v3_score'] >= 60
Xt_filt = df.loc[v3_filter_train][v3_mask_train.values][feat_cols].values
yt_filt = df.loc[v3_filter_train][v3_mask_train.values]['target_5d'].values
Xv_filt = df.loc[(df['date']>train_end)&(df['date']<=val_end)&(df['v3_score']>=60), feat_cols].values
yv_filt = df.loc[(df['date']>train_end)&(df['date']<=val_end)&(df['v3_score']>=60), 'target_5d'].values
lgb_v3filt = LGBMRegressor(n_estimators=600, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, min_child_samples=20,
    random_state=42, n_jobs=-1, verbose=-1)
lgb_v3filt.fit(Xt_filt, yt_filt, eval_set=[(Xv_filt, yv_filt)])
print(f"done (训练样本从{len(Xt):,}减至{len(Xt_filt):,})")

print(f"  训练完成 ({time.time()-t1:.0f}s)")

# ============================================================
# 3. 测试集预测
# ============================================================
print("\n[3/4] 测试集预测...")
test_mask = df['date'] > val_end
Xs = df.loc[test_mask, feat_cols].values
Xs_v3 = df.loc[test_mask, feat_cols_v3].values

# 纯V4
df.loc[test_mask, 'score_v4'] = lgb_base.predict(Xs)
# V4+V3特征
df.loc[test_mask, 'score_v4_v3feat'] = lgb_v3feat.predict(Xs_v3)
# V4(V3过滤训练) — 预测时也过滤
df.loc[test_mask, 'score_v4_v3filt'] = lgb_v3filt.predict(Xs)

# 市场择时
mkt = df.groupby('date').agg(
    mkt_breadth=('daily_ret', lambda x: (x>0).mean()),
).reset_index()
mkt['pos'] = 1.0
mkt.loc[mkt['mkt_breadth'] < 0.45, 'pos'] = 0.6
mkt.loc[mkt['mkt_breadth'] < 0.35, 'pos'] = 0.3
mkt_timing = mkt.set_index('date')['pos']

# ============================================================
# 4. 回测框架
# ============================================================
print("\n[4/4] 回测...")

def proper_backtest(test_df, score_col, top_n=15, hold_days=7, 
                    stop_loss=None, market_timing=None, v3_filter=None):
    """
    v3_filter: 如果设置，只选v3_score >= threshold的股票
    """
    dates = sorted(test_df['date'].unique())
    cash = 1.0; positions = {}; equity_curve = []; trades = []
    
    for i, date in enumerate(dates):
        day = test_df[test_df['date'] == date]
        if len(day) == 0:
            equity_curve.append({'date': date, 'equity': cash})
            continue
        
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
            if v3_filter is not None:
                avail = avail[avail['v3_score'] >= v3_filter]
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

test_df = df.loc[df['date'] > val_end, ['date','code','close','daily_ret','v3_score',
    'score_v4','score_v4_v3feat','score_v4_v3filt']].copy()

# ============================================================
# 实验矩阵
# ============================================================
exps = []

# === 基线 ===
exps.append({'name': 'V3纯公式 Top-15 7d', 'score': 'v3_score', 'tn': 15, 'hd': 7, 'sl': None, 'timing': None, 'v3f': None})
exps.append({'name': 'V4-LGB Top-15 7d', 'score': 'score_v4', 'tn': 15, 'hd': 7, 'sl': None, 'timing': None, 'v3f': None})

# === 方向1：V3过滤 + V4排序 ===
for thresh in [50, 60, 70, 80]:
    exps.append({'name': f'V3≥{thresh}+V4排序 Top-15 7d', 'score': 'score_v4', 'tn': 15, 'hd': 7, 'sl': None, 'timing': None, 'v3f': thresh})

# === 方向2：V3+V4加权融合 ===
for w_v3 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    w_v4 = 1.0 - w_v3
    col = f'blend_{w_v3}'
    test_df[col] = test_df['v3_score'] * w_v3 + test_df['score_v4'] * w_v4
    exps.append({'name': f'融合V3:{w_v3:.0%}+V4:{w_v4:.0%} Top-15 7d', 'score': col, 'tn': 15, 'hd': 7, 'sl': None, 'timing': None, 'v3f': None})

# === 方向3：V4+V3特征 ===
exps.append({'name': 'V4+V3特征 Top-15 7d', 'score': 'score_v4_v3feat', 'tn': 15, 'hd': 7, 'sl': None, 'timing': None, 'v3f': None})

# === 方向4：V3过滤训练的V4 ===
exps.append({'name': 'V4(V3过滤训练) Top-15 7d', 'score': 'score_v4_v3filt', 'tn': 15, 'hd': 7, 'sl': None, 'timing': None, 'v3f': None})

# === 方向5：最优组合 + 止损/择时 ===
# 测试几个有潜力的配置
best_combos = [
    ('V3≥60+V4排序', 'score_v4', 60),
    ('融合V3:30%+V4:70%', 'blend_0.3', None),
    ('融合V3:20%+V4:80%', 'blend_0.2', None),
    ('V4+V3特征', 'score_v4_v3feat', None),
]
for label, score, v3f in best_combos:
    exps.append({'name': f'{label} Top-15 7d SL-8%', 'score': score, 'tn': 15, 'hd': 7, 'sl': -0.08, 'timing': None, 'v3f': v3f})
    exps.append({'name': f'{label} Top-15 7d + 择时', 'score': score, 'tn': 15, 'hd': 7, 'sl': None, 'timing': mkt_timing, 'v3f': v3f})
    exps.append({'name': f'{label} Top-15 7d + 择时+SL-8%', 'score': score, 'tn': 15, 'hd': 7, 'sl': -0.08, 'timing': mkt_timing, 'v3f': v3f})

# === Top-N变化 ===
for tn in [10, 15, 20]:
    exps.append({'name': f'融合V3:30%+V4:70% Top-{tn} 7d', 'score': 'blend_0.3', 'tn': tn, 'hd': 7, 'sl': None, 'timing': None, 'v3f': None})

# === 持有期变化 ===
for hd in [5, 7, 10]:
    exps.append({'name': f'融合V3:30%+V4:70% Top-15 {hd}d', 'score': 'blend_0.3', 'tn': 15, 'hd': hd, 'sl': None, 'timing': None, 'v3f': None})

# 运行
results = []
for i, e in enumerate(exps):
    t_start = time.time()
    eq, trades = proper_backtest(test_df, e['score'], e['tn'], e['hd'], e['sl'], e['timing'], e['v3f'])
    m = metrics(eq, trades, e['name'])
    m['config'] = e
    results.append(m)
    dt = time.time()-t_start
    q = "🟢" if m['avg_drawdown'] > -0.03 else "🟡" if m['avg_drawdown'] > -0.05 else "🔴"
    print(f"  [{i+1}/{len(exps)}] {e['name']:<50} 夏普{m['sharpe']:.2f} 年化{m['annual_return']:+.1%} 平均DD{m['avg_drawdown']:.2%} {q} ({dt:.1f}s)")

# ============================================================
# 输出
# ============================================================
for r in results:
    r['efficiency'] = r['sharpe'] * max(0.01, 1 + r['avg_drawdown'])

results.sort(key=lambda x: x['efficiency'], reverse=True)

print("\n" + "=" * 130)
print("📊 综合方案排名（按效率排序 = 夏普 × (1+平均DD)）")
print("=" * 130)
print(f"\n{'#':>3} {'策略':<55} {'年化':>7} {'最大DD':>8} {'平均DD':>8} {'夏普':>6} {'Sortino':>7} {'胜率':>6}")
print("-" * 120)
for i, r in enumerate(results[:30]):
    q = "🟢" if r['avg_drawdown'] > -0.03 else "🟡" if r['avg_drawdown'] > -0.05 else "🔴"
    print(f"{i+1:>3} {r['name']:<55} {r['annual_return']:>+6.1%} {r['max_drawdown']:>7.1%} {r['avg_drawdown']:>7.2%} {r['sharpe']:>6.2f} {r['sortino']:>7.2f} {r['win_rate']:>5.1%} {q}")

# 方向分析
print("\n" + "=" * 130)
print("🔍 各方向最优方案对比")
print("=" * 130)

directions = {
    'V3纯公式': [r for r in results if r['name'] == 'V3纯公式 Top-15 7d'],
    'V4纯ML': [r for r in results if r['name'] == 'V4-LGB Top-15 7d'],
    'V3过滤+V4排序': [r for r in results if 'V3≥' in r['name'] and 'SL' not in r['name'] and '择时' not in r['name']],
    'V3+V4加权融合': [r for r in results if '融合' in r['name'] and 'Top-15 7d' in r['name'] and 'SL' not in r['name'] and '择时' not in r['name'] and 'Top-1' not in r['name'].split('Top-')[1][:2]],
    'V4+V3特征': [r for r in results if 'V3特征' in r['name'] and 'SL' not in r['name'] and '择时' not in r['name']],
    'V4(V3过滤训练)': [r for r in results if 'V3过滤训练' in r['name'] and 'SL' not in r['name'] and '择时' not in r['name']],
}

for direction, rs in directions.items():
    if rs:
        best = max(rs, key=lambda x: x['efficiency'])
        print(f"\n  {direction}: {best['name']}")
        print(f"    夏普: {best['sharpe']:.2f} | 年化: {best['annual_return']:+.1%} | 平均DD: {best['avg_drawdown']:.2%} | 效率: {best['efficiency']:.3f}")

# 最优+止损/择时
print("\n" + "=" * 130)
print("🏆 最优综合方案（含止损/择时）")
print("=" * 130)
print(f"\n{'#':>3} {'策略':<55} {'年化':>7} {'最大DD':>8} {'平均DD':>8} {'夏普':>6} {'效率':>6}")
print("-" * 100)
for i, r in enumerate(results[:15]):
    q = "🟢" if r['avg_drawdown'] > -0.03 else "🟡" if r['avg_drawdown'] > -0.05 else "🔴"
    print(f"{i+1:>3} {r['name']:<55} {r['annual_return']:>+6.1%} {r['max_drawdown']:>7.1%} {r['avg_drawdown']:>7.2%} {r['sharpe']:>6.2f} {r['efficiency']:>6.3f} {q}")

# 保存
out = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'n_experiments': len(results),
    'results': [{k:v for k,v in r.items() if k!='config'} for r in results],
}
with open(f'{OUT}/v3v4_combined_results.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)

print(f"\n保存 → analysis/v3v4_combined_results.json")
print(f"总耗时: {time.time()-t0:.1f}s")
print("=" * 90)
