#!/usr/bin/env python3
"""
蓝盾V4 深度研究 — 快速版（向量化回测）
模型层 + 决策层联合优化
"""
import pandas as pd
import numpy as np
import json, time, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

DATA = '/home/hermes/.hermes/openclaw-archive/data/us/us_hist_sp500_10y.parquet'
OUT = '/home/hermes/.hermes/openclaw-archive/analysis'

print("=" * 90)
print("蓝盾V4 深度研究 — 向量化回测版")
print("=" * 90)

# ============================================================
# 1. 数据 + 特征
# ============================================================
print("\n[1/4] 加载数据...")
t0 = time.time()

df = pd.read_parquet(DATA).rename(columns={'sym': 'code'})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['code', 'date']).reset_index(drop=True)

# 市场特征（全截面聚合）
mkt = df.groupby('date').agg(
    mkt_breadth=('close', lambda x: (x.pct_change(5) > 0).mean()),
    mkt_ret_5d=('close', lambda x: x.pct_change(5).median()),
    mkt_ret_20d=('close', lambda x: x.pct_change(20).median()),
    mkt_vol=('close', lambda x: x.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) if len(x) > 20 else np.nan),
    mkt_median_price=('close', 'median'),
).reset_index()
df = df.merge(mkt, on='date', how='left')

# 逐股特征
def feats(g):
    c, v, h, l, o = g['close'], g['volume'], g['high'], g['low'], g['open']
    for n in [1,2,3,5,10,20,60]: g[f'ret_{n}'] = c.pct_change(n)
    for n in [5,10,20,60]: g[f'vol_{n}'] = c.pct_change().rolling(n).std() * np.sqrt(252)
    
    d = c.diff(); up = d.clip(lower=0); dn = (-d).clip(lower=0)
    for w in [14]:
        rs = up.rolling(w).mean() / dn.rolling(w).mean().replace(0, np.nan)
        g[f'rsi_{w}'] = 100 - 100/(1+rs)
    
    ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
    g['macd_hist'] = (ema12-ema26) - (ema12-ema26).ewm(span=9).mean()
    
    sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
    g['bb_width'] = 4*std20/sma20.replace(0, np.nan)
    g['bb_pos'] = (c - (sma20-2*std20)) / (4*std20).replace(0, np.nan)
    
    vs20 = v.rolling(20).mean(); vs60 = v.rolling(60).mean()
    g['vol_ratio'] = v / vs20.replace(0, np.nan)
    g['vol_trend'] = vs20 / vs60.replace(0, np.nan)
    
    g['hl_range'] = (h-l)/c
    g['body_ratio'] = abs(c-o)/(h-l).replace(0, np.nan)
    
    for n in [5,10,20,50]: g[f'bias_{n}'] = (c-c.rolling(n).mean())/c.rolling(n).mean().replace(0, np.nan)
    
    g['high_52w'] = h.rolling(250).max()
    g['low_52w'] = l.rolling(250).min()
    g['dist_52w_high'] = c/g['high_52w'] - 1
    g['dist_52w_low'] = c/g['low_52w'] - 1
    
    for n in [5,10,20,60]: g[f'mom_{n}'] = c/c.shift(n) - 1
    
    tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    g['atr_pct'] = tr.rolling(14).mean() / c
    
    g['ret_skew'] = c.pct_change().rolling(20).skew()
    g['ret_ratio_5_20'] = c.pct_change(5) / c.pct_change(20).replace(0, np.nan)
    
    # 截面排名特征（当日）
    g['rank_ret_5'] = g['ret_5'].rank(pct=True)
    g['rank_vol_20'] = g['vol_20'].rank(pct=True)
    g['rank_rsi'] = g['rsi_14'].rank(pct=True)
    g['rank_bias_20'] = g['bias_20'].rank(pct=True)
    
    return g

groups = []
for code, grp in df.groupby('code'):
    groups.append(feats(grp))
df = pd.concat(groups, ignore_index=True)

# 目标
df['target_5d'] = df.groupby('code')['close'].transform(lambda x: x.shift(-5)/x - 1)
df['daily_ret'] = df.groupby('code')['close'].pct_change()

# 特征列
skip = {'date','code','open','high','low','close','volume','target_5d','daily_ret',
        'high_52w','low_52w','mkt_median_price','mkt_vol'}
feat_cols = [c for c in df.columns if c not in skip]
df = df.replace([np.inf,-np.inf], np.nan)

# 只drop关键特征NaN
core = [c for c in feat_cols if not c.startswith('dist_52w') and not c.startswith('ret_skew') 
        and not c.startswith('ret_ratio')]
df = df.dropna(subset=core + ['target_5d','daily_ret']).sort_values('date').reset_index(drop=True)

print(f"  特征: {len(feat_cols)}维, 数据: {len(df):,}行 ({time.time()-t0:.1f}s)")

# ============================================================
# 2. 训练模型
# ============================================================
print(f"\n[2/4] 训练模型...")
t0 = time.time()

train_end = pd.Timestamp('2021-12-31')
val_end = pd.Timestamp('2023-12-31')
tm = df['date'] <= train_end
vm = (df['date'] > train_end) & (df['date'] <= val_end)
testm = df['date'] > val_end

Xt = df.loc[tm, feat_cols].values; yt = df.loc[tm, 'target_5d'].values
Xv = df.loc[vm, feat_cols].values; yv = df.loc[vm, 'target_5d'].values
Xs = df.loc[testm, feat_cols].values

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

models = {}
print("  XGBoost...")
m = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10,
    random_state=42, n_jobs=-1, verbosity=0)
m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
models['XGB'] = m

print("  LightGBM...")
m = LGBMRegressor(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
    random_state=42, n_jobs=-1, verbose=-1)
m.fit(Xt, yt, eval_set=[(Xv, yv)])
models['LGB'] = m

print("  CatBoost...")
m = CatBoostRegressor(iterations=600, depth=6, learning_rate=0.03, l2_leaf_reg=3,
    random_seed=42, verbose=0)
m.fit(Xt, yt, eval_set=(Xv, yv))
models['Cat'] = m

print(f"  训练完成 ({time.time()-t0:.1f}s)")

# 预测
preds = {n: m.predict(Xs) for n, m in models.items()}

# ============================================================
# 3. 向量化 Top-N 回测
# ============================================================
print(f"\n[3/4] 向量化回测...")

test_df = df.loc[testm, ['date','code','close','daily_ret']].copy()
test_df['pred_xgb'] = preds['XGB']
test_df['pred_lgb'] = preds['LGB']
test_df['pred_cat'] = preds['Cat']
test_df['pred_avg'] = (preds['XGB'] + preds['LGB'] + preds['Cat']) / 3
test_df['pred_weighted'] = preds['XGB']*0.4 + preds['LGB']*0.3 + preds['Cat']*0.3

# 截面排名
test_df['rank_xgb'] = test_df.groupby('date')['pred_xgb'].rank(pct=True)
test_df['rank_lgb'] = test_df.groupby('date')['pred_lgb'].rank(pct=True)
test_df['rank_cat'] = test_df.groupby('date')['pred_cat'].rank(pct=True)
test_df['rank_avg'] = test_df.groupby('date')['pred_avg'].rank(pct=True)
test_df['rank_weighted'] = test_df.groupby('date')['pred_weighted'].rank(pct=True)

# 市场择时信号
mkt_signal = test_df.groupby('date').agg(
    breadth=('daily_ret', lambda x: (x > 0).mean()),
    median_ret=('daily_ret', 'median'),
).reset_index()
mkt_signal['sma20'] = mkt_signal['median_ret'].rolling(20).mean()
mkt_signal['market_pos'] = 1.0
mkt_signal.loc[mkt_signal['breadth'] < 0.45, 'market_pos'] = 0.6
mkt_signal.loc[mkt_signal['breadth'] < 0.35, 'market_pos'] = 0.3
mkt_signal = mkt_signal.set_index('date')['market_pos']

def fast_topn_backtest(test_df, score_col, top_n=15, hold_days=7, 
                       stop_loss=None, trailing_stop=None, timing=None,
                       min_hold=None):
    """向量化Top-N回测"""
    dates = sorted(test_df['date'].unique())
    n_days = len(dates)
    
    # 预计算：每日排名
    # 每只股票的持仓期收益
    # 简化模型：每天选Top-N，持有hold_days后卖出
    # 这意味着在第t天买入的股票，在第t+hold_days天的累计收益
    
    equity = 1.0
    equity_curve = []
    all_trades = []
    n_active = 0
    
    for i, date in enumerate(dates):
        day = test_df[test_df['date'] == date].copy()
        
        # 获取市场择时仓位
        pos_size = 1.0
        if timing is not None and date in timing.index:
            pos_size = timing.loc[date]
        
        # 选Top-N
        ranked = day.nlargest(top_n, score_col)
        
        # 计算这些股票的持仓期收益
        for _, row in ranked.iterrows():
            code = row['code']
            ret_1d = row['daily_ret']
            
            # 简化：用单日收益近似（真实持有期需要更复杂逻辑）
            # 但我们有target_5d，可以估算持有期
            # 这里用逐日模拟：
            trade = {
                'code': code,
                'entry_date': date,
                'entry_idx': i,
                'score': row[score_col],
                'ret_1d': ret_1d,
            }
            all_trades.append(trade)
        
        # 用简单方式计算组合日收益：Top-N股票的等权日收益
        if len(ranked) > 0:
            port_ret = ranked['daily_ret'].mean() * pos_size
            equity *= (1 + port_ret)
        
        equity_curve.append({'date': date, 'equity': equity})
    
    return equity_curve, all_trades

def fast_vectorized_backtest(test_df, score_col, top_n=15, hold_days=7,
                             stop_loss=None, timing=None):
    """
    更精确的回测：每天选Top-N，模拟持有hold_days
    用shift后的收益计算
    """
    dates = sorted(test_df['date'].unique())
    n_dates = len(dates)
    
    # 预计算每天每只股票的持有期收益
    test_df = test_df.copy()
    test_df['fwd_cumret'] = test_df.groupby('code')['daily_ret'].transform(
        lambda x: (1 + x).rolling(hold_days).apply(lambda y: y.prod() - 1, raw=True).shift(-hold_days)
    )
    
    # 每天选Top-N，记录其持有期收益
    equity = 1.0
    equity_curve = []
    trade_results = []
    peak = 1.0
    
    for i, date in enumerate(dates):
        day = test_df[test_df['date'] == date].dropna(subset=[score_col, 'fwd_cumret'])
        
        pos_size = 1.0
        if timing is not None and date in timing.index:
            pos_size = float(timing.loc[date])
        
        # 选Top-N
        if len(day) >= top_n:
            top = day.nlargest(top_n, score_col)
        else:
            top = day
        
        # 止损检查（简化：如果持有期收益 < stop_loss，截断）
        if stop_loss:
            top_ret = top['fwd_cumret'].clip(lower=stop_loss)
        else:
            top_ret = top['fwd_cumret']
        
        # 组合收益（等权）
        port_ret = top_ret.mean() * pos_size
        
        equity *= (1 + port_ret)
        peak = max(peak, equity)
        dd = equity / peak - 1
        
        trade_results.extend([
            {'date': date, 'code': row['code'], 'ret': row['fwd_cumret'], 
             'score': row[score_col], 'held': hold_days}
            for _, row in top.iterrows()
        ])
        
        equity_curve.append({
            'date': date, 'equity': equity, 'dd': dd,
            'n_stocks': len(top), 'pos_size': pos_size
        })
    
    return equity_curve, trade_results

def calc_full_metrics(eq_list, trades, name):
    """全面指标"""
    eq = pd.DataFrame(eq_list)
    eq['ret'] = eq['equity'].pct_change()
    
    days = (eq['date'].max() - eq['date'].min()).days
    total_ret = eq['equity'].iloc[-1] / eq['equity'].iloc[0] - 1
    annual_ret = (1+total_ret)**(365/max(days,1)) - 1
    
    max_dd = eq['dd'].min()
    
    # 平均回撤
    in_dd = False; dd_list = []; dd_start = 0
    for i in range(len(eq)):
        if eq['dd'].iloc[i] < -0.001 and not in_dd:
            in_dd = True; dd_start = i
        elif eq['dd'].iloc[i] >= -0.001 and in_dd:
            in_dd = False
            dd_list.append(eq['dd'].iloc[dd_start:i].min())
    if in_dd:
        dd_list.append(eq['dd'].iloc[dd_start:].min())
    avg_dd = np.mean(dd_list) if dd_list else 0
    
    daily_ret = eq['ret'].dropna()
    sharpe = daily_ret.mean()/daily_ret.std()*np.sqrt(252) if daily_ret.std()>0 else 0
    sortino = daily_ret.mean()/daily_ret[daily_ret<0].std()*np.sqrt(252) if (daily_ret<0).sum()>0 else 0
    calmar = annual_ret/abs(max_dd) if max_dd != 0 else 0
    
    # 交易统计
    if trades:
        tdf = pd.DataFrame(trades)
        wins = tdf[tdf['ret'] > 0]
        losses = tdf[tdf['ret'] <= 0]
        win_rate = len(wins)/len(tdf)
        avg_win = wins['ret'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['ret'].mean()) if len(losses) > 0 else 0.001
        
        # 每笔交易的持有期最大回撤（用ret近似）
        trade_rets = tdf['ret'].values
        avg_trade_ret = np.mean(trade_rets)
    else:
        win_rate = avg_win = avg_loss = avg_trade_ret = 0
    
    return {
        'name': name, 'annual_return': annual_ret, 'max_drawdown': max_dd,
        'avg_drawdown': avg_dd, 'n_dd_events': len(dd_list),
        'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar,
        'win_rate': win_rate, 'n_trades': len(trades),
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'avg_trade_ret': avg_trade_ret if trades else 0,
    }

# ============================================================
# 4. 实验矩阵
# ============================================================
print(f"\n[4/4] 运行实验矩阵...")

experiments = []
score_cols = {
    'XGB': 'pred_xgb', 'LGB': 'pred_lgb', 'Cat': 'pred_cat',
    '等权集成': 'pred_avg', '加权集成': 'pred_weighted'
}

# 单模型
for name, col in score_cols.items():
    for top_n in [10, 15, 20]:
        for hold in [5, 7, 10]:
            experiments.append({'name': f'{name} Top-{top_n} {hold}d', 'score': col,
                              'top_n': top_n, 'hold': hold, 'sl': None, 'timing': None})

# 止损
for sl in [-0.05, -0.08, -0.10]:
    experiments.append({'name': f'加权集成 Top-15 7d SL{int(sl*100)}%', 'score': 'pred_weighted',
                      'top_n': 15, 'hold': 7, 'sl': sl, 'timing': None})

# 市场择时
for name, col in [('加权集成', 'pred_weighted'), ('等权集成', 'pred_avg')]:
    experiments.append({'name': f'{name} Top-15 7d + 择时', 'score': col,
                      'top_n': 15, 'hold': 7, 'sl': None, 'timing': mkt_signal})

# 择时+止损
experiments.append({'name': '加权集成 Top-15 7d + 择时+SL-8%', 'score': 'pred_weighted',
                  'top_n': 15, 'hold': 7, 'sl': -0.08, 'timing': mkt_signal})

results = []
for i, exp in enumerate(experiments):
    t1 = time.time()
    try:
        eq, trades = fast_vectorized_backtest(
            test_df, exp['score'], exp['top_n'], exp['hold'],
            exp['sl'], exp['timing']
        )
        m = calc_full_metrics(eq, trades, exp['name'])
        m['config'] = exp
        results.append(m)
        dt = time.time()-t1
        print(f"  [{i+1}/{len(experiments)}] {exp['name']:<45} 夏普{m['sharpe']:.2f} 年化{m['annual_return']:+.1%} 最大DD{m['max_drawdown']:.1%} 平均DD{m['avg_drawdown']:.2%} ({dt:.1f}s)")
    except Exception as e:
        print(f"  [{i+1}/{len(experiments)}] {exp['name']:<45} ❌ {e}")

# ============================================================
# 5. 结果分析
# ============================================================
print("\n" + "=" * 120)
print("📊 全部结果（按夏普排序）")
print("=" * 120)

results.sort(key=lambda x: x['sharpe'], reverse=True)
print(f"\n{'#':>3} {'策略':<48} {'年化':>7} {'最大DD':>8} {'平均DD':>8} {'夏普':>6} {'Sortino':>7} {'胜率':>6}")
print("-" * 105)
for i, r in enumerate(results):
    q = "🟢" if r['avg_drawdown'] > -0.02 else "🟡" if r['avg_drawdown'] > -0.04 else "🔴"
    print(f"{i+1:>3} {r['name']:<48} {r['annual_return']:>+6.1%} {r['max_drawdown']:>7.1%} {r['avg_drawdown']:>7.2%} {r['sharpe']:>6.2f} {r['sortino']:>7.2f} {r['win_rate']:>5.1%} {q}")

# ============================================================
# 6. 深度分析
# ============================================================
print("\n" + "=" * 120)
print("🔍 深度分析")
print("=" * 120)

# A. 模型层：平均回撤对比
print("\n📊 A. 模型层 — 平均回撤（选股质量）")
print("-" * 80)
baseline = [r for r in results if '7d' in r['name'] and '择时' not in r['name'] and 'SL' not in r['name'] and 'Top-15' in r['name']]
for r in sorted(baseline, key=lambda x: x['avg_drawdown'], reverse=True):
    q = "🟢" if r['avg_drawdown'] > -0.02 else "🟡" if r['avg_drawdown'] > -0.04 else "🔴"
    print(f"  {q} {r['name']:<35} 平均DD: {r['avg_drawdown']:.3%} | 最大DD: {r['max_drawdown']:.1%} | 夏普: {r['sharpe']:.2f}")

# B. Top-N影响
print("\n📊 B. Top-N影响（加权集成7d）")
print("-" * 80)
topn = [r for r in results if '加权集成' in r['name'] and '7d' in r['name'] and '择时' not in r['name'] and 'SL' not in r['name']]
for r in sorted(topn, key=lambda x: x['sharpe'], reverse=True):
    print(f"  {r['name']:<35} 夏普: {r['sharpe']:.2f} | 平均DD: {r['avg_drawdown']:.3%} | 胜率: {r['win_rate']:.1%}")

# C. 持有期影响
print("\n📊 C. 持有期影响（加权集成Top-15）")
print("-" * 80)
hold_r = [r for r in results if '加权集成' in r['name'] and 'Top-15' in r['name'] and '择时' not in r['name'] and 'SL' not in r['name']]
for r in sorted(hold_r, key=lambda x: x['sharpe'], reverse=True):
    print(f"  {r['name']:<35} 夏普: {r['sharpe']:.2f} | 平均DD: {r['avg_drawdown']:.3%} | 年化: {r['annual_return']:+.1%}")

# D. 决策层效果
print("\n📊 D. 决策层 — 止损/择时效果")
print("-" * 80)
decision = [r for r in results if 'SL' in r['name'] or '择时' in r['name']]
for r in sorted(decision, key=lambda x: x['sharpe'], reverse=True):
    print(f"  {r['name']:<45} 夏普: {r['sharpe']:.2f} | 最大DD: {r['max_drawdown']:.1%} | 平均DD: {r['avg_drawdown']:.2%}")

# E. 最优推荐
print("\n" + "=" * 120)
print("🏆 最优方案推荐")
print("=" * 120)

# 综合评分 = 夏普 × (1 + 平均回撤) — 平均回撤越小(越接近0)，效率越高
for r in results:
    r['efficiency'] = r['sharpe'] * max(0.01, 1 + r['avg_drawdown'])

by_eff = sorted(results, key=lambda x: x['efficiency'], reverse=True)
for i, r in enumerate(by_eff[:5]):
    print(f"\n  #{i+1} {r['name']}")
    print(f"      年化: {r['annual_return']:+.1%} | 夏普: {r['sharpe']:.2f} | Sortino: {r['sortino']:.2f}")
    print(f"      最大DD: {r['max_drawdown']:.1%} | 平均DD: {r['avg_drawdown']:.3%} ({r['n_dd_events']}次)")
    print(f"      胜率: {r['win_rate']:.1%} | 交易数: {r['n_trades']} | 平均持仓收益: {r['avg_trade_ret']:.3%}")

# 保存
output = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'n_experiments': len(results),
    'results': [{k:v for k,v in r.items() if k != 'config'} for r in results],
    'top_5_efficiency': [{k:v for k,v in r.items() if k != 'config'} for r in by_eff[:5]],
    'top_5_sharpe': [{k:v for k,v in r.items() if k != 'config'} for r in results[:5]],
}
with open(f'{OUT}/v4_deep_research_v2.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n\n结果已保存 → analysis/v4_deep_research_v2.json")
print(f"总耗时: {time.time()-t0:.1f}s")
print("=" * 90)
