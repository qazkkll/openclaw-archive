#!/usr/bin/env python3
"""
蓝盾V4 — 10年回测 + 交易成本 + 特征重要性 + 选股重叠度
"""
import pandas as pd
import numpy as np
import json, time, warnings
warnings.filterwarnings('ignore')

DATA = '/home/hermes/.hermes/openclaw-archive/data/us/us_hist_sp500_10y.parquet'
OUT = '/home/hermes/.hermes/openclaw-archive/analysis'

print("=" * 90)
print("蓝盾V4 — 10年完整验证")
print("=" * 90)

# ============================================================
# 1. 数据+特征
# ============================================================
t0 = time.time()
print("\n[1/5] 数据...")
df = pd.read_parquet(DATA).rename(columns={'sym': 'code'})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['code', 'date']).reset_index(drop=True)

mkt = df.groupby('date').agg(
    mkt_breadth=('close', lambda x: (x.pct_change(5) > 0).mean()),
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
        'high_52w','low_52w','mkt_breadth'}
feat_cols = [c for c in df.columns if c not in skip]
df = df.replace([np.inf,-np.inf], np.nan)
core = [c for c in feat_cols if not c.startswith('dist_52w') and not c.startswith('ret_skew')
        and not c.startswith('ret_ratio')]
df = df.dropna(subset=core + ['target_5d','daily_ret']).sort_values('date').reset_index(drop=True)
print(f"  特征: {len(feat_cols)}维, 数据: {len(df):,}行 ({time.time()-t0:.1f}s)")
print(f"  时间范围: {df['date'].min().date()} ~ {df['date'].max().date()}")

# ============================================================
# 2. Walk-Forward训练（每年重训一次）
# ============================================================
print("\n[2/5] Walk-Forward训练...")

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# 每年重训：用之前所有数据训练，预测未来1年
years = sorted(df['date'].dt.year.unique())
print(f"  年份: {years[0]}~{years[-1]}")

all_preds = []
feature_importances = {'XGB': {}, 'LGB': {}, 'Cat': {}}

for i, year in enumerate(years):
    # 训练数据：year-1及之前，至少3年
    train_cutoff = pd.Timestamp(f'{year}-01-01')
    train_start = pd.Timestamp(f'{max(year-5, years[0])}-01-01')
    val_start = pd.Timestamp(f'{year-1}-01-01')
    
    train_mask = (df['date'] >= train_start) & (df['date'] < val_start)
    val_mask = (df['date'] >= val_start) & (df['date'] < train_cutoff)
    test_mask = (df['date'] >= train_cutoff) & (df['date'] < pd.Timestamp(f'{year+1}-01-01'))
    
    n_train = train_mask.sum()
    n_val = val_mask.sum()
    n_test = test_mask.sum()
    
    if n_train < 10000 or n_val < 1000 or n_test < 1000:
        print(f"  {year}: 数据不足 (train={n_train}, val={n_val}, test={n_test}), 跳过")
        continue
    
    Xt = df.loc[train_mask, feat_cols].values
    yt = df.loc[train_mask, 'target_5d'].values
    Xv = df.loc[val_mask, feat_cols].values
    yv = df.loc[val_mask, 'target_5d'].values
    Xs = df.loc[test_mask, feat_cols].values
    
    print(f"  {year}: train={n_train:,} val={n_val:,} test={n_test:,}...", end=' ', flush=True)
    t1 = time.time()
    
    # 训练3个模型
    preds_this_year = {}
    
    # XGBoost
    m_xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=0.1, min_child_weight=10, random_state=42, n_jobs=-1, verbosity=0)
    m_xgb.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
    preds_this_year['XGB'] = m_xgb.predict(Xs)
    fi = dict(zip(feat_cols, m_xgb.feature_importances_))
    for k in ['ret_5', 'vol_20', 'rsi_14', 'macd_hist', 'bb_pos', 'vol_ratio', 'mom_5', 'mom_20', 'bias_20', 'atr_pct']:
        if k in fi:
            feature_importances['XGB'][k] = feature_importances['XGB'].get(k, []) + [fi[k]]
    
    # LightGBM
    m_lgb = LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=0.1, min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1)
    m_lgb.fit(Xt, yt, eval_set=[(Xv, yv)])
    preds_this_year['LGB'] = m_lgb.predict(Xs)
    fi = dict(zip(feat_cols, m_lgb.feature_importances_))
    for k in ['ret_5', 'vol_20', 'rsi_14', 'macd_hist', 'bb_pos', 'vol_ratio', 'mom_5', 'mom_20', 'bias_20', 'atr_pct']:
        if k in fi:
            feature_importances['LGB'][k] = feature_importances['LGB'].get(k, []) + [fi[k]]
    
    # CatBoost
    m_cat = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.03, l2_leaf_reg=3,
        random_seed=42, verbose=0)
    m_cat.fit(Xt, yt, eval_set=(Xv, yv))
    preds_this_year['Cat'] = m_cat.predict(Xs)
    fi = dict(zip(feat_cols, m_cat.feature_importances_))
    for k in ['ret_5', 'vol_20', 'rsi_14', 'macd_hist', 'bb_pos', 'vol_ratio', 'mom_5', 'mom_20', 'bias_20', 'atr_pct']:
        if k in fi:
            feature_importances['Cat'][k] = feature_importances['Cat'].get(k, []) + [fi[k]]
    
    preds_this_year['ENS'] = preds_this_year['XGB']*0.4 + preds_this_year['LGB']*0.3 + preds_this_year['Cat']*0.3
    
    # 保存预测
    test_idx = df[test_mask].index
    for model_name, pred in preds_this_year.items():
        df.loc[test_idx, f'score_{model_name}'] = pred
    
    print(f"done ({time.time()-t1:.0f}s)")

# ============================================================
# 3. 回测（10年）
# ============================================================
print(f"\n[3/5] 10年回测...")

test_df = df.dropna(subset=['score_XGB','score_LGB','score_Cat','score_ENS']).copy()
print(f"  有效测试数据: {len(test_df):,}行, {test_df['date'].min().date()}~{test_df['date'].max().date()}")
print(f"  测试年数: {test_df['date'].dt.year.nunique()}")

def proper_backtest(test_df, score_col, top_n=15, hold_days=7, stop_loss=None):
    dates = sorted(test_df['date'].unique())
    cash = 1.0; positions = {}; equity_curve = []; trades = []; peak = 1.0
    
    for i, date in enumerate(dates):
        day = test_df[test_df['date'] == date]
        if len(day) == 0:
            equity_curve.append({'date': date, 'equity': cash, 'dd': cash/peak-1})
            continue
        
        # 持仓PnL
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
                    'pnl': ret, 'reason': 'stop_loss',
                    'days_held': (date - pos['entry_date']).days})
                del positions[code]
        
        # 轮换
        if i % hold_days == 0 and i > 0:
            for code in list(positions.keys()):
                row = day[day['code'] == code]
                if len(row) == 0: continue
                pos = positions.pop(code)
                price = row['close'].values[0]
                cash += pos['shares'] * price
                trades.append({'code': code, 'entry_date': pos['entry_date'], 'exit_date': date,
                    'entry_price': pos['entry_price'], 'exit_price': price,
                    'pnl': price/pos['entry_price']-1, 'reason': 'expire',
                    'days_held': (date - pos['entry_date']).days})
            
            avail = day[~day['code'].isin(positions.keys())].copy()
            if len(avail) > 0:
                top = avail.nlargest(top_n, score_col)
                size_per = cash / len(top)
                for _, row in top.iterrows():
                    if row['close'] > 0:
                        shares = size_per / row['close']
                        positions[row['code']] = {'entry_price': row['close'], 'shares': shares,
                            'entry_date': date, 'peak_price': row['close']}
                        cash -= size_per
        
        pos_val = sum(pos['shares'] * day[day['code']==c]['close'].values[0]
                      for c, pos in positions.items() if len(day[day['code']==c])>0)
        eq = cash + pos_val
        peak = max(peak, eq)
        equity_curve.append({'date': date, 'equity': eq, 'dd': eq/peak-1})
    
    return equity_curve, trades

def calc_metrics(eq_list, trades, name, cost_per_trade=0):
    eq = pd.DataFrame(eq_list)
    eq['ret'] = eq['equity'].pct_change()
    
    # 交易成本
    if cost_per_trade > 0 and trades:
        n_trades = len(trades)
        total_cost = n_trades * cost_per_trade
        eq['equity_adj'] = eq['equity'] * (1 - total_cost)
        eq['ret'] = eq['equity_adj'].pct_change()
    
    days = (eq['date'].max()-eq['date'].min()).days
    total = eq['equity'].iloc[-1]/eq['equity'].iloc[0]-1
    annual = (1+total)**(365/max(days,1))-1
    
    max_dd = eq['dd'].min()
    in_dd = False; dd_list = []; s = 0
    for j in range(len(eq)):
        if eq['dd'].iloc[j] < -0.001 and not in_dd:
            in_dd = True; s = j
        elif eq['dd'].iloc[j] >= -0.001 and in_dd:
            in_dd = False; dd_list.append(eq['dd'].iloc[s:j].min())
    if in_dd: dd_list.append(eq['dd'].iloc[s:].min())
    avg_dd = np.mean(dd_list) if dd_list else 0
    
    dr = eq['ret'].dropna()
    sharpe = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    sortino = dr.mean()/dr[dr<0].std()*np.sqrt(252) if (dr<0).sum()>0 else 0
    
    if trades:
        tdf = pd.DataFrame(trades)
        win_rate = len(tdf[tdf['pnl']>0])/len(tdf)
        avg_win = tdf[tdf['pnl']>0]['pnl'].mean() if (tdf['pnl']>0).any() else 0
        avg_loss = abs(tdf[tdf['pnl']<=0]['pnl'].mean()) if (tdf['pnl']<=0).any() else 0
    else:
        win_rate = avg_win = avg_loss = 0
    
    return {
        'name': name, 'annual_return': annual, 'max_drawdown': max_dd,
        'avg_drawdown': avg_dd, 'n_dd_events': len(dd_list),
        'sharpe': sharpe, 'sortino': sortino,
        'win_rate': win_rate, 'n_trades': len(trades),
        'avg_win': avg_win, 'avg_loss': avg_loss,
    }

# 核心实验（只跑最关键的配置）
exps = [
    # 模型对比
    ('Cat Top-15 10d', 'score_Cat', 15, 10, None),
    ('Cat Top-20 10d', 'score_Cat', 20, 10, None),
    ('Cat Top-15 7d', 'score_Cat', 15, 7, None),
    ('LGB Top-10 10d', 'score_LGB', 10, 10, None),
    ('LGB Top-15 5d', 'score_LGB', 15, 5, None),
    ('XGB Top-15 10d', 'score_XGB', 15, 10, None),
    ('集成 Top-15 10d', 'score_ENS', 15, 10, None),
    ('集成 Top-15 7d', 'score_ENS', 15, 7, None),
    # 决策层
    ('集成 Top-15 7d SL-8%', 'score_ENS', 15, 7, -0.08),
    ('集成 Top-15 7d SL-10%', 'score_ENS', 15, 7, -0.10),
    ('Cat Top-15 10d SL-10%', 'score_Cat', 15, 10, -0.10),
]

results = []
for name, col, tn, hd, sl in exps:
    eq, trades = proper_backtest(test_df, col, tn, hd, sl)
    m = calc_metrics(eq, trades, name)
    m['equity_curve'] = eq  # 保留用于年度分析
    results.append(m)
    print(f"  {name:<35} 夏普{m['sharpe']:.2f} 年化{m['annual_return']:+.1%} 最大DD{m['max_drawdown']:.1%} 平均DD{m['avg_drawdown']:.2%} 交易{m['n_trades']}")

# ============================================================
# 4. 交易成本敏感性
# ============================================================
print(f"\n[4/5] 交易成本敏感性...")

# 对前3个策略测试不同成本
cost_configs = [0, 0.0005, 0.001, 0.002, 0.005]  # 0%, 0.05%, 0.1%, 0.2%, 0.5%
cost_results = []

for name, col, tn, hd, sl in exps[:5]:
    eq, trades = proper_backtest(test_df, col, tn, hd, sl)
    for cost in cost_configs:
        m = calc_metrics(eq, trades, f"{name} cost={cost:.2%}", cost_per_trade=cost)
        cost_results.append({
            'strategy': name, 'cost_per_trade': cost,
            'sharpe': m['sharpe'], 'annual_return': m['annual_return'],
            'max_drawdown': m['max_drawdown'], 'avg_drawdown': m['avg_drawdown'],
        })

# 打印成本敏感性表
print(f"\n  {'策略':<30} {'成本/笔':>8} {'夏普':>6} {'年化':>8} {'最大DD':>8}")
print("  " + "-" * 65)
for cr in cost_results:
    print(f"  {cr['strategy']:<30} {cr['cost_per_trade']:>7.2%} {cr['sharpe']:>6.2f} {cr['annual_return']:>+7.1%} {cr['max_drawdown']:>7.1%}")

# ============================================================
# 5. 特征重要性 + 选股重叠度
# ============================================================
print(f"\n[5/5] 特征重要性 + 选股重叠度...")

# 特征重要性
print("\n📊 CatBoost 特征重要性（跨年平均）")
cat_imp = feature_importances['Cat']
cat_sorted = sorted(cat_imp.items(), key=lambda x: np.mean(x[1]), reverse=True)
for feat, vals in cat_sorted[:10]:
    print(f"  {feat:<20} {np.mean(vals):>8.1f}")

print("\n📊 模型对比特征重要性")
for model in ['XGB', 'LGB', 'Cat']:
    imp = feature_importances[model]
    top3 = sorted(imp.items(), key=lambda x: np.mean(x[1]), reverse=True)[:5]
    feats_str = ', '.join([f"{f}({np.mean(v):.0f})" for f, v in top3])
    print(f"  {model}: {feats_str}")

# 选股重叠度
print("\n📊 模型选股重叠度（测试期平均）")
test_dates = sorted(test_df['date'].unique())
overlaps = {'XGB_LGB': [], 'XGB_Cat': [], 'LGB_Cat': [], 'All_3': []}

for date in test_dates:
    day = test_df[test_df['date'] == date]
    if len(day) < 30:
        continue
    
    top_xgb = set(day.nlargest(15, 'score_XGB')['code'])
    top_lgb = set(day.nlargest(15, 'score_LGB')['code'])
    top_cat = set(day.nlargest(15, 'score_Cat')['code'])
    
    overlaps['XGB_LGB'].append(len(top_xgb & top_lgb) / 15)
    overlaps['XGB_Cat'].append(len(top_xgb & top_cat) / 15)
    overlaps['LGB_Cat'].append(len(top_lgb & top_cat) / 15)
    overlaps['All_3'].append(len(top_xgb & top_lgb & top_cat) / 15)

print(f"  XGB ∩ LGB: {np.mean(overlaps['XGB_LGB']):.1%}")
print(f"  XGB ∩ Cat: {np.mean(overlaps['XGB_Cat']):.1%}")
print(f"  LGB ∩ Cat: {np.mean(overlaps['LGB_Cat']):.1%}")
print(f"  三模型交集: {np.mean(overlaps['All_3']):.1%}")

# ============================================================
# 综合排名
# ============================================================
print("\n" + "=" * 90)
print("📊 10年回测综合排名")
print("=" * 90)

results.sort(key=lambda x: x['sharpe'] * max(0.01, 1+x['avg_drawdown']), reverse=True)
print(f"\n{'#':>2} {'策略':<35} {'年化':>7} {'最大DD':>8} {'平均DD':>8} {'夏普':>6} {'Sortino':>7} {'胜率':>6}")
print("-" * 95)
for i, r in enumerate(results):
    q = "🟢" if r['avg_drawdown'] > -0.03 else "🟡" if r['avg_drawdown'] > -0.05 else "🔴"
    print(f"{i+1:>2} {r['name']:<35} {r['annual_return']:>+6.1%} {r['max_drawdown']:>7.1%} {r['avg_drawdown']:>7.2%} {r['sharpe']:>6.2f} {r['sortino']:>7.2f} {r['win_rate']:>5.1%} {q}")

# 保存
output = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'test_period': f"{test_df['date'].min().date()} ~ {test_df['date'].max().date()}",
    'n_years': test_df['date'].dt.year.nunique(),
    'results': [{k:v for k,v in r.items() if k!='equity_curve'} for r in results],
    'cost_sensitivity': cost_results,
    'feature_importance': {k: {f: float(np.mean(v)) for f, v in feats.items()} for k, feats in feature_importances.items()},
    'pick_overlap': {k: float(np.mean(v)) for k, v in overlaps.items()},
}
with open(f'{OUT}/v4_10year_validation.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n\n保存 → analysis/v4_10year_validation.json")
print(f"总耗时: {time.time()-t0:.1f}s")
print("=" * 90)
