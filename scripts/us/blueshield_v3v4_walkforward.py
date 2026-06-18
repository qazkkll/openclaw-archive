#!/usr/bin/env python3
"""
V3≥60 + V4-LGB排序 + SL-8% — Walk-Forward验证
每年重训LGB，V3过滤始终生效
"""
import pandas as pd
import numpy as np
import json, time, warnings
warnings.filterwarnings('ignore')

DATA = '/home/hermes/.hermes/openclaw-archive/data/us/us_hist_sp500_10y.parquet'
print("=" * 90)
print("V3≥60+V4排序+SL-8% — Walk-Forward验证")
print("=" * 90)

t0 = time.time()
df = pd.read_parquet(DATA).rename(columns={'sym': 'code'})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['code', 'date']).reset_index(drop=True)

# V3评分
def calc_v3(g):
    c, h, l = g['close'], g['high'], g['low']
    ma5 = c.rolling(5).mean(); ma20 = c.rolling(20).mean(); ma60 = c.rolling(60).mean()
    trend = ((c > ma5).astype(float)*10 + (ma5 > ma20).astype(float)*10 + (ma20 > ma60).astype(float)*10)
    ret5 = c.pct_change(5); ret20 = c.pct_change(20)
    momentum = ((ret5 > 0).astype(float)*12.5 + (ret20 > 0).astype(float)*12.5)
    ema12 = c.ewm(span=12, adjust=False).mean(); ema26 = c.ewm(span=26, adjust=False).mean()
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

df['v3_score'] = df.groupby('code').apply(calc_v3).reset_index(level=0, drop=True)
print(f"V3评分完成 ({time.time()-t0:.1f}s)")

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
        'high_52w','low_52w','v3_score'}
feat_cols = [c for c in df.columns if c not in skip]
df = df.replace([np.inf,-np.inf], np.nan)
core = [c for c in feat_cols if not c.startswith('dist_52w') and not c.startswith('ret_skew') 
        and not c.startswith('ret_ratio')]
df = df.dropna(subset=core + ['target_5d','daily_ret']).sort_values('date').reset_index(drop=True)
print(f"特征: {len(feat_cols)}维, 数据: {len(df):,}行 ({time.time()-t0:.1f}s)")

from lightgbm import LGBMRegressor

# Walk-Forward：每年重训
years = sorted(df['date'].dt.year.unique())
start_year = 2018  # 需要2年训练数据
all_trades = []
all_equity = []
cash = 1.0
positions = {}

for test_year in range(start_year, 2027):
    train_end = pd.Timestamp(f'{test_year-1}-12-31')
    test_start = pd.Timestamp(f'{test_year}-01-01')
    test_end = pd.Timestamp(f'{test_year}-12-31')
    
    if test_start > df['date'].max():
        break
    
    # 训练
    train = df[df['date'] <= train_end]
    X_train = train[feat_cols].values
    y_train = train['target_5d'].values
    
    model = LGBMRegressor(n_estimators=600, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, min_child_samples=20,
        random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)
    
    # 测试
    test = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
    test['lgb_pred'] = model.predict(test[feat_cols].values)
    
    # 回测（V3≥60过滤 + LGB排序 + Top-15 7d + SL-8%）
    test_dates = sorted(test['date'].unique())
    for i, date in enumerate(test_dates):
        day = test[test['date'] == date]
        if len(day) == 0: continue
        
        # 持仓PnL + 止损
        for code in list(positions.keys()):
            row = day[day['code'] == code]
            if len(row) == 0: continue
            cur = row['close'].values[0]
            pos = positions[code]
            ret = cur / pos['entry'] - 1
            pos['peak'] = max(pos.get('peak', pos['entry']), cur)
            
            if ret <= -0.08:  # SL-8%
                pnl = (cur / pos['entry'] - 1)
                cash += pos['shares'] * cur
                all_trades.append({'year': test_year, 'code': code, 'pnl': pnl, 'reason': 'stop_loss'})
                del positions[code]
            elif pos['peak'] > pos['entry'] * 1.05:
                trail = (pos['peak'] - cur) / pos['peak']
                if trail >= 0.05:
                    pnl = (cur / pos['entry'] - 1)
                    cash += pos['shares'] * cur
                    all_trades.append({'year': test_year, 'code': code, 'pnl': pnl, 'reason': 'trailing'})
                    del positions[code]
        
        # 轮换（每7天）
        if i % 7 == 0 and i > 0:
            for code in list(positions.keys()):
                row = day[day['code'] == code]
                if len(row) == 0: continue
                price = row['close'].values[0]
                pos = positions.pop(code)
                pnl = (price / pos['entry'] - 1)
                cash += pos['shares'] * price
                all_trades.append({'year': test_year, 'code': code, 'pnl': pnl, 'reason': 'expire'})
            
            # 选股：V3≥60过滤 + LGB排序
            avail = day[~day['code'].isin(positions.keys())]
            avail = avail[avail['v3_score'] >= 60]
            if len(avail) > 0:
                avail['rank'] = avail['lgb_pred'].rank(ascending=False)
                top = avail.nsmallest(15, 'rank')
                if len(top) > 0:
                    size_per = cash / len(top)
                    for _, row in top.iterrows():
                        if row['close'] > 0:
                            shares = size_per / row['close']
                            positions[row['code']] = {
                                'entry': row['close'], 'shares': shares, 'peak': row['close']}
                            cash -= size_per
        
        # 记录权益
        pos_val = sum(
            pos['shares'] * day[day['code']==c]['close'].values[0]
            for c, pos in positions.items()
            if len(day[day['code']==c]) > 0)
        all_equity.append({'date': date, 'equity': cash + pos_val, 'year': test_year})
    
    # 年度统计
    year_eq = pd.DataFrame([e for e in all_equity if e['year'] == test_year])
    year_trades = [t for t in all_trades if t['year'] == test_year]
    if len(year_eq) > 1:
        yr = year_eq['equity'].iloc[-1] / year_eq['equity'].iloc[0] - 1
        n_sl = sum(1 for t in year_trades if t['reason'] == 'stop_loss')
        print(f"  {test_year}: {yr:>+7.1%} | 交易{len(year_trades):>4}笔 | 止损{n_sl:>3}次 | 余额${cash:.2f}")

# 总体统计
eq = pd.DataFrame(all_equity)
if len(eq) > 1:
    total = eq['equity'].iloc[-1] / eq['equity'].iloc[0] - 1
    days = (eq['date'].max() - eq['date'].min()).days
    annual = (1+total)**(365/max(days,1))-1
    eq['ret'] = eq['equity'].pct_change()
    sharpe = eq['ret'].mean() / eq['ret'].std() * np.sqrt(252) if eq['ret'].std() > 0 else 0
    rolling_max = eq['equity'].cummax()
    dd = (eq['equity'] / rolling_max - 1)
    max_dd = dd.min()
    
    in_dd = False; dd_list = []; s = 0
    for j in range(len(dd)):
        if dd.iloc[j] < -0.001 and not in_dd:
            in_dd = True; s = j
        elif dd.iloc[j] >= -0.001 and in_dd:
            in_dd = False; dd_list.append(dd.iloc[s:j].min())
    if in_dd: dd_list.append(dd.iloc[s:].min())
    avg_dd = np.mean(dd_list) if dd_list else 0
    
    tdf = pd.DataFrame(all_trades)
    win_rate = (tdf['pnl'] > 0).mean() if len(tdf) > 0 else 0
    
    print(f"\n{'='*90}")
    print(f"📊 Walk-Forward总计 (2018-2026)")
    print(f"{'='*90}")
    print(f"  夏普:     {sharpe:.2f}")
    print(f"  年化:     {annual:+.1%}")
    print(f"  总收益:   {total:+.1%}")
    print(f"  最大DD:   {max_dd:.1%}")
    print(f"  平均DD:   {avg_dd:.2%}")
    print(f"  胜率:     {win_rate:.1%}")
    print(f"  交易数:   {len(all_trades)}")
    print(f"  止损率:   {sum(1 for t in all_trades if t['reason']=='stop_loss')/max(len(all_trades),1):.1%}")
    
    # 对比
    print(f"\n  对比纯V4-LGB Walk-Forward:")
    print(f"  纯V4:     夏普1.13, 年化+48.4%, 最大DD-56.8%, 平均DD-4.77%")
    print(f"  V3+V4:    夏普{sharpe:.2f}, 年化{annual:+.1%}, 最大DD{max_dd:.1%}, 平均DD{avg_dd:.2%}")

print(f"\n总耗时: {time.time()-t0:.1f}s")
print("=" * 90)
