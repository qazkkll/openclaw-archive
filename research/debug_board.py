#!/usr/bin/env python3
"""Debug board rotation backtest"""
import pandas as pd, numpy as np, json, datetime, warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('data/a_hist_10y.parquet')
df = df.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df['date'] = df['date'].astype(int)

mf = pd.read_parquet('data/cn/moneyflow_core.parquet')
mf['sym'] = mf['ts_code'].str[:6]
mf['date'] = mf['trade_date'].astype(int)
for col in ['sm', 'md', 'lg', 'elg']:
    mf[f'{col}_net'] = mf[f'buy_{col}_amount'] - mf[f'sell_{col}_amount']
mf['total_net'] = mf['net_mf_amount']
df = df.merge(mf[['sym','date','total_net','lg_net','md_net','elg_net']], on=['sym','date'], how='left')

df = df[~df['sym'].str.startswith('688')].copy()
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
df = df[df['volume'] > 0].copy()
df = df.sort_values(['sym', 'date']).reset_index(drop=True)

df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['vol20'] = df.groupby('sym')['ret5'].transform(lambda x: x.rolling(4, min_periods=2).std())
delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan)).fillna(50)
for col in ['total_net', 'lg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())

df['board'] = df['sym'].apply(lambda c: '创业板' if c[:3] in ('300','301') else '主板')

df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')
market_avg_r20 = df.groupby('date')['mkt_ret20'].first()
market_ma60 = market_avg_r20.rolling(60, min_periods=1).mean()
market_ma120 = market_avg_r20.rolling(120, min_periods=1).mean()
market_state_map = {}
for d in sorted(df['date'].unique()):
    r20 = market_avg_r20.get(d, 0) if d in market_avg_r20.index else 0
    ma60 = market_ma60.get(d, 0) if d in market_ma60.index else 0
    ma120 = market_ma120.get(d, 0) if d in market_ma120.index else 0
    if ma60 <= ma120 and r20 <= 0:
        market_state_map[d] = 'bear'
    elif ma60 <= ma120 or r20 <= 0:
        market_state_map[d] = 'cautious'
    else:
        market_state_map[d] = 'bull'

def score_baseline(day):
    s = day.copy()
    s['score'] = 0.0
    s['score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 3
    s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 2
    s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 2
    s['score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1.5
    s['score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 1
    s['score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1
    return s

def score_board_top(day):
    s = score_baseline(day)
    if len(s) == 0:
        return s
    board_avg = s.groupby('board')['score'].mean()
    if len(board_avg) == 0:
        return s.head(0)
    best = board_avg.idxmax()
    return s[s['board'] == best]

# Run backtest
test_dates = sorted(df[(df['date'] >= 20200101) and (df['date'] <= 20260616)]['date'].unique())
price_dict = {}
for d in test_dates:
    dd = df[df['date'] == d]
    price_dict[d] = dict(zip(dd['sym'], dd['close']))

rebal_dates = test_dates[::10]
equity = 100000.0
equity_curve = [(test_dates[0], equity)]
trades = []

for i, rd in enumerate(rebal_dates):
    state = market_state_map.get(rd, 'bull')
    if state == 'bear':
        next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else test_dates[-1]
        for d in test_dates:
            if rd < d <= next_rd:
                equity_curve.append((d, equity))
        continue
    pos_pct = 0.5 if state == 'cautious' else 1.0
    day = df[df['date'] == rd].copy()
    if len(day) < 15:
        continue
    scored = score_board_top(day)
    if len(scored) == 0:
        print(f"EMPTY at {rd}, day_len={len(day)}")
        next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else test_dates[-1]
        for d in test_dates:
            if rd < d <= next_rd:
                equity_curve.append((d, equity))
        continue
    picks = scored.nlargest(15, 'score')
    entry_prices = dict(zip(picks['sym'], picks['close']))
    equity *= (1 - 0.003 * pos_pct)
    next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else test_dates[-1]
    hold_dates = [d for d in test_dates if rd < d <= next_rd]
    active = set(entry_prices.keys())
    prev_p = {s: entry_prices[s] for s in active}
    for hd in hold_dates:
        curr = price_dict.get(hd, {})
        dr = 0.0
        na = len(active)
        if na == 0:
            equity_curve.append((hd, equity))
            continue
        w = pos_pct / na
        stopped = []
        for sym in list(active):
            if sym not in curr:
                continue
            cp = curr[sym]
            ep = entry_prices[sym]
            pp = prev_p.get(sym, ep)
            cum = cp/ep - 1
            if cum <= -0.03:
                ppc = pp/ep - 1
                if ppc <= -0.03:
                    sr = 0
                else:
                    sr = -0.03 - ppc
                    stopped.append(sym)
            else:
                sr = cp/pp - 1 if pp > 0 else 0
            dr += sr * w
            prev_p[sym] = cp
        equity *= (1 + dr)
        equity_curve.append((hd, equity))
        for sym in stopped:
            active.discard(sym)
    equity *= (1 - 0.003 * pos_pct)
    for sym, ep in entry_prices.items():
        xp = price_dict.get(next_rd, {}).get(sym, ep)
        ret = xp/ep - 1
        if ret < -0.03:
            ret = -0.03
        trades.append(ret - 0.003)

eq_arr = np.array([e[1] for e in equity_curve])
eq_dates = np.array([e[0] for e in equity_curve])
daily_rets = np.diff(eq_arr) / eq_arr[:-1]
daily_rets = daily_rets[np.isfinite(daily_rets)]

dt1 = datetime.datetime.strptime(str(eq_dates[0]), '%Y%m%d')
dt2 = datetime.datetime.strptime(str(eq_dates[-1]), '%Y%m%d')
years = (dt2 - dt1).days / 365.25
cagr = (eq_arr[-1]/eq_arr[0]) ** (1/years) - 1
sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)
peak = np.maximum.accumulate(eq_arr)
dd = (eq_arr - peak) / peak

print(f"Board rotation: CAGR={cagr:.1%} Sharpe={sharpe:.2f} DD={dd.min():.1%} Trades={len(trades)}")
