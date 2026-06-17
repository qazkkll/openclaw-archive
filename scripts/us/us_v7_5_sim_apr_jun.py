#!/usr/bin/env python3
"""
us_v7_5_sim_apr_jun.py — V7.5 回测模拟 2026-04-01 ~ 2026-06-11
T5_H10_S15_R10策略，次日开盘价买卖，10万美金起始
"""
import sys, os, json, pickle, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, xgboost as xgb

BASE = '/home/hermes/.hermes/openclaw-archive'; ML = f'{BASE}/ml'; MD = f'{BASE}/data/models'; VER = 'us_v7_5'
print('=' * 70, flush=True)
print(f'V7.5 模拟回测 2026-04-01~2026-06-11', flush=True)
print(f'策略: T5_H10_S15_R10, 起始$100,000', flush=True)
print('=' * 70, flush=True)
T0 = time.time()

model = xgb.Booster(); model.load_model(f'{MD}/{VER}.json')
cal = pickle.load(open(f'{MD}/{VER}_calibrator.pkl', 'rb'))
report = json.load(open(f'{MD}/{VER}_report.json'))
FEATS = report['features']

# 特征
df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str'] = df['date'].astype(str).str[:10]
df['target'] = (df['fwd_5d_ret'] > 0.05).astype(int)
for f in FEATS:
    if f in df.columns:
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0).clip(-1e6, 1e6)
df = df.replace([np.inf, -np.inf], np.nan)
del df['date']

# 回测日期范围
BTD = sorted(df['date_str'].unique())
BTD = [d for d in BTD if d >= '2026-04-01' and d <= '2026-06-11']
print(f'回测天数: {len(BTD)}', flush=True)

# 价格数据（含open/close）
main = pd.read_parquet(f'{ML}/us_hist_yf_10y.parquet', columns=['ticker', 'date', 'open', 'close'])
main.rename(columns={'ticker': 'sym'}, inplace=True)
mega = pd.read_parquet(f'{ML}/us_hist_megacap_10y.parquet', columns=['sym', 'date', 'open', 'close'])
all_v = pd.concat([main, mega], ignore_index=True).drop_duplicates(subset=['sym', 'date'])
all_v['ds'] = all_v['date'].astype(str).str[:10]
all_v = all_v[all_v['ds'].isin(BTD)]
close_idx = {}; open_idx = {}
for s, g in all_v.groupby('sym'):
    g = g.sort_values('ds')
    open_idx[s] = dict(zip(g['ds'].values, g['open'].values.astype(float)))
    close_idx[s] = dict(zip(g['ds'].values, g['close'].values.astype(float)))
del main, mega, all_v
print(f'价格索引: {len(close_idx)}只', flush=True)

# 逐日概率计算+回测
print(f'\n逐日模拟回测...', flush=True)

CAPITAL = 100000.0
TOP_N = 5; HOLD = 10; STOP = -0.15; REBAL = 10

cash = CAPITAL
portfolio = {}
curve = []  # 每日净值
daily_log = []  # 每日操作日志
trades_log = []  # 每笔交易记录
trade_id = 0

# 交易日历
day_set = set(BTD)

for day_idx, d in enumerate(BTD):
    # 1. 当天候选股评分
    day = df[df['date_str'] == d]
    if len(day) < 30:
        curve.append(cash + sum(p['qty'] * close_idx.get(s, {}).get(d, p['bp'])
                                for s, p in portfolio.items()))
        continue
    
    X = np.nan_to_num(day[FEATS].values.astype(np.float32), nan=0)
    raw = model.predict(xgb.DMatrix(X, feature_names=FEATS))
    calib = cal.predict_proba(raw.reshape(-1, 1))[:, 1]
    day = day.copy()
    day['prob'] = calib
    day = day.sort_values('prob', ascending=False)
    
    # 2. 持仓检查——止损/到期
    for sym in list(portfolio.keys()):
        pos = portfolio[sym]
        cp = close_idx.get(sym, {}).get(d)
        if cp is None:
            continue
        ret = (cp - pos['bp']) / pos['bp']
        days_held = day_idx - pos['di']
        
        if ret <= STOP or days_held >= HOLD:
            # 卖出
            proceed = pos['qty'] * cp
            cash += proceed
            pnl = proceed - (pos['qty'] * pos['bp'])
            trades_log.append({
                'id': trade_id, 'date': d, 'sym': sym, 'action': 'sell',
                'reason': '止损' if ret <= STOP else '到期',
                'buy_price': round(pos['bp'], 2), 'sell_price': round(cp, 2),
                'qty': round(pos['qty'], 2), 'pnl': round(pnl, 2),
                'pnl_pct': round(ret * 100, 1), 'days_held': days_held,
                'buy_date': pos['bd']
            })
            trade_id += 1
            del portfolio[sym]
    
    # 3. 调仓
    if day_idx % REBAL == 0 or len(portfolio) < TOP_N:
        # 找不在持仓中的高分股
        hold_syms = set(portfolio.keys())
        cand = []
        for _, r in day.iterrows():
            if r['sym'] in hold_syms:
                continue
            nxt_idx = day_idx + 1
            if nxt_idx >= len(BTD):
                continue
            nxt_d = BTD[nxt_idx]
            bp = open_idx.get(r['sym'], {}).get(nxt_d)
            if bp is None or np.isnan(bp) or bp <= 0:
                continue
            cand.append((r['sym'], r['prob'], float(bp), nxt_d))
            if len(cand) >= TOP_N * 2:
                break
        
        # 买入（用次日开盘价）
        need = TOP_N - len(portfolio)
        buys = cand[:need]
        for sym, prob, price, nxt_d in buys:
            qty = cash / max(need, 1) / max(price, 0.01)
            if qty < 10:  # 最少10股
                continue
            cost = qty * price
            if cost > cash:
                continue
            cash -= cost
            portfolio[sym] = {
                'bp': price, 'qty': qty, 'di': day_idx,
                'bd': nxt_d, 'prob': float(prob)
            }
            trades_log.append({
                'id': trade_id, 'date': nxt_d, 'sym': sym, 'action': 'buy',
                'reason': '新开仓',
                'buy_price': round(price, 2), 'sell_price': None,
                'qty': round(qty, 2), 'pnl': None, 'pnl_pct': None,
                'days_held': None, 'buy_date': nxt_d, 'prob': round(float(prob), 4)
            })
            trade_id += 1
    
    # 4. 记录当日净值
    pv = sum(p['qty'] * close_idx.get(s, {}).get(d, p['bp'])
             for s, p in portfolio.items())
    nav = cash + pv
    curve.append(nav)
    
    # 每天记录日志
    hold_info = [(s, round(p['qty'], 1), round(p['bp'], 2),
                  round((close_idx.get(s, {}).get(d, p['bp']) - p['bp']) / p['bp'] * 100, 1))
                 for s, p in portfolio.items()]
    daily_log.append({
        'date': d, 'nav': round(nav, 2),
        'cash': round(cash, 2), 'holdings': len(portfolio),
        'top3_probs': [(r['sym'], round(float(r['prob']), 4))
                       for _, r in day.head(3).iterrows()],
        'portfolio': [{'sym': s, 'qty': q, 'bp': bp, 'ret_pct': r}
                      for s, q, bp, r in hold_info]
    })
    
    if day_idx % 10 == 0:
        print(f'  Day {day_idx}: {d} NAV=${nav:,.0f} 持仓{len(portfolio)}只', flush=True)

# 清仓
final_cash = cash
final_pv = 0
for sym, pos in portfolio.items():
    last_d = BTD[-1]
    cp = close_idx.get(sym, {}).get(last_d, pos['bp'])
    final_cash += pos['qty'] * cp
    final_pv += pos['qty'] * pos['bp']
    trades_log.append({
        'id': trade_id, 'date': last_d, 'sym': sym, 'action': 'sell',
        'reason': '模拟结束清仓',
        'buy_price': round(pos['bp'], 2), 'sell_price': round(cp, 2),
        'qty': round(pos['qty'], 2), 'pnl': round(pos['qty'] * (cp - pos['bp']), 2),
        'pnl_pct': round((cp - pos['bp']) / pos['bp'] * 100, 1),
        'days_held': len(BTD) - pos['di']
    })

final_nav = final_cash
total_return = (final_nav / CAPITAL - 1) * 100
years = len(BTD) / 252
annualized = ((final_nav / CAPITAL) ** (1 / max(years, 0.01)) - 1) * 100

# 统计
equity = np.array(curve)
peak = np.maximum.accumulate(equity)
mdd_pct = (equity - peak).min() / peak.max() * 100 if peak.max() > 0 else 0
dr = np.diff(equity) / (equity[:-1] + 1e-10)
sharpe = (dr.mean() / max(dr.std(), 1e-6)) * np.sqrt(252) if len(dr) > 20 else 0

buy_trades = [t for t in trades_log if t['action'] == 'buy']
sell_trades = [t for t in trades_log if t['action'] == 'sell' and t['pnl'] is not None]
wins = sum(1 for t in sell_trades if t['pnl'] > 0)
losses = len(sell_trades) - wins
win_rate = wins / max(len(sell_trades), 1) * 100

# ===== 输出结果 =====
print('\n' + '=' * 70)
print(f'📊 模拟回测结果: 2026-04-01 ~ 2026-06-11')
print('=' * 70)
print(f'💰 起始资金: ${CAPITAL:,.0f}')
print(f'💰 最终净值: ${final_nav:,.0f}')
print(f'📈 总收益率: {total_return:+.2f}%')
print(f'📈 年化收益: {annualized:+.2f}%')
print(f'📉 最大回撤: {mdd_pct:.2f}%')
print(f'📊 夏普比率: {sharpe:.2f}')
print(f'📐 交易笔数: {len(buy_trades)}买入 / {len(sell_trades)}卖出')
print(f'🏆 胜率: {win_rate:.1f}% ({wins}胜 / {losses}负)')
print(f'🕐 回测天数: {len(BTD)}个交易日')
print(f'💵 盈亏总额: ${sum(t["pnl"] for t in sell_trades if t["pnl"] is not None):+,.0f}')

# 每日净值表
print(f'\n{"日期":>12s} {"净值":>10s} {"现金":>10s} {"持仓":>5s} {"Top3候选":>35s}')
print('-' * 75)
for entry in daily_log:
    if entry['date'] in [BTD[0]] + [BTD[i] for i in range(0, len(BTD), 10)] + [BTD[-1]]:
        top3 = ','.join(f"{s}({p*100:.0f}%)" for s, p in entry['top3_probs'])
        print(f'{entry["date"]:>12s} ${entry["nav"]:>8,.0f} ${entry["cash"]:>8,.0f} {entry["holdings"]:>5d} {top3:>35s}')

# 关键交易记录
print(f'\n== 关键买卖记录 ==')
print(f'{"日期":>12s} {"操作":>5s} {"代码":>8s} {"理由":>10s} {"价格":>8s} {"盈亏":>10s}')
print('-' * 60)
for t in trades_log:
    if t['action'] == 'buy':
        print(f'{t["date"]:>12s} {"买入":>5s} {t["sym"]:>8s} {"新开仓":>10s} ${t["buy_price"]:<7.2f}')
    elif t['action'] == 'sell' and t['pnl'] is not None and abs(t['pnl']) > 1000:
        pnl_str = f'+${t["pnl"]:,.0f}' if t['pnl'] > 0 else f'-${abs(t["pnl"]):,.0f}'
        print(f'{t["date"]:>12s} {"卖出":>5s} {t["sym"]:>8s} {t["reason"]:>10s} ${t["sell_price"]:<7.2f} {pnl_str:>10s}')

# 保存
output = {
    'simulation': 'V7.5 2026-04-01~2026-06-11',
    'strategy': 'T5_H10_S15_R10',
    'start_capital': CAPITAL,
    'end_nav': round(final_nav, 2),
    'total_return_pct': round(total_return, 2),
    'annualized_pct': round(annualized, 2),
    'max_drawdown_pct': round(mdd_pct, 2),
    'sharpe': round(sharpe, 2),
    'win_rate_pct': round(win_rate, 1),
    'wins': wins, 'losses': losses,
    'total_trades': len(buy_trades),
    'days': len(BTD),
    'daily_nav': [{'date': e['date'], 'nav': e['nav']} for e in daily_log],
    'trades': trades_log
}
json.dump(output, open(f'{BASE}/data/sim_apr_jun_v75.json', 'w'), indent=2, ensure_ascii=False)
print(f'\n✅ 已保存: {BASE}/data/sim_apr_jun_v75.json')
print(f'⏱️ 耗时: {time.time() - T0:.1f}s')
print('=' * 70)
