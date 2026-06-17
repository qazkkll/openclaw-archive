#!/usr/bin/env python3
"""
us_v7_5_sim_apr_jun_filtered.py — 过滤版 2026-04-01~2026-06-11
与us_v7_5_sim_apr_jun.py相同逻辑，但候选池限1654只+二次确认
T5_H10_S15_R10，次日开盘价买卖
"""
import sys, os, json, pickle, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, xgboost as xgb

BASE = '/home/hermes/.hermes/openclaw-archive'; ML = f'{BASE}/ml'; MD = f'{BASE}/data/models'; VER = 'us_v7_5'
print('=' * 70, flush=True)
print(f'V7.5 过滤版模拟 2026-04-01~2026-06-11', flush=True)
print(f'策略: T5_H10_S15_R10, 起始$100,000', flush=True)
print(f'过滤: 成交额≥$5M/天 + 价格≥$3 (1654只)', flush=True)
print('=' * 70, flush=True)
T0 = time.time()

# 1. 加载模型
model = xgb.Booster(); model.load_model(f'{MD}/{VER}.json')
cal = pickle.load(open(f'{MD}/{VER}_calibrator.pkl', 'rb'))
report = json.load(open(f'{MD}/{VER}_report.json'))
FEATS = report['features']

# 2. 加载过滤名单
fs = json.load(open(f'{ML}/us_filtered_syms.json'))
filtered_syms = set(fs['syms'])
print(f'过滤名单: {len(filtered_syms)}只', flush=True)

# 3. 特征
df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str'] = df['date'].astype(str).str[:10]
df = df[df['sym'].isin(filtered_syms)]  # 关键：过滤候选池
print(f'过滤后特征: {len(df)}行', flush=True)
df['target'] = (df['fwd_5d_ret'] > 0.05).astype(int)
for f in FEATS:
    if f in df.columns:
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0).clip(-1e6, 1e6)
df = df.replace([np.inf, -np.inf], np.nan)
del df['date']

# 4. 回测日期
BTD = sorted(df['date_str'].unique())
BTD = [d for d in BTD if d >= '2026-04-01' and d <= '2026-06-11']
print(f'回测天数: {len(BTD)}', flush=True)

# 5. 价格索引
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

# 6. 回测
print(f'\n逐日模拟回测...', flush=True)
CAPITAL = 100000.0
TOP_N = 5; HOLD = 10; STOP = -0.15; REBAL = 10

cash = CAPITAL
portfolio = {}
curve = []
daily_log = []
trades_log = []
trade_id = 0

for day_idx, d in enumerate(BTD):
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
    
    # 止损/到期
    for sym in list(portfolio.keys()):
        pos = portfolio[sym]
        cp = close_idx.get(sym, {}).get(d)
        if cp is None:
            continue
        ret = (cp - pos['bp']) / pos['bp']
        days_held = day_idx - pos['di']
        
        if ret <= STOP or days_held >= HOLD:
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
    
    # 调仓
    if day_idx % REBAL == 0 or len(portfolio) < TOP_N:
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
        
        need = TOP_N - len(portfolio)
        buys = cand[:need]
        for sym, prob, price, nxt_d in buys:
            qty = cash / max(need, 1) / max(price, 0.01)
            if qty < 10:
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
    
    pv = sum(p['qty'] * close_idx.get(s, {}).get(d, p['bp'])
             for s, p in portfolio.items())
    nav = cash + pv
    curve.append(nav)
    
    daily_log.append({
        'date': d, 'nav': round(nav, 2),
        'cash': round(cash, 2), 'holdings': len(portfolio),
        'top3_probs': [(r['sym'], round(float(r['prob']), 4))
                       for _, r in day.head(3).iterrows()],
        'portfolio': [{'sym': s, 'qty': round(p['qty'], 1), 'bp': round(p['bp'], 2),
                       'ret_pct': round((close_idx.get(s, {}).get(d, p['bp']) - p['bp']) / p['bp'] * 100, 1)}
                      for s, p in portfolio.items()]
    })
    
    if day_idx % 10 == 0:
        print(f'  Day {day_idx}: {d} NAV=${nav:,.0f} 持仓{len(portfolio)}只', flush=True)

# 清仓
final_cash = cash
for sym, pos in portfolio.items():
    last_d = BTD[-1]
    cp = close_idx.get(sym, {}).get(last_d, pos['bp'])
    final_cash += pos['qty'] * cp
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

# 输出
print('\n' + '=' * 70)
print(f'📊 过滤版模拟结果: 2026-04-01 ~ 2026-06-11')
print('=' * 70)
print(f'💰 起始资金: ${CAPITAL:,.0f}')
print(f'💰 最终净值: ${final_nav:,.0f}')
print(f'📈 总收益率: {total_return:+.2f}%')
print(f'📈 年化收益: {annualized:+.2f}%')
print(f'📉 最大回撤: {mdd_pct:.2f}%')
print(f'📊 夏普比率: {sharpe:.2f}')
print(f'📐 交易笔数: {len(buy_trades)}买入 / {len(sell_trades)}卖出')
print(f'🏆 胜率: {win_rate:.1f}% ({wins}胜/{losses}负)')
print(f'💵 盈亏总额: ${sum(t["pnl"] for t in sell_trades):+,.0f}')
print(f'🕐 回测天数: {len(BTD)}个交易日')
print(f'⏱️ 耗时: {time.time()-T0:.1f}s')

print(f'\n{"日期":>12} {"净值":>10} {"现金":>8} {"持仓":>4} {"Top3候选":>30}')
print('-' * 70)
for dl in [d for d in daily_log if d['date'] in ('2026-04-01','2026-04-15','2026-04-30','2026-05-15','2026-05-29','2026-06-10')]:
    top3 = ','.join([f'{s}({int(p*100)})' for s, p in dl['top3_probs'][:3]])
    print(f'  {dl["date"]} ${dl["nav"]:>8,} ${dl["cash"]:>6,}   {dl["holdings"]}   {top3}')

print(f'\n== 关键买卖记录 ==')
buy_only = [t for t in trades_log if t['action'] == 'buy']
sell_only = [t for t in trades_log if t['action'] == 'sell' and t['pnl'] is not None]
print(f'{"日期":>10} {"操作":>4} {"代码":>6} {"理由":>10} {"价格":>6} {"盈亏":>8}')
print('-' * 50)
for t in buy_only:
    print(f'{t["date"]:>10} {"买入":>4} {t["sym"]:>6} {t["reason"]:>10} {t["buy_price"]:>6.2f}')
for t in sorted(sell_only, key=lambda x: abs(x['pnl']), reverse=True)[:10]:
    print(f'{t["date"]:>10} {"卖出":>4} {t["sym"]:>6} {t["reason"]:>10} {t["sell_price"]:>6.2f} {t["pnl"]:>+8,.0f}')

# 保存
result = {
    'capital': CAPITAL, 'final_nav': round(final_nav, 2),
    'total_return_pct': round(total_return, 2),
    'annualized_pct': round(annualized, 2),
    'mdd_pct': round(mdd_pct, 2), 'sharpe': round(sharpe, 2),
    'win_rate': round(win_rate, 1),
    'buy_count': len(buy_trades), 'sell_count': len(sell_trades),
    'days': len(BTD), 'trades': trades_log, 'curve': [round(x, 2) for x in curve],
    'daily_log': daily_log
}
with open(f'{BASE}/data/sim_apr_jun_v75_filtered.json', 'w') as f:
    json.dump(result, f)
print(f'\n✅ 已保存: {BASE}/data/sim_apr_jun_v75_filtered.json')

# 额外统计：过滤后实际买到的票
buy_syms_filtered = set(t['sym'] for t in buy_trades)
print(f'\n== 过滤后买到的股票 ({len(buy_syms_filtered)}只) ==')
for s in sorted(buy_syms_filtered):
    print(f'  {s}', end='')
print()
print('=' * 70)
