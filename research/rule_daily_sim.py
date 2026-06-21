#!/usr/bin/env python3
"""
rule-alpha-v1.0 — 每日精确回撤验证
CEO决策: 验证真实每日回撤（非仅调仓日）
- 每日跟踪持仓收益
- 每日检查止损（累计亏损>SL即止损）
- 准确计算最大回撤
"""
import pandas as pd, numpy as np, json, time, os, sys, datetime
import warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

WORKSPACE = os.path.expanduser('~/.hermes/openclaw-archive')
DATA_DIR = os.path.join(WORKSPACE, 'data')

print("="*60)
print("rule-alpha-v1.0 — 每日精确回撤验证")
print("="*60)

# ============================================================
# 1. 加载全部数据
# ============================================================
print("\n[1] 加载全部数据...")
t0 = time.time()

df = pd.read_parquet(os.path.join(DATA_DIR, 'a_hist_10y.parquet'))
df = df.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df['date'] = df['date'].astype(int)

mf = pd.read_parquet(os.path.join(DATA_DIR, 'cn/moneyflow_core.parquet'))
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

# 特征
df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['vol20'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(20, min_periods=2).std())

delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)

for col in ['total_net', 'lg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())

# 市场状态
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')

all_dates_full = sorted(df['date'].unique())
print(f"  全量: {len(df):,}行, {df['sym'].nunique()}只, {all_dates_full[0]}~{all_dates_full[-1]}, {time.time()-t0:.0f}秒")

# ============================================================
# 2. 评分函数
# ============================================================
def score_optimized(day):
    s = day.copy()
    s['score'] = 0.0
    s['score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 3
    s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 2
    s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 2
    s['score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1.5
    s['score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 1
    s['score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1
    return s

# ============================================================
# 3. 市场状态判断
# ============================================================
market_avg_r20 = df.groupby('date')['mkt_ret20'].first()
market_ma60 = market_avg_r20.rolling(60, min_periods=1).mean()
market_ma120 = market_avg_r20.rolling(120, min_periods=1).mean()

def get_market_state(date_int):
    r20 = market_avg_r20.get(date_int, 0)
    ma60 = market_ma60.get(date_int, 0)
    ma120 = market_ma120.get(date_int, 0)
    ma_bull = ma60 > ma120
    mom_pos = r20 > 0
    if not ma_bull and not mom_pos:
        return 'bear'
    elif not ma_bull or not mom_pos:
        return 'cautious'
    else:
        return 'bull'

# ============================================================
# 4. 每日精确回测引擎
# ============================================================
def run_daily_backtest(df_data, dates, hold_days=10, top_n=15, stop_loss=-0.03, 
                       cost=0.003, market_filter=True):
    """
    每日精确回测：
    - 每hold_days天选股
    - 每日跟踪持仓收益
    - 每日检查止损
    - 计算真实每日回撤
    """
    # 构建日期->数据映射
    date_data = {}
    for d in dates:
        day = df_data[df_data['date'] == d]
        if len(day) > 0:
            date_data[d] = day.set_index('sym')
    
    # 构建股价映射: (sym, date) -> close
    price_map = {}
    for d, day_df in date_data.items():
        for sym, row in day_df.iterrows():
            price_map[(sym, d)] = row['close']
    
    # 调仓日
    rebal_dates = [d for d in dates if d in date_data]
    rebal_indices = [dates.index(d) for d in rebal_dates]
    
    equity = 100000.0
    equity_curve = []
    trades = []
    
    for i, rebal_idx in enumerate(rebal_indices):
        rd = dates[rebal_idx]
        
        # 市场过滤
        if market_filter:
            state = get_market_state(rd)
            if state == 'bear':
                equity_curve.append((rd, equity))
                continue
            elif state in ['cautious', 'weak']:
                position_pct = 0.5
            else:
                position_pct = 1.0
        else:
            position_pct = 1.0
        
        # 选股
        day = date_data.get(rd)
        if day is None or len(day) < top_n:
            continue
        
        day_scored = score_optimized(day.reset_index())
        picks = day_scored.nlargest(top_n, 'score')
        selected_syms = picks['sym'].tolist()
        entry_prices = dict(zip(picks['sym'], picks['close']))
        
        # 扣除买入成本
        equity *= (1 - cost * position_pct)
        
        # 下一个调仓日
        next_rebal_idx = rebal_indices[i + 1] if i + 1 < len(rebal_indices) else len(dates) - 1
        
        # 持有期内每日跟踪
        active_positions = {sym: entry_prices[sym] for sym in selected_syms}
        
        for day_idx in range(rebal_idx + 1, next_rebal_idx + 1):
            hd = dates[day_idx]
            
            if hd not in date_data:
                equity_curve.append((hd, equity))
                continue
            
            day_prices = date_data[hd]
            
            daily_rets = []
            stopped_out = []
            
            for sym, entry_p in list(active_positions.items()):
                if sym in day_prices.index:
                    curr_p = day_prices.loc[sym, 'close']
                    cum_ret = curr_p / entry_p - 1
                    
                    if cum_ret <= stop_loss:
                        # 止损：锁定亏损
                        trades.append({'sym': sym, 'date': hd, 'return': stop_loss, 'type': 'stop_loss'})
                        stopped_out.append(sym)
                    else:
                        # 正常持有
                        if len(equity_curve) > 0 and equity_curve[-1][0] == dates[day_idx - 1]:
                            # 用前一天的价格计算日收益
                            prev_date = dates[day_idx - 1]
                            if (sym, prev_date) in price_map:
                                prev_p = price_map[(sym, prev_date)]
                                daily_ret = curr_p / prev_p - 1
                            else:
                                daily_ret = 0
                        else:
                            daily_ret = 0
                        daily_rets.append(daily_ret)
                else:
                    daily_rets.append(0)
            
            # 移除止损股票
            for sym in stopped_out:
                del active_positions[sym]
            
            # 更新权益
            if daily_rets:
                port_ret = np.mean(daily_rets) * position_pct
                equity *= (1 + port_ret)
            
            equity_curve.append((hd, equity))
        
        # 卖出剩余持仓（调仓）
        for sym in active_positions:
            if (sym, dates[next_rebal_idx]) in price_map:
                exit_p = price_map[(sym, dates[next_rebal_idx])]
                entry_p = active_positions[sym]
                ret = exit_p / entry_p - 1
                trades.append({'sym': sym, 'date': dates[next_rebal_idx], 'return': ret, 'type': 'normal'})
        
        # 扣除卖出成本
        equity *= (1 - cost * position_pct)
    
    return trades, equity_curve

# ============================================================
# 5. 运行回测
# ============================================================
print("\n[2] 运行每日精确回测 (2020-2026)...")
test_dates = sorted(df[(df['date'] >= 20200101) & (df['date'] <= 20260616)]['date'].unique())
df_test = df[(df['date'] >= 20200101) & (df['date'] <= 20260616)]

print(f"  测试期: {test_dates[0]}~{test_dates[-1]}, {len(test_dates)}天")

# 配置矩阵
configs = [
    ('SL3% + 市场过滤', -0.03, True),
    ('SL5% + 市场过滤', -0.05, True),
    ('SL8% + 市场过滤', -0.08, True),
    ('SL3% 无过滤', -0.03, False),
    ('无SL + 市场过滤', None, True),
]

results = {}
for name, sl, mkt_filter in configs:
    print(f"\n  运行: {name}...")
    t1 = time.time()
    trades, eq_curve = run_daily_backtest(df_test, test_dates, hold_days=10, top_n=15, 
                                           stop_loss=sl, cost=0.003, market_filter=mkt_filter)
    
    if not eq_curve:
        print(f"    无交易")
        continue
    
    eq_arr = np.array([e[1] for e in eq_curve])
    dates_arr = np.array([e[0] for e in eq_curve])
    
    # 每日收益率
    daily_rets = np.diff(eq_arr) / eq_arr[:-1]
    
    # 最大回撤（每日）
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    max_dd = dd.min()
    
    # 找回撤期
    dd_periods = []
    dd_start = None
    for k in range(len(dd)):
        if dd[k] < -0.001 and dd_start is None:
            dd_start = k
        elif dd[k] >= -0.001 and dd_start is not None:
            dd_periods.append((dates_arr[dd_start], dates_arr[k-1], dd[dd_start:k].min()))
            dd_start = None
    
    # CAGR
    years = len(eq_arr) / 252
    total_ret = eq_arr[-1] / eq_arr[0] - 1
    cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
    
    # Sharpe
    ann_ret = daily_rets.mean() * 252
    ann_std = daily_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0
    
    # Sortino
    downside = daily_rets[daily_rets < 0]
    downside_std = downside.std() if len(downside) > 0 else 0
    sortino = ann_ret / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
    
    # 交易统计
    trade_rets = np.array([t['return'] for t in trades])
    win_rate = (trade_rets > 0).mean()
    avg_win = trade_rets[trade_rets > 0].mean() if (trade_rets > 0).any() else 0
    avg_loss = trade_rets[trade_rets < 0].mean() if (trade_rets < 0).any() else 0
    sl_trades = [t for t in trades if t['type'] == 'stop_loss']
    sl_rate = len(sl_trades) / len(trades) if trades else 0
    
    results[name] = {
        'cagr': cagr, 'sharpe': sharpe, 'sortino': sortino, 'max_dd': max_dd,
        'win_rate': win_rate, 'avg_win': avg_win, 'avg_loss': avg_loss,
        'pl_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
        'trades': len(trades), 'sl_rate': sl_rate,
        'final_equity': eq_arr[-1],
        'dd_periods': sorted(dd_periods, key=lambda x: x[2])[:5],
    }
    
    print(f"    CAGR: {cagr:.2%} | Sharpe: {sharpe:.2f} | DD: {max_dd:.2%} | 胜率: {win_rate:.1%} | SL率: {sl_rate:.1%} | {time.time()-t1:.0f}秒")

# ============================================================
# 6. 结果汇总
# ============================================================
print("\n" + "="*80)
print("📊 rule-alpha-v1.0 每日精确回撤验证结果")
print("="*80)

print(f"\n{'配置':<25} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'胜率':>7} {'PL比':>6} {'SL率':>6}")
print("-"*80)
for name, r in results.items():
    print(f"{name:<25} {r['cagr']:>7.1%} {r['sharpe']:>8.2f} {r['sortino']:>8.2f} {r['max_dd']:>7.1%} "
          f"{r['win_rate']:>6.1%} {r['pl_ratio']:>6.2f} {r['sl_rate']:>5.1%}")

# 最佳配置详情
best_name = max(results, key=lambda x: results[x]['sharpe'])
best = results[best_name]
print(f"\n🏆 最佳配置: {best_name}")
print(f"  CAGR: {best['cagr']:.2%}")
print(f"  Sharpe: {best['sharpe']:.2f}")
print(f"  Sortino: {best['sortino']:.2f}")
print(f"  MaxDD (daily): {best['max_dd']:.2%}")
print(f"  最终权益: {best['final_equity']:,.0f}")

print(f"\n  Top 5 回撤期:")
for start, end, depth in best['dd_periods']:
    print(f"    {start}~{end}: {depth:.2%}")

# ============================================================
# 7. 保存结果
# ============================================================
output = {
    'version': 'rule-alpha-v1.0',
    'test_period': '2020-01-02 ~ 2026-06-16',
    'simulation': 'daily_exact',
    'configs': {}
}

for name, r in results.items():
    output['configs'][name] = {
        'cagr': round(r['cagr'], 4),
        'sharpe': round(r['sharpe'], 4),
        'sortino': round(r['sortino'], 4),
        'max_dd': round(r['max_dd'], 4),
        'win_rate': round(r['win_rate'], 4),
        'pl_ratio': round(r['pl_ratio'], 2),
        'trades': r['trades'],
        'sl_rate': round(r['sl_rate'], 4),
        'final_equity': round(r['final_equity'], 0),
    }

output_file = os.path.join(WORKSPACE, 'research', 'rule_alpha_v1_daily_sim.json')
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存: {output_file}")
