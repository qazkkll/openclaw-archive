#!/usr/bin/env python3
"""
rule-alpha-v1.0 — 每日回撤计算（修正版）
方法: 用每日价格直接计算持仓收益
"""
import pandas as pd, numpy as np, json, time, os, datetime
import warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("="*60)
print("rule-alpha-v1.0 — 每日回撤验证（修正版）")
print("="*60)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[1] 加载数据...")
t0 = time.time()

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

# 特征
df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['vol20'] = df.groupby('sym')['ret5'].transform(lambda x: x.rolling(4, min_periods=2).std())

delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)

for col in ['total_net', 'lg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())

# 市场状态
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')
market_avg_r20 = df.groupby('date')['mkt_ret20'].first()
market_ma60 = market_avg_r20.rolling(60, min_periods=1).mean()
market_ma120 = market_avg_r20.rolling(120, min_periods=1).mean()

market_state_map = {}
for d in sorted(df['date'].unique()):
    r20 = market_avg_r20.get(d, 0) if d in market_avg_r20.index else 0
    ma60 = market_ma60.get(d, 0) if d in market_ma60.index else 0
    ma120 = market_ma120.get(d, 0) if d in market_ma120.index else 0
    ma_bull = ma60 > ma120
    mom_pos = r20 > 0
    if not ma_bull and not mom_pos:
        market_state_map[d] = 'bear'
    elif not ma_bull or not mom_pos:
        market_state_map[d] = 'cautious'
    else:
        market_state_map[d] = 'bull'

all_dates = sorted(df['date'].unique())
print(f"  {len(df):,}行, {df['sym'].nunique()}只, {time.time()-t0:.0f}秒")

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
# 3. 每日精确回测
# ============================================================
print("\n[2] 运行每日精确回测...")

def run_daily_backtest(df_all, test_start=20200101, test_end=20260616, 
                        hold_days=10, top_n=15, stop_loss=-0.03, cost=0.003,
                        use_market_filter=True):
    """每日精确回测 — 正确的实现"""
    
    # 测试期数据
    df_test = df_all[(df_all['date'] >= test_start) & (df_all['date'] <= test_end)]
    test_dates = sorted(df_test['date'].unique())
    
    # 构建价格矩阵: date -> sym -> close
    price_dict = {}
    for d in test_dates:
        day_data = df_test[df_test['date'] == d]
        price_dict[d] = dict(zip(day_data['sym'], day_data['close']))
    
    # 调仓日
    rebal_dates = test_dates[::hold_days]
    
    equity = 100000.0
    equity_curve = [(test_dates[0], equity)]
    trades = []
    
    for i, rd in enumerate(rebal_dates):
        # 市场过滤
        if use_market_filter:
            state = market_state_map.get(rd, 'bull')
            if state == 'bear':
                # 空仓，equity不变
                next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else test_dates[-1]
                for d in test_dates:
                    if rd < d <= next_rd:
                        equity_curve.append((d, equity))
                continue
            elif state in ['cautious']:
                position_pct = 0.5
            else:
                position_pct = 1.0
        else:
            position_pct = 1.0
        
        # 选股
        day = df_test[df_test['date'] == rd].copy()
        if len(day) < top_n:
            continue
        day = score_optimized(day)
        picks = day.nlargest(top_n, 'score')
        
        # 入场价格
        entry_prices = {}
        for _, row in picks.iterrows():
            entry_prices[row['sym']] = row['close']
        
        # 扣除买入成本
        equity *= (1 - cost * position_pct)
        
        # 持有期
        next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else test_dates[-1]
        hold_dates = [d for d in test_dates if rd < d <= next_rd]
        
        # 每日跟踪
        active_syms = set(entry_prices.keys())
        prev_day_prices = {sym: entry_prices[sym] for sym in active_syms}
        
        for hd in hold_dates:
            curr_prices = price_dict.get(hd, {})
            
            # 计算当日组合收益
            daily_port_ret = 0.0
            n_active = len(active_syms)
            if n_active == 0:
                equity_curve.append((hd, equity))
                continue
            
            weight_per_stock = position_pct / n_active
            stopped_out = []
            
            for sym in list(active_syms):
                if sym not in curr_prices:
                    continue
                    
                curr_p = curr_prices[sym]
                entry_p = entry_prices[sym]
                prev_p = prev_day_prices.get(sym, entry_p)
                
                # 累计收益（从入场价）
                cum_ret = curr_p / entry_p - 1
                
                if stop_loss is not None and cum_ret <= stop_loss:
                    # 止损: 计算从入场到止损的损失
                    # 但只计算从昨天到今天的变化
                    if prev_p > 0:
                        # 今天相对昨天的收益，但整体不超过SL
                        prev_cum = prev_p / entry_p - 1
                        if prev_cum <= stop_loss:
                            # 昨天已经止损了，今天无变化
                            day_ret = 0
                        else:
                            # 今天触发止损
                            day_ret = stop_loss - prev_cum
                            stopped_out.append(sym)
                    else:
                        day_ret = 0
                else:
                    # 正常持有
                    day_ret = curr_p / prev_p - 1 if prev_p > 0 else 0
                
                daily_port_ret += day_ret * weight_per_stock
                prev_day_prices[sym] = curr_p
            
            # 更新权益
            equity *= (1 + daily_port_ret)
            equity_curve.append((hd, equity))
            
            # 移除止损股票
            for sym in stopped_out:
                active_syms.discard(sym)
        
        # 卖出剩余持仓
        equity *= (1 - cost * position_pct)
        
        # 记录交易
        for sym, entry_p in entry_prices.items():
            exit_p = price_dict.get(next_rd, {}).get(sym, entry_p)
            ret = exit_p / entry_p - 1
            if stop_loss is not None and ret < stop_loss:
                ret = stop_loss
            trades.append({'sym': sym, 'date': rd, 'return': ret - cost})
    
    return trades, equity_curve

# ============================================================
# 4. 运行测试
# ============================================================
configs = [
    ('SL3%+MF', -0.03, True),
    ('SL5%+MF', -0.05, True),
    ('SL8%+MF', -0.08, True),
    ('SL3%', -0.03, False),
    ('SL5%', -0.05, False),
    ('无SL+MF', None, True),
]

results = {}
for name, sl, mf in configs:
    print(f"\n  运行 {name}...")
    t1 = time.time()
    trades, eq_curve = run_daily_backtest(df, stop_loss=sl, use_market_filter=mf)
    
    eq_arr = np.array([e[1] for e in eq_curve])
    eq_dates = np.array([e[0] for e in eq_curve])
    
    # 每日收益
    daily_rets = np.diff(eq_arr) / eq_arr[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    
    # 最大回撤
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    max_dd = dd.min()
    
    # CAGR
    dt1 = datetime.datetime.strptime(str(eq_dates[0]), '%Y%m%d')
    dt2 = datetime.datetime.strptime(str(eq_dates[-1]), '%Y%m%d')
    years = (dt2 - dt1).days / 365.25
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
    
    # Top 5 回撤期
    dd_periods = []
    dd_start = None
    for k in range(len(dd)):
        if dd[k] < -0.005 and dd_start is None:
            dd_start = k
        elif dd[k] >= -0.005 and dd_start is not None:
            dd_periods.append((eq_dates[dd_start], eq_dates[k-1], dd[dd_start:k].min()))
            dd_start = None
    dd_periods.sort(key=lambda x: x[2])
    
    results[name] = {
        'cagr': cagr, 'sharpe': sharpe, 'sortino': sortino, 'max_dd': max_dd,
        'win_rate': win_rate, 'avg_win': avg_win, 'avg_loss': avg_loss,
        'pl_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
        'trades': len(trades), 'final_equity': eq_arr[-1],
        'dd_periods': dd_periods[:5],
    }
    
    print(f"    CAGR: {cagr:.2%} | Sharpe: {sharpe:.2f} | DD: {max_dd:.2%} | 胜率: {win_rate:.1%} | {time.time()-t1:.0f}秒")

# ============================================================
# 5. 结果汇总
# ============================================================
print("\n" + "="*80)
print("📊 rule-alpha-v1.0 每日精确回撤验证结果")
print("="*80)

print(f"\n{'配置':<15} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'胜率':>7} {'PL比':>6} {'交易':>6}")
print("-"*70)
for name, r in results.items():
    print(f"{name:<15} {r['cagr']:>7.1%} {r['sharpe']:>8.2f} {r['sortino']:>8.2f} {r['max_dd']:>7.1%} "
          f"{r['win_rate']:>6.1%} {r['pl_ratio']:>6.2f} {r['trades']:>6}")

# 最佳配置
best_name = max(results, key=lambda x: results[x]['sharpe'])
best = results[best_name]
print(f"\n🏆 最佳配置: {best_name}")
print(f"  CAGR: {best['cagr']:.2%}")
print(f"  Sharpe: {best['sharpe']:.2f}")
print(f"  Sortino: {best['sortino']:.2f}")
print(f"  MaxDD (daily): {best['max_dd']:.2%}")
print(f"  最终权益: {best['final_equity']:,.0f}")
print(f"\n  Top 5 回撤期:")
for s, e, d in best['dd_periods']:
    print(f"    {s}~{e}: {d:.2%}")

# 年度回撤（对最佳配置重新跑）
print(f"\n  年度最大回撤:")
best_trades, best_eq = run_daily_backtest(df, stop_loss=float(best_name.split('%')[0].replace('SL',''))/100 if 'SL' in best_name else None,
                                            use_market_filter='+MF' in best_name)
best_eq_arr = np.array([e[1] for e in best_eq])
best_eq_dates = np.array([e[0] for e in best_eq])
best_peak = np.maximum.accumulate(best_eq_arr)
best_dd = (best_eq_arr - best_peak) / best_peak

for year in range(2020, 2027):
    year_mask = (best_eq_dates // 10000 == year)
    if year_mask.any():
        year_dd = best_dd[year_mask].min()
        print(f"    {year}: {year_dd:.2%}")

# ============================================================
# 6. 保存
# ============================================================
output = {
    'version': 'rule-alpha-v1.0',
    'test_period': '2020-01-02 ~ 2026-06-16',
    'simulation': 'daily_exact_v2',
    'configs': {}
}
for name, r in results.items():
    output['configs'][name] = {
        'cagr': round(r['cagr'], 4), 'sharpe': round(r['sharpe'], 4),
        'sortino': round(r['sortino'], 4), 'max_dd': round(r['max_dd'], 4),
        'win_rate': round(r['win_rate'], 4), 'pl_ratio': round(r['pl_ratio'], 2),
        'trades': r['trades'], 'final_equity': round(r['final_equity'], 0),
    }

with open('research/rule_alpha_v1_daily_dd_v2.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存: research/rule_alpha_v1_daily_dd_v2.json")
