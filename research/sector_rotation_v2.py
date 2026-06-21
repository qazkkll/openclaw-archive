#!/usr/bin/env python3
"""
A股板块轮动策略 V2 — 使用验证过的回测框架
==========================================
使用 rule_daily_dd_v3.py 的回测逻辑（已被验证正确）
测试板块轮动策略
"""
import pandas as pd, numpy as np, json, time, os, datetime, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("🔄 A股板块轮动策略 V2 (验证框架)")
print("="*60)
t0 = time.time()

# ============================================================
# 1. 加载数据 (与rule_daily_dd_v3完全一致)
# ============================================================
print("加载数据...")
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

# 特征 (与验证框架一致)
df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['vol20'] = df.groupby('sym')['ret5'].transform(lambda x: x.rolling(4, min_periods=2).std())

delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)

for col in ['total_net', 'lg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())

# 市场状态

# 加载行业信息并merge
with open('data/config/stock_info.json') as f:
    info = json.load(f)
info_df = pd.DataFrame(info).T
info_df.index.name = 'sym'
info_df = info_df.reset_index()[['sym', 'industry']]
df = df.merge(info_df, on='sym', how='left')
df['industry'] = df['industry'].fillna('其他')

# 超级行业分组
industry_map = {
    '电气设备': '新能源', '元器件': '电子', '半导体': '电子', '电器仪表': '电子',
    '专用机械': '机械', '机械基件': '机械', '运输设备': '机械',
    '软件服务': 'IT', '互联网': 'IT', 'IT设备': 'IT',
    '汽车配件': '汽车', '化工原料': '化工', '塑料': '化工', '农药化肥': '化工',
    '医疗保健': '医药', '化学制药': '医药', '生物制药': '医药', '中成药': '医药',
    '通信设备': '通信', '建筑工程': '建筑', '环境保护': '环保',
    '食品': '消费', '家用电器': '消费', '服饰': '消费', '家居用品': '消费',
    '文教休闲': '消费', '广告包装': '消费',
    '证券': '金融', '银行': '金融', '保险': '金融', '多元金融': '金融',
    '小金属': '材料', '铝': '材料', '铜': '材料', '铅锌': '材料',
    '航空': '交通', '船舶': '交通', '公路': '交通', '铁路': '交通',
    '机场港口': '交通', '仓储物流': '交通',
    '房地产': '地产', '房产服务': '地产',
    '煤炭开采': '能源', '石油开采': '能源', '石油加工': '能源',
    '火力发电': '电力', '水力发电': '电力', '新型电力': '电力',
    '白酒': '白酒',
}
df['super_industry'] = df['industry'].map(industry_map).fillna('其他')

# 市场状态 (与验证框架一致)
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
print(f"  {len(df):,}行, {df['sym'].nunique()}只, {len(all_dates)}天")

# ============================================================
# 2. 评分函数 (基线)
# ============================================================
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

# ============================================================
# 3. 板块轮动评分函数
# ============================================================
def score_board_top(day):
    """选最强板块, 只在该板块内选股"""
    s = score_baseline(day)
    if 'board' not in s.columns or len(s) == 0:
        return s.head(0)
    board_avg = s.groupby('board')['score'].mean()
    if len(board_avg) == 0:
        return s.head(0)
    best = board_avg.idxmax()
    return s[s['board'] == best]

def score_industry_top2(day):
    """选最强2个超级行业"""
    s = score_baseline(day)
    if 'super_industry' not in s.columns or len(s) == 0:
        return s.head(0)
    ind_avg = s.groupby('super_industry')['score'].mean()
    if len(ind_avg) < 2:
        return s
    top2 = ind_avg.nlargest(2).index
    return s[s['super_industry'].isin(top2)]

def score_industry_top3(day):
    """选最强3个超级行业"""
    s = score_baseline(day)
    if 'super_industry' not in s.columns or len(s) == 0:
        return s.head(0)
    ind_avg = s.groupby('super_industry')['score'].mean()
    if len(ind_avg) < 3:
        return s
    top3 = ind_avg.nlargest(3).index
    return s[s['super_industry'].isin(top3)]

def score_industry_top5(day):
    """选最强5个超级行业"""
    s = score_baseline(day)
    if 'super_industry' not in s.columns or len(s) == 0:
        return s.head(0)
    ind_avg = s.groupby('super_industry')['score'].mean()
    if len(ind_avg) < 5:
        return s
    top5 = ind_avg.nlargest(5).index
    return s[s['super_industry'].isin(top5)]

def score_industry_diversified(day, max_per_ind=2):
    """行业分散: 每个行业最多max_per_ind只"""
    s = score_baseline(day)
    s = s.sort_values('score', ascending=False)
    if 'super_industry' not in s.columns:
        return s.head(15)
    result = []
    counts = {}
    for _, row in s.iterrows():
        ind = row['super_industry']
        if counts.get(ind, 0) < max_per_ind:
            result.append(row)
            counts[ind] = counts.get(ind, 0) + 1
        if len(result) >= 15:
            break
    return pd.DataFrame(result) if result else s.head(0)

def score_board_then_industry(day):
    """先选最强板块, 再在板块内选最强行业"""
    s = score_baseline(day)
    if 'board' not in s.columns or 'super_industry' not in s.columns:
        return s.head(0)
    board_avg = s.groupby('board')['score'].mean()
    if len(board_avg) == 0:
        return s.head(0)
    best_board = board_avg.idxmax()
    board_data = s[s['board'] == best_board]
    if len(board_data) == 0:
        return s.head(0)
    ind_avg = board_data.groupby('super_industry')['score'].mean()
    if len(ind_avg) < 2:
        return board_data
    top2 = ind_avg.nlargest(2).index
    return board_data[board_data['super_industry'].isin(top2)]

def score_exclude_bottom_industry(day):
    """排除最弱行业, 其余选股"""
    s = score_baseline(day)
    if 'super_industry' not in s.columns or len(s) == 0:
        return s
    ind_avg = s.groupby('super_industry')['score'].mean()
    if len(ind_avg) < 3:
        return s
    worst = ind_avg.idxmin()
    return s[s['super_industry'] != worst]

def score_weighted_by_industry(day):
    """行业加权: 最强行业股票评分+1, 最弱行业-1"""
    s = score_baseline(day)
    if 'super_industry' not in s.columns or len(s) == 0:
        return s
    ind_avg = s.groupby('super_industry')['score'].mean()
    if len(ind_avg) < 3:
        return s
    ind_rank = ind_avg.rank(pct=True)
    s['ind_bonus'] = s['super_industry'].map(ind_rank).fillna(0.5) * 2 - 1  # [-1, +1]
    s['score'] = s['score'] + s['ind_bonus']
    return s

# ============================================================
# 4. 回测框架 (与rule_daily_dd_v3完全一致)
# ============================================================
def run_backtest(df, score_fn, name, test_start=20200101, test_end=20260616,
                  hold_days=10, top_n=15, stop_loss=-0.03, cost=0.003,
                  use_market_filter=True):
    """验证过的回测框架"""
    df_test = df[(df['date'] >= test_start) & (df['date'] <= test_end)]
    test_dates = sorted(df_test['date'].unique())
    
    price_dict = {}
    for d in test_dates:
        day_data = df_test[df_test['date'] == d]
        price_dict[d] = dict(zip(day_data['sym'], day_data['close']))
    
    rebal_dates = test_dates[::hold_days]
    
    equity = 100000.0
    equity_curve = [(test_dates[0], equity)]
    trades = []
    
    for i, rd in enumerate(rebal_dates):
        if use_market_filter:
            state = market_state_map.get(rd, 'bull')
            if state == 'bear':
                next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else test_dates[-1]
                for d in test_dates:
                    if rd < d <= next_rd:
                        equity_curve.append((d, equity))
                continue
            elif state == 'cautious':
                position_pct = 0.5
            else:
                position_pct = 1.0
        else:
            position_pct = 1.0
        
        day = df_test[df_test['date'] == rd].copy()
        if len(day) < top_n:
            continue
        
        scored = score_fn(day)
        if len(scored) == 0:
            # No stocks pass filter, cash
            next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else test_dates[-1]
            for d in test_dates:
                if rd < d <= next_rd:
                    equity_curve.append((d, equity))
            continue
        
        picks = scored.nlargest(min(top_n, len(scored)), 'score')
        
        entry_prices = {}
        for _, row in picks.iterrows():
            entry_prices[row['sym']] = row['close']
        
        equity *= (1 - cost * position_pct)
        
        next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else test_dates[-1]
        hold_dates = [d for d in test_dates if rd < d <= next_rd]
        
        active_syms = set(entry_prices.keys())
        prev_day_prices = {sym: entry_prices[sym] for sym in active_syms}
        
        for hd in hold_dates:
            curr_prices = price_dict.get(hd, {})
            
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
                
                cum_ret = curr_p / entry_p - 1
                
                if stop_loss is not None and cum_ret <= stop_loss:
                    prev_cum = prev_p / entry_p - 1
                    if prev_cum <= stop_loss:
                        day_ret = 0
                    else:
                        day_ret = stop_loss - prev_cum
                        stopped_out.append(sym)
                else:
                    day_ret = curr_p / prev_p - 1 if prev_p > 0 else 0
                
                daily_port_ret += day_ret * weight_per_stock
                prev_day_prices[sym] = curr_p
            
            equity *= (1 + daily_port_ret)
            equity_curve.append((hd, equity))
            
            for sym in stopped_out:
                active_syms.discard(sym)
        
        equity *= (1 - cost * position_pct)
        
        for sym, entry_p in entry_prices.items():
            exit_p = price_dict.get(next_rd, {}).get(sym, entry_p)
            ret = exit_p / entry_p - 1
            if stop_loss is not None and ret < stop_loss:
                ret = stop_loss
            trades.append({'sym': sym, 'date': rd, 'return': ret - cost})
    
    # 计算指标
    eq_arr = np.array([e[1] for e in equity_curve])
    eq_dates = np.array([e[0] for e in equity_curve])
    daily_rets = np.diff(eq_arr) / eq_arr[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    max_dd = dd.min()
    
    dt1 = datetime.datetime.strptime(str(eq_dates[0]), '%Y%m%d')
    dt2 = datetime.datetime.strptime(str(eq_dates[-1]), '%Y%m%d')
    years = (dt2 - dt1).days / 365.25
    total_ret = eq_arr[-1] / eq_arr[0] - 1
    cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
    
    ann_ret = daily_rets.mean() * 252
    ann_std = daily_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0
    
    downside = daily_rets[daily_rets < 0]
    downside_std = downside.std() if len(downside) > 0 else 0
    sortino = ann_ret / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
    
    trade_rets = np.array([t['return'] for t in trades])
    win_rate = (trade_rets > 0).mean() if len(trade_rets) > 0 else 0
    avg_win = trade_rets[trade_rets > 0].mean() if (trade_rets > 0).any() else 0
    avg_loss = trade_rets[trade_rets < 0].mean() if (trade_rets < 0).any() else 0
    pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # 年度收益
    yearly = {}
    for year in range(2020, 2027):
        year_mask = (eq_dates // 10000 == year)
        if year_mask.any():
            year_eq = eq_arr[year_mask]
            yearly[year] = year_eq[-1] / year_eq[0] - 1
    
    return {
        'name': name, 'cagr': cagr, 'sharpe': sharpe, 'sortino': sortino,
        'max_dd': max_dd, 'win_rate': win_rate, 'pl_ratio': pl_ratio,
        'n_trades': len(trades), 'final_equity': eq_arr[-1],
        'yearly': yearly,
    }

# ============================================================
# 5. 运行所有策略
# ============================================================
print("\n" + "="*60)
print("运行回测 (2020-01 ~ 2026-06)")
print("="*60)

strategies = [
    # 基线 (已验证)
    (score_baseline, "A_baseline_SL3_MF", 15, 10, -0.03, True),
    (score_baseline, "B_baseline_SL2_MF", 15, 10, -0.02, True),
    
    # 板块轮动
    (score_board_top, "C_board_top_SL3", 15, 10, -0.03, True),
    (score_board_top, "D_board_top_SL2", 15, 10, -0.02, True),
    
    # 行业轮动 (Top N industries)
    (score_industry_top2, "E_ind_top2_SL3", 15, 10, -0.03, True),
    (score_industry_top3, "F_ind_top3_SL3", 15, 10, -0.03, True),
    (score_industry_top5, "G_ind_top5_SL3", 15, 10, -0.03, True),
    
    # 行业分散
    (lambda d: score_industry_diversified(d, 1), "H_div_max1_SL3", 15, 10, -0.03, True),
    (lambda d: score_industry_diversified(d, 2), "I_div_max2_SL3", 15, 10, -0.03, True),
    (lambda d: score_industry_diversified(d, 3), "J_div_max3_SL3", 15, 10, -0.03, True),
    
    # 板块+行业组合
    (score_board_then_industry, "K_b+ind_SL3", 15, 10, -0.03, True),
    
    # 排除最弱行业
    (score_exclude_bottom_industry, "L_excl_worst_SL3", 15, 10, -0.03, True),
    
    # 行业加权
    (score_weighted_by_industry, "M_ind_weight_SL3", 15, 10, -0.03, True),
    
    # 不同持有期
    (score_baseline, "N_baseline_5d_SL3", 15, 5, -0.03, True),
    (score_industry_top3, "O_ind3_5d_SL3", 15, 5, -0.03, True),
    (score_baseline, "P_baseline_20d_SL3", 15, 20, -0.03, True),
    (score_industry_top3, "Q_ind3_20d_SL3", 15, 20, -0.03, True),
    
    # 无市场过滤 (对照)
    (score_baseline, "R_baseline_SL3_noMF", 15, 10, -0.03, False),
    (score_industry_top3, "S_ind3_SL3_noMF", 15, 10, -0.03, False),
]

results = []
for score_fn, name, top_n, hold, sl, mf in strategies:
    print(f"  {name}...", end=" ", flush=True)
    t1 = time.time()
    try:
        r = run_backtest(df, score_fn, name, hold_days=hold, top_n=top_n,
                         stop_loss=sl, use_market_filter=mf)
        results.append(r)
        print(f"CAGR={r['cagr']:.1%} Sharpe={r['sharpe']:.2f} DD={r['max_dd']:.1%} ({time.time()-t1:.0f}s)")
    except Exception as e:
        print(f"❌ {e}")

# ============================================================
# 6. 结果汇总
# ============================================================
print("\n" + "="*70)
print("📊 板块轮动策略回测结果 (2020-01 ~ 2026-06)")
print("="*70)

results.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'策略':<25} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'胜率':>7} {'PL比':>6}")
print("-" * 75)
for r in results:
    print(f"{r['name']:<25} {r['cagr']:>7.1%} {r['sharpe']:>8.2f} {r['sortino']:>8.2f} {r['max_dd']:>7.1%} {r['win_rate']:>6.1%} {r['pl_ratio']:>6.2f}")

# 年度收益
print(f"\n📈 年度收益 (Top 8):")
years = sorted(set().union(*[set(r['yearly'].keys()) for r in results]))
print(f"{'策略':<25}", end="")
for y in years:
    print(f" {y:>6}", end="")
print()
print("-" * (25 + 7 * len(years)))
for r in results[:8]:
    print(f"{r['name']:<25}", end="")
    for y in years:
        ret = r['yearly'].get(y, 0)
        print(f" {ret:>5.0%}", end="")
    print()

# ============================================================
# 7. 分析: 哪个行业表现最好
# ============================================================
print("\n" + "="*70)
print("📊 行业表现分析 (2020-2026)")
print("="*70)

# 计算各超级行业的年度收益
industry_perf = {}
for year in range(2020, 2027):
    year_data = df[(df['date'] // 10000 == year)]
    if len(year_data) == 0:
        continue
    
    # 每个行业的平均年度收益
    year_dates = sorted(year_data['date'].unique())
    first_d = year_dates[0]
    last_d = year_dates[-1]
    
    first_prices = year_data[year_data['date'] == first_d].set_index('sym')['close']
    last_prices = year_data[year_data['date'] == last_d].set_index('sym')['close']
    
    # 需要行业信息
    year_with_ind = year_data[year_data['date'].isin([first_d, last_d])].copy()
    year_with_ind = year_with_ind.merge(info_df, on='sym', how='left')
    year_with_ind['super_industry'] = year_with_ind['industry'].map(industry_map).fillna('其他')
    
    first_with_ind = year_with_ind[year_with_ind['date'] == first_d].set_index('sym')
    last_with_ind = year_with_ind[year_with_ind['date'] == last_d].set_index('sym')
    
    common = first_with_ind.index.intersection(last_with_ind.index)
    rets = (last_with_ind.loc[common, 'close'] / first_with_ind.loc[common, 'close'] - 1)
    inds = first_with_ind.loc[common, 'super_industry']
    
    ind_rets = pd.DataFrame({'ret': rets, 'industry': inds})
    ind_avg = ind_rets.groupby('industry')['ret'].mean()
    
    for ind, ret in ind_avg.items():
        if ind not in industry_perf:
            industry_perf[ind] = {}
        industry_perf[ind][year] = ret

# 显示前10行业
ind_df = pd.DataFrame(industry_perf).T
ind_df['avg'] = ind_df.mean(axis=1)
ind_df = ind_df.sort_values('avg', ascending=False)

print(f"\n{'行业':<12}", end="")
for y in years:
    print(f" {y:>6}", end="")
print(f" {'均值':>6}")
print("-" * (12 + 7 * (len(years) + 1)))
for ind, row in ind_df.head(15).iterrows():
    print(f"{ind:<12}", end="")
    for y in years:
        ret = row.get(y, 0)
        if pd.isna(ret):
            print(f"   N/A", end="")
        else:
            print(f" {ret:>5.0%}", end="")
    avg = row['avg']
    print(f" {avg:>5.0%}" if not pd.isna(avg) else f"   N/A")

# ============================================================
# 8. 保存
# ============================================================
output = {
    'experiment': 'sector_rotation_v2',
    'date': '2026-06-21',
    'test_period': '2020-01 ~ 2026-06',
    'framework': 'validated (rule_daily_dd_v3)',
    'results': []
}
for r in results:
    output['results'].append({
        'name': r['name'],
        'cagr': round(r['cagr'], 4),
        'sharpe': round(r['sharpe'], 4),
        'sortino': round(r['sortino'], 4),
        'max_dd': round(r['max_dd'], 4),
        'win_rate': round(r['win_rate'], 4),
        'pl_ratio': round(r['pl_ratio'], 2),
        'n_trades': r['n_trades'],
        'final_equity': round(r['final_equity'], 0),
        'yearly': {str(k): round(v, 4) for k, v in r['yearly'].items()},
    })

with open('research/sector_rotation_v2_results.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ 结果已保存 research/sector_rotation_v2_results.json")
print(f"⏱️ 总耗时: {time.time()-t0:.0f}秒")
