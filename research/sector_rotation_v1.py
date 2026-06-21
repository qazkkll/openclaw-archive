#!/usr/bin/env python3
"""
A股板块轮动策略 V1 — CEO实验
=============================
测试三种板块轮动方案:
1. Board rotation: 主板/创业板/科创板, 按动量切换
2. Industry group rotation: 20个超级行业, 按动量切换
3. Industry-filtered stock selection: rule-alpha + 行业过滤

对比基线: rule-alpha-v1.0 (无板块轮动)
"""
import pandas as pd, numpy as np, json, time, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

import os
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))
np.random.seed(42)

print("🔄 A股板块轮动策略 V1 实验")
print("="*60)
t0 = time.time()

# ============================================================
# 1. 加载数据
# ============================================================
print("加载数据...")
df = pd.read_parquet('data/a_hist_10y.parquet')
df = df.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df['date'] = df['date'].astype(int)

# 加载资金流
mf = pd.read_parquet('data/cn/moneyflow_core.parquet')
mf['sym'] = mf['ts_code'].str[:6]
mf['date'] = mf['trade_date'].astype(int)
for col in ['sm', 'md', 'lg', 'elg']:
    mf[f'{col}_net'] = mf[f'buy_{col}_amount'] - mf[f'sell_{col}_amount']
mf['total_net'] = mf['net_mf_amount']
df = df.merge(mf[['sym','date','total_net','lg_net','md_net','elg_net']], on=['sym','date'], how='left')

# 加载行业信息
with open('data/config/stock_info.json') as f:
    info = json.load(f)
info_df = pd.DataFrame(info).T
info_df.index.name = 'sym'
info_df = info_df.reset_index()[['sym', 'industry', 'market']]

df = df.merge(info_df, on='sym', how='left')

# 过滤
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
df = df[df['volume'] > 0].copy()
df = df.sort_values(['sym', 'date']).reset_index(drop=True)

print(f"  数据: {len(df):,}行, {df['sym'].nunique()}只, {df['date'].nunique()}天")

# ============================================================
# 2. 特征计算
# ============================================================
print("计算特征...")

# 收益率
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ret10'] = df.groupby('sym')['close'].pct_change(10)
df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ret60'] = df.groupby('sym')['close'].pct_change(60)

# 均线偏离
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']

# 波动率
df['vol20'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(20, min_periods=2).std())

# RSI
delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)

# 资金流5日聚合
for col in ['total_net', 'lg_net', 'md_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())

# 板块标记
def get_board(code):
    if code.startswith('300') or code.startswith('301'):
        return '创业板'
    elif code.startswith('688') or code.startswith('689'):
        return '科创板'
    elif code.startswith('00') or code.startswith('60'):
        return '主板'
    else:
        return '其他'

df['board'] = df['sym'].apply(get_board)

# 超级行业分组 (110→20)
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
    '农林牧渔': '农业', '种植业': '农业', '养殖业': '农业', '饲料': '农业',
    '白酒': '白酒', '啤酒': '饮料', '软饮料': '饮料', '乳制品': '饮料',
    '出版业': '传媒', '影视音像': '传媒', '游戏': '传媒',
    '水泥': '建材', '玻璃': '建材', '陶瓷': '建材',
    '百货': '零售', '超市连锁': '零售', '电器连锁': '零售',
}
df['super_industry'] = df['industry'].map(industry_map).fillna('其他')

print(f"  超级行业: {df['super_industry'].nunique()}个")
print(f"  板块: {df['board'].value_counts().to_dict()}")

# ============================================================
# 3. 规则型评分函数 (基线)
# ============================================================
def score_rule_alpha(day):
    """rule-alpha-v1.0 评分函数"""
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
# 4. 回测框架
# ============================================================
def backtest_strategy(df, strategy_fn, strategy_name, 
                      top_n=15, hold_days=10, stop_loss=-0.03,
                      position_map={'bull': 1.0, 'cautious': 0.3, 'bear': 0},
                      test_start=20200101, cost=0.003):
    """
    标准回测框架:
    - 每hold_days天调仓
    - Top-N等权持仓
    - 止损
    - 市场过滤器
    """
    dates = sorted(df[df['date'] >= test_start]['date'].unique())
    trade_dates = sorted(df['date'].unique())
    
    # 市场状态判断
    mkt_ret = df.groupby('date')['ret20'].mean()
    mkt_ma20 = df.groupby('date')['ma20_bias'].mean()
    
    # 计算MA60/MA120
    mkt_daily = mkt_ret.sort_index()
    mkt_ma60 = mkt_daily.rolling(60, min_periods=20).mean()
    mkt_ma120 = mkt_daily.rolling(120, min_periods=40).mean()
    
    # 宽度
    breadth_series = df.groupby('date')['ret5'].apply(lambda x: (x > 0).mean())
    
    # 调仓日
    rebal_dates = dates[::hold_days]
    
    equity = 100000
    positions = {}  # sym -> (entry_price, entry_date, shares)
    daily_equity = []
    trades = []
    
    for i, d in enumerate(dates):
        # 当日数据
        day_data = df[df['date'] == d]
        if len(day_data) == 0:
            continue
        
        # 市场状态
        ma60_val = mkt_ma60.get(d, 0)
        ma120_val = mkt_ma120.get(d, 0)
        ret20_val = mkt_ret.get(d, 0)
        breadth_val = breadth_series.get(d, 0.5)
        
        ma_bull = ma60_val > ma120_val
        mom_pos = ret20_val > 0
        breadth_ok = breadth_val > 0.4
        
        if not ma_bull and not mom_pos:
            regime = 'bear'
        elif not ma_bull or not mom_pos:
            regime = 'cautious'
        elif not breadth_ok:
            regime = 'weak'
        else:
            regime = 'bull'
        
        pos_pct = position_map.get(regime, 0)
        
        # 止损检查
        to_remove = []
        for sym, (entry_p, entry_d, shares) in positions.items():
            sym_data = day_data[day_data['sym'] == sym]
            if len(sym_data) > 0:
                current_p = sym_data.iloc[0]['close']
                ret = (current_p - entry_p) / entry_p
                if stop_loss and ret <= stop_loss:
                    # 止损
                    pnl = shares * (current_p - entry_p) - abs(shares * current_p * cost)
                    equity += shares * current_p - abs(shares * current_p * cost)
                    trades.append({'sym': sym, 'entry': entry_p, 'exit': current_p, 
                                   'ret': ret, 'pnl': pnl, 'reason': 'stop_loss', 'date': d})
                    to_remove.append(sym)
        for sym in to_remove:
            del positions[sym]
        
        # 调仓日
        if d in rebal_dates:
            # 卖出所有持仓
            for sym, (entry_p, entry_d, shares) in positions.items():
                sym_data = day_data[day_data['sym'] == sym]
                if len(sym_data) > 0:
                    exit_p = sym_data.iloc[0]['close']
                    ret = (exit_p - entry_p) / entry_p
                    pnl = shares * (exit_p - entry_p) - abs(shares * exit_p * cost)
                    equity += shares * exit_p - abs(shares * exit_p * cost)
                    trades.append({'sym': sym, 'entry': entry_p, 'exit': exit_p,
                                   'ret': ret, 'pnl': pnl, 'reason': 'rebalance', 'date': d})
            positions = {}
            
            # 选股
            if pos_pct > 0:
                scored = strategy_fn(day_data)
                scored = scored[scored['close'] >= 3]
                scored = scored[~scored['sym'].str.contains('ST|退市', na=False)]
                scored = scored[scored['volume'] > 0]
                
                if 'score' in scored.columns:
                    top = scored.nlargest(top_n, 'score')
                else:
                    top = scored.head(top_n)
                
                # 买入
                cash_per_stock = equity * pos_pct / max(len(top), 1)
                for _, row in top.iterrows():
                    sym = row['sym']
                    price = row['close']
                    shares = int(cash_per_stock / price / 100) * 100  # 整手
                    if shares > 0:
                        cost_buy = shares * price * cost
                        equity -= (shares * price + cost_buy)
                        positions[sym] = (price, d, shares)
        
        # 更新每日权益
        pos_value = 0
        for sym, (entry_p, entry_d, shares) in positions.items():
            sym_data = day_data[day_data['sym'] == sym]
            if len(sym_data) > 0:
                pos_value += shares * sym_data.iloc[0]['close']
            else:
                pos_value += shares * entry_p  # 用入场价近似
        
        total_equity = equity + pos_value
        daily_equity.append({'date': d, 'equity': total_equity, 'regime': regime, 
                            'n_positions': len(positions), 'pos_pct': pos_pct})
    
    # 清仓
    if positions:
        last_day = df[df['date'] == dates[-1]]
        for sym, (entry_p, entry_d, shares) in positions.items():
            sym_data = last_day[last_day['sym'] == sym]
            if len(sym_data) > 0:
                exit_p = sym_data.iloc[0]['close']
                equity += shares * exit_p - abs(shares * exit_p * cost)
    
    # 计算指标
    eq_df = pd.DataFrame(daily_equity)
    if len(eq_df) == 0:
        return None
    
    eq_df['date_dt'] = pd.to_datetime(eq_df['date'].astype(str))
    eq_df = eq_df.set_index('date_dt')
    eq_df['ret'] = eq_df['equity'].pct_change()
    
    final = eq_df['equity'].iloc[-1]
    initial = eq_df['equity'].iloc[0]
    days = (eq_df.index[-1] - eq_df.index[0]).days
    years = days / 365.25
    
    cagr = (final / initial) ** (1 / years) - 1 if years > 0 else 0
    sharpe = eq_df['ret'].mean() / eq_df['ret'].std() * np.sqrt(252) if eq_df['ret'].std() > 0 else 0
    
    # Sortino
    neg_ret = eq_df['ret'][eq_df['ret'] < 0]
    sortino = eq_df['ret'].mean() / neg_ret.std() * np.sqrt(252) if len(neg_ret) > 0 and neg_ret.std() > 0 else 0
    
    # Max DD
    peak = eq_df['equity'].cummax()
    dd = (eq_df['equity'] - peak) / peak
    max_dd = dd.min()
    
    # Trade stats
    trade_df = pd.DataFrame(trades)
    if len(trade_df) > 0:
        win_rate = (trade_df['ret'] > 0).mean()
        avg_win = trade_df[trade_df['ret'] > 0]['ret'].mean() if (trade_df['ret'] > 0).any() else 0
        avg_loss = trade_df[trade_df['ret'] <= 0]['ret'].mean() if (trade_df['ret'] <= 0).any() else 0
        pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 999
        n_trades = len(trade_df)
        sl_trades = (trade_df['reason'] == 'stop_loss').sum()
    else:
        win_rate = avg_win = avg_loss = pl_ratio = 0
        n_trades = sl_trades = 0
    
    # 年度收益
    eq_df['year'] = eq_df.index.year
    yearly = eq_df.groupby('year')['equity'].agg(['first', 'last'])
    yearly['return'] = yearly['last'] / yearly['first'] - 1
    
    return {
        'name': strategy_name,
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'pl_ratio': pl_ratio,
        'n_trades': n_trades,
        'sl_trades': sl_trades,
        'final_equity': final,
        'yearly_returns': yearly['return'].to_dict(),
        'daily_equity': eq_df,
    }

# ============================================================
# 5. 策略定义
# ============================================================

# --- 策略A: 基线 rule-alpha-v1.0 ---
def strategy_baseline(day):
    return score_rule_alpha(day)

# --- 策略B: Board rotation (选最强板块内的股票) ---
def strategy_board_rotation(day, lookback=20):
    """选近lookback天表现最强的板块, 只在该板块内选股"""
    scored = score_rule_alpha(day)
    
    # 计算各板块平均评分
    if 'board' not in scored.columns:
        return scored.head(0)
    
    board_scores = scored.groupby('board')['score'].mean()
    if len(board_scores) == 0:
        return scored.head(0)
    
    best_board = board_scores.idxmax()
    result = scored[scored['board'] == best_board]
    return result

# --- 策略C: Industry rotation (选最强2个超级行业) ---
def strategy_industry_rotation(day, n_industries=2):
    """选评分最高的n_industries个超级行业, 只在这些行业内选股"""
    scored = score_rule_alpha(day)
    
    if 'super_industry' not in scored.columns:
        return scored.head(0)
    
    # 行业平均评分
    ind_scores = scored.groupby('super_industry')['score'].mean()
    if len(ind_scores) < n_industries:
        return scored.head(0)
    
    top_industries = ind_scores.nlargest(n_industries).index
    result = scored[scored['super_industry'].isin(top_industries)]
    return result

# --- 策略D: Industry rotation (选最强3个超级行业) ---
def strategy_industry_rotation_3(day):
    return strategy_industry_rotation(day, n_industries=3)

# --- 策略E: 板块动量+行业动量混合 ---
def strategy_board_industry_combo(day):
    """先选最强板块, 再在板块内选最强行业"""
    scored = score_rule_alpha(day)
    
    if 'board' not in scored.columns or 'super_industry' not in scored.columns:
        return scored.head(0)
    
    # 选最强板块
    board_scores = scored.groupby('board')['score'].mean()
    best_board = board_scores.idxmax()
    board_data = scored[scored['board'] == best_board]
    
    # 在板块内选最强行业
    if len(board_data) == 0:
        return scored.head(0)
    
    ind_scores = board_data.groupby('super_industry')['score'].mean()
    if len(ind_scores) < 2:
        return board_data
    
    top_industries = ind_scores.nlargest(2).index
    result = board_data[board_data['super_industry'].isin(top_industries)]
    return result

# --- 策略F: 行业分散 (每个行业最多2只) ---
def strategy_industry_diversified(day, max_per_industry=2):
    """评分排序后, 每个超级行业最多选2只, 确保行业分散"""
    scored = score_rule_alpha(day)
    scored = scored.sort_values('score', ascending=False)
    
    if 'super_industry' not in scored.columns:
        return scored.head(15)
    
    # 逐行业限制
    result = []
    counts = {}
    for _, row in scored.iterrows():
        ind = row['super_industry']
        if counts.get(ind, 0) < max_per_industry:
            result.append(row)
            counts[ind] = counts.get(ind, 0) + 1
        if len(result) >= 15:
            break
    
    return pd.DataFrame(result)

# --- 策略G: Board timing (根据板块信号调整仓位) ---
def strategy_board_timing(day, top_n=15):
    """
    不改变选股逻辑, 但根据板块信号调整仓位:
    - 创业板动量强 → 满仓
    - 主板动量强 → 半仓
    - 都弱 → 空仓
    """
    scored = score_rule_alpha(day)
    
    # 板块动量
    if 'board' not in scored.columns:
        return scored.head(0)
    
    board_mom = scored.groupby('board')['ret20'].mean()
    
    # 创业板强 → 选股
    if board_mom.get('创业板', -1) > 0.02:
        return scored  # 全量选股
    elif board_mom.get('主板', -1) > 0.02:
        return scored  # 全量选股但仓位减半 (在回测中通过position_map控制)
    else:
        return scored.head(0)  # 空仓

# ============================================================
# 6. 运行回测
# ============================================================
print("\n" + "="*60)
print("运行回测 (2020-01 ~ 2026-06)")
print("="*60)

configs = [
    # (策略函数, 名称, top_n, hold_days, stop_loss, position_map)
    (strategy_baseline, "A_baseline", 15, 10, -0.03, {'bull': 1.0, 'cautious': 0.3, 'bear': 0}),
    (strategy_board_rotation, "B_board_rot", 15, 10, -0.03, {'bull': 1.0, 'cautious': 0.3, 'bear': 0}),
    (strategy_industry_rotation, "C_ind_rot2", 15, 10, -0.03, {'bull': 1.0, 'cautious': 0.3, 'bear': 0}),
    (strategy_industry_rotation_3, "D_ind_rot3", 15, 10, -0.03, {'bull': 1.0, 'cautious': 0.3, 'bear': 0}),
    (strategy_board_industry_combo, "E_board+ind", 15, 10, -0.03, {'bull': 1.0, 'cautious': 0.3, 'bear': 0}),
    (strategy_industry_diversified, "F_ind_div", 15, 10, -0.03, {'bull': 1.0, 'cautious': 0.3, 'bear': 0}),
    # 板块timing with different position maps
    (strategy_baseline, "G_baseline_mf", 15, 10, -0.02, {'bull': 1.0, 'cautious': 0.3, 'bear': 0}),
    # 不同持有期
    (strategy_baseline, "H_hold5d", 15, 5, -0.03, {'bull': 1.0, 'cautious': 0.3, 'bear': 0}),
    (strategy_industry_diversified, "I_ind_div_5d", 15, 5, -0.03, {'bull': 1.0, 'cautious': 0.3, 'bear': 0}),
    # 行业分散+不同max_per_industry
    (lambda d: strategy_industry_diversified(d, max_per_industry=1), "J_ind_max1", 15, 10, -0.03, {'bull': 1.0, 'cautious': 0.3, 'bear': 0}),
    (lambda d: strategy_industry_diversified(d, max_per_industry=3), "K_ind_max3", 15, 10, -0.03, {'bull': 1.0, 'cautious': 0.3, 'bear': 0}),
]

results = []
for fn, name, top_n, hold, sl, pos_map in configs:
    print(f"\n  运行 {name}...")
    t1 = time.time()
    try:
        r = backtest_strategy(df, fn, name, 
                              top_n=top_n, hold_days=hold, stop_loss=sl,
                              position_map=pos_map, test_start=20200101, cost=0.003)
        if r:
            results.append(r)
            print(f"    ✅ CAGR={r['cagr']:.1%} Sharpe={r['sharpe']:.2f} DD={r['max_dd']:.1%} WR={r['win_rate']:.1%} ({time.time()-t1:.0f}s)")
        else:
            print(f"    ❌ 无结果")
    except Exception as e:
        print(f"    ❌ 错误: {e}")

# ============================================================
# 7. 结果汇总
# ============================================================
print("\n" + "="*60)
print("📊 板块轮动策略回测结果汇总")
print("="*60)

# 按Sharpe排序
results.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'策略':<20} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'胜率':>8} {'PL比':>8} {'交易':>6}")
print("-" * 90)
for r in results:
    print(f"{r['name']:<20} {r['cagr']:>7.1%} {r['sharpe']:>8.2f} {r['sortino']:>8.2f} {r['max_dd']:>7.1%} {r['win_rate']:>7.1%} {r['pl_ratio']:>8.2f} {r['n_trades']:>6}")

# 年度收益对比
print("\n📈 年度收益对比:")
years = sorted(set().union(*[set(r['yearly_returns'].keys()) for r in results]))
print(f"{'策略':<20}", end="")
for y in years:
    print(f" {y:>8}", end="")
print()
print("-" * (20 + 9 * len(years)))
for r in results[:5]:  # Top 5
    print(f"{r['name']:<20}", end="")
    for y in years:
        ret = r['yearly_returns'].get(y, 0)
        print(f" {ret:>7.1%}", end="")
    print()

# ============================================================
# 8. 保存结果
# ============================================================
output = {
    'experiment': 'sector_rotation_v1',
    'date': '2026-06-21',
    'test_period': '2020-01 ~ 2026-06',
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
        'pl_ratio': round(r['pl_ratio'], 4),
        'n_trades': r['n_trades'],
        'sl_trades': r['sl_trades'],
        'final_equity': round(r['final_equity'], 2),
        'yearly_returns': {str(k): round(v, 4) for k, v in r['yearly_returns'].items()},
    })

os.makedirs('research', exist_ok=True)
with open('research/sector_rotation_v1_results.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ 结果已保存 research/sector_rotation_v1_results.json")
print(f"⏱️ 总耗时: {time.time()-t0:.0f}秒")
