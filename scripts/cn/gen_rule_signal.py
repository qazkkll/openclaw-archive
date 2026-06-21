#!/usr/bin/env python3
"""
rule-alpha-v2.1 — 生产信号生成脚本
每日盘后运行，输出：市场状态 + Top15信号 + 仓位建议
策略: 纯规则型（反转+资金流+低波动+超卖+大单+均线偏离）
持有期: 10天, Top15, SL-1%, DD-based position sizing

v2.1改进: DD-based position sizing替代binary market filter
- DD-3%→80%, DD-6%→60%, DD-10%→40%, DD-14%→20%, DD-18%→空仓
- Walk-Forward Sharpe: 1.75, CAGR: 51.8%, DD: -13.6%
"""
import pandas as pd, numpy as np, json, time, os, sys, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

HOLD_DAYS = 10
TOP_N = 15
SL = -0.01  # 生产配置: SL1%

# DD-based position sizing thresholds (v2.1)
# Format: (dd_level, position_pct)
DD_THRESHOLDS = [
    (-0.03, 0.80),  # DD-3% → 仓位80%
    (-0.06, 0.60),  # DD-6% → 仓位60%
    (-0.10, 0.40),  # DD-10% → 仓位40%
    (-0.14, 0.20),  # DD-14% → 仓位20%
    (-0.18, 0.00),  # DD-18% → 空仓
]

# 市场状态（用于显示，不影响仓位）
def get_market_state(market_ret20, market_ma60, market_ma120, market_breadth):
    ma_bull = market_ma60 > market_ma120
    mom_pos = market_ret20 > 0
    breadth_ok = market_breadth > 0.4
    
    if not ma_bull and not mom_pos:
        return 'bear'
    elif not ma_bull or not mom_pos:
        return 'cautious'
    elif not breadth_ok:
        return 'weak'
    else:
        return 'bull'

# DD-based position sizing
def get_dd_position_pct(current_dd, dd_thresholds=DD_THRESHOLDS):
    """根据当前回撤计算仓位比例"""
    position_pct = 1.0  # 默认满仓
    for dd_level, pct in dd_thresholds:
        if current_dd <= dd_level:
            position_pct = pct
            break
    return position_pct

# ============================================================
# 1. 加载数据
# ============================================================
print(f"📊 rule-alpha-v1.0 信号生成 {time.strftime('%Y-%m-%d %H:%M')}")
print("="*60)

t0 = time.time()

# 加载历史OHLCV
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

# 过滤
df = df[~df['sym'].str.startswith('688')].copy()
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
df = df[df['volume'] > 0].copy()
df = df.sort_values(['sym', 'date']).reset_index(drop=True)

print(f"  数据加载: {len(df):,}行, {df['sym'].nunique()}只股票, {time.time()-t0:.0f}秒")

# ============================================================
# 2. 计算特征
# ============================================================
print("  计算特征...")

# 收益率
df['ret20'] = df.groupby('sym')['close'].pct_change(20)

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
for col in ['total_net', 'lg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())

# 市场状态
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')
df['mkt_ma20'] = df.groupby('date')['ma20_bias'].transform('mean')

# ============================================================
# 3. 规则型评分函数
# ============================================================
def score_optimized(day):
    """rule-alpha-v1.0 评分函数"""
    s = day.copy()
    s['score'] = 0.0
    s['score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 3      # 反转：跌幅越大越好
    s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 2    # 资金流入
    s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 2  # 低波动
    s['score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1.5 # RSI超卖
    s['score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 1       # 大单流入
    s['score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1   # 均线偏离
    return s

# ============================================================
# 4. 市场状态判断
# ============================================================
all_dates = sorted(df['date'].unique())
today = all_dates[-1]
today_data = df[df['date'] == today].copy()

# 市场宏观指标
market_breadth = today_data['breadth'].mean() if 'breadth' in today_data.columns else 0.5
market_ret20 = today_data['ret20'].mean() if 'ret20' in today_data.columns else 0
market_ma_bias = today_data['ma20_bias'].mean() if 'ma20_bias' in today_data.columns else 0

# 用最近60天的市场均值判断趋势
recent_dates = all_dates[-60:]
recent_market = df[df['date'].isin(recent_dates)].groupby('date')['ret20'].mean()
market_ma60 = recent_market.mean()
recent_120 = all_dates[-120:]
market_ma120 = df[df['date'].isin(recent_120)].groupby('date')['ret20'].mean().mean()

regime = get_market_state(market_ret20, market_ma60, market_ma120, market_breadth)

# DD-based position sizing
# 从portfolio state文件读取当前DD
portfolio_state_file = 'signals/cn/portfolio_state.json'
if os.path.exists(portfolio_state_file):
    with open(portfolio_state_file) as f:
        portfolio_state = json.load(f)
    current_dd = portfolio_state.get('current_dd', 0)
    peak_equity = portfolio_state.get('peak_equity', 100000)
    current_equity = portfolio_state.get('current_equity', 100000)
else:
    current_dd = 0
    peak_equity = 100000
    current_equity = 100000

position_pct = get_dd_position_pct(current_dd)

# 市场状态仓位调整（叠加DD-based）
# 如果市场状态是bear，进一步降低仓位
if regime == 'bear':
    position_pct = min(position_pct, 0.5)  # bear时最多50%
elif regime == 'cautious':
    position_pct = min(position_pct, 0.8)  # cautious时最多80%

# ============================================================
# 5. 生成信号
# ============================================================
print(f"\n{'='*60}")
print(f"📊 rule-alpha-v1.0 信号 {today}")
print(f"{'='*60}")

print(f"\n🌐 市场状态: {regime.upper()}")
print(f"  MA60趋势: {market_ma60:.4f} {'>' if market_ma60 > market_ma120 else '<'} MA120: {market_ma120:.4f}")
print(f"  20日动量: {market_ret20:.4f} ({'正' if market_ret20 > 0 else '负'})")
print(f"  市场宽度: {market_breadth:.2%} ({'OK' if market_breadth > 0.4 else '弱'})")
print(f"  当前回撤: {current_dd:.2%}")
print(f"  ➜ 仓位建议: {position_pct*100:.0f}% (DD-based + 市场调整)")

if position_pct > 0:
    # 评分
    scored = score_optimized(today_data)
    
    # 过滤
    scored = scored[
        (scored['close'] >= 3) & 
        (scored['close'] <= 200) & 
        (~scored['sym'].str.contains('ST|退市', na=False)) &
        (scored['volume'] > 0)
    ]
    
    # Top N
    top = scored.nlargest(TOP_N, 'score')
    
    # 信号灯（百分位）
    all_scores = scored['score']
    p95 = all_scores.quantile(0.95)
    p90 = all_scores.quantile(0.90)
    p80 = all_scores.quantile(0.80)
    
    def signal_label(score):
        if score >= p95: return '🟢🟢'
        elif score >= p90: return '🟢'
        elif score >= p80: return '🟡'
        else: return '⚪'
    
    top['signal'] = top['score'].apply(signal_label)
    
    print(f"\n📈 Top{TOP_N} 信号 (持有{HOLD_DAYS}天, 止损{SL*100:.0f}%):")
    print(f"  百分位门槛: P95={p95:.3f} P90={p90:.3f} P80={p80:.3f}")
    print(f"{'排名':>4} {'信号':>5} {'代码':>8} {'价格':>8} {'评分':>8} {'20d收益':>8} {'RSI':>6} {'资金流5d':>10}")
    print("-" * 70)
    
    for _, r in top.iterrows():
        ret20_str = f"{r.get('ret20', 0):.1%}" if not pd.isna(r.get('ret20')) else "N/A"
        rsi_str = f"{r.get('rsi_14', 50):.0f}" if not pd.isna(r.get('rsi_14')) else "N/A"
        flow_str = f"{r.get('total_net_5d', 0)/1e8:.2f}亿" if not pd.isna(r.get('total_net_5d')) else "N/A"
        print(f"{_ + 1:>4} {r['signal']:>5} {r['sym']:>8} ¥{r['close']:>7.2f} {r['score']:>8.3f} {ret20_str:>8} {rsi_str:>6} {flow_str:>10}")
    
    # 统计
    g2 = (top['signal'] == '🟢🟢').sum()
    g1 = (top['signal'] == '🟢').sum()
    y = (top['signal'] == '🟡').sum()
    print(f"\n  🟢🟢精品: {g2}只  🟢强信号: {g1}只  🟡观察: {y}只")
    
    # 止损说明
    print(f"\n  ⚠️ 止损规则: 持有期内累计亏损>{SL*100:.0f}%即止损")
    print(f"  📅 调仓日: 每{HOLD_DAYS}天")
else:
    print(f"\n⚠️ 市场状态BEAR，建议空仓观望")
    top = pd.DataFrame()

# ============================================================
# 6. 保存信号
# ============================================================
signal_output = {
    'date': str(today),
    'strategy': 'rule-alpha-v2.1',
    'hold_days': HOLD_DAYS,
    'top_n': TOP_N,
    'stop_loss': SL,
    'regime': regime,
    'position_pct': position_pct,
    'position_sizing': 'DD-based',
    'dd_thresholds': DD_THRESHOLDS,
    'current_dd': round(current_dd, 4),
    'market': {
        'breadth': round(market_breadth, 4),
        'ret20': round(market_ret20, 6),
        'ma_bias': round(market_ma_bias, 6),
        'ma60': round(market_ma60, 6),
        'ma120': round(market_ma120, 6),
    },
    'top': []
}

if len(top) > 0:
    for _, r in top.iterrows():
        signal_output['top'].append({
            'rank': _ + 1,
            'sym': r['sym'],
            'close': round(float(r['close']), 2),
            'score': round(float(r['score']), 4),
            'signal': r['signal'],
            'ret20': round(float(r.get('ret20', 0)), 4),
            'rsi': round(float(r.get('rsi_14', 50)), 1),
        })

os.makedirs('signals/cn', exist_ok=True)
with open(f'signals/cn/rule_alpha_v1_{today}.json', 'w') as f:
    json.dump(signal_output, f, indent=2, ensure_ascii=False)
with open('signals/cn/latest_rule.json', 'w') as f:
    json.dump(signal_output, f, indent=2, ensure_ascii=False)

print(f"\n✅ 信号已保存 signals/cn/rule_alpha_v1_{today}.json")
print(f"   最新信号: signals/cn/latest_rule.json")
