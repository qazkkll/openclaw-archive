#!/usr/bin/env python3
"""
Falcon 6个月逐月回测 (2026年1-6月)
100万初始资金，带仓位管理和止损
"""
import pandas as pd, numpy as np, json, warnings, sys
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats

warnings.filterwarnings('ignore')
BASE = Path('/home/hermes/.hermes/openclaw-archive')
FALCON_DIR = BASE / 'data' / 'falcon'
OUTPUT = BASE / 'data' / 'falcon' / 'backtest_results'

def load_data():
    """加载价格和特征数据"""
    prices = pd.read_parquet(FALCON_DIR / 'features_v02.parquet', columns=['ticker', 'date', 'close', 'volume'])
    prices['date'] = pd.to_datetime(prices['date'])
    
    # 加载FinBERT (如果存在)
    finbert_dir = BASE / 'data' / 'finbert_sentiment'
    finbert_files = list(finbert_dir.rglob('ticker=*.parquet'))
    finbert = None
    if finbert_files:
        dfs = []
        for f in finbert_files:
            try:
                df = pd.read_parquet(f)
                if len(df) > 0:
                    dfs.append(df)
            except: pass
        if dfs:
            finbert = pd.concat(dfs, ignore_index=True)
            finbert['published_at'] = pd.to_datetime(finbert['published_at'], utc=True, errors='coerce')
            finbert = finbert.dropna(subset=['published_at', 'sentiment'])
            finbert['date'] = finbert['published_at'].dt.date
    
    return prices, finbert

def simulate_falcon_score(prices, date):
    """模拟Falcon评分 (基于可用特征)"""
    # 用最近30天数据计算简单因子
    hist = prices[(prices['date'] <= date) & (prices['date'] > date - timedelta(days=60))]
    if len(hist) < 20:
        return pd.DataFrame()
    
    latest = hist.groupby('ticker').last().reset_index()
    
    # 动量因子 (20日收益)
    ret_20d = hist.groupby('ticker').apply(
        lambda x: x['close'].iloc[-1] / x['close'].iloc[0] - 1 if len(x) >= 20 else np.nan
    ).reset_index(name='momentum')
    
    # 波动率因子 (20日)
    vol_20d = hist.groupby('ticker').apply(
        lambda x: x['close'].pct_change().std() * np.sqrt(252) if len(x) >= 20 else np.nan
    ).reset_index(name='volatility')
    
    # 成交量因子
    if 'volume' in hist.columns:
        vol_ratio = hist.groupby('ticker').apply(
            lambda x: x['volume'].iloc[-5:].mean() / x['volume'].mean() if len(x) >= 10 else 1
        ).reset_index(name='volume_ratio')
    else:
        vol_ratio = pd.DataFrame({'ticker': latest['ticker'], 'volume_ratio': 1})
    
    # 合并因子
    factors = ret_20d.merge(vol_20d, on='ticker').merge(vol_ratio, on='ticker')
    factors = factors.merge(latest[['ticker', 'close']], on='ticker')
    
    # 简单评分 (动量高+波动率低+成交量放大)
    for col in ['momentum', 'volatility', 'volume_ratio']:
        if factors[col].std() > 0:
            factors[f'{col}_z'] = (factors[col] - factors[col].mean()) / factors[col].std()
        else:
            factors[f'{col}_z'] = 0
    
    factors['falcon_score'] = (
        factors['momentum_z'] * 0.5 + 
        factors['volatility_z'] * -0.3 + 
        factors['volume_ratio_z'] * 0.2
    )
    
    # 排名
    factors['rank'] = factors['falcon_score'].rank(ascending=False, pct=True)
    factors['signal'] = '⚪'
    factors.loc[factors['rank'] <= 0.05, 'signal'] = '🟢🟢'
    factors.loc[(factors['rank'] > 0.05) & (factors['rank'] <= 0.2), 'signal'] = '🟢'
    factors.loc[(factors['rank'] > 0.2) & (factors['rank'] <= 0.5), 'signal'] = '🟡'
    
    return factors

def run_backtest(prices, finbert=None):
    """运行6个月回测 (2026年1-6月)"""
    print("="*60)
    print("📈 Falcon 6个月逐月回测 (2026年1-6月)")
    print("="*60)
    
    # 筛选2026年数据
    start_date = pd.Timestamp('2026-01-01')
    end_date = pd.Timestamp('2026-06-27')
    
    prices_2026 = prices[(prices['date'] >= start_date) & (prices['date'] <= end_date)].copy()
    if len(prices_2026) == 0:
        print("⚠️ 无2026年数据，用最近6个月")
        max_date = prices['date'].max()
        start_date = max_date - timedelta(days=180)
        prices_2026 = prices[(prices['date'] >= start_date) & (prices['date'] <= max_date)].copy()
    
    print(f"数据范围: {prices_2026['date'].min().date()} ~ {prices_2026['date'].max().date()}")
    print(f"股票数: {prices_2026['ticker'].nunique()}")
    
    # 月度调仓
    prices_2026['month'] = prices_2026['date'].dt.to_period('M')
    months = sorted(prices_2026['month'].unique())
    
    # 回测参数
    initial_capital = 1_000_000
    capital = initial_capital
    position_size = 0.10  # 每只10%
    max_positions = 10
    stop_loss = -0.15  # -15%止损
    rebalance_days = 63  # 季度调仓
    
    # 记录
    monthly_returns = []
    positions = {}  # ticker -> {'entry_price': float, 'entry_date': date, 'shares': int}
    trade_log = []
    last_rebalance = None
    
    for month in months:
        month_data = prices_2026[prices_2026['month'] == month]
        month_start = month_data['date'].min()
        month_end = month_data['date'].max()
        
        # 检查止损
        closed_positions = []
        for ticker, pos in list(positions.items()):
            ticker_data = month_data[month_data['ticker'] == ticker]
            if len(ticker_data) == 0: continue
            
            current_price = ticker_data['close'].iloc[-1]
            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
            
            if pnl_pct <= stop_loss:
                # 止损平仓
                proceeds = pos['shares'] * current_price
                capital += proceeds
                trade_log.append({
                    'date': month_end, 'ticker': ticker, 'action': 'STOP_LOSS',
                    'price': current_price, 'shares': pos['shares'],
                    'pnl_pct': pnl_pct, 'proceeds': proceeds
                })
                closed_positions.append(ticker)
        
        for ticker in closed_positions:
            del positions[ticker]
        
        # 调仓检查 (季度)
        if last_rebalance is None or (month_start - last_rebalance).days >= rebalance_days:
            # 评分
            scores = simulate_falcon_score(prices, month_start)
            if len(scores) > 0:
                # 选Top10 (行业分散简化版：直接取score最高的)
                top10 = scores.nlargest(max_positions, 'falcon_score')
                
                # 平掉不在Top10的持仓
                for ticker in list(positions.keys()):
                    if ticker not in top10['ticker'].values:
                        pos = positions[ticker]
                        ticker_data = month_data[month_data['ticker'] == ticker]
                        if len(ticker_data) > 0:
                            current_price = ticker_data['close'].iloc[-1]
                            proceeds = pos['shares'] * current_price
                            capital += proceeds
                            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                            trade_log.append({
                                'date': month_end, 'ticker': ticker, 'action': 'REBALANCE_OUT',
                                'price': current_price, 'shares': pos['shares'],
                                'pnl_pct': pnl_pct, 'proceeds': proceeds
                            })
                            del positions[ticker]
                
                # 开新仓
                for _, row in top10.iterrows():
                    ticker = row['ticker']
                    if ticker not in positions and len(positions) < max_positions:
                        alloc = capital * position_size
                        if alloc > 0:
                            shares = int(alloc / row['close'])
                            if shares > 0:
                                cost = shares * row['close']
                                capital -= cost
                                positions[ticker] = {
                                    'entry_price': row['close'],
                                    'entry_date': month_start,
                                    'shares': shares
                                }
                                trade_log.append({
                                    'date': month_start, 'ticker': ticker, 'action': 'BUY',
                                    'price': row['close'], 'shares': shares,
                                    'score': row['falcon_score'], 'signal': row['signal']
                                })
                
                last_rebalance = month_start
        
        # 月度市值
        portfolio_value = capital
        for ticker, pos in positions.items():
            ticker_data = month_data[month_data['ticker'] == ticker]
            if len(ticker_data) > 0:
                portfolio_value += pos['shares'] * ticker_data['close'].iloc[-1]
            else:
                portfolio_value += pos['shares'] * pos['entry_price']
        
        monthly_ret = (portfolio_value / initial_capital - 1) if initial_capital > 0 else 0
        monthly_returns.append({
            'month': str(month),
            'portfolio_value': portfolio_value,
            'return': monthly_ret,
            'positions': len(positions),
            'capital': capital
        })
        
        print(f"  {month}: 市值¥{portfolio_value:,.0f}, 收益{monthly_ret*100:+.2f}%, 持仓{len(positions)}只")
    
    # 统计
    if monthly_returns:
        final_value = monthly_returns[-1]['portfolio_value']
        total_return = (final_value / initial_capital - 1)
        annualized = (1 + total_return) ** (12 / len(monthly_returns)) - 1
        
        # 最大回撤
        values = [m['portfolio_value'] for m in monthly_returns]
        peak = values[0]
        max_dd = 0
        for v in values:
            peak = max(peak, v)
            dd = (v - peak) / peak
            max_dd = min(max_dd, dd)
        
        # 交易统计
        trades_df = pd.DataFrame(trade_log)
        if len(trades_df) > 0:
            n_trades = len(trades_df[trades_df['action'] == 'BUY'])
            n_stop_loss = len(trades_df[trades_df['action'] == 'STOP_LOSS'])
            win_trades = trades_df[(trades_df['action'].isin(['REBALANCE_OUT', 'STOP_LOSS'])) & (trades_df['pnl_pct'] > 0)]
            total_closed = len(trades_df[trades_df['action'].isin(['REBALANCE_OUT', 'STOP_LOSS'])])
            win_rate = len(win_trades) / total_closed if total_closed > 0 else 0
        else:
            n_trades = n_stop_loss = 0
            win_rate = 0
        
        print(f"\n{'='*60}")
        print("📊 回测结果汇总")
        print(f"{'='*60}")
        print(f"  初始资金: ¥{initial_capital:,.0f}")
        print(f"  最终市值: ¥{final_value:,.0f}")
        print(f"  总收益: {total_return*100:+.2f}%")
        print(f"  年化收益: {annualized*100:+.2f}%")
        print(f"  最大回撤: {max_dd*100:.2f}%")
        print(f"  总交易: {n_trades}笔")
        print(f"  止损触发: {n_stop_loss}次")
        print(f"  胜率: {win_rate*100:.1f}%")
        
        # 逐月明细
        print(f"\n  逐月明细:")
        for m in monthly_returns:
            print(f"    {m['month']}: ¥{m['portfolio_value']:,.0f} ({m['return']*100:+.2f}%)")
        
        return {
            'initial': initial_capital,
            'final': final_value,
            'total_return': total_return,
            'annualized': annualized,
            'max_drawdown': max_dd,
            'n_trades': n_trades,
            'n_stop_loss': n_stop_loss,
            'win_rate': win_rate,
            'monthly': monthly_returns,
            'trades': trade_log
        }
    
    return None

if __name__ == '__main__':
    print("加载数据...")
    prices, finbert = load_data()
    print(f"价格: {len(prices)}行, {prices['ticker'].nunique()}只")
    if finbert is not None:
        print(f"FinBERT: {len(finbert)}篇")
    
    result = run_backtest(prices, finbert)
    
    if result:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT / 'falcon_6month_backtest.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n💾 结果已保存: {OUTPUT / 'falcon_6month_backtest.json'}")
