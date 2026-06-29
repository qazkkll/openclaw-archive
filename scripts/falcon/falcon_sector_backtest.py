#!/usr/bin/env python3
"""
Falcon 行业分散回测对比
比较: Top10无限制 vs Top10行业分散(≤2只/行业)
使用SPX GICS行业分类
"""
import pandas as pd, numpy as np, json, warnings, sys
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
BASE = Path('/home/hermes/.hermes/openclaw-archive')
FALCON_DIR = BASE / 'data' / 'falcon'
OUTPUT = BASE / 'data' / 'falcon' / 'backtest_results'

# SPX GICS行业映射 (Top常用)
SECTOR_MAP = {
    'XLK': ['AAPL','MSFT','NVDA','AVGO','ADBE','CRM','ACN','INTC','AMD','CSCO','ORCL','TXN','QCOM','NOW','IBM','INTU','AMAT','MU','LRCX','ADI','KLAC','SNPS','CDNS','MRVL','HPQ','FTNT','PANW','NXPI','APH','MSI'],
    'XLF': ['BRK.B','JPM','V','MA','BAC','WFC','GS','MS','AXP','SPGI','CME','ICE','BLK','SCHW','CB','PGR','MMC','AON','MET','AIG','TRV','ALL','AFL','PRU','MCO','FIS','FISV','GPN'],
    'XLV': ['UNH','JNJ','LLY','ABBV','MRK','TMO','ABT','PFE','DHR','AMGN','BMY','MDT','ISRG','GILD','SYK','VRTX','BSX','REGN','ZTS','ELV','CI','HCA','EW','BDX','BAX','HOLX','ALGN','DXCM','IQV','A'],
    'XLY': ['AMZN','TSLA','HD','MCD','NKE','LOW','SBUX','TJX','BKNG','CMG','ORLY','AZO','ROST','DHI','LEN','GM','F','MAR','YUM','DPZ','APTV','EBAY','ETSY','BBY','POOL','HAS','NCLH','CCL','RCL','MGM'],
    'XLC': ['META','GOOG','GOOGL','NFLX','DIS','CMCSA','T','VZ','TMUS','CHTR','EA','ATVI','TTWO','OMC','IPG','FOXA','LYV','MTCH','PARA','WBD','ROKU','SNAP','PINS','ZM','SPOT'],
    'XLI': ['GE','CAT','HON','UNP','RTX','BA','DE','LMT','MMM','UPS','GD','NSC','EMR','ITW','ETN','WM','CSX','FDX','ROK','CMI','PH','JCI','TDG','CTAS','FAST','PAYX','ODFL','GWW','SWK','IR'],
    'XLP': ['PG','KO','PEP','COST','WMT','PM','MO','MDLZ','CL','EL','KMB','GIS','SYY','HSY','KDP','K','CPB','CAG','SJM','HRL','TSN','CHD','CLX','MKC','TAP'],
    'XLE': ['XOM','CVX','COP','EOG','SLB','MPC','PSX','VLO','OXY','PXD','WMB','KMI','HAL','DVN','FANG','HES','BKR','TRGP','CTRA','OVV','APA','MRO','FMT','NOV','SWN'],
    'XLU': ['NEE','DUK','SO','D','SRE','AEP','EXC','XEL','ED','WEC','ES','AWK','DTE','CMS','AEE','ETR','FE','PNW','PPL','NRG','CEG','PCG','EIX','AES','ATO','CNP','LNT','NI','OGE','UGI'],
    'XLRE': ['AMT','PLD','CCI','EQIX','PSA','SPG','O','WELL','DLR','AVB','VICI','EQR','ARE','MAA','UDR','ESS','VTR','PEAK','KIM','REG','BXP','SLG','HST','CPT','KRG','MAC','HIW','CUZ','DEI','ELS'],
    'XLB': ['LIN','APD','SHW','ECL','FCX','NEM','NUE','VMC','MLM','DOW','DD','PPG','CTVA','ALB','EMN','CE','CF','MOS','IFF','FMC','SEE','IP','PKG','AVY','BLL','WRK','SON','OLN','CBT','HUN']
}

def get_sector(ticker):
    for sector, tickers in SECTOR_MAP.items():
        if ticker in tickers:
            return sector
    return 'OTHER'

def load_data():
    prices = pd.read_parquet(FALCON_DIR / 'features_v02.parquet', columns=['ticker', 'date', 'close', 'volume'])
    prices['date'] = pd.to_datetime(prices['date'])
    return prices

def score_universe(prices, date, lookback=60):
    """简单动量+波动率评分"""
    hist = prices[(prices['date'] <= date) & (prices['date'] > date - timedelta(days=lookback))]
    if len(hist) < 20:
        return pd.DataFrame()
    
    latest = hist.groupby('ticker').last().reset_index()
    
    ret_20d = hist.groupby('ticker').apply(
        lambda x: x['close'].iloc[-1] / x['close'].iloc[0] - 1 if len(x) >= 20 else np.nan
    ).reset_index(name='momentum')
    
    vol_20d = hist.groupby('ticker').apply(
        lambda x: x['close'].pct_change().std() * np.sqrt(252) if len(x) >= 20 else np.nan
    ).reset_index(name='volatility')
    
    if 'volume' in hist.columns:
        vol_ratio = hist.groupby('ticker').apply(
            lambda x: x['volume'].iloc[-5:].mean() / x['volume'].mean() if len(x) >= 10 and x['volume'].mean() > 0 else 1
        ).reset_index(name='volume_ratio')
    else:
        vol_ratio = pd.DataFrame({'ticker': latest['ticker'], 'volume_ratio': 1})
    
    factors = ret_20d.merge(vol_20d, on='ticker').merge(vol_ratio, on='ticker')
    factors = factors.merge(latest[['ticker', 'close']], on='ticker')
    factors = factors.dropna()
    
    for col in ['momentum', 'volatility', 'volume_ratio']:
        std = factors[col].std()
        if std > 0:
            factors[f'{col}_z'] = (factors[col] - factors[col].mean()) / std
        else:
            factors[f'{col}_z'] = 0
    
    factors['falcon_score'] = factors['momentum_z'] * 0.5 + factors['volatility_z'] * -0.3 + factors['volume_ratio_z'] * 0.2
    factors['sector'] = factors['ticker'].apply(get_sector)
    
    return factors

def backtest(prices, top_n=10, max_per_sector=2, rebalance_days=63, stop_loss=-0.15):
    """回测主函数"""
    start_date = pd.Timestamp('2026-01-01')
    end_date = pd.Timestamp('2026-06-27')
    
    data = prices[(prices['date'] >= start_date) & (prices['date'] <= end_date)].copy()
    if len(data) == 0:
        max_date = prices['date'].max()
        data = prices[(prices['date'] >= max_date - timedelta(days=180)) & (prices['date'] <= max_date)].copy()
    
    data['month'] = data['date'].dt.to_period('M')
    months = sorted(data['month'].unique())
    
    initial_capital = 1_000_000
    capital = initial_capital
    position_size = 1.0 / top_n
    positions = {}
    last_rebalance = None
    monthly_returns = []
    trade_count = 0
    stop_count = 0
    
    for month in months:
        mdata = data[data['month'] == month]
        m_start = mdata['date'].min()
        m_end = mdata['date'].max()
        
        # 止损
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            td = mdata[mdata['ticker'] == ticker]
            if len(td) > 0:
                cur = td['close'].iloc[-1]
                pnl = (cur - pos['entry']) / pos['entry']
                if pnl <= stop_loss:
                    capital += pos['shares'] * cur
                    del positions[ticker]
                    stop_count += 1
        
        # 调仓
        if last_rebalance is None or (m_start - last_rebalance).days >= rebalance_days:
            scores = score_universe(prices, m_start)
            if len(scores) > 0:
                if max_per_sector < 99:
                    # 行业分散选股
                    selected = []
                    sector_count = {}
                    for _, row in scores.nlargest(200, 'falcon_score').iterrows():
                        sec = row['sector']
                        if sector_count.get(sec, 0) < max_per_sector and len(selected) < top_n:
                            selected.append(row)
                            sector_count[sec] = sector_count.get(sec, 0) + 1
                    target = pd.DataFrame(selected)
                else:
                    target = scores.nlargest(top_n, 'falcon_score')
                
                # 平仓不在目标中的
                for ticker in list(positions.keys()):
                    if ticker not in target['ticker'].values:
                        td = mdata[mdata['ticker'] == ticker]
                        if len(td) > 0:
                            capital += positions[ticker]['shares'] * td['close'].iloc[-1]
                            del positions[ticker]
                
                # 开新仓
                for _, row in target.iterrows():
                    if row['ticker'] not in positions and len(positions) < top_n:
                        alloc = initial_capital * position_size
                        shares = int(alloc / row['close'])
                        if shares > 0:
                            capital -= shares * row['close']
                            positions[row['ticker']] = {'entry': row['close'], 'shares': shares}
                            trade_count += 1
                
                last_rebalance = m_start
        
        # 月末市值
        pval = capital
        for ticker, pos in positions.items():
            td = mdata[mdata['ticker'] == ticker]
            if len(td) > 0:
                pval += pos['shares'] * td['close'].iloc[-1]
            else:
                pval += pos['shares'] * pos['entry']
        
        ret = pval / initial_capital - 1
        monthly_returns.append({'month': str(month), 'value': pval, 'return': ret, 'positions': len(positions)})
    
    # 统计
    if monthly_returns:
        final = monthly_returns[-1]['value']
        total_ret = final / initial_capital - 1
        annualized = (1 + total_ret) ** (12 / len(monthly_returns)) - 1
        values = [m['value'] for m in monthly_returns]
        peak = values[0]
        max_dd = 0
        for v in values:
            peak = max(peak, v)
            max_dd = min(max_dd, (v - peak) / peak)
        sharpe_est = annualized / 0.15 if max_dd != 0 else 0  # rough estimate
        
        return {
            'total_return': total_ret,
            'annualized': annualized,
            'max_drawdown': max_dd,
            'trades': trade_count,
            'stop_losses': stop_count,
            'monthly': monthly_returns,
            'sharpe_est': sharpe_est
        }
    return None

if __name__ == '__main__':
    print("="*60)
    print("🔄 Falcon 行业分散对比测试")
    print("="*60)
    
    prices = load_data()
    print(f"数据: {len(prices)}行, {prices['ticker'].nunique()}只")
    
    # Test 1: Top10 无限制
    print("\n--- Top10 无行业限制 ---")
    r1 = backtest(prices, top_n=10, max_per_sector=99, rebalance_days=63)
    if r1:
        print(f"  总收益: {r1['total_return']*100:+.1f}%")
        print(f"  年化: {r1['annualized']*100:+.1f}%")
        print(f"  回撤: {r1['max_drawdown']*100:.1f}%")
        print(f"  交易: {r1['trades']}笔, 止损: {r1['stop_losses']}次")
        for m in r1['monthly']:
            print(f"    {m['month']}: ¥{m['value']:,.0f} ({m['return']*100:+.1f}%) 持仓{m['positions']}")
    
    # Test 2: Top10 行业分散(≤2只/行业)
    print("\n--- Top10 行业分散(≤2只/行业) ---")
    r2 = backtest(prices, top_n=10, max_per_sector=2, rebalance_days=63)
    if r2:
        print(f"  总收益: {r2['total_return']*100:+.1f}%")
        print(f"  年化: {r2['annualized']*100:+.1f}%")
        print(f"  回撤: {r2['max_drawdown']*100:.1f}%")
        print(f"  交易: {r2['trades']}笔, 止损: {r2['stop_losses']}次")
        for m in r2['monthly']:
            print(f"    {m['month']}: ¥{m['value']:,.0f} ({m['return']*100:+.1f}%) 持仓{m['positions']}")
    
    # Test 3: Top20 行业分散(≤3只/行业)
    print("\n--- Top20 行业分散(≤3只/行业) ---")
    r3 = backtest(prices, top_n=20, max_per_sector=3, rebalance_days=63)
    if r3:
        print(f"  总收益: {r3['total_return']*100:+.1f}%")
        print(f"  年化: {r3['annualized']*100:+.1f}%")
        print(f"  回撤: {r3['max_drawdown']*100:.1f}%")
        print(f"  交易: {r3['trades']}笔, 止损: {r3['stop_losses']}次")
        for m in r3['monthly']:
            print(f"    {m['month']}: ¥{m['value']:,.0f} ({m['return']*100:+.1f}%) 持仓{m['positions']}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 对比汇总")
    print("="*60)
    if r1 and r2 and r3:
        print(f"{'策略':<30} {'总收益':>8} {'年化':>8} {'回撤':>8} {'止损':>4}")
        print(f"{'Top10无限制':<30} {r1['total_return']*100:>+7.1f}% {r1['annualized']*100:>+7.1f}% {r1['max_drawdown']*100:>7.1f}% {r1['stop_losses']:>4}")
        print(f"{'Top10行业分散(≤2)':<30} {r2['total_return']*100:>+7.1f}% {r2['annualized']*100:>+7.1f}% {r2['max_drawdown']*100:>7.1f}% {r2['stop_losses']:>4}")
        print(f"{'Top20行业分散(≤3)':<30} {r3['total_return']*100:>+7.1f}% {r3['annualized']*100:>+7.1f}% {r3['max_drawdown']*100:>7.1f}% {r3['stop_losses']:>4}")
    
    # Save
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {'top10_unlimited': r1, 'top10_diversified': r2, 'top20_diversified': r3}
    with open(OUTPUT / 'sector_diversification_comparison.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n💾 已保存: {OUTPUT / 'sector_diversification_comparison.json'}")
