#!/usr/bin/env python3
"""
OpenD实时行情获取 — 替代minishare/yfinance

用法：
    from opend_data import get_realtime_prices, get_market_context, get_kline
"""

import json, os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_quote_ctx():
    """获取OpenD行情上下文"""
    from futu import OpenQuoteContext
    return OpenQuoteContext(host='127.0.0.1', port=11111)


def get_realtime_prices(tickers):
    """OpenD获取美股实时价格"""
    from futu import RET_OK, Market, SubType
    
    ctx = get_quote_ctx()
    
    # 转换代码格式（AAPL → US.AAPL）
    futu_codes = [f"US.{t}" for t in tickers]
    
    # 订阅实时报价
    ret_sub = ctx.subscribe(futu_codes, [SubType.QUOTE], subscribe_push=False)
    
    # 获取快照
    ret, data = ctx.get_market_snapshot(futu_codes)
    
    prices = {}
    if ret == RET_OK and data is not None and not data.empty:
        for _, row in data.iterrows():
            code = row['code'].replace('US.', '')
            prices[code] = {
                'price': float(row.get('last_price', 0)),
                'change_pct': float(row.get('price_spread', 0)),
                'volume': int(row.get('volume', 0)),
                'open': float(row.get('open_price', 0)),
                'high': float(row.get('high_price', 0)),
                'low': float(row.get('low_price', 0)),
                'prev_close': float(row.get('prev_close_price', 0)),
                'turnover': float(row.get('turnover', 0)),
            }
    
    ctx.close()
    return prices


def get_kline(ticker, ktype='K_DAY', count=60):
    """OpenD获取历史K线"""
    from futu import RET_OK, KLType, SubType
    
    ctx = get_quote_ctx()
    
    futu_code = f"US.{ticker}"
    kl_type = KLType.K_DAY if ktype == 'K_DAY' else KLType.K_60M
    
    # 订阅K线
    ctx.subscribe([futu_code], [kl_type], subscribe_push=False)
    
    # 获取K线
    ret, data, _ = ctx.request_history_kline(futu_code, ktype=kl_type, max_count=count)
    
    klines = []
    if ret == RET_OK and data is not None and not data.empty:
        for _, row in data.iterrows():
            klines.append({
                'date': str(row.get('time_key', '')),
                'open': float(row.get('open', 0)),
                'high': float(row.get('high', 0)),
                'low': float(row.get('low', 0)),
                'close': float(row.get('close', 0)),
                'volume': int(row.get('volume', 0)),
            })
    
    ctx.close()
    return klines


def get_market_context():
    """获取市场概览（通过指数行情）"""
    from futu import RET_OK
    
    ctx = get_quote_ctx()
    
    # 美股指数代码
    indices = {
        'S&P 500': 'US.IXIC',  # 纳斯达克（近似）
        'NASDAQ': 'US.IXIC',
        'DJI': 'US.DJI',
    }
    
    # 获取主要指数
    ret, data = ctx.get_market_snapshot(['US.IXIC', 'US.DJI'])
    
    context = {}
    if ret == RET_OK and data is not None:
        for _, row in data.iterrows():
            code = row['code']
            if code == 'US.IXIC':
                context['NASDAQ'] = f"{row.get('last_price', 0):,.0f}"
            elif code == 'US.DJI':
                context['DJI'] = f"{row.get('last_price', 0):,.0f}"
    
    # VIX需要通过yfinance（OpenD无VIX权限）
    try:
        import yfinance as yf
        vix = yf.Ticker('^VIX').fast_info
        context['VIX'] = f"{vix.last_price:.1f}"
    except:
        context['VIX'] = '—'
    
    # 10Y国债
    try:
        import yfinance as yf
        tnx = yf.Ticker('^TNX').fast_info
        context['10Y'] = f"{tnx.last_price:.2f}%"
    except:
        context['10Y'] = '—'
    
    ctx.close()
    return context


def get_portfolio():
    """获取美股持仓"""
    from futu import RET_OK, TrdEnv, TrdMarket
    
    from futu import OpenSecTradeContext
    trade_ctx = OpenSecTradeContext(host='127.0.0.1', port=11111)
    
    # 获取持仓
    ret, positions = trade_ctx.position_list_query(trd_env=TrdEnv.REAL)
    
    if ret != RET_OK or positions is None or positions.empty:
        trade_ctx.close()
        return []
    
    # 获取实时行情
    codes = positions['code'].tolist()
    quote_ctx = get_quote_ctx()
    ret, snapshots = quote_ctx.get_market_snapshot(codes)
    
    portfolio = []
    for _, pos in positions.iterrows():
        code = pos['code']
        qty = pos['qty']
        cost_price = pos['cost_price']
        market_val = pos['market_val']
        
        live_price = 0
        if snapshots is not None and not snapshots.empty:
            snap = snapshots[snapshots['code'] == code]
            if not snap.empty:
                live_price = snap.iloc[0].get('last_price', 0)
        
        unrealized_pl = (live_price - cost_price) * qty if live_price > 0 else 0
        
        portfolio.append({
            'code': code,
            'name': pos.get('stock_name', ''),
            'qty': qty,
            'cost_price': round(cost_price, 2),
            'live_price': round(live_price, 2),
            'market_val': round(market_val, 2),
            'unrealized_pl': round(unrealized_pl, 2),
        })
    
    trade_ctx.close()
    quote_ctx.close()
    return portfolio


# ── 测试 ──
if __name__ == "__main__":
    print("=== 测试OpenD数据源 ===")
    
    print("\n1. 实时价格:")
    prices = get_realtime_prices(['AAPL', 'NVDA', 'TSLA'])
    for k, v in prices.items():
        print(f"  {k}: ${v['price']:.2f}")
    
    print("\n2. 市场概览:")
    ctx = get_market_context()
    for k, v in ctx.items():
        print(f"  {k}: {v}")
    
    print("\n3. 持仓:")
    portfolio = get_portfolio()
    if portfolio:
        for p in portfolio:
            print(f"  {p['code']}: {p['qty']}股 @ ${p['cost_price']}")
    else:
        print("  无持仓")
