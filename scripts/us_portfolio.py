#!/usr/bin/env python3
"""
OpenD美股持仓查询 — 实时持仓 + 盈亏 + 行情

用法：
    python3 us_portfolio.py              # 查询当前持仓
    python3 us_portfolio.py --json       # JSON输出
"""

import sys, os, json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def query_positions():
    """查询美股持仓"""
    from futu import OpenSecTradeContext, OpenQuoteContext, RET_OK
    
    trade_ctx = OpenSecTradeContext(host='127.0.0.1', port=11111)
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    
    # 账户信息
    ret, accinfo = trade_ctx.accinfo_query()
    if ret != RET_OK:
        print(f"❌ 账户查询失败: {accinfo}")
        return None
    
    # 持仓
    ret, positions = trade_ctx.position_list_query()
    if ret != RET_OK or positions is None or positions.empty:
        print("📭 当前无持仓")
        return None
    
    # 实时行情
    codes = positions['code'].tolist()
    ret, snapshots = quote_ctx.get_market_snapshot(codes)
    
    # 组装
    portfolio = []
    for _, pos in positions.iterrows():
        code = pos['code']
        qty = pos['qty']
        cost_price = pos['cost_price']
        market_val = pos['market_val']
        
        live_price = 0
        change_pct = 0
        if snapshots is not None and not snapshots.empty:
            snap = snapshots[snapshots['code'] == code]
            if not snap.empty:
                live_price = snap.iloc[0].get('last_price', 0)
                prev_close = snap.iloc[0].get('prev_close_price', 0)
                if prev_close > 0:
                    change_pct = (live_price / prev_close - 1) * 100
        
        unrealized_pl = (live_price - cost_price) * qty if live_price > 0 and qty > 0 else 0
        unrealized_pl_ratio = ((live_price / cost_price) - 1) * 100 if cost_price > 0 and live_price > 0 else 0
        
        portfolio.append({
            'code': code,
            'name': pos.get('stock_name', ''),
            'qty': int(qty),
            'cost_price': round(float(cost_price), 2),
            'live_price': round(float(live_price), 2),
            'market_val': round(float(market_val), 2),
            'unrealized_pl': round(float(unrealized_pl), 2),
            'unrealized_pl_ratio': round(float(unrealized_pl_ratio), 2),
            'change_pct': round(float(change_pct), 2),
        })
    
    portfolio.sort(key=lambda x: x['market_val'], reverse=True)
    
    # 过滤空仓
    portfolio = [p for p in portfolio if p['qty'] > 0]
    
    account = {
        'total_assets': round(float(accinfo.iloc[0].get('total_assets', 0)), 2),
        'cash': round(float(accinfo.iloc[0].get('cash', 0)), 2),
        'market_val': round(float(accinfo.iloc[0].get('market_val', 0)), 2),
        'power': round(float(accinfo.iloc[0].get('power', 0)), 2),
    }
    
    trade_ctx.close()
    quote_ctx.close()
    
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'account': account,
        'positions': portfolio,
    }


def format_portfolio(data):
    """格式化持仓报告"""
    if data is None:
        return "无持仓数据"
    
    account = data['account']
    positions = data['positions']
    
    lines = []
    lines.append("━" * 45)
    lines.append("📊 美股持仓报告")
    lines.append(f"⏰ {data['timestamp']}")
    lines.append("━" * 45)
    
    # 账户概览
    lines.append("")
    lines.append("💰 账户概览")
    lines.append(f"  总资产:  ${account['total_assets']:>12,.2f}")
    lines.append(f"  现金:    ${account['cash']:>12,.2f}")
    lines.append(f"  持仓市值: ${account['market_val']:>11,.2f}")
    lines.append(f"  购买力:  ${account['power']:>11,.2f}")
    
    # 持仓列表
    if positions:
        lines.append("")
        lines.append("📋 持仓明细")
        lines.append(f"{'标的':<7} {'名称':<12} {'数量':<6} {'成本':<9} {'现价':<9} {'盈亏':>10} {'盈亏%':>7} {'日涨跌':>7}")
        lines.append("─" * 80)
        
        total_pl = 0
        total_mv = 0
        for p in positions:
            pl_emoji = "🟢" if p['unrealized_pl'] > 0 else "🔴" if p['unrealized_pl'] < 0 else "⚪"
            chg_emoji = "↑" if p['change_pct'] > 0 else "↓" if p['change_pct'] < 0 else "→"
            
            lines.append(
                f" {pl_emoji} {p['code']:<5} {p['name']:<12} {p['qty']:<6} "
                f"${p['cost_price']:<8} ${p['live_price']:<8} "
                f"${p['unrealized_pl']:>+9,.2f} {p['unrealized_pl_ratio']:>+6.1f}% "
                f"{chg_emoji}{p['change_pct']:>+5.1f}%"
            )
            total_pl += p['unrealized_pl']
            total_mv += p['market_val']
        
        lines.append("─" * 80)
        total_pl_ratio = (total_pl / total_mv * 100) if total_mv > 0 else 0
        lines.append(f"  {'合计':<21} {'':6} {'':9} {'':9} ${total_pl:>+9,.2f} {total_pl_ratio:>+6.1f}%")
    
    lines.append("")
    lines.append("━" * 45)
    
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OpenD美股持仓查询")
    parser.add_argument("--json", action="store_true", help="JSON输出")
    args = parser.parse_args()
    
    data = query_positions()
    
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(format_portfolio(data))


if __name__ == "__main__":
    main()
