#!/usr/bin/env python3
"""
sync_portfolio_from_opend.py — 从Futu OpenD拉实时持仓，写入dashboard格式的portfolio.json

链路: Futu OpenD (127.0.0.1:11111) → 查询持仓+行情 → output/state/portfolio.json

分类逻辑:
  蓝盾(>=$10) → large_cap
  绿箭(<$10)  → small_cap

用法:
  cd ~/.hermes/openclaw-archive && python3 scripts/sync_portfolio_from_opend.py
"""

import json, sys, os
from datetime import datetime
from pathlib import Path

# 路径
ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "output" / "state"
PORTFOLIO_PATH = STATE_DIR / "portfolio.json"

STATE_DIR.mkdir(parents=True, exist_ok=True)


def query_opend():
    """连接OpenD查询持仓+账户"""
    from futu import OpenSecTradeContext, OpenQuoteContext, RET_OK, TrdEnv

    trade_ctx = OpenSecTradeContext(host='127.0.0.1', port=11111)
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

    # 账户
    ret, accinfo = trade_ctx.accinfo_query(trd_env=TrdEnv.REAL)
    if ret != RET_OK:
        print(f"❌ 账户查询失败: {accinfo}")
        trade_ctx.close()
        quote_ctx.close()
        return None, None, None

    # 持仓
    ret, positions = trade_ctx.position_list_query(trd_env=TrdEnv.REAL)
    if ret != RET_OK or positions is None or positions.empty:
        print("📭 无持仓")
        trade_ctx.close()
        quote_ctx.close()
        return accinfo, [], {}

    # 过滤qty>0
    positions = positions[positions['qty'] > 0].copy()
    if positions.empty:
        trade_ctx.close()
        quote_ctx.close()
        return accinfo, [], {}

    # 实时行情快照
    codes = positions['code'].tolist()
    ret, snapshots = quote_ctx.get_market_snapshot(codes)
    
    price_map = {}
    if ret == RET_OK and snapshots is not None and not snapshots.empty:
        for _, snap in snapshots.iterrows():
            price_map[snap['code']] = {
                'last_price': float(snap.get('last_price', 0)),
                'prev_close': float(snap.get('prev_close_price', 0)),
                'name': snap.get('name', ''),
            }

    trade_ctx.close()
    quote_ctx.close()

    return accinfo, positions, price_map


def transform_to_dashboard(accinfo, positions, price_map):
    """转换为dashboard期望的格式"""
    large_cap = []
    small_cap = []

    for _, pos in positions.iterrows():
        code = pos['code']  # e.g. "US.ASML"
        ticker = code.replace('US.', '')  # "ASML"
        qty = int(pos['qty'])
        cost_price = round(float(pos['cost_price']), 3)
        market_val = round(float(pos['market_val']), 2)

        # 从快照获取实时价
        snap = price_map.get(code, {})
        current_price = round(snap.get('last_price', 0), 2)
        stock_name = snap.get('name', ticker)

        pnl_pct = round((current_price / cost_price - 1) * 100, 2) if cost_price > 0 else 0
        pnl_usd = round((current_price - cost_price) * qty, 2)

        entry = {
            'ticker': ticker,
            'name': stock_name,
            'qty': qty,
            'cost_price': cost_price,
            'current_price': current_price,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'market_val': market_val,
            'days_held': 0,  # OpenD不提供，后续可从交易记录推算
            'hold_days': 20 if current_price >= 10 else 5,
            'model': '🛡️ 蓝盾V6' if current_price >= 10 else '🎯 绿箭V11',
        }

        if current_price >= 10:
            large_cap.append(entry)
        else:
            small_cap.append(entry)

    # 按市值降序
    large_cap.sort(key=lambda x: x.get('market_val', 0), reverse=True)
    small_cap.sort(key=lambda x: x.get('market_val', 0), reverse=True)

    return large_cap, small_cap


def main():
    print(f"[sync] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 从OpenD同步持仓...")

    result = query_opend()
    if result is None or result[0] is None:
        print("[sync] ❌ OpenD查询失败")
        sys.exit(1)

    accinfo, positions, price_map = result

    if isinstance(positions, list) and not positions:
        print("[sync] 📭 无持仓")
        # 写空文件
        with open(PORTFOLIO_PATH, 'w') as f:
            json.dump({'large_cap': [], 'small_cap': [], 'account': {}, 'last_sync': datetime.now().isoformat()}, f, indent=2)
        return

    large_cap, small_cap = transform_to_dashboard(accinfo, positions, price_map)

    # 账户信息
    acc_row = accinfo.iloc[0] if hasattr(accinfo, 'iloc') else accinfo
    account = {
        'total_assets': round(float(acc_row.get('total_assets', 0)), 2),
        'cash': round(float(acc_row.get('cash', 0)), 2),
        'market_val': round(float(acc_row.get('market_val', 0)), 2),
        'power': round(float(acc_row.get('power', 0)), 2),
    }

    portfolio = {
        'large_cap': large_cap,
        'small_cap': small_cap,
        'account': account,
        'last_sync': datetime.now().isoformat(),
    }

    with open(PORTFOLIO_PATH, 'w', encoding='utf-8') as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)

    print(f"[sync] ✅ 写入 {PORTFOLIO_PATH}")
    print(f"[sync] 蓝盾: {len(large_cap)}只 | 绿箭: {len(small_cap)}只")
    print(f"[sync] 总资产: ${account['total_assets']:,.2f} | 现金: ${account['cash']:,.2f} | 持仓: ${account['market_val']:,.2f}")

    for h in large_cap + small_cap:
        emoji = '🟢' if h['pnl_pct'] >= 0 else '🔴'
        print(f"  {emoji} {h['ticker']}: {h['qty']}股 @ ${h['current_price']:.2f} | {h['pnl_pct']:+.1f}% (${h['pnl_usd']:+.2f})")


if __name__ == '__main__':
    main()
