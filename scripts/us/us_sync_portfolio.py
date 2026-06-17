"""
us_sync_portfolio.py — 持仓同步引擎
1. 从Futu OpenD拉最新持仓
2. 用minishare批量拉实时价
3. 更新 data/portfolio.json（持仓市值+现金+总资产）
4. 输出变动摘要

用法: python scripts/us_sync_portfolio.py
"""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

try:
    import minishare as ms
except ImportError:
    print("[ERROR] minishare not installed. Run: pip install minishare")
    sys.exit(1)

workspace = Path.cwd()
PORTFOLIO_PATH = workspace / 'data' / 'portfolio.json'
MS_TOKEN = 'Jarvne6fmgArRa46Xfon0e1kw55E6hes5IB2Fy2X0ndqnvrL48jsVOtTbf014f06'

def load_portfolio():
    if PORTFOLIO_PATH.exists():
        with open(PORTFOLIO_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'holdings': [], 'available_cash': 0, 'last_sync': None}

def fetch_prices(codes):
    """Minishare批量拉实时价"""
    api = ms.pro_api(MS_TOKEN)
    ts_code = ','.join(codes)
    df = api.rt_us_k(ts_code=ts_code, extFields='date')
    if df is None or df.empty:
        print("[WARN] minishare返回空数据")
        return {}
    result = {}
    for _, row in df.iterrows():
        code = row.get('ts_code', '').strip()
        if code:
            result[code] = {
                'price': float(row.get('close', 0)),
                'change': float(row.get('pct_chg', 0)),
                'date': str(row.get('date', ''))
            }
    return result

def main():
    print("[持仓同步] 开始...")

    # 1. 读当前持仓
    pf = load_portfolio()
    holdings = pf.get('holdings', [])
    if not holdings:
        print("[WARN] 持仓为空")
        return

    codes = [h['code'] for h in holdings]
    print(f"[持仓同步] {len(codes)}只: {', '.join(codes)}")

    # 2. 拉实时价
    prices = fetch_prices(codes)
    if not prices:
        print("[WRAN] 无实时价数据，跳过更新")
        return

    print(f"[持仓同步] 获取到{len(prices)}只的实时价")

    # 3. 更新持仓
    total_market_value = 0
    changes = []
    for h in holdings:
        code = h['code']
        if code in prices:
            old_price = h.get('price', 0)
            new_price = prices[code]['price']
            qty = h.get('qty', 0)
            mv = round(new_price * qty, 2)
            change_pct = round((new_price - old_price) / old_price * 100, 2) if old_price else 0
            h['price'] = new_price
            h['market_value'] = mv
            h['last_price_date'] = prices[code]['date']
            total_market_value += mv
            changes.append(f"{code}: {old_price}->{new_price} ({change_pct:+.2f}%)")
        else:
            total_market_value += h.get('market_value', 0)

    # 4. 更新汇总
    pf['total_market_value'] = round(total_market_value, 2)
    pf['total_assets'] = round(total_market_value + pf.get('available_cash', 0), 2)
    pf['last_sync'] = prices.get(codes[0], {}).get('date', '') if codes else ''

    for h in holdings:
        mv = h.get('market_value', 0)
        h['weight_pct'] = round(mv / total_market_value * 100, 1) if total_market_value else 0

    # 5. 写入
    with open(PORTFOLIO_PATH, 'w', encoding='utf-8') as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)
    print(f"[持仓同步] ✅ 已更新 {PORTFOLIO_PATH}")

    # 6. 输出摘要
    print(f"\n{'='*50}")
    print(f"[持仓摘要]")
    for h in holdings:
        print(f"  {h['code']}: {h.get('qty',0)}股 @ ${h.get('price',0):.2f} | 市值${h.get('market_value',0):.2f} | {h.get('weight_pct',0):.1f}%")
    print(f"  现金: ${pf.get('available_cash',0):.2f}")
    print(f"  总资产: ${pf['total_assets']:.2f}")
    print(f"  持仓占比: {round(total_market_value/pf['total_assets']*100,1) if pf['total_assets'] else 0}%")
    for c in changes:
        print(f"  📈 {c}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
