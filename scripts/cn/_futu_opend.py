#!/usr/bin/env python3
"""
_futu_opend.py — Futu OpenD 连接模板
======================================
封装成独立模块，任何脚本只需要:
    from _futu_opend import get_holdings, test_connection, OPEND_HOST, OPEND_PORT

用法:
    python _futu_opend.py              # 测试连接 + 打印持仓
    python _futu_opend.py --list       # 同上
    python _futu_opend.py --code NVDA  # 查单只持仓信息

依赖: futu (pip install futu)
"""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# ====== OpenD 连接参数（一改全改） ======
OPEND_HOST = '127.0.0.1'
OPEND_PORT = 11111
OPEND_TIMEOUT = 5  # 连接超时秒数

# ====== 持仓缓存路径（OpenD离线时的降级方案） ======
WORKSPACE = Path(r'/home/hermes/.hermes/openclaw-archive')
PORTFOLIO_CACHE = WORKSPACE / 'data' / 'portfolio.json'


def test_connection(silent=False):
    """
    测试 OpenD 连通性。
    返回: (ok: bool, msg: str)
    如果 silent=True, 不打印日志。
    """
    try:
        import futu as ft
        ctx = ft.OpenQuoteContext(
            host=OPEND_HOST,
            port=OPEND_PORT
        )
        ret, data = ctx.get_global_state()
        ctx.close()
        if ret == ft.RET_OK:
            if not silent:
                print(f'🔌 OpenD 已连接  ({OPEND_HOST}:{OPEND_PORT})')
            return True, 'connected'
        else:
            if not silent:
                print(f'⚠️  OpenD 返回异常: {data}')
            return False, str(data)
    except ImportError:
        return False, 'futu SDK未安装'
    except ConnectionRefusedError:
        if not silent:
            print(f'🔌 OpenD 未启动  ({OPEND_HOST}:{OPEND_PORT})')
        return False, 'connection refused'
    except Exception as e:
        if not silent:
            print(f'🔌 OpenD 连接异常: {e}')
        return False, str(e)


def get_holdings(silent=False, cache_fallback=True):
    """
    从 OpenD 拉实时持仓。
    
    返回: [ (code, qty, cost, market_val, pl_ratio, pl_val), ... ]
        code = 纯股票代码 (如 'NVDA', 不带 US. 前缀)
    
    如果 OpenD 连接失败 + cache_fallback=True:
        读 portfolio.json 缓存
    全部失败:
        返回 []
    
    silent=True → 不打印日志
    """
    # 尝试 OpenD
    try:
        import futu as ft
        ctx = ft.OpenSecTradeContext(
            filter_trdmarket=ft.TrdMarket.US,
            host=OPEND_HOST,
            port=OPEND_PORT,
            security_firm=ft.SecurityFirm.FUTUSECURITIES
        )
        ret, data = ctx.position_list_query(trd_env=ft.TrdEnv.REAL)
        ctx.close()
        
        if ret != ft.RET_OK:
            raise RuntimeError(f'position_list 返回: {ret}')
        
        holdings = []
        for _, row in data.iterrows():
            code = row['code'].replace('US.', '')
            holdings.append((
                code,
                int(row['qty']),
                float(row['cost_price']),
                float(row['market_val']),
                float(row['pl_ratio']),
                float(row['pl_val']),
            ))
        
        if not silent:
            print(f'📡 OpenD 持仓: {len(holdings)} 只')
            for c, q, cost, mv, plr, plv in holdings:
                print(f'  {c:>6s}  {q:>4d}股  成本${cost:<8.2f}  市值${mv:<8.2f}  '
                      f'盈亏{plr:+.2f}%(${plv:+.2f})')
        return holdings
    
    except Exception as e:
        if not silent:
            print(f'⚠️  OpenD: {e}')
        
        if cache_fallback:
            return _load_cache_holdings(silent)
        return []


def get_holding_by_code(code, silent=True):
    """
    查单只股票在不在持仓里。
    返回: (code, qty, cost, market_val, pl_ratio) 或 None
    """
    holdings = get_holdings(silent=True)
    code = code.upper().replace('US.', '')
    for h in holdings:
        if h[0] == code:
            return h
    return None


def _load_cache_holdings(silent=False):
    """读 portfolio.json 缓存"""
    if not PORTFOLIO_CACHE.exists():
        if not silent:
            print(f'📡 无持仓数据 (OpenD离线 + 无缓存)')
        return []
    try:
        with open(PORTFOLIO_CACHE, encoding='utf-8') as f:
            data = json.load(f)
        holdings_raw = data.get('holdings', [])
        # 转成统一格式
        holdings = []
        for h in holdings_raw:
            holdings.append((
                h['code'],
                int(h.get('qty', 0)),
                float(h.get('cost', 0)),
                float(h.get('market_val', h.get('price', 0)) * h.get('qty', 0)) if h.get('market_val') else 0,
                float(h.get('pl_ratio', 0)),
                float(h.get('pl_val', 0)),
            ))
        if not silent:
            print(f'📡 持仓缓存(OpenD离线): {len(holdings)} 只')
            for c, q, cost, mv, plr, plv in holdings:
                print(f'  {c:>6s}  {q:>4d}股  成本${cost:<8.2f}  市值${mv:<8.2f}  '
                      f'盈亏{plr:+.2f}%(${plv:+.2f})')
        return holdings
    except Exception as e:
        if not silent:
            print(f'📡 持仓缓存读取失败: {e}')
        return []


def get_codes_only(silent=False):
    """只拿持仓股票代码列表 → ['NVDA', 'ON', ...]"""
    holdings = get_holdings(silent=silent, cache_fallback=True)
    return [h[0] for h in holdings]


# ====== 独立运行 ======

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Futu OpenD 连接工具')
    parser.add_argument('--list', action='store_true', help='列出持仓')
    parser.add_argument('--code', type=str, help='查单只持仓')
    parser.add_argument('--test', action='store_true', default=True, help='测试连接(默认)')
    args = parser.parse_args()
    
    # 默认行为: 测连接 + 列持仓
    ok, msg = test_connection()
    
    if args.code:
        h = get_holding_by_code(args.code)
        if h:
            print(f'\n📋 {h[0]}:')
            print(f'  数量:   {h[1]}股')
            print(f'  成本价: ${h[2]:.2f}')
            print(f'  市值:   ${h[3]:.2f}')
            print(f'  盈亏比: {h[4]:+.2f}%')
            print(f'  盈亏额: ${h[5]:+.2f}')
        else:
            print(f'\n❌ {args.code} 不在持仓中')
    else:
        get_holdings()


if __name__ == '__main__':
    main()
