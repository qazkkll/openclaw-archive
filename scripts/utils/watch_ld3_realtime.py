#!/usr/bin/env python3
"""
watch_ld3_realtime.py — 蓝盾3.0 盘中实时监控
==============================================
数据源:
 ① _futu_opend.py → OpenD 持仓模板
 ② 桌面 watchlist.txt → 自定义自选
 ③ minishare → 批量实时价

用法:
  python scripts/watch_ld3_realtime.py           # 默认10分钟轮询
  python scripts/watch_ld3_realtime.py --interval 300  # 5分钟
  python scripts/watch_ld3_realtime.py --once          # 跑一轮就退出

桌面自选文件: /home/hermes/Desktop/watchlist.txt
  一行一个代码, #开头为注释
  例:
    # 我的自选
    AAPL
    TSLA
    NVDA

OpenD 模板: scripts/_futu_opend.py
  所有脚本从这里导入, 一改全改
"""
import sys, os, json, time, datetime, argparse, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

# ====== 路径配置 ======
WORKSPACE  = Path(r'/home/hermes/.hermes/openclaw-archive')
DATA_DIR   = Path(r'/home/hermes/.hermes/openclaw-archive/data')
DESKTOP    = Path(r'C:\Users\admin\Desktop')
WATCHLIST_FILE = DESKTOP / 'watchlist.txt'
SIGNALS_FILE   = DATA_DIR  / 'realtime_signals.json'

# ====== OpenD 模板（一劳永逸） ======
sys.path.insert(0, str(WORKSPACE / 'scripts'))
from _futu_opend import (
    get_holdings as _get_opend_raw,
    get_codes_only as _get_opend_codes,
    test_connection as _test_opend,
    PORTFOLIO_CACHE,
)

# ====== 报警阈值 ======
ENTRY_THRESHOLD = 80
EXIT_THRESHOLD  = 70
STOP_LOSS_PCT   = -15.0
STOP_WARN_PCT   = -12.0


# ─────────────────────────────────────────────
#  ① OpenD 持仓（统一走模板）
# ─────────────────────────────────────────────

def get_holdings_with_meta():
    """
    拉持仓(OpenD) + 元数据(是否实时, 来源)
    返回 (holdings_list, source_str)
    holdings_list = [{'code','qty','cost','market_val','pl_ratio'}, ...]
    """
    raw = _get_opend_raw(silent=True, cache_fallback=False)
    if raw:
        holdings = []
        for h in raw:
            holdings.append({
                'code':       h[0],
                'qty':        h[1],
                'cost':       h[2],
                'market_val': h[3],
                'pl_ratio':   h[4],
            })
        return holdings, 'OpenD'

    # OpenD 失败 → 读缓存
    if PORTFOLIO_CACHE.exists():
        try:
            with open(PORTFOLIO_CACHE, encoding='utf-8') as f:
                data = json.load(f)
            cached = data.get('holdings', [])
            holdings = []
            for h in cached:
                holdings.append({
                    'code':       h['code'],
                    'qty':        int(h.get('qty', 0)),
                    'cost':       float(h.get('cost', 0)),
                    'market_val': float(h.get('market_val', 0)),
                    'pl_ratio':   float(h.get('pl_ratio', 0)),
                })
            return holdings, 'cache'
        except:
            pass
    return [], 'none'


# ─────────────────────────────────────────────
#  ② 桌面自选文件 watchlist.txt
# ─────────────────────────────────────────────

def read_watchlist():
    """从桌面 watchlist.txt 读取自选股代码"""
    if not WATCHLIST_FILE.exists():
        return []
    codes = []
    with open(WATCHLIST_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            code = line.upper().strip()
            if code and code.isalpha():
                codes.append(code)
    return list(set(codes))


def ensure_watchlist_file():
    """如果文件不存在, 创建模板"""
    if not WATCHLIST_FILE.exists():
        template = """# 蓝盾监控自选股 — 一行一个代码, #开头为注释
# 改完存盘后脚本会在下一轮自动重读
#
# --- 常用大盘股 ---
# AAPL
# MSFT
# NVDA
# GOOGL
# META
#
# --- 持仓由OpenD自动获取, 不用写这里 ---
"""
        with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f:
            f.write(template)
        return True
    return False


# ─────────────────────────────────────────────
#  ③ minishare 实时价
# ─────────────────────────────────────────────

MS_TOKEN = 'Jarvne6fmgArRa46Xfon0e1kw55E6hes5IB2Fy2X0ndqnvrL48jsVOtTbf014f06'
_MS_API = None

def _ms_api():
    global _MS_API
    if _MS_API is None:
        import minishare as ms
        _MS_API = ms.pro_api(MS_TOKEN)
    return _MS_API


def get_realtime_prices(codes):
    """minishare 批量拉实时价 → {code: {close, pct_chg, ...}}"""
    if not codes:
        return {}
    try:
        codes_str = ','.join(codes)
        df = _ms_api().rt_us_k(ts_code=codes_str, extFields='date,open,high,low,volume')
        result = {}
        for _, row in df.iterrows():
            result[row['ts_code']] = {
                'close':   float(row['close']),
                'open':    float(row['open']),
                'high':    float(row['high']),
                'low':     float(row['low']),
                'volume':  int(row.get('volume', 0)),
                'pct_chg': float(row.get('pct_chg', 0)),
                'change':  float(row.get('change', 0)),
                'time':    str(row.get('date', '')),
            }
        return result
    except Exception as e:
        print(f'  ⚠️ minishare: {e}')
        return {}


# ─────────────────────────────────────────────
#  ④ 评分估算
# ─────────────────────────────────────────────

def load_baseline():
    """加载最新评分文件"""
    scored_dir = DATA_DIR
    files = sorted(scored_dir.glob('ld3_scored_*.json'), reverse=True)
    if not files:
        files = sorted(scored_dir.glob('fusion_rec_*.json'), reverse=True)
    if not files:
        print(f'❌ 未找到评分文件')
        return {}, '', ''
    with open(files[0], encoding='utf-8') as f:
        data = json.load(f)
    scores_map = {}
    for s in data.get('scores', []):
        scores_map[s['code']] = s
    return scores_map, files[0].name, data.get('date', '')


def estimate_score(baseline_score, pct_chg):
    """
    盘中估算评分。
    核心假设: 大盘股弹性约每±1% = ±2分。
    大幅波动边际递减:
        >5%  →  +10 + (pct-5)*1
        <-5%  →  -10 + (pct+5)*1
    """
    if pct_chg is None:
        return baseline_score, 0.0
    delta = pct_chg * 2.0
    if pct_chg > 5:
        delta = 10.0 + (pct_chg - 5) * 1.0
    elif pct_chg < -5:
        delta = -10.0 + (pct_chg + 5) * 1.0
    est = baseline_score + delta
    return max(0, min(100, round(est, 1))), round(delta, 1)


# ─────────────────────────────────────────────
#  ⑤ 信号检测 + 写文件
# ─────────────────────────────────────────────

def detect_signals(monitor_items, holdings_codes, holdings_full, baseline_map):
    """
    核心逻辑: 对比估算评分 vs 阈值 → 返回信号列表
    holdings_full = [ {code, qty, cost, market_val, pl_ratio}, ... ]
    """
    signals = []
    cost_map = {h['code']: h for h in holdings_full}

    for item in monitor_items:
        code   = item['code']
        rp     = item.get('realtime_price')
        pct    = item.get('pct_chg')
        base_s = item.get('base_score', 0)
        est_s  = item.get('est_score', 0)
        is_held = code in holdings_codes

        # --- 止损预警 ---
        if is_held and rp and code in cost_map:
            h = cost_map[code]
            cost = float(h.get('cost', 0))
            if cost > 0:
                loss_pct = (rp - cost) / cost * 100
                if loss_pct <= STOP_WARN_PCT:
                    signals.append({
                        'type': 'stop_loss',
                        'code': code,
                        'msg': f'🔴 {code} 止损预警: {loss_pct:.1f}% (止损线{STOP_LOSS_PCT}%)',
                        'detail': {
                            'loss_pct': round(loss_pct, 1),
                            'price': rp,
                            'cost': cost,
                            'qty': h.get('qty', 0),
                        }
                    })

        # 跳过无基线分的股票（不在蓝盾评分池里）
        if base_s <= 0:
            continue

        # --- 买入机会 (未持仓 && 评分≥80) ---
        if not is_held and est_s >= ENTRY_THRESHOLD:
            signals.append({
                'type': 'entry',
                'code': code,
                'msg': f'🟢 {code} 买入机会: 评分{est_s:.0f}, ${rp}, ({pct:+.2f}%)',
                'detail': {
                    'est_score': round(est_s, 0),
                    'price': rp,
                    'pct_chg': pct,
                }
            })

        # --- 卖出信号 (持仓 && 评分<70) ---
        if is_held and est_s < EXIT_THRESHOLD:
            signals.append({
                'type': 'exit',
                'code': code,
                'msg': f'🔵 {code} 卖出信号: 当前评分{est_s:.0f}, 低于{EXIT_THRESHOLD}线',
                'detail': {
                    'est_score': round(est_s, 0),
                    'price': rp,
                    'pct_chg': pct,
                }
            })

    return signals


def write_signals(signals, monitor_items, holdings_codes):
    """写信号文件 + 去重"""
    old_sigs = []
    if SIGNALS_FILE.exists():
        try:
            with open(SIGNALS_FILE, encoding='utf-8') as f:
                old_sigs = json.load(f).get('all', [])
        except:
            pass

    new_sigs = []
    for s in signals:
        dup = any(o.get('code') == s['code'] and o.get('type') == s['type']
                  for o in old_sigs[-20:])  # 只查最近20条
        if not dup:
            s['time'] = datetime.datetime.now().isoformat()
            new_sigs.append(s)

    if not new_sigs:
        return 0

    all_sigs = old_sigs + new_sigs
    if len(all_sigs) > 100:
        all_sigs = all_sigs[-100:]

    snapshot = {
        'time': datetime.datetime.now().isoformat(),
        'holdings': [],
    }
    for code in holdings_codes:
        for item in monitor_items:
            if item['code'] == code:
                snapshot['holdings'].append({
                    'code': code,
                    'price': item.get('realtime_price'),
                    'pct_chg': item.get('pct_chg'),
                    'est_score': item.get('est_score'),
                })
                break

    payload = {
        'signals': new_sigs,
        'all': all_sigs,
        'snapshot': snapshot,
    }

    with open(SIGNALS_FILE, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return len(new_sigs)


# ─────────────────────────────────────────────
#  ⑥ 终端显示
# ─────────────────────────────────────────────

def color(text, code):
    return f'\033[{code}m{text}\033[0m'


def print_status(monitor_items, holdings_codes, signals):
    now = datetime.datetime.now().strftime('%H:%M:%S')
    print(f'\n{"─"*60}')
    print(f'  ⏱️  {now} HKT')
    print(f'{"─"*60}')

    sorted_items = sorted(monitor_items, key=lambda x: -x['est_score'])

    print(f'  {"代码":>6s}  评分  实时价    涨跌    Δ   持仓 来源')
    print(f'  {"─"*42}')

    for item in sorted_items:
        code = item['code']
        es   = item['est_score']
        rp   = item.get('realtime_price')
        pct  = item.get('pct_chg')
        dd   = item.get('delta', 0)
        src  = item.get('source', '')
        is_h = '🟢' if code in holdings_codes else '  '

        p_str = f'{pct:+.2f}%' if pct is not None else ' N/A '
        r_str = f'${rp:.2f}' if rp else '  N/A  '

        if es >= 80:   clr = '92'
        elif es >= 70: clr = '93'
        else:          clr = '91'
        delta_clr = '96' if dd >= 0 else '91'

        c_code  = color(code.rjust(6), clr)
        c_score = color(f'{es:>3.0f}', clr)
        c_delta = color(f'{dd:>+3.0f}', delta_clr)
        line = f'  {c_code}  {c_score:>4s}  {r_str:>7s}  {p_str:>7s}  {c_delta:>4s}  {is_h}  {src}'
        print(line)

    if signals:
        print(f'\n  ┌─── 信号 ────')
        for s in signals:
            clr = '91' if s['type'] == 'stop_loss' else '92' if s['type'] == 'entry' else '94'
            print(f'  │ {color(s["msg"], clr)}')
        print(f'  └────────────')
    else:
        print(f'\n  ✅ 无触发信号')

    print(f'\n  📋 持仓:')
    for code in holdings_codes:
        for item in sorted_items:
            if item['code'] == code:
                es = item['est_score']
                rp = item.get('realtime_price')
                pc = item.get('pct_chg')
                icon = '🟢' if es >= 80 else '🟡' if es >= 70 else '🔴'
                print(f'     {icon} {code:6s} 测{es:>3.0f}  ${rp:>6.2f} ({pc:+.2f}%)' if rp
                      else f'     ⚫ {code:6s} 测{es:>3.0f}  N/A')
                break
        else:
            print(f'     ⚫ {code:6s}  无数据')


# ─────────────────────────────────────────────
#  ⑦ 主循环
# ─────────────────────────────────────────────

def build_monitor_list(watchlist_codes, holdings_codes):
    """合并监控列表: 持仓 + 自选 (不含全量SP500)"""
    codes = set()
    codes.update(holdings_codes)
    codes.update(watchlist_codes)
    return sorted(codes)


def main():
    parser = argparse.ArgumentParser(description='蓝盾3.0 盘中实时监控')
    parser.add_argument('--interval', type=int, default=600, help='轮询间隔秒数(默认600=10分)')
    parser.add_argument('--once', action='store_true', help='只跑一轮')
    args = parser.parse_args()

    # === 启动检查 ===
    ensure_watchlist_file()

    try:
        import minishare as ms
    except ImportError:
        print('❌ minishare 未安装: pip install minishare')
        sys.exit(1)



    baseline_map, fname, bdate = load_baseline()
    if not baseline_map:
        sys.exit(1)
    print(f'📄 基线: {fname} (日期: {bdate})')

    # OpenD 持仓
    all_holdings, hsrc = get_holdings_with_meta()
    holdings_codes = [h['code'] for h in all_holdings]
    print(f'📡 持仓({hsrc}): {len(all_holdings)} 只' if all_holdings else '📡 持仓: 无数据')

    watchlist_codes = read_watchlist()
    if watchlist_codes:
        print(f'📋 自选({WATCHLIST_FILE.name}): {", ".join(watchlist_codes)}')

    monitor_codes = build_monitor_list(watchlist_codes, holdings_codes)
    print(f'🎯 监控: {len(monitor_codes)} 只 (持仓{len(holdings_codes)}+自选{len(watchlist_codes)})')
    print(f'{"🟢" if hsrc=="OpenD" else "🟡"} OpenD: {"已连接" if hsrc=="OpenD" else "离线, 使用缓存"}')
    print(f'⏱️  轮询: 每{args.interval}s  (Ctrl+C中断)')
    print(f'📬 信号文件: {SIGNALS_FILE}')
    print(f'📝 自选修改: 编辑桌面 watchlist.txt → 自动生效')
    print(f'{"="*50}')

    cycle = 0
    while True:
        cycle += 1
        print(f'\n🔄 第{cycle}轮 — {datetime.datetime.now().strftime("%H:%M:%S")}')

        # 每轮重读自选
        watchlist_codes = read_watchlist()
        monitor_codes = build_monitor_list(watchlist_codes, holdings_codes)

        rt = get_realtime_prices(monitor_codes)
        if not rt:
            print('  ⚠️ 未获取实时数据')
        else:
            items = []
            for code in monitor_codes:
                bs = baseline_map.get(code, {}).get('score', 0)
                rtd = rt.get(code)
                pct = rtd.get('pct_chg') if rtd else None
                es, dd = estimate_score(bs, pct)

                if code in holdings_codes:
                    src = '持仓'
                elif code in watchlist_codes:
                    src = '自选'
                else:
                    src = ''

                items.append({
                    'code': code,
                    'base_score': bs,
                    'est_score': es,
                    'delta': dd,
                    'realtime_price': rtd.get('close') if rtd else None,
                    'pct_chg': pct,
                    'source': src,
                })

            signals = detect_signals(items, holdings_codes, all_holdings, baseline_map)
            new_count = write_signals(signals, items, holdings_codes)
            if new_count > 0:
                print(f'  📬 新信号 {new_count} 条 → {SIGNALS_FILE}')

            print_status(items, holdings_codes, signals)

        if args.once:
            break

        wait = max(args.interval - (time.time() % args.interval), 1)
        print(f'\n{"·"*40}')
        print(f'  下一轮 {wait:.0f}s... (编辑桌面watchlist.txt立即生效 | Ctrl+C停止)')
        print(f'{"·"*40}')
        try:
            time.sleep(min(wait, args.interval))
        except KeyboardInterrupt:
            print('\n\n  🛑 监控已停止')
            break


if __name__ == '__main__':
    main()
