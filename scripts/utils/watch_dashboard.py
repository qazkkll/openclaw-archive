#!/usr/bin/env python3
"""
watch_dashboard.py — 小钳盯盘仪 v2
====================================
数据流:
  OpenD → 持仓成本/盈亏
  minishare → 实时价（持仓+自选+SP500扫描）
  us_hist_clean.parquet → 历史K线 → v5s_calc出指标
  v5s_score → 实时评分

终端布局:
  📊 大盘风向标
  🛡️ 蓝盾持仓 (逐项指标灯 + 综合灯)
  💹 其他持仓（绿箭/小盘）
  📋 自选股
  🔥 SP500 盘中突破
  💡 建议关注 (写入 buy_signals.txt)

用法:
  python scripts/watch_dashboard.py           # 每600s轮询
  python scripts/watch_dashboard.py --once     # 跑一轮
  python scripts/watch_dashboard.py --demo     # 模拟盘(测试用)

输出文件:
  /home/hermes/.hermes/openclaw-project/data/realtime_signals.json → cron推Telegram
  /home/hermes/.hermes/openclaw-project/data/buy_signals.txt       → 你读文件来找我
"""
import sys, os, json, ctypes, time, datetime, warnings, traceback

# Enable ANSI on Windows
if sys.platform == "win32":
    _kd = ctypes.windll.kernel32
    _mode = ctypes.c_uint32()
    if _kd.GetConsoleMode(_kd.GetStdHandle(-11), ctypes.byref(_mode)):
        _kd.SetConsoleMode(_kd.GetStdHandle(-11), _mode.value | 0x0004)
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd

# ====== 路径 ======
WORKSPACE = Path(r'/home/hermes/.hermes/openclaw-archive')
DATA_DIR  = Path(r'/home/hermes/.hermes/openclaw-archive/data')
DESKTOP   = Path(r'C:\Users\admin\Desktop')
SCRIPTS   = WORKSPACE / 'scripts'
sys.path.insert(0, str(SCRIPTS))

WATCHLIST_FILE   = DESKTOP / 'watchlist.txt'
SIGNALS_FILE     = DATA_DIR  / 'realtime_signals.json'
BUY_SIGNALS_FILE = DATA_DIR  / 'buy_signals.txt'
HIST_FILE        = DATA_DIR  / 'us_hist_clean.parquet'
SP500_FILE       = DATA_DIR  / 'sp500_list.json'

# ====== OpenD 模板 ======
from _futu_opend import get_holdings as _get_opend

# ====== 蓝盾评分引擎 ======
from us_score_engine import v5s_calc, v5s_score

# ====== 参数 ======
ENTRY_THRESHOLD = 80
EXIT_THRESHOLD  = 70
STOP_LOSS_PCT   = -15.0
STOP_WARN_PCT   = -12.0

GREEN_TICK = '>=' + str(ENTRY_THRESHOLD)
RED_CROSS  = '<' + str(EXIT_THRESHOLD)
SL_PCT_STR = str(abs(STOP_LOSS_PCT))

# ====== ANSI ======
CYAN  = '\033[96m'
GREEN = '\033[92m'
YELL  = '\033[93m'
RED   = '\033[91m'
BLUE  = '\033[94m'
BOLD  = '\033[1m'
ENDC  = '\033[0m'

def c(text, code):
    return code + text + ENDC

def hbar(title, w=57):
    return '╔══ ' + title + ' ═' + '═' * (w - 6 - len(title)) + '╗'

def fbar(w=57):
    return '╚' + '═' * (w - 2) + '╝'

def sep(w=57):
    return '║  ' + '─' * (w - 5)

def pct_str(v):
    if v is None:
        return '  N/A  '
    s = f'{v:+.2f}%'
    if v > 2:
        return c(s, GREEN)
    if v > 0:
        return c(s, CYAN)
    if v > -2:
        return c(s, RED)
    return c(s, BOLD + RED)

# ════════════════════════════════════════════
#  数据加载
# ════════════════════════════════════════════

_HIST_CACHE = None

def load_hist(codes):
    """从缓存或文件加载指定股票的历史K线（全局缓存，只读一次文件）"""
    global _HIST_CACHE
    if _HIST_CACHE is None:
        if not HIST_FILE.exists():
            print('  [ERROR] 历史数据不存在: ' + str(HIST_FILE))
            return {}
        with open(HIST_FILE, encoding='utf-8') as f:
            _HIST_CACHE = json.load(f)
        print('  📂 历史数据已缓存 (' + str(len(_HIST_CACHE)) + ' 只股票)')
    result = {}
    for code in codes:
        cu = code.upper()
        if cu in _HIST_CACHE:
            result[cu] = _HIST_CACHE[cu]
    return result

def load_baseline():
    files = sorted(DATA_DIR.glob('ld3_scored_*.json'), reverse=True)
    if not files:
        return {}, '', ''
    with open(files[0], encoding='utf-8') as f:
        data = json.load(f)
    sm = {}
    for s in data.get('scores', []):
        sm[s['code']] = s
    return sm, files[0].name, data.get('date', '')

def load_sp500():
    if not SP500_FILE.exists():
        return []
    with open(SP500_FILE, encoding='utf-8') as f:
        d = json.load(f)
    if isinstance(d, dict):
        return d.get('syms', [])
    return d

# ════════════════════════════════════════════
#  OpenD持仓
# ════════════════════════════════════════════

def get_holdings_data():
    raw = _get_opend(silent=True, cache_fallback=False)
    if raw:
        hl = [{'code':h[0],'qty':h[1],'cost':h[2],
               'market_val':h[3],'pl_ratio':h[4],'pl_val':h[5]}
              for h in raw]
        return hl, 'OpenD'
    cp = WORKSPACE / 'data' / 'portfolio.json'
    if cp.exists():
        with open(cp, encoding='utf-8') as f:
            dd = json.load(f)
        ca = dd.get('holdings', [])
        hl = [{'code':h['code'],'qty':int(h.get('qty',0)),
               'cost':float(h.get('cost',0)),
               'market_val':float(h.get('market_val',0)),
               'pl_ratio':float(h.get('pl_ratio',0)),
               'pl_val':float(h.get('pl_val',0))}
              for h in ca]
        return hl, 'cache'
    return [], 'none'

# ════════════════════════════════════════════
#  minishare
# ════════════════════════════════════════════

MS_TOKEN = 'Jarvne6fmgArRa46Xfon0e1kw55E6hes5IB2Fy2X0ndqnvrL48jsVOtTbf014f06'
_MS_API = None

def ms_api():
    global _MS_API
    if _MS_API is None:
        import minishare as ms
        _MS_API = ms.pro_api(MS_TOKEN)
    return _MS_API

def get_realtime(codes):
    if not codes:
        return {}
    try:
        s = ','.join(codes)
        df = ms_api().rt_us_k(ts_code=s, extFields='date,open,high,low,volume')
        r = {}
        for _, row in df.iterrows():
            r[row['ts_code']] = {
                'close': float(row['close']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'volume': int(row.get('volume', 0)),
                'pct_chg': float(row.get('pct_chg', 0)),
                'change': float(row.get('change', 0)),
                'time': str(row.get('date', '')),
            }
        return r
    except:
        return {}

# ════════════════════════════════════════════
#  评分引擎
# ════════════════════════════════════════════

def _norm_hist(hist):
    """Normalize hist dict: short keys (c/h/l/v) -> full keys (close/high/low/volume)"""
    if not hist or 'open' in hist:
        return hist
    return {'open': hist.get('c', []), 'high': hist.get('h', []),
            'low': hist.get('l', []), 'close': hist.get('c', []),
            'volume': hist.get('v', [])}

def compute_rt(code, hist, rt_price):
    if not hist:
        return None
    o = [float(x) for x in hist['open']]
    h = [float(x) for x in hist['high']]
    lo = [float(x) for x in hist['low']]
    c = [float(x) for x in hist['close']]
    c.append(rt_price)
    o.append(rt_price)
    h.append(max(rt_price, h[-1]))
    lo.append(min(rt_price, lo[-1]))
    if len(c) < 70:
        return None
    ind = v5s_calc(c, h, lo)
    if ind is None:
        return None
    score = v5s_score(ind, len(c) - 1)
    score = max(0, min(100, int(round(score))))

    # 均线 (use ma20 from v5s_calc)
    ma20 = ind.get('ma20', [])
    ma20v = ma20[-1] if ma20 and ma20[-1] is not None else 0
    pct_vs_ma = (rt_price / ma20v - 1) * 100 if ma20v else 0
    ma_sig = '🟢' if pct_vs_ma > 1.0 else ('🟡' if pct_vs_ma > -0.5 else '🔴')

    # MACD
    macd_a = ind.get('macd', [])
    macd_sl = ind.get('macd_signal', [])
    if macd_a and macd_sl and len(macd_a) > 0 and len(macd_sl) > 0:
        mc, ms_ = macd_a[-1], macd_sl[-1]
        if mc is not None and ms_ is not None:
            macd_sig = '🟢' if (mc > ms_ and mc > 0) else ('🟡' if mc > ms_ else '🔴')
        else:
            macd_sig = '🟡'
    else:
        macd_sig = '🟡'

    # RSI
    rsi_arr = ind.get('rsi', [])
    rsi_val = rsi_arr[-1] if rsi_arr and rsi_arr[-1] is not None else 50
    rsi_sig = '🟢' if 50 <= rsi_val <= 70 else ('🔴' if (rsi_val < 30 or rsi_val > 80) else '🟡')

    # KDJ - compute from local price data
    kdj_sig = '🟡'
    j_val = k_val = d_val = 50
    try:
        cp = [float(x) for x in c]
        hp = [float(x) for x in h]
        lp = [float(x) for x in lo]
        if len(cp) > 14:
            ll = [min(lp[i-9:i]) for i in range(9, len(lp))]
            hh = [max(hp[i-9:i]) for i in range(9, len(hp))]
            rsv = [(cp[i] - ll[i]) / (hh[i] - ll[i]) * 100 if hh[i] > ll[i] else 50 for i in range(min(len(ll), len(hp)-9))]
            k_arr = [50]
            for r in rsv: k_arr.append(2/3 * k_arr[-1] + 1/3 * r)
            d_arr = [50]
            for kk in k_arr: d_arr.append(2/3 * d_arr[-1] + 1/3 * kk)
            j_arr = [3*k - 2*d for k,d in zip(k_arr, d_arr)]
            if j_arr: j_val = j_arr[-1]
            if k_arr: k_val = k_arr[-1]
            if d_arr: d_val = d_arr[-1]
            if j_val > 80 or j_val < 20:
                kdj_sig = '🔴'
            elif k_val > d_val and j_val > 50:
                kdj_sig = '🟢'
            elif k_val < d_val and j_val < 50:
                kdj_sig = '🔴'
            else:
                kdj_sig = '🟡'
    except Exception:
        pass

    # 52周
    p52_arr = ind.get('p52', [])
    p52 = p52_arr[-1] if p52_arr and p52_arr[-1] is not None else 0
    p52_sig = '🟢' if p52 > 70 else ('🟡' if p52 > 40 else '🔴')

    # composite = 模型评分信号，指标灯是参考信息不做决策
    if score >= 80:
        composite = '🟢'
    elif score >= 70:
        composite = '🟡'
    else:
        composite = '🔴'

    # 量比（当日量 / 20日均量）
    vols = hist.get('volume', [])
    if len(vols) > 0:
        cur_vol = float(vols[-1])
        avg_vol = sum(float(vols[i]) for i in range(-21, -1)) / 20 if len(vols) >= 21 else cur_vol
        vol_ratio = round(cur_vol / avg_vol, 1) if avg_vol > 0 else 1.0
    else:
        vol_ratio = 1.0

    return {
        'score': score,
        'ma': ma_sig, 'macd': macd_sig, 'rsi': rsi_sig,
        'kdj': kdj_sig, 'p52': p52_sig, 'composite': composite,
        'ma_pct': round(pct_vs_ma, 1),
        'rsi_val': round(rsi_val, 1),
        'macd_val': round(macd_a[-1] - macd_sl[-1], 2) if (macd_a and macd_sl and len(macd_a) > 0 and len(macd_sl) > 0 and macd_a[-1] is not None and macd_sl[-1] is not None) else 0,
        'j_val': round(j_val, 1),
        'p52_val': round(p52, 1),
        'vol_ratio': vol_ratio,
    }

# ════════════════════════════════════════════
#  绿箭轻量指标
# ════════════════════════════════════════════

def compute_green(code, hist, rt):
    if not hist or not rt:
        return None
    hist = _norm_hist(hist)
    c = [float(x) for x in hist['close']]
    v = [float(x) for x in hist.get('volume', [0] * len(c))]
    avg_vol = sum(v[-20:]) / max(len(v[-20:]), 1)
    vol_ratio = round(rt['volume'] / max(avg_vol, 1), 1) if avg_vol > 0 else 1
    vol_sig = '🟢' if vol_ratio > 1.5 else ('🔴' if vol_ratio < 0.5 else '🟡')
    prev_close = c[-1] if c else None
    gap_pct = (rt['open'] / prev_close - 1) * 100 if prev_close else 0
    h52 = max(c[-252:]) if len(c) > 252 else max(c)
    l52 = min(c[-252:]) if len(c) > 252 else min(c)
    p52 = (rt['close'] - l52) / max((h52 - l52), 0.01) * 100
    return {
        'price': rt['close'],
        'pct': rt.get('pct_chg', 0),
        'vol_ratio': vol_ratio,
        'gap': round(gap_pct, 2),
        'p52': round(p52, 1),
        'vol_sig': vol_sig,
        'gap_sig': '🟢' if abs(gap_pct) < 0.5 else ('🟡' if abs(gap_pct) < 2 else '🔴'),
        'p52_sig': '🟢' if p52 > 70 else ('🟡' if p52 > 40 else '🔴'),
        'composite': '🟢' if vol_ratio > 1.5 and p52 > 50 else \
                     ('🔴' if vol_ratio < 0.5 and p52 < 30 else '🟡'),
    }

# ════════════════════════════════════════════
#  大盘
# ════════════════════════════════════════════

def get_market_data():
    try:
        import yfinance as yf
        sp = yf.Ticker('SPY')
        sp_h = sp.history(period='2d')
        q = yf.Ticker('QQQ')
        q_h = q.history(period='2d')
        v = yf.Ticker('^VIX')
        v_h = v.history(period='2d')
        spx = sp_h['Close'].iloc[-1] if not sp_h.empty else 0
        spx_c = (sp_h['Close'].iloc[-1] / sp_h['Close'].iloc[-2] - 1) * 100 if len(sp_h) > 1 else 0
        ndx = q_h['Close'].iloc[-1] if not q_h.empty else 0
        ndx_c = (q_h['Close'].iloc[-1] / q_h['Close'].iloc[-2] - 1) * 100 if len(q_h) > 1 else 0
        vix = v_h['Close'].iloc[-1] if not v_h.empty else 0
        return {
            'spx': round(spx, 2), 'spx_chg': round(spx_c, 2),
            'spx_sig': '🟢' if spx_c > 0 else '🔴',
            'ndx': round(ndx, 2), 'ndx_chg': round(ndx_c, 2),
            'ndx_sig': '🟢' if ndx_c > 0 else '🔴',
            'vix': round(vix, 2),
            'vix_sig': '🟢' if vix < 20 else ('🟡' if vix < 30 else '🔴'),
        }
    except:
        return {'spx':0,'spx_chg':0,'spx_sig':'🟡',
                'ndx':0,'ndx_chg':0,'ndx_sig':'🟡',
                'vix':0,'vix_sig':'🟡'}

# ════════════════════════════════════════════
#  渲染
# ════════════════════════════════════════════

def rd(market, monitors, hcodes, watch_codes, breakouts, buy_list):
    lines = []
    w = 57
    now = datetime.datetime.now()
    day_name = now.strftime('%a')
    # ── 时间显示: 盘中→实时; 休市→冻结最后交易时间 ──
    now_utc = datetime.datetime.utcnow()
    et_hour = (now_utc.hour - 4 + 24) % 24 + now_utc.minute / 60
    et_day = now_utc.weekday()
    is_open = (et_day < 5 and 9.5 <= et_hour < 16)
    
    # 计算最后交易时间
    if et_day >= 5:  # 周末 → 周五
        last_trade = now - datetime.timedelta(days=et_day - 4)
    elif et_hour < 9.5:  # 盘前 → 昨天
        last_trade = now - datetime.timedelta(days=1)
        if last_trade.weekday() >= 5:
            last_trade -= datetime.timedelta(days=last_trade.weekday() - 4)
    elif et_hour >= 16:  # 盘后 → 今天
        last_trade = now
    else:
        last_trade = now
    lt_str = last_trade.strftime('%a %m/%d')
    
    lines.append('')
    lines.append(c(BOLD + '  📡 小钳盯盘仪', CYAN))
    if is_open:
        time_str = '  ' + day_name + ' ' + now.strftime('%m/%d') + '  ' + now.strftime('%H:%M:%S') + ' HKT  📈 盘中'
    elif et_day >= 5:
        time_str = '  ' + lt_str + '  4:00 PM EDT  🕐 周末休市（最后交易: 周五收盘）'
    elif et_hour < 9.5:
        time_str = '  ' + lt_str + '  4:00 PM EDT  🕐 盘前（最后交易: 上一交易日收盘）'
    else:
        time_str = '  ' + lt_str + '  4:00 PM EDT  🕐 盘后（最后交易: 今日收盘）'
    lines.append(time_str)
    lines.append('')

    # ── 大盘 ──
    lines.append(hbar('📊 大盘风向标', w))
    if market:
        sc = GREEN if market['spx_sig'] == '🟢' else RED
        vc = GREEN if market['vix_sig'] == '🟢' else (YELL if market['vix_sig'] == '🟡' else RED)
        sp = '$' + str(market['spx'])
        np_ = '$' + str(market['ndx'])
        vx = str(market['vix'])
        lines.append('║  SPY ' + c(sp, CYAN) + '  ' + c(market['spx_sig'], sc) + ' ' +
                     pct_str(market['spx_chg']) + '  |  QQQ ' + c(np_, CYAN) + '  ' +
                     c(market['ndx_sig'], sc) + ' ' + pct_str(market['ndx_chg']))
        lines.append('║  VIX ' + c(vx, vc) + '  ' + c(market['vix_sig'], vc) +
                     '  |  💡 ' + GREEN_TICK + '买 | ' + RED_CROSS + '卖 | -' + SL_PCT_STR + '%止损')
    lines.append(fbar(w))

    # ── 蓝盾持仓 ──
    lines.append('')
    lines.append(hbar('🛡️ 蓝盾持仓', w))
    lines.append('║  ' + '代码'.rjust(5) + ' 评分 实时价    日涨   持仓赚  信号')
    lines.append(sep(w))
    bs_list = [m for m in monitors if m.get('sec') == 'blue' and m['code'] in hcodes]
    for m in bs_list:
        sc = m.get('rd', {})
        code = m['code']
        sc_score2 = sc.get('score', 0)
        s_color = GREEN if sc_score2 >= 80 else (YELL if sc_score2 >= 70 else RED)
        line = '║  ' + c(code.rjust(5), s_color) + '  ' + c(str(sc.get('score', 0)).rjust(3), s_color)
        rp = m.get('rp', 0)
        pct = m.get('pct', 0)
        hi = m.get('hi', {})
        hp = hi.get('pl_ratio')
        line += '  $' + str(round(rp, 2)).rjust(7) if rp else '   N/A  '
        line += ' ' + pct_str(pct) + ' '
        if hp is not None:
            line += pct_str(hp) + ' '
        else:
            line += '  N/A   '
        comp = sc.get('composite', '🟡')
        bs_score = sc.get('score', 0)
        if bs_score >= 80:
            line += c('✅购买  ', GREEN)
        elif bs_score >= 70:
            line += c('🟡持有观望', YELL)
        else:
            line += c('🔴关注退出', RED)
        lines.append(line)
        # 指标行
        sigs = '均线' + sc.get('ma', '🟡') + '  MACD' + sc.get('macd', '🟡') + \
               '  RSI强弱' + sc.get('rsi', '🟡') + '  KDJ随机' + sc.get('kdj', '🟡') + \
               '  52周' + sc.get('p52', '🟡') + '  →  ' + comp + '综合'
        lines.append('║        ' + sigs)
        rv = sc.get('rsi_val', '')
        mv = sc.get('macd_val', '')
        mp = sc.get('ma_pct', '')
        vr = sc.get('vol_ratio', '')
        r_str = ('%.1f' % rv) if rv != '' else 'N/A  '
        m_str = ('%.2f' % mv) if mv != '' else 'N/A  '
        mp_str = ('%.1f%%' % mp) if mp != '' else 'N/A  '
        v_str = ('%.1fx' % vr) if vr != '' else 'N/A '
        lines.append('║        ' + c(f'RSI:{r_str:>6s} | MACD:{m_str:>6s} | 均线位:{mp_str:>6s} | 量比:{v_str:>5s}', BLUE))
    if not bs_list:
        lines.append('║  (无蓝盾持仓)')
    lines.append(fbar(w))

    # ── 其他持仓 ──
    lines.append('')
    lines.append(hbar('💹 其他持仓（绿箭/小盘）', w))
    lines.append('║  ' + '代码'.rjust(5) + ' 实时价    日涨    量比   52周%  信号')
    lines.append(sep(w))
    oc = [c for c in hcodes if c not in [m['code'] for m in bs_list]]
    for code in oc:
        m = next((x for x in monitors if x['code'] == code), None)
        if not m:
            lines.append('║  ' + code.rjust(5) + '  N/A')
            continue
        gl = m.get('gl', {})
        rp = m.get('rp', 0)
        pct = m.get('pct', 0)
        vr = gl.get('vol_ratio', '?')
        p52 = gl.get('p52', '?')
        cm = gl.get('composite', '🟡') if gl else '🟡'
        lines.append('║  ' + code.rjust(5) + '  $' + str(round(rp, 2)).rjust(6) +
                     ' ' + pct_str(pct) + '  ' + str(vr) + 'x  ' + str(p52) + '%  ' + cm)
    if not oc:
        lines.append('║  (无其他持仓)')
    lines.append(fbar(w))

    # ── 自选 ──
    lines.append('')
    lines.append(hbar('📋 自选 (' + WATCHLIST_FILE.name + ')', w))
    wl_list = [m for m in monitors if m.get('sec') == 'watch']
    if wl_list:
        lines.append('║     评分  均线  MACD  RSI  随机 52周  综合  价格')
        lines.append(sep(w))
        for m in wl_list:
            sc = m.get('rd', {})
            lines.append('║  ' + m['code'].rjust(5) + '  ' +
                         str(sc.get('score', '?')).rjust(3) + '  ' +
                         sc.get('ma', '🟡') + sc.get('macd', '🟡') +
                         sc.get('rsi', '🟡') + sc.get('kdj', '🟡') + '  ' +
                         sc.get('p52', '🟡') + '  ' + sc.get('composite', '🟡') +
                         '  $' + str(round(m.get('rp', 0), 2)))
    else:
        lines.append('║  (无自选)')
    lines.append(fbar(w))

    # ── SP500突破 ──
    if breakouts:
        lines.append('')
        lines.append(hbar('🔥 SP500 盘中突破', w))
        lines.append('║     评分  均线  MACD  RSI  随机 52周  综合  价格')
        lines.append(sep(w))
        for b in breakouts:
            if 'ma' in b:
                line = '║  ' + c('🚀', GREEN) + ' ' + b['code'].rjust(5) + '  ' + \
                       c(str(b['score']), GREEN).rjust(5) + '  ' + \
                       b.get('ma','🟡') + b.get('macd','🟡') + \
                       b.get('rsi','🟡') + b.get('kdj','🟡') + '  ' + \
                       b.get('p52','🟡') + '  ' + \
                       c(b['composite'], GREEN if b['composite']=='🟢' else RED) + \
                       '  $' + str(round(b['price'], 2)).rjust(7) + '  ' + c('新突破!', BOLD + GREEN)
            else:
                line = '║  ' + c('🚀', GREEN) + ' ' + b['code'].rjust(5) + \
                       '  评分' + c(str(b['score']), GREEN) + \
                       '  $' + str(round(b['price'], 2)).rjust(7) + '  ' + c('新突破!', BOLD + GREEN)
            lines.append(line)
        lines.append(fbar(w))

    # ── 建议关注 ──
    if buy_list:
        lines.append('')
        lines.append(hbar('💡 建议关注', w))
        for b in buy_list[:10]:
            lines.append('║  ' + b['code'].rjust(5) +
                         '  评分' + c(str(b['score']), GREEN) +
                         '  $' + str(round(b['price'], 2)).rjust(7) +
                         '  ' + c('✅全绿灯', GREEN) + '  ' + b.get('note', ''))
        lines.append('║  📝 看 /home/hermes/.hermes/openclaw-project/data/buy_signals.txt')
        lines.append(fbar(w))

    # ── 底部提示 ──
    lines.append(c('  💡 有问题直接发我Telegram', BLUE))
    lines.append(c('  Ctrl+C 关闭监控', YELL))
    lines.append('')
    return '\n'.join(lines)

# ════════════════════════════════════════════
#  信号写入
# ════════════════════════════════════════════

def write_signals(monitors, hcodes, breakouts, buy_list):
    signals = []
    for m in monitors:
        sc = m.get('rd', {})
        code = m['code']
        held = code in hcodes
        comp = sc.get('composite', '🟡')
        score = sc.get('score', 0)
        rp = m.get('rp')
        pct = m.get('pct')
        hi = m.get('hi', {})
        if held and rp and hi.get('cost', 0) > 0:
            loss = (rp - hi['cost']) / hi['cost'] * 100
            if loss <= STOP_WARN_PCT:
                signals.append({
                    'type': 'stop_loss', 'code': code,
                    'msg': '🛑 ' + code + ' 止损预警: ' + ('%.1f' % loss) +
                           '% (止损线-' + SL_PCT_STR + '%)'})
        if not held and comp == '🟢' and score >= ENTRY_THRESHOLD:
            signals.append({
                'type': 'entry', 'code': code,
                'msg': '🟢 ' + code + ' 买入信号: 评分' + str(score) +
                       ', 全绿灯, $' + str(round(rp, 2)) + ', ' + ('%.2f%%' % pct)})
        if held and comp == '🔴' and score < EXIT_THRESHOLD:
            signals.append({
                'type': 'exit', 'code': code,
                'msg': '🔴 ' + code + ' 卖出信号: 评分' + str(score) +
                       ', 全红灯, $' + str(round(rp, 2)) + ', ' + ('%.2f%%' % pct)})
    for b in breakouts:
        signals.append({
            'type': 'breakout', 'code': b['code'],
            'msg': '🚀 ' + b['code'] + ' 盘中突破: 评分' + str(b['score']) +
                   ', $' + str(round(b['price'], 2))})

    old = []
    if SIGNALS_FILE.exists():
        try:
            with open(SIGNALS_FILE, encoding='utf-8') as f:
                old = json.load(f).get('all', [])[-30:]
        except:
            pass
    new_s = []
    for s in signals:
        dup = any(o.get('type') == s['type'] and o.get('code') == s['code'] for o in old[-10:])
        if not dup:
            s['time'] = datetime.datetime.now().isoformat()
            new_s.append(s)
    if new_s:
        all_s = old + new_s
        if len(all_s) > 100:
            all_s = all_s[-100:]
        with open(SIGNALS_FILE, 'w', encoding='utf-8') as f:
            json.dump({'signals': new_s, 'all': all_s}, f, ensure_ascii=False, indent=2)

    if buy_list:
        sl = ['# ' + now_str() + ' 小钳盯盘买入建议',
              '# 看到感兴趣的代码 → 编辑 desktop/ask_xiaoqian.txt 找我',
              '#' + '=' * 50,
              '代码'.rjust(6) + ' | 评分 | 价格     | 日涨     | 状态     | 说明']
        for b in buy_list[:10]:
            label = b.get('label', '')
            sl.append(b['code'].rjust(6) + ' | ' + str(b['score']).rjust(3) +
                      ' | $' + str(round(b['price'], 2)).rjust(7) +
                      ' | ' + ('%.2f%%' % b['pct']).rjust(7) +
                      ' | ' + label.rjust(6) + ' | ' + b.get('note', ''))
        with open(BUY_SIGNALS_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sl))
    return len(new_s)

def now_str():
    return datetime.datetime.now().strftime('%m/%d %H:%M')

# ════════════════════════════════════════════
#  自选
# ════════════════════════════════════════════

def read_watchlist():
    if not WATCHLIST_FILE.exists():
        return []
    codes = []
    with open(WATCHLIST_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            cu = line.upper().strip()
            if cu and cu.isalpha():
                codes.append(cu)
    return list(set(codes))

def ensure_watchlist():
    if not WATCHLIST_FILE.exists():
        with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f:
            f.write('''# 美股专属 · 小钳盯盘仪自选
# 编辑后下一轮自动生效
#
# ─── 蓝盾池（S&P 500 自动评分）───
# 填你关注的代码：
# AAPL
# TSLA
#
# ─── 绿箭池（$1-10小盘轻量监控）───
# BBAI
# UGRO
#
# 格式：一行一个代码，#开头为注释
''')

# ════════════════════════════════════════════
#  Demo
# ════════════════════════════════════════════

def run_demo():
    print('')
    print(c('  ' + BOLD + '🧪 模拟模式 — 展示盯盘仪面板布局', YELL))
    print(c('  ' + BOLD + '图中数据为模拟示例，非实时行情', YELL))
    baseline, _, _ = load_baseline()

    dl = [
        {'code':'NVDA','qty':16,'cost':199.26,'market_val':3283,'pl_ratio':2.97,'pl_val':95},
        {'code':'ON','qty':42,'cost':129.81,'market_val':4903,'pl_ratio':-10.03,'pl_val':-548},
        {'code':'TRGP','qty':8,'cost':272.83,'market_val':2181,'pl_ratio':-0.08,'pl_val':-2},
        {'code':'GS','qty':2,'cost':1007.98,'market_val':2126,'pl_ratio':5.43,'pl_val':109},
        {'code':'ASML','qty':2,'cost':1786.59,'market_val':3727,'pl_ratio':4.31,'pl_val':154},
        {'code':'BBAI','qty':300,'cost':4.07,'market_val':1251,'pl_ratio':-1.23,'pl_val':-15},
        {'code':'UGRO','qty':400,'cost':3.07,'market_val':1128,'pl_ratio':-8.14,'pl_val':-100},
    ]
    dp = {
        'NVDA': {'close':205.19,'pct_chg':0.16,'open':204.80,'high':206.50,'low':204.10,'volume':42500000},
        'ON':   {'close':116.79,'pct_chg':0.72,'open':116.50,'high':117.80,'low':116.00,'volume':3200000},
        'TRGP': {'close':272.60,'pct_chg':1.20,'open':271.50,'high':273.80,'low':271.00,'volume':1800000},
        'GS':   {'close':1062.75,'pct_chg':0.03,'open':1060.00,'high':1065.00,'low':1058.00,'volume':1900000},
        'ASML': {'close':1863.55,'pct_chg':-1.89,'open':1860.00,'high':1870.00,'low':1850.00,'volume':850000},
        'BBAI': {'close':4.02,'pct_chg':-2.90,'open':4.10,'high':4.12,'low':4.01,'volume':3500000},
        'UGRO': {'close':2.82,'pct_chg':-5.69,'open':2.95,'high':2.98,'low':2.80,'volume':4200000},
    }
    demo_breakouts = [
        {'code':'CARR','score':93,'price':69.91,'pct_chg':2.34,'composite':'🟢',
         'ma':'🟢','macd':'🟢','rsi':'🟢','kdj':'🟡','p52':'🟢','rsi_val':59,'ma_pct':1.2},
        {'code':'TER','score':91,'price':403.20,'pct_chg':3.12,'composite':'🟢',
         'ma':'🟢','macd':'🟢','rsi':'🟡','kdj':'🟢','p52':'🟢','rsi_val':57,'ma_pct':0.8},
    ]
    demo_buys = [
        {'code':'CARR','score':93,'price':69.91,'pct':2.34,'composite':'🟢','note':'新突破买入线','source':'SP500','label':'新突破'},
        {'code':'TER','score':91,'price':403.20,'pct':3.12,'composite':'🟢','note':'全绿灯','source':'SP500','label':'全绿灯'},
    ]
    dm = {
        'spx': 5432.15,'spx_chg':0.82,'spx_sig':'🟢',
        'ndx': 19021.80,'ndx_chg':1.12,'ndx_sig':'🟢',
        'vix': 14.20,'vix_sig':'🟢',
    }

    monitors = []
    sc_map = {
        'NVDA': {'score':51,'ma':'🔴','macd':'🔴','rsi':'🔴','kdj':'🔴','p52':'🔴','composite':'🔴',
                 'rsi_val':35,'macd_val':-1.2,'ma_pct':-2.1,'j_val':25,'p52_val':30},
        'ON':   {'score':58,'ma':'🔴','macd':'🔴','rsi':'🟡','kdj':'🔴','p52':'🔴','composite':'🔴',
                 'rsi_val':42,'macd_val':-0.8,'ma_pct':-3.5,'j_val':18,'p52_val':25},
        'TRGP': {'score':68,'ma':'🔴','macd':'🟡','rsi':'🟡','kdj':'🔴','p52':'🟡','composite':'🔴',
                 'rsi_val':45,'macd_val':-0.4,'ma_pct':-1.8,'j_val':35,'p52_val':45},
        'GS':   {'score':77,'ma':'🟢','macd':'🟡','rsi':'🟢','kdj':'🟡','p52':'🟢','composite':'🟡',
                 'rsi_val':58,'macd_val':0.5,'ma_pct':2.1,'j_val':72,'p52_val':68},
        'ASML': {'score':75,'ma':'🟢','macd':'🟡','rsi':'🟡','kdj':'🟢','p52':'🟢','composite':'🟡',
                 'rsi_val':58,'macd_val':0.3,'ma_pct':1.9,'j_val':68,'p52_val':65},
        'BBAI': {},
        'UGRO': {},
    }
    for h in dl:
        code = h['code']
        rt = dp[code]
        sc = sc_map.get(code, {})
        gl = None
        sec = 'blue'
        if code in ('BBAI','UGRO'):
            sec = 'green'
            gl = compute_green(code, None, None)
            if gl is None:
                gl = {'price':rt['close'],'pct':rt['pct_chg'],'vol_ratio':1.2,'p52':45,'gap':0.5,
                      'composite':'🟡','vol_sig':'🟡','gap_sig':'🟢','p52_sig':'🟡'}
        monitors.append({
            'code': code, 'sec': sec, 'rp': rt['close'],
            'pct': rt['pct_chg'], 'hi': h, 'rd': sc, 'gl': gl,
        })

    dashboard = rd(dm, monitors, [h['code'] for h in dl], [], demo_breakouts, demo_buys)
    print(dashboard)
    n = write_signals(monitors, [h['code'] for h in dl], demo_breakouts, demo_buys)
    print('  📬 信号写入: ' + str(n) + ' 条 (demo)')

# ════════════════════════════════════════════
#  主函数
# ════════════════════════════════════════════

def is_td():
    return datetime.datetime.now().weekday() < 5

def main():
    import argparse
    ap = argparse.ArgumentParser(description='小钳盯盘仪 v2')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--interval', type=int, default=600)
    args = ap.parse_args()

    if args.demo:
        run_demo()
        return

    print(c(BOLD + '  📡 小钳盯盘仪 v2 启动中...', CYAN))
    ensure_watchlist()
    if not is_td():
        print(c('  📅 非交易时段 — 显示最近收盘数据', YELL))

    bm, fn, bn = load_baseline()
    if not bm:
        print(c('  ❌ 无基线评分文件', RED))
        sys.exit(1)
    print('  📄 基线: ' + fn)

    all_h, hsrc = get_holdings_data()
    hcodes = [h['code'] for h in all_h]
    print('  📡 持仓(' + hsrc + '): ' + str(len(all_h)) + ' 只')

    wcodes = read_watchlist()
    if wcodes:
        print('  📋 自选: ' + ', '.join(wcodes))

    sp500_list = load_sp500()
    print('  🏛️  SP500: ' + str(len(sp500_list)) + ' 只')

    print('  ⏱️  ' + ('一次' if args.once else ('每' + str(args.interval) + 's轮询')))
    print('  ' + '=' * 50)

    cycle = 0
    while True:
        cycle += 1
        print('\n  🔄 第' + str(cycle) + '轮 — ' + datetime.datetime.now().strftime('%H:%M:%S'))
        try:
            wcodes = read_watchlist()
            mcodes = list(set(hcodes + wcodes))
            market = get_market_data()
            all_hist = load_hist(mcodes)
            scan_codes = [c for c in sp500_list[:100] if c not in mcodes]
            rt_all = get_realtime(mcodes + scan_codes)
            if not rt_all:
                print(c('  ⚠️ 无实时数据 — 使用基线分展示', YELL))
                # 用基线数据兜底
                monitors = []
                for code in mcodes:
                    bs = bm.get(code, {})
                    bsc = bs.get('score', 0)
                    if bsc <= 0:
                        continue
                    hist = all_hist.get(code.upper(), {})
                    bs_price = bs.get('price', 0)
                    if hist and bs_price > 0:
                        sc = compute_rt(code.upper(), _norm_hist(hist), bs_price)
                        if sc is None:
                            sc = {}
                        sc['score'] = bsc  # use baseline score
                    else:
                        sc = {}
                        sc['score'] = bsc
                        sc['composite'] = bs.get('composite', '🟡')
                    sc.setdefault('composite', bs.get('composite', '🟡'))
                    sc.setdefault('ma','🟡'); sc.setdefault('macd','🟡')
                    sc.setdefault('rsi','🟡'); sc.setdefault('kdj','🟡')
                    sc.setdefault('p52','🟡')
                    sc.setdefault('rsi_val',''); sc.setdefault('macd_val','')
                    sc.setdefault('ma_pct',''); sc.setdefault('j_val','')
                    sc.setdefault('p52_val',''); sc.setdefault('vol_ratio','')
                    # composite = 模型评分信号
                    if bsc >= 80:
                        sc['composite'] = '🟢'
                    elif bsc >= 70:
                        sc['composite'] = '🟡'
                    else:
                        sc['composite'] = '🔴'
                    # from baseline rsi/pct52
                    rv = bs.get('rsi')
                    if rv is not None:
                        sc['rsi'] = '🟢' if 50 <= rv <= 70 else ('🔴' if rv < 30 or rv > 80 else '🟡')
                        sc['rsi_val'] = round(rv, 1)
                    pv = bs.get('pct52')
                    if pv is not None:
                        sc['p52'] = '🟢' if pv > 70 else ('🟡' if pv > 40 else '🔴')
                        sc['p52_val'] = round(pv, 1)
                    hi = next((h for h in all_h if h['code'] == code), None)
                    monitors.append({'code':code, 'sec':'blue', 'rp':hi.get('cost',0) if hi else 0,
                                     'pct':0, 'hi':hi, 'rd':sc, 'gl':None})
                breakouts = []
                for code in sp500_list[:100]:
                    if code in mcodes:
                        continue
                    bs2 = bm.get(code, {})
                    bsc2 = bs2.get('score', 0)
                    if bsc2 >= ENTRY_THRESHOLD:
                        breakouts.append({'code': code, 'score': int(bsc2), 'price': bs2.get('price', 0),
                                          'pct_chg': 0, 'composite': bs2.get('composite', '🟡')})
                breakouts.sort(key=lambda x: -x['score'])
                buy_list = []
                for m in monitors:
                    sc3 = m.get('rd', {})
                    if m['code'] not in hcodes and sc3.get('score', 0) >= ENTRY_THRESHOLD:
                        buy_list.append({'code': m['code'], 'score': int(sc3.get('score', 0)),
                                         'price': m.get('rp', 0), 'pct': m.get('pct', 0),
                                         'composite': sc3.get('composite', '🟡'), 'note': '评分达标', 'label': '评分达标'})
                for b in breakouts:
                    if len(buy_list) < 10:
                        buy_list.append({'code': b['code'], 'score': b['score'],
                                         'price': b.get('price', 0), 'pct': 0,
                                         'composite': b.get('composite', '🟡'), 'note': '新突破买入线', 'label': '新突破'})
            else:
                monitors = []
                for code in mcodes:
                    hist = all_hist.get(code.upper(), {})
                    if not hist:
                        continue
                    rt = rt_all.get(code)
                    if not rt:
                        continue
                    sc = compute_rt(code.upper(), _norm_hist(hist), rt.get('close', 0))
                    if sc is None:
                        continue
                    hi = next((h for h in all_h if h['code'] == code), None)
                    gl = None
                    if code.lower() in ('bbai','ugro','hpk','mrdn'):
                        gl = compute_green(code, hist, rt)
                    monitors.append({
                        'code': code, 'sec': 'blue',
                        'rp': rt.get('close'), 'pct': rt.get('pct_chg'),
                        'hi': hi, 'rd': sc, 'gl': gl,
                    })

                breakouts = []
                if mcodes:
                    all_hist = load_hist(mcodes)
                for code in scan_codes:
                    rt = rt_all.get(code)
                    if not rt:
                        continue
                    bs = bm.get(code, {})
                    bsc = bs.get('score', 0)
                    if bsc <= 0:
                        continue
                    est = bsc + rt['pct_chg'] * 2
                    if est >= ENTRY_THRESHOLD and bsc < ENTRY_THRESHOLD - 5:
                        # 只用估算值，不加载全历史（加快速度）
                        breakouts.append({'code': code, 'score': int(est), 'price': rt['close'],
                                          'pct_chg': rt['pct_chg'], 'composite': '🟢'})

                buy_list = []
                for m in monitors:
                    sc = m.get('rd', {})
                    if m['code'] not in hcodes and sc.get('composite') == '🟢' and sc.get('score', 0) >= ENTRY_THRESHOLD:
                        buy_list.append({'code': m['code'], 'score': int(sc['score']),
                                         'price': m.get('rp', 0), 'pct': m.get('pct', 0),
                                         'composite': sc['composite'], 'note': '全绿灯', 'label': '全绿灯'})
                for b in breakouts:
                    if len(buy_list) < 10:
                        buy_list.append({'code': b['code'], 'score': b['score'],
                                         'price': b['price'], 'pct': b['pct_chg'],
                                         'composite': b['composite'], 'note': '新突破买入线', 'label': '新突破'})

            dashboard = rd(market, monitors, hcodes, wcodes, breakouts, buy_list)
            print(dashboard)
            ns = write_signals(monitors, hcodes, breakouts, buy_list)
            if ns > 0:
                print('  📬 新信号 ' + str(ns) + ' 条 → 已通知')
        except KeyboardInterrupt:
            print('\n  🛑 已关闭')
            break
        except Exception as e:
            print(c('  ❌ 异常: ' + str(e), RED))
            traceback.print_exc()
        if args.once:
            break
        for _ in range(args.interval):
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                print('\n  🛑 已关闭')
                return

if __name__ == '__main__':
    main()
