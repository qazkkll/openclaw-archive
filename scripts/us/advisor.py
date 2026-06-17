#!/usr/bin/env python3
"""
🦐 小钳投顾 — 统一股票推荐引擎
A股 + 美股 共用评分框架，市场差异化阈值

用法:
  python3 scripts/advisor.py 600081       # A股，自动识别
  python3 scripts/advisor.py NVDA         # 美股，自动识别
  python3 scripts/advisor.py 000997.SZ    # 指定后缀
  python3 scripts/advisor.py --json NVDA  # JSON输出

输出格式:
  [🟢买入/🔴卖出/🟡持有/⏳观望] 评分+理由
"""
import sys, json, math, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================
# 1. 市场配置
# ============================================================
CONFIG = {
    'A': {
        'name': 'A股',
        'score_buy': 62,
        'score_watch': 57,
        'score_hold': 51,
        'score_caution': 47,
        'score_sell': 0,
        'signal_labels': { 62: '🟢 买入', 57: '🔵 关注', 51: '🟡 持有', 47: '🟠 警惕', 0: '🔴 卖出' },
        'max_positions': 8,
        'rebalance_days': 7,
    },
    'US': {
        'name': '美股',
        'score_buy': 60,
        'score_watch': 50,
        'score_hold': 35,
        'score_caution': 25,
        'score_sell': 0,
        'signal_labels': { 60: '🟢 买入', 50: '🔵 关注', 35: '🟡 持有', 25: '🟠 警惕', 0: '🔴 卖出' },
        'max_positions': 5,
        'rebalance_days': 20,
    }
}

# ============================================================
# 2. 数据获取
# ============================================================
def detect_market(ticker):
    """自动判断市场"""
    t = ticker.upper()
    if t.endswith('.SS') or t.endswith('.SH'):
        return 'A'
    if t.endswith('.SZ'):
        return 'A'
    if t in ('SPY', 'QQQ', 'DIA', 'IWM'):
        return 'US'
    if t.endswith('.TO'):
        return 'US'  # 加拿大
    # 数字开头通常是A股代码
    if t[0].isdigit():
        return 'A'
    # 字母开头 + 不带后缀 = 美股
    return 'US'

def to_yahoo_ticker(ticker, market):
    """转换为Yahoo Finance ticker"""
    t = ticker.upper().strip()
    if market == 'A':
        if '.' in t:
            return t
        if t.startswith('6') or t.startswith('5'):
            return t + '.SS'
        return t + '.SZ'
    return t

def fetch_data(ticker):
    """获取股票历史数据"""
    import yfinance as yf
    h = yf.Ticker(ticker).history(period='1y')
    if h.empty:
        return None
    return {
        'close': [float(x) for x in h['Close'].tolist()],
        'high': [float(x) for x in h['High'].tolist()],
        'low': [float(x) for x in h['Low'].tolist()],
        'open': [float(x) for x in h['Open'].tolist()],
        'volume': [int(x) for x in h['Volume'].tolist()],
        'dates': [d.strftime('%Y-%m-%d') for d in h.index],
    }

# ============================================================
# 3. 技术指标计算（共用）
# ============================================================
def ema(arr, period):
    k = 2 / (period + 1)
    r = [arr[0]]
    for v in arr[1:]:
        r.append(v * k + r[-1] * (1 - k))
    return r

def sma(arr, period):
    if len(arr) < period:
        return []
    return [sum(arr[i-period+1:i+1])/period for i in range(period-1, len(arr))]

def calc_macd(closes):
    e12 = ema(closes, 12)
    e26 = ema(closes, 26)
    n = min(len(e12), len(e26))
    ml = [e12[i] - e26[i] for i in range(n)]
    sl = ema(ml, 9)
    hist = [ml[i] - sl[i] for i in range(min(len(ml), len(sl)))]
    return {'macd': ml[-1], 'signal': sl[-1], 'histogram': hist[-1], 'hist_prev': hist[-2] if len(hist)>=2 else hist[-1]}

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    deltas = [closes[i]-closes[i-1] for i in range(-period, 0)]
    gains = sum(x for x in deltas if x > 0) / period
    losses = sum(abs(x) for x in deltas if x < 0) / period
    return 100 - 100 / (1 + gains/losses) if losses > 0 else 100

def calc_adx(closes, highs, lows):
    """精确匹配 quick_scan.js calcADX"""
    n = len(closes)
    if n < 28:
        return 20
    period = 14
    start = n - period
    tr_sum = dm_plus_sum = dm_minus_sum = 0
    for i in range(start, n):
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        dm_plus = max(0, highs[i] - highs[i-1])
        dm_minus = max(0, lows[i-1] - lows[i])
        tr_sum += tr
        dm_plus_sum += dm_plus
        dm_minus_sum += dm_minus
    atr = tr_sum / period
    if atr == 0:
        return 20
    di_plus = 100 * (dm_plus_sum / period) / atr
    di_minus = 100 * (dm_minus_sum / period) / atr
    if di_plus + di_minus == 0:
        return 20
    if di_plus != di_plus or di_minus != di_minus:
        return 20
    return round(100 * abs(di_plus - di_minus) / (di_plus + di_minus))

# ============================================================
# 4. V1评分（A股评分引擎，也用于美股评分模式）
# ============================================================
def v1_score(closes, highs, lows):
    """V1评分系统 — 精确匹配 quick_scan.js 的 scoreV1 逻辑"""
    c = closes
    n = len(c)
    if n < 60:
        return {'total': 0, 'factors': {}, 'macd_pass': False, 'error': '数据不足60天'}

    cp = c[-1]

    # 均线
    _ma5 = sma(c, 5)
    _ma20 = sma(c, 20)
    _ma60 = sma(c, 60)
    ma5 = _ma5[-1] if len(_ma5) > 0 else cp
    ma20 = _ma20[-1] if len(_ma20) > 0 else cp
    ma60 = _ma60[-1] if len(_ma60) > 0 else cp

    h52 = max(c[-252:]) if len(c) >= 252 else max(c)
    l52 = min(c[-252:]) if len(c) >= 252 else min(c)
    pos52 = (cp - l52) / (h52 - l52) * 100 if h52 > l52 else 50

    macd = calc_macd(c)
    rsi = calc_rsi(c, 14)
    adx = calc_adx(c, highs, lows)

    # MACD门: 柱必须为正否则0分
    if macd['histogram'] <= 0:
        return {
            'total': 0, 'macd_pass': False,
            'rsi': round(rsi), 'adx': round(adx), 'pos52': round(pos52, 1),
            'macd_hist': round(macd['histogram'], 4),
            'current_price': round(cp, 2), 'ma20': round(ma20, 2)
        }

    # 各因子评分 (满分20)
    # MACD状态
    prev_macd = {'histogram': 0}  # 简化
    if len(c) >= 28:
        prev_c = c[:-1]
        pe12 = ema(prev_c, 12)
        pe26 = ema(prev_c, 26)
        pml = [pe12[i]-pe26[i] for i in range(min(len(pe12), len(pe26)))]
        psl = ema(pml, 9)
        prev_macd['histogram'] = pml[-1] - psl[-1] if len(pml) == len(psl) else 0

    if macd['histogram'] > 0 and prev_macd['histogram'] <= 0:
        ms = 20
    elif macd['histogram'] > 0 and macd['macd'] > macd['signal']:
        ms = 12
    elif macd['histogram'] > 0:
        ms = 6
    else:
        ms = 0

    # 52周位置
    if pos52 < 20: ws = 20
    elif pos52 < 35: ws = 15
    elif pos52 < 50: ws = 10
    elif pos52 < 65: ws = 6
    elif pos52 < 80: ws = 3
    else: ws = 0

    # 均线系统
    mas = 0
    if cp > ma20: mas += 7
    if ma5 > ma20: mas += 7
    if ma20 > ma60: mas += 6

    # ADX
    if adx >= 35: ads = 20
    elif adx >= 28: ads = 15
    elif adx >= 22: ads = 10
    elif adx >= 18: ads = 5
    else: ads = -5

    # RSI
    if rsi < 25: rs = 20
    elif rsi < 35: rs = 14
    elif rsi < 50: rs = 10
    elif rsi < 65: rs = 6
    elif rsi < 75: rs = 2
    else: rs = -5

    # 动态权重: 趋势/震荡
    is_trending = adx >= 22
    wl = [25, 15, 15, 25, 20] if is_trending else [10, 30, 15, 10, 35]
    sw = sum(wl)

    total = (ms * wl[0] / 20) + (ws * wl[1] / 20) + (mas * wl[2] / 20) + (ads * wl[3] / 20) + (rs * wl[4] / 20)
    normalized = min(total / sw * 100, 100)

    return {
        'total': round(normalized),
        'macd_pass': True,
        'macd_hist': round(macd['histogram'], 4),
        'macd_trend': '扩张' if macd['histogram'] > macd['hist_prev'] else '萎缩',
        'rsi': round(rsi), 'adx': round(adx),
        'pos52': round(pos52, 1),
        'ma5': round(ma5, 2), 'ma20': round(ma20, 2), 'ma60': round(ma60, 2),
        'weight_mode': '趋势' if is_trending else '震荡',
        'factors': {'macd': ms, 'pos52': ws, 'ma': mas, 'adx': ads, 'rsi': rs},
    }

# ============================================================
# 5. 美股市场状态检测
# ============================================================
def check_us_market_mode():
    """检测美股市场模式: 牛市/熊市"""
    import yfinance as yf
    try:
        spy = yf.Ticker('SPY').history(period='1y')
        c = spy['Close'].tolist()
        cp = c[-1]
        ma200 = sum(c[-200:]) / 200 if len(c) >= 200 else cp
        bull = cp > ma200
        return {
            'mode': 'BULL' if bull else 'BEAR',
            'spy': round(cp, 2),
            'spy_ma200': round(ma200, 2),
            'dist': round((cp/ma200 - 1)*100, 2) if ma200 else 0,
            'label': '🟢 牛市模式（动量追涨）' if bull else '🔴 熊市模式（逆向防守）'
        }
    except:
        return {'mode': 'UNKNOWN', 'label': '⚠️ 无法检测'}

# ============================================================
# 6. 统一推荐引擎
# ============================================================
def get_signal_label(score, market):
    cfg = CONFIG[market]
    for threshold in [cfg['score_buy'], cfg['score_watch'], cfg['score_hold'], cfg['score_caution']]:
        if score >= threshold:
            return cfg['signal_labels'][threshold]
    return cfg['signal_labels'][0]

def recommend(ticker, output_json=False):
    """统一推荐入口"""
    market = detect_market(ticker)
    yahoo_ticker = to_yahoo_ticker(ticker, market)
    cfg = CONFIG[market]

    # 获取数据
    data = fetch_data(yahoo_ticker)
    if not data:
        result = {'ticker': ticker, 'error': f'无法获取 {yahoo_ticker} 数据'}
        return json.dumps(result) if output_json else f"❌ 无法获取 {ticker} 数据"

    c = data['close']
    h = data['high']
    l = data['low']
    v = data['volume']
    cp = c[-1]
    dates = data['dates']

    # 基础指标（共用）
    rsi14 = calc_rsi(c, 14)
    macd = calc_macd(c)
    ma20 = sma(c, 20)[-1] if len(sma(c, 20)) > 0 else cp
    ma50 = sma(c, 50)[-1] if len(sma(c, 50)) > 0 else cp
    ma200 = sma(c, 200)[-1] if len(sma(c, 200)) > 0 else None
    r20 = (cp/c[-21]-1)*100 if len(c) >= 21 else 0
    r10 = (cp/c[-11]-1)*100 if len(c) >= 11 else 0
    r5 = (cp/c[-6]-1)*100 if len(c) >= 6 else 0
    h52 = max(c); l52 = min(c)
    pos52 = (cp-l52)/(h52-l52)*100 if h52!=l52 else 50
    avg_vol = sum(v[-20:])/20 if len(v)>=20 else 1
    vol_ratio = v[-1]/avg_vol if avg_vol>0 else 1

    # 评分
    if market == 'A':
        # A股: V1评分（MACD门控）
        score_result = v1_score(c, h, l)
        score = score_result['total']
        signal = get_signal_label(score, 'A')
        factors = score_result.get('factors', {})
        macd_pass = score_result.get('macd_pass', False)
        macd_detail = f"MACD柱{score_result.get('macd_hist', 0):.4f} ({'通过' if macd_pass else '未通过'})"

    else:
        # 美股: 检测市场模式
        market_mode = check_us_market_mode()
        if market_mode['mode'] == 'BULL':
            # 牛市模式: 不评分，用动量
            score = 0  # 动量排名，非评分
            macd_detail = "牛市模式·动量排序"
            signal = '🔵 动量排序' if r20 > 0 else '🟡 动量偏弱'
            factors = {'20日动量': round(r20, 1)}
        else:
            # 熊市模式: V2逆向评分
            score_result = v1_score(c, h, l)
            score = score_result['total']
            signal = get_signal_label(score, 'US')
            factors = score_result.get('factors', {})
            macd_detail = f"MACD柱{score_result.get('macd_hist', 0):.4f}"

    # 构建标准化输出
    # 买卖建议
    if market == 'A':
        if score >= cfg['score_buy']:
            action = '🟢 买入'
            reason = f"V1评分{score}分，达标买入线{cfg['score_buy']}分"
        elif score < cfg['score_caution']:
            action = '🔴 卖出'
            reason = f"评分{score}分，低于警惕线{cfg['score_caution']}分，建议离场"
        elif score < cfg['score_hold']:
            action = '🟠 减仓'
            reason = f"评分{score}分，低于持有线{cfg['score_hold']}分，建议减仓"
        else:
            action = '🟡 持有'
            reason = f"评分{score}分，在持有区间({cfg['score_hold']}-{cfg['score_buy']-1})"
    else:
        if market_mode['mode'] == 'BULL':
            if r20 > 10:
                action = '🟢 关注'
                reason = f"20日动量+{r20:.1f}%，强势标的"
            elif r20 > 0:
                action = '🟡 持有'
                reason = f"20日动量+{r20:.1f}%，趋势向上"
            else:
                action = '⏳ 观望'
                reason = f"20日动量{r20:+.1f}%，偏弱"
        else:
            if score >= cfg['score_buy']:
                action = '🟢 买入'
                reason = f"熊市评分{score}分，达标买入线{cfg['score_buy']}分"
            elif score < cfg['score_caution']:
                action = '🔴 卖出'
                reason = f"熊市评分{score}分，低于卖出线{cfg['score_caution']}分"
            else:
                action = '🟡 持有'
                reason = f"熊市评分{score}分，持有区间"

    # 风险等级
    if rsi14 > 75:
        risk = '🔴 偏高（超买）'
    elif rsi14 > 65:
        risk = '🟡 中等偏高'
    elif rsi14 < 30:
        risk = '🟢 偏低（超卖机会）'
    elif rsi14 < 40:
        risk = '🟢 偏低'
    else:
        risk = '🟢 正常'

    # ===== 主观评估 (🧠 我说) =====
    subjective_green = 0
    subjective_yellow = 0
    subjective_red = 0
    sub_factors = []

    # ============= 红灯信号 (硬伤, 直接🐂) =============
    # MACD门未过 = 最大红灯
    if (market == 'A' and (not score_result.get('macd_pass', True))) or (market == 'A' and score == 0):
        subjective_red += 3
        sub_factors.append('MACD未过')
    # 跌破MA20超过3%
    ma20_dist = (cp/ma20 - 1)*100 if ma20 else 0
    if ma20_dist < -3:
        subjective_red += 1
        sub_factors.append('跌破MA20')

    # ============= 黄灯信号 (顾虑, 轻仓) =============
    # 缩量
    if vol_ratio < 0.6:
        subjective_yellow += 1
        sub_factors.append('缩量')
    # RSI超买
    if rsi14 > 75:
        subjective_yellow += 1
        sub_factors.append('RSI超买')
    # MACD柱萎缩(正值但缩小)
    if macd['histogram'] > 0 and macd['histogram'] <= macd['hist_prev']:
        subjective_yellow += 1
        sub_factors.append('MACD萎缩')
    # 高位
    if pos52 > 85:
        subjective_yellow += 1
        sub_factors.append('高位')
    # 远离MA20
    if ma20_dist > 10:
        subjective_yellow += 1
        sub_factors.append('远离均线')
    # 美股熊市模式
    if market == 'US' and market_mode['mode'] == 'BEAR':
        subjective_yellow += 1
        sub_factors.append('熊市')

    # ============= 绿灯信号 (加分, 确认) =============
    if vol_ratio > 1.5:
        subjective_green += 1
        sub_factors.append('放量')
    if rsi14 < 35:
        subjective_green += 1
    if pos52 < 30 and (market == 'A' and score > 0):
        subjective_green += 1
        sub_factors.append('低位安全垫')
    if market == 'US' and market_mode['mode'] == 'BULL':
        subjective_green += 1
    if macd['histogram'] > 0 and macd['histogram'] > macd['hist_prev']:
        subjective_green += 1
        sub_factors.append('MACD扩张')

    # ============= 模型灯 =============
    if market == 'A':
        if score >= cfg['score_buy']:
            model_light = '🟢'
        elif score < cfg['score_caution']:
            model_light = '🔴'
        else:
            model_light = '🟡'
    else:  # US
        if market_mode['mode'] == 'BULL':
            if r20 > 10: model_light = '🟢'
            elif r20 > 0: model_light = '🟡'
            else: model_light = '🔴'
        else:
            if score >= cfg['score_buy']: model_light = '🟢'
            elif score < cfg['score_caution']: model_light = '🔴'
            else: model_light = '🟡'

    # ============= 主观灯 =============
    # 有红灯 => 主观🔴; 黄>=绿且黄>0 => 🟡; 否则🟢
    if subjective_red > 0:
        my_light = '🔴'
        my_summary = ' '.join(sub_factors[:3])
    elif (subjective_yellow >= subjective_green and subjective_yellow > 0) or subjective_green == 0:
        my_light = '🟡'
        my_summary = ' '.join(sub_factors[:3]) if sub_factors else '信号一般'
    else:
        my_light = '🟢'
        my_summary = ' ✅' if sub_factors else '✅'

    # ============= 组合灯 + 最终行动 =============
    combined = model_light + my_light
    combo_map = {
        '🟢🟢': ('⚡ 双重推荐 · 重仓',),   # 模型🟢+我🟢
        '🟢🟡': ('💡 模型推荐 · 轻仓',),   # 模型🟢+我🟡
        '🟢🔴': ('⚠️ 模型推荐 · 我反对',), # 模型🟢+我🔴
        '🟡🟢': ('👀 关注 · 等信号',),     # 模型🟡+我🟢
        '🟡🟡': ('⏳ 观望',),             # 模型🟡+我🟡
        '🟡🔴': ('🔴 不推荐',),           # 模型🟡+我🔴
        '🔴🟢': ('⏳ 等MACD转正',),        # 模型🔴+我🟢
        '🔴🟡': ('🔴 不推荐',),           # 模型🔴+我🟡
        '🔴🔴': ('🔴 远离',),             # 模型🔴+我🔴
    }
    combo_action = combo_map.get(combined, ('❓',))[0]

    result = {
        'ticker': ticker,
        'market': market,
        'price': round(cp, 2),
        'date': dates[-1] if dates else '',
        'model_light': model_light,
        'my_light': my_light,
        'combined': combined,
        'combo_action': combo_action,
        'score': score if market == 'A' else (round(r20, 1) if market_mode['mode'] == 'BULL' else score),
        'signal': signal,
        'rsi14': round(rsi14),
        'r20': round(r20, 1),
        'ma20': round(ma20, 2),
        'pos52': round(pos52, 1),
        'vol_ratio': round(vol_ratio, 2),
        'macd_hist': round(macd['histogram'], 4),
        'macd_trend': '扩张' if macd['histogram'] > macd['hist_prev'] else '萎缩',
        'factors': factors,
        'sub_note': ' '.join(sub_factors[:3]) if sub_factors else '—',
        'market_mode': market_mode.get('label', '') if market == 'US' else '',
        'config': {
            'buy_threshold': cfg['score_buy'],
            'sell_threshold': cfg['score_caution'],
            'max_positions': cfg['max_positions'],
            'rebalance_days': cfg['rebalance_days'],
        }
    }

    if output_json:
        return json.dumps(result, ensure_ascii=False, indent=2)

    # 个股深度格式 (参考长电科技模板)
    flag = '🇨🇳' if market == 'A' else '🇺🇸'
    lines = []
    lines.append(f"{flag} **{ticker.upper()}** · ${cp:.2f}")
    lines.append('')
    lines.append(f"{combined} **{combo_action}**")
    lines.append('')
    # 关键指标表格
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|:---|---:|")
    lines.append(f"| 评分 | **{result['score']}**分 |")
    lines.append(f"| RSI14 | **{rsi14:.0f}** {'🔴 超买' if rsi14>70 else '🟢 超卖机会' if rsi14<30 else '中性'} |")
    lines.append(f"| 20日动量 | **{r20:+.1f}%** |")
    lines.append(f"| MA20 | ${ma20:.2f} (距**{(cp/ma20-1)*100:+.1f}%**) |")
    lines.append(f"| 52周位 | **{pos52:.0f}%** |")
    lines.append(f"| 量比 | **{vol_ratio:.2f}x** |")
    lines.append(f"| MACD柱 | **{macd['histogram']:.3f}** ({result['macd_trend']}) |")
    if market == 'US':
        lines.append(f"| 市场模式 | {result.get('market_mode', '?')} |")
    lines.append('')
    # 主观
    lines.append(f"🧠 {my_summary}")
    sub_note = result['sub_note'] if result['sub_note'] != '—' else '无异常'
    lines.append(f"   关键信号: {sub_note}")

    return '\n'.join(lines)

# ============================================================
# 7. 速评（压缩版，适合Telegram）
# ============================================================
def quick_advice(ticker):
    """压缩版推荐（单行结论+关键数字）"""
    market = detect_market(ticker)
    yahoo_ticker = to_yahoo_ticker(ticker, market)
    data = fetch_data(yahoo_ticker)
    if not data:
        return f"❌ {ticker}: 无数据"

    c = data['close']
    cp = c[-1]
    rsi14 = calc_rsi(c, 14)
    r20 = (cp/c[-21]-1)*100 if len(c) >= 21 else 0

    # 用V1评分
    sr = v1_score(c, data['high'], data['low'])
    score = sr['total']
    cfg = CONFIG[market]

    # 信号
    if market == 'A':
        if score >= cfg['score_buy']:
            sig = '🟢 买入'
        elif score < cfg['score_caution']:
            sig = '🔴 卖出'
        elif score < cfg['score_hold']:
            sig = '🟠 减仓'
        else:
            sig = '🟡 持有'
    else:
        mm = check_us_market_mode()
        if mm['mode'] == 'BULL':
            if r20 > 15: sig = '🟢 买入'
            elif r20 > 5: sig = '🟡 持有'
            elif r20 > -5: sig = '⏳ 观望'
            else: sig = '🔴 规避'
        else:
            if score >= cfg['score_buy']: sig = '🟢 买入'
            elif score < cfg['score_caution']: sig = '🔴 卖出'
            else: sig = '🟡 持有'

    if market == 'US' and check_us_market_mode()['mode'] == 'BULL':
        return f"${cp:.2f} | {sig} | 20日{r20:+.1f}% | RSI{rsi14:.0f} | 52周位{sr.get('pos52','?')}%"
    return f"${cp:.2f} | {sig} | V1评分{score} | RSI{rsi14:.0f} | 20日{r20:+.1f}% | 52周位{sr.get('pos52','?')}%"

# ============================================================
# 8. 批量分析（读取 quick_scan_result.json）
# ============================================================
def check_a_market():
    """检测A股大盘状态（独立于quick_scan，避免平安银行bug）"""
    import yfinance as yf
    try:
        sh = yf.Ticker('000001.SS').history(period='6mo')
        c = sh['Close'].tolist()
        cp = c[-1]
        ma20 = sum(c[-20:])/20
        ma50 = sum(c[-50:])/50
        r20 = (cp/c[-21]-1)*100 if len(c)>=21 else 0
        h52 = max(c); l52 = min(c)
        pos52 = (cp-l52)/(h52-l52)*100 if h52!=l52 else 50
        high = sh['High'].tolist(); low = sh['Low'].tolist()
        
        # ADX
        period = 14
        start = len(c) - period
        tr_sum = dm_plus_sum = dm_minus_sum = 0
        for i in range(start, len(c)):
            tr = max(high[i]-low[i], abs(high[i]-c[i-1]), abs(low[i]-c[i-1]))
            dm_plus = max(0, high[i] - high[i-1])
            dm_minus = max(0, low[i-1] - low[i])
            tr_sum += tr; dm_plus_sum += dm_plus; dm_minus_sum += dm_minus
        atr = tr_sum/period if tr_sum else 1
        adx = round(100*abs(dm_plus_sum-dm_minus_sum)/(dm_plus_sum+dm_minus_sum)) if (dm_plus_sum+dm_minus_sum)>0 else 20
        
        # 市场状态
        if cp > ma20*1.02 and adx > 22: state = '🟢 牛市'; entry_extra = 35
        elif cp < ma20*0.98 and adx > 22: state = '🔴 熊市'; entry_extra = 5
        else: state = '🟡 震荡'; entry_extra = 20
        
        entry = entry_extra + min(adx/40*20, 20) + (15 if cp>ma20*1.03 else 5 if cp>ma20 else 0) + (10 if cp>ma20 else 3)
        entry = min(round(entry), 100)
        
        if entry >= 80: entry_label = '🟢 极佳'
        elif entry >= 60: entry_label = '🟢 适合'
        elif entry >= 40: entry_label = '🟡 谨慎'
        elif entry >= 20: entry_label = '🟠 不宜'
        else: entry_label = '🔴 禁止'
        
        return {
            'state': state, 'entry': entry, 'entry_label': entry_label,
            'price': round(cp, 2), 'ma20': round(ma20, 2),
            'ma50': round(ma50, 2), 'adx': adx,
            'pos52': round(pos52, 1), 'r20': round(r20, 1),
        }
    except Exception as e:
        return {'state': '⚠️', 'entry': 0, 'entry_label': '❓', 'error': str(e)}


def batch_scan():
    """A股全扫描 + 双层推荐（重构版）"""
    import os
    ws = '/home/admin/.openclaw/workspace'
    path = os.path.join(ws, 'data', 'quick_scan_result.json')
    if not os.path.exists(path):
        return '❌ 没有扫描结果文件'
    with open(path) as f:
        data = json.load(f)
    
    top8 = data.get('top8', [])
    
    # ===== 大盘全景（独立检测，不走scan数据） =====
    mk = check_a_market()
    
    lines = []
    lines.append(f"📊 A股 · {datetime.now().strftime('%m/%d')} 市场全景")
    lines.append(f"{'─'*30}")
    
    # 第一行: 大盘 + 入场分
    arrow = '⬆' if mk['price'] > mk['ma20'] else '⬇'
    dist20 = (mk['price']/mk['ma20']-1)*100
    lines.append(f"{mk['state']} 上证{mk['price']} MA20{mk['ma20']}({dist20:+.2f}%){arrow} | 入场分{mk['entry']} {mk['entry_label']}")
    lines.append(f"  MA50:{mk['ma50']} ADX:{mk['adx']} 52周位:{mk['pos52']}% 20日:{mk['r20']:+.1f}%")
    
    # 入场建议
    if mk['entry'] < 40:
        lines.append(f"  ⚠️ 入场分偏低，控制仓位，等回调或放量再进")
    elif mk['entry'] >= 60:
        lines.append(f"  ✅ 适合入场，可积极选股")
    lines.append('')
    
    # ===== 个股双层推荐 =====
    results = []
    for s in top8:
        r = json.loads(recommend(s['code'], output_json=True))
        r['name'] = s.get('name', '')
        r['sector'] = s.get('sector', '')
        results.append(r)
    
    priority = {'🟢🟢': 0, '🟢🟡': 1, '🟡🟢': 2, '🟡🟡': 3, '🟢🔴': 4, '🟡🔴': 5, '🔴🟢': 6, '🔴🟡': 7, '🔴🔴': 8}
    results.sort(key=lambda x: priority.get(x['combined'], 99))
    
    lines.append('🏆 双层推荐 · Top8')
    flag = '🇨🇳'
    for r in results:
        name = r.get('name', r['ticker'])
        combo = r['combined']
        action = r['combo_action']
        score = r['score']
        sec = r.get('sector', '')
        line = f"  {flag} {combo} {name}({r['ticker']}) {action}"
        line += f" · 📊{score}分 RSI{r['rsi14']} 20日{r['r20']:+.1f}%"
        if sec and sec != '其他':
            line += f' {sec}'
        if r['sub_note'] != '—':
            line += f' | {r["sub_note"]}'
        lines.append(line)
    
    greens = [r for r in results if r['combined'] == '🟢🟢']
    yellows = [r for r in results if r['combined'] in ('🟢🟡', '🟡🟢')]
    if greens:
        lines.append('')
        lines.append('⚡ 双重推荐解析:')
        for r in greens:
            name = r.get('name', r['ticker'])
            ticker = r['ticker']
            price = r['price']
            score = r['score']
            ma20 = r['ma20']
            note = r['sub_note'] if r['sub_note'] != '—' else '信号配合'
            buy_low = round(ma20 * 0.98, 2)
            buy_high = round(price * 1.01, 2)
            stop = round(ma20 * 0.97, 2)
            lines.append(f"  {flag} {name}({ticker}) · 现¥{price}")
            lines.append(f"    模型评分{score}分+{note} → 🟢🟢 双重确认")
            lines.append(f"    建议: ¥{buy_low}-{buy_high}分批进 | 止损¥{stop} | 目标看前高")
    if yellows:
        lines.append(f"💡 轻仓关注: {'  '.join(r['ticker'] for r in yellows)}")
    lines.append('')
    lines.append('─'*30)
    lines.append('🟢🟢 = 重仓(模型🟢+主观🟢)  🟢🟡 = 轻仓(达标但有顾虑)')
    lines.append('🟡🟢 = 关注(未达标但形态好)  🟡🟡/🔴🔴 = 观望/远离')
    lines.append('📊 模型灯=A股V1评分≥62 / 美股动量前20 | 🧠 主观灯=量价MACD位置综合')
    
    return '\n'.join(lines)

# ============================================================
# 9. 美股批量扫描
# ============================================================
def batch_scan_us():
    """美股质量池批量扫描 + 双层推荐"""
    import os, yfinance as yf
    ws = '/home/admin/.openclaw/workspace'
    path = os.path.join(ws, 'data', 'sp500_universe.json')
    if not os.path.exists(path):
        return '❌ 没有美股候选池文件'
    with open(path) as f:
        data = json.load(f)
    pool = data.get('pool', [])
    if not pool:
        return '❌ 候选池为空'
    
    lines = []
    
    # 市场模式
    mm = check_us_market_mode()
    lines.append(f"📊 美股 · {datetime.now().strftime('%m/%d')} 市场全景")
    lines.append(f"{'─'*30}")
    
    # 大盘状态
    lines.append(f"{mm['label']} | SPY ${mm['spy']} MA200 ${mm['spy_ma200']} (距{mm['dist']:+.2f}%)")
    lines.append(f"  牛市模式 = 动量追涨，熊市模式 = 逆向防守")
    if mm['dist'] < 0:
        lines.append(f"  ⚠️ 大盘低于MA200，熊市模式，注意防守")
    elif mm['dist'] > 10:
        lines.append(f"  ⚠️ 大盘远离MA200+{mm['dist']:.0f}%，注意回调风险")
    else:
        lines.append(f"  ✅ 牛市延续，积极选股")
    lines.append('')
    
    tickers = [s['ticker'] for s in pool]
    lines.append(f"📡 扫描 {len(tickers)} 只质量池...")
    
    scored = []
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        for t in batch:
            try:
                h = yf.Ticker(t).history(period='3mo')
                if h.empty:
                    continue
                c = h['Close'].tolist()
                if len(c) < 30:
                    continue
                cp = c[-1]
                r20 = (cp/c[-21]-1)*100 if len(c)>=21 else 0
                r10 = (cp/c[-11]-1)*100 if len(c)>=11 else 0
                ma20 = sum(c[-20:])/20
                rsi14 = calc_rsi(c, 14)
                h52 = max(c); l52 = min(c)
                pos52 = (cp-l52)/(h52-l52)*100 if h52!=l52 else 50
                
                if mm['mode'] == 'BULL':
                    score_val = round(r20, 1)
                else:
                    # 熊市模式需要完整V1
                    sco = v1_score(c, h['High'].tolist(), h['Low'].tolist())
                    score_val = sco['total']
                
                scored.append({
                    'ticker': t,
                    'score': score_val,
                    'price': round(cp, 2),
                    'r20': round(r20, 1),
                    'r10': round(r10, 1),
                    'ma20': round(ma20, 2),
                    'rsi14': round(rsi14),
                    'pos52': round(pos52, 1),
                })
            except Exception as e:
                pass
        lines.append(f"  📡 {min(i+batch_size, len(tickers))}/{len(tickers)}")
    
    # 排序
    if mm['mode'] == 'BULL':
        scored.sort(key=lambda x: x['r20'], reverse=True)
    else:
        scored.sort(key=lambda x: x['score'], reverse=True)
    
    top = scored[:20]
    
    # 对Top20跑完整分析（含主观评估）
    results = []
    for s in top:
        try:
            r = json.loads(recommend(s['ticker'], output_json=True))
            # 合并行业信息
            pool_info = next((p for p in pool if p['ticker'] == s['ticker']), {})
            r['sector'] = pool_info.get('sector', '')
            r['quality_score'] = pool_info.get('quality_score', '')
            results.append(r)
        except:
            pass
    
    # 按双灯排序
    priority = {'🟢🟢': 0, '🟢🟡': 1, '🟡🟢': 2, '🟡🟡': 3, '🟢🔴': 4, '🟡🔴': 5, '🔴🟢': 6, '🔴🟡': 7, '🔴🔴': 8}
    results.sort(key=lambda x: priority.get(x['combined'], 99))
    
    lines.append('')
    us_flag = '🇺🇸'
    lines.append(f'🏆 双层推荐 · Top10 (质量池{len(pool)}只)')
    for r in results[:10]:
        combo = r['combined']
        action = r['combo_action']
        sec = r.get('sector', '')
        score = r['score']
        line = f"  {us_flag} {combo} {r['ticker']} {action} · 📊{score} RSI{r['rsi14']} 20日{r['r20']:+.1f}%"
        if sec:
            line += f' | {sec}'
        if r['sub_note'] != '—':
            line += f' | {r["sub_note"]}'
        lines.append(line)
    
    # 双重推荐汇总
    greens = [r for r in results if r['combined'] == '🟢🟢']
    yellows = [r for r in results if r['combined'] in ('🟢🟡', '🟡🟢')]
    if greens:
        lines.append('')
        lines.append('⚡ 双重推荐解析:')
        for r in greens:
            ticker = r['ticker']
            price = r['price']
            score = r['score']
            ma20 = r['ma20']
            note = r['sub_note'] if r['sub_note'] != '—' else '动量强势'
            buy_low = round(ma20 * 0.98, 2)
            buy_high = round(price * 1.02, 2)
            stop = round(ma20 * 0.95, 2)
            lines.append(f"  🇺🇸 {ticker} · 现${price}")
            lines.append(f"    20日动量+{score}%+{note} → 🟢🟢 双重确认")
            lines.append(f"    建议: ${buy_low}-{buy_high}区间进 | 止损${stop} | 动量趋势向好")
    if yellows:
        lines.append(f"💡 轻仓关注: {'/'.join(r['ticker'] for r in yellows)}")
    lines.append('')
    lines.append('─'*30)
    lines.append('🟢🟢 = 重仓(动量🟢+主观🟢)  🟢🟡 = 轻仓(动量达标但有高位风险)')
    lines.append('🟡🟢 = 关注(动量一般但形态好)  🟡🟡/🔴🔴 = 观望/远离')
    lines.append('📊 模型灯=20日动量(牛市)或V1评分(熊市) | 🧠 主观灯=量价位置MACD综合')
    
    return '\n'.join(lines)


# ============================================================
# 10. 主入口
# ============================================================
if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print("用法: python3 scripts/advisor.py <ticker> [--json|--quick|--batch]")
        sys.exit(1)

    if args[0] == '--batch':
        print(batch_scan())
        sys.exit(0)
    if args[0] == '--batch-us':
        print(batch_scan_us())
        sys.exit(0)

    ticker = args[0]
    mode = '--quick' if '--quick' in args else ('--json' if '--json' in args else 'normal')

    if mode == '--quick':
        print(quick_advice(ticker))
    else:
        print(recommend(ticker, output_json=(mode == '--json')))

    if mode == 'normal':
        # 如果是A股，检查是否在扫描结果中
        try:
            with open('/home/admin/.openclaw/workspace/data/quick_scan_result.json') as f:
                scan = json.load(f)
            for s in scan.get('top8', []):
                if s['code'] == ticker:
                    print(f"  (扫描排名: Top{next(i+1 for i,t in enumerate(scan['top8']) if t['code']==ticker)})")
                    break
        except: pass
