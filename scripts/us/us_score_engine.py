#!/usr/bin/env python3
"""
共用评分引擎 — 小火轮 v4 V1评分系统

从 bt_v3_round3.py 提取并封装为独立模块，用于验证/回放/交叉测试。
"""
import json

def compute_indicators(close, high, low, volume=None):
    """
    从 OHLC 数据计算全部技术指标。
    close/high/low: list of float，长度一致。
    volume: list of float（可选），用于量能分析。
    返回 dict 或 None（数据不足时）。
    """
    n = len(close)
    if n < 60:
        return None

    def sma(arr, p):
        return [None] * (p - 1) + [sum(arr[i - p + 1:i + 1]) / p for i in range(p - 1, len(arr))]

    def ema(arr, p):
        k = 2 / (p + 1)
        r = [arr[0]]
        for v in arr[1:]:
            r.append(v * k + r[-1] * (1 - k))
        return r

    # --- 均线 ---
    m5 = sma(close, 5)
    m20 = sma(close, 20)
    m60 = sma(close, 60)

    # --- MACD ---
    e12 = ema(close, 12)
    e26 = ema(close, 26)
    macd_line = [e12[i] - e26[i] for i in range(n)]
    signal = ema(macd_line, 9)
    macd_hist = [macd_line[i] - signal[i] for i in range(n)]

    # --- RSI(14) ---
    gains, losses = [], []
    for i in range(1, n):
        diff = close[i] - close[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    rsi = [None] * 14
    avg_gain = sum(gains[:14]) / 14 if len(gains) >= 14 else 0
    avg_loss = sum(losses[:14]) / 14 if len(losses) >= 14 else 0
    for i in range(14, n):
        rsi.append(100 - 100 / (1 + avg_gain / avg_loss) if avg_loss > 0 else 100)
        if i < len(gains):
            avg_gain = (avg_gain * 13 + gains[i]) / 14
            avg_loss = (avg_loss * 13 + losses[i]) / 14

    # --- ADX(14, 周期27起步) ---
    adx = [None] * 27
    tr_list, dp_list, dm_list = [], [], []
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        dp = max(0, high[i] - high[i - 1])
        dm = max(0, low[i - 1] - low[i])
        tr_list.append(tr)
        dp_list.append(dp)
        dm_list.append(dm)
        if i < 14:
            continue
        tr14 = sum(tr_list[-14:])
        dp14 = sum(dp_list[-14:])
        dm14 = sum(dm_list[-14:])
        atr = tr14 / 14
        if atr == 0:
            adx.append(0)
            continue
        dip = dp14 / 14 / atr * 100
        dim = dm14 / 14 / atr * 100
        if dip + dim == 0:
            adx.append(0)
            continue
        dx = abs(dip - dim) / (dip + dim) * 100
        if i < 27:
            adx.append(dx)
            continue
        prev_adx_avg = sum(a for a in adx[-13:] if a is not None) / 13
        adx.append((prev_adx_avg * 13 + dx) / 14)
    while len(adx) < n:
        adx.append(None)
    adx = adx[:n]  # 截断到输入长度

    # --- 52周百分位 ---
    p52 = [None] * 251
    for i in range(251, n):
        lo = min(close[i - 250:i + 1])
        hi = max(close[i - 250:i + 1])
        p52.append((close[i] - lo) / (hi - lo) * 100 if hi > lo else 50)
    while len(p52) < n:
        p52.append(None)

    # --- 动量(20日/60日) ---
    mom20 = [None] * 20
    for i in range(20, n):
        mom20.append((close[i] - close[i-20]) / close[i-20] * 100 if close[i-20] > 0 else 0)
    while len(mom20) < n:
        mom20.append(None)

    mom60 = [None] * 60
    for i in range(60, n):
        mom60.append((close[i] - close[i-60]) / close[i-60] * 100 if close[i-60] > 0 else 0)
    while len(mom60) < n:
        mom60.append(None)

    # 量比（volume / 20日均量）
    vol_ratio = [None] * 20
    if volume and len(volume) >= 20:
        for i in range(20, len(volume)):
            avg_vol = sum(volume[i-20:i]) / 20
            vol_ratio.append(volume[i] / avg_vol if avg_vol > 0 else 1.0)
    else:
        vol_ratio = [None] * len(close)
    
    return {
        'close': close,
        'high': high,
        'low': low,
        'volume': volume,
        'vol_ratio': vol_ratio,
        'm5': m5,
        'm20': m20,
        'm60': m60,
        'macd_hist': macd_hist,
        'macd_line': macd_line,
        'signal': signal,
        'rsi': rsi,
        'adx': adx,
        'p52': p52,
        'mom20': mom20,
        'mom60': mom60,
    }


def safe(arr, i):
    """越界安全取值（支持负索引）。"""
    if not arr:
        return None
    idx = i if i >= 0 else len(arr) + i
    if 0 <= idx < len(arr) and arr[idx] is not None:
        v = arr[idx]
        if isinstance(v, float) and v != v:
            return None
        return v
    return None


def v1_score(ind, di):
    """
    V1 评分算法。
    ind: compute_indicators() 返回的 dict。
    di: 数据索引（当日）。
    返回 float 评分（0~100）。
    """

    # ------ MACD 门控 ------
    mh = safe(ind['macd_hist'], di)
    mhp = safe(ind['macd_hist'], di - 1)
    ms = 0
    if mh is not None and mhp is not None:
        if mh > 0 and mhp <= 0:
            ms = 20           # 金叉/上穿零轴
        elif mh > 0 and mh > mhp:
            ms = 12           # 柱线扩大
        elif mh > 0:
            ms = 6            # 柱线缩小但仍在正区
    if ms <= 0:
        return 0.0

    # ------ 位置分 (52周百分位) ------
    p52 = safe(ind['p52'], di)
    ws = 0
    if p52 is not None:
        if p52 < 20:
            ws = 20
        elif p52 < 35:
            ws = 15
        elif p52 < 50:
            ws = 10
        elif p52 < 65:
            ws = 6
        elif p52 < 80:
            ws = 3
        else:
            ws = 0            # pos52>=80 → 0 而非负数

    # ------ 均线分 ------
    pr = safe(ind['close'], di)
    m5 = safe(ind['m5'], di)
    m20 = safe(ind['m20'], di)
    m60 = safe(ind['m60'], di)
    mas = 0
    if pr is not None and m20 is not None and pr > m20:
        mas += 7
    if m5 is not None and m20 is not None and m5 > m20:
        mas += 7
    if m20 is not None and m60 is not None and m20 > m60:
        mas += 6

    # ------ ADX 分 ------
    av = safe(ind['adx'], di)
    ads = -5
    if av is not None:
        if av >= 35:
            ads = 20
        elif av >= 28:
            ads = 15
        elif av >= 22:
            ads = 10
        elif av >= 18:
            ads = 5

    # ------ RSI 分 ------
    rv = safe(ind['rsi'], di)
    rs = 0
    if rv is not None:
        if rv < 25:
            rs = 20
        elif rv < 35:
            rs = 14
        elif rv < 50:
            rs = 10
        elif rv < 65:
            rs = 6
        elif rv < 75:
            rs = 2
        else:
            rs = -5

    # ------ 动量分(20日+60日, LightGBM确认预测力73%) ------
    m20_v = safe(ind.get('mom20'), di)
    m60_v = safe(ind.get('mom60'), di)
    moms = 0
    if m20_v is not None:
        if m20_v > 15: moms += 12
        elif m20_v > 8: moms += 8
        elif m20_v > 3: moms += 4
        elif m20_v < -10: moms -= 5
    if m60_v is not None:
        if m60_v > 20: moms += 10
        elif m60_v > 10: moms += 6
        elif m60_v > 3: moms += 3
        elif m60_v < -15: moms -= 5

    # ------ 量比分(倍量+8分) ------
    vr = safe(ind.get('vol_ratio'), di)
    vols = 0
    if vr is not None:
        if vr >= 3.0: vols = 12
        elif vr >= 2.0: vols = 8
        elif vr >= 1.5: vols = 4
        elif vr < 0.5: vols = -3

    # ------ 权重选择(原5因子不变) ------
    is_trend = av is not None and av >= 22
    w = [25, 15, 15, 25, 20] if is_trend else [10, 30, 15, 10, 35]

    base = (
        ms * (w[0] / 20.0)
        + ws * (w[1] / 20.0)
        + mas * (w[2] / 20.0)
        + ads * (w[3] / 20.0)
        + rs * (w[4] / 20.0)
    )
    base_score = min(base / sum(w) * 100.0, 100.0)

    # ------ 动量奖金(不稀释原有评分) ------
    bonus = 0.0
    if m20_v is not None and m20_v > 0:
        bonus += min(m20_v * 0.5, 8.0)
    if m60_v is not None and m60_v > 0:
        bonus += min(m60_v * 0.3, 6.0)
    # 量比奖金
    if vr is not None and vr >= 1.5:
        bonus += min((vr - 1.0) * 3, 6.0)
    # 动量惩罚(下跌时扣分)
    if m20_v is not None and m20_v < -10:
        bonus -= 5
    if m60_v is not None and m60_v < -15:
        bonus -= 5

    return min(base_score + bonus, 100.0)


def v1_score_from_data(close, high, low, volume=None, idx=-1):
    """从原始 K 线数据直接评分（便捷入口）"""
    ind = compute_indicators(close, high, low, volume)
    if ind is None:
        return 0.0
    di = idx if idx >= 0 else len(close) - 1
    return v1_score(ind, di)


def get_raw_scores(ind, di):
    """返回各子项分数明细，用于调试和验证。"""
    mh = safe(ind['macd_hist'], di)
    mhp = safe(ind['macd_hist'], di - 1)
    ms = 0
    if mh is not None and mhp is not None:
        if mh > 0 and mhp <= 0:
            ms = 20
        elif mh > 0 and mh > mhp:
            ms = 12
        elif mh > 0:
            ms = 6
    p52 = safe(ind['p52'], di)
    ws = 0
    if p52 is not None:
        if p52 < 20:
            ws = 20
        elif p52 < 35:
            ws = 15
        elif p52 < 50:
            ws = 10
        elif p52 < 65:
            ws = 6
        elif p52 < 80:
            ws = 3
    pr = safe(ind['close'], di)
    m5v = safe(ind['m5'], di)
    m20v = safe(ind['m20'], di)
    m60v = safe(ind['m60'], di)
    mas = 0
    if pr and m20v and pr > m20v:
        mas += 7
    if m5v and m20v and m5v > m20v:
        mas += 7
    if m20v and m60v and m20v > m60v:
        mas += 6
    av = safe(ind['adx'], di)
    ads = -5
    if av is not None:
        if av >= 35:
            ads = 20
        elif av >= 28:
            ads = 15
        elif av >= 22:
            ads = 10
        elif av >= 18:
            ads = 5
    rv = safe(ind['rsi'], di)
    rs = 0
    if rv is not None:
        if rv < 25:
            rs = 20
        elif rv < 35:
            rs = 14
        elif rv < 50:
            rs = 10
        elif rv < 65:
            rs = 6
        elif rv < 75:
            rs = 2
        else:
            rs = -5
    is_trend = av is not None and av >= 22
    w = [25, 15, 15, 25, 20] if is_trend else [10, 30, 15, 10, 35]
    return {
        'ms': ms, 'ws': ws, 'mas': mas, 'ads': ads, 'rs': rs,
        'adx': av, 'rsi': rv, 'p52': p52, 'macd_hist': mh,
        'macd_hist_prev': mhp,
        'is_trend': is_trend, 'weights': w,
        'price': pr, 'm5': m5v, 'm20': m20v, 'm60': m60v,
        'total_weight_sum': sum(w),
    }


# ────────────────────────────────────────────
# V5-S 评分（进攻动量）— 2026-06-03 回测定稿
# 最优参数: hd=20 tn=5 ms=60 mh=8 sl=-15
# 来源: us_hist_v5.json 2686只全量回测
# 年化+78.0%, 夏普1.81, 回撤22.8%, 152笔交易
# ────────────────────────────────────────────

def v5s_calc(close, high, low):
    """V5-S 全指标计算（不含回测引擎）"""
    n = len(close)
    if n < 120:
        return None

    def sma(a, p):
        return [None]*(p-1) + [sum(a[i-p+1:i+1])/p for i in range(p-1, len(a))]
    def ema(a, p):
        k = 2/(p+1); r = [a[0]]
        for v in a[1:]: r.append(v*k + r[-1]*(1-k))
        return r

    # 均线
    ma5 = sma(close, 5)
    ma20 = sma(close, 20)
    ma60 = sma(close, 60)
    ma120 = sma(close, 120)

    # MACD
    e12 = ema(close, 12)
    e26 = ema(close, 26)
    macd = [e12[i]-e26[i] for i in range(n)]
    sig = ema(macd, 9)
    macd_hist = [macd[i]-sig[i] for i in range(n)]

    # RSI(14)
    gains, losses = [], []
    for i in range(1, n):
        d = close[i]-close[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    rsi = [None]*14
    if len(gains) >= 14:
        ag = sum(gains[:14])/14
        al = sum(losses[:14])/14
        for i in range(14, n):
            rsi.append(100-100/(1+ag/al) if al > 0 else 100)
            if i < len(gains):
                ag = (ag*13 + gains[i])/14
                al = (al*13 + losses[i])/14

    # P52
    p52 = [None]*252
    for i in range(252, n):
        lo = min(close[i-251:i+1])
        hi = max(close[i-251:i+1])
        p52.append((close[i]-lo)/(hi-lo)*100 if hi > lo else 50)

    return {
        'close': close, 'high': high, 'low': low,
        'ma5': ma5, 'ma20': ma20, 'ma60': ma60, 'ma120': ma120,
        'macd': macd, 'macd_signal': sig, 'macd_hist': macd_hist,
        'rsi': rsi, 'p52': p52
    }


def v5s_score(ind, di):
    """V5-S 统一评分（2026-06-08重构版）
    融合原V5-S/M/L的三维优点构建单模型
    维度: 趋势排列(30) + 动量持续性(25) + MACD能量(25) + 均线偏离(10) + RSI(10) + 52周位置(10)
    总分上限110，和原模型可比
    """
    def sf(a, i):
        if not a: return 0
        idx = i if i >= 0 else len(a) + i
        if 0 <= idx < len(a) and a[idx] is not None:
            v = a[idx]
            if isinstance(v, float) and v != v:
                return 0
            return v
        return 0

    c = ind['close']
    p = sf(c, di)
    if p <= 0:
        return 0

    ma5 = sf(ind['ma5'], di)
    ma20 = sf(ind['ma20'], di)
    ma60 = sf(ind['ma60'], di)
    ma120 = sf(ind['ma120'], di)

    # ─── 1. 趋势排列 (30分) ───
    # 三层排列: MA5>MA20>MA60>MA120 给满
    tr = 0
    if ma5 > ma20: tr += 6
    if ma20 > ma60: tr += 8
    if ma60 > ma120: tr += 10
    if p > ma20: tr += 3
    if p > ma60: tr += 3
    # 多头排列奖励：三级排列全对（5>20 AND 20>60 AND 60>120）
    if ma5 > ma20 and ma20 > ma60 and ma60 > ma120:
        tr = min(tr + 5, 30)
    tr = min(tr, 30)

    # ─── 2. 动量持续性 (25分) ───
    # 原V5-S强项：20/60日价格延续 + 30日动量
    # 加入V5-L的长期趋势保护：不追过于陡峭的高位
    p20 = sf(c, di-20)
    p60 = sf(c, di-60)
    m20 = (p-p20)/p20*100 if p20 > 0 else 0
    m60 = (p-p60)/p60*100 if p60 > 0 else 0
    mo = 10  # 基础动量分
    if m20 > 3: mo += 5
    if m20 > 10: mo += 5
    if m60 > 5: mo += 3
    if m60 > 15: mo += 3
    if m60 > -5 and m60 < 5:
        mo -= 3  # 60日横盘无动量
    # 30日动量衰减（高位回落保护）
    p30 = sf(c, di-30)
    m30 = (p-p30)/p30*100 if p30 > 0 else 0
    if m30 > 40:
        overheat = (m30 - 40) / 5
        mo = max(mo - min(overheat, 10), 0)
    mo = min(mo, 25)

    # ─── 3. MACD能量 (25分) ───
    # V5-S核心，保留金叉+柱线扩大逻辑
    mh = sf(ind['macd_hist'], di)
    mhp = sf(ind['macd_hist'], di-1)
    ms = 0
    macd_line = sf(ind['macd'], di)
    macd_sig = sf(ind['macd_signal'], di)
    if macd_line > macd_sig: ms += 8
    if mh > 0 and mhp <= 0: ms += 10  # 金叉/上穿
    elif mh > 0 and mh > mhp: ms += 7  # 柱线扩大中
    elif mh > 0: ms += 4  # 正区但缩小
    if mh < 0 and mhp > 0: ms -= 5  # 高位回落
    ms = min(ms, 25)

    # ─── 4. 均线偏离度 (10分) ───
    # 价格在MA20上方但不过分遥远
    # 太近=没空间，太远=追高风险
    ma20_dev = (p - ma20) / ma20 * 100 if ma20 > 0 else 0
    ma_dev_s = 0
    if 1 <= ma20_dev <= 8:
        ma_dev_s = 10
    elif -2 <= ma20_dev < 1:
        ma_dev_s = 5
    elif 8 < ma20_dev <= 15:
        ma_dev_s = 6
    elif ma20_dev > 15:
        ma_dev_s = 3
    elif ma20_dev < -5:
        ma_dev_s = 0

    # ─── 5. RSI位置 (10分) ───
    rsi = sf(ind['rsi'], di)
    rs = 5
    if 50 <= rsi <= 65:
        rs = 10  # 主升段最佳区域
    elif 35 <= rsi < 50:
        rs = 7  # 刚脱离超卖
    elif 65 < rsi <= 75:
        rs = 5  # 偏热但可控
    elif rsi > 80:
        rs = 2  # 超买
    elif rsi < 30:
        rs = 4  # 超卖区可能有反弹

    # ─── 6. 52周位置 (10分) ───
    p52 = sf(ind['p52'], di)
    ps = 0
    if 30 <= p52 <= 70:
        ps = 10  # 中位区域，空间最大
    elif 70 < p52 <= 85:
        ps = 6
    elif 85 < p52 <= 100:
        ps = 2  # 高位，风险大
    elif 15 <= p52 < 30:
        ps = 7  # 低位，可能反转
    elif p52 < 15:
        ps = 4  # 极低位

    total = tr + mo + ms + ma_dev_s + rs + ps
    return max(total, 0)


def v5m_calc(close, high, low):

    """V5-M 已废弃（2026-06-08），请使用v5s_calc"""

    return None



def v5m_score(ind, di):

    """V5-M 已废弃（2026-06-08），请使用v5s_score"""

    return 0



def v5l_calc(close, high, low):

    """V5-L 已废弃（2026-06-08），请使用v5s_calc"""

    return None



def v5l_score(ind, di):

    """V5-L 已废弃（2026-06-08），请使用v5s_score"""

    return 0







def _indicators_for_v5(close):
    """共用指标计算（V5三模型）"""
    n = len(close)
    def sma(a, p):
        return [None]*(p-1) + [sum(a[i-p+1:i+1])/p for i in range(p-1, len(a))]
    def ema(a, p):
        k = 2/(p+1); r = [a[0]]
        for v in a[1:]: r.append(v*k + r[-1]*(1-k))
        return r
    ma5 = sma(close, 5); ma20 = sma(close, 20)
    ma60 = sma(close, 60); ma120 = sma(close, 120)
    e12 = ema(close, 12); e26 = ema(close, 26)
    macd = [e12[i]-e26[i] for i in range(n)]
    sig = ema(macd, 9)
    macd_hist = [macd[i]-sig[i] for i in range(n)]
    gains, losses = [], []
    for i in range(1, n):
        d = close[i]-close[i-1]
        gains.append(max(d, 0)); losses.append(max(-d, 0))
    rsi = [None]*14
    if len(gains) >= 14:
        ag = sum(gains[:14])/14; al = sum(losses[:14])/14
        for i in range(14, n):
            rsi.append(100-100/(1+ag/al) if al > 0 else 100)
            if i < len(gains):
                ag = (ag*13 + gains[i])/14; al = (al*13 + losses[i])/14
    p52 = [None]*252
    for i in range(252, n):
        lo = min(close[i-251:i+1]); hi = max(close[i-251:i+1])
        p52.append((close[i]-lo)/(hi-lo)*100 if hi > lo else 50)
    return {'close':close,'ma5':ma5,'ma20':ma20,'ma60':ma60,'ma120':ma120,
            'macd':macd,'macd_signal':sig,'macd_hist':macd_hist,'rsi':rsi,'p52':p52}

def _sf(a, i):
    if not a: return 0
    idx = i if i >= 0 else len(a) + i
    if 0 <= idx < len(a) and a[idx] is not None: return a[idx]
    return 0



