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

    # --- 52周百分位 ---
    p52 = [None] * 251
    for i in range(251, n):
        lo = min(close[i - 250:i + 1])
        hi = max(close[i - 250:i + 1])
        p52.append((close[i] - lo) / (hi - lo) * 100 if hi > lo else 50)
    while len(p52) < n:
        p52.append(None)

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
    }


def safe(arr, i):
    """越界安全取值（支持负索引）。"""
    if not arr:
        return None
    idx = i if i >= 0 else len(arr) + i
    if 0 <= idx < len(arr) and arr[idx] is not None:
        return arr[idx]
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

    # ------ 权重选择 ------
    is_trend = av is not None and av >= 22
    w = [25, 15, 15, 25, 20] if is_trend else [10, 30, 15, 10, 35]

    ttl = (
        ms * (w[0] / 20.0)
        + ws * (w[1] / 20.0)
        + mas * (w[2] / 20.0)
        + ads * (w[3] / 20.0)
        + rs * (w[4] / 20.0)
    )
    return min(ttl / sum(w) * 100.0, 100.0)


def v1_score_from_data(close, high, low, idx=-1):
    """从原始 K 线数据直接评分（便捷入口）"""
    ind = compute_indicators(close, high, low)
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


# bear_score 已废弃（2026-05-25）。
# 结论：V1的MACD门控天然防熊，不需要单独的熊市评分。
# 原代码保留在 git history 中，需要时 git show 即可恢复。
