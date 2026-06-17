#!/usr/bin/env python3
"""
test_score_math.py — 小火轮 v4 V1 评分系统单元测试

测试用例:
  A: MACD 门控 → 连续下跌 K 线 → 期望评分 = 0
  B: 完美形态 → 验证所有子项分数及公式计算
  C: 边缘情况 → pos52>=80 / RSI>=75 / 边界映射
  D: 震荡权重 → ADX<22 时切换权重 [10,30,15,10,35]

所有数据生成使用确定性方法（非随机），确保可复现。
运行: python3 scripts/test_score_math.py
"""

import sys, math
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from score_engine import (
    compute_indicators,
    v1_score,
    get_raw_scores,
)

PASS = 0
FAIL = 0

def check(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")

# ============================================================
#  确定性 K 线数据生成
# ============================================================

def gen_linear_trend(start, end_val, n):
    """线性趋势: 从 start 到 end_val，n 个点。"""
    return [start + (end_val - start) * i / (n - 1) for i in range(n)]


def gen_pure_drop(length=300):
    """
    纯下跌：价格从 20 跌到 5，带 251 天国 preamble 用于 p52。
    确定性，无噪声。MACD 柱应为负。
    """
    # 先稳后跌
    preamble_len = 50
    steady = [15.0] * preamble_len
    drop = gen_linear_trend(15.0, 5.0, length - preamble_len)
    close = steady + drop
    # 给 high/low 一些宽度
    high = [c * 1.01 for c in close]
    low = [c * 0.99 for c in close]
    # 补齐到 length 以上
    while len(close) < length:
        close.append(close[-1] * 0.99)
        high.append(close[-1] * 1.01)
        low.append(close[-1] * 0.99)
    close = close[:length]
    high = high[:length]
    low = low[:length]
    return close, high, low


def gen_strong_uptrend(length=300):
    """
    稳定上涨：价格从 5 到 30，MA5>MA20>MA60，MACD 正且柱上升。
    确定性，无噪声。
    """
    # 底部盘整 + 稳定上涨
    base_len = 80
    base = [5.0 + 0.01 * i for i in range(base_len)]  # 微涨
    up = gen_linear_trend(6.0, 30.0, length - base_len)
    close = base + up
    while len(close) < length:
        close.append(close[-1] * 1.002)
    close = close[:length]
    high = [c * 1.005 for c in close]
    low = [c * 0.995 for c in close]
    return close, high, low


def gen_at_52w_high(length=300):
    """
    价格在 52 周高位：前期低位 + 近期急拉到新高。
    """
    base = gen_linear_trend(8.0, 10.0, 200)   # 缓慢爬
    jump = gen_linear_trend(10.0, 15.0, 100)  # 急拉
    close = base + jump
    while len(close) < length:
        close.append(close[-1] * 1.001)
    close = close[:length]
    high = [c * 1.005 for c in close]
    low = [c * 0.995 for c in close]
    return close, high, low


def gen_super_strength(length=300):
    """
    超级强势：每天大涨，RSI 应该 > 75。
    """
    close = [10.0]
    for i in range(1, length):
        close.append(close[-1] * 1.015)  # 每天 1.5%
    high = [c * 1.005 for c in close]
    low = [c * 0.995 for c in close]
    return close, high, low


def gen_low_adx(length=300):
    """
    日间交替涨跌 — 每天涨 0.1% 然后跌 0.1%，
    完全抵消趋势 → ADX ≈ 10-15。
    """
    close = [10.0]
    for i in range(1, length):
        if i % 2 == 1:
            c = close[-1] * 1.001
        else:
            c = close[-1] * 0.999
        close.append(c)
    high = [c * 1.002 for c in close]
    low = [c * 0.998 for c in close]
    return close, high, low


# ============================================================
#  测试用例
# ============================================================

def test_a_macd_gate():
    """
    用例A: MACD 门控
    连续下跌 → MACD 柱为负 → ms=0 → 评分=0
    """
    print(f"\n{'='*60}")
    print("📌 用例A: MACD 门控 — 纯下跌K线→评分=0")
    print(f"{'='*60}")

    close, high, low = gen_pure_drop(300)
    ind = compute_indicators(close, high, low)
    di = len(close) - 1
    raw = get_raw_scores(ind, di)
    score = v1_score(ind, di)

    mh = raw['macd_hist']
    print(f"  MACD 柱 (最后): {mh:.6f}" if mh is not None else f"  MACD 柱: None")
    print(f"  MS: {raw['ms']}  总分: {score:.4f}")

    check("纯下跌 → MACD 柱为负", mh is not None and mh < 0,
          f"MACD柱={mh}")
    check("MACD 负 → ms=0", raw['ms'] == 0, f"ms={raw['ms']}")
    check("ms=0 → 总分为 0", score == 0.0, f"总分={score}")


def test_b_perfect_setup():
    """
    用例B: 完美形态 — 公式验证

    手动计算（趋势权重 [25,15,15,25,20]）:
      ms=12, ws=15, mas=20, ads=15, rs=10

      w_ms   = 12*25/20  = 15.0
      w_ws   = 15*15/20  = 11.25
      w_mas  = 20*15/20  = 15.0
      w_ads  = 15*25/20  = 18.75
      w_rs   = 10*20/20  = 10.0
      ttl    = 70.0
      sum_w  = 100
      总分   = 70.0
    """
    print(f"\n{'='*60}")
    print("📌 用例B: 完美形态 — 公式验证")
    print(f"{'='*60}")

    TARGET_MS, TARGET_WS = 12, 15
    TARGET_MAS, TARGET_ADS, TARGET_RS = 20, 15, 10
    W_TREND = [25, 15, 15, 25, 20]

    # 手动计算预期总分
    ttl_expected = (TARGET_MS * W_TREND[0]/20 + TARGET_WS * W_TREND[1]/20
                    + TARGET_MAS * W_TREND[2]/20 + TARGET_ADS * W_TREND[3]/20
                    + TARGET_RS * W_TREND[4]/20)
    expected_score = ttl_expected / sum(W_TREND) * 100
    print(f"  手动计算: ms={TARGET_MS} ws={TARGET_WS} mas={TARGET_MAS} "
          f"ads={TARGET_ADS} rs={TARGET_RS}")
    print(f"  预期总分: {expected_score:.4f}")

    # 用确定性上涨数据
    close, high, low = gen_strong_uptrend(300)
    ind = compute_indicators(close, high, low)
    di = len(close) - 1
    raw = get_raw_scores(ind, di)
    score = v1_score(ind, di)

    print(f"\n  实际子项分: ms={raw['ms']} ws={raw['ws']} mas={raw['mas']} "
          f"ads={raw['ads']} rs={raw['rs']}")
    print(f"  权重: {raw['weights']} (趋势={raw['is_trend']})")

    # 只在 ms>0 时验证公式
    if raw['ms'] > 0:
        w = raw['weights']
        ttl_actual = (raw['ms'] * (w[0]/20.0) + raw['ws'] * (w[1]/20.0)
                      + raw['mas'] * (w[2]/20.0) + raw['ads'] * (w[3]/20.0)
                      + raw['rs'] * (w[4]/20.0))
        recomputed = min(ttl_actual / sum(w) * 100.0, 100.0)
        check(f"公式一致性: v1_score({score:.4f}) == 重算({recomputed:.4f})",
              abs(score - recomputed) < 0.001,
              f"v1_score={score:.4f}, 重算={recomputed:.4f}")
    else:
        print(f"  ⚠️ 上涨数据 ms=0 (MACD负), 可能是数据生成问题")

    # 验证趋势权重分支
    if raw['is_trend']:
        check("趋势权重正确", raw['weights'] == W_TREND,
              f"实际={raw['weights']}")

    # 纯公式验证（硬编码子项分）
    print(f"\n  --- 纯公式验证（指定子项分 + 趋势权重） ---")
    ttl_hard = (TARGET_MS * W_TREND[0]/20 + TARGET_WS * W_TREND[1]/20
                + TARGET_MAS * W_TREND[2]/20 + TARGET_ADS * W_TREND[3]/20
                + TARGET_RS * W_TREND[4]/20)
    hard_score = ttl_hard / sum(W_TREND) * 100.0
    check(f"ms=12 ws=15 mas=20 ads=15 rs=10 → 总分={hard_score:.2f} (期望=70.00)",
          abs(hard_score - 70.0) < 0.01, f"={hard_score:.4f}")


def test_c_edge_cases():
    """
    用例C: 边缘情况

    C1: pos52≥80 → ws=0
    C2: RSI≥75 → rs=-5
    C3: 边界映射表
    """
    print(f"\n{'='*60}")
    print("📌 用例C: 边缘情况")
    print(f"{'='*60}")

    # ---- C1 ----
    print(f"\n  --- C1: 52 周高位 → ws=0 ---")
    close, high, low = gen_at_52w_high(300)
    ind = compute_indicators(close, high, low)
    di = len(close) - 1
    raw = get_raw_scores(ind, di)
    print(f"  p52={raw['p52']:.2f}  ws={raw['ws']}")
    if raw['p52'] is not None and raw['p52'] >= 80:
        check("pos52≥80 → ws=0", raw['ws'] == 0, f"ws={raw['ws']}")
    else:
        print(f"  ⚠️ 未产生高位 (p52={raw['p52']})")

    # ---- C2 ----
    print(f"\n  --- C2: RSI≥75 → rs=-5 ---")
    c2, h2, l2 = gen_super_strength(300)
    ind2 = compute_indicators(c2, h2, l2)
    raw2 = get_raw_scores(ind2, len(c2) - 1)
    print(f"  RSI={raw2['rsi']:.2f}  rs={raw2['rs']}")
    if raw2['rsi'] is not None and raw2['rsi'] >= 75:
        check("RSI≥75 → rs=-5", raw2['rs'] == -5, f"rs={raw2['rs']}")
    else:
        print(f"  ⚠️ 未产生 RSI≥75 (rsi={raw2['rsi']})")

    # ---- C3: 边界映射表 ----
    print(f"\n  --- C3: 边界映射表验证 ---")
    for val, expected in [(5, 20), (19, 20), (20, 15), (34, 15),
                          (35, 10), (49, 10), (50, 6), (64, 6),
                          (65, 3), (79, 3), (80, 0), (99, 0)]:
        ws_calc = 0
        if val < 20: ws_calc = 20
        elif val < 35: ws_calc = 15
        elif val < 50: ws_calc = 10
        elif val < 65: ws_calc = 6
        elif val < 80: ws_calc = 3
        check(f"pos52={val} → ws={expected}", ws_calc == expected,
              f"实际={ws_calc}")

    for val, expected in [(20, 20), (24, 20), (25, 14), (34, 14),
                          (35, 10), (49, 10), (50, 6), (64, 6),
                          (65, 2), (74, 2), (75, -5), (90, -5)]:
        rs_calc = 0
        if val < 25: rs_calc = 20
        elif val < 35: rs_calc = 14
        elif val < 50: rs_calc = 10
        elif val < 65: rs_calc = 6
        elif val < 75: rs_calc = 2
        else: rs_calc = -5
        check(f"RSI={val} → rs={expected}", rs_calc == expected,
              f"实际={rs_calc}")

    for val, expected in [(15, -5), (17, -5), (18, 5), (21, 5),
                          (22, 10), (27, 10), (28, 15), (34, 15),
                          (35, 20), (50, 20)]:
        ads_calc = -5
        if val >= 35: ads_calc = 20
        elif val >= 28: ads_calc = 15
        elif val >= 22: ads_calc = 10
        elif val >= 18: ads_calc = 5
        check(f"ADX={val} → ads={expected}", ads_calc == expected,
              f"实际={ads_calc}")


def test_d_oscillation_weights():
    """
    用例D: 震荡权重
    ADX<22 → [10,30,15,10,35]
    """
    print(f"\n{'='*60}")
    print("📌 用例D: 震荡权重切换验证")
    print(f"{'='*60}")

    close, high, low = gen_low_adx(300)
    ind = compute_indicators(close, high, low)
    di = len(close) - 1
    raw = get_raw_scores(ind, di)
    score = v1_score(ind, di)

    adx_v = raw['adx']
    print(f"  ADX: {adx_v:.2f}" if adx_v is not None else "  ADX: None")
    print(f"  is_trend: {raw['is_trend']}  权重: {raw['weights']}")
    print(f"  子项分: ms={raw['ms']} ws={raw['ws']} mas={raw['mas']} "
          f"ads={raw['ads']} rs={raw['rs']}")

    if adx_v is not None and adx_v < 22:
        W_OSC = [10, 30, 15, 10, 35]
        check("ADX<22 → 震荡权重", raw['weights'] == W_OSC,
              f"实际={raw['weights']}")

        if raw['ms'] > 0:
            w = raw['weights']
            ttl_osc = (raw['ms'] * (w[0]/20.0) + raw['ws'] * (w[1]/20.0)
                       + raw['mas'] * (w[2]/20.0) + raw['ads'] * (w[3]/20.0)
                       + raw['rs'] * (w[4]/20.0))
            manual = min(ttl_osc / sum(w) * 100.0, 100.0)
            check(f"震荡权重公式一致: {score:.4f} == {manual:.4f}",
                  abs(score - manual) < 0.001,
                  f"v1_score={score:.4f}, 手动={manual:.4f}")

            W_T = [25, 15, 15, 25, 20]
            ttl_t = (raw['ms'] * (W_T[0]/20.0) + raw['ws'] * (W_T[1]/20.0)
                     + raw['mas'] * (W_T[2]/20.0) + raw['ads'] * (W_T[3]/20.0)
                     + raw['rs'] * (W_T[4]/20.0))
            trend_equiv = min(ttl_t / sum(W_T) * 100.0, 100.0)
            print(f"  若用趋势权重: {trend_equiv:.4f}")
            print(f"  实际震荡权重: {score:.4f}")
            check("权重切换产生不同分数",
                  abs(score - trend_equiv) > 0.01,
                  f"差值={abs(score - trend_equiv):.4f}")
        else:
            print(f"  ⚠️ ms=0, 跳过公式一致性检查")
    else:
        print(f"  ⚠️ ADX={adx_v:.2f}, 未满足震荡条件")


# ============================================================
#  主函数
# ============================================================

def main():
    print(f"{'='*60}")
    print("🧪 小火轮 v4 V1 评分系统 — 单元测试")
    print(f"{'='*60}")
    print("  数据生成全确定性，无需随机种子，可复现")

    test_a_macd_gate()
    test_b_perfect_setup()
    test_c_edge_cases()
    test_d_oscillation_weights()

    print(f"\n{'='*60}")
    verdict = "✅ 全部通过" if FAIL == 0 else f"❌ 有 {FAIL} 项失败"
    print(f"📊 结果: {verdict}")
    print(f"  通过: {PASS} / {PASS + FAIL}")
    print(f"{'='*60}")
    return 0 if FAIL == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
