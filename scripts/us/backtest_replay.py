#!/usr/bin/env python3
"""
backtest_replay.py — 回放验证：回测 vs 实时评分一致性

对 5 个指定日期：
  1. 用该日期之前的所有数据计算 V1 评分
  2. 输出 Top-8 评分股票
  3. 运行微型回测到该日期，记录持仓
  4. 对比持仓股是否在 Top 候选列表中
  5. 检查评分排名稳定性

运行: python3 scripts/backtest_replay.py
"""

import sys, json, time
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from score_engine import compute_indicators, v1_score, safe

# ============================================================
#  加载数据
# ============================================================

print("📥 加载 backtest_hist_yahoo.json ...")
t0 = time.time()
with open('data/backtest_hist_yahoo.json') as f:
    hist = json.load(f)
print(f"  共 {len(hist)} 只股票, 耗时 {time.time()-t0:.1f}s")

# 过滤 ETFS
ETFS = {'515000', '512480', '512800', '512880', '512010',
        '515030', '510300', '511010', '518880', '159915'}
codes = [c for c in hist if c not in ETFS and len(hist[c].get('close', [])) > 500]
print(f"  有效候选股: {len(codes)} 只")

# 所有交易日期
all_dates = sorted(set(d for c in codes for d in hist[c].get('dates', [])
                       if '2015-01-01' <= d <= '2026-05-14'))
print(f"  交易日数: {len(all_dates)}")

# 建立 code → date_index 映射
cdates = {}
for c in codes:
    dates = hist[c].get('dates', [])
    cdates[c] = {dt: i for i, dt in enumerate(dates)} if dates else {}

# ============================================================
#  单日评分
# ============================================================

def score_all_at(target_date):
    """
    对 target_date 评分所有股票，返回 [(code, score, detail), ...] 降序。
    使用截止到 target_date（含）的数据。
    """
    results = []
    for code in codes:
        d = hist.get(code)
        if not d:
            continue
        dates = d.get('dates', [])
        cmap = cdates.get(code, {})
        di = cmap.get(target_date)

        # 允许向前取最近交易日
        if di is None:
            for dt in reversed(dates):
                if dt <= target_date and cmap.get(dt) is not None:
                    di = cmap[dt]
                    break
        if di is None or di < 0:
            continue

        close = d['close'][:di + 1]
        high = d['high'][:di + 1]
        low = d['low'][:di + 1]

        if len(close) < 260:
            continue

        ind = compute_indicators(close, high, low)
        if ind is None:
            continue

        sc = v1_score(ind, len(close) - 1)
        if sc > 0:
            # 获取子项明细
            p52 = safe(ind['p52'], -1)
            adx = safe(ind['adx'], -1)
            rsi = safe(ind['rsi'], -1)
            mh = safe(ind['macd_hist'], -1)
            mhp = safe(ind['macd_hist'], -2)
            results.append((code, round(sc, 2), {
                'p52': round(p52, 1) if p52 is not None else None,
                'adx': round(adx, 1) if adx is not None else None,
                'rsi': round(rsi, 1) if rsi is not None else None,
                'macd_h': round(mh, 4) if mh is not None else None,
            }))
    results.sort(key=lambda x: -x[1])
    return results


# ============================================================
#  微型回测：模拟到目标日期
# ============================================================

def mini_backtest_to(target_date, buy_thresh=62, sell_thresh=50,
                     rebal_days=7, max_pos=8, pct=0.125):
    """
    从 2015-01-01 到 target_date 运行简化回测。
    返回持仓字典 {code: {entry_idx: ..., entry_score: ...}}
    以及最后一个调仓日的 Top 候选列表。
    """
    cash = 1_000_000.0
    pos = {}  # code -> {entry_idx, entry_score, shares_value}

    # 准备评分缓存
    ind_cache = {}  # code -> indicators_dict

    def get_ind(code, end_idx):
        key = (code, end_idx)
        if key in ind_cache:
            return ind_cache[key]
        d = hist.get(code)
        if not d:
            ind_cache[key] = None
            return None
        close = d['close'][:end_idx + 1]
        high = d['high'][:end_idx + 1]
        low = d['low'][:end_idx + 1]
        if len(close) < 260:
            ind_cache[key] = None
            return None
        ind = compute_indicators(close, high, low)
        ind_cache[key] = ind
        return ind

    target_idx = all_dates.index(target_date)

    for di in range(260, target_idx + 1):
        dt = all_dates[di]

        # 调仓日判断
        if (di - 260) % rebal_days == 0:
            # 评分所有股票
            scorings = []
            for code in codes:
                ind = get_ind(code, di)
                if ind is None:
                    continue
                sc = v1_score(ind, len(ind['close']) - 1)
                if sc >= buy_thresh:
                    pr = safe(ind['close'], -1)
                    if pr and pr > 0:
                        scorings.append((code, sc, pr))
            scorings.sort(key=lambda x: -x[1])
            top_codes = set(x[0] for x in scorings[:max_pos * 2])

            # 清仓不在 top 的
            for c in list(pos.keys()):
                if c not in top_codes:
                    ind2 = get_ind(c, di)
                    pr = safe(ind2['close'], -1) if ind2 else None
                    if pr and pr > 0:
                        cash += pos[c].get('val', 0) * pr / pos[c].get('entry_pr', 1)
                    del pos[c]

            # 买入 top
            for code, sc, pr in scorings:
                if len(pos) >= max_pos:
                    break
                if code in pos:
                    continue
                inv = min(cash * pct, cash * 0.95)
                if inv < 20000:
                    continue
                pos[code] = {'val': inv, 'entry_pr': pr, 'entry_score': sc}
                cash -= inv

        # 每日卖出检查
        for c in list(pos.keys()):
            ind2 = get_ind(c, di)
            if ind2 is None:
                continue
            sc = v1_score(ind2, len(ind2['close']) - 1)
            if sc < sell_thresh:
                pr = safe(ind2['close'], -1)
                if pr and pr > 0:
                    cash += pos[c]['val'] * pr / pos[c].get('entry_pr', 1)
                del pos[c]

    # 获取最后一个调仓日的 Top-8 候选
    last_rebal_day = max(260 + ((target_idx - 260) // rebal_days) * rebal_days, 260)
    if last_rebal_day <= target_idx:
        scorings = []
        for code in codes:
            ind = get_ind(code, last_rebal_day)
            if ind is None:
                continue
            sc = v1_score(ind, len(ind['close']) - 1)
            if sc >= buy_thresh:
                pr = safe(ind['close'], -1)
                if pr and pr > 0:
                    scorings.append((code, sc, pr))
        scorings.sort(key=lambda x: -x[1])
        last_top = [(c, round(s, 2)) for c, s, _ in scorings[:max_pos]]
    else:
        last_top = []

    return pos, last_top


# ============================================================
#  主验证流程
# ============================================================

TEST_DATES = ['2017-03-01', '2019-05-01', '2021-01-04',
              '2022-06-01', '2024-03-01']

def main():
    ALL_PASS = True

    for target_date in TEST_DATES:
        print(f"\n{'='*70}")
        print(f"📅 回放日期: {target_date}")
        print(f"{'='*70}")

        # --- 第1步：该日期评分 Top-8 ---
        t1 = time.time()
        all_scores = score_all_at(target_date)
        print(f"  ⏱️ 评分耗时: {time.time()-t1:.1f}s")
        print(f"  评分 >0 的股票: {len(all_scores)} 只")

        top8 = all_scores[:8]
        top8_codes = set(c for c, _, _ in top8)

        print(f"\n  📊 V1 评分 Top-8:")
        print(f"  {'排名':>4} {'代码':>10} {'评分':>7} {'P52':>6} {'ADX':>6} {'RSI':>6} {'MACD柱':>10}")
        print(f"  {'─'*4} {'─'*10} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*10}")
        for i, (code, sc, detail) in enumerate(top8, 1):
            d = detail
            print(f"  {i:>4} {code:>10} {sc:>7.2f} "
                  f"{str(d.get('p52', 'N/A')):>6} "
                  f"{str(d.get('adx', 'N/A')):>6} "
                  f"{str(d.get('rsi', 'N/A')):>6} "
                  f"{str(d.get('macd_h', 'N/A')):>10}")

        # --- 第2步：微型回测到该日期 ---
        t2 = time.time()
        pos, last_top = mini_backtest_to(target_date)
        print(f"\n  ⏱️ 回测耗时: {time.time()-t2:.1f}s")

        pos_codes = set(pos.keys())
        print(f"\n  📋 回测持仓 ({len(pos)} 只):")
        for c, p in sorted(pos.items()):
            print(f"    {c:>10} — entry_score={p.get('entry_score', 'N/A')}")

        # --- 第3步: 对比 ---
        print(f"\n  🎯 对比分析:")

        # 3a: 持仓是否在 Top-8 候选
        in_top8 = pos_codes & top8_codes
        not_in_top8 = pos_codes - top8_codes
        print(f"    持仓在 Top-8 中: {len(in_top8)}/{len(pos)} 只")
        for c in sorted(in_top8):
            rank = next(i + 1 for i, (cc, _, _) in enumerate(top8) if cc == c)
            print(f"      ✅ {c:>10} (排名 #{rank})")
        for c in sorted(not_in_top8):
            print(f"      ⚠️  {c:>10} (不在 Top-8, 但可能较早买入且未触发卖出)")

        # 3b: 持仓评分排名（用当前日期评分）
        pos_ranks = []
        for c in pos_codes:
            for i, (cc, sc, _) in enumerate(all_scores):
                if cc == c:
                    pos_ranks.append((c, sc, i + 1))
                    break
            else:
                pos_ranks.append((c, 0, len(all_scores) + 1))

        print(f"\n    持仓评分排名:")
        pos_ranks.sort(key=lambda x: x[1], reverse=True)
        for c, sc, r in pos_ranks:
            print(f"      {c:>10} — 评分={sc:.1f}, 全市场排名=#{r}")

        # 3c: 评分一致性检查
        # 检查：评分靠前的股票是否在持仓中
        top8_in_pos = sum(1 for c, _, _ in top8 if c in pos_codes)
        coverage = top8_in_pos / min(len(pos), 8) * 100 if pos else 0
        print(f"\n    评分一致性: Top-8 中有 {top8_in_pos} 只被持仓 (覆盖度 {coverage:.0f}%)")

        if coverage < 30 and pos:
            print(f"    ❌ 覆盖度低于 30%，可能评分逻辑与回测不一致！")
            ALL_PASS = False
        elif pos:
            print(f"    ✅ 覆盖度可接受")
        else:
            print(f"    ⚠️ 持仓为空（该日期可能是交易日外或无持仓）")

        # 3d: 最后调仓日的 Top 候选对比
        if last_top:
            print(f"\n    最后一个调仓日候选 Top-8:")
            for c, sc in last_top:
                tag = "✅持仓" if c in pos_codes else ""
                print(f"      {c:>10} — 评分={sc:.1f}  {tag}")

    print(f"\n{'='*70}")
    if ALL_PASS:
        print("✅ 所有日期回放验证通过")
    else:
        print("⚠️ 存在差异，见上方详情")
    print(f"{'='*70}")

    # 附加检查：评分排名排序正确性
    print(f"\n{'='*70}")
    print("📌 附加检查: 评分排序正确性")
    print(f"{'='*70}")

    # 随机选一天验证排序
    all_scores = score_all_at('2024-03-01')
    for i in range(len(all_scores) - 1):
        if all_scores[i][1] < all_scores[i + 1][1]:
            print(f"  ❌ 排序错误: 第{i+1}名({all_scores[i][1]}) < 第{i+2}名({all_scores[i+1][1]})")
            ALL_PASS = False
            break
    else:
        print("  ✅ 评分排序正确 (降序)")

    return 0 if ALL_PASS else 1


if __name__ == '__main__':
    sys.exit(main())
