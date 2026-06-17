#!/usr/bin/env python3
"""
测试 V5三模型 在 SP500 大盘股上的评分表现
用 yfinance 拉取 大盘股 K线，跑 V5评分看分布
"""
import sys, json, yfinance as yf
sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')

ENGINE = r'/home/hermes/.hermes/openclaw-archive\scripts\us_score_engine.py'
with open(ENGINE, encoding='utf-8') as f:
    code = f.read()
engine = {}
exec(code, engine)

v5s_calc  = engine['v5s_calc']   # 全指标计算
v5s_score = engine['v5s_score']  # 统一评分

indicators_v5 = engine['_indicators_for_v5']
_sf = engine['_sf']

# ───────────────────────────────────────
# 复现 V5 三模型的主力评分公式
# ───────────────────────────────────────
def v5_3model_score(ind, di):
    """
    V5三模型（6月2日定稿）的主力评分
    在 _indicators_for_v5 计算的基础上做评分
    源自原始 V5 评分设计：趋势排列+动量持续性+MACD能量+均线偏离+RSI位置+52周位置
    夏普3.87的核心逻辑
    """
    c = ind['close']
    p = _sf(c, di)
    if p <= 0:
        return 0

    ma5  = _sf(ind['ma5'], di)
    ma20 = _sf(ind['ma20'], di)
    ma60 = _sf(ind['ma60'], di)
    ma120 = _sf(ind['ma120'], di)

    # ─── 1. 趋势排列 (30分) ───
    tr = 0
    if ma5 > ma20: tr += 6
    if ma20 > ma60: tr += 8
    if ma60 > ma120: tr += 10
    if p > ma20: tr += 3
    if p > ma60: tr += 3
    if ma5 > ma20 and ma20 > ma60 and ma60 > ma120:
        tr = min(tr + 5, 30)
    tr = min(tr, 30)

    # ─── 2. 动量持续性 (25分) ───
    p20 = _sf(c, di-20)
    p60 = _sf(c, di-60)
    m20 = (p-p20)/p20*100 if p20 > 0 else 0
    m60 = (p-p60)/p60*100 if p60 > 0 else 0
    mo = 10
    if m20 > 3: mo += 5
    if m20 > 10: mo += 5
    if m60 > 5: mo += 3
    if m60 > 15: mo += 3
    if -5 <= m60 <= 5: mo -= 3
    p30 = _sf(c, di-30)
    m30 = (p-p30)/p30*100 if p30 > 0 else 0
    if m30 > 40:
        overheat = (m30 - 40) / 5
        mo = max(mo - min(overheat, 10), 0)
    mo = min(mo, 25)

    # ─── 3. MACD能量 (25分) ───
    mh = _sf(ind['macd_hist'], di)
    mhp = _sf(ind['macd_hist'], di-1)
    macd_line = _sf(ind['macd'], di)
    macd_sig = _sf(ind['macd_signal'], di)
    ms = 0
    if macd_line > macd_sig: ms += 8
    if mh > 0 and mhp <= 0: ms += 10
    elif mh > 0 and mh > mhp: ms += 7
    elif mh > 0: ms += 4
    if mh < 0 and mhp > 0: ms -= 5
    ms = min(ms, 25)

    # ─── 4. 均线偏离度 (10分) ───
    ma20_dev = (p - ma20) / ma20 * 100 if ma20 > 0 else 0
    ma_dev_s = 0
    if 1 <= ma20_dev <= 8: ma_dev_s = 10
    elif -2 <= ma20_dev < 1: ma_dev_s = 5
    elif 8 < ma20_dev <= 15: ma_dev_s = 6
    elif ma20_dev > 15: ma_dev_s = 3
    elif ma20_dev < -5: ma_dev_s = 0

    # ─── 5. RSI位置 (10分) ───
    rsi = _sf(ind['rsi'], di)
    rs = 5
    if 50 <= rsi <= 65: rs = 10
    elif 35 <= rsi < 50: rs = 7
    elif 65 < rsi <= 75: rs = 5
    elif rsi > 80: rs = 2
    elif rsi < 30: rs = 4

    # ─── 6. 52周位置 (10分) ───
    p52 = _sf(ind['p52'], di)
    ps = 0
    if 30 <= p52 <= 70: ps = 10
    elif 70 < p52 <= 85: ps = 6
    elif 85 < p52 <= 100: ps = 2
    elif 15 <= p52 < 30: ps = 7
    elif p52 < 15: ps = 4

    total = tr + mo + ms + ma_dev_s + rs + ps
    return max(total, 0)


# ───────────────────────────────────────
# 测试候选股
# ───────────────────────────────────────
large_caps = {
    'NVDA':  '英伟达',
    'AAPL':  '苹果',
    'MSFT':  '微软',
    'GOOGL': '谷歌',
    'AMZN':  '亚马逊',
    'META':  'Meta',
    'TSLA':  '特斯拉',
    'AVGO':  '博通',
    'JPM':   '摩根大通',
    'V':     'Visa',
    'WMT':   '沃尔玛',
    'PG':    '宝洁',
    'JNJ':   '强生',
    'XOM':   '埃克森美孚',
    'KO':    '可口可乐',
}

# 再加几个中等市值/小盘参考
mids = {
    'SMCI':  '超微电脑',
    'PLTR':  'Palantir',
    'DKNG':  'DraftKings',
}

all_stocks = {**large_caps, **mids}

print(f"{'='*70}")
print(f"V5三模型 大盘股评分测试")
print(f"评分范围: 0-110  (趋势30+动量25+MACD25+偏离10+RSI10+位置10)")
print(f"{'='*70}")
print(f"{'代码':<6} {'名称':<10} {'总分':>5} {'趋':>4} {'动':>4} {'MACD':>5} {'偏离':>4} {'RSI':>4} {'位':>4} {'价格':>8} {'MA20%':>6} {'RSI':>5} {'52w%':>5}")
print(f"{'-'*70}")

results = []
for code, name in all_stocks.items():
    try:
        t = yf.Ticker(code)
        h = t.history(period='2y')
        if len(h) < 252:
            print(f"{code:<6} {'数据不足':<10}  (仅{len(h)}天)")
            continue
        c = h['Close'].tolist()
        hi = h['High'].tolist()
        lo = h['Low'].tolist()
        vo = h['Volume'].tolist()

        ind = indicators_v5(c)
        if not ind:
            print(f"{code:<6} {'指标失败':<10}")
            continue

        # 多天评分，看稳定性
        scores = []
        for offset in range(-1, -21, -1):
            s = v5_3model_score(ind, offset)
            scores.append(s)

        cur_score = scores[0]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        std_dev = (sum((s-avg_score)**2 for s in scores) / len(scores))**0.5

        # 信号质量
        above60 = sum(1 for s in scores if s >= 60)  # V4/V5 买入信号线

        results.append({
            'code': code,
            'name': name,
            'score': cur_score,
            'avg': round(avg_score, 1),
            'max': max_score,
            'min': min_score,
            'std': round(std_dev, 1),
            'above60': above60,
            'type': '大盘' if code in large_caps else '中盘'
        })

        # 本日详情
        p = ind['close'][-1]
        ma20 = ind['ma20'][-1]
        ma20_dev = (p-ma20)/ma20*100 if ma20 else 0
        rsi = ind['rsi'][-1]
        p52 = ind['p52'][-1]

        # 子项详情
        detail = v5_3model_score(ind, -1)

        print(f"{code:<6} {name:<10} {cur_score:>5.0f} {min(tr if 'tr' in dir() else 0, 0):>4}")

        # 第二个 pass 取子项
        c2 = ind['close']; p2 = _sf(c2, -1)
        m5 = _sf(ind['ma5'], -1); m20v = _sf(ind['ma20'], -1); m60v = _sf(ind['ma60'], -1); m120v = _sf(ind['ma120'], -1)
        tr2 = 0
        if m5 > m20v: tr2 += 6
        if m20v > m60v: tr2 += 8
        if m60v > m120v: tr2 += 10
        if p2 > m20v: tr2 += 3
        if p2 > m60v: tr2 += 3
        if m5 > m20v and m20v > m60v and m60v > m120v: tr2 = min(tr2+5, 30)
        tr2 = min(tr2, 30)

        p20 = _sf(c2, -21); p60 = _sf(c2, -61)
        m20v2 = (p2-p20)/p20*100 if p20>0 else 0; m60v2 = (p2-p60)/p60*100 if p60>0 else 0
        mo2 = 10
        if m20v2>3: mo2+=5; 
        if m20v2>10: mo2+=5
        if m60v2>5: mo2+=3
        if m60v2>15: mo2+=3
        if -5<=m60v2<=5: mo2-=3
        p30 = _sf(c2, -31)
        m30v = (p2-p30)/p30*100 if p30>0 else 0
        if m30v>40: mo2 = max(mo2-(m30v-40)/5, 0)
        mo2 = min(mo2, 25)

        mh = _sf(ind['macd_hist'], -1); mhp = _sf(ind['macd_hist'], -2)
        ml = _sf(ind['macd'], -1); msig = _sf(ind['macd_signal'], -1)
        ms2 = 0
        if ml > msig: ms2 += 8
        if mh>0 and mhp<=0: ms2+=10
        elif mh>0 and mh>mhp: ms2+=7
        elif mh>0: ms2+=4
        if mh<0 and mhp>0: ms2-=5
        ms2 = min(ms2, 25)

        md2 = 0
        if 1<=ma20_dev<=8: md2=10
        elif -2<=ma20_dev<1: md2=5
        elif 8<ma20_dev<=15: md2=6
        elif ma20_dev>15: md2=3
        elif ma20_dev<-5: md2=0

        rs2 = 5
        if 50<=rsi<=65: rs2=10
        elif 35<=rsi<50: rs2=7
        elif 65<rsi<=75: rs2=5
        elif rsi>80: rs2=2
        elif rsi<30: rs2=4

        ps2 = 0
        if 30<=p52<=70: ps2=10
        elif 70<p52<=85: ps2=6
        elif 85<p52<=100: ps2=2
        elif 15<=p52<30: ps2=7
        elif p52<15: ps2=4

        total2 = tr2 + mo2 + ms2 + md2 + rs2 + ps2

        print(f"{code:<6} {name:<10} {total2:>5.0f} {tr2:>4} {mo2:>4} {ms2:>5} {md2:>4} {rs2:>4} {ps2:>4} {p:>8.2f} {ma20_dev:>6.1f} {rsi:>5.1f} {p52:>5.0f}")

    except Exception as e:
        print(f"{code:<6} {name:<10} ❌ {str(e)[:40]}")

print(f"{'='*70}")
print()

# 汇总分析
print(f"\n{'='*50}")
print(f"汇总分析")
print(f"{'='*50}")

large_results = [r for r in results if r['type'] == '大盘']
mid_results = [r for r in results if r['type'] == '中盘']

if large_results:
    avg_cur = sum(r['score'] for r in large_results)/len(large_results)
    avg_avg = sum(r['avg'] for r in large_results)/len(large_results)
    avg_above60 = sum(r['above60'] for r in large_results)/len(large_results)*100/20
    print(f"\n📊 大盘股 (15只)")
    print(f"   当前评分均值: {avg_cur:.1f} / 110")
    print(f"   20日均分均值: {avg_avg:.1f}")
    print(f"   评分≥60的交易日占比: {avg_above60:.0f}%")
    above_60 = [r for r in large_results if r['score'] >= 60]
    print(f"   当前≥60分(买入线): {len(above_60)}/{len(large_results)}")
    if above_60:
        for r in above_60:
            print(f"     - {r['code']} ({r['name']}) 评分={r['score']:.0f}")

if mid_results:
    print(f"\n📊 参考中盘 (几只)")
    for r in mid_results:
        print(f"   {r['code']} ({r['name']}) 评分={r['score']:.0f} avg={r['avg']}")

print(f"\n{'='*50}")
print(f"信号质量判断")
print(f"{'='*50}")
if large_results:
    # 评分波动
    avg_std = sum(r['std'] for r in large_results)/len(large_results)
    print(f"大盘股评分20日波动均值: {avg_std:.1f} 分")
    if avg_std < 8:
        print(f"  ✅ 评分稳定度好")
    elif avg_std < 15:
        print(f"  ⚠️ 评分中等波动")
    else:
        print(f"  ❌ 评分波动大，信号噪音高")
    
    # 信号区分度
    max_score = max(r['score'] for r in large_results)
    min_score = min(r['score'] for r in large_results)
    print(f"大盘股当前评分范围: {min_score:.0f}~{max_score:.0f} / 110")
    if max_score - min_score > 40:
        print(f"  ✅ 区分度好（能分出强弱）")
    elif max_score - min_score > 20:
        print(f"  ⚠️ 有一定区分度")
    else:
        print(f"  ❌ 区分度差（都集中在窄区间）")

print(f"\n建议: 评分≥60 = 买入信号, ≥80 = 强势买入, <40 = 考虑卖出")
print(f"V5三模型回测买入门槛: 60分 (hd=10, tn=2, rmax=60)")
