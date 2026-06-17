#!/usr/bin/env python3
"""
小火轮 v4 准确性验证系统 🔍
每天跑一次，确保 Node.js 评分 ≈ Python 评分

检查项:
1. 评分一致性 — 同一份Sina数据，Node.js和Python算分是否一致
2. 数据源稳定性 — Sina API是否正常返回
3. 结果合理性 — 分数分布是否正常
"""

import json, urllib.request, subprocess, os, sys, math
from datetime import datetime

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERBOSE = '--verbose' in sys.argv

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def http_get(url, timeout=10):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        return urllib.request.urlopen(req, timeout=timeout).read().decode('utf-8')
    except Exception as e:
        return None

def fetch_sina_kline(code, days=250):
    prefix = 'sh' if code.startswith('6') or code.startswith('5') else 'sz'
    url = f"https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketData.getKLineData?symbol={prefix}{code}&scale=240&datalen={days}"
    data = http_get(url)
    if not data: return None
    try:
        arr = json.loads(data)
        if not isinstance(arr, list) or len(arr) < 20: return None
        # 转换成统一格式 (oldest-first)
        return [{'close': float(x['close']), 'high': float(x['high']), 
                 'low': float(x['low']), 'open': float(x['open']),
                 'volume': int(x.get('volume',0)), 'day': x.get('day','')} for x in arr]
    except: return None

# ===== Python版 V1评分（与回测完全一致）=====
def calc_sma(data, period):
    if len(data) < period: return None
    return sum(d['close'] for d in data[-period:]) / period

def calc_ema(values, period):
    k = 2 / (period + 1)
    result = values[0]
    for v in values[1:]: result = v * k + result * (1 - k)
    return result

def calc_macd(data):
    closes = [d['close'] for d in data]
    if len(closes) < 26: return {'macd': 0, 'signal': 0, 'histogram': 0}
    ema12 = calc_ema(closes, 12)
    ema26 = calc_ema(closes, 26)
    macd_line = ema12 - ema26
    # Simplified signal
    signals = []
    for i in range(max(0, len(closes) - 26), len(closes)):
        e12 = calc_ema(closes[:i+1], 12)
        e26 = calc_ema(closes[:i+1], 26)
        signals.append(e12 - e26)
    signal = sum(signals) / len(signals)
    return {'macd': macd_line, 'signal': signal, 'histogram': macd_line - signal}

def calc_rsi(data, period=14):
    closes = [d['close'] for d in data]
    if len(closes) < period + 1: return 50
    gains, losses = 0, 0
    for i in range(len(closes) - period, len(closes)):
        diff = closes[i] - closes[i-1]
        if diff > 0: gains += diff
        else: losses += abs(diff)
    avg_gain, avg_loss = gains/period, losses/period
    if avg_loss == 0: return 100
    return 100 - (100 / (1 + avg_gain/avg_loss))

def calc_adx(data, period=14):
    if len(data) < period * 2: return 20
    highs = [d['high'] for d in data]
    lows = [d['low'] for d in data]
    closes = [d['close'] for d in data]
    tr_sum = dm_plus_sum = dm_minus_sum = 0
    start = len(data) - period
    for i in range(start, len(data)):
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        dm_plus = max(0, highs[i]-highs[i-1])
        dm_minus = max(0, lows[i-1]-lows[i])
        tr_sum += tr; dm_plus_sum += dm_plus; dm_minus_sum += dm_minus
    atr = tr_sum / period
    if atr == 0: return 20
    di_plus = 100 * (dm_plus_sum/period) / atr
    di_minus = 100 * (dm_minus_sum/period) / atr
    if di_plus + di_minus == 0: return 20
    return round(100 * abs(di_plus - di_minus) / (di_plus + di_minus))

def score_v1(data):
    """V1评分（纯Python，与回测一致）"""
    if not data or len(data) < 60: return 0
    
    closes = [d['close'] for d in data]
    price = closes[-1]
    ma5 = calc_sma(data, 5)
    ma20 = calc_sma(data, 20)
    ma60 = calc_sma(data, 60)
    
    high52 = max(closes[-252:]) if len(closes) >= 252 else max(closes)
    low52 = min(closes[-252:]) if len(closes) >= 252 else min(closes)
    pos52 = ((price - low52) / (high52 - low52)) * 100 if high52 > low52 else 50
    
    macd = calc_macd(data)
    rsi = calc_rsi(data)
    adx = calc_adx(data)
    
    if macd['histogram'] <= 0: return 0
    
    # MACD状态
    ms = 0
    if macd['histogram'] > 0 and macd['macd'] > macd['signal']: ms = 12
    elif macd['histogram'] > 0: ms = 6
    
    # 52周位置
    ws = 0
    if pos52 < 20: ws = 20
    elif pos52 < 35: ws = 15
    elif pos52 < 50: ws = 10
    elif pos52 < 65: ws = 6
    elif pos52 < 80: ws = 3
    
    # 均线系统
    mas = 0
    if price and ma20 and price > ma20: mas += 7
    if ma5 and ma20 and ma5 > ma20: mas += 7
    if ma20 and ma60 and ma20 > ma60: mas += 6
    
    # ADX
    ads = -5
    if adx >= 35: ads = 20
    elif adx >= 28: ads = 15
    elif adx >= 22: ads = 10
    elif adx >= 18: ads = 5
    
    # RSI
    rs = 0
    if rsi < 25: rs = 20
    elif rsi < 35: rs = 14
    elif rsi < 50: rs = 10
    elif rsi < 65: rs = 6
    elif rsi < 75: rs = 2
    else: rs = -5
    
    # 动态权重
    is_trending = adx >= 22
    wl = [25, 15, 15, 25, 20] if is_trending else [10, 30, 15, 10, 35]
    sw = sum(wl)
    
    total = (ms * wl[0]/20) + (ws * wl[1]/20) + (mas * wl[2]/20) + (ads * wl[3]/20) + (rs * wl[4]/20)
    return round(min(total / sw * 100, 100))

def compare_with_node():
    """读取Node.js评分结果，与Python对比"""
    try:
        with open(f'{WORKSPACE}/data/verify_node_scores.json') as f:
            node_data = json.load(f)
    except:
        log("没有Node.js评分数据，跳过对比")
        return None, None
    
    node_scores = {s['code']: s['score'] for s in node_data.get('stocks', [])}
    
    # 用相同股票列表重新取数打分
    py_scores = {}
    for code in node_scores:
        data = fetch_sina_kline(code)
        if data:
            py_scores[code] = score_v1(data)
    
    common = [c for c in node_scores if c in py_scores]
    if not common:
        log("无共同股票可对比")
        return None, None
    
    diffs = []
    for code in common:
        ns = node_scores[code]
        ps = py_scores[code]
        diffs.append((code, ns, ps, abs(ns - ps)))
    
    diffs.sort(key=lambda x: -x[3])
    max_diff = diffs[0][3]
    avg_diff = sum(d[3] for d in diffs) / len(diffs)
    mismatch = sum(1 for d in diffs if d[3] > 2)
    
    print(f"  📊 共同股票: {len(common)}只")
    print(f"  📊 平均差异: {avg_diff:.1f}分, 最大差异: {max_diff}分")
    print(f"  📊 偏差>2分: {mismatch}只")
    
    if mismatch > 0:
        print(f"  ⚠️  以下股票评分差异 > 2分:")
        for code, ns, ps, d in diffs[:mismatch]:
            print(f"    {code}: Node={ns} Python={ps} 差{d}")
    
    if avg_diff < 0.5 and mismatch == 0:
        print(f"  ✅ Node.js与Python评分完全一致")
    elif avg_diff < 1.0 and mismatch <= 2:
        print(f"  ✅ 基本一致 (微小偏差)")
    else:
        print(f"  ⚠️  评分系统存在偏差，需排查")
    
    return avg_diff, mismatch

def verify():
    print("=" * 60)
    print("🔍 小火轮 v4 准确性验证")
    print(f"   日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # ===== 1. 数据源可用性 =====
    print("\n[1/4] 数据源检查")
    test_codes = ['000001', '600519', '300750']
    data_ok = 0
    for code in test_codes:
        data = fetch_sina_kline(code)
        if data and len(data) >= 100:
            data_ok += 1
            if VERBOSE:
                print(f"  ✅ {code}: {len(data)}根K线, {data[0]['day']}~{data[-1]['day']}")
        else:
            print(f"  ❌ {code}: 数据异常 ({len(data) if data else 0})")
    print(f"  📊 数据源可用率: {data_ok}/{len(test_codes)} ✅" if data_ok == len(test_codes) 
          else f"  ❌ 数据源异常: {data_ok}/{len(test_codes)}")
    
    if data_ok < len(test_codes):
        print("\n  ⚠️ 数据源不稳定，继续其他检查...")
    
    # ===== 2. Node.js vs Python 评分一致性 =====
    print("\n[2/4] 评分一致性检查 (Node.js vs Python)")
    
    # 先尝试跑Node.js验证脚本获取独立评分
    log("运行Node.js验证脚本...")
    try:
        subprocess.run(['node', f'{WORKSPACE}/scripts/verify_now.js'],
                       capture_output=True, timeout=180, cwd=WORKSPACE)
    except subprocess.TimeoutExpired:
        log("⚠️ Node.js验证超时")
    except Exception as e:
        log(f"⚠️ Node.js验证失败: {e}")
    
    avg_diff, mismatch = compare_with_node()
    
    # ===== 3. 结果合理性 =====
    print("\n[3/4] 结果合理性检查")
    
    all_py_scores = [s for c, s in py_scores.items()]
    if all_py_scores:
        avg = sum(all_py_scores) / len(all_py_scores)
        max_s = max(all_py_scores)
        above_62 = sum(1 for s in all_py_scores if s >= 62)
        above_50 = sum(1 for s in all_py_scores if s >= 50)
        print(f"  📊 扫描{len(all_py_scores)}只: 平均{avg:.0f}分, 最高{max_s}分")
        print(f"  📊 ≥62分(可买入): {above_62}只 ({above_62/len(all_py_scores)*100:.0f}%)")
        print(f"  📊 ≥50分(可持有): {above_50}只 ({above_50/len(all_py_scores)*100:.0f}%)")
        
        # 回测中每调仓日平均167只≥62分，所以要有合理的数量
        if above_62 < 3:
            print(f"  ⚠️  可买入标的过少({above_62}只)，可能市场极差或数据异常")
        elif above_62 > 200:
            print(f"  ⚠️  可买入标的过多({above_62}只)，可能评分偏乐观")
        else:
            print(f"  ✅  标的数量合理 (范围3-200，实际{above_62})")
    else:
        print(f"  ⚠️  无评分数据")
    
    # ===== 4. 整体结论 =====
    print(f"\n[4/4] 整体结论")
    
    all_checks_passed = (data_ok == len(test_codes) and 
                         'avg_diff' in locals() and avg_diff < 1.0 and
                         'above_62' in locals() and 3 <= above_62 <= 200)
    
    if all_checks_passed:
        print(f"\n  ✅ 验证通过 — 系统运行正常")
    else:
        issues = []
        if data_ok < len(test_codes): issues.append("数据源异常")
        if 'avg_diff' in locals() and avg_diff >= 1.0: issues.append(f"评分偏差{avg_diff:.1f}分")
        if 'above_62' in locals():
            if above_62 < 3: issues.append("可买入标的过少")
            elif above_62 > 200: issues.append("评分偏乐观")
        print(f"\n  ⚠️  发现潜在问题: {'; '.join(issues)}")

if __name__ == '__main__':
    verify()
