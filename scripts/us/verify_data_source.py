#!/usr/bin/env python3
"""
verify_data_source.py — 数据源交叉验证

取 10 只股票的同一日期段 (2024-01 ~ 2024-12)，
分别从以下来源获取收盘价：
  a) Yahoo Finance (backtest_hist_yahoo.json)
  b) 新浪财经 API

对比：
  1. 同一日收盘价差异 (%) — 超过 2% 报警
  2. MACD 方向一致性（涨/跌方向是否相同）

运行: python3 scripts/verify_data_source.py
"""

import sys, json, time, urllib.request, urllib.parse
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from score_engine import compute_indicators

# ============================================================
#  选择 10 只测试股票
# ============================================================

# 从历史数据中选取不同板块的代表性A股
TEST_CODES = [
    '600519',  # 贵州茅台 (消费)
    '000858',  # 五粮液 (消费)
    '600036',  # 招商银行 (金融)
    '000002',  # 万科A (地产)
    '601318',  # 中国平安 (保险)
    '300750',  # 宁德时代 (新能源)
    '000333',  # 美的集团 (家电)
    '600887',  # 伊利股份 (乳业)
    '002415',  # 海康威视 (安防)
    '601012',  # 隆基绿能 (光伏)
]


def code_prefix(code):
    """根据股票代码返回新浪 API 的前缀。"""
    if code.startswith('6') or code.startswith('9'):
        return 'sh'
    return 'sz'


# ============================================================
#  读取 Yahoo Finance 数据
# ============================================================

print("📥 加载 Yahoo Finance 数据 ...")
with open('data/backtest_hist_yahoo.json') as f:
    yahoo_data = json.load(f)

print(f"  共 {len(yahoo_data)} 只股票")


def get_yahoo_close(code, start_date='2024-01-01', end_date='2024-12-31'):
    """从 Yahoo 数据提取收盘价 DataFrame。"""
    d = yahoo_data.get(code)
    if not d:
        return None
    dates = d.get('dates', [])
    close = d.get('close', [])
    result = {}
    for dt, pr in zip(dates, close):
        if start_date <= dt <= end_date:
            result[dt] = pr
    return result


# ============================================================
#  新浪 API 获取
# ============================================================

def fetch_sina_klines(code, datalen=1500):
    """
    从新浪财经 K 线 API 获取历史日K数据。
    返回 [(date, close), ...] 或 None。
    """
    prefix = code_prefix(code)
    url = (f"https://quotes.sina.cn/cn/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={prefix}{code}"
           f"&scale=240&datalen={datalen}")
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Accept': '*/*',
        })
        resp = urllib.request.urlopen(req, timeout=15)
        raw = resp.read().decode('gbk')
        data = json.loads(raw)
        result = []
        for item in data:
            dt = item.get('date', '')
            close = float(item.get('close', 0))
            if dt and close > 0:
                result.append((dt, close))
        result.sort(key=lambda x: x[0])
        return result
    except Exception as e:
        print(f"  ⚠️ 新浪API失败 ({code}): {e}")
        return None


def get_sina_close(code, start_date='2024-01-01', end_date='2024-12-31'):
    """从新浪 API 提取收盘价 dict {date: close}。"""
    raw = fetch_sina_klines(code)
    if not raw:
        return None
    result = {}
    for dt, pr in raw:
        if start_date <= dt <= end_date:
            result[dt] = pr
    return result


# ============================================================
#  MACD 方向计算
# ============================================================

def compute_macd_direction(close_prices):
    """计算 MACD 柱线方向: 1=上升, -1=下降, 0=持平。"""
    if len(close_prices) < 27:
        return None

    def ema(arr, p):
        k = 2 / (p + 1)
        r = [arr[0]]
        for v in arr[1:]:
            r.append(v * k + r[-1] * (1 - k))
        return r

    e12 = ema(close_prices, 12)
    e26 = ema(close_prices, 26)
    macd_line = [e12[i] - e26[i] for i in range(len(close_prices))]
    signal = ema(macd_line, 9)
    macd_hist = [macd_line[i] - signal[i] for i in range(len(close_prices))]

    # 最后两天的柱线方向
    if len(macd_hist) < 2:
        return None
    last = macd_hist[-1]
    prev = macd_hist[-2]
    if last > prev:
        return 1
    elif last < prev:
        return -1
    return 0


# ============================================================
#  主验证
# ============================================================

def main():
    print(f"\n{'='*70}")
    print("🔬 数据源交叉验证: Yahoo Finance vs 新浪API")
    print(f"  测试期间: 2024-01 ~ 2024-12")
    print(f"  测试股票数: {len(TEST_CODES)}")
    print(f"{'='*70}")

    total_dates = 0
    big_diff_count = 0
    total_diff_pct = []
    yahoo_macd_dir = {}
    sina_macd_dir = {}
    macd_mismatch = 0
    macd_total = 0

    for code in TEST_CODES:
        print(f"\n{'─'*60}")
        print(f"📊 {code}")

        # Yahoo 数据
        yahoo_close = get_yahoo_close(code)
        if not yahoo_close:
            print(f"  ❌ Yahoo: 无数据")
            continue
        common_dates = set(yahoo_close.keys())
        print(f"  Yahoo: {min(yahoo_close.keys())} ~ {max(yahoo_close.keys())}, "
              f"{len(yahoo_close)} 天")

        # 新浪数据
        sina_close = get_sina_close(code)
        if not sina_close:
            print(f"  ⚠️ 新浪: 获取失败，跳过该股")
            continue
        common_dates &= set(sina_close.keys())
        print(f"  新浪: {min(sina_close.keys())} ~ {max(sina_close.keys())}, "
              f"{len(sina_close)} 天")

        if len(common_dates) == 0:
            print(f"  ❌ 无共同交易日期")
            continue

        print(f"  共同日期: {len(common_dates)} 天")

        # --- 对比收盘价 ---
        diffs = []
        for dt in sorted(common_dates):
            yp = yahoo_close[dt]
            sp = sina_close[dt]
            diff_pct = abs(yp - sp) / ((yp + sp) / 2) * 100
            diffs.append(diff_pct)
            total_diff_pct.append(diff_pct)

        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        big_diffs = sum(1 for d in diffs if d > 2.0)
        big_diff_count += big_diffs

        print(f"  平均差异: {avg_diff:.3f}%  |  最大差异: {max_diff:.3f}%")
        print(f"  差异 >2% 天数: {big_diffs}/{len(diffs)}")

        if max_diff > 5:
            print(f"  ⚠️ 差异 >5% 的日期:")
            for dt in sorted(common_dates):
                yp = yahoo_close[dt]
                sp = sina_close[dt]
                dp = abs(yp - sp) / ((yp + sp) / 2) * 100
                if dp > 5:
                    print(f"    {dt}: Yahoo={yp:.2f} 新浪={sp:.2f} 差异={dp:.2f}%")

        if avg_diff < 0.5:
            print(f"  ✅ 价格一致性良好")
        elif avg_diff < 2:
            print(f"  🟡 价格差异可接受")
        else:
            print(f"  ❌ 价格差异偏大")

        # --- 对比 MACD 方向 ---
        yahoo_prices_list = [yahoo_close[dt] for dt in sorted(common_dates)]
        sina_prices_list = [sina_close[dt] for dt in sorted(common_dates)]

        y_dir = compute_macd_direction(yahoo_prices_list)
        s_dir = compute_macd_direction(sina_prices_list)

        if y_dir is not None and s_dir is not None:
            yahoo_macd_dir[code] = y_dir
            sina_macd_dir[code] = s_dir
            macd_total += 1
            match = y_dir == s_dir
            if not match:
                macd_mismatch += 1

            dir_label = {1: '↑上升', -1: '↓下降', 0: '→持平'}
            print(f"  MACD方向 | Yahoo: {dir_label.get(y_dir, '?')} "
                  f"| 新浪: {dir_label.get(s_dir, '?')} "
                  f"| {'✅ 一致' if match else '❌ 不一致'}")

        total_dates += len(common_dates)

    # ============================================================
    #  汇总
    # ============================================================
    print(f"\n{'='*70}")
    print("📊 汇总报告")
    print(f"{'='*70}")

    if total_diff_pct:
        avg_all = sum(total_diff_pct) / len(total_diff_pct)
        max_all = max(total_diff_pct)
        print(f"  总有效对比天数: {total_dates}")
        print(f"  平均价格差异: {avg_all:.3f}%")
        print(f"  最大价格差异: {max_all:.3f}%")
        print(f"  差异 >2% 累计天数: {big_diff_count}/{total_dates} "
              f"({big_diff_count/total_dates*100:.1f}%)")

        if avg_all < 0.5:
            print(f"  ✅ 整体价格一致性: 优秀")
        elif avg_all < 1:
            print(f"  ✅ 整体价格一致性: 良好")
        elif avg_all < 2:
            print(f"  🟡 整体价格一致性: 可接受")
        else:
            print(f"  ❌ 整体价格一致性: 偏差较大")

    print(f"\n  MACD 方向验证:")
    print(f"    方向一致性: {macd_total - macd_mismatch}/{macd_total} 一致")
    if macd_mismatch == 0:
        print(f"    ✅ MACD 方向完全一致")
    else:
        print(f"    ❌ {macd_mismatch} 只股票方向不一致")
        for code in yahoo_macd_dir:
            if yahoo_macd_dir[code] != sina_macd_dir.get(code):
                print(f"       {code}: Yahoo={yahoo_macd_dir[code]} 新浪={sina_macd_dir[code]}")

    print(f"\n{'='*70}")
    verdict = "✅ 数据源交叉验证通过" if (avg_all < 2 and macd_mismatch == 0) else "⚠️ 存在差异需关注"
    print(f"  结论: {verdict}")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
