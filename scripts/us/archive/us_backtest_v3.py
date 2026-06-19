#!/usr/bin/env python3
"""
美股量化回测 — 6组模型全方位对比
小钳轮动框架移植 + 双动量 + 行业轮动 + 多因子 + 均线跟踪 + 混合

数据: yfinance 5年 (2021-01 ~ 2026-05)
股票: SP500 Top 100 + 11行业ETF + 债券ETF
"""
import json, math, sys, time, os
from datetime import datetime
from collections import defaultdict

print("=" * 60)
print("📥 美股量化回测 v3 — 数据准备")
print("=" * 60)

# ===== 股票池 =====
# SP500 Top 100 by market cap (消费/科技/医药/金融/能源全覆盖)
SP500_TOP = [
    'AAPL','MSFT','NVDA','AMZN','META','GOOGL','GOOG','BRK-B','TSLA','AVGO',
    'JPM','V','XOM','UNH','LLY','COST','PG','HD','MA','JNJ',
    'MRK','CVX','ABBV','CRM','BAC','ORCL','NFLX','KO','AMD','PEP',
    'WMT','ADBE','MCD','CSCO','ABT','TMO','GE','IBM','DHR','CAT',
    'TXN','INTU','QCOM','VZ','CMCSA','WFC','PM','NEE','RTX','SPGI',
    'LOW','MS','BA','PM','AMAT','AXP','T','UNP','HON','BKNG',
    'SYK','LMT','ISRG','PLTR','BLK','BKNG','AMGN','PANW','MDT','DE',
    'SCHW','C','MU','PGR','NOW','GILD','ADP','CB','ETN','UBER',
    'MDLZ','TMUS','EQIX','MMC','ICE','SO','CI','COF','INTC','ATVI',
    'DUK','ELV','AON','ZTS','REGN','SHW','PYPL','USB','VRTX','NOC',
][:100]

# 行业ETF + 债券ETF
ETFS = {
    'XLK': '科技', 'XLC': '通信', 'XLY': '消费可选', 'XLP': '消费必选',
    'XLV': '医药', 'XLF': '金融', 'XLE': '能源', 'XLU': '公用',
    'XLI': '工业', 'XLB': '材料', 'XLRE': '地产',
    'QQQ': '纳指ETF', 'SPY': '标普ETF', 'SHY': '短债ETF', 'IEF': '中期国债',
    'GLD': '黄金', 'TLT': '长债ETF',
}

ALL_TICKERS = SP500_TOP + list(ETFS.keys())

# ===== 数据获取 =====
print(f"📡 获取 {len(ALL_TICKERS)} 只股票5年数据...")

def fetch_data(ticker, period='5y'):
    """获取单个股票的K线数据"""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period)
        if hist.empty or len(hist) < 100:
            return None
        return {
            'close': hist['Close'].tolist(),
            'high': hist['High'].tolist(),
            'low': hist['Low'].tolist(),
            'volume': hist['Volume'].tolist(),
            'dates': [str(d.date()) for d in hist.index],
        }
    except:
        return None

# 分批获取，每批间隔1秒
data = {}
success = 0
errors = []

for i, t in enumerate(ALL_TICKERS):
    d = fetch_data(t)
    if d:
        data[t] = d
        success += 1
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(ALL_TICKERS)}] {success}成功 {len(errors)}失败")
    else:
        errors.append(t)
    time.sleep(0.3)

print(f"\n✅ 数据获取完成: {success}/{len(ALL_TICKERS)}")
if errors:
    print(f"❌ 失败: {errors[:10]}...")

if success < 50:
    print("数据量不足，终止")
    sys.exit(1)

# 统一N
N = min(len(data[t]['close']) for t in data)
dates = data[list(data.keys())[0]]['dates']
print(f"  统一N={N}天 ({dates[0]}~{dates[-1]})")

# 保存原始数据
os.makedirs('/home/admin/.openclaw/workspace/data', exist_ok=True)
with open('/home/admin/.openclaw/workspace/data/us_hist_v3.json', 'w') as f:
    json.dump(data, f)
print(f"✅ 数据已保存")
