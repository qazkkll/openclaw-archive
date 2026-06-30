#!/usr/bin/env python3
"""A股超卖反弹分析 - 000801/601727/600941"""
import tushare as ts
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 配置
with open('/home/hermes/.hermes/openclaw-archive/data/config/tushare.json') as f:
    cfg = json.load(f)
ts.set_token(cfg['token'])
pro = ts.pro_api()

stocks = [
    {'code': '000801.SZ', 'name': '四川九洲', 'price': 11.76, 'rsi': 37.2, 'drop20d': -12, 'fund': 97, 'sector': '家用电器'},
    {'code': '601727.SH', 'name': '上海电气', 'price': 7.08, 'rsi': 24.9, 'drop20d': -13.1, 'fund': 95, 'sector': '电气设备'},
    {'code': '600941.SH', 'name': '中国移动', 'price': 87.60, 'rsi': 22.0, 'drop20d': -9.5, 'fund': 94, 'sector': '电信运营'},
]

end_date = '20260630'
start_date = '20260601'  # ~20 trading days

results = []

for s in stocks:
    code = s['code']
    print(f"\n{'='*60}")
    print(f"分析: {s['name']} ({code})")
    print(f"{'='*60}")
    
    # 1. 日K线
    try:
        df = pro.daily(ts_code=code, start_date='20260525', end_date=end_date)
        df = df.sort_values('trade_date').reset_index(drop=True)
        print(f"\n近20日K线数据 ({len(df)}条):")
        print(df[['trade_date','open','high','low','close','vol','amount']].to_string())
    except Exception as e:
        print(f"K线获取失败: {e}")
        df = None
    
    # 2. 每日资金流向
    try:
        mf = pro.moneyflow(ts_code=code, start_date='20260525', end_date=end_date)
        mf = mf.sort_values('trade_date').reset_index(drop=True)
        print(f"\n资金流向:")
        print(mf[['trade_date','buy_sm_amount','sell_sm_amount','buy_md_amount','sell_md_amount',
                  'buy_lg_amount','sell_lg_amount','buy_elg_amount','sell_elg_amount']].tail(10).to_string())
    except Exception as e:
        print(f"资金流向获取失败: {e}")
        mf = None
    
    # 3. 基本面 - daily_basic获取PE/PB等
    try:
        basic = pro.daily_basic(ts_code=code, start_date='20260620', end_date=end_date)
        if basic is not None and len(basic) > 0:
            b = basic.sort_values('trade_date').iloc[-1]
            print(f"\n基本面: pe_ttm={b.get('pe_ttm')}, pb={b.get('pb')}, total_mv={b.get('total_mv')}, turnover_rate={b.get('turnover_rate')}")
    except Exception as e:
        print(f"基本面获取失败: {e}")
    
    # 4. 个股资金流排名
    try:
        rank = pro.moneyflow_hsgt(start_date='20260625', end_date=end_date)
        # This is market-level, skip
    except:
        pass
    
    # K线形态分析
    if df is not None and len(df) >= 5:
        last5 = df.tail(5)
        closes = last5['close'].values
        opens = last5['open'].values
        highs = last5['high'].values
        lows = last5['low'].values
        
        # 判断趋势
        trend = "下跌" if closes[-1] < closes[0] else "上涨" if closes[-1] > closes[0] else "横盘"
        
        # 下影线判断（探底回升信号）
        lower_shadows = [(o - l) if c > o else (c - l) for o, l, c in zip(opens, lows, closes)]
        has_long_lower_shadow = any(ls / (h - l + 0.001) > 0.5 for ls, h, l in zip(lower_shadows, highs, lows))
        
        # 成交量变化
        vols = df.tail(10)['vol'].values
        vol_trend = "放量" if np.mean(vols[-3:]) > np.mean(vols[:3]) * 1.2 else "缩量"
        
        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - 100 / (1 + rs)).iloc[-1]
        
        # MA支撑
        ma5 = df['close'].rolling(5).mean().iloc[-1]
        ma10 = df['close'].rolling(10).mean().iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else None
        
        print(f"\nK线形态分析:")
        print(f"  近5日趋势: {trend}, 收盘: {closes[-1]:.2f}")
        print(f"  下影线信号: {'有长下影线(探底回升)' if has_long_lower_shadow else '无明显下影线'}")
        print(f"  量能: {vol_trend}")
        print(f"  计算RSI(14): {rsi:.1f}")
        ma20_str = f"{ma20:.2f}" if ma20 else "N/A"
        print(f"  MA5={ma5:.2f}, MA10={ma10:.2f}, MA20={ma20_str}")
    
    results.append(s)

# 汇总判断
print("\n" + "="*60)
print("综合评估与建议")
print("="*60)
