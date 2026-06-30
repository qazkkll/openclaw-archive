#!/usr/bin/env python3
"""技术指标计算"""
import tushare as ts
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

with open('/home/hermes/.hermes/openclaw-archive/data/config/tushare.json') as f:
    cfg = json.load(f)
ts.set_token(cfg['token'])
pro = ts.pro_api()

ts_code = '300693.SZ'
df = pro.daily(ts_code=ts_code, start_date='20260101', end_date='20260630')
df = df.sort_values('trade_date').reset_index(drop=True)

# RSI计算
delta = df['close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df['rsi'] = 100 - (100 / (1 + rs))

# MACD
exp12 = df['close'].ewm(span=12).mean()
exp26 = df['close'].ewm(span=26).mean()
df['macd'] = exp12 - exp26
df['signal'] = df['macd'].ewm(span=9).mean()
df['hist'] = df['macd'] - df['signal']

# KDJ
low_min = df['low'].rolling(9).min()
high_max = df['high'].rolling(9).max()
rsv = (df['close'] - low_min) / (high_max - low_min) * 100
df['k'] = rsv.ewm(com=2).mean()
df['d'] = df['k'].ewm(com=2).mean()
df['j'] = 3 * df['k'] - 2 * df['d']

# 布林带
df['bb_mid'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

# 成交量均线
df['vol_ma5'] = df['vol'].rolling(5).mean()
df['vol_ma20'] = df['vol'].rolling(20).mean()

print("=" * 80)
print("📊 300693 盛弘股份 技术指标分析")
print("=" * 80)

last = df.iloc[-1]
print(f"\n最新交易日: {last['trade_date']}")
print(f"收盘价: {last['close']}")
print(f"\n--- 技术指标 ---")
print(f"RSI(14): {last['rsi']:.2f}")
print(f"MACD: {last['macd']:.4f}")
print(f"MACD Signal: {last['signal']:.4f}")
print(f"MACD Histogram: {last['hist']:.4f}")
print(f"KDJ K: {last['k']:.2f}")
print(f"KDJ D: {last['d']:.2f}")
print(f"KDJ J: {last['j']:.2f}")
print(f"布林上轨: {last['bb_upper']:.2f}")
print(f"布林中轨: {last['bb_mid']:.2f}")
print(f"布林下轨: {last['bb_lower']:.2f}")
print(f"成交量: {last['vol']:.0f}")
print(f"成交额: {last['amount']:.0f}万")
print(f"量比(vs MA5): {last['vol'] / last['vol_ma5']:.2f}")
print(f"量比(vs MA20): {last['vol'] / last['vol_ma20']:.2f}")

# 近期MACD交叉
recent = df.tail(20)
macd_cross_up = []
macd_cross_down = []
for i in range(1, len(recent)):
    prev = recent.iloc[i-1]
    curr = recent.iloc[i]
    if prev['macd'] < prev['signal'] and curr['macd'] > curr['signal']:
        macd_cross_up.append(curr['trade_date'])
    elif prev['macd'] > prev['signal'] and curr['macd'] < curr['signal']:
        macd_cross_down.append(curr['trade_date'])

print(f"\n近20日MACD金叉: {macd_cross_up if macd_cross_up else '无'}")
print(f"近20日MACD死叉: {macd_cross_down if macd_cross_down else '无'}")

# KDJ交叉
kdj_cross_up = []
for i in range(1, len(recent)):
    prev = recent.iloc[i-1]
    curr = recent.iloc[i]
    if prev['k'] < prev['d'] and curr['k'] > curr['d'] and curr['k'] < 30:
        kdj_cross_up.append(curr['trade_date'])
print(f"近20日KDJ低位金叉(<30): {kdj_cross_up if kdj_cross_up else '无'}")

# 波动率
returns = df['close'].pct_change().tail(20)
vol_ann = returns.std() * (252 ** 0.5) * 100
print(f"\n20日年化波动率: {vol_ann:.2f}%")

# 前期关键支撑/阻力
recent30 = df.tail(30)
print(f"\n30日价格范围:")
print(f"  最高: {recent30['high'].max()} ({recent30.loc[recent30['high'].idxmax(), 'trade_date']})")
print(f"  最低: {recent30['low'].min()} ({recent30.loc[recent30['low'].idxmin(), 'trade_date']})")
print(f"  中位价: {recent30['close'].median():.2f}")

# 涨跌幅统计
print(f"\n近期涨跌幅:")
for i in range(-5, 0):
    r = df.iloc[i]
    print(f"  {r['trade_date']}: {r['pct_chg']:+.2f}% (O:{r['open']} H:{r['high']} L:{r['low']} C:{r['close']})")

print("\n" + "=" * 80)
