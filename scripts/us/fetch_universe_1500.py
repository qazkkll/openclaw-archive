#!/usr/bin/env python3
"""
抓取1500只A股历史日K线 — 优化版
分批抓取、每100只保存、可断点续传
"""
import json, re, time, os, urllib.request

UNIVERSE_FILE = '/home/admin/.openclaw/workspace/data/universe_1500.json'
OUTPUT_FILE = '/home/admin/.openclaw/workspace/data/backtest_hist_v3.json'
EXISTING_FILE = '/home/admin/.openclaw/workspace/data/backtest_hist_v2_filtered.json'

def get_prefix(code):
    if code.startswith('6') or code.startswith('5'): return 'sh'
    elif code.startswith('0') or code.startswith('3'): return 'sz'
    elif code.startswith('8') or code.startswith('4'): return 'bj'
    return 'sz'

def fetch_sina(code):
    prefix = get_prefix(code)
    url = f'https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_x_{prefix}{code}=/CN_MarketData.getKLineData?symbol={prefix}{code}&scale=240&ma=no&datalen=1024'
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        resp = urllib.request.urlopen(req, timeout=20)
        raw = resp.read().decode('utf-8', errors='replace')
        m = re.search(r'\[.*\]', raw)
        if not m: return None
        klines = json.loads(m.group())
        if not klines or len(klines) < 100: return None
        return {
            'dates': [k['day'][:10] for k in klines],
            'close': [float(k['close']) for k in klines],
            'high': [float(k['high']) for k in klines],
            'low': [float(k['low']) for k in klines],
            'volume': [int(float(k.get('volume', 0))) for k in klines],
        }
    except:
        return None

# 加载
with open(UNIVERSE_FILE) as f:
    universe = json.load(f)

# 加载已有的数据（如果有）
existing = {}
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        existing = json.load(f)
    print(f"已有输出文件: {len(existing)} 只")

# 加载之前的191只数据（作为备用，如果新抓取失败）
legacy = {}
if os.path.exists(EXISTING_FILE):
    with open(EXISTING_FILE) as f:
        legacy = json.load(f)
    print(f"已有历史数据(legacy): {len(legacy)} 只")

# 要抓取的列表
to_fetch = [c for c in universe if c not in existing]
print(f"需要抓取: {len(to_fetch)} 只 (共{len(universe)}只)")
print(f"开始时间: {time.strftime('%H:%M')}")

success = len(existing)
errors = 0
batch = 0

for i, code in enumerate(to_fetch):
    # 优先用legacy数据
    if code in legacy:
        existing[code] = legacy[code]
        success += 1
        continue
    
    data = fetch_sina(code)
    if data:
        existing[code] = data
        success += 1
        days = len(data['close'])
        print(f"  [{i+1}/{len(to_fetch)}] {code} ✅ {days}天")
    else:
        errors += 1
        print(f"  [{i+1}/{len(to_fetch)}] {code} ❌")
    
    # 每批保存一次
    if (i+1) % 100 == 0:
        batch += 1
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(existing, f)
        print(f"  --- 第{batch}批保存: {success}成功 {errors}失败 ---")
        time.sleep(1)
    
    time.sleep(0.3)

# 最终保存
with open(OUTPUT_FILE, 'w') as f:
    json.dump(existing, f)

print(f"\n{'='*60}")
print(f"完成!")
print(f"  目标: {len(universe)} 只")
print(f"  成功: {success} 只")
print(f"  失败: {errors} 只")
print(f"  已保存: {OUTPUT_FILE}")
print(f"  完成时间: {time.strftime('%H:%M')}")
