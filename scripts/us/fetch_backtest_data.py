#!/usr/bin/env python3
"""
批量获取A股历史日K线（用于回测扩展）
源: Sina Finance API（免费，无需令牌）
每只股票1024条日K线（约4年）
"""
import json, re, time, sys, os, urllib.request

# 加载股票列表
with open('/home/admin/.openclaw/workspace/data/backtest_universe.json') as f:
    codes = json.load(f)

def get_prefix(code):
    if code.startswith('6') or code.startswith('5'):
        return 'sh'
    elif code.startswith('0') or code.startswith('3'):
        return 'sz'
    elif code.startswith('8') or code.startswith('4'):
        return 'bj'
    return 'sz'

def fetch_sina(code):
    """从Sina获取日K线（1024条）"""
    prefix = get_prefix(code)
    url = f'https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_x_{prefix}{code}=/CN_MarketData.getKLineData?symbol={prefix}{code}&scale=240&ma=no&datalen=1024'
    
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        resp = urllib.request.urlopen(req, timeout=15)
        data = resp.read().decode('utf-8', errors='replace')
        
        # 提取JSON
        m = re.search(r'\[.*\]', data)
        if not m:
            return None
        
        klines = json.loads(m.group())
        if not klines:
            return None
        
        # 转换为标准格式
        result = {
            'dates': [],
            'close': [],
            'high': [],
            'low': [],
            'volume': [],
        }
        
        for k in klines:
            result['dates'].append(k['day'][:10])
            result['close'].append(float(k['close']))
            result['high'].append(float(k['high']))
            result['low'].append(float(k['low']))
            result['volume'].append(int(float(k.get('volume', 0))))
        
        return result
    except Exception as e:
        return None

# 开始采集
print(f"开始采集 {len(codes)} 只股票的日K线...")
print(f"来源: Sina Finance (免费)")
print(f"每只股票: 约1024条日K线 (~4年)")
print()

results = {}
errors = []
existing = {}

# 先加载已有的test_50_hist.json数据（已有500天）
if os.path.exists('/home/admin/.openclaw/workspace/data/test_50_hist.json'):
    with open('/home/admin/.openclaw/workspace/data/test_50_hist.json') as f:
        existing = json.load(f)
    print(f"已加载现有 test_50_hist.json: {len(existing)} 只")

for i, code in enumerate(codes):
    # 检查是否已有数据（优先用已有的，只抓取缺失的）
    if code in existing:
        results[code] = existing[code]
        print(f"  [{i+1}/{len(codes)}] {code} → 使用已有数据 ({len(existing[code].get('close',[]))}天)")
        continue
    
    # 从Sina抓取
    data = fetch_sina(code)
    if data and len(data['close']) >= 200:  # 至少200天有效
        results[code] = data
        days = len(data['close'])
        print(f"  [{i+1}/{len(codes)}] {code} → ✅ {days}天 ({data['dates'][0]} ~ {data['dates'][-1]})")
    else:
        errors.append(code)
        print(f"  [{i+1}/{len(codes)}] {code} → ❌ 获取失败")
    
    # 每20只休息2秒，避免被封
    if (i+1) % 20 == 0:
        print(f"  --- 休息2秒 ---")
        time.sleep(2)
    
    # 每只之间间隔0.5秒
    time.sleep(0.5)

# 保存
output = {}
for code in codes:
    if code in results:
        output[code] = results[code]

with open('/home/admin/.openclaw/workspace/data/backtest_hist_v2.json', 'w') as f:
    json.dump(output, f)

print(f"\n{'='*60}")
print(f"采集完成!")
print(f"  成功: {len(output)} 只")
print(f"  失败: {len(errors)} 只")
if errors:
    print(f"  失败列表: {errors}")
print(f"  已保存: data/backtest_hist_v2.json")
print(f"{'='*60}")
