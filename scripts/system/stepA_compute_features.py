"""
stepA_compute_features.py — 特征计算 + 存中间文件（可在任意位置卡死恢复）

输出: /home/hermes/.hermes/openclaw-project/data/a_ml_feats_cache.json
  {"X": [[特征...], ...], "y": [0/1, ...]}
  约 5000只 * ~800天 * 16特征 ≈ ~500MB

用法: python stepA_compute_features.py
"""
import json, sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

CACHE = '/home/hermes/.hermes/openclaw-project/data/a_ml_feats_cache.json'
STOCKS_FILE = '/home/hermes/.hermes/openclaw-project/data/a_ml_stocks_done.json'

t0 = time.time()

# 1. 加载K线
print('1. 加载K线...', flush=True)
with open('/home/hermes/.hermes/openclaw-project/data/a_hist_10y.json', 'r', encoding='utf-8') as f:
    hist = json.load(f)
print(f'  加载: {len(hist)}只', flush=True)

# 2. 筛选主板 + 500天以上
codes = [c for c in hist if c.startswith(('60','00')) and len(hist[c].get('dates',[])) >= 500]
print(f'  主板+500天: {len(codes)}只', flush=True)

# 检查断点
done_set = set()
if os.path.exists(STOCKS_FILE):
    with open(STOCKS_FILE, 'r') as f:
        done_set = set(json.load(f))
    print(f'  断点: {len(done_set)}只已完成', flush=True)

# 3. 特征计算
print('2. 计算特征...', flush=True)
X_list = []
y_list = []
sc = 0

for code in codes:
    if code in done_set:
        sc += 1
        continue
    
    h = hist[code]
    try:
        c = np.array(h['c'][::-1], dtype=np.float64)
        hi = np.array(h['h'][::-1], dtype=np.float64)
        lo = np.array(h['l'][::-1], dtype=np.float64)
        v = np.array(h['v'][::-1], dtype=np.float64)
    except:
        done_set.add(code)
        continue
    
    n = len(c)
    if n < 200:
        done_set.add(code)
        continue
    
    rows_x = []
    rows_y = []
    
    for i in range(100, n - 5):
        # 动量
        r1 = c[i]/c[i-1]-1 if c[i-1] > 0 else 0
        r5 = c[i]/c[i-5]-1 if c[i-5] > 0 else 0
        r20 = c[i]/c[i-20]-1 if c[i-20] > 0 else 0
        
        # 均线 + 偏离度
        m5 = np.mean(c[i-4:i+1]); m10 = np.mean(c[i-9:i+1])
        m20 = np.mean(c[i-19:i+1]); m60 = np.mean(c[i-59:i+1]) if i >= 59 else m20
        d5 = c[i]/m5-1; d20 = c[i]/m20-1; d60 = c[i]/m60-1
        align = 1 if m5 > m10 > m20 else (-1 if m5 < m10 < m20 else 0)
        
        # RSI
        chgs = np.diff(c[i-13:i+1])
        ag = np.mean(chgs[chgs > 0]) if np.any(chgs > 0) else 0.001
        al = -np.mean(chgs[chgs < 0]) if np.any(chgs < 0) else 0.001
        rsi = 100 - 100 / (1 + ag/al)
        
        # MACD（简化均线）
        macd = np.mean(c[i-11:i+1]) - np.mean(c[i-25:i+1])
        
        # 成交量 + 位置
        vr = v[i] / np.mean(v[i-4:i+1]) if np.mean(v[i-4:i+1]) > 0 else 1
        h20 = np.max(hi[i-19:i+1]); l20 = np.min(lo[i-19:i+1])
        pos = (c[i]-l20)/(h20-l20) if h20 > l20 else 0.5
        
        # 波动率
        v5 = np.std([c[j]/c[j-1]-1 for j in range(i-4, i+1)])
        v20 = np.std([c[j]/c[j-1]-1 for j in range(i-19, i+1)])
        
        # 标签
        rf = c[i+5]/c[i]-1
        if c[i] > 0 and c[i+5] > 0:
            y = 1.0 if rf > 0.02 else 0.0
            rows_x.append([r1, r5, r20, d5, d20, d60, align, v5, v20, rsi, macd, vr, pos, c[i]/m60])
            rows_y.append(y)
    
    if rows_x:
        X_list.extend(rows_x)
        y_list.extend(rows_y)
    
    done_set.add(code)
    sc += 1
    
    # 每200只存盘
    if sc % 200 == 0:
        cache = {'X': X_list, 'y': y_list}
        with open(CACHE, 'w') as f:
            json.dump(cache, f)
        with open(STOCKS_FILE, 'w') as f:
            json.dump(list(done_set), f)
        print(f'  {sc}/{len(codes)}只, X: {len(X_list)}行', flush=True)

# 最终存盘
cache = {'X': X_list, 'y': y_list}
with open(CACHE, 'w') as f:
    json.dump(cache, f)
with open(STOCKS_FILE, 'w') as f:
    json.dump(list(done_set), f)
print(f'\n✅ 特征完成: {len(codes)}只, {len(X_list)}行', flush=True)
print(f'缓存: {CACHE}', flush=True)
print(f'耗时: {(time.time()-t0)/60:.1f}分钟', flush=True)
