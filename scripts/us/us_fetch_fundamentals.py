#!/usr/bin/env python3
"""
拉取美股2435只的做空比例+行业信息，存本地JSON
之后特征预计算时直接读，不用每次等yfinance限速
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _paths
import yfinance as yf
import pandas as pd

T0 = time.time()
print("═══ 拉取做空比例+行业信息 ═══")

# 从旧precalc parquet拿股票列表（2435只）
df = pd.read_parquet(_paths.US_ML_FEATS)
codes = df['sym'].unique().tolist()
print(f"  股票数: {len(codes)}只")

# 分批拉取，防止yfinance限速
batch_size = 50
all_data = {}
failed = []
batch_time = T0

for i in range(0, len(codes), batch_size):
    batch = codes[i:i+batch_size]
    for c in batch:
        try:
            info = yf.Ticker(c).info
            all_data[c] = {
                'shortRatio': info.get('shortRatio'),
                'shortPct': info.get('sharesPercentSharesOut'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'marketCap': info.get('marketCap'),
                'price': info.get('currentPrice', info.get('regularMarketPrice')),
                'trailingPE': info.get('trailingPE'),
                'forwardPE': info.get('forwardPE'),
                'dividendYield': info.get('dividendYield'),
                'beta': info.get('beta')
            }
        except Exception:
            failed.append(c)
    
    # 每批打印进度
    pct = min(100, (i+batch_size)/len(codes)*100)
    elapsed = time.time()-T0
    rate = (i+batch_size)/elapsed if elapsed>0 else 0
    eta = (len(codes)-(i+batch_size))/rate if rate>0 else 0
    print(f"  [{i+batch_size}/{len(codes)}] {pct:.0f}% | {rate:.1f}只/s | ETA {eta:.0f}s", flush=True)
    time.sleep(0.5)  # 每次批间隔防限速

save_path = _paths.ML_DIR + "/us_fundamentals.json"
with open(save_path, 'w') as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

TOTAL = time.time()-T0
print(f"\n✅ 完成! ({TOTAL:.0f}s)")
print(f"  成功: {len(all_data)}只, 失败: {len(failed)}只")
print(f"  保存: {save_path}")
if failed:
    print(f"  失败股票: {failed[:10]}...")
