"""
a_ml_train_v4a_feats.py — 第1步：特征计算 + 双重门控标签
输出: /home/hermes/.hermes/openclaw-project/scripts/system/v4_feats_label.parquet
"""
import json, os, time, concurrent.futures, datetime
import numpy as np
import pandas as pd

LOG_FILE = f'/home/hermes/.hermes/openclaw-project/scripts/system/train_v4a_feats_{datetime.date.today().strftime("%Y%m%d")}.log'
log = lambda msg: (print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True),
                    open(LOG_FILE,'a',encoding='utf-8').write(f'[{time.strftime("%H:%M:%S")}] {msg}\n'))
open(LOG_FILE,'w',encoding='utf-8').close()

t0 = time.time()

FEAT_NAMES = [
    'r1','r5','r20','m5_div_m20',
    'd5','d20','d60','align',
    'v5','v20','rsi','macd','vr','pos','c_div_m60',
    'vp_signal','vr20','vol_ratio','price_norm'
]

def compute_stock(code, h):
    try:
        for k in ['c','h','l','v','dates']:
            if k not in h or not isinstance(h[k], list) or len(h[k]) < 200:
                return None
        c = np.array(h['c'][::-1], dtype=np.float64)
        hi = np.array(h['h'][::-1], dtype=np.float64)
        lo = np.array(h['l'][::-1], dtype=np.float64)
        v = np.array(h['v'][::-1], dtype=np.float64)
        dates = h['dates'][::-1]
    except Exception:
        return None

    n = len(c)
    if n < 200:
        return None

    rows = []
    for i in range(100, n-5):
        try:
            r1 = c[i]/c[i-1]-1 if c[i-1]>0 else 0
            r5 = c[i]/c[i-5]-1 if i>=5 and c[i-5]>0 else 0
            r20 = c[i]/c[i-20]-1 if i>=20 and c[i-20]>0 else 0
            m5 = np.mean(c[i-4:i+1]); m10 = np.mean(c[i-9:i+1])
            m20 = np.mean(c[i-19:i+1]); m60 = np.mean(c[i-59:i+1]) if i>=59 else m20
            d5 = c[i]/m5-1; d20 = c[i]/m20-1; d60 = c[i]/m60-1
            align = 1 if m5>m10>m20 else (-1 if m5<m10<m20 else 0)
            m5_div_m20 = m5/m20 - 1
            chgs = np.diff(c[i-13:i+1])
            avg_g = np.mean(chgs[chgs>0]) if np.any(chgs>0) else 0.001
            avg_l = -np.mean(chgs[chgs<0]) if np.any(chgs<0) else 0.001
            rsi = 100-100/(1+avg_g/avg_l)
            e12 = np.mean(c[i-11:i+1]); e26 = np.mean(c[i-25:i+1])
            macd = e12 - e26
            vr = v[i]/np.mean(v[i-4:i+1]) if np.mean(v[i-4:i+1])>0 else 1
            v5 = np.std([c[j]/c[j-1]-1 for j in range(i-4,i+1)])
            v20 = np.std([c[j]/c[j-1]-1 for j in range(i-19,i+1)])
            h20 = np.max(hi[i-19:i+1]); l20 = np.min(lo[i-19:i+1])
            pos = (c[i]-l20)/(h20-l20) if h20>l20 else 0.5
            vol_ratio = v5/v20 if v20>0 else 1.0
            vr20 = v[i]/np.mean(v[i-19:i+1]) if np.mean(v[i-19:i+1])>0 else 1
            price_norm = c[i]/m60 - 1
            c_div_m60 = c[i]/m60 - 1
            vm5 = np.mean(v[i-4:i+1]); pm5 = np.mean(c[i-4:i+1])
            if v[i] > vm5 and c[i] > pm5: vp_s = 1.0
            elif v[i] < vm5 and c[i] < pm5: vp_s = -1.0
            elif v[i] > vm5 and c[i] < pm5: vp_s = -0.5
            else: vp_s = 0.5
            ret_f = c[i+5]/c[i]-1
            if c[i] > 0 and c[i+5] > 0 and not (np.isnan(ret_f) or np.isinf(ret_f)):
                rows.append({
                    'code': code, 'date': dates[i],
                    'r1': r1, 'r5': r5, 'r20': r20,
                    'm5_div_m20': m5_div_m20,
                    'd5': d5, 'd20': d20, 'd60': d60,
                    'align': align, 'v5': v5, 'v20': v20,
                    'rsi': rsi, 'macd': macd, 'vr': vr,
                    'pos': pos, 'c_div_m60': c_div_m60,
                    'vp_signal': vp_s, 'vr20': vr20,
                    'vol_ratio': vol_ratio, 'price_norm': price_norm,
                    'fwd_ret_5d': ret_f
                })
        except Exception:
            continue
    return rows if len(rows) > 10 else None


# ─── Step 1: 加载 ───
log('Step 1/3: 加载K线数据...')
with open('/home/hermes/.hermes/openclaw-project/data/a_hist_10y.json', 'r', encoding='utf-8') as f:
    hist = json.load(f)

codes = [c for c in hist if c.startswith(('60','00')) and len(hist[c].get('dates',[])) >= 750]
log(f'  主板且>=3年: {len(codes)}只')

# ─── Step 2: 并行特征计算 ───
log(f'Step 2/3: 并行计算特征 ({len(codes)}只)...')

all_rows = []
batch_size = 200
total = (len(codes) + batch_size - 1) // batch_size

for bi in range(0, len(codes), batch_size):
    batch = codes[bi:bi+batch_size]
    batch_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futs = {ex.submit(compute_stock, code, hist[code]): code for code in batch}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            try:
                r = fut.result(timeout=60)
                if r:
                    batch_rows.extend(r)
            except Exception:
                pass
            done += 1
            if done % 100 == 0:
                log(f'  批次{bi//batch_size+1}/{total}: {done}/{len(batch)}只')
    if batch_rows:
        all_rows.extend(batch_rows)
        log(f'  批次{bi//batch_size+1}/{total}: +{len(batch_rows)}行')

log(f'  完成! 总行数: {len(all_rows)}')

del hist  # 释放大内存

# ─── Step 2b: 构建双重门控标签 ───
log('Step 2b/3: 构建双重门控标签...')

df = pd.DataFrame(all_rows)
del all_rows

# 按日期做横截面排名
df['pct_rank'] = df.groupby('date')['fwd_ret_5d'].rank(pct=True, ascending=False)
df['label'] = ((df['pct_rank'] < 0.15) & (df['fwd_ret_5d'] > 0.05)).astype(np.float32)

log(f'  正例率: {df["label"].mean():.4f} ({int(df["label"].sum())}/{len(df)})')
log(f'  特征列: 19, 总行数: {len(df)}')
log(f'  日期数: {df["date"].nunique()}, 日均样本: {len(df)/df["date"].nunique():.0f}')

# 保存
out = '/home/hermes/.hermes/openclaw-project/scripts/system/v4_feats_label.parquet'
df.to_parquet(out, index=False)
log(f'  保存: {out} ({(os.path.getsize(out)/1024**3):.1f}GB)')

log(f'✅ 完成! 耗时: {(time.time()-t0)/60:.1f}分钟')
