"""Debug run - just process a few batches"""
import json, os, sys, time, concurrent.futures, traceback
import numpy as np
import xgboost as xgb

LOG = '/home/hermes/.hermes/openclaw-project/scripts/system/train_v3_debug.txt'
open(LOG, 'w', encoding='utf-8').close()
log = lambda msg: (open(LOG, 'a', encoding='utf-8').write(f'[{time.strftime("%H:%M:%S")}] {msg}\n'), print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True))

log('Loading...')
with open('/home/hermes/.hermes/openclaw-project/data/a_hist_10y.json', 'r', encoding='utf-8') as f:
    hist = json.load(f)
codes = [c for c in hist if c.startswith(('60','00')) and len(hist[c].get('dates',[])) >= 750]
log(f'Codes: {len(codes)}')

def compute_stock(code, h):
    """单只股票特征计算"""
    try:
        c = np.array(h['c'][::-1], dtype=np.float64)
        hi = np.array(h['h'][::-1], dtype=np.float64)
        lo = np.array(h['l'][::-1], dtype=np.float64)
        v = np.array(h['v'][::-1], dtype=np.float64)
    except Exception as e:
        return None, f"array_error:{e}"

    n = len(c)
    if n < 200:
        return None, f"too_short:{n}"

    rows_x, rows_y = [], []
    for i in range(100, n-5):
        try:
            r1 = c[i]/c[i-1]-1 if c[i-1]>0 else 0
            r5 = c[i]/c[i-5]-1 if i>=5 and c[i-5]>0 else 0
            r20 = c[i]/c[i-20]-1 if i>=20 and c[i-20]>0 else 0
            
            m5 = np.mean(c[i-4:i+1]); m10 = np.mean(c[i-9:i+1])
            m20 = np.mean(c[i-19:i+1]); m60 = np.mean(c[i-59:i+1]) if i>=59 else m20
            
            d5 = c[i]/m5-1; d20 = c[i]/m20-1; d60 = c[i]/m60-1
            align = 1 if m5>m10>m20 else (-1 if m5<m10<m20 else 0)
            
            chgs = np.diff(c[i-13:i+1])
            avg_g = np.mean(chgs[chgs>0]) if np.any(chgs>0) else 0.001
            avg_l = -np.mean(chgs[chgs<0]) if np.any(chgs<0) else 0.001
            rsi = 100 - 100/(1+avg_g/avg_l)
            
            e12 = np.mean(c[i-11:i+1]); e26 = np.mean(c[i-25:i+1])
            macd = e12 - e26
            
            vr = v[i] / np.mean(v[i-4:i+1]) if np.mean(v[i-4:i+1])>0 else 1
            
            h20 = np.max(hi[i-19:i+1]); l20 = np.min(lo[i-19:i+1])
            pos = (c[i]-l20)/(h20-l20) if h20>l20 else 0.5
            
            v5 = np.std([c[j]/c[j-1]-1 for j in range(i-4,i+1)])
            v20 = np.std([c[j]/c[j-1]-1 for j in range(i-19,i+1)])
            
            vol_ratio = v5/v20 if v20>0 else 1.0
            vr20 = v[i] / np.mean(v[i-19:i+1]) if np.mean(v[i-19:i+1])>0 else 1
            price_norm = c[i]/m60 - 1
            
            if v[i] > np.mean(v[i-4:i+1]) and c[i] > np.mean(c[i-4:i+1]):
                vp_s = 1.0
            elif v[i] < np.mean(v[i-4:i+1]) and c[i] < np.mean(c[i-4:i+1]):
                vp_s = -1.0
            elif v[i] > np.mean(v[i-4:i+1]) and c[i] < np.mean(c[i-4:i+1]):
                vp_s = -0.5
            else:
                vp_s = 0.5
            
            ret_f = c[i+5]/c[i]-1
            if c[i]>0 and c[i+5]>0:
                y_val = 1.0 if ret_f > 0.02 else 0.0
                feat = [r1,r5,r20,d5,d20,d60,align,v5,v20,rsi,macd,vr,pos,price_norm,
                        vp_s, vr20, vol_ratio, price_norm]
                rows_x.append(feat)
                rows_y.append(y_val)
        except Exception as e:
            continue
    
    if len(rows_x) > 10:
        return (np.array(rows_x, dtype=np.float32), np.array(rows_y, dtype=np.float32)), None
    return None, f"too_few_rows:{len(rows_x)}"

# Process batch 1
batch_codes = codes[:200]
log(f'Batch 1: {len(batch_codes)} codes')
batch_X, batch_y = [], []
errors = []

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
    futures = {ex.submit(compute_stock, code, hist[code]): code for code in batch_codes}
    done = 0
    for fut in concurrent.futures.as_completed(futures):
        done += 1
        try:
            result = fut.result(timeout=30)
            if result[0] is not None:
                batch_X.append(result[0][0])
                batch_y.append(result[0][1])
            if result[1] is not None:
                errors.append((futures[fut], result[1]))
        except Exception as e:
            errors.append((futures[fut], f"exception:{e}"))
        if done % 50 == 0:
            log(f'  {done}/{len(batch_codes)} done')

log(f'Batch 1 done: {len(batch_X)} stocks contributed, {len(errors)} errors')
if errors:
    log(f'Sample errors:')
    for c, e in errors[:5]:
        log(f'  {c}: {e}')

if batch_X:
    X = np.vstack(batch_X)
    y = np.concatenate(batch_y)
    log(f'X shape: {X.shape}, y shape: {y.shape}, pos_rate: {y.mean():.3f}')
    
    # Quick XGBoost test
    log('Testing XGBoost GPU...')
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    try:
        m = xgb.XGBClassifier(
            n_estimators=50, max_depth=6, learning_rate=0.2,
            subsample=0.7, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
            tree_method='hist', device='cuda', predictor='gpu_predictor',
            eval_metric='auc', early_stopping_rounds=10
        )
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
        log('XGBoost GPU training succeeded!')
    except Exception as e:
        log(f'XGBoost GPU failed: {e}')
        import traceback
        log(traceback.format_exc())
else:
    log('No data!')

log('Debug run complete')
