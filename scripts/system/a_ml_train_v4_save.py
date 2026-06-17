"""
a_ml_train_v4_save.py — 用v4参数扫描的最优参数重新训练+保存模型
最优: 300树 d10 lr0.10 AUC=0.7154
性价比: 200树 d10 lr0.10 AUC=0.7152 (选这个)
"""
import json, os, time, concurrent.futures, pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import traceback, sys

# 直接用标准输出，不用lambda
t0 = time.time()

FEAT_NAMES = [
    'r1','r5','r20','d5','d20','d60','align',
    'v5','v20','rsi','macd','vr','pos','c_div_m60',
    'vp_signal','vr20','vol_ratio','price_norm'
]

def log(msg):
    s = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(s, flush=True)

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

    rows_x, rows_ret, rows_date = [], [], []
    
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
            pn = c[i]/m60 - 1
            
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
                rows_x.append([r1,r5,r20,d5,d20,d60,align,v5,v20,rsi,macd,vr,
                               pos,pn,vp_s,vr20,vol_ratio,pn])
                rows_ret.append(ret_f)
                rows_date.append(dates[i+5])
        except Exception:
            continue
    
    if len(rows_x) > 10:
        return (np.array(rows_x, dtype=np.float32), rets := np.array(rows_ret, dtype=np.float64), dates := list(rows_date))
    return None


log('加载数据...')
with open('/home/hermes/.hermes/openclaw-project/data/a_hist_10y.json', 'r', encoding='utf-8') as f:
    hist = json.load(f)

codes = [c for c in hist if c.startswith(('60','00')) and len(hist[c].get('dates',[])) >= 750]
log(f'  {len(codes)}只主板>=3年')

log('并行特征计算...')
all_X, all_rets, all_dates = [], [], []
batch_size = 200

for batch_start in range(0, len(codes), batch_size):
    batch_codes = codes[batch_start:batch_start+batch_size]
    batch_X, batch_r, batch_d = [], [], []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = {ex.submit(compute_stock, code, hist[code]): code for code in batch_codes}
        for fut in concurrent.futures.as_completed(futures):
            try:
                result = fut.result(timeout=60)
                if result is not None:
                    batch_X.append(result[0])
                    batch_r.extend(result[1].tolist())
                    batch_d.extend(result[2])
            except Exception:
                pass
    
    if batch_X:
        all_X.append(np.vstack(batch_X))
        all_rets.extend(batch_r)
        all_dates.extend(batch_d)
        log(f'  批次完成: +{sum(len(x) for x in batch_X)}行')

del hist

X = np.vstack(all_X).astype(np.float32)
rets = np.array(all_rets, dtype=np.float64)
dates_arr = np.array(all_dates)
log(f'特征完成: {X.shape}')

# 双重门控标签
unique_dates = sorted(set(dates_arr))
y = np.zeros(len(rets), dtype=np.float64)

for d in unique_dates:
    mask = dates_arr == d
    if mask.sum() < 10:
        continue
    d_rets = rets[mask]
    rank_thresh = np.percentile(d_rets, 85)
    for idx in np.where(mask)[0]:
        if rets[idx] >= rank_thresh and rets[idx] > 0.05:
            y[idx] = 1.0

pos_rate = y.mean()
log(f'双重门控标签: 正例率={pos_rate:.4f} ({int(y.sum())}/{len(y)})')

# 训练 (最优参数: 200树 d10 lr0.10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log('训练最终模型...')
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=10, learning_rate=0.1,
    subsample=0.7, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
    tree_method='hist', device='cuda',
    eval_metric='auc'
)
model.fit(X_train, y_train, verbose=False)

p = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, p)
acc = accuracy_score(y_test, (p >= 0.5).astype(int))
log(f'原始模型: AUC={auc:.4f} Acc={acc:.4f}')

# 校准
log('校准...')
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(p.reshape(-1, 1), y_test)
cp = log_reg.predict_proba(p.reshape(-1, 1))[:,1]
c_auc = roc_auc_score(y_test, cp)
c_acc = accuracy_score(y_test, (cp >= 0.5).astype(int))
log(f'校准后: AUC={c_auc:.4f} Acc={c_acc:.4f}')

# 校准质量
log('校准质量:')
for lo in np.arange(0, 1, 0.1):
    hi = lo + 0.1
    mask = (cp >= lo) & (cp < hi)
    if mask.sum() > 10:
        pred = cp[mask].mean()
        actual = y_test[mask].mean()
        diff = abs(pred-actual)
        flag = ' OK' if diff < 0.03 else (' WARN' if diff < 0.05 else ' BAD')
        log(f'  [{lo:.1f},{hi:.1f}) n={mask.sum()}  pred={pred:.3f} actual={actual:.3f} diff={diff:.3f}{flag}')

# 保存
log('保存模型...')
model_path = '/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v3.json'
cal_path = '/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v3_cal.pkl'
meta_path = '/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v3_meta.json'

model.save_model(model_path)
with open(cal_path, 'wb') as f:
    pickle.dump(log_reg, f)

meta = {
    'model': 'a_xgb_tech_v3',
    'date': time.strftime('%Y-%m-%d'),
    'features': FEAT_NAMES,
    'n_features': len(FEAT_NAMES),
    'target': 'rank_p85_and_ret_gte_5pct',
    'target_desc': 'dual gate: cross-section top 15% AND 5d return >5%',
    'params': {
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.7,
        'colsample_bytree': 0.8
    },
    'performance': {
        'acc': round(float(c_acc), 4),
        'auc': round(float(c_auc), 4),
        'uncal_auc': round(float(auc), 4)
    },
    'n_train': len(X_train),
    'n_test': len(X_test),
    'pos_rate': float(pos_rate),
    'features_importance': {
        name: round(float(imp), 4)
        for name, imp in zip(FEAT_NAMES, model.feature_importances_)
    }
}

with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

elapsed = time.time() - t0
log(f'\n完成! 耗时: {elapsed/60:.1f}分钟')
log(f'模型: {model_path}')
log(f'校准器: {cal_path}')
log(f'元数据: {meta_path}')
log(f'AUC: {c_auc:.4f}')
