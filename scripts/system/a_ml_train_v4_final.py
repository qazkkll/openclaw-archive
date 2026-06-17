"""
a_ml_train_v4_final.py — 用v4参数扫描最优参数重训+保存
最优: 200树 d10 lr0.10 AUC=0.7152 (性价比)
文件日志(UTF8)避开编码问题
"""
import json, os, time, concurrent.futures, pickle, sys
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

LOG = '/home/hermes/.hermes/openclaw-project/scripts/system/train_v4_final_log.txt'
open(LOG, 'w', encoding='utf-8').close()
def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        # write without emoji
        clean = msg.replace('\u2705','[OK]').replace('\u26a0\ufe0f','[WARN]').replace('\u274c','[FAIL]')
        f.write(f'[{time.strftime("%H:%M:%S")}] {clean}\n')
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

t0 = time.time()

FEAT_NAMES = [
    'r1','r5','r20','m5_div_m20','d5','d20','d60','align',
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
                rows_x.append([r1,r5,r20,m5_div_m20:=r5/r20 if abs(r20)>1e-6 else 0,
                               d5,d20,d60,align,v5,v20,rsi,macd,vr,pos,
                               c_div_m60:=c[i]/m60-1, vp_s, vr20, vol_ratio, pn])
                rows_ret.append(ret_f)
                rows_date.append(dates[i+5])
        except Exception:
            continue
    if len(rows_x) > 10:
        return (np.array(rows_x, dtype=np.float32), np.array(rows_ret, dtype=np.float64), rows_date)
    return None

log('Step 1: Load data...')
with open('/home/hermes/.hermes/openclaw-project/data/a_hist_10y.json', 'r', encoding='utf-8') as f:
    hist = json.load(f)
codes = [c for c in hist if c.startswith(('60','00')) and len(hist[c].get('dates',[])) >= 750]
log(f'  {len(codes)} stocks')

log('Step 2: Feature computation...')
all_X, all_ret, all_date = [], [], []
batch_size = 200
for bs in range(0, len(codes), batch_size):
    bc = codes[bs:bs+batch_size]
    bx, br, bd = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        ff = {ex.submit(compute_stock, c, hist[c]): c for c in bc}
        for f in concurrent.futures.as_completed(ff):
            try:
                r = f.result(timeout=60)
                if r is not None:
                    bx.append(r[0]); br.extend(r[1].tolist()); bd.extend(r[2])
            except Exception:
                pass
    if bx:
        all_X.append(np.vstack(bx))
        all_ret.extend(br); all_date.extend(bd)
    log(f'  batch {bs//batch_size+1}/{(len(codes)+batch_size-1)//batch_size}: +{sum(len(x) for x in bx)} rows')
del hist

Xarr = np.vstack(all_X).astype(np.float32)
rets = np.array(all_ret, dtype=np.float64)
darr = np.array(all_date)
log(f'  Feature matrix: {Xarr.shape}')
del all_X, all_ret, all_date

log('Step 3: Dual-gate labels...')
ud = sorted(set(darr))
y = np.zeros(len(rets), dtype=np.float64)
for d in ud:
    mask = darr == d
    if mask.sum() < 10:
        continue
    dr = rets[mask]
    th = np.percentile(dr, 85)
    for idx in np.where(mask)[0]:
        if rets[idx] >= th and rets[idx] > 0.05:
            y[idx] = 1.0
log(f'  Pos rate: {y.mean():.4f} ({int(y.sum())}/{len(y)})')
del rets, darr, ud

log('Step 4: Train final model (200trees d10 lr0.1)...')
X_tr, X_te, y_tr, y_te = train_test_split(Xarr, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=10, learning_rate=0.1,
    subsample=0.7, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, tree_method='hist', device='cuda'
)
model.fit(X_tr, y_tr, verbose=False)
p = model.predict_proba(X_te)[:,1]
auc_raw = roc_auc_score(y_te, p)
log(f'  Raw AUC: {auc_raw:.4f}')

log('  Calibrating...')
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(p.reshape(-1,1), y_te)
cp = lr.predict_proba(p.reshape(-1,1))[:,1]
c_auc = roc_auc_score(y_te, cp)
c_acc = accuracy_score(y_te, (cp >= 0.5).astype(int))
log(f'  Cal AUC: {c_auc:.4f} Acc: {c_acc:.4f}')

log('  Calibration check:')
for lo in np.arange(0, 1, 0.1):
    hi = lo + 0.1
    mask = (cp >= lo) & (cp < hi)
    if mask.sum() > 10:
        pred = cp[mask].mean()
        actual = y_te[mask].mean()
        diff = abs(pred-actual)
        flag = '[OK]' if diff < 0.03 else ('[WARN]' if diff < 0.05 else '[BAD]')
        log(f'    [{lo:.1f},{hi:.1f}) n={mask.sum()} pred={pred:.3f} actual={actual:.3f} diff={diff:.3f} {flag}')

log('Saving model...')
mp = '/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v3.json'
cp_path = '/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v3_cal.pkl'
metap = '/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v3_meta.json'

model.save_model(mp)
with open(cp_path, 'wb') as f:
    pickle.dump(lr, f)

meta = {
    'model': 'a_xgb_tech_v3',
    'date': time.strftime('%Y-%m-%d'),
    'features': FEAT_NAMES,
    'n_features': len(FEAT_NAMES),
    'target': 'dual_gate_p85_ret5pct',
    'params': {'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1,
               'subsample': 0.7, 'colsample_bytree': 0.8},
    'performance': {'acc': round(float(c_acc), 4), 'auc': round(float(c_auc), 4)},
    'n_train': len(X_tr), 'n_test': len(X_te), 'pos_rate': float(y.sum()/len(y)),
    'features_importance': {n: round(float(imp), 4) for n, imp in zip(FEAT_NAMES, model.feature_importances_)}
}
with open(metap, 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

elapsed = time.time() - t0
log(f'Done! {elapsed/60:.1f}min')
log(f'Model: {mp}')
log(f'Cal: {cp_path}')
log(f'Meta: {metap}')
log(f'AUC: {c_auc:.4f}')
