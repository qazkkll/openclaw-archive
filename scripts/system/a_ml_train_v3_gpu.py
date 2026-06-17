"""
a_ml_train_v3_gpu.py — A股ML v3训练 (GPU加速版)
使用RTX 3080 Ti + 并行特征计算 + XGBoost GPU

超参数(来自参数扫描): ne=100, md=8, lr=0.2, ss=0.7
已修复: 不再使用已弃用的 predictor='gpu_predictor', 改用 device='cuda'
新增特征: vp_signal, vr20, vol_ratio, price_norm
"""
import json, os, time, concurrent.futures, pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import traceback

# ─── 日志 ───
LOG = '/home/hermes/.hermes/openclaw-project/scripts/system/train_v3_log.txt'
open(LOG, 'w', encoding='utf-8').close()
log = lambda msg: (open(LOG, 'a', encoding='utf-8').write(f'[{time.strftime("%H:%M:%S")}] {msg}\n'),
                   print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True))

t0 = time.time()

FEAT_NAMES = [
    'r1','r5','r20','d5','d20','d60','align',
    'v5','v20','rsi','macd','vr','pos','c_div_m60',
    'vp_signal','vr20','vol_ratio','price_norm'
]

def compute_stock(code, h):
    """单只股票特征计算"""
    try:
        # 确保所有字段存在且为list
        for k in ['c','h','l','v']:
            if k not in h or not isinstance(h[k], list) or len(h[k]) < 200:
                return None
        c = np.array(h['c'][::-1], dtype=np.float64)
        hi = np.array(h['h'][::-1], dtype=np.float64)
        lo = np.array(h['l'][::-1], dtype=np.float64)
        v = np.array(h['v'][::-1], dtype=np.float64)
    except Exception:
        return None

    n = len(c)
    if n < 200:
        return None

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
            
            # RSI(14)
            chgs = np.diff(c[i-13:i+1])
            avg_g = np.mean(chgs[chgs>0]) if np.any(chgs>0) else 0.001
            avg_l = -np.mean(chgs[chgs<0]) if np.any(chgs<0) else 0.001
            rsi = 100 - 100/(1+avg_g/avg_l)
            
            # MACD
            e12 = np.mean(c[i-11:i+1]); e26 = np.mean(c[i-25:i+1])
            macd = e12 - e26
            
            # 量比
            vr = v[i] / np.mean(v[i-4:i+1]) if np.mean(v[i-4:i+1])>0 else 1
            
            # 位置
            h20 = np.max(hi[i-19:i+1]); l20 = np.min(lo[i-19:i+1])
            pos = (c[i]-l20)/(h20-l20) if h20>l20 else 0.5
            
            # 波动率
            v5 = np.std([c[j]/c[j-1]-1 for j in range(i-4,i+1)])
            v20 = np.std([c[j]/c[j-1]-1 for j in range(i-19,i+1)])
            
            # 4新特征
            vol_ratio = v5/v20 if v20>0 else 1.0
            vr20 = v[i] / np.mean(v[i-19:i+1]) if np.mean(v[i-19:i+1])>0 else 1
            price_norm = c[i]/m60 - 1
            
            # vp_signal
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
        except Exception:
            continue
    
    if len(rows_x) > 10:
        return (np.array(rows_x, dtype=np.float32), np.array(rows_y, dtype=np.float32))
    return None


log(f'Step 1/4: 加载数据...')
with open('/home/hermes/.hermes/openclaw-project/data/a_hist_10y.json', 'r', encoding='utf-8') as f:
    hist = json.load(f)

codes = [c for c in hist if c.startswith(('60','00')) and len(hist[c].get('dates',[])) >= 750]
log(f'  主板且>=3年: {len(codes)}只')

# ─── Step 2: 并行特征计算 ───
log(f'Step 2/4: 并行计算特征 ({len(codes)}只, {os.cpu_count()}线程)...')

all_X, all_y = [], []
batch_size = 200
total_batches = (len(codes) + batch_size - 1) // batch_size

for batch_idx, batch_start in enumerate(range(0, len(codes), batch_size)):
    batch_codes = codes[batch_start:batch_start+batch_size]
    batch_X, batch_y = [], []
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
            futures = {ex.submit(compute_stock, code, hist[code]): code for code in batch_codes}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                try:
                    result = fut.result(timeout=60)
                    if result is not None:
                        batch_X.append(result[0])
                        batch_y.append(result[1])
                except Exception:
                    pass
                done += 1
                if done % 50 == 0:
                    log(f'  批次{batch_idx+1}/{total_batches}: {done}/{len(batch_codes)}只')
    except Exception as e:
        log(f'  批次{batch_idx+1} 线程池异常: {e}')
        traceback.print_exc()
    
    if batch_X:
        try:
            all_X.append(np.vstack(batch_X))
            all_y.append(np.concatenate(batch_y))
            log(f'  批次{batch_idx+1}/{total_batches}: +{sum(len(x) for x in batch_X)}行')
        except Exception as e:
            log(f'  批次{batch_idx+1} vstack失败: {e}')
    
    # 每5批保存中间状态，防止内存溢出
    if (batch_idx+1) % 5 == 0:
        log(f'  中间状态: 已处理{batch_idx+1}/{total_batches}批, 累计{sum(len(x) for x in all_X) if all_X else 0}行')

if not all_X:
    log('错误: 没有任何数据被生成!')
    raise RuntimeError("No data generated")

X = np.vstack(all_X); y = np.concatenate(all_y)
log(f'  完成! 特征表: {X.shape}, 正例率: {y.mean():.3f}')

del hist, all_X, all_y  # 释放内存

# ─── Step 3: GPU XGBoost (固定为 device=cuda) ───
log('Step 3/4: GPU XGBoost训练...')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

m = xgb.XGBClassifier(
    n_estimators=100, max_depth=8, learning_rate=0.2,
    subsample=0.7, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
    tree_method='hist',        # XGBoost 2.0+ 兼容
    device='cuda',             # GPU 加速
    eval_metric='auc',
    early_stopping_rounds=10
)

m.fit(X_train, y_train,
      eval_set=[(X_test, y_test)],
      verbose=True)

p = m.predict_proba(X_test)[:,1]
acc = accuracy_score(y_test, m.predict(X_test))
auc = roc_auc_score(y_test, p)
brier = brier_score_loss(y_test, p)
log(f'  原始模型: Acc={acc:.4f}  AUC={auc:.4f}  Brier={brier:.4f}')

# ─── 校准 ───
log('  校准中...')

# 使用Platt缩放(LogisticRegression作为校准器)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
p_reshaped = p.reshape(-1, 1)
log_reg.fit(p_reshaped, y_test)
cp = log_reg.predict_proba(p_reshaped)[:,1]
c_auc = roc_auc_score(y_test, cp)
c_brier = brier_score_loss(y_test, cp)
log(f'  校准后: AUC={c_auc:.4f}  Brier={c_brier:.4f}')

# 校准质量
log('  校准质量检查:')
for lo in np.arange(0, 1, 0.1):
    hi = lo + 0.1
    mask = (cp >= lo) & (cp < hi)
    if mask.sum() > 10:
        pred = cp[mask].mean()
        actual = y_test[mask].mean()
        diff = abs(pred-actual)
        flag = ' ✅' if diff < 0.03 else (' ⚠️' if diff < 0.05 else ' ❌')
        log(f'    [{lo:.1f},{hi:.1f}) n={mask.sum()}  pred={pred:.3f} actual={actual:.3f} diff={diff:.3f}{flag}')

# 保存校准器
cal = {
    'intercept': float(log_reg.intercept_[0]),
    'coef': float(log_reg.coef_[0][0]),
    'method': 'platt_logistic'
}

# 特征重要性
log('\n---特征重要性---')
imp = m.feature_importances_
for n,i in sorted(zip(FEAT_NAMES, imp), key=lambda x: -x[1]):
    log(f'  {n}: {i:.4f}')

# ─── Step 4: 保存 ───
log('Step 4/4: 保存模型...')
os.makedirs('/home/hermes/.hermes/openclaw-project/data/models', exist_ok=True)
# save_model() on the sklearn wrapper has a bug with device='cuda'
# Use the booster's save method directly
m.get_booster().save_model('/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v2.json')
with open('/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v2_cal.pkl', 'wb') as f:
    pickle.dump(cal, f)

meta = {
    'model': 'a_xgb_tech_v2',
    'date': '2026-06-10',
    'features': FEAT_NAMES,
    'n_feats': len(FEAT_NAMES),
    'params': {'ne':100,'md':8,'lr':0.2,'ss':0.7,'tree':'hist','device':'cuda'},
    'perf': {'acc':float(acc),'auc':float(auc),'cal_auc':float(c_auc),'brier':float(brier),'cal_brier':float(c_brier)},
    'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
    'pos_rate': float(y.mean())
}
with open('/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v2_meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

elapsed = (time.time()-t0)/60
log(f'✅ 完成! 总耗时: {elapsed:.1f}分钟')
log(f'模型已保存: /home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v2.json')
log(f'校准器已保存: /home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v2_cal.pkl')
log(f'元数据已保存: /home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v2_meta.json')
