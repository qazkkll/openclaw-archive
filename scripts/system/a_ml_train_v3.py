"""
a_ml_train_v3.py — A股ML v3训练
基于参数扫描结论:
  - 超参数: ne=100, md=8, lr=0.2, ss=0.7
  - 特征: 14原特征 + vp_signal, vr20, vol_ratio, price_norm = 18-19特征
  - 预期AUC: 0.655~0.660
  - 校准: Platt缩放 (sigmoid)
"""
import json, sys, os, time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from scipy.special import expit

t0 = time.time()
LOG_FILE = '/home/hermes/.hermes/openclaw-project/scripts/system/train_v3_log.txt'
# 使用专属日志，末尾加_YYYYMMDD避免与其他进程冲突
import datetime
LOG_FILE = f'/home/hermes/.hermes/openclaw-project/scripts/system/train_v3_{datetime.date.today().strftime("%Y%m%d")}.log'
log = lambda msg: (print(f'[{time.time()-t0:.0f}s] {msg}', flush=True),
                    open(LOG_FILE,'a',encoding='utf-8').write(f'[{time.strftime("%H:%M:%S")}] {msg}\n'))

# 清空日志
open(LOG_FILE,'w').close()

# ─── Step 1: 加载 ───
log('Step 1/4: 加载K线数据...')
with open('/home/hermes/.hermes/openclaw-project/data/a_hist_10y.json', 'r', encoding='utf-8') as f:
    hist = json.load(f)
log(f'  加载: {len(hist)}只股票')

# 只取主板(60/00开头), 且至少750天(3年)
codes = [c for c in hist 
         if c.startswith(('60','00')) 
         and len(hist[c].get('dates',[])) >= 750]
log(f'  主板且>=3年: {len(codes)}只')

# 用全部数据，不限制500只（参数扫描已验证全量可跑）
NPART = len(codes)

# ─── Step 2: 特征工程 ───
log(f'Step 2/4: 计算特征 ({NPART}只)...')

FEAT_NAMES = [
    'r1','r5','r20','m5_div_m20',     # 收益率
    'd5','d20','d60','align',          # 均线偏移+排列
    'v5','v20','rsi','macd','vr','pos',# 波动率+RSI+MACD+量比+位置
    'c_div_m60',                       # 价格/60日线
    'vp_signal',                       # 量价配合
    'vr20',                            # 20日均量比
    'vol_ratio',                       # 波动率比(短/长)
    'price_norm',                      # 价格归一化
]

all_X, all_y = [], []
skipped = 0

for idx, code in enumerate(codes):
    if (idx+1) % 200 == 0:
        log(f'  进度: {idx+1}/{NPART} (skipped {skipped})')
    
    try:
        h = hist[code]
        c = np.array(h['c'][::-1], dtype=np.float64)
        hi = np.array(h['h'][::-1], dtype=np.float64)
        lo = np.array(h['l'][::-1], dtype=np.float64)
        v = np.array(h['v'][::-1], dtype=np.float64)
    except Exception as e:
        skipped += 1
        if skipped <= 5:
            log(f'  skip {code} load: {e}')
        continue
    
    n = len(c)
    if n < 200:
        skipped += 1
        continue
    
    rows_x, rows_y = [], []
    stock_ok = True
    
    for i in range(100, n-5):
        try:
            # 基础收益率
            r1 = c[i]/c[i-1]-1 if c[i-1]>0 else 0
            r5 = c[i]/c[i-5]-1 if i>=5 and c[i-5]>0 else 0
            r20 = c[i]/c[i-20]-1 if i>=20 and c[i-20]>0 else 0
            
            # 均线
            m5 = np.mean(c[i-4:i+1])
            m10 = np.mean(c[i-9:i+1])
            m20 = np.mean(c[i-19:i+1])
            m60 = np.mean(c[i-59:i+1]) if i>=59 else m20
            
            d5 = c[i]/m5-1
            d20 = c[i]/m20-1
            d60 = c[i]/m60-1
            align = 1 if m5>m10>m20 else (-1 if m5<m10<m20 else 0)
            
            # RSI(14)
            chgs = np.diff(c[i-13:i+1])
            avg_g = np.mean(chgs[chgs>0]) if np.any(chgs>0) else 0.001
            avg_l = -np.mean(chgs[chgs<0]) if np.any(chgs<0) else 0.001
            rsi = 100 - 100/(1+avg_g/avg_l)
            
            # MACD(简化: 12/26均线差)
            e12 = np.mean(c[i-11:i+1])
            e26 = np.mean(c[i-25:i+1])
            macd = e12 - e26
            
            # 量比(5日)
            vr = v[i] / np.mean(v[i-4:i+1]) if np.mean(v[i-4:i+1])>0 else 1
            
            # 20日位置
            h20 = np.max(hi[i-19:i+1]); l20 = np.min(lo[i-19:i+1])
            pos = (c[i]-l20)/(h20-l20) if h20>l20 else 0.5
            
            # 波动率(5日/20日)
            v5 = np.std([c[j]/c[j-1]-1 for j in range(i-4,i+1)])
            v20 = np.std([c[j]/c[j-1]-1 for j in range(i-19,i+1)])
            
            # ─── 4个新增特征 ───
            # vp_signal: 量价配合
            if v[i] > np.mean(v[i-4:i+1]) and c[i] > np.mean(c[i-4:i+1]):
                vp_s = 1.0  # 放量上涨
            elif v[i] < np.mean(v[i-4:i+1]) and c[i] < np.mean(c[i-4:i+1]):
                vp_s = -1.0  # 缩量下跌
            elif v[i] > np.mean(v[i-4:i+1]) and c[i] < np.mean(c[i-4:i+1]):
                vp_s = -0.5  # 放量下跌(坏)
            else:
                vp_s = 0.5  # 缩量上涨
            
            # vr20: 20日均量比
            vr20 = v[i] / np.mean(v[i-19:i+1]) if np.mean(v[i-19:i+1])>0 else 1
            
            # vol_ratio: 短/长波动率比
            vol_ratio = v5 / v20 if v20 > 0 else 1.0
            
            # price_norm: 价格/60日线归一化
            price_norm = c[i] / m60 - 1
            
            if i+5 < n:
                ret_f = c[i+5]/c[i]-1
                if c[i]>0 and c[i+5]>0:
                    y = 1.0 if ret_f > 0.02 else 0.0
                    feat = [r1,r5,r20,d5,d20,d60,align,v5,v20,rsi,macd,vr,pos,price_norm,
                            vp_s, vr20, vol_ratio, price_norm]
                    rows_x.append(feat)
                    rows_y.append(y)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                log(f'  skip {code}@{i}: {e}')
            stock_ok = False
            break
    
    if stock_ok and len(rows_x) > 10:
        all_X.append(np.array(rows_x, dtype=np.float32))
        all_y.append(np.array(rows_y, dtype=np.float32))
    elif not stock_ok and len(rows_x) <= 10:
        pass  # already counted as skipped
    elif len(rows_x) <= 10:
        skipped += 1

X = np.vstack(all_X); y = np.concatenate(all_y)
log(f'  特征表: {X.shape}, 正例率: {y.mean():.3f}')
log(f'  特征数: {len(FEAT_NAMES)}')

# ─── Step 3: 训练 ───
log('Step 3/4: 训练XGBoost (参数扫描最优配置)...')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

m = xgb.XGBClassifier(
    n_estimators=100, max_depth=8, learning_rate=0.2,
    subsample=0.7, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
    tree_method='hist'  # 加速
)

m.fit(X_train, y_train)

p = m.predict_proba(X_test)[:,1]
acc = accuracy_score(y_test, m.predict(X_test))
auc = roc_auc_score(y_test, p)
log(f'  原始模型: Acc={acc:.4f}  AUC={auc:.4f}')

# ─── 校准 ───
log('Step 3b: Platt校准...')
# 使用sklearn 1.8兼容方法: 手动Platt缩放 (LogisticRegression on raw scores)
raw_scores = m.predict_proba(X_train)[:,1].reshape(-1,1)
platt = LogisticRegression()
platt.fit(raw_scores, y_train)

val_scores = m.predict_proba(X_val)[:,1].reshape(-1,1)
cp = platt.predict_proba(val_scores)[:,1]
c_auc = roc_auc_score(y_val, cp)

# 校准质量检查
for lo in np.arange(0, 1, 0.1):
    hi = lo + 0.1
    mask = (cp >= lo) & (cp < hi)
    if mask.sum() > 10:
        pred = cp[mask].mean()
        actual = y_val[mask].mean()
        log(f'  prob [{lo:.1f},{hi:.1f}) -> pred={pred:.3f}, actual={actual:.3f}, diff={abs(pred-actual):.3f}')

log(f'  校准后AUC: {c_auc:.4f}')

# ─── 特征重要性 ───
log('\n--- 特征重要性 ---')
imp = m.feature_importances_
for n, i in sorted(zip(FEAT_NAMES, imp), key=lambda x: -x[1]):
    log(f'  {n}: {i:.4f}')

# ─── Step 4: 保存 ───
log('Step 4/4: 保存模型...')
os.makedirs('/home/hermes/.hermes/openclaw-project/data/models', exist_ok=True)
model_path = '/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v2.json'
m.get_booster().save_model(model_path)
log(f'  模型: {model_path}')

# 保存校准器(Platt)
import pickle
cal_path = '/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v2_cal.pkl'
with open(cal_path, 'wb') as f:
    pickle.dump(platt, f)
log(f'  校准器: {cal_path}')

# 保存特征名
meta = {
    'model': 'a_xgb_tech_v2',
    'date': '2026-06-10',
    'features': FEAT_NAMES,
    'n_features': len(FEAT_NAMES),
    'params': {'n_estimators':100, 'max_depth':8, 'learning_rate':0.2, 'subsample':0.7},
    'performance': {'acc': float(acc), 'auc': float(auc), 'cal_auc': float(c_auc)},
    'n_train': len(y_train),
    'n_val': len(y_val),
    'n_test': len(y_test),
    'pos_rate': float(y.mean())
}
meta_path = '/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v2_meta.json'
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)
log(f'  元数据: {meta_path}')

log(f'\n完成! 总耗时: {(time.time()-t0)/60:.1f}分钟')
log(f'日志: {LOG_FILE}')
print(f'\nv3训练完成！AUC={auc:.4f}, Cal AUC={c_auc:.4f}')
