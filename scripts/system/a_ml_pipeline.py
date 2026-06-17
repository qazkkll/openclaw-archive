"""
a_ml_pipeline.py — A股ML训练管道
用法: python a_ml_pipeline.py

三步:
1. 加载K线数据 + 计算特征
2. 训练XGBoost
3. 保存模型

日志输出到 /home/hermes/.hermes/openclaw-project/scripts/system/train_log.txt
"""
import json, sys, os, time, logging
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# 日志
LOG = '/home/hermes/.hermes/openclaw-project/scripts/system/train_log.txt'
logging.basicConfig(filename=LOG, level=logging.INFO, 
    format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = lambda msg: (logging.info(msg), print(msg, flush=True))

t0 = time.time()

# ─── Step 1: 加载K线 ───
log('Step 1/4: 加载K线数据...')
with open('/home/hermes/.hermes/openclaw-project/data/a_hist_10y.json', 'r', encoding='utf-8') as f:
    hist = json.load(f)
log(f'  加载完成: {len(hist)}只股票')

codes = [c for c in hist if c.startswith(('60','00')) and len(hist[c].get('dates',[])) >= 500]
codes = codes[:500]  # 先500只测试
log(f'  训练池: {len(codes)}只股票')

# ─── Step 2: 计算特征 ───
log('Step 2/4: 计算技术面特征...')
all_X, all_y = [], []
sc = 0

for code in codes:
    h = hist[code]
    try:
        c = np.array(h['c'][::-1], dtype=np.float64)
        hi = np.array(h['h'][::-1], dtype=np.float64)
        lo = np.array(h['l'][::-1], dtype=np.float64)
        v = np.array(h['v'][::-1], dtype=np.float64)
    except:
        continue
    
    n = len(c)
    if n < 200: continue
    
    rows_x, rows_y = [], []
    for i in range(100, n-5):
        r1 = c[i]/c[i-1]-1 if c[i-1]>0 else 0
        r5 = c[i]/c[i-5]-1 if i>=5 and c[i-5]>0 else 0
        r20 = c[i]/c[i-20]-1 if i>=20 and c[i-20]>0 else 0
        
        m5 = np.mean(c[i-4:i+1]); m10 = np.mean(c[i-9:i+1])
        m20 = np.mean(c[i-19:i+1]); m60 = np.mean(c[i-59:i+1]) if i>=59 else m20
        
        d5 = c[i]/m5-1; d20 = c[i]/m20-1; d60 = c[i]/m60-1
        align = 1 if m5>m10>m20 else (-1 if m5<m10<m20 else 0)
        
        # RSI
        chgs = np.diff(c[i-13:i+1])
        avg_g = np.mean(chgs[chgs>0]) if np.any(chgs>0) else 0.001
        avg_l = -np.mean(chgs[chgs<0]) if np.any(chgs<0) else 0.001
        rsi = 100 - 100/(1+avg_g/avg_l)
        
        # MACD
        e12 = np.mean(c[i-11:i+1])  # 简化: 12日均线
        e26 = np.mean(c[i-25:i+1])  # 简化: 26日均线
        macd = e12 - e26
        
        vr = v[i] / np.mean(v[i-4:i+1]) if np.mean(v[i-4:i+1])>0 else 1
        h20 = np.max(hi[i-19:i+1]); l20 = np.min(lo[i-19:i+1])
        pos = (c[i]-l20)/(h20-l20) if h20>l20 else 0.5
        
        v5 = np.std([c[j]/c[j-1]-1 for j in range(i-4,i+1)]); v20 = np.std([c[j]/c[j-1]-1 for j in range(i-19,i+1)])
        
        if i+5 < n:
            ret_f = c[i+5]/c[i]-1
            if c[i]>0 and c[i+5]>0:
                y = 1.0 if ret_f > 0.02 else 0.0
                rows_x.append([r1,r5,r20,m5/m20,d5,d20,d60,align,v5,v20,rsi,macd,vr,pos,c[i]/m60])
                rows_y.append(y)
    
    if len(rows_x) > 10:
        all_X.append(np.array(rows_x, dtype=np.float32))
        all_y.append(np.array(rows_y, dtype=np.float32))
        sc += 1
        if sc % 100 == 0: log(f'  {sc}只完成')

X = np.vstack(all_X); y = np.concatenate(all_y)
log(f'  特征: {X.shape}, 正例率: {y.mean():.2%}')

# ─── Step 3: 训练 ───
log('Step 3/4: 训练XGBoost...')
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

m = xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.05,
    subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1)
m.fit(X_train, y_train)

p = m.predict_proba(X_test)[:,1]
acc = accuracy_score(y_test, m.predict(X_test))
auc = roc_auc_score(y_test, p)
log(f'  准确率: {acc:.4f}  AUC: {auc:.4f}')

# 校准
cal = CalibratedClassifierCV(m, method='sigmoid', cv='prefit')
cal.fit(X_test, y_test)
cp = cal.predict_proba(X_test)[:,1]
log(f'  校准后概率: {cp.mean():.4f}')

# ─── Step 4: 保存 ───
log('Step 4/4: 保存模型...')
os.makedirs('/home/hermes/.hermes/openclaw-project/data/models', exist_ok=True)
m.save_model('/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v1.json')
log(f'  模型: /home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v1.json')

fn = ['r1','r5','r20','m5/m20','d5','d20','d60','align','v5','v20','rsi','macd','vr','pos','c/m60']
imp = m.feature_importances_
for n,i in sorted(zip(fn,imp), key=lambda x:-x[1]):
    log(f'   {n}: {i:.4f}')

log(f'✅ 完成! 总耗时: {(time.time()-t0)/60:.1f}分钟')
log(f'日志: {LOG}')
