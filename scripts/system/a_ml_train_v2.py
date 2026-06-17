"""A股ML训练 — K线版 v2"""
import json, sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

t0 = time.time()
print('1. 加载K线数据...', flush=True)
with open('/home/hermes/.hermes/openclaw-project/data/a_hist_10y.json', 'r', encoding='utf-8') as f:
    hist = json.load(f)

codes = [c for c in hist.keys() if c.startswith(('60','00')) and len(hist[c].get('dates',[])) >= 500]
codes = codes[:500]
print(f'  主板+500天: {len(codes)}只', flush=True)

print('2. 计算特征...', flush=True)
all_X, all_y = [], []
sc = 0
for code in codes:
    h = hist[code]
    c = np.array(h['c'][::-1], dtype=np.float64)
    hi = np.array(h['h'][::-1], dtype=np.float64)
    lo = np.array(h['l'][::-1], dtype=np.float64)
    v = np.array(h['v'][::-1], dtype=np.float64)
    
    n = len(c)
    if n < 200: continue
    
    rows_x, rows_y = [], []
    for i in range(100, n-5):
        # 价格动量
        r1 = c[i]/c[i-1]-1 if c[i-1]>0 else 0
        r5 = c[i]/c[i-5]-1 if c[i-5]>0 else 0
        r20 = c[i]/c[i-20]-1 if c[i-20]>0 else 0
        
        # 均线
        m5 = np.mean(c[i-4:i+1]); m10 = np.mean(c[i-9:i+1])
        m20 = np.mean(c[i-19:i+1]); m60 = np.mean(c[i-59:i+1]) if i>=59 else m20
        
        # 偏离度
        d5 = c[i]/m5-1; d20 = c[i]/m20-1; d60 = c[i]/m60-1
        
        # 排列
        align = 1 if m5>m10>m20 else (-1 if m5<m10<m20 else 0)
        
        # RSI(14)
        chgs = np.diff(c[i-13:i+1])
        avg_g = np.mean(chgs[chgs>0]) if np.any(chgs>0) else 0
        avg_l = -np.mean(chgs[chgs<0]) if np.any(chgs<0) else 0.001
        rsi = 100 - 100/(1+avg_g/avg_l) if avg_l>1e-8 else 50
        
        # MACD (简化)
        def ema_s(arr, p):
            if len(arr) < p: return arr[-1]
            r = np.mean(arr[:p])
            a = 2/(p+1)
            for val in arr[p:]: r = val*a + r*(1-a)
            return r
        e12 = ema_s(c[max(0,i-25):i+1],12)
        e26 = ema_s(c[max(0,i-49):i+1],26)
        macd = e12-e26
        
        # 成交量
        vr = v[i]/np.mean(v[i-4:i+1]) if np.mean(v[i-4:i+1])>0 else 1
        h20 = np.max(hi[i-19:i+1]); l20 = np.min(lo[i-19:i+1])
        pos = (c[i]-l20)/(h20-l20) if h20>l20 else 0.5
        
        # 波动率
        v5 = np.std([c[j]/c[j-1]-1 for j in range(i-4,i+1)])
        v20 = np.std([c[j]/c[j-1]-1 for j in range(i-19,i+1)])
        
        # Y
        if i+5 < n:
            ret_f = c[i+5]/c[i]-1
            if c[i]>0 and c[i+5]>0:
                y = 1.0 if ret_f>0.02 else 0.0
                rows_x.append([r1,r5,r20,m5/m20,e12/e26,d5,d20,d60,align,v5,v20,rsi,macd,vr,pos,c[i]/m60])
                rows_y.append(y)
    
    if len(rows_x) > 10:
        all_X.append(np.array(rows_x, dtype=np.float32))
        all_y.append(np.array(rows_y, dtype=np.float32))
        sc += 1
        if sc % 100 == 0: print(f'  {sc}: {len(rows_x)}行', flush=True)

X = np.vstack(all_X); y = np.concatenate(all_y)
print(f'\n特征: {X.shape}, 正例率: {y.mean():.2%}', flush=True)

print('3. 训练...', flush=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
m = xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.05,
    subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1)
m.fit(X_train,y_train)
p = m.predict_proba(X_test)[:,1]
print(f'  AUC: {roc_auc_score(y_test,p):.4f}  Acc: {accuracy_score(y_test,m.predict(X_test)):.4f}', flush=True)

# 校准
cal = CalibratedClassifierCV(m,method='sigmoid',cv='prefit')
cal.fit(X_test,y_test)
cp = cal.predict_proba(X_test)[:,1]
print(f'  校准后prob: {cp.mean():.4f}', flush=True)

# 保存
os.makedirs('/home/hermes/.hermes/openclaw-project/data/models',exist_ok=True)
m.save_model('/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v1.json')
print(f'✅ 模型: a_xgb_tech_v1.json', flush=True)

fn = ['r1','r5','r20','m5/m20','m12/m26','d5','d20','d60','align','v5','v20','rsi','macd','vr','pos','c/m60']
imp = m.feature_importances_
for n,i in sorted(zip(fn,imp),key=lambda x:-x[1]):
    print(f'  {n}: {i:.4f}', flush=True)

print(f'总耗时: {(time.time()-t0)/60:.1f}分', flush=True)
