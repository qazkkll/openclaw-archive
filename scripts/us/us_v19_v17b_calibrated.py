"""绿箭v17b — Platt校准版，cv=None，让模型输出校准后概率"""
import sys, os, math, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
import _paths

T0 = time.time()
print("═══ 绿箭v17b: Platt校准 ═══")

df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v2.parquet")

# ETF
with open(_paths.ML_DIR+"/us_sector_etf.json") as f: etf_data = json.load(f)
s2e = {'Technology':'XLK','Financial Services':'XLF','Financial':'XLF','Energy':'XLE',
       'Healthcare':'XLV','Industrials':'XLI','Consumer Defensive':'XLP',
       'Consumer Cyclical':'XLY','Utilities':'XLU','Basic Materials':'XLB',
       'Materials':'XLB','Real Estate':'XLRE','Communication Services':'XLC','Semiconductor':'SMH'}
def get_er(s):
    e=s2e.get(s)
    return etf_data[e]['ret5'] if e and e in etf_data else etf_data['SPY']['ret5']

df['sector_etf_ret5'] = df['sector'].apply(get_er)
for k in ['SPY','QQQ','IWM']: df[f'{k.lower()}_ret5'] = etf_data[k]['ret5']

base = ['price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
        'ret1','ret5','ret20','ret60','macd','macd_signal','macd_hist',
        'vol_ratio','ma_bias20','vol5','trend_accel',
        'short_ratio','short_pct','market_cap','pe_trailing','pe_forward','beta']
df['sc'] = df['sector'].astype('category').cat.codes.astype(int)
all_feats = base + ['sector_etf_ret5','spy_ret5','qqq_ret5','iwm_ret5','sc']

df = df.dropna(subset=all_feats+['label_5d_5class']).copy()
X=df[all_feats].values; y=df['label_5d_5class'].values; n=len(df)
classes=np.array([0,1,2,3,4])
wts=compute_class_weight('balanced',classes=classes,y=y)
wd={i:w for i,w in enumerate(wts)}

# WF
for name,ts,te,vs,ve,tst,tste in [
    ('WF1',0,0.60,0.60,0.75,0.75,0.85),
    ('WF2',0.15,0.65,0.65,0.80,0.80,0.90),
    ('WF3',0.30,0.70,0.70,0.85,0.85,1.00)]:
    sw=np.array([wd[yi] for yi in y[int(ts*n):int(te*n)]])
    sw*=np.linspace(0.3,1.0,len(sw))
    m=xgb.XGBClassifier(n_estimators=500,max_depth=5,lr=0.1,subsample=0.8,
        colsample_bytree=0.8,random_state=42,n_jobs=-1,verbosity=0,num_class=5, device='cuda')
    m.fit(X[int(ts*n):int(te*n)],y[int(ts*n):int(te*n)],sample_weight=sw,verbose=0)
    cc=CalibratedClassifierCV(m,method='sigmoid',cv=None)
    cc.fit(X[int(vs*n):int(ve*n)],y[int(vs*n):int(ve*n)])
    pu5=cc.predict_proba(X[int(tst*n):int(tste*n)])[:,4]
    t10=pu5>=np.percentile(pu5,90)
    ap=df['label_5d_pct'].values[int(tst*n):int(tste*n)][t10]
    sp=ap.mean()/ap.std()*math.sqrt(252/5) if ap.std()>0 else 0
    print(f"  {name}: 夏普={sp:.3f}",flush=True)

# 全量+校准
sw_f=np.array([wd[yi] for yi in y[:int(n*0.85)]])*np.linspace(0.3,1.0,int(n*0.85))
fb=xgb.XGBClassifier(n_estimators=500,max_depth=5,lr=0.1,subsample=0.8,
    colsample_bytree=0.8,random_state=42,n_jobs=-1,verbosity=0,num_class=5, device='cuda')
fb.fit(X[:int(n*0.85)],y[:int(n*0.85)],sample_weight=sw_f,verbose=0)
fc=CalibratedClassifierCV(fb,method='sigmoid',cv=None)
fc.fit(X[int(n*0.85):],y[int(n*0.85):])

yp=fc.predict_proba(X[int(n*0.85):])
pu5=yp[:,4]; ta=df['label_5d_pct'].values[int(n*0.85):]
t10=pu5>=np.percentile(pu5,90); r=ta[t10]
sf=r.mean()/r.std()*math.sqrt(252/5) if r.std()>0 else 0
hit5=(df['label_5d_5class'].values[int(n*0.85):][t10]==4).mean()
print(f"  测试: 夏普={sf:.3f} 涨>5%命中={hit5:.1%}")

# 校准检查
for thr in [0.2,0.3,0.4]:
    mc=pu5>=thr
    if mc.sum()>=10:
        actual=(df['label_5d_5class'].values[int(n*0.85):][mc]==4).mean()
        predicted=pu5[mc].mean()
        print(f"   >{thr:.0%}(n={mc.sum()}): 预测={predicted:.1%} 实际={actual:.1%}")

# 今日预测
latest=df.dropna(subset=all_feats).drop_duplicates(subset='sym',keep='last')
Xl=latest[all_feats].values
ypl=fc.predict_proba(Xl)
preds=[{'sym':row['sym'],'price':float(row['price']),
        'up5':float(ypl[i][4]),'dn5':float(ypl[i][0])}
       for i,(_,row) in enumerate(latest.iterrows())]
preds.sort(key=lambda x:-x['up5'])

print(f"\n{'═'*70}")
print(f"{'排名':>3} {'代码':>8} {'价格':>8} {'涨>5%(校准)':>10} {'跌>5%':>8}")
print(f"{'─'*70}")
for i,r in enumerate(preds[:20]):
    print(f"{i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {r['up5']*100:>9.1f}% {r['dn5']*100:>7.1f}%")

import joblib
joblib.dump(fc, _paths.US_MODEL_DIR+"/greenshaft_v17b.pkl")
fb.save_model(_paths.US_MODEL_DIR+"/greenshaft_v17b_base.json")

print(f"\n✅ v17b完成! ({time.time()-T0:.0f}s)")
