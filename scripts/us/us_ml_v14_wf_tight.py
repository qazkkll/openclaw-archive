#!/usr/bin/env python3
"""
美股ML v14 — 重新设计Walk-Forward，让WF3专用验证窗口
WF3一直是最弱环（0.654），可能是因为验证窗口不够"新"
v14: 每个WF窗口的验证集紧贴测试集之前
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import json, time, math, pandas as pd, numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
import _paths

T0 = time.time()
print("═══ v14 WF专用验证贴近期 ═══")

df = pd.read_parquet(_paths.US_ML_FEATS)

base = ['ma5','ma10','ma20','ma60','rsi14','vol20','p52',
        'ret1','ret5','ret20','ret60',
        'macd','macd_signal','macd_hist','vol_ratio','ma_bias20',
        'price','volume']
df['vol20_ma20'] = df['vol20'].fillna(0) - (df['volume']/df['volume'].rolling(20).mean()).fillna(0)
df['vol20_ma60'] = df['vol20'].fillna(0) - (df['volume']/df['volume'].rolling(60).mean()).fillna(0)
df['ret60_vol20'] = df['ret60'].fillna(0)*df['vol20'].fillna(0)
df['trend_accel']=df['ret20'].fillna(0)-df['ret5'].fillna(0)
df['macd_accel']=df['macd_hist'].fillna(0)-df['macd_hist'].shift(1).fillna(0)
df['vol_price_div']=df['ret1'].fillna(0)*df['vol_ratio'].fillna(0)
df['vol_surge']=(df['vol20'].fillna(0)-df['vol20'].shift(5).fillna(0))/df['vol20'].shift(5).fillna(df['vol20']).replace(0,np.nan).fillna(1)
df['p52_momentum']=df['ret1'].fillna(0)*(df['p52'].fillna(50)-50)

feats = base + ['vol20_ma20','vol20_ma60','ret60_vol20',
                'trend_accel','macd_accel','vol_price_div','vol_surge','p52_momentum']
print(f"  特征: {len(feats)}列")

df=df.dropna(subset=['label_pct']+feats).copy()
def b3(p):
    if p>2:return 2
    if p>-2:return 1
    return 0
df['lb']=df['label_pct'].apply(b3)

X=df[feats].values; yb=df['lb'].values; ya=df['label_pct'].values
classes=np.array([0,1,2])
wts=compute_class_weight('balanced',classes=classes,y=yb)
wd={i:w for i,w in enumerate(wts)}

# v14 Walk-Forward: 三个窗口，验证集紧贴测试集
wfs=[
    # 训练:0-60%, 验证:60-75%, 测试:75-85%
    ('WF1',0,0.60,0.60,0.75,0.75,0.85),
    # 训练:15-65%, 验证:65-80%, 测试:80-90%
    ('WF2',0.15,0.65,0.65,0.80,0.80,0.90),
    # 训练:30-70%, 验证:70-85%, 测试:85-100%   <- WF3重点
    ('WF3',0.30,0.70,0.70,0.85,0.85,1.00),
]

# v14强化: 每个窗口用不同的early_stopping轮数
# WF3需要更短的停止（最新数据噪声大）
es_map = {'WF1':30, 'WF2':25, 'WF3':15}

all_p, all_a = [], []
for name,ts,te,vs,ve,tst,tste in wfs:
    n=len(df)
    idx_tr=slice(int(ts*n),int(te*n))
    idx_va=slice(int(vs*n),int(ve*n))
    idx_te=slice(int(tst*n),int(tste*n))
    
    Xtr,ytr=X[idx_tr],yb[idx_tr]
    Xva,yva=X[idx_va],yb[idx_va]
    Xte,yte_act=X[idx_te],ya[idx_te]
    
    sw=np.array([wd[yi] for yi in ytr])
    decay=np.linspace(0.3,1.0,len(sw))
    sw*=decay
    
    es=es_map[name]
    m=xgb.XGBClassifier(n_estimators=500, max_depth=5, lr=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', early_stopping_rounds=es,
        random_state=42, n_jobs=-1, verbosity=0, num_class=3, device='cuda')
    m.fit(Xtr,ytr,sample_weight=sw,eval_set=[(Xva,yva)],verbose=0)
    
    best_n=m.best_iteration+1 if hasattr(m,'best_iteration') else 500
    proba=m.predict_proba(Xte)
    all_p.extend(proba[:,2].tolist())
    all_a.extend(yte_act.tolist())
    
    top10=proba[:,2]>=np.percentile(proba[:,2],90)
    r=yte_act[top10]
    sp=r.mean()/r.std()*math.sqrt(252) if r.std()>0 else 0
    wr=(r>0).mean()
    print(f"  {name}: best_tree={best_n}, 夏普={sp:.3f}, 胜率={wr:.1%}", flush=True)

preds,acts=np.array(all_p),np.array(all_a)
te=preds>=np.percentile(preds,90)
r=acts[te]
wfs=r.mean()/r.std()*math.sqrt(252) if r.std()>0 else 0
wf_wr=(r>0).mean()
print(f"\n  合成: 夏普={wfs:.3f}, 胜率={wf_wr:.1%}")

# 全量
print("\n[全量]")
train_end=int(len(df)*0.85)
sw_full=np.array([wd[yi] for yi in yb[:train_end]])
decay=np.linspace(0.3,1.0,train_end)
sw_full*=decay
final=xgb.XGBClassifier(n_estimators=500, max_depth=5, lr=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0, num_class=3, device='cuda')
final.fit(X[:train_end],yb[:train_end],sample_weight=sw_full,
          eval_set=[(X[train_end:],yb[train_end:])],verbose=0)

yp=final.predict_proba(X[train_end:])[:,2]; ta=ya[train_end:]
t10=yp>=np.percentile(yp,90); r=ta[t10]
sf=r.mean()/r.std()*math.sqrt(252) if r.std()>0 else 0
wr=(r>0).mean()
print(f"  测试: 夏普={sf:.3f}, 胜率={wr:.1%}")

# 今日预测
latest=df.dropna(subset=feats).drop_duplicates(subset='sym',keep='last')
Xl=latest[feats].values; ypl=final.predict_proba(Xl)

results=[]
for i,(_,row) in enumerate(latest.iterrows()):
    probs=ypl[i].tolist()
    results.append({'sym':row['sym'],'price':float(row['price']),
                    'prob_up':probs[2]})
results.sort(key=lambda x:-x['prob_up'])

print(f"\n{'═'*50}")
for i,r in enumerate(results[:20]):
    print(f"  {i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {r['prob_up']*100:>5.1f}%")

# 保存
final.save_model(_paths.US_MODEL_DIR + "/us_xgb_v14.json")
out={
    'timestamp':str(__import__('datetime').datetime.now()),
    'model':'us_xgb_v14',
    'features':feats,'es_map':es_map,
    'wf_sharpe':round(wfs,4),'wf_win_rate':round(wf_wr,4),
    'test_sharpe':round(sf,4),'test_win_rate':round(wr,4),
}
with open(_paths.US_MODEL_DIR+"/us_xgb_v14_prediction.json",'w') as f:
    json.dump(out,f,indent=2)

print(f"\n✅ v14 完成! ({time.time()-T0:.0f}s)")
