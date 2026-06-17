#!/usr/bin/env python3
"""
美股ML v15 — 5天预测 + 5档分类
基于预计算v2.1的label_5d_5class
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
print("═══ v15 5天预测+5档分类 ═══")

df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v2.parquet")
feats = ['price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
         'ret1','ret5','ret20','ret60',
         'macd','macd_signal','macd_hist','vol_ratio','ma_bias20',
         'vol5','trend_accel']
print(f"  数据: {len(df):,}行, {df['sym'].nunique()}只")
print(f"  特征: {len(feats)}个")

df = df.dropna(subset=feats + ['label_5d_5class']).copy()
X = df[feats].values
y = df['label_5d_5class'].values

classes = np.array([0,1,2,3,4])
wts = compute_class_weight('balanced', classes=classes, y=y)
wd = {i:w for i,w in enumerate(wts)}
print(f"  5档权重: {[f'{w:.2f}' for w in wts]}")

# Walk-Forward
wfs = [('WF1',0,0.60,0.60,0.75,0.75,0.85),
       ('WF2',0.15,0.65,0.65,0.80,0.80,0.90),
       ('WF3',0.30,0.70,0.70,0.85,0.85,1.00)]

n = len(df)
all_p_up5, all_act = [], []
all_p_dn5 = []  # 跌>5%概率

for name,ts,te,vs,ve,tst,tste in wfs:
    Xtr, ytr = X[int(ts*n):int(te*n)], y[int(ts*n):int(te*n)]
    Xva, yva = X[int(vs*n):int(ve*n)], y[int(vs*n):int(ve*n)]
    Xte = X[int(tst*n):int(tste*n)]
    yte_act = y[int(tst*n):int(tste*n)]
    
    sw = np.array([wd[yi] for yi in ytr])
    decay = np.linspace(0.3, 1.0, len(sw))
    sw *= decay
    
    m = xgb.XGBClassifier(n_estimators=800, max_depth=5, lr=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', early_stopping_rounds=30,
        random_state=42, n_jobs=-1, verbosity=0, num_class=5, device='cuda')
    m.fit(Xtr, ytr, sample_weight=sw,
          eval_set=[(Xva, yva)], verbose=0)
    
    proba = m.predict_proba(Xte)
    proba_up5 = proba[:, 4]  # 涨>5%
    proba_dn5 = proba[:, 0]  # 跌>5%
    all_p_up5.extend(proba_up5.tolist())
    all_p_dn5.extend(proba_dn5.tolist())
    all_act.extend(yte_act.tolist())
    
    # 涨>5% top10%
    top10 = proba_up5 >= np.percentile(proba_up5, 90)
    n_top = top10.sum()
    r = yte_act[top10]
    # 实际涨>5%的比率
    hit5 = (r == 4).mean()
    # 夏普（用5天收益值）
    # 需要label_5d_pct值... 重新从df找
    _end=int(tste*n)
    _start=int(tst*n)
    actual_pct = df['label_5d_pct'].values[_start:_end]
    r_pct = actual_pct[top10]
    sp = r_pct.mean() / r_pct.std() * math.sqrt(252/5) if r_pct.std()>0 else 0
    wr = (r_pct > 0).mean()
    print(f"  {name}: Top10%涨>5%命中={hit5:.1%} 夏普={sp:.3f} 胜率={wr:.1%}", flush=True)

# 合成
preds=np.array(all_p_up5); acts=np.array(all_act)
te=preds>=np.percentile(preds,90)
# 需要label_5d_pct数值回算... 但合成时用df整体切片算
r=df['label_5d_pct'].values[-len(acts):][te]
wf_s=r.mean()/r.std()*math.sqrt(252/5) if r.std()>0 else 0
wf_w=(r>0).mean()
print(f"\n  合成: 夏普={wf_s:.3f}, 胜率={wf_w:.1%}")

# 全量
print("\n[全量]")
train_end=int(n*0.85)
sw_f=[wd[yi] for yi in y[:train_end]]
decay=np.linspace(0.3,1.0,train_end)
sw_f*=decay

final=xgb.XGBClassifier(n_estimators=800, max_depth=5, lr=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0, num_class=5, device='cuda')
final.fit(X[:train_end], y[:train_end], sample_weight=sw_f,
          eval_set=[(X[train_end:], y[train_end:])], verbose=0)

yp=final.predict_proba(X[train_end:])
pu5=yp[:,4]; pd5=yp[:,0]; ta=df['label_5d_pct'].values[train_end:]
t10=pu5>=np.percentile(pu5,90); r=ta[t10]
sf=r.mean()/r.std()*math.sqrt(252/5) if r.std()>0 else 0
wr=(r>0).mean()
hit5=(df['label_5d_5class'].values[train_end:][t10]==4).mean()
print(f"  测试: 夏普={sf:.3f} 胜率={wr:.1%} 涨>5%命中={hit5:.1%}")

# 今日预测
latest=df.dropna(subset=feats).drop_duplicates(subset='sym',keep='last')
Xl=latest[feats].values; ypl=final.predict_proba(Xl)

results=[]
for i,(_,row) in enumerate(latest.iterrows()):
    p=ypl[i]
    results.append({'sym':row['sym'],'price':float(row['price']),
                    'prob_up5':float(p[4]),'prob_dn5':float(p[0])})
results.sort(key=lambda x:-x['prob_up5'])

print(f"\n{'═'*60}")
print(f"{'排名':>3} {'代码':>8} {'价格':>8} {'涨>5%':>7} {'跌>5%':>7}")
print(f"{'─'*60}")
for i,r in enumerate(results[:20]):
    print(f"{i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {r['prob_up5']*100:>6.1f}% {r['prob_dn5']*100:>6.1f}%")

final.save_model(_paths.US_MODEL_DIR + "/us_xgb_v15.json")
out={
    'timestamp':str(__import__('datetime').datetime.now()),
    'model':'us_xgb_v15','features':feats,
    'wf_sharpe':round(wf_s,4),'wf_win_rate':round(wf_w,4),
    'test_sharpe':round(sf,4),'test_win_rate':round(wr,4),
    'test_hit_up5':round(hit5,4),
    'predictions':[{'rank':i+1,**r} for i,r in enumerate(results[:50])],
}
with open(_paths.US_MODEL_DIR+"/us_xgb_v15_prediction.json",'w') as f:
    json.dump(out,f,indent=2,ensure_ascii=False)

print(f"\n✅ v15 完成! ({time.time()-T0:.0f}s)")
