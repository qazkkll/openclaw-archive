"""
绿箭v16 �?基本面特征增强版训练
5�?档分�?+ short_ratio + sector + market_cap + pe + beta
"""
import sys, os, math, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
import _paths

T0 = time.time()
print("══�?绿箭v16: 基本面增�?══�?)

df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v2.parquet")

# 特征列：基础20 + 基本�?
base_feats = ['price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
              'ret1','ret5','ret20','ret60',
              'macd','macd_signal','macd_hist','vol_ratio','ma_bias20',
              'vol5','trend_accel']

# sector需要编码为数�?sectors = df['sector'].dropna().unique()
sector_map = {s:i for i,s in enumerate(sorted(sectors))}
df['sector_code'] = df['sector'].map(sector_map)

fund_feats = ['short_ratio','short_pct','market_cap','pe_trailing','pe_forward','beta','sector_code']
all_feats = base_feats + fund_feats

print(f"特征: {len(base_feats)}基础 + {len(fund_feats)}基本�?= {len(all_feats)}�?)

# 过滤空数�?df = df.dropna(subset=all_feats + ['label_5d_5class']).copy()
X = df[all_feats].values
y = df['label_5d_5class'].values
n = len(df)
print(f"有效数据: {n:,}�?)

# 5档分�?for cl in range(5):
    cnt = (y == cl).sum()
    print(f"  {cl}: {cnt:,} ({cnt/n*100:.1f}%)")

# 类别权重
classes = np.array([0,1,2,3,4])
wts = compute_class_weight('balanced', classes=classes, y=y)
wd = {i:w for i,w in enumerate(wts)}
print(f"权重: {[f'{w:.2f}' for w in wts]}")

# Walk-Forward
wfs = [('WF1',0,0.60,0.60,0.75,0.75,0.85),
       ('WF2',0.15,0.65,0.65,0.80,0.80,0.90),
       ('WF3',0.30,0.70,0.70,0.85,0.85,1.00)]

results = []
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
    pu5 = proba[:,4]  # �?5%
    pd5 = proba[:,0]  # �?5%
    
    # �?5%命中
    top10 = pu5 >= np.percentile(pu5, 90)
    n_top = top10.sum()
    hit5 = (yte_act[top10] == 4).mean() if n_top > 0 else 0
    
    # 实际return的夏�?    _end = int(tste*n); _start = int(tst*n)
    actual_pct = df['label_5d_pct'].values[_start:_end]
    r_pct = actual_pct[top10]
    sp = r_pct.mean()/r_pct.std()*math.sqrt(252/5) if r_pct.std()>0 else 0
    wr = (r_pct > 0).mean()
    
    results.append({'name':name, 'sharpe':sp, 'hit_rate':hit5, 
                    'win_rate':wr, 'trades':n_top})
    print(f"  {name}: 夏普={sp:.3f} �?5%命中={hit5:.1%} 胜率={wr:.1%}", flush=True)

# 合成
preds_list = []; acts_list = []
for name,ts,te,vs,ve,tst,tste in wfs:
    _s = int(tst*n); _e = int(tste*n)
    Xte = X[_s:_e]
    # 需要重新训练来获取合成预测... 简化：用最后一个WF的预�?    pass

# 全量训练
print("\n[全量训练]")
train_end = int(n * 0.85)
sw_f = np.array([wd[yi] for yi in y[:train_end]])
sw_f *= np.linspace(0.3, 1.0, train_end)

final = xgb.XGBClassifier(n_estimators=800, max_depth=5, lr=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0, num_class=5, device='cuda')
final.fit(X[:train_end], y[:train_end], sample_weight=sw_f,
          eval_set=[(X[train_end:], y[train_end:])], verbose=0)

yp = final.predict_proba(X[train_end:])
pu5 = yp[:,4]; pd5 = yp[:,0]
ta = df['label_5d_pct'].values[train_end:]
t10 = pu5 >= np.percentile(pu5, 90)
r = ta[t10]
sf = r.mean()/r.std()*math.sqrt(252/5) if r.std()>0 else 0
wr = (r > 0).mean()
hit5 = (df['label_5d_5class'].values[train_end:][t10] == 4).mean()
print(f"  测试夏普={sf:.3f} 胜率={wr:.1%} �?5%命中={hit5:.1%}")

# 今日预测
latest = df.dropna(subset=all_feats).drop_duplicates(subset='sym', keep='last')
Xl = latest[all_feats].values
ypl = final.predict_proba(Xl)

preds = []
for i, (_, row) in enumerate(latest.iterrows()):
    p = ypl[i]
    preds.append({
        'sym': row['sym'], 'price': float(row['price']),
        'prob_up5': float(p[4]), 'prob_dn5': float(p[0]),
        'prob_flat': float(p[2]),
        'prob_slight_up': float(p[3]),
        'prob_slight_dn': float(p[1]),
    })
preds.sort(key=lambda x: -x['prob_up5'])

print(f"\n{'�?*60}")
print(f"{'排名':>3} {'代码':>8} {'价格':>8} {'�?5%':>7} {'�?5%':>7} {'平�?%':>7}")
print(f"{'─'*60}")
for i, r in enumerate(preds[:20]):
    print(f"{i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {r['prob_up5']*100:>6.1f}% {r['prob_dn5']*100:>6.1f}% {r['prob_flat']*100:>6.1f}%")

# 保存
final.save_model(_paths.US_MODEL_DIR + "/greenshaft_v16.json")
out = {
    'timestamp': str(__import__('datetime').datetime.now()),
    'model': 'greenshaft_v16',
    'features': all_feats,
    'wf_results': results,
    'test_sharpe': round(sf, 4),
    'test_win_rate': round(wr, 4),
    'test_hit_up5': round(hit5, 4),
    'predictions': [{'rank': int(i+1), **{k: (float(v) if hasattr(v, 'item') else v) for k,v in r.items()}} for i, r in enumerate(preds[:50])],
}
with open(_paths.US_MODEL_DIR + "/greenshaft_v16_prediction.json", 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"\n�?绿箭v16 完成! ({time.time()-T0:.0f}s)")
print(f"  保存: {_paths.win(_paths.US_MODEL_DIR + '/greenshaft_v16.json')}")
