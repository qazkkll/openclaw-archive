"""
绿箭v19 — scale_pos_weight + 完整Isotonic校准
根因：涨>5%样本仅18%，模型不确定度高
方案：对涨>5%类别加权重 (1:5)，拉高涨>5%样本的影响
"""
import sys, os, math, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.isotonic import IsotonicRegression
import _paths

T0 = time.time()
print("═══ 绿箭v19: scale_pos_weight校准 ═══")

df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v2.parquet")
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
df['sc'] = df['sector'].astype('category').cat.codes.astype(int)

all_feats = ['price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
             'ret1','ret5','ret20','ret60','macd','macd_signal','macd_hist',
             'vol_ratio','ma_bias20','vol5','trend_accel',
             'short_ratio','short_pct','market_cap','pe_trailing','pe_forward','beta',
             'sector_etf_ret5','spy_ret5','qqq_ret5','iwm_ret5','sc']

df = df.dropna(subset=all_feats+['label_5d_5class']).copy()
X=df[all_feats].values; y5=df['label_5d_5class'].values
y_bin = (y5 == 4).astype(int)  # 涨>5%二值标签
n = len(df)

# 统计类别分布
for cl in range(5):
    cnt = (y5 == cl).sum()
    print(f"  类别{cl}: {cnt:,} ({cnt/n*100:.1f}%)")

# scale_pos_weight: 涨>5% vs 非涨>5%
pos = (y5 == 4).sum()
neg = n - pos
scale_pos = neg / pos
print(f"\n  scale_pos_weight = {scale_pos:.1f} (正类=涨>5%, {pos}/{neg})")

# 权重: 对涨>5%类别额外加权
# 类别用compute_class_weight + scale_pos
classes=np.array([0,1,2,3,4])
wts=compute_class_weight('balanced',classes=classes,y=y5)
# 对类别4(涨>5%)加倍
wts[4] *= 2.5
# 对类别0(跌>5%)也略微加
wts[0] *= 1.3
wd={i:w for i,w in enumerate(wts)}
print(f"  类别权重: {[f'{w:.2f}' for w in wts]}")

# Walk-Forward
wfs = [('WF1',0,0.60,0.60,0.75,0.75,0.85),
       ('WF2',0.15,0.65,0.65,0.80,0.80,0.90),
       ('WF3',0.30,0.70,0.70,0.85,0.85,1.00)]

print("\n[Walk-Forward]")
wf_results = []
for name,ts,te,vs,ve,tst,tste in wfs:
    sw = np.array([wd[yi] for yi in y5[int(ts*n):int(te*n)]])
    sw *= np.linspace(0.3, 1.0, len(sw))
    
    m = xgb.XGBClassifier(n_estimators=600, max_depth=5, lr=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', early_stopping_rounds=25,
        random_state=42, n_jobs=-1, verbosity=0, num_class=5, device='cuda')
    m.fit(X[int(ts*n):int(te*n)], y5[int(ts*n):int(te*n)],
          sample_weight=sw,
          eval_set=[(X[int(vs*n):int(ve*n)], y5[int(vs*n):int(ve*n)])],
          verbose=0)
    
    # Isotonic 校准
    raw_val = m.predict_proba(X[int(vs*n):int(ve*n)])[:, 4]
    actual_val = y_bin[int(vs*n):int(ve*n)]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_val, actual_val)
    
    # 测试
    Xte = X[int(tst*n):int(tste*n)]
    raw_test = m.predict_proba(Xte)[:, 4]
    cal_test = calibrator.predict(raw_test)
    actual_test = y_bin[int(tst*n):int(tste*n)]
    
    ece_before = np.mean(np.abs(raw_test - actual_test))
    ece_after = np.mean(np.abs(cal_test - actual_test))
    
    top10 = cal_test >= np.percentile(cal_test, 90)
    r = df['label_5d_pct'].values[int(tst*n):int(tste*n)][top10]
    sp = r.mean()/r.std()*math.sqrt(252/5) if r.std()>0 else 0
    hit5 = actual_test[top10].mean()
    
    # 分桶
    buckets = {}
    for lo, hi in [(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),
                   (0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]:
        mc = (cal_test >= lo) & (cal_test < hi)
        nm = mc.sum()
        if nm < 20: continue
        buckets[f'{lo:.0%}-{hi:.0%}'] = {
            'n': int(nm), 'predicted': round(cal_test[mc].mean(),4),
            'actual': round(actual_test[mc].mean(),4)
        }
    
    wf_results.append({'name':name,'sharpe':round(sp,3),'hit_rate':round(hit5,4),
                       'ece_before':round(ece_before,4),'ece_after':round(ece_after,4),
                       'buckets':buckets})
    
    print(f"  {name}: 夏普={sp:.3f} 命中={hit5:.1%} ECE={ece_before:.4f}→{ece_after:.4f}")
    # 显示分桶
    for bk, bv in sorted(buckets.items()):
        print(f"    {bk}: n={bv['n']:>6} 预测={bv['predicted']:.1%} 实际={bv['actual']:.1%} 偏差={bv['predicted']-bv['actual']:+.1%}")
    print(flush=True)

# 全量训练
print("[全量+校准]")
train_end = int(n * 0.85)
sw_f = np.array([wd[yi] for yi in y5[:train_end]])
sw_f *= np.linspace(0.3, 1.0, train_end)

final = xgb.XGBClassifier(n_estimators=600, max_depth=5, lr=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', early_stopping_rounds=25,
    random_state=42, n_jobs=-1, verbosity=0, num_class=5, device='cuda')
final.fit(X[:train_end], y5[:train_end], sample_weight=sw_f,
          eval_set=[(X[train_end:], y5[train_end:])], verbose=0)

# 校准
raw_test_full = final.predict_proba(X[train_end:])[:, 4]
actual_test_full = y_bin[train_end:]
X_val = X[int(n*0.7):train_end]
raw_val_full = final.predict_proba(X_val)[:, 4]
actual_val_full = y_bin[int(n*0.7):train_end]

cal_full = IsotonicRegression(out_of_bounds='clip')
cal_full.fit(raw_val_full, actual_val_full)
cal_test_full = cal_full.predict(raw_test_full)

ece_before = np.mean(np.abs(raw_test_full - actual_test_full))
ece_after = np.mean(np.abs(cal_test_full - actual_test_full))

top10 = cal_test_full >= np.percentile(cal_test_full, 90)
r = df['label_5d_pct'].values[train_end:][top10]
sf = r.mean()/r.std()*math.sqrt(252/5) if r.std()>0 else 0
hit5 = actual_test_full[top10].mean()

print(f"  测试: 夏普={sf:.3f} 命中={hit5:.1%}")
print(f"  ECE: {ece_before:.4f} → {ece_after:.4f}")

# 完整分桶检查
print(f"\n[校准后完整分桶]")
print(f"{'区间':>10} {'n':>8} {'预测均值':>10} {'实际占比':>10} {'偏差':>8} {'合格?':>6}")
print(f"{'─'*52}")
approved = True
buckets_final = {}
for lo, hi in [(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),
               (0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]:
    mc = (cal_test_full >= lo) & (cal_test_full < hi)
    nm = mc.sum()
    if nm < 20: continue
    actual = actual_test_full[mc].mean()
    pred = cal_test_full[mc].mean()
    bias = pred - actual
    ok = abs(bias) < 0.05
    if not ok and lo >= 0.2: approved = False
    buckets_final[f'{lo:.0%}-{hi:.0%}'] = {
        'n': int(nm), 'predicted': round(pred,4), 'actual': round(actual,4), 'bias': round(bias,4)
    }
    print(f"  {lo:.0%}-{hi:.0%}: n={nm:>7}  {pred:.1%}    {actual:.1%}     {bias:+.1%}   {'✅' if ok else '❌'}")

# 样本充足性检查
high_n = sum(v['n'] for k,v in buckets_final.items() 
              if any(th in k for th in ['50%','60%','70%','80%','90%']))
print(f"\n  高概率区间(>50%)样本总量: {high_n}")
if high_n < 500:
    print(f"  ❌ 不足500, 校准统计不可靠")
    approved = False

print(f"\n  {'✅ 校准验收通过' if approved else '❌ 校准未验收'}")

# 今日预测
latest = df.dropna(subset=all_feats).drop_duplicates(subset='sym', keep='last')
Xl = latest[all_feats].values
raw_prob = final.predict_proba(Xl)[:, 4]
cal_prob = cal_full.predict(raw_prob)
dn_prob = final.predict_proba(Xl)[:, 0]

preds = []
for i, (_, row) in enumerate(latest.iterrows()):
    preds.append({
        'sym': row['sym'], 'price': float(row['price']),
        'up5': float(cal_prob[i]), 'dn5': float(dn_prob[i]),
    })
preds.sort(key=lambda x: -x['up5'])

print(f"\n{'═'*70}")
print(f"{'排名':>3} {'代码':>8} {'价格':>8} {'涨>5%':>8} {'跌>5%':>8}")
print(f"{'─'*70}")
for i, r in enumerate(preds[:20]):
    print(f"{i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {r['up5']*100:>7.1f}% {r['dn5']*100:>7.1f}%")

# 保存
final.save_model(_paths.US_MODEL_DIR + "/greenshaft_v19_base.json")
import joblib
joblib.dump(cal_full, _paths.US_MODEL_DIR + "/greenshaft_v19_calib.pkl")

out = {
    'timestamp': str(__import__('datetime').datetime.now()),
    'model': 'greenshaft_v19',
    'features': all_feats,
    'class_weights': [round(w,2) for w in wts],
    'wf_results': wf_results,
    'test': {'sharpe': round(sf,4), 'hit_up5': round(hit5,4),
             'ece_before': round(ece_before,4), 'ece_after': round(ece_after,4)},
    'calibration_accepted': approved,
    'buckets': buckets_final,
    'predictions': [{'rank':i+1,'sym':r['sym'],'price':float(r['price']),
                      'up5':float(r['up5']),'dn5':float(r['dn5'])}
                     for i,r in enumerate(preds[:50])],
}
with open(_paths.US_MODEL_DIR + "/greenshaft_v19_prediction.json", 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)

TOTAL = time.time() - T0
print(f"\n✅ 绿箭v19 完成! ({TOTAL:.0f}s)")
print(f"  校准验收: {'通过' if approved else '❌ 待优化'}")
