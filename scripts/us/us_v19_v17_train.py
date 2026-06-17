"""
绿箭v17 �?加行业ETF特征
把每只股票的sector→ETF映射，ETF的ret5作为特征
"""
import sys, json, os, math, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
import _paths

T0 = time.time()
print("══�?绿箭v17: 加行业ETF特征 ══�?)

# 读现有特�?df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v2.parquet")

# 行业ETF映射 (sector �?etf代码)
sector_to_etf = {
    'Technology': 'XLK',
    'Financial Services': 'XLF', 'Financial': 'XLF',
    'Energy': 'XLE',
    'Healthcare': 'XLV',
    'Industrials': 'XLI',
    'Consumer Defensive': 'XLP',
    'Consumer Cyclical': 'XLY',
    'Utilities': 'XLU',
    'Basic Materials': 'XLB', 'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC',
    'Semiconductor': 'SMH',  # 如sector直接是Semiconductor
}
broad_etfs = ['SPY','QQQ','IWM']

# 加载ETF数据
with open(_paths.ML_DIR + "/us_sector_etf.json", 'r') as f:
    etf_data = json.load(f)

# 创建行业ETF特征�?# 对于每只股票, sector→etf→取该etf的ret5
# 如果sector无匹�? 用SPY作为默认
def get_etf_ret5(sector):
    etf = sector_to_etf.get(sector)
    if etf and etf in etf_data:
        return etf_data[etf]['ret5']
    return None

df['sector_etf_ret5'] = df['sector'].apply(get_etf_ret5)
df['spy_ret5'] = etf_data['SPY']['ret5']
df['qqq_ret5'] = etf_data['QQQ']['ret5']
df['iwm_ret5'] = etf_data['IWM']['ret5']

# 填充缺失的sector_etf_ret5（用SPY�?df['sector_etf_ret5'] = df['sector_etf_ret5'].fillna(etf_data['SPY']['ret5'])

# 特征�?base_feats = ['price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
              'ret1','ret5','ret20','ret60','macd','macd_signal','macd_hist',
              'vol_ratio','ma_bias20','vol5','trend_accel',
              'short_ratio','short_pct','market_cap','pe_trailing','pe_forward','beta']
new_feats = ['sector_etf_ret5','spy_ret5','qqq_ret5','iwm_ret5']
df['sector_code'] = df['sector'].astype('category').cat.codes.astype(int)
all_feats = base_feats + new_feats + ['sector_code']

print(f"特征: {len(base_feats)}基础+基本�?+ {len(new_feats)}ETF = {len(all_feats)}�?)

df = df.dropna(subset=all_feats + ['label_5d_5class']).copy()
X = df[all_feats].values
y = df['label_5d_5class'].values
n = len(df)

for cl in range(5):
    cnt = (y == cl).sum()
    print(f"  {cl}: {cnt:,} ({cnt/n*100:.1f}%)")

# Walk-Forward
classes = np.array([0,1,2,3,4])
wts = compute_class_weight('balanced', classes=classes, y=y)
wd = {i:w for i,w in enumerate(wts)}

wfs = [('WF1',0,0.60,0.60,0.75,0.75,0.85),
       ('WF2',0.15,0.65,0.65,0.80,0.80,0.90),
       ('WF3',0.30,0.70,0.70,0.85,0.85,1.00)]

for name,ts,te,vs,ve,tst,tste in wfs:
    Xtr, ytr = X[int(ts*n):int(te*n)], y[int(ts*n):int(te*n)]
    Xva, yva = X[int(vs*n):int(ve*n)], y[int(vs*n):int(ve*n)]
    Xte = X[int(tst*n):int(tste*n)]
    yte_act = y[int(tst*n):int(tste*n)]
    
    sw = np.array([wd[yi] for yi in ytr])
    sw *= np.linspace(0.3, 1.0, len(sw))
    
    m = xgb.XGBClassifier(n_estimators=800, max_depth=5, lr=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', early_stopping_rounds=30,
        random_state=42, n_jobs=-1, verbosity=0, num_class=5, device='cuda')
    m.fit(Xtr, ytr, sample_weight=sw,
          eval_set=[(Xva, yva)], verbose=0)
    
    proba = m.predict_proba(Xte)
    pu5 = proba[:,4]
    
    top10 = pu5 >= np.percentile(pu5, 90)
    _end = int(tste*n); _start = int(tst*n)
    actual_pct = df['label_5d_pct'].values[_start:_end]
    r_pct = actual_pct[top10]
    sp = r_pct.mean()/r_pct.std()*math.sqrt(252/5) if r_pct.std()>0 else 0
    hit5 = (yte_act[top10] == 4).mean() if top10.sum() > 0 else 0
    wr = (r_pct > 0).mean()
    
    print(f"  {name}: 夏普={sp:.3f} �?5%命中={hit5:.1%} 胜率={wr:.1%}", flush=True)

# 全量训练+校准
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
pu5 = yp[:,4]
ta = df['label_5d_pct'].values[train_end:]
t10 = pu5 >= np.percentile(pu5, 90)
r = ta[t10]
sf = r.mean()/r.std()*math.sqrt(252/5) if r.std()>0 else 0
wr = (r > 0).mean()
hit5 = (df['label_5d_5class'].values[train_end:][t10] == 4).mean()
print(f"  测试夏普={sf:.3f} 胜率={wr:.1%} �?5%命中={hit5:.1%}")

# 校准�?bins = [(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]
cal_table = {}
for lo, hi in bins:
    mask = (pu5 >= lo) & (pu5 < hi)
    n_bin = mask.sum()
    if n_bin >= 10:
        hit = (df['label_5d_5class'].values[train_end:][mask] == 4).mean()
        avg_prob = pu5[mask].mean()
        cal_table[f'{lo:.0%}-{hi:.0%}'] = {
            'predicted': round(avg_prob,4),
            'actual': round(hit,4),
            'factor': round(hit/avg_prob,4) if avg_prob>0 else 0,
            'n': int(n_bin)
        }
print(f"\n[校准表]")
for k,v in cal_table.items():
    print(f"  {k}: 预测={v['predicted']:.1%} 实际={v['actual']:.1%} 校准因子={v['factor']:.2f} 样本={v['n']}")

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
    })
preds.sort(key=lambda x: -x['prob_up5'])

print(f"\n{'�?*70}")
print(f"{'排名':>3} {'代码':>8} {'价格':>8} {'�?5%':>7} {'校准�?:>7} {'�?5%':>7}")
print(f"{'─'*70}")
for i, r in enumerate(preds[:20]):
    # 校准
    prob = r['prob_up5']
    if prob >= 0.9:
        adj = prob * cal_table.get('90%-100%',{}).get('factor', 0.56)
    elif prob >= 0.8:
        adj = prob * cal_table.get('80%-90%',{}).get('factor', 0.71)
    elif prob >= 0.7:
        adj = prob * cal_table.get('70%-80%',{}).get('factor', 0.73)
    elif prob >= 0.6:
        adj = prob * cal_table.get('60%-70%',{}).get('factor', 0.79)
    elif prob >= 0.5:
        adj = prob * cal_table.get('50%-60%',{}).get('factor', 0.85)
    else:
        adj = prob
    print(f"{i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {prob*100:>6.1f}% {adj*100:>6.1f}% {r['prob_dn5']*100:>6.1f}%")

# 保存模型
final.save_model(_paths.US_MODEL_DIR + "/greenshaft_v17.json")
out = {
    'timestamp': str(__import__('datetime').datetime.now()),
    'model': 'greenshaft_v17',
    'features': all_feats,
    'test_sharpe': round(sf,4),
    'test_win_rate': round(wr,4),
    'test_hit_up5': round(hit5,4),
    'calibration': cal_table,
    'predictions': [{'rank': i+1, 'sym': r['sym'], 'price': r['price'],
                      'up5': float(round(r['prob_up5'],4))} 
                     for i,r in enumerate(preds[:50])],
}
with open(_paths.US_MODEL_DIR + "/us_v19_v17_prediction.json", 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"\n�?绿箭v17 完成! ({time.time()-T0:.0f}s)")
print(f"  模型: {_paths.win(_paths.US_MODEL_DIR + '/greenshaft_v17.json')}")
print(f"  预测: {_paths.win(_paths.US_MODEL_DIR + '/us_v19_v17_prediction.json')}")
