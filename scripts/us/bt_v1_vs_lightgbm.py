#!/usr/bin/env python3
"""V1 vs LightGBM因子权重对比 - 轻量版"""
import json, numpy as np, sys, os, time
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from score_engine import compute_indicators, get_raw_scores

print('🍤 V1 vs LightGBM 因子权重对比 (轻量版)')
t0 = time.time()

with open(os.path.join(ROOT, 'data', 'backtest_hist_yahoo.json')) as f:
    raw = json.load(f)

codes = list(raw.keys())[:30]
all_features = []; all_labels = []

for ci, code in enumerate(codes):
    item = raw[code]
    if not isinstance(item, dict): continue
    dates = item.get('dates',[])
    c = item.get('close',[]); h = item.get('high',[]); l = item.get('low',[])
    if len(c) < 250: continue
    
    for i in range(200, len(c)-20, 20):
        ind = compute_indicators(c[:i+1], h[:i+1], l[:i+1])
        if ind is None: continue
        rs = get_raw_scores(ind, len(c[:i+1])-1)
        if rs.get('ms',0) == 0 and rs.get('mas',0) == 0:
            continue  # 无有效信号跳过
        
        # 归一化因子: 各因子分/20.0
        f = [
            rs.get('ms',0)/20.0, rs.get('ws',0)/20.0,
            rs.get('mas',0)/20.0, rs.get('ads',0)/20.0,
            rs.get('rs',0)/20.0,
            (c[i]/c[i-20]-1)*100 if i>=20 else 0,
            (c[i]/c[i-60]-1)*100 if i>=60 else 0,
        ]
        all_features.append(f)
        all_labels.append((c[i+10]/c[i]-1)*100)
    
    print(f'  [{ci+1}/{len(codes)}] {code}', flush=True)

print(f'\n样本: {len(all_features)}条')
if len(all_features) < 50:
    print('样本不足，跳过训练')
    sys.exit(0)

X = np.array(all_features); y = np.array(all_labels)
split = int(len(X)*0.8)

model = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, verbose=-1)
model.fit(X[:split], y[:split])
test_pred = model.predict(X[split:])
test_r = np.corrcoef(y[split:], test_pred)[0,1]

names = ['MACD','52周位置','均线','ADX','RSI','20日动量','60日动量']
imp = model.feature_importances_
total = sum(imp)

print(f'\n时间: {time.time()-t0:.0f}s  LightGBM预测R={test_r:.3f}')
print(f'\n{"因子":<12} {"LightGBM":>8} {"V1权重":>8}')
print('-' * 32)
v1 = {'MACD':25,'52周位置':15,'均线':15,'ADX':25,'RSI':20}
for name, imp_val in zip(names[:5], imp[:5]):
    lgb_pct = imp_val/total*100
    v1_w = v1.get(name,0)
    d = '↑' if lgb_pct>v1_w+3 else('↓' if lgb_pct<v1_w-3 else '→')
    print(f'{name:<12} {lgb_pct:>5.1f}%    {v1_w:>3}%  {d}')
for name, imp_val in zip(names[5:], imp[5:]):
    print(f'{name:<12} {imp_val/total*100:>5.1f}%    {"-":>3}  V1无')

model.booster_.save_model(os.path.join(ROOT,'data','lgb_model.txt'))
print(f'\n✅ 模型保存到 data/lgb_model.txt')
