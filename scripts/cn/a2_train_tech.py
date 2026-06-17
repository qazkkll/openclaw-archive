# -*- coding: utf-8 -*-
"""
A2专用训练脚本 — 纯技术指标版本
特征: 30个技术指标 (18旧+12新: KDJ/MACD/BB/OBV/动量)
数据: a_hist_10y.parquet (无资金流依赖)
输出: /home/hermes/.hermes/openclaw-project/data/layer3_checkpoints/a2_tech_30f.json
"""
import json, sys, os, time, numpy as np, xgboost as xgb
from datetime import datetime as dt
sys.stdout.reconfigure(encoding='utf-8')

D_DATA = '/home/hermes/.hermes/openclaw-archive/data'
MODEL_DIR = os.path.join(D_DATA, 'layer3_checkpoints')

# Quality pool (top 400 liquid stocks)
QUALITY_POOL = None  # Load from quality_pool.json if exists
pool_path = os.path.join(D_DATA, 'quality_pool.json')
if os.path.exists(pool_path):
    QUALITY_POOL = json.load(open(pool_path, 'r', encoding='utf-8'))
    print("Quality pool:", len(QUALITY_POOL), "stocks")

# ─── Features ───
FEATURE_KEYS = [
    # 18 old tech
    'pct_ma5','pct_ma10','pct_ma20','pct_ma60','pct_ma120',
    'ma20_slope','ma60_slope','ma_align',
    'vol_10d','vol_60d','vol_ratio','atr20_pct',
    'ret_5d','ret_10d','ret_20d','ret_60d','rsi14',
    'vol_ratio_5_20',
    # 12 new tech
    'kdj_k','kdj_d','kdj_j',
    'macd_dif','macd_dea','macd_bar',
    'bb_width','bb_position',
    'obv_ratio_5_20',
    'ret5_max','ret3_vs_ema12',
]

def calc_features(code, hist_rec):
    """Compute features for one stock. Returns [(date, feat_dict, fwd_10d), ...]"""
    c = hist_rec['c']
    h = hist_rec['h']
    l = hist_rec['l']
    o = hist_rec['o']
    v = hist_rec['v']
    dates = hist_rec.get('dates', [str(i) for i in range(len(c))])
    n = len(c)
    results = []
    
    for i in range(120, n-10):  # Need 120 warmup
        price = c[i]
        rec = {'code': code, 'date': dates[i], 'close': price}
        
        # ─── MA features ───
        ma5 = sum(c[i-4:i+1])/5; ma10 = sum(c[i-9:i+1])/10
        ma20 = sum(c[i-19:i+1])/20; ma60 = sum(c[i-59:i+1])/60
        ma120 = sum(c[i-119:i+1])/120 if i >= 119 else ma60
        
        rec['pct_ma5'] = (price/ma5-1)*100 if ma5>0 else 0
        rec['pct_ma10'] = (price/ma10-1)*100 if ma10>0 else 0
        rec['pct_ma20'] = (price/ma20-1)*100 if ma20>0 else 0
        rec['pct_ma60'] = (price/ma60-1)*100 if ma60>0 else 0
        rec['pct_ma120'] = (price/ma120-1)*100 if ma120>0 else 0
        
        if i >= 25:
            ma20b = sum(c[i-25:i-4])/20
            rec['ma20_slope'] = (ma20/ma20b-1)*100 if ma20b>0 else 0
        else: rec['ma20_slope'] = 0
        if i >= 65:
            ma60b = sum(c[i-65:i-4])/60
            rec['ma60_slope'] = (ma60/ma60b-1)*100 if ma60b>0 else 0
        else: rec['ma60_slope'] = 0
        
        rec['ma_align'] = (ma5>ma10)+(ma10>ma20)+(ma20>ma60)+(price>ma5)+(price>ma10)+(price>ma60)
        
        # ─── Volatility ───
        rets = [abs(c[j]/c[j-1]-1)*100 for j in range(max(1,i-9),i+1) if c[j-1]>0]
        rec['vol_10d'] = sum(rets)/len(rets) if rets else 0
        rets60 = [abs(c[j]/c[j-1]-1)*100 for j in range(max(1,i-59),i+1) if c[j-1]>0]
        rec['vol_60d'] = sum(rets60)/len(rets60) if rets60 else 0
        rec['vol_ratio'] = rec['vol_10d']/rec['vol_60d'] if rec['vol_60d']>0 else 1
        
        trs = [max(h[j]-l[j],abs(h[j]-c[j-1]),abs(l[j]-c[j-1])) for j in range(max(1,i-19),i+1)]
        rec['atr20_pct'] = sum(trs)/len(trs)/price*100 if price>0 else 0
        
        # ─── Returns ───
        rec['ret_5d'] = (price/c[i-5]-1)*100 if i>=5 else 0
        rec['ret_10d'] = (price/c[i-10]-1)*100 if i>=10 else 0
        rec['ret_20d'] = (price/c[i-20]-1)*100 if i>=20 else 0
        rec['ret_60d'] = (price/c[i-60]-1)*100 if i>=60 else 0
        
        # ─── RSI ───
        changes = [c[j]-c[j-1] for j in range(max(1,i-13),i+1)]
        gains = sum(x for x in changes if x>0)
        losses = abs(sum(x for x in changes if x<0))
        rec['rsi14'] = 100-100/(1+gains/losses) if losses>0 else 100
        
        # ─── Volume ratio ───
        vol5 = sum(v[i-4:i+1])/5
        vol20 = sum(v[max(0,i-19):i+1])/20
        rec['vol_ratio_5_20'] = vol5/vol20 if vol20>0 else 1
        
        # ─── KDJ (9,3,3) ───
        if i >= 8:
            hh9 = max(h[i-8:i+1])
            ll9 = min(l[i-8:i+1])
            rsv = (price-ll9)/(hh9-ll9)*100 if (hh9-ll9)>0 else 50
        else: rsv = 50
        rec['kdj_k'] = round(2/3*50 + 1/3*rsv, 2)
        rec['kdj_d'] = round(2/3*50 + 1/3*rec['kdj_k'], 2)
        rec['kdj_j'] = round(3*rec['kdj_k']-2*rec['kdj_d'], 2)
        
        # ─── MACD (12,26,9) ───
        ema12 = price; ema26 = price
        for jj in range(min(34, i)):
            ema12 = c[i-jj-1]*(2/13)+ema12*(11/13)
            ema26 = c[i-jj-1]*(2/27)+ema26*(25/27)
        dif = ema12-ema26
        dea = dif
        for jj in range(min(8, i)):
            idx2 = i-jj-1; e12=c[idx2]; e26=c[idx2]
            for kk in range(min(26, idx2)):
                e12=c[idx2-kk-1]*(2/13)+e12*(11/13)
                e26=c[idx2-kk-1]*(2/27)+e26*(25/27)
            dea = (e12-e26)*(2/10)+dea*(8/10)
        rec['macd_dif'] = round(dif, 4)
        rec['macd_dea'] = round(dea, 4)
        rec['macd_bar'] = round(2*(dif-dea), 4)
        
        # ─── Bollinger Bands (20,2) ───
        if i >= 19:
            ma20bb = sum(c[i-19:i+1])/20
            var20 = sum((c[j]-ma20bb)**2 for j in range(i-19,i+1))/20
            std20 = var20**0.5
            rec['bb_width'] = round(4*std20/ma20bb, 4) if ma20bb>0 else 0
            rec['bb_position'] = round((price-(ma20bb-2*std20))/(4*std20), 4) if std20>0 else 0.5
        else:
            rec['bb_width'], rec['bb_position'] = 0, 0.5
        
        # ─── OBV ───
        obv_seq = [0]
        for j in range(1, i+1):
            dv = v[j] if c[j]>c[j-1] else (-v[j] if c[j]<c[j-1] else 0)
            obv_seq.append(obv_seq[-1]+dv)
        obv5 = abs(obv_seq[-1]-obv_seq[max(0,len(obv_seq)-6)])
        obv20 = abs(obv_seq[-1]-obv_seq[max(0,len(obv_seq)-21)])
        rec['obv_ratio_5_20'] = round(obv5/obv20, 4) if obv20>0 else 1.0
        
        # ─── Momentum ───
        win5 = [c[j]/c[j-1]-1 for j in range(max(1,i-4),i+1)]
        rec['ret5_max'] = round(max(win5)*100, 2) if win5 else 0
        ret3 = price/c[i-3]-1 if i>=3 else 0
        ema12h = sum(c[i-11:i+1])/12 if i>=11 else 0.01
        rec['ret3_vs_ema12'] = round((ret3*100)/(ema12h*100+1),4) if abs(ema12h*100)>0.001 else 0
        
        # ─── Target: forward 10d return ───
        # Separate train/val/test
        fwd = (c[i+10]/price-1)*100
        
        results.append((dates[i], rec, fwd))
    
    return results

# ─── Load data ───
print("Loading hist data...", flush=True)
t0 = time.time()
with open(os.path.join(D_DATA, 'a_hist_10y.parquet'), 'rb') as f:
    hist = json.load(f)
print("  %d stocks, %.1fs" % (len(hist), time.time()-t0), flush=True)

# Filter to quality pool
if QUALITY_POOL:
    # quality_pool maps ts_code to something
    pool_stocks = []
    for item in QUALITY_POOL:
        code = item.get('ts_code', item.get('code', item.get('symbol', '')))
        # Try different formats: 000001, 000001.SZ, etc.
        for fmt in [code, code.replace('.SZ','').replace('.SH','')]:
            if fmt in hist:
                pool_stocks.append(fmt)
                break
    stock_codes = pool_stocks
    print("  Quality pool: %d/%d in hist" % (len(stock_codes), len(QUALITY_POOL)))
else:
    # Filter: must have 200+ bars, price >= 1, not ST
    stock_codes = []
    for c in hist:
        sd = hist[c]
        if len(sd['c']) >= 200 and sd['c'][-1] >= 1.0:
            stock_codes.append(c)
    stock_codes = sorted(stock_codes)
    print("  Filtered: %d stocks (>=200 bars, >=$1)" % len(stock_codes))

# ─── Build feature matrix ───
print("\nBuilding feature matrix...", flush=True)
t1 = time.time()
all_rows = []
skipped = 0

for idx, code in enumerate(stock_codes):
    if idx % 50 == 0 and idx > 0:
        print("  [%d/%d] %.0fs" % (idx, len(stock_codes), time.time()-t1), flush=True)
    
    try:
        results = calc_features(code, hist[code])
        for date, rec, fwd in results:
            row = [rec.get(k, 0) for k in FEATURE_KEYS]
            all_rows.append(row + [fwd, date, code])
    except Exception as e:
        skipped += 1
        if skipped <= 5:
            print("  Skip %s: %s" % (code, e), flush=True)

print("  %d samples, skipped %d stocks, %.0fs" % (len(all_rows), skipped, time.time()-t1), flush=True)

if len(all_rows) == 0:
    print("ERROR: No data")
    sys.exit(1)

# ─── Build matrices ───
X = np.array([row[:len(FEATURE_KEYS)] for row in all_rows], dtype=np.float32)
y = np.array([row[len(FEATURE_KEYS)] for row in all_rows], dtype=np.float32)
dates_arr = [row[len(FEATURE_KEYS)+1] for row in all_rows]
codes_arr = [row[len(FEATURE_KEYS)+2] for row in all_rows]

# Check for inf/nan
bad = ~np.isfinite(X).all(axis=1) | ~np.isfinite(y)
bad_count = bad.sum()
if bad_count > 0:
    print("Removing %d samples with inf/nan" % bad_count, flush=True)
    X = X[~bad]
    y = y[~bad]
    dates_arr = [d for i,d in enumerate(dates_arr) if not bad[i]]
    codes_arr = [c for i,c in enumerate(codes_arr) if not bad[i]]

# ─── Train/Val/Test split ───
# 10-year period: train<=2023, val 2024-2025, test>=2025
# Use date string comparison
train_mask = np.array([d < '2024-01-01' for d in dates_arr])
val_mask = np.array([('2024-01-01' <= d < '2025-07-01') for d in dates_arr])
test_mask = np.array([d >= '2025-07-01' for d in dates_arr])

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print("\nData split: train=%d val=%d test=%d" % (len(X_train), len(X_val), len(X_test)), flush=True)

# ─── Grid search ───
print("\nGrid search...", flush=True)
best_score = -999
best_params = {}
best_model = None

param_grid = {'max_depth': [4, 6], 'learning_rate': [0.03, 0.05], 'min_child_weight': [3, 5]}

from itertools import product
for depth, lr, mw in product(*param_grid.values()):
    params = {
        'objective': 'reg:squarederror', 'max_depth': depth,
        'learning_rate': lr, 'min_child_weight': mw,
        'subsample': 0.7, 'colsample_bytree': 0.6,
        'eval_metric': 'rmse', 'seed': 42,
        'device': 'cuda'
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_KEYS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_KEYS)
    
    model = xgb.train(params, dtrain, num_boost_round=500,
                      evals=[(dval, 'val')], early_stopping_rounds=30,
                      verbose_eval=False)
    
    pred_val = model.predict(dval)
    ss_res = np.sum((y_val-pred_val)**2)
    ss_tot = np.sum((y_val-y_val.mean())**2)
    r2 = 1 - ss_res/ss_tot
    print("  depth=%d lr=%.3f mw=%d : val_R2=%.4f best_iter=%d" % (
        depth, lr, mw, r2, model.best_iteration), flush=True)
    
    if r2 > best_score:
        best_score = r2
        best_params = {'max_depth': depth, 'learning_rate': lr, 'min_child_weight': mw}
        best_model = model

print("Best: depth=%d lr=%.3f mw=%d val_R2=%.4f" % (
    best_params['max_depth'], best_params['learning_rate'],
    best_params['min_child_weight'], best_score), flush=True)

# ─── Final eval with best model ───
print("\n" + "=" * 55, flush=True)
print("Final Evaluation (30 tech features):", flush=True)
print("%8s %8s %8s %8s %10s %10s" % ('Dataset','RMSE','MAE','R2','Top5_avg','Top20_avg'), flush=True)
print("-" * 55, flush=True)

for name, X_set, y_set in [('Train',X_train,y_train), ('Val',X_val,y_val), ('Test',X_test,y_test)]:
    md = xgb.DMatrix(X_set, feature_names=FEATURE_KEYS)
    pred = best_model.predict(md)
    
    rmse = np.sqrt(np.mean((pred-y_set)**2))
    mae = np.mean(np.abs(pred-y_set))
    ss_res = np.sum((y_set-pred)**2)
    ss_tot = np.sum((y_set-y_set.mean())**2)
    r2 = 1-ss_res/ss_tot if ss_tot>0 else 0
    
    top5_avg = y_set[np.argsort(pred)[-5:]].mean() if len(pred)>=5 else 0
    top20_avg = y_set[np.argsort(pred)[-min(20,len(pred)):]].mean()
    
    print("%8s %8.3f %8.3f %8.4f %+10.2f%% %+10.2f%%" % (name, rmse, mae, r2, top5_avg, top20_avg), flush=True)

# Feature importance
imp = best_model.get_score(importance_type='gain')
imp_sorted = sorted(imp.items(), key=lambda x: -x[1])
print("\nFeature importance (gain):", flush=True)
for name, gain in imp_sorted[:15]:
    pct = gain/sum(v for _,v in imp_sorted)*100
    print("  %-20s: %8.1f (%.1f%%)" % (name, gain, pct), flush=True)

# Test quintile analysis
print("\nTest quintile:", flush=True)
dtest = xgb.DMatrix(X_test, feature_names=FEATURE_KEYS)
pred_test = best_model.predict(dtest)
order = np.argsort(pred_test)
n = len(pred_test)
print("%8s %10s %12s %8s %8s" % ('Group','Avg_pred','Avg_actual','Win%','Median'), flush=True)
for g in range(5):
    start, end = g*n//5, (g+1)*n//5
    idx = order[start:end]
    ap = pred_test[idx].mean()
    aa = y_test[idx].mean()
    w = (y_test[idx]>0).mean()*100
    md = np.median(y_test[idx])
    label = 'Q%d'%(g+1) if g<4 else '(High)'
    print("%8s %+9.3f%% %+11.2f%% %7.1f%% %+7.2f%%" % (label, ap, aa, w, md), flush=True)

# ─── Save model ───
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, 'a2_tech_30f.json')
best_model.save_model(model_path)

meta = {
    'model': 'a2_tech_30f',
    'features': FEATURE_KEYS,
    'n_features': len(FEATURE_KEYS),
    'best_iteration': best_model.best_iteration,
    'best_params': best_params,
    'val_R2': best_score,
    'data': 'a_hist_10y.parquet (2016-2026)',
    'split': 'train<=2023, val=2024-2025H1, test>=2025H2',
    'pool': 'quality_pool.json (%d stocks)' % len(stock_codes),
    'generated': '2026-06-14'
}
meta_path = os.path.join(MODEL_DIR, 'a2_tech_30f_meta.json')
json.dump(meta, open(meta_path,'w'), ensure_ascii=False, indent=2)

print("\nModel saved: %s" % model_path, flush=True)
print("Meta: %s" % meta_path, flush=True)
print("Total time: %.0fs" % (time.time()-t0), flush=True)
