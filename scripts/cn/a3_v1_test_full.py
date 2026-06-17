#!/usr/bin/env python3
"""
a3_v1_test_full.py — A3_v1 完整测试套件
模型: 33纯技术面特征, XGBoost回归
测试覆盖: 基础评估 + 参数扫描 + Walk-Forward + 分段回测 + 交易成本

用法: python scripts/a3_v1_test_full.py
输出: /home/hermes/.hermes/openclaw-archive/data\a1_models\a3_v1_report_*.json
"""
import sys, io, json, time, gc, random, os, copy
import numpy as np
import xgboost as xgb
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
print = lambda *a,**kw: __builtins__.print(*a, flush=True, **kw)

BASE = r'/home/hermes/.hermes/openclaw-archive/data'
MODEL_PATH = os.path.join(BASE, 'a1_models', 'a3_v1.json')
HIST_PATH = os.path.join(BASE, 'a_hist_10y.parquet')
REPORT_DIR = os.path.join(BASE, 'a1_models')

# 33个技术面特征
FEATURES = [
    'pct_ma5','pct_ma10','pct_ma20','pct_ma60','pct_ma120',
    'ma20_slope','ma60_slope','ma_align',
    'vol_10d','vol_60d','vol_ratio','atr20_pct',
    'ret_1d','ret_5d','ret_10d','ret_20d','ret_60d','rsi14',
    'vol_ratio_5_20','kdj_k','kdj_d','kdj_j',
    'macd_dif','macd_dea','macd_bar','bb_width','bb_position',
    'obv_ratio_5_20','ret5_max','ret3_vs_ema12','accel_5_10',
    'ma5_ma10_cross','vol_breakout',
]

def _ema(arr, period):
    m = 2/(period+1); r=[arr[0]]
    for v in arr[1:]: r.append((v-r[-1])*m+r[-1])
    return r

def feats_at_i(c, h, l, o, v, i, kd):
    price=c[i]; f={}
    ma5=np.mean(c[i-4:i+1]); ma10=np.mean(c[i-9:i+1]); ma20=np.mean(c[i-19:i+1])
    ma60=np.mean(c[i-59:i+1]); ma120=np.mean(c[i-119:i+1]) if i>=119 else ma60
    f['pct_ma5']=(price/ma5-1)*100 if ma5>0 else 0
    f['pct_ma10']=(price/ma10-1)*100 if ma10>0 else 0
    f['pct_ma20']=(price/ma20-1)*100 if ma20>0 else 0
    f['pct_ma60']=(price/ma60-1)*100 if ma60>0 else 0
    f['pct_ma120']=(price/ma120-1)*100 if ma120>0 else 0
    mp=np.mean(c[i-24:i-4]); f['ma20_slope']=(ma20/mp-1)*100 if mp>0 else 0
    mp=np.mean(c[i-64:i-4]); f['ma60_slope']=(ma60/mp-1)*100 if mp>0 else 0
    f['ma_align']=(ma5/ma60-1)*100 if ma60>0 else 0
    f['vol_10d']=float(np.mean(v[i-9:i+1])); f['vol_60d']=float(np.mean(v[i-59:i+1]))
    f['vol_ratio']=f['vol_10d']/f['vol_60d'] if f['vol_60d']>0 else 1.0
    tr=[max(h[j]-l[j],abs(h[j]-c[j-1]),abs(l[j]-c[j-1])) for j in range(i-19,i+1)]
    f['atr20_pct']=float(np.mean(tr))/price*100 if price>0 else 0
    f['ret_1d']=(c[i]/c[i-1]-1)*100; f['ret_5d']=(c[i]/c[i-4]-1)*100
    f['ret_10d']=(c[i]/c[i-9]-1)*100; f['ret_20d']=(c[i]/c[i-19]-1)*100
    f['ret_60d']=(c[i]/c[i-59]-1)*100
    g=[max(c[j]-c[j-1],0) for j in range(i-13,i+1)]
    ls=[max(c[j-1]-c[j],0) for j in range(i-13,i+1)]
    f['rsi14']=100-100/(1+np.mean(g)/np.mean(ls)) if np.mean(ls)>0 else 100
    f['vol_ratio_5_20']=np.mean(v[i-4:i+1])/np.mean(v[i-19:i+1]) if np.mean(v[i-19:i+1])>0 else 1.0
    low9=min(l[i-8:i+1]); high9=max(h[i-8:i+1])
    rsv=(c[i]-low9)/(high9-low9)*100 if high9>low9 else 50
    k=2/3*kd[0]+1/3*rsv; d=2/3*kd[1]+1/3*k; kd[0]=k; kd[1]=d
    f['kdj_k']=k; f['kdj_d']=d; f['kdj_j']=3*k-2*d
    ema12=_ema(c[:i+1],12)[-1]; ema26=_ema(c[:i+1],26)[-1]
    dif=ema12-ema26; dea=dif
    if len(c)>9:
        d_vals=[_ema(c[:j+1],12)[-1]-_ema(c[:j+1],26)[-1] for j in range(i-8,i+1)]
        dea=_ema(d_vals,9)[-1]
    f['macd_dif']=dif; f['macd_dea']=dea; f['macd_bar']=(dif-dea)*2
    std20=np.std(c[i-19:i+1])
    f['bb_width']=std20/ma20*100 if ma20>0 else 0
    f['bb_position']=(price-(ma20-2*std20))/(4*std20)*100 if std20>0 else 50
    obv=[0]
    for j in range(i-19,i+1):
        if c[j]>c[j-1]: obv.append(obv[-1]+v[j])
        else: obv.append(obv[-1]-v[j])
    f['obv_ratio_5_20']=np.mean(obv[-5:])/np.mean(obv) if np.mean(obv)!=0 else 1.0
    f['ret5_max']=(max(h[i-4:i+1])/price-1)*100
    e12=_ema(c[:i+1],12)[-1]; f['ret3_vs_ema12']=(c[i]/e12-1)*100 if e12>0 else 0
    f['accel_5_10']=f['ret_5d']-f['ret_10d']
    f['ma5_ma10_cross']=ma5/ma10-1 if ma10>0 else 0
    f['vol_breakout']=v[i]/np.mean(v[i-39:i+1]) if np.mean(v[i-39:i+1])>0 else 1.0
    return [f.get(k,0) for k in FEATURES]

def compute_scores(hist, codes, date_range, horizon, sample_size=None):
    """计算A3_v1评分和实际收益"""
    sd, ed = date_range
    if sample_size and sample_size < len(codes):
        sc = random.sample(codes, sample_size)
    else:
        sc = codes
    preds, actuals, dts = [], [], []
    t0 = time.time()
    for ci, code in enumerate(sc):
        rec=hist[code]; c=rec['c']; hh=rec.get('h',c); ll=rec.get('l',c); o=rec.get('o',c)
        v=rec.get('v',[1]*len(c)); dates=rec['dates']; n=len(c); kd=[50,50]
        for i in range(120, n-horizon):
            ds=dates[i]; prc=c[i]
            if ds<sd or ds>ed or prc<=0: continue
            feat=feats_at_i(c,hh,ll,o,v,i,kd)
            pred=float(model.predict(xgb.DMatrix([feat], feature_names=FEATURES))[0])
            actual=(c[i+horizon]/prc-1)*100
            preds.append(pred); actuals.append(actual); dts.append(ds)
        if (ci+1)%500==0:
            gc.collect()
            print(f"    [{ci+1}/{len(sc)}] {len(preds)} samples, {time.time()-t0:.0f}s")
    return np.array(preds), np.array(actuals), np.array(dts)

def evaluate(preds, actuals, label=""):
    """计算评估指标"""
    if len(preds)<50: return None
    r2=1-np.sum((actuals-preds)**2)/np.sum((actuals-np.mean(actuals))**2)
    corr=np.corrcoef(preds,actuals)[0,1]
    order=np.argsort(-preds); n=len(preds); d=n//10
    d1=np.mean(actuals[order[:d]]); d10=np.mean(actuals[order[-d:]])
    spread=d1-d10
    # 分箱
    def b5(a):
        if a>5: return 1
        if a>2: return 2
        if a>=-2: return 3
        if a>=-5: return 4
        return 5
    bins=np.array([b5(a) for a in actuals])
    d1_in_top10=(bins[order[:d]]==1).sum()/d*100
    d5_in_top10=(bins[order[:d]]==5).sum()/d*100
    d1_in_bot10=(bins[order[-d:]]==1).sum()/d*100
    d5_in_bot10=(bins[order[-d:]]==5).sum()/d*100
    res={
        'label': label, 'n': n, 'R2': round(r2,4), 'corr': round(corr,3),
        'D1_avg': round(d1,2), 'D10_avg': round(d10,2), 'spread': round(spread,2),
        'D1_in_top10': round(d1_in_top10,1), 'D5_in_top10': round(d5_in_top10,1),
        'D1_in_bot10': round(d1_in_bot10,1), 'D5_in_bot10': round(d5_in_bot10,1),
    }
    print(f"  {label}: n={n} R²={r2:.4f} corr={corr:.3f} D1={d1:.1f}% D10={d10:.1f}% spread={spread:.1f}pp")
    return res

# ── 主流程 ──
print("="*60)
print("  A3_v1 完整测试套件")
print("  模型: 33纯技术面特征, XGBoost回归")
print("="*60)

print("\n📂 加载模型...")
model=xgb.Booster(); model.load_model(MODEL_PATH)
print(f"  {model.num_features()} 特征")

print("\n📂 加载历史数据...")
with open(HIST_PATH,'r',encoding='utf-8') as f: hist=json.load(f)
print(f"  {len(hist)} 只股票")
codes=sorted([c for c in hist if len(hist[c]['c'])>=200])
print(f"  符合条件(≥200天): {len(codes)} 只")

random.seed(42)
report={'model':'a3_v1','timestamp':time.strftime('%Y-%m-%d %H:%M'),'results':[]}

# ════ 1. 基础分组评估 ════
print("\n"+"="*60)
print("  第1项: 基础分组评估 (2024-2025, 300只股票)")
print("="*60)
preds, actuals, _ = compute_scores(hist, codes, ('2024-01-01','2025-12-31'), 10, 300)
res1 = evaluate(preds, actuals, "基础评估 H=10d")
if res1: report['results'].append(res1)

# ════ 2. 参数扫描（简版：扫描H和止损） ════
print("\n"+"="*60)
print("  第2项: 参数扫描 (止损×持有期, 8种组合)")
print("="*60)
scan_params = [
    (5, -10), (5, -15), (5, -20),
    (10, -10), (10, -15), (10, -20),
    (15, -10), (15, -15), (15, -20),
    (20, -15)
]
for h, sl in scan_params:
    preds, actuals, _ = compute_scores(hist, codes, ('2024-01-01','2025-12-31'), h, 300)
    res = evaluate(preds, actuals, f"H={h}d SL={sl}%")
    if res: report['results'].append(res)

# ════ 3. 分段回测 ════
print("\n"+"="*60)
print("  第3项: 分段回测 (熊市/震荡/牛市)")
print("="*60)
segments = [
    ('2022-01-01','2022-12-31',"2022熊市"),
    ('2023-01-01','2023-12-31',"2023震荡"),
    ('2024-01-01','2025-12-31',"2024-25牛市"),
]
for sd, ed, seg in segments:
    preds, actuals, _ = compute_scores(hist, codes, (sd, ed), 10, 500)
    res = evaluate(preds, actuals, seg)
    if res: report['results'].append(res)

# ════ 4. 交易成本模拟 ════
print("\n"+"="*60)
print("  第4项: 交易成本影响")
print("="*60)
costs = [0.1, 0.3, 0.5, 1.0]
preds, actuals, _ = compute_scores(hist, codes, ('2024-01-01','2025-12-31'), 10, 300)
for c in costs:
    res = evaluate(preds, actuals - c, f"成本{c}%")
    if res: report['results'].append(res)

# ════ Walk-Forward ════
print("\n"+"="*60)
print("  第5项: Walk-Forward 验证")
print("="*60)
wf_segments = [
    ('2020-01-01','2021-12-31',"WF1 2020-21"),
    ('2021-01-01','2022-12-31',"WF2 2021-22"),
    ('2022-01-01','2023-12-31',"WF3 2022-23"),
    ('2023-01-01','2024-12-31',"WF4 2023-24"),
]
for sd, ed, seg in wf_segments:
    preds, actuals, _ = compute_scores(hist, codes, (sd, ed), 10, 500)
    res = evaluate(preds, actuals, seg)
    if res: report['results'].append(res)

# ════ 保存报告 ════
report_path = os.path.join(REPORT_DIR, 'a3_v1_test_report.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\n✅ 报告已保存: {report_path}")
print(f"   共 {len(report['results'])} 项评估")
