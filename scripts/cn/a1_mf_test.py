#!/usr/bin/env python3
"""a1_mf_test.py — A3(33技术) vs A3+MF(33技术+21资金流) 对比验证

更新: 2026-06-14
- parquet读时过滤（不加载全量720万行）
- 只加载必需列，减少内存
- 保留原有逻辑不变
"""
import json, time, gc, math, os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
import numpy as np
import pandas as pd
import xgboost as xgb

H = 15

# 日志同时输出到文件
import io
log_file = open(r'/home/hermes/.hermes/openclaw-archive/data\a3_mf_test_log.txt', 'w', encoding='utf-8')
class TeeWriter:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for stream in self.streams:
            stream.write(s)
            stream.flush()
    def flush(self):
        for stream in self.streams:
            stream.flush()
sys.stdout = TeeWriter(sys.__stdout__, log_file)

print(f"A3 vs A3+MF 对比 (H={H}d)", flush=True)

# ============================================================
# 1. 加载历史K线 → 确定样本 → 再读资金流（只加载样本）
# ============================================================
print("Loading hist data...", flush=True)
t_start = time.time()
with open(r'/home/hermes/.hermes/openclaw-archive/data\a_hist_10y.parquet') as f:
    hist = json.load(f)
codes = sorted(c for c in hist if len(hist[c]['c']) >= 200)
import random; random.seed(42); sample = random.sample(codes, 150)
print(f"样本: {len(sample)}只, {time.time()-t_start:.0f}s", flush=True)

# 构建ts_code → 原始code映射
sample_ts_codes = set()
code_to_ts = {}
for code in sample:
    ts_code = code + ('.SZ' if code.startswith(('0','3')) else '.SH')
    sample_ts_codes.add(ts_code)
    code_to_ts[code] = ts_code
print(f"样本股票代码: {len(sample_ts_codes)}只", flush=True)

# ============================================================
# 2. 加载资金流 — parquet读时过滤（只读样本股票的+必需列）
# ============================================================
print("Loading moneyflow (filtered at parquet read level)...", flush=True)
t0 = time.time()

# 必需列列表
MF_COLS = ['ts_code', 'trade_date',
           'buy_lg_amount', 'sell_lg_amount',
           'buy_elg_amount', 'sell_elg_amount',
           'buy_md_amount', 'sell_md_amount',
           'net_mf_amount']

# pd.read_parquet 支持 filters 参数（pyarrow引擎默认）
# filters = [('ts_code', 'in', sample_ts_codes)] 只在parquet_reader层面过滤
mf = pd.read_parquet(
    r'/home/hermes/.hermes/openclaw-archive/data\moneyflow_core.parquet',
    columns=MF_COLS,
    filters=[('ts_code', 'in', sample_ts_codes)]
)
print(f"  过滤后: {len(mf):,} rows, {time.time()-t0:.0f}s", flush=True)

if len(mf) == 0:
    print("  ⚠️ 过滤后无数据 — 检查ts_code格式是否匹配", flush=True)
    # 应急: 检查第一只code的格式
    first_code = list(sample_ts_codes)[0]
    print(f"  例: {first_code}", flush=True)
    # 尝试全量加载一小块看看格式
    mf_check = pd.read_parquet(r'/home/hermes/.hermes/openclaw-archive/data\moneyflow_core.parquet', columns=['ts_code'], nrows=5)
    print(f"  资金流ts_code格式样例: {mf_check['ts_code'].tolist()}", flush=True)
    sys.exit(1)

# 计算派生资金流特征
print("Computing derived MF features...", flush=True)
mf['lg_net_1d'] = mf['buy_lg_amount'] - mf['sell_lg_amount']
mf['elg_net_1d'] = mf['buy_elg_amount'] - mf['sell_elg_amount']
mf['md_net_1d'] = mf['buy_md_amount'] - mf['sell_md_amount']
mf['major_net_1d'] = mf['lg_net_1d'] + mf['elg_net_1d']
total_vol = (mf['buy_lg_amount'] + mf['sell_lg_amount'] +
             mf['buy_elg_amount'] + mf['sell_elg_amount'])
mf['lg_pct'] = np.where(total_vol > 0, mf['lg_net_1d'] / total_vol * 100, 0.0)
mf['elg_pct'] = np.where(total_vol > 0, mf['elg_net_1d'] / total_vol * 100, 0.0)
mf['major_ratio'] = np.where(total_vol > 0, mf['major_net_1d'] / total_vol * 100, 0.0)
mf['net_mf_1d'] = mf['net_mf_amount']

# 排序 + 滚动求和
mf = mf.sort_values(['ts_code','trade_date']).reset_index(drop=True)
for base, src in [('net_mf','net_mf_amount'), ('lg_net','lg_net_1d'), ('major_net','major_net_1d')]:
    for w in [5, 10, 20, 60]:
        col = f'{base}_{w}d'
        mf[col] = mf.groupby('ts_code')[src].transform(
            lambda x: x.rolling(w, min_periods=1).sum())
print(f"  Derived features done, {time.time()-t0:.0f}s", flush=True)

# 构建dict查询: (ts_code, date) → mf feature values (向量化构建)
MF_FEATS = ['net_mf_1d','lg_net_1d','elg_net_1d','md_net_1d','lg_pct','elg_pct',
            'major_net_1d','major_ratio','net_mf_5d','major_net_5d','lg_net_5d',
            'net_mf_10d','major_net_10d','lg_net_10d','net_mf_20d','major_net_20d',
            'lg_net_20d','net_mf_60d','major_net_60d','lg_net_60d','ret20d_pct']

feat_names = [f for f in MF_FEATS if f != 'ret20d_pct']
ts_arr = mf['ts_code'].values
dt_arr = mf['trade_date'].values.astype(str)  # 统一为str
feat_arrs = [mf[f].values for f in feat_names]

mf_dict = {}
for idx in range(len(mf)):
    key = (ts_arr[idx], dt_arr[idx])
    vals = [float(a[idx]) if not (isinstance(a[idx], float) and np.isnan(a[idx])) else 0.0
            for a in feat_arrs]
    mf_dict[key] = vals

del mf
gc.collect()
print(f"  MF dict: {len(mf_dict):,} entries, {time.time()-t0:.0f}s", flush=True)

# ============================================================
# 3. 以下：技术特征(33) + MF特征 + 训练对比（保持原有逻辑）
# ============================================================

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
ALL_FEATS = FEATURES + MF_FEATS

def _ema(a, p):
    m = 2/(p+1); r = [a[0]]
    for v in a[1:]: r.append((v-r[-1])*m+r[-1])
    return r

def compute_tech(c, h, l, o, v, i, kd):
    """计算33个技术特征"""
    p = c[i]; f = {}
    ma5=np.mean(c[i-4:i+1]);ma10=np.mean(c[i-9:i+1]);ma20=np.mean(c[i-19:i+1])
    ma60=np.mean(c[i-59:i+1]);ma120=np.mean(c[i-119:i+1]) if i>=119 else ma60
    f['pct_ma5']=(p/ma5-1)*100 if ma5>0 else 0;f['pct_ma10']=(p/ma10-1)*100 if ma10>0 else 0
    f['pct_ma20']=(p/ma20-1)*100 if ma20>0 else 0;f['pct_ma60']=(p/ma60-1)*100 if ma60>0 else 0
    f['pct_ma120']=(p/ma120-1)*100 if ma120>0 else 0
    mp=np.mean(c[i-24:i-4]);f['ma20_slope']=(ma20/mp-1)*100 if mp>0 else 0
    mp=np.mean(c[i-64:i-4]);f['ma60_slope']=(ma60/mp-1)*100 if mp>0 else 0
    f['ma_align']=(ma5/ma60-1)*100 if ma60>0 else 0
    f['vol_10d']=float(np.mean(v[i-9:i+1]));f['vol_60d']=float(np.mean(v[i-59:i+1]))
    f['vol_ratio']=f['vol_10d']/f['vol_60d'] if f['vol_60d']>0 else 1.0
    f['vol_ratio_5_20']=np.mean(v[i-4:i+1])/np.mean(v[i-19:i+1]) if np.mean(v[i-19:i+1])>0 else 1.0
    tr=[max(h[j]-l[j],abs(h[j]-c[j-1]),abs(l[j]-c[j-1])) for j in range(i-19,i+1)]
    f['atr20_pct']=float(np.mean(tr))/p*100 if p>0 else 0
    f['ret_1d']=(c[i]/c[i-1]-1)*100;f['ret_5d']=(c[i]/c[i-4]-1)*100
    f['ret_10d']=(c[i]/c[i-9]-1)*100;f['ret_20d']=(c[i]/c[i-19]-1)*100
    f['ret_60d']=(c[i]/c[i-59]-1)*100

    # RSI14
    g=0;l_=0
    for j in range(i-13,i+1):
        d=c[j]-c[j-1]
        if d>=0: g+=d; l_+=0
        else: g+=0; l_-=d
    rs=g/l_ if l_>0 else 999
    f['rsi14']=100-100/(1+rs)

    # KDJ
    hh=max(h[i-8:i+1]);ll=min(l[i-8:i+1])
    rsv=(c[i]-ll)/(hh-ll)*100 if hh-ll>0 else 50
    k=kd['k'] if kd else 50
    d=kd['d'] if kd else 50
    f['kdj_j']=3*rsv-2*k if kd else rsv
    f['kdj_k']=k*2/3+rsv/3 if kd else 50
    f['kdj_d']=d*2/3+rsv/3 if kd else 50

    # MACD
    e12=_ema(c[i-25:i+1],12);e26=_ema(c[i-25:i+1],26)
    dif=e12[-1]-e26[-1];dea=_ema([e12[j]-e26[j] for j in range(len(e12))],9)
    f['macd_dif']=dif;f['macd_dea']=dea[-1] if len(dea)>0 else 0
    f['macd_bar']=2*(dif-(dea[-1] if len(dea)>0 else 0))

    # Bollinger
    sma=np.mean(c[i-19:i+1]);std=np.std(c[i-19:i+1])
    f['bb_width']=4*std/sma*100 if sma>0 else 0
    f['bb_position']=(p-(sma-2*std))/(4*std)*100 if std>0 else 50

    # OBV & others
    obv=[v[0]]
    for j in range(1,i+1):
        if c[j]>c[j-1]: obv.append(obv[-1]+v[j])
        elif c[j]<c[j-1]: obv.append(obv[-1]-v[j])
        else: obv.append(obv[-1])
    obv5=np.mean(obv[i-4:i+1]);obv20=np.mean(obv[i-19:i+1])
    f['obv_ratio_5_20']=obv5/obv20 if obv20>0 else 1
    f['ret5_max']=max([c[i]/c[max(0,i-4):i+1][j]-1 for j in range(5)]+[0])*100
    ema12=_ema(c[max(0,i-11):i+1],12)
    f['ret3_vs_ema12']=(c[i]/ema12[-1]-1)*100 if len(ema12)>0 else 0
    f['accel_5_10']=(c[i]/c[i-4]-c[i-5]/c[i-9])*100 if i>=9 else 0
    f['ma5_ma10_cross']=1 if ma5>ma10 else (-1 if ma5<ma10 else 0)
    f['vol_breakout']=v[i]/f['vol_10d'] if f['vol_10d']>0 else 1
    return f

# ── 构建数据集 ──
print("Building features...", flush=True)
X3, Xm, y = [], [], []
t0 = time.time()
for code in sample:
    d = hist[code]; c, h, l, o, v = d['c'], d['h'], d['l'], d['o'], d['v']
    dates = d['dates']
    ts = code_to_ts[code]
    kd = {}
    for i in range(199, len(c) - H):
        # 技术特征
        f = compute_tech(c, h, l, o, v, i, kd)
        kd['k']=f['kdj_k']; kd['d']=f['kdj_d']

        # 标签
        future_ret = (c[i+H-1]/c[i]-1)*100
        if abs(future_ret) > 200: continue  # 异常

        # 资金流特征（如果今日有数据）
        date_str = dates[i]
        mf_key = (ts, date_str)
        mf_vals = mf_dict.get(mf_key)

        if mf_vals is not None:
            X3.append([f[k] for k in FEATURES])
            Xm.append([f[k] for k in FEATURES] + mf_vals)
            y.append(future_ret)
        else:
            # 无MF数据时用零填充（维度与feat_names一致）
            X3.append([f[k] for k in FEATURES])
            Xm.append([f[k] for k in FEATURES] + [0.0]*len(feat_names))
            y.append(future_ret)

    if (sample.index(code)+1) % 30 == 0:
        print(f"  {sample.index(code)+1}/{len(sample)} stocks, {time.time()-t0:.0f}s", flush=True)

X3 = np.array(X3, dtype=np.float32)
Xm = np.array(Xm, dtype=np.float32)
y = np.array(y, dtype=np.float32)
n = len(X3)
split = int(n * 0.8)
print(f"  Total samples: {n}, time: {time.time()-t0:.0f}s", flush=True)
print(f"  Split: train={split}, test={n-split}", flush=True)
idx = np.random.permutation(n)
X3_tr, X3_te = X3[idx[:split]], X3[idx[split:]]
Xm_tr, Xm_te = Xm[idx[:split]], Xm[idx[split:]]
y_tr, y_te = y[idx[:split]], y[idx[split:]]

# ── 训练 + 评估 ──
def evaluate(X_tr, X_te, y_tr, y_te, name):
    m = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=2, reg_alpha=1,
        random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_tr, y_tr)
    p_te = m.predict(X_te)
    r2 = 1 - np.var(y_te-p_te)/np.var(y_te)
    corr = float(np.corrcoef(p_te, y_te)[0,1])

    def bin5(a):
        if a > 5: return 1
        if a > 2: return 2
        if a >= -2: return 3
        if a >= -5: return 4
        return 5
    bins = np.array([bin5(a) for a in y_te])
    n_te = len(y_te); d = n_te // 10
    order = np.argsort(-p_te)
    top10 = order[:d]; bot10 = order[-d:]
    d1_top = (bins[top10]==1).sum()/d*100
    d5_top = (bins[top10]==5).sum()/d*100
    d1_bot = (bins[bot10]==1).sum()/d*100
    d5_bot = (bins[bot10]==5).sum()/d*100
    avg_top = float(np.mean(y_te[top10]))
    avg_bot = float(np.mean(y_te[bot10]))
    try:
        imp = m.get_booster().get_score(importance_type='weight')
    except Exception:
        imp = {}
    top5 = sorted(imp.items(), key=lambda x:-x[1])[:5] if imp else []
    imp_sorted = sorted(imp.items(), key=lambda x:-x[1]) if imp else []
    mf_in_top20 = sum(1 for k,v in imp_sorted[:20] if any(f in k for f in ['net_','lg_','elg_','md_','major_','ret20d'])) if imp else 0

    print(f"\n{name}:", flush=True)
    print(f"  R²={r2:.4f} corr={corr:.3f}", flush=True)
    print(f"  D1评分段:D1大涨={d1_top:.1f}% D5大跌={d5_top:.1f}%", flush=True)
    print(f"  D10评分段:D1大涨={d1_bot:.1f}% D5大跌={d5_bot:.1f}%", flush=True)
    print(f"  Top10%={avg_top:+.2f}% Bot10%={avg_bot:+.2f}% Spread={avg_top-avg_bot:+.2f}%", flush=True)
    print(f"  特征Top5:{[(k,float(v)) for k,v in top5]}", flush=True)
    print(f"  资金流因子进入Top20: {mf_in_top20}个", flush=True)
    return {
        'd1_accuracy': float(d1_top), 'd5_risk': float(d5_top),
        'r2': r2, 'corr': corr,
        'top10_avg_return': avg_top, 'bot10_avg_return': avg_bot,
        'spread': avg_top - avg_bot,
        'mf_in_top20': mf_in_top20,
        'top5_features': [(k, float(v)) for k,v in top5],
    }

try:
    r3 = evaluate(X3_tr, X3_te, y_tr, y_te, "A3(33技术)")
except Exception as e:
    print(f"A3训练失败: {e}", flush=True)
    r3 = {'error': str(e)}

try:
    rm = evaluate(Xm_tr, Xm_te, y_tr, y_te, "A3+MF(33技术+21资金)")
except Exception as e:
    print(f"A3+MF训练失败: {e}", flush=True)
    rm = {'error': str(e)}

improve = rm.get('d1_accuracy', 0) - r3.get('d1_accuracy', 0)
result = {
    'horizon': H,
    'method': 'train_full_model',
    'A3_33tech': r3,
    'A3_plus_MF': rm,
    'd1_improvement_pp': float(improve),
    'verdict': 'add_moneyflow' if improve > 3 else 'tech_only',
    'status': 'completed' if 'error' not in r3 and 'error' not in rm else 'partial',
}
if 'error' not in r3 and 'error' not in rm:
    print(f"\n结论:D1改善={improve:+.1f}pp → {'✅加入资金流' if improve>3 else '❌资金流收益不大'}", flush=True)
else:
    print(f"\n部分失败: A3={r3.get('error','OK')}, A3+MF={rm.get('error','OK')}", flush=True)

with open('data/a3_mf_result.json','w') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"结果写至 data/a3_mf_result.json", flush=True)
print(f"总耗时: {time.time()-t_start:.0f}s", flush=True)
