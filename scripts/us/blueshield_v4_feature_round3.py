# -*- coding: utf-8 -*-
"""
蓝盾V4 特征工程优化 第三轮（聚焦）
基于前两轮发现：
- 截面排名特征是核心：16维CS夏普1.203
- 最优截面组合：ret_5d, vol_20d, rsi_14, macd_hist, tq_10, tq_20, vol_ratio_20, atr_pct,
                ma_20_ratio, ma_50_ratio, ret_quality_10, ret_quality_20, bb_width, price_vol_div, mom_accel, ret_1d
- 基线XGB 0.976 vs 最佳1.203

本轮目标：
1. 精选最优CS子集
2. CS+原V4最优组合
3. 参数优化
4. Top-N + 持有期精细搜索
"""
import warnings, json, os, time
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data', 'us', 'us_hist_sp500_10y.parquet')
OUT = os.path.join(ROOT, 'analysis')

print("=" * 90)
print("蓝盾V4 特征工程优化 — 第三轮（聚焦）")
print("=" * 90)
t_total = time.time()

# ════════════════════════════════════════
#  1. 数据+特征
# ════════════════════════════════════════
print("\n[1/4] 数据+特征...")
t0 = time.time()
df = pd.read_parquet(DATA)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['sym', 'date']).reset_index(drop=True)

def feats(g):
    c = g['close'].values.astype(np.float64)
    h = g['high'].values.astype(np.float64)
    l = g['low'].values.astype(np.float64)
    v = g['volume'].values.astype(np.float64)
    cs = pd.Series(c); dr = cs.pct_change()
    for d in [1,2,3,5,10,20,60]:
        g[f'ret_{d}d'] = cs.pct_change(d).values
    for w in [5,10,20,50,120]:
        g[f'ma_{w}_ratio'] = (c / cs.rolling(w).mean().values)
    ma5=cs.rolling(5).mean(); ma20=cs.rolling(20).mean()
    ma50=cs.rolling(50).mean(); ma120=cs.rolling(120).mean()
    g['ma5_ma20_cross']=((ma5>ma20).astype(float)).values
    g['ma20_ma50_cross']=((ma20>ma50).astype(float)).values
    g['ma50_ma120_cross']=((ma50>ma120).astype(float)).values
    g['ma_align_score']=g['ma5_ma20_cross']+g['ma20_ma50_cross']+g['ma50_ma120_cross']
    for w in [5,10,20,60]:
        g[f'vol_{w}d'] = dr.rolling(w).std().values
    v5=dr.rolling(5).std(); v20=dr.rolling(20).std(); v60=dr.rolling(60).std()
    g['vol_ratio_5_20']=(v5/v20.replace(0,0.001)).values
    g['vol_ratio_5_60']=(v5/v60.replace(0,0.001)).values
    g['vol_regime']=np.where(g['vol_20d']>g['vol_20d'].rolling(60).mean().values,1.0,0.0)
    delta=cs.diff(); gain=delta.where(delta>0,0).rolling(14).mean()
    loss=(-delta.where(delta<0,0)).rolling(14).mean()
    rs=gain/loss.replace(0,0.001)
    g['rsi_14']=(100-100/(1+rs)).values
    g['rsi_50_pct']=((g['rsi_14']-50)/50)
    ema12=cs.ewm(span=12).mean(); ema26=cs.ewm(span=26).mean()
    g['macd']=(ema12-ema26).values
    g['macd_sig']=g['macd'].ewm(span=9).mean().values
    g['macd_hist']=(g['macd']-g['macd_sig'])
    g['macd_cross']=((g['macd']>g['macd_sig']).astype(float))
    vs5=pd.Series(v).rolling(5).mean(); vs20=pd.Series(v).rolling(20).mean()
    g['vol_ratio_5']=(v/vs5.values); g['vol_ratio_20']=(v/vs20.values)
    g['price_vol_div']=(g['ret_5d']*-1*(g['vol_ratio_5']-1))
    for w in [20,50,100,252]:
        hh=pd.Series(h).rolling(w,min_periods=20).max().values
        ll=pd.Series(l).rolling(w,min_periods=20).min().values
        rng=np.where(hh-ll==0,0.001,hh-ll)
        g[f'price_pos_{w}']=(c-ll)/rng
    tr=np.maximum(h-l,np.maximum(abs(h-np.roll(c,1)),abs(l-np.roll(c,1))))
    tr[0]=h[0]-l[0]
    g['atr_pct']=(pd.Series(tr).rolling(20).mean()/c*100).values
    dv=c*v; dm5=pd.Series(dv).rolling(5).mean()
    g['dvol_ratio']=np.where(dm5.values>0,dv/dm5.values,1.0)
    rm10=dr.rolling(10).mean(); rs10=dr.rolling(10).std()
    g['ret_quality_10']=(rm10/rs10.replace(0,0.001)).values
    rm20=dr.rolling(20).mean(); rs20=dr.rolling(20).std()
    g['ret_quality_20']=(rm20/rs20.replace(0,0.001)).values
    g['trend_strength']=(g['ma5_ma20_cross']*2+g['ma20_ma50_cross']*3+g['ma50_ma120_cross']*4)
    g['tq_5']=g['ret_5d']/(g['vol_5d']+0.001)
    g['tq_10']=g['ret_10d']/(g['vol_10d']+0.001)
    g['tq_20']=g['ret_20d']/(g['vol_20d']+0.001)
    g['mom_accel']=g['ret_5d']/(g['ret_20d'].abs()+0.001)
    g['mom_accel_10_60']=g['ret_10d']/(g['ret_60d'].abs()+0.001)
    g['vol_chg_5_20']=g['vol_5d']/(g['vol_20d']+0.001)-1
    g['vol_chg_5_60']=g['vol_5d']/(g['vol_60d']+0.001)-1
    sma20=cs.rolling(20).mean(); std20=cs.rolling(20).std()
    bb_w=(4*std20/sma20).values
    g['bb_width']=bb_w
    g['bb_squeeze']=bb_w / pd.Series(bb_w).rolling(60).mean().replace(0,0.001).values
    dist20=(cs-sma20)/sma20; g['ma20_dist']=dist20.values; g['ma20_dist_chg5']=dist20.diff(5).values
    ma50_s=cs.rolling(50).mean(); dist50=(cs-ma50_s)/ma50_s
    g['ma50_dist']=dist50.values; g['ma50_dist_chg10']=dist50.diff(10).values
    g['ret_skew_20']=dr.rolling(20).skew().values
    g['ret_kurt_20']=dr.rolling(20).kurt().values
    return g

groups = []
for sym, grp in df.groupby('sym'):
    groups.append(feats(grp))
df = pd.concat(groups, ignore_index=True)
df['fwd_5d_ret'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-5)/x-1)
print(f"  特征: {time.time()-t0:.1f}s")

# ════════════════════════════════════════
#  2. 截面排名
# ════════════════════════════════════════
print("\n[2/4] 截面排名...")
t0 = time.time()
rank_all = ['ret_1d','ret_5d','ret_20d','vol_20d','rsi_14','macd_hist',
            'tq_10','tq_20','vol_ratio_20','atr_pct','ma_20_ratio','ma_50_ratio',
            'ret_quality_10','ret_quality_20','bb_width','price_vol_div','mom_accel',
            'macd','macd_sig','vol_chg_5_20','bb_squeeze','ma20_dist']
cs_cols = {}
for f in rank_all:
    if f in df.columns:
        col = f'cs_{f}'
        df[col] = df.groupby('date')[f].rank(pct=True)
        cs_cols[f] = col
all_cs = list(cs_cols.values())
print(f"  截面: {len(all_cs)}个 ({time.time()-t0:.1f}s)")

# ════════════════════════════════════════
#  3. 特征定义
# ════════════════════════════════════════
base_v4 = [f for f in [
    'ret_1d','ret_2d','ret_3d','ret_5d','ret_10d','ret_20d','ret_60d',
    'ma_5_ratio','ma_10_ratio','ma_20_ratio','ma_50_ratio','ma_120_ratio',
    'ma5_ma20_cross','ma20_ma50_cross','ma50_ma120_cross','ma_align_score',
    'vol_5d','vol_10d','vol_20d','vol_60d',
    'vol_ratio_5_20','vol_ratio_5_60','vol_regime',
    'rsi_14','rsi_50_pct',
    'macd','macd_sig','macd_hist','macd_cross',
    'vol_ratio_5','vol_ratio_20','price_vol_div',
    'price_pos_20','price_pos_50','price_pos_100','price_pos_252',
    'atr_pct','dvol_ratio','ret_quality_10','ret_quality_20','trend_strength',
] if f in df.columns]

enhanced = [f for f in [
    'tq_5','tq_10','tq_20','mom_accel','mom_accel_10_60',
    'vol_chg_5_20','vol_chg_5_60','bb_width','bb_squeeze',
    'ma20_dist','ma20_dist_chg5','ma50_dist','ma50_dist_chg10',
    'ret_skew_20','ret_kurt_20',
] if f in df.columns]

# ════════════════════════════════════════
#  4. XGB WF引擎
# ════════════════════════════════════════
def eval_topn(test_df, pred, top_n=15):
    tc = test_df.copy(); tc['pred'] = pred
    daily = []
    for d, day in tc.groupby('date'):
        if len(day) < top_n: continue
        top = day.nlargest(top_n, 'pred')
        daily.append({'r': top['fwd_5d_ret'].mean(),
                      'w': (top['fwd_5d_ret']>0).mean()})
    if not daily: return None
    tdf = pd.DataFrame(daily)
    geo = np.exp(np.log(1+tdf['r']).mean())-1
    ann = geo*252/5
    sh = tdf['r'].mean()/max(tdf['r'].std(),0.001)*np.sqrt(252/5)
    dd = (1+tdf['r']).cumprod(); dd_max=(dd/dd.cummax()-1).min()
    return {'ann':ann,'sharpe':sh,'dd':dd_max,'wr':tdf['w'].mean(),'n':len(tdf)}

def wf(data, feat_cols, top_n=15, params_override=None):
    cols = feat_cols + ['date','fwd_5d_ret']
    sub = data[cols].dropna().copy()
    if len(sub) < 10000: return None
    dates = sorted(sub['date'].unique()); n = len(dates); step = n // 5
    params = {'device':'cuda','tree_method':'hist','max_depth':6,
              'learning_rate':0.03,'subsample':0.8,'colsample_bytree':0.7,
              'reg_alpha':0.1,'min_child_weight':10,'verbosity':0}
    if params_override: params.update(params_override)
    results = []
    for i in range(4):
        tr_end=(i+1)*step; te_end=min((i+2)*step,n-1)
        if te_end<=tr_end: continue
        tr=sub[sub['date']<=dates[tr_end]]
        te=sub[(sub['date']>dates[tr_end])&(sub['date']<=dates[te_end])]
        if len(te)<1000: continue
        dtrain=xgb.DMatrix(tr[feat_cols].values,label=tr['fwd_5d_ret'].values)
        dtest=xgb.DMatrix(te[feat_cols].values)
        m=xgb.train(params,dtrain,num_boost_round=500,
                     evals=[(dtrain,'train')],early_stopping_rounds=50,verbose_eval=False)
        pred=m.predict(dtest,iteration_range=(0,m.best_iteration+1))
        r=eval_topn(te,pred,top_n)
        if r: results.append(r)
    if not results: return None
    return {k:np.mean([r[k] for r in results]) for k in results[0]}

# ════════════════════════════════════════
#  5. 实验矩阵（聚焦）
# ════════════════════════════════════════
print("\n[3/4] 实验...")
print(f"\n{'='*95}")

# 第二轮最优CS特征（按累加分析的贡献排序）
# ret_5d→vol_20d→rsi_14→macd_hist→tq_10→tq_20→vol_ratio_20→atr_pct→ma_20_ratio→
# ma_50_ratio→ret_quality_10→ret_quality_20→bb_width→price_vol_div→mom_accel→ret_1d

cs_best_order = [
    'cs_ret_5d','cs_vol_20d','cs_rsi_14','cs_macd_hist',
    'cs_tq_10','cs_tq_20','cs_vol_ratio_20','cs_atr_pct',
    'cs_ma_20_ratio','cs_ma_50_ratio','cs_ret_quality_10','cs_ret_quality_20',
    'cs_bb_width','cs_price_vol_div','cs_mom_accel','cs_ret_1d',
]
cs_best_order = [c for c in cs_best_order if c in all_cs]

exps = []

# A. 精选CS子集
for n in [3,4,5,6,8,10,12,16]:
    exps.append((f"精选CS Top-{n}", cs_best_order[:n], 15, None))

# B. CS + V4 组合（逐步添加）
for n in [3,4,5,6,8]:
    exps.append((f"V4+CS Top-{n}", base_v4 + cs_best_order[:n], 15, None))

# C. V4+增强+CS组合
for n in [3,5,8]:
    exps.append((f"V4+增强+CS Top-{n}", base_v4 + enhanced + cs_best_order[:n], 15, None))

# D. Top-N网格（用最优特征集）
best_feat = cs_best_order[:16]  # 纯CS 16维是第二轮最佳
for tn in [10, 12, 15, 20, 25]:
    exps.append((f"CS16 Top-{tn}", best_feat, tn, None))

# V4+增强+CS16 也做Top-N
combo_feat = base_v4 + enhanced + cs_best_order[:8]
for tn in [10, 15, 20]:
    exps.append((f"V4+增强+CS8 Top-{tn}", combo_feat, tn, None))

# E. 参数优化（用最佳特征集）
print("  E. 参数优化...")
param_grid = [
    ({'max_depth':4,'learning_rate':0.05,'min_child_weight':20}, "depth4_lr005"),
    ({'max_depth':5,'learning_rate':0.03,'min_child_weight':15}, "depth5_lr003"),
    ({'max_depth':6,'learning_rate':0.03,'min_child_weight':10}, "depth6_lr003_def"),
    ({'max_depth':8,'learning_rate':0.02,'min_child_weight':5}, "depth8_lr002"),
    ({'max_depth':6,'learning_rate':0.01,'subsample':0.7}, "depth6_lr001"),
    ({'max_depth':4,'learning_rate':0.01,'colsample_bytree':0.5}, "depth4_lr001_cs05"),
]
best_feats_for_tune = cs_best_order[:16]
for params, pname in param_grid:
    exps.append((f"CS16 {pname}", best_feats_for_tune, 15, params))

print(f"  总实验数: {len(exps)}")

all_r = []
for i, (name, feat_list, tn, params) in enumerate(exps):
    feat_list = [f for f in feat_list if f in df.columns]
    t1 = time.time()
    print(f"  [{i+1}/{len(exps)}] {name} ({len(feat_list)}d, T{tn})...", end=' ', flush=True)
    try:
        r = wf(df, feat_list, top_n=tn, params_override=params)
        dt = time.time()-t1
        if r:
            r['name'] = name; r['n_feat'] = len(feat_list); r['top_n'] = tn
            all_r.append(r)
            tag = "🏆" if r['sharpe']>=1.13 else "  "
            print(f"夏普{r['sharpe']:.3f} 年化{r['ann']*100:+.1f}% DD{r['dd']*100:.1f}% 胜{r['wr']*100:.1f}% {tag}({dt:.0f}s)")
        else:
            print(f"无结果({dt:.0f}s)")
    except Exception as e:
        print(f"失败: {e}")

# ════════════════════════════════════════
#  6. 汇总
# ════════════════════════════════════════
print(f"\n{'='*95}")
print("📊 第三轮完整结果（按夏普排序）")
print(f"{'='*95}")
print(f"{'#':>2} {'实验':<38} {'维':>4} {'TN':>3} {'夏普':>7} {'年化':>8} {'DD':>8} {'胜率':>6}")
print("-"*85)
for i,r in enumerate(sorted(all_r,key=lambda x:-x['sharpe'])):
    tag = "🏆" if r['sharpe']>=1.13 else "  "
    print(f"{i+1:>2} {r['name']:<38} {r['n_feat']:>4} {r['top_n']:>3} "
          f"{r['sharpe']:>7.3f} {r['ann']*100:>+7.1f}% {r['dd']*100:>7.1f}% {r['wr']*100:>5.1f}% {tag}")

if all_r:
    best = max(all_r,key=lambda x:x['sharpe'])
    delta = best['sharpe']-1.13
    print(f"\n🏆 最佳: {best['name']}")
    print(f"  夏普: {best['sharpe']:.3f} | 年化: {best['ann']*100:+.1f}% | DD: {best['dd']*100:.1f}% | 胜率: {best['wr']*100:.1f}%")
    print(f"  vs基线1.13: {delta:+.3f} ({delta/1.13*100:+.1f}%)")

# 三轮汇总
print(f"\n{'='*95}")
print("📈 三轮优化轨迹")
print(f"{'='*95}")
baseline = 1.13
print(f"  第零轮（生产基线）:     夏普 {baseline:.3f}")
round1_best = 1.154
print(f"  第一轮（基础特征工程）: 夏普 {round1_best:.3f} ({round1_best-baseline:+.3f})")
if all_r:
    round3_best = max(r['sharpe'] for r in all_r)
    print(f"  第三轮（聚焦优化）:     夏普 {round3_best:.3f} ({round3_best-baseline:+.3f})")
    print(f"  总提升: {round3_best-baseline:+.3f} ({(round3_best-baseline)/baseline*100:+.1f}%)")

with open(os.path.join(OUT,'v4_feature_round3.json'),'w') as f:
    json.dump({'timestamp':pd.Timestamp.now().isoformat(),'results':all_r},f,indent=2,default=str)
print(f"\n保存 → analysis/v4_feature_round3.json")
print(f"总耗时: {time.time()-t_total:.1f}s")
print("="*90)
