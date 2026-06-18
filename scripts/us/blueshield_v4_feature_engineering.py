# -*- coding: utf-8 -*-
"""
蓝盾V4 特征工程优化 — GPU加速版 (XGBoost CUDA)
"""
import warnings, json, os, time
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data', 'us', 'us_hist_sp500_10y.parquet')
OUT = os.path.join(ROOT, 'analysis')

print("=" * 80)
print("蓝盾V4 特征工程优化 — XGBoost CUDA")
print("=" * 80)
t_total = time.time()

# ════════════════════════════════════════
#  1. 数据+特征（全向量化）
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
    cs = pd.Series(c)
    dr = cs.pct_change()

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

    # 增强
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

    dist20=(cs-sma20)/sma20
    g['ma20_dist']=dist20.values
    g['ma20_dist_chg5']=dist20.diff(5).values
    ma50_s=cs.rolling(50).mean()
    dist50=(cs-ma50_s)/ma50_s
    g['ma50_dist']=dist50.values
    g['ma50_dist_chg10']=dist50.diff(10).values

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
rank_feats = ['ret_1d','ret_5d','ret_20d','vol_20d','rsi_14','macd_hist',
              'tq_10','tq_20','vol_ratio_20','atr_pct','ma_20_ratio','ma_50_ratio',
              'ret_quality_10','ret_quality_20','bb_width','price_vol_div','mom_accel']
cs_cols = []
for f in rank_feats:
    if f in df.columns:
        col = f'cs_{f}'
        df[col] = df.groupby('date')[f].rank(pct=True)
        cs_cols.append(col)
print(f"  截面: {len(cs_cols)}个 ({time.time()-t0:.1f}s)")

# ════════════════════════════════════════
#  3. 特征分组
# ════════════════════════════════════════
base_v4 = [
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
]
base_v4 = [f for f in base_v4 if f in df.columns]

new_enhanced = [
    'tq_5','tq_10','tq_20','mom_accel','mom_accel_10_60',
    'vol_chg_5_20','vol_chg_5_60','bb_width','bb_squeeze',
    'ma20_dist','ma20_dist_chg5','ma50_dist','ma50_dist_chg10',
    'ret_skew_20','ret_kurt_20',
]
new_enhanced = [f for f in new_enhanced if f in df.columns]

print(f"\n[3/4] 特征分组:")
print(f"  原V4: {len(base_v4)}维 | 增强: {len(new_enhanced)}维 | 截面: {len(cs_cols)}维")

# ════════════════════════════════════════
#  4. Walk-Forward (XGBoost CUDA)
# ════════════════════════════════════════
print(f"\n[4/4] Walk-Forward (XGBoost CUDA)...")

XGB_PARAMS = {
    'device': 'cuda',
    'tree_method': 'hist',
    'max_depth': 6,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'min_child_weight': 10,
    'verbosity': 0,
    'nthread': -1,
}

def eval_top15(test_df, pred):
    tc = test_df.copy()
    tc['pred'] = pred
    daily = []
    for d, day in tc.groupby('date'):
        if len(day) < 15: continue
        top15 = day.nlargest(15, 'pred')
        daily.append({'r': top15['fwd_5d_ret'].mean(),
                      'w': (top15['fwd_5d_ret']>0).mean()})
    if not daily: return None
    tdf = pd.DataFrame(daily)
    geo = np.exp(np.log(1+tdf['r']).mean())-1
    ann = geo*252/5
    sh = tdf['r'].mean()/max(tdf['r'].std(),0.001)*np.sqrt(252/5)
    dd = (1+tdf['r']).cumprod(); dd_max=(dd/dd.cummax()-1).min()
    return {'ann':ann,'sharpe':sh,'dd':dd_max,'wr':tdf['w'].mean(),'n':len(tdf)}

def wf(data, feat_cols, name):
    cols = feat_cols + ['date','fwd_5d_ret']
    sub = data[cols].dropna().copy()
    if len(sub) < 10000: return None
    dates = sorted(sub['date'].unique())
    n = len(dates)
    step = n // 5

    results = []
    for i in range(4):
        tr_end = (i+1)*step
        te_end = min((i+2)*step, n-1)
        if te_end <= tr_end: continue
        tr = sub[sub['date']<=dates[tr_end]]
        te = sub[(sub['date']>dates[tr_end])&(sub['date']<=dates[te_end])]
        if len(te) < 1000: continue

        dtrain = xgb.DMatrix(tr[feat_cols].values, label=tr['fwd_5d_ret'].values)
        dtest = xgb.DMatrix(te[feat_cols].values)
        m = xgb.train(XGB_PARAMS, dtrain, num_boost_round=500,
                      evals=[(dtrain,'train')], early_stopping_rounds=50,
                      verbose_eval=False)
        pred = m.predict(dtest, iteration_range=(0, m.best_iteration+1))
        r = eval_top15(te, pred)
        if r: results.append(r)

    if not results: return None
    avg = {k:np.mean([r[k] for r in results]) for k in results[0]}
    return {'name':name,'folds':len(results),**avg}

# 实验
exps = [
    ("① 原V4基线", base_v4),
    ("② 仅增强特征", new_enhanced),
    ("③ 原V4+增强", base_v4 + new_enhanced),
    ("④ 原V4+截面排名", base_v4 + cs_cols),
    ("⑤ 原V4+增强+截面", base_v4 + new_enhanced + cs_cols),
    ("⑥ 仅截面排名", cs_cols),
]

print(f"\n{'='*90}")
all_r = []
for name, feat_list in exps:
    feat_list = [f for f in feat_list if f in df.columns]
    t1 = time.time()
    print(f"  [{name}] ({len(feat_list)}维)...", end=' ', flush=True)
    try:
        r = wf(df, feat_list, name)
        dt = time.time()-t1
        if r:
            all_r.append(r)
            tag = "🏆" if r['sharpe']>=1.13 else "  "
            print(f"夏普{r['sharpe']:.3f} 年化{r['ann']*100:+.1f}% DD{r['dd']*100:.1f}% 胜{r['wr']*100:.1f}% {tag}({dt:.0f}s)")
        else:
            print(f"无结果({dt:.0f}s)")
    except Exception as e:
        print(f"失败: {e}")
        import traceback; traceback.print_exc()

# 汇总
print(f"\n{'='*90}")
print("📊 结果汇总")
print(f"{'='*90}")
print(f"{'#':>2} {'实验':<35} {'维':>4} {'夏普':>7} {'年化':>8} {'DD':>8} {'胜率':>6}")
print("-"*75)
for i,r in enumerate(sorted(all_r,key=lambda x:-x['sharpe'])):
    nf = len([f for nm,f in exps if nm==r['name']][0])
    tag = "🏆" if r['sharpe']>=1.13 else "  "
    print(f"{i+1:>2} {r['name']:<35} {nf:>4} {r['sharpe']:>7.3f} {r['ann']*100:>+7.1f}% {r['dd']*100:>7.1f}% {r['wr']*100:>5.1f}% {tag}")

if all_r:
    best = max(all_r,key=lambda x:x['sharpe'])
    delta = best['sharpe']-1.13
    print(f"\n🏆 最佳: {best['name']}")
    print(f"  夏普: {best['sharpe']:.3f} (vs基线1.13, {delta:+.3f})")

with open(os.path.join(OUT,'v4_feature_engineering_v2.json'),'w') as f:
    json.dump({'timestamp':pd.Timestamp.now().isoformat(),'results':all_r},f,indent=2,default=str)
print(f"\n保存 → analysis/v4_feature_engineering_v2.json")
print(f"总耗时: {time.time()-t_total:.1f}s")
print("="*90)
