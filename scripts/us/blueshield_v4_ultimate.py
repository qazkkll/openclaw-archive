# -*- coding: utf-8 -*-
"""
蓝盾V4 终极优化 v2 — 修复多持有期 + 新方向
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
print("蓝盾V4 终极优化 v2")
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
    g['vol_chg_5_20']=g['vol_5d']/(g['vol_20d']+0.001)-1
    sma20=cs.rolling(20).mean(); std20=cs.rolling(20).std()
    bb_w=(4*std20/sma20).values
    g['bb_width']=bb_w
    g['bb_squeeze']=bb_w / pd.Series(bb_w).rolling(60).mean().replace(0,0.001).values
    g['ret_skew_20']=dr.rolling(20).skew().values
    g['ret_kurt_20']=dr.rolling(20).kurt().values
    return g

groups = []
for sym, grp in df.groupby('sym'):
    groups.append(feats(grp))
df = pd.concat(groups, ignore_index=True)
print(f"  特征: {time.time()-t0:.1f}s")

# ════════════════════════════════════════
#  2. 截面排名 + 多持有期标签
# ════════════════════════════════════════
print("\n[2/4] 截面排名+标签...")
t0 = time.time()
rank_all = ['ret_1d','ret_5d','ret_20d','vol_20d','rsi_14','macd_hist',
            'tq_10','tq_20','vol_ratio_20','atr_pct','ma_20_ratio','ma_50_ratio',
            'ret_quality_10','ret_quality_20','bb_width','price_vol_div','mom_accel']
cs_cols = []
for f in rank_all:
    if f in df.columns:
        col = f'cs_{f}'
        df[col] = df.groupby('date')[f].rank(pct=True)
        cs_cols.append(col)

for days in [3, 5, 10, 15]:
    df[f'fwd_{days}d_ret'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-days)/x-1)
print(f"  截面: {len(cs_cols)} | 标签: 3d/5d/10d/15d ({time.time()-t0:.1f}s)")

# ════════════════════════════════════════
#  3. Walk-Forward
# ════════════════════════════════════════
print("\n[3/4] Walk-Forward...")

def eval_topn(test_df, pred, label_col, top_n=15):
    """修复：使用动态label_col"""
    tc = test_df.copy(); tc['pred'] = pred
    daily = []
    for d, day in tc.groupby('date'):
        if len(day) < top_n: continue
        top = day.nlargest(top_n, 'pred')
        daily.append({'r': top[label_col].mean(),
                      'w': (top[label_col]>0).mean()})
    if not daily: return None
    tdf = pd.DataFrame(daily)
    geo = np.exp(np.log(1+tdf['r']).mean())-1
    hold_days = int(label_col.split('_')[1].replace('d',''))
    ann_factor = 252 / hold_days
    ann = geo * ann_factor
    sh = tdf['r'].mean()/max(tdf['r'].std(),0.001)*np.sqrt(ann_factor)
    dd = (1+tdf['r']).cumprod(); dd_max=(dd/dd.cummax()-1).min()
    return {'ann':ann,'sharpe':sh,'dd':dd_max,'wr':tdf['w'].mean(),'n':len(tdf)}

def wf(data, feat_cols, label_col, top_n=15, params=None):
    cols = feat_cols + ['date', label_col]
    sub = data[cols].dropna().copy()
    if len(sub) < 10000: return None
    dates = sorted(sub['date'].unique()); n = len(dates); step = n // 5
    if params is None:
        params = {'device':'cuda','tree_method':'hist','max_depth':6,
                  'learning_rate':0.03,'subsample':0.8,'colsample_bytree':0.7,
                  'reg_alpha':0.1,'min_child_weight':10,'verbosity':0}
    results = []
    for i in range(4):
        tr_end=(i+1)*step; te_end=min((i+2)*step,n-1)
        if te_end<=tr_end: continue
        tr=sub[sub['date']<=dates[tr_end]]
        te=sub[(sub['date']>dates[tr_end])&(sub['date']<=dates[te_end])]
        if len(te)<1000: continue
        dtrain=xgb.DMatrix(tr[feat_cols].values,label=tr[label_col].values)
        dtest=xgb.DMatrix(te[feat_cols].values)
        m=xgb.train(params,dtrain,num_boost_round=500,
                     evals=[(dtrain,'train')],early_stopping_rounds=50,verbose_eval=False)
        pred=m.predict(dtest,iteration_range=(0,m.best_iteration+1))
        r=eval_topn(te,pred,label_col,top_n)
        if r: results.append(r)
    if not results: return None
    return {k:np.mean([r[k] for r in results]) for k in results[0]}

# ════════════════════════════════════════
#  4. 实验
# ════════════════════════════════════════
exps = [
    # 持有期对比
    ("CS16 3d", cs_cols, 'fwd_3d_ret', 15, None),
    ("CS16 5d", cs_cols, 'fwd_5d_ret', 15, None),
    ("CS16 10d", cs_cols, 'fwd_10d_ret', 15, None),
    ("CS16 15d", cs_cols, 'fwd_15d_ret', 15, None),
    
    # Top-N对比（5d持有期）
    ("CS16 Top-10 5d", cs_cols, 'fwd_5d_ret', 10, None),
    ("CS16 Top-20 5d", cs_cols, 'fwd_5d_ret', 20, None),
    ("CS16 Top-25 5d", cs_cols, 'fwd_5d_ret', 25, None),
    
    # 最优持有期 × Top-N
    ("CS16 Top-10 10d", cs_cols, 'fwd_10d_ret', 10, None),
    ("CS16 Top-15 10d", cs_cols, 'fwd_10d_ret', 15, None),
    ("CS16 Top-20 10d", cs_cols, 'fwd_10d_ret', 20, None),
    
    # 参数微调
    ("CS16 depth4 5d", cs_cols, 'fwd_5d_ret', 15, 
     {'device':'cuda','tree_method':'hist','max_depth':4,'learning_rate':0.05,
      'subsample':0.8,'colsample_bytree':0.7,'reg_alpha':0.1,'min_child_weight':20,'verbosity':0}),
    ("CS16 depth5 5d", cs_cols, 'fwd_5d_ret', 15,
     {'device':'cuda','tree_method':'hist','max_depth':5,'learning_rate':0.03,
      'subsample':0.8,'colsample_bytree':0.7,'reg_alpha':0.1,'min_child_weight':15,'verbosity':0}),
]

all_r = []
for i, (name, feat_list, label, tn, params) in enumerate(exps):
    feat_list = [f for f in feat_list if f in df.columns]
    t1 = time.time()
    print(f"  [{i+1}/{len(exps)}] {name} ({len(feat_list)}d)...", end=' ', flush=True)
    try:
        r = wf(df, feat_list, label, tn, params)
        dt = time.time()-t1
        if r:
            r['name'] = name; r['n_feat'] = len(feat_list); r['top_n'] = tn
            r['label'] = label
            all_r.append(r)
            tag = "🏆" if r['sharpe']>=1.20 else "  "
            print(f"夏普{r['sharpe']:.3f} 年化{r['ann']*100:+.1f}% DD{r['dd']*100:.1f}% 胜{r['wr']*100:.1f}% {tag}({dt:.0f}s)")
        else:
            print(f"无结果({dt:.0f}s)")
    except Exception as e:
        print(f"失败: {e}")

# 汇总
print(f"\n{'='*95}")
print("📊 终极优化 v2 结果")
print(f"{'='*95}")
print(f"{'#':>2} {'实验':<30} {'维':>4} {'TN':>3} {'标签':>8} {'夏普':>7} {'年化':>8} {'DD':>8} {'胜率':>6}")
print("-"*90)
for i,r in enumerate(sorted(all_r,key=lambda x:-x['sharpe'])):
    tag = "🏆" if r['sharpe']>=1.20 else "  "
    print(f"{i+1:>2} {r['name']:<30} {r['n_feat']:>4} {r['top_n']:>3} {r['label']:>8} "
          f"{r['sharpe']:>7.3f} {r['ann']*100:>+7.1f}% {r['dd']*100:>7.1f}% {r['wr']*100:>5.1f}% {tag}")

if all_r:
    best = max(all_r,key=lambda x:x['sharpe'])
    print(f"\n🏆 最佳: {best['name']}")
    print(f"  夏普: {best['sharpe']:.3f} | 年化: {best['ann']*100:+.1f}% | DD: {best['dd']*100:.1f}%")
    print(f"  vs基线1.13: {best['sharpe']-1.13:+.3f}")

with open(os.path.join(OUT,'v4_ultimate_v2.json'),'w') as f:
    json.dump({'timestamp':pd.Timestamp.now().isoformat(),'results':all_r},f,indent=2,default=str)
print(f"\n保存 → analysis/v4_ultimate_v2.json")
print(f"总耗时: {time.time()-t_total:.1f}s")
print("="*90)
