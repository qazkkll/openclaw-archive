# -*- coding: utf-8 -*-
"""
蓝盾V4 特征工程优化 第二轮
基于第一轮发现：截面排名特征是最强信号源
本轮目标：
1. 截面特征逐一贡献度分析
2. 最优特征组合搜索
3. XGB vs LGB对比
4. Top-N + 持有期网格搜索
"""
import warnings, json, os, time
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data', 'us', 'us_hist_sp500_10y.parquet')
OUT = os.path.join(ROOT, 'analysis')

print("=" * 90)
print("蓝盾V4 特征工程优化 — 第二轮")
print("=" * 90)
t_total = time.time()

# ════════════════════════════════════════
#  1. 数据+全部特征（一次算完）
# ════════════════════════════════════════
print("\n[1/5] 数据+特征...")
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
#  2. 截面排名（更多维度）
# ════════════════════════════════════════
print("\n[2/5] 截面排名...")
t0 = time.time()
rank_all = [
    'ret_1d','ret_5d','ret_20d',
    'vol_20d','rsi_14','macd_hist',
    'tq_10','tq_20',
    'vol_ratio_20','atr_pct',
    'ma_20_ratio','ma_50_ratio',
    'ret_quality_10','ret_quality_20',
    'bb_width','price_vol_div','mom_accel',
    'macd','macd_sig',
    'vol_chg_5_20',
    'bb_squeeze',
    'ma20_dist',
]
cs_cols = {}
for f in rank_all:
    if f in df.columns:
        col = f'cs_{f}'
        df[col] = df.groupby('date')[f].rank(pct=True)
        cs_cols[f] = col
print(f"  截面特征: {len(cs_cols)}个 ({time.time()-t0:.1f}s)")

# ════════════════════════════════════════
#  3. 特征分组定义
# ════════════════════════════════════════
print("\n[3/5] 特征分组...")

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

all_cs = list(cs_cols.values())
all_enhanced = [
    'tq_5','tq_10','tq_20','mom_accel','mom_accel_10_60',
    'vol_chg_5_20','vol_chg_5_60','bb_width','bb_squeeze',
    'ma20_dist','ma20_dist_chg5','ma50_dist','ma50_dist_chg10',
    'ret_skew_20','ret_kurt_20',
]
all_enhanced = [f for f in all_enhanced if f in df.columns]

print(f"  原V4: {len(base_v4)} | 截面: {len(all_cs)} | 增强: {len(all_enhanced)}")

# ════════════════════════════════════════
#  4. Walk-Forward 引擎
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

def wf_xgb(data, feat_cols, top_n=15, n_rounds=500):
    cols = feat_cols + ['date','fwd_5d_ret']
    sub = data[cols].dropna().copy()
    if len(sub) < 10000: return None
    dates = sorted(sub['date'].unique()); n = len(dates); step = n // 5

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
        dtrain=xgb.DMatrix(tr[feat_cols].values,label=tr['fwd_5d_ret'].values)
        dtest=xgb.DMatrix(te[feat_cols].values)
        m=xgb.train(params,dtrain,num_boost_round=n_rounds,
                     evals=[(dtrain,'train')],early_stopping_rounds=50,verbose_eval=False)
        pred=m.predict(dtest,iteration_range=(0,m.best_iteration+1))
        r=eval_topn(te,pred,top_n)
        if r: results.append(r)
    if not results: return None
    avg={k:np.mean([r[k] for r in results]) for k in results[0]}
    return avg

def wf_lgb(data, feat_cols, top_n=15, n_rounds=500):
    cols = feat_cols + ['date','fwd_5d_ret']
    sub = data[cols].dropna().copy()
    if len(sub) < 10000: return None
    dates = sorted(sub['date'].unique()); n = len(dates); step = n // 5

    params = {'objective':'regression','metric':'rmse',
              'num_leaves':63,'learning_rate':0.03,
              'feature_fraction':0.5,'bagging_fraction':0.7,
              'lambda_l1':0.1,'lambda_l2':1.0,'verbose':-1,'n_jobs':-1}
    results = []
    for i in range(4):
        tr_end=(i+1)*step; te_end=min((i+2)*step,n-1)
        if te_end<=tr_end: continue
        tr=sub[sub['date']<=dates[tr_end]]
        te=sub[(sub['date']>dates[tr_end])&(sub['date']<=dates[te_end])]
        if len(te)<1000: continue
        dtrain=lgb.Dataset(tr[feat_cols].values,label=tr['fwd_5d_ret'].values)
        dval=lgb.Dataset(te[feat_cols].values,label=te['fwd_5d_ret'].values)
        m=lgb.train(params,dtrain,num_boost_round=n_rounds,
                     valid_sets=[dval],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
        pred=m.predict(te[feat_cols].values,num_iteration=m.best_iteration)
        r=eval_topn(te,pred,top_n)
        if r: results.append(r)
    if not results: return None
    avg={k:np.mean([r[k] for r in results]) for k in results[0]}
    return avg

# ════════════════════════════════════════
#  5. 实验矩阵
# ════════════════════════════════════════
print("\n[4/5] 实验...")
print(f"\n{'='*95}")

exps = []

# A. 截面特征逐一贡献度（加法分析）
print("  A. 截面特征加法分析...")
cs_rank_list = list(cs_cols.values())
# 按重要性排序（用第一轮的结果）
cs_importance_order = [
    'cs_ret_5d','cs_vol_20d','cs_rsi_14','cs_macd_hist',
    'cs_tq_10','cs_tq_20','cs_vol_ratio_20','cs_atr_pct',
    'cs_ma_20_ratio','cs_ma_50_ratio','cs_ret_quality_10','cs_ret_quality_20',
    'cs_bb_width','cs_price_vol_div','cs_mom_accel','cs_ret_1d','cs_ret_20d',
]
cumulative = []
for cs_f in cs_importance_order:
    if cs_f in all_cs:
        cumulative.append(cs_f)
        exps.append((f"CS累加{len(cumulative)}维", list(cumulative), 'xgb'))

# B. 截面特征分组测试
print("  B. 截面特征分组...")
# 动量类截面
cs_momentum = [c for c in all_cs if any(x in c for x in ['ret_','mom_','tq_'])]
# 波动/风险类截面
cs_risk = [c for c in all_cs if any(x in c for x in ['vol_','atr_','bb_','rsi_'])]
# 价格位置截面
cs_price = [c for c in all_cs if any(x in c for x in ['ma_','price_'])]

exps.append(("CS动量类", cs_momentum, 'xgb'))
exps.append(("CS风险类", cs_risk, 'xgb'))
exps.append(("CS价格类", cs_price, 'xgb'))

# C. 最优组合
print("  C. 最优组合...")
# 取加法分析中贡献最大的截面特征（前8个）
top_cs = [c for c in cs_importance_order[:8] if c in all_cs]
exps.append(("Top8截面", top_cs, 'xgb'))
exps.append(("原V4+Top8截面", base_v4 + top_cs, 'xgb'))
exps.append(("原V4+增强+Top8截面", base_v4 + all_enhanced + top_cs, 'xgb'))

# D. 全截面对比XGB vs LGB
exps.append(("全截面 XGB", all_cs, 'xgb'))
exps.append(("全截面 LGB", all_cs, 'lgb'))
exps.append(("原V4+全截面 XGB", base_v4 + all_cs, 'xgb'))
exps.append(("原V4+全截面 LGB", base_v4 + all_cs, 'lgb'))
exps.append(("原V4+增强+全截面 XGB", base_v4 + all_enhanced + all_cs, 'xgb'))
exps.append(("原V4+增强+全截面 LGB", base_v4 + all_enhanced + all_cs, 'lgb'))

# E. Top-N网格
print("  D. Top-N网格...")
best_feat = base_v4 + all_enhanced + all_cs  # 用最大特征集
for tn in [10, 15, 20]:
    for model in ['xgb', 'lgb']:
        exps.append((f"{model.upper()} Top-{tn} 5d", best_feat, model, tn))

print(f"  总实验数: {len(exps)}")

# 运行
all_r = []
for i, exp in enumerate(exps):
    if len(exp) == 4:
        name, feat_list, model, tn = exp
    else:
        name, feat_list, model = exp
        tn = 15
    feat_list = [f for f in feat_list if f in df.columns]
    t1 = time.time()
    print(f"  [{i+1}/{len(exps)}] {name} ({len(feat_list)}维, Top-{tn})...", end=' ', flush=True)
    try:
        if model == 'lgb':
            r = wf_lgb(df, feat_list, top_n=tn)
        else:
            r = wf_xgb(df, feat_list, top_n=tn)
        dt = time.time()-t1
        if r:
            r['name'] = name
            r['model'] = model
            r['n_feat'] = len(feat_list)
            r['top_n'] = tn
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
print("📊 第二轮完整结果（按夏普排序）")
print(f"{'='*95}")
print(f"{'#':>2} {'实验':<38} {'模型':>4} {'维':>4} {'TN':>3} {'夏普':>7} {'年化':>8} {'DD':>8} {'胜率':>6}")
print("-"*90)
for i,r in enumerate(sorted(all_r,key=lambda x:-x['sharpe'])):
    tag = "🏆" if r['sharpe']>=1.13 else "  "
    print(f"{i+1:>2} {r['name']:<38} {r['model']:>4} {r['n_feat']:>4} {r['top_n']:>3} "
          f"{r['sharpe']:>7.3f} {r['ann']*100:>+7.1f}% {r['dd']*100:>7.1f}% {r['wr']*100:>5.1f}% {tag}")

if all_r:
    best = max(all_r,key=lambda x:x['sharpe'])
    delta = best['sharpe']-1.13
    print(f"\n🏆 最佳: {best['name']} ({best['model']}, Top-{best['top_n']})")
    print(f"  夏普: {best['sharpe']:.3f} | 年化: {best['ann']*100:+.1f}% | DD: {best['dd']*100:.1f}% | 胜率: {best['wr']*100:.1f}%")
    print(f"  vs基线1.13: {delta:+.3f} ({delta/1.13*100:+.1f}%)")

# 保存
with open(os.path.join(OUT,'v4_feature_round2.json'),'w') as f:
    json.dump({'timestamp':pd.Timestamp.now().isoformat(),'results':all_r},f,indent=2,default=str)
print(f"\n保存 → analysis/v4_feature_round2.json")
print(f"总耗时: {time.time()-t_total:.1f}s")
print("="*90)
