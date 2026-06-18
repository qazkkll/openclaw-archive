# -*- coding: utf-8 -*-
"""
绿箭V10 — 彩票模型升级版
定位：高波动、高收益、识别极端涨幅机会

改进方向：
1. 截面排名特征（蓝盾验证有效）
2. 从回归→分类（V9已验证）
3. 更激进的正则化（彩票需要泛化）
4. 多持有期目标（3d/5d/10d）
"""
import warnings, json, os, time
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data', 'us', 'us_hist_sp500_10y.parquet')
OUT = os.path.join(ROOT, 'analysis')
MODEL_DIR = os.path.join(ROOT, 'models', 'us')

print("=" * 90)
print("绿箭V10 — 彩票模型升级")
print("=" * 90)
t_total = time.time()

# ════════════════════════════════════════
#  1. 数据+特征
# ════════════════════════════════════════
print("\n[1/5] 数据+特征...")
t0 = time.time()
df = pd.read_parquet(DATA)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['sym', 'date']).reset_index(drop=True)

# 过滤低价股（彩票池）
df['price'] = df['close']
df = df[df['price'] < 50].copy()  # 聚焦<50的股票
print(f"  低价股: {df['sym'].nunique()}只, {len(df):,}行")

def feats(g):
    c = g['close'].values.astype(np.float64)
    h = g['high'].values.astype(np.float64)
    l = g['low'].values.astype(np.float64)
    v = g['volume'].values.astype(np.float64)
    cs = pd.Series(c); dr = cs.pct_change()
    
    # 收益率
    for d in [1,2,3,5,10,20]:
        g[f'ret_{d}d'] = cs.pct_change(d).values
    
    # 均线
    for w in [5,10,20,50]:
        g[f'ma_{w}_ratio'] = (c / cs.rolling(w).mean().values)
    
    # 均线交叉
    ma5=cs.rolling(5).mean(); ma20=cs.rolling(20).mean(); ma50=cs.rolling(50).mean()
    g['ma5_ma20_cross']=((ma5>ma20).astype(float)).values
    g['ma20_ma50_cross']=((ma20>ma50).astype(float)).values
    
    # 波动率
    for w in [5,10,20]:
        g[f'vol_{w}d'] = dr.rolling(w).std().values
    v5=dr.rolling(5).std(); v20=dr.rolling(20).std()
    g['vol_ratio_5_20']=(v5/v20.replace(0,0.001)).values
    
    # RSI
    delta=cs.diff(); gain=delta.where(delta>0,0).rolling(14).mean()
    loss=(-delta.where(delta<0,0)).rolling(14).mean()
    rs=gain/loss.replace(0,0.001)
    g['rsi_14']=(100-100/(1+rs)).values
    
    # MACD
    ema12=cs.ewm(span=12).mean(); ema26=cs.ewm(span=26).mean()
    g['macd']=(ema12-ema26).values
    g['macd_hist']=(g['macd']-g['macd'].ewm(span=9).mean().values)
    
    # 布林带
    sma20=cs.rolling(20).mean(); std20=cs.rolling(20).std()
    g['bb_width']=(4*std20/sma20).values
    g['bb_pos']=(c-sma20)/(2*std20.replace(0,1))
    
    # 成交量
    vs5=pd.Series(v).rolling(5).mean(); vs20=pd.Series(v).rolling(20).mean()
    g['vol_ratio_5']=(v/vs5.values); g['vol_ratio_20']=(v/vs20.values)
    
    # 价格位置
    for w in [20,60]:
        hh=pd.Series(h).rolling(w,min_periods=20).max().values
        ll=pd.Series(l).rolling(w,min_periods=20).min().values
        rng=np.where(hh-ll==0,0.001,hh-ll)
        g[f'price_pos_{w}']=(c-ll)/rng
    
    # ATR
    tr=np.maximum(h-l,np.maximum(abs(h-np.roll(c,1)),abs(l-np.roll(c,1))))
    tr[0]=h[0]-l[0]
    g['atr_pct']=(pd.Series(tr).rolling(14).mean()/c*100).values
    
    # 动量
    g['mom_accel']=g['ret_5d']/(g['ret_20d'].abs()+0.001)
    
    # 趋势质量
    g['tq_5']=g['ret_5d']/(g['vol_5d']+0.001)
    g['tq_10']=g['ret_10d']/(g['vol_10d']+0.001)
    
    return g

groups = []
for sym, grp in df.groupby('sym'):
    groups.append(feats(grp))
df = pd.concat(groups, ignore_index=True)
print(f"  特征: {time.time()-t0:.1f}s")

# ════════════════════════════════════════
#  2. 截面排名
# ════════════════════════════════════════
print("\n[2/5] 截面排名...")
t0 = time.time()
rank_feats = ['ret_1d','ret_5d','ret_20d','vol_20d','rsi_14','macd_hist',
              'tq_5','tq_10','vol_ratio_20','atr_pct','ma_20_ratio','mom_accel']
cs_cols = []
for f in rank_feats:
    if f in df.columns:
        col = f'cs_{f}'
        df[col] = df.groupby('date')[f].rank(pct=True)
        cs_cols.append(col)
print(f"  截面: {len(cs_cols)}个 ({time.time()-t0:.1f}s)")

# ════════════════════════════════════════
#  3. 彩票标签（二分类）
# ════════════════════════════════════════
print("\n[3/5] 彩票标签...")

# 多个阈值
for days, threshold in [(3, 0.20), (5, 0.20), (5, 0.50), (10, 0.30)]:
    col_name = f'lottery_{days}d_{int(threshold*100)}'
    df[f'fwd_{days}d_ret'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-days)/x-1)
    df[col_name] = (df[f'fwd_{days}d_ret'] >= threshold).astype(int)
    hit_rate = df[col_name].mean()
    print(f"  {col_name}: {hit_rate:.2%} hit rate")

# 最终标签：5天涨20%
TARGET = 'lottery_5d_20'
df = df.dropna(subset=[TARGET] + cs_cols)

print(f"  总样本: {len(df):,}, 正样本: {df[TARGET].sum():,} ({df[TARGET].mean():.2%})")

# ════════════════════════════════════════
#  4. Walk-Forward
# ════════════════════════════════════════
print("\n[4/5] Walk-Forward...")

def eval_lottery(test_df, pred, top_n=20, threshold=0.5):
    """彩票模型评估：Top-N命中率 + 收益"""
    tc = test_df.copy(); tc['pred'] = pred
    
    daily_stats = []
    for d, day in tc.groupby('date'):
        if len(day) < top_n: continue
        top = day.nlargest(top_n, 'pred')
        
        # 命中率
        hit_rate = (top[TARGET] > 0).mean()
        
        # 平均收益（用fwd_5d_ret）
        avg_ret = top['fwd_5d_ret'].mean() if 'fwd_5d_ret' in top.columns else 0
        
        # 最大单票收益
        max_ret = top['fwd_5d_ret'].max() if 'fwd_5d_ret' in top.columns else 0
        
        daily_stats.append({
            'hit_rate': hit_rate,
            'avg_ret': avg_ret,
            'max_ret': max_ret,
            'n_lottos': top[TARGET].sum()
        })
    
    if not daily_stats: return None
    
    tdf = pd.DataFrame(daily_stats)
    avg_hit = tdf['hit_rate'].mean()
    avg_ret = tdf['avg_ret'].mean()
    avg_max = tdf['max_ret'].mean()
    total_lottos = tdf['n_lottos'].sum()
    
    # 计算夏普（基于日收益）
    ann = np.exp(np.log(1+tdf['avg_ret']).mean()*252) - 1
    sh = tdf['avg_ret'].mean()/max(tdf['avg_ret'].std(),0.001)*np.sqrt(252)
    dd = (1+tdf['avg_ret']).cumprod()
    dd_max = (dd/dd.cummax()-1).min()
    
    return {
        'hit_rate': avg_hit,
        'ann': ann, 'sharpe': sh, 'dd': dd_max,
        'avg_ret': avg_ret, 'avg_max': avg_max,
        'total_lottos': total_lottos, 'n_days': len(tdf)
    }

def wf(data, feat_cols, top_n=20, params=None):
    cols = feat_cols + ['date', TARGET, 'fwd_5d_ret']
    sub = data[cols].dropna().copy()
    if len(sub) < 10000: return None
    dates = sorted(sub['date'].unique()); n = len(dates); step = n // 5
    
    if params is None:
        params = {'device':'cuda','tree_method':'hist','max_depth':4,
                  'learning_rate':0.03,'subsample':0.5,'colsample_bytree':0.7,
                  'reg_alpha':1.0,'min_child_weight':20,'verbosity':0,
                  'scale_pos_weight':10}  # 正样本权重
    
    results = []
    for i in range(4):
        tr_end=(i+1)*step; te_end=min((i+2)*step,n-1)
        if te_end<=tr_end: continue
        tr=sub[sub['date']<=dates[tr_end]]
        te=sub[(sub['date']>dates[tr_end])&(sub['date']<=dates[te_end])]
        if len(te)<1000: continue
        
        dtrain=xgb.DMatrix(tr[feat_cols].values,label=tr[TARGET].values)
        dtest=xgb.DMatrix(te[feat_cols].values)
        m=xgb.train(params,dtrain,num_boost_round=500,
                     evals=[(dtrain,'train')],early_stopping_rounds=50,verbose_eval=False)
        pred=m.predict(dtest,iteration_range=(0,m.best_iteration+1))
        r=eval_lottery(te,pred,top_n)
        if r: results.append(r)
    
    if not results: return None
    return {k:np.mean([r[k] for r in results]) for k in results[0]}

# ════════════════════════════════════════
#  5. 实验
# ════════════════════════════════════════
print("\n[5/5] 实验...")

exps = [
    # 纯截面特征
    ("V10 CS12", cs_cols, 20, None),
    ("V10 CS12 Top-10", cs_cols, 10, None),
    ("V10 CS12 Top-30", cs_cols, 30, None),
    
    # 不同正样本权重
    ("V10 CS12 w5", cs_cols, 20, {'device':'cuda','tree_method':'hist','max_depth':4,
        'learning_rate':0.03,'subsample':0.5,'colsample_bytree':0.7,
        'reg_alpha':1.0,'min_child_weight':20,'verbosity':0,'scale_pos_weight':5}),
    ("V10 CS12 w20", cs_cols, 20, {'device':'cuda','tree_method':'hist','max_depth':4,
        'learning_rate':0.03,'subsample':0.5,'colsample_bytree':0.7,
        'reg_alpha':1.0,'min_child_weight':20,'verbosity':0,'scale_pos_weight':20}),
    
    # 更深的模型
    ("V10 CS12 depth6", cs_cols, 20, {'device':'cuda','tree_method':'hist','max_depth':6,
        'learning_rate':0.03,'subsample':0.5,'colsample_bytree':0.7,
        'reg_alpha':1.0,'min_child_weight':10,'verbosity':0,'scale_pos_weight':10}),
]

all_r = []
for i, (name, feat_list, tn, params) in enumerate(exps):
    feat_list = [f for f in feat_list if f in df.columns]
    t1 = time.time()
    print(f"  [{i+1}/{len(exps)}] {name} ({len(feat_list)}d, Top-{tn})...", end=' ', flush=True)
    try:
        r = wf(df, feat_list, tn, params)
        dt = time.time()-t1
        if r:
            r['name'] = name; r['n_feat'] = len(feat_list); r['top_n'] = tn
            all_r.append(r)
            tag = "!" if r['hit_rate'] > 0.15 else "  "
            print(f"命中{r['hit_rate']:.1%} 夏普{r['sharpe']:.3f} 年化{r['ann']*100:+.1f}% DD{r['dd']*100:.1f}% ({dt:.0f}s) {tag}")
        else:
            print(f"无结果({dt:.0f}s)")
    except Exception as e:
        print(f"失败: {e}")

# 汇总
print(f"\n{'='*95}")
print("📊 绿箭V10结果")
print(f"{'='*95}")
print(f"{'#':>2} {'实验':<25} {'维':>4} {'TN':>3} {'命中':>6} {'夏普':>7} {'年化':>8} {'DD':>8}")
print("-"*75)
for i,r in enumerate(sorted(all_r,key=lambda x:-x['hit_rate'])):
    print(f"{i+1:>2} {r['name']:<25} {r['n_feat']:>4} {r['top_n']:>3} {r['hit_rate']:>5.1%} "
          f"{r['sharpe']:>7.3f} {r['ann']*100:>+7.1f}% {r['dd']*100:>7.1f}%")

if all_r:
    best = max(all_r,key=lambda x:x['hit_rate'])
    print(f"\n🏆 最佳命中率: {best['name']}")
    print(f"  命中: {best['hit_rate']:.1%} | 夏普: {best['sharpe']:.3f} | 年化: {best['ann']*100:+.1f}%")

with open(os.path.join(OUT,'arrow_v10_results.json'),'w') as f:
    json.dump({'timestamp':pd.Timestamp.now().isoformat(),'results':all_r},f,indent=2,default=str)
print(f"\n保存 → analysis/arrow_v10_results.json")
print(f"总耗时: {time.time()-t_total:.1f}s")
print("="*90)
