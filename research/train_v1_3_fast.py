#!/usr/bin/env python3
"""V1.3优化：灌基本面+截面排名标签（优化版，避免慢groupby）"""
import pandas as pd, numpy as np, xgboost as xgb, json, os, time, sys, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

def log(msg):
    print(msg, flush=True)

HOLD_DAYS = 10
TOP_K = 15

log("=" * 60)
log("V1.3 优化训练（基本面+截面排名标签）")
log("=" * 60)

# 1. 加载
log("\n[1] 加载数据...")
t0 = time.time()
df = pd.read_parquet('data/cn/features_v2.parquet')
df['date'] = pd.to_datetime(df['date'])
df['date_int'] = df['date'].dt.strftime('%Y%m%d').astype(int)
df = df[df['date_int'] >= 20180101].copy()  # 缩减到2018年起，避免OOM
log(f"  features_v2: {len(df):,}行 ({time.time()-t0:.0f}s)")

# merge基本面
t1 = time.time()
db = pd.read_parquet('data/cn/daily_basic.parquet')
db['sym'] = db['ts_code'].str[:6]
db['date_int'] = db['trade_date'].astype(int)
db = db[db['date_int'] >= 20180101][['sym','date_int','pe_ttm','pb','ps_ttm','dv_ratio','total_mv']].copy()
df = df.merge(db, on=['sym','date_int'], how='left')
log(f"  merge daily_basic: {time.time()-t1:.0f}s")

# 2. 特征（一次性计算所有rank，避免多次groupby）
log("\n[2] 计算特征...")
t2 = time.time()

# 基础派生
df['rev_5d'] = -df['r5']
df['rev_10d'] = -df['r10']
df['rev_20d'] = -df['r20']
df['rsi_reversal'] = -(df['rsi14'] - 50)
df['macd_reversal'] = -df['macd']
df['low_vol_5d'] = -df['vol5']
df['low_vol_20d'] = -df['vol20']
df['low_atr'] = -df['atr_pct']
df['small_cap'] = -np.log(df['circ_mv'].clip(lower=1))

# 残差动量
for col, src in [('residual_mom_5d','r5'),('residual_mom_20d','r20')]:
    df[col] = df[src] - df.groupby('date_int')[src].transform('mean')

df['lg_flow_momentum'] = df['lg_net_5'] - df['lg_net_20'] / 4
df['total_flow_momentum'] = df['total_net_5'] - df['total_net_20'] / 4

# 基本面清洗
pe = df['pe_ttm'].where((df['pe_ttm'] > 0) & (df['pe_ttm'] < 500))
pb = df['pb'].where((df['pb'] > 0) & (df['pb'] < 100))
ps = df['ps_ttm'].where((df['ps_ttm'] > 0) & (df['ps_ttm'] < 200))
df['pe_inverse'] = 1.0 / pe.clip(lower=1)
df['pb_inverse'] = 1.0 / pb.clip(lower=0.1)

# 批量rank（一次groupby搞定所有）
log("  批量rank计算...")
rank_cols = {
    'lg_net_20': 'lg_net_20_rank',
    'md_net_20': 'md_net_20_rank',
    'total_net_20': 'total_net_20_rank',
    'vol_r': 'turnover_rank',
    'pe_ttm': 'pe_rank',
    'pb': 'pb_rank',
    'dv_ratio': 'div_rank',
    'ps_ttm': 'ps_rank',
}

for src, dst in rank_cols.items():
    if src in df.columns:
        ascending = True if src in ['pe_ttm','pb','ps_ttm'] else (False if src == 'dv_ratio' else True)
        df[dst] = df.groupby('date_int')[src].rank(pct=True, ascending=ascending)
    else:
        df[dst] = 0.5

df['rev_flow_interaction'] = df['rev_20d'] * df['lg_net_20_rank']
log(f"  特征计算完成 ({time.time()-t2:.0f}s)")

features = [
    'rev_5d','rev_10d','rev_20d','rsi_reversal','macd_reversal','macd_hist',
    'low_vol_5d','low_vol_20d','low_atr',
    'md_net_5','md_net_20','lg_net_5','lg_net_20','total_net_5','total_net_20',
    'small_cap','residual_mom_5d','residual_mom_20d',
    'lg_flow_momentum','total_flow_momentum',
    'lg_net_20_rank','md_net_20_rank','total_net_20_rank',
    'rev_flow_interaction','turnover_rank',
    'pe_rank','pe_inverse','pb_rank','pb_inverse','div_rank','ps_rank',
    'vol_r','sm_net_5','sm_net_20','elg_net_5','elg_net_20',
]

for f in features:
    if f not in df.columns: df[f] = 0
    df[f] = df[f].fillna(0).replace([np.inf,-np.inf], 0)

# 市场状态
mkt = df.groupby('date_int')['close'].mean()
mkt_ma60 = mkt.rolling(60).mean()
mkt_ma120 = mkt.rolling(120).mean()
adv = df.groupby('date_int').apply(lambda x: (x['r5']>0).sum()/max(len(x),1))
mkt_score = ((mkt_ma60 > mkt_ma120).astype(int) + (mkt.rolling(20).mean() > mkt.shift(20).rolling(20).mean()).astype(int) + (adv > 0.5).astype(int))
df = df.merge(mkt_score.rename('mkt_score').reset_index(), on='date_int', how='left')

# 3. 标签
log("\n[3] 计算标签...")
t3 = time.time()
df = df.sort_values(['sym','date_int'])
df['fwd_ret'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-HOLD_DAYS)/x - 1)
df['fwd_rank'] = df.groupby('date_int')['fwd_ret'].rank(pct=True)
df_valid = df.dropna(subset=['fwd_ret']).copy()
log(f"  有效样本: {len(df_valid):,} ({time.time()-t3:.0f}s)")

# 4. Walk-Forward
log("\n[4] Walk-Forward验证...")

folds = [
    (20180101,20211231,20220101,20230630),
    (20190101,20220630,20220701,20231231),
    (20190101,20230630,20230701,20241231),
    (20200101,20240630,20240701,20260630),
]

params = {'max_depth':6,'eta':0.05,'subsample':0.8,'colsample_bytree':0.8,
          'min_child_weight':100,'objective':'reg:squarederror','tree_method':'hist'}

results = {}
for label, target in [('原始标签','fwd_ret'),('排名标签','fwd_rank')]:
    log(f"\n  === {label} ===")
    wf = []
    for fi,(tr_s,tr_e,te_s,te_e) in enumerate(folds):
        t1 = time.time()
        tr = df_valid[(df_valid['date_int']>=tr_s)&(df_valid['date_int']<=tr_e)]
        te = df_valid[(df_valid['date_int']>=te_s)&(df_valid['date_int']<=te_e)]
        if len(tr)<10000 or len(te)<1000: continue
        
        dtrain = xgb.DMatrix(tr[features].fillna(0), label=tr[target])
        dtest = xgb.DMatrix(te[features].fillna(0))
        model = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
        te = te.copy()
        te['pred'] = model.predict(dtest)
        
        ic = te.groupby('date_int').apply(lambda x: x['pred'].corr(x['fwd_ret'])).mean()
        ric = te.groupby('date_int').apply(lambda x: x['pred'].corr(x['fwd_ret'],method='spearman')).mean()
        top_ret = te.groupby('date_int').apply(lambda x: x.nlargest(TOP_K,'pred')['fwd_ret'].mean()).mean()
        bot_ret = te.groupby('date_int').apply(lambda x: x.nsmallest(TOP_K,'pred')['fwd_ret'].mean()).mean()
        ls = top_ret - bot_ret
        wf.append({'fold':fi+1,'ic':ic,'ric':ric,'ls':ls})
        log(f"    Fold{fi+1}: IC={ic:.4f} RankIC={ric:.4f} LS={ls:.4f} ({time.time()-t1:.0f}s)")
    
    results[label] = {'ic':np.mean([r['ic'] for r in wf]), 'ric':np.mean([r['ric'] for r in wf]), 'ls':np.mean([r['ls'] for r in wf])}

# 5. Paper Trade
log("\n[5] Paper Trade（排名标签+市场过滤）...")

train_f = df_valid[df_valid['date_int']<=20240630]
final_model = xgb.train(params, xgb.DMatrix(train_f[features].fillna(0), label=train_f['fwd_rank']), num_boost_round=500, verbose_eval=False)

quarter_starts = []
for year in range(2021,2027):
    for month in [1,4,7,10]:
        qdate = int(f"{year}{month:02d}01")
        cands = [d for d in sorted(df_valid['date_int'].unique()) if abs(d-qdate)<2000]
        if cands: quarter_starts.append(min(cands, key=lambda x: abs(x-qdate)))
quarter_starts = sorted(set(quarter_starts))

paper = []
for sd in quarter_starts:
    day = df_valid[df_valid['date_int']==sd].copy()
    if len(day)<50: continue
    day = day[day['close']>3]
    
    ms = day['mkt_score'].iloc[0] if 'mkt_score' in day.columns else 3
    if ms <= 1: pos, regime = 0, 'bear'
    elif ms <= 2: pos, regime = 0.5, 'cautious'
    else: pos, regime = 1.0, 'bull'
    
    if pos == 0:
        paper.append({'date':sd,'port':0,'alpha':0,'wr':0,'pos':0,'regime':regime})
        continue
    
    day['pred'] = final_model.predict(xgb.DMatrix(day[features].fillna(0)))
    top = day.nlargest(TOP_K,'pred')
    rets = top['fwd_ret'].fillna(0).tolist()
    port = np.mean(rets) * pos
    bench = day['fwd_ret'].dropna().mean()
    wr = sum(1 for r in rets if r>0)/len(rets)*100
    paper.append({'date':sd,'port':port,'alpha':port-bench,'wr':wr,'pos':pos,'regime':regime})

# 6. 汇总
log("\n[6] 结果\n")
rdf = pd.DataFrame(paper)
active = rdf[rdf['pos']>0]
alpha_pct = (active['alpha']>0).sum()/len(active)*100
cum = (1+active['port']).prod()-1
n_yrs = len(active)*HOLD_DAYS/365
ann = (1+cum)**(1/max(n_yrs,0.5))-1
sharpe = active['port'].mean()/active['port'].std()*np.sqrt(365/HOLD_DAYS) if active['port'].std()>0 else 0
dd = (1+active['port']).cumprod()
max_dd = ((dd-dd.expanding().max())/dd.expanding().max()).min()

log("="*60)
log("📊 V1.3 Walk-Forward对比:")
for l,r in results.items():
    log(f"  {l}: IC={r['ic']:.4f} RankIC={r['ric']:.4f} LS={r['ls']:.4f}")

log(f"\n📊 Paper Trade:")
log(f"  活跃期: {len(active)}/{len(rdf)}")
log(f"  Alpha正: {(active['alpha']>0).sum()}/{len(active)} = {alpha_pct:.1f}%")
log(f"  年化: {ann*100:+.1f}%  Sharpe: {sharpe:.2f}  MaxDD: {max_dd*100:.1f}%")

log(f"\n  分年:")
ac = active.copy()
ac['year'] = ac['date']//10000
for y,g in ac.groupby('year'):
    log(f"    {y}: 收益={g['port'].mean()*100:+.2f}% Alpha={g['alpha'].mean()*100:+.2f}% WR={g['wr'].mean():.0f}%")

log(f"\n📊 版本对比:")
log(f"  V1.0:       年化+13%   Sharpe 0.72  DD-26.9%  Alpha正67%")
log(f"  V1.1+过滤:  年化+12.7% Sharpe 0.55  DD-14.7%  Alpha正66.7%")
log(f"  V1.3:       年化{ann*100:+.1f}%  Sharpe {sharpe:.2f}  DD{max_dd*100:.1f}%  Alpha正{alpha_pct:.0f}%")

imp = final_model.get_score(importance_type='gain')
total = sum(imp.values())
fund_feats = ['pe_rank','pe_inverse','pb_rank','pb_inverse','div_rank','ps_rank']
fund_gain = sum(imp.get(f,0) for f in fund_feats)
log(f"\n  基本面特征贡献: {fund_gain/total*100:.1f}%")
for f in sorted(fund_feats, key=lambda x: imp.get(x,0), reverse=True):
    log(f"    {f}: {imp.get(f,0)/total*100:.1f}%")

# 保存
final_model.save_model('models/cn/cn_alpha_v1.3.json')
with open('models/cn/cn_alpha_v1.3_summary.json','w') as f:
    json.dump({'version':'cn-alpha-v1.3','date':time.strftime('%Y-%m-%d'),
        'features':len(features),'hold_days':HOLD_DAYS,
        'label':'cross_sectional_rank',
        'paper_trade':{'ann_return':round(ann*100,2),'sharpe':round(sharpe,3),
            'max_dd':round(max_dd*100,2),'alpha_positive_pct':round(alpha_pct,1)},
        'wf':results}, f, indent=2, ensure_ascii=False, default=str)
log(f"\n✅ 模型已保存: models/cn/cn_alpha_v1.3.json")
