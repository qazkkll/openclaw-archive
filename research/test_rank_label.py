#!/usr/bin/env python3
"""V1.3轻量版：只测试排名标签效果，不加新数据"""
import pandas as pd, numpy as np, xgboost as xgb, json, os, time, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

def log(msg): print(msg, flush=True)

HOLD_DAYS = 10
TOP_K = 15

log("V1.3 轻量测试：排名标签 vs 原始标签")

# 加载
log("\n[1] 加载数据...")
df = pd.read_parquet('data/cn/features_v2.parquet')
df['date'] = pd.to_datetime(df['date'])
df['date_int'] = df['date'].dt.strftime('%Y%m%d').astype(int)

# 只用2019年起（减少内存）
df = df[df['date_int'] >= 20190101].copy()
log(f"  {len(df):,}行, {df['sym'].nunique()}只")

# 计算v1.1特征
log("\n[2] 计算特征...")
df['rev_5d'] = -df['r5']
df['rev_10d'] = -df['r10']
df['rev_20d'] = -df['r20']
df['rsi_reversal'] = -(df['rsi14'] - 50)
df['macd_reversal'] = -df['macd']
df['low_vol_5d'] = -df['vol5']
df['low_vol_20d'] = -df['vol20']
df['low_atr'] = -df['atr_pct']
df['small_cap'] = -np.log(df['circ_mv'].clip(lower=1))
for col, src in [('residual_mom_5d','r5'),('residual_mom_20d','r20')]:
    df[col] = df[src] - df.groupby('date_int')[src].transform('mean')
df['lg_flow_momentum'] = df['lg_net_5'] - df['lg_net_20'] / 4
df['total_flow_momentum'] = df['total_net_5'] - df['total_net_20'] / 4
for col in ['lg_net_20','md_net_20','total_net_20']:
    df[f'{col}_rank'] = df.groupby('date_int')[col].rank(pct=True)
df['rev_flow_interaction'] = df['rev_20d'] * df['lg_net_20_rank']
df['turnover_rank'] = df.groupby('date_int')['vol_r'].rank(pct=True)
# 基本面填0（后续再灌真实数据）
for f in ['pe_rank','pe_inverse','pb_rank','pb_inverse','div_rank','ps_rank']:
    df[f] = 0

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

# 标签
log("\n[3] 标签...")
df = df.sort_values(['sym','date_int'])
df['fwd_ret'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-HOLD_DAYS)/x-1)
df['fwd_rank'] = df.groupby('date_int')['fwd_ret'].rank(pct=True)
df_valid = df.dropna(subset=['fwd_ret']).copy()
log(f"  有效: {len(df_valid):,}")

# WF
log("\n[4] Walk-Forward...")
folds = [
    (20190101,20211231,20220101,20230630),
    (20200101,20220630,20220701,20231231),
    (20200101,20230630,20230701,20241231),
    (20210101,20240630,20240701,20260630),
]

params = {'max_depth':6,'eta':0.05,'subsample':0.8,'colsample_bytree':0.8,
          'min_child_weight':100,'objective':'reg:squarederror','tree_method':'hist'}

for label, target in [('原始标签fwd_ret','fwd_ret'),('排名标签fwd_rank','fwd_rank')]:
    log(f"\n  === {label} ===")
    wf = []
    for fi,(tr_s,tr_e,te_s,te_e) in enumerate(folds):
        t1 = time.time()
        tr = df_valid[(df_valid['date_int']>=tr_s)&(df_valid['date_int']<=tr_e)]
        te = df_valid[(df_valid['date_int']>=te_s)&(df_valid['date_int']<=te_e)]
        if len(tr)<5000 or len(te)<500: continue
        
        dtrain = xgb.DMatrix(tr[features].fillna(0), label=tr[target])
        dtest = xgb.DMatrix(te[features].fillna(0))
        m = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
        te = te.copy()
        te['pred'] = m.predict(dtest)
        
        ic = te.groupby('date_int').apply(lambda x: x['pred'].corr(x['fwd_ret'])).mean()
        ric = te.groupby('date_int').apply(lambda x: x['pred'].corr(x['fwd_ret'],method='spearman')).mean()
        top_ret = te.groupby('date_int').apply(lambda x: x.nlargest(TOP_K,'pred')['fwd_ret'].mean()).mean()
        bot_ret = te.groupby('date_int').apply(lambda x: x.nsmallest(TOP_K,'pred')['fwd_ret'].mean()).mean()
        ls = top_ret - bot_ret
        wf.append({'ic':ic,'ric':ric,'ls':ls,'top':top_ret})
        log(f"    Fold{fi+1}: IC={ic:.4f} RankIC={ric:.4f} LS={ls:.4f} Top={top_ret:.4f} ({time.time()-t1:.0f}s)")
    
    log(f"    汇总: IC={np.mean([r['ic'] for r in wf]):.4f} RankIC={np.mean([r['ric'] for r in wf]):.4f} LS={np.mean([r['ls'] for r in wf]):.4f} Top={np.mean([r['top'] for r in wf]):.4f}")

log("\n[完成]")
