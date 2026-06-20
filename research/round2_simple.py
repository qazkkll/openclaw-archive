#!/usr/bin/env python3
"""第二轮实验：资金流去噪 + 市值控制"""

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
import json, time, warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/home/hermes/.hermes/openclaw-archive/data'
OUTPUT_DIR = '/home/hermes/.hermes/openclaw-archive/research'

print("=" * 60)
print("第二轮：资金流去噪 + 市值控制")
print("=" * 60)

# 加载
print("\n[1] 加载数据...")
t0 = time.time()

df_hist = pd.read_parquet(f'{DATA_DIR}/a_hist_10y.parquet')
df_hist = df_hist.rename(columns={'Code':'sym','Date':'date','O':'open','H':'high','L':'low','C':'close','V':'volume'})
df_hist['date'] = pd.to_datetime(df_hist['date'].astype(str), format='%Y%m%d')

df_mf = pd.read_parquet(f'{DATA_DIR}/moneyflow_core.parquet')
df_mf['sym'] = df_mf['ts_code'].str.replace(r'\.\w+$', '', regex=True)
df_mf['date'] = pd.to_datetime(df_mf['trade_date'].astype(str), format='%Y%m%d')
for c in ['sm','md','lg','elg']:
    df_mf[f'{c}_net'] = df_mf[f'buy_{c}_amount'] - df_mf[f'sell_{c}_amount']
df_mf['total_net'] = df_mf['net_mf_amount']
df_mf = df_mf[['sym','date','sm_net','md_net','lg_net','elg_net','total_net']].drop_duplicates(subset=['sym','date'])

df = pd.merge(df_hist, df_mf, on=['sym','date'], how='inner')
df = df.sort_values(['sym','date']).reset_index(drop=True)
df = df[df['close'] > 0]
print(f"  {len(df):,} 行, {df['sym'].nunique()} 股")

# 特征
print("\n[2] 特征工程...")
df['r1'] = df.groupby('sym')['close'].pct_change(1)
df['r5'] = df.groupby('sym')['close'].pct_change(5)
df['r10'] = df.groupby('sym')['close'].pct_change(10)
df['r20'] = df.groupby('sym')['close'].pct_change(20)
df['ma5'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(5).mean())
df['ma10'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(10).mean())
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20).mean())
df['d5'] = (df['close'] - df['ma5']) / df['ma5']
df['d10'] = (df['close'] - df['ma10']) / df['ma10']
df['d20'] = (df['close'] - df['ma20']) / df['ma20']
df['vol5'] = df.groupby('sym')['r1'].transform(lambda x: x.rolling(5).std())
df['vol20'] = df.groupby('sym')['r1'].transform(lambda x: x.rolling(20).std())
df['hl'] = df['high'] - df['low']
df['atr'] = df.groupby('sym')['hl'].transform(lambda x: x.rolling(14).mean())
df['atr_pct'] = df['atr'] / df['close']
df['vol_r'] = df.groupby('sym')['volume'].transform(lambda x: x.rolling(5).mean()) / (df.groupby('sym')['volume'].transform(lambda x: x.rolling(20).mean()) + 1)

delta = df.groupby('sym')['close'].diff()
df['rsi14'] = 100 - (100 / (1 + delta.clip(lower=0).rolling(14).mean() / ((-delta).clip(lower=0).rolling(14).mean() + 1e-10)))
# 上面的RSI需要按sym分组，用更简单的方式
df['_g'] = delta.clip(lower=0)
df['_l'] = (-delta).clip(lower=0)
df['_ag'] = df.groupby('sym')['_g'].transform(lambda x: x.rolling(14).mean())
df['_al'] = df.groupby('sym')['_l'].transform(lambda x: x.rolling(14).mean())
df['rsi14'] = 100 - (100 / (1 + df['_ag'] / (df['_al'] + 1e-10)))

df['ema12'] = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=12).mean())
df['ema26'] = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=26).mean())
df['macd'] = df['ema12'] - df['ema26']
df['macd_sig'] = df.groupby('sym')['macd'].transform(lambda x: x.ewm(span=9).mean())
df['macd_hist'] = df['macd'] - df['macd_sig']

for c in ['sm_net','md_net','lg_net','elg_net','total_net']:
    df[f'{c}_5'] = df.groupby('sym')[c].transform(lambda x: x.rolling(5).sum())
    df[f'{c}_20'] = df.groupby('sym')[c].transform(lambda x: x.rolling(20).sum())

# 市值代理
df['log_amt'] = np.log1p(df['volume'] * df['close'])
df['log_amt_20'] = df.groupby('sym')['log_amt'].transform(lambda x: x.rolling(20).mean())

df['fwd_ret_20'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-20) / x - 1)
print(f"  特征完成")

# 特征集
tech = ['r1','r5','r10','r20','d5','d10','d20','vol5','vol20','atr_pct','vol_r','rsi14','macd','macd_sig','macd_hist']
mf = ['sm_net_5','sm_net_20','md_net_5','md_net_20','lg_net_5','lg_net_20','elg_net_5','elg_net_20','total_net_5','total_net_20']
sz = ['log_amt_20']

experiments = {
    'tech_only': tech,
    'tech+mf': tech + mf,
    'tech+mf+size': tech + mf + sz,
}

all_feats = list(set(tech + mf + sz))
df = df.dropna(subset=all_feats + ['fwd_ret_20'])
print(f"  有效: {len(df):,} 行, {df['sym'].nunique()} 股")

# Walk-Forward
print("\n[3] Walk-Forward...")
min_d, max_d = df['date'].min(), df['date'].max()
splits = []
c = min_d
while True:
    te = c + pd.DateOffset(years=1, months=6)
    vs = te + pd.Timedelta(days=1)
    ve = vs + pd.DateOffset(months=6)
    if ve > max_d:
        ve = max_d
        if vs < max_d: splits.append((c, te, vs, ve))
        break
    splits.append((c, te, vs, ve))
    c += pd.DateOffset(years=2)
print(f"  {len(splits)} folds")

all_res = {}

for name, feats in experiments.items():
    avail = [f for f in feats if f in df.columns]
    print(f"\n  === {name} ({len(avail)} feat) ===")
    res = []
    for i, (ts, te, vs, ve) in enumerate(splits):
        tr = df[(df['date']>=ts)&(df['date']<=te)]
        tt = df[(df['date']>=vs)&(df['date']<=ve)]
        if len(tr)<50000 or len(tt)<10000: continue
        
        X_tr, y_tr = tr[avail].values, tr['fwd_ret_20'].values
        X_te = tt[avail].values
        
        t1 = time.time()
        m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=42, verbosity=0)
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        
        tc = tt.copy()
        tc['pred'] = p
        ics, rics = [], []
        for d in tc['date'].unique():
            dd = tc[tc['date']==d]
            if len(dd)<30: continue
            ics.append(np.corrcoef(dd['fwd_ret_20'], dd['pred'])[0,1])
            rics.append(spearmanr(dd['fwd_ret_20'], dd['pred'])[0])
        
        ic, ric = np.nanmean(ics), np.nanmean(rics)
        ic_s, ric_s = np.nanstd(ics), np.nanstd(rics)
        
        tc['pct'] = tc.groupby('date')['pred'].rank(pct=True)
        top = tc[tc['pct']>=0.9]['fwd_ret_20'].mean()
        bot = tc[tc['pct']<=0.1]['fwd_ret_20'].mean()
        ls = top - bot
        
        top_d = tc[tc['pct']>=0.9].groupby('date')['fwd_ret_20'].mean()
        if len(top_d)>5:
            cu = (1+top_d).cumprod()
            dd_v = (cu/cu.cummax()-1).min()
            ann = cu.iloc[-1]**(252/max(len(cu),1))-1
            cal = ann/abs(dd_v) if dd_v!=0 else 0
        else:
            ann, dd_v, cal = 0, 0, 0
        
        res.append({'ic':ic,'ic_std':ic_s,'icir':ic/(ic_s+1e-10),'rank_ic':ric,'rank_ic_std':ric_s,
            'rank_icir':ric/(ric_s+1e-10),'top10':top,'bot10':bot,'ls':ls,'ann':ann,'dd':dd_v,'cal':cal,'t':time.time()-t1})
        print(f"    F{i+1}: IC={ic:.4f} RIC={ric:.4f} LS={ls*100:.2f}% ({time.time()-t1:.0f}s)")
    
    s = {}
    for k in ['ic','rank_ic','icir','rank_icir','top10','bot10','ls','ann','dd','cal','t']:
        vs = [r[k] for r in res if not (np.isnan(r[k]) or np.isinf(r[k]))]
        if vs: s[k]=float(np.mean(vs)); s[k+'_std']=float(np.std(vs))
    all_res[name] = {'summary':s, 'nfeat':len(avail)}
    print(f"  → IC={s.get('ic',0)*100:.2f}% LS={s.get('ls',0)*100:.2f}%")

# 对比
print(f"\n{'='*65}")
print("3种特征组合对比")
print(f"{'='*65}")
print(f"{'实验':<16} {'IC':>8} {'RankIC':>8} {'ICIR':>8} {'多空':>8} {'回撤':>8} {'#feat'}")
print("-"*65)
for n, d in all_res.items():
    s = d['summary']
    print(f"{n:<16} {s.get('ic',0)*100:>7.2f}% {s.get('rank_ic',0)*100:>7.2f}% {s.get('icir',0)*100:>7.1f}% {s.get('ls',0)*100:>7.2f}% {s.get('dd',0)*100:>7.1f}% {d['nfeat']:>5}")

best = max(all_res.items(), key=lambda x: x[1]['summary'].get('ls',0))
print(f"\n最优: {best[0]} (多空 {best[1]['summary']['ls']*100:.2f}%)")

# 特征重要性（最优方案的最后一个fold）
print(f"\n最优方案特征重要性 TOP 10:")
imp = m.feature_importances_
fi = sorted(zip(avail, imp), key=lambda x: -x[1])
for fn, fv in fi[:10]:
    print(f"  {fn:<18} {fv:.4f}")

with open(f'{OUTPUT_DIR}/round2_results.json', 'w') as f:
    json.dump(all_res, f, indent=2, default=str)
print(f"\n保存: {OUTPUT_DIR}/round2_results.json")
print(f"耗时: {time.time()-t0:.0f}s")
