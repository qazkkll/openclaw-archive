#!/usr/bin/env python3
"""
方向1 v2: 有意义的样本外回测
分割点设在2024-06-30（特征数据在2024年6月前只有45只，之后才有2200只）
训练: 2024-06 ~ 2025-06 (64,009行, 2212只)
回测: 2025-07 ~ 2026-06 (完全样本外)
"""
import sys,os,json,pickle,time,itertools,warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd,numpy as np,xgboost as xgb

BASE='/home/hermes/.hermes/openclaw-archive'; ML_DIR=f'{BASE}/ml'; MD=f'{BASE}/data/models'; VER='us_v7_4_oos2'
print('='*70,flush=True); print('方向1 v2: 样本外回测 (2025H2~2026)',flush=True); print('='*70,flush=True)
T0=time.time()

# 1. 加载
print('\n[1] 加载数据...',flush=True)
df=pd.read_parquet(f'{ML_DIR}/us_ml_feats_v71_v19.parquet')
FEATS=['price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
    'ret1','ret5','ret20','ret60','macd','macd_signal','macd_hist',
    'vol_ratio','ma_bias20','vol5','trend_accel',
    'short_ratio','short_pct','market_cap',
    'sector_etf_ret5','spy_ret5','qqq_ret5','iwm_ret5','sc']

df=df.replace([np.inf,-np.inf],np.nan)
df=df.dropna(subset=FEATS+['label_5d_5class','label_5d_pct'])
for f in FEATS: df[f]=pd.to_numeric(df[f],errors='coerce').fillna(0).clip(-1e6,1e6)
df=df.sort_values(['sym','date']).reset_index(drop=True)
df['ds']=df['date'].astype(str)

# 分割点
TRAIN_END='2025-06-30'
train_dates=[d for d in sorted(df['ds'].unique()) if d<=TRAIN_END]
test_dates=[d for d in sorted(df['ds'].unique()) if d>TRAIN_END]

train_df=df[df['ds'].isin(train_dates)].copy()
test_df=df[df['ds'].isin(test_dates)].copy()

print(f'  全部: {len(df):,}行, {df.sym.nunique()}只',flush=True)
print(f'  训练: {len(train_df):,}行, {train_df.sym.nunique()}只 ({train_dates[0]}~{TRAIN_END})',flush=True)
print(f'  回测: {len(test_df):,}行, {test_df.sym.nunique()}只 ({test_dates[0]}~{test_dates[-1]})',flush=True)

# 2. 训练
print('\n[2] 模型训练...',flush=True)
X_train=np.nan_to_num(train_df[FEATS].values.astype(np.float32),nan=0)
y_train=train_df['label_5d_5class'].values.astype(int)

dtrain=xgb.DMatrix(X_train,label=y_train)
params={
    'eta':0.07,'gamma':0.5,'max_depth':7,'min_child_weight':5,
    'subsample':0.7,'colsample_bytree':0.7,'lambda':1.5,'alpha':0.5,
    'objective':'multi:softprob','eval_metric':'mlogloss','num_class':5,'seed':42,
}
clf=xgb.train(params,dtrain,num_boost_round=400,verbose_eval=False)
print(f'  训练完成 ✓',flush=True)

# 校准
from sklearn.isotonic import IsotonicRegression
print('  校准...',flush=True)
calib_start=int(len(train_df)*0.8)
X_cal=X_train[calib_start:]
raw=clf.predict(xgb.DMatrix(X_cal))
raw_up5=raw[:,4]
y_cal=(train_df['label_5d_5class'].values[calib_start:]>=4).astype(float)

# 重新采样平衡（校准数据通常严重偏0）
pos_count=int(y_cal.sum())
neg_count=len(y_cal)-pos_count
print(f'  校准: pos={pos_count}({pos_count/len(y_cal):.1%}), neg={neg_count}',flush=True)

cal=IsotonicRegression(out_of_bounds='clip')
cal.fit(raw_up5,y_cal)

# 3. 概率计算
print('\n[3] 样本外概率计算...',flush=True)
probs={}
n_batch=5000
for i in range(0,len(test_df),n_batch):
    if i%(n_batch*5)==0: print(f'  {100*i//max(len(test_df),1)}%...',flush=True)
    batch=test_df.iloc[i:i+n_batch].dropna(subset=FEATS)
    if len(batch)==0: continue
    X=np.nan_to_num(batch[FEATS].values.astype(np.float32),nan=0)
    raw=clf.predict(xgb.DMatrix(X))
    if raw.ndim>1: raw=raw[:,4]
    calib=cal.predict(raw)
    for j,idx in enumerate(batch.index): probs[idx]=float(calib[j])
print(f'  完成: {len(probs):,}行',flush=True)

# 4. 索引
print('\n[4] 构建回测索引...',flush=True)
day_cands={}
for d in test_dates:
    day=df[df['ds']==d].copy()
    if len(day)<30: continue
    day['p']=day.index.map(lambda i:probs.get(i,0))
    day=day.dropna(subset=['p'])
    cands=[(r['sym'],r['p'],float(r['price'])) for _,r in day.iterrows() if r['p']>0]
    cands.sort(key=lambda x:-x[1])
    day_cands[d]=cands

date_prices={d:{} for d in test_dates}
for _,r in df[df['ds'].isin(test_dates)].iterrows():
    date_prices[r['ds']][r['sym']]=float(r['price'])
print(f'  日索引: {len(day_cands)}天',flush=True)

# 5. 参数回测
print('\n[5] 样本外参数回测...',flush=True)
PARAM_TOP=[5,10,15]; PARAM_HOLD=[5,10]; PARAM_STOP=[5,10,15]; PARAM_REB=[5,10]
results=[]

for top_n,hold,stop,rebal in itertools.product(PARAM_TOP,PARAM_HOLD,PARAM_STOP,PARAM_REB):
    if hold<rebal: continue
    cap=10000.0; cash=cap; portfolio={}; trades=0; wins=0; curve=[cap]; sl=stop/100.0
    day_list=sorted(day_cands.keys())
    for day_idx,d in enumerate(day_list):
        pt=date_prices.get(d,{})
        for sym in list(portfolio.keys()):
            pos=portfolio[sym]; cp=pt.get(sym)
            if cp is None: continue
            dh=(pd.to_datetime(d)-pd.to_datetime(pos['bd'])).days
            ret=(cp-pos['bp'])/pos['bp']
            if ret<=-sl or dh>=hold:
                cash+=pos['qty']*cp; trades+=1
                if cp>=pos['bp']: wins+=1
                del portfolio[sym]
        if day_idx%rebal==0:
            buys=[c for c in day_cands.get(d,[]) if c[0] not in portfolio][:top_n]
            for sym,prob,price in buys:
                qty=cash/max(top_n,1)/max(price,0.01)
                if qty<1: continue
                portfolio[sym]={'bd':d,'bp':price,'qty':qty}
                cash-=qty*price
        curve.append(cash+sum(p['qty']*pt.get(s,p['bp']) for s,p in portfolio.items()))
    final=cash+sum(p['qty']*p['bp'] for p in portfolio.values())
    ec=np.array(curve); tr=(final/10000-1)*100; yrs=len(day_list)/252
    an=((final/10000)**(1/max(yrs,0.01))-1)*100
    peak=np.maximum.accumulate(ec)
    mdd=(ec-peak).min()/peak.max()*100 if peak.max()>0 else 0
    dr=np.diff(ec)/(ec[:-1]+1e-10)
    sh=(dr.mean()/max(dr.std(),1e-6))*np.sqrt(252) if len(dr)>20 else 0
    wr=wins/max(trades,1)
    tag=f'T{top_n}_H{hold}_S{stop}_R{rebal}'
    results.append({'tag':tag,'tr':round(tr,1),'an':round(an,1),
        'sh':round(sh,2),'mdd':round(mdd,1),'wr':round(wr,3),'trades':trades})

# 6. 结果
print('\n[6] 样本外结果',flush=True)
rdf=pd.DataFrame(results).sort_values('sh',ascending=False)
print(f'{"参数":20s} {"收益":>7s} {"年化":>7s} {"夏普":>6s} {"回撤":>7s} {"胜率":>6s} {"交易":>5s}')
print('-'*60)
for _,r in rdf.iterrows():
    print(f'{r["tag"]:20s} {r["tr"]:>6.1f}% {r["an"]:>6.1f}% {r["sh"]:>6.2f} {r["mdd"]:>6.1f}% {r["wr"]:>5.1%} {r["trades"]:>5}')

print('\n=== 夏普Top5 ===')
for _,r in rdf.head(5).iterrows():
    print(f'  {r["tag"]:20s} 年化{r["an"]:>5.1f}% 夏普{r["sh"]:>5.2f} 回撤{r["mdd"]:>5.1f}%')
print('\n=== 年化Top5 ===')
for _,r in rdf.sort_values('an',ascending=False).head(5).iterrows():
    print(f'  {r["tag"]:20s} 年化{r["an"]:>5.1f}% 夏普{r["sh"]:>5.2f} 回撤{r["mdd"]:>5.1f}%')

# 保存
json.dump({
    'timestamp':'2026-06-11 10:45','model':'样本外训练(2024-06~2025-06)',
    'train_range':'2024-06~2025-06','backtest_range':'2025-07~2026-06',
    'capital':10000,'days':len(test_dates),
    'train_stocks':int(train_df.sym.nunique()),
    'test_stocks':int(test_df.sym.nunique()),
    'all':rdf.to_dict('records'),
    'sharpe_top':rdf.head(5).to_dict('records'),
},open(f'{MD}/{VER}_backtest.json','w'),indent=2)
print(f'\n保存: {VER}_backtest.json')
print(f'耗时: {time.time()-T0:.0f}s')
print('='*70)
