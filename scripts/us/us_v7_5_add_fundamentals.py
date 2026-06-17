#!/usr/bin/env python3
"""
us_v7_5_add_fundamentals.py v2 — V7.5 基本面过滤+重训+回测
不使用yfinance info（太慢），直接用现有数据中的成交量+价格做过滤
还能从PE/beta这些已有特征中做质量筛选
"""
import sys, os, json, pickle, time, itertools, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

BASE='/home/hermes/.hermes/openclaw-archive'; ML=f'{BASE}/ml'; MD=f'{BASE}/data/models'; VER='us_v7_5'
print('='*70,flush=True); print('V7.5 基本面过滤 v2',flush=True); print('='*70,flush=True)
T0=time.time()

# ===== 1. 加载特征 =====
df=pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str']=df['date'].astype(str).str[:10]
print(f'[1] 特征: {len(df):,}行, {df.sym.nunique()}只, {df.date.min()}~{df.date.max()}',flush=True)

# ===== 2. 从原始10年数据计算成交量 + 价格过滤 =====
print('\n[2] 成交量/价格过滤...',flush=True)
main=pd.read_parquet(f'{ML}/us_hist_yf_10y.parquet', columns=['ticker','date','close','volume'])
main.rename(columns={'ticker':'sym'},inplace=True)
mega=pd.read_parquet(f'{ML}/us_hist_megacap_10y.parquet', columns=['sym','date','close','volume'])
all_v=pd.concat([main,mega],ignore_index=True).drop_duplicates(subset=['sym','date'])
# 过滤用最近60天日均成交量 + 最新收盘价
latest_v=all_v.sort_values('date').groupby('sym').tail(60)
vol_stats=latest_v.groupby('sym').agg(
    avg_vol_60d=('volume','mean'),
    close_60d_avg=('close','mean')
).reset_index()
# 最新close
last_close=all_v.sort_values('date').groupby('sym').last()['close'].reset_index()
last_close.rename(columns={'close':'last_close'},inplace=True)
del main,mega,all_v,latest_v

# 合并
filters=vol_stats.merge(last_close,on='sym')

# 过滤条件
MIN_VOL_USD=5e6     # 日均成交量>$5M的量（不是成交额）
MIN_PRICE=3.0        # 不低于$3
# 注意：avg_vol_60d是股数，需要x价格≈金额
filters['avg_vol_60d']=filters['avg_vol_60d'].fillna(0)
filters['avg_vol_usd']=filters['avg_vol_60d']*filters['close_60d_avg']
filters['last_close']=filters['last_close'].fillna(0)

# 使用PE/估值已有特征做质量筛选（从特征集判断）
# 从V7.5特征中获取每只股票的最后一期PE和股息率
latest_feats=df[df['date_str']==sorted(df['date_str'].unique())[-1]][['sym','pe_trailing','div_yield','sc']].copy()
filters=filters.merge(latest_feats,on='sym',how='left')
filters['pe_trailing']=filters['pe_trailing'].fillna(0)
filters['div_yield']=filters['div_yield'].fillna(0)

# 条件组合
vol_ok=filters['avg_vol_usd']>=MIN_VOL_USD
price_ok=filters['last_close']>=MIN_PRICE
pe_ok=((filters['pe_trailing']>0)&(filters['pe_trailing']<200))|(filters['pe_trailing']<=0)
# pe_ok: 正PE或者负PE都行（不排除亏损公司），但排除超高PE>200的

valid_syms=set(filters[vol_ok&price_ok]['sym'].unique())
print(f'  全池: {filters.sym.nunique()}只')
print(f'  成交量>={MIN_VOL_USD/1e6:.0f}M美元: {vol_ok.sum()}')
print(f'  价格>={MIN_PRICE}: {price_ok.sum()}')
print(f'  全部满足: {len(valid_syms)}只')

# ===== 3. 过滤特征集 =====
df_f=df[df['sym'].isin(valid_syms)].copy()
print(f'\n[3] 过滤后特征: {len(df_f):,}行',flush=True)

# ===== 4. 训练（过滤版） =====
print('\n[4] 训练XGBoost (过滤版)...',flush=True)
FEATS=json.load(open(f'{MD}/{VER}_report.json'))['features']
df_f['target']=(df_f['fwd_5d_ret']>0.05).astype(int)
for f in FEATS:
    if f in df_f.columns: df_f[f]=pd.to_numeric(df_f[f],errors='coerce').fillna(0).clip(-1e6,1e6)
df_f=df_f.replace([np.inf,-np.inf],np.nan)
del df

TRAIN_END='2023-12-31';VAL_END='2024-12-31'
train=df_f[df_f['date_str']<TRAIN_END]
val=df_f[(df_f['date_str']>=TRAIN_END)&(df_f['date_str']<VAL_END)]
test=df_f[df_f['date_str']>=VAL_END]

X_train=np.nan_to_num(train[FEATS].values.astype(np.float32),nan=0)
y_train=train['target'].values.astype(int)
X_val=np.nan_to_num(val[FEATS].values.astype(np.float32),nan=0)
y_val=val['target'].values.astype(int)
X_test=np.nan_to_num(test[FEATS].values.astype(np.float32),nan=0)
y_test=test['target'].values.astype(int)

pos_c=y_train.sum(); neg_c=len(y_train)-pos_c
spw=neg_c/max(pos_c,1)
print(f'  训练: {len(X_train):,} 正:{pos_c:,} 负:{neg_c:,} 权重:{spw:.1f}',flush=True)

dtrain=xgb.DMatrix(X_train,y_train)
dval=xgb.DMatrix(X_val,y_val)
model=xgb.train({
    'objective':'binary:logistic','eval_metric':'auc',
    'max_depth':6,'learning_rate':0.05,'subsample':0.8,
    'colsample_bytree':0.8,'scale_pos_weight':spw,
    'min_child_weight':3,'gamma':0.2,'reg_alpha':0.5,'reg_lambda':3,
    'random_state':42,, 'device':'cuda'
},dtrain,500,evals=[(dtrain,'train'),(dval,'val')],early_stopping_rounds=100,verbose_eval=50)

print(f'\n验证集AUC: {roc_auc_score(y_val,model.predict(dval)):.4f}',flush=True)
test_auc=roc_auc_score(y_test,model.predict(xgb.DMatrix(X_test)))
print(f'测试集AUC: {test_auc:.4f}',flush=True)

# 校准
cal_sz=min(len(X_test)//2,100000)
y_cp=model.predict(xgb.DMatrix(X_test[:cal_sz]))
cal=LogisticRegression(C=1.0,solver='lbfgs').fit(y_cp.reshape(-1,1),y_test[:cal_sz])
y_tr=model.predict(xgb.DMatrix(X_test[cal_sz:]))
y_tc=cal.predict_proba(y_tr.reshape(-1,1))[:,1]
cal_auc=roc_auc_score(y_test[cal_sz:],y_tc)
print(f'校准后AUC: {cal_auc:.4f}',flush=True)

model.save_model(f'{MD}/{VER}_filtered.json')
pickle.dump(cal,open(f'{MD}/{VER}_filtered_calibrator.pkl','wb'))
json.dump({'model':f'{VER}_filtered','features':FEATS,
    'val_auc':round(roc_auc_score(y_val,model.predict(dval)),4),
    'test_auc':round(test_auc,4),'calibrated_auc':round(cal_auc,4),
    'filtered_syms':len(valid_syms),'original_syms':2474,
    'filters':{'min_vol_usd':5e6,'min_price':3}},
    open(f'{MD}/{VER}_filtered_report.json','w'),indent=2)
print(f'\n[5] 模型保存: us_v7_5_filtered.json',flush=True)

# ===== 5. 带过滤回测 =====
print('\n[6] 带过滤回测...',flush=True)
del df_f['date']
BTD=sorted(df_f['date_str'].unique())
BTD=[d for d in BTD if d>='2022-01-01']

# 价格索引(只回测日期)
main2=pd.read_parquet(f'{ML}/us_hist_yf_10y.parquet',columns=['ticker','date','open','close'])
main2.rename(columns={'ticker':'sym'},inplace=True)
mega2=pd.read_parquet(f'{ML}/us_hist_megacap_10y.parquet',columns=['sym','date','open','close'])
av=pd.concat([main2,mega2],ignore_index=True).drop_duplicates(subset=['sym','date'])
av['ds']=av['date'].astype(str).str[:10];av=av[av['ds'].isin(BTD)]
ci={};oi={}
for s,g in av.groupby('sym'):
    g=g.sort_values('ds');oi[s]=dict(zip(g['ds'].values,g['open'].values.astype(float)))
    ci[s]=dict(zip(g['ds'].values,g['close'].values.astype(float)))
del main2,mega2,av

# 概率
print('  概率计算...',flush=True)
probs={}
for i in range(0,len(df_f),10000):
    c=df_f.iloc[i:i+10000];c=c[c['date_str'].isin(BTD)]
    if len(c)==0:continue
    X=np.nan_to_num(c[FEATS].values.astype(np.float32),nan=0)
    r=model.predict(xgb.DMatrix(X,feature_names=FEATS))
    cb=cal.predict_proba(r.reshape(-1,1))[:,1]
    for j,(_,r2) in enumerate(c.iterrows()):probs[r2.name]=float(cb[j])

# 候选
print('  候选索引...',flush=True)
dc={}
for d in BTD:
    day=df_f[df_f['date_str']==d]
    if len(day)<10:continue
    day=day.copy();day['p']=day.index.map(lambda i:probs.get(i,0));day=day[day['p']>0]
    cands=[]
    for _,r in day.iterrows():
        ni=BTD.index(d)+1;nd=BTD[ni] if ni<len(BTD) else None
        if nd is None:continue
        np=oi.get(r['sym'],{}).get(nd)
        if np is None:continue
        cands.append((r['sym'],r['p'],float(np)))
    cands.sort(key=lambda x:-x[1]);dc[d]=cands
print(f'  {sum(len(v) for v in dc.values()):,}条候选',flush=True)

# 回测
PARAMS=[('T5_H10_S15_R10',5,10,15,10),('T5_H10_S10_R10',5,10,10,10),
        ('T5_H5_S15_R10',5,5,15,10),('T5_H10_S5_R10',5,10,5,10),
        ('T10_H10_S15_R10',10,10,15,10)]
results=[]
for tag,tn,hold,stop,rebal in PARAMS:
    cap=10000.0;cash=cap;pf={};t=0;w=0;cv=[float(cap)];sl=stop/100.0
    for di,d in enumerate(BTD):
        for s in list(pf.keys()):
            p=pf[s];cp=ci.get(s,{}).get(d)
            if cp is None:continue
            ret=(cp-p['bp'])/p['bp']
            if ret<=-sl or (di-p['di'])>=hold:
                cash+=p['qty']*cp;t+=1
                if cp>=p['bp']:w+=1
                del pf[s]
        if di%rebal==0:
            cands=[c for c in dc.get(d,[]) if c[0] not in pf]
            for s,pr,prc in cands[:tn]:
                qty=cash/max(tn,1)/max(prc,0.01)
                if qty<1:continue
                pf[s]={'bp':prc,'qty':qty,'di':di}
                cash-=qty*prc
        pv=sum(p['qty']*ci.get(s,{}).get(d,p['bp']) for s,p in pf.items())
        cv.append(cash+pv)
    final=cash+sum(p['qty']*p['bp'] for p in pf.values())
    ec=np.array(cv);tr=(final/10000-1)*100
    yrs=len(BTD)/252
    an=((final/10000)**(1/max(yrs,0.01))-1)*100 if yrs>=0.01 else tr
    peak=np.maximum.accumulate(ec)
    mdd=(ec-peak).min()/peak.max()*100 if peak.max()>0 else 0
    dr=np.diff(ec)/(ec[:-1]+1e-10)
    sh=(dr.mean()/max(dr.std(),1e-6))*np.sqrt(252) if len(dr)>20 else 0
    wr=w/max(t,1)
    results.append({'tag':tag,'tr':round(tr,1),'an':round(an,1),
        'sh':round(sh,2),'mdd':round(mdd,1),'wr':round(wr,3),'trades':t,
        'filtered_syms':len(valid_syms),'day_stats':{'min_cands_per_day':min(len(v) for v in dc.values())}})
    print(f'  {tag}: 年化{an:.1f}% 夏普{sh:.2f} 回撤{mdd:.1f}% 胜率{wr:.1%}',flush=True)

print(f'\n{"="*70}')
rdf=pd.DataFrame(results).sort_values('sh',ascending=False)
print(f'{"参数":20s} {"收益":>7s} {"年化":>7s} {"夏普":>6s} {"回撤":>7s} {"胜率":>7s} {"交易":>6s}')
print('-'*60)
for _,r in rdf.iterrows():
    print(f'{r["tag"]:20s} {r["tr"]:>6.1f}% {r["an"]:>6.1f}% {r["sh"]:>6.2f} {r["mdd"]:>6.1f}% {r["wr"]:>6.1%} {r["trades"]:>6}')

# 对比原版
print(f'\n=== 与原版对比 ===')
print(f'  过滤前: 2474只 (全池)')
print(f'  过滤后: {len(valid_syms)}只 (成交量>={MIN_VOL_USD/1e6:.0f}M美元 + 价格>={MIN_PRICE})')
og=json.load(open(f'{MD}/us_v7_5_backtest_v4.json'))
for o in og['all']:
    for f in results:
        if o['tag']==f['tag']:
            print(f'  {o["tag"]}: 过滤前夏普{o["sh"]} 年化{o["an"]}% → 过滤后夏普{f["sh"]} 年化{f["an"]}%')

json.dump({'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),'model':f'{VER}_filtered',
    'range':f'{BTD[0]}~{BTD[-1]}','days':len(BTD),'method':'次日开盘+成交量+价格过滤',
    'filtered_syms':len(valid_syms),'original_syms':2474,
    'all':rdf.to_dict('records')},open(f'{MD}/us_v7_5_backtest_filtered.json','w'),indent=2)
print(f'\n耗时: {(time.time()-T0)/60:.1f}分钟')
print('='*70)
