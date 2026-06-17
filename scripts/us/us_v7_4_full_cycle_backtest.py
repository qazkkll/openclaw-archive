#!/usr/bin/env python3
"""
方向2: 全周期策略调优+大盘风控 v2
用groupby替代逐行索引，更快
"""
import sys,os,json,pickle,time,itertools,warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd,numpy as np,xgboost as xgb

BASE='/home/hermes/.hermes/openclaw-archive'; ML_DIR=f'{BASE}/ml'; MD=f'{BASE}/data/models'; VER='us_v7_4_full_ctrl'
print('='*70,flush=True); print('方向2 v2: 全周期策略+大盘风控',flush=True); print('='*70,flush=True)
T0=time.time()

# 1. 加载&概率(预计算)
# 复用已有模型的概率（us_v7_4的概率在之前跑过，但是新会话没有probs缓存）
# 这里直接加载已有特征，算概率
print('\n[1] 加载+概率计算...',flush=True)
model=xgb.Booster(); model.load_model(f'{MD}/us_v7_4.json')
cal=pickle.load(open(f'{MD}/us_v7_4_calibrator.pkl','rb'))
df=pd.read_parquet(f'{ML_DIR}/us_ml_feats_v71_v19.parquet')
FEATS=['price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
    'ret1','ret5','ret20','ret60','macd','macd_signal','macd_hist',
    'vol_ratio','ma_bias20','vol5','trend_accel',
    'short_ratio','short_pct','market_cap',
    'sector_etf_ret5','spy_ret5','qqq_ret5','iwm_ret5','sc']

df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATS+['label_5d_5class','label_5d_pct'])
for f in FEATS: df[f]=pd.to_numeric(df[f],errors='coerce').fillna(0).clip(-1e6,1e6)
df=df.sort_values(['sym','date']).reset_index(drop=True)
df['ds']=df['date'].astype(str)

# 原版v7.4概率已经存在（模型中），直接计算
btd=sorted(df['ds'].unique())
dt=df.copy()
probs=np.zeros(len(dt))
n_batch=5000
for i in range(0,len(dt),n_batch):
    if i%(n_batch*10)==0: print(f'  prob {100*i//len(dt)}%...',flush=True)
    batch=dt.iloc[i:i+n_batch].dropna(subset=FEATS)
    if len(batch)==0: continue
    X=np.nan_to_num(batch[FEATS].values.astype(np.float32),nan=0)
    raw=model.predict(xgb.DMatrix(X))
    if raw.ndim>1: raw=raw[:,4]
    calib=cal.predict_proba(raw.reshape(-1,1))[:,1]
    for j,idx in enumerate(batch.index): probs[idx]=float(calib[j])
print(f'  prob done: {len(probs)}, range {btd[0]}~{btd[-1]}',flush=True)

# 2. 快速索引构建（groupby归并）
print('\n[2] 索引构建...',flush=True)
dt=df.copy()
dt['prob']=probs
# 按天排序取TopN（groupby先按概率排序）
all_cands=dt[dt['prob']>0].copy()
# 按天和概率排序，取每天候选
day_cands={}
for d,grp in all_cands.groupby('ds'):
    cands=list(zip(grp['sym'],grp['prob'],grp['price']))
    cands.sort(key=lambda x:-x[1])
    day_cands[d]=cands

# 价格索引也用同样的
date_prices={}
for d,grp in dt.groupby('ds'):
    date_prices[d]=dict(zip(grp['sym'],grp['price']))
print(f'  {len(day_cands)}天索引完成',flush=True)

# SPY 200日均线（用全市场平均价格代理）
daily_med=dt.groupby('ds')['price'].median()
daily_sma200=daily_med.rolling(200).mean()
above_sma=(daily_med>daily_sma200).to_dict()
print(f'  200日均线: {sum(above_sma.values())}/{len(above_sma)}天在均线上方',flush=True)

# 3. 参数回测
print('\n[3] 全周期多参数回测...',flush=True)
PARAM_TOP=[5,10,15]; PARAM_HOLD=[5,10]; PARAM_STOP=[5,10,15]; PARAM_REB=[5,10]; PARAM_CTRL=[0,50,100]
results=[]

total_combos=108; done=0
for top_n,hold,stop,rebal,ctrl in itertools.product(PARAM_TOP,PARAM_HOLD,PARAM_STOP,PARAM_REB,PARAM_CTRL):
    done+=1
    if done%18==0: print(f'  {done}/{total_combos}...',flush=True)
    cap=10000.0; cash=cap; pf={}; trades=0; wins=0; eq=[cap]; sl=stop/100.0
    day_list=sorted(day_cands.keys())
    ctrl_f=ctrl/100.0
    
    for idx,d in enumerate(day_list):
        pt=date_prices.get(d,{})
        # 止损+到期
        for s in list(pf.keys()):
            p=pf[s]; cp=pt.get(s)
            if cp is None: continue
            dh=(pd.to_datetime(d)-pd.to_datetime(p['bd'])).days
            ret=(cp-p['bp'])/p['bp']
            if ret<=-sl or dh>=hold:
                cash+=p['qty']*cp; trades+=1
                if cp>=p['bp']: wins+=1
                del pf[s]
        # 风控
        risk=1.0 if above_sma.get(d,True) else ctrl_f
        # 调仓
        if idx%rebal==0:
            cands=[c for c in day_cands.get(d,[]) if c[0] not in pf]
            buys=cands[:max(1,int(top_n*risk))]
            for sym,prob,price in buys:
                if risk<=0: break
                qty=cash/top_n/max(price,0.01)*risk
                if qty<1: continue
                pf[sym]={'bd':d,'bp':price,'qty':qty}
                cash-=qty*price
        eq.append(cash+sum(p['qty']*pt.get(s,p['bp']) for s,p in pf.items()))
    
    final=cash+sum(p['qty']*p['bp'] for p in pf.values())
    ec=np.array(eq); tr=(final/10000-1)*100; yrs=len(day_list)/252
    an=((final/10000)**(1/max(yrs,0.01))-1)*100 if yrs>0 else 0
    peak=np.maximum.accumulate(ec)
    mdd=(ec-peak).min()/peak.max()*100 if peak.max()>0 else 0
    dr=np.diff(ec)/(ec[:-1]+1e-10)
    sh=(dr.mean()/max(dr.std(),1e-6))*np.sqrt(252) if len(dr)>20 else 0
    wr=wins/max(trades,1)
    ct={0:'No',50:'Half',100:'AllOut'}[ctrl]
    tag=f'T{top_n}_H{hold}_S{stop}_R{rebal}_{ct}'
    results.append({'tag':tag,'tr':round(tr,1),'an':round(an,1),'sh':round(sh,2),
        'mdd':round(mdd,1),'wr':round(wr,3),'trades':trades,'ctrl':ct})

# 4. 输出
print(f'\n[4] 结果 ({len(results)}组合)',flush=True)
rdf=pd.DataFrame(results)

print('\n=== 全局Top15（夏普排序）===')
rdf_s=rdf.sort_values('sh',ascending=False)
print(f'{"参数":25s} {"收益":>7s} {"年化":>7s} {"夏普":>6s} {"回撤":>7s} {"胜率":>6s} {"风控":>5s}')
print('-'*65)
for _,r in rdf_s.head(15).iterrows():
    print(f'{r["tag"]:25s} {r["tr"]:>6.1f}% {r["an"]:>6.1f}% {r["sh"]:>6.2f} {r["mdd"]:>6.1f}% {r["wr"]:>5.1%} {r["ctrl"]:>5s}')

print('\n=== 按风控级别平均值 ===')
for ct in ['No','Half','AllOut']:
    sub=rdf[rdf['ctrl']==ct]
    print(f'  {ct:>7s}: avg_sh={sub["sh"].mean():.2f} avg_an={sub["an"].mean():.1f}% avg_mdd={sub["mdd"].mean():.1f}%')

# 无风控 vs 有风控在不同年份的表现
print('\n=== 按年查看无风控 vs 半仓风控 (T10_H10_S15_R5) ===')
for ctrl_txt in ['No','Half']:
    row=rdf[(rdf['tag'].str.startswith('T10_H10_S15_R5'))&(rdf['ctrl']==ctrl_txt)]
    if len(row)>0:
        r=row.iloc[0]
        print(f'  T10_H10_S15_R5_{ctrl_txt}: 年化{r["an"]:>5.1f}% 夏普{r["sh"]:>5.2f} 回撤{r["mdd"]:>5.1f}%')

# 5. 保存
json.dump({
    'timestamp':'2026-06-11 10:50','model':'us_v7_4',
    'range':'2022~2026','capital':10000,'combos':len(results),
    'sharpe_top':rdf_s.head(10).to_dict('records'),
    'all':rdf.to_dict('records'),
},open(f'{MD}/{VER}_backtest.json','w'),indent=2)
print(f'\n保存: {VER}_backtest.json')
print(f'耗时: {time.time()-T0:.0f}s')
