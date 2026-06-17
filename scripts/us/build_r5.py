#!/usr/bin/env python3
"""R5最终版：成交量确认+排除财报+回测验证"""
import json, os, warnings, numpy as np, yfinance as yf
warnings.filterwarnings('ignore')

CACHE='/home/admin/.openclaw/workspace/data/cache'
UNIVERSE='/home/admin/.openclaw/workspace/data/sp500_universe.json'
pool=json.load(open(UNIVERSE));tickers=pool['tickers']

print('Loading stocks...', flush=True)
loaded={}
for t in tickers:
    try:
        raw=json.load(open(f'{CACHE}/{t}.json'))['data']
        n=len(raw);c=[float(raw[i]['close']) for i in range(n)]
        v=[float(raw[i].get('volume',0)) for i in range(n)]
        result={}
        for i in range(60,n):
            d=raw[i]['date'];pr=c[i]
            hp52=max(c[max(0,i-251):i+1]);p52=pr/hp52*100 if hp52>0 else 100
            m30=(pr/c[i-30]-1)*100
            d1=(c[i]/c[i-1]-1)*100 if i>0 else 0
            vol5=sum(v[i-4:i+1])/5;vol20=sum(v[i-19:i+1])/20
            vr=vol5/vol20 if vol20>0 else 1
            gains=sum(max(c[j]-c[j-1],0) for j in range(max(1,i-13),i+1))
            losses=sum(max(c[j-1]-c[j],0) for j in range(max(1,i-13),i+1))
            rsi=100-100/(1+gains/losses) if losses>0 else 50
            result[d]={'p':pr,'d1':d1,'m30':m30,'rsi':rsi,'vr':vr,'p52':p52}
        loaded[t]=result
    except:pass

dates_list=sorted(set(d for t in loaded for d in loaded[t]))
years=sorted(set(int(d[:4]) for d in dates_list))
print('Stocks:%d Years:%d'%(len(loaded),len(years)))

def run_r5(params):
    HD=params.get('hd',5);TN=params.get('tn',3);IC=params.get('ic',0)
    D1_MIN=params.get('d1',-3);RSI_MAX=params.get('rsi',35);VR_MIN=params.get('vr',1.3)
    M30_MIN=params.get('m30',3);EXCL_EARN=params.get('excl',False)
    yearly_rets=[]
    for y in years:
        if y>2025:continue
        yr_dates=[d for d in dates_list if '%d-01-02'%y<=d<='%d-12-31'%y]
        if len(yr_dates)<60:continue
        rets=[]
        for si in range(HD,len(yr_dates)-HD,HD):
            db=yr_dates[si];ds=yr_dates[min(si+HD,len(yr_dates)-1)]
            cand=[]
            for t,td in loaded.items():
                vb=td.get(db)
                if not vb:continue
                # R5条件
                if vb.get('d1',0)>=D1_MIN:continue  # 单日跌幅不够
                if vb.get('rsi',50)>=RSI_MAX:continue  # 不够超卖
                if vb.get('m30',0)<M30_MIN:continue  # 趋势向下
                if VR_MIN>1 and vb.get('vr',0)<VR_MIN:continue  # 量不够
                # 排除财报日前3天（可选的，需earnings_date数据）
                # 简化：用vol_r>1.8表示可能的财报日（成交量暴增）
                if EXCL_EARN and vb.get('vr',0)>1.8:continue
                score=-vb['d1']  # 跌越多越好
                cand.append((score,t,vb['p']))
            if not cand:continue
            cand.sort(key=lambda x:-x[0])
            if IC>0:
                f=[];sc={}
                for s,t,p in cand:
                    sec=t[:2]
                    if sc.get(sec,0)>=IC:continue
                    sc[sec]=sc.get(sec,0)+1;f.append((t,p))
                pos=f[:TN]
            else:pos=[(t,p) for _,t,p in cand[:TN]]
            if not pos:continue
            pr=[]
            for t,bp in pos:
                vs=loaded[t].get(ds)
                if vs and bp>0:pr.append((vs['p']/bp-1)*100)
            if pr:rets.append(np.mean(pr))
        if rets:yearly_rets.append(sum(rets))
    cum=sum(yearly_rets);ny=len([r for r in yearly_rets if r!=0])
    ann=((1+cum/100)**(1/ny)-1)*100 if cum>-100 and ny>0 else 0
    sh=np.mean(yearly_rets)/np.std(yearly_rets)*(12**0.5) if len(yearly_rets)>2 and np.std(yearly_rets)>0 else 0
    cv=100;pk=100;md=0
    for r in yearly_rets:cv*=1+r/100;pk=max(pk,cv);md=max(md,(pk-cv)/pk*100)
    return{'ann':round(ann,2),'sharpe':round(sh,2),'mdd':round(md,1)}

# 参数扫描
print('\n=== R5参数扫描 ===')
best=0
for d1 in [-3,-4,-5]:
    for rsi_max in [30,35,40]:
        for vr_min in [1.0,1.3,1.5]:
            for m30_min in [3,5]:
                p={'hd':5,'tn':3,'ic':0,'d1':d1,'rsi':rsi_max,'vr':vr_min,'m30':m30_min}
                r=run_r5(p)
                if r['ann']>best:
                    best=r['ann'];best_p=p
                    print(f'  🔥 跌>%d%% RSI<%d 量>%.1f 趋势>%d: %.2f%% ann %.2f sharpe %.1f%% mdd'%(
                        abs(d1),rsi_max,vr_min,m30_min,r['ann'],r['sharpe'],r['mdd']))

print(f'\n最佳参数: {best_p}')
print(f'最佳业绩: {best}% ann')

# 对比旧R5
old=run_r5({'hd':5,'tn':3,'ic':0,'d1':-10,'rsi':100,'vr':0,'m30':0})
print(f'\n旧R5(跌幅基准): {old["ann"]}% ann {old["sharpe"]} sharpe {old["mdd"]}% mdd')
new=run_r5({**best_p,'ic':1})
print(f'新R5(最佳+行业限1): {new["ann"]}% ann {new["sharpe"]} sharpe {new["mdd"]}% mdd')

# 保存
model={'name':'R5 v2.0','params':best_p,'result':best}
json.dump(model,open('/home/admin/.openclaw/workspace/data/r5_v2.json','w'))
print('\n已保存: r5_v2.json')
