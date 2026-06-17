#!/usr/bin/env python3
"""修正版回测对比（正确的回撤+净值计算）"""
import sys, json, os, time, math
ws = '/home/hermes/.hermes/openclaw-archive'
sys.path.insert(0, ws+'/scripts')
from score_engine import v5s_calc as new_calc, v5s_score as new_score

t0 = time.time()
data = json.load(open(ws+'/data/us_hist_clean.parquet','rb'))
syms = list(data.keys())

def common_sf(a,i):
    if not a: return 0
    idx=i if i>=0 else len(a)+i
    if 0<=idx<len(a) and a[idx] is not None:
        v=a[idx]; return 0 if (isinstance(v,float) and v!=v) else v
    return 0

# 旧指标
def old_ind(c):
    n=len(c)
    def sm(a,p): return [None]*(p-1)+[sum(a[i-p+1:i+1])/p for i in range(p-1,len(a))]
    def em(a,p): k=2/(p+1); r=[a[0]]; [r.append(v*k+r[-1]*(1-k)) for v in a[1:]]; return r
    m5=sm(c,5);m20=sm(c,20);m60=sm(c,60);m120=sm(c,120)
    e12=em(c,12);e26=em(c,26);md=[e12[i]-e26[i] for i in range(n)]
    sg=sm(md,9);hst=[md[i]-(sg[i] if i<len(sg) and sg[i] is not None else 0) for i in range(n)]
    gl=[max(c[i]-c[i-1],0) for i in range(1,n)]
    ls=[max(c[i-1]-c[i],0) for i in range(1,n)]
    rsi=[None]*14
    if len(gl)>=14:
        ag=sum(gl[:14])/14;al=sum(ls[:14])/14
        for i in range(14,n):
            rsi.append(100-100/(1+ag/al) if al>0 else 100)
            if i<len(gl): ag=(ag*13+gl[i])/14; al=(al*13+ls[i])/14
    p52=[None]*252
    for i in range(252,n):
        lo=min(c[i-251:i+1]);hi=max(c[i-251:i+1])
        p52.append((c[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    return {'ma5':m5,'ma20':m20,'ma60':m60,'ma120':m120,'md':md,'sg':sg,'hst':hst,'rsi':rsi,'p52':p52}

def old_sc(ind,di):
    def sf(a,i):
        if not a: return 0
        idx=i if i>=0 else len(a)+i
        v=a[idx] if 0<=idx<len(a) and a[idx] is not None else 0
        return 0 if (isinstance(v,float) and v!=v) else v
    c=ind['c'];p=sf(c,di)
    if p<=0: return 0
    m5=sf(ind['ma5'],di);m20=sf(ind['ma20'],di);m60=sf(ind['ma60'],di);m120=sf(ind['ma120'],di)
    tr=(10 if m5>m20 else 0)+(10 if m20>m60 else 0)+(10 if m60>m120 else 0)+(5 if p>m20 else 0)+(5 if p>m60 else 0)+(5 if p>m120 else 0)+(5 if m5>m20 and m20>m60 else 0)
    mo=15;p5=sf(c,di-5);p20x=sf(c,di-20);p60x=sf(c,di-60)
    if p5>0 and p>p5: mo+=5
    if p20x>0 and p>p20x: mo+=5
    if p60x>0 and p>p60x: mo+=5
    p30=sf(c,di-30);m30=(p-p30)/p30*100 if p30>0 else 0; mo+=m30/10
    if m30>50: mo=max(mo-(m30-50)/5,0)
    mh_=sf(ind['hst'],di);mhp=sf(ind['hst'],di-1);ms=8 if sf(ind['md'],di)>sf(ind['sg'],di) else 0
    if mh_>0 and mhp<=0: ms+=12
    elif mh_>0: ms+=5
    if mh_>mhp: ms+=5
    rsi=sf(ind['rsi'],di)
    rs=5+(5 if 50<=rsi<=70 else 3 if rsi>70 else -5 if rsi<30 else 0)
    p52=sf(ind['p52'],di)
    ps=10 if 70<=p52<=100 else 7 if 50<=p52<70 else 4 if 30<=p52<50 else 0
    return max(tr+mo+ms+rs+ps,0)

def backtest(score_fn, ind_all, name=''):
    """统一回测引擎。score_fn(ind, di, ticker) → float"""
    N=min(len(data[t]['c']) for t in tickers)
    c=100000; pos={}; trades=[]; navs=[1.0]
    
    for i in range(W, N-20):
        # 评分
        ca=[]
        for t in tickers:
            sc=score_fn(ind_all[t], i, t)
            if sc>=MS: ca.append((sc,t,common_sf(data[t]['c'],i)))
        
        # 卖出（每日净值在卖出后结算）
        for t in list(pos.keys()):
            cp=common_sf(data[t]['c'], i)
            r=(cp-pos[t]['ep'])/pos[t]['ep']*100
            if r<SL or i-pos[t]['ei']>=HD:
                trades.append({'t':t,'ep':pos[t]['ei'],'ev':pos[t]['ep'],'xv':cp,'ret':r,'h':i-pos[t]['ei']})
                c+=pos[t]['sh']*cp; del pos[t]
        
        # 买入
        ca.sort(key=lambda x:-x[0])
        for sc,t,bp in ca:
            if len(pos)>=MH: break
            if t in pos: continue
            al=c/(MH-len(pos))
            if al<100 or bp<=0: continue
            pos[t]={'ep':bp,'sh':al/bp,'ei':i}; c-=al
        
        # 每日净值（按MH均仓虚拟）
        ev=c
        for t,pd in pos.items():
            cp=common_sf(data[t]['c'], i)
            if cp>0: ev+=pd['sh']*cp
        navs.append(ev/100000)
    
    # 清仓
    for t,pd in list(pos.items()):
        cp=common_sf(data[t]['c'], N-21)
        if cp>0:
            r=(cp-pd['ep'])/pd['ep']*100
            trades.append({'t':t,'ep':pd['ei'],'ev':pd['ep'],'xv':cp,'ret':r,'h':N-21-pd['ei']})
            c+=pd['sh']*cp
    
    ev=c
    for t,pd in pos.items():
        cp=common_sf(data[t]['c'], min(N-21,len(data[t]['c'])-1))
        if cp>0: ev+=pd['sh']*cp
    navs.append(ev/100000)
    
    nt=len(trades)
    wr=round(len([x for x in trades if x['ret']>0])/nt*100,1) if nt>0 else 0
    fn=ev/100000
    yr=(N-W)/252
    ann=round((fn**(1/yr)-1)*100,2) if yr>0 else 0
    
    # 回撤（逐日净值）
    pk=max(navs); mdd=0.0
    for v in navs:
        if v>pk: pk=v
        dd=(pk-v)/pk
        if dd>mdd: mdd=dd
    mdd=round(mdd*100,1)
    
    # 夏普（日收益率）
    drs=[]
    for j in range(1,len(navs)):
        if navs[j-1]>0: drs.append((navs[j]/navs[j-1]-1)*100)
    am=sum(drs)/len(drs) if drs else 0
    sd=math.sqrt(sum((x-am)**2 for x in drs)/len(drs)) if len(drs)>1 else 1
    sh=round(am/sd*math.sqrt(252),2) if sd>1e-10 else 0
    
    avg_win=round(sum(x['ret'] for x in trades if x['ret']>0)/max(1,sum(1 for x in trades if x['ret']>0)),2)
    avg_loss=round(sum(x['ret'] for x in trades if x['ret']<=0)/max(1,sum(1 for x in trades if x['ret']<=0)),2)
    
    return {'params':{'hd':HD,'ms':MS,'mh':MH,'sl':SL},'ann':ann,'sh':sh,'dd':mdd,'wr':wr,
            'trades_count':nt,'avg_win':avg_win,'avg_loss':avg_loss,'fn':round(fn,4)}

def old_wrap(ind, di, t):
    return old_sc(ind, di)
def new_wrap(ind, di, t):
    return new_score(ind, di)

HD=20; TN=5; MS=60; MH=8; SL=-15; W=400

tickers=[]
old={}
for i,sy in enumerate(syms):
    if i%500==0: print('old %d/%d' % (i,len(syms)), flush=True)
    d=data[sy]; c=[float(x) for x in d.get('c',[])]
    if len(c)<520: continue
    ind=old_ind(c); ind['c']=c
    old[sy]=ind; tickers.append(sy)
print('old pool: %d (%ds)' % (len(tickers), time.time()-t0), flush=True)

print('=== OLD ===', flush=True)
r_old=backtest(old_wrap, old, 'old')
print(json.dumps(r_old, ensure_ascii=False))

new={}
for i,sy in enumerate(tickers):
    if i%500==0: print('new %d/%d' % (i,len(tickers)), flush=True)
    d=data[sy]; c=[float(x) for x in d.get('c',[])]
    h=[float(x) for x in d.get('h',[])]; l=[float(x) for x in d.get('l',[])]
    try:
        ind=new_calc(c,h,l)
        if ind: new[sy]=ind
    except: pass
print('new pool: %d (%ds)' % (len(new), time.time()-t0), flush=True)

print('=== NEW ===', flush=True)
r_new=backtest(new_wrap, new, 'new')
print(json.dumps(r_new, ensure_ascii=False))

print('\n' + '='*55)
print('  新旧模型最终对比')
print('='*55)
fmt='  %-12s %8s %8s %8s'
print(fmt % ('','旧模型','新模型','差距'))
print(fmt % ('年化', str(r_old['ann'])+'%', str(r_new['ann'])+'%', ('+'+str(round(r_new['ann']-r_old['ann'],1))+'%')))
print(fmt % ('夏普', str(r_old['sh']), str(r_new['sh']), ('+' if r_new['sh']>=r_old['sh'] else '')+str(round(r_new['sh']-r_old['sh'],2))))
print(fmt % ('回撤', str(r_old['dd'])+'%', str(r_new['dd'])+'%', ('+' if r_new['dd']>=r_old['dd'] else '')+str(round(r_new['dd']-r_old['dd'],1))+'%'))
print(fmt % ('胜率', str(r_old['wr'])+'%', str(r_new['wr'])+'%', ('+' if r_new['wr']>=r_old['wr'] else '')+str(round(r_new['wr']-r_old['wr'],1))+'%'))
print(fmt % ('交易', str(r_old['trades_count']), str(r_new['trades_count']), ('+' if r_new['trades_count']>=r_old['trades_count'] else '')+str(r_new['trades_count']-r_old['trades_count'])))
print(fmt % ('均胜', str(r_old['avg_win'])+'%', str(r_new['avg_win'])+'%', ''))
print(fmt % ('均亏', str(r_old['avg_loss'])+'%', str(r_new['avg_loss'])+'%', ''))
print(fmt % ('终净值', str(r_old['fn']), str(r_new['fn']), ''))
print('\n耗时: %ds' % (time.time()-t0))
