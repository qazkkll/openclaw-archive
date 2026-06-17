#!/usr/bin/env python3
"""V5-S 回测总框架（原版逻辑精简版）+ 新旧模型对比"""
import sys, json, os, time, math
ws = '/home/hermes/.hermes/openclaw-archive'
sys.path.insert(0, ws+'/scripts')
from score_engine import v5s_calc as new_calc, v5s_score as new_score

t0 = time.time()
data = json.load(open(ws+'/data/us_hist_clean.parquet','rb'))
syms = list(data.keys())

def old_indicators(c):
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

def old_score(ind, di):
    def sf(a,i):
        if not a: return 0
        idx=i if i>=0 else len(a)+i
        v=a[idx] if 0<=idx<len(a) and a[idx] is not None else 0
        return 0 if (isinstance(v,float) and v!=v) else v
    c=ind['c'];p=sf(c,di)
    if p<=0: return 0
    m5=sf(ind['ma5'],di);m20=sf(ind['ma20'],di);m60=sf(ind['ma60'],di);m120=sf(ind['ma120'],di)
    tr=(10 if m5>m20 else 0)+(10 if m20>m60 else 0)+(10 if m60>m120 else 0)+(5 if p>m20 else 0)+(5 if p>m60 else 0)+(5 if p>m120 else 0)+(5 if m5>m20 and m20>m60 else 0)
    mo=15
    p5=sf(c,di-5);p20x=sf(c,di-20);p60x=sf(c,di-60)
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

def backtest(score_fn, ind_all, price_fn=None):
    """统一回测引擎。score_fn(ind, di, ticker) → float"""
    N = min(len(data[t]['c']) for t in tickers)
    c=100000; pos={}; trades=[]
    
    for i in range(W, N-20):
        # 评分
        ca=[]
        for t in tickers:
            sc=score_fn(ind_all[t], i, t)
            if sc>=MS: ca.append((sc,t,common_sf(data[t]['c'],i)))
        
        # 卖出
        for t in list(pos.keys()):
            cp=common_sf(data[t]['c'], i)
            r=(cp-pos[t]['ep'])/pos[t]['ep']*100
            if r<SL or i-pos[t]['ei']>=HD:
                trades.append({'t':t,'ep':pos[t]['ei'],'ev':pos[t]['ep'],'xv':cp,'ret':r,'h':i-pos[t]['ei'],'ei':pos[t]['ei']})
                c+=pos[t]['sh']*cp; del pos[t]
        
        # 买入
        ca.sort(key=lambda x:-x[0])
        for sc,t,bp in ca:
            if len(pos)>=MH: break
            if t in pos: continue
            al=c/(MH-len(pos))
            if al<100: continue
            if bp<=0: continue
            pos[t]={'ep':bp,'sh':al/bp,'ei':i}; c-=al
    
    # 清仓
    for t,pd in pos.items():
        cp=common_sf(data[t]['c'], N-21)
        trades.append({'t':t,'ep':pd['ei'],'ev':pd['ep'],'xv':cp,'ret':(cp-pd['ep'])/pd['ep']*100,'h':N-21-pd['ei'],'ei':pd['ei']})
        c+=pd['sh']*cp
    
    nt=len(trades)
    wr=round(len([x for x in trades if x['ret']>0])/nt*100,1) if nt>0 else 0
    ar=sum(x['ret'] for x in trades)/nt if nt>0 else 0
    fn=c/100000
    yr=(N-W)/252
    ann=round((fn**(1/yr)-1)*100,2) if yr>0 else 0
    # 回撤（逐笔）
    stops=[max(1,t['ep']) for t in trades if t['ret']==t['ret']]
    # 其实直接按trade的ret算净值曲线更简单——但backtest.py没保存净值
    # 用equity近似：sum(未实现) or 逐笔累加
    # 用最简单的方式：模拟均仓MH的净值变化
    ns=[1.0];pk=1.0;mdd=0.0
    for t in trades:
        if isinstance(t['ret'],float) and t['ret']!=t['ret']: continue
        ns.append(ns[-1]*(1+t['ret']/MH))
        if ns[-1]>pk: pk=ns[-1]
        dd=(pk-ns[-1])/pk; 
        if dd>mdd: mdd=dd
    mdd=round(mdd*100,1)
    # 夏普
    rs=[t['ret'] for t in trades if not (isinstance(t['ret'],float) and t['ret']!=t['ret'])]
    am=sum(rs)/len(rs) if rs else 0
    sm=math.sqrt(sum((x-am)**2 for x in rs)/len(rs)) if len(rs)>1 else 1
    sh=round(am/sm*math.sqrt(252),2) if sm>1e-10 else 0
    return {'params':{'hd':HD,'ms':MS,'mh':MH,'sl':SL},'ann':ann,'sh':sh,'dd':mdd,'wr':wr,'trades_count':nt,'trades':trades}

def common_sf(a,i):
    if not a: return 0
    idx=i if i>=0 else len(a)+i
    if 0<=idx<len(a) and a[idx] is not None:
        v=a[idx]; return 0 if (isinstance(v,float) and v!=v) else v
    return 0

# 参数
HD=20; TN=5; MS=60; MH=8; SL=-15; W=400

# 加载+算旧指标
tickers=[];
old={}
for i,sy in enumerate(syms):
    if i%500==0: print('old %d/%d' % (i,len(syms)), flush=True)
    d=data[sy];c=[float(x) for x in d.get('c',[])]
    if len(c)<520: continue
    ind=old_indicators(c); ind['c']=c
    old[sy]=ind; tickers.append(sy)
print('old pool: %d (%ds)' % (len(tickers), time.time()-t0), flush=True)

def old_score_wrap(ind, di, t):
    return old_score(ind, di)

print('=== OLD backtest ===', flush=True)
r_old=backtest(old_score_wrap, old)
print(json.dumps({k:r_old[k] for k in ['ann','sh','dd','wr','trades_count']}, ensure_ascii=False), flush=True)

# 新模型
new={}
for i,sy in enumerate(tickers):
    if i%500==0: print('new %d/%d' % (i,len(tickers)), flush=True)
    d=data[sy]
    c=[float(x) for x in d.get('c',[])]
    h=[float(x) for x in d.get('h',[])]
    l=[float(x) for x in d.get('l',[])]
    try:
        ind=new_calc(c,h,l)
        if ind: new[sy]=ind
    except: pass
print('new pool: %d (%ds)' % (len(new), time.time()-t0), flush=True)

def new_score_wrap(ind, di, t):
    return new_score(ind, di)

print('=== NEW backtest ===', flush=True)
r_new=backtest(new_score_wrap, new)
print(json.dumps({k:r_new[k] for k in ['ann','sh','dd','wr','trades_count']}, ensure_ascii=False), flush=True)

print('')
print('='*50)
print('  对比')
print('='*50)
print('           旧模型     新模型     差距')
print('年化:      %6.1f%%   %6.1f%%   %+6.1f%%' % (r_old['ann'], r_new['ann'], r_new['ann']-r_old['ann']))
print('夏普:      %6.2f     %6.2f    %+6.2f' % (r_old['sh'], r_new['sh'], r_new['sh']-r_old['sh']))
print('回撤:      %6.1f%%   %6.1f%%   %+6.1f%%' % (r_old['dd'], r_new['dd'], r_new['dd']-r_old['dd']))
print('胜率:      %6.1f%%   %6.1f%%   %+6.1f%%' % (r_old['wr'], r_new['wr'], r_new['wr']-r_old['wr']))
print('交易:      %6d      %6d    %+d' % (r_old['trades_count'], r_new['trades_count'], r_new['trades_count']-r_old['trades_count']))
print('')
print('耗时: %ds' % (time.time()-t0))
