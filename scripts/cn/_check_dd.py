import sys, json, os
ws = '/home/hermes/.hermes/openclaw-archive'
sys.path.insert(0, ws + '/scripts')
from score_engine import v5s_calc as nc, v5s_score as ns

all_d = json.load(open(ws + '/data/us_hist_clean.parquet','r'))
syms = list(all_d.keys())

def old_ind(c, h, l):
    n=len(c)
    def sm(a,p): return [None]*(p-1)+[sum(a[i-p+1:i+1])/p for i in range(p-1,len(a))]
    def em(a,p): k=2/(p+1); r=[a[0]]; [r.append(v*k+r[-1]*(1-k)) for v in a[1:]]; return r
    m5=sm(c,5);m20=sm(c,20);m60=sm(c,60);m120=sm(c,120)
    e12=em(c,12);e26=em(c,26);macd=[e12[i]-e26[i] for i in range(n)]
    sig=sm(macd,9)
    hst=[macd[i]-(sig[i] if i<len(sig) and sig[i] is not None else 0) for i in range(n)]
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
    return {'close':c,'ma5':m5,'ma20':m20,'ma60':m60,'ma120':m120,
            'macd':macd,'macd_signal':sig,'macd_hist':hst,'rsi':rsi,'p52':p52}

def old_sc(ind, di):
    def sf(a,i):
        if not a: return 0
        idx=i if i>=0 else len(a)+i
        return a[idx] if 0<=idx<len(a) and a[idx] is not None else 0
    c=ind['close'];p=sf(c,di)
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
    mh=sf(ind['macd_hist'],di);mhp=sf(ind['macd_hist'],di-1);ms=8 if sf(ind['macd'],di)>sf(ind['macd_signal'],di) else 0
    if mh>0 and mhp<=0: ms+=12
    elif mh>0: ms+=5
    if mh>mhp: ms+=5
    rsi=sf(ind['rsi'],di)
    rs=5+(5 if 50<=rsi<=70 else 3 if rsi>70 else -5 if rsi<30 else 0)
    p52=sf(ind['p52'],di)
    ps=10 if 70<=p52<=100 else 7 if 50<=p52<70 else 4 if 30<=p52<50 else 0
    return max(tr+mo+ms+rs+ps,0)

H=20; SL=-0.15; T5=5

cache={}
for i,sy in enumerate(syms):
    if i%500==0: print('  %d/%d' % (i,len(syms)), flush=True)
    d=all_d[sy];c=d.get('c',[]);h=d.get('h',[]);l=d.get('l',[])
    if len(c)<520: continue
    ind=old_ind(c,h,l)
    if ind is not None: cache[sy]=(ind,c)
print('有效池: %d' % len(cache), flush=True)

max_n=max(len(c) for _,c in cache.values())
all_t=[]
for day in range(252, max_n-H, 5):
    cand=[]
    for sy,(ind,c) in cache.items():
        if day>=len(c): continue
        sc=old_sc(ind,day)
        if sc>0: cand.append((sc,sy,c[day]))
    cand.sort(key=lambda x:-x[0])
    for sc,sy,bp in cand[:T5]:
        ic=cache[sy]; cc=ic[1]
        sd=min(day+H,len(cc)-1)
        for d2 in range(day+1,sd+1):
            if (cc[d2]-bp)/bp<=SL: sd=d2; sp=cc[d2]; break
        else:
            sp=cc[sd]
        ret=(sp-bp)/bp
        if ret is None or (isinstance(ret, float) and ret != ret):
            continue
        all_t.append(ret)

nt=len(all_t)
print('交易: %d' % nt, flush=True)

# 逐笔净值法（正确）
# 均仓：每笔T5只各投1/T5
pos = T5
nav=1.0; peak=1.0; mdd=0.0
for r in all_t:
    nav*=(1+r/pos)
    if nav>peak: peak=nav
    dd=(peak-nav)/peak
    if dd>mdd: mdd=dd

fn=nav
ann=fn**(1/5)-1
wr=sum(1 for r in all_t if r>0)/nt*100
ar=sum(all_t)/nt*100

# 夏普用单笔收益，年化乘sqrt(252)
am=sum(all_t)/nt
sm=((sum((r-am)**2 for r in all_t)/nt)**0.5) if nt>1 else 1
sharpe=(am/sm)*(252**0.5) if sm>1e-10 else 0

print('')
print('=== 旧模型 正确逐笔净值计算 ===')
print('交易: %d  胜率: %.1f%%  均收益: %.2f%%' % (nt, wr, ar))
print('总收益: %.1f%%' % ((fn-1)*100))
print('年化: %.1f%%' % (ann*100))
print('夏普(年化): %.2f' % sharpe)
print('最大回撤(逐笔净值): %.1f%%' % (mdd*100))
