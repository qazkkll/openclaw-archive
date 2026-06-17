"""V5-S 止损优化回测 — 对比几种止损策略
使用原版backtest.py框架，只改止损逻辑"""
import sys, os, json, math, time, itertools
ws = '/home/hermes/.hermes/openclaw-archive'
sys.path.insert(0, os.path.join(ws, 'scripts'))
sys.path.insert(0, os.path.join(ws, 'scripts', 'recovered'))
import common
sys.modules['lib'] = type(sys)('lib')
sys.modules['lib'].common = common
sys.modules['lib.common'] = common

# 预计算指标（只跑一次）
CACHE = ws + '/data/indicators_old_v5s.json'
if not os.path.exists(CACHE):
    print('预计算指标...', flush=True)
    data = json.load(open(ws+'/data/us_hist_clean.parquet','rb'))
    syms = list(data.keys())
    inds = {}; tickers = []
    def sm(a,p): return [None]*(p-1)+[sum(a[i-p+1:i+1])/p for i in range(p-1,len(a))]
    def em(a,p): k=2/(p+1); r=[a[0]]; [r.append(v*k+r[-1]*(1-k)) for v in a[1:]]; return r
    for i,sy in enumerate(syms):
        if i%500==0: print('  %d/%d' % (i,len(syms)), flush=True)
        d=data[sy];c=[float(x) for x in d.get('c',[])]
        h=[float(x) for x in d.get('h',[])]; l=[float(x) for x in d.get('l',[])]
        if len(c)<520: continue
        n=len(c)
        m5=sm(c,5);m20=sm(c,20);m60=sm(c,60);m120=sm(c,120)
        e12=em(c,12);e26=em(c,26);macd=[e12[i]-e26[i] for i in range(n)]
        sig=sm(macd,9);hst=[macd[i]-(sig[i] if i<len(sig) and sig[i] is not None else 0) for i in range(n)]
        gl=[max(c[i]-c[i-1],0) for i in range(1,n)]; ls=[max(c[i-1]-c[i],0) for i in range(1,n)]
        rsi=[None]*14
        if len(gl)>=14:
            ag=sum(gl[:14])/14;al=sum(ls[:14])/14
            for j in range(14,n):
                rsi.append(100-100/(1+ag/al) if al>0 else 100)
                if j<len(gl): ag=(ag*13+gl[j])/14; al=(al*13+ls[j])/14
        p52=[None]*252
        for j in range(252,n):
            lo=min(c[j-251:j+1]);hi=max(c[j-251:j+1])
            p52.append((c[j]-lo)/(hi-lo)*100 if hi>lo else 50)
        inds[sy] = {'close':c,'high':h,'low':l,'ma5':m5,'ma20':m20,'ma60':m60,'ma120':m120,
                    'macd':macd,'macd_signal':sig,'macd_hist':hst,'rsi':rsi,'p52':p52}
        tickers.append(sy)
    json.dump({'inds':{t:inds[t] for t in tickers},'tickers':tickers}, open(CACHE,'w'))
    print('指标缓存: %d只' % len(tickers), flush=True)
else:
    cache = json.load(open(CACHE,'rb'))
    tickers = cache['tickers']
    inds = cache['inds']
    print('从缓存加载: %d只' % len(tickers), flush=True)

data = json.load(open(ws+'/data/us_hist_clean.parquet','rb'))
N = min(len(data[t]['c']) for t in tickers)
print('N=%d天' % N, flush=True)

# 原版评分（旧模型）
def score(sy, di):
    def sf(a,i):
        if not a: return 0
        idx=i if i>=0 else len(a)+i
        v=a[idx] if 0<=idx<len(a) and a[idx] is not None else 0
        return 0 if (isinstance(v,float) and v!=v) else v
    c=inds[sy]['close'];p=sf(c,di)
    if p<=0: return 0
    m5=sf(inds[sy]['ma5'],di);m20=sf(inds[sy]['ma20'],di);m60=sf(inds[sy]['ma60'],di);m120=sf(inds[sy]['ma120'],di)
    tr=(10 if m5>m20 else 0)+(10 if m20>m60 else 0)+(10 if m60>m120 else 0)+(5 if p>m20 else 0)+(5 if p>m60 else 0)+(5 if p>m120 else 0)+(5 if m5>m20 and m20>m60 else 0)
    mo=15
    p5=sf(c,di-5);p20x=sf(c,di-20);p60x=sf(c,di-60)
    if p5>0 and p>p5: mo+=5
    if p20x>0 and p>p20x: mo+=5
    if p60x>0 and p>p60x: mo+=5
    p30=sf(c,di-30);m30=(p-p30)/p30*100 if p30>0 else 0; mo+=m30/10
    if m30>50: mo=max(mo-(m30-50)/5,0)
    mh=sf(inds[sy]['macd_hist'],di);mhp=sf(inds[sy]['macd_hist'],di-1)
    ms=8 if sf(inds[sy]['macd'],di)>sf(inds[sy]['macd_signal'],di) else 0
    if mh>0 and mhp<=0: ms+=12
    elif mh>0: ms+=5
    if mh>mhp: ms+=5
    rsi=sf(inds[sy]['rsi'],di)
    rs=5+(5 if 50<=rsi<=70 else 3 if rsi>70 else -5 if rsi<30 else 0)
    p52=sf(inds[sy]['p52'],di)
    ps=10 if 70<=p52<=100 else 7 if 50<=p52<70 else 4 if 30<=p52<50 else 0
    return max(tr+mo+ms+rs+ps,0)

# 止损策略
STOP_STRATEGIES = {
    'sl_15': lambda sy, ep, hp, di: (common.sf(data[sy]['c'], di) - ep) / ep * 100 < -15,
    'sl_10': lambda sy, ep, hp, di: (common.sf(data[sy]['c'], di) - ep) / ep * 100 < -10,
    'sl_20': lambda sy, ep, hp, di: (common.sf(data[sy]['c'], di) - ep) / ep * 100 < -20,
    'sl_8': lambda sy, ep, hp, di: (common.sf(data[sy]['c'], di) - ep) / ep * 100 < -8,
    'sl_trail_15': lambda sy, ep, hp, di: (common.sf(data[sy]['c'], di) - hp) / hp * 100 < -15 or (common.sf(data[sy]['c'], di) - ep) / ep * 100 < -15,
    'sl_trail_10': lambda sy, ep, hp, di: (common.sf(data[sy]['c'], di) - hp) / hp * 100 < -10 or (common.sf(data[sy]['c'], di) - ep) / ep * 100 < -15,
}

# 回测函数（可配置止损）
def run_bt(stop_fn, hd=20, ms=60, mh=8, W=400):
    c=100000.; pos={}; trades=[]; dv=[]
    for i in range(W, N-20):
        scored=[(sy, score(sy,i)) for sy in tickers]
        ca=sorted([x for x in scored if x[1]>=ms], key=lambda x:-x[1])
        for sy in list(pos.keys()):
            cp=common.sf(data[sy]['c'], i)
            r=(cp-pos[sy]['ep'])/pos[sy]['ep']*100
            triggered=stop_fn(sy, pos[sy]['ep'], pos[sy].get('hp',pos[sy]['ep']), i)
            if triggered or i-pos[sy]['ei']>=hd:
                trades.append({'t':sy,'ep':pos[sy]['ei'],'xp':i,'ev':round(pos[sy]['ep'],2),'xv':round(cp,2),'ret':round(r,2),'h':i-pos[sy]['ei']})
                c+=pos[sy]['sh']*cp; del pos[sy]
        for sy, sc in ca:
            if len(pos)>=mh: break
            if sy in pos: continue
            al=c/(mh-len(pos))
            if al<100: continue
            ep=common.sf(data[sy]['c'], i)
            if ep<=0: continue
            pos[sy]={'ep':ep,'sh':al/ep,'ei':i,'hp':ep}; c-=al
        for pd in pos.values():
            cp=common.sf(data[pd['ei' if 'ei' in pd else ''].replace('ei','') if False else list(pos.keys())[list(pos.values()).index(pd)] if pd in pos.values() else ''], i) if False else 0
        # 更新最高价
        for sy in list(pos.keys()):
            cp=common.sf(data[sy]['c'], i)
            if cp>pos[sy].get('hp',0): pos[sy]['hp']=cp
        v=c+sum(pd['sh']*common.sf(data[t]['c'],i) for t,pd in pos.items() if common.sf(data[t]['c'],i)>0)
        dv.append(v)
    for sy, pd in pos.items():
        cp=common.sf(data[sy]['c'],N-21)
        trades.append({'t':sy,'ep':pd['ei'],'xp':N-21,'ev':round(pd['ep'],2),'xv':round(cp,2),'ret':round((cp-pd['ep'])/pd['ep']*100,2),'h':N-21-pd['ei']})
        c+=pd['sh']*cp
    total_r=(c/100000-1)*100; yr=(N-W)/252
    ann=round(((c/100000)**(1/yr)-1)*100,2) if yr>0 else 0
    pk=dv[0];md=0
    for v in dv:
        if v>pk: pk=v
        dd=(pk-v)/pk*100
        if dd>md: md=dd
    md=round(md,1)
    dr=[(dv[i]/dv[i-1]-1)*100 for i in range(1,len(dv))]
    mr=sum(dr)/len(dr) if dr else 0
    sd=math.sqrt(sum((x-mr)**2 for x in dr)/len(dr)) if dr else 0
    sh=round(mr/sd*math.sqrt(252),2) if sd>0 else 0
    wr=round(len([x for x in trades if x['ret']>0])/len(trades)*100,1) if trades else 0
    return {'ann':ann,'sh':sh,'dd':md,'wr':wr,'trades_count':len(trades),'trades':trades}

# 跑所有止损策略
results = []
for name, fn in STOP_STRATEGIES.items():
    t1=time.time()
    r=run_bt(fn)
    r['strategy']=name
    r['time']=round(time.time()-t1,1)
    results.append(r)
    print('%s: ann=%6.1f%% sh=%5.2f dd=%5.1f%% wr=%5.1f%% tr=%4d (%ds)' % (
        name, r['ann'], r['sh'], r['dd'], r['wr'], r['trades_count'], r['time']), flush=True)

print('\n' + '='*70)
print('止损策略对比')
print('='*70)
print('策略          年化     夏普     回撤     胜率     笔数')
print('-'*55)
for r in sorted(results, key=lambda x:-x['ann']):
    print('%-14s %6.1f%% %6.2f %6.1f%% %6.1f%% %5d' % (
        r['strategy'], r['ann'], r['sh'], r['dd'], r['wr'], r['trades_count']))

best = max(results, key=lambda x:x['ann'])
print('\n🏆 最佳年化: %s (%.1f%%)' % (best['strategy'], best['ann']))
best_sh = max(results, key=lambda x:x['sh'])
print('🏆 最佳夏普: %s (%.2f)' % (best_sh['strategy'], best_sh['sh']))

json.dump(results, open(ws+'/data/bt_stop_strategies.json','w'), indent=2)
print('保存: data/bt_stop_strategies.json')
