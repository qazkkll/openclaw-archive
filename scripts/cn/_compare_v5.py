"""V5-S 新旧模型对比 — 简洁版，预计算一次，分两次回测"""
import sys, os, json, math, time
ws = '/home/hermes/.hermes/openclaw-archive'
sys.path.insert(0, os.path.join(ws, 'scripts'))
sys.path.insert(0, os.path.join(ws, 'scripts', 'recovered'))

import common
sys.modules['lib'] = type(sys)('lib')
sys.modules['lib'].common = common
sys.modules['lib.common'] = common

from score_engine import v5s_calc as new_calc, v5s_score as new_score

# 旧版指标
def old_ind(c, h, l):
    n=len(c)
    def sm(a,p): return [None]*(p-1)+[sum(a[i-p+1:i+1])/p for i in range(p-1,len(a))]
    def em(a,p): k=2/(p+1); r=[a[0]]; [r.append(v*k+r[-1]*(1-k)) for v in a[1:]]; return r
    m5=sm(c,5);m20=sm(c,20);m60=sm(c,60);m120=sm(c,120)
    e12=em(c,12);e26=em(c,26);macd=[e12[i]-e26[i] for i in range(n)]
    sig=sm(macd,9);hst=[macd[i]-(sig[i] if i<len(sig) and sig[i] is not None else 0) for i in range(n)]
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
    return {'ma5':m5,'ma20':m20,'ma60':m60,'ma120':m120,'macd':macd,
            'macd_signal':sig,'macd_hist':hst,'rsi':rsi,'p52':p52}

# 原版评分
def old_score(ind, di):
    def sf(a,i):
        if not a: return 0
        idx=i if i>=0 else len(a)+i
        v=a[idx] if 0<=idx<len(a) and a[idx] is not None else 0
        return 0 if (isinstance(v,float) and v!=v) else v
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

# 回测引擎（可切换评分函数）
def run_bt(data, ind_all, tickers, score_fn, hd=20, ms=60, mh=8, sl=-15, W=400):
    N = min(len(data[t]['c']) for t in tickers)
    c = 100000.; pos = {}; trades = []; dv = []
    
    for i in range(W, N-20):
        scored = [(t, score_fn(ind_all[t], i)) for t in tickers]
        ca = sorted([x for x in scored if x[1] >= ms], key=lambda x: -x[1])
        
        for t in list(pos.keys()):
            cp = common.sf(data[t]['c'], i)
            r = (cp - pos[t]['ep']) / pos[t]['ep'] * 100
            if r < sl or i - pos[t]['ei'] >= hd:
                trades.append({'t':t,'ep':pos[t]['ei'],'xp':i,'ev':round(pos[t]['ep'],2),'xv':round(cp,2),'ret':round(r,2),'h':i-pos[t]['ei']})
                c += pos[t]['sh'] * cp
                del pos[t]
        
        for t, sc in ca:
            if len(pos) >= mh: break
            if t in pos: continue
            al = c / (mh - len(pos))
            if al < 100: continue
            ep = common.sf(data[t]['c'], i)
            if ep <= 0: continue
            pos[t] = {'ep': ep, 'sh': al/ep, 'ei': i}; c -= al
        
        v = c + sum(pd['sh'] * common.sf(data[t]['c'], i) for t, pd in pos.items() if common.sf(data[t]['c'], i) > 0)
        dv.append(v)
    
    for t, pd in pos.items():
        cp = common.sf(data[t]['c'], N-21)
        trades.append({'t':t,'ep':pd['ei'],'xp':N-21,'ev':round(pd['ep'],2),'xv':round(cp,2),'ret':round((cp-pd['ep'])/pd['ep']*100,2),'h':N-21-pd['ei']})
        c += pd['sh'] * cp
    
    total_r = (c/100000-1)*100
    yr = (N-W)/252
    ann = round(((c/100000)**(1/yr)-1)*100, 2) if yr > 0 else 0
    pk = dv[0]; md = 0
    for v in dv:
        if v > pk: pk = v
        dd = (pk-v)/pk*100
        if dd > md: md = dd
    md = round(md, 1)
    dr = [(dv[i]/dv[i-1]-1)*100 for i in range(1, len(dv))]
    mr = sum(dr)/len(dr) if dr else 0
    sd = math.sqrt(sum((x-mr)**2 for x in dr)/len(dr)) if dr else 0
    sh = round(mr/sd*math.sqrt(252), 2) if sd > 0 else 0
    wr = round(len([x for x in trades if x['ret'] > 0])/len(trades)*100, 1) if trades else 0
    
    return {'ann':ann,'sh':sh,'dd':md,'wr':wr,'trades_count':len(trades),'trades':trades}

# 加载数据
t0 = time.time()
data = json.load(open(ws + '/data/us_hist_clean.parquet', 'rb'))
syms = list(data.keys())
print('总池: %d只' % len(syms), flush=True)

# 预计算旧版指标
old_inds = {}; new_inds = {}; tickers = []
for i,sy in enumerate(syms):
    if i%200==0: print('  指标 %d/%d' % (i,len(syms)), flush=True)
    d=data[sy];c=[float(x) for x in d.get('c',[])]
    h=[float(x) for x in d.get('h',[])]
    l=[float(x) for x in d.get('l',[])]
    if len(c)<520: continue
    try:
        oi = old_ind(c,h,l)
        if oi is None: continue
        oi['close'] = c
        old_inds[sy]=oi
        old_inds[sy]=oi
        try:
            ni = new_calc(c,h,l)
            if ni is not None: new_inds[sy]=ni
        except:
            pass
        tickers.append(sy)
    except:
        continue

print('有效: %d只, 新模型: %d只 (%ds)' % (len(tickers), len(new_inds), time.time()-t0), flush=True)

# 跑旧模型回测
print('\n=== 旧模型回测 ===', flush=True)
t1 = time.time()
r_old = run_bt(data, old_inds, tickers, old_score, hd=20, ms=60, mh=8, sl=-15, W=400)
print('h20_m60_mh8_sl-15: %.1f%% ann sh=%.2f dd=%.1f%% wr=%.1f%% tr=%d (%ds)' % (
    r_old['ann'], r_old['sh'], r_old['dd'], r_old['wr'], r_old['trades_count'], time.time()-t1), flush=True)

# 跑新模型回测（只用new_inds里的股票）
print('\n=== 新模型回测 ===', flush=True)
t2 = time.time()
new_tickers = list(new_inds.keys())
r_new = run_bt(data, new_inds, new_tickers, new_score, hd=20, ms=60, mh=8, sl=-15, W=400)
print('h20_m60_mh8_sl-15: %.1f%% ann sh=%.2f dd=%.1f%% wr=%.1f%% tr=%d (%ds)' % (
    r_new['ann'], r_new['sh'], r_new['dd'], r_new['wr'], r_new['trades_count'], time.time()-t2), flush=True)

# 对比
print('\n' + '='*55)
print('新旧模型对比')
print('='*55)
print('')
print('              旧模型         新模型       差距')
print('年化:         %6.1f%%      %6.1f%%     %+6.1f%%' % (r_old['ann'], r_new['ann'], r_new['ann']-r_old['ann']))
print('夏普:         %6.2f        %6.2f      %+6.2f' % (r_old['sh'], r_new['sh'], r_new['sh']-r_old['sh']))
print('回撤:         %6.1f%%      %6.1f%%     %+6.1f%%' % (r_old['dd'], r_new['dd'], r_new['dd']-r_old['dd']))
print('胜率:         %6.1f%%      %6.1f%%     %+6.1f%%' % (r_old['wr'], r_new['wr'], r_new['wr']-r_old['wr']))
print('交易:         %6d         %6d      %+d' % (r_old['trades_count'], r_new['trades_count'], r_new['trades_count']-r_old['trades_count']))
print('股票池:       %d只         %d只' % (len(tickers), len(new_tickers)))
print('')
print('总耗时: %ds' % (time.time()-t0))

# 保存
result = {
    'old': {'ann':r_old['ann'],'sh':r_old['sh'],'dd':r_old['dd'],'wr':r_old['wr'],'trades':r_old['trades_count'],'pool':len(tickers)},
    'new': {'ann':r_new['ann'],'sh':r_new['sh'],'dd':r_new['dd'],'wr':r_new['wr'],'trades':r_new['trades_count'],'pool':len(new_tickers)},
    'params': {'hd':20,'ms':60,'mh':8,'sl':-15},
    'ts': time.strftime('%Y-%m-%d %H:%M:%S')
}
json.dump(result, open(ws+'/data/bt_v5_new_vs_old.json','w'), indent=2)
print('\n结果已保存: data/bt_v5_new_vs_old.json')
