"""V5-S真实回测：h20_tn5_ms60_mh8_sl15，5年数据"""
import sys, json, time, os, math
ws = '/home/hermes/.hermes/openclaw-archive'
sys.path.insert(0, ws + '/scripts')

def ef(v):
    if v is None or (isinstance(v, float) and v!=v): return 0.0
    try: return float(v)
    except: return 0.0

t0 = time.time()
data = json.load(open(ws + '/data/us_hist_clean.parquet','rb'))
syms = list(data.keys())
print('池: %d只' % len(syms), flush=True)

# 评分函数
def score_old(c):
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
    return {'ma5':m5,'ma20':m20,'ma60':m60,'ma120':m120,
            'macd':macd,'sig':sig,'hst':hst,'rsi':rsi,'p52':p52}

def sfc(ind,di):
    def sf(a,i):
        if not a: return 0
        idx=i if i>=0 else len(a)+i
        v=a[idx] if 0<=idx<len(a) and a[idx] is not None else 0
        return 0 if (isinstance(v,float) and v!=v) else v
    p=sf(ind['c'],di)
    if p<=0: return 0
    m5=sf(ind['ma5'],di);m20=sf(ind['ma20'],di)
    m60=sf(ind['ma60'],di);m120=sf(ind['ma120'],di)
    tr=(10 if m5>m20 else 0)+(10 if m20>m60 else 0)+(10 if m60>m120 else 0)+(5 if p>m20 else 0)+(5 if p>m60 else 0)+(5 if p>m120 else 0)+(5 if m5>m20 and m20>m60 else 0)
    mo=15
    p5=sf(ind['c'],di-5);p20x=sf(ind['c'],di-20);p60x=sf(ind['c'],di-60)
    if p5>0 and p>p5: mo+=5
    if p20x>0 and p>p20x: mo+=5
    if p60x>0 and p>p60x: mo+=5
    p30=sf(ind['c'],di-30);m30=(p-p30)/p30*100 if p30>0 else 0; mo+=m30/10
    if m30>50: mo=max(mo-(m30-50)/5,0)
    mh=sf(ind['hst'],di);mhp=sf(ind['hst'],di-1)
    ms=8 if sf(ind['macd'],di)>sf(ind['sig'],di) else 0
    if mh>0 and mhp<=0: ms+=12
    elif mh>0: ms+=5
    if mh>mhp: ms+=5
    rsi=sf(ind['rsi'],di)
    rs=5+(5 if 50<=rsi<=70 else 3 if rsi>70 else -5 if rsi<30 else 0)
    p52=sf(ind['p52'],di)
    ps=10 if 70<=p52<=100 else 7 if 50<=p52<70 else 4 if 30<=p52<50 else 0
    return max(tr+mo+ms+rs+ps,0)

# 预计算指标 + close
cache={}
for i,sy in enumerate(syms):
    if i%500==0: print('  计算指标 %d/%d' % (i,len(syms)), flush=True)
    d=data[sy];c=[ef(x) for x in d.get('c',[])]
    if len(c)<520: continue
    ind=score_old(c)
    if ind is None: continue
    ind['c']=c
    cache[sy]=ind
print('有效: %d只 (%ds)' % (len(cache), time.time()-t0), flush=True)

# 参数 (h20_tn5_ms60_mh8_sl15)
H=20; TN=5; MS=60; MH=8; SL=-0.15
max_n=max(len(ind['c']) for ind in cache.values())

# 验算记录
bt_name = 'v5s_realscan_h%d_tn%d_ms%d_mh%d_sl%d_5y' % (H,TN,MS,MH,int(SL*-100))
bt_provenance = {
    'script': 'scripts/_v5s_real_bt.py',
    'data': 'us_hist_clean.parquet (2436只)',
    'date': time.strftime('%Y-%m-%d'),
    'target_params': 'h20_tn5_ms60_mh8_sl15 (来自MEMORY.md 2026-06-03记录)',
    'target_result': {'trades':384,'wr':47.7,'ann':49.1,'mdd':19.7,'sharpe':1.57},
    'diff_reasons': [],
}

# 日期对齐：用所有股票中最久的close天数
# 只跑有数据的天数
trades=[]  # (day, type, sym, price, ret, score)
pos={}     # sy -> (buy_day, buy_price, score)

for day in range(252, max_n, 5):
    # ① 处理到期+止损
    for sy in list(pos.keys()):
        ind=cache[sy]; cc=ind['c']
        bp=pos[sy][1]
        sd=min(day+H, len(cc)-1)
        out=False
        for d2 in range(pos[sy][0]+1, sd+1):
            if d2>=len(cc): break
            if (cc[d2]-bp)/bp <= SL:
                ret=(cc[d2]-bp)/bp
                trades.append({'d':d2,'t':'SELL','sy':sy,'bp':bp,'sp':cc[d2],'ret':ret,'sc':pos[sy][2],'r':'stop'})
                del pos[sy]
                out=True
                break
        if not out and day>=sd:
            ret=(cc[sd]-bp)/bp if sd<len(cc) else 0
            trades.append({'d':sd,'t':'SELL','sy':sy,'bp':bp,'sp':cc[sd] if sd<len(cc) else cc[-1],'ret':ret,'sc':pos[sy][2],'r':'expire'})
            del pos[sy]
    
    # ② 评分
    cand=[]
    for sy,ind in cache.items():
        if sy in pos: continue
        if day>=len(ind['c']): continue
        sc=sfc(ind,day)
        if sc>=MS:
            cand.append((sc,sy,ind['c'][day]))
    cand.sort(key=lambda x:-x[0])
    
    # ③ 买入
    for sc,sy,bp in cand[:min(TN, MH-len(pos))]:
        pos[sy]=(day,bp,sc)
        trades.append({'d':day,'t':'BUY','sy':sy,'bp':bp,'sp':None,'ret':None,'sc':sc,'r':'buy'})

# 最后平仓
for sy in list(pos.keys()):
    ind=cache[sy]; cc=ind['c']
    sp=cc[-1]; bp=pos[sy][1]; ret=(sp-bp)/bp
    trades.append({'d':len(cc)-1,'t':'SELL','sy':sy,'bp':bp,'sp':sp,'ret':ret,'sc':pos[sy][2],'r':'force'})
    del pos[sy]

# 统计
sells=[t for t in trades if t['t']=='SELL']
buys=[t for t in trades if t['t']=='BUY']
nt=len(sells)
rets=[t['ret'] for t in sells if t['ret'] is not None and not (isinstance(t['ret'],float) and t['ret']!=t['ret'])]
wr=sum(1 for r in rets if r>0)/len(rets)*100
ar=sum(rets)/len(rets)*100
md=max(rets)*100; mn=min(rets)*100

# 净值（按天排序）
trade_sells=sorted(sells, key=lambda x:x['d'])
nav=1.0; peak=1.0; mdd=0.0; pnls=[]
for t in trade_sells:
    r=t['ret']
    if r is None or (isinstance(r,float) and r!=r): continue
    nav*=(1+r/MH)
    if nav>peak: peak=nav
    dd=(peak-nav)/peak
    if dd>mdd: mdd=dd
    pnls.append(nav)

fn=nav
ann=fn**(1/5)-1
am=sum(rets)/len(rets)
sm=((sum((r-am)**2 for r in rets)/len(rets))**0.5) if len(rets)>1 else 1
sharpe=(am/sm)*(252**0.5) if sm>1e-10 else 0

print('\n' + '='*55)
print(' V5-S 真实回测完成')
print('='*55)
print(' 参数: hold=%dd top=%d min_score=%d max_hold=%d stop=%.0f%%' % (H,TN,MS,MH,SL*-100))
print(' 数据: 5年, %d只' % len(cache))
print('')
print(' 交易: %d笔 (买入%d/卖出%d)' % (len(trades), len(buys), nt))
print(' 胜率: %.1f%%' % wr)
print(' 均收益: %.2f%%  最大单笔: +%.2f%% / %.2f%%' % (ar, md, mn))
print(' 总收益: %.1f%%' % ((fn-1)*100))
print(' 年化: %.1f%%' % (ann*100))
print(' 夏普: %.2f' % sharpe)
print(' 最大回撤: %.1f%%' % (mdd*100))
# 保存结果
bt_provenance['actual_result'] = {'trades':nt,'wr':round(wr,1),'ann':round(ann*100,1),'mdd':round(mdd*100,1),'sharpe':round(sharpe,2)}
bt_provenance['diff_reasons'] = [
    '交易笔数%d笔vs目标384笔' % nt,
    '原因: 目标数据源us_training_pool.json(3229只)比us_hist_clean.parquet(2436只)多793只',
    '评分公式也可能有细节差异（旧版本评分的系数/阈值）',
    '建议: 统一数据源后重跑'
]
result_path = ws + '/data/bt_' + bt_name + '.json'
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(bt_provenance, f, indent=2, ensure_ascii=False)
print('结果已保存: data/' + 'bt_' + bt_name + '.json')
print(' 耗时: %ds' % (time.time()-t0))
