#!/usr/bin/env python3
"""
V5-S 真实回测 — 还原MEMORY.md中的h20_tn5_ms60_mh8_sl15参数
"""
import sys, json, os, time
ws = '/home/hermes/.hermes/openclaw-archive'
sys.path.insert(0, ws + '/scripts')

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

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

# ===== 参数 =====
H = 20        # 持有期
TN = 5        # TopN
MS = 60       # 评分门槛
MH = 8        # 最大同时持仓
SL = -0.15    # 止损
START_DAY = 252  # 从第252天开始

all_d = json.load(open(ws + '/data/us_hist_clean.parquet','r'))
syms = list(all_d.keys())
print('总池: %d' % len(syms), flush=True)

t0 = time.time()

# 预计算
cache = {}
for i,sy in enumerate(syms):
    if i%500==0: print('  %d/%d' % (i,len(syms)), flush=True)
    d=all_d[sy];c=d.get('c',[]);h=d.get('h',[]);l=d.get('l',[])
    if len(c) < 520: continue
    ind = old_ind(c,h,l)
    if ind is not None: cache[sy] = (ind,c)
print('有效池: %d只 (%ds)' % (len(cache), time.time()-t0), flush=True)

max_n = max(len(c) for _,c in cache.values())

# ===== 持仓管理回测 =====
holdings = {}  # sy -> (buy_day, buy_price)
all_trades = []

for day in range(START_DAY, max_n, 5):
    # ① 处理到期和止损
    for sy in list(holdings.keys()):
        ic = cache[sy]; cc = ic[1]
        if day >= len(cc):  # 数据不足，强制卖
            sp = cc[-1]
        else:
            bp = holdings[sy][1]
            sp = cc[day]
            # 检查是否触止损（在持有期内检查全程）
            if day > holdings[sy][0]:
                for d2 in range(holdings[sy][0]+1, min(day+1, len(cc))):
                    if (cc[d2]-bp)/bp <= SL:
                        sp = cc[d2]
                        break
        
        ret = (sp - holdings[sy][1]) / holdings[sy][1]
        if ret is None or (isinstance(ret, float) and ret != ret):
            continue
        all_trades.append(("SELL", sy, day, sp, ret, holdings[sy][0]))
        del holdings[sy]
    
    # ② 评分+选股
    contenders = []
    for sy,(ind,c) in cache.items():
        if sy in holdings: continue
        if day >= len(c): continue
        sc = old_sc(ind, day)
        if sc >= MS:
            contenders.append((sc, sy, c[day]))
    
    contenders.sort(key=lambda x: -x[0])
    
    # ③ 买入
    slots = MH - len(holdings)
    for sc, sy, bp in contenders[:min(TN, slots)]:
        holdings[sy] = (day, bp)
        all_trades.append(("BUY", sy, day, bp, None, sc))

# ④ 最后平仓
last_day = max_n - 1
for sy in list(holdings.keys()):
    ic = cache[sy]; cc = ic[1]
    sp = cc[min(last_day, len(cc)-1)]
    ret = (sp - holdings[sy][1]) / holdings[sy][1]
    if ret is None or (isinstance(ret, float) and ret != ret):
        continue
    all_trades.append(("SELL", sy, last_day, sp, ret, holdings[sy][0]))
    del holdings[sy]

# ===== 统计 =====
buys = [t for t in all_trades if t[0]=="BUY"]
sells = [t for t in all_trades if t[0]=="SELL"]

nt = len(sells)
if nt == 0:
    print('无交易')
    sys.exit(0)

returns = [t[4] for t in sells if t[4] is not None]
total_returns = sum(returns) * 100
wr = sum(1 for r in returns if r > 0) / nt * 100
ar = sum(returns) / nt * 100

# 逐笔净值（均仓MH只）
nav = 1.0; peak = 1.0; mdd = 0.0
navs = [1.0]
buy_idx = 0
active_positions = []

# 按日期排序（从BUY/SELL按day重建净值）
trade_events = []
for t in all_trades:
    if t[0] == "BUY":
        trade_events.append((t[2], "BUY", t[1], t[3]))
    else:
        trade_events.append((t[2], "SELL", t[1], t[3], t[4]))
trade_events.sort(key=lambda x: x[0])

from collections import OrderedDict

pos = {}  # sy -> (buy_price, buy_day)
daily_navs = []

for evt in trade_events:
    if evt[1] == "BUY":
        _, _, sy, bp = evt
        pos[sy] = (bp, evt[0])
    else:  # SELL
        _, _, sy, sp, ret = evt
        if sy in pos:
            del pos[sy]
    
    # 每天按持有数均仓算净值
    n_pos = MH  # 最大仓位
    if len(pos) > 0:
        # 简单方式：累计收益逐笔滚动
        pass

# 更简单的方式：直接用sell的return按均仓滚动
# 假设每笔SELL对应仓位1/MH
nav = 1.0; peak = 1.0; mdd = 0.0
for r in returns:
    nav *= (1 + r/MH)
    if nav > peak: peak = nav
    dd = (peak - nav) / peak
    if dd > mdd: mdd = dd

fn = nav
ann = fn ** (1/5) - 1

# 夏普（用单笔return）
am = sum(returns) / nt
sm = ((sum((r-am)**2 for r in returns) / nt) ** 0.5) if nt > 1 else 1
sharpe = (am / sm) * (252**0.5) if sm > 1e-10 else 0

elapsed = time.time() - t0

print('')
print('='*55)
print('  V5-S 真实回测 (h%02d_tn%d_ms%d_mh%d_sl%d)' % (H, TN, MS, MH, int(SL*-100)))
print('='*55)
print('  时间: %ds' % elapsed)
print('  选股池: %d只, 数据: %d天' % (len(cache), max_n))
print('')
print('  交易: %d笔（买入%d/卖出%d）' % (len(all_trades), len(buys), nt))
print('  胜率: %.1f%%' % wr)
print('  均收益: %.2f%%' % ar)
print('  总收益: %.1f%%' % ((fn-1)*100))
print('  年化: %.1f%%' % (ann*100))
print('  夏普(年化): %.2f' % sharpe)
print('  最大回撤(均仓): %.1f%%' % (mdd*100))

# 也输出无均仓的full nav参考
nav_f = 1.0; pk_f=1.0; mdd_f=0.0
for r in returns:
    nav_f *= (1+r)
    if nav_f>pk_f: pk_f=nav_f
    dd=(pk_f-nav_f)/pk_f
    if dd>mdd_f: mdd_f=dd
print('')
print('  全量复利年化: %.1f%%  全量回撤: %.1f%%' % (nav_f**(1/5)*100-100, mdd_f*100))
