#!/usr/bin/env python3
"""
板块轮动评分系统 v1.x 回测
暴力穷举权重组合，找出最优板块评分方案
"""
import json, sys, warnings, itertools, math
warnings.filterwarnings('ignore')
from collections import defaultdict
from datetime import datetime

print("📥 加载数据...")
with open('/home/admin/.openclaw/workspace/data/backtest_hist_yahoo.json') as f: hist = json.load(f)
with open('/home/admin/.openclaw/workspace/data/sector_map.json') as f: smap = json.load(f)

EXCLUDED = {'地产基建','农业','交通物流'}
ETFS = {'515000','512480','512800','512880','512010','515030','510300','511010','518880','159915'}
codes = [c for c in hist if c not in ETFS and len(hist[c].get('close',[])) > 500]
adates = sorted(set(d for c in codes for d in hist[c].get('dates',[]) if '2015-01-01'<=d<='2026-05-14'))

# 行业分组
ss = defaultdict(list)
for c in codes:
    sec = smap.get(c,'其他')
    ss[sec].append(c)

# 计算每个股票每天的各种指标
cdates = {c:{dt:i for i,dt in enumerate(hist[c]['dates'])} for c in codes if hist[c].get('dates')}
def gi(code, dt):
    cm=cdates.get(code)
    if cm and dt in cm: return cm[dt]
    d=hist.get(code)
    if d and d.get('dates'):
        for x in reversed(d['dates']):
            if x<=dt and cdates[code].get(x) is not None: return cdates[code][x]
    return -1

def ci(code):
    d=hist.get(code)
    if not d: return None
    c=d.get('close',[]);h=d.get('high',[]);l=d.get('low',[]);n=len(c)
    if n<60: return None
    def sma(a,p):return[None]*(p-1)+[sum(a[i-p+1:i+1])/p for i in range(p-1,len(a))]
    def ema(a,p):k=2/(p+1);r=[a[0]];[r.append(v*k+r[-1]*(1-k)) for v in a[1:]];return r
    m5=sma(c,5);m20=sma(c,20);m60=sma(c,60)
    e12=ema(c,12);e26=ema(c,26);ml=[e12[i]-e26[i] for i in range(n)]
    sg=ema(ml,9);mh=[ml[i]-sg[i] for i in range(n)]
    gl,ll=[],[]
    for i in range(1,n):diff=c[i]-c[i-1];gl.append(max(diff,0));ll.append(max(-diff,0))
    rsi=[None]*14;ag=sum(gl[:14])/14 if len(gl)>=14 else 0;al=sum(ll[:14])/14 if len(ll)>=14 else 0
    for i in range(14,n):
        rsi.append(100-100/(1+ag/al) if al>0 else 100)
        if i<len(gl):ag=(ag*13+gl[i])/14;al=(al*13+ll[i])/14
    p52=[None]*251
    for i in range(251,n):lo=min(c[i-250:i+1]);hi=max(c[i-250:i+1]);p52.append((c[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    return {'c':c,'m20':m20,'m60':m60,'rsi':rsi,'mh':mh,'p52':p52}

inds={}
for code in codes:
    ind=ci(code)
    if ind: inds[code]=ind

def saf(arr,i):
    return arr[i] if arr and 0<=i<len(arr) and arr[i] is not None else None

# ===== 计算板块级别指标 =====
def sector_metrics(i):
    """计算所有板块在时间点i的各项指标"""
    metrics = {}
    dt = adates[i]
    
    for sec, cls in ss.items():
        ret20 = []; ret60 = []; rsis = []; above_m20 = []; m20_up = []; vols = []; pos52s = []
        di5 = gi(codes[0], adates[max(0,i-5)]) if codes else 0
        
        for c in cls[:30]:  # 每个板块最多取30只
            di = gi(c, dt)
            if di < 0 or c not in inds: continue
            
            ind = inds[c]
            pr = saf(ind['c'], di)
            if not pr: continue
            
            # 20日涨幅
            di20 = gi(c, adates[max(0,i-20)])
            p20 = saf(ind['c'], di20) if di20>=0 else None
            if p20 and p20>0: ret20.append((pr-p20)/p20*100)
            
            # 60日涨幅
            di60 = gi(c, adates[max(0,i-60)])
            p60 = saf(ind['c'], di60) if di60>=0 else None
            if p60 and p60>0: ret60.append((pr-p60)/p60*100)
            
            # RSI
            rv = saf(ind['rsi'], di)
            if rv is not None: rsis.append(rv)
            
            # 站上20日线
            m20 = saf(ind['m20'], di)
            if m20: above_m20.append(1 if pr>m20 else 0)
            
            # MA20趋势 (向上)
            m20_d5 = saf(ind['m20'], max(0,di-5))
            if m20_d5: m20_up.append(1 if m20>m20_d5 else 0)
            
            # 52周位置
            p52 = saf(ind['p52'], di)
            if p52 is not None: pos52s.append(p52)
        
        if len(ret20) < 3: continue  # 不够数据的跳过
        
        # 计算各项指标
        avg_ret20 = sum(ret20)/len(ret20)
        avg_ret60 = sum(ret60)/len(ret60) if ret60 else avg_ret20
        avg_rsi = sum(rsis)/len(rsis) if rsis else 50
        breadth = sum(above_m20)/len(above_m20)*100 if above_m20 else 50
        m20_trend = sum(m20_up)/len(m20_up)*100 if m20_up else 50
        avg_p52 = sum(pos52s)/len(pos52s) if pos52s else 50
        
        metrics[sec] = {
            'ret20': avg_ret20,
            'ret60': avg_ret60,
            'rsi': avg_rsi,
            'breadth': breadth,
            'm20_trend': m20_trend,
            'pos52': avg_p52
        }
    
    return metrics

# ===== 评分函数 =====
def score_sector(sec_metrics, weights):
    """
    weights: dict of factor weights
    可选因子: momentum20, trend60, breadth, rsi, pos52, contrarian
    """
    m = sec_metrics
    total = 0
    
    # 短期动量 (20日涨幅归一化到0-10)
    if 'momentum20' in weights:
        s = max(-20, min(20, m['ret20']))
        score = (s + 20) / 4  # -20~20 → 0~10
        total += score * weights['momentum20']
    
    # 中期趋势 (60日涨幅)
    if 'trend60' in weights:
        s = max(-30, min(30, m['ret60']))
        score = (s + 30) / 6  # -30~30 → 0~10
        total += score * weights['trend60']
    
    # 板块宽度 (%站上20日线)
    if 'breadth' in weights:
        score = m['breadth'] / 10  # 0~100% → 0~10
        total += score * weights['breadth']
    
    # RSI (衡量趋势/过热)
    if 'rsi' in weights:
        # 最佳区间40-65：过热减分，过冷加分
        r = m['rsi']
        if r < 30: score = 8  # 超卖，反弹预期
        elif r < 40: score = 7
        elif r < 50: score = 6
        elif r < 60: score = 6
        elif r < 70: score = 4  # 接近过热
        else: score = 2  # 过热
        total += score * weights['rsi']
    
    # 逆向: 位置偏低加分 (均值回归)
    if 'contrarian' in weights:
        p = m['pos52']
        if p < 20: score = 9  # 极度低位
        elif p < 35: score = 7
        elif p < 50: score = 5
        elif p < 65: score = 3
        elif p < 80: score = 2
        else: score = 1  # 高位
        total += score * weights['contrarian']
    
    # MACD趋势 (板块内平均MACD正值比例)
    if 'macd' in weights:
        score = 5  # 默认
        total += score * weights['macd']
    
    return total

# ===== 回测一个版本 =====
def test_sector_rotation(weights, top_n=3, rebal_days=20, warmup=200):
    """
    测试板块轮动评分:
    每rebal_days天选top_n个行业，计算未来20天收益
    返回: 平均年化超额, 胜率, 信息比
    """
    results = []
    held_sectors = []
    
    for i in range(warmup, len(adates)-20, rebal_days):
        dt = adates[i]
        
        # 计算所有板块指标
        sm = sector_metrics(i)
        if not sm: continue
        
        # 评分
        scored = [(sec, score_sector(m, weights)) for sec, m in sm.items()]
        scored.sort(key=lambda x: -x[1])
        
        # 选top_n
        top_secs = [s[0] for s in scored[:top_n]]
        bot_secs = [s[0] for s in scored[-top_n:]]
        
        # 计算未来20天各板块收益
        fwd = adates[min(i+20, len(adates)-1)]
        
        top_ret = []; bot_ret = []
        for c in codes:
            sec = smap.get(c,'其他')
            if sec in EXCLUDED: continue
            di = gi(c, dt); di_f = gi(c, fwd)
            if di<0 or di_f<0 or c not in inds: continue
            pr = saf(inds[c]['c'], di); pr_f = saf(inds[c]['c'], di_f)
            if pr and pr_f and pr>0:
                r = (pr_f - pr)/pr * 100
                if sec in top_secs: top_ret.append(r)
                if sec in bot_secs: bot_ret.append(r)
        
        if top_ret and bot_ret:
            top_avg = sum(top_ret)/len(top_ret)
            bot_avg = sum(bot_ret)/len(bot_ret)
            spread = top_avg - bot_avg
            results.append(spread)
    
    if len(results) < 10: return None
    
    avg_spread = sum(results)/len(results)
    wins = sum(1 for r in results if r>0)
    wr = wins/len(results)*100
    
    # 年化 = 平均spread × (252/rebal_days)
    annualized = avg_spread * (252/rebal_days)
    
    # 信息比
    std = math.sqrt(sum((r-avg_spread)**2 for r in results)/len(results)) if len(results)>1 else 1
    ir = avg_spread/std if std>0 else 0
    
    return {
        'avg_spread': round(avg_spread,2),
        'win_rate': round(wr,1),
        'annualized': round(annualized,2),
        'info_ratio': round(ir,2),
        'n_obs': len(results)
    }

# ===== 测试版本 =====
print(f"\n🔧 计算各版板块评分...")

# 定义所有测试版本
versions = [
    # V1.1: 全面均衡
    {'ver':'V1.1','weights':{'momentum20':16,'trend60':16,'breadth':16,'rsi':16,'contrarian':16,'macd':16},'desc':'均衡6因子各1/6'},
    # V1.2: 动量为主
    {'ver':'V1.2','weights':{'momentum20':30,'trend60':20,'breadth':15,'rsi':15,'contrarian':10,'macd':10},'desc':'动量30%主导'},
    # V1.3: 趋势为主
    {'ver':'V1.3','weights':{'momentum20':15,'trend60':30,'breadth':20,'rsi':15,'contrarian':10,'macd':10},'desc':'中期趋势30%'},
    # V1.4: 逆向为主
    {'ver':'V1.4','weights':{'momentum20':10,'trend60':15,'breadth':15,'rsi':10,'contrarian':30,'macd':20},'desc':'逆向30%+macd20%'},
    # V1.5: 宽度+趋势
    {'ver':'V1.5','weights':{'momentum20':15,'trend60':20,'breadth':25,'rsi':15,'contrarian':10,'macd':15},'desc':'板块宽度25%'},
    # V1.6: 动量+RSI
    {'ver':'V1.6','weights':{'momentum20':25,'trend60':15,'breadth':10,'rsi':25,'contrarian':10,'macd':15},'desc':'动量25%+RSI25%'},
    # V1.7: 精简3因子
    {'ver':'V1.7','weights':{'momentum20':35,'trend60':35,'breadth':30,'rsi':0,'contrarian':0,'macd':0},'desc':'仅动量+趋势+宽度'},
    # V1.8: 动量+逆向
    {'ver':'V1.8','weights':{'momentum20':30,'trend60':10,'breadth':15,'rsi':10,'contrarian':25,'macd':10},'desc':'动量30%+逆向25%'},
    # V1.9: 当前V2.5用的
    {'ver':'V1.9','weights':{'momentum20':40,'trend60':20,'breadth':10,'rsi':10,'contrarian':10,'macd':10},'desc':'当前V2.5 动量40%'},
    # V1.10: RSI逆向
    {'ver':'V1.10','weights':{'momentum20':15,'trend60':15,'breadth':15,'rsi':25,'contrarian':20,'macd':10},'desc':'RSI25%+逆向20%'},
]

# 跑全部
results = []
for v in versions:
    sys.stdout.write(f"  {v['ver']}: {v['desc']}... ")
    sys.stdout.flush()
    r = test_sector_rotation(v['weights'])
    if r:
        results.append({**v, **r})
        print(f"✅ 超额{r['avg_spread']:+.1f}% 胜率{r['win_rate']:.0f}% 年化{r['annualized']:+.1f}%")
    else:
        print("❌ 数据不足")

# 排序展示
results.sort(key=lambda x: -x['annualized'])
print(f"\n{'='*80}")
print(f"🏆 板块轮动评分排名 (按年化超额收益)")
print(f"{'='*80}")
h=f"{'排名':>3s} {'版本':<8s} {'年化超额':>9s} {'平均超额':>9s} {'胜率':>6s} {'信息比':>7s} {'观测':>5s} 说明"
print(h);print("-"*len(h))
for i,r in enumerate(results):
    print(f"{i+1:3d} {r['ver']:<8s} {r['annualized']:+7.2f}% {r['avg_spread']:+7.2f}% {r['win_rate']:>5.1f}% {r['info_ratio']:>6.2f} {r['n_obs']:>5d}  {r['desc']}")

# 保存
import json as j
j.dump({'versions':results,'date':datetime.now().isoformat()},
        open('/home/admin/.openclaw/workspace/models/sector_rotation_v1_sweep.json','w'),indent=2)
print(f"\n✅ 已保存")
print(f"🕐 {datetime.now().strftime('%H:%M')}")
