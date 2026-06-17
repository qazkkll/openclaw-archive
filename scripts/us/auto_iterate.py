#!/usr/bin/env python3
"""
自动迭代引擎 — 整晚跑, 不断尝试新模型, 只保留更好的
每次迭代: 尝试一个改动 → 全时段回测 → 比前一个强就保留
"""
import json, math, time, copy, sys, random, os
from datetime import datetime

# ==== 加载数据 ====
with open('data/test_50_hist.json') as f:
    hist = json.load(f)
with open('data/test_stocks_50.json') as f:
    test_stocks = json.load(f)

# ==== 指标计算 ====
def ema(arr,n):
    k=2/(n+1);r=[arr[0]]
    for v in arr[1:]:r.append(v*k+r[-1]*(1-k))
    return r
def sma(arr,n):
    return [None]*(n-1)+[sum(arr[i-n+1:i+1])/n for i in range(n-1,len(arr))]
def calc_all(closes,highs,lows):
    n=len(closes)
    ma5,ma20,ma60=sma(closes,5),sma(closes,20),sma(closes,60)
    g,l=[],[]
    for i in range(1,n):
        d=closes[i]-closes[i-1];g.append(max(d,0));l.append(max(-d,0))
    rsi=[None]*14;ag,al=sum(g[:14])/14,sum(l[:14])/14
    for i in range(14,n):
        rsi.append(100-100/(1+ag/al) if al else 100)
        if i<len(g):ag=(ag*13+g[i])/14;al=(al*13+l[i])/14
    e12,e26=ema(closes,12),ema(closes,26)
    macd=[e12[i]-e26[i] for i in range(n)]
    sig=ema(macd,9);hist_m=[macd[i]-sig[i] for i in range(n)]
    pos52=[None]*252;vol20=sma(closes,20)  # reuse for now
    for i in range(252,n):
        lo,hi=min(closes[i-251:i+1]),max(closes[i-251:i+1])
        pos52.append((closes[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    # 量比
    vols=highs  # reuse highs as proxy for volume if no volume data
    vol5=[sum(closes[max(0,i-4):i+1])/min(5,i+1) for i in range(n)]
    vol20=[sum(closes[max(0,i-19):i+1])/min(20,i+1) for i in range(n)]
    vol_ratio=[vol5[i]/vol20[i] if vol20[i]>0 else 1 for i in range(n)]
    return {'close':closes,'ma5':ma5,'ma20':ma20,'ma60':ma60,'rsi':rsi,'macd_hist':hist_m,'pos52':pos52,'vol_ratio':vol_ratio}

def sf(arr,i):
    return arr[i] if 0<=i<len(arr) and arr[i] is not None else None

# ==== 计算所有股票指标 ====
ind_data = {}
for code, d in hist.items():
    if len(d['close']) < 300: continue
    ind_data[code] = calc_all(d['close'], d['high'], d['low'])

N = min(len(d['close']) for d in ind_data.values())
dates = hist[list(hist.keys())[0]]['dates']

# ==== 时段定义 ====
def find_date_idx(target):
    for i, d in enumerate(dates):
        if d >= target: return i
    return N//2

WARMUP = 260  # 指标预热天数
period_start = WARMUP
periods = [
    ("训练期1", period_start, period_start+150),
    ("训练期2", period_start+150, period_start+300),
    ("验证期1", period_start+300, period_start+420),
    ("验证期2", period_start+420, period_start+500),
    ("测试期", period_start+500, N-20),
    ("熊市段", period_start, period_start+60),  # 最早的时段(可能有熊市)
]

# ==== 评分函数(v6 baseline) ====
def score_fn(i, ind, w, config):
    """config = {'macd_form','rsi_form','w52_form','ma_form','adx_est'}"""
    s=0
    h=sf(ind['macd_hist'],i);hp=sf(ind['macd_hist'],i-1)
    # MACD
    if config.get('macd_form','weighted') == 'weighted':
        if h and hp and h>0 and hp<=0: s+=w[0]*0.8
        elif h and hp and h>hp and h>0: s+=w[0]*0.6
        elif h and h>0: s+=w[0]*0.25
        else: s-=w[0]*0.1
    else:  # momentum
        if h and hp: s+=min(w[0],max(-w[0]*2,h*50))

    # 52W
    p=sf(ind['pos52'],i)
    form = config.get('w52_form','sqrt')
    if p:
        if form == 'sqrt': s+=w[1]*(1-math.sqrt(min(p,100)/100))
        elif form == 'cubic': s+=w[1]*(1-(p/100)**(1/3))
        elif form == 'exp': s+=w[1]*math.exp(-p/25)
        elif form == 'exp15': s+=w[1]*math.exp(-p/15)
        elif form == 'sqrt03': s+=w[1]*(1-(p/100)**0.3)
        elif form == 'linear': s+=w[1]*(1-p/100)
        elif form == 'step': s+=w[1]*(1 if p<20 else 0.75 if p<40 else 0.5 if p<60 else 0.25 if p<80 else 0)
        else: s+=w[1]*(1-math.sqrt(min(p,100)/100))

    # MA
    price=sf(ind['close'],i);m20=sf(ind['ma20'],i);m5=sf(ind['ma5'],i);m60=sf(ind['ma60'],i)
    ma_form=config.get('ma_form','distance')
    if ma_form=='distance':
        if price and m20 and price>m20: s+=w[2]*0.35
        if m5 and m20 and m5>m20: s+=w[2]*0.35
        s+=w[2]*0.3
    elif ma_form=='bullish':
        bull=price and m20 and m60 and price>m20 and m20>m60
        s+=w[2]*(0.8 if bull else 0.3)

    # ADX (estimated)
    s+=w[3]*0.5

    # RSI
    r=sf(ind['rsi'],i)
    rsi_form=config.get('rsi_form','hyperbolic')
    if r:
        if rsi_form=='hyperbolic':
            if r<50: s+=w[4]/(1+((r-25)/15)**2)
            else: s+=w[4]/(1+((75-r)/15)**2)
        elif rsi_form=='sigmoid':
            s+=w[4]/(1+math.exp((r-50)/8))
        elif rsi_form=='linear':
            if r<25: s+=w[4]
            elif r<35: s+=w[4]*0.7
            elif r<50: s+=w[4]*0.5
            elif r<65: s+=w[4]*0.3
            elif r<75: s+=w[4]*0.1
    
    # Volume bonus (if enabled)
    if config.get('vol_bonus', False):
        vr=sf(ind['vol_ratio'],i)
        if vr:
            s+=5*min(1,math.log(vr+0.5,2))
    
    
    # Interaction effects
    if config.get('w52adx_interaction') and p and config.get('adx_est'):
        s+=w[3]*0.2*(1-p/100)  # 52W低位+ADX高分
    if config.get('macdw52_interaction') and p:
        if h and h>0: s+=w[0]*0.15*(1-p/100)
    if config.get('rsivol_bonus'):
        vr2=sf(ind['vol_ratio'],i)
        if vr2 and r and r<30: s+=8*(vr2-1)
    if config.get('sector_relative'):
        pass  # 暂不实现
    if config.get('macd_strength') and h:
        s+=min(8,max(-4,h*3))
    if config.get('vol_trend'):
        vr3=sf(ind['vol_ratio'],i)
        if vr3 and vr3>1.2 and price and m20 and price>m20: s+=5
    if config.get('vol_breakout'):
        vr4=sf(ind['vol_ratio'],i)
        if vr4 and vr4>1.8: s+=8
    return s

# ==== 资金回测(同前, 但加 config) ====
def run_backtest(ind_data, w, config, buy_pct=55, sell_pct=35, pos_size=0.15, start=None, end=None):
    FIXED_POS = 1000000*pos_size
    cash=1000000.0; positions={}; trades=[]
    if start is None: start=max(252, min(p[1] for p in periods))
    if end is None: end=min(N, max(p[2] for p in periods))
    
    for i in range(start, end):
        # Sell
        for code, pos in list(positions.items()):
            ind=ind_data.get(code)
            if not ind: continue
            sc=score_fn(i,ind,w,config)
            p=sf(ind['close'],i);m20=sf(ind['ma20'],i);mh=sf(ind['macd_hist'],i)
            if (sc<sell_pct) or (p and m20 and p<m20 and mh and mh<0):
                if p and p>0:
                    pnl=(p-pos['entry_price'])/pos['entry_price']
                    cash+=FIXED_POS*(1+pnl); trades.append(pnl*100)
                    del positions[code]
        # Buy
        slots = 5-len(positions)
        available = int(cash/FIXED_POS)
        if available>0 and slots>0:
            cands=[]
            for code in ind_data:
                if code in positions: continue
                sc=score_fn(i,ind_data[code],w,config)
                if sc>=buy_pct:
                    p=sf(ind_data[code]['close'],i)
                    if p and p>0: cands.append((code,sc,p))
            cands.sort(key=lambda x:x[1],reverse=True)
            for code,sc,p in cands[:min(available,slots)]:
                if code in positions: continue
                positions[code]={'entry_price':p}; cash-=FIXED_POS

    for code,pos in list(positions.items()):
        ind=ind_data.get(code)
        if ind:
            p=sf(ind['close'],end-1)
            if p and p>0:
                pnl=(p-pos['entry_price'])/pos['entry_price']
                cash+=FIXED_POS*(1+pnl); trades.append(pnl*100)
    ret=(cash-1000000)/1000000*100
    wins=[t for t in trades if t>0]
    return {'ret':round(ret,2),'wr':round(len(wins)/len(trades)*100,1) if trades else 0,'trades':len(trades)}

# ==== 多时段评价 ====
def evaluate(ind_data, w, config, buy=55, sell=35):
    """返回综合评分: avg_ret*0.5 + positive_ratio*20*0.3 + worst_period_penalty"""
    results=[]
    for label,s,e in periods:
        if s>=e or e>N: continue
        r=run_backtest(ind_data,w,config,buy,sell,start=s,end=e)
        results.append({'period':label,'ret':r['ret'],'wr':r['wr'],'trades':r['trades']})
    
    rets=[r['ret'] for r in results]
    avg_ret=sum(rets)/len(rets)
    positive=sum(1 for r in rets if r>=0)
    worst=min(rets)
    
    score = avg_ret*0.5 + (positive/len(rets)*30)*0.3 + max(0,worst)*0.2  # worst period bonus
    # Penalize very negative worst periods
    if worst < -5: score += worst * 0.3
    
    return {'avg_ret':round(avg_ret,2),'positive':positive,'total':len(rets),'score':round(score,2),
            'worst':round(worst,2),'details':results}

# ==== 试验池 ====
baseline_w = [20,30,15,20,15]
baseline_config = {'macd_form':'weighted','w52_form':'sqrt','ma_form':'distance','rsi_form':'hyperbolic','adx_est':True,'vol_bonus':False}

# Log
log = []
iteration = 0
best_score = -999
best_model = None

def log_model(label, w, config, eval_res):
    log.append({'iter':len(log),'time':datetime.now().strftime('%H:%M'),'label':label,
                'w':w,'config':config,'eval':eval_res})

# Baseline
baseline_eval = evaluate(ind_data, baseline_w, baseline_config)
log_model("BASELINE v6", baseline_w, baseline_config, baseline_eval)
best_score = baseline_eval['score']
best_model = (baseline_w.copy(), dict(baseline_config))
print(f"[BASELINE] score={best_score} avg_ret={baseline_eval['avg_ret']}% pos={baseline_eval['positive']}/{baseline_eval['total']}")

# ===== 自动迭代 =====
mutations_pool = [
    # (type, description, apply_function)
    # Phase 1: Weight variations around optimal
    ('w', '[18,30,17,20,15]', lambda w,c: ([18,30,17,20,15],c)),
    ('w', '[20,28,15,22,15]', lambda w,c: ([20,28,15,22,15],c)),
    ('w', '[20,25,15,25,15]', lambda w,c: ([20,25,15,25,15],c)),
    ('w', '[22,30,13,20,15]', lambda w,c: ([22,30,13,20,15],c)),
    ('w', '[15,35,15,20,15]', lambda w,c: ([15,35,15,20,15],c)),
    ('w', '[20,30,10,25,15]', lambda w,c: ([20,30,10,25,15],c)),
    ('w', '[25,25,15,20,15]', lambda w,c: ([25,25,15,20,15],c)),
    ('w', '[15,30,20,20,15]', lambda w,c: ([15,30,20,20,15],c)),
    
    # Phase 2: Factor form changes
    ('f', 'rsi=sigmoid', lambda w,c: (w, {**c,'rsi_form':'sigmoid'})),
    ('f', 'rsi=linear', lambda w,c: (w, {**c,'rsi_form':'linear'})),
    ('f', 'w52=exp', lambda w,c: (w, {**c,'w52_form':'exp'})),
    ('f', 'w52=cubic', lambda w,c: (w, {**c,'w52_form':'cubic'})),
    ('f', 'w52=step', lambda w,c: (w, {**c,'w52_form':'step'})),
    ('f', 'w52=linear', lambda w,c: (w, {**c,'w52_form':'linear'})),
    ('f', 'macd=momentum', lambda w,c: (w, {**c,'macd_form':'momentum'})),
    ('f', 'ma=bullish', lambda w,c: (w, {**c,'ma_form':'bullish'})),
    
    # Phase 3: New factors
    ('n', '+vol_bonus', lambda w,c: (w, {**c,'vol_bonus':True})),
    
    # Phase 4: Threshold changes (iterating with best weight)
    ('t', 'buy50/sell30', lambda w,c: None),  # special handling
    ('t', 'buy60/sell40', lambda w,c: None),
    ('t', 'buy55/sell30', lambda w,c: None),
    ('t', 'buy55/sell40', lambda w,c: None),
    
    # Phase 5: Threshold fine-tuning
    ('t', 'buy50/sell40', lambda w,c: None),
    ('t', 'buy60/sell35', lambda w,c: None),
    ('t', 'buy55/sell45', lambda w,c: None),
    ('t', 'buy58/sell38', lambda w,c: None),
    
    # Phase 6: Interaction effects (multiplicative)
    ('i', 'w52*w_adx bonus', lambda w,c: (w, {**c,'w52adx_interaction':True})),
    ('i', 'macd*w52 bonus', lambda w,c: (w, {**c,'macdw52_interaction':True})),
    ('i', 'rsi*vol_bonus', lambda w,c: (w, {**c,'rsivol_bonus':True})),
    
    # Phase 7: Sector-relative scoring (stock score minus sector avg)
    ('i', 'sector_relative', lambda w,c: (w, {**c,'sector_relative':True})),
    
    # Phase 8: MACD strength scaling
    ('f', 'macd_strength', lambda w,c: (w, {**c,'macd_strength':True})),
    ('f', 'macd_confirmed', lambda w,c: (w, {**c,'macd_form':'confirmed'})),
    
    # Phase 9: Volume-weighted
    ('n', '+vol_trend', lambda w,c: (w, {**c,'vol_trend':True})),
    ('n', '+vol_breakout', lambda w,c: (w, {**c,'vol_breakout':True})),
    
    # Phase 10: Position sizing strategies
    ('t', 'pos10pct', lambda w,c: None),
    ('t', 'pos20pct', lambda w,c: None),
    
    # Phase 11: Factor exponent experiments
    ('f', 'w52=sqrt(x^0.3)', lambda w,c: (w, {**c,'w52_form':'sqrt03'})),
    ('f', 'w52=exp(-x/15)', lambda w,c: (w, {**c,'w52_form':'exp15'})),
]

# Main iteration loop
no_improvement = 0
tried = set()

while no_improvement < 30 and iteration < 150:
    iteration += 1
    mutation = mutations_pool[(iteration-1) % len(mutations_pool)]
    mt, label, apply_fn = mutation
    
    # Skip if already tried
    if label in tried:
        continue
    tried.add(label)
    
    try:
        if mt in ('w','f','n'):
            new_w, new_config = apply_fn(best_model[0], best_model[1])
            eval_res = evaluate(ind_data, new_w, new_config)
        elif mt == 't':
            # Test different buy/sell thresholds with current best weight
            thresholds = {'buy50/sell30':(50,30),'buy60/sell40':(60,40),'buy55/sell30':(55,30),'buy55/sell40':(55,40)}
            if label in thresholds:
                buy,sell = thresholds[label]
                w,c = best_model
                eval_res = evaluate(ind_data, w, c, buy=buy, sell=sell)
                label = f"{label}(best_weight)"
        elif mt == 'i':
            config_copy = dict(best_model[1])
            if 'w52adx_interaction' in label:
                config_copy['w52adx_interaction'] = True
            else:
                config_copy['macdw52_interaction'] = True
            eval_res = evaluate(ind_data, best_model[0], config_copy)
        
        score = eval_res['score']
        
        if score > best_score * 0.98:  # within 2% = keep
            is_better = score > best_score
            if is_better:
                best_score = score
                best_model = (new_w if mt in ('w','f','n') else best_model[0], 
                            new_config if mt in ('w','f','n','i') else best_model[1])
                no_improvement = 0
            else:
                no_improvement += 1
            
            marker = "✅ NEW BEST!" if is_better else "  +kept "
            print(f"[{iteration:2d}] {marker} {label:25s} score={score:+.2f} ret={eval_res['avg_ret']:+.2f}% pos={eval_res['positive']}/{eval_res['total']} wr={eval_res['details'][-1]['wr']}%")
            log_model(f"{'⭐' if is_better else '  '}{label}", 
                     new_w if mt in ('w','f','n') else best_model[0],
                     new_config if mt in ('w','f','n','i') else best_model[1], eval_res)
        else:
            print(f"[{iteration:2d}]     rejected {label:25s} score={score:+.2f} < {best_score:+.2f}")
            no_improvement += 1
        
    except Exception as e:
        print(f"[{iteration:2d}] ❌ error {label}: {e}")
        no_improvement += 1
    
    # Reset no_improvement counter periodically to keep trying
    if iteration % 20 == 0:
        no_improvement = max(0, no_improvement - 8)

# Final report
print(f"\n\n{'='*70}")
print(f"最终结果: {len(log)} 次试验")
print(f"{'='*70}")
print(f"最优模型: {best_model}")
final_eval = evaluate(ind_data, best_model[0], best_model[1])
print(f"最终评分: {final_eval['score']}")
print(f"最终收益: {final_eval['avg_ret']}%")
print(f"正向时段: {final_eval['positive']}/{final_eval['total']}")
print(f"最差时段: {final_eval['worst']}%")

# Write log
with open('iteration_log/full_log.json','w') as f:
    json.dump(log, f, indent=2)
with open('iteration_log/final_model.json','w') as f:
    json.dump({'weights':best_model[0],'config':best_model[1],'eval':final_eval}, f, indent=2)
print(f"\n✅ 日志已保存")
