#!/usr/bin/env python3
"""
us_v9_daily_score.py — 绿箭V9-Lottery 每日评分

基于us_all_ohlcv.json + V9模型(us_v9_lottery.json) + 最新日数据
输出50维特征 → V9概率(0-1) → 排序出票
"""
import sys, os, json, time, warnings, numpy as np, pandas as pd, xgboost as xgb
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = '/home/hermes/.hermes/openclaw-archive/data'
MD_DIR = f'{DATA_DIR}/models'
DATA_FILE = f'{DATA_DIR}/us_all_ohlcv.json'
MODEL_FILE = f'{MD_DIR}/us_v9_lottery.json'

FEAT_V9 = ["ma5","ma5_ratio","ma20_ratio","ma60_ratio","vol5","vol20","vol_ratio",
    "ema12","ema26","macd","macd_signal","macd_hist","rsi14","k","d","j",
    "bb_upper","bb_lower","bb_width","bb_position","vol_ratio_ma5","vol_ratio_ma20",
    "adx","plus_di","minus_di","price_position","price_position_60","cmf",
    "vix_close","close_log","close_x_vol","plus_di_x_low_vol","adx_x_rsi","bb_x_vol","rsi_x_kdj","low_price",
    "price_range_norm","price_accel","oversold","trend_strength","volatility_expansion",
    "pos_60d_channel","kdj_j","bb_squeeze","rsi_trend_5d","ma5_x_ma20_cross",
    "price_vs_vwap","consecutive_up","consecutive_down","reversal_pattern"]

def rsi(p, period=14):
    if len(p)<period+1: return 50.0
    d=np.diff(p[-(period+1):]); g=np.sum(d[d>0]); lv=abs(np.sum(d[d<0]))
    return 100-100/(1+g/lv) if lv>0 else 100

def feats50(c,h,l,o,v,pos):
    """Compute 50 features for one stock at position pos"""
    if pos < 100 or pos > len(c): return None
    c=np.array(c[:pos],dtype=float); ct=float(c[-1]); n5=min(5,len(c)); n20=min(20,len(c)); n60=min(60,len(c))
    ma5=float(np.mean(c[-n5:])); ma20=float(np.mean(c[-n20:])); ma60=float(np.mean(c[-n60:]))
    r5=float(np.std(c[-n5:])); r20=float(np.std(c[-n20:]))
    rsi14=rsi(list(c),14); low14=float(np.min(c[-14:])); high14=float(np.max(c[-14:]))
    rsv=(ct-low14)/(high14-low14)*100 if high14>low14 else 50; k=d=j=rsv
    bb_mid=ma20; bb_std=r20 if r20>0 else 1; bu=bb_mid+2*bb_std; bl=bb_mid-2*bb_std; bb_w=(bu-bl)/bb_mid if bb_mid>0 else 0
    h=np.array(h[:pos],dtype=float)[-n20:]; l=np.array(l[:pos],dtype=float)[-n20:]
    o=o[:pos] if o is not None and len(o)>=pos else None
    v=v[:pos] if v is not None and len(v)>=pos else None
    low20=float(np.min(c[-n20:])); high20=float(np.max(c[-n20:]))
    low60=float(np.min(c[-60:])); high60=float(np.max(c[-60:]))
    pp20=(ct-low20)/(high20-low20) if high20>low20 else 0.5; pp60=(ct-low60)/(high60-low60) if high60>low60 else 0.5
    cu=cd=0
    for i in range(-1,-min(11,len(c)),-1):
        if abs(i)>=1:
            if c[i]>c[i-1]: cu+=1;cd=0
            else: cd+=1;cu=0
    ret5d=(ct-c[-6])/c[-6]*100 if len(c)>5 and c[-6]!=0 else 0
    ret1d=(ct-c[-2])/c[-2]*100 if len(c)>1 and c[-2]!=0 else 0
    rev=1 if ret5d<-10 and ret1d>0 else 0; gp=0
    if o is not None and len(o)>=2:
        pc=float(c[-2]); to=float(o[-1])
        gp=(to-pc)/pc*100 if pc>0 else 0
    m5p=float(np.mean(c[-10:-5])) if len(c)>=10 else ma5
    m20p=float(np.mean(c[-25:-20])) if len(c)>=25 else ma20
    vl5=0;vl20=0;vlr=1.0
    if v is not None and len(v)>=20:
        va=np.array(v[-20:],dtype=float); va=va[~np.isnan(va)]
        if len(va)>=10: vl5=float(np.mean(va[-5:])); vl20=float(np.mean(va)); vlr=vl5/vl20 if vl20>0 else 1.0
    
    f=[0.0]*50
    f[0]=ma5;f[1]=ct/ma5 if ma5>0 else 1;f[2]=ct/ma20 if ma20>0 else 1;f[3]=ct/ma60 if ma60>0 else 1
    f[4]=r5 if not np.isnan(r5) else 0;f[5]=r20 if not np.isnan(r20) else 0;f[6]=f[4]/f[5] if f[5]>0 else 1
    f[7]=ct;f[8]=ct;f[12]=rsi14;f[13]=k;f[14]=d;f[15]=j
    f[16]=bu;f[17]=bl;f[18]=bb_w;f[19]=pp20
    f[20]=vlr;f[21]=vlr;f[22]=25;f[23]=50;f[24]=50;f[25]=pp20;f[26]=pp60;f[27]=0;f[28]=18
    f[29]=float(np.log(ct)) if ct>0 else 0;f[30]=ct*f[4];f[31]=float(gp/10)
    f[32]=0;f[33]=0;f[34]=float(rsi14/100);f[35]=1 if ct<10 else 0
    f[36]=f[4]/ct if ct>0 else 0
    acc=((ma5-m20p)-(m5p-m20p))/ma20 if ma20>0 else 0
    f[37]=float(acc);f[38]=1 if (rsi14<30 and ct<bb_mid) else 0
    f[39]=(ma5-ma20)/ma20 if ma20>0 else 0;f[40]=f[4]/f[5] if f[5]>0 else 1;f[41]=pp60;f[42]=j
    bh=float(np.std(c[-60:]/np.mean(c[-60:]))) if len(c)>=60 and np.mean(c[-60:])>0 else 0.1
    f[43]=1 if bb_w<bh*0.8 else 0
    rs5=rsi(list(c[:-15]),14) if len(c)>15 else rsi14
    f[44]=rsi14-rs5
    cr=1 if (ma5>ma20 and m5p<=m20p) else (-1 if (ma5<ma20 and m5p>=m20p) else 0)
    f[45]=float(cr)
    vwap=float(np.mean(np.mean((h+l)/2))) if len(h)>=20 else ct
    f[46]=ct/vwap if vwap>0 else 1;f[47]=float(cu);f[48]=float(cd);f[49]=float(rev)
    for i,x in enumerate(f):
        if np.isnan(x) or np.isinf(x): f[i]=0.0
    return f, float(ct)

def main():
    t0 = time.time()
    today = time.strftime('%Y-%m-%d')
    print(f'绿箭 V9-Lottery — {today}')
    print('=' * 45)
    
    # Load model
    model = xgb.Booster()
    model.load_model(MODEL_FILE)
    model.set_param({'device':'cuda'})  # GPU预测
    print(f'模型: {MODEL_FILE} ({len(FEAT_V9)}特征)')
    
    # Load data
    data = json.load(open(DATA_FILE, 'r', encoding='utf-8'))
    
    # Find latest date's position for each stock
    # us_all_ohlcv.json data ends at ~2026-06-12
    from datetime import datetime
    data_end = datetime(2026, 6, 12)
    
    scored = []
    for code, sd in data.items():
        n = len(sd['c'])
        pos = n  # Use all data, predict next bar
        
        if pos < 100: continue
        
        ct = float(sd['c'][-1])  # Latest close
        if ct < 1.0 or ct > 10: continue  # $1-10 lottery pool
        c_full = sd['c']; h_full = sd['h']; l_full = sd['l']
        o_full = sd.get('o', [])
        v_full = sd.get('v', [])
        if len(o_full) < 100: o_full = None
        if len(v_full) < 100: v_full = None
        
        try:
            result = feats50(c_full, h_full, l_full, o_full, v_full, pos)
            if result is None: continue
            f, price = result
        except Exception as e:
            print(f'  SKIP {code}: {str(e)[:50]}')
            continue
        
        try:
            df = pd.DataFrame([f], columns=FEAT_V9)
            prob = float(model.predict(xgb.DMatrix(df))[0])
        except:
            continue
        
        # Estimate sector by price
        sector = 'Penny' if price<1 else 'Micro' if price<3 else 'Small' if price<5 else 'MidLo' if price<8 else 'MidHi'
        
        ratio_bp = round(float(prob) / float(price), 4) if price > 0 else 0
        
        scored.append({
            'sym': code,
            'prob': round(float(prob), 4),
            'price': round(float(price), 2),
            'ratio_bp': ratio_bp,
            'sector': sector,
        })
    
    if not scored:
        print('❌ 没有候选股')
        return
    
    # 方案F: prob/price top30 -> prob top10 (2026-06-15定版)
    # 先用prob/price筛出低价高概率池，再在池中按prob降序取top10
    scored.sort(key=lambda x: x['ratio_bp'], reverse=True)
    pool_30 = scored[:30]
    top10 = sorted(pool_30, key=lambda x: x['prob'], reverse=True)[:10]
    
    print(f'候选池: {len(scored)}只 ($1-10彩票池) | 排序: prob/price top30 -> prob top10')
    max_p = max(s['price'] for s in scored)
    min_p = min(s['price'] for s in scored)
    print(f'价格范围: ${min_p:.2f} - ${max_p:.2f}')
    print(f'≥0.90: {sum(1 for s in scored if s["prob"]>=0.90)}只')
    print(f'≥0.85: {sum(1 for s in scored if s["prob"]>=0.85)}只')
    print()
    print('Top 10 (prob/price top30池 -> prob降序):')
    print(f'  {"#":4s} {"代号":6s} {"prob":>6s} {"ratio":>7s} {"价格":>5s}  类别')
    for i, s in enumerate(top10):
        print(f'  {i+1:2d}. {s["sym"]:6s} {s["prob"]:.4f} {s["ratio_bp"]:.4f} ${s["price"]:.2f}  {s["sector"]}')
    
    # Save
    output = {
        'date': today,
        'model': 'V9-Lottery',
        'sort_method': 'prob/price_top30->prob_top10',
        'buy_signals': scored,
        'top10_picks': top10,
        'top_pick': top10[0] if top10 else None,
        'prob_thresholds': {
            '0.90+': sum(1 for s in scored if s['prob'] >= 0.90),
            '0.85+': sum(1 for s in scored if s['prob'] >= 0.85),
            '0.80+': sum(1 for s in scored if s['prob'] >= 0.80),
            '0.70+': sum(1 for s in scored if s['prob'] >= 0.70),
        }
    }
    
    out_path = f'{DATA_DIR}/scored_v9_lottery_{today}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f'\n保存: {out_path}')
    print(f'耗时: {time.time()-t0:.1f}s')

if __name__ == '__main__':
    main()
