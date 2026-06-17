#!/usr/bin/env python3
"""
🍤 V1+ETF双模 v3 — 全量1177只预计算，准确定位熊市切换点

切换阈值: V1≥62的票数不足 → ETF模式
"""
import sys, os, json, time, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_engine import compute_indicators, v1_score
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'backtest_hist_yahoo.json')
RB_DAYS = 7
BUY = 62
POS = 8
ETF_W = {'511010.SS': 0.6, '518880.SS': 0.4}

def load_etf():
    print("加载ETF...", flush=True)
    r = {}
    for c in ['511010.SS','518880.SS']:
        d = yf.download(c, period='10y', progress=False)
        r[c] = {'c': list(d['Close'].values), 'd': [str(x)[:10] for x in d.index]}
    return r

def precompute_all(all_s, codes):
    print(f"预计算{len(codes)}只V1...", flush=True)
    t0 = time.time()
    cache = {}
    for i, code in enumerate(codes):
        d = all_s[code]
        if len(d['close']) < 60:
            continue
        ind = compute_indicators(d['close'], d['high'], d['low'])
        if ind is None:
            continue
        scores = [0.0]*60
        for di in range(60, len(d['close'])):
            s = v1_score(ind, di)
            scores.append(round(s,1) if s else 0.0)
        cache[code] = scores
        if (i+1)%400 == 0:
            print(f"  {i+1}/{len(codes)} {time.time()-t0:.0f}s", flush=True)
    print(f"完成: {len(cache)}只 {time.time()-t0:.0f}s", flush=True)
    return cache

###### 参数扫描：ETF入场阈值 + ETF配置 #######
def run_sweep():
    t0 = time.time()
    
    all_s = json.load(open(DATA_PATH))
    codes = list(all_s.keys())
    n = len(all_s[codes[0]]['close'])
    dates = all_s[codes[0]]['dates']
    
    etf = load_etf()
    etf_start = etf['511010.SS']['d'][0]
    
    # 回测起点
    si = 250
    while si < n and dates[si] < etf_start:
        si += 1
    print(f"回测: {dates[si]}~{dates[-1]}\n", flush=True)
    
    # 预计算全量1177
    cache = precompute_all(all_s, codes)
    valid = [c for c in codes if c in cache]
    
    # ETF日收益
    etf_r = {}
    for c in etf:
        cls = etf[c]['c']
        ds = etf[c]['d']
        rets = {}
        for i in range(1, len(cls)):
            rets[ds[i]] = cls[i]/cls[i-1]-1
        etf_r[c] = rets
    
    # 要扫描的参数
    # min_stocks: 选票少于N只时切ETF（全市场1177只）
    # etf_mix: ETF组合
    min_stock_opts = [5, 10, 20, 30, 50]  # 多少只达标以下切ETF
    etf_opts = [
        {'511010.SS': 1.0, '518880.SS': 0.0},           # 全国债
        {'511010.SS': 0.0, '518880.SS': 1.0},           # 全黄金
        {'511010.SS': 0.6, '518880.SS': 0.4},           # 国债60黄金40
        {'511010.SS': 0.5, '518880.SS': 0.5},           # 各半
    ]
    
    results = []
    
    for ms in min_stock_opts:
        for em in etf_opts:
            etf_label = f"国债{int(em['511010.SS']*100)}%+黄金{int(em['518880.SS']*100)}%"
            
            holdings = {}
            etf_holdings = {}
            last_rb = -RB_DAYS
            dv = {}
            md = {'stock':0,'etf':0}
            switches = []
            
            for di in range(si, n):
                today = dates[di]
                
                if di - last_rb >= RB_DAYS:
                    good = sum(1 for c in valid if di < len(cache[c]) and cache[c][di] >= BUY)
                    
                    if good >= ms:
                        scored = [(cache[c][di], c) for c in valid if di < len(cache[c]) and cache[c][di] >= BUY]
                        scored.sort(reverse=True)
                        top = scored[:POS]
                        holdings = {c: 1.0/max(1,len(top)) for _,c in top}
                        etf_holdings = {}
                        if md['etf'] > 0 and md['stock'] == 0:
                            switches.append(f"{today} →股票(V1≥62:{good}只)")
                    else:
                        holdings = {}
                        etf_holdings = dict(em)
                        if md['stock'] > 0 and md['etf'] == 0:
                            switches.append(f"{today} →ETF(V1≥62仅{good}只)")
                    
                    last_rb = di
                
                # 净值
                if holdings:
                    v = 0
                    for c, w in holdings.items():
                        arr = all_s[c]['close']
                        if last_rb < len(arr) and di < len(arr):
                            v += w * arr[di]/arr[last_rb]
                        else:
                            v += w
                    dv[today] = v
                    md['stock'] += 1
                elif etf_holdings:
                    v = 0
                    for c, w in etf_holdings.items():
                        ret = etf_r[c].get(today, 0)
                        v += w * (1 + ret)
                    dv[today] = v
                    md['etf'] += 1
                else:
                    dv[today] = 1.0
            
            if not dv:
                continue
            
            ds = sorted(dv.keys())
            cum = (dv[ds[-1]]-1)*100
            
            # 逐年
            years = {}
            for yr in range(2017, 2027):
                y = str(yr)
                yd = {d:v for d,v in dv.items() if d.startswith(y)}
                if len(yd) < 20: continue
                ds2 = sorted(yd.keys())
                years[y] = float(round((yd[ds2[-1]]/yd[ds2[0]]-1)*100, 1))
            
            bear = float(years.get('2018',0)+years.get('2022',0))
            results.append({'ms':ms,'etf':etf_label,'cum':round(cum,1),'bear_sum':bear,
                           'years':years,'switches':len(switches)})
            
            yr3 = ' '.join([f'{y}{years.get(y,"x"):+.1f}' for y in ['2018','2020','2022','2024']])
            print(f"ms={ms:2d} {etf_label:15s} | 累计{cum:>+6.1f}% 熊{bear:>+5.1f}% | {yr3}", flush=True)
    
    print(f"\n扫描: {len(results)}组合 {time.time()-t0:.0f}s\n", flush=True)
    
    # 排序输出
    print("="*70)
    print("🏆 熊市表现最佳(2018+2022)")
    print("="*70)
    results.sort(key=lambda r: (-r['bear_sum'], -r['cum']))
    for i, r in enumerate(results[:5]):
        yr = ' '.join([f'{y}{r["years"].get(y,"?"):+5.1f}' for y in ['2018','2020','2022','2024']])
        print(f"#{i+1} ms={r['ms']:2d} {r['etf']:15s} | 累计{r['cum']:+6.1f}% 熊{r['bear_sum']:+5.1f}% | {yr}")
    
    print(f"\n{'='*70}")
    print("🏆 累计收益最佳")
    print("="*70)
    results.sort(key=lambda r: -r['cum'])
    for i, r in enumerate(results[:5]):
        yr = ' '.join([f'{y}{r["years"].get(y,"?"):+5.1f}' for y in ['2018','2020','2022','2024']])
        print(f"#{i+1} ms={r['ms']:2d} {r['etf']:15s} | 累计{r['cum']:+6.1f}% 熊{r['bear_sum']:+5.1f}% | {yr}")
    
    # 保存
    results.sort(key=lambda r: (-r['bear_sum'], -r['cum']))
    json.dump({'time':time.strftime('%Y-%m-%d %H:%M'),'total':len(results),'top':results[:20]},
              open(os.path.join(ROOT,'data','bt_v1_etf_sweep.json'),'w'),indent=2)
    print(f"\n→ data/bt_v1_etf_sweep.json")

if __name__ == '__main__':
    run_sweep()
