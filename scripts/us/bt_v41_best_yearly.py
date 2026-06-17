#!/usr/bin/env python3
"""
最佳参数 · 逐年详细对比
"""
import json, os, warnings, time
import numpy as np

warnings.filterwarnings('ignore')

CACHE = "/home/admin/.openclaw/workspace/data/cache"
UNIVERSE = "/home/admin/.openclaw/workspace/data/sp500_universe.json"

pool_data = json.load(open(UNIVERSE))
tickers = pool_data['tickers']

# ── 加载数据 ──
def get_metric(raw):
    result = {}
    n = len(raw)
    for i in range(60, n):
        row = raw[i]; d = row['date']; pr = float(row['close'])
        hp52 = max(float(raw[j]['close']) for j in range(max(0,i-251), i+1))
        p52 = pr/hp52*100 if hp52>0 else 100
        m = {}
        for p in [15,20,25,30]:
            if i >= p: m[p] = (pr/float(raw[i-p]['close'])-1)*100
        result[d] = {'p':pr,'p52':p52,**m}
    return result

loaded = {}
for t in tickers:
    fpath = f"{CACHE}/{t}.json"
    if os.path.exists(fpath):
        try:
            data = json.load(open(fpath))['data']
            if len(data) > 200: loaded[t] = get_metric(data)
        except: pass

all_dates = sorted(set(d for td in loaded.values() for d in td.keys() if '2014-01-01'<=d<='2025-12-31'))

# ── 基准 ──
import yfinance as yf
import datetime
spy_raw = yf.download('SPY',start="2013-06-01",end="2026-06-01",progress=False)
spy_close = spy_raw['Close'].squeeze()
spy_dates = set(d.strftime('%Y-%m-%d') for d in spy_raw.index)
spy_map = {d.strftime('%Y-%m-%d'): float(spy_close.iloc[i]) for i, d in enumerate(spy_raw.index)}

qqq_raw = yf.download('QQQ',start="2013-06-01",end="2026-06-01",progress=False)
qqq_close = qqq_raw['Close'].squeeze()
qqq_map = {d.strftime('%Y-%m-%d'): float(qqq_close.iloc[i]) for i, d in enumerate(qqq_raw.index)}

def bench_ret(m, y):
    def get_b(dt_str):
        dt = datetime.date(int(dt_str[:4]), int(dt_str[5:7]), int(dt_str[8:10]))
        for off in range(-3, 4):
            d = (dt + datetime.timedelta(days=off)).strftime('%Y-%m-%d')
            if d in m: return m[d]
        return None
    s = get_b(f"{y}-01-02")
    e = get_b(f"{y}-12-31")
    return (e/s-1)*100 if s and e else 0

# ── 回测 ──
def run_strategy(ds, dc, md, tn, hd, years=range(2014,2026)):
    yearly = {}
    for y in years:
        sd, ed = f"{y}-01-02", f"{y}-12-31"
        yr_dates = [d for d in all_dates if sd<=d<=ed]
        if len(yr_dates) < 60: yearly[y]=0; continue
        rets = []
        for si in range(hd, len(yr_dates)-hd, hd):
            db = yr_dates[si]; dsell = yr_dates[min(si+hd,len(yr_dates)-1)]
            dm = yr_dates[max(0,si-md)]
            cand = []
            for t,td in loaded.items():
                vb=td.get(db); vp=td.get(dm)
                if not vb or not vp or vb['p']<1: continue
                mom = (vb['p']/vp['p']-1)*100 if md not in vb else vb[md]
                p52=vb['p52']
                deduction = max(0,(p52-ds)/(100-ds))*dc
                score = mom*(1-min(deduction,1))
                cand.append((score,t,vb['p']))
            if len(cand)<tn: continue
            cand.sort(key=lambda x:-x[0])
            pr=[]
            for _,t,bp in cand[:tn]:
                vs=loaded[t].get(dsell)
                if vs and bp>0: pr.append((vs['p']/bp-1)*100)
            if pr: rets.append(np.mean(pr))
        yearly[y] = sum(rets) if rets else 0
    return yearly

# ── 方案定义 ──
CONFIGS = [
    ("V4原始(ds50/0.7/20日/5只/20天)", dict(ds=50,dc=0.7,md=20,tn=5,hd=20)),
    ("V4.1(ds40/0.5/25日/5只/20天)", dict(ds=40,dc=0.5,md=25,tn=5,hd=20)),
    ("🏆 最佳累计tn=5(ds40/0.7/30日/5/20)", dict(ds=40,dc=0.7,md=30,tn=5,hd=20)),
    ("最佳平衡(ds60/0.7/15日/5/10)", dict(ds=60,dc=0.7,md=15,tn=5,hd=10)),
    ("稳健8只(ds60/0.5/15日/8/10)", dict(ds=60,dc=0.5,md=15,tn=8,hd=10)),
    ("V3纯动量(20日/5只/20天)", dict(ds=0,dc=0,md=20,tn=5,hd=20)),
]

YEARS = list(range(2014,2026))

# Run all
all_results = {}
for name, params in CONFIGS:
    ds = params['ds']; dc = params['dc']; md = params['md']; tn = params['tn']; hd = params['hd']
    
    if ds == 0:  # V3纯动量: use md=20 momentum, no deduction
        yearly = {}
        for y in YEARS:
            sd, ed = f"{y}-01-02", f"{y}-12-31"
            yr_dates = [d for d in all_dates if sd<=d<=ed]
            if len(yr_dates) < 60: yearly[y]=0; continue
            rets = []
            for si in range(hd, len(yr_dates)-hd, hd):
                db = yr_dates[si]; dsell = yr_dates[min(si+hd,len(yr_dates)-1)]
                cand = []
                for t,td in loaded.items():
                    vb=td.get(db)
                    if not vb or vb['p']<1: continue
                    mom = vb.get(f'm{md}',0)
                    cand.append((mom,t,vb['p']))
                if len(cand)<tn: continue
                cand.sort(key=lambda x:-x[0])
                pr=[]
                for _,t,bp in cand[:tn]:
                    vs=loaded[t].get(dsell)
                    if vs and bp>0: pr.append((vs['p']/bp-1)*100)
                if pr: rets.append(np.mean(pr))
            yearly[y] = sum(rets) if rets else 0
        all_results[name] = yearly
    else:
        all_results[name] = run_strategy(ds, dc, md, tn, hd)

# SPY/QQQ
spy_yr = {y: bench_ret(spy_map, y) for y in YEARS}
qqq_yr = {y: bench_ret(qqq_map, y) for y in YEARS}

# ── 打印表格 ──
print("=" * 130)
print("最佳方案逐年对比 ·  140只质量池 · 2014-2025")
print("=" * 130)

headers = ["年份"] + [n.split('(')[0] for n,_ in CONFIGS] + ["SPY", "QQQ"]
header_fmt = f"{'年份':>6s}"
for n,_ in CONFIGS:
    short = n.split('(')[0]
    header_fmt += f" {short:>28s}"
header_fmt += f" {'SPY':>8s} {'QQQ':>8s}"
print(header_fmt)
print("-" * 130)

# For cumulative
cum_data = {n: 0 for n,_ in CONFIGS}
cum_spy = 0; cum_qqq = 0

for y in YEARS:
    line = f"{y:>6d}"
    for name,_ in CONFIGS:
        r = all_results[name].get(y, 0)
        line += f" {r:>+28.1f}%"
    line += f" {spy_yr[y]:>+8.1f}% {qqq_yr[y]:>+8.1f}%"
    print(line)

print("-" * 130)

# Cumulative & stats
print()
print("=" * 130)
print("汇总统计")
print("=" * 130)

# Header
stat_header = f"{'指标':>25s}"
for n,_ in CONFIGS:
    short = n.split('(')[0]
    stat_header += f" {short:>28s}"
stat_header += f" {'SPY':>8s} {'QQQ':>8s}"
print(stat_header)
print("-" * 130)

# 累计
line = f"{'12年累计':>25s}"
for name,_ in CONFIGS:
    t = sum(all_results[name].values())
    line += f" {t:>+28.1f}%"
line += f" {sum(spy_yr.values()):>+8.1f}% {sum(qqq_yr.values()):>+8.1f}%"
print(line)

# 年化
line = f"{'年化':>25s}"
for name,_ in CONFIGS:
    vals = [v for v in all_results[name].values() if v != 0]
    t = sum(vals)
    n_y = len(vals)
    ann = ((1+t/100)**(1/n_y)-1)*100 if t>-100 else 0
    line += f" {ann:>+27.1f}%"
print(line)

# 夏普
line = f"{'夏普(年)':>25s}"
for name,_ in CONFIGS:
    vals = [v for v in all_results[name].values() if v != 0]
    if len(vals) > 2 and np.std(vals) > 0:
        sp = np.mean(vals)/np.std(vals)*(12**0.5)
    else: sp = 0
    line += f" {sp:>28.2f}"
print(line)

# 回撤
line = f"{'最大回撤':>25s}"
for name,_ in CONFIGS:
    vals = [v for v in all_results[name].values() if v != 0]
    cv = 100; pk = 100; mdd = 0
    for r in vals:
        cv *= (1+r/100)
        if cv > pk: pk = cv
        dd = (pk-cv)/pk*100
        if dd > mdd: mdd = dd
    line += f" {mdd:>27.1f}%"
print(line)

# 胜率
line = f"{'胜率(年)':>25s}"
for name,_ in CONFIGS:
    vals = [v for v in all_results[name].values() if v != 0]
    wr = sum(1 for r in vals if r > 0)/len(vals)*100
    line += f" {wr:>27.1f}%"
print(line)

# 跑赢QQQ
line = f"{'跑赢QQQ年数':>25s}"
for name,_ in CONFIGS:
    beat = sum(1 for y in YEARS if all_results[name].get(y, 0) > qqq_yr[y])
    line += f" {beat:>27d}/12"
print(line)

# ── 累积曲线 ──
print()
print("=" * 130)
print("累积收益曲线")
print("=" * 130)
header2 = f"{'年份':>6s}"
for n,_ in CONFIGS:
    short = n.split('(')[0]
    header2 += f" {short:>28s}"
header2 += f" {'SPY':>8s} {'QQQ':>8s}"
print(header2)
print("-" * 130)

cum_data = {n: 0 for n,_ in CONFIGS}
cum_spy = 0; cum_qqq = 0
for y in YEARS:
    line = f"{y:>6d}"
    for name,_ in CONFIGS:
        cum_data[name] += all_results[name].get(y, 0)
        line += f" {cum_data[name]:>+28.1f}%"
    cum_spy += spy_yr[y]
    cum_qqq += qqq_yr[y]
    line += f" {cum_spy:>+8.1f}% {cum_qqq:>+8.1f}%"
    print(line)

# ── 最佳方案逐年明细 ──
print()
print("=" * 130)
print("🏆  最佳方案逐年明细 (ds=40 dc=0.7 md=30 tn=5 hd=20)")
print("=" * 130)
best_name = "🏆 最佳累计tn=5(ds40/0.7/30日/5/20)"
stable_name = "最佳平衡(ds60/0.7/15日/5/10)"

print(f"\n{'年份':>6s}  {'最佳累计tn=5':>13s}  {'最佳平衡':>13s}  {'V3纯动量':>13s}  {'SPY':>10s}  {'QQQ':>10s}")
print("-" * 70)
for y in YEARS:
    b = all_results[best_name].get(y, 0)
    s = all_results[stable_name].get(y, 0)
    v3 = all_results["V3纯动量(20日/5只/20天)"].get(y, 0)
    sp = spy_yr[y]; qq = qqq_yr[y]
    e1 = "🏆" if b > sp and b > qq else "✅" if b > sp else "🟡" if b > qq else ""
    e2 = "🏆" if s > sp and s > qq else "✅" if s > sp else ""
    print(f"{y:>6d}  {b:>+13.1f}%  {s:>+13.1f}%  {v3:>+13.1f}%  {sp:>+10.1f}%  {qq:>+10.1f}%  {e1} {e2}")

# 对比汇总
b_total = sum(all_results[best_name].values())
s_total = sum(all_results[stable_name].values())
v3_total = sum(all_results["V3纯动量(20日/5只/20天)"].values())
sp_total = sum(spy_yr.values())
qq_total = sum(qqq_yr.values())

print("-" * 70)
print(f"{'12年累计':>6s}  {b_total:>+13.1f}%  {s_total:>+13.1f}%  {v3_total:>+13.1f}%  {sp_total:>+10.1f}%  {qq_total:>+10.1f}%")

# 指标对比
def calc_stats(vals_dict):
    vals = [v for v in vals_dict.values() if v != 0]
    n = len(vals)
    t = sum(vals)
    ann = ((1+t/100)**(1/n)-1)*100 if t>-100 else 0
    sp = np.mean(vals)/np.std(vals)*(12**0.5) if len(vals)>2 and np.std(vals)>0 else 0
    cv=100;pk=100;mdd=0
    for r in vals:
        cv*=(1+r/100)
        if cv>pk:pk=cv
        dd=(pk-cv)/pk*100
        if dd>mdd:mdd=dd
    wr=sum(1 for r in vals if r>0)/n*100
    return ann, sp, mdd, wr

print(f"\n{'指标':>20s}  {'最佳累计tn=5':>15s}  {'最佳平衡':>15s}  {'V3纯动量':>15s}")
print("-" * 65)
for name in [best_name, stable_name, "V3纯动量(20日/5只/20天)"]:
    ann, sp, mdd, wr = calc_stats(all_results[name])
    if name == best_name:
        print(f"{'年化':>20s}  {ann:>+14.1f}%")
        print(f"{'夏普':>20s}  {sp:>14.2f}")
        print(f"{'最大回撤':>20s}  {mdd:>13.1f}%")
        print(f"{'胜率':>20s}  {wr:>13.1f}%")
        print()
