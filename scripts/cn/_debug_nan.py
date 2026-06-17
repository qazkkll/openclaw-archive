#!/usr/bin/env python3
"""快速debug nan原因 — flush版"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ws = "C:\\Users\\admin\\.openclaw\\workspace"
sys.path.insert(0, os.path.join(ws, "scripts"))
from score_engine import v5s_calc, v5s_score

print("加载数据...", flush=True)
d = json.load(open(f"{ws}/data/us_hist_clean.parquet","r"))
syms = list(d.keys())
print(f"总池: {len(syms)}只", flush=True)

cache={}
for i,sy in enumerate(syms):
    if i%500==0: print(f"  加载 {i}/{len(syms)}", flush=True)
    c=d[sy].get("c",[]); h=d[sy].get("h",[]); l=d[sy].get("l",[])
    if len(c)>=520:
        ind=v5s_calc(c,h,l)
        if ind: cache[sy]=(ind,c)

print(f"有效池: {len(cache)}只", flush=True)
max_n=max(len(c) for _,c in cache.values())
all_t=[]

for day in range(252, min(max_n-20, 1550), 5):
    if day%200==0: print(f"  day {day}, trades={len(all_t)}", flush=True)
    cand=[]
    for sy,(ind,c) in cache.items():
        if day>=len(c): continue
        sc=v5s_score(ind,day)
        if sc>0: cand.append((sc,sy,c[day]))
    cand.sort(key=lambda x:-x[0])
    for sc,sy,bp in cand[:5]:
        ic=cache[sy]; cc=ic[1]
        sd=min(day+20,len(cc)-1)
        for d2 in range(day+1,sd+1):
            if (cc[d2]-bp)/bp<=-0.15: sd=d2; break
        sp=cc[sd]
        ret=(sp-bp)/bp
        all_t.append(ret)

nt=len(all_t)
print(f"\nTotal trades: {nt}", flush=True)
print(f"Has nan: {sum(1 for r in all_t if r!=r)}", flush=True)
print(f"Has inf: {sum(1 for r in all_t if r==float('inf') or r==float('-inf'))}", flush=True)
print(f"Range: min={min(all_t):.4f} max={max(all_t):.4f}", flush=True)
print(f"Sum: {sum(all_t)}", flush=True)
print(f"Avg: {sum(all_t)/nt}", flush=True)
