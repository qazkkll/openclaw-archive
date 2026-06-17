"""
预计算资金流因子分 - mmap版本
用内存映射 + 正则提取每只股票数据，一次处理一只。
"""
import json, os, sys, time, re, mmap, gc
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

WORKSPACE = "D:\\openclaw-workspace"
MF_PATH = os.path.join(WORKSPACE, 'data', 'moneyflow_data.parquet')
OUT_PATH = os.path.join(WORKSPACE, 'data', 'precomputed_scores.json')

def ensure_float(v):
    if v is None: return 0.0
    try: return float(v)
    except: return 0.0

def _total_amt(r):
    """资金流没有直接amount字段，从细项加总"""
    return (ensure_float(r.get('buy_sm_amount', 0)) +
            ensure_float(r.get('buy_md_amount', 0)) +
            ensure_float(r.get('buy_lg_amount', 0)) +
            ensure_float(r.get('buy_elg_amount', 0)))

def calc_mf_score(records):
    recent = records[-20:]
    if len(recent) < 5: return None
    nets_5 = []
    for r in recent[-5:]:
        amt = _total_amt(r)
        if amt == 0: continue
        buy = ensure_float(r.get('buy_lg_amount', 0)) + ensure_float(r.get('buy_elg_amount', 0))
        sell = ensure_float(r.get('sell_lg_amount', 0)) + ensure_float(r.get('sell_elg_amount', 0))
        nets_5.append((buy - sell) / amt * 100)
    if not nets_5: return None
    avg_net = sum(nets_5) / len(nets_5)
    
    elg_nets = []
    for r in recent[-5:]:
        amt = _total_amt(r)
        if amt == 0: continue
        be = ensure_float(r.get('buy_elg_amount', 0))
        se = ensure_float(r.get('sell_elg_amount', 0))
        elg_nets.append((be - se) / amt * 100)
    avg_elg = sum(elg_nets) / len(elg_nets) if elg_nets else 0
    
    trend = 0
    if len(recent) >= 20:
        prev = []
        for r in recent[:15]:
            amt = _total_amt(r)
            if amt == 0: continue
            buy = ensure_float(r.get('buy_lg_amount', 0)) + ensure_float(r.get('buy_elg_amount', 0))
            sell = ensure_float(r.get('sell_lg_amount', 0)) + ensure_float(r.get('sell_elg_amount', 0))
            prev.append((buy - sell) / amt * 100)
        if prev: trend = avg_net - (sum(prev) / len(prev))
    sc = 50 + avg_net * 3 + avg_elg * 2 + trend * 2
    return {'mf_score': round(max(10, min(100, sc)), 1),
            'net_pct': round(avg_net, 2), 'elg_net_pct': round(avg_elg, 2),
            'trend': round(trend, 2), 'n_records': len(records)}


def main():
    t0 = time.time()
    size_gb = os.path.getsize(MF_PATH) / 1e9
    print(f"Money flow file: {size_gb:.1f}GB")
    print("Memory-mapping...", flush=True)
    
    with open(MF_PATH, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            print(f"Mapped {len(mm)/1e9:.1f}GB", flush=True)
            
            # 在二进制中找股票代码 pattern
            # Pattern: b'"XXXXXX.SH":['
            pat = re.compile(rb'"(\d{6}\.(?:SH|SZ))"\s*:\s*\[')
            
            results = {}
            count = 0
            scan_pos = 0
            arr_limit = 3 * 1024 * 1024  # 3MB max per stock
            
            while True:
                m = pat.search(mm, scan_pos)
                if m is None:
                    break
                
                code = m.group(1).decode('ascii')
                arr_start = m.end() - 1  # position of [
                
                # 找匹配的 ]
                depth = 1  # already counted the opening [
                arr_end = -1
                
                # Search up to 3MB ahead
                search_end = min(arr_start + arr_limit, len(mm))
                for j in range(arr_start + 1, search_end):
                    ch = mm[j]
                    if ch == 91:  # '['
                        depth += 1
                    elif ch == 93:  # ']'
                        depth -= 1
                        if depth == 0:
                            arr_end = j + 1
                            break
                
                if arr_end < 0:
                    scan_pos = arr_start + 1
                    continue
                
                # Parse the array
                try:
                    arr_bytes = mm[arr_start:arr_end]
                    records = json.loads(arr_bytes)
                    if isinstance(records, list) and len(records) >= 5:
                        score = calc_mf_score(records)
                        if score:
                            results[code] = score
                except:
                    pass
                
                count += 1
                scan_pos = arr_end
                
                if count % 500 == 0:
                    pct = (scan_pos / len(mm)) * 100
                    print(f"  {count} stocks ({pct:.0f}%)", end='\r', flush=True)
            
            print(f"\n  Total found: {count} stocks, {len(results)} scored")
    
    # Save
    print(f"\nSaving {len(results)} scores...", flush=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)
    
    out_size = os.path.getsize(OUT_PATH) / 1e6
    print(f"  Output: {out_size:.1f}MB")
    
    # Stats
    all_sc = [s['mf_score'] for s in results.values() if s]
    if all_sc:
        print(f"\nScore distribution ({len(all_sc)} stocks):")
        for thr in [80, 70, 60, 50, 40, 30]:
            cnt = sum(1 for s in all_sc if s >= thr)
            print(f"  >= {thr}: {cnt} ({cnt/len(all_sc)*100:.0f}%)")
        print(f"  Mean: {sum(all_sc)/len(all_sc):.1f}")
        all_sc.sort()
        print(f"  Median: {all_sc[len(all_sc)//2]:.1f}")
    
    print(f"\nTime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
