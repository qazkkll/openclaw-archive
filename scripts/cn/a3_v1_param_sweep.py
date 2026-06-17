#!/usr/bin/env python3
"""A3_v1 参数扫描 + 市场状态过滤 — 全量4536只
基于已有特征计算，测试多组参数 + 大盘过滤
"""
import sys, io, json, time, os, gc, warnings
import numpy as np
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
print = lambda *a,**kw: (__import__('builtins').print(*a, flush=True, **kw))
import xgboost as xgb

BASE = r'/home/hermes/.hermes/openclaw-archive/data'
MODEL_PATH = os.path.join(BASE, 'a1_models', 'a3_v1.json')
HIST_PATH = os.path.join(BASE, 'a_hist_10y.parquet')
FEAT_CACHE = os.path.join(BASE, 'a1_models', 'a3_v1_feats_cache.npz')

FEATURES = [
    'pct_ma5','pct_ma10','pct_ma20','pct_ma60','pct_ma120',
    'ma20_slope','ma60_slope','ma_align',
    'vol_10d','vol_60d','vol_ratio','atr20_pct',
    'ret_1d','ret_5d','ret_10d','ret_20d','ret_60d','rsi14',
    'vol_ratio_5_20','kdj_k','kdj_d','kdj_j',
    'macd_dif','macd_dea','macd_bar','bb_width','bb_position',
    'obv_ratio_5_20','ret5_max','ret3_vs_ema12','accel_5_10',
    'ma5_ma10_cross','vol_breakout',
]

def _ema_np(arr, period):
    alpha = 2.0 / (period + 1)
    result = np.empty_like(arr, dtype=np.float64)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = arr[i] * alpha + result[i-1] * (1 - alpha)
    return result

def compute_features_vectorized(c, h, l, o, v, dates, start_idx=120):
    n = len(c)
    if n < start_idx + 20: return None
    c = c.astype(np.float64); h = h.astype(np.float64)
    l = l.astype(np.float64); v = v.astype(np.float64)
    
    def ma(arr, w):
        cs = np.cumsum(arr); cs[w:] = cs[w:] - cs[:-w]
        r = np.full(n, np.nan); r[w-1:] = cs[w-1:] / w; return r
    
    ma5=ma(c,5); ma10=ma(c,10); ma20=ma(c,20); ma60=ma(c,60); ma120=ma(c,120)
    ma120 = np.where(np.isnan(ma120), ma60, ma120)
    ma20_prev = np.full(n, np.nan); ma60_prev = np.full(n, np.nan)
    for i in range(start_idx, n):
        if i>=24: ma20_prev[i]=np.mean(c[i-24:i-4])
        if i>=64: ma60_prev[i]=np.mean(c[i-64:i-4])
    tr=np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    tr_full=np.zeros(n); tr_full[1:]=tr
    diff=np.diff(c)
    ema12=_ema_np(c,12); ema26=_ema_np(c,26)
    dif_arr=ema12-ema26; dea_arr=_ema_np(dif_arr,9)
    obv_full=np.zeros(n)
    for j in range(1,n):
        if c[j]>c[j-1]: obv_full[j]=v[j]
        elif c[j]<c[j-1]: obv_full[j]=-v[j]
    obv_cum=np.cumsum(obv_full)
    
    valid_range = range(start_idx, n-10)
    n_rows = len(valid_range)
    if n_rows < 10: return None
    feat_matrix = np.zeros((n_rows, 33))
    date_list = []
    prev_k=50.0; prev_d=50.0
    
    for idx, i in enumerate(valid_range):
        price=c[i]
        feat_matrix[idx,0]=(price/ma5[i]-1)*100 if ma5[i]>0 else 0
        feat_matrix[idx,1]=(price/ma10[i]-1)*100 if ma10[i]>0 else 0
        feat_matrix[idx,2]=(price/ma20[i]-1)*100 if ma20[i]>0 else 0
        feat_matrix[idx,3]=(price/ma60[i]-1)*100 if ma60[i]>0 else 0
        feat_matrix[idx,4]=(price/ma120[i]-1)*100 if ma120[i]>0 else 0
        feat_matrix[idx,5]=(ma20[i]/ma20_prev[i]-1)*100 if not np.isnan(ma20_prev[i]) and ma20_prev[i]>0 else 0
        feat_matrix[idx,6]=(ma60[i]/ma60_prev[i]-1)*100 if not np.isnan(ma60_prev[i]) and ma60_prev[i]>0 else 0
        feat_matrix[idx,7]=(ma5[i]/ma60[i]-1)*100 if ma60[i]>0 else 0
        vol10=np.mean(v[i-9:i+1]); vol60=np.mean(v[i-59:i+1])
        feat_matrix[idx,8]=vol10; feat_matrix[idx,9]=vol60
        feat_matrix[idx,10]=vol10/vol60 if vol60>0 else 1.0
        feat_matrix[idx,11]=np.mean(tr_full[i-19:i+1])/price*100 if price>0 else 0
        feat_matrix[idx,12]=(c[i]/c[i-1]-1)*100 if i>=1 else 0
        feat_matrix[idx,13]=(c[i]/c[i-4]-1)*100 if i>=4 else 0
        feat_matrix[idx,14]=(c[i]/c[i-9]-1)*100 if i>=9 else 0
        feat_matrix[idx,15]=(c[i]/c[i-19]-1)*100 if i>=19 else 0
        feat_matrix[idx,16]=(c[i]/c[i-59]-1)*100 if i>=59 else 0
        g=diff[i-13:i+1]; ag=np.mean(np.maximum(g,0)); al=np.mean(np.maximum(-g,0))
        feat_matrix[idx,17]=100-100/(1+ag/al) if al>0 else 100
        vol5=np.mean(v[i-4:i+1]); vol20=np.mean(v[i-19:i+1])
        feat_matrix[idx,18]=vol5/vol20 if vol20>0 else 1.0
        low9=np.min(l[i-8:i+1]); high9=np.max(h[i-8:i+1])
        rsv=(c[i]-low9)/(high9-low9)*100 if high9>low9 else 50
        k_val=2/3*prev_k+1/3*rsv; d_val=2/3*prev_d+1/3*k_val
        prev_k=k_val; prev_d=d_val
        feat_matrix[idx,19]=round(k_val,2); feat_matrix[idx,20]=round(d_val,2)
        feat_matrix[idx,21]=round(3*k_val-2*d_val,2)
        feat_matrix[idx,22]=round(dif_arr[i],4); feat_matrix[idx,23]=round(dea_arr[i],4)
        feat_matrix[idx,24]=round((dif_arr[i]-dea_arr[i])*2,4)
        std20=np.std(c[i-19:i+1])
        feat_matrix[idx,25]=std20/ma20[i]*100 if ma20[i]>0 else 0
        feat_matrix[idx,26]=(price-(ma20[i]-2*std20))/(4*std20)*100 if std20>0 else 50
        obv5=np.mean(obv_cum[i-4:i+1]); obv20=np.mean(obv_cum[i-19:i+1])
        feat_matrix[idx,27]=obv5/obv20 if abs(obv20)>0 else 1.0
        feat_matrix[idx,28]=(np.max(h[i-4:i+1])/price-1)*100
        feat_matrix[idx,29]=(c[i]/ema12[i]-1)*100 if ema12[i]>0 else 0
        feat_matrix[idx,30]=feat_matrix[idx,13]-feat_matrix[idx,14]
        feat_matrix[idx,31]=ma5[i]/ma10[i]-1 if ma10[i]>0 else 0
        vol40=np.mean(v[i-39:i+1]) if i>=39 else np.mean(v[:i+1])
        feat_matrix[idx,32]=v[i]/vol40 if vol40>0 else 1.0
        date_list.append(dates[i])
    
    return {'dates': date_list, 'features': feat_matrix}

def process_stock(code, data):
    try:
        c=np.array(data['c']); h=np.array(data['h'])
        l=np.array(data['l']); o=np.array(data['o']); v=np.array(data['v'])
        if len(data['dates'])<250: return code, None
        return code, compute_features_vectorized(c,h,l,o,v,data['dates'])
    except: return code, None

def run_wf(stock_feats, hist, n_splits, hold_days, top_k, sl_pct, market_filter=False):
    """Walk-Forward with configurable params + optional market filter"""
    all_dates = set()
    for data in stock_feats.values():
        all_dates.update(data['dates'])
    dates_sorted = sorted(all_dates)
    
    # 计算大盘MA20（用000001或所有股票均值）
    # 简化：用全市场当日平均收益判断牛熊
    if market_filter:
        # 用000300(沪深300)或000001(上证)的MA20判断
        ref = hist.get('000001', hist.get('399300'))
        if ref:
            ref_c = np.array(ref['c'], dtype=np.float64)
            ref_dates = ref['dates']
            ref_ma20 = np.convolve(ref_c, np.ones(20)/20, mode='valid')
            bull_dates = set()
            for i in range(len(ref_ma20)):
                d_idx = i + 19
                if d_idx < len(ref_dates) and ref_c[d_idx] > ref_ma20[i]:
                    bull_dates.add(ref_dates[d_idx])
        else:
            bull_dates = None
    else:
        bull_dates = None
    
    split_size = len(dates_sorted) // (n_splits + 1)
    results = []
    
    for fold in range(n_splits):
        train_end = dates_sorted[split_size*(fold+1)-1]
        test_start_idx = split_size*(fold+1)
        test_end_idx = min(test_start_idx + hold_days + 5, len(dates_sorted)-1)
        test_dates = dates_sorted[test_start_idx:test_end_idx+1]
        if len(test_dates) < hold_days: continue
        
        scores = []
        for code, data in stock_feats.items():
            feat_dates = data['dates']
            feat_matrix = data['features']
            
            insert_pos = 0
            for j, dt in enumerate(feat_dates):
                if dt <= train_end: insert_pos = j + 1
                else: break
            if insert_pos == 0: continue
            
            # 市场过滤：只在大盘>MA20时买入
            if bull_dates is not None:
                entry_date = feat_dates[min(insert_pos, len(feat_dates)-1)]
                if entry_date not in bull_dates:
                    continue
            
            X = feat_matrix[insert_pos-1:insert_pos]
            dmat = xgb.DMatrix(X, feature_names=FEATURES)
            pred = model.predict(dmat)[0]
            
            entry_date_idx = insert_pos
            if entry_date_idx >= len(feat_dates): continue
            
            scores.append({
                'code': code,
                'score': float(pred),
                'entry_date': feat_dates[entry_date_idx],
            })
        
        if not scores: continue
        scores.sort(key=lambda x: -x['score'])
        top = scores[:top_k]
        
        rets = []
        for s in top:
            code = s['code']
            d = hist.get(code)
            if d is None: continue
            dates_arr = d['dates']; c_arr = d['c']
            entry_date = s['entry_date']
            entry_idx = None
            for j, dt in enumerate(dates_arr):
                if dt == entry_date: entry_idx = j; break
            if entry_idx is None or entry_idx + hold_days >= len(dates_arr): continue
            entry_p = c_arr[entry_idx]
            if entry_p <= 0: continue
            exit_p = c_arr[entry_idx + hold_days]
            ret = (exit_p/entry_p - 1) * 100
            min_p = min(c_arr[entry_idx:entry_idx+hold_days+1])
            max_loss = (min_p/entry_p - 1) * 100
            if max_loss <= sl_pct: ret = sl_pct
            rets.append(ret)
        
        if not rets: continue
        avg_ret = np.mean(rets)
        results.append({
            'fold': fold+1, 'train_end': train_end,
            'avg_ret': round(avg_ret, 2),
            'wins': sum(1 for r in rets if r > 0),
            'total': len(rets),
        })
    
    if not results: return None
    avg = np.mean([r['avg_ret'] for r in results])
    total_w = sum(r['wins'] for r in results)
    total_n = sum(r['total'] for r in results)
    # 去掉fold5（异常牛市）
    non_boom = [r['avg_ret'] for r in results[:-1]] if len(results) > 1 else [r['avg_ret'] for r in results]
    avg_no_boom = np.mean(non_boom) if non_boom else 0
    
    return {
        'avg_ret': round(avg, 2),
        'avg_no_boom': round(avg_no_boom, 2),
        'win_rate': round(total_w/total_n*100, 1) if total_n > 0 else 0,
        'n_folds': len(results),
        'folds': results,
    }

if __name__ == '__main__':
    t0 = time.time()
    print("=" * 60)
    print("  A3_v1 参数扫描 + 市场状态过滤")
    print("=" * 60)
    
    # [1] 加载
    print("\n[1] 加载模型+数据...")
    model = xgb.Booster(); model.load_model(MODEL_PATH)
    with open(HIST_PATH, 'r', encoding='utf-8') as f:
        hist = json.load(f)
    valid = {k:v for k,v in hist.items() if len(v.get('dates',[]))>=250}
    print(f"  有效: {len(valid)} 只")
    
    # [2] 特征计算
    print(f"[2] 特征计算（{len(valid)}只，{min(cpu_count(),8)}进程）...")
    tasks = [(c,d) for c,d in valid.items()]
    with Pool(min(cpu_count(),8)) as pool:
        results = pool.starmap(process_stock, tasks, chunksize=50)
    
    stock_feats = {c:r for c,r in results if r is not None}
    del results; gc.collect()
    print(f"  完成: {len(stock_feats)} 只")
    
    # [3] 参数扫描
    print("\n[3] 参数扫描...")
    param_grid = [
        # (hold_days, top_k, sl_pct, market_filter, label)
        (5, 5, -10, False, "5d/K5/SL10"),
        (5, 5, -15, False, "5d/K5/SL15"),
        (5, 10, -10, False, "5d/K10/SL10"),
        (5, 10, -15, False, "5d/K10/SL15"),
        (5, 10, -20, False, "5d/K10/SL20"),
        (5, 20, -15, False, "5d/K20/SL15"),
        (3, 10, -10, False, "3d/K10/SL10"),
        (3, 10, -15, False, "3d/K10/SL15"),
        (10, 10, -15, False, "10d/K10/SL15"),
        (10, 10, -20, False, "10d/K10/SL20"),
        # 加大盘过滤
        (5, 10, -15, True, "5d/K10/SL15+大盘过滤"),
        (5, 5, -10, True, "5d/K5/SL10+大盘过滤"),
        (3, 10, -10, True, "3d/K10/SL10+大盘过滤"),
        (10, 10, -15, True, "10d/K10/SL15+大盘过滤"),
    ]
    
    all_results = []
    for hold, k, sl, mf, label in param_grid:
        wf = run_wf(stock_feats, hist, 5, hold, k, sl, mf)
        if wf:
            all_results.append({
                'label': label,
                'hold': hold, 'top_k': k, 'sl': sl,
                'market_filter': mf,
                **wf
            })
            tag = "🟢" if wf['win_rate'] >= 55 else ("🟡" if wf['win_rate'] >= 50 else "🔴")
            print(f"  {tag} {label:30s} | avg={wf['avg_ret']:+.2f}% (去boom={wf['avg_no_boom']:+.2f}%) | 胜率={wf['win_rate']:.0f}%")
    
    # 排序
    all_results.sort(key=lambda x: -x['avg_no_boom'])
    
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  排名（按去boom平均收益）")
    print("=" * 60)
    for i, r in enumerate(all_results):
        tag = "🟢" if r['win_rate'] >= 55 else ("🟡" if r['win_rate'] >= 50 else "🔴")
        print(f"  {i+1:2d}. {tag} {r['label']:30s} | avg={r['avg_ret']:+.2f}% | 去boom={r['avg_no_boom']:+.2f}% | 胜率={r['win_rate']:.0f}%")
    
    # 保存
    report = {
        'type': 'param_sweep',
        'n_stocks': len(stock_feats),
        'best_by_no_boom': all_results[0]['label'] if all_results else None,
        'results': all_results,
        'elapsed_sec': round(elapsed, 1),
    }
    out_path = os.path.join(BASE, 'a1_models', 'a3_v1_param_sweep.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告: {out_path}")
    print(f"  耗时: {elapsed:.0f}s")
