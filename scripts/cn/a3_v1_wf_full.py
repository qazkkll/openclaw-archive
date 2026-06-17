#!/usr/bin/env python3
"""A3_v1 Walk-Forward 全量版 — 向量化特征计算 + 多进程
精确复制训练pipeline特征，跑全量4536只股票
"""
import sys, io, json, time, os, gc, warnings
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
print = lambda *a,**kw: (__import__('builtins').print(*a, flush=True, **kw))

import xgboost as xgb

BASE = r'/home/hermes/.hermes/openclaw-archive/data'
MODEL_PATH = os.path.join(BASE, 'a1_models', 'a3_v1.json')
HIST_PATH = os.path.join(BASE, 'a_hist_10y.parquet')

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
    """向量化EMA计算"""
    alpha = 2.0 / (period + 1)
    result = np.empty_like(arr, dtype=np.float64)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = arr[i] * alpha + result[i-1] * (1 - alpha)
    return result

def compute_features_vectorized(c, h, l, o, v, dates, start_idx=120):
    """向量化特征计算 — 精确复制训练pipeline"""
    n = len(c)
    if n < start_idx + 20:
        return None
    
    c = c.astype(np.float64)
    h = h.astype(np.float64)
    l = l.astype(np.float64)
    v = v.astype(np.float64)
    
    # === MA ===
    def ma(arr, w):
        cs = np.cumsum(arr)
        cs[w:] = cs[w:] - cs[:-w]
        result = np.full(n, np.nan)
        result[w-1:] = cs[w-1:] / w
        return result
    
    ma5 = ma(c, 5)
    ma10 = ma(c, 10)
    ma20 = ma(c, 20)
    ma60 = ma(c, 60)
    ma120 = ma(c, 120)
    ma120 = np.where(np.isnan(ma120), ma60, ma120)
    
    # === MA slope ===
    ma20_prev = np.full(n, np.nan)
    ma60_prev = np.full(n, np.nan)
    for i in range(start_idx, n):
        if i >= 24: ma20_prev[i] = np.mean(c[i-24:i-4])
        if i >= 64: ma60_prev[i] = np.mean(c[i-64:i-4])
    
    # === ATR ===
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    tr_full = np.zeros(n)
    tr_full[1:] = tr
    
    # === RSI ===
    diff = np.diff(c)
    
    # === EMA for MACD ===
    ema12 = _ema_np(c, 12)
    ema26 = _ema_np(c, 26)
    dif_arr = ema12 - ema26
    
    # DEA = 9-day EMA of DIF
    dea_arr = _ema_np(dif_arr, 9)
    
    # === OBV ===
    obv_full = np.zeros(n)
    for j in range(1, n):
        if c[j] > c[j-1]: obv_full[j] = v[j]
        elif c[j] < c[j-1]: obv_full[j] = -v[j]
    obv_cum = np.cumsum(obv_full)
    
    # === 逐行组装特征 (start_idx 到 n-11) ===
    valid_range = range(start_idx, n - 10)
    n_rows = len(valid_range)
    if n_rows < 10:
        return None
    
    feat_matrix = np.zeros((n_rows, 33))
    date_list = []
    
    # KDJ stateful
    prev_k = 50.0
    prev_d = 50.0
    
    for idx, i in enumerate(valid_range):
        price = c[i]
        
        # MA pct
        feat_matrix[idx, 0] = (price/ma5[i]-1)*100 if ma5[i] > 0 else 0
        feat_matrix[idx, 1] = (price/ma10[i]-1)*100 if ma10[i] > 0 else 0
        feat_matrix[idx, 2] = (price/ma20[i]-1)*100 if ma20[i] > 0 else 0
        feat_matrix[idx, 3] = (price/ma60[i]-1)*100 if ma60[i] > 0 else 0
        feat_matrix[idx, 4] = (price/ma120[i]-1)*100 if ma120[i] > 0 else 0
        
        # MA slope
        feat_matrix[idx, 5] = (ma20[i]/ma20_prev[i]-1)*100 if not np.isnan(ma20_prev[i]) and ma20_prev[i] > 0 else 0
        feat_matrix[idx, 6] = (ma60[i]/ma60_prev[i]-1)*100 if not np.isnan(ma60_prev[i]) and ma60_prev[i] > 0 else 0
        
        # MA align
        feat_matrix[idx, 7] = (ma5[i]/ma60[i]-1)*100 if ma60[i] > 0 else 0
        
        # Volume
        vol10 = np.mean(v[i-9:i+1])
        vol60 = np.mean(v[i-59:i+1])
        feat_matrix[idx, 8] = vol10
        feat_matrix[idx, 9] = vol60
        feat_matrix[idx, 10] = vol10 / vol60 if vol60 > 0 else 1.0
        
        # ATR
        tr20 = np.mean(tr_full[i-19:i+1])
        feat_matrix[idx, 11] = tr20 / price * 100 if price > 0 else 0
        
        # Returns
        feat_matrix[idx, 12] = (c[i]/c[i-1]-1)*100 if i >= 1 else 0
        feat_matrix[idx, 13] = (c[i]/c[i-4]-1)*100 if i >= 4 else 0
        feat_matrix[idx, 14] = (c[i]/c[i-9]-1)*100 if i >= 9 else 0
        feat_matrix[idx, 15] = (c[i]/c[i-19]-1)*100 if i >= 19 else 0
        feat_matrix[idx, 16] = (c[i]/c[i-59]-1)*100 if i >= 59 else 0
        
        # RSI
        g = diff[i-13:i+1]
        gains = np.maximum(g, 0)
        losses = np.maximum(-g, 0)
        ag = np.mean(gains)
        al = np.mean(losses)
        if al > 0:
            feat_matrix[idx, 17] = 100 - 100/(1 + ag/al)
        else:
            feat_matrix[idx, 17] = 100
        
        # Vol ratio 5/20
        vol5 = np.mean(v[i-4:i+1])
        vol20 = np.mean(v[i-19:i+1])
        feat_matrix[idx, 18] = vol5 / vol20 if vol20 > 0 else 1.0
        
        # KDJ (stateful)
        low9 = np.min(l[i-8:i+1])
        high9 = np.max(h[i-8:i+1])
        rsv = (c[i] - low9) / (high9 - low9) * 100 if high9 > low9 else 50
        k_val = 2/3 * prev_k + 1/3 * rsv
        d_val = 2/3 * prev_d + 1/3 * k_val
        prev_k, prev_d = k_val, d_val
        feat_matrix[idx, 19] = round(k_val, 2)
        feat_matrix[idx, 20] = round(d_val, 2)
        feat_matrix[idx, 21] = round(3*k_val - 2*d_val, 2)
        
        # MACD
        feat_matrix[idx, 22] = round(dif_arr[i], 4)
        feat_matrix[idx, 23] = round(dea_arr[i], 4)
        feat_matrix[idx, 24] = round((dif_arr[i] - dea_arr[i])*2, 4)
        
        # Bollinger
        std20 = np.std(c[i-19:i+1])
        feat_matrix[idx, 25] = std20 / ma20[i] * 100 if ma20[i] > 0 else 0
        feat_matrix[idx, 26] = (price - (ma20[i] - 2*std20)) / (4*std20) * 100 if std20 > 0 else 50
        
        # OBV ratio
        obv5 = np.mean(obv_cum[i-4:i+1])
        obv20 = np.mean(obv_cum[i-19:i+1])
        feat_matrix[idx, 27] = obv5 / obv20 if abs(obv20) > 0 else 1.0
        
        # ret5_max (用high)
        feat_matrix[idx, 28] = (np.max(h[i-4:i+1]) / price - 1) * 100
        
        # ret3_vs_ema12
        feat_matrix[idx, 29] = (c[i]/ema12[i]-1)*100 if ema12[i] > 0 else 0
        
        # accel
        feat_matrix[idx, 30] = feat_matrix[idx, 13] - feat_matrix[idx, 14]
        
        # ma5_ma10_cross
        feat_matrix[idx, 31] = ma5[i] / ma10[i] - 1 if ma10[i] > 0 else 0
        
        # vol_breakout
        vol40 = np.mean(v[i-39:i+1]) if i >= 39 else np.mean(v[:i+1])
        feat_matrix[idx, 32] = v[i] / vol40 if vol40 > 0 else 1.0
        
        date_list.append(dates[i])
    
    return {'dates': date_list, 'features': feat_matrix}


def process_stock(code, data):
    """处理单只股票"""
    try:
        c = np.array(data['c'])
        h = np.array(data['h'])
        l = np.array(data['l'])
        o = np.array(data['o'])
        v = np.array(data['v'])
        dates = data['dates']
        
        if len(dates) < 250:
            return code, None
        
        result = compute_features_vectorized(c, h, l, o, v, dates)
        return code, result
    except Exception as e:
        return code, None


def main():
    t0 = time.time()
    print("=" * 60)
    print("  A3_v1 Walk-Forward 全量版（向量化）")
    print("=" * 60)
    
    # [1] 加载模型
    print("\n[1] 加载模型...")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    
    # [2] 加载数据
    print("[2] 加载数据...")
    with open(HIST_PATH, 'r', encoding='utf-8') as f:
        hist = json.load(f)
    
    valid = {k: v for k, v in hist.items() if len(v.get('dates', [])) >= 250}
    print(f"  有效股票: {len(valid)} 只")
    
    # [3] 向量化特征计算（多进程）
    print(f"[3] 特征计算（全量 {len(valid)} 只，{cpu_count()} 进程）...")
    
    tasks = [(code, data) for code, data in valid.items()]
    
    n_cpus = min(cpu_count(), 8)
    with Pool(n_cpus) as pool:
        results = pool.starmap(process_stock, tasks, chunksize=50)
    
    # 收集结果
    stock_feats = {}
    for code, result in results:
        if result is not None:
            stock_feats[code] = result
    
    del results, valid
    gc.collect()
    
    print(f"  完成: {len(stock_feats)} 只有有效特征")
    
    # [4] 收集所有日期
    all_dates = set()
    for code, data in stock_feats.items():
        all_dates.update(data['dates'])
    dates_sorted = sorted(all_dates)
    print(f"  日期范围: {dates_sorted[0]} ~ {dates_sorted[-1]}, 共{len(dates_sorted)}天")
    
    # [5] Walk-Forward
    print("\n[4] Walk-Forward 评估 (5折, 持有5天, top10, 止损-15%)...")
    n_splits = 5
    split_size = len(dates_sorted) // (n_splits + 1)
    
    wf_results = []
    for fold in range(n_splits):
        train_end = dates_sorted[split_size * (fold + 1) - 1]
        test_start_idx = split_size * (fold + 1)
        test_end_idx = min(test_start_idx + 10, len(dates_sorted) - 1)
        test_dates = dates_sorted[test_start_idx:test_end_idx+1]
        if len(test_dates) < 5:
            continue
        
        # 对每只股票打分
        scores = []
        for code, data in stock_feats.items():
            # 找 train_end 日或之前最近的特征
            feat_dates = data['dates']
            feat_matrix = data['features']
            
            # 二分查找 train_end
            insert_pos = 0
            for j, dt in enumerate(feat_dates):
                if dt <= train_end:
                    insert_pos = j + 1
                else:
                    break
            
            if insert_pos == 0:
                continue
            
            feat_idx = insert_pos - 1
            X = feat_matrix[feat_idx:feat_idx+1]
            dmat = xgb.DMatrix(X, feature_names=FEATURES)
            pred = model.predict(dmat)[0]
            
            # 找 entry: train_end 后第一个有数据的日期
            entry_date_idx = insert_pos
            if entry_date_idx >= len(feat_dates):
                continue
            
            scores.append({
                'code': code,
                'score': float(pred),
                'entry_date': feat_dates[entry_date_idx],
            })
        
        if not scores:
            continue
        
        # 选 top10
        scores.sort(key=lambda x: -x['score'])
        top10 = scores[:10]
        
        # 从原始数据取5日收益
        rets = []
        for s in top10:
            code = s['code']
            d = hist.get(code)
            if d is None:
                continue
            
            dates_arr = d['dates']
            c_arr = d['c']
            
            # 找 entry_date 的索引
            entry_date = s['entry_date']
            entry_idx = None
            for j, dt in enumerate(dates_arr):
                if dt == entry_date:
                    entry_idx = j
                    break
            
            if entry_idx is None or entry_idx + 5 >= len(dates_arr):
                continue
            
            entry_p = c_arr[entry_idx]
            if entry_p <= 0:
                continue
            
            # 5日收益
            exit_p = c_arr[entry_idx + 5]
            ret = (exit_p / entry_p - 1) * 100
            
            # 止损检查
            min_p = min(c_arr[entry_idx:entry_idx+6])
            max_loss = (min_p / entry_p - 1) * 100
            if max_loss <= -15:
                ret = -15.0
            
            rets.append(ret)
        
        if not rets:
            continue
        
        avg_ret = np.mean(rets)
        wf_results.append({
            'fold': fold + 1,
            'train_end': train_end,
            'n_candidates': len(scores),
            'avg_ret_5d': round(avg_ret, 2),
            'wins': sum(1 for r in rets if r > 0),
            'total': len(rets),
        })
        print(f"  Fold {fold+1}: train_end={train_end}, candidates={len(scores)}, top10_avg={avg_ret:.2f}%, wins={sum(1 for r in rets if r > 0)}/{len(rets)}")
    
    elapsed = time.time() - t0
    
    print("\n" + "=" * 60)
    if wf_results:
        avg = np.mean([r['avg_ret_5d'] for r in wf_results])
        total_wins = sum(r['wins'] for r in wf_results)
        total_all = sum(r['total'] for r in wf_results)
        print(f"  平均5日收益: {avg:.2f}%")
        print(f"  总胜率: {total_wins}/{total_all} = {total_wins/total_all*100:.1f}%")
        print(f"  折数: {len(wf_results)}")
        
        report = {
            'model': 'a3_v1',
            'type': 'walk_forward_full_vectorized',
            'params': {'n_splits': 5, 'hold_days': 5, 'top_k': 10, 'sl_pct': -15},
            'n_stocks': len(stock_feats),
            'avg_ret_5d': round(avg, 2),
            'win_rate': round(total_wins/total_all*100, 1),
            'folds': wf_results,
            'elapsed_sec': round(elapsed, 1),
        }
        
        out_path = os.path.join(BASE, 'a1_models', 'a3_v1_wf_full.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  报告: {out_path}")
    else:
        print("  无有效结果")
    
    print(f"  耗时: {elapsed:.0f}s")

if __name__ == '__main__':
    main()
