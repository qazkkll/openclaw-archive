#!/usr/bin/env python3
"""A3_v1 Walk-Forward — 精确复制训练pipeline特征计算"""
import sys, io, json, time, os, gc
import numpy as np
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

def _ema(arr, period):
    if len(arr) < 1: return []
    m = 2 / (period + 1)
    r = [arr[0]]
    for i in range(1, len(arr)):
        r.append(arr[i] * m + r[-1] * (1 - m))
    return r

def compute_features(c, h, l, o, v, dates, start_idx=120):
    """精确复制 a1_train_pipeline.py 的 compute_features"""
    n = len(c)
    results = []
    prev_k, prev_d = 50.0, 50.0
    
    for i in range(start_idx, n - 10):
        price = c[i]
        rec = {}
        
        ma5  = np.mean(c[i-4:i+1])
        ma10 = np.mean(c[i-9:i+1])
        ma20 = np.mean(c[i-19:i+1])
        ma60 = np.mean(c[i-59:i+1])
        ma120 = np.mean(c[i-119:i+1]) if i >= 119 else ma60
        
        rec['pct_ma5']   = (price/ma5-1)*100 if ma5 > 0 else 0
        rec['pct_ma10']  = (price/ma10-1)*100 if ma10 > 0 else 0
        rec['pct_ma20']  = (price/ma20-1)*100 if ma20 > 0 else 0
        rec['pct_ma60']  = (price/ma60-1)*100 if ma60 > 0 else 0
        rec['pct_ma120'] = (price/ma120-1)*100 if ma120 > 0 else 0
        
        ma20_prev = np.mean(c[i-24:i-4])
        ma60_prev = np.mean(c[i-64:i-4])
        rec['ma20_slope'] = (ma20/ma20_prev - 1)*100 if ma20_prev > 0 else 0
        rec['ma60_slope'] = (ma60/ma60_prev - 1)*100 if ma60_prev > 0 else 0
        
        rec['ma_align'] = (ma5/ma60 - 1)*100 if ma60 > 0 else 0
        
        rec['vol_10d'] = np.mean(v[i-9:i+1])
        rec['vol_60d'] = np.mean(v[i-59:i+1])
        rec['vol_ratio'] = rec['vol_10d'] / rec['vol_60d'] if rec['vol_60d'] > 0 else 1.0
        
        tr = [max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(i-19, i+1)]
        rec['atr20_pct'] = np.mean(tr) / price * 100 if price > 0 else 0
        
        rec['ret_1d']  = (c[i]/c[i-1]-1)*100 if i >= 1 else 0
        rec['ret_5d']  = (c[i]/c[i-4]-1)*100 if i >= 4 else 0
        rec['ret_10d'] = (c[i]/c[i-9]-1)*100 if i >= 9 else 0
        rec['ret_20d'] = (c[i]/c[i-19]-1)*100 if i >= 19 else 0
        rec['ret_60d'] = (c[i]/c[i-59]-1)*100 if i >= 59 else 0
        
        gains = [max(c[j]-c[j-1], 0) for j in range(i-13, i+1)]
        losses = [max(c[j-1]-c[j], 0) for j in range(i-13, i+1)]
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rec['rsi14'] = 100 - 100 / (1 + rs)
        else:
            rec['rsi14'] = 100
        
        vol5 = np.mean(v[i-4:i+1])
        vol20 = np.mean(v[i-19:i+1])
        rec['vol_ratio_5_20'] = vol5 / vol20 if vol20 > 0 else 1.0
        
        low9 = min(l[i-8:i+1])
        high9 = max(h[i-8:i+1])
        if high9 > low9:
            rsv = (c[i] - low9) / (high9 - low9) * 100
        else:
            rsv = 50
        k_val = 2/3 * prev_k + 1/3 * rsv
        d_val = 2/3 * prev_d + 1/3 * k_val
        prev_k, prev_d = k_val, d_val
        rec['kdj_k'] = round(k_val, 2)
        rec['kdj_d'] = round(d_val, 2)
        rec['kdj_j'] = round(3 * k_val - 2 * d_val, 2)
        
        ema12_arr = _ema(c[:i+1], 12)
        ema26_arr = _ema(c[:i+1], 26)
        ema12 = ema12_arr[-1]
        ema26 = ema26_arr[-1]
        dif = ema12 - ema26
        dea_start = max(0, len(ema12_arr) - 9)
        dea_vals = [ema12_arr[j] - ema26_arr[j] for j in range(dea_start, len(ema12_arr))]
        dea_arr = _ema(dea_vals, 9)
        dea = dea_arr[-1] if len(dea_arr) > 0 else dif
        rec['macd_dif'] = round(dif, 4)
        rec['macd_dea'] = round(dea, 4)
        rec['macd_bar'] = round((dif - dea)*2, 4)
        
        ma = ma20
        std20 = np.std(c[i-19:i+1])
        rec['bb_width'] = std20 / ma * 100 if ma > 0 else 0
        rec['bb_position'] = (price - (ma - 2*std20)) / (4*std20) * 100 if std20 > 0 else 50
        
        obv_vals = []
        obv = 0
        for j in range(i-19, i+1):
            if j > 0:
                if c[j] > c[j-1]: obv += v[j]
                elif c[j] < c[j-1]: obv -= v[j]
            obv_vals.append(obv)
        obv5 = np.mean(obv_vals[-5:]) if len(obv_vals) >= 5 else 0
        obv20 = np.mean(obv_vals) if obv_vals else 0
        rec['obv_ratio_5_20'] = obv5 / obv20 if obv20 > 0 else 1.0
        
        rec['ret5_max'] = (max(h[i-4:i+1]) / price - 1) * 100
        
        ema12_val = _ema(c[:i+1], 12)[-1]
        rec['ret3_vs_ema12'] = (c[i]/ema12_val - 1)*100 if ema12_val > 0 else 0
        
        rec['accel_5_10'] = rec['ret_5d'] - rec['ret_10d']
        
        rec['ma5_ma10_cross'] = ma5 / ma10 - 1 if ma10 > 0 else 0
        
        vol_avg = np.mean(v[i-39:i+1])
        rec['vol_breakout'] = v[i] / vol_avg if vol_avg > 0 else 1.0
        
        results.append((dates[i], rec))
    
    return results

def main():
    t0 = time.time()
    print("=" * 60)
    print("  A3_v1 Walk-Forward（精确特征版）")
    print("=" * 60)
    
    print("\n[1] 加载模型...")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    
    print("[2] 加载数据...")
    with open(HIST_PATH, 'r', encoding='utf-8') as f:
        hist = json.load(f)
    
    # 过滤 >=250天
    valid = {k: v for k, v in hist.items() if len(v.get('dates', [])) >= 250}
    del hist; gc.collect()
    print(f"  有效股票: {len(valid)} 只")
    
    # 抽样300只
    import random; random.seed(42)
    codes = random.sample(list(valid.keys()), min(300, len(valid)))
    
    # 预计算所有股票的全部特征序列
    print("[3] 计算特征（300只，精确复制训练pipeline）...")
    stock_feats = {}  # code -> [(date, feat_dict), ...]
    for idx, code in enumerate(codes):
        d = valid[code]
        c, h, l, o, v_arr = d['c'], d['h'], d['l'], d['o'], d['v']
        dates = d['dates']
        feats = compute_features(c, h, l, o, v_arr, dates)
        if feats:
            stock_feats[code] = feats
        if (idx+1) % 50 == 0:
            print(f"  {idx+1}/{len(codes)} done")
    
    print(f"  完成: {len(stock_feats)} 只有有效特征")
    del valid; gc.collect()
    
    # 收集所有日期
    all_dates = set()
    for feats in stock_feats.values():
        for dt, _ in feats:
            all_dates.add(dt)
    dates_sorted = sorted(all_dates)
    print(f"  日期范围: {dates_sorted[0]} ~ {dates_sorted[-1]}, 共{len(dates_sorted)}天")
    
    # Walk-Forward: 5折
    print("\n[4] Walk-Forward 评估...")
    n_splits = 5
    split_size = len(dates_sorted) // (n_splits + 1)
    
    wf_results = []
    for fold in range(n_splits):
        train_end = dates_sorted[split_size * (fold + 1) - 1]
        test_start_idx = split_size * (fold + 1)
        # 找 train_end 之后最近的一个交易日的5天收益
        test_end_idx = min(test_start_idx + 10, len(dates_sorted) - 1)
        test_dates = dates_sorted[test_start_idx:test_end_idx+1]
        if len(test_dates) < 5:
            continue
        
        # 对每只股票：取 train_end 日的特征 -> 预测 -> 算5日收益
        scores = []
        for code, feats in stock_feats.items():
            # 找 train_end 日或之前最近的特征
            train_feat = None
            for dt, fd in feats:
                if dt <= train_end:
                    train_feat = (dt, fd)
                else:
                    break
            
            if train_feat is None:
                continue
            
            # 找 entry: train_end 之后第一个交易日
            entry_date = None
            entry_price = None
            for dt, fd in feats:
                if dt > train_end:
                    entry_date = dt
                    # 用当天close作为entry（简化：用前一天close+ret_1d反推不准，直接用open近似）
                    break
            
            if entry_date is None:
                continue
            
            # 找5日后价格
            feat_dates = [dt for dt, _ in feats]
            if entry_date not in feat_dates:
                continue
            entry_idx = feat_dates.index(entry_date)
            if entry_idx + 5 >= len(feat_dates):
                continue
            
            # 用特征中的价格反推 entry price
            # 更简单：直接用 close 数组
            # 但我们没存close... 用 pct_ma5 反推太复杂
            # 简化：用 train_end 日的 ret_1d 和价格关系
            # 实际上我们需要原始价格
            
            # 重新：直接从原始数据取
            pass
        
        # 上面的方法太复杂，换个思路：
        # 直接在 compute_features 时同时返回 close 价格
        # 但为了不改太多，用另一种方式：
        # 在 train_end 日打分，然后从原始hist取5日收益
        
        scores = []
        for code, feats in stock_feats.items():
            # 找 train_end 日最近的特征
            best = None
            for dt, fd in feats:
                if dt <= train_end:
                    best = (dt, fd)
            if best is None:
                continue
            
            _, feat_dict = best
            X = [[feat_dict.get(f, 0) for f in FEATURES]]
            dmat = xgb.DMatrix(X, feature_names=FEATURES)
            pred = model.predict(dmat)[0]
            
            scores.append({'code': code, 'score': float(pred), 'feat_date': best[0]})
        
        if not scores:
            continue
        
        scores.sort(key=lambda x: -x['score'])
        top10 = scores[:10]
        
        # 从原始数据取这10只的5日收益
        rets = []
        for s in top10:
            code = s['code']
            d = hist_raw.get(code)
            if d is None:
                continue
            dates_arr = d['dates']
            c_arr = d['c']
            # 找 train_end 后第一天买入（用close）
            entry_idx = None
            for i, dt in enumerate(dates_arr):
                if dt > train_end:
                    entry_idx = i
                    break
            if entry_idx is None or entry_idx + 5 >= len(dates_arr):
                continue
            entry_p = c_arr[entry_idx]
            exit_p = c_arr[entry_idx + 5]
            ret = (exit_p / entry_p - 1) * 100
            # 止损
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
            'win_count': sum(1 for r in rets if r > 0),
        })
        print(f"  Fold {fold+1}: train_end={train_end}, top10_avg_5d={avg_ret:.2f}%, wins={sum(1 for r in rets if r > 0)}/{len(rets)}")
    
    elapsed = time.time() - t0
    
    print("\n" + "=" * 60)
    if wf_results:
        avg = np.mean([r['avg_ret_5d'] for r in wf_results])
        wr = np.mean([r['win_count']/r['n_candidates'][:10] if False else r['win_count']/min(10,r['n_candidates']) for r in wf_results])
        print(f"  平均5日收益: {avg:.2f}%")
        print(f"  折数: {len(wf_results)}")
        
        report = {
            'model': 'a3_v1',
            'type': 'walk_forward_exact',
            'params': {'n_splits': 5, 'hold_days': 5, 'top_k': 10, 'sl_pct': -15, 'n_stocks': 300},
            'avg_ret_5d': round(avg, 2),
            'folds': wf_results,
            'elapsed_sec': round(elapsed, 1),
        }
        
        out_path = os.path.join(BASE, 'a1_models', 'a3_v1_wf_exact.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  报告: {out_path}")
    else:
        print("  无有效结果")
    
    print(f"  耗时: {elapsed:.0f}s")

if __name__ == '__main__':
    # 需要保留原始hist来取收益
    with open(HIST_PATH, 'r', encoding='utf-8') as f:
        hist_raw = json.load(f)
    main()
