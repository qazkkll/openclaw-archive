#!/usr/bin/env python3
"""
a3_v1_wf_only.py — A3_v1 Walk-Forward 验证（轻量版）
只跑 Walk-Forward，不跑全量参数扫描
"""
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

def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def calc_features(ohlcv):
    """从 OHLCV 计算特征矩阵"""
    n = len(ohlcv)
    if n < 120:
        return None, None
    
    c = np.array([x['c'] for x in ohlcv], dtype=np.float64)
    h = np.array([x['h'] for x in ohlcv], dtype=np.float64)
    l = np.array([x['l'] for x in ohlcv], dtype=np.float64)
    v = np.array([x['v'] for x in ohlcv], dtype=np.float64)
    
    # 均线
    ma5 = np.convolve(c, np.ones(5)/5, mode='valid')
    ma10 = np.convolve(c, np.ones(10)/10, mode='valid')
    ma20 = np.convolve(c, np.ones(20)/20, mode='valid')
    ma60 = np.convolve(c, np.ones(60)/60, mode='valid')
    ma120 = np.convolve(c, np.ones(120)/120, mode='valid')
    
    # 对齐长度
    min_len = min(len(ma5), len(ma10), len(ma20), len(ma60), len(ma120))
    offset = n - min_len
    
    c = c[offset:]
    h = h[offset:]
    l = l[offset:]
    v = v[offset:]
    ma5 = ma5[-min_len:]
    ma10 = ma10[-min_len:]
    ma20 = ma20[-min_len:]
    ma60 = ma60[-min_len:]
    ma120 = ma120[-min_len:]
    
    # 特征计算
    feats = []
    labels_5d = []
    labels_10d = []
    
    for i in range(120, min_len - 10):
        try:
            pct_ma5 = (c[i] - ma5[i]) / ma5[i] * 100
            pct_ma10 = (c[i] - ma10[i]) / ma10[i] * 100
            pct_ma20 = (c[i] - ma20[i]) / ma20[i] * 100
            pct_ma60 = (c[i] - ma60[i]) / ma60[i] * 100
            pct_ma120 = (c[i] - ma120[i]) / ma120[i] * 100
            
            ma20_slope = (ma20[i] - ma20[i-5]) / ma20[i-5] * 100 if i >= 5 else 0
            ma60_slope = (ma60[i] - ma60[i-10]) / ma60[i-10] * 100 if i >= 10 else 0
            
            ma_align = 1.0 if ma5[i] > ma10[i] > ma20[i] > ma60[i] else (
                -1.0 if ma5[i] < ma10[i] < ma20[i] < ma60[i] else 0.0)
            
            vol_10d = np.std(v[max(0,i-9):i+1]) / (np.mean(v[max(0,i-9):i+1]) + 1e-8)
            vol_60d = np.std(v[max(0,i-59):i+1]) / (np.mean(v[max(0,i-59):i+1]) + 1e-8)
            vol_ratio = np.mean(v[max(0,i-4):i+1]) / (np.mean(v[max(0,i-19):i+1]) + 1e-8)
            
            tr_list = []
            for j in range(max(1, i-19), i+1):
                tr_list.append(max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])))
            atr20 = np.mean(tr_list) if tr_list else 0
            atr20_pct = atr20 / c[i] * 100
            
            ret_1d = (c[i] - c[i-1]) / c[i-1] * 100
            ret_5d = (c[i] - c[i-5]) / c[i-5] * 100 if i >= 5 else 0
            ret_10d = (c[i] - c[i-10]) / c[i-10] * 100 if i >= 10 else 0
            ret_20d = (c[i] - c[i-20]) / c[i-20] * 100 if i >= 20 else 0
            ret_60d = (c[i] - c[i-60]) / c[i-60] * 100 if i >= 60 else 0
            
            # RSI14
            gains, losses = [], []
            for j in range(max(1, i-13), i+1):
                diff = c[j] - c[j-1]
                gains.append(max(diff, 0))
                losses.append(max(-diff, 0))
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 1e-8
            rs = avg_gain / (avg_loss + 1e-8)
            rsi14 = 100 - 100 / (1 + rs)
            
            vol_ratio_5_20 = np.mean(v[max(0,i-4):i+1]) / (np.mean(v[max(0,i-19):i+1]) + 1e-8)
            
            # KDJ
            period = 9
            if i >= period:
                low_n = np.min(l[i-period+1:i+1])
                high_n = np.max(h[i-period+1:i+1])
                rsv = (c[i] - low_n) / (high_n - low_n + 1e-8) * 100
            else:
                rsv = 50
            kdj_k = rsv  # 简化
            kdj_d = rsv
            kdj_j = 3 * rsv - 2 * rsv
            
            # MACD (简化)
            ema12 = c[i]
            ema26 = c[i]
            macd_dif = 0
            macd_dea = 0
            macd_bar = 0
            
            # BB
            bb_mid = ma20[i]
            bb_std = np.std(c[max(0,i-19):i+1])
            bb_width = (bb_std * 2) / (bb_mid + 1e-8) * 100
            bb_position = (c[i] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-8)
            
            # OBV ratio
            obv_up = sum(v[j] for j in range(max(1,i-4), i+1) if c[j] > c[j-1])
            obv_dn = sum(v[j] for j in range(max(1,i-4), i+1) if c[j] <= c[j-1])
            obv_ratio_5_20_val = obv_up / (obv_dn + 1e-8)
            
            ret5_max = max((c[j] - c[i]) / c[i] * 100 for j in range(i+1, min(i+6, min_len)))
            ret3_vs_ema12 = (c[i] - ema12) / (ema12 + 1e-8) * 100
            accel_5_10 = ret_5d - ret_10d if i >= 10 else 0
            ma5_ma10_cross = 1.0 if ma5[i] > ma10[i] and ma5[i-1] <= ma10[i-1] else (
                -1.0 if ma5[i] < ma10[i] and ma5[i-1] >= ma10[i-1] else 0.0)
            vol_breakout = 1.0 if vol_ratio > 2.0 else 0.0
            
            feat = [
                pct_ma5, pct_ma10, pct_ma20, pct_ma60, pct_ma120,
                ma20_slope, ma60_slope, ma_align,
                vol_10d, vol_60d, vol_ratio, atr20_pct,
                ret_1d, ret_5d, ret_10d, ret_20d, ret_60d, rsi14,
                vol_ratio_5_20, kdj_k, kdj_d, kdj_j,
                macd_dif, macd_dea, macd_bar, bb_width, bb_position,
                obv_ratio_5_20_val, ret5_max, ret3_vs_ema12, accel_5_10,
                ma5_ma10_cross, vol_breakout,
            ]
            
            # 标签：未来5日/10日收益
            if i + 5 < min_len:
                ret5 = (c[i+5] - c[i]) / c[i] * 100
            else:
                continue
            if i + 10 < min_len:
                ret10 = (c[i+10] - c[i]) / c[i] * 100
            else:
                ret10 = ret5
            
            feats.append(feat)
            labels_5d.append(ret5)
            labels_10d.append(ret10)
            
        except:
            continue
    
    if len(feats) < 50:
        return None, None
    
    return np.array(feats), {'ret5': np.array(labels_5d), 'ret10': np.array(labels_10d)}

def walk_forward_eval(model, stocks_data, n_splits=5, hold_days=5, top_k=10, sl_pct=-15):
    """Walk-Forward 评估"""
    all_dates = set()
    for code, data in stocks_data.items():
        for d in data.get('dates', []):
            all_dates.add(d)
    
    dates = sorted(all_dates)
    if len(dates) < 300:
        return None
    
    # 分成 n_splits 段
    split_size = len(dates) // (n_splits + 1)
    results = []
    
    for fold in range(n_splits):
        train_end = dates[split_size * (fold + 1) - 1]
        test_start_idx = split_size * (fold + 1)
        test_end_idx = min(test_start_idx + hold_days + 5, len(dates))
        
        if test_end_idx >= len(dates):
            break
        
        test_dates = dates[test_start_idx:test_end_idx]
        
        # 对每只股票在 train_end 日打分
        scores = []
        for code, data in stocks_data.items():
            dates_arr = data.get('dates', [])
            c_arr = data.get('c', [])
            h_arr = data.get('h', [])
            l_arr = data.get('l', [])
            o_arr = data.get('o', [])
            v_arr = data.get('v', [])
            
            # 构建 train bars
            train_bars = [{'d': dates_arr[i], 'c': c_arr[i], 'h': h_arr[i], 'l': l_arr[i], 'o': o_arr[i], 'v': v_arr[i]} for i in range(len(dates_arr)) if dates_arr[i] <= train_end]
            if len(train_bars) < 200:
                continue
            
            feats, labels = calc_features(train_bars)
            if feats is None:
                continue
            
            # 取最后一行
            X = feats[-1:].tolist()
            dmat = xgb.DMatrix(X, feature_names=FEATURES)
            pred = model.predict(dmat)[0]
            
            # 找到 test_dates 中的 entry price
            date_to_idx = {dates_arr[i]: i for i in range(len(dates_arr))}
            test_indices = [date_to_idx[d] for d in test_dates if d in date_to_idx]
            if len(test_indices) < 2:
                continue
            
            entry = c_arr[test_indices[0]]
            highs = [h_arr[i] for i in test_indices]
            lows = [l_arr[i] for i in test_indices]
            final = c_arr[test_indices[-1]]
            
            max_gain = (max(highs) - entry) / entry * 100
            max_loss = (min(lows) - entry) / entry * 100
            actual_ret = (final - entry) / entry * 100
            
            # 止损判断
            if max_loss <= sl_pct:
                actual_ret = sl_pct
            
            scores.append({
                'code': code,
                'score': float(pred),
                'ret': actual_ret,
            })
        
        if not scores:
            continue
        
        # 选 top_k
        scores.sort(key=lambda x: -x['score'])
        top = scores[:top_k]
        avg_ret = np.mean([s['ret'] for s in top])
        
        results.append({
            'fold': fold + 1,
            'train_end': train_end,
            'test_period': f"{test_dates[0]}~{test_dates[-1]}",
            'n_candidates': len(scores),
            'avg_ret': round(avg_ret, 2),
        })
        
        print(f"  Fold {fold+1}/{n_splits}: train_end={train_end}, avg_ret={avg_ret:.2f}%")
    
    if not results:
        return None
    
    avg_ret = np.mean([r['avg_ret'] for r in results])
    win_count = sum(1 for r in results if r['avg_ret'] > 0)
    
    return {
        'n_folds': len(results),
        'avg_ret_per_fold': round(avg_ret, 2),
        'win_rate': round(win_count / len(results) * 100, 1),
        'folds': results,
    }

def main():
    t0 = time.time()
    
    print("=" * 60)
    print("  A3_v1 Walk-Forward 验证（轻量版）")
    print("=" * 60)
    
    # 加载模型
    print("\n[1/3] 加载模型...")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    print(f"  OK: {MODEL_PATH}")
    
    # 加载数据（只取200只）
    print("\n[2/3] 加载历史数据（抽样200只）...")
    hist = load_json(HIST_PATH)
    codes = list(hist.keys())
    
    # 过滤：至少250天数据（列式格式）
    valid = []
    for code in codes:
        dates = hist[code].get('dates', [])
        if len(dates) >= 250:
            valid.append(code)
    
    print(f"  总计 {len(codes)} 只, 符合条件(>=250天): {len(valid)} 只")
    
    # 抽样200只
    import random
    random.seed(42)
    sample_codes = random.sample(valid, min(200, len(valid)))
    sample_data = {code: hist[code] for code in sample_codes}
    
    del hist
    gc.collect()
    
    print(f"  抽样: {len(sample_codes)} 只")
    
    # Walk-Forward
    print("\n[3/3] Walk-Forward 评估 (5折, 持有5天, top10, 止损-15%)...")
    wf = walk_forward_eval(model, sample_data, n_splits=5, hold_days=5, top_k=10, sl_pct=-15)
    
    elapsed = time.time() - t0
    
    # 输出结果
    print("\n" + "=" * 60)
    print("  结果")
    print("=" * 60)
    
    if wf:
        print(f"  折数: {wf['n_folds']}")
        print(f"  平均每折收益: {wf['avg_ret_per_fold']:.2f}%")
        print(f"  胜率: {wf['win_rate']:.1f}%")
        print(f"  耗时: {elapsed:.1f}s")
        
        # 保存
        report = {
            'model': 'a3_v1',
            'type': 'walk_forward_lite',
            'params': {'n_splits': 5, 'hold_days': 5, 'top_k': 10, 'sl_pct': -15},
            'n_stocks': len(sample_codes),
            'result': wf,
            'elapsed_sec': round(elapsed, 1),
        }
        
        out_path = os.path.join(BASE, 'a1_models', 'a3_v1_wf_lite.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n  报告已保存: {out_path}")
    else:
        print("  Walk-Forward 失败（无有效结果）")
    
    print(f"\n总耗时: {elapsed:.1f}s")

if __name__ == '__main__':
    main()
