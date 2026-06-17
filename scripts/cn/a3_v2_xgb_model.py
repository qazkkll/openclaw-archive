#!/usr/bin/env python3
"""A3_v2 模型 — 技术特征 + 资金流特征 + 市场状态特征
基于A3_v1的33个技术特征，新增：
- 资金流特征：主力/超大单/大单/中单/小单净流入占比、5日/10日均值、斜率、超大单/大单比值
- 市场状态特征：大盘MA20趋势、涨跌家数比、市场波动率、涨停/跌停家数、市场情绪
"""
import sys, json, time, os, gc, warnings
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings('ignore')

# 内存守卫
MEMORY_THRESHOLD_GB = 10  # 可用内存低于10GB时触发
MEMORY_CHECK_INTERVAL = 50  # 每处理50只股票检查一次

def check_memory():
    """检查可用内存，返回(可用GB, 是否安全)"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        free_gb = mem.available / (1024**3)
        return free_gb, free_gb >= MEMORY_THRESHOLD_GB
    except ImportError:
        # fallback: 假设安全
        return 20.0, True

def emergency_cleanup():
    """紧急清理：强制GC + 清空缓存"""
    gc.collect()
    if hasattr(np, 'core') and hasattr(np.core, 'get_ndarray_cacher'):
        np.core.get_ndarray_cacher().clear()
    print(f"🧹 紧急清理完成，可用内存: {check_memory()[0]:.1f}GB")

import xgboost as xgb

BASE = r'/home/hermes/.hermes/openclaw-archive/data'
HIST_PATH = os.path.join(BASE, 'a_hist_10y.parquet')
MONEYFLOW_PATH = os.path.join(BASE, 'moneyflow_pool.json')
MODEL_PATH = os.path.join(BASE, 'models', 'a3_v2_xgb.json')
POOL_PATH = os.path.join(BASE, 'quality_pool.json')

# === 特征列表 ===
TECH_FEATURES = [
    'pct_ma5','pct_ma10','pct_ma20','pct_ma60','pct_ma120',
    'ma20_slope','ma60_slope','ma_align',
    'vol_10d','vol_60d','vol_ratio','atr20_pct',
    'ret_1d','ret_5d','ret_10d','ret_20d','ret_60d','rsi14',
    'vol_ratio_5_20','kdj_k','kdj_d','kdj_j',
    'macd_dif','macd_dea','macd_bar','bb_width','bb_position',
    'obv_ratio_5_20','ret5_max','ret3_vs_ema12','accel_5_10',
    'ma5_ma10_cross','vol_breakout',
]

MONEYFLOW_FEATURES = [
    'mf_net_pct',        # 主力净流入占比
    'mf_super_pct',      # 超大单净流入占比
    'mf_big_pct',        # 大单净流入占比
    'mf_mid_pct',        # 中单净流入占比
    'mf_small_pct',      # 小单净流入占比
    'mf_net_ma5',        # 5日主力净流入均值
    'mf_net_ma10',       # 10日主力净流入均值
    'mf_net_slope',      # 主力净流入MA5斜率
    'mf_super_big_ratio',# 超大单/大单比值
]

MARKET_FEATURES = [
    'mkt_ma20_slope',    # 大盘MA20趋势
    'mkt_up_down_ratio', # 涨跌家数比
    'mkt_volatility',    # 市场波动率
    'mkt_limit_up',      # 涨停家数
    'mkt_limit_down',    # 跌停家数
    'mkt_sentiment',     # 市场情绪（成交额/MA20）
]

ALL_FEATURES = TECH_FEATURES + MONEYFLOW_FEATURES + MARKET_FEATURES

def _ema_np(arr, period):
    """向量化EMA计算"""
    alpha = 2.0 / (period + 1)
    result = np.empty_like(arr, dtype=np.float64)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = arr[i] * alpha + result[i-1] * (1 - alpha)
    return result

def compute_market_state(hist_data):
    """计算市场状态特征（全市场统计）"""
    print("计算市场状态特征...")
    
    # 用000001(上证指数)作为大盘参考
    ref = hist_data.get('000001')
    if not ref:
        print("  警告: 无000001数据，跳过市场状态计算")
        return {}
    
    ref_c = np.array(ref['c'], dtype=np.float64)
    ref_v = np.array(ref['v'], dtype=np.float64)
    ref_dates = ref['dates']
    
    # 大盘MA20
    ref_ma20 = np.convolve(ref_c, np.ones(20)/20, mode='valid')
    
    # 全市场统计（每日）
    all_dates = set()
    for code, data in hist_data.items():
        if code.startswith('000') or code.startswith('399'):
            continue  # 跳过指数
        all_dates.update(data['dates'])
    dates_sorted = sorted(all_dates)
    
    # 构建日期索引
    date_stats = {}
    for dt in dates_sorted:
        date_stats[dt] = {
            'up': 0, 'down': 0, 'limit_up': 0, 'limit_down': 0,
            'rets': [], 'volume': 0.0
        }
    
    # 统计每日涨跌家数、涨停跌停
    for code, data in hist_data.items():
        if code.startswith('000') or code.startswith('399'):
            continue
        c = np.array(data['c'], dtype=np.float64)
        v = np.array(data['v'], dtype=np.float64)
        dates = data['dates']
        
        for i in range(1, len(dates)):
            dt = dates[i]
            if dt not in date_stats:
                continue
            ret = (c[i] / c[i-1] - 1) * 100 if c[i-1] > 0 else 0
            date_stats[dt]['rets'].append(ret)
            date_stats[dt]['volume'] += v[i]
            
            if ret > 0:
                date_stats[dt]['up'] += 1
            elif ret < 0:
                date_stats[dt]['down'] += 1
            
            if ret >= 9.9:
                date_stats[dt]['limit_up'] += 1
            if ret <= -9.9:
                date_stats[dt]['limit_down'] += 1
    
    # 计算市场特征
    market_features = {}
    for i, dt in enumerate(dates_sorted):
        stats = date_stats[dt]
        if not stats['rets']:
            continue
        
        # 大盘MA20斜率（找最近的）
        ma20_slope = 0
        if i >= 24 and i < len(ref_ma20) + 20:
            idx = min(i - 20, len(ref_ma20) - 1)
            if idx >= 4:
                ma20_slope = (ref_ma20[idx] / ref_ma20[idx-4] - 1) * 100
        
        # 涨跌家数比
        up_down = stats['up'] / stats['down'] if stats['down'] > 0 else 1.0
        
        # 市场波动率
        volatility = np.std(stats['rets']) if len(stats['rets']) > 10 else 0
        
        # 市场情绪（成交额用全市场总成交量近似）
        sentiment = 1.0  # 简化处理
        
        market_features[dt] = {
            'mkt_ma20_slope': ma20_slope,
            'mkt_up_down_ratio': up_down,
            'mkt_volatility': volatility,
            'mkt_limit_up': stats['limit_up'],
            'mkt_limit_down': stats['limit_down'],
            'mkt_sentiment': sentiment,
        }
    
    print(f"  计算完成: {len(market_features)} 个交易日")
    return market_features

def compute_features_v2(code, hist_data, moneyflow_data, market_features, start_idx=120):
    """计算A3_v2特征：技术 + 资金流 + 市场状态"""
    data = hist_data.get(code)
    if not data or len(data['dates']) < start_idx + 20:
        return None
    
    c = np.array(data['c'], dtype=np.float64)
    h = np.array(data['h'], dtype=np.float64)
    l = np.array(data['l'], dtype=np.float64)
    o = np.array(data['o'], dtype=np.float64)
    v = np.array(data['v'], dtype=np.float64)
    dates = data['dates']
    n = len(c)
    
    # === 技术特征（复制A3_v1） ===
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
    
    ma20_prev = np.full(n, np.nan)
    ma60_prev = np.full(n, np.nan)
    for i in range(start_idx, n):
        if i >= 24: ma20_prev[i] = np.mean(c[i-24:i-4])
        if i >= 64: ma60_prev[i] = np.mean(c[i-64:i-4])
    
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    tr_full = np.zeros(n)
    tr_full[1:] = tr
    
    diff = np.diff(c)
    ema12 = _ema_np(c, 12)
    ema26 = _ema_np(c, 26)
    dif_arr = ema12 - ema26
    dea_arr = _ema_np(dif_arr, 9)
    
    obv_full = np.zeros(n)
    for j in range(1, n):
        if c[j] > c[j-1]: obv_full[j] = v[j]
        elif c[j] < c[j-1]: obv_full[j] = -v[j]
    obv_cum = np.cumsum(obv_full)
    
    # === 资金流特征 ===
    mf_data = moneyflow_data.get(code)
    has_moneyflow = mf_data is not None and 'dates' in mf_data
    
    if has_moneyflow:
        mf_dates = mf_data['dates']
        # 资金流字段（假设存在，需验证）
        mf_net = np.array(mf_data.get('net_mf_amount', [0]*len(mf_dates)), dtype=np.float64)
        mf_super = np.array(mf_data.get('super_large_amount', [0]*len(mf_dates)), dtype=np.float64)
        mf_big = np.array(mf_data.get('large_amount', [0]*len(mf_dates)), dtype=np.float64)
        mf_mid = np.array(mf_data.get('medium_amount', [0]*len(mf_dates)), dtype=np.float64)
        mf_small = np.array(mf_data.get('small_amount', [0]*len(mf_dates)), dtype=np.float64)
        
        # 计算净流入占比
        mf_net_pct = mf_net / c[:len(mf_dates)] * 100 if len(c) >= len(mf_dates) else np.zeros(len(mf_dates))
        mf_super_pct = mf_super / c[:len(mf_dates)] * 100 if len(c) >= len(mf_dates) else np.zeros(len(mf_dates))
        mf_big_pct = mf_big / c[:len(mf_dates)] * 100 if len(c) >= len(mf_dates) else np.zeros(len(mf_dates))
        mf_mid_pct = mf_mid / c[:len(mf_dates)] * 100 if len(c) >= len(mf_dates) else np.zeros(len(mf_dates))
        mf_small_pct = mf_small / c[:len(mf_dates)] * 100 if len(c) >= len(mf_dates) else np.zeros(len(mf_dates))
        
        # 5日/10日均值
        mf_net_ma5 = np.convolve(mf_net_pct, np.ones(5)/5, mode='valid')
        mf_net_ma10 = np.convolve(mf_net_pct, np.ones(10)/10, mode='valid')
        
        # MA5斜率
        mf_net_slope = np.zeros(len(mf_net_ma5))
        for i in range(4, len(mf_net_ma5)):
            mf_net_slope[i] = (mf_net_ma5[i] / mf_net_ma5[i-4] - 1) * 100 if mf_net_ma5[i-4] > 0 else 0
        
        # 超大单/大单比值
        mf_super_big_ratio = mf_super / mf_big if mf_big > 0 else np.ones(len(mf_super))
    else:
        # 无资金流数据，填0
        mf_dates = []
        mf_net_pct = mf_super_pct = mf_big_pct = mf_mid_pct = mf_small_pct = np.array([])
        mf_net_ma5 = mf_net_ma10 = mf_net_slope = mf_super_big_ratio = np.array([])
    
    # === 组装特征矩阵 ===
    valid_range = range(start_idx, n - 10)
    n_rows = len(valid_range)
    if n_rows < 10:
        return None
    
    n_features = len(ALL_FEATURES)
    feat_matrix = np.zeros((n_rows, n_features))
    date_list = []
    prev_k = 50.0
    prev_d = 50.0
    
    for idx, i in enumerate(valid_range):
        price = c[i]
        dt = dates[i]
        
        # === 技术特征 (0-32) ===
        feat_matrix[idx, 0] = (price/ma5[i]-1)*100 if ma5[i] > 0 else 0
        feat_matrix[idx, 1] = (price/ma10[i]-1)*100 if ma10[i] > 0 else 0
        feat_matrix[idx, 2] = (price/ma20[i]-1)*100 if ma20[i] > 0 else 0
        feat_matrix[idx, 3] = (price/ma60[i]-1)*100 if ma60[i] > 0 else 0
        feat_matrix[idx, 4] = (price/ma120[i]-1)*100 if ma120[i] > 0 else 0
        
        feat_matrix[idx, 5] = (ma20[i]/ma20_prev[i]-1)*100 if not np.isnan(ma20_prev[i]) and ma20_prev[i] > 0 else 0
        feat_matrix[idx, 6] = (ma60[i]/ma60_prev[i]-1)*100 if not np.isnan(ma60_prev[i]) and ma60_prev[i] > 0 else 0
        
        feat_matrix[idx, 7] = (ma5[i]/ma60[i]-1)*100 if ma60[i] > 0 else 0
        
        vol10 = np.mean(v[i-9:i+1])
        vol60 = np.mean(v[i-59:i+1])
        feat_matrix[idx, 8] = vol10
        feat_matrix[idx, 9] = vol60
        feat_matrix[idx, 10] = vol10 / vol60 if vol60 > 0 else 1.0
        
        feat_matrix[idx, 11] = np.mean(tr_full[i-19:i+1]) / price * 100 if price > 0 else 0
        
        feat_matrix[idx, 12] = (c[i]/c[i-1]-1)*100 if i >= 1 else 0
        feat_matrix[idx, 13] = (c[i]/c[i-4]-1)*100 if i >= 4 else 0
        feat_matrix[idx, 14] = (c[i]/c[i-9]-1)*100 if i >= 9 else 0
        feat_matrix[idx, 15] = (c[i]/c[i-19]-1)*100 if i >= 19 else 0
        feat_matrix[idx, 16] = (c[i]/c[i-59]-1)*100 if i >= 59 else 0
        
        g = diff[i-13:i+1]
        ag = np.mean(np.maximum(g, 0))
        al = np.mean(np.maximum(-g, 0))
        feat_matrix[idx, 17] = 100 - 100/(1 + ag/al) if al > 0 else 100
        
        vol5 = np.mean(v[i-4:i+1])
        vol20 = np.mean(v[i-19:i+1])
        feat_matrix[idx, 18] = vol5 / vol20 if vol20 > 0 else 1.0
        
        low9 = np.min(l[i-8:i+1])
        high9 = np.max(h[i-8:i+1])
        rsv = (c[i] - low9) / (high9 - low9) * 100 if high9 > low9 else 50
        k_val = 2/3 * prev_k + 1/3 * rsv
        d_val = 2/3 * prev_d + 1/3 * k_val
        prev_k, prev_d = k_val, d_val
        feat_matrix[idx, 19] = k_val
        feat_matrix[idx, 20] = d_val
        feat_matrix[idx, 21] = 3*k_val - 2*d_val
        
        feat_matrix[idx, 22] = dif_arr[i]
        feat_matrix[idx, 23] = dea_arr[i]
        feat_matrix[idx, 24] = (dif_arr[i] - dea_arr[i]) * 2
        
        std20 = np.std(c[i-19:i+1])
        feat_matrix[idx, 25] = std20 / ma20[i] * 100 if ma20[i] > 0 else 0
        feat_matrix[idx, 26] = (price - (ma20[i] - 2*std20)) / (4*std20) * 100 if std20 > 0 else 50
        
        obv5 = np.mean(obv_cum[i-4:i+1])
        obv20 = np.mean(obv_cum[i-19:i+1])
        feat_matrix[idx, 27] = obv5 / obv20 if abs(obv20) > 0 else 1.0
        
        feat_matrix[idx, 28] = (np.max(h[i-4:i+1]) / price - 1) * 100
        feat_matrix[idx, 29] = (c[i] / ema12[i] - 1) * 100 if ema12[i] > 0 else 0
        feat_matrix[idx, 30] = feat_matrix[idx, 13] - feat_matrix[idx, 14]
        feat_matrix[idx, 31] = ma5[i] / ma10[i] - 1 if ma10[i] > 0 else 0
        vol40 = np.mean(v[i-39:i+1]) if i >= 39 else np.mean(v[:i+1])
        feat_matrix[idx, 32] = v[i] / vol40 if vol40 > 0 else 1.0
        
        # === 资金流特征 (33-41) ===
        if has_moneyflow and dt in mf_dates:
            mf_idx = mf_dates.index(dt)
            if mf_idx < len(mf_net_pct):
                feat_matrix[idx, 33] = mf_net_pct[mf_idx]
                feat_matrix[idx, 34] = mf_super_pct[mf_idx]
                feat_matrix[idx, 35] = mf_big_pct[mf_idx]
                feat_matrix[idx, 36] = mf_mid_pct[mf_idx]
                feat_matrix[idx, 37] = mf_small_pct[mf_idx]
                
                if mf_idx >= 4 and mf_idx - 4 < len(mf_net_ma5):
                    feat_matrix[idx, 38] = mf_net_ma5[mf_idx - 4]
                if mf_idx >= 9 and mf_idx - 9 < len(mf_net_ma10):
                    feat_matrix[idx, 39] = mf_net_ma10[mf_idx - 9]
                if mf_idx >= 4 and mf_idx - 4 < len(mf_net_slope):
                    feat_matrix[idx, 40] = mf_net_slope[mf_idx - 4]
                if mf_idx < len(mf_super_big_ratio):
                    feat_matrix[idx, 41] = mf_super_big_ratio[mf_idx]
        
        # === 市场状态特征 (42-47) ===
        if dt in market_features:
            mkt = market_features[dt]
            feat_matrix[idx, 42] = mkt['mkt_ma20_slope']
            feat_matrix[idx, 43] = mkt['mkt_up_down_ratio']
            feat_matrix[idx, 44] = mkt['mkt_volatility']
            feat_matrix[idx, 45] = mkt['mkt_limit_up']
            feat_matrix[idx, 46] = mkt['mkt_limit_down']
            feat_matrix[idx, 47] = mkt['mkt_sentiment']
        
        date_list.append(dt)
    
    return {'dates': date_list, 'features': feat_matrix}

def process_stock(code, hist_data, moneyflow_data, market_features):
    """单只股票特征计算（多进程用）"""
    try:
        return code, compute_features_v2(code, hist_data, moneyflow_data, market_features)
    except Exception as e:
        return code, None

def walk_forward_validation(stock_feats, n_splits=5, hold_days=5, top_k=10, sl_pct=-15):
    """Walk-Forward 5折验证"""
    print(f"\n=== Walk-Forward 验证 ({n_splits}折) ===")
    
    all_dates = set()
    for data in stock_feats.values():
        all_dates.update(data['dates'])
    dates_sorted = sorted(all_dates)
    
    split_size = len(dates_sorted) // (n_splits + 1)
    wf_results = []
    
    for fold in range(n_splits):
        train_end = dates_sorted[split_size*(fold+1)-1]
        test_start_idx = split_size*(fold+1)
        test_end_idx = min(test_start_idx + hold_days + 5, len(dates_sorted)-1)
        test_dates = dates_sorted[test_start_idx:test_end_idx+1]
        
        if len(test_dates) < hold_days:
            continue
        
        print(f"\nFold {fold+1}: train_end={train_end}, test={test_dates[0]}~{test_dates[-1]}")
        
        # 收集训练/测试样本
        X_train, y_train = [], []
        X_test, y_test = [], []
        
        for code, data in stock_feats.items():
            feat_dates = data['dates']
            feat_matrix = data['features']
            
            # 找到train_end的位置
            insert_pos = 0
            for j, dt in enumerate(feat_dates):
                if dt <= train_end:
                    insert_pos = j + 1
                else:
                    break
            
            if insert_pos == 0 or insert_pos + 5 >= len(feat_dates):
                continue
            
            # 训练样本：train_end前5日收益率
            if insert_pos >= 5:
                c = np.array(hist_data[code]['c'], dtype=np.float64)
                c_dates = hist_data[code]['dates']
                
                # 找到对应的价格索引
                price_idx = None
                for k, dt in enumerate(c_dates):
                    if dt == feat_dates[insert_pos-1]:
                        price_idx = k
                        break
                
                if price_idx and price_idx >= 5 and price_idx + 5 < len(c):
                    entry_p = c[price_idx]
                    exit_p = c[price_idx + 5]
                    ret = (exit_p / entry_p - 1) * 100
                    
                    # 止损检查
                    min_p = min(c[price_idx:price_idx+6])
                    if (min_p / entry_p - 1) * 100 <= sl_pct:
                        ret = sl_pct
                    
                    label = 1 if ret > 5 else 0
                    
                    X_train.append(feat_matrix[insert_pos-1])
                    y_train.append(label)
            
            # 测试样本：test_dates中的每个日期
            for test_dt in test_dates:
                test_pos = None
                for j, dt in enumerate(feat_dates):
                    if dt == test_dt:
                        test_pos = j
                        break
                
                if test_pos is None or test_pos + 5 >= len(feat_dates):
                    continue
                
                c = np.array(hist_data[code]['c'], dtype=np.float64)
                c_dates = hist_data[code]['dates']
                
                price_idx = None
                for k, dt in enumerate(c_dates):
                    if dt == feat_dates[test_pos]:
                        price_idx = k
                        break
                
                if price_idx and price_idx + 5 < len(c):
                    entry_p = c[price_idx]
                    exit_p = c[price_idx + 5]
                    ret = (exit_p / entry_p - 1) * 100
                    
                    min_p = min(c[price_idx:price_idx+6])
                    if (min_p / entry_p - 1) * 100 <= sl_pct:
                        ret = sl_pct
                    
                    label = 1 if ret > 5 else 0
                    
                    X_test.append(feat_matrix[test_pos])
                    y_test.append(label)
        
        if not X_train or not X_test:
            print(f"  Fold {fold+1}: 无训练/测试数据，跳过")
            continue
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"  训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
        print(f"  正例比例: 训练{sum(y_train)/len(y_train)*100:.1f}%, 测试{sum(y_test)/len(y_test)*100:.1f}%")
        
        # 训练XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=ALL_FEATURES)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=ALL_FEATURES)
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'device': 'cuda',
        }
        
        model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'eval')], verbose_eval=False)
        
        # 预测
        y_pred = model.predict(dtest)
        y_pred_label = (y_pred > 0.5).astype(int)
        
        # 计算指标
        accuracy = np.mean(y_pred_label == y_test)
        
        # 模拟交易：选top_k只预测为正例的股票
        scores = list(zip(y_pred, y_test))
        scores.sort(key=lambda x: x[0], reverse=True)
        top_k_actual = scores[:top_k]
        
        if top_k_actual:
            avg_ret = np.mean([1 if s[1] == 1 else 0 for s in top_k_actual]) * 100
            wins = sum(1 for s in top_k_actual if s[1] == 1)
        else:
            avg_ret = 0
            wins = 0
        
        print(f"  Fold {fold+1} 结果:")
        print(f"    准确率: {accuracy*100:.1f}%")
        print(f"    Top-{top_k} 命中率: {wins}/{top_k} = {wins/top_k*100:.1f}%")
        
        wf_results.append({
            'fold': fold + 1,
            'train_end': train_end,
            'accuracy': round(accuracy * 100, 1),
            'top_k_hits': wins,
            'top_k_total': top_k,
            'hit_rate': round(wins / top_k * 100, 1),
        })
    
    return wf_results

def main():
    t0 = time.time()
    print("=" * 60)
    print("A3_v2 模型训练 — 技术 + 资金流 + 市场状态")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    with open(HIST_PATH, 'r', encoding='utf-8') as f:
        hist_data = json.load(f)
    print(f"  K线数据: {len(hist_data)} 只股票")
    
    with open(MONEYFLOW_PATH, 'r', encoding='utf-8') as f:
        moneyflow_data = json.load(f)
    print(f"  资金流数据: {len(moneyflow_data)} 只股票")
    
    # 2. 计算市场状态
    print("\n[2/5] 计算市场状态特征...")
    market_features = compute_market_state(hist_data)
    
    # 3. 加载候选池
    print("\n[3/5] 加载候选池...")
    if os.path.exists(POOL_PATH):
        with open(POOL_PATH, 'r', encoding='utf-8') as f:
            pool = json.load(f)
        candidate_codes = [s['symbol'] for s in pool['a']['a_stocks']]
        print(f"  候选池: {len(candidate_codes)} 只股票")
    else:
        candidate_codes = list(hist_data.keys())[:500]
        print(f"  无候选池文件，使用前500只: {len(candidate_codes)}")
    
    # 4. 计算特征（多进程）
    print("\n[4/5] 计算特征（多进程）...")
    n_cpus = cpu_count()
    print(f"  CPU核心: {n_cpus}")
    
    process_func = partial(process_stock, hist_data=hist_data, 
                          moneyflow_data=moneyflow_data, 
                          market_features=market_features)
    
    with Pool(n_cpus) as pool:
        results = pool.starmap(process_func, [(code,) for code in candidate_codes])
    
    stock_feats = {code: data for code, data in results if data is not None}
    print(f"  成功计算: {len(stock_feats)} 只股票")
    
    # 5. Walk-Forward 验证
    print("\n[5/5] Walk-Forward 验证...")
    wf_results = walk_forward_validation(stock_feats, n_splits=5, hold_days=5, top_k=10)
    
    # 6. 汇总结果
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    
    if wf_results:
        avg_acc = np.mean([r['accuracy'] for r in wf_results])
        avg_hit = np.mean([r['hit_rate'] for r in wf_results])
        total_hits = sum(r['top_k_hits'] for r in wf_results)
        total_top_k = sum(r['top_k_total'] for r in wf_results)
        
        print(f"\n总体统计:")
        print(f"  平均准确率: {avg_acc:.1f}%")
        print(f"  平均Top-10命中率: {avg_hit:.1f}%")
        print(f"  总命中: {total_hits}/{total_top_k} = {total_hits/total_top_k*100:.1f}%")
        print(f"  耗时: {elapsed:.0f}s")
        
        # 保存模型和报告
        report = {
            'model': 'a3_v2',
            'type': 'walk_forward_v2',
            'features': ALL_FEATURES,
            'n_tech': len(TECH_FEATURES),
            'n_moneyflow': len(MONEYFLOW_FEATURES),
            'n_market': len(MARKET_FEATURES),
            'n_stocks': len(stock_feats),
            'params': {
                'n_splits': 5,
                'hold_days': 5,
                'top_k': 10,
                'sl_pct': -15,
            },
            'avg_accuracy': round(avg_acc, 1),
            'avg_hit_rate': round(avg_hit, 1),
            'total_hits': total_hits,
            'total_top_k': total_top_k,
            'folds': wf_results,
            'elapsed_sec': round(elapsed, 1),
        }
        
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存: {MODEL_PATH}")
        
        # 验证标准
        print("\n=== 验证标准 ===")
        win_rate = total_hits / total_top_k * 100
        if win_rate > 55:
            print(f"✅ 胜率 {win_rate:.1f}% > 55% — 通过")
        else:
            print(f"❌ 胜率 {win_rate:.1f}% < 55% — 未通过")
            print("\n结论: 资金流+市场状态特征仍不足以预测A股个股")
    else:
        print("\n无有效验证结果")
    
    print(f"\n总耗时: {elapsed:.0f}s")

if __name__ == '__main__':
    main()
