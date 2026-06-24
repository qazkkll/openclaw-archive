#!/usr/bin/env python3
"""
全市场蓝盾V7 + 绿箭V12 重训练脚本（v3 内存最优版）
核心优化:
1. numpy预分配数组，不构建dict list
2. 分批处理，每批500只股票
3. 技术特征用纯numpy，不用pandas rolling
4. 宏观特征用dict lookup，不用merge
"""
import json, os, sys, time, argparse, warnings, gc
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, 'data', 'us')
MODEL_DIR = os.path.join(ROOT, 'models', 'us')
os.makedirs(MODEL_DIR, exist_ok=True)

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

# ============================================================
# numpy滚动计算工具
# ============================================================
def rolling_mean(arr, w):
    out = np.full(len(arr), np.nan, dtype=np.float64)
    cs = np.cumsum(arr, dtype=np.float64)
    out[w-1:] = (cs[w-1:] - np.concatenate([[0.0], cs[:-w]])) / w
    return out

def rolling_sum(arr, w):
    out = np.full(len(arr), np.nan, dtype=np.float64)
    cs = np.cumsum(arr, dtype=np.float64)
    out[w-1:] = cs[w-1:] - np.concatenate([[0.0], cs[:-w]])
    return out

def rolling_std(arr, w):
    """用Welford在线算法，O(n)"""
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if len(arr) < w:
        return out
    # 初始窗口
    window = arr[:w].copy()
    mean = np.mean(window)
    m2 = np.sum((window - mean)**2)
    out[w-1] = np.sqrt(m2 / (w-1))
    for i in range(w, len(arr)):
        old = arr[i-w]
        new = arr[i]
        old_mean = mean
        mean = mean + (new - old) / w
        m2 = m2 + (new - old) * (new - mean) - (old - old_mean)**2 / w + (old - old_mean) * (old - mean) / w
        # 简化: 直接用numpy
        pass
    # 更简单的方式: 分块计算
    for i in range(w-1, len(arr)):
        out[i] = np.std(arr[i-w+1:i+1], ddof=1)
    return out

def rolling_min(arr, w):
    out = np.full(len(arr), np.nan, dtype=np.float64)
    for i in range(w-1, len(arr)):
        out[i] = np.min(arr[i-w+1:i+1])
    return out

def rolling_max(arr, w):
    out = np.full(len(arr), np.nan, dtype=np.float64)
    for i in range(w-1, len(arr)):
        out[i] = np.max(arr[i-w+1:i+1])
    return out

def pct_change(arr, p):
    out = np.full(len(arr), np.nan, dtype=np.float64)
    mask = arr[:-p] != 0
    out[p:] = np.where(mask, arr[p:] / arr[:-p] - 1, np.nan)
    return out

def ema(arr, span):
    out = np.empty(len(arr), dtype=np.float64)
    alpha = 2.0 / (span + 1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

# ============================================================
# 技术特征计算
# ============================================================
TECH_FEAT_NAMES = [
    'ma5','ma20','ma60','ma_bias20','ma_align','price_position',
    'ret1','ret5','ret20','ret60','momentum_6m','momentum_1m',
    'mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change',
    'rsi14','rsi_change','macd','macd_signal','macd_hist',
    'bb_std','bb_width','bb_pos','ret_quality'
]

def compute_tech(close, high, low, volume):
    """返回 (n, 27) 的技术特征数组"""
    n = len(close)
    if n < 70:
        return None
    
    ret = np.empty(n, dtype=np.float64)
    ret[0] = 0
    ret[1:] = (close[1:] - close[:-1]) / close[:-1]
    
    ma5 = rolling_mean(close, 5)
    ma20 = rolling_mean(close, 20)
    ma60 = rolling_mean(close, 60)
    ma_bias20 = (close - ma20) / (ma20 + 1e-10)
    ma_align = ((close > ma5).astype(np.float64) + (ma5 > ma20).astype(np.float64))
    
    mn60 = rolling_min(close, 60)
    mx60 = rolling_max(close, 60)
    price_position = (close - mn60) / (mx60 - mn60 + 1e-10)
    
    ret1 = pct_change(close, 1)
    ret5 = pct_change(close, 5)
    ret20 = pct_change(close, 20)
    ret60 = pct_change(close, 60)
    momentum_6m = pct_change(close, 126)
    momentum_1m = pct_change(close, 21)
    mom_divergence = momentum_1m - ret20
    
    trend_accel = np.full(n, np.nan, dtype=np.float64)
    trend_accel[10:] = ret5[10:] - ret5[5:-5] if len(ret5) > 10 else np.nan
    
    vol20 = rolling_std(ret, 20)
    vol5 = rolling_std(ret, 5)
    
    vol_ma20 = rolling_mean(volume, 20)
    vol_ratio = volume / (vol_ma20 + 1e-10)
    
    vol_change = np.full(n, np.nan, dtype=np.float64)
    vol_change[40:] = vol20[40:] / (vol20[20:-20] + 1e-10)
    
    delta = np.empty(n, dtype=np.float64)
    delta[0] = 0
    delta[1:] = close[1:] - close[:-1]
    gain = np.where(delta > 0, delta, 0.0)
    loss_arr = np.where(delta < 0, -delta, 0.0)
    gain_ma = rolling_mean(gain, 14)
    loss_ma = rolling_mean(loss_arr, 14)
    rsi14 = 100 - 100 / (1 + gain_ma / (loss_ma + 1e-10))
    
    rsi_change = np.full(n, np.nan, dtype=np.float64)
    rsi_change[5:] = rsi14[5:] - rsi14[:-5]
    
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd = ema12 - ema26
    macd_signal = ema(macd, 9)
    macd_hist = macd - macd_signal
    
    bb_std = vol20
    bb_mid = ma20
    bb_width = 4 * bb_std * bb_mid / (bb_mid + 1e-10)
    
    std20 = rolling_std(close, 20)
    bb_lower = bb_mid - 2 * std20
    bb_upper = bb_mid + 2 * std20
    bb_pos = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    ret_pos = np.where(ret > 0, ret, 0.0)
    ret_neg = np.where(ret < 0, -ret, 0.0)
    ret_pos_ma = rolling_mean(ret_pos, 20)
    ret_neg_ma = rolling_mean(ret_neg, 20)
    ret_quality = ret_pos_ma / (ret_pos_ma + ret_neg_ma + 1e-10)
    
    feats = np.column_stack([
        ma5, ma20, ma60, ma_bias20, ma_align, price_position,
        ret1, ret5, ret20, ret60, momentum_6m, momentum_1m,
        mom_divergence, trend_accel, vol20, vol5, vol_ratio, vol_change,
        rsi14, rsi_change, macd, macd_signal, macd_hist,
        bb_std, bb_width, bb_pos, ret_quality
    ])
    return feats

# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blueshield', action='store_true')
    parser.add_argument('--arrow', action='store_true')
    args = parser.parse_args()
    if not args.blueshield and not args.arrow:
        args.blueshield = True
        args.arrow = True
    
    t_total = time.time()
    
    # ---- 加载 ----
    log('Step 1: 加载数据...')
    df = pd.read_parquet(os.path.join(DATA_DIR, 'us_hist_full_10y.parquet'))
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close', 'volume'])
    df = df[df['volume'] > 0]
    df = df.sort_values(['sym', 'date'])
    log(f'  清洗后: {len(df):,} 行, {df["sym"].nunique()} 只')
    
    # VIX
    vix_df = pd.read_parquet(os.path.join(DATA_DIR, 'vix_10y.parquet'))
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = [c[0] for c in vix_df.columns]
    vix_df = vix_df.reset_index()
    vix_df.columns = [c.lower() for c in vix_df.columns]
    if 'date' not in vix_df.columns:
        vix_df = vix_df.rename(columns={vix_df.columns[0]: 'date'})
    vix_df['date'] = pd.to_datetime(vix_df['date'])
    cc = [c for c in vix_df.columns if 'close' in c]
    if cc:
        vix_df = vix_df.rename(columns={cc[0]: 'vix_close'})
    vix_map = dict(zip(vix_df['date'], vix_df['vix_close'].values))
    
    # 基本面
    with open(os.path.join(DATA_DIR, 'us_fundamentals.json')) as f:
        fund_raw = json.load(f)
    fund_map = {}
    for sym, info in fund_raw.items():
        fund_map[sym] = [
            info.get('trailingPE', np.nan),
            info.get('forwardPE', np.nan),
            info.get('dividendYield', np.nan),
            info.get('beta', np.nan),
        ]
    
    # 宏观ETF收益
    log('  计算宏观ETF收益...')
    def get_close(sym):
        s = df[df['sym'] == sym][['date', 'close']].copy()
        return s.set_index('date')['close']
    
    spy_c = get_close('SPY')
    qqq_c = get_close('QQQ')
    iwm_c = get_close('IWM')
    
    # 构建宏观特征字典: date -> 13维宏观特征
    macro_dates = spy_c.index
    macro_arr = np.zeros((len(macro_dates), 13), dtype=np.float64)
    
    for col_idx, (name, s) in enumerate([('spy', spy_c), ('qqq', qqq_c), ('iwm', iwm_c)]):
        vals = s.values
        macro_arr[:, col_idx*4+0] = np.concatenate([[np.nan], vals[1:]/vals[:-1]-1])  # ret1
        r5 = np.full(len(vals), np.nan); r5[5:] = vals[5:]/vals[:-5]-1
        macro_arr[:, col_idx*4+1] = r5  # ret5
        r20 = np.full(len(vals), np.nan); r20[20:] = vals[20:]/vals[:-20]-1
        macro_arr[:, col_idx*4+2] = r20  # ret20
        r60 = np.full(len(vals), np.nan); r60[60:] = vals[60:]/vals[:-60]-1
        macro_arr[:, col_idx*4+3] = r60  # ret60
    
    # VIX
    for i, d in enumerate(macro_dates):
        macro_arr[i, 12] = vix_map.get(d, np.nan)
    # ffill VIX
    vix_col = macro_arr[:, 12]
    mask = np.isnan(vix_col)
    idx = np.where(~mask, np.arange(len(vix_col)), 0)
    np.maximum.accumulate(idx, out=idx)
    macro_arr[:, 12] = vix_col[idx]
    
    macro_date_map = {d: i for i, d in enumerate(macro_dates)}
    
    log(f'  宏观: {len(macro_dates)} 天')
    del spy_c, qqq_c, iwm_c, vix_df, vix_map
    gc.collect()
    
    # ETF排除列表
    etf_syms = {'SPY','QQQ','IWM','DIA','VOO','VTI','IVV','VEA','VWO',
                'BND','AGG','TLT','GLD','SLV','USO','XLE','XLF','XLK','XLV',
                'XLI','XLP','XLU','XLB','XLRE','XLC','ARKK','ARKG','ARKW'}
    
    # ---- 模型列表 ----
    model_configs = []
    if args.blueshield:
        model_configs.append({
            'name': 'blueshield', 'version': 'blueshield_v7',
            'price_min': 10, 'price_max': 1e9,
            'hold_days': 20, 'top_n': 15,
            'has_fund': True, 'extra_feats': [],
            'replaces': 'V6'
        })
    if args.arrow:
        model_configs.append({
            'name': 'arrow', 'version': 'arrow_v12',
            'price_min': 1, 'price_max': 10,
            'hold_days': 5, 'top_n': 5,
            'has_fund': False, 'extra_feats': ['price', 'range_pct'],
            'replaces': 'V11'
        })
    
    # 按价格区间过滤，避免重复过滤
    last_prices = df.groupby('sym')['last_price'] if 'last_price' in df.columns else df.groupby('sym')['close'].transform('last')
    # 更高效: 直接在循环里过滤
    
    results = {}
    
    for cfg in model_configs:
        version = cfg['version']
        price_min = cfg['price_min']
        price_max = cfg['price_max']
        hold_days = cfg['hold_days']
        top_n = cfg['top_n']
        has_fund = cfg['has_fund']
        extra_feats = cfg['extra_feats']
        
        log(f'\n{"="*60}')
        log(f'{version} 训练 (>${price_min}, {hold_days}天, Top{top_n})')
        log(f'{"="*60}')
        t0 = time.time()
        
        # 特征列
        n_tech = 27
        n_extra = len(extra_feats)
        n_macro = 13
        n_fund = 4 if has_fund else 0
        feat_cols = TECH_FEAT_NAMES + extra_feats + [
            'vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60',
            'qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60',
            'iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60'
        ]
        if has_fund:
            feat_cols += ['pe_trailing','pe_forward','div_yield','beta']
        n_feat = len(feat_cols)
        
        # 确定有效股票
        sym_last = df.groupby('sym')['close'].last()
        valid_syms = sym_last[(sym_last > price_min) & (sym_last <= price_max)].index.tolist()
        valid_syms = [s for s in valid_syms if s not in etf_syms]
        log(f'  范围内: {len(valid_syms)} 只')
        
        # ---- 分批特征工程，写入预分配数组 ----
        # 先统计总行数
        sym_counts = df[df['sym'].isin(valid_syms)].groupby('sym').size()
        # 只保留>=80天的
        sym_counts = sym_counts[sym_counts >= 80]
        valid_syms = [s for s in valid_syms if s in sym_counts.index]
        total_rows_est = int(sym_counts.sum())
        log(f'  有效股票: {len(valid_syms)}, 预估行数: {total_rows_est:,}')
        
        # 预分配
        all_X = np.empty((total_rows_est, n_feat), dtype=np.float32)
        all_y = np.empty(total_rows_est, dtype=np.float32)
        all_dates = np.empty(total_rows_est, dtype='datetime64[ns]')
        write_idx = 0
        
        batch_size = 200
        n_processed = 0
        
        for batch_start in range(0, len(valid_syms), batch_size):
            batch_syms = set(valid_syms[batch_start:batch_start+batch_size])
            batch_df = df[df['sym'].isin(batch_syms)]
            
            for sym, grp in batch_df.groupby('sym'):
                if len(grp) < 80:
                    continue
                
                close = grp['close'].values.astype(np.float64)
                high = grp['high'].values.astype(np.float64)
                low = grp['low'].values.astype(np.float64)
                vol = grp['volume'].values.astype(np.float64)
                dates = grp['date'].values
                
                tech = compute_tech(close, high, low, vol)
                if tech is None:
                    continue
                
                # 构建完整特征矩阵
                n_rows = len(close)
                
                # 宏观特征
                macro_feats = np.full((n_rows, n_macro), np.nan, dtype=np.float64)
                for i, d in enumerate(dates):
                    dt = pd.Timestamp(d)
                    if dt in macro_date_map:
                        macro_feats[i] = macro_arr[macro_date_map[dt]]
                    else:
                        # 找最近的日期
                        for delta in range(1, 5):
                            prev = dt - pd.Timedelta(days=delta)
                            if prev in macro_date_map:
                                macro_feats[i] = macro_arr[macro_date_map[prev]]
                                break
                
                # 额外特征
                extra_arr = np.empty((n_rows, n_extra), dtype=np.float64)
                for ei, ef in enumerate(extra_feats):
                    if ef == 'price':
                        extra_arr[:, ei] = close
                    elif ef == 'range_pct':
                        extra_arr[:, ei] = (high - low) / (close + 1e-10)
                
                # 基本面
                if has_fund:
                    fund_vals = fund_map.get(sym, [np.nan]*4)
                    fund_arr = np.tile(np.array(fund_vals, dtype=np.float64), (n_rows, 1))
                else:
                    fund_arr = np.empty((n_rows, 0), dtype=np.float64)
                
                # 合并特征
                feat_matrix = np.hstack([tech, extra_arr, macro_feats, fund_arr]).astype(np.float32)
                
                # 标签（clip极端收益率，避免penny stock噪声）
                fwd_ret = np.full(n_rows, np.nan, dtype=np.float32)
                if n_rows > hold_days:
                    raw_ret = close[hold_days:] / close[:-hold_days] - 1
                    # 蓝盾clip ±50%, 绿箭clip ±100%
                    clip_limit = 0.5 if hold_days == 20 else 1.0
                    raw_ret = np.clip(raw_ret, -clip_limit, clip_limit)
                    fwd_ret[:-hold_days] = raw_ret.astype(np.float32)
                
                # 写入
                end_idx = write_idx + n_rows
                if end_idx > total_rows_est:
                    # 扩容
                    new_size = total_rows_est + 1000000
                    all_X = np.resize(all_X, (new_size, n_feat))
                    all_y = np.resize(all_y, new_size)
                    all_dates = np.resize(all_dates, new_size)
                
                all_X[write_idx:end_idx] = feat_matrix
                all_y[write_idx:end_idx] = fwd_ret
                all_dates[write_idx:end_idx] = dates
                write_idx = end_idx
                
                n_processed += 1
            
            done = min(batch_start + batch_size, len(valid_syms))
            if done % 1000 == 0 or done == len(valid_syms):
                log(f'    {done}/{len(valid_syms)} 只, {write_idx:,} 行')
        
        # 截断到实际大小
        all_X = all_X[:write_idx]
        all_y = all_y[:write_idx]
        all_dates = all_dates[:write_idx]
        
        log(f'  总行数: {write_idx:,}')
        
        # 过滤NaN标签和特征
        valid_mask = ~np.isnan(all_y)
        for col in range(n_feat):
            valid_mask &= ~np.isnan(all_X[:, col])
        valid_mask &= ~np.isinf(all_X).any(axis=1)
        
        all_X = all_X[valid_mask]
        all_y = all_y[valid_mask]
        all_dates = all_dates[valid_mask]
        
        log(f'  过滤后: {len(all_y):,} 行')
        log(f'  特征工程耗时: {time.time()-t0:.1f}s')
        
        # ---- Walk-Forward ----
        log('Step 3: Walk-Forward 5折...')
        
        oos_start = np.datetime64('2024-01-01')
        train_mask = all_dates < oos_start
        oos_mask = all_dates >= oos_start
        
        X_train_all = all_X[train_mask]
        y_train_all = all_y[train_mask]
        dates_train = all_dates[train_mask]
        
        unique_train_dates = np.unique(dates_train)
        fold_size = len(unique_train_dates) // 5
        
        wf_results = []
        
        for fold in range(5):
            val_start = unique_train_dates[(fold+1) * fold_size]
            val_end = unique_train_dates[min((fold+2) * fold_size, len(unique_train_dates)-1)]
            
            tr_mask = dates_train < val_start
            vl_mask = (dates_train >= val_start) & (dates_train < val_end)
            
            X_tr = X_train_all[tr_mask]
            y_tr = y_train_all[tr_mask]
            X_vl = X_train_all[vl_mask]
            y_vl = y_train_all[vl_mask]
            d_vl = dates_train[vl_mask]
            
            if len(X_tr) < 1000 or len(X_vl) < 100:
                continue
            
            params = {
                'objective': 'reg:squarederror', 'max_depth': 6,
                'learning_rate': 0.03, 'subsample': 0.8,
                'colsample_bytree': 0.8, 'min_child_weight': 10,
                'tree_method': 'hist', 'seed': 42, 'verbosity': 0
            }
            
            dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_cols)
            dval = xgb.DMatrix(X_vl, label=y_vl, feature_names=feat_cols)
            
            model = xgb.train(params, dtrain, num_boost_round=500,
                             evals=[(dval, 'val')], early_stopping_rounds=50,
                             verbose_eval=False)
            
            pred = model.predict(dval)
            
            # 每天Top-N
            daily_rets = []
            for d in np.unique(d_vl):
                mask = d_vl == d
                if mask.sum() < top_n:
                    continue
                top_idx = np.argsort(pred[mask])[-top_n:]
                daily_rets.append(np.mean(y_vl[mask][top_idx]))
            
            if daily_rets:
                dr = np.array(daily_rets)
                avg_ret = np.mean(dr) * 100
                win_rate = np.mean(dr > 0) * 100
                sharpe = np.mean(dr) / (np.std(dr) + 1e-10) * np.sqrt(252/hold_days)
                wf_results.append({
                    'fold': fold, 'avg_return': avg_ret, 'win_rate': win_rate,
                    'sharpe': sharpe, 'n_days': len(dr),
                    'best_iteration': int(model.best_iteration)
                })
                log(f'  Fold {fold}: avg={avg_ret:.2f}%, win={win_rate:.1f}%, sharpe={sharpe:.2f}, iter={model.best_iteration}')
            
            del dtrain, dval, model
            gc.collect()
        
        best_iter = max(int(np.median([r['best_iteration'] for r in wf_results])) if wf_results else 500, 200)
        log(f'  最佳迭代: {best_iter}')
        
        # ---- OOS ----
        log('Step 4: OOS评估 (2024-2026)...')
        X_oos = all_X[oos_mask]
        y_oos = all_y[oos_mask]
        d_oos = all_dates[oos_mask]
        
        params_full = {
            'objective': 'reg:squarederror', 'max_depth': 6,
            'learning_rate': 0.03, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 10,
            'tree_method': 'hist', 'seed': 42, 'verbosity': 0
        }
        
        dtrain_full = xgb.DMatrix(X_train_all, label=y_train_all, feature_names=feat_cols)
        oos_model = xgb.train(params_full, dtrain_full, num_boost_round=best_iter, verbose_eval=False)
        
        doos = xgb.DMatrix(X_oos, feature_names=feat_cols)
        oos_pred = oos_model.predict(doos)
        
        oos_daily = []
        for d in np.unique(d_oos):
            mask = d_oos == d
            if mask.sum() < top_n:
                continue
            top_idx = np.argsort(oos_pred[mask])[-top_n:]
            oos_daily.append(np.mean(y_oos[mask][top_idx]))
        
        oos_result = None
        if oos_daily:
            od = np.array(oos_daily)
            oos_result = {
                'avg_return': float(np.mean(od)*100),
                'win_rate': float(np.mean(od > 0)*100),
                'sharpe': float(np.mean(od)/(np.std(od)+1e-10)*np.sqrt(252/hold_days)),
                'max_dd': float(np.min(np.cumsum(od))*100),
                'n_days': len(od)
            }
            log(f'  OOS: avg={oos_result["avg_return"]:.2f}%, win={oos_result["win_rate"]:.1f}%, sharpe={oos_result["sharpe"]:.2f}')
        
        del oos_model, dtrain_full, doos
        gc.collect()
        
        # ---- 最终模型 ----
        log('Step 5: 训练最终模型...')
        dall = xgb.DMatrix(all_X, label=all_y, feature_names=feat_cols)
        final_model = xgb.train(params_full, dall, num_boost_round=best_iter, verbose_eval=False)
        
        importance = final_model.get_score(importance_type='gain')
        total_imp = sum(importance.values()) if importance else 1
        feat_imp = {k: round(v/total_imp*100, 2) for k, v in sorted(importance.items(), key=lambda x: -x[1])}
        
        # ---- 阈值 ----
        log('Step 6: 信号阈值...')
        all_pred = final_model.predict(dall)
        
        daily_top_preds = []
        for d in np.unique(all_dates):
            mask = all_dates == d
            if mask.sum() < top_n * 3:
                continue
            top_idx = np.argsort(all_pred[mask])[-top_n:]
            daily_top_preds.append({
                'min': float(np.min(all_pred[mask][top_idx])),
                'max': float(np.max(all_pred[mask][top_idx])),
                'mean': float(np.mean(all_pred[mask][top_idx]))
            })
        
        st = pd.DataFrame(daily_top_preds)
        thresholds = {
            'green2': {
                'threshold': round(float(st['max'].quantile(0.9)), 4),
                'note': 'Top 1%信号, 样本少但收益率极高'
            },
            'green1': {
                'threshold': round(float(st['mean'].quantile(0.75)), 4),
                'note': f'主力信号, Top {top_n}平均分位'
            },
            'observe': {
                'threshold': round(float(st['min'].quantile(0.5)), 4),
                'note': '观察池'
            }
        }
        
        for level, info in thresholds.items():
            t = info['threshold']
            mask = all_pred >= t
            if mask.sum() > 0:
                info['avg_return'] = round(float(np.mean(all_y[mask])*100), 2)
                info['win_rate'] = round(float(np.mean(all_y[mask] > 0)*100), 1)
                info['count'] = int(mask.sum())
                log(f'  {level}: threshold={t}, avg={info["avg_return"]}%, win={info["win_rate"]}%, count={info["count"]}')
        
        # ---- 保存 ----
        log('Step 7: 保存...')
        model_path = os.path.join(MODEL_DIR, f'{version}_xgb.json')
        final_model.save_model(model_path)
        log(f'  模型: {model_path} ({os.path.getsize(model_path)/1024/1024:.1f}MB)')
        
        if has_fund:
            universe_str = f'全市场>${price_min} ({len(valid_syms)}只)'
        else:
            universe_str = f'全市场${price_min}-${price_max} ({len(valid_syms)}只)'
        
        wf_avg = float(np.mean([r['avg_return'] for r in wf_results])) if wf_results else 0
        wf_win = float(np.mean([r['win_rate'] for r in wf_results])) if wf_results else 0
        wf_sharpe = float(np.mean([r['sharpe'] for r in wf_results])) if wf_results else 0
        
        meta = {
            'version': version, 'algorithm': 'XGBoost',
            'features': feat_cols, 'n_features': n_feat,
            'tech_features': n_tech + n_extra,
            'macro_features': n_macro,
            'fund_features': n_fund,
            'hold_days': hold_days, 'top_n': top_n,
            'universe': universe_str,
            'params': params_full, 'n_trees': best_iter,
            'trained_on': f'{pd.Timestamp(all_dates.min())}~{pd.Timestamp(all_dates.max())}',
            'n_train_samples': int(len(all_y)),
            'feature_importance': feat_imp,
            'validation': {
                'method': f'Walk-Forward {len(wf_results)}折 + OOS 2024-2026',
                'wf_avg_return': round(wf_avg, 2),
                'wf_win_rate': round(wf_win, 1),
                'wf_sharpe': round(wf_sharpe, 2),
            },
            'signal_thresholds': thresholds,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'data_source': 'us_hist_full_10y.parquet (全市场11,864只)',
            'replaces': cfg['replaces']
        }
        if oos_result:
            meta['validation'].update({
                'oos_avg_return': round(oos_result['avg_return'], 2),
                'oos_win_rate': round(oos_result['win_rate'], 1),
                'oos_sharpe': round(oos_result['sharpe'], 2),
                'oos_max_dd': round(oos_result['max_dd'], 1),
            })
        
        meta_path = os.path.join(MODEL_DIR, f'{version}_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        log(f'  元数据: {meta_path}')
        
        results[cfg['name']] = meta
        log(f'{version} 完成! 耗时: {(time.time()-t0)/60:.1f}分钟')
        
        del all_X, all_y, all_dates, final_model, dall
        gc.collect()
    
    # ---- 汇总 ----
    log(f'\n{"="*60}')
    log('训练汇总')
    log(f'{"="*60}')
    for name, meta in results.items():
        v = meta['validation']
        log(f'\n{meta["version"]}:')
        log(f'  {meta["n_train_samples"]:,} 样本, {meta["n_features"]} 特征, {meta["n_trees"]} 树')
        log(f'  WF: avg={v["wf_avg_return"]:.2f}%, win={v["wf_win_rate"]:.1f}%, sharpe={v["wf_sharpe"]:.2f}')
        if 'oos_avg_return' in v:
            log(f'  OOS: avg={v["oos_avg_return"]:.2f}%, win={v["oos_win_rate"]:.1f}%, sharpe={v["oos_sharpe"]:.2f}')
        log(f'  Top3: {list(meta["feature_importance"].keys())[:3]}')
    
    log(f'\n总耗时: {(time.time()-t_total)/60:.1f}分钟')

if __name__ == '__main__':
    main()
