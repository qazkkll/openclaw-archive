"""
A1 Layer 3 — A股个股多因子评分模型（XGBoost）

数据：a_hist_10y.parquet (4581只, 2016-2026) + moneyflow_data.parquet (5761只, 2016-2026)
目标：预测每只股票未来5/10/20日涨幅，生成评分

策略：不做宏观方向预测，直接对个股做多因子评分。
和V8-Lottery一致思路，但有A股独有的资金流数据优势。

三期分离：训练2016-2020 | 验证2021-2023H1 | 测试2023H2-2026

用法：
  python scripts/a1_layer3_xgb.py                  # 完整训练
  python scripts/a1_layer3_xgb.py --quick          # 快速测试（只取500只股票）
"""

import json, os, sys, time, math
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

D_DATA = r'/home/hermes/.hermes/openclaw-archive/data'
MODEL_DIR = os.path.join(D_DATA, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 加载数据（懒加载，只读需要的部分） ───

HIST_CACHE = {}
MF_CACHE = {}
STOCK_CODE = 'ts_code'

def load_hist():
    global HIST_CACHE
    if HIST_CACHE:
        return HIST_CACHE
    t0 = time.time()
    with open(os.path.join(D_DATA, 'a_hist_10y.parquet'), 'rb') as f:
        HIST_CACHE = json.load(f)
    print(f"  历史数据: {len(HIST_CACHE)} 只股票, {time.time()-t0:.1f}s")
    return HIST_CACHE

def load_moneyflow():
    global MF_CACHE
    if MF_CACHE:
        return MF_CACHE
    t0 = time.time()
    with open(os.path.join(D_DATA, 'moneyflow_data.parquet'), 'rb') as f:
        MF_CACHE = json.load(f)
    print(f"  资金流数据: {len(MF_CACHE)} 只股票, {time.time()-t0:.1f}s")
    return MF_CACHE


# ─── 特征计算（单只股票） ───

def calc_stock_features(code, hist_rec, mf_by_date, mf_rollup):
    """
    计算单只股票的完整特征集（优化版，O(n)日期查找）。
    
    hist_rec: {c:[], h:[], l:[], o:[], v:[], dates:[]}
    mf_by_date: {date: {buy_lg_amount, ...}} — 每日资金流数据
    mf_rollup: {date: {net_mf_5d, major_net_5d, lg_net_5d, net_mf_10d, ...}} — 预计算的累加
    
    返回: [{date, features_dict, fwd_ret_10d, fwd_ret_20d}]
    """
    c = hist_rec['c']
    h = hist_rec['h']
    l = hist_rec['l']
    o = hist_rec['o']
    v = hist_rec['v']
    dates = hist_rec['dates']
    
    n = len(c)
    results = []
    
    for i in range(60, n-20):
        d = dates[i]
        price = c[i]
        
        rec = {'code': code, 'date': d, 'close': price}
        
        # ─── 技术指标 ───
        ma5 = sum(c[i-4:i+1])/5
        ma10 = sum(c[i-9:i+1])/10
        ma20 = sum(c[i-19:i+1])/20
        ma60 = sum(c[i-59:i+1])/60
        ma120 = sum(c[i-119:i+1])/120 if i >= 119 else ma60
        
        rec['pct_ma5'] = (price/ma5 - 1)*100 if ma5 > 0 else 0
        rec['pct_ma10'] = (price/ma10 - 1)*100 if ma10 > 0 else 0
        rec['pct_ma20'] = (price/ma20 - 1)*100 if ma20 > 0 else 0
        rec['pct_ma60'] = (price/ma60 - 1)*100 if ma60 > 0 else 0
        rec['pct_ma120'] = (price/ma120 - 1)*100 if ma120 > 0 else 0
        
        if i >= 25:
            ma20_before = sum(c[i-25:i-4])/20
            rec['ma20_slope'] = (ma20/ma20_before - 1)*100 if ma20_before > 0 else 0
        else:
            rec['ma20_slope'] = 0
        if i >= 65:
            ma60_before = sum(c[i-65:i-4])/60
            rec['ma60_slope'] = (ma60/ma60_before - 1)*100 if ma60_before > 0 else 0
        else:
            rec['ma60_slope'] = 0
        
        align = (ma5 > ma10) + (ma10 > ma20) + (ma20 > ma60) + (price > ma5) + (price > ma10) + (price > ma60)
        rec['ma_align'] = align
        
        rets = [abs(c[j]/c[j-1] - 1)*100 if c[j-1] > 0 else 0 for j in range(max(1,i-9), i+1)]
        rec['vol_10d'] = sum(rets)/len(rets)
        rets60 = [abs(c[j]/c[j-1] - 1)*100 if c[j-1] > 0 else 0 for j in range(max(1,i-59), i+1)]
        rec['vol_60d'] = sum(rets60)/len(rets60)
        rec['vol_ratio'] = rec['vol_10d']/rec['vol_60d'] if rec['vol_60d'] > 0 else 1
        
        trs = [max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(max(1,i-19), i+1)]
        rec['atr20_pct'] = sum(trs)/len(trs)/price*100 if price > 0 else 0
        
        rec['ret_5d'] = (price/c[i-5] - 1)*100 if i >= 5 else 0
        rec['ret_10d'] = (price/c[i-10] - 1)*100 if i >= 10 else 0
        rec['ret_20d'] = (price/c[i-20] - 1)*100 if i >= 20 else 0
        rec['ret_60d'] = (price/c[i-60] - 1)*100 if i >= 60 else 0
        
        changes = [c[j] - c[j-1] for j in range(max(1,i-13), i+1)]
        gains = sum(x for x in changes if x > 0)
        losses = sum(-x for x in changes if x < 0)
        rec['rsi14'] = 100 - 100/(1 + gains/losses/14) if losses > 0 else 100
        
        vol5 = sum(v[i-4:i+1])/5
        vol20 = sum(v[i-19:i+1])/20
        rec['vol_ratio_5_20'] = vol5/vol20 if vol20 > 0 else 1
        
        if i >= 40:
            past_rets = [(c[j]/c[j-20] - 1)*100 for j in range(20, i+1)]
            cur_ret = (price/c[i-20] - 1)*100
            rec['ret20d_pct'] = sum(1 for r in past_rets if r < cur_ret)/len(past_rets)*100
        else:
            rec['ret20d_pct'] = 50
        
        # ─── KDJ (9,3,3) ───
        if i >= 8:
            hh9 = max(h[i-8:i+1])
            ll9 = min(l[i-8:i+1])
            rsv = (price - ll9) / (hh9 - ll9) * 100 if (hh9 - ll9) > 0 else 50.0
            k_kdj = 2/3 * (50.0) + 1/3 * rsv
            d_kdj = 2/3 * (50.0) + 1/3 * k_kdj
            j_kdj = 3 * k_kdj - 2 * d_kdj
        else:
            k_kdj, d_kdj, j_kdj = 50.0, 50.0, 50.0
        rec['kdj_k'] = round(k_kdj, 2)
        rec['kdj_d'] = round(d_kdj, 2)
        rec['kdj_j'] = round(j_kdj, 2)
        
        # ─── MACD (12,26,9) ───
        ema12 = price; ema26 = price
        for j in range(min(34, i)):
            ema12 = c[i-j-1] * (2/13) + ema12 * (11/13)
            ema26 = c[i-j-1] * (2/27) + ema26 * (25/27)
        dif = ema12 - ema26
        # DEA: EMA of DIF (9 periods)
        dea = dif
        for j in range(min(8, i)):
            prev_dif = 0
            idx2 = i - j - 1
            # Approximate: use close-based re-calculation
            e12_prev = c[idx2]; e26_prev = c[idx2]
            for k in range(min(26, idx2)):
                pk = idx2 - k - 1
                e12_prev = c[pk] * (2/13) + e12_prev * (11/13)
                e26_prev = c[pk] * (2/27) + e26_prev * (25/27)
            prev_dif = e12_prev - e26_prev
            dea = prev_dif * (2/10) + dea * (8/10)
        macd_bar = 2 * (dif - dea)
        rec['macd_dif'] = round(dif, 4)
        rec['macd_dea'] = round(dea, 4)
        rec['macd_bar'] = round(macd_bar, 4)
        
        # ─── 布林带 (20,2) ───
        if i >= 19:
            ma20_bb = sum(c[i-19:i+1]) / 20
            var20_bb = sum((c[j]-ma20_bb)**2 for j in range(i-19,i+1)) / 20
            std20_bb = var20_bb ** 0.5
            upper_bb = ma20_bb + 2 * std20_bb
            lower_bb = ma20_bb - 2 * std20_bb
            bb_w = (upper_bb - lower_bb) / ma20_bb if ma20_bb > 0 else 0
            bb_p = (price - lower_bb) / (upper_bb - lower_bb) if (upper_bb - lower_bb) > 0 else 0.5
        else:
            bb_w, bb_p = 0, 0.5
        rec['bb_width'] = round(bb_w, 4)
        rec['bb_position'] = round(bb_p, 4)
        
        # ─── OBV ───
        obv_seq = [0]
        for j in range(1, i+1):
            delta = v[j] if c[j] > c[j-1] else (-v[j] if c[j] < c[j-1] else 0)
            obv_seq.append(obv_seq[-1] + delta)
        obv5 = abs(obv_seq[-1] - obv_seq[max(0, len(obv_seq)-6)])
        obv20 = abs(obv_seq[-1] - obv_seq[max(0, len(obv_seq)-21)])
        rec['obv_ratio_5_20'] = round(obv5 / obv20, 4) if obv20 > 0 else 1.0
        
        # ─── 动量补充 ───
        win5 = [c[j]/c[j-1]-1 for j in range(max(1,i-4),i+1)]
        rec['ret5_max'] = round(max(win5)*100, 2) if win5 else 0
        ret3 = price / c[i-3] - 1 if i >= 3 else 0
        ema12_hist = sum(c[i-11:i+1]) / 12 if i >= 11 else 0.01
        rec['ret3_vs_ema12'] = round((ret3*100) / (ema12_hist*100 + 1), 4) if abs(ema12_hist*100) > 0.001 else 0
        
        # ─── 资金流因子（从预计算的rollup直接取） ───
        ru = mf_rollup.get(d, {})
        rec['net_mf'] = ru.get('net_mf_1d', 0)
        rec['lg_net'] = ru.get('lg_net_1d', 0)
        rec['elg_net'] = ru.get('elg_net_1d', 0)
        rec['md_net'] = ru.get('md_net_1d', 0)
        rec['lg_pct'] = ru.get('lg_pct', 50)
        rec['elg_pct'] = ru.get('elg_pct', 25)
        rec['major_net'] = ru.get('major_net_1d', 0)
        rec['major_ratio'] = ru.get('major_ratio', 0)
        rec['net_mf_5d'] = ru.get('net_mf_5d', 0)
        rec['net_mf_10d'] = ru.get('net_mf_10d', 0)
        rec['net_mf_20d'] = ru.get('net_mf_20d', 0)
        rec['net_mf_60d'] = ru.get('net_mf_60d', 0)
        rec['major_net_5d'] = ru.get('major_net_5d', 0)
        rec['major_net_10d'] = ru.get('major_net_10d', 0)
        rec['major_net_20d'] = ru.get('major_net_20d', 0)
        rec['lg_net_5d'] = ru.get('lg_net_5d', 0)
        rec['lg_net_10d'] = ru.get('lg_net_10d', 0)
        rec['lg_net_20d'] = ru.get('lg_net_20d', 0)
        
        # ─── 目标值 ───
        rec['fwd_5d'] = (c[i+5] / price - 1) * 100 if i + 5 < n else None
        rec['fwd_10d'] = (c[i+10] / price - 1) * 100 if i + 10 < n else None
        rec['fwd_20d'] = (c[i+20] / price - 1) * 100 if i + 20 < n else None
        
        results.append(rec)
    
    return results


# ─── 主构建流程 ───

def precompute_mf_rollup(mf_records):
    """
    预计算资金流的滑动窗口累加值。
    mf_records: [{trade_date, buy_lg_amount, ...}] (按日期正序)
    返回: {date: {net_mf_5d, major_net_10d, ...}}
    """
    n = len(mf_records)
    result = {}
    
    # 预取所有数值
    dates = [r['trade_date'] for r in mf_records]
    net_mf = [(r.get('net_mf_amount', 0) or 0) for r in mf_records]
    buy_lg = [(r.get('buy_lg_amount', 0) or 0) for r in mf_records]
    sell_lg = [(r.get('sell_lg_amount', 0) or 0) for r in mf_records]
    buy_elg = [(r.get('buy_elg_amount', 0) or 0) for r in mf_records]
    sell_elg = [(r.get('sell_elg_amount', 0) or 0) for r in mf_records]
    buy_md = [(r.get('buy_md_amount', 0) or 0) for r in mf_records]
    sell_md = [(r.get('sell_md_amount', 0) or 0) for r in mf_records]
    buy_sm = [(r.get('buy_sm_amount', 0) or 0) for r in mf_records]
    sell_sm = [(r.get('sell_sm_amount', 0) or 0) for r in mf_records]
    buy_lg_elg = [buy_lg[i] + buy_elg[i] for i in range(n)]
    sell_lg_elg = [sell_lg[i] + sell_elg[i] for i in range(n)]
    
    for i in range(1, n):
        d = dates[i]
        entry = {
            'net_mf_1d': net_mf[i],
            'lg_net_1d': buy_lg[i] - sell_lg[i],
            'elg_net_1d': buy_elg[i] - sell_elg[i],
            'md_net_1d': buy_md[i] - sell_md[i],
            'major_net_1d': buy_lg_elg[i] - sell_lg_elg[i],
        }
        
        # 大单比例
        total = buy_sm[i] + sell_sm[i]
        entry['lg_pct'] = (buy_lg_elg[i]) / total * 100 if total > 0 else 50
        entry['elg_pct'] = buy_elg[i] / total * 100 if total > 0 else 25
        major = buy_lg_elg[i] + sell_lg_elg[i]
        entry['major_ratio'] = (buy_lg_elg[i] - sell_lg_elg[i]) / major * 100 if major > 0 else 0
        
        # 滑动累加（比每次sum O(k)快）
        for lookback, label in [(5, 5), (10, 10), (20, 20), (60, 60)]:
            start = max(0, i - lookback + 1)
            entry[f'net_mf_{label}d'] = sum(net_mf[start:i+1])
            entry[f'major_net_{label}d'] = sum(buy_lg_elg[j] - sell_lg_elg[j] for j in range(start, i+1))
            entry[f'lg_net_{label}d'] = sum(buy_lg[j] - sell_lg[j] for j in range(start, i+1))
        
        result[d] = entry
    
    return result


def build_feature_matrix(stocks_subset=None):
    """构建A股多因子特征矩阵"""
    t0 = time.time()
    hist = load_hist()
    mf = load_moneyflow()
    
    stock_codes = sorted(hist.keys())
    if stocks_subset:
        stock_codes = stock_codes[:stocks_subset]
    
    print(f"\n计算特征: {len(stock_codes)} 只股票")
    
    all_rows = []
    feat_names = None
    skipped = 0
    
    for idx, code in enumerate(stock_codes):
        if code not in hist:
            skipped += 1
            continue
        
        # 对齐股票代码格式
        mf_code = code + '.SZ' if code.startswith('0') or code.startswith('3') else code + '.SH'
        mf_records = mf.get(mf_code, [])
        
        if not mf_records:
            skipped += 1
            continue
        
        # 预计算资金流rollup
        mf_rollup = precompute_mf_rollup(mf_records)
        
        if not mf_rollup:
            skipped += 1
            continue
        
        results = calc_stock_features(code, hist[code], {}, mf_rollup)
        
        if results:
            fn = [k for k in results[0].keys() if k not in ('code', 'date', 'close', 'fwd_5d', 'fwd_10d', 'fwd_20d')]
            if feat_names is None:
                feat_names = fn
            
            for r in results:
                if r['fwd_20d'] is not None:
                    row = {'code': r['code'], 'date': r['date']}
                    for f in feat_names:
                        row[f] = r.get(f, 0)
                    row['fwd_5d'] = r['fwd_5d']
                    row['fwd_10d'] = r['fwd_10d']
                    row['fwd_20d'] = r['fwd_20d']
                    all_rows.append(row)
        
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(stock_codes) - idx - 1)
            print(f"  {idx+1}/{len(stock_codes)} ({len(all_rows)} rows, {elapsed:.0f}s, ETA {eta:.0f}s)")
    
    print(f"  跳过: {skipped} (无资金流数据或数据不足)")
    
    df = pd.DataFrame(all_rows)
    print(f"\n特征矩阵: {df.shape}")
    print(f"  特征: {len(feat_names) if feat_names else 0} 个")
    print(f"  时间: {time.time()-t0:.0f}s")
    return df


def train_model(df, hold_days=20, tune=False, classify=False):
    """
    训练XGBoost模型，预测未来hold_days天收益
    classify=True → binary:logistic 分类，预测涨跌概率
    classify=False → reg:squarederror 回归，预测涨跌幅
    """
    import xgboost as xgb
    
    # 标签
    target_col = f'fwd_{hold_days}d'
    
    # 三期分离
    train_df = df[df['date'] <= '20201231']
    val_df = df[(df['date'] >= '20210101') & (df['date'] <= '20230630')]
    test_df = df[df['date'] >= '20230701']
    
    feat_cols = [c for c in df.columns if c not in ('code', 'date', 'fwd_5d', 'fwd_10d', 'fwd_20d')]
    
    # 分类模式：label转二进制（涨/跌）
    if classify:
        print(f"\n⚠️  分类模式: label = (fwd_{hold_days}d > 0)")
    
    print(f"\n三期分离 ({target_col}):")
    print(f"  训练: {len(train_df)} 样本")
    print(f"  验证: {len(val_df)} 样本")
    print(f"  测试: {len(test_df)} 样本")
    print(f"  特征: {len(feat_cols)} 个")
    
    X_train = train_df[feat_cols].values.astype(np.float32)
    
    # 分类模式：label转二进制
    if classify:
        y_train_raw = train_df[target_col].values.astype(np.float32)
        y_train = (y_train_raw > 0).astype(np.float32)
        y_val_raw = val_df[target_col].values.astype(np.float32)
        y_val = (y_val_raw > 0).astype(np.float32)
        y_test_raw = test_df[target_col].values.astype(np.float32)
        y_test = (y_test_raw > 0).astype(np.float32)
    else:
        y_train = train_df[target_col].values.astype(np.float32)
        y_val = val_df[target_col].values.astype(np.float32)
        y_test = test_df[target_col].values.astype(np.float32)
    
    X_val = val_df[feat_cols].values.astype(np.float32)
    X_test = test_df[feat_cols].values.astype(np.float32)
    
    # 标签统计
    if classify:
        print(f"\n标签分布(二分类 1=涨/0=跌):")
        for name, y in [('训练', y_train), ('验证', y_val), ('测试', y_test)]:
            win_pct = y.mean()*100
            print(f"  {name}: 胜率={win_pct:.1f}%, 样本={len(y)}")
    else:
        print(f"\n标签分布:")
        for name, y in [('训练', y_train), ('验证', y_val), ('测试', y_test)]:
            pct = np.percentile(y, [5, 25, 50, 75, 95])
            print(f"  {name}: mean={y.mean():+.3f}%, median={pct[2]:+.3f}%, 5%-95%={pct[0]:+.2f}%~{pct[4]:+.2f}%")
    
    if tune:
        # 网格搜索
        from itertools import product
        param_grid = list(product([3, 5, 7], [0.03, 0.05], [3, 5]))
        best_val_r = 0
        best_params = None
        best_model = None
        
        print(f"\n网格搜索: {len(param_grid)} 组")
        for depth, lr, mw in param_grid:
            params = {
                'objective': 'reg:squarederror',
                'max_depth': depth,
                'learning_rate': lr,
                'min_child_weight': mw,
                'subsample': 0.8,
                'colsample_bytree': 0.6,
                'eval_metric': 'rmse',
                'seed': 42,
                'n_jobs': -1,
                'device': 'cuda'
            }
            
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_cols)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_cols)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=500,
                evals=[(dval, 'val')],
                early_stopping_rounds=30,
                verbose_eval=False
            )
            
            # R² on val
            pred_val = model.predict(dval)
            ss_res = np.sum((y_val - pred_val)**2)
            ss_tot = np.sum((y_val - y_val.mean())**2)
            r2 = 1 - ss_res/ss_tot
            
            if r2 > best_val_r:
                best_val_r = r2
                best_params = (depth, lr, mw)
                best_model = model
        
        print(f"最佳: depth={best_params[0]}, lr={best_params[1]}, mw={best_params[2]}, val_R²={best_val_r:.4f}")
        return best_model, feat_cols
    
    # 默认参数（深度适中的回归森林）
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 5,
        'learning_rate': 0.05,
        'min_child_weight': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'eval_metric': 'rmse',
        'seed': 42,
        'n_jobs': -1,
        'device': 'cuda'
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feat_cols)
    
    # 用更多树，配合early stopping
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # ── 评估 ──
    print(f"\n{'='*60}")
    print(f"评估结果 (预测目标: {target_col}):")
    print(f"{'Dataset':>8} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Top5_avg':>10} {'Top20_avg':>10}")
    print("-" * 55)
    
    for name, X, y in [('Train', X_train, y_train), ('Val', X_val, y_val), ('Test', X_test, y_test)]:
        md = xgb.DMatrix(X, feature_names=feat_cols)
        pred = model.predict(md)
        
        rmse = np.sqrt(np.mean((pred - y)**2))
        mae = np.mean(np.abs(pred - y))
        ss_res = np.sum((y - pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        
        # TopN平均：选出pred最高的N只，看实际收益
        top_n = min(5, len(pred))
        top_idx = np.argsort(pred)[-top_n:]
        top5_avg = y[top_idx].mean()
        
        top20_idx = np.argsort(pred)[-min(20, len(pred)):]
        top20_avg = y[top20_idx].mean()
        
        print(f"{name:>8} {rmse:>8.3f} {mae:>8.3f} {r2:>8.4f} {top5_avg:>+10.2f}% {top20_avg:>+10.2f}%")
    
    # ── 前十特征 ──
    imp = model.get_score(importance_type='gain')
    imp_sorted = sorted(imp.items(), key=lambda x: -x[1])
    print(f"\n特征重要性 (gain):")
    total = sum(imp_sorted[i][1] for i in range(min(15, len(imp_sorted))))
    for name, gain in imp_sorted[:15]:
        print(f"  {name:<20}: {gain:>10.1f} ({gain/total*100:.1f}%)")
    
    # ── 测试集分位表现 ──
    pred_test = model.predict(dtest)
    
    # 按预测值分组
    pred_order = np.argsort(pred_test)
    n = len(pred_test)
    
    print(f"\n测试集分位表现:")
    print(f"{'Group':>8} {'Avg_pred':>10} {'Avg_actual':>12} {'Win_rate':>8} {'Median':>8}")
    print("-" * 50)
    
    for g in range(5):
        start = g * n // 5
        end = (g + 1) * n // 5
        idx = pred_order[start:end]
        avg_pred = pred_test[idx].mean()
        avg_actual = y_test[idx].mean()
        win = (y_test[idx] > 0).mean() * 100
        median = np.median(y_test[idx])
        label = f'Q{g+1}'
        if g == 0: label = '(Low)'
        if g == 4: label = '(High)'
        print(f"{label:>8} {avg_pred:>+9.3f}% {avg_actual:>+11.2f}% {win:>7.1f}% {median:>+7.2f}%")
    
    # 彩票视角：预测涨幅最高的5%的实际表现
    top_5pct = int(n * 0.05)
    idx_top = pred_order[-top_5pct:]
    print(f"\n彩票视角 (top 5% 预测涨幅):")
    print(f"  实际平均: {y_test[idx_top].mean():+.2f}%")
    print(f"  胜率: {(y_test[idx_top] > 0).mean()*100:.1f}%")
    print(f"  max: {y_test[idx_top].max():+.2f}%")
    
    return model, feat_cols


def save_model(model, feat_cols, hold_days, test_metrics):
    """保存模型+元数据"""
    model_path = os.path.join(MODEL_DIR, f'a1_layer3_xgb_{hold_days}d.json')
    model.save_model(model_path)
    
    meta = {
        'model': f'a1_layer3_xgb_{hold_days}d',
        'target': f'fwd_{hold_days}d',
        'features': feat_cols,
        'n_features': len(feat_cols),
        'hold_days': hold_days,
        'test_metrics': test_metrics,
        'data': 'a_hist_10y + moneyflow_data',
        'data_range': '2016-2026',
        'date_cut': {'train': '<=20201231', 'val': '20210101-20230630', 'test': '>=20230701'},
        'generated': '2026-06-12'
    }
    
    meta_path = os.path.join(MODEL_DIR, f'a1_layer3_xgb_{hold_days}d_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print(f"\n模型保存: {model_path}")
    print(f"元数据: {meta_path}")
    return model_path


# ─── 主程序 ───

def main():
    quick = '--quick' in sys.argv
    tune = '--tune' in sys.argv
    hold_days = 10  # 预测10日收益（A股短线，与V8的5日不同）
    
    # 解析--hold参数
    for arg in sys.argv:
        if arg.startswith('--hold='):
            hold_days = int(arg.split('=')[1])
    
    print("=" * 60)
    print(f"A1 Layer 3 — A股个股多因子评分模型")
    print(f"  预测: 未来{hold_days}日涨幅")
    print(f"  数据: a_hist_10y (OHLCV) + moneyflow_data (资金流)")
    print(f"  模式: {'快速(500只)' if quick else '全量'}")
    if tune:
        print(f"  模式: 网格搜索调参")
    print("=" * 60)
    
    t0 = time.time()
    
    # 构建特征矩阵
    subset = 500 if quick else None
    df = build_feature_matrix(stocks_subset=subset)
    
    if len(df) == 0:
        print("❌ 无数据，退出")
        return
    
    # 训练
    model, feat_cols = train_model(df, hold_days=hold_days, tune=tune)
    
    # 保存
    metrics = {}  # 评估结果已在train中打印
    save_model(model, feat_cols, hold_days, metrics)
    
    print(f"\n总耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
