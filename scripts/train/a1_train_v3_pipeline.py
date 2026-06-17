#!/usr/bin/env python3
"""
A3_v3 — 资金流因子增强模型训练
============================
基于A1训练管线架构 + 资金流因子拼接

特征集:
- 33个技术特征 (同A3_v1)
- 15个资金流因子 (从a1_factor_moneyflow.py输出)
- 总计: 48个特征

数据分割:
- Train: 2016-01 ~ 2021-12-31 (6年)
- Val: 2022-01-01 ~ 2023-12-31 (2年)
- Test: 2024-01-01 ~ 2026-06 (held out)

对比基线: A3_v1 (纯33技术特征)
目的: 验证资金流因子是否提升选股能力
"""
import json, sys, os, time, gc, math, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict
from datetime import datetime
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ═══ Config ═══
D_DATA = r'/home/hermes/.hermes/openclaw-archive/data'
MODEL_DIR = os.path.join(D_DATA, 'a1_models')
A1_FACTOR_DIR = os.path.join(D_DATA, 'a1_factors')
os.makedirs(MODEL_DIR, exist_ok=True)

# Date splits (strict!)
TRAIN_END = '2021-12-31'
VAL_END = '2023-12-31'

QUICK = '--quick' in sys.argv
DO_GRID = '--grid' in sys.argv

MIN_CANDLES = 250

# XGBoost params
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'seed': 42,
}
NUM_BOOST = 500
EARLY_STOP = 30

HYPER_PARAM_GRID = {
    'max_depth': [4, 5, 6],
    'learning_rate': [0.03, 0.05, 0.08],
    'subsample': [0.7, 0.8],
    'min_child_weight': [2, 3, 5],
}

# ═══ Feature Definitions ═══
TECH_FEATURES = [
    'pct_ma5', 'pct_ma10', 'pct_ma20', 'pct_ma60', 'pct_ma120',
    'ma20_slope', 'ma60_slope',
    'ma_align',
    'vol_10d', 'vol_60d', 'vol_ratio',
    'atr20_pct',
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d',
    'rsi14',
    'vol_ratio_5_20',
    'kdj_k', 'kdj_d', 'kdj_j',
    'macd_dif', 'macd_dea', 'macd_bar',
    'bb_width', 'bb_position',
    'obv_ratio_5_20',
    'ret5_max', 'ret3_vs_ema12',
    'accel_5_10',
    'ma5_ma10_cross',
    'vol_breakout',
]

MF_FACTORS = [
    'mf_net_ma5', 'mf_net_ma10', 'mf_net_ma20',
    'elg_ratio_ma5', 'elg_ratio_ma10', 'elg_ratio_ma20',
    'panic_ma5', 'panic_ma10', 'panic_ma20',
    'accel_3v3', 'accel_3v7', 'accel_5v20',
    'scissor_ma5',
    'consecutive_inflow',
    'mf_intensity_ma5',
]

ALL_FEATURES = TECH_FEATURES + MF_FACTORS

print(f"特征总计: {len(ALL_FEATURES)}个 (技术{len(TECH_FEATURES)} + 资金流{len(MF_FACTORS)})")


def _ema(arr, period):
    if len(arr) < 1:
        return []
    multiplier = 2 / (period + 1)
    result = [arr[0]]
    for val in arr[1:]:
        result.append((val - result[-1]) * multiplier + result[-1])
    return result


def compute_tech_features(c, h, l, o, v, dates, start_idx=120):
    """Compute 33 technical features for one stock."""
    n = len(c)
    results = []
    
    for i in range(start_idx, n - 10):
        price = c[i]
        rec = {}
        
        # ── MA features ──
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
        
        # ── MA slope ──
        ma20_prev = np.mean(c[i-24:i-4])
        ma60_prev = np.mean(c[i-64:i-4])
        rec['ma20_slope'] = (ma20/ma20_prev - 1)*100 if ma20_prev > 0 else 0
        rec['ma60_slope'] = (ma60/ma60_prev - 1)*100 if ma60_prev > 0 else 0
        
        # ── MA alignment ──
        rec['ma_align'] = (ma5/ma60 - 1)*100 if ma60 > 0 else 0
        
        # ── Volume ──
        rec['vol_10d'] = np.mean(v[i-9:i+1])
        rec['vol_60d'] = np.mean(v[i-59:i+1])
        rec['vol_ratio'] = rec['vol_10d'] / rec['vol_60d'] if rec['vol_60d'] > 0 else 1.0
        
        # ── ATR ──
        tr = [max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(i-19, i+1)]
        rec['atr20_pct'] = np.mean(tr) / price * 100 if price > 0 else 0
        
        # ── Returns ──
        rec['ret_1d']  = (c[i]/c[i-1]-1)*100 if i >= 1 else 0
        rec['ret_5d']  = (c[i]/c[i-4]-1)*100 if i >= 4 else 0
        rec['ret_10d'] = (c[i]/c[i-9]-1)*100 if i >= 9 else 0
        rec['ret_20d'] = (c[i]/c[i-19]-1)*100 if i >= 19 else 0
        rec['ret_60d'] = (c[i]/c[i-59]-1)*100 if i >= 59 else 0
        
        # ── RSI(14) ──
        gains = [max(c[j]-c[j-1], 0) for j in range(i-13, i+1)]
        losses = [max(c[j-1]-c[j], 0) for j in range(i-13, i+1)]
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        rec['rsi14'] = 100 - 100 / (1 + avg_gain/avg_loss) if avg_loss > 0 else 100
        
        # ── Volume ratio (5d vs 20d) ──
        vol5 = np.mean(v[i-4:i+1])
        vol20 = np.mean(v[i-19:i+1])
        rec['vol_ratio_5_20'] = vol5 / vol20 if vol20 > 0 else 1.0
        
        # ── KDJ (stateful) ──
        low9 = min(l[i-8:i+1])
        high9 = max(h[i-8:i+1])
        rsv = (c[i] - low9) / (high9 - low9) * 100 if high9 > low9 else 50
        if i == start_idx:
            k_val = d_val = 50.0
        else:
            prev_rec = results[-1][1]
            prev_k = prev_rec.get('kdj_k', 50.0)
            prev_d = prev_rec.get('kdj_d', 50.0)
            k_val = 2/3 * prev_k + 1/3 * rsv
            d_val = 2/3 * prev_d + 1/3 * k_val
        rec['kdj_k'] = round(k_val, 2)
        rec['kdj_d'] = round(d_val, 2)
        rec['kdj_j'] = round(3 * k_val - 2 * d_val, 2)
        
        # ── MACD ──
        ema12_arr = _ema(c[:i+1], 12)
        ema26_arr = _ema(c[:i+1], 26)
        ema12 = ema12_arr[-1]; ema26 = ema26_arr[-1]
        dif = ema12 - ema26
        dea_start = max(0, len(ema12_arr) - 9)
        dea_vals = [ema12_arr[j] - ema26_arr[j] for j in range(dea_start, len(ema12_arr))]
        dea_arr = _ema(dea_vals, 9)
        dea = dea_arr[-1] if len(dea_arr) > 0 else dif
        rec['macd_dif'] = round(dif, 4)
        rec['macd_dea'] = round(dea, 4)
        rec['macd_bar'] = round((dif - dea)*2, 4)
        
        # ── Bollinger ──
        std20 = np.std(c[i-19:i+1])
        rec['bb_width'] = std20 / ma20 * 100 if ma20 > 0 else 0
        rec['bb_position'] = (price - (ma20 - 2*std20)) / (4*std20) * 100 if std20 > 0 else 50
        
        # ── OBV ──
        obv_vals = []; obv = 0
        for j in range(i-19, i+1):
            if j > 0:
                if c[j] > c[j-1]: obv += v[j]
                elif c[j] < c[j-1]: obv -= v[j]
            obv_vals.append(obv)
        obv5 = np.mean(obv_vals[-5:]) if len(obv_vals) >= 5 else 0
        obv20 = np.mean(obv_vals) if obv_vals else 0
        rec['obv_ratio_5_20'] = obv5 / obv20 if obv20 > 0 else 1.0
        
        # ── Others ──
        rec['ret5_max'] = (max(h[i-4:i+1]) / price - 1) * 100
        ema12_val = _ema(c[:i+1], 12)[-1]
        rec['ret3_vs_ema12'] = (c[i]/ema12_val - 1)*100 if ema12_val > 0 else 0
        rec['accel_5_10'] = rec['ret_5d'] - rec['ret_10d']
        rec['ma5_ma10_cross'] = ma5 / ma10 - 1 if ma10 > 0 else 0
        vol_avg = np.mean(v[i-39:i+1])
        rec['vol_breakout'] = v[i] / vol_avg if vol_avg > 0 else 1.0
        
        # Target: 10-day forward return
        fwd_10d = (c[i+10] / price - 1) * 100
        
        # Date string (YYYY-MM-DD)
        date_str = dates[i] if isinstance(dates[i], str) else str(dates[i])
        
        results.append((date_str, rec, fwd_10d))
    
    return results


def main():
    t0 = time.time()
    print("="*65)
    print("A3_v3 — 资金流因子增强模型训练")
    print(f"模式: {'快速' if QUICK else '全量'}")
    print(f"特征: {len(ALL_FEATURES)}个 (技术{len(TECH_FEATURES)} + 资金流{len(MF_FACTORS)})")
    print("="*65)
    
    # ── 1. Load data ──
    print(f"\n[1/6] 加载数据...")
    hist_path = os.path.join(D_DATA, 'a_hist_10y.parquet')
    print(f"  正在加载 {hist_path}...")
    sys.stdout.flush()
    t_json = time.time()
    with open(hist_path, 'r', encoding='utf-8') as f:
        hist = json.load(f)
    print(f"  JSON加载完成: {time.time()-t_json:.1f}s")
    print(f"  K线: {len(hist)} 只股票")
    
    codes_zb = sorted([c for c in hist if c.startswith(('6','0'))])
    if QUICK:
        codes_zb = codes_zb[:500]
    print(f"  主板: {len(codes_zb)} 只")
    
    # ── 2. Load moneyflow factors ──
    print(f"\n[2/6] 加载资金流因子...")
    mf_path = os.path.join(A1_FACTOR_DIR, 'moneyflow_factors.parquet')
    t_mf = time.time()
    
    if os.path.exists(mf_path):
        # Read row groups iteratively to avoid MemoryError (file is ~1GB)
        import pyarrow.parquet as pq
        mf_cols = ['ts_code', 'trade_date'] + MF_FACTORS
        print(f"  读取列: {len(mf_cols)}个...")
        
        pf = pq.ParquetFile(mf_path)
        print(f"  Row groups: {pf.num_row_groups}")
        
        chunks = []
        for i in range(pf.num_row_groups):
            rg = pf.read_row_group(i, columns=mf_cols)
            rg_df = rg.to_pandas()
            rg_df['trade_date'] = pd.to_datetime(rg_df['trade_date']).dt.strftime('%Y-%m-%d')
            rg_df['ts_code'] = rg_df['ts_code'].astype(str)
            # Downcast to float32 to save memory
            for col in MF_FACTORS:
                if col in rg_df.columns:
                    rg_df[col] = rg_df[col].astype('float32')
            chunks.append(rg_df)
            del rg, rg_df
            gc.collect()
        
        mf_df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
        
        mf_df = mf_df.set_index(['ts_code', 'trade_date'])
        print(f"  资金流因子: {len(mf_df):,} 行, {time.time()-t_mf:.1f}s")
    else:
        print(f"  ❌ 资金流因子文件未找到: {mf_path}")
        print(f"  请先运行 a1_factor_moneyflow.py")
        return
    
    # ── 3. Compute features (tech + moneyflow join) ──
    print(f"\n[3/6] 计算特征（技术因子+资金流因子拼接）...")
    t1 = time.time()
    
    all_records = []  # list of (date, code, tech_feat_vec, mf_feat_vec, target)
    total_candles = 0
    skipped = 0
    mf_matched = 0
    mf_missed = 0
    
    for ci, code in enumerate(codes_zb):
        rec = hist[code]
        c_arr = rec.get('c', [])
        h_arr = rec.get('h', [])
        l_arr = rec.get('l', [])
        o_arr = rec.get('o', [])
        v_arr = rec.get('v', [])
        d_arr = rec.get('dates', [])
        
        if len(c_arr) < MIN_CANDLES:
            skipped += 1
            continue
        
        # Compute tech features
        fwd_results = compute_tech_features(c_arr, h_arr, l_arr, o_arr, v_arr, d_arr)
        
        # For each sample, look up moneyflow factors
        for date_str, feat_dict, target in fwd_results:
            # Get moneyflow factors
            try:
                mf_row = mf_df.loc[(code, date_str)]
                mf_vec = [float(mf_row.get(f, 0)) for f in MF_FACTORS]
                mf_matched += 1
            except KeyError:
                mf_vec = [0.0] * len(MF_FACTORS)
                mf_missed += 1
            
            tech_vec = [feat_dict.get(k, 0) for k in TECH_FEATURES]
            all_records.append((date_str, code, tech_vec, mf_vec, target))
        
        total_candles += len(c_arr)
        
        if (ci+1) % 200 == 0:
            cur_sz = len(all_records)
            mem = __import__('psutil').Process().memory_info().rss / 1024 / 1024
            print(f"  [{ci+1}/{len(codes_zb)}] {cur_sz:,} 样本, RSS {mem:.0f}MB, "
                  f"资金流匹配率 {mf_matched/(mf_matched+mf_missed+1)*100:.1f}%")
    
    print(f"\n  完成: {len(all_records):,} 样本, {skipped} 跳过")
    print(f"  资金流匹配: {mf_matched} 匹配, {mf_missed} 缺失 ({(mf_missed/(mf_matched+mf_missed)*100) if (mf_matched+mf_missed)>0 else 0:.1f}%)")
    print(f"  特征计算时间: {time.time()-t1:.1f}s")
    
    # ── 4. Split into train/val/test ──
    print(f"\n[4/6] 时间分层...")
    t2 = time.time()
    
    train_feats = []; train_targets = []
    val_feats = []; val_targets = []
    test_feats = []; test_targets = []
    
    for date_str, code, tech_vec, mf_vec, target in all_records:
        full_vec = tech_vec + mf_vec  # 拼接
        if date_str <= TRAIN_END:
            train_feats.append(full_vec)
            train_targets.append(target)
        elif date_str <= VAL_END:
            val_feats.append(full_vec)
            val_targets.append(target)
        else:
            test_feats.append(full_vec)
            test_targets.append(target)
    
    print(f"  训练: {len(train_feats):,} 样本 ({TRAIN_END}前)")
    print(f"  验证: {len(val_feats):,} 样本 ({TRAIN_END} ~ {VAL_END})")
    print(f"  测试: {len(test_feats):,} 样本 (HELD OUT)")
    
    X_train = np.array(train_feats, dtype=np.float32)
    y_train = np.array(train_targets, dtype=np.float32)
    X_val = np.array(val_feats, dtype=np.float32)
    y_val = np.array(val_targets, dtype=np.float32)
    
    print(f"  特征矩阵: {X_train.shape[1]} 维")
    print(f"  时间分层: {time.time()-t2:.1f}s")
    
    # ── 5. Train XGBoost ──
    print(f"\n[5/6] 训练XGBoost...")
    t3 = time.time()
    
    if DO_GRID:
        from itertools import product
        keys = list(HYPER_PARAM_GRID.keys())
        combos = list(product(*HYPER_PARAM_GRID.values()))
        print(f"  扫描 {len(combos)} 组超参...")
        
        best_rmse = float('inf')
        best_model = None
        best_params = None
        best_round = 0
        
        for idx, combo in enumerate(combos):
            params = XGB_PARAMS.copy()
            params.update(dict(zip(keys, combo)))
            
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=ALL_FEATURES)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=ALL_FEATURES)
            
            model = xgb.train(
                params, dtrain, num_boost_round=NUM_BOOST,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=EARLY_STOP,
                verbose_eval=False
            )
            
            val_pred = model.predict(dval)
            rmse = math.sqrt(np.mean((val_pred - y_val) ** 2))
            print(f"  [{idx+1}/{len(combos)}] {dict(zip(keys,combo))} val_rmse={rmse:.4f}, rounds={model.best_iteration}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_params = params
                best_round = model.best_iteration
        
        model = best_model
    else:
        params = XGB_PARAMS.copy()
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=ALL_FEATURES)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=ALL_FEATURES)
        
        model = xgb.train(
            params, dtrain, num_boost_round=NUM_BOOST,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=100
        )
        best_round = model.best_iteration
        best_params = params
    
    print(f"  最佳参数: {best_params}")
    print(f"  最佳轮数: {best_round}")
    print(f"  训练时间: {time.time()-t3:.1f}s")
    
    # ── 6. Evaluate ──
    print(f"\n[6/6] 评估...")
    
    dtest = xgb.DMatrix(np.array(test_feats, dtype=np.float32), feature_names=ALL_FEATURES)
    test_pred = model.predict(dtest)
    
    r2_train = 1 - np.var(y_train - model.predict(dtrain)) / np.var(y_train)
    r2_val = 1 - np.var(y_val - model.predict(dval)) / np.var(y_val)
    r2_test = 1 - np.var(np.array(test_targets) - test_pred) / np.var(test_targets)
    
    print(f"\n  📊 R²:")
    print(f"     Train: {r2_train:.4f}")
    print(f"     Val:   {r2_val:.4f}")
    print(f"     Test:  {r2_test:.4f}")
    
    # Feature importance
    importance = model.get_score(importance_type='weight')
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  📊 特征重要性 (Top 20):")
    for feat, imp in sorted_imp[:20]:
        marker = " 🔹" if feat in MF_FACTORS else ""
        print(f"     {feat:25s}: {imp:5d}{marker}")
    
    # Check: how many moneyflow factors in top 20?
    mf_in_top20 = sum(1 for f, _ in sorted_imp[:20] if f in MF_FACTORS)
    print(f"\n  资金流因子在Top20中: {mf_in_top20}/{len(MF_FACTORS)}")
    
    # ── Save model ──
    model_path = os.path.join(MODEL_DIR, 'a3_v3_mf_model.json')
    model.save_model(model_path)
    print(f"\n  💾 模型已保存: {model_path}")
    
    meta = {
        'version': 'a3_v3_mf',
        'features': ALL_FEATURES,
        'tech_features': TECH_FEATURES,
        'mf_factors': MF_FACTORS,
        'best_params': {k: str(v) if isinstance(v, (np.integer, np.floating)) else v 
                       for k, v in best_params.items()},
        'best_round': int(best_round) if hasattr(best_round, '__int__') else best_round,
        'r2_train': round(float(r2_train), 4),
        'r2_val': round(float(r2_val), 4),
        'r2_test': round(float(r2_test), 4),
        'n_train': len(train_feats),
        'n_val': len(val_feats),
        'n_test': len(test_feats),
        'mf_match_rate': mf_matched / (mf_matched + mf_missed) if (mf_matched + mf_missed) > 0 else 0,
        'timestamp': datetime.now().isoformat(),
    }
    meta_path = os.path.join(MODEL_DIR, 'a3_v3_mf_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    print(f"  📄 元数据已保存: {meta_path}")
    
    # ── Compare with A3_v1 (pure tech) ──
    print(f"\n{'='*65}")
    print(f"  📊 A3_v3 vs A3_v1 对比")
    print(f"{'='*65}")
    
    # Check if A3_v1 model exists
    old_model_path = os.path.join(MODEL_DIR, 'a3_v1.json')
    if os.path.exists(old_model_path):
        old_b = xgb.Booster()
        old_b.load_model(old_model_path)
        old_pred = old_b.predict(dtest)
        old_r2_test = 1 - np.var(np.array(test_targets) - old_pred) / np.var(test_targets)
        
        test_targets_arr = np.array(test_targets)
        new_order = np.argsort(-test_pred)
        old_order = np.argsort(-old_pred)
        
        print(f"\n  {'指标':^25} {'A3_v3(技术+资金流)':^18} {'A3_v1(纯技术)':^14}")
        print(f"  {'-'*57}")
        print(f"  {'Test R²':>25} {r2_test:>+12.4f} {old_r2_test:>+14.4f}")
        
        top10 = len(test_targets_arr) // 10
        new_top_ret = np.mean(test_targets_arr[new_order[:top10]])
        old_top_ret = np.mean(test_targets_arr[old_order[:top10]])
        print(f"  {'Top10% avg ret':>25} {new_top_ret:>+12.2f}% {old_top_ret:>+14.2f}%")
        
        new_bot_ret = np.mean(test_targets_arr[new_order[-top10:]])
        old_bot_ret = np.mean(test_targets_arr[old_order[-top10:]])
        print(f"  {'Bottom10% avg ret':>25} {new_bot_ret:>+12.2f}% {old_bot_ret:>+14.2f}%")
        
        spread_new = new_top_ret - new_bot_ret
        spread_old = old_top_ret - old_bot_ret
        print(f"  {'Spread(T-B)':>25} {spread_new:>+12.2f}% {spread_old:>+14.2f}%")
        
        pct_improve = (spread_new / spread_old - 1) * 100 if spread_old != 0 else 0
        print(f"  {'提升幅度':>25} {'':>12} {pct_improve:>+13.1f}%")
        
        meta['compare_v1_r2'] = round(float(old_r2_test), 4)
        meta['compare_v1_top10'] = round(float(old_top_ret), 2)
        meta['compare_v3_top10'] = round(float(new_top_ret), 2)
        meta['improvement_pct'] = round(float(pct_improve), 1)
        
        # Re-save meta with comparison
        with open(meta_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    else:
        print(f"\n  A3_v1模型未找到，跳过对比")
        print(f"  路径: {old_model_path}")
    
    total_time = time.time() - t0
    print(f"\n{'='*65}")
    print(f"全部完成! 用时 {total_time:.0f}s ({total_time/60:.1f}分)")
    print(f"{'='*65}")
    
    return meta


if __name__ == '__main__':
    main()
