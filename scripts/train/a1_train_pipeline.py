#!/usr/bin/env python3
"""
A1 模型重建 — Phase 2: 特征计算 + 模型训练管线
============================================
设计原则：
  - 纯技术面基线（不用资金流，后续独立实验）
  - 严格Train/Val/Test分离 (2016-2021 / 2022-2023 / 2024-2026)
  - 10年K线数据 (a_hist_10y.parquet, 2016-01 ~ 2026-06)
  - XGBoost回归，目标=10日涨幅
  - 系统特征工程（30+技术指标 + 相关性筛选）
  - 超参网格搜索在验证集上选参
  - 最终在Held-Out测试集上报告

用法: python scripts/a1_train_pipeline.py [--quick]
"""
import json, sys, os, time, gc, math, random
import numpy as np
import xgboost as xgb
from collections import defaultdict
from datetime import datetime, timedelta
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ═══ Config ═══
D_DATA = r'/home/hermes/.hermes/openclaw-archive/data'
MODEL_DIR = os.path.join(D_DATA, 'a1_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Date splits (strict! test set NEVER touched during training)
TRAIN_END = '2021-12-31'
VAL_END   = '2023-12-31'
# TEST: 2024-01-01 ~ 2026-06-12 (held out)

# Candidate pool: liquid main board stocks
MIN_CANDLES = 250  # need ~1 year of warmup

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

QUICK = '--quick' in sys.argv

# ═══ Feature Definitions ═══
TECH_FEATURES = [
    # MA deviation
    'pct_ma5', 'pct_ma10', 'pct_ma20', 'pct_ma60', 'pct_ma120',
    # MA slope
    'ma20_slope', 'ma60_slope',
    # MA alignment (bull/bear spread)
    'ma_align',
    # Volume
    'vol_10d', 'vol_60d', 'vol_ratio',
    # Volatility
    'atr20_pct',
    # Returns
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d',
    # RSI
    'rsi14',
    # Volume ratio
    'vol_ratio_5_20',
    # KDJ
    'kdj_k', 'kdj_d', 'kdj_j',
    # MACD
    'macd_dif', 'macd_dea', 'macd_bar',
    # Bollinger
    'bb_width', 'bb_position',
    # OBV
    'obv_ratio_5_20',
    # Momentum
    'ret5_max', 'ret3_vs_ema12',
    # Price acceleration
    'accel_5_10',        # ret_5d vs ret_10d acceleration
    'ma5_ma10_cross',    # MA5/MA10 cross signal
    'vol_breakout',      # volume > 2x recent avg
]

def compute_features(c, h, l, o, v, dates, start_idx=120):
    """Compute technical features for one stock.
    Returns list of (date_str, feature_dict, target_10d_return)
    """
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
        
        # ── MA slope (rate of change of MA) ──
        ma20_prev = np.mean(c[i-24:i-4])
        ma60_prev = np.mean(c[i-64:i-4])
        rec['ma20_slope'] = (ma20/ma20_prev - 1)*100 if ma20_prev > 0 else 0
        rec['ma60_slope'] = (ma60/ma60_prev - 1)*100 if ma60_prev > 0 else 0
        
        # ── MA alignment ──
        # Positive when short MA > long MA (bullish), negative otherwise
        rec['ma_align'] = (ma5/ma60 - 1)*100 if ma60 > 0 else 0
        
        # ── Volume ──
        rec['vol_10d'] = np.mean(v[i-9:i+1])
        rec['vol_60d'] = np.mean(v[i-59:i+1])
        rec['vol_ratio'] = rec['vol_10d'] / rec['vol_60d'] if rec['vol_60d'] > 0 else 1.0
        
        # ── ATR (volatility) ──
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
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rec['rsi14'] = 100 - 100 / (1 + rs)
        else:
            rec['rsi14'] = 100
        
        # ── Volume ratio (5d vs 20d) ──
        vol5 = np.mean(v[i-4:i+1])
        vol20 = np.mean(v[i-19:i+1])
        rec['vol_ratio_5_20'] = vol5 / vol20 if vol20 > 0 else 1.0
        
        # ── KDJ (stateful) ──
        low9 = min(l[i-8:i+1])
        high9 = max(h[i-8:i+1])
        if high9 > low9:
            rsv = (c[i] - low9) / (high9 - low9) * 100
        else:
            rsv = 50
        if i == start_idx:
            k_val = d_val = 50.0
        else:
            # Use previous computed K/D values stored in last result
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
        ema12 = ema12_arr[-1]
        ema26 = ema26_arr[-1]
        dif = ema12 - ema26
        # DEA = 9-day EMA of DIF
        dea_start = max(0, len(ema12_arr) - 9)
        dea_vals = [ema12_arr[j] - ema26_arr[j] for j in range(dea_start, len(ema12_arr))]
        dea_arr = _ema(dea_vals, 9)
        dea = dea_arr[-1] if len(dea_arr) > 0 else dif
        rec['macd_dif'] = round(dif, 4)
        rec['macd_dea'] = round(dea, 4)
        rec['macd_bar'] = round((dif - dea)*2, 4)
        
        # ── Bollinger ──
        ma = ma20
        std20 = np.std(c[i-19:i+1])
        rec['bb_width'] = std20 / ma * 100 if ma > 0 else 0
        rec['bb_position'] = (price - (ma - 2*std20)) / (4*std20) * 100 if std20 > 0 else 50
        
        # ── OBV ──
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
        
        # ── 5-day max ──
        rec['ret5_max'] = (max(h[i-4:i+1]) / price - 1) * 100
        
        # ── 3d vs ema12 ──
        ema12_val = _ema(c[:i+1], 12)[-1]
        rec['ret3_vs_ema12'] = (c[i]/ema12_val - 1)*100 if ema12_val > 0 else 0
        
        # ── Acceleration ──
        rec['accel_5_10'] = rec['ret_5d'] - rec['ret_10d']
        
        # ── MA Cross ──
        rec['ma5_ma10_cross'] = ma5 / ma10 - 1 if ma10 > 0 else 0
        
        # ── Volume breakout ──
        vol_avg = np.mean(v[i-39:i+1])
        rec['vol_breakout'] = v[i] / vol_avg if vol_avg > 0 else 1.0
        
        # ── Target: 10-day forward return ──
        fwd_10d = (c[i+10] / price - 1) * 100
        
        results.append((dates[i], rec, fwd_10d))
    
    return results


def _ema(arr, period):
    """Simple EMA calculation - returns list of same length as arr"""
    if len(arr) < 1:
        return []
    multiplier = 2 / (period + 1)
    result = [arr[0]]
    for val in arr[1:]:
        result.append((val - result[-1]) * multiplier + result[-1])
    return result


# ═══ Main Pipeline ═══
def main():
    t0 = time.time()
    print("="*65)
    print("A1 模型重建 — 纯技术面管线")
    print(f"模式: {'快速' if QUICK else '全量'}")
    print("="*65)
    
    # 1. Load data
    print(f"\n[1/5] 加载数据...")
    hist_path = os.path.join(D_DATA, 'a_hist_10y.parquet')
    with open(hist_path, 'r', encoding='utf-8') as f:
        hist = json.load(f)
    print(f"  K线: {len(hist)} 只股票")
    
    # Filter main board
    codes_zb = sorted([c for c in hist if c.startswith(('6','0'))])
    if QUICK:
        # Take a diverse sample for quick testing
        codes_zb = codes_zb[:500]
    print(f"  主板: {len(codes_zb)} 只")
    print(f"  总数据加载: {time.time()-t0:.1f}s")
    
    # 2. Compute features for all stocks
    print(f"\n[2/5] 计算特征（{len(TECH_FEATURES)}个指标）...")
    t1 = time.time()
    all_features = []  # list of (date, code, feat_vec, target)
    total_candles = 0
    skipped = 0
    
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
        
        # Compute features
        fwd_result = compute_features(c_arr, h_arr, l_arr, o_arr, v_arr, d_arr)
        
        for date_str, feat_dict, target in fwd_result:
            feat_vec = [feat_dict.get(k, 0) for k in TECH_FEATURES]
            all_features.append((date_str, code, feat_vec, target))
        
        total_candles += len(c_arr)
        
        if (ci+1) % 200 == 0:
            cur_sz = len(all_features)
            mem = __import__('psutil').Process().memory_info().rss / 1024 / 1024
            print(f"  [{ci+1}/{len(codes_zb)}] 已处理 {cur_sz:,} 样本, RSS {mem:.0f}MB, elapsed {time.time()-t1:.0f}s")
    
    print(f"  完成: {len(all_features):,} 样本, {skipped} 跳过")
    print(f"  特征计算时间: {time.time()-t1:.1f}s")
    
    # 3. Split into train/val/test by time
    print(f"\n[3/5] 时间分层...")
    t2 = time.time()
    
    train_feats, train_targets = [], []
    val_feats, val_targets = [], []
    test_feats, test_targets = [], []
    
    for date_str, code, feat_vec, target in all_features:
        if date_str <= TRAIN_END:
            train_feats.append(feat_vec)
            train_targets.append(target)
        elif date_str <= VAL_END:
            val_feats.append(feat_vec)
            val_targets.append(target)
        else:
            test_feats.append(feat_vec)
            test_targets.append(target)
    
    print(f"  训练: {len(train_feats):,} 样本 ({TRAIN_END}前)")
    print(f"  验证: {len(val_feats):,} 样本 ({TRAIN_END} ~ {VAL_END})")
    print(f"  测试: {len(test_feats):,} 样本 (HELD OUT, {VAL_END}后)")
    
    X_train = np.array(train_feats, dtype=np.float32)
    y_train = np.array(train_targets, dtype=np.float32)
    X_val = np.array(val_feats, dtype=np.float32)
    y_val = np.array(val_targets, dtype=np.float32)
    
    print(f"  特征矩阵: {X_train.shape[1]} 特征")
    print(f"  时间分层: {time.time()-t2:.1f}s")
    
    # 4. Train XGBoost with hyperparameter search
    print(f"\n[4/5] 训练XGBoost...")
    t3 = time.time()
    
    DO_GRID = '--grid' in sys.argv
    
    if DO_GRID:
        # Hyperparameter grid search
        best_rmse = float('inf')
        best_model = None
        best_params = None
        best_round = 0
        
        from itertools import product
        keys = list(HYPER_PARAM_GRID.keys())
        combos = list(product(*HYPER_PARAM_GRID.values()))
        print(f"  扫描 {len(combos)} 组超参...")
        
        for idx, combo in enumerate(combos):
            params = XGB_PARAMS.copy()
            params.update(dict(zip(keys, combo)))
            
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=TECH_FEATURES)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=TECH_FEATURES)
            
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
        # Base params only (faster)
        params = XGB_PARAMS.copy()
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=TECH_FEATURES)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=TECH_FEATURES)
        
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
    
    # 5. Evaluate cleanly
    print(f"\n[5/5] 评估...")
    
    dtest = xgb.DMatrix(np.array(test_feats, dtype=np.float32), feature_names=TECH_FEATURES)
    test_pred = model.predict(dtest)
    
    # Benchmark: baseline model
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
    print(f"\n  📊 特征重要性 (Top 15):")
    for feat, imp in sorted_imp[:15]:
        print(f"     {feat:20s}: {imp}")
    
    # Save model
    model_path = os.path.join(MODEL_DIR, 'a1_tech_v1.json')
    model.save_model(model_path)
    print(f"\n  💾 模型已保存: {model_path}")
    
    # Save metadata
    meta = {
        'train_end': TRAIN_END,
        'val_end': VAL_END,
        'features': TECH_FEATURES,
        'best_params': best_params,
        'best_round': best_round,
        'r2_train': round(r2_train, 4),
        'r2_val': round(r2_val, 4),
        'r2_test': round(r2_test, 4),
        'n_train': len(train_feats),
        'n_val': len(val_feats),
        'n_test': len(test_feats),
        'timestamp': datetime.now().isoformat(),
    }
    meta_path = os.path.join(MODEL_DIR, 'a1_tech_v1_meta.json')
    with open(meta_path, 'w') as f:
        f.write(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    print(f"  📄 元数据已保存: {meta_path}")
    
    # Compare with old model
    print(f"\n  📊 新旧模型对比...")
    old_model_path = os.path.join(D_DATA, 'layer3_checkpoints', 'model_batch_5.json')
    if os.path.exists(old_model_path):
        old_b = xgb.Booster()
        old_b.load_model(old_model_path)
        old_pred = old_b.predict(dtest)
        
        old_r2_test = 1 - np.var(np.array(test_targets) - old_pred) / np.var(test_targets)
        
        # Compare by return bins (deciles)
        test_targets_arr = np.array(test_targets)
        
        # New model: top-decile average return
        new_order = np.argsort(-test_pred)
        old_order = np.argsort(-old_pred)
        
        print(f"  {'指标':^20} {'新模型':^12} {'旧模型':^12}")
        print(f"  {'-'*44}")
        print(f"  {'Test R²':>20} {r2_test:>+.4f}       {old_r2_test:>+.4f}")
        
        # Top decile actual return
        top10 = len(test_targets_arr) // 10
        new_top_ret = np.mean(test_targets_arr[new_order[:top10]])
        old_top_ret = np.mean(test_targets_arr[old_order[:top10]])
        print(f"  {'Top10% avg ret':>20} {new_top_ret:>+.2f}%     {old_top_ret:>+.2f}%")
        
        # Bottom decile
        new_bot_ret = np.mean(test_targets_arr[new_order[-top10:]])
        old_bot_ret = np.mean(test_targets_arr[old_order[-top10:]])
        print(f"  {'Bottom10% avg ret':>20} {new_bot_ret:>+.2f}%     {old_bot_ret:>+.2f}%")
        
        # Spread
        print(f"  {'Spread(T-B)':>20} {new_top_ret-new_bot_ret:>+.2f}%     {old_top_ret-old_bot_ret:>+.2f}%")
    
    total_time = time.time() - t0
    print(f"\n{'='*65}")
    print(f"全部完成! 用时 {total_time:.0f}s ({total_time/60:.1f}分)")
    print(f"{'='*65}")
    
    return meta


if __name__ == '__main__':
    main()
