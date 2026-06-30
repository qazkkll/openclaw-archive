#!/usr/bin/env python3
"""
GPU参数优化 v2 — 内存优化版
先采样快速搜索，再全量验证top参数
"""
import gc, json, os, time, warnings, random
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime

warnings.filterwarnings('ignore')
ROOT = '/home/hermes/.hermes/openclaw-archive'

TECH_FEATS = [
    'ma5','ma20','ma60','ma_bias20','ma_align','price_position',
    'ret1','ret5','ret20','ret60','momentum_6m','momentum_1m',
    'mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change',
    'rsi14','rsi_change','macd','macd_signal','macd_hist',
    'bb_std','bb_width','bb_pos','ret_quality'
]
EXTRA_FEATS = ['price', 'range_pct']
MACRO_FEATS = ['vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60',
               'qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60',
               'iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60']

BASELINE = {'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 10, 'n_estimators': 200}

GRID = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [5, 10, 20],
    'n_estimators': [200, 400, 600],
}

random.seed(42)
ALL_FEATURES = TECH_FEATS + EXTRA_FEATS + MACRO_FEATS

def compute_features(g):
    g = g.sort_values('date').copy()
    c = g['close'].astype(np.float32)
    g['ma5'] = c.rolling(5).mean()
    g['ma20'] = c.rolling(20).mean()
    g['ma60'] = c.rolling(60).mean()
    g['ma_bias20'] = (c - g['ma20']) / g['ma20']
    g['ma_align'] = ((c > g['ma5']).astype(np.int8) + (g['ma5'] > g['ma20']).astype(np.int8))
    mn60 = c.rolling(60).min(); mx60 = c.rolling(60).max()
    g['price_position'] = (c - mn60) / (mx60 - mn60 + 1e-10)
    g['ret1'] = c.pct_change(1); g['ret5'] = c.pct_change(5)
    g['ret20'] = c.pct_change(20); g['ret60'] = c.pct_change(60)
    g['momentum_6m'] = c.pct_change(126); g['momentum_1m'] = c.pct_change(21)
    g['mom_divergence'] = g['momentum_1m'] - g['ret20']
    g['trend_accel'] = g['ret5'] - g['ret5'].shift(5)
    dr = c.pct_change(1)
    g['vol20'] = dr.rolling(20).std(); g['vol5'] = dr.rolling(5).std()
    g['vol_ratio'] = g['volume'].astype(np.float32) / g['volume'].astype(np.float32).rolling(20).mean()
    g['vol_change'] = g['vol20'] / g['vol20'].shift(20)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    g['rsi14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    g['rsi_change'] = g['rsi14'].diff(5)
    e12 = c.ewm(span=12).mean(); e26 = c.ewm(span=26).mean()
    g['macd'] = e12 - e26; g['macd_signal'] = g['macd'].ewm(span=9).mean()
    g['macd_hist'] = g['macd'] - g['macd_signal']
    g['bb_std'] = c.rolling(20).std()
    g['bb_width'] = 2 * g['bb_std'] / g['ma20']
    g['bb_pos'] = (c - g['ma20']) / (2 * g['bb_std'] + 1e-10)
    g['ret_quality'] = g['ret20'] / (g['vol20'] + 1e-10)
    g['price'] = c
    g['range_pct'] = (g['high'] - g['low']) / (c + 1e-10)
    return g

def load_data_sampled(min_price, max_price, sample_n=2000):
    """加载数据+特征，随机采样sample_n只股票减内存"""
    df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'),
                         columns=['sym','date','open','high','low','close','volume'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close','volume'])
    df = df[(df['close'] > min_price) & (df['close'] < max_price) & (df['volume'] > 0)]
    
    # 随机采样
    all_syms = df['sym'].unique()
    if len(all_syms) > sample_n:
        rng = np.random.RandomState(42)
        sampled = rng.choice(all_syms, sample_n, replace=False)
        df = df[df['sym'].isin(sampled)]
    
    # 按批次计算特征
    syms = list(df.groupby('sym'))
    batches = []
    for i in range(0, len(syms), 1000):
        parts = []
        for sym, g in syms[i:i+1000]:
            if len(g) < 130: continue
            parts.append(compute_features(g))
        if parts: batches.append(pd.concat(parts, ignore_index=True))
        del parts; gc.collect()
    result = pd.concat(batches, ignore_index=True)
    del batches, syms, df; gc.collect()
    return result

def load_macro():
    vix_raw = pd.read_parquet(os.path.join(ROOT, 'data/us/vix_10y.parquet'))
    vix = pd.DataFrame({'date': pd.to_datetime(vix_raw['date']), 'vix_close': vix_raw['close'].astype(np.float32)})
    
    spy_data = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'),
                                columns=['sym','date','close'])
    spy_data['date'] = pd.to_datetime(spy_data['date'])
    
    macro = vix
    for sym in ['SPY','QQQ','IWM']:
        s = spy_data[spy_data['sym'] == sym][['date','close']].copy().sort_values('date')
        for d in [1,5,20,60]:
            s[f'{sym.lower()}_ret{d}'] = s['close'].pct_change(d).astype(np.float32)
        s = s.drop(columns=['close'])
        macro = macro.merge(s, on='date', how='outer')
    del spy_data, vix; gc.collect()
    return macro.sort_values('date').ffill()

def prepare_valid(feat, features, hold_days, macro=None):
    if macro is not None:
        macro_cols = [c for c in features if c in MACRO_FEATS]
        if macro_cols:
            feat = feat.merge(macro[['date'] + macro_cols], on='date', how='left')
            feat[macro_cols] = feat[macro_cols].ffill().fillna(0)
    feat = feat.sort_values(['sym','date'])
    feat['fwd_ret'] = feat.groupby('sym')['close'].pct_change(hold_days).shift(-hold_days)
    feat['hit'] = (feat['fwd_ret'] > 0.02).astype(np.int8)
    valid = feat.dropna(subset=features + ['fwd_ret'])
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    # 转float32省内存
    for c in features:
        if c in valid.columns:
            valid[c] = valid[c].astype(np.float32)
    return valid

def eval_config(valid, features, hold_days, params):
    train_end = pd.Timestamp('2023-12-31')
    val_end = pd.Timestamp('2024-12-31')
    
    train_df = valid[valid['date'] <= train_end]
    val_df = valid[(valid['date'] > train_end) & (valid['date'] <= val_end)]
    test_df = valid[valid['date'] > val_end].copy()
    
    if len(train_df) < 1000 or len(test_df) < 1000:
        return None
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        min_child_weight=params['min_child_weight'],
        n_estimators=params['n_estimators'],
        tree_method='hist', device='cuda',
        random_state=42, verbosity=0,
    )
    model.fit(train_df[features].values, train_df['hit'].values,
              eval_set=[(val_df[features].values, val_df['hit'].values)], verbose=False)
    
    test_df['pred'] = model.predict_proba(test_df[features].values)[:, 1]
    test_df['rank_pct'] = test_df.groupby('date')['pred'].rank(pct=True)
    
    ic_list = []
    for dt in sorted(test_df['date'].unique()):
        day = test_df[test_df['date'] == dt]
        if len(day) < 50: continue
        ic = day['pred'].corr(day['fwd_ret'], method='spearman')
        ic_list.append(ic)
    if len(ic_list) < 10:
        return None
    ic_mean = float(np.mean(ic_list))
    ic_std = float(np.std(ic_list))
    icir = ic_mean / ic_std if ic_std > 0 else 0
    ic_pos = float(np.mean([x > 0 for x in ic_list]))
    
    t5 = test_df[test_df['rank_pct'] >= 0.95]
    t5_win = float(t5['hit'].mean()) if len(t5) > 10 else 0
    t5_ret = float(t5['fwd_ret'].mean()) if len(t5) > 10 else 0
    
    # Portfolio
    test_dates = sorted(test_df['date'].unique())
    equity = 100000.0; equity_curve = [equity]; cost = 0.001
    for i in range(0, len(test_dates) - hold_days, hold_days):
        day = test_df[test_df['date'] == test_dates[i]]
        if len(day) < 50: continue
        top15 = day.nlargest(15, 'pred')
        avg_ret = top15['fwd_ret'].mean()
        if np.isnan(avg_ret): continue
        equity *= (1 + avg_ret - 2 * cost)
        equity_curve.append(equity)
    
    eq = np.array(equity_curve)
    n_years = len(equity_curve) * hold_days / 365.25
    cagr = (eq[-1] / eq[0]) ** (1/max(n_years, 0.1)) - 1
    returns = np.diff(eq) / eq[:-1]
    sharpe = (returns.mean() / returns.std() * np.sqrt(252/hold_days)) if returns.std() > 0 else 0
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    neg_r = returns[returns < 0]
    sortino = (returns.mean() / neg_r.std() * np.sqrt(252/hold_days)) if len(neg_r) > 0 and neg_r.std() > 0 else 0
    
    return {
        'ic': ic_mean, 'icir': float(icir), 'ic_pos': ic_pos,
        't5_win': t5_win, 't5_ret': t5_ret,
        'sharpe': float(sharpe), 'sortino': float(sortino),
        'cagr': float(cagr), 'max_dd': float(dd),
    }

def random_search(n=60):
    return [{k: random.choice(v) for k, v in GRID.items()} for _ in range(n)]

def run_search(name, min_price, max_price, hold_days, macro, n_sample=1500, n_search=60):
    print(f"\n{'='*70}")
    print(f"{name} (price {'$'+str(min_price)+'-'+str(max_price) if max_price<10000 else '>$'+str(min_price)}, hold={hold_days}d)")
    print(f"{'='*70}")
    
    feat = load_data_sampled(min_price, max_price, sample_n=n_sample)
    print(f"  样本: {len(feat):,}行, {feat['sym'].nunique()}只")
    
    # 合并宏特征
    macro_cols = ['date'] + [c for c in MACRO_FEATS if c in macro.columns]
    feat = feat.merge(macro[macro_cols], on='date', how='left')
    for c in MACRO_FEATS:
        if c in feat.columns:
            feat[c] = feat[c].ffill().fillna(0)
    gc.collect()
    
    valid = prepare_valid(feat.copy(), ALL_FEATURES, hold_days)
    print(f"  有效: {len(valid):,}")
    del feat; gc.collect()
    
    # Baseline
    print("\n  ▶ 基线...")
    t0 = time.time()
    bl = eval_config(valid, ALL_FEATURES, hold_days, BASELINE)
    print(f"    ICIR={bl['icir']:.3f} Sharpe={bl['sharpe']:.2f} T5win={bl['t5_win']*100:.1f}% T5ret={bl['t5_ret']*100:+.2f}% ({time.time()-t0:.0f}s)")
    
    # Search
    configs = random_search(n_search)
    best = {'icir': -999, 'params': None, 'result': None}
    all_configs = []
    
    for i, cfg in enumerate(configs):
        t0 = time.time()
        r = eval_config(valid, ALL_FEATURES, hold_days, cfg)
        elapsed = time.time() - t0
        if r is None:
            continue
        all_configs.append({'params': cfg, **r})
        if r['icir'] > best['icir']:
            best = {'icir': r['icir'], 'params': cfg, 'result': r}
            print(f"  [{i+1}/{n_search}] ★ BEST ICIR={r['icir']:.3f} Sharpe={r['sharpe']:.2f} T5win={r['t5_win']*100:.1f}% ({elapsed:.0f}s)")
        elif (i+1) % 15 == 0:
            print(f"  [{i+1}/{n_search}] best={best['icir']:.3f}")
    
    del valid; gc.collect()
    
    return {
        'baseline': bl,
        'best': {'params': best['params'], **best['result']},
        'all_configs': all_configs,
    }

def main():
    print("="*70)
    print("GPU参数优化 v2 — 采样搜索+全量验证")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)
    
    try:
        import torch
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("GPU: XGBoost CUDA mode")
    
    print("加载宏观数据...")
    macro = load_macro()
    
    # 蓝盾
    blue = run_search("蓝盾 Blueshield", 10, 10000, 20, macro, n_sample=1500, n_search=60)
    
    # 绿箭
    arrow = run_search("绿箭 Arrow", 1, 10, 5, macro, n_sample=1000, n_search=60)
    
    # Save
    with open(os.path.join(ROOT, 'data/blueshield_best_params.json'), 'w') as f:
        json.dump({'params': blue['best']['params'], 'metrics': {k:v for k,v in blue['best'].items() if k != 'params'}}, f, indent=2)
    with open(os.path.join(ROOT, 'data/arrow_best_params.json'), 'w') as f:
        json.dump({'params': arrow['best']['params'], 'metrics': {k:v for k,v in arrow['best'].items() if k != 'params'}}, f, indent=2)
    with open(os.path.join(ROOT, 'data/param_search_results.json'), 'w') as f:
        json.dump({'blueshield': blue, 'arrow': arrow}, f, indent=2, default=str)
    
    # Summary
    print(f"\n{'='*70}")
    print("优化结果汇总")
    print(f"{'='*70}")
    
    for name, data in [('蓝盾', blue), ('绿箭', arrow)]:
        bl = data['baseline']
        best = data['best']
        print(f"\n{name}:")
        print(f"  基线: ICIR={bl['icir']:.3f} Sharpe={bl['sharpe']:.2f} T5win={bl['t5_win']*100:.1f}% T5ret={bl['t5_ret']*100:+.2f}% DD={bl['max_dd']*100:.1f}%")
        print(f"  最优: ICIR={best['icir']:.3f} Sharpe={best['sharpe']:.2f} T5win={best['t5_win']*100:.1f}% T5ret={best['t5_ret']*100:+.2f}% DD={best['max_dd']*100:.1f}%")
        print(f"  Δ:    ICIR {best['icir']-bl['icir']:+.3f} Sharpe {best['sharpe']-bl['sharpe']:+.2f}")
        print(f"  参数: {best['params']}")
    
    print(f"\n✅ 保存: data/param_search_results.json")

if __name__ == '__main__':
    main()
