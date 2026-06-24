#!/usr/bin/env python3
"""
模型极致优化 + 端到端审计（内存优化版）
逐模型处理，训练完一个释放一个
"""
import gc, json, os, time, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
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

def load_macro():
    all_data = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'),
                                columns=['sym','date','close'])
    all_data['date'] = pd.to_datetime(all_data['date'])
    try:
        vix_raw = pd.read_parquet(os.path.join(ROOT, 'data/us/vix_10y.parquet'))
        vix_raw.columns = [c[0] if isinstance(c, tuple) else c for c in vix_raw.columns]
        vix_raw = vix_raw.reset_index()
        vix = pd.DataFrame({'date': pd.to_datetime(vix_raw['Date']), 'vix_close': vix_raw['Close'].astype(float)})
    except:
        vix = all_data[all_data['sym'] == '^VIX'][['date','close']].rename(columns={'close': 'vix_close'})
    macro_dfs = {}
    for sym in ['SPY','QQQ','IWM']:
        s = all_data[all_data['sym'] == sym][['date','close']].copy().sort_values('date')
        for d in [1,5,20,60]:
            s[f'{sym.lower()}_ret{d}'] = s['close'].pct_change(d)
        macro_dfs[sym] = s.drop(columns=['close'])
    macro = vix
    for sym in ['SPY','QQQ','IWM']:
        macro = macro.merge(macro_dfs[sym], on='date', how='outer')
    del all_data, macro_dfs, vix; gc.collect()
    return macro.sort_values('date').ffill()

def load_and_compute(min_price=10, max_price=10000):
    """加载数据+计算特征，一次性完成"""
    df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'),
                         columns=['sym','date','open','high','low','close','volume'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close','volume'])
    df = df[(df['close'] > min_price) & (df['close'] < max_price) & (df['volume'] > 0)]
    
    syms = list(df.groupby('sym'))
    batches = []
    for i in range(0, len(syms), 2000):
        parts = []
        for sym, g in syms[i:i+2000]:
            if len(g) < 130: continue
            parts.append(compute_features(g))
        if parts: batches.append(pd.concat(parts, ignore_index=True))
        del parts; gc.collect()
    result = pd.concat(batches, ignore_index=True)
    del batches, syms, df; gc.collect()
    return result

def prepare_valid(feat, features, hold_days, macro=None):
    """准备验证数据集"""
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
    return valid

def run_config(valid, features, hold_days, label):
    """训练+全市场回测+IC分析+Portfolio模拟"""
    print(f"\n{'='*60}")
    print(f"  {label} ({len(features)}特征)")
    print(f"{'='*60}")
    
    train_end = pd.Timestamp('2023-12-31')
    val_end = pd.Timestamp('2024-12-31')
    
    train_df = valid[valid['date'] <= train_end]
    val_df = valid[(valid['date'] > train_end) & (valid['date'] <= val_end)]
    test_df = valid[valid['date'] > val_end].copy()
    print(f"  训练: {len(train_df):,} | 验证: {len(val_df):,} | 测试: {len(test_df):,}")
    
    # 训练
    t0 = time.time()
    model = xgb.XGBClassifier(
        objective='binary:logistic', max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        n_estimators=200, tree_method='hist', device='cuda',
        random_state=42, verbosity=0,
    )
    model.fit(train_df[features].values, train_df['hit'].values,
              eval_set=[(val_df[features].values, val_df['hit'].values)], verbose=False)
    print(f"  训练: {time.time()-t0:.0f}s")
    
    # 打分
    test_df['pred'] = model.predict_proba(test_df[features].values)[:, 1]
    test_df['rank_pct'] = test_df.groupby('date')['pred'].rank(pct=True)
    
    # IC分析
    ic_list = []
    for dt in sorted(test_df['date'].unique()):
        day = test_df[test_df['date'] == dt]
        if len(day) < 100: continue
        ic = day['pred'].corr(day['fwd_ret'], method='spearman')
        ic_list.append({'date': dt, 'ic': ic})
    ic_df = pd.DataFrame(ic_list)
    ic_mean = ic_df['ic'].mean()
    ic_std = ic_df['ic'].std()
    icir = ic_mean / ic_std if ic_std > 0 else 0
    ic_pos = (ic_df['ic'] > 0).mean()
    ic_df['month'] = ic_df['date'].dt.to_period('M')
    monthly_ic = ic_df.groupby('month')['ic'].mean()
    monthly_pos = (monthly_ic > 0).mean()
    
    # 分层收益
    layers = {}
    for tier, lo, hi in [('Top5%', 0.95, 1.0), ('Top10%', 0.90, 1.0),
                          ('Top20%', 0.80, 1.0), ('Mid60%', 0.20, 0.80),
                          ('Bot20%', 0.0, 0.20)]:
        sub = test_df[(test_df['rank_pct'] >= lo) & (test_df['rank_pct'] < hi + 0.001)]
        if len(sub) > 10:
            layers[tier] = {'win': float(sub['hit'].mean()), 'avg_ret': float(sub['fwd_ret'].mean()),
                            'med_ret': float(sub['fwd_ret'].median()), 'n': int(len(sub))}
    
    # 年度
    test_df['year'] = test_df['date'].dt.year
    yearly = {}
    for yr in sorted(test_df['year'].unique()):
        y = test_df[test_df['year'] == yr]
        t5 = y[y['rank_pct'] >= 0.95]; b20 = y[y['rank_pct'] < 0.20]
        if len(t5) > 10:
            yearly[str(yr)] = {'t5_win': float(t5['hit'].mean()), 't5_ret': float(t5['fwd_ret'].mean()),
                               'b20_win': float(b20['hit'].mean()), 'b20_ret': float(b20['fwd_ret'].mean())}
    
    # Portfolio模拟
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
    calmar = cagr / abs(dd) if dd != 0 else 0
    
    # 统计检验
    t5_d = test_df[test_df['rank_pct'] >= 0.95]['fwd_ret']
    b20_d = test_df[test_df['rank_pct'] < 0.20]['fwd_ret']
    _, p_val = stats.ttest_ind(t5_d, b20_d) if len(t5_d) > 10 and len(b20_d) > 10 else (0, 1)
    
    # 特征重要性
    imp = dict(zip(features, model.feature_importances_))
    macro_imp = sum(v for k, v in imp.items() if k in MACRO_FEATS)
    tech_imp = sum(v for k, v in imp.items() if k not in MACRO_FEATS)
    
    print(f"  IC={ic_mean:.4f} ICIR={icir:.3f} IC>0={ic_pos*100:.0f}% 月IC>0={monthly_pos*100:.0f}%")
    print(f"  Top5%: win={layers.get('Top5%',{}).get('win',0)*100:.1f}% ret={layers.get('Top5%',{}).get('avg_ret',0)*100:+.2f}%")
    print(f"  Portfolio: CAGR={cagr*100:.1f}% Sharpe={sharpe:.2f} DD={dd*100:.1f}% Sortino={sortino:.2f}")
    
    return {
        'label': label, 'features': features,
        'ic': float(ic_mean), 'icir': float(icir), 'ic_pos': float(ic_pos),
        'monthly_ic_pos': float(monthly_pos),
        'monthly_ic': {str(k): float(v) for k, v in monthly_ic.items()},
        'layers': layers, 'yearly': yearly,
        'portfolio': {'cagr': float(cagr), 'sharpe': float(sharpe), 'max_dd': float(dd),
                       'sortino': float(sortino), 'calmar': float(calmar), 'trades': len(equity_curve)-1},
        'p_val': float(p_val),
        'top15_imp': {k: round(float(v), 4) for k, v in sorted(imp.items(), key=lambda x: -x[1])[:15]},
        'macro_pct': float(macro_imp / (macro_imp + tech_imp) * 100) if (macro_imp + tech_imp) > 0 else 0,
        'model': model,
    }

def main():
    print("="*70)
    print("模型极致优化 + 端到端审计 (内存优化版)")
    print("="*70)
    
    macro = load_macro()
    all_results = {}
    
    # ========== 蓝盾V8 ==========
    print("\n" + "="*70)
    print("蓝盾V8 优化 (4种配置)")
    print("="*70)
    
    feat = load_and_compute(min_price=10)
    print(f"  蓝盾: {len(feat):,}行, {feat['sym'].nunique()}只")
    
    configs_blue = {
        'V8_tech': {'features': TECH_FEATS, 'use_macro': False},
        'V8_tech+vix': {'features': TECH_FEATS + ['vix_close'], 'use_macro': True},
        'V8_tech+vix+spy': {'features': TECH_FEATS + ['vix_close','spy_ret20'], 'use_macro': True},
        'V8_tech+macro4': {'features': TECH_FEATS + ['vix_close','spy_ret20','qqq_ret20','iwm_ret20'], 'use_macro': True},
    }
    
    best_blue_icir = -999
    best_blue_model = None
    best_blue_name = None
    
    for name, cfg in configs_blue.items():
        m = macro if cfg['use_macro'] else None
        valid = prepare_valid(feat.copy(), cfg['features'], hold_days=20, macro=m)
        r = run_config(valid, cfg['features'], 20, name)
        all_results[name] = r
        if r['icir'] > best_blue_icir:
            best_blue_icir = r['icir']
            best_blue_model = r['model']
            best_blue_name = name
        del valid; gc.collect()
    
    # 保存最优蓝盾
    if best_blue_model:
        best_blue_model.save_model(os.path.join(ROOT, 'models/us/blueshield_v8_xgb.json'))
        print(f"\n✅ 最优蓝盾: {best_blue_name} (ICIR={best_blue_icir:.3f})")
    
    del feat, best_blue_model; gc.collect()
    
    # ========== 绿箭V12 ==========
    print("\n" + "="*70)
    print("绿箭V12 优化 (2种配置)")
    print("="*70)
    
    feat = load_and_compute(min_price=1, max_price=10)
    print(f"  绿箭: {len(feat):,}行, {feat['sym'].nunique()}只")
    
    configs_arrow = {
        'A_full': {'features': TECH_FEATS + EXTRA_FEATS + MACRO_FEATS, 'use_macro': True},
        'A_tech': {'features': TECH_FEATS + EXTRA_FEATS, 'use_macro': False},
    }
    
    best_arrow_icir = -999
    best_arrow_model = None
    best_arrow_name = None
    
    for name, cfg in configs_arrow.items():
        m = macro if cfg['use_macro'] else None
        valid = prepare_valid(feat.copy(), cfg['features'], hold_days=5, macro=m)
        r = run_config(valid, cfg['features'], 5, name)
        all_results[name] = r
        if r['icir'] > best_arrow_icir:
            best_arrow_icir = r['icir']
            best_arrow_model = r['model']
            best_arrow_name = name
        del valid; gc.collect()
    
    # 保存最优绿箭
    if best_arrow_model:
        best_arrow_model.save_model(os.path.join(ROOT, 'models/us/arrow_v12_xgb.json'))
        print(f"\n✅ 最优绿箭: {best_arrow_name} (ICIR={best_arrow_icir:.3f})")
    
    del feat, best_arrow_model; gc.collect()
    
    # ========== 最终排名 ==========
    print("\n" + "="*70)
    print("最终排名 (按ICIR)")
    print("="*70)
    
    ranked = sorted([(n, r) for n, r in all_results.items()], key=lambda x: x[1]['icir'], reverse=True)
    print(f"\n{'配置':<20s} {'IC':>7s} {'ICIR':>7s} {'月>0':>6s} {'T5胜':>6s} {'T5收':>7s} {'Sharpe':>7s} {'DD':>7s} {'宏%':>5s}")
    print("-" * 85)
    for name, r in ranked:
        t5w = r['layers'].get('Top5%',{}).get('win',0)*100
        t5r = r['layers'].get('Top5%',{}).get('avg_ret',0)*100
        print(f"{name:<20s} {r['ic']:>7.4f} {r['icir']:>7.3f} {r['monthly_ic_pos']*100:>5.0f}% "
              f"{t5w:>5.1f}% {t5r:>+6.2f}% {r['portfolio']['sharpe']:>7.2f} {r['portfolio']['max_dd']*100:>6.1f}% {r['macro_pct']:>4.0f}%")
    
    # 保存
    save = {n: {k: v for k, v in r.items() if k != 'model'} for n, r in all_results.items()}
    with open(os.path.join(ROOT, 'data/model-optimization-results.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    
    # 更新meta
    best_blue_r = all_results[best_blue_name]
    best_arrow_r = all_results[best_arrow_name]
    
    for prefix, r, name in [('blueshield_v8', best_blue_r, best_blue_name), ('arrow_v12', best_arrow_r, best_arrow_name)]:
        meta = {
            'version': prefix, 'config': name,
            'features': r['features'], 'n_features': len(r['features']),
            'macro_features': sum(1 for f in r['features'] if f in MACRO_FEATS),
            'hold_days': 20 if 'blueshield' in prefix else 5,
            'ic': r['ic'], 'icir': r['icir'], 'monthly_ic_pos': r['monthly_ic_pos'],
            'portfolio': r['portfolio'], 'feature_importance': r['top15_imp'],
            'macro_imp_pct': r['macro_pct'], 'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
        }
        with open(os.path.join(ROOT, f'models/us/{prefix}_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
    
    print(f"\n✅ 全部完成。最优蓝盾={best_blue_name}, 最优绿箭={best_arrow_name}")
    print(f"结果: data/model-optimization-results.json")

if __name__ == '__main__':
    main()
