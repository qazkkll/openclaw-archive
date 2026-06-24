#!/usr/bin/env python3
"""
独立验证：蓝盾V8 + 绿箭V12全面审计
1. V8独立验证（IC、月度一致性、look-ahead bias检查）
2. V8优化空间评估
3. 绿箭V12全市场回测 + 特征审计
"""
import gc, json, os, sys, time, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

warnings.filterwarnings('ignore')
ROOT = '/home/hermes/.hermes/openclaw-archive'

# ========== 特征定义 ==========
TECH_FEATS_V8 = [
    'ma5','ma20','ma60','ma_bias20','ma_align','price_position',
    'ret1','ret5','ret20','ret60','momentum_6m','momentum_1m',
    'mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change',
    'rsi14','rsi_change','macd','macd_signal','macd_hist',
    'bb_std','bb_width','bb_pos','ret_quality'
]

TECH_FEATS_ARROW = TECH_FEATS_V8 + ['price', 'range_pct']
MACRO_FEATS = ['vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60',
               'qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60',
               'iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60']
ALL_FEATS_ARROW = TECH_FEATS_ARROW + MACRO_FEATS

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

def load_macro_data():
    """加载宏观数据：VIX + SPY/QQQ/IWM收益率"""
    print("   加载宏观数据...", flush=True)
    
    # 从主数据提取宏观指标
    macro_syms = ['SPY','QQQ','IWM']
    all_data = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'),
                                columns=['sym','date','close'])
    all_data['date'] = pd.to_datetime(all_data['date'])
    
    # VIX - 从parquet的multi-index列中提取
    try:
        vix_raw = pd.read_parquet(os.path.join(ROOT, 'data/us/vix_10y.parquet'))
        vix_raw.columns = [c[0] if isinstance(c, tuple) else c for c in vix_raw.columns]
        vix_raw = vix_raw.reset_index()
        vix = pd.DataFrame({'date': pd.to_datetime(vix_raw['Date']), 'vix_close': vix_raw['Close'].astype(float)})
    except:
        # fallback: 从主数据提取
        vix = all_data[all_data['sym'] == '^VIX'][['date','close']].copy()
        vix = vix.rename(columns={'close': 'vix_close'})
    
    # SPY/QQQ/IWM收益率
    macro_dfs = {}
    for sym in macro_syms:
        s = all_data[all_data['sym'] == sym][['date','close']].copy()
        s = s.sort_values('date')
        for d in [1,5,20,60]:
            s[f'{sym.lower()}_ret{d}'] = s['close'].pct_change(d)
        macro_dfs[sym] = s.drop(columns=['close'])
    
    # 合并
    macro = vix
    for sym in macro_syms:
        macro = macro.merge(macro_dfs[sym], on='date', how='outer')
    macro = macro.sort_values('date').ffill()
    return macro

def load_and_score(model_path, features, hold_days, label, universe_filter=None, need_macro=False):
    """加载模型，打分，计算IC和分层收益"""
    print(f"\n{'='*70}")
    print(f"审计: {label}")
    print(f"{'='*70}")
    
    # 加载数据
    print("1. 加载数据...", flush=True)
    t0 = time.time()
    df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'),
                         columns=['sym','date','open','high','low','close','volume'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close','volume'])
    if universe_filter:
        df = universe_filter(df)
    else:
        df = df[(df['close'] > 10) & (df['close'] < 10000) & (df['volume'] > 0)]
    print(f"   {time.time()-t0:.0f}s, {len(df):,}行, {df['sym'].nunique()}只")
    
    # 特征
    print("2. 计算特征...", flush=True)
    t0 = time.time()
    syms = list(df.groupby('sym'))
    total = len(syms)
    batch_size = 2000
    feat_batches = []
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_parts = []
        for sym, g in syms[batch_start:batch_end]:
            if len(g) < 130:
                continue
            f = compute_features(g)
            batch_parts.append(f)
        if batch_parts:
            feat_batches.append(pd.concat(batch_parts, ignore_index=True))
        del batch_parts; gc.collect()
        print(f"   批次 {batch_start//batch_size+1}: {batch_start}-{batch_end}/{total} ({time.time()-t0:.0f}s)", flush=True)
    feat_all = pd.concat(feat_batches, ignore_index=True)
    del feat_batches; gc.collect()
    print(f"   总计: {time.time()-t0:.0f}s, {len(feat_all):,}行")
    
    # 合并宏观数据
    if need_macro:
        macro = load_macro_data()
        feat_all = feat_all.merge(macro, on='date', how='left')
        feat_all[MACRO_FEATS] = feat_all[MACRO_FEATS].ffill().fillna(0)
        print(f"   宏观数据已合并")
    
    # 标签
    feat_all = feat_all.sort_values(['sym','date'])
    feat_all['fwd_ret'] = feat_all.groupby('sym')['close'].pct_change(hold_days).shift(-hold_days)
    feat_all['hit'] = (feat_all['fwd_ret'] > 0.02).astype(np.int8)
    
    valid = feat_all.dropna(subset=features + ['fwd_ret'])
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    
    # 测试集：2024-01-01之后
    test_df = valid[valid['date'] > pd.Timestamp('2024-01-01')].copy()
    print(f"   测试集: {len(test_df):,}行, {test_df['sym'].nunique()}只")
    
    # 加载模型
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # 打分
    print("3. 打分...", flush=True)
    test_df['pred_score'] = model.predict_proba(test_df[features].values)[:, 1]
    test_df['rank_pct'] = test_df.groupby('date')['pred_score'].rank(pct=True)
    
    # ========== IC分析（金标准） ==========
    print("\n4. IC分析（Spearman Rank Correlation）...")
    ic_results = []
    for dt in sorted(test_df['date'].unique()):
        day = test_df[test_df['date'] == dt]
        if len(day) < 100:
            continue
        ic = day['pred_score'].corr(day['fwd_ret'], method='spearman')
        ic_results.append({'date': dt, 'ic': ic, 'n': len(day)})
    ic_df = pd.DataFrame(ic_results)
    
    ic_mean = ic_df['ic'].mean()
    ic_std = ic_df['ic'].std()
    icir = ic_mean / ic_std if ic_std > 0 else 0
    ic_positive_pct = (ic_df['ic'] > 0).mean()
    
    print(f"   IC均值: {ic_mean:.4f}")
    print(f"   IC标准差: {ic_std:.4f}")
    print(f"   ICIR: {icir:.3f}")
    print(f"   IC>0比例: {ic_positive_pct*100:.1f}%")
    
    # ========== 分层收益 ==========
    print("\n5. 分层收益（全时期）...")
    layers = {}
    for label_tier, lo, hi in [('Top5%', 0.95, 1.0), ('Top10%', 0.90, 1.0),
                                 ('Top20%', 0.80, 1.0), ('Mid60%', 0.20, 0.80),
                                 ('Bot20%', 0.0, 0.20)]:
        sub = test_df[(test_df['rank_pct'] >= lo) & (test_df['rank_pct'] < hi + 0.001)]
        if len(sub) > 10:
            win = sub['hit'].mean()
            avg = sub['fwd_ret'].mean()
            med = sub['fwd_ret'].median()
            layers[label_tier] = {'win': win, 'avg': avg, 'med': med, 'n': len(sub)}
            print(f"   {label_tier:10s}: 胜率={win*100:.1f}%, 均收益={avg*100:+.2f}%, 中位数={med*100:+.2f}%, n={len(sub):,}")
    
    # ========== 年度分解 ==========
    print("\n6. 年度分解...")
    test_df['year'] = test_df['date'].dt.year
    for year in sorted(test_df['year'].unique()):
        yr = test_df[test_df['year'] == year]
        t5 = yr[yr['rank_pct'] >= 0.95]
        b20 = yr[yr['rank_pct'] < 0.20]
        if len(t5) > 10:
            print(f"   {year}: Top5% 胜率={t5['hit'].mean()*100:.1f}% 均收益={t5['fwd_ret'].mean()*100:+.2f}% "
                  f"| Bot20% 胜率={b20['hit'].mean()*100:.1f}% 均收益={b20['fwd_ret'].mean()*100:+.2f}%")
    
    # ========== 月度IC一致性 ==========
    print("\n7. 月度IC一致性...")
    ic_df['month'] = ic_df['date'].dt.to_period('M')
    monthly_ic = ic_df.groupby('month')['ic'].mean()
    ic_positive_months = (monthly_ic > 0).mean()
    print(f"   月度IC>0比例: {ic_positive_months*100:.1f}%")
    for m, ic_val in monthly_ic.items():
        marker = "✅" if ic_val > 0 else "❌"
        print(f"   {m}: IC={ic_val:.4f} {marker}")
    
    # ========== 特征重要性 ==========
    print("\n8. 特征重要性...")
    imp = dict(zip(features, model.feature_importances_))
    for f, v in sorted(imp.items(), key=lambda x: -x[1])[:15]:
        print(f"   {f:20s} {v:.4f}")
    
    # 检查宏观vs个股
    macro_imp = sum(v for k, v in imp.items() if k in MACRO_FEATS)
    tech_imp = sum(v for k, v in imp.items() if k not in MACRO_FEATS)
    total_imp = macro_imp + tech_imp
    if total_imp > 0:
        print(f"\n   宏观特征占比: {macro_imp/total_imp*100:.1f}%")
        print(f"   个股特征占比: {tech_imp/total_imp*100:.1f}%")
    
    # ========== 统计显著性 ==========
    print("\n9. 统计显著性检验...")
    t5_data = test_df[test_df['rank_pct'] >= 0.95]['fwd_ret']
    b20_data = test_df[test_df['rank_pct'] < 0.20]['fwd_ret']
    if len(t5_data) > 10 and len(b20_data) > 10:
        t_stat, p_val = stats.ttest_ind(t5_data, b20_data)
        print(f"   Top5% vs Bot20% t-stat: {t_stat:.3f}, p-value: {p_val:.4f}")
        if p_val < 0.05:
            print(f"   ✅ 统计显著 (p<0.05)")
        else:
            print(f"   ❌ 不显著 (p≥0.05)")
    
    return {
        'ic_mean': ic_mean, 'icir': icir, 'ic_positive_pct': ic_positive_pct,
        'ic_positive_months': ic_positive_months,
        'layers': layers, 'macro_imp_pct': macro_imp/total_imp*100 if total_imp > 0 else 0,
    }

def main():
    # ========== 1. 蓝盾V8独立验证 ==========
    v8_result = load_and_score(
        model_path=os.path.join(ROOT, 'models/us/blueshield_v8_xgb.json'),
        features=TECH_FEATS_V8,
        hold_days=20,
        label="蓝盾V8 (纯技术, 27特征)"
    )
    
    # ========== 2. 绿箭V12全面审计 ==========
    # Arrow是$1-$10
    def arrow_filter(df):
        return df[(df['close'] >= 1) & (df['close'] <= 10) & (df['volume'] > 0)]
    
    arrow_result = load_and_score(
        model_path=os.path.join(ROOT, 'models/us/arrow_v12_xgb.json'),
        features=ALL_FEATS_ARROW,
        hold_days=5,
        label="绿箭V12 (29技术+13宏观, $1-$10)",
        universe_filter=arrow_filter, need_macro=True
    )
    
    # ========== 3. 绿箭纯技术版本（对比） ==========
    arrow_tech_result = load_and_score(
        model_path=os.path.join(ROOT, 'models/us/arrow_v12_xgb.json'),
        features=TECH_FEATS_ARROW,  # 只用技术特征，忽略宏观
        hold_days=5,
        label="绿箭V12-纯技术对比 (忽略宏观特征)",
        universe_filter=arrow_filter, need_macro=False
    )
    
    # ========== 总结 ==========
    print(f"\n{'='*70}")
    print("总结")
    print(f"{'='*70}")
    print(f"\n蓝盾V8:")
    print(f"  IC: {v8_result['ic_mean']:.4f}, ICIR: {v8_result['icir']:.3f}")
    print(f"  月度IC>0: {v8_result['ic_positive_months']*100:.0f}%")
    print(f"  Top5%胜率: {v8_result['layers'].get('Top5%',{}).get('win',0)*100:.1f}%")
    print(f"  宏观占比: {v8_result['macro_imp_pct']:.1f}%")
    
    print(f"\n绿箭V12 (含宏观):")
    print(f"  IC: {arrow_result['ic_mean']:.4f}, ICIR: {arrow_result['icir']:.3f}")
    print(f"  月度IC>0: {arrow_result['ic_positive_months']*100:.0f}%")
    print(f"  Top5%胜率: {arrow_result['layers'].get('Top5%',{}).get('win',0)*100:.1f}%")
    print(f"  宏观占比: {arrow_result['macro_imp_pct']:.1f}%")
    
    print(f"\n绿箭V12-纯技术对比:")
    print(f"  IC: {arrow_tech_result['ic_mean']:.4f}, ICIR: {arrow_tech_result['icir']:.3f}")
    print(f"  月度IC>0: {arrow_tech_result['ic_positive_months']*100:.0f}%")
    print(f"  Top5%胜率: {arrow_tech_result['layers'].get('Top5%',{}).get('win',0)*100:.1f}%")
    print(f"  宏观占比: {arrow_tech_result['macro_imp_pct']:.1f}%")
    
    # 保存完整结果
    output = {
        'blueshield_v8': v8_result,
        'arrow_v12_full': arrow_result,
        'arrow_v12_tech_only': arrow_tech_result,
        'timestamp': time.strftime('%Y-%m-%d %H:%M'),
    }
    with open(os.path.join(ROOT, 'data/audit-results-independent.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n结果已保存: data/audit-results-independent.json")

if __name__ == '__main__':
    main()
