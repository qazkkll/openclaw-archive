#!/usr/bin/env python3
"""
蓝盾V8重训（内存优化版）
float32 + 分批计算特征 + 增量打分
"""
import gc, json, os, sys, time, warnings
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
    # 只保留需要的列
    return g[['date','sym','close'] + TECH_FEATS + ['volume']].astype({c: np.float32 for c in TECH_FEATS})

def main():
    print("="*70)
    print("蓝盾V8重训：纯技术特征，全市场（内存优化）")
    print("="*70)
    
    # 1. 加载数据
    print("\n1. 加载数据...", flush=True)
    t0 = time.time()
    df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'),
                         columns=['sym','date','open','high','low','close','volume'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close','volume'])
    df = df[(df['close'] > 10) & (df['close'] < 10000) & (df['volume'] > 0)]
    print(f"   {time.time()-t0:.0f}s, {len(df):,}行, {df['sym'].nunique()}只")
    
    # 2. 分批计算特征
    print("\n2. 分批计算特征...", flush=True)
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
            batch_df = pd.concat(batch_parts, ignore_index=True)
            feat_batches.append(batch_df)
        
        del batch_parts
        gc.collect()
        
        elapsed = time.time() - t0
        print(f"   批次 {batch_start//batch_size+1}: {batch_start}-{batch_end}/{total} ({elapsed:.0f}s)", flush=True)
    
    feat_all = pd.concat(feat_batches, ignore_index=True)
    del feat_batches
    gc.collect()
    print(f"   总计: {time.time()-t0:.0f}s, {len(feat_all):,}行, {feat_all['sym'].nunique()}只")
    
    # 3. 标签
    print("\n3. 构造标签...", flush=True)
    feat_all = feat_all.sort_values(['sym','date'])
    feat_all['fwd_ret20'] = feat_all.groupby('sym')['close'].pct_change(20).shift(-20)
    feat_all['hit'] = (feat_all['fwd_ret20'] > 0.02).astype(np.int8)
    
    valid = feat_all.dropna(subset=TECH_FEATS + ['hit'])
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=TECH_FEATS)
    print(f"   有效: {len(valid):,}, 正样本: {valid['hit'].mean()*100:.1f}%")
    
    # 4. 分割
    print("\n4. Walk-Forward分割...", flush=True)
    train_end = pd.Timestamp('2023-12-31')
    val_end = pd.Timestamp('2024-12-31')
    
    train_df = valid[valid['date'] <= train_end]
    val_df = valid[(valid['date'] > train_end) & (valid['date'] <= val_end)]
    test_df = valid[valid['date'] > val_end]
    
    print(f"   训练: {len(train_df):,} ({train_df['date'].min().date()} ~ {train_df['date'].max().date()})")
    print(f"   验证: {len(val_df):,} ({val_df['date'].min().date()} ~ {val_df['date'].max().date()})")
    print(f"   测试: {len(test_df):,} ({test_df['date'].min().date()} ~ {test_df['date'].max().date()})")
    
    # 5. 训练
    print("\n5. 训练...", flush=True)
    t0 = time.time()
    
    X_train = train_df[TECH_FEATS].values
    y_train = train_df['hit'].values
    X_val = val_df[TECH_FEATS].values
    y_val = val_df['hit'].values
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=6, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=10,
        n_estimators=200, tree_method='hist', device='cuda',
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"   完成: {time.time()-t0:.0f}s")
    
    # 释放训练数据
    del X_train, y_train, X_val, y_val
    gc.collect()
    
    # 6. 验证集评估
    print("\n6. 验证集评估...", flush=True)
    val_pred = model.predict_proba(val_df[TECH_FEATS].values)[:, 1]
    val_df = val_df.copy()
    val_df['pred_score'] = val_pred
    val_df['rank_pct'] = val_df.groupby('date')['pred_score'].rank(pct=True)
    
    for label, lo, hi in [('Top5%', 0.95, 1.0), ('Top10%', 0.90, 1.0),
                           ('Top20%', 0.80, 1.0), ('Mid60%', 0.20, 0.80),
                           ('Bot20%', 0.0, 0.20)]:
        sub = val_df[(val_df['rank_pct'] >= lo) & (val_df['rank_pct'] < hi + 0.001)]
        if len(sub) > 10:
            win = sub['hit'].mean()
            avg = sub['fwd_ret20'].mean()
            print(f"   {label:10s}: 胜率={win*100:.1f}%, 均收益={avg*100:+.2f}%, n={len(sub):,}")
    
    # 7. 全市场回测
    print("\n7. 全市场回测 (测试集)...", flush=True)
    t0 = time.time()
    
    all_dates = sorted(test_df['date'].unique())
    df_dates = pd.DataFrame({'date': all_dates})
    df_dates['ym'] = df_dates['date'].dt.to_period('M')
    bt_dates = df_dates.groupby('ym')['date'].last().sort_values().values[-24:]
    
    # 预计算测试集打分
    test_pred = model.predict_proba(test_df[TECH_FEATS].values)[:, 1]
    test_df = test_df.copy()
    test_df['pred_score'] = test_pred
    
    results = []
    for i, bt_date in enumerate(bt_dates):
        bt_date = pd.Timestamp(bt_date)
        
        window = test_df[test_df['date'] <= bt_date]
        latest_idx = window.groupby('sym')['date'].idxmax()
        score_df = window.loc[latest_idx].copy()
        
        if len(score_df) < 100:
            continue
        
        score_df['rank_pct'] = score_df['pred_score'].rank(pct=True)
        
        future_start = bt_date + pd.Timedelta(days=1)
        future_end = bt_date + pd.Timedelta(days=25)
        future = test_df[(test_df['date'] > future_start) & (test_df['date'] <= future_end)]
        
        for _, row in score_df.iterrows():
            sym = row['sym']
            base_price = row['close']
            sym_future = future[future['sym'] == sym].sort_values('date')
            
            if len(sym_future) < 5:
                continue
            
            fwd_idx = min(19, len(sym_future) - 1)
            fwd_ret = (sym_future.iloc[fwd_idx]['close'] - base_price) / base_price
            hit = 1 if fwd_ret > 0.02 else 0
            
            results.append({
                'date': bt_date.strftime('%Y-%m-%d'),
                'sym': sym,
                'pred_score': float(row['pred_score']),
                'rank_pct': float(row['rank_pct']),
                'fwd_ret': float(fwd_ret),
                'hit': hit,
            })
        
        n_r = len([r for r in results if r['date'] == bt_date.strftime('%Y-%m-%d')])
        print(f"   [{i+1}/{len(bt_dates)}] {bt_date.strftime('%Y-%m-%d')}: {n_r}只", flush=True)
    
    # 8. 结果
    res_df = pd.DataFrame(results)
    print(f"\n{'='*70}")
    print(f"全市场回测: 蓝盾V8 (纯技术, 27特征)")
    print(f"{'='*70}")
    print(f"记录: {len(res_df):,}, 股票: {res_df['sym'].nunique()}, 月份: {res_df['date'].nunique()}")
    
    print(f"\n分层胜率:")
    for label, lo, hi in [('Top5%', 0.95, 1.0), ('Top10%', 0.90, 1.0),
                           ('Top20%', 0.80, 1.0), ('Mid60%', 0.20, 0.80),
                           ('Bot20%', 0.0, 0.20)]:
        sub = res_df[(res_df['rank_pct'] >= lo) & (res_df['rank_pct'] < hi + 0.001)]
        if len(sub) > 10:
            win = sub['hit'].mean()
            avg = sub['fwd_ret'].mean()
            print(f"  {label:10s}: 胜率={win*100:.1f}%, 均收益={avg*100:+.2f}%, n={len(sub):,}")
    
    # 年度
    res_df['year'] = pd.to_datetime(res_df['date']).dt.year
    print(f"\n年度:")
    for year in sorted(res_df['year'].unique()):
        yr = res_df[res_df['year'] == year]
        t5 = yr[yr['rank_pct'] >= 0.95]
        b20 = yr[yr['rank_pct'] < 0.20]
        if len(t5) > 10:
            print(f"  {year}: Top5%胜率={t5['hit'].mean()*100:.1f}% 均收益={t5['fwd_ret'].mean()*100:+.2f}% "
                  f"| Bot20%胜率={b20['hit'].mean()*100:.1f}% 均收益={b20['fwd_ret'].mean()*100:+.2f}%")
    
    # 特征重要性
    print(f"\n特征重要性 Top10:")
    imp = dict(zip(TECH_FEATS, model.feature_importances_))
    for f, v in sorted(imp.items(), key=lambda x: -x[1])[:10]:
        print(f"  {f:20s} {v:.4f}")
    
    # 保存
    model.save_model(os.path.join(ROOT, 'models/us/blueshield_v8_xgb.json'))
    
    meta = {
        'version': 'blueshield_v8', 'algorithm': 'XGBoost',
        'features': TECH_FEATS, 'n_features': len(TECH_FEATS),
        'macro_features': 0, 'hold_days': 20,
        'universe': '全市场>$10', 'n_stocks': int(feat_all['sym'].nunique()),
        'trained_on': f'{train_df["date"].min().date()} ~ {train_df["date"].max().date()}',
        'n_train': len(train_df),
        'feature_importance': {k: round(v, 4) for k, v in sorted(imp.items(), key=lambda x: -x[1])[:15]},
        'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'replaces': 'blueshield_v7',
    }
    with open(os.path.join(ROOT, 'models/us/blueshield_v8_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    # 最终判断
    t5 = res_df[res_df['rank_pct'] >= 0.95]
    b20 = res_df[res_df['rank_pct'] < 0.20]
    print(f"\n{'='*70}")
    if t5['hit'].mean() > b20['hit'].mean() and t5['fwd_ret'].mean() > b20['fwd_ret'].mean():
        print("✅ V8排名方向正确 (Top > Bot)，模型可用")
    elif t5['hit'].mean() > 0.5:
        print("⚠️ 胜率>50%但排序需优化")
    else:
        print("❌ 排序仍有问题，需继续迭代")

if __name__ == '__main__':
    main()
