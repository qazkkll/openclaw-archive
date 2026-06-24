#!/usr/bin/env python3
"""
全市场Walk-Forward回测（优化版）
加载一次数据 → 全量计算特征 → 逐月打分
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
ROOT = '/home/hermes/.hermes/openclaw-archive'

MACRO_COLS = ['vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60',
              'qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60',
              'iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60']
FUND_COLS = ['pe_trailing','pe_forward','div_yield','beta']
TECH_FEATS = ['ma5','ma20','ma60','ma_bias20','ma_align','price_position',
    'ret1','ret5','ret20','ret60','momentum_6m','momentum_1m',
    'mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change',
    'rsi14','rsi_change','macd','macd_signal','macd_hist',
    'bb_std','bb_width','bb_pos','ret_quality']
ALL_FEATS = TECH_FEATS + MACRO_COLS + FUND_COLS

def compute_features(g):
    g = g.sort_values('date').copy()
    c = g['close']
    g['ma5'] = c.rolling(5).mean(); g['ma20'] = c.rolling(20).mean(); g['ma60'] = c.rolling(60).mean()
    g['ma_bias20'] = (c - g['ma20']) / g['ma20']
    g['ma_align'] = ((c > g['ma5']).astype(int) + (g['ma5'] > g['ma20']).astype(int))
    mn60 = c.rolling(60).min(); mx60 = c.rolling(60).max()
    g['price_position'] = (c - mn60) / (mx60 - mn60 + 1e-10)
    g['ret1'] = c.pct_change(1); g['ret5'] = c.pct_change(5)
    g['ret20'] = c.pct_change(20); g['ret60'] = c.pct_change(60)
    g['momentum_6m'] = c.pct_change(126); g['momentum_1m'] = c.pct_change(21)
    g['mom_divergence'] = g['momentum_1m'] - g['ret20']
    g['trend_accel'] = g['ret5'] - g['ret5'].shift(5)
    dr = c.pct_change(1)
    g['vol20'] = dr.rolling(20).std(); g['vol5'] = dr.rolling(5).std()
    g['vol_ratio'] = g['volume'] / g['volume'].rolling(20).mean()
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
    return g

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='blueshield', choices=['blueshield', 'arrow'])
    parser.add_argument('--months', type=int, default=36)
    parser.add_argument('--sample', type=int, default=2000, help='随机采样股票数(0=全量)')
    args = parser.parse_args()
    
    model_name = args.model
    n_months = args.months
    sample_n = args.sample
    
    if model_name == 'blueshield':
        model_path = os.path.join(ROOT, 'models/us/blueshield_v7_xgb.json')
        price_min = 10; hold_days = 20
    else:
        model_path = os.path.join(ROOT, 'models/us/arrow_v12_xgb.json')
        price_min = 1; hold_days = 5
    
    # 加载模型
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    feat_names = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else ALL_FEATS
    
    print(f"{'='*70}")
    print(f"全市场Walk-Forward回测: {model_name}")
    print(f"{'='*70}")
    print(f"模型: {model_path}")
    print(f"Hold: {hold_days}天 | Price: >${price_min} | Sample: {sample_n or '全量'}")
    
    # 1. 加载数据（只加载一次）
    print(f"\n1. 加载数据...", flush=True)
    t0 = time.time()
    df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'))
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close', 'volume'])
    df = df[(df['close'] > 0.5) & (df['close'] < 10000) & (df['volume'] > 0)]
    
    # 取最近5年的数据（回测期+特征窗口）
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365*n_months//12 + 400)
    df = df[df['date'] >= cutoff]
    print(f"   加载: {time.time()-t0:.0f}s, {len(df):,}行, {df['sym'].nunique()}只")
    
    # 采样
    all_syms = df['sym'].unique()
    if sample_n > 0 and len(all_syms) > sample_n:
        np.random.seed(42)
        sampled = np.random.choice(all_syms, sample_n, replace=False)
        df = df[df['sym'].isin(sampled)]
        print(f"   采样: {sample_n}只, {len(df):,}行")
    
    # 2. 计算特征（只算一次）
    print(f"\n2. 计算特征...", flush=True)
    t0 = time.time()
    parts = []
    n_syms = df['sym'].nunique()
    for i, (sym, g) in enumerate(df.groupby('sym')):
        if len(g) < 130:
            continue
        f = compute_features(g)
        parts.append(f)
        if (i+1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (n_syms - i - 1)
            print(f"   {i+1}/{n_syms} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)
    
    feat_all = pd.concat(parts, ignore_index=True)
    print(f"   完成: {time.time()-t0:.0f}s, {len(feat_all):,}行, {feat_all['sym'].nunique()}只")
    
    # 3. 确定回测日期（每月最后一个交易日）
    all_dates = sorted(feat_all['date'].unique())
    df_dates = pd.DataFrame({'date': all_dates})
    df_dates['ym'] = df_dates['date'].dt.to_period('M')
    bt_dates = df_dates.groupby('ym')['date'].last().sort_values().values[-n_months:]
    
    print(f"\n3. Walk-forward打分 ({len(bt_dates)}个月)...")
    
    # 4. 逐月打分
    results = []
    for i, bt_date in enumerate(bt_dates):
        bt_date = pd.Timestamp(bt_date)
        
        # 取每个股票在bt_date当天或之前最近的行
        score_df = feat_all[feat_all['date'] <= bt_date].copy()
        latest_idx = score_df.groupby('sym')['date'].idxmax()
        score_df = score_df.loc[latest_idx].copy()
        
        # 价格筛选
        score_df = score_df[score_df['close'] > price_min]
        
        if len(score_df) < 100:
            continue
        
        # 补缺失特征列
        for col in feat_names:
            if col not in score_df.columns:
                score_df[col] = 0
        
        # 打分
        try:
            X = score_df[feat_names].fillna(0).replace([np.inf, -np.inf], 0)
            scores = model.predict_proba(X)[:, 1]
            score_df['pred_score'] = scores
        except Exception as e:
            print(f"   [{i+1}] {bt_date.strftime('%Y-%m-%d')} 打分失败: {e}")
            continue
        
        score_df['rank_pct'] = score_df['pred_score'].rank(pct=True)
        
        # 计算未来收益
        future_start = bt_date + pd.Timedelta(days=1)
        future_end = bt_date + pd.Timedelta(days=hold_days + 15)
        future = feat_all[(feat_all['date'] > future_start) & (feat_all['date'] <= future_end)]
        
        for _, row in score_df.iterrows():
            sym = row['sym']
            base_price = row['close']
            sym_future = future[future['sym'] == sym].sort_values('date')
            
            if len(sym_future) < min(hold_days, 5):
                continue
            
            fwd_idx = min(hold_days - 1, len(sym_future) - 1)
            fwd_ret = (sym_future.iloc[fwd_idx]['close'] - base_price) / base_price
            hit = 1 if fwd_ret > 0.02 else 0
            
            results.append({
                'date': bt_date.strftime('%Y-%m-%d'),
                'sym': sym,
                'pred_score': row['pred_score'],
                'rank_pct': row['rank_pct'],
                'fwd_ret': fwd_ret,
                'hit': hit,
                'base_price': base_price,
            })
        
        n_scored = len(score_df)
        n_results = len([r for r in results if r['date'] == bt_date.strftime('%Y-%m-%d')])
        print(f"   [{i+1}/{len(bt_dates)}] {bt_date.strftime('%Y-%m-%d')}: "
              f"评分{n_scored}只, 有结果{n_results}只", flush=True)
    
    # 5. 分析
    res_df = pd.DataFrame(results)
    print(f"\n{'='*70}")
    print(f"全市场回测结果: {model_name}")
    print(f"{'='*70}")
    print(f"总记录: {len(res_df):,}")
    print(f"股票数: {res_df['sym'].nunique()}")
    print(f"月份: {res_df['date'].nunique()}")
    
    print(f"\n{'='*70}")
    print(f"分层胜率（全局）")
    print(f"{'='*70}")
    
    for label, lo, hi in [('Top5%', 0.95, 1.0), ('Top10%', 0.90, 1.0), 
                           ('Top20%', 0.80, 1.0), ('Mid60%', 0.20, 0.80),
                           ('Bot20%', 0.0, 0.20)]:
        sub = res_df[(res_df['rank_pct'] >= lo) & (res_df['rank_pct'] < hi + 0.001)]
        if len(sub) > 10:
            win = sub['hit'].mean()
            avg = sub['fwd_ret'].mean()
            med = sub['fwd_ret'].median()
            print(f"  {label:10s}: 胜率={win*100:.1f}%, 均收益={avg*100:+.2f}%, "
                  f"中位数={med*100:+.2f}%, n={len(sub):,}")
    
    # 年度
    print(f"\n{'='*70}")
    print(f"年度分析")
    print(f"{'='*70}")
    res_df['year'] = pd.to_datetime(res_df['date']).dt.year
    
    header = f"  {'年份':6s} {'Top5%胜率':>10s} {'Top5%收益':>10s} {'Top20%胜率':>10s} {'Bot20%胜率':>10s} {'样本':>8s}"
    print(header)
    
    for year in sorted(res_df['year'].unique()):
        yr = res_df[res_df['year'] == year]
        t5 = yr[yr['rank_pct'] >= 0.95]
        t20 = yr[yr['rank_pct'] >= 0.80]
        b20 = yr[yr['rank_pct'] < 0.20]
        
        if len(t5) > 10:
            print(f"  {year:6d} {t5['hit'].mean()*100:9.1f}% {t5['fwd_ret'].mean()*100:+9.2f}% "
                  f"{t20['hit'].mean()*100:9.1f}% {b20['hit'].mean()*100:9.1f}% {len(yr):>8,}")
    
    # 保存
    out_path = os.path.join(ROOT, f'data/backtest-rounds/full-market-{model_name}-bt.json')
    summary = {
        'model': model_name,
        'hold_days': hold_days,
        'period': f'{bt_dates[0]} ~ {bt_dates[-1]}',
        'total_records': len(res_df),
        'unique_stocks': int(res_df['sym'].nunique()),
        'months': int(res_df['date'].nunique()),
        'top5': {
            'win_rate': float(res_df[res_df['rank_pct']>=0.95]['hit'].mean()),
            'avg_ret': float(res_df[res_df['rank_pct']>=0.95]['fwd_ret'].mean()),
            'count': int(len(res_df[res_df['rank_pct']>=0.95])),
        },
        'top20': {
            'win_rate': float(res_df[res_df['rank_pct']>=0.80]['hit'].mean()),
            'avg_ret': float(res_df[res_df['rank_pct']>=0.80]['fwd_ret'].mean()),
        },
        'bot20': {
            'win_rate': float(res_df[res_df['rank_pct']<0.20]['hit'].mean()),
            'avg_ret': float(res_df[res_df['rank_pct']<0.20]['fwd_ret'].mean()),
        },
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n保存: {out_path}")

if __name__ == '__main__':
    main()
