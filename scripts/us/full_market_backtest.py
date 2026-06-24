#!/usr/bin/env python3
"""
全市场Walk-Forward回测
加载11,864只股票 → 逐月打分 → 计算Top5%/10%/20%胜率
"""
import json, os, sys, time, warnings, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
ROOT = '/home/hermes/.hermes/openclaw-archive'

# 特征定义（与评分脚本一致）
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

def compute_macro_features(df):
    """从SPY/QQQ/IWM计算宏观特征"""
    spy = df[df['sym'] == 'SPY'].set_index('date')['close']
    qqq = df[df['sym'] == 'QQQ'].set_index('date')['close']
    iwm = df[df['sym'] == 'IWM'].set_index('date')['close']
    
    if len(spy) < 60:
        return df
    
    for col, prices in [('spy', spy), ('qqq', qqq), ('iwm', iwm)]:
        for d, suffix in [(1,'1'),(5,'5'),(20,'20'),(60,'60')]:
            ret = prices.pct_change(d)
            df[f'{col}_ret{suffix}'] = df['date'].map(ret)
    
    vix_like = (spy.pct_change(1).rolling(20).std() * np.sqrt(252) * 100)
    df['vix_close'] = df['date'].map(vix_like)
    
    return df

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='blueshield', choices=['blueshield', 'arrow'])
    parser.add_argument('--months', type=int, default=36, help='回测月数')
    parser.add_argument('--sample', type=int, default=0, help='每期采样股票数(0=全量)')
    args = parser.parse_args()
    
    model_name = args.model
    n_months = args.months
    sample_n = args.sample
    
    # 加载模型
    if model_name == 'blueshield':
        model_path = os.path.join(ROOT, 'models/us/blueshield_v7_xgb.json')
        price_min = 10
        hold_days = 20
    else:
        model_path = os.path.join(ROOT, 'models/us/arrow_v12_xgb.json')
        price_min = 1
        hold_days = 5
    
    print(f"{'='*70}")
    print(f"全市场Walk-Forward回测: {model_name}")
    print(f"{'='*70}")
    
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print(f"模型加载: {model_path}")
    print(f"特征数: {len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 'unknown'}")
    
    # 加载全量数据
    print(f"\n1. 加载全量数据...")
    t0 = time.time()
    df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'))
    df = df.dropna(subset=['close', 'volume'])
    df = df[(df['close'] > 0.5) & (df['close'] < 10000) & (df['volume'] > 0)]
    df['date'] = pd.to_datetime(df['date'])
    print(f"   加载: {time.time()-t0:.0f}s, {len(df)}行, {df['sym'].nunique()}只")
    
    # 确定回测月份
    all_dates = sorted(df['date'].unique())
    latest = all_dates[-1]
    
    # 每月取一个日期（每月第一个交易日）
    df_dates = pd.DataFrame({'date': all_dates})
    df_dates['ym'] = df_dates['date'].dt.to_period('M')
    month_ends = df_dates.groupby('ym')['date'].last().values
    bt_dates = sorted(month_ends)[-n_months:]
    
    print(f"\n2. 回测参数:")
    print(f"   模型: {model_name} (hold={hold_days}天)")
    print(f"   价格门槛: >${price_min}")
    print(f"   回测期: {bt_dates[0]} ~ {bt_dates[-1]} ({len(bt_dates)}个月)")
    print(f"   每期计算特征+打分...")
    
    # 加载宏观/基本面特征（如果有的话）
    macro_df = None
    macro_path = os.path.join(ROOT, 'data/us/features/us_ml_feats_v75_filtered.parquet')
    if os.path.exists(macro_path):
        macro_df = pd.read_parquet(macro_path)
        macro_df['date'] = pd.to_datetime(macro_df['date'])
        print(f"   宏观特征: {len(macro_df)}行")
    
    # Walk-forward回测
    results = []
    for i, bt_date in enumerate(bt_dates):
        bt_date = pd.Timestamp(bt_date)
        print(f"\r   [{i+1}/{len(bt_dates)}] {bt_date.strftime('%Y-%m-%d')}...", end='', flush=True)
        
        # 取bt_date前250天的数据来计算特征
        start = bt_date - timedelta(days=365)
        window = df[(df['date'] >= start) & (df['date'] <= bt_date)].copy()
        
        if len(window) < 10000:
            continue
        
        # 价格筛选
        latest_prices = window.groupby('sym')['close'].last()
        valid_syms = latest_prices[latest_prices > price_min].index
        window = window[window['sym'].isin(valid_syms)]
        
        if sample_n > 0 and len(valid_syms) > sample_n:
            sampled = np.random.choice(valid_syms, sample_n, replace=False)
            window = window[window['sym'].isin(sampled)]
        
        # 计算特征
        parts = []
        for sym, g in window.groupby('sym'):
            if len(g) < 130:
                continue
            f = compute_features(g)
            parts.append(f)
        
        if not parts:
            continue
        
        feat_df = pd.concat(parts, ignore_index=True)
        
        # 取每个股票最新一行
        latest_idx = feat_df.groupby('sym')['date'].idxmax()
        score_df = feat_df.loc[latest_idx].copy()
        
        # 补宏观特征（用SPY/QQQ/IWM的当前值填充所有行）
        for col in MACRO_COLS + FUND_COLS:
            if col not in score_df.columns:
                score_df[col] = 0
        
        # 确保特征列存在
        feat_cols = [c for c in ALL_FEATS if c in score_df.columns]
        missing = [c for c in ALL_FEATS if c not in score_df.columns]
        if missing:
            for m in missing:
                score_df[m] = 0
        
        # 打分
        try:
            X = score_df[ALL_FEATS].fillna(0).replace([np.inf, -np.inf], 0)
            scores = model.predict_proba(X)[:, 1]
            score_df['pred_score'] = scores
        except Exception as e:
            print(f"\n   模型打分失败: {e}")
            continue
        
        # 排名
        score_df['rank_pct'] = score_df['pred_score'].rank(pct=True)
        
        # 计算未来收益（需要后续数据）
        future = df[(df['date'] > bt_date) & (df['date'] <= bt_date + timedelta(days=hold_days+10))]
        
        for _, row in score_df.iterrows():
            sym = row['sym']
            base_price = row['close']
            base_score = row['pred_score']
            rank_pct = row['rank_pct']
            
            sym_future = future[future['sym'] == sym].sort_values('date')
            if len(sym_future) < hold_days:
                continue
            
            fwd_price = sym_future.iloc[hold_days-1]['close']
            fwd_ret = (fwd_price - base_price) / base_price
            hit = 1 if fwd_ret > 0.02 else 0
            
            results.append({
                'date': bt_date.strftime('%Y-%m-%d'),
                'sym': sym,
                'pred_score': base_score,
                'rank_pct': rank_pct,
                'fwd_ret': fwd_ret,
                'hit': hit,
                'base_price': base_price,
            })
    
    print(f"\n\n3. 分析结果...")
    
    if not results:
        print("   没有结果！")
        return
    
    res_df = pd.DataFrame(results)
    print(f"   总记录: {len(res_df)}")
    print(f"   股票数: {res_df['sym'].nunique()}")
    print(f"   月数: {res_df['date'].nunique()}")
    
    # 按分数分桶
    print(f"\n{'='*70}")
    print(f"全市场回测结果 ({model_name})")
    print(f"{'='*70}")
    
    res_df['score_bin'] = pd.cut(res_df['rank_pct'], 
        bins=[0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 1.0],
        labels=['Bot5%','Bot5-10%','Bot10-20%','Bot20-40%','Mid40-60%',
                'Top40-20%','Top20-10%','Top10-5%','Top5%'])
    
    for bin_name in ['Top5%','Top10-5%','Top20-10%','Top40-20%','Mid40-60%',
                     'Bot20-40%','Bot10-20%','Bot5-10%','Bot5%']:
        sub = res_df[res_df['score_bin'] == bin_name]
        if len(sub) > 10:
            win = sub['hit'].mean()
            avg = sub['fwd_ret'].mean()
            med = sub['fwd_ret'].median()
            print(f"  {bin_name:12s}: 胜率={win*100:5.1f}%, 均收益={avg*100:+6.2f}%, "
                  f"中位数={med*100:+.2f}%, n={len(sub):>6}")
    
    # 直接按Top5/10/20%统计
    print(f"\n{'='*70}")
    print(f"简洁版")
    print(f"{'='*70}")
    
    for label, lo, hi in [('Top5%', 0.95, 1.0), ('Top10%', 0.90, 1.0), 
                           ('Top20%', 0.80, 1.0), ('Mid', 0.20, 0.80),
                           ('Bot20%', 0.0, 0.20)]:
        sub = res_df[(res_df['rank_pct'] >= lo) & (res_df['rank_pct'] < hi)]
        if len(sub) > 10:
            win = sub['hit'].mean()
            avg = sub['fwd_ret'].mean()
            print(f"  {label:10s}: 胜率={win*100:.1f}%, 均收益={avg*100:+.2f}%, n={len(sub)}")
    
    # 年度分析
    print(f"\n{'='*70}")
    print(f"年度分析 (Top5%)")
    print(f"{'='*70}")
    
    res_df['year'] = pd.to_datetime(res_df['date']).dt.year
    top5 = res_df[res_df['rank_pct'] >= 0.95]
    
    for year in sorted(res_df['year'].unique()):
        sub = top5[top5['year'] == year]
        if len(sub) > 10:
            win = sub['hit'].mean()
            avg = sub['fwd_ret'].mean()
            print(f"  {year}: 胜率={win*100:.1f}%, 均收益={avg*100:+.2f}%, n={len(sub)}")
    
    # 保存结果
    out_path = os.path.join(ROOT, f'data/backtest-rounds/full-market-{model_name}-bt.json')
    with open(out_path, 'w') as f:
        json.dump({
            'model': model_name,
            'period': f'{bt_dates[0]} ~ {bt_dates[-1]}',
            'total_records': len(res_df),
            'unique_stocks': res_df['sym'].nunique(),
            'results': results[:1000],  # 只保存前1000条（太大）
            'summary': {
                'top5_win': float(res_df[res_df['rank_pct']>=0.95]['hit'].mean()),
                'top5_avg_ret': float(res_df[res_df['rank_pct']>=0.95]['fwd_ret'].mean()),
                'bot20_win': float(res_df[res_df['rank_pct']<0.2]['hit'].mean()),
                'bot20_avg_ret': float(res_df[res_df['rank_pct']<0.2]['fwd_ret'].mean()),
            }
        }, f, indent=2)
    print(f"\n保存: {out_path}")

if __name__ == '__main__':
    main()
EOF