#!/usr/bin/env python3
"""
信号阈值自动校准
用最近N天的实际分数分布和前瞻收益，计算最优信号阈值。
不依赖训练期的固定阈值。

用法:
    python3 calibrate_thresholds.py              # 蓝盾
    python3 calibrate_thresholds.py --model arrow # 绿箭
    python3 calibrate_thresholds.py --apply       # 直接写入meta文件
"""
import json, os, sys, argparse, time, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MACRO_COLS = ['vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60',
              'qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60',
              'iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60']
FUND_COLS = ['pe_trailing','pe_forward','div_yield','beta']

def calibrate(model_name='blueshield'):
    """校准指定模型的信号阈值"""
    
    if model_name == 'blueshield':
        model_path = os.path.join(ROOT, 'models/us/blueshield_v7_xgb.json')
        meta_path = os.path.join(ROOT, 'models/us/blueshield_v7_meta.json')
        price_min, price_max = 10, 1e6
        label = '蓝盾V7'
    else:
        model_path = os.path.join(ROOT, 'models/us/arrow_v12_xgb.json')
        meta_path = os.path.join(ROOT, 'models/us/arrow_v12_meta.json')
        price_min, price_max = 1, 10
        label = '绿箭V12'
    
    import xgboost as xgb
    
    # Import compute_features from the appropriate score script
    if model_name == 'blueshield':
        script = os.path.join(ROOT, 'scripts/us/blueshield_v6_score.py')
    else:
        script = os.path.join(ROOT, 'scripts/us/arrow_v11_score.py')
    
    import importlib.util
    spec = importlib.util.spec_from_file_location('score_mod', script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    print(f'🔧 {label} 阈值校准')
    print('='*50)
    
    # Load and compute features (6 months)
    print('1. 加载数据...')
    df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'))
    df = df.dropna(subset=['close', 'volume'])
    df = df[(df['close'] > 0.5) & (df['close'] < 10000) & (df['volume'] > 0)]
    cutoff = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
    df = df[df['date'] >= cutoff]
    
    print('2. 计算特征...')
    t0 = time.time()
    parts = []
    for sym, g in df.groupby('sym'):
        f = mod.compute_features(g); f['sym'] = sym; parts.append(f)
    df = pd.concat(parts, ignore_index=True)
    
    # Macro
    v75 = pd.read_parquet(os.path.join(ROOT, 'data/us/features/us_ml_feats_v75_filtered.parquet'))
    macro_daily = v75[['date']+MACRO_COLS].drop_duplicates(subset=['date'])
    df = pd.merge(df, macro_daily, on='date', how='left')
    for col in MACRO_COLS: df[col] = df[col].ffill().fillna(0)
    
    # Fund (blueshield only)
    if model_name == 'blueshield':
        fund_latest = v75.sort_values('date').groupby('sym').tail(1)[['sym']+FUND_COLS]
        df = pd.merge(df, fund_latest, on='sym', how='left')
        for col in FUND_COLS: df[col] = df[col].fillna(0)
    
    # Filter
    df = df[(df['close'] >= price_min) & (df['close'] < price_max)]
    df = df[df['volume'] > 50000]
    ALL_FEATS = mod.TECH_FEATS + MACRO_COLS + (FUND_COLS if model_name == 'blueshield' else [])
    df = df.dropna(subset=ALL_FEATS)
    print(f'   {df["sym"].nunique()}只, {len(df)}行 ({time.time()-t0:.0f}s)')
    
    # Score
    print('3. 评分...')
    model = xgb.Booster()
    model.load_model(model_path)
    with open(meta_path) as f: meta = json.load(f)
    feats = meta['features']
    X = df[feats].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    df['pred'] = model.predict(xgb.DMatrix(X, feature_names=feats))
    
    # Forward returns
    df = df.sort_values(['sym', 'date'])
    df['fwd_5d'] = df.groupby('sym')['close'].shift(-5) / df['close'] - 1
    df['fwd_20d'] = df.groupby('sym')['close'].shift(-20) / df['close'] - 1
    cutoff_fwd = (datetime.now() - timedelta(days=25)).strftime('%Y-%m-%d')
    df_eval = df[df['date'] < cutoff_fwd].dropna(subset=['fwd_5d'])
    
    # Current score distribution (latest day only)
    latest = df.sort_values('date').groupby('sym').tail(1)
    scores = latest['pred'].values
    
    print(f'\n4. 校准结果:')
    print(f'   当前分数分布: P50={np.percentile(scores,50):.4f} P90={np.percentile(scores,90):.4f} P95={np.percentile(scores,95):.4f} P99={np.percentile(scores,99):.4f} Max={scores.max():.4f}')
    
    # 校准方法: 用前瞻收益找最优阈值
    # 对每个可能的阈值，计算过该阈值的股票的前瞻收益
    thresholds = np.percentile(df_eval['pred'].values, np.arange(80, 100, 1))
    results = []
    for t in thresholds:
        above = df_eval[df_eval['pred'] >= t]
        if len(above) < 50: continue
        r5 = above['fwd_5d'].mean()
        wr5 = (above['fwd_5d'] > 0).mean()
        n = len(above)
        results.append({
            'threshold': t, 'pct': (df_eval['pred'] < t).mean() * 100,
            'n': n, 'avg_5d': r5, 'win_5d': wr5,
            'avg_20d': above['fwd_20d'].mean() if 'fwd_20d' in above else 0,
        })
    
    # Find optimal thresholds
    # green2: 最高平均收益的阈值 (Top ~5%)
    # green1: 正edge的最低阈值 (Top ~10-20%)
    # observe: 略正edge (Top ~20-30%)
    
    results_df = pd.DataFrame(results)
    
    # Find threshold where 5d avg > 0.5% (meaningful edge)
    positive_edge = results_df[results_df['avg_5d'] > 0.005]
    if len(positive_edge) > 0:
        observe_thresh = positive_edge['threshold'].min()
    else:
        observe_thresh = np.percentile(scores, 80)
    
    # green1: Top ~10% with avg > 1%
    g1_candidates = results_df[results_df['avg_5d'] > 0.01]
    if len(g1_candidates) > 0:
        green1_thresh = g1_candidates['threshold'].min()
    else:
        green1_thresh = np.percentile(scores, 90)
    
    # green2: Top ~5% - highest avg return
    top5 = results_df.nlargest(5, 'avg_5d')
    green2_thresh = top5['threshold'].min() if len(top5) > 0 else np.percentile(scores, 95)
    
    # Check actual performance at these thresholds
    for name, thresh in [('observe', observe_thresh), ('green1', green1_thresh), ('green2', green2_thresh)]:
        above = df_eval[df_eval['pred'] >= thresh]
        r5 = above['fwd_5d'].mean() * 100
        wr5 = (above['fwd_5d'] > 0).mean() * 100
        r20 = above['fwd_20d'].mean() * 100 if 'fwd_20d' in above else 0
        pct = (scores < thresh).mean() * 100
        print(f'   {name}: threshold={thresh:.4f} (Top{100-pct:.0f}%) → 5d={r5:+.2f}% win={wr5:.1f}% 20d={r20:+.2f}%')
    
    new_thresholds = {
        'green2': {
            'threshold': round(float(green2_thresh), 4),
            'min_win_rate': 0,
            'min_avg_return': 0,
            'note': f'自动校准 {datetime.now().strftime("%Y-%m-%d")}. 基于最近6个月实际前瞻收益.'
        },
        'green1': {
            'threshold': round(float(green1_thresh), 4),
            'min_win_rate': 0,
            'min_avg_return': 0,
            'note': f'自动校准 {datetime.now().strftime("%Y-%m-%d")}.'
        },
        'observe': {
            'threshold': round(float(observe_thresh), 4),
            'min_win_rate': 0,
            'min_avg_return': 0,
            'note': f'自动校准 {datetime.now().strftime("%Y-%m-%d")}.'
        }
    }
    
    return new_thresholds, meta_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='blueshield', choices=['blueshield', 'arrow'])
    parser.add_argument('--apply', action='store_true', help='直接写入meta文件')
    args = parser.parse_args()
    
    new_thresh, meta_path = calibrate(args.model)
    
    if args.apply:
        with open(meta_path) as f:
            meta = json.load(f)
        old = meta.get('signal_thresholds', {})
        meta['signal_thresholds'] = new_thresh
        meta['threshold_calibrated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'\n✅ 已写入 {meta_path}')
        print(f'   旧阈值: green2={old.get("green2",{}).get("threshold","?")} green1={old.get("green1",{}).get("threshold","?")}')
        print(f'   新阈值: green2={new_thresh["green2"]["threshold"]} green1={new_thresh["green1"]["threshold"]}')
    else:
        print(f'\n--apply 以写入meta文件')

if __name__ == '__main__':
    main()
