#!/usr/bin/env python3
"""
模型极致优化实验Runner
每轮实验：Walk-Forward验证 + IC/ICIR + 分层收益 + OOS评估
用法: python3 experiment_runner.py --exp <experiment_name>
"""
import json, os, sys, time, warnings, argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

warnings.filterwarnings('ignore')
ROOT = '/home/hermes/.hermes/openclaw-archive'
DATA_DIR = os.path.join(ROOT, 'data', 'us')
MODEL_DIR = os.path.join(ROOT, 'models', 'us')
RESULT_DIR = os.path.join(ROOT, 'data', 'experiments')
os.makedirs(RESULT_DIR, exist_ok=True)

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

# ============================================================
# 特征计算
# ============================================================
def compute_tech_features(g):
    c = g['close']
    g['ma5'] = c.rolling(5).mean(); g['ma20'] = c.rolling(20).mean(); g['ma60'] = c.rolling(60).mean()
    g['ma_bias20'] = (c - g['ma20']) / g['ma20']
    g['ma_align'] = ((c > g['ma5']).astype(int) + (g['ma5'] > g['ma20']).astype(int))
    mn60 = c.rolling(60).min(); mx60 = c.rolling(60).max()
    g['price_position'] = (c - mn60) / (mx60 - mn60 + 1e-10)
    g['ret1'] = c.pct_change(1); g['ret5'] = c.pct_change(5); g['ret20'] = c.pct_change(20); g['ret60'] = c.pct_change(60)
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
    ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
    g['macd'] = ema12 - ema26; g['macd_signal'] = g['macd'].ewm(span=9).mean()
    g['macd_hist'] = g['macd'] - g['macd_signal']
    g['bb_std'] = dr.rolling(20).std()
    bb_mid = c.rolling(20).mean()
    g['bb_width'] = 4 * g['bb_std'] * bb_mid / (bb_mid + 1e-10)
    g['bb_pos'] = (c - (bb_mid - 2 * c.rolling(20).std())) / (4 * c.rolling(20).std() + 1e-10)
    ret_pos = dr.clip(lower=0).rolling(20).mean()
    ret_neg = (-dr).clip(lower=0).rolling(20).mean()
    g['ret_quality'] = ret_pos / (ret_pos + ret_neg + 1e-10)
    g['price'] = c
    g['range_pct'] = (g['high'] - g['low']) / (c + 1e-10)
    # 资金流特征（新增）
    g['cmf'] = ((c - g['low']) - (g['high'] - c)) / (g['high'] - g['low'] + 1e-10)
    g['cmf'] = (g['cmf'] * g['volume']).rolling(20).sum() / g['volume'].rolling(20).sum()
    obv = (np.sign(c.diff()) * g['volume']).fillna(0).cumsum()
    g['obv_slope'] = obv.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0, raw=False)
    g['vol_price_corr'] = c.rolling(20).corr(g['volume'])
    return g

TECH_FEATS = ['ma5', 'ma20', 'ma60', 'ma_bias20', 'ma_align', 'price_position',
    'ret1', 'ret5', 'ret20', 'ret60', 'momentum_6m', 'momentum_1m',
    'mom_divergence', 'trend_accel', 'vol20', 'vol5', 'vol_ratio', 'vol_change',
    'rsi14', 'rsi_change', 'macd', 'macd_signal', 'macd_hist',
    'bb_std', 'bb_width', 'bb_pos', 'ret_quality']
EXTRA_FEATS = ['price', 'range_pct', 'cmf', 'obv_slope', 'vol_price_corr']
MACRO_FEATS = ['vix_close', 'spy_ret1', 'spy_ret5', 'spy_ret20', 'spy_ret60',
    'qqq_ret1', 'qqq_ret5', 'qqq_ret20', 'qqq_ret60',
    'iwm_ret1', 'iwm_ret5', 'iwm_ret20', 'iwm_ret60']

# ============================================================
# 数据加载
# ============================================================
def load_all_data():
    log('加载数据...')
    df = pd.read_parquet(os.path.join(DATA_DIR, 'us_hist_full_10y.parquet'))
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close', 'volume'])
    df = df[(df['close'] > 0.5) & (df['volume'] > 0)]
    
    # VIX
    vix_df = pd.read_parquet(os.path.join(DATA_DIR, 'vix_10y.parquet'))
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = [c[0] for c in vix_df.columns]
    vix_df = vix_df.reset_index()
    vix_df.columns = [c.lower().replace('ticker','') for c in vix_df.columns]
    if 'date' not in vix_df.columns:
        vix_df = vix_df.rename(columns={vix_df.columns[0]: 'date'})
    vix_df['date'] = pd.to_datetime(vix_df['date'])
    vix_close_col = [c for c in vix_df.columns if 'close' in c.lower()]
    if vix_close_col:
        vix_df = vix_df.rename(columns={vix_close_col[0]: 'vix_close'})
    vix_df = vix_df[['date', 'vix_close']].dropna()
    
    # 宏观ETF
    spy = df[df['sym'] == 'SPY'][['date', 'close']].copy()
    qqq = df[df['sym'] == 'QQQ'][['date', 'close']].copy()
    iwm = df[df['sym'] == 'IWM'][['date', 'close']].copy()
    macro = pd.DataFrame({'date': spy['date']})
    for name, series in [('spy', spy['close']), ('qqq', qqq['close']), ('iwm', iwm['close'])]:
        macro[f'{name}_ret1'] = series.pct_change(1)
        macro[f'{name}_ret5'] = series.pct_change(5)
        macro[f'{name}_ret20'] = series.pct_change(20)
        macro[f'{name}_ret60'] = series.pct_change(60)
    macro = macro.merge(vix_df, on='date', how='left')
    macro['vix_close'] = macro['vix_close'].ffill().fillna(20)
    
    log(f'  {df["sym"].nunique()}只, {len(df):,}行')
    return df, macro

# ============================================================
# 特征工程
# ============================================================
def build_features(df, macro, price_range, hold_days, feat_set='full'):
    df = df.copy()
    if price_range == 'blueshield':
        last_prices = df.groupby('sym')['close'].last()
        valid_syms = last_prices[last_prices > 10].index
    else:
        last_prices = df.groupby('sym')['close'].last()
        valid_syms = last_prices[(last_prices >= 1) & (last_prices <= 10)].index
    df = df[df['sym'].isin(valid_syms)]
    
    log(f'  特征工程: {df["sym"].nunique()}只')
    t0 = time.time()
    df = df.sort_values(['sym', 'date'])
    groups = []
    for sym, group in df.groupby('sym'):
        if len(group) < 80: continue
        groups.append(compute_tech_features(group.copy()))
    df = pd.concat(groups, ignore_index=True)
    
    df = df.merge(macro, on='date', how='left')
    for col in MACRO_FEATS:
        if col not in df.columns: df[col] = 0
        df[col] = df[col].ffill().fillna(0)
    for col in EXTRA_FEATS:
        if col not in df.columns: df[col] = 0
    
    # 标签
    orig_sym = df['sym'].values.copy()
    def calc_fwd(group):
        group = group.sort_values('date')
        group['fwd_ret'] = group['close'].shift(-hold_days) / group['close'] - 1
        return group
    df = df.groupby('sym', group_keys=False).apply(calc_fwd)
    df = df.reset_index(drop=True)
    if 'sym' not in df.columns:
        df['sym'] = orig_sym[:len(df)]
    df = df.dropna(subset=['fwd_ret'])
    
    etf_syms = {'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'IVV', 'VEA', 'VWO',
                'BND', 'AGG', 'TLT', 'GLD', 'SLV', 'USO', 'XLE', 'XLF', 'XLK', 'XLV'}
    df = df[~df['sym'].isin(etf_syms)]
    
    # 特征集选择
    if feat_set == 'full':
        feat_cols = TECH_FEATS + EXTRA_FEATS + MACRO_FEATS
    elif feat_set == 'tech_macro':
        feat_cols = TECH_FEATS + MACRO_FEATS
    elif feat_set == 'tech_only':
        feat_cols = TECH_FEATS + EXTRA_FEATS
    elif feat_set == 'tech_flow':
        feat_cols = TECH_FEATS + ['cmf', 'obv_slope', 'vol_price_corr']
    else:
        feat_cols = TECH_FEATS + EXTRA_FEATS + MACRO_FEATS
    
    for col in feat_cols:
        if col not in df.columns: df[col] = 0
    feat_present = df[feat_cols].notna().sum(axis=1)
    df = df[feat_present >= len(feat_cols) * 0.8]
    for col in feat_cols:
        if df[col].isna().any(): df[col] = df[col].fillna(df[col].median())
    
    log(f'  完成: {time.time()-t0:.1f}s, {len(df):,}行, {len(feat_cols)}特征')
    return df, feat_cols

# ============================================================
# Walk-Forward验证（4折，避免最后一折数据不足）
# ============================================================
def walk_forward_eval(df, feat_cols, hold_days, params, n_folds=4, top_n=5):
    df = df.sort_values('date')
    dates = np.sort(df['date'].unique())
    oos_start = pd.Timestamp('2024-01-01')
    train_dates = dates[dates < oos_start]
    
    # 4折，每折~470天
    fold_size = len(train_dates) // (n_folds + 1)  # 留最后一段作为buffer
    wf_results = []
    
    for fold in range(n_folds):
        train_end_idx = (fold + 1) * fold_size
        val_start_idx = train_end_idx
        val_end_idx = val_start_idx + fold_size
        if val_end_idx > len(train_dates):
            val_end_idx = len(train_dates)
        if val_end_idx <= val_start_idx:
            continue
        
        train_mask = df['date'].isin(train_dates[:train_end_idx])
        val_mask = df['date'].isin(train_dates[val_start_idx:val_end_idx])
        
        X_train = np.nan_to_num(df.loc[train_mask, feat_cols].values, nan=0, posinf=0, neginf=0).astype(np.float32)
        y_train = df.loc[train_mask, 'fwd_ret'].values
        X_val = np.nan_to_num(df.loc[val_mask, feat_cols].values, nan=0, posinf=0, neginf=0).astype(np.float32)
        y_val = df.loc[val_mask, 'fwd_ret'].values
        
        if len(X_train) < 1000 or len(X_val) < 100:
            continue
        
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_cols)
        model = xgb.train(params, dtrain, num_boost_round=500,
                         evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
        pred = model.predict(dval)
        
        val_df = df.loc[val_mask].copy()
        val_df['pred'] = pred
        
        ic_values = []
        for d, group in val_df.groupby('date'):
            if len(group) < 20: continue
            ic, _ = stats.spearmanr(group['pred'], group['fwd_ret'])
            if not np.isnan(ic): ic_values.append(ic)
        
        if ic_values:
            ic_mean = np.mean(ic_values)
            icir = ic_mean / (np.std(ic_values) + 1e-10)
            ic_pos = np.mean([x > 0 for x in ic_values])
            
            spreads = []
            for d, group in val_df.groupby('date'):
                if len(group) < top_n * 2: continue
                n5 = max(1, int(len(group) * 0.05))
                n20 = max(1, int(len(group) * 0.20))
                spreads.append(group.nlargest(n5, 'pred')['fwd_ret'].mean() - group.nsmallest(n20, 'pred')['fwd_ret'].mean())
            
            val_start = pd.Timestamp(train_dates[val_start_idx]).date()
            val_end = pd.Timestamp(train_dates[val_end_idx-1]).date()
            wf_results.append({
                'fold': fold, 'ic': round(ic_mean, 4), 'icir': round(icir, 3),
                'ic_pos': round(ic_pos, 3), 'spread': round(np.mean(spreads)*100, 2),
                'n_days': len(ic_values), 'best_iter': model.best_iteration,
                'period': f'{val_start}~{val_end}'
            })
    
    return wf_results

# ============================================================
# OOS评估
# ============================================================
def oos_eval(df, feat_cols, hold_days, params, best_iter, top_n=5):
    df = df.sort_values('date')
    oos_start = pd.Timestamp('2024-01-01')
    train_mask = df['date'] < oos_start
    oos_mask = df['date'] >= oos_start
    
    X_train = np.nan_to_num(df.loc[train_mask, feat_cols].values, nan=0, posinf=0, neginf=0).astype(np.float32)
    y_train = df.loc[train_mask, 'fwd_ret'].values
    X_oos = np.nan_to_num(df.loc[oos_mask, feat_cols].values, nan=0, posinf=0, neginf=0).astype(np.float32)
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_cols)
    doos = xgb.DMatrix(X_oos, feature_names=feat_cols)
    model = xgb.train(params, dtrain, num_boost_round=best_iter, verbose_eval=False)
    pred = model.predict(doos)
    
    oos_df = df.loc[oos_mask].copy()
    oos_df['pred'] = pred
    
    ic_values = []
    for d, group in oos_df.groupby('date'):
        if len(group) < 20: continue
        ic, _ = stats.spearmanr(group['pred'], group['fwd_ret'])
        if not np.isnan(ic): ic_values.append(ic)
    
    if not ic_values:
        return None
    
    ic_mean = np.mean(ic_values)
    icir = ic_mean / (np.std(ic_values) + 1e-10)
    ic_pos = np.mean([x > 0 for x in ic_values])
    
    spreads = []
    top5_rets = []
    for d, group in oos_df.groupby('date'):
        if len(group) < top_n * 2: continue
        n5 = max(1, int(len(group) * 0.05))
        n20 = max(1, int(len(group) * 0.20))
        top5 = group.nlargest(n5, 'pred')['fwd_ret'].mean()
        bot20 = group.nsmallest(n20, 'pred')['fwd_ret'].mean()
        spreads.append(top5 - bot20)
        top5_rets.append(top5)
    
    # 年度分解
    oos_df['year'] = oos_df['date'].dt.year
    yearly = {}
    for year, ydf in oos_df.groupby('year'):
        y_ic = []
        for d, group in ydf.groupby('date'):
            if len(group) < 20: continue
            ic, _ = stats.spearmanr(group['pred'], group['fwd_ret'])
            if not np.isnan(ic): y_ic.append(ic)
        if y_ic:
            yearly[int(year)] = {'ic': round(np.mean(y_ic), 4), 'n_days': len(y_ic)}
    
    return {
        'ic': round(ic_mean, 4), 'icir': round(icir, 3),
        'ic_pos': round(ic_pos, 3), 'spread': round(np.mean(spreads)*100, 2),
        'top5_avg': round(np.mean(top5_rets)*100, 2),
        'top5_win': round(np.mean([r > 0 for r in top5_rets])*100, 1),
        'n_days': len(ic_values), 'yearly': yearly
    }

# ============================================================
# 主实验
# ============================================================
EXPERIMENTS = {
    # V12实验
    'v12_baseline': {'price_range': 'arrow', 'hold_days': 5, 'feat_set': 'tech_macro', 'objective': 'reg:squarederror'},
    'v12_with_flow': {'price_range': 'arrow', 'hold_days': 5, 'feat_set': 'full', 'objective': 'reg:squarederror'},
    'v12_tech_only': {'price_range': 'arrow', 'hold_days': 5, 'feat_set': 'tech_only', 'objective': 'reg:squarederror'},
    'v12_3d_hold': {'price_range': 'arrow', 'hold_days': 3, 'feat_set': 'tech_macro', 'objective': 'reg:squarederror'},
    'v12_10d_hold': {'price_range': 'arrow', 'hold_days': 10, 'feat_set': 'tech_macro', 'objective': 'reg:squarederror'},
    'v12_classify': {'price_range': 'arrow', 'hold_days': 5, 'feat_set': 'tech_macro', 'objective': 'binary:logistic'},
    # V8实验
    'v8_baseline': {'price_range': 'blueshield', 'hold_days': 20, 'feat_set': 'tech_macro', 'objective': 'reg:squarederror'},
    'v8_with_flow': {'price_range': 'blueshield', 'hold_days': 20, 'feat_set': 'full', 'objective': 'reg:squarederror'},
    'v8_10d_hold': {'price_range': 'blueshield', 'hold_days': 10, 'feat_set': 'tech_macro', 'objective': 'reg:squarederror'},
    'v8_5d_hold': {'price_range': 'blueshield', 'hold_days': 5, 'feat_set': 'tech_macro', 'objective': 'reg:squarederror'},
    'v8_classify': {'price_range': 'blueshield', 'hold_days': 20, 'feat_set': 'tech_macro', 'objective': 'binary:logistic'},
}

def run_experiment(exp_name, df_full, macro):
    if exp_name not in EXPERIMENTS:
        log(f'未知实验: {exp_name}')
        return None
    
    cfg = EXPERIMENTS[exp_name]
    log(f'\n{"="*60}')
    log(f'实验: {exp_name}')
    log(f'  价格区间: {cfg["price_range"]} | 持有期: {cfg["hold_days"]}天 | 特征集: {cfg["feat_set"]} | 目标: {cfg["objective"]}')
    log(f'{"="*60}')
    
    df, feat_cols = build_features(df_full, macro, cfg['price_range'], cfg['hold_days'], cfg['feat_set'])
    
    if cfg['objective'] == 'binary:logistic':
        df['fwd_ret'] = (df['fwd_ret'] > 0.02).astype(int)
    
    top_n = 15 if cfg['price_range'] == 'blueshield' else 5
    
    params = {
        'objective': cfg['objective'],
        'max_depth': 6, 'learning_rate': 0.03,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 10, 'tree_method': 'hist', 'seed': 42, 'verbosity': 0
    }
    if cfg['objective'] == 'binary:logistic':
        params['eval_metric'] = 'logloss'
    
    # Walk-Forward
    wf = walk_forward_eval(df, feat_cols, cfg['hold_days'], params, top_n=top_n)
    
    if wf:
        valid_wf = [r for r in wf if r['n_days'] >= 30]
        if valid_wf:
            avg_ic = np.mean([r['ic'] for r in valid_wf])
            avg_icir = np.mean([r['icir'] for r in valid_wf])
            avg_ic_pos = np.mean([r['ic_pos'] for r in valid_wf])
            avg_spread = np.mean([r['spread'] for r in valid_wf])
            best_iter = max(int(np.median([r['best_iter'] for r in valid_wf])), 200)
        else:
            avg_ic = avg_icir = avg_ic_pos = avg_spread = 0
            best_iter = 500
    else:
        avg_ic = avg_icir = avg_ic_pos = avg_spread = 0
        best_iter = 500
    
    for r in wf:
        log(f'  Fold {r["fold"]}: IC={r["ic"]:.4f} ICIR={r["icir"]:.3f} IC>0={r["ic_pos"]*100:.0f}% spread={r["spread"]:+.2f}% [{r["period"]}]')
    log(f'  WF汇总: IC={avg_ic:.4f} ICIR={avg_icir:.3f} IC>0={avg_ic_pos*100:.0f}% spread={avg_spread:+.2f}%')
    
    # OOS
    oos = oos_eval(df, feat_cols, cfg['hold_days'], params, best_iter, top_n)
    if oos:
        log(f'  OOS: IC={oos["ic"]:.4f} ICIR={oos["icir"]:.3f} IC>0={oos["ic_pos"]*100:.0f}% spread={oos["spread"]:+.2f}% Top5%={oos["top5_avg"]:+.2f}% win={oos["top5_win"]:.0f}%')
        for year, yd in sorted(oos.get('yearly', {}).items()):
            log(f'    {year}: IC={yd["ic"]:.4f} ({yd["n_days"]}天)')
    
    result = {
        'experiment': exp_name,
        'config': cfg,
        'n_features': len(feat_cols),
        'n_samples': len(df),
        'features': feat_cols,
        'wf': wf,
        'wf_summary': {'ic': round(avg_ic, 4), 'icir': round(avg_icir, 3), 'ic_pos': round(avg_ic_pos, 3), 'spread': round(avg_spread, 2)},
        'oos': oos,
        'best_iter': best_iter,
        'params': params,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存
    result_path = os.path.join(RESULT_DIR, f'{exp_name}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    log(f'  保存: {result_path}')
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help='实验名称（逗号分隔多个）')
    parser.add_argument('--all', action='store_true', help='运行所有实验')
    parser.add_argument('--list', action='store_true', help='列出所有实验')
    args = parser.parse_args()
    
    if args.list:
        for name, cfg in EXPERIMENTS.items():
            print(f'  {name}: {cfg["price_range"]} {cfg["hold_days"]}d {cfg["feat_set"]} {cfg["objective"]}')
        return
    
    df_full, macro = load_all_data()
    
    if args.all:
        exps = list(EXPERIMENTS.keys())
    elif args.exp:
        exps = args.exp.split(',')
    else:
        print('用法: --exp <name> 或 --all 或 --list')
        return
    
    results = {}
    for exp_name in exps:
        try:
            results[exp_name] = run_experiment(exp_name, df_full, macro)
        except Exception as e:
            log(f'  [ERROR] {exp_name}: {e}')
            import traceback
            traceback.print_exc()
    
    # 汇总
    log(f'\n{"="*60}')
    log('实验汇总排名 (按OOS ICIR)')
    log(f'{"="*60}')
    ranked = []
    for name, r in results.items():
        if r and r.get('oos'):
            ranked.append({
                'name': name,
                'wf_icir': r['wf_summary']['icir'],
                'oos_icir': r['oos']['icir'],
                'oos_ic': r['oos']['ic'],
                'oos_spread': r['oos']['spread'],
                'oos_win': r['oos']['top5_win']
            })
    ranked.sort(key=lambda x: -x['oos_icir'])
    for i, r in enumerate(ranked):
        log(f'  {i+1}. {r["name"]:20s} WF_ICIR={r["wf_icir"]:.3f} OOS_ICIR={r["oos_icir"]:.3f} OOS_IC={r["oos_ic"]:.4f} spread={r["oos_spread"]:+.2f}% win={r["oos_win"]:.0f}%')

if __name__ == '__main__':
    main()
