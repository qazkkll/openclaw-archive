#!/usr/bin/env python3
"""
绿箭V12 重训练 + IC/ICIR验证
Walk-Forward 5折 + OOS 2024-2026
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

warnings.filterwarnings('ignore')
ROOT = '/home/hermes/.hermes/openclaw-archive'
DATA_DIR = os.path.join(ROOT, 'data', 'us')
MODEL_DIR = os.path.join(ROOT, 'models', 'us')

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

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
    return g

feat_cols = ['ma5', 'ma20', 'ma60', 'ma_bias20', 'ma_align', 'price_position',
    'ret1', 'ret5', 'ret20', 'ret60', 'momentum_6m', 'momentum_1m',
    'mom_divergence', 'trend_accel', 'vol20', 'vol5', 'vol_ratio', 'vol_change',
    'rsi14', 'rsi_change', 'macd', 'macd_signal', 'macd_hist',
    'bb_std', 'bb_width', 'bb_pos', 'ret_quality', 'price', 'range_pct',
    'vix_close', 'spy_ret1', 'spy_ret5', 'spy_ret20', 'spy_ret60',
    'qqq_ret1', 'qqq_ret5', 'qqq_ret20', 'qqq_ret60',
    'iwm_ret1', 'iwm_ret5', 'iwm_ret20', 'iwm_ret60']
hold_days = 5

log('='*60)
log('绿箭V12 重训练 (Walk-Forward + IC/ICIR)')
log('='*60)

# 1. 加载全量数据
log('Step 1: 加载数据...')
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

# 2. 宏观特征（从全量数据，包含SPY/QQQ/IWM）
log('Step 2: 计算宏观特征...')
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
log(f'  宏观: {len(macro)}天, SPY/QQQ/IWM/VIX')

# 3. 过滤绿箭范围
last_prices = df.groupby('sym')['close'].last()
valid_syms = last_prices[(last_prices >= 1) & (last_prices <= 10)].index
df = df[df['sym'].isin(valid_syms)]
log(f'  绿箭: {df["sym"].nunique()}只, {len(df):,}行')

# 4. 技术特征
log('Step 3: 计算技术特征...')
t0 = time.time()
df = df.sort_values(['sym', 'date'])
groups = []
for sym, group in df.groupby('sym'):
    if len(group) < 80:
        continue
    groups.append(compute_tech_features(group.copy()))
df = pd.concat(groups, ignore_index=True)
log(f'  完成: {time.time()-t0:.1f}s, {len(df):,}行')

# 5. 合并宏观
df = df.merge(macro, on='date', how='left')
for col in feat_cols:
    if col not in df.columns:
        df[col] = 0
    df[col] = df[col].ffill().fillna(0)

# 6. 标签
log('Step 4: 创建标签...')
orig_sym = df['sym'].values.copy()
def calc_fwd_return(group):
    group = group.sort_values('date')
    group['fwd_ret'] = group['close'].shift(-hold_days) / group['close'] - 1
    return group
df = df.groupby('sym', group_keys=False).apply(calc_fwd_return)
df = df.reset_index(drop=True)
if 'sym' not in df.columns:
    df['sym'] = orig_sym[:len(df)]
df = df.dropna(subset=['fwd_ret'])

# 排除ETF
etf_syms = {'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'IVV', 'VEA', 'VWO',
            'BND', 'AGG', 'TLT', 'GLD', 'SLV', 'USO', 'XLE', 'XLF', 'XLK', 'XLV'}
df = df[~df['sym'].isin(etf_syms)]

# 特征完整性过滤
feat_present = df[feat_cols].notna().sum(axis=1)
df = df[feat_present >= len(feat_cols) * 0.8]
for col in feat_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

log(f'  最终: {len(df):,}行, {df["sym"].nunique()}只, {len(feat_cols)}特征')

# 7. Walk-Forward 验证
log('Step 5: Walk-Forward 5折验证...')
df = df.sort_values('date')
dates = np.sort(df['date'].unique())
oos_start = pd.Timestamp('2024-01-01')
train_dates = dates[dates < oos_start]
oos_dates = dates[dates >= oos_start]
log(f'  训练: {pd.Timestamp(train_dates[0]).date()} ~ {pd.Timestamp(train_dates[-1]).date()}')
log(f'  OOS: {pd.Timestamp(oos_dates[0]).date()} ~ {pd.Timestamp(oos_dates[-1]).date()}')

params = {
    'objective': 'reg:squarederror', 'max_depth': 6, 'learning_rate': 0.03,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 10,
    'tree_method': 'hist', 'seed': 42, 'verbosity': 0
}

n_folds = 5
fold_size = len(train_dates) // n_folds
wf_results = []

for fold in range(n_folds):
    train_end_idx = (fold + 1) * fold_size
    val_start_idx = train_end_idx
    val_end_idx = min(val_start_idx + fold_size, len(train_dates))
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
    
    # IC
    ic_values = []
    for d, group in val_df.groupby('date'):
        if len(group) < 20: continue
        ic, _ = stats.spearmanr(group['pred'], group['fwd_ret'])
        if not np.isnan(ic): ic_values.append(ic)
    
    if ic_values:
        ic_mean = np.mean(ic_values)
        icir = ic_mean / (np.std(ic_values) + 1e-10)
        ic_pos = np.mean([x > 0 for x in ic_values])
        
        # Top5% vs Bot20% 分层
        spreads = []
        for d, group in val_df.groupby('date'):
            if len(group) < 20: continue
            n5 = max(1, int(len(group) * 0.05))
            n20 = max(1, int(len(group) * 0.20))
            top5 = group.nlargest(n5, 'pred')['fwd_ret'].mean()
            bot20 = group.nsmallest(n20, 'pred')['fwd_ret'].mean()
            spreads.append(top5 - bot20)
        
        wf_results.append({
            'fold': fold, 'ic': round(ic_mean, 4), 'icir': round(icir, 3),
            'ic_pos': round(ic_pos, 3), 'spread': round(np.mean(spreads)*100, 2),
            'n_days': len(ic_values), 'best_iter': model.best_iteration
        })
        log(f'  Fold {fold}: IC={ic_mean:.4f} ICIR={icir:.3f} IC>0={ic_pos*100:.0f}% spread={np.mean(spreads)*100:+.2f}%')

if wf_results:
    avg_ic = np.mean([r['ic'] for r in wf_results])
    avg_icir = np.mean([r['icir'] for r in wf_results])
    avg_ic_pos = np.mean([r['ic_pos'] for r in wf_results])
    avg_spread = np.mean([r['spread'] for r in wf_results])
    best_iter = max(int(np.median([r['best_iter'] for r in wf_results])), 200)
    log(f'\n  WF汇总: IC={avg_ic:.4f} ICIR={avg_icir:.3f} IC>0={avg_ic_pos*100:.0f}% spread={avg_spread:+.2f}%')
else:
    best_iter = 500

# 8. OOS评估
log('\nStep 6: OOS评估...')
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

oos_ic_values = []
for d, group in oos_df.groupby('date'):
    if len(group) < 20: continue
    ic, _ = stats.spearmanr(group['pred'], group['fwd_ret'])
    if not np.isnan(ic): oos_ic_values.append(ic)

oos_ic = np.mean(oos_ic_values)
oos_icir = oos_ic / (np.std(oos_ic_values) + 1e-10)
oos_ic_pos = np.mean([x > 0 for x in oos_ic_values])

# 分层
oos_spreads = []
oos_top5_rets = []
for d, group in oos_df.groupby('date'):
    if len(group) < 20: continue
    n5 = max(1, int(len(group) * 0.05))
    n20 = max(1, int(len(group) * 0.20))
    top5 = group.nlargest(n5, 'pred')['fwd_ret'].mean()
    bot20 = group.nsmallest(n20, 'pred')['fwd_ret'].mean()
    oos_spreads.append(top5 - bot20)
    oos_top5_rets.append(top5)

log(f'  OOS IC={oos_ic:.4f} ICIR={oos_icir:.3f} IC>0={oos_ic_pos*100:.0f}%')
log(f'  Top5%={np.mean(oos_top5_rets)*100:+.2f}% spread={np.mean(oos_spreads)*100:+.2f}%')
log(f'  Top5 win={np.mean([r > 0 for r in oos_top5_rets])*100:.1f}%')

# 9. 训练最终模型
log('\nStep 7: 训练最终模型...')
X_all = np.nan_to_num(df[feat_cols].values, nan=0, posinf=0, neginf=0).astype(np.float32)
y_all = df['fwd_ret'].values
dall = xgb.DMatrix(X_all, label=y_all, feature_names=feat_cols)
final_model = xgb.train(params, dall, num_boost_round=best_iter, verbose_eval=False)

importance = final_model.get_score(importance_type='gain')
total_imp = sum(importance.values())
feat_imp = {k: round(v / total_imp * 100, 2) for k, v in sorted(importance.items(), key=lambda x: -x[1])}

# 10. 保存
log('\nStep 8: 保存...')
model_path = os.path.join(MODEL_DIR, 'arrow_v12_xgb.json')
final_model.save_model(model_path)
log(f'  模型: {model_path} ({os.path.getsize(model_path)/1024:.0f}KB)')

meta = {
    'version': 'arrow_v12', 'algorithm': 'XGBoost',
    'features': feat_cols, 'n_features': len(feat_cols),
    'hold_days': 5, 'top_n': 5,
    'universe': f'$1-$10 ({df["sym"].nunique()}只)',
    'params': params, 'n_trees': best_iter,
    'trained_on': f'{df["date"].min().date()} ~ {df["date"].max().date()}',
    'train_end': str(df['date'].max().date()),
    'n_train_samples': len(df),
    'feature_importance': feat_imp,
    'validation': {
        'method': 'Walk-Forward 5折 + OOS 2024-2026',
        'wf_ic': round(avg_ic, 4), 'wf_icir': round(avg_icir, 3),
        'wf_ic_pos': round(avg_ic_pos, 3), 'wf_spread': round(avg_spread, 2),
        'oos_ic': round(oos_ic, 4), 'oos_icir': round(oos_icir, 3),
        'oos_ic_pos': round(oos_ic_pos, 3),
        'oos_top5_avg': round(np.mean(oos_top5_rets)*100, 2),
        'oos_spread': round(np.mean(oos_spreads)*100, 2),
        'oos_n_days': len(oos_ic_values),
    },
    'signal_thresholds': {
        'green2': {'threshold': round(float(np.percentile(pred, 99)), 4), 'note': 'Top 1%'},
        'green1': {'threshold': round(float(np.percentile(pred, 95)), 4), 'note': 'Top 5%'},
        'observe': {'threshold': round(float(np.percentile(pred, 90)), 4), 'note': 'Top 10%'}
    },
    'created': time.strftime('%Y-%m-%d %H:%M'),
    'data_source': 'us_hist_full_10y.parquet',
    'data_leakage_warning': '原V12训练含测试期数据(ICIR=0.573虚高)。本次为正确WF训练。'
}

meta_path = os.path.join(MODEL_DIR, 'arrow_v12_meta.json')
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

log(f'\n{"="*60}')
log('绿箭V12 重训练完成')
log(f'{"="*60}')
log(f'  WF: IC={avg_ic:.4f} ICIR={avg_icir:.3f} IC>0={avg_ic_pos*100:.0f}%')
log(f'  OOS: IC={oos_ic:.4f} ICIR={oos_icir:.3f} IC>0={oos_ic_pos*100:.0f}%')
log(f'  Top5%={np.mean(oos_top5_rets)*100:+.2f}% spread={np.mean(oos_spreads)*100:+.2f}%')
log(f'  Top3特征: {list(feat_imp.keys())[:3]}')
