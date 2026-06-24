#!/usr/bin/env python3
"""
全市场蓝盾V6 + 绿箭V11 重训练脚本
数据: us_hist_full_10y.parquet (29.8M行, 11,864只股票)
特征: 与评分脚本完全一致的技术面+宏观面+基本面
验证: Walk-Forward 5折 + 样本外2024-2026
输出: models/us/blueshield_v7_xgb.json, models/us/arrow_v12_xgb.json

用法:
    python3 retrain_full_market.py                # 训练两个模型
    python3 retrain_full_market.py --blueshield   # 只训练蓝盾
    python3 retrain_full_market.py --arrow        # 只训练绿箭
"""
import json, os, sys, time, argparse, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, 'data', 'us')
MODEL_DIR = os.path.join(ROOT, 'models', 'us')
os.makedirs(MODEL_DIR, exist_ok=True)

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

# ============================================================
# 1. 加载数据
# ============================================================
def load_data():
    log('Step 1: 加载数据...')
    df = pd.read_parquet(os.path.join(DATA_DIR, 'us_hist_full_10y.parquet'))
    log(f'  总行数: {len(df):,}')
    log(f'  股票数: {df["sym"].nunique()}')
    
    # 加载VIX
    vix_df = pd.read_parquet(os.path.join(DATA_DIR, 'vix_10y.parquet'))
    # 处理MultiIndex列
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
    log(f'  VIX: {len(vix_df)} 天')
    
    # 加载基本面
    with open(os.path.join(DATA_DIR, 'us_fundamentals.json')) as f:
        fund = json.load(f)
    fund_df = pd.DataFrame(fund).T
    fund_df.index.name = 'sym'
    fund_df = fund_df.reset_index()
    fund_cols = ['sym', 'trailingPE', 'forwardPE', 'dividendYield', 'beta']
    fund_df = fund_df[[c for c in fund_cols if c in fund_df.columns]]
    fund_df = fund_df.rename(columns={
        'trailingPE': 'pe_trailing',
        'forwardPE': 'pe_forward',
        'dividendYield': 'div_yield'
    })
    log(f'  基本面: {len(fund_df)} 只股票')
    
    return df, vix_df, fund_df

# ============================================================
# 2. 技术特征计算（与评分脚本完全一致）
# ============================================================
def compute_tech_features(g):
    """单只股票的技术特征，输入必须按date排序"""
    c = g['close']
    g['ma5'] = c.rolling(5).mean()
    g['ma20'] = c.rolling(20).mean()
    g['ma60'] = c.rolling(60).mean()
    g['ma_bias20'] = (c - g['ma20']) / g['ma20']
    g['ma_align'] = ((c > g['ma5']).astype(int) + (g['ma5'] > g['ma20']).astype(int))
    mn60 = c.rolling(60).min()
    mx60 = c.rolling(60).max()
    g['price_position'] = (c - mn60) / (mx60 - mn60 + 1e-10)
    g['ret1'] = c.pct_change(1)
    g['ret5'] = c.pct_change(5)
    g['ret20'] = c.pct_change(20)
    g['ret60'] = c.pct_change(60)
    g['momentum_6m'] = c.pct_change(126)
    g['momentum_1m'] = c.pct_change(21)
    g['mom_divergence'] = g['momentum_1m'] - g['ret20']
    g['trend_accel'] = g['ret5'] - g['ret5'].shift(5)
    dr = c.pct_change(1)
    g['vol20'] = dr.rolling(20).std()
    g['vol5'] = dr.rolling(5).std()
    g['vol_ratio'] = g['volume'] / g['volume'].rolling(20).mean()
    g['vol_change'] = g['vol20'] / g['vol20'].shift(20)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    g['rsi14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    g['rsi_change'] = g['rsi14'].diff(5)
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    g['macd'] = ema12 - ema26
    g['macd_signal'] = g['macd'].ewm(span=9).mean()
    g['macd_hist'] = g['macd'] - g['macd_signal']
    g['bb_std'] = dr.rolling(20).std()
    bb_mid = c.rolling(20).mean()
    g['bb_width'] = 4 * g['bb_std'] * bb_mid / (bb_mid + 1e-10)
    g['bb_pos'] = (c - (bb_mid - 2 * c.rolling(20).std())) / (4 * c.rolling(20).std() + 1e-10)
    ret_pos = dr.clip(lower=0).rolling(20).mean()
    ret_neg = (-dr).clip(lower=0).rolling(20).mean()
    g['ret_quality'] = ret_pos / (ret_pos + ret_neg + 1e-10)
    # Arrow V11 专用特征
    g['price'] = c
    g['range_pct'] = (g['high'] - g['low']) / (c + 1e-10)
    return g

# ============================================================
# 3. 宏观特征计算
# ============================================================
def compute_macro_features(spy, qqq, iwm, vix):
    """计算SPY/QQQ/IWM的多周期收益率 + VIX"""
    macro = pd.DataFrame()
    macro['date'] = spy['date']
    
    for name, series in [('spy', spy['close']), ('qqq', qqq['close']), ('iwm', iwm['close'])]:
        macro[f'{name}_ret1'] = series.pct_change(1)
        macro[f'{name}_ret5'] = series.pct_change(5)
        macro[f'{name}_ret20'] = series.pct_change(20)
        macro[f'{name}_ret60'] = series.pct_change(60)
    
    macro = macro.merge(vix, on='date', how='left')
    macro['vix_close'] = macro['vix_close'].ffill()
    return macro

# ============================================================
# 4. 全流程特征工程
# ============================================================
def build_features(df, vix_df, fund_df, price_range):
    """
    price_range: 'blueshield' (>$10) 或 'arrow' ($1-$10)
    返回: (X, y, dates, tickers) 用于训练
    """
    log(f'Step 2: 特征工程 [{price_range}]...')
    
    # 过滤价格范围
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close', 'volume'])
    df = df[df['volume'] > 0]
    
    if price_range == 'blueshield':
        # 蓝盾: >$10, 用最近价格过滤
        last_prices = df.groupby('sym')['close'].last()
        valid_syms = last_prices[last_prices > 10].index
        hold_days = 20
    else:
        # 绿箭: $1-$10
        last_prices = df.groupby('sym')['close'].last()
        valid_syms = last_prices[(last_prices >= 1) & (last_prices <= 10)].index
        hold_days = 5
    
    df = df[df['sym'].isin(valid_syms)]
    log(f'  价格过滤后: {df["sym"].nunique()} 只股票, {len(df):,} 行')
    
    # 分组计算技术特征
    log('  计算技术特征...')
    t0 = time.time()
    
    # 用groupby + apply避免循环
    df = df.sort_values(['sym', 'date'])
    groups = []
    for sym, group in df.groupby('sym'):
        if len(group) < 80:  # 需要至少60天数据计算ma60
            continue
        g = compute_tech_features(group.copy())
        groups.append(g)
    
    df = pd.concat(groups, ignore_index=True)
    log(f'  技术特征完成: {time.time()-t0:.1f}s, {len(df):,} 行')
    
    # 提取宏观ETF数据
    log('  计算宏观特征...')
    spy = df[df['sym'] == 'SPY'][['date', 'close']].copy()
    qqq = df[df['sym'] == 'QQQ'][['date', 'close']].copy()
    iwm = df[df['sym'] == 'IWM'][['date', 'close']].copy()
    
    if len(spy) == 0 or len(qqq) == 0 or len(iwm) == 0:
        log('  [ERROR] 缺少SPY/QQQ/IWM数据')
        return None
    
    macro = compute_macro_features(spy, qqq, iwm, vix_df)
    
    # 合并宏观特征
    df = df.merge(macro, on='date', how='left')
    
    # 合并基本面
    if price_range == 'blueshield':
        df = df.merge(fund_df, on='sym', how='left')
        for col in ['pe_trailing', 'pe_forward', 'div_yield', 'beta']:
            if col not in df.columns:
                df[col] = np.nan
    
    # 创建标签: 前瞻收益率
    log(f'  创建标签 (hold_days={hold_days})...')
    def calc_fwd_return(group):
        group = group.sort_values('date')
        group['fwd_ret'] = group['close'].shift(-hold_days) / group['close'] - 1
        return group
    
    df = df.groupby('sym', group_keys=False).apply(calc_fwd_return)
    
    # 过滤无标签的行
    df = df.dropna(subset=['fwd_ret'])
    
    # 过滤ETF和非普通股（只保留看起来像股票的）
    # 排除SPY/QQQ/IWM/DIA等ETF
    etf_syms = {'SPY', 'QQQ', 'IWM', 'DIA', 'SPDR', 'VOO', 'VTI', 'IVV', 'VEA', 'VWO',
                'BND', 'AGG', 'TLT', 'GLD', 'SLV', 'USO', 'XLE', 'XLF', 'XLK', 'XLV',
                'XLI', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC', 'ARKK', 'ARKG', 'ARKW'}
    df = df[~df['sym'].isin(etf_syms)]
    
    # 定义特征列
    tech_feats = ['ma5', 'ma20', 'ma60', 'ma_bias20', 'ma_align', 'price_position',
        'ret1', 'ret5', 'ret20', 'ret60', 'momentum_6m', 'momentum_1m',
        'mom_divergence', 'trend_accel', 'vol20', 'vol5', 'vol_ratio', 'vol_change',
        'rsi14', 'rsi_change', 'macd', 'macd_signal', 'macd_hist',
        'bb_std', 'bb_width', 'bb_pos', 'ret_quality']
    macro_cols = ['vix_close', 'spy_ret1', 'spy_ret5', 'spy_ret20', 'spy_ret60',
                  'qqq_ret1', 'qqq_ret5', 'qqq_ret20', 'qqq_ret60',
                  'iwm_ret1', 'iwm_ret5', 'iwm_ret20', 'iwm_ret60']
    
    if price_range == 'blueshield':
        fund_cols = ['pe_trailing', 'pe_forward', 'div_yield', 'beta']
        extra_feats = []
        feat_cols = tech_feats + macro_cols + fund_cols
    else:
        fund_cols = []
        extra_feats = ['price', 'range_pct']
        feat_cols = tech_feats + extra_feats + macro_cols
    
    # 确保所有特征列存在
    for col in feat_cols:
        if col not in df.columns:
            df[col] = np.nan
    
    # 过滤行: 至少80%特征非空
    feat_present = df[feat_cols].notna().sum(axis=1)
    df = df[feat_present >= len(feat_cols) * 0.8]
    
    # 填充剩余NaN为中位数
    for col in feat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    log(f'  最终数据: {len(df):,} 行, {df["sym"].nunique()} 只股票, {len(feat_cols)} 特征')
    
    return df, feat_cols, hold_days

# ============================================================
# 5. Walk-Forward 验证
# ============================================================
def walk_forward_validate(df, feat_cols, hold_days, n_folds=5):
    """Walk-Forward n折验证"""
    log(f'Step 3: Walk-Forward {n_folds}折验证...')
    
    df = df.sort_values('date')
    dates = df['date'].unique()
    dates = np.sort(dates)
    
    # 划分: 2024-01-01之前为训练+验证，之后为OOS
    oos_start = pd.Timestamp('2024-01-01')
    train_dates = dates[dates < oos_start]
    oos_dates = dates[dates >= oos_start]
    
    log(f'  训练+验证期: {train_dates[0].date()} ~ {train_dates[-1].date()} ({len(train_dates)} 天)')
    log(f'  样本外: {oos_dates[0].date()} ~ {oos_dates[-1].date()} ({len(oos_dates)} 天)')
    
    # Walk-Forward折叠
    fold_size = len(train_dates) // n_folds
    wf_results = []
    
    for fold in range(n_folds):
        # 训练: 前fold+1个区间
        train_end_idx = (fold + 1) * fold_size
        if train_end_idx >= len(train_dates):
            train_end_idx = len(train_dates) - 1
        
        # 验证: 下一个区间
        val_start_idx = train_end_idx
        val_end_idx = min(val_start_idx + fold_size, len(train_dates))
        
        if val_end_idx <= val_start_idx:
            continue
        
        train_mask = df['date'].isin(train_dates[:train_end_idx])
        val_mask = df['date'].isin(train_dates[val_start_idx:val_end_idx])
        
        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, 'fwd_ret'].values
        X_val = df.loc[val_mask, feat_cols].values
        y_val = df.loc[val_mask, 'fwd_ret'].values
        
        if len(X_train) < 1000 or len(X_val) < 100:
            continue
        
        # 训练XGBoost回归
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 10,
            'tree_method': 'hist',
            'seed': 42,
            'verbosity': 0
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_cols)
        
        model = xgb.train(params, dtrain, num_boost_round=500,
                         evals=[(dval, 'val')], early_stopping_rounds=50,
                         verbose_eval=False)
        
        pred = model.predict(dval)
        
        # 评估: Top N的平均收益
        val_df = df.loc[val_mask].copy()
        val_df['pred'] = pred
        
        # 每天选Top N
        top_n = 15 if hold_days == 20 else 5
        daily_returns = []
        for d, group in val_df.groupby('date'):
            if len(group) < top_n:
                continue
            top = group.nlargest(top_n, 'pred')
            daily_returns.append(top['fwd_ret'].mean())
        
        if daily_returns:
            avg_ret = np.mean(daily_returns) * 100
            win_rate = np.mean([r > 0 for r in daily_returns]) * 100
            sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252 / hold_days)
            wf_results.append({
                'fold': fold,
                'avg_return': avg_ret,
                'win_rate': win_rate,
                'sharpe': sharpe,
                'n_days': len(daily_returns),
                'best_iter': model.best_iteration
            })
            log(f'  Fold {fold}: avg={avg_ret:.2f}%, win={win_rate:.1f}%, sharpe={sharpe:.2f}, '
                f'days={len(daily_returns)}, best_iter={model.best_iteration}')
    
    return wf_results

# ============================================================
# 6. OOS评估
# ============================================================
def evaluate_oos(df, feat_cols, hold_days, best_iteration):
    """样本外评估"""
    log('Step 4: 样本外评估 (2024-2026)...')
    
    oos_start = pd.Timestamp('2024-01-01')
    train_mask = df['date'] < oos_start
    oos_mask = df['date'] >= oos_start
    
    X_train = df.loc[train_mask, feat_cols].values
    y_train = df.loc[train_mask, 'fwd_ret'].values
    X_oos = df.loc[oos_mask, feat_cols].values
    y_oos = df.loc[oos_mask, 'fwd_ret'].values
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'tree_method': 'hist',
        'seed': 42,
        'verbosity': 0
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_cols)
    doos = xgb.DMatrix(X_oos, label=y_oos, feature_names=feat_cols)
    
    model = xgb.train(params, dtrain, num_boost_round=best_iteration,
                     verbose_eval=False)
    
    pred = model.predict(doos)
    
    oos_df = df.loc[oos_mask].copy()
    oos_df['pred'] = pred
    
    top_n = 15 if hold_days == 20 else 5
    daily_returns = []
    for d, group in oos_df.groupby('date'):
        if len(group) < top_n:
            continue
        top = group.nlargest(top_n, 'pred')
        daily_returns.append(top['fwd_ret'].mean())
    
    if daily_returns:
        avg_ret = np.mean(daily_returns) * 100
        win_rate = np.mean([r > 0 for r in daily_returns]) * 100
        sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252 / hold_days)
        max_dd = np.min(np.cumsum(daily_returns)) * 100
        log(f'  OOS: avg={avg_ret:.2f}%, win={win_rate:.1f}%, sharpe={sharpe:.2f}, max_dd={max_dd:.1f}%')
        return {
            'avg_return': avg_ret,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'n_days': len(daily_returns),
            'n_stocks': oos_df['sym'].nunique()
        }
    return None

# ============================================================
# 7. 训练最终模型
# ============================================================
def train_final_model(df, feat_cols, hold_days, best_iteration):
    """用全部数据训练最终模型"""
    log('Step 5: 训练最终模型...')
    
    X = df[feat_cols].values
    y = df['fwd_ret'].values
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'tree_method': 'hist',
        'seed': 42,
        'verbosity': 0
    }
    
    dtrain = xgb.DMatrix(X, label=y, feature_names=feat_cols)
    model = xgb.train(params, dtrain, num_boost_round=best_iteration,
                     verbose_eval=False)
    
    # 特征重要性
    importance = model.get_score(importance_type='gain')
    total_imp = sum(importance.values())
    feat_imp = {k: round(v / total_imp * 100, 2) for k, v in 
                sorted(importance.items(), key=lambda x: -x[1])}
    
    return model, feat_imp

# ============================================================
# 8. 信号阈值计算
# ============================================================
def compute_signal_thresholds(model, df, feat_cols, hold_days):
    """基于历史数据计算信号阈值"""
    log('Step 6: 计算信号阈值...')
    
    X = df[feat_cols].values
    dmat = xgb.DMatrix(X, feature_names=feat_cols)
    pred = model.predict(dmat)
    
    df_pred = df.copy()
    df_pred['pred'] = pred
    
    # 每天排名
    top_n = 15 if hold_days == 20 else 5
    daily_stats = []
    for d, group in df_pred.groupby('date'):
        if len(group) < top_n * 3:
            continue
        top = group.nlargest(top_n, 'pred')
        rest = group[~group.index.isin(top.index)]
        daily_stats.append({
            'top_avg': top['fwd_ret'].mean(),
            'rest_avg': rest['fwd_ret'].mean(),
            'top_pred_min': top['pred'].min(),
            'top_pred_max': top['pred'].max(),
            'top_pred_mean': top['pred'].mean()
        })
    
    stats_df = pd.DataFrame(daily_stats)
    
    # 阈值: 用预测分数的百分位
    thresholds = {
        'green2': {
            'threshold': round(float(stats_df['top_pred_max'].quantile(0.9)), 4),
            'note': f'Top 1%信号, 样本少但收益率极高'
        },
        'green1': {
            'threshold': round(float(stats_df['top_pred_mean'].quantile(0.75)), 4),
            'note': f'主力信号, Top {top_n}平均分位'
        },
        'observe': {
            'threshold': round(float(stats_df['top_pred_min'].quantile(0.5)), 4),
            'note': '观察池'
        }
    }
    
    # 信号分级统计
    for level, info in thresholds.items():
        t = info['threshold']
        mask = df_pred['pred'] >= t
        if mask.sum() > 0:
            avg_ret = df_pred.loc[mask, 'fwd_ret'].mean() * 100
            win_rate = (df_pred.loc[mask, 'fwd_ret'] > 0).mean() * 100
            info['avg_return'] = round(avg_ret, 2)
            info['win_rate'] = round(win_rate, 1)
            info['count'] = int(mask.sum())
            log(f'  {level}: threshold={t}, avg={avg_ret:.2f}%, win={win_rate:.1f}%, count={mask.sum()}')
    
    return thresholds

# ============================================================
# 9. 保存模型
# ============================================================
def save_model(model, feat_cols, hold_days, version, price_range, 
               wf_results, oos_result, feat_imp, thresholds, df, n_trees):
    """保存XGBoost模型+元数据"""
    
    # 保存模型文件
    model_path = os.path.join(MODEL_DIR, f'{version}_xgb.json')
    model.save_model(model_path)
    log(f'  模型保存: {model_path} ({os.path.getsize(model_path)/1024/1024:.1f}MB)')
    
    # 分离特征类型
    tech_feats = ['ma5', 'ma20', 'ma60', 'ma_bias20', 'ma_align', 'price_position',
        'ret1', 'ret5', 'ret20', 'ret60', 'momentum_6m', 'momentum_1m',
        'mom_divergence', 'trend_accel', 'vol20', 'vol5', 'vol_ratio', 'vol_change',
        'rsi14', 'rsi_change', 'macd', 'macd_signal', 'macd_hist',
        'bb_std', 'bb_width', 'bb_pos', 'ret_quality']
    macro_cols = ['vix_close', 'spy_ret1', 'spy_ret5', 'spy_ret20', 'spy_ret60',
                  'qqq_ret1', 'qqq_ret5', 'qqq_ret20', 'qqq_ret60',
                  'iwm_ret1', 'iwm_ret5', 'iwm_ret20', 'iwm_ret60']
    fund_cols = ['pe_trailing', 'pe_forward', 'div_yield', 'beta']
    extra_feats = ['price', 'range_pct']
    
    if price_range == 'blueshield':
        universe = f'全市场>${10} ({df["sym"].nunique()}只)'
        top_n = 15
    else:
        universe = f'全市场$1-$10 ({df["sym"].nunique()}只)'
        top_n = 5
    
    # WF汇总
    if wf_results:
        wf_avg = np.mean([r['avg_return'] for r in wf_results])
        wf_win = np.mean([r['win_rate'] for r in wf_results])
        wf_sharpe = np.mean([r['sharpe'] for r in wf_results])
    else:
        wf_avg = wf_win = wf_sharpe = 0
    
    # 元数据
    meta = {
        'version': version,
        'algorithm': 'XGBoost',
        'features': feat_cols,
        'n_features': len(feat_cols),
        'tech_features': len([f for f in feat_cols if f in tech_feats]),
        'macro_features': len([f for f in feat_cols if f in macro_cols]),
        'fund_features': len([f for f in feat_cols if f in fund_cols + extra_feats]),
        'hold_days': hold_days,
        'top_n': top_n,
        'universe': universe,
        'params': {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 10,
            'tree_method': 'hist',
            'seed': 42,
            'verbosity': 0
        },
        'n_trees': n_trees,
        'trained_on': f'{df["date"].min()}~{df["date"].max()}',
        'n_train_samples': len(df),
        'feature_importance': feat_imp,
        'validation': {
            'method': f'Walk-Forward {len(wf_results)}折 + 样本外2024-2026',
            'wf_avg_return': round(wf_avg, 2),
            'wf_win_rate': round(wf_win, 1),
            'wf_sharpe': round(wf_sharpe, 2),
        },
        'signal_thresholds': thresholds,
        'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'data_source': 'us_hist_full_10y.parquet (全市场11,864只)',
        'replaces': 'V6' if price_range == 'blueshield' else 'V11'
    }
    
    if oos_result:
        meta['validation']['oos_avg_return'] = round(oos_result['avg_return'], 2)
        meta['validation']['oos_win_rate'] = round(oos_result['win_rate'], 1)
        meta['validation']['oos_sharpe'] = round(oos_result['sharpe'], 2)
        meta['validation']['oos_max_dd'] = round(oos_result['max_dd'], 1)
        meta['validation']['oos_n_days'] = oos_result['n_days']
    
    meta_path = os.path.join(MODEL_DIR, f'{version}_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    log(f'  元数据保存: {meta_path}')
    
    return meta

# ============================================================
# 主流程
# ============================================================
def train_model(price_range, df, vix_df, fund_df):
    """训练单个模型的完整流程"""
    
    if price_range == 'blueshield':
        version = 'blueshield_v7'
        log(f'\n{"="*60}')
        log(f'蓝盾V7 训练 (>$10, 20天持有期, Top 15)')
        log(f'{"="*60}')
    else:
        version = 'arrow_v12'
        log(f'\n{"="*60}')
        log(f'绿箭V12 训练 ($1-$10, 5天持有期, Top 5)')
        log(f'{"="*60}')
    
    t0 = time.time()
    
    # 特征工程
    result = build_features(df, vix_df, fund_df, price_range)
    if result is None:
        log('[ERROR] 特征工程失败')
        return None
    
    df_feat, feat_cols, hold_days = result
    
    # Walk-Forward验证
    wf_results = walk_forward_validate(df_feat, feat_cols, hold_days)
    
    # 确定最佳迭代次数
    if wf_results:
        best_iter = int(np.median([r['best_iter'] for r in wf_results]))
        best_iter = max(best_iter, 200)  # 至少200轮
    else:
        best_iter = 500
    
    log(f'  最佳迭代次数: {best_iter}')
    
    # OOS评估
    oos_result = evaluate_oos(df_feat, feat_cols, hold_days, best_iter)
    
    # 训练最终模型
    model, feat_imp = train_final_model(df_feat, feat_cols, hold_days, best_iter)
    
    # 计算信号阈值
    thresholds = compute_signal_thresholds(model, df_feat, feat_cols, hold_days)
    
    # 保存
    meta = save_model(model, feat_cols, hold_days, version, price_range,
                     wf_results, oos_result, feat_imp, thresholds, df_feat, best_iter)
    
    elapsed = time.time() - t0
    log(f'\n{version} 完成! 耗时: {elapsed/60:.1f}分钟')
    
    return meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blueshield', action='store_true', help='只训练蓝盾')
    parser.add_argument('--arrow', action='store_true', help='只训练绿箭')
    args = parser.parse_args()
    
    if not args.blueshield and not args.arrow:
        args.blueshield = True
        args.arrow = True
    
    t_total = time.time()
    
    # 加载数据
    df, vix_df, fund_df = load_data()
    
    results = {}
    
    if args.blueshield:
        results['blueshield'] = train_model('blueshield', df, vix_df, fund_df)
    
    if args.arrow:
        results['arrow'] = train_model('arrow', df, vix_df, fund_df)
    
    # 汇总报告
    log(f'\n{"="*60}')
    log('训练汇总报告')
    log(f'{"="*60}')
    
    for name, meta in results.items():
        if meta is None:
            log(f'{name}: [FAILED]')
            continue
        v = meta['validation']
        log(f'\n{meta["version"]}:')
        log(f'  数据: {meta["n_train_samples"]:,} 样本, {meta["n_features"]} 特征')
        log(f'  WF: avg={v.get("wf_avg_return",0):.2f}%, win={v.get("wf_win_rate",0):.1f}%, sharpe={v.get("wf_sharpe",0):.2f}')
        if 'oos_avg_return' in v:
            log(f'  OOS: avg={v["oos_avg_return"]:.2f}%, win={v["oos_win_rate"]:.1f}%, sharpe={v["oos_sharpe"]:.2f}')
        log(f'  Top 3特征: {list(meta["feature_importance"].keys())[:3]}')
    
    log(f'\n总耗时: {(time.time()-t_total)/60:.1f}分钟')

if __name__ == '__main__':
    main()
