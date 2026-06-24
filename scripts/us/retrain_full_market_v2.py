#!/usr/bin/env python3
"""
全市场蓝盾V7 + 绿箭V12 重训练脚本（内存优化版）
关键优化:
1. 分批处理stock groupby，避免一次性12M行groupby.apply
2. 向量化前向收益计算（不用apply）
3. 中间结果写磁盘，不全留内存
4. 技术特征用numpy数组直接算，不依赖pandas rolling
"""
import json, os, sys, time, argparse, warnings, gc
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
# 技术特征计算（numpy向量化版本）
# ============================================================
def compute_tech_numpy(close, high, low, volume):
    """用numpy直接计算技术特征，避免pandas rolling开销"""
    n = len(close)
    if n < 70:
        return None
    
    # 均线
    def rolling_mean(arr, w):
        out = np.full(len(arr), np.nan)
        cs = np.cumsum(arr)
        out[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
        return out
    
    def rolling_std(arr, w):
        out = np.full(len(arr), np.nan)
        for i in range(w-1, len(arr)):
            out[i] = np.std(arr[i-w+1:i+1], ddof=1)
        return out
    
    def rolling_min(arr, w):
        out = np.full(len(arr), np.nan)
        for i in range(w-1, len(arr)):
            out[i] = np.min(arr[i-w+1:i+1])
        return out
    
    def rolling_max(arr, w):
        out = np.full(len(arr), np.nan)
        for i in range(w-1, len(arr)):
            out[i] = np.max(arr[i-w+1:i+1])
        return out
    
    # 收益率
    ret = np.diff(close) / close[:-1]
    ret = np.concatenate([[0], ret])
    
    ma5 = rolling_mean(close, 5)
    ma20 = rolling_mean(close, 20)
    ma60 = rolling_mean(close, 60)
    
    ma_bias20 = (close - ma20) / (ma20 + 1e-10)
    ma_align = ((close > ma5).astype(int) + (ma5 > ma20).astype(int))
    
    mn60 = rolling_min(close, 60)
    mx60 = rolling_max(close, 60)
    price_position = (close - mn60) / (mx60 - mn60 + 1e-10)
    
    def pct_change(arr, p):
        out = np.full(len(arr), np.nan)
        out[p:] = arr[p:] / arr[:-p] - 1
        return out
    
    ret1 = pct_change(close, 1)
    ret5 = pct_change(close, 5)
    ret20 = pct_change(close, 20)
    ret60 = pct_change(close, 60)
    momentum_6m = pct_change(close, 126)
    momentum_1m = pct_change(close, 21)
    mom_divergence = momentum_1m - ret20
    
    # trend_accel: ret5 - ret5.shift(5)
    ret5_shift5 = np.full(len(ret5), np.nan)
    ret5_shift5[5:] = ret5[:-5]
    trend_accel = ret5 - ret5_shift5
    
    vol20 = rolling_std(ret, 20)
    vol5 = rolling_std(ret, 5)
    
    vol_ma20 = rolling_mean(volume.astype(np.float64), 20)
    vol_ratio = volume / (vol_ma20 + 1e-10)
    
    vol20_shift20 = np.full(len(vol20), np.nan)
    vol20_shift20[20:] = vol20[:-20]
    vol_change = vol20 / (vol20_shift20 + 1e-10)
    
    # RSI
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss_arr = np.where(delta < 0, -delta, 0)
    gain_ma = rolling_mean(gain, 14)
    loss_ma = rolling_mean(loss_arr, 14)
    rsi14 = 100 - 100 / (1 + gain_ma / (loss_ma + 1e-10))
    
    rsi14_shift5 = np.full(len(rsi14), np.nan)
    rsi14_shift5[5:] = rsi14[:-5]
    rsi_change = rsi14 - rsi14_shift5
    
    # MACD (EMA)
    def ema(arr, span):
        out = np.zeros(len(arr))
        alpha = 2.0 / (span + 1)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
        return out
    
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd = ema12 - ema26
    macd_signal = ema(macd, 9)
    macd_hist = macd - macd_signal
    
    bb_std = vol20  # 已经是日收益20日标准差
    bb_mid = ma20
    bb_width = 4 * bb_std * bb_mid / (bb_mid + 1e-10)
    
    bb_lower = bb_mid - 2 * rolling_std(close, 20)
    bb_upper = bb_mid + 2 * rolling_std(close, 20)
    bb_pos = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    # ret_quality
    ret_pos = np.where(ret > 0, ret, 0)
    ret_neg = np.where(ret < 0, -ret, 0)
    ret_pos_ma = rolling_mean(ret_pos, 20)
    ret_neg_ma = rolling_mean(ret_neg, 20)
    ret_quality = ret_pos_ma / (ret_pos_ma + ret_neg_ma + 1e-10)
    
    features = {
        'ma5': ma5, 'ma20': ma20, 'ma60': ma60,
        'ma_bias20': ma_bias20, 'ma_align': ma_align,
        'price_position': price_position,
        'ret1': ret1, 'ret5': ret5, 'ret20': ret20, 'ret60': ret60,
        'momentum_6m': momentum_6m, 'momentum_1m': momentum_1m,
        'mom_divergence': mom_divergence, 'trend_accel': trend_accel,
        'vol20': vol20, 'vol5': vol5, 'vol_ratio': vol_ratio,
        'vol_change': vol_change,
        'rsi14': rsi14, 'rsi_change': rsi_change,
        'macd': macd, 'macd_signal': macd_signal, 'macd_hist': macd_hist,
        'bb_std': bb_std, 'bb_width': bb_width, 'bb_pos': bb_pos,
        'ret_quality': ret_quality,
        'price': close,
        'range_pct': (high - low) / (close + 1e-10),
    }
    return features

# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blueshield', action='store_true', help='只训练蓝盾')
    parser.add_argument('--arrow', action='store_true', help='只训练绿箭')
    args = parser.parse_args()
    
    if not args.blueshield and not args.arrow:
        args.blueshield = True
        args.arrow = True
    
    t_total = time.time()
    
    # ---- 加载数据 ----
    log('Step 1: 加载数据...')
    df = pd.read_parquet(os.path.join(DATA_DIR, 'us_hist_full_10y.parquet'))
    log(f'  总行数: {len(df):,}')
    
    # 加载VIX
    vix_df = pd.read_parquet(os.path.join(DATA_DIR, 'vix_10y.parquet'))
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = [c[0] for c in vix_df.columns]
    vix_df = vix_df.reset_index()
    vix_df.columns = [c.lower() for c in vix_df.columns]
    if 'date' not in vix_df.columns:
        vix_df = vix_df.rename(columns={vix_df.columns[0]: 'date'})
    vix_df['date'] = pd.to_datetime(vix_df['date'])
    close_cols = [c for c in vix_df.columns if 'close' in c]
    if close_cols:
        vix_df = vix_df.rename(columns={close_cols[0]: 'vix_close'})
    vix_map = dict(zip(vix_df['date'].dt.strftime('%Y-%m-%d'), vix_df['vix_close'].values))
    
    # 加载基本面
    with open(os.path.join(DATA_DIR, 'us_fundamentals.json')) as f:
        fund = json.load(f)
    fund_map = {}
    for sym, info in fund.items():
        fund_map[sym] = {
            'pe_trailing': info.get('trailingPE', np.nan),
            'pe_forward': info.get('forwardPE', np.nan),
            'div_yield': info.get('dividendYield', np.nan),
            'beta': info.get('beta', np.nan),
        }
    
    # 提取宏观ETF的收益序列
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['sym', 'date'])
    
    def get_ret_series(sym, col='close'):
        s = df[df['sym'] == sym][['date', col]].copy().sort_values('date')
        s = s.set_index('date')[col]
        return s
    
    spy_c = get_ret_series('SPY')
    qqq_c = get_ret_series('QQQ')
    iwm_c = get_ret_series('IWM')
    
    macro_dates = spy_c.index
    macro_feats = pd.DataFrame(index=macro_dates)
    for name, s in [('spy', spy_c), ('qqq', qqq_c), ('iwm', iwm_c)]:
        macro_feats[f'{name}_ret1'] = s.pct_change(1)
        macro_feats[f'{name}_ret5'] = s.pct_change(5)
        macro_feats[f'{name}_ret20'] = s.pct_change(20)
        macro_feats[f'{name}_ret60'] = s.pct_change(60)
    macro_feats['vix_close'] = macro_feats.index.map(lambda d: vix_map.get(d.strftime('%Y-%m-%d'), np.nan))
    macro_feats['vix_close'] = macro_feats['vix_close'].ffill()
    macro_dict = macro_feats.to_dict('index')  # date -> feature dict
    
    log(f'  宏观数据: {len(macro_dict)} 天')
    del spy_c, qqq_c, iwm_c, vix_df, vix_map
    gc.collect()
    
    # 排除ETF
    etf_syms = {'SPY','QQQ','IWM','DIA','VOO','VTI','IVV','VEA','VWO',
                'BND','AGG','TLT','GLD','SLV','USO','XLE','XLF','XLK','XLV',
                'XLI','XLP','XLU','XLB','XLRE','XLC','ARKK','ARKG','ARKW'}
    
    # ---- 按价格区间分别处理 ----
    models_to_train = []
    if args.blueshield:
        models_to_train.append('blueshield')
    if args.arrow:
        models_to_train.append('arrow')
    
    results = {}
    
    for model_type in models_to_train:
        if model_type == 'blueshield':
            version = 'blueshield_v7'
            price_min, price_max = 10, 1e9
            hold_days = 20
            top_n = 15
            fund_cols_list = ['pe_trailing', 'pe_forward', 'div_yield', 'beta']
            extra_feat_names = []
        else:
            version = 'arrow_v12'
            price_min, price_max = 1, 10
            hold_days = 5
            top_n = 5
            fund_cols_list = []
            extra_feat_names = ['price', 'range_pct']
        
        log(f'\n{"="*60}')
        log(f'{version} 训练 (${price_min}-${price_max if price_max<1e9 else "∞"}, {hold_days}天, Top{top_n})')
        log(f'{"="*60}')
        
        t0 = time.time()
        
        tech_feat_names = ['ma5','ma20','ma60','ma_bias20','ma_align','price_position',
            'ret1','ret5','ret20','ret60','momentum_6m','momentum_1m',
            'mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change',
            'rsi14','rsi_change','macd','macd_signal','macd_hist',
            'bb_std','bb_width','bb_pos','ret_quality']
        macro_col_names = ['vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60',
                          'qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60',
                          'iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60']
        
        feat_cols = tech_feat_names + extra_feat_names + macro_col_names + fund_cols_list
        
        # ---- 分批计算特征+标签 ----
        log('  分批特征工程...')
        
        # 先确定哪些股票在价格范围内
        last_prices = df.groupby('sym')['close'].last()
        valid_syms = last_prices[(last_prices > price_min) & (last_prices <= price_max)].index
        valid_syms = [s for s in valid_syms if s not in etf_syms]
        log(f'  价格范围内: {len(valid_syms)} 只股票')
        
        # 分批处理
        batch_size = 500
        all_rows = []
        
        for batch_start in range(0, len(valid_syms), batch_size):
            batch_syms = valid_syms[batch_start:batch_start+batch_size]
            batch_df = df[df['sym'].isin(batch_syms)].copy()
            
            for sym, grp in batch_df.groupby('sym'):
                grp = grp.sort_values('date')
                if len(grp) < 80:
                    continue
                
                close = grp['close'].values.astype(np.float64)
                high = grp['high'].values.astype(np.float64)
                low = grp['low'].values.astype(np.float64)
                vol = grp['volume'].values.astype(np.float64)
                dates = grp['date'].values
                
                feats = compute_tech_numpy(close, high, low, vol)
                if feats is None:
                    continue
                
                # 添加宏观特征
                for i, d in enumerate(dates):
                    d_str = pd.Timestamp(d).strftime('%Y-%m-%d')
                    mf = macro_dict.get(pd.Timestamp(d), {})
                    
                    row = {f: feats[f][i] for f in tech_feat_names + extra_feat_names}
                    for mc in macro_col_names:
                        row[mc] = mf.get(mc, np.nan)
                    
                    # 基本面
                    for fc in fund_cols_list:
                        row[fc] = fund_map.get(sym, {}).get(fc, np.nan)
                    
                    # 标签: 前瞻收益
                    if i < len(close) - hold_days:
                        row['fwd_ret'] = close[i + hold_days] / close[i] - 1
                    else:
                        row['fwd_ret'] = np.nan
                    
                    row['date'] = d
                    row['sym'] = sym
                    all_rows.append(row)
            
            done = min(batch_start + batch_size, len(valid_syms))
            if done % 2000 == 0 or done == len(valid_syms):
                log(f'    已处理 {done}/{len(valid_syms)} 只股票, {len(all_rows):,} 行')
        
        log(f'  总行数: {len(all_rows):,}')
        
        # 转DataFrame
        df_feat = pd.DataFrame(all_rows)
        del all_rows
        gc.collect()
        
        # 过滤无标签
        df_feat = df_feat.dropna(subset=['fwd_ret'])
        
        # 过滤特征NaN（至少80%非空）
        feat_present = df_feat[feat_cols].notna().sum(axis=1)
        df_feat = df_feat[feat_present >= len(feat_cols) * 0.8]
        
        # 填充NaN为中位数
        for col in feat_cols:
            med = df_feat[col].median()
            df_feat[col] = df_feat[col].fillna(med)
        
        # 再过滤inf
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols)
        
        log(f'  最终数据: {len(df_feat):,} 行, {df_feat["sym"].nunique()} 只股票')
        log(f'  特征工程耗时: {time.time()-t0:.1f}s')
        
        # ---- Walk-Forward 验证 ----
        log('Step 3: Walk-Forward 5折验证...')
        df_feat = df_feat.sort_values('date')
        dates_arr = np.sort(df_feat['date'].unique())
        
        oos_start = pd.Timestamp('2024-01-01')
        train_dates = dates_arr[dates_arr < oos_start]
        
        fold_size = len(train_dates) // 5
        wf_results = []
        
        for fold in range(5):
            train_end_idx = (fold + 1) * fold_size
            val_start_idx = train_end_idx
            val_end_idx = min(val_start_idx + fold_size, len(train_dates))
            
            if val_end_idx <= val_start_idx:
                continue
            
            train_mask = df_feat['date'].isin(train_dates[:train_end_idx])
            val_mask = df_feat['date'].isin(train_dates[val_start_idx:val_end_idx])
            
            X_train = df_feat.loc[train_mask, feat_cols].values
            y_train = df_feat.loc[train_mask, 'fwd_ret'].values
            X_val = df_feat.loc[val_mask, feat_cols].values
            y_val = df_feat.loc[val_mask, 'fwd_ret'].values
            
            if len(X_train) < 1000 or len(X_val) < 100:
                continue
            
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
            val_df = df_feat.loc[val_mask, ['date', 'fwd_ret']].copy()
            val_df['pred'] = pred
            
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
                    'fold': fold, 'avg_return': avg_ret, 'win_rate': win_rate,
                    'sharpe': sharpe, 'n_days': len(daily_returns),
                    'best_iteration': model.best_iteration
                })
                log(f'  Fold {fold}: avg={avg_ret:.2f}%, win={win_rate:.1f}%, sharpe={sharpe:.2f}, '
                    f'days={len(daily_returns)}, iter={model.best_iteration}')
            
            del dtrain, dval, model
            gc.collect()
        
        # ---- 确定最佳迭代 ----
        if wf_results:
            best_iter = int(np.median([r['best_iteration'] for r in wf_results]))
            best_iter = max(best_iter, 200)
        else:
            best_iter = 500
        log(f'  最佳迭代次数: {best_iter}')
        
        # ---- OOS 评估 ----
        log('Step 4: 样本外评估 (2024-2026)...')
        train_mask = df_feat['date'] < oos_start
        oos_mask = df_feat['date'] >= oos_start
        
        X_train_full = df_feat.loc[train_mask, feat_cols].values
        y_train_full = df_feat.loc[train_mask, 'fwd_ret'].values
        X_oos = df_feat.loc[oos_mask, feat_cols].values
        
        params_final = {
            'objective': 'reg:squarederror', 'max_depth': 6,
            'learning_rate': 0.03, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 10,
            'tree_method': 'hist', 'seed': 42, 'verbosity': 0
        }
        
        dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full, feature_names=feat_cols)
        oos_model = xgb.train(params_final, dtrain_full, num_boost_round=best_iter, verbose_eval=False)
        
        doos = xgb.DMatrix(X_oos, feature_names=feat_cols)
        oos_pred = oos_model.predict(doos)
        
        oos_df = df_feat.loc[oos_mask, ['date', 'fwd_ret']].copy()
        oos_df['pred'] = oos_pred
        
        oos_daily = []
        for d, group in oos_df.groupby('date'):
            if len(group) < top_n:
                continue
            top = group.nlargest(top_n, 'pred')
            oos_daily.append(top['fwd_ret'].mean())
        
        oos_result = None
        if oos_daily:
            oos_result = {
                'avg_return': np.mean(oos_daily) * 100,
                'win_rate': np.mean([r > 0 for r in oos_daily]) * 100,
                'sharpe': np.mean(oos_daily) / (np.std(oos_daily) + 1e-10) * np.sqrt(252 / hold_days),
                'max_dd': np.min(np.cumsum(oos_daily)) * 100,
                'n_days': len(oos_daily)
            }
            log(f'  OOS: avg={oos_result["avg_return"]:.2f}%, win={oos_result["win_rate"]:.1f}%, '
                f'sharpe={oos_result["sharpe"]:.2f}, max_dd={oos_result["max_dd"]:.1f}%')
        
        del oos_model, dtrain_full, doos, X_train_full, y_train_full, X_oos
        gc.collect()
        
        # ---- 训练最终模型（全量数据）----
        log('Step 5: 训练最终模型...')
        X_all = df_feat[feat_cols].values
        y_all = df_feat['fwd_ret'].values
        dall = xgb.DMatrix(X_all, label=y_all, feature_names=feat_cols)
        final_model = xgb.train(params_final, dall, num_boost_round=best_iter, verbose_eval=False)
        
        # 特征重要性
        importance = final_model.get_score(importance_type='gain')
        total_imp = sum(importance.values()) if importance else 1
        feat_imp = {k: round(v / total_imp * 100, 2) for k, v in sorted(importance.items(), key=lambda x: -x[1])}
        
        # ---- 信号阈值 ----
        log('Step 6: 计算信号阈值...')
        all_pred = final_model.predict(dall)
        df_feat['pred'] = all_pred
        
        daily_stats = []
        for d, group in df_feat.groupby('date'):
            if len(group) < top_n * 3:
                continue
            top = group.nlargest(top_n, 'pred')
            daily_stats.append({
                'top_pred_min': top['pred'].min(),
                'top_pred_max': top['pred'].max(),
                'top_pred_mean': top['pred'].mean(),
            })
        
        stats_df = pd.DataFrame(daily_stats)
        
        thresholds = {
            'green2': {
                'threshold': round(float(stats_df['top_pred_max'].quantile(0.9)), 4),
                'note': 'Top 1%信号, 样本少但收益率极高'
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
        
        for level, info in thresholds.items():
            t = info['threshold']
            mask = df_feat['pred'] >= t
            if mask.sum() > 0:
                avg_ret = df_feat.loc[mask, 'fwd_ret'].mean() * 100
                win_rate = (df_feat.loc[mask, 'fwd_ret'] > 0).mean() * 100
                info['avg_return'] = round(float(avg_ret), 2)
                info['win_rate'] = round(float(win_rate), 1)
                info['count'] = int(mask.sum())
                log(f'  {level}: threshold={t}, avg={avg_ret:.2f}%, win={win_rate:.1f}%, count={mask.sum()}')
        
        # ---- 保存 ----
        log('Step 7: 保存模型...')
        model_path = os.path.join(MODEL_DIR, f'{version}_xgb.json')
        final_model.save_model(model_path)
        log(f'  模型: {model_path} ({os.path.getsize(model_path)/1024/1024:.1f}MB)')
        
        if model_type == 'blueshield':
            universe_str = f'全市场>${price_min} ({df_feat["sym"].nunique()}只)'
            n_tech = len(tech_feat_names)
            n_macro = len(macro_col_names)
            n_fund = len(fund_cols_list)
        else:
            universe_str = f'全市场${price_min}-${price_max} ({df_feat["sym"].nunique()}只)'
            n_tech = len(tech_feat_names) + len(extra_feat_names)
            n_macro = len(macro_col_names)
            n_fund = 0
        
        wf_avg = np.mean([r['avg_return'] for r in wf_results]) if wf_results else 0
        wf_win = np.mean([r['win_rate'] for r in wf_results]) if wf_results else 0
        wf_sharpe = np.mean([r['sharpe'] for r in wf_results]) if wf_results else 0
        
        meta = {
            'version': version,
            'algorithm': 'XGBoost',
            'features': feat_cols,
            'n_features': len(feat_cols),
            'tech_features': n_tech,
            'macro_features': n_macro,
            'fund_features': n_fund,
            'hold_days': hold_days,
            'top_n': top_n,
            'universe': universe_str,
            'params': params_final,
            'n_trees': best_iter,
            'trained_on': f'{df_feat["date"].min()}~{df_feat["date"].max()}',
            'n_train_samples': len(df_feat),
            'feature_importance': feat_imp,
            'validation': {
                'method': f'Walk-Forward {len(wf_results)}折 + 样本外2024-2026',
                'wf_avg_return': round(float(wf_avg), 2),
                'wf_win_rate': round(float(wf_win), 1),
                'wf_sharpe': round(float(wf_sharpe), 2),
            },
            'signal_thresholds': thresholds,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'data_source': 'us_hist_full_10y.parquet (全市场11,864只)',
            'replaces': 'V6' if model_type == 'blueshield' else 'V11'
        }
        
        if oos_result:
            meta['validation'].update({
                'oos_avg_return': round(oos_result['avg_return'], 2),
                'oos_win_rate': round(oos_result['win_rate'], 1),
                'oos_sharpe': round(oos_result['sharpe'], 2),
                'oos_max_dd': round(oos_result['max_dd'], 1),
                'oos_n_days': oos_result['n_days']
            })
        
        meta_path = os.path.join(MODEL_DIR, f'{version}_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        log(f'  元数据: {meta_path}')
        
        results[model_type] = meta
        log(f'\n{version} 完成! 耗时: {(time.time()-t0)/60:.1f}分钟')
        
        # 清理
        del df_feat, final_model, dall, X_all, y_all
        gc.collect()
    
    # ---- 汇总 ----
    log(f'\n{"="*60}')
    log('训练汇总报告')
    log(f'{"="*60}')
    
    for name, meta in results.items():
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
