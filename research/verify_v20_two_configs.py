#!/usr/bin/env python3
"""独立验证：cn-alpha-v2.0 两条配置对比
1) 无止损 (原V2.0)
2) SL-3% (CEO版)
用相同的WF folds、相同的特征、相同的模型参数，独立重跑。
"""
import pandas as pd, numpy as np, json, time, os, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

t0 = time.time()
print(f"[验证] cn-alpha-v2.0 两条路对比 {time.strftime('%Y-%m-%d %H:%M')}")

# ========== 加载数据 ==========
print("[1/5] 加载数据...")
df = pd.read_parquet('data/a_hist_10y.parquet')
df = df.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df['date'] = df['date'].astype(int)
mf = pd.read_parquet('data/cn/moneyflow_core.parquet')
mf['sym'] = mf['ts_code'].str[:6]
mf['date'] = mf['trade_date'].astype(int)
for col in ['sm', 'md', 'lg', 'elg']:
    mf[f'{col}_net'] = mf[f'buy_{col}_amount'] - mf[f'sell_{col}_amount']
mf['total_net'] = mf['net_mf_amount']
df = df.merge(mf[['sym', 'date', 'total_net', 'lg_net', 'md_net', 'elg_net']], on=['sym', 'date'], how='left')
df = df[~df['sym'].str.startswith('688')].copy()
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
df = df[df['volume'] > 0].copy()
df = df.sort_values(['sym', 'date']).reset_index(drop=True)
print(f"  Data: {len(df):,} rows, {df['sym'].nunique()} stocks")

# ========== Price lookup ==========
print("[2/5] Price lookup...")
price_lookup = dict(zip(zip(df['sym'], df['date']), df['close']))

# ========== 特征 ==========
print("[3/5] 计算特征...")
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ret10'] = df.groupby('sym')['close'].pct_change(10)
df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma60'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(60, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['ma60_bias'] = (df['close'] - df['ma60']) / df['ma60']
df['vol5'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(5, min_periods=2).std())
df['vol20'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(20, min_periods=2).std())
delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)
ema12 = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=12, min_periods=1).mean())
ema26 = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=26, min_periods=1).mean())
df['macd'] = ema12 - ema26
df['macd_signal'] = df.groupby('sym')['macd'].transform(lambda x: x.ewm(span=9, min_periods=1).mean())
df['macd_hist'] = df['macd'] - df['macd_signal']
df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df.groupby('sym')['close'].shift(1)), abs(df['low'] - df.groupby('sym')['close'].shift(1))))
df['atr14'] = df.groupby('sym')['tr'].transform(lambda x: x.rolling(14, min_periods=1).mean())
df['atr_pct'] = df['atr14'] / df['close']
df['vol_ratio'] = df.groupby('sym')['volume'].transform(lambda x: x.rolling(5).mean()) / df.groupby('sym')['volume'].transform(lambda x: x.rolling(20).mean())
for col in ['total_net', 'lg_net', 'md_net', 'elg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())
    df[f'{col}_20d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(20, min_periods=1).sum())
    df[f'{col}_5d_rk'] = df.groupby('date')[f'{col}_5d'].rank(pct=True)
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')
df['fwd_10d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-10) / x - 1)

XGB_FEATURES = [
    'ret5', 'ret10', 'ret20', 'ma20_bias', 'ma60_bias',
    'vol5', 'vol20', 'rsi_14', 'macd_hist', 'atr_pct', 'vol_ratio',
    'total_net_5d', 'lg_net_5d', 'md_net_5d', 'elg_net_5d',
    'total_net_20d', 'lg_net_20d', 'md_net_20d', 'elg_net_20d',
    'total_net_5d_rk', 'lg_net_5d_rk', 'md_net_5d_rk', 'elg_net_5d_rk',
    'breadth', 'mkt_ret20'
]
TOP_N = 15
COST = 0.0015  # 0.15% 单边

# ========== WF folds ==========
print("[4/5] 构建WF folds...")
all_dates = sorted(df['date'].unique())
def int_to_dt(d): return datetime(int(str(d)[:4]), int(str(d)[4:6]), int(str(d)[6:8]))
def dt_to_int(d): return int(d.strftime('%Y%m%d'))

folds = []
train_start_dt = datetime(2016, 1, 1)
while True:
    train_end_dt = train_start_dt + timedelta(days=365*2)
    test_start_dt = train_end_dt
    test_end_dt = test_start_dt + timedelta(days=182)
    if test_end_dt > int_to_dt(all_dates[-1]):
        break
    td = [d for d in all_dates if dt_to_int(train_start_dt) <= d <= dt_to_int(train_end_dt)]
    ted = [d for d in all_dates if dt_to_int(test_start_dt) <= d <= dt_to_int(test_end_dt)]
    if len(td) >= 200 and len(ted) >= 20:
        folds.append({'train_dates': td, 'test_dates': ted, 'train_start': dt_to_int(train_start_dt), 'test_end': dt_to_int(test_end_dt)})
    train_start_dt += timedelta(days=182)

print(f"  {len(folds)} folds")

# ========== 验证 ==========
print("[5/5] WF验证 (两条路)...")
import xgboost as xgb

configs = [
    {'hold': 10, 'sl': None, 'label': 'V2.0_无止损'},
    {'hold': 10, 'sl': -0.03, 'label': 'V2.0_SL3%'},
]

results = {c['label']: {'fold_sharpes': [], 'all_rets': [], 'fold_details': []} for c in configs}

for fold_idx, fold in enumerate(folds):
    t1 = time.time()
    train = df[df['date'].isin(fold['train_dates'])].dropna(subset=XGB_FEATURES + ['fwd_10d'])
    if len(train) < 1000:
        print(f"  Fold {fold_idx+1}: SKIP (only {len(train)} rows)")
        continue
    
    # 用和CEO完全相同的模型参数
    model = xgb.XGBRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=4, verbosity=0
    )
    model.fit(train[XGB_FEATURES].fillna(0), train['fwd_10d'])
    
    test_data = df[df['date'].isin(fold['test_dates'])].copy()
    test_data['xgb_score'] = model.predict(test_data[XGB_FEATURES].fillna(0))
    test_data = test_data[
        (test_data['close'] >= 3) & (test_data['close'] <= 200) &
        (~test_data['sym'].str.contains('ST|退市', na=False)) & (test_data['volume'] > 0)
    ]
    
    test_dates = fold['test_dates']
    
    for cfg in configs:
        hold = cfg['hold']
        sl = cfg['sl']
        label = cfg['label']
        
        rebal_dates = test_dates[::hold]
        strat_rets = []
        
        for rd in rebal_dates:
            day = test_data[test_data['date'] == rd]
            if len(day) < 50:
                continue
            top = day.nlargest(TOP_N, 'xgb_score')
            
            rebal_idx = test_dates.index(rd)
            exit_idx = min(rebal_idx + hold, len(test_dates) - 1)
            exit_date = test_dates[exit_idx]
            if exit_date == rd:
                continue
            
            rets = []
            for _, row in top.iterrows():
                exit_price = price_lookup.get((row['sym'], exit_date))
                if exit_price is None:
                    continue
                ret = exit_price / row['close'] - 1
                
                # Stop loss (daily check)
                if sl is not None:
                    for i in range(rebal_idx + 1, exit_idx + 1):
                        ip = price_lookup.get((row['sym'], test_dates[i]))
                        if ip is not None and ip / row['close'] - 1 <= sl:
                            ret = sl
                            break
                
                ret -= COST
                rets.append(ret)
            
            if rets:
                strat_rets.append(np.mean(rets))
        
        if len(strat_rets) > 2:
            avg = np.mean(strat_rets)
            std = np.std(strat_rets)
            rebal_per_year = 252 / hold
            fold_sharpe = (avg * rebal_per_year) / (std * np.sqrt(rebal_per_year)) if std > 0 else 0
            results[label]['fold_sharpes'].append(fold_sharpe)
            results[label]['all_rets'].extend(strat_rets)
            results[label]['fold_details'].append({
                'fold': fold_idx + 1,
                'period': f"{fold['train_start']}-{fold['test_end']}",
                'n_rebal': len(strat_rets),
                'avg_ret': round(avg, 6),
                'std': round(std, 6),
                'sharpe': round(fold_sharpe, 4)
            })
    
    print(f"  Fold {fold_idx+1}: {time.time()-t1:.0f}s")

# ========== 输出 ==========
print("\n" + "=" * 90)
print("📊 验证结果：cn-alpha-v2.0 两条路对比")
print("=" * 90)

for cfg in configs:
    label = cfg['label']
    d = results[label]
    fs = d['fold_sharpes']
    ar = d['all_rets']
    if not fs:
        print(f"\n{label}: 无数据")
        continue
    
    avg_ret = np.mean(ar)
    std_ret = np.std(ar)
    wr = np.mean([r > 0 for r in ar])
    rebal_per_year = 252 / cfg['hold']
    ann_ret = avg_ret * rebal_per_year
    ann_std = std_ret * np.sqrt(rebal_per_year)
    overall_sharpe = ann_ret / ann_std if ann_std > 0 else 0
    
    print(f"\n{label}")
    print(f"  WF折数:     {len(fs)}")
    print(f"  平均Sharpe: {np.mean(fs):.3f} ± {np.std(fs):.3f}")
    print(f"  Sharpe中位: {np.median(fs):.3f}")
    print(f"  Sharpe/Std: {np.mean(fs)/np.std(fs):.2f}" if np.std(fs) > 0 else "  Sharpe/Std: inf")
    print(f"  正Sharpe:   {sum(1 for s in fs if s > 0)}/{len(fs)} ({sum(1 for s in fs if s > 0)/len(fs)*100:.0f}%)")
    print(f"  总交易数:   {len(ar)}")
    print(f"  胜率:       {wr:.1%}")
    print(f"  平均收益:   {avg_ret:.4f} ({avg_ret*100:.2f}%)")
    print(f"  年化收益:   {ann_ret:.1%}")
    print(f"  年化波动:   {ann_std:.1%}")
    print(f"  整体Sharpe: {overall_sharpe:.3f}")
    print(f"  每折详情:")
    for fd in d['fold_details']:
        flag = "✅" if fd['sharpe'] > 1.0 else "⚠️" if fd['sharpe'] > 0 else "❌"
        print(f"    {flag} F{fd['fold']:>2}: {fd['period']} | Rebal={fd['n_rebal']:>3} | Ret={fd['avg_ret']*100:>6.2f}% | Std={fd['std']*100:>5.2f}% | Sharpe={fd['sharpe']:>6.2f}")

# 对比
print("\n" + "=" * 90)
print("📊 对比总结")
print("=" * 90)
d1 = results['V2.0_无止损']
d2 = results['V2.0_SL3%']
if d1['fold_sharpes'] and d2['fold_sharpes']:
        fs1, fs2 = d1['fold_sharpes'], d2['fold_sharpes']
        print(f"{'指标':<20} {'无止损':>12} {'SL-3%':>12} {'差异':>12}")
        print("-" * 60)
        print(f"{'WF Sharpe均值':<20} {np.mean(fs1):>12.3f} {np.mean(fs2):>12.3f} {np.mean(fs2)-np.mean(fs1):>+12.3f}")
        print(f"{'WF Sharpe标准差':<20} {np.std(fs1):>12.3f} {np.std(fs2):>12.3f} {np.std(fs2)-np.std(fs1):>+12.3f}")
        print(f"{'Sharpe/Std(稳定性)':<20} {np.mean(fs1)/np.std(fs1) if np.std(fs1)>0 else 99:>12.2f} {np.mean(fs2)/np.std(fs2) if np.std(fs2)>0 else 99:>12.2f}")
        print(f"{'正Sharpe折数':<20} {sum(1 for s in fs1 if s>0):>10}/{len(fs1)} {sum(1 for s in fs2 if s>0):>10}/{len(fs2)}")
        print(f"{'胜率':<20} {np.mean([r>0 for r in d1['all_rets']]):>12.1%} {np.mean([r>0 for r in d2['all_rets']]):>12.1%}")

print(f"\n⏱️ 总耗时: {time.time()-t0:.0f}s")
