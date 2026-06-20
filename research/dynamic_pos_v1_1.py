#!/usr/bin/env python3
"""
cn-alpha-v1.1 动态仓位回测
方案B: Bull100% / Cautious50% / Weak50% / Bear0%
"""
import pandas as pd, numpy as np, xgboost as xgb, json, time, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("=" * 60)
print("cn-alpha-v1.1 动态仓位回测")
print("=" * 60)
t0 = time.time()

# 数据加载（同前）
df = pd.read_parquet('data/cn/features_v2.parquet')
df['date'] = pd.to_datetime(df['date'])
df['date_int'] = df['date'].dt.strftime('%Y%m%d').astype(int)
basic = pd.read_parquet('data/cn/daily_basic.parquet')
basic['sym'] = basic['ts_code'].str[:6]
basic['date_int'] = basic['trade_date'].astype(int)
basic = basic[['sym', 'date_int', 'pe_ttm', 'pb', 'ps_ttm', 'dv_ratio']].drop_duplicates(['sym', 'date_int'])
df = df.merge(basic, on=['sym', 'date_int'], how='left')
df = df[df['close'] > 1]
df = df[df['volume'] > 0]
df['daily_ret'] = df.groupby('sym')['close'].pct_change()
df = df[df['daily_ret'].abs() < 0.12]
df = df.drop(columns=['daily_ret'])

# 特征
df['rev_5d'] = -df['r5']; df['rev_10d'] = -df['r10']; df['rev_20d'] = -df['r20']
df['rsi_reversal'] = -df['rsi14']; df['macd_reversal'] = -df['macd']
df['low_vol_5d'] = -df['vol5']; df['low_vol_20d'] = -df['vol20']; df['low_atr'] = -df['atr_pct']
df['small_cap'] = -df['log_circ_mv']
df['residual_mom_5d'] = df['r5'] - df.groupby('date')['r5'].transform('mean')
df['residual_mom_20d'] = df['r20'] - df.groupby('date')['r20'].transform('mean')
df['lg_flow_momentum'] = df['lg_net_5'] - df['lg_net_20'] / 4
df['total_flow_momentum'] = df['total_net_5'] - df['total_net_20'] / 4
for col in ['lg_net_20', 'md_net_20', 'total_net_20']:
    df[f'{col}_rank'] = df.groupby('date')[col].rank(pct=True)
df['rev_flow_interaction'] = df['rev_20d'] * df['lg_net_20_rank']
df['turnover_rank'] = df.groupby('date')['vol_r'].rank(pct=True)
df['pe_clean'] = df['pe_ttm'].where((df['pe_ttm'] > 0) & (df['pe_ttm'] < 500), np.nan)
df['pe_rank'] = df.groupby('date')['pe_clean'].rank(pct=True, ascending=True)
df['pe_inverse'] = 1.0 / df['pe_clean'].clip(lower=1)
df['pb_clean'] = df['pb'].where((df['pb'] > 0) & (df['pb'] < 100), np.nan)
df['pb_rank'] = df.groupby('date')['pb_clean'].rank(pct=True, ascending=True)
df['pb_inverse'] = 1.0 / df['pb_clean'].clip(lower=0.1)
df['div_rank'] = df.groupby('date')['dv_ratio'].rank(pct=True, ascending=False)
df['ps_clean'] = df['ps_ttm'].where((df['ps_ttm'] > 0) & (df['ps_ttm'] < 200), np.nan)
df['ps_rank'] = df.groupby('date')['ps_clean'].rank(pct=True, ascending=True)

features = [
    'rev_5d','rev_10d','rev_20d','rsi_reversal','macd_reversal','macd_hist',
    'low_vol_5d','low_vol_20d','low_atr',
    'md_net_5','md_net_20','lg_net_5','lg_net_20','total_net_5','total_net_20',
    'small_cap','residual_mom_5d','residual_mom_20d',
    'lg_flow_momentum','total_flow_momentum',
    'lg_net_20_rank','md_net_20_rank','total_net_20_rank',
    'rev_flow_interaction','turnover_rank',
    'pe_rank','pe_inverse','pb_rank','pb_inverse','div_rank','ps_rank',
    'vol_r','sm_net_5','sm_net_20','elg_net_5','elg_net_20',
]

# 市场状态
market_daily = df.groupby('date_int')['r1'].mean()
market_ma60 = market_daily.rolling(60).mean()
market_ma120 = market_daily.rolling(120).mean()
market_ret20 = market_daily.rolling(20).sum()
adv_dec = df.groupby('date_int').apply(lambda x: (x['r1'] > 0).sum() / max((x['r1'] < 0).sum(), 1))
market_state = pd.DataFrame({'ret20': market_ret20, 'ma60': market_ma60, 'ma120': market_ma120, 'adv_dec': adv_dec})

def classify(row):
    if pd.isna(row['ma60']) or pd.isna(row['ma120']): return 'unknown'
    ma_bull = row['ma60'] > row['ma120']
    mom_pos = row['ret20'] > 0 if not pd.isna(row['ret20']) else True
    breadth = row['adv_dec'] > 0.4 if not pd.isna(row['adv_dec']) else True
    if not ma_bull and not mom_pos: return 'bear'
    elif not ma_bull or not mom_pos: return 'cautious'
    elif not breadth: return 'weak'
    else: return 'bull'

market_state['regime'] = market_state.apply(classify, axis=1)

all_dates = sorted(df['date_int'].unique())
df_model = df.dropna(subset=features + ['fwd20'])

train_window = 504
step = 21
top_n = 30
hold = 20

# Walk-Forward
print(f"Walk-Forward训练中...")
wf_results = []
predictions = []
start_idx = train_window

while start_idx + hold <= len(all_dates):
    train_dates = all_dates[start_idx - train_window:start_idx]
    test_dates = all_dates[start_idx:start_idx + hold]
    train = df_model[df_model['date_int'].isin(train_dates)]
    test = df_model[df_model['date_int'].isin(test_dates)]
    if len(train) < 1000 or len(test) < 100:
        start_idx += step; continue
    X_train = train[features].fillna(0)
    y_train_rank = train['fwd20'].groupby(train['date_int']).rank(pct=True)
    X_test = test[features].fillna(0)
    model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method='hist', device='cuda',
        random_state=42, verbosity=0)
    model.fit(X_train, y_train_rank)
    test = test.copy()
    test['pred'] = model.predict(X_test)
    for td in test['date_int'].unique():
        t = test[test['date_int'] == td]
        if len(t) > 50:
            ic = t['pred'].corr(t['fwd20'], method='spearman')
            wf_results.append({'date': td, 'ic': ic})
    predictions.append(test[['date_int', 'sym', 'close', 'pred', 'fwd20']])
    start_idx += step

wf_df = pd.DataFrame(wf_results)
print(f"IC={wf_df['ic'].mean():.4f} ICIR={wf_df['ic'].mean()/wf_df['ic'].std():.3f}")

# ============================================================
# 三种方案对比
# ============================================================
all_pred = pd.concat(predictions)
test_periods = sorted(all_pred['date_int'].unique())
rebalance_dates = test_periods[::hold]

configs = {
    'A: Bull30%/Cau30%': {'bull': 0.30, 'cautious': 0.30, 'weak': 0.30, 'bear': 0},
    'B: Bull100%/Cau50%': {'bull': 1.00, 'cautious': 0.50, 'weak': 0.50, 'bear': 0},
    'C: Bull70%/Cau35%': {'bull': 0.70, 'cautious': 0.35, 'weak': 0.35, 'bear': 0},
    'D: Bull100%/Cau75%': {'bull': 1.00, 'cautious': 0.75, 'weak': 0.75, 'bear': 0},
}

ann_factor = 252 / hold
all_results = {}

for cfg_name, positions in configs.items():
    portfolio_value = 1.0
    prev_syms = set()
    history = [{'date': rebalance_dates[0], 'value': 1.0, 'regime': 'init', 'pos': 0}]
    
    for i, rd in enumerate(rebalance_dates):
        regime = market_state.loc[rd, 'regime'] if rd in market_state.index else 'unknown'
        pos_pct = positions.get(regime, 0)
        
        # 上期收益
        if i > 0 and prev_syms and pos_pct > 0:
            prev_rd = rebalance_dates[i-1]
            entry_df = all_pred[all_pred['date_int'] == prev_rd]
            exit_df = all_pred[all_pred['date_int'] == rd]
            
            rets = []
            for sym in prev_syms:
                ep_row = entry_df[entry_df['sym'] == sym]
                ex_row = exit_df[exit_df['sym'] == sym]
                if len(ep_row) > 0 and len(ex_row) > 0:
                    ret = (ex_row.iloc[0]['close'] - ep_row.iloc[0]['close']) / ep_row.iloc[0]['close']
                    rets.append(ret)
            
            if rets:
                port_ret = np.mean(rets) * pos_pct
                # 交易成本
                new_signal = all_pred[all_pred['date_int'] == rd]
                new_signal = new_signal[new_signal['close'] > 3]
                if len(new_signal) >= top_n:
                    top = new_signal.nlargest(top_n, 'pred')
                    new_syms = set(top['sym'])
                    turnover = len(new_syms - prev_syms) / top_n if prev_syms else 1.0
                    cost = turnover * 0.0015 * pos_pct
                    port_ret -= cost
                
                portfolio_value *= (1 + port_ret)
        
        # 选股
        if pos_pct > 0:
            signal = all_pred[all_pred['date_int'] == rd]
            signal = signal[signal['close'] > 3]
            if len(signal) >= top_n:
                top = signal.nlargest(top_n, 'pred')
                prev_syms = set(top['sym'])
            else:
                prev_syms = set()
        else:
            prev_syms = set()
        
        history.append({'date': rd, 'value': portfolio_value, 'regime': regime, 'pos': pos_pct})
    
    hdf = pd.DataFrame(history)
    total_ret = portfolio_value - 1.0
    years = len(history) * hold / 252
    ann_ret = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
    
    hdf['peak'] = hdf['value'].cummax()
    hdf['dd'] = (hdf['value'] - hdf['peak']) / hdf['peak']
    max_dd = hdf['dd'].min()
    
    period_rets = hdf['value'].pct_change().dropna()
    sharpe = period_rets.mean() / period_rets.std() * np.sqrt(ann_factor) if period_rets.std() > 0 else 0
    
    # 分年
    hdf['year'] = hdf['date'] // 10000
    yearly = []
    for y, g in hdf.groupby('year'):
        if len(g) < 2: continue
        y_ret = g['value'].iloc[-1] / g['value'].iloc[0] - 1
        y_dd = g['dd'].min()
        cash_pct = (g['pos'] == 0).mean() * 100
        yearly.append({'year': y, 'ret': y_ret, 'dd': y_dd, 'cash_pct': cash_pct})
    
    all_results[cfg_name] = {
        'ann_ret': ann_ret, 'sharpe': sharpe, 'max_dd': max_dd,
        'total_ret': total_ret, 'yearly': yearly
    }

# ============================================================
# 结果
# ============================================================
print(f"\n{'='*80}")
print(f"{'方案':<22} {'年化':>8} {'Sharpe':>8} {'DD':>8} {'总收益':>10}")
print("-" * 80)
for name, r in all_results.items():
    print(f"{name:<22} {r['ann_ret']*100:>+7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>+7.1f}% {r['total_ret']*100:>+9.1f}%")

# 方案B分年详情
print(f"\n方案B分年度:")
for y in all_results['B: Bull100%/Cau50%']['yearly']:
    emoji = "🟢" if y['ret'] > 0 else "🔴"
    print(f"  {emoji} {y['year']}: 收益{y['ret']*100:+.1f}% DD{y['dd']*100:.1f}% 空仓{y['cash_pct']:.0f}%")

print(f"\n耗时: {time.time()-t0:.0f}秒")
