#!/usr/bin/env python3
"""
cn-alpha-v1.1: 反转+资金流+基本面因子
用已拉取的daily_basic数据
"""
import pandas as pd, numpy as np, xgboost as xgb, json, time, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("=" * 60)
print("cn-alpha-v1.1 — 反转+资金流+基本面")
print("=" * 60)
t0 = time.time()

# 加载特征数据
df = pd.read_parquet('data/cn/features_v2.parquet')
df['date'] = pd.to_datetime(df['date'])
df['date_int'] = df['date'].dt.strftime('%Y%m%d').astype(int)

# 加载daily_basic
basic = pd.read_parquet('data/cn/daily_basic.parquet')
basic['sym'] = basic['ts_code'].str[:6]
basic['date_int'] = basic['trade_date'].astype(int)
basic_cols = ['sym', 'date_int', 'pe_ttm', 'pb', 'ps_ttm', 'dv_ratio']
basic = basic[basic_cols].drop_duplicates(['sym', 'date_int'])

# 合并
df = df.merge(basic, on=['sym', 'date_int'], how='left')
print(f"数据: {len(df):,}行, PE非空{df['pe_ttm'].notna().mean()*100:.1f}%, PB非空{df['pb'].notna().mean()*100:.1f}%")

# 清洗
df = df[df['close'] > 1]
df = df[df['volume'] > 0]
df['daily_ret'] = df.groupby('sym')['close'].pct_change()
df = df[df['daily_ret'].abs() < 0.12]
df = df.drop(columns=['daily_ret'])

# === 反转特征 ===
df['rev_5d'] = -df['r5']
df['rev_10d'] = -df['r10']
df['rev_20d'] = -df['r20']
df['rsi_reversal'] = -df['rsi14']
df['macd_reversal'] = -df['macd']
df['low_vol_5d'] = -df['vol5']
df['low_vol_20d'] = -df['vol20']
df['low_atr'] = -df['atr_pct']
df['small_cap'] = -df['log_circ_mv']
df['residual_mom_5d'] = df['r5'] - df.groupby('date')['r5'].transform('mean')
df['residual_mom_20d'] = df['r20'] - df.groupby('date')['r20'].transform('mean')

# === 资金流交互 ===
df['lg_flow_momentum'] = df['lg_net_5'] - df['lg_net_20'] / 4
df['total_flow_momentum'] = df['total_net_5'] - df['total_net_20'] / 4
for col in ['lg_net_20', 'md_net_20', 'total_net_20']:
    df[f'{col}_rank'] = df.groupby('date')[col].rank(pct=True)
df['rev_flow_interaction'] = df['rev_20d'] * df['lg_net_20_rank']
df['turnover_rank'] = df.groupby('date')['vol_r'].rank(pct=True)

# === 基本面特征 ===
df['pe_clean'] = df['pe_ttm'].where(df['pe_ttm'] > 0, np.nan)
df['pe_rank'] = df.groupby('date')['pe_clean'].rank(pct=True, ascending=True)
df['pe_inverse'] = 1.0 / df['pe_clean'].clip(lower=1)
df['pb_clean'] = df['pb'].where(df['pb'] > 0, np.nan)
df['pb_rank'] = df.groupby('date')['pb_clean'].rank(pct=True, ascending=True)
df['pb_inverse'] = 1.0 / df['pb_clean'].clip(lower=0.1)
df['div_rank'] = df.groupby('date')['dv_ratio'].rank(pct=True, ascending=False)
df['ps_clean'] = df['ps_ttm'].where(df['ps_ttm'] > 0, np.nan)
df['ps_rank'] = df.groupby('date')['ps_clean'].rank(pct=True, ascending=True)

features = [
    'rev_5d', 'rev_10d', 'rev_20d', 'rsi_reversal', 'macd_reversal', 'macd_hist',
    'low_vol_5d', 'low_vol_20d', 'low_atr',
    'md_net_5', 'md_net_20', 'lg_net_5', 'lg_net_20', 'total_net_5', 'total_net_20',
    'small_cap', 'residual_mom_5d', 'residual_mom_20d',
    'lg_flow_momentum', 'total_flow_momentum',
    'lg_net_20_rank', 'md_net_20_rank', 'total_net_20_rank',
    'rev_flow_interaction', 'turnover_rank',
    'pe_rank', 'pe_inverse', 'pb_rank', 'pb_inverse', 'div_rank', 'ps_rank',
    'vol_r', 'sm_net_5', 'sm_net_20', 'elg_net_5', 'elg_net_20',
]
print(f"特征: {len(features)}")

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

# Walk-Forward
all_dates = sorted(df['date_int'].unique())
df_model = df.dropna(subset=features + ['fwd20'])
print(f"有效样本: {len(df_model):,}")

train_window = 504
step = 21
configs = [
    {'name': 'v1.1-30-20d', 'top_n': 30, 'hold': 20},
    {'name': 'v1.1-20-20d', 'top_n': 20, 'hold': 20},
    {'name': 'v1.1-30-10d', 'top_n': 30, 'hold': 10},
]

all_wf = {}
all_pt = {}

for cfg in configs:
    print(f"\n--- {cfg['name']} ---")
    wf_results = []
    predictions = []
    start_idx = train_window
    
    while start_idx + cfg['hold'] <= len(all_dates):
        train_dates = all_dates[start_idx - train_window:start_idx]
        test_dates = all_dates[start_idx:start_idx + cfg['hold']]
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
    all_wf[cfg['name']] = {
        'ic': round(wf_df['ic'].mean(), 4),
        'icir': round(wf_df['ic'].mean() / wf_df['ic'].std(), 3) if wf_df['ic'].std() > 0 else 0,
        'ic_pos': round((wf_df['ic'] > 0).mean() * 100, 1),
    }
    
    # Paper Trading
    all_pred = pd.concat(predictions)
    test_periods = sorted(all_pred['date_int'].unique())
    rebalance_dates = test_periods[::cfg['hold']]
    pt_results = []
    
    for rd in rebalance_dates:
        regime = market_state.loc[rd, 'regime'] if rd in market_state.index else 'unknown'
        if regime == 'bear':
            pt_results.append({'port_ret': 0, 'bench_ret': 0, 'alpha': 0, 'regime': regime, 'action': 'CASH'})
            continue
        signal = all_pred[all_pred['date_int'] == rd]
        signal = signal[signal['close'] > 3]
        if len(signal) < cfg['top_n']: continue
        top = signal.nlargest(cfg['top_n'], 'pred')
        entry_prices = dict(zip(top['sym'], top['close'].astype(float)))
        rd_idx = test_periods.index(rd) if rd in test_periods else -1
        if rd_idx + cfg['hold'] >= len(test_periods): continue
        exit_date = test_periods[rd_idx + cfg['hold']]
        exit_df = all_pred[all_pred['date_int'] == exit_date]
        rets = []
        for sym in top['sym']:
            ep = entry_prices[sym]
            ex = exit_df[exit_df['sym'] == sym]
            rets.append((ex.iloc[0]['close'] - ep) / ep if len(ex) > 0 else 0)
        bm = all_pred[all_pred['date_int'] == rd][['sym','close']].merge(
            exit_df[['sym','close']], on='sym', suffixes=('_s','_e'))
        bench_ret = ((bm['close_e'] - bm['close_s'])/bm['close_s']).mean() if len(bm) > 0 else 0
        port_ret = np.mean(rets)
        if regime in ['cautious', 'weak']:
            port_ret *= 0.5; bench_ret *= 0.5; action = f'{regime}_HALF'
        else: action = 'FULL'
        alpha = port_ret - bench_ret
        wr = len([r for r in rets if r > 0]) / len(rets) * 100
        pt_results.append({'port_ret': port_ret, 'bench_ret': bench_ret, 'alpha': alpha, 'win_rate': wr, 'regime': regime, 'action': action})
    
    pt_df = pd.DataFrame(pt_results)
    active = pt_df[pt_df['action'] != 'CASH']
    cash = pt_df[pt_df['action'] == 'CASH']
    if len(active) > 0:
        ann_factor = 252 / cfg['hold']
        avg_port = active['port_ret'].mean()
        avg_alpha = active['alpha'].mean()
        sharpe = avg_port / active['port_ret'].std() * np.sqrt(ann_factor) if active['port_ret'].std() > 0 else 0
        cum = np.cumprod(1 + active['port_ret'].values)
        peak = np.maximum.accumulate(cum)
        max_dd = ((cum - peak) / peak).min()
        alpha_pos = (active['alpha'] > 0).mean() * 100
        all_pt[cfg['name']] = {
            'avg_return': round(avg_port*100, 2), 'avg_alpha': round(avg_alpha*100, 2),
            'ann_return': round(avg_port*ann_factor*100, 1), 'ann_alpha': round(avg_alpha*ann_factor*100, 1),
            'sharpe': round(sharpe, 2), 'max_dd': round(max_dd*100, 1),
            'win_rate': round(active['win_rate'].mean(), 0), 'alpha_pos_pct': round(alpha_pos, 0),
            'active': len(active), 'cash': len(cash)
        }
        print(f"  IC={all_wf[cfg['name']]['ic']:.4f} ICIR={all_wf[cfg['name']]['icir']:.3f}")
        print(f"  年化{avg_port*ann_factor*100:+.1f}% Sharpe:{sharpe:.2f} DD:{max_dd*100:.1f}% Alpha正:{alpha_pos:.0f}%")

# 保存
model.save_model('models/cn/cn_alpha_v1.1.json')
best_name = max(all_pt, key=lambda x: all_pt[x]['sharpe'])
summary = {
    'version': 'cn-alpha-v1.1', 'date': '2026-06-20',
    'best_config': best_name,
    'all_configs': {n: {**all_wf[n], **all_pt[n]} for n in all_wf},
    'features': features,
    'fundamental_features': ['pe_rank', 'pe_inverse', 'pb_rank', 'pb_inverse', 'div_rank', 'ps_rank'],
}
with open('models/cn/cn_alpha_v1.1_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*70}")
print(f"{'配置':<16} {'IC':>8} {'ICIR':>8} {'年化%':>8} {'Sharpe':>8} {'DD%':>8} {'Alpha正%':>8}")
print("-" * 70)
for n in all_wf:
    w = all_wf[n]; p = all_pt.get(n, {})
    print(f"{n:<16} {w['ic']:>8.4f} {w['icir']:>8.3f} {p.get('ann_return',0):>+8.1f} {p.get('sharpe',0):>8.2f} {p.get('max_dd',0):>8.1f} {p.get('alpha_pos_pct',0):>8.0f}")
print(f"\n🏆 最优: {best_name}")
print(f"总耗时: {time.time()-t0:.0f}秒")
