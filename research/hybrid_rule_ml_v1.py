#!/usr/bin/env python3
"""
混合方案: 规则型评分 + ML分类过滤
CEO方向: 用XGBoost分类器过滤规则型策略的"假阳性"
- 规则型选出候选 → ML判断"会涨还是会跌" → 只买ML判涨的
- 如果ML不能改善 → 放弃混合方案
"""
import sys, os, time, json, datetime
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import xgboost as xgb

WORKSPACE = os.path.expanduser('~/.hermes/openclaw-archive')
DATA_DIR = os.path.join(WORKSPACE, 'data')

print("="*60)
print("混合方案: 规则型 + ML过滤")
print("="*60)

# === 1. 加载+特征(同v5) ===
print("\n[1] 加载+特征...")
t0 = time.time()

df = pd.read_parquet(os.path.join(DATA_DIR, 'a_hist_10y.parquet'))
df = df.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df['date'] = df['date'].astype(int)

mf = pd.read_parquet(os.path.join(DATA_DIR, 'cn/moneyflow_core.parquet'))
mf['sym'] = mf['ts_code'].str[:6]
mf['date'] = mf['trade_date'].astype(int)
for col in ['sm', 'md', 'lg', 'elg']:
    mf[f'{col}_net'] = mf[f'buy_{col}_amount'] - mf[f'sell_{col}_amount']
mf['total_net'] = mf['net_mf_amount']
df = df.merge(mf[['sym','date','total_net','lg_net','md_net','elg_net']], on=['sym','date'], how='left')

df = df[~df['sym'].str.startswith('688')].copy()
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
df = df[df['volume'] > 0].copy()
df = df.sort_values(['sym', 'date']).reset_index(drop=True)

# 特征
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ret10'] = df.groupby('sym')['close'].pct_change(10)
df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ret60'] = df.groupby('sym')['close'].pct_change(60)
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma60'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(60, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['ma60_bias'] = (df['close'] - df['ma60']) / df['ma60']
df['vol20'] = df.groupby('sym')['ret5'].transform(lambda x: x.rolling(4, min_periods=2).std())

delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)

for col in ['total_net', 'lg_net', 'md_net', 'elg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())

df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_bias'] = df.groupby('date')['ma60_bias'].transform('mean')
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')

for hd in [5, 10, 20]:
    df[f'fwd_{hd}d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-hd) / x - 1)

# ML特征
ml_features = ['ret5', 'ret10', 'ret20', 'ret60', 'ma20_bias', 'ma60_bias', 'vol20',
               'rsi_14', 'total_net_5d', 'lg_net_5d', 'md_net_5d', 'elg_net_5d']
df['label_10d'] = (df['fwd_10d'] > 0.03).astype(int)  # 涨>3%为正类

# 清理
for f in ml_features:
    df[f] = df[f].fillna(0).replace([np.inf, -np.inf], 0)

print(f"  {len(df):,}行, {df['sym'].nunique()}只, {time.time()-t0:.0f}秒")

# === 2. 评分函数 ===
def score_optimized(day):
    s = day.copy()
    s['score'] = 0.0
    s['score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 3
    s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 2
    s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 2
    s['score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1.5
    s['score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 1
    s['score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1
    return s

# === 3. Walk-Forward混合回测 ===
print("\n[2] Walk-Forward混合回测...")
print("="*60)

# 每2年训练一次ML, 在测试期用规则型+ML过滤
folds = [
    ('2020', 20160101, 20191231, 20200101, 20201231),
    ('2021', 20170101, 20201231, 20210101, 20211231),
    ('2022', 20180101, 20211231, 20220101, 20221231),
    ('2023', 20190101, 20221231, 20230101, 20231231),
    ('2024', 20200101, 20231231, 20240101, 20241231),
    ('2025', 20210101, 20241231, 20250101, 20251231),
    ('2026', 20220101, 20251231, 20260101, 20260616),
]

hold_days = 10
top_n = 15
stop_loss = -0.03
cost = 0.003

# 三种策略对比
strategies = {
    'pure_rule': '纯规则型(基线)',
    'rule_ml_filter': '规则型+ML过滤',
    'rule_ml_rank': '规则型+ML重排',
}

all_results = {k: [] for k in strategies}
all_trades = {k: [] for k in strategies}

for fold_name, train_start, train_end, test_start, test_end in folds:
    # 训练ML
    train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
    train_df = train_df.dropna(subset=ml_features + ['label_10d'])
    
    if len(train_df) < 10000:
        print(f"  {fold_name}: 训练数据不足, 跳过")
        continue
    
    X_train = train_df[ml_features]
    y_train = train_df['label_10d']
    
    ml_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    )
    ml_model.fit(X_train, y_train, verbose=False)
    
    # 测试期
    test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)]
    test_dates = sorted(test_df['date'].unique())
    
    # ML预测
    test_df = test_df.copy()
    test_df['ml_prob'] = ml_model.predict_proba(test_df[ml_features])[:, 1]
    
    rebal_dates = test_dates[::hold_days]
    
    for strategy_name in strategies:
        trades_fold = []
        equity = 100000.0
        equity_curve = []
        
        for rd in rebal_dates:
            day = test_df[test_df['date'] == rd].copy()
            if len(day) < top_n * 2:
                continue
            
            # 规则型评分
            day = score_optimized(day)
            
            if strategy_name == 'pure_rule':
                # 纯规则型: 取Top N
                picks = day.nlargest(top_n, 'score')
            elif strategy_name == 'rule_ml_filter':
                # 规则型Top 30, 然后ML过滤取Top N
                candidates = day.nlargest(top_n * 2, 'score')
                picks = candidates.nlargest(top_n, 'ml_prob')
            elif strategy_name == 'rule_ml_rank':
                # 规则型Top 30, 用ML概率重排
                candidates = day.nlargest(top_n * 2, 'score')
                candidates['combined'] = candidates['score'].rank(pct=True) * 0.5 + candidates['ml_prob'].rank(pct=True) * 0.5
                picks = candidates.nlargest(top_n, 'combined')
            
            fwd_col = f'fwd_{hold_days}d'
            rets = picks[fwd_col].fillna(0).values
            rets = np.where(rets < stop_loss, stop_loss, rets)
            rets = rets - cost
            
            for i, (_, row) in enumerate(picks.iterrows()):
                trades_fold.append({'sym': row['sym'], 'date': rd, 'net_ret': rets[i], 'year': int(str(rd)[:4])})
            
            avg_ret = rets.mean()
            equity *= (1 + avg_ret)
            equity_curve.append((rd, equity))
        
        all_results[strategy_name].append({
            'fold': fold_name,
            'trades': len(trades_fold),
            'win_rate': round((np.array([t['net_ret'] for t in trades_fold]) > 0).mean(), 4) if trades_fold else 0,
            'avg_return': round(np.array([t['net_ret'] for t in trades_fold]).mean(), 4) if trades_fold else 0,
        })
        all_trades[strategy_name].extend(trades_fold)
        
        rets_arr = np.array([t['net_ret'] for t in trades_fold])
        wr = (rets_arr > 0).mean() if len(rets_arr) > 0 else 0
        avg = rets_arr.mean() if len(rets_arr) > 0 else 0
        print(f"  {fold_name} [{strategy_name}]: {len(trades_fold)}笔 胜率{wr:.1%} 均收{avg:.2%}")

# === 4. 汇总 ===
print("\n" + "="*60)
print("[3] 策略对比汇总")
print("="*60)

for strategy_name, label in strategies.items():
    trades = all_trades[strategy_name]
    if not trades:
        continue
    
    rets = np.array([t['net_ret'] for t in trades])
    eq = [100000.0]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    eq = np.array(eq)
    
    win_rate = (rets > 0).mean()
    avg_ret = rets.mean()
    years = len(folds) * 1  # 大约每年一个fold
    tpy = 252 / hold_days
    ann_ret = avg_ret * tpy
    ann_std = rets.std() * np.sqrt(tpy)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0
    
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = dd.max()
    
    avg_win = rets[rets > 0].mean() if (rets > 0).any() else 0
    avg_loss = rets[rets < 0].mean() if (rets < 0).any() else 0
    pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    cagr = (eq[-1] / eq[0]) ** (1/years) - 1
    
    print(f"\n  {label}:")
    print(f"    交易: {len(trades)}")
    print(f"    胜率: {win_rate:.1%}")
    print(f"    均收: {avg_ret:.2%}")
    print(f"    年化: {cagr:.1%}")
    print(f"    Sharpe: {sharpe:.2f}")
    print(f"    DD: {max_dd:.1%}")
    print(f"    盈亏比: {pl_ratio:.2f}")

# === 5. CEO判断 ===
print("\n" + "="*60)
print("[4] CEO判断")
print("="*60)

# 比较Sharpe
sharpe_compare = {}
for sn in strategies:
    trades = all_trades[sn]
    if not trades:
        continue
    rets = np.array([t['net_ret'] for t in trades])
    tpy = 252 / hold_days
    ann_ret = rets.mean() * tpy
    ann_std = rets.std() * np.sqrt(tpy)
    sharpe_compare[sn] = ann_ret / ann_std if ann_std > 0 else 0

best_sn = max(sharpe_compare, key=sharpe_compare.get)
print(f"\n最佳策略: {strategies[best_sn]} (Sharpe {sharpe_compare[best_sn]:.2f})")

improvement = sharpe_compare.get('rule_ml_filter', 0) - sharpe_compare.get('pure_rule', 0)
print(f"ML过滤改善: {improvement:+.2f} Sharpe")
if improvement < 0.05:
    print("⚠️ ML过滤改善不足0.05, 不值得增加复杂度")
else:
    print("✅ ML过滤有效, 值得继续优化")

improvement2 = sharpe_compare.get('rule_ml_rank', 0) - sharpe_compare.get('pure_rule', 0)
print(f"ML重排改善: {improvement2:+.2f} Sharpe")
