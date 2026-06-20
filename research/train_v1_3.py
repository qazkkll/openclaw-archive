#!/usr/bin/env python3
"""
cn-alpha-v1.3 优化训练
CEO方案：灌真实基本面 + 截面排名标签
目标：Sharpe从0.55提升到0.8+
"""
import pandas as pd, numpy as np, xgboost as xgb, json, os, time, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

HOLD_DAYS = 10
TOP_K = 15
VERSION = "cn-alpha-v1.3"

print("=" * 70)
print(f"{VERSION} 优化训练（基本面+截面排名标签）")
print("=" * 70)

# ============================================================
# 1. 加载数据 + merge基本面
# ============================================================
print("\n[1/6] 加载数据 + merge基本面...")
t0 = time.time()

df = pd.read_parquet('data/cn/features_v2.parquet')
df['date'] = pd.to_datetime(df['date'])
df['date_int'] = df['date'].dt.strftime('%Y%m%d').astype(int)
df['sym'] = df['ts_code'].str[:6] if 'ts_code' in df.columns else df['sym']

# merge daily_basic
db = pd.read_parquet('data/cn/daily_basic.parquet')
db['sym'] = db['ts_code'].str[:6]
db['date_int'] = db['trade_date'].astype(int)
db = db[['sym', 'date_int', 'pe_ttm', 'pb', 'ps_ttm', 'dv_ratio', 'total_mv', 'circ_mv', 'turnover_rate']].copy()

df = df.merge(db, on=['sym', 'date_int'], how='left', suffixes=('', '_db'))

# 用daily_basic的circ_mv和turnover_rate覆盖（更准确）
for col in ['circ_mv_db', 'turnover_rate_db']:
    base = col.replace('_db', '')
    if col in df.columns:
        df[base] = df[col].fillna(df[base])
        df.drop(col, axis=1, inplace=True)

print(f"  数据: {len(df):,}行, {df['sym'].nunique()}只")
print(f"  基本面覆盖率: PE={df['pe_ttm'].notna().mean()*100:.0f}%, PB={df['pb'].notna().mean()*100:.0f}%, PS={df['ps_ttm'].notna().mean()*100:.0f}%, 股息={df['dv_ratio'].notna().mean()*100:.0f}%")
print(f"  耗时: {time.time()-t0:.0f}s")

# ============================================================
# 2. 计算36个特征（真实基本面）
# ============================================================
print("\n[2/6] 计算特征（真实基本面）...")

# 反转
df['rev_5d'] = -df['r5']
df['rev_10d'] = -df['r10']
df['rev_20d'] = -df['r20']
df['rsi_reversal'] = -(df['rsi14'] - 50)
df['macd_reversal'] = -df['macd']
df['low_vol_5d'] = -df['vol5']
df['low_vol_20d'] = -df['vol20']
df['low_atr'] = -df['atr_pct']
df['small_cap'] = -np.log(df['circ_mv'].clip(lower=1))

# 残差动量
for col, src in [('residual_mom_5d', 'r5'), ('residual_mom_20d', 'r20')]:
    grp = df.groupby('date_int')[src]
    df[col] = df[src] - grp.transform('mean')

# 资金流动量
df['lg_flow_momentum'] = df['lg_net_5'] - df['lg_net_20'] / 4
df['total_flow_momentum'] = df['total_net_5'] - df['total_net_20'] / 4

# 截面排名（资金流）
for col in ['lg_net_20', 'md_net_20', 'total_net_20']:
    df[f'{col}_rank'] = df.groupby('date_int')[col].rank(pct=True)

df['rev_flow_interaction'] = df['rev_20d'] * df['lg_net_20_rank']
df['turnover_rank'] = df.groupby('date_int')['vol_r'].rank(pct=True)

# === 真实基本面特征（之前全填0，现在用真实数据） ===
pe = df['pe_ttm'].where((df['pe_ttm'] > 0) & (df['pe_ttm'] < 500))
df['pe_rank'] = df.groupby('date_int')['pe_ttm'].rank(pct=True, ascending=True)
df['pe_inverse'] = 1.0 / pe.clip(lower=1)

pb = df['pb'].where((df['pb'] > 0) & (df['pb'] < 100))
df['pb_rank'] = df.groupby('date_int')['pb'].rank(pct=True, ascending=True)
df['pb_inverse'] = 1.0 / pb.clip(lower=0.1)

df['div_rank'] = df.groupby('date_int')['dv_ratio'].rank(pct=True, ascending=False)

ps = df['ps_ttm'].where((df['ps_ttm'] > 0) & (df['ps_ttm'] < 200))
df['ps_rank'] = df.groupby('date_int')['ps_ttm'].rank(pct=True, ascending=True)

# 特征列表（与v1.1相同的36个）
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

for f in features:
    if f not in df.columns:
        df[f] = 0
    df[f] = df[f].fillna(0).replace([np.inf, -np.inf], 0)

# 市场状态特征
mkt = df.groupby('date_int').agg(
    avg_close=('close', 'mean'),
    avg_r20=('r20', 'mean'),
).reset_index()
for w in [20, 60, 120]:
    mkt[f'ma{w}'] = mkt['avg_close'].rolling(w).mean()
mkt['trend'] = (mkt['ma60'] > mkt['ma120']).astype(int)
mkt['momentum'] = (mkt['avg_r20'] > 0).astype(int)

adv_dec = df.groupby('date_int').apply(lambda x: (x['r5'] > 0).sum() / max(len(x), 1)).reset_index()
adv_dec.columns = ['date_int', 'adv_ratio']
mkt = mkt.merge(adv_dec, on='date_int', how='left')
mkt['breadth'] = (mkt['adv_ratio'] > 0.5).astype(int)
mkt['score'] = mkt['trend'] + mkt['momentum'] + mkt['breadth']

df = df.merge(mkt[['date_int', 'score']], on='date_int', how='left')

print(f"  特征: {len(features)}个")
print(f"  基本面特征非零: PE={df['pe_rank'].ne(0.5).mean()*100:.0f}%, PB={df['pb_rank'].ne(0.5).mean()*100:.0f}%")
print(f"  耗时: {time.time()-t0:.0f}s")

# ============================================================
# 3. 标签：截面排名（替代原始收益）
# ============================================================
print("\n[3/6] 计算截面排名标签...")

df = df.sort_values(['sym', 'date_int'])
df['fwd_ret'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-HOLD_DAYS) / x - 1)
df['fwd_rank'] = df.groupby('date_int')['fwd_ret'].rank(pct=True)

df_valid = df.dropna(subset=['fwd_ret', 'fwd_rank'])
print(f"  有效样本: {len(df_valid):,}")

# ============================================================
# 4. Walk-Forward验证（对比原始标签 vs 排名标签）
# ============================================================
print("\n[4/6] Walk-Forward验证...")

folds = [
    (20160101, 20201231, 20210101, 20220630),
    (20170101, 20210630, 20210701, 20221231),
    (20180101, 20220630, 20220701, 20231231),
    (20190101, 20230630, 20230701, 20241231),
    (20200101, 20240630, 20240701, 20260630),
]

params = {
    'max_depth': 6, 'eta': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 100, 'objective': 'reg:squarederror',
    'eval_metric': 'rmse', 'tree_method': 'hist',
}

results = {}
for label, target_col in [('V1.3-原始标签', 'fwd_ret'), ('V1.3-排名标签', 'fwd_rank')]:
    print(f"\n  === {label} ===")
    wf = []
    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        t1 = time.time()
        tr = df_valid[(df_valid['date_int'] >= tr_s) & (df_valid['date_int'] <= tr_e)]
        te = df_valid[(df_valid['date_int'] >= te_s) & (df_valid['date_int'] <= te_e)]
        if len(tr) < 10000 or len(te) < 1000:
            continue
        
        X_tr = tr[features].fillna(0)
        X_te = te[features].fillna(0)
        dtrain = xgb.DMatrix(X_tr, label=tr[target_col])
        dtest = xgb.DMatrix(X_te, label=te[target_col])
        
        model = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
        te = te.copy()
        te['pred'] = model.predict(dtest)
        
        # IC用原始收益衡量（不管训练标签是什么）
        ic = te.groupby('date_int').apply(lambda x: x['pred'].corr(x['fwd_ret'])).mean()
        ric = te.groupby('date_int').apply(lambda x: x['pred'].corr(x['fwd_ret'], method='spearman')).mean()
        
        top_ret = te.groupby('date_int').apply(lambda x: x.nlargest(TOP_K, 'pred')['fwd_ret'].mean()).mean()
        bot_ret = te.groupby('date_int').apply(lambda x: x.nsmallest(TOP_K, 'pred')['fwd_ret'].mean()).mean()
        ls = top_ret - bot_ret
        
        wf.append({'fold': fi+1, 'ic': ic, 'ric': ric, 'ls': ls, 'top': top_ret})
        print(f"    Fold {fi+1}: IC={ic:.4f}, RankIC={ric:.4f}, LS={ls:.4f} ({time.time()-t1:.0f}s)")
    
    avg_ic = np.mean([r['ic'] for r in wf])
    avg_ric = np.mean([r['ric'] for r in wf])
    avg_ls = np.mean([r['ls'] for r in wf])
    results[label] = {'ic': avg_ic, 'ric': avg_ric, 'ls': avg_ls, 'wf': wf}
    print(f"    汇总: IC={avg_ic:.4f}, RankIC={avg_ric:.4f}, LS={avg_ls:.4f}")

# ============================================================
# 5. Paper Trade（排名标签模型 + 市场过滤器）
# ============================================================
print("\n[5/6] Paper Trade（排名标签 + 市场过滤器）...")

# 用排名标签训练最终模型
train_final = df_valid[df_valid['date_int'] <= 20240630]
X_final = train_final[features].fillna(0)
dtrain_final = xgb.DMatrix(X_final, label=train_final['fwd_rank'])
final_model = xgb.train(params, dtrain_final, num_boost_round=500, verbose_eval=False)

quarter_starts = []
for year in range(2021, 2027):
    for month in [1, 4, 7, 10]:
        qdate = int(f"{year}{month:02d}01")
        candidates = [d for d in sorted(df_valid['date_int'].unique()) if abs(d - qdate) < 2000]
        if candidates:
            quarter_starts.append(min(candidates, key=lambda x: abs(x - qdate)))
quarter_starts = sorted(set(quarter_starts))

paper = []
for signal_date in quarter_starts:
    day = df_valid[df_valid['date_int'] == signal_date].copy()
    if len(day) < 50:
        continue
    day = day[day['close'] > 3]
    
    # 市场过滤
    mkt_score = day['score'].iloc[0] if 'score' in day.columns else 3
    if mkt_score <= 1:
        regime = 'bear'
        position = 0
    elif mkt_score <= 2:
        regime = 'cautious'
        position = 0.5
    else:
        regime = 'bull'
        position = 1.0
    
    if position == 0:
        paper.append({'date': signal_date, 'port': 0, 'bench': 0, 'alpha': 0, 'wr': 0, 'pos': 0, 'regime': regime})
        continue
    
    X_day = day[features].fillna(0)
    day['pred'] = final_model.predict(xgb.DMatrix(X_day))
    
    top = day.nlargest(TOP_K, 'pred')
    rets = top['fwd_ret'].fillna(0).tolist()
    port_ret = np.mean(rets) * position
    bench_ret = day['fwd_ret'].dropna().mean()
    winners = sum(1 for r in rets if r > 0)
    
    paper.append({'date': signal_date, 'port': port_ret, 'bench': bench_ret,
                  'alpha': port_ret - bench_ret, 'wr': winners/len(rets)*100,
                  'pos': position, 'regime': regime})

# 汇总
rdf = pd.DataFrame(paper)
active = rdf[rdf['pos'] > 0]

alpha_pos = (active['alpha'] > 0).sum()
alpha_pct = alpha_pos / len(active) * 100
cum = (1 + active['port']).prod() - 1
n_yrs = len(active) * HOLD_DAYS / 365
ann = (1 + cum) ** (1 / max(n_yrs, 0.5)) - 1
sharpe = active['port'].mean() / active['port'].std() * np.sqrt(365/HOLD_DAYS) if active['port'].std() > 0 else 0
dd = (1 + active['port']).cumprod()
max_dd = ((dd - dd.expanding().max()) / dd.expanding().max()).min()
downside = active['port'][active['port'] < 0]
sortino = active['port'].mean() / downside.std() * np.sqrt(365/HOLD_DAYS) if len(downside) > 0 and downside.std() > 0 else 0

# ============================================================
# 6. 输出报告
# ============================================================
print("\n[6/6] 结果\n")
print("=" * 70)
print(f"📊 {VERSION} 验证报告")
print("=" * 70)

print(f"\n📈 Walk-Forward对比:")
print(f"  {'模型':<20} {'IC':>8} {'RankIC':>8} {'L/S':>8}")
print(f"  {'-'*44}")
for label, r in results.items():
    print(f"  {label:<20} {r['ic']:>8.4f} {r['ric']:>8.4f} {r['ls']:>8.4f}")

print(f"\n📈 Paper Trade（排名标签 + 市场过滤）:")
print(f"  活跃期: {len(active)}/{len(rdf)}")
print(f"  Alpha正: {alpha_pos}/{len(active)} = {alpha_pct:.1f}%")
print(f"  年化: {ann*100:+.1f}%")
print(f"  Sharpe: {sharpe:.2f}")
print(f"  Sortino: {sortino:.2f}")
print(f"  MaxDD: {max_dd*100:.1f}%")

print(f"\n  分年:")
active_c = active.copy()
active_c['year'] = active_c['date'] // 10000
for year, grp in active_c.groupby('year'):
    print(f"    {year}: 收益={grp['port'].mean()*100:+.2f}%, Alpha={grp['alpha'].mean()*100:+.2f}%, WR={grp['wr'].mean():.0f}%")

print(f"\n  逐期:")
for _, r in rdf.iterrows():
    if r['pos'] == 0:
        print(f"    {int(r['date'])} {r['regime']:>8} CASH")
    else:
        print(f"    {int(r['date'])} {r['regime']:>8} {r['port']*100:>+6.2f}% Alpha={r['alpha']*100:>+6.2f}% WR={r['wr']:.0f}%")

# vs 历史版本
print(f"\n📊 版本对比:")
print(f"  {'版本':<15} {'年化':>8} {'Sharpe':>8} {'MaxDD':>8} {'Alpha正':>8}")
print(f"  {'-'*47}")
print(f"  {'V1.0':<15} {'+13%':>8} {'0.72':>8} {'-26.9%':>8} {'67%':>8}")
print(f"  {'V1.1(无过滤)':<15} {'-4.2%':>8} {'0.02':>8} {'-26.7%':>8} {'-':>8}")
print(f"  {'V1.1(+过滤)':<15} {'+12.7%':>8} {'0.55':>8} {'-14.7%':>8} {'66.7%':>8}")
print(f"  {'V1.3(基本面+排名)':<15} {f'{ann*100:+.1f}%':>8} {f'{sharpe:.2f}':>8} {f'{max_dd*100:.1f}%':>8} {f'{alpha_pct:.0f}%':>8}")

# 特征重要性
imp = final_model.get_score(importance_type='gain')
total = sum(imp.values())
print(f"\n  特征重要性Top10:")
for feat, gain in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]:
    is_fund = feat in ['pe_rank','pe_inverse','pb_rank','pb_inverse','div_rank','ps_rank']
    tag = "🆕基本面" if is_fund else ""
    print(f"    {feat:<25} {gain/total*100:>5.1f}% {tag}")

fund_gain = sum(imp.get(f, 0) for f in ['pe_rank','pe_inverse','pb_rank','pb_inverse','div_rank','ps_rank'])
print(f"\n  基本面特征总贡献: {fund_gain/total*100:.1f}% (V1.1时=0%)")

# 保存
final_model.save_model('models/cn/cn_alpha_v1.3.json')
summary = {
    'version': VERSION, 'date': time.strftime('%Y-%m-%d'),
    'features': len(features), 'feature_list': features,
    'hold_days': HOLD_DAYS, 'top_k': TOP_K,
    'label': 'cross_sectional_rank',
    'paper_trade': {
        'active_periods': len(active), 'total_periods': len(rdf),
        'alpha_positive_pct': round(alpha_pct, 1),
        'ann_return': round(ann*100, 2),
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'max_dd': round(max_dd*100, 2),
    },
    'wf': results,
}
with open('models/cn/cn_alpha_v1.3_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

print(f"\n✅ 模型: models/cn/cn_alpha_v1.3.json")
print(f"✅ 报告: models/cn/cn_alpha_v1.3_summary.json")
