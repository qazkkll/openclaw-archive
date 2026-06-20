#!/usr/bin/env python3
"""
A股Alpha研究框架
CEO方法：小数据集快速实验 → 找到有效方向 → 再扩大验证
选取20个代表性时点覆盖牛熊震荡
"""
import pandas as pd, numpy as np, xgboost as xgb, json, os, time, sys, warnings, itertools
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

def log(msg): print(msg, flush=True)

log("=" * 60)
log("A股Alpha研究框架")
log("=" * 60)

# ============================================================
# 1. 代表性数据集：20个时点覆盖各市场状态
# ============================================================
log("\n[1] 构建代表性数据集...")

df = pd.read_parquet('data/cn/features_v2.parquet')
df['date'] = pd.to_datetime(df['date'])
df['date_int'] = df['date'].dt.strftime('%Y%m%d').astype(int)

# 手动选20个覆盖牛熊震荡的时点
representative_dates = [
    # 2020 牛市
    20200203, 20200323, 20200706, 20201231,
    # 2021 震荡
    20210218, 20210525, 20210901, 20211231,
    # 2022 熊市
    20220128, 20220427, 20220801, 20221031,
    # 2023 震荡偏弱
    20230130, 20230630, 20231020, 20231229,
    # 2024-2025 混合
    20240208, 20240930, 20250106, 20250407,
]

# 找最近的实际交易日
all_dates = sorted(df['date_int'].unique())
actual_dates = []
for target in representative_dates:
    candidates = [d for d in all_dates if abs(d - target) < 15]
    if candidates:
        actual_dates.append(min(candidates, key=lambda x: abs(x - target)))

actual_dates = sorted(set(actual_dates))
log(f"  选定{len(actual_dates)}个时点: {actual_dates[:5]}...{actual_dates[-3:]}")

# 只加载这些时点前后20天的数据（用于滚动特征）
date_range = (min(actual_dates) - 3000, max(actual_dates) + 3000)  # 前后约30天
df_sample = df[(df['date_int'] >= date_range[0]) & (df['date_int'] <= date_range[1])].copy()
log(f"  样本: {len(df_sample):,}行 ({time.time():.0f}s)")

# ============================================================
# 2. 特征组定义
# ============================================================
log("\n[2] 定义特征组...")

feature_groups = {
    # 基础技术面
    'momentum': ['r5', 'r10', 'r20'],
    'reversal': ['rev_5d', 'rev_10d', 'rev_20d'],  # 需要计算
    'volatility': ['vol5', 'vol20', 'atr_pct'],
    'macd': ['macd', 'macd_hist', 'macd_sig'],
    'rsi': ['rsi14'],
    
    # 资金流（Andy觉得可能准）
    'money_flow_raw': ['sm_net', 'md_net', 'lg_net', 'elg_net', 'total_net'],
    'money_flow_5d': ['sm_net_5', 'md_net_5', 'lg_net_5', 'elg_net_5', 'total_net_5'],
    'money_flow_20d': ['sm_net_20', 'md_net_20', 'lg_net_20', 'elg_net_20', 'total_net_20'],
    'money_flow_ratio': [],  # 需要计算
    
    # 市值/换手
    'size': ['log_circ_mv', 'circ_mv'],
    'turnover': ['vol_r', 'turnover_rate', 'turnover_20'],
    
    # 交互（可能有效）
    'flow_x_reversal': [],  # 需要计算
    'flow_x_size': [],  # 需要计算
}

# 计算派生特征
df_sample['rev_5d'] = -df_sample['r5']
df_sample['rev_10d'] = -df_sample['r10']
df_sample['rev_20d'] = -df_sample['r20']

# 资金流比率
if 'lg_net' in df_sample.columns and 'total_net' in df_sample.columns:
    df_sample['lg_flow_ratio'] = df_sample['lg_net'] / df_sample['total_net'].clip(lower=1)
    df_sample['elg_flow_ratio'] = df_sample['elg_net'] / df_sample['total_net'].clip(lower=1)
    feature_groups['money_flow_ratio'] = ['lg_flow_ratio', 'elg_flow_ratio']

# 资金流动量（5d vs 20d）
df_sample['lg_flow_mom'] = df_sample['lg_net_5'] - df_sample['lg_net_20'] / 4
df_sample['total_flow_mom'] = df_sample['total_net_5'] - df_sample['total_net_20'] / 4
feature_groups['money_flow_momentum'] = ['lg_flow_mom', 'total_flow_mom']

# 交互特征
df_sample['rev_x_lg_flow'] = df_sample['rev_20d'] * df_sample['lg_net_20']
df_sample['rev_x_total_flow'] = df_sample['rev_20d'] * df_sample['total_net_20']
df_sample['small_x_flow'] = (-df_sample['log_circ_mv']) * df_sample['lg_net_20']
feature_groups['flow_x_reversal'] = ['rev_x_lg_flow', 'rev_x_total_flow']
feature_groups['flow_x_size'] = ['small_x_flow']

# 资金流方向信号
df_sample['lg_net_positive'] = (df_sample['lg_net'] > 0).astype(float)
df_sample['lg_net_5_positive'] = (df_sample['lg_net_5'] > 0).astype(float)
df_sample['lg_accel'] = (df_sample['lg_net_5'] > df_sample['lg_net_20'] / 4).astype(float)
feature_groups['flow_direction'] = ['lg_net_positive', 'lg_net_5_positive', 'lg_accel']

# 资金流集中度（大单占比）
df_sample['lg_concentration'] = df_sample['lg_net'].abs() / df_sample['total_net'].abs().clip(lower=1)
df_sample['elg_concentration'] = df_sample['elg_net'].abs() / df_sample['total_net'].abs().clip(lower=1)
feature_groups['flow_concentration'] = ['lg_concentration', 'elg_concentration']

# 截面排名
for col in ['lg_net_20', 'total_net_20', 'vol_r', 'rev_20d']:
    df_sample[f'{col}_rank'] = df_sample.groupby('date_int')[col].rank(pct=True)
feature_groups['cross_section_rank'] = ['lg_net_20_rank', 'total_net_20_rank', 'vol_r_rank', 'rev_20d_rank']

# 填充
all_feats = set()
for v in feature_groups.values():
    all_feats.update(v)
for f in all_feats:
    if f not in df_sample.columns:
        df_sample[f] = 0
    df_sample[f] = df_sample[f].fillna(0).replace([np.inf, -np.inf], 0)

log(f"  特征组: {len(feature_groups)}组, 总特征: {len(all_feats)}个")

# ============================================================
# 3. 计算标签（多个持有期）
# ============================================================
log("\n[3] 计算标签...")
df_sample = df_sample.sort_values(['sym', 'date_int'])

for hd in [5, 10, 20]:
    df_sample[f'fwd_{hd}d'] = df_sample.groupby('sym')['close'].transform(
        lambda x: x.shift(-hd) / x - 1)

# ============================================================
# 4. 实验矩阵
# ============================================================
log("\n[4] 实验矩阵...")

params = {
    'max_depth': 5, 'eta': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 50, 'objective': 'reg:squarederror', 'tree_method': 'hist',
}

# 分割训练/测试
train_dates = [d for d in actual_dates if d <= 20231231]
test_dates = [d for d in actual_dates if d > 20231231]
log(f"  训练: {len(train_dates)}时点, 测试: {len(test_dates)}时点")

results = []

for group_name, feat_list in feature_groups.items():
    valid_feats = [f for f in feat_list if f in df_sample.columns]
    if len(valid_feats) == 0:
        continue
    
    for hold_days in [5, 10, 20]:
        target = f'fwd_{hold_days}d'
        
        # 训练集
        tr = df_sample[df_sample['date_int'].isin(train_dates)].dropna(subset=[target])
        te = df_sample[df_sample['date_int'].isin(test_dates)].dropna(subset=[target])
        
        if len(tr) < 500 or len(te) < 200:
            continue
        
        try:
            dtrain = xgb.DMatrix(tr[valid_feats].fillna(0), label=tr[target])
            dtest = xgb.DMatrix(te[valid_feats].fillna(0))
            m = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
            te = te.copy()
            te['pred'] = m.predict(dtest)
            
            ic = te.groupby('date_int').apply(lambda x: x['pred'].corr(x[target])).mean()
            ric = te.groupby('date_int').apply(lambda x: x['pred'].corr(x[target], method='spearman')).mean()
            top_ret = te.groupby('date_int').apply(lambda x: x.nlargest(15, 'pred')[target].mean()).mean()
            bot_ret = te.groupby('date_int').apply(lambda x: x.nsmallest(15, 'pred')[target].mean()).mean()
            ls = top_ret - bot_ret
            
            results.append({
                'group': group_name,
                'n_feats': len(valid_feats),
                'hold_days': hold_days,
                'ic': ic,
                'ric': ric,
                'ls': ls,
                'top': top_ret,
            })
            
            status = "OK" if ic > 0.05 else ("WEAK" if ic > 0.02 else "NOISE")
            log(f"  {group_name:<25} hd={hold_days:>2} IC={ic:>7.4f} LS={ls:>7.4f} Top={top_ret:>7.4f} [{status}]")
        except Exception as e:
            log(f"  {group_name:<25} hd={hold_days:>2} ERROR: {e}")

# ============================================================
# 5. 汇总分析
# ============================================================
log("\n" + "=" * 60)
log("[5] 汇总分析")
log("=" * 60)

rdf = pd.DataFrame(results)
if len(rdf) > 0:
    log("\n按IC排序:")
    rdf_sorted = rdf.sort_values('ic', ascending=False)
    for _, r in rdf_sorted.head(15).iterrows():
        tag = "⭐" if r['ic'] > 0.10 else ("✅" if r['ic'] > 0.05 else "⚠️" if r['ic'] > 0.02 else "❌")
        log(f"  {tag} {r['group']:<25} hd={int(r['hold_days']):>2} IC={r['ic']:.4f} LS={r['ls']:.4f} feats={int(r['n_feats'])}")
    
    log("\n按持有期分组:")
    for hd in [5, 10, 20]:
        sub = rdf[rdf['hold_days'] == hd].sort_values('ic', ascending=False)
        if len(sub) > 0:
            log(f"\n  持有{hd}天 Top5:")
            for _, r in sub.head(5).iterrows():
                log(f"    {r['group']:<25} IC={r['ic']:.4f} LS={r['ls']:.4f}")
    
    log("\n最强特征组:")
    best = rdf.loc[rdf['ic'].idxmax()]
    log(f"  {best['group']} (持有{int(best['hold_days'])}天): IC={best['ic']:.4f}, LS={best['ls']:.4f}")

# 保存
with open('research/experiment_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
log(f"\n✅ 结果已保存 research/experiment_results.json")
