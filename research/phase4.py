#!/usr/bin/env python3
"""
Phase 4: 分类模型 + 行业轮动
CEO方向：二分类(涨/跌) + 行业动量排名
"""
import pandas as pd, numpy as np, xgboost as xgb, json, time

def log(msg):
    print(msg, flush=True)
    with open('research/phase4_log.txt','a') as f: f.write(msg+'\n')

open('research/phase4_log.txt','w').close()
log('Phase 4: 分类模型 + 行业轮动')

# === 1. 加载数据 + 行业分类 ===
log('\n[1] 加载数据...')
df = pd.read_parquet('data/cn/features_v2.parquet')
df['date_int'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d').astype(int)
df = df[(df['date_int'] >= 20200101) & (df['date_int'] <= 20260601)]

# 行业分类
import tushare as ts
ts.set_token('ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db')
pro = ts.pro_api()
sb = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')
sb['sym'] = sb['ts_code'].str[:6]
df = df.merge(sb[['sym','industry']], on='sym', how='left')
df['industry'] = df['industry'].fillna('未知')

log(f'  {len(df):,}行, {df["sym"].nunique()}只, {df["industry"].nunique()}个行业')

# 采样（每10天）
dates = sorted(df['date_int'].unique())[::10]
df = df[df['date_int'].isin(dates)].copy()
log(f'  采样后: {len(df):,}行, {len(dates)}天')

# === 2. 计算特征 ===
log('\n[2] 计算特征...')
df = df.sort_values(['sym','date_int'])

# 基础特征
df['rev_20d'] = -df['r20']
df['rsi_signal'] = 100 - df['rsi14']

# 资金流
for w in [5]:
    for col in ['sm_net','md_net','lg_net','elg_net','total_net']:
        df[f'{col}_{w}d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(w, min_periods=1).sum())

df['lg_mom'] = df['lg_net_5d'] - df.groupby('sym')['lg_net_5d'].shift(20)/4
df['lg_accel'] = df['lg_net_5d'] - df.groupby('sym')['lg_net_5d'].shift(5)

for col in ['lg_net_5d','total_net_5d']:
    df[f'{col}_rk'] = df.groupby('date_int')[col].rank(pct=True)

# 标签：分类（涨=1, 跌=0）
for hd in [5, 10, 20]:
    df[f'fwd_{hd}d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-hd)/x-1)
    df[f'label_{hd}d'] = (df[f'fwd_{hd}d'] > 0).astype(int)

# === 3. 行业轮动特征 ===
log('\n[3] 行业轮动特征...')

# 行业平均收益和资金流
for col in ['r5','r20','lg_net_5d','total_net_5d']:
    df[f'ind_avg_{col}'] = df.groupby(['date_int','industry'])[col].transform('mean')
    df[f'ind_rank_{col}'] = df.groupby('date_int')[f'ind_avg_{col}'].rank(pct=True)

# 行业内个股排名
df['ind_rank_in_stock'] = df.groupby(['date_int','industry'])['lg_net_5d'].rank(pct=True)

# 行业动量（行业过去20天平均收益）
df['ind_momentum'] = df['ind_avg_r20']

# 行业资金流入强度
df['ind_flow_strength'] = df['ind_avg_lg_net_5d']

# 行业分散度（行业内个股分歧）
df['ind_dispersion'] = df.groupby(['date_int','industry'])['r5'].transform('std')

# === 4. 特征列表 ===
base_features = ['lg_net_5d','elg_net_5d','lg_mom','lg_accel','lg_net_5d_rk',
                 'rev_20d','rsi_signal','macd_hist','log_circ_mv']

industry_features = ['ind_avg_r5','ind_avg_r20','ind_avg_lg_net_5d','ind_avg_total_net_5d',
                     'ind_rank_r5','ind_rank_r20','ind_rank_lg_net_5d','ind_rank_total_net_5d',
                     'ind_rank_in_stock','ind_momentum','ind_flow_strength','ind_dispersion']

all_features = base_features + industry_features

# 填充
for f in all_features:
    if f not in df.columns: df[f] = 0
    df[f] = df[f].fillna(0).replace([np.inf,-np.inf], 0)

log(f'  基础特征: {len(base_features)}, 行业特征: {len(industry_features)}, 总计: {len(all_features)}')

# === 5. 实验矩阵 ===
log('\n[4] 实验矩阵...')

train_d = dates[:int(len(dates)*0.7)]
test_d = dates[int(len(dates)*0.7):]
tr = df[df['date_int'].isin(train_d)]
te = df[df['date_int'].isin(test_d)]
log(f'  Train: {len(train_d)}天, Test: {len(test_d)}天')

params_reg = {'max_depth':5,'eta':0.1,'subsample':0.8,'colsample_bytree':0.8,
              'min_child_weight':50,'objective':'reg:squarederror','tree_method':'hist'}
params_clf = {'max_depth':5,'eta':0.1,'subsample':0.8,'colsample_bytree':0.8,
              'min_child_weight':50,'objective':'binary:logistic','eval_metric':'auc','tree_method':'hist'}

experiments = {
    # 基线
    'A_regress_base': ('reg', base_features, 10, 'fwd_10d'),
    'B_classif_base': ('clf', base_features, 10, 'label_10d'),
    
    # 分类 vs 回归（不同持有期）
    'C_clf_5d': ('clf', base_features, 5, 'label_5d'),
    'D_clf_20d': ('clf', base_features, 20, 'label_20d'),
    'E_reg_20d': ('reg', base_features, 20, 'fwd_20d'),
    
    # +行业特征
    'F_clf_industry': ('clf', all_features, 10, 'label_10d'),
    'G_reg_industry': ('reg', all_features, 10, 'fwd_10d'),
    
    # 只行业特征
    'H_clf_indonly': ('clf', industry_features, 10, 'label_10d'),
    
    # 行业特征+20天
    'I_clf_ind_20d': ('clf', all_features, 20, 'label_20d'),
}

results = []
for name, (mode, feats, hd, target) in experiments.items():
    valid_feats = [f for f in feats if f in df.columns]
    tr2 = tr.dropna(subset=[target])
    te2 = te.dropna(subset=[target])
    if len(tr2)<200 or len(te2)<100: continue
    
    t0 = time.time()
    params = params_clf if mode == 'clf' else params_reg
    m = xgb.train(params, xgb.DMatrix(tr2[valid_feats], label=tr2[target]), num_boost_round=200, verbose_eval=False)
    te2 = te2.copy()
    te2['pred'] = m.predict(xgb.DMatrix(te2[valid_feats]))
    
    # 用fwd收益衡量（不管训练标签是什么）
    fwd_col = f'fwd_{hd}d'
    if fwd_col not in te2.columns:
        fwd_col = 'fwd_10d'
    
    ic = te2.groupby('date_int').apply(lambda x: x['pred'].corr(x[fwd_col])).mean()
    top = te2.groupby('date_int').apply(lambda x: x.nlargest(15,'pred')[fwd_col].mean()).mean()
    bot = te2.groupby('date_int').apply(lambda x: x.nsmallest(15,'pred')[fwd_col].mean()).mean()
    ls = top - bot
    
    # 分类准确率（如果是分类模型）
    if mode == 'clf':
        te2['pred_label'] = (te2['pred'] > 0.5).astype(int)
        if f'label_{hd}d' in te2.columns:
            acc = (te2['pred_label'] == te2[f'label_{hd}d']).mean()
        else:
            acc = 0
    else:
        acc = 0
    
    elapsed = time.time() - t0
    row = {'name':name,'mode':mode,'n_feats':len(valid_feats),'hd':hd,'ic':ic,'ls':ls,'top':top,'acc':acc}
    results.append(row)
    
    acc_str = f' Acc={acc:.3f}' if mode=='clf' else ''
    log(f'  {name:<20} {mode} hd={hd:>2} {len(valid_feats):>2}f IC={ic:.4f} LS={ls:.4f}{acc_str} ({elapsed:.0f}s)')

# === 6. 汇总 ===
log(f'\n{"="*60}')
log('结果汇总:')

# 按IC排序
log('\n按IC排序:')
for r in sorted(results, key=lambda x: x['ic'], reverse=True):
    tag = '*' if r['ic']>0.10 else ('+' if r['ic']>0.05 else '-')
    log(f'  {tag} {r["name"]:<20} IC={r["ic"]:.4f} LS={r["ls"]:.4f} Acc={r.get("acc",0):.3f}')

# 分类 vs 回归对比
log('\n分类 vs 回归:')
for hd in [5,10,20]:
    clf_r = [r for r in results if r['mode']=='clf' and r['hd']==hd and 'base' in r['name'] or r['name']==f'C_clf_{hd}d' or r['name']==f'D_clf_{hd}d']
    reg_r = [r for r in results if r['mode']=='reg' and r['hd']==hd and 'base' in r['name'] or r['name']==f'E_reg_{hd}d']
    if clf_r and reg_r:
        log(f'  hd={hd}: Clf IC={clf_r[0]["ic"]:.4f} vs Reg IC={reg_r[0]["ic"]:.4f}')

# 行业特征效果
log('\n行业特征效果:')
base_clf = [r for r in results if r['name']=='B_classif_base']
ind_clf = [r for r in results if r['name']=='F_clf_industry']
if base_clf and ind_clf:
    log(f'  基础: IC={base_clf[0]["ic"]:.4f}')
    log(f'  +行业: IC={ind_clf[0]["ic"]:.4f}')
    log(f'  提升: {(ind_clf[0]["ic"]-base_clf[0]["ic"])/base_clf[0]["ic"]*100:+.1f}%')

with open('research/phase4_results.json','w') as f:
    json.dump(results, f, indent=2, default=str)
log('\nDone.')
