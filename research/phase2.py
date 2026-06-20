#!/usr/bin/env python3
"""Phase 2: 深度实验 — 资金流细分 × 持有期"""
import pandas as pd, numpy as np, xgboost as xgb, time, json, sys
def log(msg):
    print(msg, flush=True)
    with open('research/phase2_log.txt', 'a') as f:
        f.write(msg + '\n')

# 清空日志
open('research/phase2_log.txt', 'w').close()

log('Phase 2: 深度实验')

# 加载
df = pd.read_parquet('data/cn/features_v2.parquet')
df['date_int'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d').astype(int)
df = df[(df['date_int']>=20210101)&(df['date_int']<=20251231)]
dates = sorted(df['date_int'].unique())[::10]
df = df[df['date_int'].isin(dates)].copy()
log(f'Sample: {len(df):,} rows, {len(dates)} dates')

# 计算特征
df = df.sort_values(['sym','date_int'])
for w in [5,10,20]:
    for col in ['sm_net','md_net','lg_net','elg_net','total_net']:
        df[f'{col}_{w}d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(w, min_periods=1).sum())

df['lg_mom'] = df['lg_net_5d'] - df['lg_net_20d']/4
df['elg_mom'] = df['elg_net_5d'] - df['elg_net_20d']/4
df['total_mom'] = df['total_net_5d'] - df['total_net_20d']/4
df['lg_accel'] = df['lg_net_5d'] - df.groupby('sym')['lg_net_5d'].shift(5)
df['elg_accel'] = df['elg_net_5d'] - df.groupby('sym')['elg_net_5d'].shift(5)
df['lg_streak'] = df.groupby('sym')['lg_net'].transform(lambda x: (x>0).rolling(5).sum())
df['elg_streak'] = df.groupby('sym')['elg_net'].transform(lambda x: (x>0).rolling(5).sum())

for col in ['lg_net_5d','lg_net_20d','elg_net_5d','total_net_5d','total_net_20d']:
    df[f'{col}_rk'] = df.groupby('date_int')[col].rank(pct=True)

df['rev_5d'] = -df['r5']
df['rev_10d'] = -df['r10']
df['rev_20d'] = -df['r20']
df['rsi_signal'] = 100 - df['rsi14']
df['macd_cross'] = (df['macd'] > df['macd_sig']).astype(float)
df['rev_x_lg'] = df['rev_20d'] * df['lg_net_5d_rk']
df['rsi_x_flow'] = df['rsi_signal'] * df['lg_net_5d_rk']

# 填充
num_cols = df.select_dtypes(include=[np.number]).columns
for c in num_cols:
    df[c] = df[c].fillna(0).replace([np.inf,-np.inf],0)

# 标签
for hd in [5,10,20,30]:
    df[f'fwd_{hd}d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-hd)/x-1)

# 分割
train_d = dates[:int(len(dates)*0.7)]
test_d = dates[int(len(dates)*0.7):]
tr = df[df['date_int'].isin(train_d)]
te = df[df['date_int'].isin(test_d)]
log(f'Train: {len(train_d)} dates, Test: {len(test_d)} dates')

params = {'max_depth':5,'eta':0.1,'subsample':0.8,'colsample_bytree':0.8,
          'min_child_weight':50,'objective':'reg:squarederror','tree_method':'hist'}

groups = {
    # 资金流细分
    'flow_lg_5d': ['lg_net_5d'],
    'flow_lg_20d': ['lg_net_20d'],
    'flow_elg_5d': ['elg_net_5d'],
    'flow_elg_20d': ['elg_net_20d'],
    'flow_sm_5d': ['sm_net_5d'],
    'flow_md_5d': ['md_net_5d'],
    'flow_total_5d': ['total_net_5d'],
    'flow_total_20d': ['total_net_20d'],
    'flow_mom_lg': ['lg_mom'],
    'flow_mom_elg': ['elg_mom'],
    'flow_mom_total': ['total_mom'],
    'flow_accel_lg': ['lg_accel'],
    'flow_accel_elg': ['elg_accel'],
    'flow_streak_lg': ['lg_streak'],
    'flow_streak_elg': ['elg_streak'],
    'flow_lg_5d_rk': ['lg_net_5d_rk'],
    'flow_lg_20d_rk': ['lg_net_20d_rk'],
    'flow_elg_5d_rk': ['elg_net_5d_rk'],
    'flow_total_5d_rk': ['total_net_5d_rk'],
    'flow_total_20d_rk': ['total_net_20d_rk'],
    # 技术面
    'tech_reversal': ['rev_5d','rev_10d','rev_20d'],
    'tech_rsi': ['rsi_signal'],
    'tech_macd': ['macd','macd_hist','macd_cross'],
    'tech_vol': ['vol5','vol20','atr_pct'],
    # 交互
    'interact_rev_flow': ['rev_x_lg'],
    'interact_rsi_flow': ['rsi_x_flow'],
    # 组合
    'combo_flow': ['lg_net_5d','elg_net_5d','lg_mom','lg_accel','lg_net_5d_rk'],
    'combo_tech': ['rev_20d','rsi_signal','macd_hist'],
    'combo_all': ['lg_net_5d','elg_net_5d','lg_mom','lg_accel','lg_net_5d_rk',
                   'rev_20d','rsi_signal','macd_hist','log_circ_mv'],
}

log(f'\n{"Group":<25} {"N":>2} {"5d IC":>8} {"10d IC":>8} {"20d IC":>8} {"30d IC":>8}')
log('-'*65)

all_results = []
for gname, flist in groups.items():
    valid = [f for f in flist if f in df.columns]
    if not valid: continue
    row = {'group': gname, 'n': len(valid)}
    for hd in [5,10,20,30]:
        target = f'fwd_{hd}d'
        tr2 = tr.dropna(subset=[target])
        te2 = te.dropna(subset=[target])
        if len(tr2)<200 or len(te2)<100: continue
        try:
            m = xgb.train(params, xgb.DMatrix(tr2[valid], label=tr2[target]), num_boost_round=150, verbose_eval=False)
            te2 = te2.copy()
            te2['pred'] = m.predict(xgb.DMatrix(te2[valid]))
            ic = te2.groupby('date_int').apply(lambda x: x['pred'].corr(x[target])).mean()
            top = te2.groupby('date_int').apply(lambda x: x.nlargest(15,'pred')[target].mean()).mean()
            bot = te2.groupby('date_int').apply(lambda x: x.nsmallest(15,'pred')[target].mean()).mean()
            row[f'ic_{hd}'] = round(ic, 4)
            row[f'ls_{hd}'] = round(top-bot, 4)
        except: pass
    all_results.append(row)
    ic5 = f'{row.get("ic_5",0):.4f}' if 'ic_5' in row else '  N/A'
    ic10 = f'{row.get("ic_10",0):.4f}' if 'ic_10' in row else '  N/A'
    ic20 = f'{row.get("ic_20",0):.4f}' if 'ic_20' in row else '  N/A'
    ic30 = f'{row.get("ic_30",0):.4f}' if 'ic_30' in row else '  N/A'
    log(f'{gname:<25} {len(valid):>2} {ic5:>8} {ic10:>8} {ic20:>8} {ic30:>8}')

# Top findings
log(f'\n{"="*60}')
for hd in [5,10,20,30]:
    key = f'ic_{hd}'
    ranked = sorted([r for r in all_results if key in r], key=lambda x: x[key], reverse=True)
    if ranked:
        log(f'\n持有{hd}天 Top5:')
        for r in ranked[:5]:
            log(f'  {r["group"]:<25} IC={r[key]:.4f} LS={r.get(f"ls_{hd}",0):.4f}')

with open('research/phase2_results.json','w') as f:
    json.dump(all_results, f, indent=2)
log('\nDone.')
