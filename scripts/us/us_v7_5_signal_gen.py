#!/usr/bin/env python3
"""
us_v7_5_signal_gen.py — V7.5 调仓信号生成器
根据V7.5评分和T5_H10_S15_R10策略，生成具体买入/卖出/持有信号。
比较持仓和候选列表，输出明确的调仓指令。
"""
import sys, os, json, pickle, time
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, xgboost as xgb

T0=time.time()
BASE='/home/hermes/.hermes/openclaw-archive'; ML=f'{BASE}/ml'; MD=f'{BASE}/data/models'; VER='us_v7_5'
print('='*60); print(f'V7.5 调仓信号 {time.strftime("%Y-%m-%d %H:%M")}'); print('='*60)

# 加载模型
model=xgb.Booster(); model.load_model(f'{MD}/{VER}.json')
cal=pickle.load(open(f'{MD}/{VER}_calibrator.pkl','rb'))
report=json.load(open(f'{MD}/{VER}_report.json'))
FEATS=report['features']

# 特征
df=pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str']=df['date'].astype(str).str[:10]
for f in FEATS:
    if f in df.columns: df[f]=pd.to_numeric(df[f],errors='coerce').fillna(0).clip(-1e6,1e6)
df=df.replace([np.inf,-np.inf],np.nan)

latest_date=sorted(df['date_str'].unique())[-1]
latest=df[df['date_str']==latest_date].copy()

# 基本面过滤
FILTER_PATH=f'{ML}/us_filtered_syms.json'
if os.path.exists(FILTER_PATH):
    flist=json.load(open(FILTER_PATH))
    valid_syms=set(flist['syms'])
    before=len(latest)
    latest=latest[latest['sym'].isin(valid_syms)].copy()
    after=len(latest)
    print(f'  基本面过滤: {before}->{after}只',flush=True)
else:
    print(f'  警告: 未找到过滤名单',flush=True)

# 评分
X=np.nan_to_num(latest[FEATS].values.astype(np.float32),nan=0)
raw=model.predict(xgb.DMatrix(X,feature_names=FEATS))
calib=cal.predict_proba(raw.reshape(-1,1))[:,1]
latest['prob_5pct']=calib
latest=latest.sort_values('prob_5pct',ascending=False)

# 读取当前持仓（从OpenD或缓存文件）
portfolio_file=f'{BASE}/data/portfolio_v75.json'
if os.path.exists(portfolio_file):
    portfolio=json.load(open(portfolio_file))
else:
    portfolio=[]
    print('  无持仓文件，开始全新周期',flush=True)

# 策略参数
TOP_N=5; HOLD=10; STOP=-0.15; REBAL=10

# 从评分列表生成买入候选
buy_cands=[r['sym'] for _,r in latest.head(TOP_N*2).iterrows()]
hold_syms=[p['sym'] for p in portfolio]

# 区分：持有的票继续持有（未触发止损+未到期）
keep_signals=[]
sell_signals=[]
for p in portfolio:
    sym=p['sym']; bp=p['bp']; days_held=p.get('days_held',0)
    # 取最新收盘价
    cp=latest.loc[latest['sym']==sym,'close'].values[0] if 'close' in latest.columns else None
    if cp is None: cp=p.get('last_price',bp)
    ret=(cp-bp)/bp
    if ret<=STOP or days_held>=HOLD:
        sell_signals.append({'sym':sym,'action':'sell','reason':'止损' if ret<=STOP else '到期',
            'ret':f'{ret*100:.1f}%','qty':p['qty'],'est_price':cp})
    else:
        keep_signals.append({'sym':sym,'action':'hold','days_held':days_held,
            'ret':f'{ret*100:.1f}%','prob':p.get('prob',0)})

# 需要卖出的股票 → 腾出现金
# 买新的TopN中不在持仓中的、且得分最高
sell_syms={s['sym'] for s in sell_signals}
keep_syms={s['sym'] for s in keep_signals}
existing_syms=keep_syms|sell_syms

new_buys=[s for s in buy_cands if s not in existing_syms]
# 检查是否到了重平衡日
# （简化：每次运行检查，如果现有持仓<TOP_N就补）
buy_targets=new_buys[:max(0,TOP_N-len(keep_signals))]

buy_signals=[]
for sym in buy_targets:
    prob_row=latest[latest['sym']==sym].iloc[0]
    buy_signals.append({'sym':sym,'action':'buy','prob':round(float(prob_row['prob_5pct']),4),
        'est_price':float(prob_row.get('close',0)) if 'close' in prob_row.index else '次日开盘'})

# 输出
print(f'\n评分日: {latest_date}')
print(f'持仓: {len(portfolio)}只')
print(f'候选Top5: {buy_cands[:5]}')
print()

print(f'=== 持有 ({len(keep_signals)}只) ===')
for s in keep_signals:
    print(f'  保持 {s["sym"]} 持仓{s["days_held"]}天, 涨跌{s["ret"]}')

print(f'\n=== 卖出 ({len(sell_signals)}只) ===')
for s in sell_signals:
    print(f'  卖出 {s["sym"]} ({s["reason"]}, {s["ret"]})')

print(f'\n=== 买入 ({len(buy_signals)}只) ===')
for s in buy_signals:
    print(f'  买入 {s["sym"]} (概率{s["prob"]:.1%})')

# 保存持仓
new_portfolio=keep_signals+[{'sym':s['sym'],'bp':s['est_price'],'qty':s.get('qty','TBD'),
    'prob':s['prob'],'days_held':0,'buy_date':latest_date} for s in buy_signals]
json.dump(new_portfolio,open(portfolio_file,'w'),indent=2)
print(f'\n持仓已更新: {portfolio_file}')
print(f'总耗时: {time.time()-T0:.1f}s')
