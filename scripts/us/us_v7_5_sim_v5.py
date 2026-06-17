"""$5版过滤模拟 2026-04-01~2026-06-11"""
import sys, os, json, pickle, time, warnings; warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, xgboost as xgb

BASE='/home/hermes/.hermes/openclaw-archive'; ML=f'{BASE}/ml'; MD=f'{BASE}/data/models'; VER='us_v7_5'
print('='*70,flush=True)
print('V7.5 $5版过滤模拟 2026-04-01~2026-06-11',flush=True)
print('策略: T5_H10_S15_R10, 起始$100,000',flush=True)
print('过滤: 成交额≥$5M/天 + 价格≥$5 (1602只)',flush=True)
print('='*70,flush=True)
T0=time.time()

model=xgb.Booster(); model.load_model(f'{MD}/{VER}.json')
cal=pickle.load(open(f'{MD}/{VER}_calibrator.pkl','rb'))
report=json.load(open(f'{MD}/{VER}_report.json'))
FEATS=report['features']

fs=json.load(open(f'{ML}/us_filtered_syms_v5.json'))
filtered_syms=set(fs['syms'])
print(f'过滤名单: {len(filtered_syms)}只',flush=True)

df=pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str']=df['date'].astype(str).str[:10]
df=df[df['sym'].isin(filtered_syms)]
print(f'过滤后特征: {len(df)}行',flush=True)
df['target']=(df['fwd_5d_ret']>0.05).astype(int)
for f in FEATS:
    if f in df.columns:
        df[f]=pd.to_numeric(df[f],errors='coerce').fillna(0).clip(-1e6,1e6)
df=df.replace([np.inf,-np.inf],np.nan)
del df['date']

BTD=sorted(df['date_str'].unique())
BTD=[d for d in BTD if d>='2026-04-01' and d<='2026-06-11']
print(f'回测天数: {len(BTD)}',flush=True)

main=pd.read_parquet(f'{ML}/us_hist_yf_10y.parquet',columns=['ticker','date','open','close'])
main.rename(columns={'ticker':'sym'},inplace=True)
mega=pd.read_parquet(f'{ML}/us_hist_megacap_10y.parquet',columns=['sym','date','open','close'])
all_v=pd.concat([main,mega],ignore_index=True).drop_duplicates(subset=['sym','date'])
all_v['ds']=all_v['date'].astype(str).str[:10]
all_v=all_v[all_v['ds'].isin(BTD)]
close_idx={}; open_idx={}
for s,g in all_v.groupby('sym'):
    g=g.sort_values('ds')
    open_idx[s]=dict(zip(g['ds'].values,g['open'].values.astype(float)))
    close_idx[s]=dict(zip(g['ds'].values,g['close'].values.astype(float)))
del main,mega,all_v
print(f'价格索引: {len(close_idx)}只',flush=True)

print('\n逐日模拟回测...',flush=True)
CAPITAL=100000.0; TOP_N=5; HOLD=10; STOP=-0.15; REBAL=10
cash=CAPITAL; portfolio={}; curve=[]; trades_log=[]; trade_id=0

for day_idx,d in enumerate(BTD):
    day=df[df['date_str']==d]
    if len(day)<30:
        curve.append(cash+sum(p['qty']*close_idx.get(s,{}).get(d,p['bp']) for s,p in portfolio.items()))
        continue
    X=np.nan_to_num(day[FEATS].values.astype(np.float32),nan=0)
    raw=model.predict(xgb.DMatrix(X,feature_names=FEATS))
    calib=cal.predict_proba(raw.reshape(-1,1))[:,1]
    day=day.copy(); day['prob']=calib; day=day.sort_values('prob',ascending=False)
    
    for sym in list(portfolio.keys()):
        pos=portfolio[sym]
        cp=close_idx.get(sym,{}).get(d)
        if cp is None: continue
        ret=(cp-pos['bp'])/pos['bp']
        days_held=day_idx-pos['di']
        if ret<=STOP or days_held>=HOLD:
            proceed=pos['qty']*cp; cash+=proceed
            pnl=proceed-(pos['qty']*pos['bp'])
            trades_log.append({'id':trade_id,'date':d,'sym':sym,'action':'sell',
                'reason':'止损' if ret<=STOP else '到期','buy_price':round(pos['bp'],2),
                'sell_price':round(cp,2),'qty':round(pos['qty'],2),'pnl':round(pnl,2),
                'pnl_pct':round(ret*100,1),'days_held':days_held,'buy_date':pos['bd']})
            trade_id+=1; del portfolio[sym]
    
    if day_idx%REBAL==0 or len(portfolio)<TOP_N:
        hold_syms=set(portfolio.keys()); cand=[]
        for _,r in day.iterrows():
            if r['sym'] in hold_syms: continue
            nxt_idx=day_idx+1
            if nxt_idx>=len(BTD): continue
            nxt_d=BTD[nxt_idx]
            bp=open_idx.get(r['sym'],{}).get(nxt_d)
            if bp is None or np.isnan(bp) or bp<=0: continue
            cand.append((r['sym'],r['prob'],float(bp),nxt_d))
            if len(cand)>=TOP_N*2: break
        need=TOP_N-len(portfolio)
        for sym,prob,price,nxt_d in cand[:need]:
            qty=cash/max(need,1)/max(price,0.01)
            if qty<10: continue
            cost=qty*price
            if cost>cash: continue
            cash-=cost
            portfolio[sym]={'bp':price,'qty':qty,'di':day_idx,'bd':nxt_d,'prob':float(prob)}
            trades_log.append({'id':trade_id,'date':nxt_d,'sym':sym,'action':'buy',
                'reason':'新开仓','buy_price':round(price,2),'sell_price':None,
                'qty':round(qty,2),'pnl':None,'pnl_pct':None,
                'days_held':None,'buy_date':nxt_d,'prob':round(float(prob),4)})
            trade_id+=1
    
    pv=sum(p['qty']*close_idx.get(s,{}).get(d,p['bp']) for s,p in portfolio.items())
    curve.append(cash+pv)
    if day_idx%10==0:
        print(f'  Day {day_idx}: {d} NAV=${curve[-1]:,.0f} 持仓{len(portfolio)}只',flush=True)

final_cash=cash
for sym,pos in portfolio.items():
    last_d=BTD[-1]
    cp=close_idx.get(sym,{}).get(last_d,pos['bp'])
    final_cash+=pos['qty']*cp
    trades_log.append({'id':trade_id,'date':last_d,'sym':sym,'action':'sell',
        'reason':'模拟结束清仓','buy_price':round(pos['bp'],2),'sell_price':round(cp,2),
        'qty':round(pos['qty'],2),'pnl':round(pos['qty']*(cp-pos['bp']),2),
        'pnl_pct':round((cp-pos['bp'])/pos['bp']*100,1),'days_held':len(BTD)-pos['di']})

final_nav=final_cash
total_return=(final_nav/CAPITAL-1)*100
years=len(BTD)/252
annualized=((final_nav/CAPITAL)**(1/max(years,0.01))-1)*100
equity=np.array(curve)
peak=np.maximum.accumulate(equity)
mdd_pct=(equity-peak).min()/peak.max()*100 if peak.max()>0 else 0
dr=np.diff(equity)/(equity[:-1]+1e-10)
sharpe=(dr.mean()/max(dr.std(),1e-6))*np.sqrt(252) if len(dr)>20 else 0

buy_trades=[t for t in trades_log if t['action']=='buy']
sell_trades=[t for t in trades_log if t['action']=='sell' and t['pnl'] is not None]
wins=sum(1 for t in sell_trades if t['pnl']>0)
win_rate=wins/max(len(sell_trades),1)*100

print('\n'+'='*70)
print('$5版过滤模拟结果: 2026-04-01 ~ 2026-06-11')
print('='*70)
print(f'起始资金: ${CAPITAL:,.0f}')
print(f'最终净值: ${final_nav:,.0f}')
print(f'总收益率: {total_return:+.2f}%')
print(f'年化: {annualized:+.2f}%')
print(f'最大回撤: {mdd_pct:.2f}%')
print(f'夏普: {sharpe:.2f}')
print(f'交易: {len(buy_trades)}买入/{len(sell_trades)}卖出')
print(f'胜率: {win_rate:.1f}% ({wins}胜/{len(sell_trades)-wins}负)')

# 盈亏统计
pnl_by_sym={}
for t in sell_trades:
    pnl_by_sym[t['sym']]=pnl_by_sym.get(t['sym'],0)+t['pnl']

print(f'\n== 主要盈亏 ==')
for s in sorted(pnl_by_sym,key=lambda x:abs(pnl_by_sym[x]),reverse=True)[:10]:
    print(f'  {s}: ${pnl_by_sym[s]:+,.0f}')

# CODX是否被过滤掉
codx_found=[t for t in buy_trades if t['sym']=='CODX']
print(f'\nCODX 买入次数: {len(codx_found)}')
if codx_found:
    codx_pnl=pnl_by_sym.get('CODX',0)
    print(f'CODX 总盈亏: ${codx_pnl:+,.0f}')

excl_pnl=sum(v for k,v in pnl_by_sym.items() if k not in ('QH','CODX'))
print(f'\n剔除QH+CODX后的利润: ${excl_pnl:+,.0f}')
print(f'剔除QH+CODX后净值: ${100000+excl_pnl:,.0f}')

print(f'\n⏱️ {time.time()-T0:.1f}s')
