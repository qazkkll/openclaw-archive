#!/usr/bin/env python3
"""
us_7.1_s2d_fetch_signals_fast.py — 快速拉信号（跳过404股票）
只拉市值排名前1000的股票+指定持仓
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, yfinance as yf

BASE='/home/hermes/.hermes/openclaw-archive'; ML_DIR=f'{BASE}/ml'
CACHE_FILE=f'{ML_DIR}/us_signals_v71.json'

print('='*60)
print('us_7.1_s2d — 快速拉信号（跳过垃圾股）')
print('='*60)

# 加载股票 + 按市值排序
df=pd.read_parquet(f'{ML_DIR}/us_ml_feats_v3_dated.parquet')
# 取每只股票最后一行的市值
last_rows=df.sort_values(['sym','date']).groupby('sym').last().reset_index()
wd=last_rows[['sym','price','market_cap']].fillna(0)
wd=wd.sort_values('market_cap',ascending=False)
print(f'总股票: {len(wd)}只')

# 加载缓存
cache={}
if os.path.exists(CACHE_FILE):
    cache=json.load(open(CACHE_FILE,'r'))
    print(f'缓存: {len(cache)}只')

# 排除已有
syms_to_fetch=[s for s in wd['sym'].tolist() if s not in cache]
print(f'还需拉: {len(syms_to_fetch)}只')

# 只拉前500大+持仓股票
holdings={'NOK','NVDA','GNRC','ON','QCOM'}
top_syms=list(wd.head(500)['sym'].values)+[s for s in holdings if s not in set(wd.head(500)['sym'].values)]
syms_to_fetch=[s for s in top_syms if s not in cache]
print(f'实际拉取: {len(syms_to_fetch)}只（500大盘+持仓）')

new=0; errs=0; done=len(cache)
for i,sym in enumerate(syms_to_fetch):
    try:
        t=yf.Ticker(sym)
        sdata={}
        
        # 机构持仓
        try:
            ih=t.institutional_holders
            if ih is not None and len(ih)>0:
                sdata['inst_pct_change']=float(ih.iloc[0]['pctChange'])
                sdata['inst_pct_held']=float(ih['pctHeld'].sum())
                pct_ch=ih['pctChange'].iloc[:5].mean() if len(ih)>=5 else ih['pctChange'].mean()
                sdata['inst_avg_change']=float(pct_ch)
        except: pass
        
        # 内幕交易
        try:
            it=t.insider_transactions
            if it is not None and len(it)>0:
                it=it.copy()
                if 'Shares' in it.columns:
                    it['TType']=it.get('Transaction','').astype(str).str.lower()
                    buy_s=(it[it['TType'].str.contains('purchase|buy|acquire',na=False)] if 'Transaction' in it.columns else pd.DataFrame()).index
                    all_s=it.index
                    if len(buy_s)>0:
                        net=it.loc[buy_s,'Shares'].sum()-it.loc[~it.index.isin(buy_s),'Shares'].sum()
                        sdata['insider_net']=int(net) if not pd.isna(net) else 0
                    else:
                        sdata['insider_net']=0
        except: pass
        
        # 流通股变化
        try:
            sh=t.get_shares_full()
            if sh is not None and len(sh)>2:
                sdata['shares_out']=int(sh.iloc[-1])
                sdata['shares_chg_3m']=float(sh.iloc[-1]/sh.iloc[-min(4,len(sh))]-1)
        except: pass
        
        # 升级降级
        try:
            ud=t.upgrades_downgrades
            if ud is not None and len(ud)>0:
                if 'currentPriceTarget' in ud.columns and 'priorPriceTarget' in ud.columns:
                    pt=ud['currentPriceTarget']-ud['priorPriceTarget']
                    pt=pt.dropna()
                    if len(pt)>0:
                        sdata['pt_avg_chg']=float(pt.mean())
                        sdata['pt_net']=int((pt>0).sum()-(pt<0).sum())
        except: pass
        
        # EPS修正
        try:
            er=t.eps_revisions
            if er is not None:
                if hasattr(er,'columns') and 'upLast7days' in er.columns:
                    sdata['eps_up_7d']=int(er['upLast7days'].iloc[0])
                    sdata['eps_down_7d']=int(er['downLast7Days'].iloc[0])
        except: pass
        
        # 期权IV（仅大盘+持仓）
        if sym in holdings:
            try:
                opt=t.option_chain()
                if opt is not None:
                    calls,puts=opt
                    cp_val=float(wd[wd['sym']==sym]['price'].values[0]) if len(wd[wd['sym']==sym])>0 else 100.0
                    info_p=t.info.get('regularMarketPrice',cp_val)
                    cp=float(info_p) if info_p else cp_val
                    cc=calls.iloc[(calls['strike']-cp).abs().argsort()[:1]]
                    pp=puts.iloc[(puts['strike']-cp).abs().argsort()[:1]]
                    if len(cc)>0:
                        sdata['iv_call']=float(cc['impliedVolatility'].iloc[0])
                    if len(pp)>0:
                        sdata['iv_put']=float(pp['impliedVolatility'].iloc[0])
                    if 'iv_call' in sdata and 'iv_put' in sdata:
                        sdata['iv_skew']=sdata['iv_put']/max(sdata['iv_call'],0.001)-1
            except Exception as oe:
                pass
        
        # 做空数据（从info获取，不需要额外请求）
        try:
            info=t.info
            for k,key in [('short_ratio','shortRatio'),('short_pct','shortPercentOfFloat')]:
                if info.get(key) is not None:
                    sdata[k]=info[key]
        except: pass
        
        cache[sym]=sdata
        new+=1
    except:
        cache[sym]={}
        errs+=1
    
    done+=1
    if (i+1)%20==0:
        json.dump(cache,open(CACHE_FILE,'w'))
        print(f'  [{i+1}/{len(syms_to_fetch)}] 新{new} 错{errs} 累积{done}',flush=True)

json.dump(cache,open(CACHE_FILE,'w'))

# 统计
has_data=sum(1 for v in cache.values() if len(v)>1)
print(f'\n完成! {len(cache)}只在缓存, {has_data}只有数据')
print(f'新拉{new} 错误{errs}')

# 各字段覆盖率
fields=['inst_pct_change','inst_pct_held','insider_net','shares_out','shares_chg_3m',
        'pt_avg_chg','pt_net','eps_up_7d','iv_call','iv_put','iv_skew','short_ratio','short_pct']
for f in fields:
    cnt=sum(1 for v in cache.values() if f in v)
    print(f'  {f:20s}: {cnt}/{len(cache)} ({cnt/len(cache)*100:.0f}%)')

print('\n持仓:')
for s in holdings:
    d=cache.get(s,{})
    print(f'  {s}: {list(d.keys()) if d else "空"}')
print('='*60)
