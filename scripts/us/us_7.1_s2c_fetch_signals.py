#!/usr/bin/env python3
"""
us_7.1_s2c_fetch_signals.py
拉yfinance强信号数据: 机构持仓变化 + 内幕交易量 + 升级降级 + EPS修正 + 期权IV
输出: /home/hermes/.hermes/openclaw-project/scripts/system/us_signals_v71.json (缓存)
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, yfinance as yf

BASE='/home/hermes/.hermes/openclaw-archive'; ML_DIR=f'{BASE}/ml'
CACHE_FILE=f'{ML_DIR}/us_signals_v71.json'; LOG_FILE=f'{ML_DIR}/us_signals_v71_log.txt'

print('='*60)
print('us_7.1_s2c — 拉强信号: 机构/内幕/EPS/期权IV')
print('='*60)

# 加载股票列表
df=pd.read_parquet(f'{ML_DIR}/us_ml_feats_v3_dated.parquet')
syms=df['sym'].unique()
print(f'股票: {len(syms)}只')

# 加载已有缓存
cache={}
if os.path.exists(CACHE_FILE):
    cache=json.load(open(CACHE_FILE,'r'))
    print(f'已有缓存: {len(cache)}只')

# 日志
def log(msg):
    with open(LOG_FILE,'a') as f: f.write(f'{time.strftime("%H:%M:%S")} {msg}\n')

new=0; errs=0
for i,sym in enumerate(syms):
    if sym in cache:
        continue
    
    try:
        t=yf.Ticker(sym)
        sdata={}
        
        # 机构持仓
        ih=t.institutional_holders
        if ih is not None and len(ih)>0:
            sdata['inst_pct_change']=ih.iloc[0]['pctChange']  # Top1机构变化
            sdata['inst_pct_held']=ih['pctHeld'].sum()  # 机构总持股比例
            # 总机构变化(取前5均值)
            pct_ch=ih['pctChange'].iloc[:5].mean() if len(ih)>=5 else ih['pctChange'].mean()
            sdata['inst_avg_change']=pct_ch
        
        # 内幕交易（过去30天净买入）
        it=t.insider_transactions
        if it is not None and len(it)>0:
            it=it.copy()
            # 检测列名 - 实际上含Start Date
            if 'Start Date' in it.columns:
                it=it.rename(columns={'Start Date':'Date'})
            elif 'date' in it.columns:
                pass
            
            if 'Shares' in it.columns and 'Transaction' in it.columns:
                buys=it[it['Transaction'].str.contains('Purchase|Buy|Acquire',na=False,case=False)]
                sells=it[it['Transaction'].str.contains('Sale|Sell',na=False,case=False)]
                net=buys['Shares'].sum()-sells['Shares'].sum()
                sdata['insider_net_shares']=int(net)
            
            # 内幕交易总次数
            sdata['insider_txns']=len(it)
        
        # 流通股（最近2次变化率）
        sh=t.get_shares_full()
        if sh is not None and len(sh)>1:
            sdata['shares_out']=int(sh.iloc[-1])
            # 计算变化率
            sdata['shares_chg_3m']=float((sh.iloc[-1]/sh.iloc[-min(3,len(sh))])-1)
        
        # 升级降级（最近30天净上调）
        ud=t.upgrades_downgrades
        if ud is not None and len(ud)>0:
            # 检测目标价变化
            if 'currentPriceTarget' in ud.columns and 'priorPriceTarget' in ud.columns:
                pt_chg=(ud['currentPriceTarget']-ud['priorPriceTarget']).dropna()
                if len(pt_chg)>0:
                    sdata['pt_avg_chg']=float(pt_chg.mean())
                    sdata['pt_net_up']=int((pt_chg>0).sum()-(pt_chg<0).sum())
            
            # 升级vs降级
            if 'ToGrade' in ud.columns and 'FromGrade' in ud.columns:
                # 粗略分类
                upgrades=(ud['ToGrade'].str.contains('Buy|Overweight|Outperform|Strong',na=False,case=False)).sum()
                downs=(ud['ToGrade'].str.contains('Sell|Underweight|Reduce|Underperform',na=False,case=False)).sum()
                sdata['upgrades']=int(upgrades); sdata['downgrades']=int(downs)
        
        # EPS修正
        er=t.eps_revisions
        if er is not None:
            if isinstance(er,pd.DataFrame) and 'upLast7days' in er.columns:
                sdata['eps_up_7d']=int(er['upLast7days'].iloc[0])
                sdata['eps_down_7d']=int(er['downLast7Days'].iloc[0])
        
        # 期权: 近月平值IV
        try:
            opt=t.option_chain()
            if opt is not None:
                calls,puts=opt
                atm_c=calls.iloc[(calls['strike']-t.info.get('regularMarketPrice',0)).abs().argsort()[:1]]
                atm_p=puts.iloc[(puts['strike']-t.info.get('regularMarketPrice',0)).abs().argsort()[:1]]
                if len(atm_c)>0 and len(atm_p)>0:
                    sdata['iv_call']=float(atm_c['impliedVolatility'].iloc[0])
                    sdata['iv_put']=float(atm_p['impliedVolatility'].iloc[0])
                    sdata['iv_skew']=sdata['iv_put']/max(sdata['iv_call'],0.001)-1
        
        except Exception:
            pass  # 期权可能没有
        
        cache[sym]=sdata
        new+=1
    except Exception as e:
        cache[sym]={}
        errs+=1
    
    if (i+1)%50==0:
        json.dump(cache,open(CACHE_FILE,'w'))
        log(f'[{i+1}/{len(syms)}] 新{new} 错{errs}')
        print(f'  [{i+1}/{len(syms)}] 新拉{new}只, 错误{errs}只',flush=True)

# 最终保存
json.dump(cache,open(CACHE_FILE,'w'))
print(f'\n完成! {len(cache)}/{len(syms)} 新拉{new} 错误{errs}')
print(f'→ {CACHE_FILE}')
print('='*60)
