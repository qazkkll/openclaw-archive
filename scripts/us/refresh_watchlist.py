#!/usr/bin/env python3
"""每季度财报季后刷新关注列表排名"""
import yfinance as yf, warnings, numpy as np, json, datetime, sys
warnings.filterwarnings('ignore')

UNIVERSE = "/home/admin/.openclaw/workspace/data/sp500_universe.json"
pool = json.load(open(UNIVERSE))
pool_tickers = set(pool.get('tickers', []))

def ema(arr,p):
    if len(arr)<p: return []
    k=2/(p+1); r=[arr[0]]
    for x in arr[1:]: r.append(x*k+r[-1]*(1-k))
    return r

master_list = {
    "NVDA":"英伟达","AMD":"AMD","MU":"美光","INTC":"英特尔","ORCL":"甲骨文",
    "CSCO":"思科","PANW":"Palo Alto","NET":"Cloudflare","LITE":"Lumentum",
    "ZS":"Zscaler","CRM":"Salesforce","GOOGL":"谷歌","AMZN":"亚马逊",
    "META":"Meta","MSFT":"微软","AAPL":"苹果","AVGO":"博通","QCOM":"高通",
    "AMAT":"应用材料","CAT":"卡特彼勒","ELV":"Elevance","CL":"高露洁",
    "CBOE":"CBOE","ADBE":"Adobe","AIZ":"Assurant","TXN":"德州仪器","TSLA":"特斯拉",
}

data = []
for t, name in master_list.items():
    try:
        raw = yf.download(t, period="3mo", progress=False)
        if raw is None or len(raw)<60: continue
        c=[float(raw['Close'].squeeze().iloc[i]) for i in range(len(raw))]
        pr=c[-1]
        e12=ema(c,12);e26=ema(c,26)
        ml=[e12[j]-e26[j] for j in range(min(len(e12),len(e26)))]
        sg=ema(ml,9)
        mh=[ml[j]-sg[j] for j in range(min(len(ml),len(sg)))]
        hn=mh[-1] if mh else 0
        hp52=max(c[-252:]);lp52=min(c[-252:])
        p52=(pr-lp52)/(hp52-lp52)*100 if hp52>lp52 else 50
        m30=(c[-1]/c[-31]-1)*100;m5=(c[-1]/c[-6]-1)*100
        ded=max(0,(p52-40)/60)*0.7
        v42=m30*(1-min(ded,1)) if hn>0 else 0
        ma20=sum(c[-20:])/20
        data.append({'ticker':t,'name':name,'price':round(pr,2),'v42':round(v42,1),'macd':round(hn,2),'mom5':round(m5,1),'p52':round(p52,1),'ma20':round(ma20,2),'in_pool':t in pool_tickers,'market':'US'})
    except: pass

data.sort(key=lambda x:-x['v42'])
output={'timestamp':str(datetime.datetime.now()),'description':'全关注标的排名（每季度财报后刷新）','refresh_type':sys.argv[1] if len(sys.argv)>1 else 'manual','stocks':data}
with open('/home/admin/.openclaw/workspace/data/watchlist_ranked.json','w') as f: json.dump(output,f,indent=2)
print(f"✅ 刷新完成: {len(data)}只标的")
