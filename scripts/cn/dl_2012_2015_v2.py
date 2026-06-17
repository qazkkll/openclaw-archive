"""
下载2012-2015数据 ver2 — 从现有股票池出发
"""
import sys,json,os,time
sys.stdout.reconfigure(encoding='utf-8')
BASE=r'/home/hermes/.hermes/openclaw-archive/data'
sys.path.insert(0, BASE.replace('data','scripts'))
from config_keys import TUSHARE_TOKEN
import tushare as ts
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()
t0=time.time()

# 交易日
cal=pro.trade_cal(exchange='SSE',start_date='20120101',end_date='20151231')
cl=json.loads(cal.to_json(orient='records'))
open_days=sorted([r['cal_date'] for r in cl if r['is_open']==1])
print(f'交易日: {len(open_days)}天')

# 主板股票代码（从现有a_hist_10y.parquet取）
# 从K线数据中取前1000个key来解析
import orjson
with open(BASE+'/a_hist_10y.parquet','rb')as f:
    chunk=f.read(50*1024*1024)  # 读50MB
# 提取股票代码(6位数字)
import re
codes=list(set(re.findall(r'"(\d{6})":',chunk.decode('utf-8'))))
codes=[c for c in codes if c.startswith('6')or c.startswith('0')]
print(f'主板: {len(codes)}只')
old_codes=codes  # 都下,到时再过滤
print(f'主板有10年K线的: {len(codes)}只')
print(f'预计2012前上市的: {len(old_codes)}只')

# 1. 下载K线(2012-2015)
print('\n--- K线下载 ---')
kl_file=BASE+'/kl_12_15.json'
if os.path.exists(kl_file):
    with open(kl_file,'r')as f:kl=json.load(f)
    print(f'  已有{len(kl)}只')
else:
    kl={}

kl_pending=[c for c in old_codes if c not in kl]
print(f'  需下{len(kl_pending)}只')
for i,c in enumerate(kl_pending[:500]):  # 先下500只
    try:
        full=c+('.SH'if c.startswith('6')else'.SZ')
        df=pro.daily(ts_code=full,start_date='20120101',end_date='20151231',fields='trade_date,close,open,high,low,vol')
        if df is not None and len(df)>0:
            recs=json.loads(df.to_json(orient='records'))
            recs.sort(key=lambda x:x['trade_date'])
            kl[c]={'c':[r['close']for r in recs],'dates':[r['trade_date']for r in recs]}
        time.sleep(0.3)
    except:
        time.sleep(2)
    if(i+1)%100==0:
        et=time.time()-t0
        print(f'  K线: {i+1}/{len(kl_pending)}只, {et:.0f}s')

with open(kl_file,'w')as f:json.dump(kl,f)
print(f'  K线完成({len(kl)}只), {time.time()-t0:.0f}s')

# 2. 下载资金流
print('\n--- 资金流下载 ---')
mf_file=BASE+'/mf_12_15.json'
if os.path.exists(mf_file):
    with open(mf_file,'r')as f:mf=json.load(f)
    done=set(mf.keys())
else:
    mf={};done=set()

need=[d for d in open_days if d not in done]
print(f'  需下{len(need)}/{len(open_days)}天')
for i,d in enumerate(need):
    try:
        df=pro.moneyflow(trade_date=d)
        if df is not None and len(df)>0:
            recs=json.loads(df.to_json(orient='records'))
            daily={}
            for r in recs:
                c6=r['ts_code'][:6]
                if c6 not in old_codes:continue
                daily[c6]={'net_mf':float(r.get('net_mf_amount',0)or 0),
                          'be':float(r.get('buy_elg_amount',0)or 0),'se':float(r.get('sell_elg_amount',0)or 0),
                          'bl':float(r.get('buy_lg_amount',0)or 0),'sl':float(r.get('sell_lg_amount',0)or 0)}
            if daily:mf[d]=daily
        time.sleep(0.25)
    except:
        time.sleep(2)
    if(i+1)%100==0:
        et=time.time()-t0
        print(f'  资金流: {i+1}/{len(need)}天, {et:.0f}s')

with open(mf_file,'w')as f:json.dump(mf,f)
print(f'  资金流完成({len(mf)}天), {time.time()-t0:.0f}s')
print(f'\n总: {time.time()-t0:.0f}s')
