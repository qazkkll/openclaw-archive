"""
下载2012-2015年数据（A股全量K线+资金流）
用于A1-B模型纯外样本验证
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

# 交易日历
cal=pro.trade_cal(exchange='SSE', start_date='20120101', end_date='20151231')
cl=json.loads(cal.to_json(orient='records'))
open_days=sorted([r['cal_date'] for r in cl if r['is_open']==1])
print(f'交易日: {len(open_days)}天 ({open_days[0]}~{open_days[-1]})')

# 1. 下载K线（主板股票）
print('\n[1/3] 下载K线...')
hist_file=BASE+'/a_hist_12_15.json'
if os.path.exists(hist_file):
    with open(hist_file,'r')as f:hist=json.load(f)
    print(f'  已有{len(hist)}只, 续下...')
else:
    hist={}

# 取主板股票列表
sdf=pro.stock_basic(exchange='',list_status='L',fields='ts_code,name,market')
all_stocks=json.loads(sdf.to_json(orient='records'))
main_board=[s for s in all_stocks if (s['ts_code'].startswith('6')or s['ts_code'].startswith('0'))and s['market']=='主板']
print(f'  主板: {len(main_board)}只')

downloaded=0
for s in main_board:
    tc=s['ts_code']
    if tc[:6] in hist:continue
    try:
        df=pro.daily(ts_code=tc,start_date='20120101',end_date='20151231',fields='trade_date,open,high,low,close,vol')
        if df is not None and len(df)>0:
            recs=json.loads(df.to_json(orient='records'))
            c_arr=[r['close'] for r in recs];h_arr=[r['high'] for r in recs]
            l_arr=[r['low'] for r in recs];d_arr=[r['trade_date'] for r in recs]
            v_arr=[r['vol'] for r in recs]
            hist[tc[:6]]={'c':c_arr,'h':h_arr,'l':l_arr,'dates':d_arr,'v':v_arr}
            downloaded+=1
        time.sleep(0.35)
    except:
        time.sleep(2)
    if (downloaded+1)%100==0:
        elapsed=time.time()-t0
        print(f'  K线: {downloaded}/{len(main_board)}只, {elapsed:.0f}s')

with open(hist_file,'w',encoding='utf-8')as f:json.dump(hist,f)
print(f'  K线完成: {len(hist)}只, {time.time()-t0:.0f}s')

# 2. 下载资金流
print('\n[2/3] 下载资金流...')
mf_file=BASE+'/mf_12_15.json'
if os.path.exists(mf_file):
    with open(mf_file,'r')as f:mf=json.load(f)
    existing_dates=set(mf.keys())
else:
    mf={};existing_dates=set()

pending=[d for d in open_days if d not in existing_dates]
print(f'  待下载: {len(pending)}/{len(open_days)}天')

for i,d in enumerate(pending):
    try:
        df=pro.moneyflow(trade_date=d)
        if df is not None and len(df)>0:
            recs=json.loads(df.to_json(orient='records'))
            daily={}
            for r in recs:
                c6=r['ts_code'][:6]
                daily[c6]={
                    'net_mf':r.get('net_mf_amount',0),
                    'buy_elg':r.get('buy_elg_amount',0),
                    'sell_elg':r.get('sell_elg_amount',0),
                    'buy_lg':r.get('buy_lg_amount',0),
                    'sell_lg':r.get('sell_lg_amount',0),
                    'buy_md':r.get('buy_md_amount',0),
                    'sell_md':r.get('sell_md_amount',0),
                }
            mf[d]=daily
        time.sleep(0.35)
    except:
        time.sleep(2)
    if (i+1)%100==0:
        elapsed=time.time()-t0
        print(f'  资金流: {i+1}/{len(pending)}天, {elapsed:.0f}s')

with open(mf_file,'w',encoding='utf-8')as f:json.dump(mf,f)
print(f'  资金流完成: {len(mf)}天, {time.time()-t0:.0f}s')

print(f'\n总耗时: {time.time()-t0:.0f}s')
