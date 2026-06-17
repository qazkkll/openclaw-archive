"""验证原版backtest.py框架，跑原版参数看能否复现384笔"""
import sys, os, json, time
ws = '/home/hermes/.hermes/openclaw-archive'
sys.path.insert(0, os.path.join(ws, 'scripts'))
sys.path.insert(0, os.path.join(ws, 'scripts', 'recovered'))

# import common and backtest
import common  # sf, sma, ema
sys.modules['lib'] = type(sys)('lib')
sys.modules['lib'].common = common
sys.modules['lib.common'] = common
from backtest import run_backtest, v5s_score

# 构建指标（原版backtest.py需要指标字典）
def build_indicators(c, h, l):
    """原版指标函数（对应lib/score_runner中的老版本）"""
    n = len(c)
    # sma/ema
    def sm(a,p): return [None]*(p-1)+[sum(a[i-p+1:i+1])/p for i in range(p-1,len(a))]
    def em(a,p): k=2/(p+1); r=[a[0]]; [r.append(v*k+r[-1]*(1-k)) for v in a[1:]]; return r
    
    m5=sm(c,5);m20=sm(c,20);m60=sm(c,60);m120=sm(c,120)
    e12=em(c,12);e26=em(c,26);macd=[e12[i]-e26[i] for i in range(n)]
    sig=sm(macd,9)
    hst=[macd[i]-(sig[i] if i<len(sig) and sig[i] is not None else 0) for i in range(n)]
    
    gl=[max(c[i]-c[i-1],0) for i in range(1,n)]
    ls=[max(c[i-1]-c[i],0) for i in range(1,n)]
    rsi=[None]*14
    if len(gl)>=14:
        ag=sum(gl[:14])/14;al=sum(ls[:14])/14
        for i in range(14,n):
            rsi.append(100-100/(1+ag/al) if al>0 else 100)
            if i<len(gl): ag=(ag*13+gl[i])/14; al=(al*13+ls[i])/14
    
    p52=[None]*252
    for i in range(252,n):
        lo=min(c[i-251:i+1]);hi=max(c[i-251:i+1])
        p52.append((c[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    
    return {'ma5':m5,'ma20':m20,'ma60':m60,'ma120':m120,'macd':macd,
            'macd_signal':sig,'macd_hist':hst,'rsi':rsi,'p52':p52}

t0 = time.time()
data = json.load(open(ws + '/data/us_hist_clean.parquet', 'rb'))
syms = list(data.keys())
print('总池: %d' % len(syms), flush=True)

# 预计算指标
ind_all = {}
tickers = []
for i,sy in enumerate(syms):
    if i%500==0: print('  指标 %d/%d' % (i,len(syms)), flush=True)
    d = data[sy]
    c=[float(x) for x in d.get('c',[])]
    h=[float(x) for x in d.get('h',[])]
    l=[float(x) for x in d.get('l',[])]
    if len(c) < 520: continue
    ind = build_indicators(c,h,l)
    if ind is None: continue
    ind_all[sy] = ind
    tickers.append(sy)

print('有效: %d只 (%ds)' % (len(tickers), time.time()-t0), flush=True)

# 原版backtest.py用N = min(lengths) - 第一步先打印N
N = min(len(data[t]['c']) for t in tickers)
print('N (最短股长度): %d' % N, flush=True)

# 跑回测
print('开始回测...', flush=True)
result = run_backtest(data, ind_all, tickers, hd=20, ms=60, mh=8, sl=-15, W=400)
elapsed = time.time()-t0

print('\n' + '='*55)
print('原版backtest.py复现验证')
print('='*55)
print('参数: hd=%d ms=%d mh=%d sl=%d' % (result['params']['hd'], result['params']['ms'], result['params']['mh'], result['params']['sl']))
print('数据: 2436只, N=%d天' % N)
print('')
print('年化: %.1f%%' % result['ann'])
print('夏普: %.2f' % result['sh'])
print('回撤: %.1f%%' % result['dd'])
print('胜率: %.1f%%' % result['wr'])
print('交易: %d笔' % result['trades_count'])
print('')
print('目标(bt_v5_5y_clean.json):')
print('年化: 49.1%  夏普: 1.57  回撤: 19.7%  胜率: 47.7%  交易: 384')
print('')
print('耗时: %ds' % elapsed)
