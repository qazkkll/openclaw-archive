"""
美股双模型扫描 V2 — 使用 yfinance 拉数据
V5-S（进攻趋势动量）+ V5-D（防御超卖抄底）+ 🎫 彩票
"""
import json, time, os, sys
sys.stdout.reconfigure(encoding='utf-8')
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

OUT = r'/home/hermes/.hermes/openclaw-archive/data'
LOG = r'/home/hermes/.hermes/openclaw-archive\logs\us_scan.log'
os.makedirs(OUT, exist_ok=True)

def log(m):
    t = time.strftime('%H:%M:%S')
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write('[%s] %s\n' % (t,m))
    print(m, flush=True)

# Load tickers
try:
    with open(r'/home/hermes/.hermes/openclaw-archive/data\us_active_pool.json') as f:
        tickers = json.load(f)['tickers']
except:
    with open(os.path.join(OUT, 'watchlist.json'), encoding='utf-8') as f:
        watchlist = json.load(f)
    tickers = [s['ticker'] for s in watchlist if s.get('pool')=='us_active']

tickers = list(dict.fromkeys(tickers))  # 去重
log('美股池: %d只' % len(tickers))

# ========== 扫描 ==========
log('===== yfinance批量扫描 =====')
results = []
errors = 0
batch_size = 100

for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    # 用yfinance批量downloader下载
    try:
        data = yf.download(batch, period='3mo', interval='1d', progress=False, auto_adjust=True, group_by='ticker')
    except Exception as e:
        log(f'  批{i}下载失败: {str(e)[:60]}')
        errors += len(batch)
        continue
    
    for t in batch:
        try:
            if t not in data or data[t].empty or len(data[t]) < 15:
                continue
            df = data[t]
            closes = df['Close'].dropna().values
            volumes = df['Volume'].dropna().values
            if len(closes) < 15:
                continue
            
            pr = float(closes[-1])
            d1 = (closes[-1]/closes[-2]-1)*100
            m5 = (closes[-1]/closes[min(-5, -len(closes))]-1)*100
            m10 = (closes[-1]/closes[min(-10, -len(closes))]-1)*100
            m20 = (closes[-1]/closes[min(-20, -len(closes))]-1)*100
            m30 = (closes[-1]/closes[min(-30, -len(closes))]-1)*100
            
            # RSI 14
            gains = sum(max(closes[i]-closes[i-1],0) for i in range(-14,0))
            losses = sum(max(closes[i-1]-closes[i],0) for i in range(-14,0))
            rsi = 100-100/(1+gains/losses) if losses>0 else 100
            
            # 52周高低
            hp52 = max(closes)
            lp52 = min(closes)
            p52 = pr/hp52*100 if hp52>0 else 50
            p52_range = ((pr-lp52)/(hp52-lp52))*100 if hp52>lp52 else 50
            
            # MA
            ma20 = sum(closes[-20:])/20 if len(closes)>=20 else pr
            ma20_dev = (pr/ma20-1)*100
            
            # vol ratio
            vol_avg = sum(volumes[-20:])/20 if len(volumes)>=20 else volumes.mean()
            vol_ratio = volumes[-1]/vol_avg if vol_avg>0 else 1
            
            # ===== V5-S (进攻趋势动量) =====
            # 条件：30d动量>0 + 5d动量>0 + RSI不过热 + 价格不在绝对底部
            v5_s_pass = m30 > 0 and m5 > 0 and rsi < 68 and p52 > 30
            # 评分：动量30*0.5 + 动量5*0.2 + RSI位置*0.15 + 52周位置*0.15
            v5_s_score = m30*0.5 + m5*0.2 + (68-max(rsi,30))/(68-30)*15 + (p52/100)*15 if v5_s_pass else 0
            
            # ===== V5-D (防御超卖抄底) =====
            # 条件：近日大跌 + RSI低 + 30d仍有正趋势
            v5_d_pass = d1 < -2.0 and rsi < 40 and m30 > 0
            # 评分：超卖程度
            v5_d_score = abs(d1)*10 + (40-rsi)*5 + m30*0.3 if v5_d_pass else 0
            
            # 综合
            total = round(v5_s_score + v5_d_score, 1)
            if pr < 2:
                total = 0
                v5_s_score = 0
                v5_d_score = 0
            
            results.append({
                'ticker': t,
                'price': round(pr, 2),
                'total_score': total,
                'v5_s_score': round(v5_s_score, 1),
                'v5_s_pass': v5_s_pass,
                'v5_d_score': round(v5_d_score, 1),
                'v5_d_pass': v5_d_pass,
                'm30': round(m30, 1),
                'm5': round(m5, 1),
                'd1': round(d1, 1),
                'rsi': round(rsi, 1),
                'p52': round(p52, 1),
                'ma20_dev': round(ma20_dev, 1)
            })
        except Exception as e:
            errors += 1
            continue
    
    if (i+batch_size) % 500 == 0:
        log('  进度 %d/%d' % (i+batch_size, len(tickers)))

log('扫描完成: %d只收录 ✓  %d只失败' % (len(results), errors))

if not results:
    log('ERROR: 0 results, aborting')
    exit(1)

# ========== 排序输出 ==========
results.sort(key=lambda x: -x['total_score'])

v5_s_list = [r for r in results if r['v5_s_pass']]
v5_d_list = [r for r in results if r['v5_d_pass']]

log('V5-S(进攻动量): %d只' % len(v5_s_list))
log('V5-D(防御抄底): %d只' % len(v5_d_list))

log('\n=== V5-S 综合Top20 ===')
for r in results[:20]:
    tag = 'S' if r['v5_s_pass'] else 'D' if r['v5_d_pass'] else ' '
    log('  %s %-6s $%.0f score=%-5.1f 30d%+.1f%% 5d%+.1f%% RSI%.0f p52=%.0f%%' % (
        tag, r['ticker'], r['price'], r['total_score'],
        r['m30'], r['m5'], r['rsi'], r['p52']))

# V5-D top10
log('\n=== V5-D 防守买入Top10 ===')
v5_d_sorted = sorted(v5_d_list, key=lambda x: -x['v5_d_score'])
for r in v5_d_sorted[:10]:
    log('  %-6s $%.0f score_d=%.1f d1%+.1f%% RSI%.0f 30d%+.1f%%' % (
        r['ticker'], r['price'], r['v5_d_score'],
        r['d1'], r['rsi'], r['m30']))

# ========== 🎫 彩票候选 ==========
# 从全量里找：低价(<50) + RSI超卖(<30) + 非S非D
lottery = [r for r in results if not r['v5_s_pass'] and not r['v5_d_pass']
           and r['rsi'] < 30 and r['price'] < 200]
lottery.sort(key=lambda x: x['rsi'])

log('\n=== 🎫 彩票候选(%d只) ===' % len(lottery))
for r in lottery[:15]:
    log('  🎫 %-6s $%.0f RSI=%.0f p52=%.0f%% d1%+.1f%% 30d%+.1f%%' % (
        r['ticker'], r['price'], r['rsi'], r['p52'], r['d1'], r['m30']))

# ========== 评分区间分布 ==========
log('\n=== V5总评分分布 ===')
for lo in range(0, 200, 20):
    cnt = sum(1 for r in results if lo <= r['total_score'] < lo+20)
    if cnt:
        log('  %3d-%3d: %d只' % (lo, lo+19, cnt))

# ========== 保存 ==========
# us_scored.json
us_scored = [{
    'ticker': r['ticker'],
    'score': r['total_score'],
    'v5_s_score': r['v5_s_score'],
    'v5_d_score': r['v5_d_score'],
    'price': r['price'],
    'm30': r['m30'],
    'm5': r['m5'],
    'd1': r['d1'],
    'rsi': r['rsi'],
    'p52': r['p52'],
    'ma20_dev': r['ma20_dev'],
    'v5_s_pass': int(r['v5_s_pass']),
    'v5_d_pass': int(r['v5_d_pass']),
    'updated': time.strftime('%m-%d %H:%M')
} for r in results]

with open(os.path.join(OUT, 'us_scored.json'), 'w', encoding='utf-8') as f:
    json.dump(us_scored, f, indent=2)
log('us_scored.json 保存完成 (%d只)' % len(us_scored))

# us_quality_pool.json
quality = {
    'updated': time.strftime('2026-06-03T%H:%M:%S.000000'),
    'us_stocks': [{
        'symbol': r['ticker'],
        'name': '',
        'sector': '',
        'price': r['price'],
        'v5': r['total_score'],
        'rsi': r['rsi'],
        'is_holding': 'false',
        'comment': ''
    } for r in results[:200]]
}
with open(os.path.join(OUT, 'us_quality_pool.json'), 'w', encoding='utf-8') as f:
    json.dump(quality, f, indent=2)
log('us_quality_pool.json 保存完成 (%d只)' % len(quality['us_stocks']))

# 🎫 彩票
lottery_out = [{
    'ticker': r['ticker'],
    'price': r['price'],
    'rsi': r['rsi'],
    'p52': r['p52'],
    'd1': r['d1'],
    'm30': r['m30'],
    'reason': '深度超卖' if r['rsi']<20 else 'RSI超卖'
} for r in lottery[:30]]
with open(os.path.join(OUT, 'us_lottery.json'), 'w', encoding='utf-8') as f:
    json.dump(lottery_out, f, indent=2)
log('us_lottery.json 保存完成 (%d只)' % len(lottery_out))

log('\n===== 完成 =====')
log('生成: us_scored.json + us_quality_pool.json + us_lottery.json')
