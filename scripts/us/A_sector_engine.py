#!/usr/bin/env python3
"""
行业轮动引擎 — A股核心框架
=========================
三层结构:
  1) 行业动量(短期+中期) → 识别强势/反转行业
  2) 政策/新闻信号 → 识别政策受益方向
  3) 行业拥挤度 → 规避过热板块

轻量运行, CPU自控

用法: python3 scripts/sector_engine.py
"""
import json, os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_engine import v1_score_from_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_API_KEY = "7d8e0ca352664b6d9ccd96405949b5ea"

t0 = time.time()

# ============================================================
# 加载数据
# ============================================================
with open(f'{ROOT}/data/backtest_hist_yahoo.json') as f:
    YAHOO = json.load(f)
with open(f'{ROOT}/data/sector_map_v3.json') as f:
    SECTOR_MAP = json.load(f)

# 行业→股票
sector_stocks = {}
for code, sector in SECTOR_MAP.items():
    if isinstance(sector, str):
        sector_stocks.setdefault(sector, []).append(code)

for s in list(sector_stocks.keys()):
    valid = [c for c in sector_stocks[s] if c in YAHOO and len(YAHOO[c].get('close',[]))>=200]
    sector_stocks[s] = valid
    if len(valid) < 3:
        del sector_stocks[s]

print(f'📊 有效行业: {len(sector_stocks)}个')
for s, v in sorted(sector_stocks.items(), key=lambda x:-len(x[1])):
    print(f'  {s}: {len(v)}只')

# ============================================================
# 1. 行业动量 + 拥挤度计算
# ============================================================
print()
print('═══ 1. 行业动量分析 ═══', flush=True)

def sector_momentum(code):
    """计算单只股票的动量数据"""
    d = YAHOO[code]
    c = d.get('close', [])
    if len(c) < 200: return None
    
    price = c[-1]
    if price <= 0: return None
    
    # 各周期动量
    mom = {}
    for p in [5, 10, 20, 60]:
        if len(c) > p:
            mom[f'mom{p}'] = (price / c[-p] - 1) * 100
    
    # 波动率 (近20日)
    if len(c) >= 25:
        rets_20 = [c[i]/c[i-1]-1 for i in range(-20, 0)]
        vol = math.sqrt(sum(r*r for r in rets_20)/20) * 100
        mom['vol'] = vol
    
    # 多空比对: 站上均线的比例
    ma20 = sum(c[-20:])/20
    ma50 = sum(c[-50:])/50
    mom['above_ma20'] = 1 if price > ma20 else 0
    mom['above_ma50'] = 1 if price > ma50 else 0
    
    # 动量加速/减速 (20日vs60日)
    mom20 = mom.get('mom20', 0)
    mom60 = mom.get('mom60', 0)
    
    # 短期动量是否在加速(最近5日 > 20日)
    mom5 = mom.get('mom5', 0)
    mom_accel = mom5 - mom20/4 if mom20 != 0 else 0  # 5日动量vs20日折算
    mom['accelerating'] = 1 if mom_accel > 0 else 0
    
    # 反转潜力: 20日跌幅较大时
    mom['reversal_potential'] = max(0, -mom20) / 2 if mom20 < -3 else 0
    
    # V1评分 (如果有足够数据)
    try:
        s = v1_score_from_data(c[-200:], 
                                (d.get('high',c))[-200:], 
                                (d.get('low',c))[-200:]) or 0
        mom['v1'] = s
    except:
        mom['v1'] = 0
    
    return mom

# 每个行业汇总
sectors_data = []
for sector, stocks in sorted(sector_stocks.items()):
    all_mom = [sector_momentum(c) for c in stocks]
    all_mom = [m for m in all_mom if m]
    
    if len(all_mom) < 3:
        continue
    
    # 平均
    avg = {}
    for key in ['mom5','mom10','mom20','mom60','vol','above_ma20','above_ma50',
                'accelerating','reversal_potential','v1']:
        vals = [m.get(key, 0) for m in all_mom]
        avg[key] = sum(vals) / len(vals)
    
    # 行业内部一致性 (高一致性=趋势可延续)
    mom20_vals = [m.get('mom20',0) for m in all_mom]
    pos_pct = sum(1 for v in mom20_vals if v > 0) / len(mom20_vals) * 100
    
    # 得分体系
    # 动力分: 短中期动量 + 加速信号
    power = avg['mom20'] * 0.4 + avg['mom60'] * 0.2 + avg['mom10'] * 0.3 + avg['mom5'] * 0.1
    power = max(0, power)
    
    # 健康分: 均线上方占比 + 动量加速度
    health = (avg['above_ma20'] * 20 + avg['above_ma50'] * 15 + avg['accelerating'] * 15)
    
    # 反转潜力: 当跌幅大+有反弹信号
    reversal = avg['reversal_potential'] * 2
    
    # 一致性加分: >60%股票同方向
    consistency_bonus = 5 if pos_pct > 60 else (3 if pos_pct > 50 else 0)
    
    # V1品质加分
    quality = avg['v1'] * 0.3
    
    # 拥挤度惩罚: 短期涨太快(5日>5%)且波动率高
    crowd_penalty = 0
    if avg['mom5'] > 5 and avg['vol'] > 3:
        crowd_penalty = (avg['mom5'] - 5) * 1.5
    
    total = power + health + reversal + consistency_bonus + quality - crowd_penalty
    
    sectors_data.append({
        'sector': sector, 'n': len(stocks),
        'mom5': round(avg['mom5'],1), 'mom10': round(avg['mom10'],1),
        'mom20': round(avg['mom20'],1), 'mom60': round(avg['mom60'],1),
        'above_ma20': f"{avg['above_ma20']*100:.0f}%",
        'accelerating': f"{avg['accelerating']*100:.0f}%",
        'pos_pct': f"{pos_pct:.0f}%",
        'v1_avg': round(avg['v1'],1),
        'reversal': round(avg['reversal_potential'],1),
        'power': round(power,1), 'health': round(health,1),
        'crowd_penalty': round(crowd_penalty,1),
        'total': round(total,1),
        'signal': '',
        'action': ''
    })

# 判断信号
sectors_data.sort(key=lambda x: -x['total'])

# Top 30%: 强势信号
top_threshold = sectors_data[len(sectors_data)//3]['total'] if len(sectors_data) >= 3 else 0
# Bottom 30%: 弱势信号
bot_threshold = sectors_data[-len(sectors_data)//3]['total'] if len(sectors_data) >= 3 else 0

for r in sectors_data:
    if r['reversal'] > r['total'] * 0.3 and r['mom20'] < -3:
        r['signal'] = '🔄'
        r['action'] = '反转机会'
    elif r['total'] >= top_threshold:
        r['signal'] = '🟢'
        r['action'] = '强势持有'
    elif r['total'] <= bot_threshold:
        r['signal'] = '🔴'
        r['action'] = '规避/减仓'
    else:
        r['signal'] = '🟡'
        r['action'] = '观望'

# 输出
print(f'{"":<4}{"行业":<10} {"N":>3} {"5日%":>6} {"20日%":>6} {"60日%":>6} {"均线上":>6} {"加速":>5} {"正向比":>5} {"V1":>4} {"反转":>4} {"拥挤惩罚":>5} {"总分":>5} {"操作":<8}')
print('─' * 90)
for i, r in enumerate(sectors_data):
    print(f'{r["signal"]} {r["sector"]:<8} {r["n"]:>3} {r["mom5"]:>+5.1f} {r["mom20"]:>+5.1f} {r["mom60"]:>+5.1f} {r["above_ma20"]:>5} {r["accelerating"]:>4} {r["pos_pct"]:>4} {r["v1_avg"]:>3.0f} {r["reversal"]:>+4.1f} {r["crowd_penalty"]:>+5.1f} {r["total"]:>+5.1f} {r["action"]:<8}')

# 新闻扫描
print()
print('═══ 2. 政策新闻扫描 (Finnhub) ═══', flush=True)
try:
    import urllib.request
    import urllib.parse
    
    query = "A股 政策 利好 行业"
    encoded = urllib.parse.quote(query)
    url = f"https://finnhub.io/api/v1/news?category=general&token={NEWS_API_KEY}"
    
    with urllib.request.urlopen(url, timeout=10) as resp:
        news = json.loads(resp.read().decode())
    
    # 关键政策词
    policy_keywords = {
        '半导体': ['半导体','芯片','集成电路','AI芯片','光刻'],
        '新能源': ['新能源','光伏','风电','储能','锂电','充电桩'],
        '汽车': ['新能源汽车','智能驾驶','车路云','自动驾驶'],
        '科技': ['人工智能','AI','大模型','算力','数据要素','数字经济'],
        '军工': ['军工','国防','航天','卫星','低空经济'],
        '消费': ['消费','促消费','内需','以旧换新'],
        '金融': ['金融','券商','保险','降准','降息','货币政策'],
        '地产': ['房地产','楼市','住房','保障房'],
        '医药': ['医药','医疗','创新药','生物医药'],
        '有色': ['有色','钢铁','稀土','锂矿','铜'],
        '政策': ['政策','政治局','国务院','总理','两会','中央经济'],
    }
    
    # 扫描新闻
    matched = {}
    for item in news[:30]:
        title = (item.get('headline','') + ' ' + item.get('summary','')).lower()
        for sector, keywords in policy_keywords.items():
            for kw in keywords:
                if kw.lower() in title:
                    matched.setdefault(sector, []).append(item.get('headline','')[:80])
                    break
    
    if matched:
        print(f'📰 匹配到{len(matched)}个行业的热点:')
        for sector in sorted(matched.keys()):
            print(f'  [{sector}] {matched[sector][0][:100]}')
            if len(matched[sector]) > 1:
                print(f'    +{len(matched[sector])-1}条相关新闻')
    else:
        print('📰 无匹配政策新闻')
    
except Exception as e:
    print(f'⚠️ 新闻扫描暂不可用: {e}')

# 总结
print()
print('═══ 3. 操作建议 ═══', flush=True)
strong = [r for r in sectors_data if r['signal'] == '🟢']
reversal = [r for r in sectors_data if r['signal'] == '🔄']
avoid = [r for r in sectors_data if r['signal'] == '🔴']

print(f'🟢 强势继续: {", ".join(r["sector"] for r in strong[:5])}')
if reversal:
    print(f'🔄 反转机会: {", ".join(r["sector"] for r in reversal[:3])}')
print(f'🔴 规避: {", ".join(r["sector"] for r in avoid[:3])}')

print(f'\n⏱ {time.time()-t0:.0f}s | CPU:{open("/proc/loadavg").read().split()[0]}')
