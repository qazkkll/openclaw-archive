#!/usr/bin/env python3
"""
us_ld2_daily_score.py — 蓝盾2.0 每日大盘股评分
基于V5三模型-主力S部分（6月2日定稿-纯评分公式，非ML）
用途：大盘股（SP500核心成分股）技术面评分

评分架构（满分110）：
  趋势排列 30分 + 动量持续性 25分 + MACD能量 25分
  + 均线偏离度 10分 + RSI位置 10分 + 52周位置 10分

买入线: ≥60分  / 强势买入: ≥80分 / 危险: <40分

用法:
  python us_ld2_daily_score.py [--stocks NVDA,AAPL,MSFT ...]
    不传--stocks则评分预定义的大盘股池
"""
import sys, json, time, os, warnings, yfinance as yf
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# 导入评分引擎
ENGINE = r'/home/hermes/.hermes/openclaw-archive\scripts\us_score_engine.py'
with open(ENGINE, encoding='utf-8') as f:
    code = f.read()
engine = {}
exec(code, engine)
indicators_v5 = engine['_indicators_for_v5']
_sf = engine['_sf']

# ─── 默认大盘股池 (SP500核心，手动维护) ───
DEFAULT_LARGE_CAPS = {
    # Mag7
    'AAPL': '苹果', 'MSFT': '微软', 'GOOGL': '谷歌', 'AMZN': '亚马逊',
    'NVDA': '英伟达', 'META': 'Meta', 'TSLA': '特斯拉',
    # 半导体
    'AVGO': '博通', 'AMD': 'AMD', 'INTC': '英特尔', 'QCOM': '高通',
    # 金融
    'JPM': '摩根大通', 'GS': '高盛', 'BAC': '美国银行', 'V': 'Visa',
    'MA': '万事达', 'BLK': '贝莱德',
    # 消费
    'WMT': '沃尔玛', 'PG': '宝洁', 'KO': '可口可乐', 'PEP': '百事',
    'COST': '好市多', 'HD': '家得宝',
    # 医疗
    'JNJ': '强生', 'PFE': '辉瑞', 'UNH': '联合健康', 'LLY': '礼来',
    'MRK': '默克', 'ABBV': '艾伯维',
    # 能源
    'XOM': '埃克森美孚', 'CVX': '雪佛龙',
    # 科技
    'ORCL': '甲骨文', 'CRM': 'Salesforce', 'ADBE': 'Adobe',
    'NFLX': '奈飞', 'DIS': '迪士尼',
    # 工业
    'CAT': '卡特彼勒', 'GE': '通用电气', 'BA': '波音',
    # 电信
    'T': 'AT&T', 'VZ': '威瑞森',
}


def score_single(code, hist):
    """对单只股票评分，返回详细评分结果"""
    c = hist['Close'].tolist()
    hi = hist['High'].tolist()
    lo = hist['Low'].tolist()

    ind = indicators_v5(c)
    if not ind:
        return None

    # 计算评分子项
    def _score_detail(ind, di):
        c2 = ind['close']
        p = _sf(c2, di)
        if p <= 0:
            return {'total': 0, 'trend': 0, 'momentum': 0, 'macd': 0,
                    'ma_dev': 0, 'rsi_score': 0, 'p52_score': 0}
        m5 = _sf(ind['ma5'], di); m20 = _sf(ind['ma20'], di)
        m60 = _sf(ind['ma60'], di); m120 = _sf(ind['ma120'], di)

        # 趋势排列 (30)
        tr = 0
        if m5 > m20: tr += 6
        if m20 > m60: tr += 8
        if m60 > m120: tr += 10
        if p > m20: tr += 3
        if p > m60: tr += 3
        if m5 > m20 and m20 > m60 and m60 > m120:
            tr = min(tr + 5, 30)
        tr = min(tr, 30)

        # 动量 (25)
        p20 = _sf(c2, di-20); p60 = _sf(c2, di-60)
        m20v = (p-p20)/p20*100 if p20 > 0 else 0
        m60v = (p-p60)/p60*100 if p60 > 0 else 0
        mo = 10
        if m20v > 3: mo += 5
        if m20v > 10: mo += 5
        if m60v > 5: mo += 3
        if m60v > 15: mo += 3
        if -5 <= m60v <= 5: mo -= 3
        p30 = _sf(c2, di-30)
        m30v = (p-p30)/p30*100 if p30 > 0 else 0
        if m30v > 40:
            overheat = (m30v - 40) / 5
            mo = max(mo - min(overheat, 10), 0)
        mo = min(mo, 25)

        # MACD (25)
        mh = _sf(ind['macd_hist'], di); mhp = _sf(ind['macd_hist'], di-1)
        ml = _sf(ind['macd'], di); msig = _sf(ind['macd_signal'], di)
        ms2 = 0
        if ml > msig: ms2 += 8
        if mh > 0 and mhp <= 0: ms2 += 10
        elif mh > 0 and mh > mhp: ms2 += 7
        elif mh > 0: ms2 += 4
        if mh < 0 and mhp > 0: ms2 -= 5
        ms2 = min(ms2, 25)

        # 均线偏离 (10)
        ma20_dev = (p-m20)/m20*100 if m20 > 0 else 0
        md = 0
        if 1 <= ma20_dev <= 8: md = 10
        elif -2 <= ma20_dev < 1: md = 5
        elif 8 < ma20_dev <= 15: md = 6
        elif ma20_dev > 15: md = 3
        elif ma20_dev < -5: md = 0

        # RSI (10)
        rsi = _sf(ind['rsi'], di)
        rs = 5
        if 50 <= rsi <= 65: rs = 10
        elif 35 <= rsi < 50: rs = 7
        elif 65 < rsi <= 75: rs = 5
        elif rsi > 80: rs = 2
        elif rsi < 30: rs = 4

        # 52周 (10)
        p52 = _sf(ind['p52'], di)
        ps = 0
        if 30 <= p52 <= 70: ps = 10
        elif 70 < p52 <= 85: ps = 6
        elif 85 < p52 <= 100: ps = 2
        elif 15 <= p52 < 30: ps = 7
        elif p52 < 15: ps = 4

        total = tr + mo + ms2 + md + rs + ps
        return {
            'total': total, 'trend': tr, 'momentum': mo,
            'macd': ms2, 'ma_dev': md, 'rsi_score': rs, 'p52_score': ps,
            'price': round(p, 2),
            'ma20_dev_pct': round(ma20_dev, 2),
            'rsi': round(rsi, 1),
            'p52': round(p52, 0),
        }

    cur = _score_detail(ind, -1)

    # 20日评分统计（稳定性评估）
    scores_20d = []
    for offset in range(-1, -21, -1):
        d = _score_detail(ind, offset)
        scores_20d.append(d['total'])

    cur['avg_20d'] = round(sum(scores_20d)/len(scores_20d), 1)
    cur['max_20d'] = max(scores_20d)
    cur['min_20d'] = min(scores_20d)
    cur['std_20d'] = round((sum((s-sum(scores_20d)/len(scores_20d))**2
                                 for s in scores_20d)/len(scores_20d))**0.5, 1)

    return cur


def run(args=None):
    T0 = time.time()

    # 解析股票池
    SP500_PATH = '/home/hermes/.hermes/openclaw-project/data/sp500_syms.json'
    if args and '--stocks' in sys.argv:
        idx = sys.argv.index('--stocks')
        codes = [c.strip() for c in sys.argv[idx+1].split(',')]
        stock_map = {c: c for c in codes}
    else:
        # 默认：SP500全量
        if os.path.exists(SP500_PATH):
            sp500 = json.load(open(SP500_PATH))
            stock_map = {c: c for c in sp500}
        else:
            stock_map = DEFAULT_LARGE_CAPS.copy()
        print(f'  注意: 蓝盾2.0默认池 = SP500全量 ({len(stock_map)}只)')

    print('='*65)
    print(f'  🛡️  蓝盾2.0 大盘股评分  {time.strftime("%Y-%m-%d %H:%M")}')
    print(f'  评分引擎: V5三模型-主力S  |  满分110  |  买入线≥60')
    print(f'  池: {len(stock_map)}只')
    print('='*65)

    # ─── 分批下载数据 ───
    codes = list(stock_map.keys())
    print(f'\n📥 下载K线 (2年, 分批50只)...', end='', flush=True)
    
    all_data = {}
    batch_size = 50
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        print(f' {i//batch_size+1}', end='', flush=True)
        try:
            data = yf.download(batch, period='2y', group_by='ticker',
                              progress=False, auto_adjust=True)
            if batch_size == 1 and len(batch) == 1:
                all_data[batch[0]] = data
            else:
                for c in batch:
                    if c in data.columns.levels[0]:
                        all_data[c] = data[c].dropna()
        except Exception as e:
            print(f'!', end='', flush=True)
            for c in batch:
                try:
                    t = yf.Ticker(c)
                    h = t.history(period='2y')
                    if len(h) > 200:
                        all_data[c] = h
                except:
                    pass
    print(f' ✅  (完成, {len(all_data)}只有数据)')

    results = []
    errors = []
    for code in codes:
        name = stock_map[code]
        try:
            h = all_data.get(code)
            if h is None or len(h) < 260:  # 约1年数据不足以计算52w位置
                errors.append(f'{code} ({name}): 仅{len(h) if h is not None else 0}天数据')
                continue

            score = score_single(code, h)
            if not score:
                errors.append(f'{code} ({name}): 指标计算失败')
                continue

            score['code'] = code
            score['name'] = name
            results.append(score)
        except Exception as e:
            errors.append(f'{code} ({name}): {str(e)[:50]}')

    # ─── 输出结果 ───
    results.sort(key=lambda x: x['total'], reverse=True)

    print(f'\n{"─"*65}')
    print(f'{"代码":>6} {"名称":<10} {"总分":>5} {"趋":>4} {"动":>4} {"MACD":>5} {"偏离":>4} {"RSI":>4} {"位":>4} {"价格":>8} {"MA20%":>6} {"RSI":>5} {"52w%":>5} {"20日均":>6} {"波动":>5}')
    print(f'{"─"*65}')

    for r in results:
        flag = ''
        if r['total'] >= 80:
            flag = '🟢💪'  # 强势买入
        elif r['total'] >= 60:
            flag = '🟢'    # 买入
        elif r['total'] >= 40:
            flag = '🟡'    # 观望
        else:
            flag = '🔴'    # 规避
        print(f'{r["code"]:>6} {r["name"]:<10} {r["total"]:>5} {r["trend"]:>4} {r["momentum"]:>4} {r["macd"]:>5} {r["ma_dev"]:>4} {r["rsi_score"]:>4} {r["p52_score"]:>4} {r["price"]:>8.2f} {r["ma20_dev_pct"]:>6.1f} {r["rsi"]:>5.1f} {r["p52"]:>5.0f} {r["avg_20d"]:>6.1f} {r["std_20d"]:>5.1f}')

    print(f'{"─"*65}')

    # 分类统计
    buy = [r for r in results if r['total'] >= 60]
    strong = [r for r in results if r['total'] >= 80]
    watch = [r for r in results if 40 <= r['total'] < 60]
    danger = [r for r in results if r['total'] < 40]

    # 精选核心大盘龙头
    MEGA_CORE = ['NVDA','AAPL','MSFT','GOOGL','GOOG','AMZN','META','TSLA','AVGO',
                 'JPM','V','WMT','JNJ','KO','XOM','LLY','ORCL','HD','PG',
                 'PG','COST','BAC','MA','DIS','NFLX']
    core_results = [r for r in results if r['code'] in MEGA_CORE]

    print(f'\n📊 汇总 ({len(results)}/{len(stock_map)})  蓝盾线: ≥60买入')
    print(f'  🟢💪 强势买入(≥80): {len(strong)}只')
    print(f'  🟢 买入(60-79): {len(buy)-len(strong)}只')
    print(f'  🟡 观望(40-59): {len(watch)}只')
    print(f'  🔴 规避(<40): {len(danger)}只')
    
    print(f'\n  🏆 核心大盘排序 (评级+评分/RSI/52w):')
    core_results.sort(key=lambda x: x['total'], reverse=True)
    for r in core_results:
        tag = '🟢💪' if r['total']>=80 else '🟢' if r['total']>=60 else '🟡' if r['total']>=40 else '🔴'
        print(f'    {tag} {r["code"]:>6} {r["name"]:<6}  {r["total"]:3d}分  RSI{r["rsi"]:5.1f}  52w{r["p52"]:3.0f}%')

    print(f'\n📊 稳定性评估')
    avg_std = sum(r['std_20d'] for r in results)/len(results) if results else 0
    if avg_std < 8:
        print(f'  20日评分平均波动: {avg_std:.1f} ✅ 稳定')
    else:
        print(f'  20日评分平均波动: {avg_std:.1f} ⚠️ 中等')

    if errors:
        print(f'\n⚠️ 失败{len(errors)}只:')
        for e in errors[:5]:
            print(f'  {e}')

    # 保存结果
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pool_size': len(stock_map),
        'scored': len(results),
        'buy': [{'code': r['code'], 'name': r['name'], 'score': r['total'],
                 'price': r['price'], 'rsi': r['rsi'], 'p52': r['p52']}
                for r in buy],
        'strong_buy': [{'code': r['code'], 'name': r['name'], 'score': r['total'],
                        'price': r['price'], 'rsi': r['rsi'], 'p52': r['p52']}
                       for r in strong],
        'watch': [{'code': r['code'], 'name': r['name'], 'score': r['total']} for r in watch],
        'danger': [{'code': r['code'], 'name': r['name'], 'score': r['total']} for r in danger],
        'all_scores': [{'code': r['code'], 'score': r['total'], 'trend': r['trend'],
                        'momentum': r['momentum'], 'macd': r['macd'], 'avg20d': r['avg_20d'],
                        'std20d': r['std_20d'], 'price': r['price'], 'rsi': r['rsi'], 'p52': r['p52']}
                       for r in results],
    }
    out_path = f'/home/hermes/.hermes/openclaw-project/data/ld2_scored_{time.strftime("%Y-%m-%d")}.json'
    json.dump(output, open(out_path, 'w'), indent=2, ensure_ascii=False)
    print(f'\n💾 保存: {out_path}')
    print(f'⏱️  耗时: {time.time()-T0:.1f}s')
    print('='*65)


if __name__ == '__main__':
    run()
