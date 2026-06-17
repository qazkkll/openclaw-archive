#!/usr/bin/env python3
"""
美股V4.2扫描 → 生成 dashboard JSON
"""
import json, os, sys
from datetime import datetime, timezone
import yfinance as yf

WORKSPACE = "/home/admin/.openclaw/workspace"
OUTPUT_FILE = f"{WORKSPACE}/data/us_scan_result.json"
EXTRA_TICKERS = ['ZS', 'NET', 'LITE']

QUALITY_POOL_FILE = f"{WORKSPACE}/data/sp500_universe.json"
if os.path.exists(QUALITY_POOL_FILE):
    with open(QUALITY_POOL_FILE) as f:
        data = json.load(f)
        CANDIDATES = data.get('tickers', []) + EXTRA_TICKERS
else:
    CANDIDATES = ['NVDA','AMD','MU','AVGO','QCOM','AMAT','MSFT','CRM',
        'GOOGL','AMZN','META','HD','COST','JPM','V','MA','UNH','LLY',
        'AAPL','NFLX','ORCL','CAT','GE','WMT','DIS','BA','XOM','CVX',
        'KO','PEP','MCD','ABBV','MRK','TMO','TXN','LOW','NEE','SPGI'] + EXTRA_TICKERS

DS, DC, MD = 40, 0.7, 30

SECTOR_MAP = {
    'NVDA': '半导体', 'AMD': '半导体', 'MU': '半导体', 'AVGO': '半导体',
    'QCOM': '半导体', 'AMAT': '半导体', 'TXN': '半导体', 'INTC': '半导体',
    'MRVL': '半导体', 'ON': '半导体', 'STM': '半导体', 'NXPI': '半导体',
    'KLAC': '半导体', 'LRCX': '半导体', 'MCHP': '半导体',
    'AAPL': '科技消费品', 'MSFT': '科技软件', 'CRM': '科技软件',
    'ORCL': '科技软件', 'ADBE': '科技软件', 'NOW': '科技软件',
    'INTU': '科技软件', 'PANW': '科技软件', 'SNPS': '科技软件',
    'CDNS': '科技软件', 'ROP': '科技软件', 'FTNT': '科技软件',
    'META': '互联网', 'GOOGL': '互联网', 'AMZN': '互联网',
    'NFLX': '互联网', 'SNAP': '互联网', 'PINS': '互联网',
    'HD': '消费零售', 'COST': '消费零售', 'WMT': '消费零售',
    'LOW': '消费零售', 'TGT': '消费零售', 'AMGN': '医药',
    'UNH': '医药', 'LLY': '医药', 'ABBV': '医药', 'MRK': '医药',
    'TMO': '医药', 'JNJ': '医药', 'PFE': '医药',
    'JPM': '金融', 'V': '金融', 'MA': '金融', 'GS': '金融',
    'MS': '金融', 'BRK.B': '金融', 'AXP': '金融', 'BAC': '金融',
    'WFC': '金融', 'C': '金融', 'SCHW': '金融', 'BLK': '金融',
    'CAT': '工业制造', 'GE': '工业制造', 'BA': '工业制造',
    'HON': '工业制造', 'MMM': '工业制造', 'UPS': '工业制造',
    'RTX': '工业制造', 'DE': '工业制造',
    'XOM': '能源', 'CVX': '能源', 'COP': '能源', 'EOG': '能源',
    'SLB': '能源', 'OXY': '能源',
    'KO': '消费必需', 'PEP': '消费必需', 'MCD': '消费必需',
    'PG': '消费必需', 'COST': '消费必需',
    'NEE': '公用事业', 'DUK': '公用事业', 'SO': '公用事业',
    'DIS': '媒体娱乐', 'TTWO': '游戏', 'EA': '游戏',
    'ZS': '科技软件', 'NET': '科技软件', 'LITE': '科技软件',
}

def load_portfolio():
    try:
        pf = json.load(open(f"{WORKSPACE}/data/portfolio.json"))
        return [(s['code'], s.get('name', s['code'])) for s in pf.get('us_stock', [])]
    except:
        return []

def calc_v42_score(closes, cp):
    if len(closes) < 60:
        return {'score': 0, 'mom30': 0, 'p52': 100}
    mom30 = (cp / closes[-(MD+1)] - 1) * 100 if len(closes) >= MD + 1 else (cp / closes[0] - 1) * 100
    hp, lp = max(closes[-252:]), min(closes[-252:])
    p52 = ((cp - lp) / (hp - lp)) * 100 if hp > lp else 50
    deduction = max(0, (p52 - DS) / 60 * DC)
    score = mom30 * (1 - min(deduction, 1))
    return {'score': round(score, 1), 'mom30': round(mom30, 1), 'p52': round(p52, 1)}

def get_signal(rank, score, is_holding):
    if rank is not None:
        if rank <= 5:     return '🟢', '加仓'
        elif rank <= 10:  return '🔵', '关注'
        elif is_holding:
            if rank <= 15: return '🟡', '持有'
            else:          return '🟠', '警惕'
        else:             return '🟡', '观望'
    if score < 0: return '🔴', '卖出'
    if score < 10: return '🟠', '警惕'
    return '🟡', '观望'

def main():
    print(f"美股V4.2扫描...")
    portfolio = load_portfolio()
    portfolio_codes = [p[0] for p in portfolio]

    # 下载SPY判断大盘
    sp_state = "bull"
    sp_price, sp_ma200 = 0, 0
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="1y")
        sp_price = spy_hist['Close'].iloc[-1]
        sp_ma200 = spy_hist['Close'].rolling(200).mean().iloc[-1]
        sp_state = "bull" if sp_price > sp_ma200 else "bear"
    except:
        pass

    pool = [{'code': c, 'name': c} for c in CANDIDATES if c not in portfolio_codes]
    for code, name in portfolio:
        pool.append({'code': code, 'name': name})
    pool.sort(key=lambda p: (0 if p['code'] in portfolio_codes else 1))

    results = []
    total = len(pool)
    for idx, item in enumerate(pool):
        code = item['code']
        sys.stdout.write(f"\r  📡 {idx+1}/{total}")
        sys.stdout.flush()
        try:
            ticker = yf.Ticker(code)
            hist = ticker.history(period="6mo")
            if len(hist) < 60:
                continue
            closes = [float(x) for x in hist['Close'].tolist()]
            info = ticker.info
            rp = info.get("regularMarketPrice") or info.get("currentPrice") or closes[-1]
            prev_close = info.get("previousClose")
            chg = ((rp - prev_close) / prev_close * 100) if prev_close else 0
            name = info.get("shortName") or info.get("longName") or code
            sector = SECTOR_MAP.get(code) or info.get("sector", "其他") or "其他"
            result = calc_v42_score(closes, rp)
            results.append({
                'code': code, 'name': str(name).split('(')[0].strip(),
                'score': result['score'], 'mom30': result['mom30'],
                'p52': result['p52'], 'price': round(rp, 2),
                'chg': round(chg, 2), 'sector': sector,
                'is_holding': code in portfolio_codes
            })
        except:
            pass

    print()

    results.sort(key=lambda x: x['score'], reverse=True)
    for i, s in enumerate(results):
        s['rank'] = i + 1
        sig_em, sig_label = get_signal(s['rank'], s['score'], s['is_holding'])
        s['signal'] = sig_em
        s['signal_label'] = sig_label

    top20 = results[:20]
    holdings = [s for s in results if s['is_holding']]

    output = {
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "market": {
            "state": sp_state,
            "sp_price": round(sp_price, 2),
            "sp_ma200": round(sp_ma200, 2),
            "total_scored": len(results)
        },
        "top20": top20,
        "holdings": holdings
    }

    json.dump(output, open(OUTPUT_FILE, "w"), ensure_ascii=False, indent=2)
    print(f"✅ 扫描完成 ({len(results)}只有效)")
    print(f"📊 SPY={sp_price:.2f} MA200={sp_ma200:.2f} → {'🟢牛市' if sp_state=='bull' else '🔴熊市'}")
    print(f"🏆 Top3: {top20[0]['code']}({top20[0]['score']}) {top20[1]['code']}({top20[1]['score']}) {top20[2]['code']}({top20[2]['score']})")
    if holdings:
        for h in holdings:
            print(f"  {'✅' if h['rank']<=10 else '⚠️'} {h['code']:6s} #{h['rank']:2d}  ${h['price']}  {h['signal']} {h['mom30']:+.1f}%30d {h['p52']:.0f}%52w")

if __name__ == "__main__":
    main()
