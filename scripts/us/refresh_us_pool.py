#!/usr/bin/env python3
"""
🍤 美股质量池刷新 — 每天收盘后自动更新评分排名

用minishare获取实时行情（50只一批，几秒扫完）
用yfinance获取日K线跑V4.2评分
保存到 data/us_scored.json
"""
import sys, os, json, time, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# S&P 500头部+关注(500+只)
TICKERS = [
    'AAPL','MSFT','AMZN','NVDA','GOOGL','META','TSLA','AVGO','JPM','V',
    'JNJ','WMT','MA','PG','XOM','UNH','CVX','HD','MRK','BAC','PEP',
    'ABBV','KO','COST','ADBE','CRM','NFLX','TMO','PFE','ABT','ACN','DHR',
    'WFC','NKE','DIS','LIN','TXN','PM','QCOM','IBM','SPGI','UPS','CAT',
    'LOW','RTX','BA','MDT','GS','SCHW','HON','AMD','DE','AXP','TMUS',
    'C','BLK','LMT','NOW','BKNG','SYK','ELV','PLD','MDLZ','GILD','ADP',
    'ISRG','MMC','CL','ZTS','WM','EOG','APD','MS','CB','NOC','CI',
    'MO','GD','ITW','GE','SHW','MCO','AMAT','SBUX','BDX','TGT','REGN',
    'PGR','HCA','ADI','CME','ATVI','MU','CSCO','INTC','NOK','ZS','SNOW',
    'TEAM','PANW','ORCL','CRWD','DDOG','ENPH','SNDK','STX','FTNT','ON',
    'HUM','CNC','TWLO','APP','ARES','BXP','FICO','GPN','BAX','DXCM',
    'DKNG','KDP','PAYX','ZBRA','SMCI','MSCI','ANET','APH','AXON','COIN',
    'CPRT','DASH','DDOG','DLTR','DOCU','EA','EBAY','EXPE','F','FANG',
    'FAST','FCX','FISV','FSLR','FTV','GDDY','GEN','GRMN','HES','HLT',
    'HPE','HPQ','HST','IBM','IDXX','INCY','IR','JBL','JCI','JNPR',
    'KEYS','KHC','KMI','KMX','L','LEN','LHX','LII','LNC','LNT',
    'LRCX','LULU','LUV','LVS','MAA','MANH','MAR','MAS','MCD','MCHP',
    'MCK','MELI','MET','MGM','MKC','MKTX','MLM','MMM','MNST','MOH',
    'MOS','MPC','MPWR','MRNA','MRO','MSCI','MSI','MTB','MTCH','MTD',
    'NCLH','NDAQ','NDSN','NEE','NEM','NFLX','NI','NKE','NOC','NOW',
    'NRG','NSC','NTAP','NTRS','NUE','NVR','NWSA','NXPI','ODFL','OKE',
    'OMC','ON','ORLY','OTIS','OXY','PARA','PAYX','PCAR','PCG','PEG',
    'PFE','PFG','PG','PGR','PH','PHM','PINS','PKG','PLD','PLTR',
    'PM','PNC','PNR','PNW','PODD','POOL','PPG','PPL','PRU','PSA',
    'PSX','PTC','PWR','PYPL','QRVO','RCL','REG','REGN','RF','RJF',
    'RL','RMD','ROK','ROL','ROP','ROST','RPM','RS','RSG','RTX',
    'SBUX','SCHW','SHW','SJM','SLB','SMCI','SNA','SNAP','SNPS','SO',
    'SPG','SPGI','SRE','STE','STLD','STT','STX','STZ','SWK','SWKS',
    'SYF','SYK','SYY','T','TAP','TDG','TDY','TECH','TEL','TER',
    'TFC','TFX','TGT','TJX','TMO','TMUS','TPL','TRGP','TRMB','TROW',
    'TRV','TSCO','TSLA','TSN','TT','TTWO','TWLO','TYL','UAL','UBER',
    'UDR','UHS','ULTA','UNH','UNP','UPS','URI','USB','V','VICI',
    'VLO','VMC','VRSK','VRSN','VRT','VRTX','VST','VTR','VZ','WAB',
    'WAT','WBA','WBD','WDC','WEC','WELL','WFC','WMB','WMS','WMT',
    'WY','XEL','XOM','XYL','YUM','ZBH','ZBRA','ZTS'
]

def get_score_v42(close):
    """V4.2简版评分：30日动量 - 距离52周高位扣分"""
    if len(close) < 30:
        return 0, 0
    cur = close[-1]
    mom30 = (cur - close[-31]) / close[-31] * 100 if len(close) > 31 else 0
    h52 = max(close[-252:]) if len(close) >= 252 else max(close)
    dist = (h52 - cur) / h52 * 100 if h52 else 0
    
    ds, dc = 40, 0.7
    deduction = max(0, (dist - ds) * dc) if dist > ds else 0
    score = round(mom30 - deduction, 1)
    return score, round(mom30, 1)

if __name__ == '__main__':
    print(f'📊 美股质量池刷新 · {len(TICKERS)}只')
    
    import yfinance as yf
    results = []
    errors = 0
    
    for i, t in enumerate(TICKERS):
        try:
            d = yf.download(t, period='1y', interval='1d', progress=False)
            if len(d) < 30:
                errors += 1
                continue
            close = list(d['Close'].values.flatten())
            cur = close[-1]
            if cur < 3:
                errors += 1
                continue
            
            score, mom30 = get_score_v42(close)
            ma20 = sum(close[-20:]) / 20
            h52 = max(close[-252:]) if len(close) >= 252 else max(close)
            l52 = min(close[-252:]) if len(close) >= 252 else min(close)
            pos52 = (cur - l52) / (h52 - l52) * 100 if h52 != l52 else 50
            
            results.append({
                'ticker': t,
                'score': score,
                'price': round(cur, 1),
                'ma20': round(ma20, 1),
                'pos52': round(pos52, 0),
                'mom30': round(mom30, 1),
                'updated': datetime.datetime.now().strftime('%m-%d %H:%M')
            })
            
            if (i+1) % 100 == 0:
                print(f'  进度: {i+1}/{len(TICKERS)} | 有效: {len(results)}')
        except:
            errors += 1
            continue
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    with open(os.path.join(ROOT, 'data', 'us_scored.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n完成: {len(results)}只有效 / {len(TICKERS)}只总')
    print(f'跳过: {errors}只')
    print()
    print('=== Top 10 ===')
    for r in results[:10]:
        em = '🟢' if r['score'] >= 20 else ('🟡' if r['score'] >= 0 else '🔴')
        print(f'  {em} {r["ticker"]}: {r["score"]}分  ${r["price"]}  {r["pos52"]}%位')

# 审计记录
try:
    from audit_engine import audit
    audit('refresh_us_pool', 'success', '美股质量池刷新完成')
except:
    pass
