#!/usr/bin/env python3
"""
S&P 500 全量下载 + 质量因子筛选
分阶段:
  1) 下载S&P 500成分股列表
  2) 分批下载日线数据(2014-2026)
  3) 下载财务数据(ROE/毛利率)
  4) 质量评分筛选
"""
import json, os, sys, time

CACHE = "/home/admin/.openclaw/workspace/data/cache"
OUTPUT = "/home/admin/.openclaw/workspace/data/sp500_universe.json"
WORKSPACE = "/home/admin/.openclaw/workspace"

def log(msg):
    print(f"[{time.strftime('%H:%M')}] {msg}", flush=True)

# 步1: 获取S&P 500成分股
log("步1: 获取S&P 500成分股...")
try:
    import yfinance as yf
    sp500 = yf.Ticker("^GSPC")  # 用这个先占位
except: pass

# 使用已知的S&P 500列表来源: Wikipedia或静态列表
# 从之前我们的83只扩展，用爬虫获取最新列表
import urllib.request
import re

try:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=15)
    html = resp.read().decode('utf-8')
    
    # 提取表格中的股票代码
    symbols = re.findall(r'<td><a[^>]*>([A-Z]+)</a></td>', html)
    # 去重（第一个td通常是ticker）
    tickers = list(dict.fromkeys(symbols))
    log(f"  获取到 {len(tickers)} 只")
except Exception as e:
    log(f"  Wikipedia获取失败: {e}")
    # 降级: 使用已有的83只 + 手动补一些
    tickers = None

if not tickers or len(tickers) < 400:
    # 从已有缓存中读取
    cached = [f.replace('.json','') for f in os.listdir(CACHE) if f != 'spy.json']
    tickers = cached + ['MMM','A','AAL','AA','ACN','ADBE','ADI','ADM','ADP','AEE','AEP','AES',
        'AFL','AGN','AIG','AIV','AIZ','AJG','ALK','ALL','ALLE','ALXN','AMAT','AME','AMGN',
        'AMP','AMT','ANET','ANSS','AON','AOS','APA','APD','APH','APO','APTV','ARE','ARES',
        'ARM','ARW','ASGN','ASH','ATI','ATO','ATVI','AVB','AVGO','AVY','AWK','AXON','AZO',
        'AZPN','BA','BAC','BAH','BALL','BAX','BBY','BDX','BEN','BERY','BF-B','BIIB','BIO',
        'BK','BKNG','BKR','BLDR','BLK','BLL','BMY','BR','BRK-B','BRO','BSX','BWA','BXP',
        'C','CAG','CAH','CARR','CAT','CB','CBOE','CBRE','CC','CCK','CCL','CDNS','CDW','CE',
        'CEG','CHD','CHRW','CHTR','CI','CINF','CL','CLX','CMCSA','CME','CMG','CMI','CMS',
        'CNA','CNC','CNP','COF','COG','COHU','COL','COO','COP','COST','COTY','CPB','CPRT',
        'CPT','CRL','CRM','CRWD','CSCO','CSGP','CSL','CSX','CTAS','CTLT','CTRA','CTSH',
        'CTVA','CVS','CVX','CZR','D','DAVA','DAL','DAR','DD','DDOG','DE','DECK','DELL',
        'DFS','DG','DGX','DHI','DHR','DIS','DISH','DKNG','DLR','DLTR','DOV','DOW','DPZ',
        'DRE','DRI','DT','DTE','DUK','DVA','DVN','DXCM','EA','EBAY','ECL','ED','EFX',
        'EG','EIX','EL','ELV','EMN','EMR','ENB','ENPH','EOG','EPAM','EQIX','EQR','ES',
        'ESS','ETN','ETR','EVRG','EW','EXC','EXPD','EXPE','EXR','F','FANG','FAST','FCX',
        'FDS','FDX','FE','FFIV','FICO','FIS','FISV','FITB','FIVE','FL','FMC','FNF','FOXA',
        'FRT','FSLR','FTNT','FTV','GD','GDDY','GE','GEHC','GEN','GILD','GIS','GL','GLW',
        'GM','GNRC','GOLD','GPC','GPN','GRMN','GRUB','GS','GWW','HAL','HAS','HBAN','HBI',
        'HCA','HCP','HD','HES','HIG','HII','HIW','HLT','HOG','HOLX','HON','HPE','HPQ',
        'HRL','HSIC','HST','HSY','HTZ','HUBB','HUBS','HUM','HWM','IAA','IBM','ICE','IDXX',
        'IEX','IFF','INCY','INFO','INTC','INTU','IP','IPG','IQV','IR','IRM','ISRG','IT',
        'ITW','IVZ','J','JBHT','JCI','JEF','JKHY','JKS','JMOM','JNPR','JPM','JWN','K',
        'KDP','KEY','KEYS','KHC','KIM','KKR','KLAC','KMB','KMI','KMX','KO','KR','KSS',
        'KSU','L','LDOS','LEA','LEG','LEN','LH','LHA','LHX','LII','LIN','LKQ','LLY',
        'LMT','LNC','LNT','LOW','LRCX','LUMN','LUV','LVS','LW','LYB','LYV','M','MA',
        'MAA','MAN','MAR','MAS','MASI','MCD','MCHP','MCK','MCO','MDLZ','MDT','MET','META',
        'MGM','MHK','MKC','MKTX','MLM','MMC','MMM','MNST','MO','MOH','MOS','MPC','MPWR',
        'MRK','MRNA','MRO','MS','MSCI','MSFT','MSI','MTB','MTCH','MTD','MU','MUSA','NCLH',
        'NDAQ','NDSN','NEE','NEM','NFLX','NI','NKE','NLOK','NLSN','NOC','NOV','NOW','NRG',
        'NSC','NTAP','NTNX','NU','NUE','NVDA','NVR','NWL','NWSA','NXPI','O','ODFL','OKE',
        'OLN','OMC','ON','ONEO','ORCL','ORLY','OXY','PACW','PARA','PATH','PAYC','PAYX',
        'PCAR','PCG','PCTY','PDCO','PEAK','PEG','PENN','PEP','PFE','PFG','PG','PGR',
        'PH','PHM','PINS','PKI','PLD','PLTR','PM','PNC','PNR','PNW','POOL','POST','PPG',
        'PPL','PRGO','PRU','PSA','PSX','PTC','PWR','PXD','PYPL','QCOM','QRVO','RCL','RDIV',
        'RE','REG','REGN','RF','RGEN','RHI','RJF','RL','RMD','ROK','ROL','ROP','ROST',
        'RPM','RRC','RS','RSG','RTX','RUN','RWT','RYAN','S','SBAC','SBUX','SCCO','SCHW',
        'SCI','SEE','SEIC','SEM','SHW','SIG','SIRI','SIVB','SJM','SKX','SLB','SLG','SLM',
        'SNA','SNAP','SNOW','SNPS','SO','SPG','SPGI','SPLK','SQ','SRCL','SRE','STE','STLD',
        'STT','STX','STZ','SWK','SWKS','SYY','T','TAP','TDG','TDY','TEAM','TECH','TEL',
        'TER','TFC','TFX','TGT','TJX','TMO','TMUS','TPR','TRGP','TROW','TRV','TSCO','TSLA',
        'TSN','TT','TTWO','TW','TWLO','TXN','TXT','TYL','UA','UAA','UAL','UBER','UDR',
        'UHS','ULTA','UNH','UNM','UNP','UPS','URI','USB','V','VLO','VMC','VNO','VRSK',
        'VRSN','VRTX','VSAT','VTR','VTRS','VZ','WAB','WAT','WBA','WBD','WCC','WDC','WEC',
        'WELL','WFC','WH','WHR','WLTW','WM','WMB','WMT','WRB','WRK','WST','WTW','WU','WY',
        'WYNN','XEC','XEL','XLNX','XOM','XRAY','XRX','XYL','YUM','ZBRA','ZEN','ZG','ZION',
        'ZMH','ZTS','ZWS']
    # 去重
    seen, tickers = set(), []
    for t in tickers:
        if t not in seen: seen.add(t); tickers.append(t)
    log(f"  使用降级池: {len(tickers)} 只")

save_list = {'source': 'sp500_wikipedia', 'count': len(tickers), 'date': time.strftime('%Y-%m-%d'),
             'tickers': tickers}
json.dump(save_list, open(f"{WORKSPACE}/data/sp500_list.json", 'w'))
log(f"  列表已保存")

# 步2: 下载日线数据
log("\n步2: 下载日线数据(分批)...")
existing = [f.replace('.json','') for f in os.listdir(CACHE) if f != 'spy.json' and 
            os.path.getsize(f"{CACHE}/{f}") > 100000]

to_download = [t for t in tickers if t not in existing]
log(f"  已有: {len(existing)}只  待下: {len(to_download)}只")

# 分批下载，每批10只
batch_size = 10
success = 0
for i in range(0, len(to_download), batch_size):
    batch = to_download[i:i+batch_size]
    for t in batch:
        try:
            h = yf.download(t, start="2014-01-01", end="2026-05-18", progress=False)
            if h is not None and len(h) > 200:
                records = []
                for idx in range(len(h)):
                    row = h.iloc[idx]
                    records.append({'date': str(h.index[idx].date()),
                        'close': float(row.iloc[3])})
                json.dump({'data': records}, open(f"{CACHE}/{t}.json", 'w'))
                success += 1
            else:
                log(f"  {t}: 数据不足({len(h) if h is not None else 0})")
        except Exception as e:
            log(f"  {t}: {str(e)[:40]}")
        time.sleep(0.3)  # 礼貌等待
    
    log(f"  批次{i//batch_size+1}: {len(batch)}只完成 (累计成功{success}/{len(to_download)})")
    
    # 每50只保存一次进度
    if (i + batch_size) % 50 == 0:
        progress = {'downloaded': len(existing) + success, 'total': len(existing) + len(to_download),
                   'remaining': len(to_download) - success}
        json.dump(progress, open(f"{WORKSPACE}/data/sp500_progress.json", 'w'))

log(f"\n日线下载完毕: 新增{success}只")

# 步3: 下载财务数据
log("\n步3: 下载财务数据...")
fin_data = {}
all_cached = [f.replace('.json','') for f in os.listdir(CACHE) if f != 'spy.json' and 
              os.path.getsize(f"{CACHE}/{f}") > 100000]

fin_file = f"{WORKSPACE}/data/sp500_financials.json"
if os.path.exists(fin_file):
    fin_data = json.load(open(fin_file))
    log(f"  已有{len(fin_data)}只财务数据")

to_fetch = [t for t in all_cached if t not in fin_data]
log(f"  待获取: {len(to_fetch)}只")

for i, t in enumerate(to_fetch):
    try:
        tk = yf.Ticker(t)
        info = tk.info
        fin_data[t] = {
            'roe': info.get('returnOnEquity', None),
            'gross_margin': info.get('grossMargins', None),
            'pe': info.get('trailingPE', None),
            'pb': info.get('priceToBook', None),
            'debt_equity': info.get('debtToEquity', None),
            'revenue_growth': info.get('revenueGrowth', None),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'market_cap': info.get('marketCap', 0),
        }
    except:
        pass
    
    if (i+1) % 20 == 0:
        json.dump(fin_data, open(fin_file, 'w'))
        log(f"  财务: {i+1}/{len(to_fetch)}")

json.dump(fin_data, open(fin_file, 'w'))

# 步4: 质量评分筛选
log("\n步4: 质量评分筛选...")
QUALIFIED = []
for t, f in fin_data.items():
    score = 0
    if f.get('roe') and f['roe'] > 0.15: score += 30  # ROE>15%
    elif f.get('roe') and f['roe'] > 0.10: score += 20
    elif f.get('roe') and f['roe'] > 0.05: score += 10
    
    if f.get('gross_margin') and f['gross_margin'] > 0.40: score += 25
    elif f.get('gross_margin') and f['gross_margin'] > 0.30: score += 15
    
    if f.get('market_cap', 0) > 1e10: score += 25  # 市值>100亿
    elif f.get('market_cap', 0) > 5e9: score += 15
    
    if f.get('debt_equity') and f['debt_equity'] < 1.0: score += 10
    if f.get('pe') and 0 < f['pe'] < 40: score += 10
    
    QUALIFIED.append({'ticker': t, 'quality_score': score, 'sector': f.get('sector',''),
                     'industry': f.get('industry','')})

QUALIFIED.sort(key=lambda x: x['quality_score'], reverse=True)

# 保留质量评分≥60的
final_pool = [q for q in QUALIFIED if q['quality_score'] >= 60]
log(f"  总筛选: {len(QUALIFIED)}只")
log(f"  合格池: {len(final_pool)}只")

# 保存最终池
result = {
    'date': time.strftime('%Y-%m-%d'),
    'total_screened': len(QUALIFIED),
    'qualified_count': len(final_pool),
    'quality_threshold': 60,
    'criteria': {'ROE>15%': 30, '毛利率>40%': 25, '市值>100亿': 25, 
                 '负债率<1': 10, 'PE在0-40': 10, '满分': 100},
    'pool': final_pool,
    'tickers': [q['ticker'] for q in final_pool]
}
json.dump(result, open(OUTPUT, 'w'), indent=2)

log(f"\n✅ 完成! 最终候选池: {len(final_pool)}只 → {OUTPUT}")
log(f"SPY_DATES等数据在 {CACHE}/spy.json")
