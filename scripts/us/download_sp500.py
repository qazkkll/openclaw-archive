#!/usr/bin/env python3
"""S&P 500 数据补充下载 — 只补新ticker，不重复下已有的"""
import json, os, time, sys, warnings
warnings.filterwarnings('ignore')

CACHE = "/home/admin/.openclaw/workspace/data/cache"
os.makedirs(CACHE, exist_ok=True)

# 硬编码S&P 500列表（从Wikipedia获取失败时的备用）
ALL_TICKERS = sorted(list(set([
    'MMM','A','AAL','AAP','AAPL','ABBV','ABT','ACN','ADBE','ADI','ADM','ADP','AEE','AEP',
    'AES','AFL','AIG','AIV','AIZ','AJG','AKAM','ALK','ALL','ALLE','ALGN','ALXN','AMAT',
    'AMD','AME','AMGN','AMP','AMT','AMZN','ANET','ANSS','AON','AOS','APA','APD','APH',
    'APO','APTV','ARE','ARM','ATO','ATVI','AVB','AVGO','AVY','AWK','AXP','AZO','AZPN',
    'BA','BAC','BAH','BALL','BAX','BBY','BDX','BEN','BF-B','BIIB','BIO','BK','BKNG',
    'BKR','BLDR','BLK','BLL','BMY','BR','BRK-B','BRO','BSX','BWA','BXP','C','CAG',
    'CAH','CARR','CAT','CB','CBOE','CBRE','CC','CCK','CCL','CDNS','CDW','CE','CEG',
    'CHD','CHRW','CHTR','CI','CINF','CL','CLX','CMCSA','CME','CMG','CMI','CMS','CNA',
    'CNC','CNP','COF','COG','COO','COP','COST','COTY','CPB','CPRT','CPT','CRL','CRM',
    'CRWD','CSCO','CSGP','CSL','CSX','CTAS','CTLT','CTRA','CTSH','CTVA','CVS','CVX',
    'D','DAL','DD','DDOG','DE','DECK','DELL','DFS','DG','DGX','DHI','DHR','DIS','DISH',
    'DKNG','DLR','DLTR','DOV','DOW','DPZ','DRE','DRI','DT','DTE','DUK','DVA','DVN',
    'DXCM','EA','EBAY','ECL','ED','EFX','EG','EIX','EL','ELV','EMN','EMR','ENPH','EOG',
    'EPAM','EQIX','EQR','ES','ESS','ETN','ETR','EVRG','EW','EXC','EXPD','EXPE','EXR',
    'F','FANG','FAST','FCX','FDS','FDX','FE','FFIV','FICO','FIS','FISV','FITB','FIVE',
    'FL','FMC','FNF','FOXA','FRT','FSLR','FTNT','FTV','GD','GDDY','GE','GEHC','GEN',
    'GILD','GIS','GL','GLW','GM','GNRC','GPC','GPN','GRMN','GS','GWW','HAL','HAS',
    'HBAN','HCA','HD','HES','HIG','HII','HLT','HOG','HOLX','HON','HPE','HPQ','HRL',
    'HSIC','HST','HSY','HUBB','HUBS','HUM','HWM','IAA','IBM','ICE','IDXX','IEX','IFF',
    'INCY','INFO','INTC','INTU','IP','IPG','IQV','IR','IRM','ISRG','IT','ITW','IVZ',
    'J','JBHT','JCI','JKHY','JNPR','JPM','K','KDP','KEY','KEYS','KHC','KIM','KKR',
    'KLAC','KMB','KMI','KMX','KO','KR','L','LDOS','LEA','LEG','LEN','LH','LHX','LII',
    'LIN','LKQ','LLY','LMT','LNC','LNT','LOW','LRCX','LUV','LVS','LW','LYB','MA','MAA',
    'MAN','MAR','MAS','MASI','MCD','MCHP','MCK','MCO','MDLZ','MDT','MET','META','MGM',
    'MHK','MKC','MKTX','MLM','MMC','MNST','MO','MOH','MOS','MPC','MPWR','MRK','MRO',
    'MS','MSCI','MSFT','MSI','MTB','MTCH','MTD','MU','NCLH','NDAQ','NDSN','NEE','NEM',
    'NFLX','NI','NKE','NOC','NOV','NOW','NRG','NSC','NTAP','NU','NUE','NVDA','NVR',
    'NWL','NWSA','NXPI','O','ODFL','OKE','OLN','OMC','ON','ORCL','ORLY','OXY','PARA',
    'PAYC','PAYX','PCAR','PCG','PEG','PEP','PFE','PFG','PG','PGR','PH','PHM','PINS',
    'PKI','PLD','PLTR','PM','PNC','PNR','PNW','POOL','PPG','PPL','PRU','PSA','PSX',
    'PTC','PWR','PXD','PYPL','QCOM','QRVO','RCL','RE','REG','REGN','RF','RHI','RJF',
    'RL','RMD','ROK','ROL','ROP','ROST','RPM','RRC','RS','RSG','RTX','SBAC','SBUX',
    'SCCO','SCHW','SCI','SEE','SHW','SJM','SKX','SLB','SLG','SNA','SNAP','SNOW','SNPS',
    'SO','SPG','SPGI','SRE','STE','STLD','STT','STX','STZ','SWK','SWKS','SYY','T',
    'TAP','TDG','TDY','TEAM','TECH','TEL','TER','TFC','TFX','TGT','TJX','TMO','TMUS',
    'TPR','TRGP','TROW','TRV','TSCO','TSLA','TSN','TT','TTWO','TXN','TXT','TYL','UA',
    'UAA','UAL','UBER','UDR','UHS','ULTA','UNH','UNM','UNP','UPS','URI','USB','V',
    'VLO','VMC','VNO','VRSK','VRSN','VRTX','VTR','VTRS','VZ','WAB','WAT','WBA','WBD',
    'WCC','WDC','WEC','WELL','WFC','WH','WHR','WM','WMB','WMT','WRB','WRK','WST','WTW',
    'WU','WY','WYNN','XEL','XOM','XRAY','XRX','XYL','YUM','ZBRA','ZION','ZMH','ZTS',
    # 已有的83只
    'AAPL','AMAT','AMD','AMZN','AVGO','BA','CAT','COST','CRM','DIS','GE','GOOGL','HD',
    'INTC','JPM','LLY','MA','META','MSFT','MU','NFLX','NVDA','ORCL','QCOM','TSLA','UNH',
    'V','WMT','ABBV','ABT','ADP','ADSK','AMGN','BRK-B','BSX','C','CB','CL','CMCSA','CME',
    'COF','COP','CSCO','CVX','DE','DHR','ELV','ETN','GILD','HON','IBM','ICE','INTU',
    'KO','LOW','MCD','MDLZ','MDT','MRK','MS','NEE','NOC','NOW','PANW','PEP','PG','PGR',
    'REGN','RTX','SCHW','SHW','SO','SPGI','T','TMO','TMUS','TXN','UBER','USB','VRTX',
    'VZ','WFC','XOM','ZTS'
])))

existing = [f.replace('.json','') for f in os.listdir(CACHE) if f != 'spy.json' and 
            os.path.getsize(f"{CACHE}/{f}") > 100000]
to_down = [t for t in ALL_TICKERS if t not in existing]

print(f"已有: {len(existing)}只  待下: {len(to_down)}只", flush=True)

import yfinance as yf
success = 0
fail = 0
batch_size = 20

for i in range(0, len(to_down), batch_size):
    batch = to_down[i:i+batch_size]
    for t in batch:
        try:
            h = yf.download(t, start="2014-01-01", end="2026-05-18", progress=False)
            if h is not None and len(h) > 200:
                records = []
                for idx in range(len(h)):
                    row = h.iloc[idx]
                    d = str(h.index[idx].date()) if hasattr(h.index[idx],'date') else str(h.index[idx])[:10]
                    records.append({'date': d, 'close': float(row.iloc[3])})
                json.dump({'data': records}, open(f"{CACHE}/{t}.json", 'w'))
                success += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
    
    print(f"  批次{i//batch_size+1}/{(len(to_down)-1)//batch_size+1}: +{success} ✓ / {fail} ✗", flush=True)
    
    # 每100只保存进度
    if (i + batch_size) % 100 == 0:
        json.dump({'done': len(existing)+success, 'total': len(ALL_TICKERS), 'remain': len(to_down)-success},
                  open(f"{CACHE}/../data/sp500_progress.json", 'w'))

total = len(existing) + success
print(f"\n✅ 完成: 共{total}/{len(ALL_TICKERS)}只", flush=True)
