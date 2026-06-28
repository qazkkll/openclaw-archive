#!/usr/bin/env python3
"""
🦅 Falcon 10年回测 (2016-2026)
Top5 vs Top20 × 有/无行业分散, 476只SPX
"""
import sys, json, time
sys.path.insert(0, 'scripts/falcon')
from pathlib import Path
import pandas as pd, numpy as np

DATA_DIR = Path('data/falcon')
t0 = time.time()

# 行业分类(同上)
SECTOR_MAP = {}
for t in ['AAPL','MSFT','GOOG','GOOGL','META','AMZN','NFLX','CRM','ADBE','NOW','INTU','SNPS','CDNS','PLTR','APP','SNOW','CRWD','ZS','PANW','FTNT','DDOG','NET','MDB','TEAM','WDAY','VEEV','HUBS','PAYC','ZM','SHOP','SQ','PYPL','TTD','ABNB','UBER','DASH','DKNG','RBLX','DUOL','BILL','SPT','GTLB','BR','CDAY','PCTY','SSNC','TYL','EPAM','GLOB','ORCL','SAP','ADSK','ANSS','PTC','IQV']:
    SECTOR_MAP[t] = 'Software/AI'
for t in ['NVDA','AVGO','TXN','QCOM','AMD','INTC','MU','MRVL','ON','AMAT','LRCX','KLAC','ASML','NXPI','SWKS','MCHP','ADI','MPWR','ONTO','COHR','ENTG','CRUS','SLAB','POWI','ACLS','AMKR','TER','FORM','STX','WDC','MKSI']:
    SECTOR_MAP[t] = 'Semiconductor'
for t in ['LLY','JNJ','UNH','PFE','ABBV','MRK','TMO','ABT','DHR','BMY','AMGN','GILD','ISRG','MDT','SYK','BSX','ZTS','REGN','VRTX','BIIB','MRNA','HOLX','DXCM','IDXX','PODD','EW','BAX','ALGN','MEDP','DOCS','RPRX','UTHR','NBIX','IONS','RARE','SRPT']:
    SECTOR_MAP[t] = 'Healthcare'
for t in ['JPM','BAC','GS','MS','BLK','SCHW','C','AXP','USB','PNC','TFC','BK','STT','COF','DFS','MA','V','FIS','GPN','ADP','ICE','CME','SPGI','MCO','MSCI','FDS','CBOE','NDAQ','BRK.B','MET','PRU','AFL','ALL','TRV','PGR','CB','MMC','AON','WTW','BRO','BX','KKR','APO','CG','ARES','TROW','VRSN']:
    SECTOR_MAP[t] = 'Financial'
for t in ['PG','KO','PEP','WMT','COST','HD','LOW','MCD','SBUX','NKE','TGT','DG','DLTR','CL','EL','GIS','K','SYY','ADM','MNST','KDP','STZ','CMG','YUM','DPZ','QSR','ORLY','AZO','ROST','TJX','GPS','ANF','URBN','TSCO','BBY','WSM','RH','LEN','DHI','NVR','PHM']:
    SECTOR_MAP[t] = 'Consumer'
for t in ['XOM','CVX','COP','EOG','SLB','PSX','VLO','MPC','OXY','DVN','HAL','FANG','APA','CTRA','EQT','AR','SM','BKR','LBRT','HP','NOV','FTI','WMB','KMI','OKE','ET','NEE','DUK','SO','D','AEP','SRE','EXC','XEL','ED','WEC','CEG','VST','NRG','AES']:
    SECTOR_MAP[t] = 'Energy/Utility'
for t in ['CAT','DE','HON','UNP','CSX','UPS','FDX','GE','RTX','LMT','NOC','GD','BA','LHX','WM','RSG','ITW','ETN','EMR','ROK','PH','CARR','OTIS','JCI','IR','TT','GNRC','AME','FTV','HWM','NDSN','SWK','SNA','FAST','GWW','TPL']:
    SECTOR_MAP[t] = 'Industrial'
for t in ['DIS','CMCSA','T','VZ','TMUS','CHTR','PARA','WBD','LYV','MTCH','ROKU','TTWO','EA','SPOT','IPG','OMC']:
    SECTOR_MAP[t] = 'Media/Comm'
for t in ['LIN','APD','SHW','ECL','DD','PPG','NEM','FCX','NTR','MOS','ALB','VMC','MLM','CE','IFF','SEE','AVY','PKG','IP','STLD','NUE','RS','CLF','AA','CTVA']:
    SECTOR_MAP[t] = 'Material'
for t in ['AMT','PLD','CCI','EQIX','DLR','SPG','O','WELL','ARE','AVB','EQR','UDR','MAA','VTR','BXP','SLG','KIM','REG','VICI','GLPI','SBAC','PSA','EXR','RYN','WY']:
    SECTOR_MAP[t] = 'RealEstate'

def get_sector(t): return SECTOR_MAP.get(t, 'Other')

def select_stocks(sorted_scores, top_n, max_per_sector=2):
    sel, sc = [], {}
    for t in sorted_scores.index:
        s = get_sector(t)
        if sc.get(s, 0) < max_per_sector:
            sel.append(t); sc[s] = sc.get(s, 0) + 1
            if len(sel) >= top_n: break
    return sel

# 加载数据
print("📊 加载10年数据...")
master = pd.read_parquet(DATA_DIR / 'features_v02.parquet')
master['date'] = master['date'].astype(str)
print(f"   {master['ticker'].nunique()}只, {master['date'].min()} ~ {master['date'].max()}")

fmp = {}
for n, f in [('fmp_ratios_historical','fmp_ratios_historical.json'),('analyst_historical','analyst_historical.json'),('fmp_key_metrics','fmp_key_metrics.json'),('fmp_financial_growth','fmp_financial_growth.json')]:
    fmp[n] = json.load(open(DATA_DIR/f))

weights = {'fund_ratio': 0.70, 'analyst': 0.20, 'fund_metric': 0.10}
price_pivot = master.pivot_table(index='date', columns='ticker', values='close').sort_index()
all_dates = sorted(price_pivot.index)
rebalance_dates = [all_dates[i] for i in range(0, len(all_dates), 21)]
print(f"   {len(rebalance_dates)} 个调仓日")

# PIT查询
RATIO_FIELDS = ["priceToEarningsRatio","priceToBookRatio","priceToSalesRatio","priceToFreeCashFlowRatio","enterpriseValueMultiple","grossProfitMargin","netProfitMargin","operatingProfitMargin","ebitdaMargin","assetTurnover","inventoryTurnover","receivablesTurnover","debtToEquityRatio","currentRatio","quickRatio","financialLeverageRatio","freeCashFlowOperatingCashFlowRatio","operatingCashFlowRatio","dividendYieldPercentage","dividendPayoutRatio"]
METRIC_FIELDS = ["earningsYield","evToEBITDA","evToFreeCashFlow","evToSales","freeCashFlowYield","returnOnEquity","returnOnAssets","returnOnCapitalEmployed","returnOnInvestedCapital","returnOnTangibleAssets","incomeQuality","grahamNumber","cashConversionCycle","capexToRevenue","capexToDepreciation","researchAndDevelopementToRevenue","stockBasedCompensationToRevenue","netDebtToEBITDA","operatingReturnOnAssets"]
GROWTH_FIELDS = ["revenueGrowth","grossProfitGrowth","ebitgrowth","operatingIncomeGrowth","netIncomeGrowth","epsdilutedGrowth","freeCashFlowGrowth","tenYRevenueGrowthPerShare","fiveYRevenueGrowthPerShare","threeYRevenueGrowthPerShare","receivablesGrowth","inventoryGrowth","assetGrowth","bookValueperShareGrowth","debtGrowth"]
ANALYST_FIELDS = ["eps_revision","revenue_revision","eps_dispersion"]

def get_pit(data, date):
    if not data: return {}
    latest = {}
    for q in data:
        if isinstance(q, dict) and q.get("date", "") <= date:
            latest = q
    return latest

def compute_scores(date, tickers):
    r_fr, r_fm, r_fg, r_an = {}, {}, {}, {}
    for t in tickers:
        p = get_pit(fmp['fmp_ratios_historical'].get(t, []), date)
        r_fr[t] = np.mean([p.get(f) for f in RATIO_FIELDS if p.get(f) is not None]) if p else None
        p_m = get_pit(fmp['fmp_key_metrics'].get(t, []), date)
        r_fm[t] = np.mean([p_m.get(f) for f in METRIC_FIELDS if p_m.get(f) is not None]) if p_m else None
        p_g = get_pit(fmp['fmp_financial_growth'].get(t, []), date)
        r_fg[t] = np.mean([p_g.get(f) for f in GROWTH_FIELDS if p_g.get(f) is not None]) if p_g else None
        p_a = get_pit(fmp['analyst_historical'].get(t, []), date)
        r_an[t] = np.mean([p_a.get(f) for f in ANALYST_FIELDS if p_a.get(f) is not None]) if p_a else None
    
    df = pd.DataFrame({'fund_ratio': pd.Series(r_fr), 'fund_metric': pd.Series(r_fm),
                       'fund_growth': pd.Series(r_fg), 'analyst': pd.Series(r_an)})
    for col in df.columns:
        v = df[col].dropna()
        if len(v) > 5: df[col] = df[col].rank(pct=True)
    combined = sum(weights.get(f, 0) * df[f] for f in weights if f in df.columns)
    return combined.dropna().sort_values(ascending=False)

print("📊 计算调仓日scores...")
scores_dict = {}
for i, date in enumerate(rebalance_dates):
    if date not in price_pivot.index: continue
    scores_dict[date] = compute_scores(date, list(price_pivot.columns))
    if (i+1) % 20 == 0: print(f"   {i+1}/{len(rebalance_dates)}")
print(f"✅ {len(scores_dict)} 个调仓日")

def run_backtest(top_n, sector_limit=False, max_per_sector=2):
    results = []
    rebalance_list = sorted(scores_dict.keys())
    for idx, entry_date in enumerate(rebalance_list):
        exit_date = rebalance_list[idx+1] if idx+1 < len(rebalance_list) else all_dates[min(all_dates.index(entry_date)+21, len(all_dates)-1)]
        if entry_date not in price_pivot.index or exit_date not in price_pivot.index: continue
        ep = price_pivot.loc[entry_date]; xp = price_pivot.loc[exit_date]
        sorted_scores = scores_dict[entry_date]
        picks = select_stocks(sorted_scores, top_n, max_per_sector) if sector_limit else list(sorted_scores.head(top_n).index)
        rets = [(xp[t]-ep[t])/ep[t] for t in picks if t in ep.index and t in xp.index and not pd.isna(ep[t]) and not pd.isna(xp[t]) and ep[t]>0]
        spx = [(xp[t]-ep[t])/ep[t] for t in price_pivot.columns if t in ep.index and t in xp.index and not pd.isna(ep[t]) and not pd.isna(xp[t]) and ep[t]>0]
        sectors = [get_sector(t) for t in picks]
        results.append({'date': entry_date, 'ret': np.mean(rets)*100 if rets else 0,
                       'spx_ret': np.mean(spx)*100 if spx else 0,
                       'excess': (np.mean(rets)-np.mean(spx))*100 if rets and spx else 0,
                       'n_sectors': len(set(sectors))})
    return results

print("\n📊 运行回测...")
strategies = {
    'Top5 无分散': run_backtest(5, False),
    'Top5 有分散': run_backtest(5, True, 2),
    'Top20 无分散': run_backtest(20, False),
    'Top20 有分散': run_backtest(20, True, 3),
}

# 汇总
print(f"\n{'='*95}")
print("📊 Falcon 10年回测 (2016-01 ~ 2026-06)")
print(f"{'='*95}")
print(f"\n{'策略':<16} {'年化收益':>8} {'年化超额':>8} {'波动率':>7} {'Sharpe':>7} {'IR':>6} {'胜率':>6} {'最大回撤':>8}")
print("-" * 80)

for name, data in strategies.items():
    df = pd.DataFrame(data); mr = df['ret']; me = df['excess']
    ar = mr.mean()*12; ae = me.mean()*12; vol = mr.std()*np.sqrt(12)
    sh = ar/vol if vol>0 else 0; ir = me.mean()/me.std()*np.sqrt(12) if me.std()>0 else 0
    wr = (me>0).mean()*100; nav = (1+mr/100).cumprod(); dd = (nav/nav.cummax()-1).min()*100
    print(f"{name:<16} {ar:>+7.1f}% {ae:>+7.1f}% {vol:>6.1f}% {sh:>7.2f} {ir:>5.2f} {wr:>5.0f}% {dd:>+7.1f}%")
    if name == 'Top5 无分散':
        spx_ar = df['spx_ret'].mean()*12; spx_vol = df['spx_ret'].std()*np.sqrt(12)
print(f"{'SPX等权':<16} {spx_ar:>+7.1f}% {'':>8} {spx_vol:>6.1f}%")

# 分年
print(f"\n{'='*95}")
print("📅 分年超额收益")
print(f"{'='*95}")
print(f"\n{'年份':<6}", end='')
for name in strategies: print(f" {name:>14}", end='')
print(f" {'SPX':>8}")
print("-"*80)
for year in range(2016, 2027):
    print(f"{year:<6}", end='')
    spx_y = 0
    for name, data in strategies.items():
        df = pd.DataFrame(data); sub = df[df['date'].str.startswith(str(year))]
        if len(sub)==0: continue
        print(f" {sub['excess'].mean()*12:>+13.1f}%", end='')
        spx_y = sub['spx_ret'].mean()*12
    print(f" {spx_y:>+7.1f}%")

# 行业分散效果
print(f"\n{'='*95}")
print("🏢 行业分散效果")
print(f"{'='*95}")
for n in [5, 20]:
    no = strategies[f'Top{n} 无分散']; w = strategies[f'Top{n} 有分散']
    no_dd = pd.Series([d['ret'] for d in no]); w_dd = pd.Series([d['ret'] for d in w])
    no_max = ((1+no_dd/100).cumprod()/(1+no_dd/100).cumprod().cummax()-1).min()*100
    w_max = ((1+w_dd/100).cumprod()/(1+w_dd/100).cumprod().cummax()-1).min()*100
    print(f"\nTop{n}: 最大回撤 无分散{no_max:+.1f}% vs 有分散{w_max:+.1f}% (改善{w_max-no_max:+.1f}%)")

print(f"\n⏱️ 总耗时: {time.time()-t0:.0f}秒")
