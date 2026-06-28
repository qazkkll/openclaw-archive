#!/usr/bin/env python3
"""
🦅 Falcon 全量回测: Top5 vs Top20 × 有/无行业分散
2022-01 ~ 2026-06 (4.5年) — 优化版: 只算调仓日PIT
"""
import sys
sys.path.insert(0, 'scripts/falcon')
import pandas as pd, numpy as np, json, time
from pathlib import Path

DATA_DIR = Path('data/falcon')
t0 = time.time()

# ═══════════════════════════════════════════════════
# 行业分类
# ═══════════════════════════════════════════════════
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

def get_sector(ticker):
    return SECTOR_MAP.get(ticker, 'Other')

# ═══════════════════════════════════════════════════
# 行业分散选股
# ═══════════════════════════════════════════════════
def select_stocks(sorted_scores, top_n, max_per_sector=2):
    selected = []
    sector_count = {}
    for ticker in sorted_scores.index:
        sector = get_sector(ticker)
        if sector_count.get(sector, 0) < max_per_sector:
            selected.append(ticker)
            sector_count[sector] = sector_count.get(sector, 0) + 1
            if len(selected) >= top_n:
                break
    return selected

# ═══════════════════════════════════════════════════
# 加载数据
# ═══════════════════════════════════════════════════
print("📊 加载数据...")
master = pd.read_parquet(DATA_DIR / 'features_v02.parquet')
master['date'] = master['date'].astype(str)
master_all = master[(master['date'] >= '2022-01-01') & (master['date'] <= '2026-06-26')]
print(f"   {master_all['ticker'].nunique()} 只, {master_all['date'].nunique()} 天")

fmp = {}
for n, f in [('fmp_ratios_historical','fmp_ratios_historical.json'),('analyst_historical','analyst_historical.json'),('fmp_key_metrics','fmp_key_metrics.json'),('fmp_financial_growth','fmp_financial_growth.json')]:
    fmp[n] = json.load(open(DATA_DIR/f))
fmp['fmp_insider'] = {}
fmp['fmp_dcf'] = {}
fmp['fmp_price_target'] = {}

weights = {'fund_ratio': 0.70, 'analyst': 0.20, 'fund_metric': 0.10}
price_pivot = master_all.pivot_table(index='date', columns='ticker', values='close').sort_index()
all_dates = sorted(price_pivot.index)

# ═══════════════════════════════════════════════════
# 优化PIT: 只算调仓日
# ═══════════════════════════════════════════════════
rebalance_dates = [all_dates[i] for i in range(0, len(all_dates), 21)]
print(f"📊 调仓日: {len(rebalance_dates)} 个")

# 内联PIT查询（避免加载全量engine）
RATIO_FIELDS = ["priceToEarningsRatio","priceToBookRatio","priceToSalesRatio","priceToFreeCashFlowRatio","enterpriseValueMultiple","grossProfitMargin","netProfitMargin","operatingProfitMargin","ebitdaMargin","assetTurnover","inventoryTurnover","receivablesTurnover","debtToEquityRatio","currentRatio","quickRatio","financialLeverageRatio","freeCashFlowOperatingCashFlowRatio","operatingCashFlowRatio","dividendYieldPercentage","dividendPayoutRatio"]
METRIC_FIELDS = ["earningsYield","evToEBITDA","evToFreeCashFlow","evToSales","freeCashFlowYield","returnOnEquity","returnOnAssets","returnOnCapitalEmployed","returnOnInvestedCapital","returnOnTangibleAssets","incomeQuality","grahamNumber","cashConversionCycle","capexToRevenue","capexToDepreciation","researchAndDevelopementToRevenue","stockBasedCompensationToRevenue","netDebtToEBITDA","operatingReturnOnAssets"]
GROWTH_FIELDS = ["revenueGrowth","grossProfitGrowth","ebitgrowth","operatingIncomeGrowth","netIncomeGrowth","epsdilutedGrowth","freeCashFlowGrowth","tenYRevenueGrowthPerShare","fiveYRevenueGrowthPerShare","threeYRevenueGrowthPerShare","receivablesGrowth","inventoryGrowth","assetGrowth","bookValueperShareGrowth","debtGrowth"]
ANALYST_FIELDS = ["eps_revision","revenue_revision","eps_dispersion"]
ALL_FMP = RATIO_FIELDS + METRIC_FIELDS + GROWTH_FIELDS

def get_pit(quarterly_data, date):
    if not quarterly_data: return {}
    latest = {}
    for q in quarterly_data:
        if isinstance(q, dict) and q.get("date", "") <= date:
            latest = q
    return latest

def compute_scores_for_date(date, feat_day, tickers):
    """计算某天的Falcon scores。"""
    r_fund_ratio = {}
    r_fund_metric = {}
    r_fund_growth = {}
    r_analyst = {}
    
    for t in tickers:
        # Ratios
        pit = get_pit(fmp['fmp_ratios_historical'].get(t, []), date)
        vals = [pit.get(f) for f in RATIO_FIELDS if pit.get(f) is not None]
        r_fund_ratio[t] = np.mean(vals) if vals else None
        
        # Metrics
        pit_m = get_pit(fmp['fmp_key_metrics'].get(t, []), date)
        vals = [pit_m.get(f) for f in METRIC_FIELDS if pit_m.get(f) is not None]
        r_fund_metric[t] = np.mean(vals) if vals else None
        
        # Growth
        pit_g = get_pit(fmp['fmp_financial_growth'].get(t, []), date)
        vals = [pit_g.get(f) for f in GROWTH_FIELDS if pit_g.get(f) is not None]
        r_fund_growth[t] = np.mean(vals) if vals else None
        
        # Analyst
        pit_a = get_pit(fmp['analyst_historical'].get(t, []), date)
        vals_a = []
        for f in ANALYST_FIELDS:
            v = pit_a.get(f)
            if v is not None: vals_a.append(v)
        r_analyst[t] = np.mean(vals_a) if vals_a else None
    
    # Rank (percentile)
    df = pd.DataFrame({
        'fund_ratio': pd.Series(r_fund_ratio),
        'fund_metric': pd.Series(r_fund_metric),
        'fund_growth': pd.Series(r_fund_growth),
        'analyst': pd.Series(r_analyst),
    })
    # Rank each column
    for col in df.columns:
        valid = df[col].dropna()
        if len(valid) > 5:
            df[col] = df[col].rank(pct=True)
    
    # Combined score
    combined = sum(weights.get(f, 0) * df[f] for f in weights if f in df.columns)
    return combined.dropna().sort_values(ascending=False)

# 预计算每个调仓日的scores
print("📊 计算调仓日scores...")
scores_dict = {}
for i, date in enumerate(rebalance_dates):
    if date not in price_pivot.index: continue
    tickers = list(price_pivot.columns)
    scores_dict[date] = compute_scores_for_date(date, None, tickers)
    if (i+1) % 10 == 0:
        print(f"   {i+1}/{len(rebalance_dates)}")

print(f"✅ {len(scores_dict)} 个调仓日")

# ═══════════════════════════════════════════════════
# 回测引擎
# ═══════════════════════════════════════════════════
def run_backtest(top_n, sector_limit=False, max_per_sector=2):
    results = []
    rebalance_list = sorted(scores_dict.keys())
    
    for idx, entry_date in enumerate(rebalance_list):
        # 下一个调仓日 = 出场日
        if idx + 1 < len(rebalance_list):
            exit_date = rebalance_list[idx + 1]
        else:
            # 最后一段，用21天后的日期
            ei = all_dates.index(entry_date)
            exit_date = all_dates[min(ei + 21, len(all_dates) - 1)]
        
        if entry_date not in price_pivot.index or exit_date not in price_pivot.index:
            continue
        
        entry_prices = price_pivot.loc[entry_date]
        exit_prices = price_pivot.loc[exit_date]
        
        sorted_scores = scores_dict[entry_date]
        
        if sector_limit:
            picks = select_stocks(sorted_scores, top_n, max_per_sector)
        else:
            picks = list(sorted_scores.head(top_n).index)
        
        # 组合收益
        rets = []
        for t in picks:
            if t in entry_prices.index and t in exit_prices.index:
                ep = entry_prices[t]
                xp = exit_prices[t]
                if not pd.isna(ep) and not pd.isna(xp) and ep > 0:
                    rets.append((xp - ep) / ep)
        port_ret = np.mean(rets) * 100 if rets else 0
        
        # SPX等权
        spx_rets = []
        for t in price_pivot.columns:
            if t in entry_prices.index and t in exit_prices.index:
                ep = entry_prices[t]
                xp = exit_prices[t]
                if not pd.isna(ep) and not pd.isna(xp) and ep > 0:
                    spx_rets.append((xp - ep) / ep)
        spx_ret = np.mean(spx_rets) * 100 if spx_rets else 0
        
        sectors = [get_sector(t) for t in picks]
        results.append({
            'date': entry_date, 'ret': port_ret, 'spx_ret': spx_ret,
            'excess': port_ret - spx_ret, 'picks': picks,
            'n_sectors': len(set(sectors)),
        })
    return results

# ═══════════════════════════════════════════════════
# 跑4组
# ═══════════════════════════════════════════════════
print("\n📊 运行回测...")
strategies = {
    'Top5 无分散': run_backtest(5, sector_limit=False),
    'Top5 有分散': run_backtest(5, sector_limit=True, max_per_sector=2),
    'Top20 无分散': run_backtest(20, sector_limit=False),
    'Top20 有分散': run_backtest(20, sector_limit=True, max_per_sector=3),
}

# ═══════════════════════════════════════════════════
# 汇总统计
# ═══════════════════════════════════════════════════
print("\n" + "=" * 95)
print("📊 Falcon 4组策略对比 (2022-01 ~ 2026-06)")
print("=" * 95)

print(f"\n{'策略':<16} {'年化收益':>8} {'年化超额':>8} {'波动率':>7} {'Sharpe':>7} {'IR':>6} {'胜率':>6} {'最大回撤':>8} {'月均':>6}")
print("-" * 86)

spx_ann = 0
for name, data in strategies.items():
    df = pd.DataFrame(data)
    mr = df['ret']
    me = df['excess']
    
    ann_ret = mr.mean() * 12
    ann_excess = me.mean() * 12
    vol = mr.std() * np.sqrt(12)
    sharpe = ann_ret / vol if vol > 0 else 0
    ir = me.mean() / me.std() * np.sqrt(12) if me.std() > 0 else 0
    win = (me > 0).mean() * 100
    worst = mr.min()
    
    # 最大回撤（累计NAV）
    nav = (1 + mr/100).cumprod()
    dd = (nav / nav.cummax() - 1).min() * 100
    
    print(f"{name:<16} {ann_ret:>+7.1f}% {ann_excess:>+7.1f}% {vol:>6.1f}% {sharpe:>7.2f} {ir:>5.2f} {win:>5.0f}% {dd:>+7.1f}% {mr.mean():>+5.1f}%")
    if name == 'Top5 无分散':
        spx_ann = df['spx_ret'].mean() * 12
        spx_vol = df['spx_ret'].std() * np.sqrt(12)

print(f"{'SPX等权':<16} {spx_ann:>+7.1f}% {'':>8} {spx_vol:>6.1f}%")

# ═══════════════════════════════════════════════════
# 分年
# ═══════════════════════════════════════════════════
print(f"\n{'='*95}")
print("📅 分年超额收益")
print(f"{'='*95}")
print(f"\n{'年份':<6}", end='')
for name in strategies: print(f" {name:>14}", end='')
print(f" {'SPX':>8}")
print("-" * 80)

for year in range(2022, 2027):
    print(f"{year:<6}", end='')
    spx_y = 0
    for name, data in strategies.items():
        df = pd.DataFrame(data)
        sub = df[df['date'].str.startswith(str(year))]
        if len(sub) == 0: continue
        print(f" {sub['excess'].mean()*12:>+13.1f}%", end='')
        spx_y = sub['spx_ret'].mean() * 12
    print(f" {spx_y:>+7.1f}%")

# ═══════════════════════════════════════════════════
# 行业分散效果
# ═══════════════════════════════════════════════════
print(f"\n{'='*95}")
print("🏢 行业分散效果")
print(f"{'='*95}")

for top_n in [5, 20]:
    no_lim = strategies[f'Top{top_n} 无分散']
    with_lim = strategies[f'Top{top_n} 有分散']
    
    no_sec = [d['n_sectors'] for d in no_lim]
    with_sec = [d['n_sectors'] for d in with_lim]
    
    print(f"\nTop{top_n}:")
    print(f"  无分散: 平均{np.mean(no_sec):.1f}个行业, 最少{min(no_sec)}个")
    print(f"  有分散: 平均{np.mean(with_sec):.1f}个行业, 最少{min(with_sec)}个")
    
    no_worst = min(d['ret'] for d in no_lim)
    with_worst = min(d['ret'] for d in with_lim)
    no_nav = pd.Series([d['ret'] for d in no_lim])
    with_nav = pd.Series([d['ret'] for d in with_lim])
    no_dd = ((1+no_nav/100).cumprod() / (1+no_nav/100).cumprod().cummax() - 1).min() * 100
    with_dd = ((1+with_nav/100).cumprod() / (1+with_nav/100).cumprod().cummax() - 1).min() * 100
    
    print(f"  最差单月: 无分散{no_worst:+.1f}% vs 有分散{with_worst:+.1f}% (改善{with_worst-no_worst:+.1f}%)")
    print(f"  最大回撤: 无分散{no_dd:+.1f}% vs 有分散{with_dd:+.1f}% (改善{with_dd-no_dd:+.1f}%)")

# ═══════════════════════════════════════════════════
# 最近12个月
# ═══════════════════════════════════════════════════
print(f"\n{'='*95}")
print("📋 最近12个月逐月对比")
print(f"{'='*95}")
print(f"\n{'月份':<8}", end='')
for name in strategies: print(f" {name:>14}", end='')
print(f" {'SPX':>8}")
print("-" * 80)

recent = sorted(set(d['date'] for d in strategies['Top5 无分散']))[-12:]
for entry_date in recent:
    print(f"{entry_date[:7]:<8}", end='')
    spx_v = 0
    for name, data in strategies.items():
        row = [d for d in data if d['date'] == entry_date][0]
        print(f" {row['ret']:>+13.1f}%", end='')
        spx_v = row['spx_ret']
    print(f" {spx_v:>+7.1f}%")

# ═══════════════════════════════════════════════════
# 重叠度
# ═══════════════════════════════════════════════════
print(f"\n{'='*95}")
print("🔄 Top5在Top20中的重叠度")
print(f"{'='*95}")
no5 = strategies['Top5 无分散']
no20 = strategies['Top20 无分散']
overlaps = [len(set(d5['picks']) & set(d20['picks'])) for d5, d20 in zip(no5, no20)]
print(f"平均重叠: {np.mean(overlaps):.1f}/5 ({np.mean(overlaps)/5*100:.0f}%)")

print(f"\n⏱️ 总耗时: {time.time()-t0:.0f}秒")
