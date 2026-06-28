#!/usr/bin/env python3
"""
🦅 Falcon 综合验证测试 — 7项全量验证
2016-2026, 476只SPX
"""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts/falcon')
from pathlib import Path
import pandas as pd, numpy as np

DATA_DIR = Path('data/falcon')
t0 = time.time()

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

print("=" * 90)
print("🦅 Falcon 综合验证测试 — 7项全量验证")
print("=" * 90)

master = pd.read_parquet(DATA_DIR / 'features_v02.parquet')
master['date'] = master['date'].astype(str)
print(f"\n📊 数据: {master['ticker'].nunique()}只, {master['date'].min()} ~ {master['date'].max()}")

fmp = {}
for n, f in [('fmp_ratios_historical','fmp_ratios_historical.json'),('analyst_historical','analyst_historical.json'),('fmp_key_metrics','fmp_key_metrics.json'),('fmp_financial_growth','fmp_financial_growth.json')]:
    fmp[n] = json.load(open(DATA_DIR/f))

weights = {'fund_ratio': 0.70, 'analyst': 0.20, 'fund_metric': 0.10}
price_pivot = master.pivot_table(index='date', columns='ticker', values='close').sort_index()
all_dates = sorted(price_pivot.index)

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
        vals = [p.get(f) for f in RATIO_FIELDS if p.get(f) is not None]
        r_fr[t] = np.mean(vals) if vals else None
        p_m = get_pit(fmp['fmp_key_metrics'].get(t, []), date)
        vals = [p_m.get(f) for f in METRIC_FIELDS if p_m.get(f) is not None]
        r_fm[t] = np.mean(vals) if vals else None
        p_g = get_pit(fmp['fmp_financial_growth'].get(t, []), date)
        vals = [p_g.get(f) for f in GROWTH_FIELDS if p_g.get(f) is not None]
        r_fg[t] = np.mean(vals) if vals else None
        p_a = get_pit(fmp['analyst_historical'].get(t, []), date)
        vals = [p_a.get(f) for f in ANALYST_FIELDS if p_a.get(f) is not None]
        r_an[t] = np.mean(vals) if vals else None
    df = pd.DataFrame({'fund_ratio': pd.Series(r_fr), 'fund_metric': pd.Series(r_fm),
                       'fund_growth': pd.Series(r_fg), 'analyst': pd.Series(r_an)})
    for col in df.columns:
        v = df[col].dropna()
        if len(v) > 5: df[col] = df[col].rank(pct=True)
    combined = sum(weights.get(f, 0) * df[f] for f in weights if f in df.columns)
    return combined.dropna().sort_values(ascending=False)

rebalance_dates = [all_dates[i] for i in range(0, len(all_dates), 21)]
print(f"📊 {len(rebalance_dates)} 个调仓日")

scores_dict = {}
for i, date in enumerate(rebalance_dates):
    if date not in price_pivot.index: continue
    scores_dict[date] = compute_scores(date, list(price_pivot.columns))
    if (i+1) % 30 == 0: print(f"   {i+1}/{len(rebalance_dates)}")
print(f"✅ {len(scores_dict)} 个调仓日scores")

def run_backtest(top_n, sector_limit=False, max_per_sector=2, hold_days=21, cost_pct=0.0, start_year=None, end_year=None, stop_loss=None):
    results = []
    rebalance_list = sorted(scores_dict.keys())
    for idx, entry_date in enumerate(rebalance_list):
        if start_year and entry_date[:4] < str(start_year): continue
        if end_year and entry_date[:4] > str(end_year): continue
        ei = all_dates.index(entry_date)
        exit_date = all_dates[min(ei + hold_days, len(all_dates) - 1)]
        if entry_date not in price_pivot.index or exit_date not in price_pivot.index: continue
        ep = price_pivot.loc[entry_date]; xp = price_pivot.loc[exit_date]
        sorted_scores = scores_dict[entry_date]
        picks = select_stocks(sorted_scores, top_n, max_per_sector) if sector_limit else list(sorted_scores.head(top_n).index)
        rets = []
        for t in picks:
            if t in ep.index and t in xp.index and not pd.isna(ep[t]) and not pd.isna(xp[t]) and ep[t] > 0:
                r = (xp[t] - ep[t]) / ep[t]
                if stop_loss and r < stop_loss: r = stop_loss
                rets.append(r - cost_pct)
        spx = [(xp[t]-ep[t])/ep[t] for t in price_pivot.columns if t in ep.index and t in xp.index and not pd.isna(ep[t]) and not pd.isna(xp[t]) and ep[t]>0]
        results.append({'date': entry_date, 'ret': np.mean(rets)*100 if rets else 0,
                       'spx_ret': np.mean(spx)*100 if spx else 0,
                       'excess': (np.mean(rets)-np.mean(spx))*100 if rets and spx else 0})
    return results

def stats(data):
    df = pd.DataFrame(data); mr = df['ret']; me = df['excess']
    ar = mr.mean()*12; vol = mr.std()*np.sqrt(12)
    sh = ar/vol if vol>0 else 0; ir = me.mean()/me.std()*np.sqrt(12) if me.std()>0 else 0
    wr = (me>0).mean()*100; nav = (1+mr/100).cumprod(); dd = (nav/nav.cummax()-1).min()*100
    return {'ann_ret': ar, 'excess': me.mean()*12, 'vol': vol, 'sharpe': sh, 'ir': ir, 'wr': wr, 'dd': dd}

# TEST 1
print(f"\n{'='*90}")
print("TEST 1: 持仓数扫描 (5~20)")
print(f"{'='*90}")
print(f"{'持仓数':>6} {'年化收益':>8} {'超额':>7} {'Sharpe':>7} {'IR':>6} {'回撤':>7} {'胜率':>6}")
print("-" * 60)
best_sharpe = 0; best_n = 5
for n in range(5, 21):
    r = run_backtest(n, sector_limit=True, max_per_sector=3)
    s = stats(r)
    print(f"{n:>6} {s['ann_ret']:>+7.1f}% {s['excess']:>+6.1f}% {s['sharpe']:>7.2f} {s['ir']:>5.2f} {s['dd']:>+6.1f}% {s['wr']:>5.0f}%")
    if s['sharpe'] > best_sharpe: best_sharpe = s['sharpe']; best_n = n
print(f"\n🏆 最优: Top{best_n} (Sharpe={best_sharpe:.2f})")

# TEST 2
print(f"\n{'='*90}")
print("TEST 2: 新闻波动率测试")
print(f"{'='*90}")
finbert_dir = Path('data/finbert_sentiment')
sent_data = {}
for pf in finbert_dir.rglob('*.parquet'):
    try:
        df = pd.read_parquet(pf)
        if 'published_at' in df.columns and 'sentiment' in df.columns and 'ticker' in df.columns:
            df['date'] = pd.to_datetime(df['published_at'], errors='coerce').dt.strftime('%Y-%m-%d')
            for _, row in df.iterrows():
                t = row['ticker']; d = row['date']
                if pd.isna(d) or pd.isna(row['sentiment']): continue
                if t not in sent_data: sent_data[t] = {}
                if d not in sent_data[t]: sent_data[t][d] = []
                sent_data[t][d].append(row['sentiment'])
    except: continue

sent_vol = {}
for t, dates in sent_data.items():
    for d, sents in dates.items():
        if len(sents) >= 2:
            if t not in sent_vol: sent_vol[t] = {}
            sent_vol[t][d] = np.std(sents)

print(f"   情绪数据: {len(sent_data)} tickers, {sum(len(d) for d in sent_data.values())} ticker-dates")

if len(sent_vol) > 10:
    test_dates = [d for d in sorted(scores_dict.keys()) if d >= '2024-01-01']
    high_vol_rets, low_vol_rets = [], []
    all_vols = [np.mean(list(sent_vol[t].values())) for t in sent_vol if sent_vol[t]]
    median_vol = np.median(all_vols) if all_vols else 0
    
    for entry_date in test_dates:
        if entry_date not in price_pivot.index: continue
        sorted_scores = scores_dict[entry_date]
        picks = list(sorted_scores.head(10).index)
        vol_vals = []
        for t in picks:
            if t in sent_vol:
                for off in range(7):
                    d = str(pd.Timestamp(entry_date) - pd.Timedelta(days=off))[:10]
                    if d in sent_vol[t]: vol_vals.append(sent_vol[t][d])
        ei = all_dates.index(entry_date)
        exit_date = all_dates[min(ei + 21, len(all_dates) - 1)]
        ep = price_pivot.loc[entry_date]; xp = price_pivot.loc[exit_date]
        rets = [(xp[t]-ep[t])/ep[t] for t in picks if t in ep.index and t in xp.index and not pd.isna(ep[t]) and not pd.isna(xp[t]) and ep[t]>0]
        port_ret = np.mean(rets)*100 if rets else 0
        if vol_vals:
            if np.mean(vol_vals) > median_vol: high_vol_rets.append(port_ret)
            else: low_vol_rets.append(port_ret)
    
    if high_vol_rets and low_vol_rets:
        print(f"   高新闻波动: {len(high_vol_rets)}次, 平均{np.mean(high_vol_rets):+.1f}%")
        print(f"   低新闻波动: {len(low_vol_rets)}次, 平均{np.mean(low_vol_rets):+.1f}%")
        diff = np.mean(low_vol_rets) - np.mean(high_vol_rets)
        print(f"   差异: {diff:+.1f}% {'✅ 低波动更好=有效' if diff > 0 else '❌ 无效或反向'}")
    else:
        print(f"   ⚠️ 样本不足(高:{len(high_vol_rets)}, 低:{len(low_vol_rets)})")
else:
    print("   ❌ 情绪数据覆盖不足")

# TEST 3
print(f"\n{'='*90}")
print("TEST 3: FMP数据质量检查")
print(f"{'='*90}")
print(f"{'年份':>6} {'Ratios':>8} {'Analyst':>9} {'Metrics':>9} {'有效score':>10}")
print("-" * 50)
for year in range(2016, 2027):
    rc = sum(1 for t, r in fmp['fmp_ratios_historical'].items() if any(d.get('date','').startswith(str(year)) for d in r if isinstance(d, dict)))
    ac = sum(1 for t, r in fmp['analyst_historical'].items() if any(d.get('date','').startswith(str(year)) for d in r if isinstance(d, dict)))
    mc = sum(1 for t, r in fmp['fmp_key_metrics'].items() if any(d.get('date','').startswith(str(year)) for d in r if isinstance(d, dict)))
    yd = [d for d in sorted(scores_dict.keys()) if d.startswith(str(year))]
    avg_v = np.mean([len(scores_dict[d].dropna()) for d in yd]) if yd else 0
    print(f"{year:>6} {rc:>8} {ac:>9} {mc:>9} {avg_v:>9.0f}")

# TEST 4
print(f"\n{'='*90}")
print("TEST 4: 存活偏差")
print(f"{'='*90}")
import requests, os
from dotenv import load_dotenv
load_dotenv()
fmp_key = os.getenv('FMP_API_KEY')
try:
    r = requests.get(f'https://financialmodelingprep.com/stable/sp500-constituent?apikey={fmp_key}', timeout=10)
    if r.status_code == 200:
        current = r.json()
        added_2016 = [c for c in current if c.get('dateFirstAdded','') >= '2016-01-01']
        added_2020 = [c for c in current if c.get('dateFirstAdded','') >= '2020-01-01']
        our = set(master['ticker'].unique())
        new_in_our = our & set(c['symbol'] for c in added_2016)
        print(f"   当前SP500: {len(current)}只")
        print(f"   2016后加入: {len(added_2016)}只 ({len(added_2016)/len(current)*100:.0f}%)")
        print(f"   2020后加入: {len(added_2020)}只 ({len(added_2020)/len(current)*100:.0f}%)")
        print(f"   我们476只中2016后加入: {len(new_in_our)}只")
        print(f"   ⚠️ 存活偏差约{len(new_in_our)/476*100:.0f}% (这些股票2016不在SP500)")
except Exception as e:
    print(f"   API错误: {e}")

# TEST 5
print(f"\n{'='*90}")
print("TEST 5: 交易成本敏感度")
print(f"{'='*90}")
print(f"{'成本':>8} {'年化收益':>8} {'超额':>7} {'Sharpe':>7} {'IR':>6}")
print("-" * 45)
for cost in [0.0, 0.0005, 0.001, 0.002, 0.005]:
    r = run_backtest(10, sector_limit=True, max_per_sector=3, cost_pct=cost)
    s = stats(r)
    print(f"{cost*100:>7.2f}% {s['ann_ret']:>+7.1f}% {s['excess']:>+6.1f}% {s['sharpe']:>7.2f} {s['ir']:>5.2f}")

# TEST 6
print(f"\n{'='*90}")
print("TEST 6: 调仓频率优化")
print(f"{'='*90}")
print(f"{'频率':>10} {'持仓天':>6} {'年化收益':>8} {'超额':>7} {'Sharpe':>7} {'IR':>6}")
print("-" * 50)
for hold in [5, 10, 15, 21, 42, 63]:
    label = {5:'周频', 10:'双周', 15:'3周', 21:'月频', 42:'双月', 63:'季频'}.get(hold, f'{hold}天')
    r = run_backtest(10, sector_limit=True, max_per_sector=3, hold_days=hold)
    s = stats(r)
    print(f"{label:>10} {hold:>6} {s['ann_ret']:>+7.1f}% {s['excess']:>+6.1f}% {s['sharpe']:>7.2f} {s['ir']:>5.2f}")

# TEST 7
print(f"\n{'='*90}")
print("TEST 7: OOS验证 (IS: 2016-2022, OOS: 2023-2026)")
print(f"{'='*90}")
print(f"{'策略':<16} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'比率':>6} {'IS回撤':>8} {'OOS回撤':>8}")
print("-" * 65)
for tn, sl, label in [(5, False, 'Top5无分散'), (5, True, 'Top5有分散'), (10, True, 'Top10有分散'), (20, True, 'Top20有分散')]:
    is_d = run_backtest(tn, sector_limit=sl, max_per_sector=3 if sl else 99, start_year=2016, end_year=2022)
    oos_d = run_backtest(tn, sector_limit=sl, max_per_sector=3 if sl else 99, start_year=2023, end_year=2026)
    is_s = stats(is_d); oos_s = stats(oos_d)
    ratio = oos_s['sharpe'] / is_s['sharpe'] if is_s['sharpe'] > 0 else 0
    flag = "✅" if ratio > 0.3 else "⚠️" if ratio > 0 else "❌"
    print(f"{label:<16} {is_s['sharpe']:>9.2f} {oos_s['sharpe']:>10.2f} {ratio:>5.1%} {is_s['dd']:>+7.1f}% {oos_s['dd']:>+7.1f}% {flag}")

print(f"\n⏱️ 总耗时: {time.time()-t0:.0f}秒")
