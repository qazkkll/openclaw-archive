#!/usr/bin/env python3
"""
🦅 Falcon V0.4.9 — Insider Cluster Buy Factor Validation
==========================================================
V0.4.6 + cluster_buy (3+ insiders buying in 90 days)

Test: Does adding cluster_buy improve Walk-Forward Sharpe?

Steps:
1. Download insider data for all 476 tickers
2. Compute rolling cluster_buy signal per date per ticker
3. Add to factor universe
4. Run WF: V0.4.6 vs V0.4.6+cluster_buy at various weights
"""

import sys, json, time, os, re
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_engine import BacktestEngine, BacktestResult, DataQualityError

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
DATA_DIR = WORKSPACE / "data" / "falcon"
INSIDER_DIR = DATA_DIR / "insider_trading"
INSIDER_DIR.mkdir(exist_ok=True)

SPX_PATH = WORKSPACE / "data" / "config" / "sp500_symbols.json"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"
OUTPUT_PATH = DATA_DIR / "v049_validation_results.json"

import urllib.request

# ── Factor groups (same as V0.4.6 + insider) ──
FACTOR_GROUPS = {
    'fund_ratio': [
        'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
        'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
        'r_grossProfitMargin', 'r_netProfitMargin', 'r_operatingProfitMargin', 'r_ebitdaMargin',
        'r_assetTurnover', 'r_inventoryTurnover', 'r_receivablesTurnover',
        'r_debtToEquityRatio', 'r_currentRatio', 'r_quickRatio', 'r_financialLeverageRatio',
        'r_freeCashFlowOperatingCashFlowRatio', 'r_operatingCashFlowRatio',
        'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    ],
    'fund_growth': [
        'g_revenueGrowth', 'g_grossProfitGrowth', 'g_ebitgrowth',
        'g_operatingIncomeGrowth', 'g_netIncomeGrowth', 'g_epsdilutedGrowth',
        'g_freeCashFlowGrowth', 'g_tenYRevenueGrowthPerShare',
        'g_fiveYRevenueGrowthPerShare', 'g_threeYRevenueGrowthPerShare',
        'g_receivablesGrowth', 'g_inventoryGrowth', 'g_assetGrowth',
        'g_bookValueperShareGrowth', 'g_debtGrowth',
    ],
    'analyst': ['a_eps_revision', 'a_revenue_revision', 'a_eps_dispersion', 'a_num_analysts_eps'],
    'income': ['i_gross_margin', 'i_operating_margin', 'i_net_margin', 'i_ebitda_margin',
               'i_revenue_growth_yoy', 'i_gross_margin_delta'],
    'qoq': ['r_grossProfitMargin_qoq', 'r_netProfitMargin_qoq',
            'r_operatingProfitMargin_qoq', 'r_ebitdaMargin_qoq'],
    'cashflow': ['c_fcf_margin', 'c_capex_intensity', 'c_fcf_to_income', 'c_buyback_yield'],
    'insider': ['insider_cluster_buy'],  # NEW for V0.4.9
}

GC_WEIGHTS = {'fund_growth': 0.60, 'analyst': 0.25, 'income': 0.15}

FLIP_FACTORS = {
    'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
    'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
    'r_debtToEquityRatio', 'r_financialLeverageRatio', 'r_inventoryTurnover',
    'c_capex_intensity', 'g_debtGrowth', 'g_receivablesGrowth', 'g_inventoryGrowth',
    'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
}

IC_LOOKBACK = 126
IC_POWER = 0.5

# ── Configs to test ──
CONFIGS = {
    'V0.4.6': {
        'weights': {'fund_ratio': 0.45, 'gc': 0.20, 'qoq': 0.20, 'cf': 0.15, 'insider': 0.0},
        'hold_days': 21, 'top_n': 10,
    },
    'V0.4.9_insider5': {
        'weights': {'fund_ratio': 0.43, 'gc': 0.19, 'qoq': 0.19, 'cf': 0.14, 'insider': 0.05},
        'hold_days': 21, 'top_n': 10,
    },
    'V0.4.9_insider10': {
        'weights': {'fund_ratio': 0.41, 'gc': 0.18, 'qoq': 0.18, 'cf': 0.13, 'insider': 0.10},
        'hold_days': 21, 'top_n': 10,
    },
    'V0.4.9_insider15': {
        'weights': {'fund_ratio': 0.38, 'gc': 0.17, 'qoq': 0.17, 'cf': 0.13, 'insider': 0.15},
        'hold_days': 21, 'top_n': 10,
    },
}


# ══════════════════════════════════════════════════
# Insider Data Scraping
# ══════════════════════════════════════════════════

def scrape_openinsider(ticker):
    """Scrape insider trading from OpenInsider."""
    url = f'http://openinsider.com/search?q={ticker}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        body = resp.read().decode()
    except:
        return []
    
    tables = re.findall(r'<table[^>]*>(.*?)</table>', body, re.DOTALL)
    data_table = None
    max_rows = 0
    for t in tables:
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', t, re.DOTALL)
        if len(rows) > max_rows:
            max_rows = len(rows)
            data_table = t
    
    if not data_table or max_rows < 3:
        return []
    
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', data_table, re.DOTALL)
    records = []
    
    for row in rows[1:]:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        if len(cells) < 10:
            continue
        clean = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        
        try:
            trade_date = clean[2][:10] if len(clean) > 2 and clean[2] else ''
            insider_name = clean[4] if len(clean) > 4 else ''
            trade_type = clean[6] if len(clean) > 6 else ''
            
            is_buy = any(kw in trade_type.upper() for kw in ['PURCHASE', 'P -', 'BUY'])
            
            records.append({
                'trade_date': trade_date,
                'insider': insider_name,
                'type': trade_type,
                'is_buy': is_buy,
            })
        except:
            continue
    
    return records


def download_all_insider(tickers):
    """Download insider data with caching and parallel fetching."""
    cache_file = INSIDER_DIR / "v049_cache.json"
    
    cache = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
    
    to_fetch = [t for t in tickers if t not in cache]
    print(f"📥 Fetching insider data: {len(to_fetch)} new, {len(cache)} cached")
    
    def fetch_one(ticker):
        return ticker, scrape_openinsider(ticker)
    
    # Parallel fetching (4 threads to be polite)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_one, t): t for t in to_fetch}
        done = 0
        for future in as_completed(futures):
            ticker, records = future.result()
            cache[ticker] = records
            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{len(to_fetch)}] {ticker}: {len(records)} records")
                with open(cache_file, 'w') as f:
                    json.dump(cache, f)
            time.sleep(0.2)
    
    with open(cache_file, 'w') as f:
        json.dump(cache, f)
    
    total = sum(len(v) for v in cache.values())
    with_data = sum(1 for v in cache.values() if len(v) > 0)
    print(f"  ✅ {with_data} tickers with data, {total} total records")
    return cache


# ══════════════════════════════════════════════════
# Feature Engineering
# ══════════════════════════════════════════════════

def compute_cluster_buy_features(cache, all_dates, lookback_days=90):
    """Compute rolling cluster_buy feature for each date × ticker.
    
    Returns: dict of {date: {ticker: 0 or 1}}
    """
    print(f"📊 Computing cluster_buy features for {len(all_dates)} dates...")
    t0 = time.time()
    
    # Pre-index: ticker -> sorted list of (trade_date, is_buy, insider)
    ticker_index = {}
    for ticker, records in cache.items():
        if not records:
            continue
        buys = [(r['trade_date'], r.get('insider', '')) for r in records 
                if r.get('is_buy') and r.get('trade_date')]
        buys.sort()
        ticker_index[ticker] = buys
    
    features = {}
    step = 5  # Compute every 5 days to save time
    
    for i in range(0, len(all_dates), step):
        date = all_dates[i]
        cutoff = (pd.Timestamp(date) - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        day_features = {}
        for ticker, buys in ticker_index.items():
            # Count unique buyers in window
            recent_buyers = set()
            for trade_date, insider in buys:
                if trade_date > date:
                    break
                if trade_date >= cutoff:
                    recent_buyers.add(insider)
            
            day_features[ticker] = 1.0 if len(recent_buyers) >= 3 else 0.0
        
        features[date] = day_features
    
    # Fill gaps: forward-fill for dates we skipped
    filled = {}
    sorted_feature_dates = sorted(features.keys())
    for date in all_dates:
        cands = [d for d in sorted_feature_dates if d <= date]
        if cands:
            filled[date] = features[cands[-1]]
    
    print(f"  ✅ {len(filled)} dates ({time.time()-t0:.0f}s)")
    return filled


def build_features_with_insider(df, cluster_features, all_dates):
    """Add insider_cluster_buy to the features DataFrame."""
    print("📊 Merging insider features into main DataFrame...")
    
    # Create insider column
    insider_col = []
    for _, row in df.iterrows():
        date = row['date']
        ticker = row['ticker']
        if date in cluster_features and ticker in cluster_features[date]:
            insider_col.append(cluster_features[date][ticker])
        else:
            insider_col.append(np.nan)
    
    df = df.copy()
    df['insider_cluster_buy'] = insider_col
    
    coverage = df['insider_cluster_buy'].notna().mean()
    print(f"  ✅ insider_cluster_buy coverage: {coverage:.1%}")
    return df


# ══════════════════════════════════════════════════
# Ranks & IC
# ══════════════════════════════════════════════════

def compute_ranks(df, factor_cols, sample_dates):
    print(f"📊 Computing ranks for {len(sample_dates)} days...")
    t0 = time.time()
    ranks = {}
    for date in sample_dates:
        day_df = df[df['date'] == date]
        if len(day_df) < 10:
            continue
        tickers = day_df['ticker'].values
        rank_df = pd.DataFrame(index=tickers)
        for col in factor_cols:
            if col not in day_df.columns:
                continue
            vals = day_df[col].values.astype(float)
            valid = ~np.isnan(vals)
            if valid.sum() < 10:
                continue
            r = np.full_like(vals, np.nan)
            r[valid] = rankdata(vals[valid], method='average') / valid.sum()
            if col in FLIP_FACTORS:
                mask = ~np.isnan(r)
                r[mask] = 1.0 - r[mask]
            rank_df[col] = r
        ranks[date] = rank_df
    print(f"  ✅ {len(ranks)} days ({time.time()-t0:.0f}s)")
    return ranks


def compute_daily_ic(ranks, prices, factor_cols):
    print("📊 Computing daily IC...")
    t0 = time.time()
    all_dates = sorted(ranks.keys())
    price_dates = sorted(prices.index.astype(str))
    fwd_cache = {}
    for date in all_dates:
        fc = [d for d in price_dates if d > date]
        if len(fc) < 20:
            continue
        ff = fc[min(29, len(fc)-1)]
        if ff not in prices.index or date not in prices.index:
            continue
        fwd_cache[date] = ((prices.loc[ff] / prices.loc[date]) - 1).dropna()
    daily_ic = {}
    for date in all_dates:
        if date not in fwd_cache or date not in ranks:
            continue
        rd = ranks[date]
        fw = fwd_cache[date]
        cm = rd.index.intersection(fw.index)
        if len(cm) < 30:
            continue
        fv = fw[cm].values
        daily_ic[date] = {}
        for col in factor_cols:
            if col not in rd.columns:
                continue
            r = rd.loc[cm, col].values
            valid = ~(np.isnan(r) | np.isnan(fv))
            if valid.sum() < 30:
                continue
            ic, _ = spearmanr(r[valid], fv[valid])
            if not np.isnan(ic):
                daily_ic[date][col] = ic
    print(f"  ✅ {len(daily_ic)} days ({time.time()-t0:.0f}s)")
    return daily_ic


def rolling_ic(daily_ic, all_dates, factor_cols, lookback, step=5):
    print(f"📊 Rolling IC (lookback={lookback})...")
    t0 = time.time()
    ic_dates = sorted(daily_ic.keys())
    ic_history = {}
    for i in range(0, len(ic_dates), step):
        date = ic_dates[i]
        ws = max(0, i - lookback // step)
        wd = ic_dates[ws:i+1]
        fi = {}
        for col in factor_cols:
            vals = [daily_ic[d].get(col, np.nan) for d in wd if col in daily_ic.get(d, {})]
            vals = [v for v in vals if not np.isnan(v)]
            if len(vals) >= 10:
                fi[col] = np.mean(vals)
        if fi:
            ic_history[date] = fi
    all_ic_dates = sorted(ic_history.keys())
    filled = {}
    for date in all_dates:
        cands = [d for d in all_ic_dates if d <= date]
        if cands:
            filled[date] = ic_history[cands[-1]]
    print(f"  ✅ {len(filled)} days ({time.time()-t0:.0f}s)")
    return filled


# ══════════════════════════════════════════════════
# IC-Weighted Engine
# ══════════════════════════════════════════════════

from scipy.stats import rankdata

class ICWeightedEngine(BacktestEngine):
    def __init__(self, ic_history, group_weights, gc_weights=GC_WEIGHTS, **kwargs):
        super().__init__(**kwargs)
        self.ic_history = ic_history
        self.group_weights = group_weights
        self.gc_weights = gc_weights
    
    def _get_scores(self, ranks, date, weights):
        if date not in ranks:
            return None
        ic = self.ic_history.get(date, {})
        rd = ranks[date]
        gs = {}
        for gn, factors in FACTOR_GROUPS.items():
            av = [f for f in factors if f in rd.columns]
            if not av:
                gs[gn] = pd.Series(0., index=rd.index)
                continue
            ic_available = [f for f in av if f in ic]
            if ic_available:
                iv = {f: max(0, ic.get(f, 0)) ** IC_POWER for f in ic_available}
                total = sum(iv.values())
                if total > 0:
                    w = {f: iv[f] / total for f in ic_available}
                    wt = pd.Series(0., index=rd.index)
                    for f in ic_available:
                        wt += w[f] * rd[f]
                    gs[gn] = wt
                else:
                    gs[gn] = rd[av].mean(axis=1)
            else:
                gs[gn] = rd[av].mean(axis=1)
        
        gc = (self.gc_weights.get('fund_growth', 0) * gs.get('fund_growth', 0) +
              self.gc_weights.get('analyst', 0) * gs.get('analyst', 0) +
              self.gc_weights.get('income', 0) * gs.get('income', 0))
        
        final = (self.group_weights['fund_ratio'] * gs.get('fund_ratio', 0) +
                 self.group_weights['gc'] * gc +
                 self.group_weights['qoq'] * gs.get('qoq', 0) +
                 self.group_weights['cf'] * gs.get('cashflow', 0) +
                 self.group_weights.get('insider', 0) * gs.get('insider', 0))
        return final.dropna().sort_values(ascending=False)


# ══════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  🦅 Falcon V0.4.9 — Insider Cluster Buy Validation")
    print("=" * 60)
    t_total = time.time()
    
    # Load tickers
    with open(SPX_PATH) as f:
        spx = json.load(f)
    tickers = spx if isinstance(spx, list) else spx.get('tickers', [])
    print(f"📋 SPX: {len(tickers)} tickers")
    
    # Download insider data
    cache = download_all_insider(tickers)
    
    # Load features & prices
    print("\n📂 Loading data...")
    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = df['date'].astype(str)
    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = prices_df['date'].astype(str)
    prices = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
    
    all_dates = sorted(df['date'].unique())
    five_years_ago = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    sample_dates = [d for d in all_dates if d >= five_years_ago]
    print(f"  Features: {df.shape}, Prices: {prices.shape}, Dates: {len(sample_dates)}")
    
    # Compute insider features
    cluster_features = compute_cluster_buy_features(cache, sample_dates)
    df = build_features_with_insider(df, cluster_features, sample_dates)
    
    # All factors
    all_factors = list(set(f for fg in FACTOR_GROUPS.values() for f in fg))
    
    # Compute ranks
    ranks = compute_ranks(df, all_factors, sample_dates)
    
    # Compute IC
    daily_ic = compute_daily_ic(ranks, prices, all_factors)
    ic_hist = rolling_ic(daily_ic, sorted(ranks.keys()), all_factors, IC_LOOKBACK)
    
    # ══════════════════════════════════════════════════
    # Walk-Forward: All configs
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Walk-Forward Validation")
    print("=" * 60)
    
    results = {}
    
    for name, config in CONFIGS.items():
        print(f"\n  ▶ {name}: weights={config['weights']}, hold={config['hold_days']}, top_n={config['top_n']}")
        
        engine = ICWeightedEngine(
            ic_history=ic_hist,
            group_weights=config['weights'],
            cost=0.001,
            stop_loss=-0.15,
        )
        
        try:
            wf = engine.walk_forward(
                ranks, prices, weights={},
                hold_days=config['hold_days'],
                top_n=config['top_n'],
                train_years=1, test_months=6,
            )
            
            windows = wf.window_details or []
            valid_windows = [w for w in windows if 'sharpe' in w]
            
            results[name] = {
                'sharpe': wf.sharpe,
                'max_dd': wf.max_dd,
                'cagr': wf.cagr,
                'win_rate': wf.win_rate,
                'n_windows': len(valid_windows),
                'windows': windows,
                'warnings': wf.warnings,
                'config': config,
            }
            
            print(f"    Sharpe={wf.sharpe:.3f}  MaxDD={wf.max_dd:.1%}  CAGR={wf.cagr:.1%}  Windows={len(valid_windows)}")
            if wf.warnings:
                for w in wf.warnings:
                    print(f"    ⚠️ {w}")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results[name] = {'error': str(e)}
    
    # ══════════════════════════════════════════════════
    # Comparison
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  📊 COMPARISON")
    print("=" * 60)
    
    baseline_sharpe = results.get('V0.4.6', {}).get('sharpe', 0)
    
    print(f"\n  {'Config':<30} {'Sharpe':>8} {'Delta':>8} {'MaxDD':>8} {'CAGR':>8}")
    print("  " + "-" * 65)
    
    for name, r in results.items():
        if 'error' in r:
            print(f"  {name:<30} {'ERROR':>8}")
            continue
        delta = r['sharpe'] - baseline_sharpe
        delta_pct = (r['sharpe'] / baseline_sharpe - 1) * 100 if baseline_sharpe > 0 else 0
        marker = '✅' if delta > 0.05 * baseline_sharpe else '❌' if delta < 0 else '≈'
        print(f"  {name:<30} {r['sharpe']:>8.3f} {delta:>+7.3f} ({delta_pct:+.1f}%) {r['max_dd']:>7.1%} {r['cagr']:>7.1%} {marker}")
    
    # Window-by-window for best config
    best_name = max(results.keys(), key=lambda k: results[k].get('sharpe', 0) if 'error' not in results[k] else 0)
    best = results.get(best_name, {})
    
    if 'windows' in best and 'windows' in results.get('V0.4.6', {}):
        print(f"\n  Window-by-Window: V0.4.6 vs {best_name}")
        print(f"  {'Window':<35} {'V0.4.6':>8} {best_name:>8} {'Delta':>8}")
        print("  " + "-" * 65)
        
        v046_windows = results['V0.4.6']['windows']
        best_windows = best['windows']
        
        for i in range(min(len(v046_windows), len(best_windows))):
            w46 = v046_windows[i]
            wb = best_windows[i]
            if 'sharpe' in w46 and 'sharpe' in wb:
                d = wb['sharpe'] - w46['sharpe']
                print(f"  {w46.get('period', f'W{i}'):<35} {w46['sharpe']:>8.3f} {wb['sharpe']:>8.3f} {d:>+8.3f}")
    
    # Verdict
    print("\n" + "=" * 60)
    print("  📋 VERDICT")
    print("=" * 60)
    
    if best_name != 'V0.4.6' and 'error' not in best:
        improvement = (best['sharpe'] / baseline_sharpe - 1) * 100
        if improvement > 5:
            print(f"\n  ✅ {best_name} improves over V0.4.6 by {improvement:+.1f}%")
            print(f"  Sharpe: {baseline_sharpe:.3f} → {best['sharpe']:.3f}")
            print(f"  → Recommend upgrading to {best_name}")
        else:
            print(f"\n  ≈ {best_name} marginally different ({improvement:+.1f}%)")
            print(f"  → Not enough improvement to justify change")
    else:
        print(f"\n  ❌ No config beats V0.4.6 by >5%")
        print(f"  → Insider cluster_buy does not add value to Falcon")
    
    elapsed = time.time() - t_total
    print(f"\n⏱️ Total: {elapsed:.0f}s")
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'configs': {k: v.get('config', {}) for k, v in results.items()},
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'windows'} for k, v in results.items()},
        'best': best_name,
        'baseline_sharpe': baseline_sharpe,
        'improvement_pct': (best.get('sharpe', 0) / baseline_sharpe - 1) * 100 if baseline_sharpe > 0 else 0,
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
