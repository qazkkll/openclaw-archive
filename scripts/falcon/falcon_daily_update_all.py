#!/usr/bin/env python3
"""
🦅 Falcon V0.4.6 每日全量更新脚本
=================================
单一入口，更新所有管线数据并同步到dashboard。

流程:
  1. 下载最新价格 (yfinance + Polygon补充)
  2. 下载最新FMP基本面数据 (ratios/metrics/growth)
  3. 重建 features_v02.parquet (技术指标)
  4. 重建 features_v04_1.parquet (PIT因子矩阵)
  5. 计算IC权重 (compute_rolling_ic.py)
  6. 评分 (falcon_score.py)
  7. 同步dashboard
  8. 输出摘要

用法:
  python3 scripts/falcon/falcon_daily_update_all.py
  python3 scripts/falcon/falcon_daily_update_all.py --skip-download  # 跳过下载，只重建
  python3 scripts/falcon/falcon_daily_update_all.py --dry-run        # 只检查不执行
"""
import sys
import os
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# ─── 路径配置 ─────────────────────────────────────────
PROJECT = Path('/home/hermes/.hermes/openclaw-archive')
DATA_DIR = PROJECT / 'data' / 'falcon'
SNAPSHOTS_DIR = PROJECT / 'data' / 'fmp_premium' / 'snapshots'
SCRIPTS_DIR = PROJECT / 'scripts' / 'falcon'

FMP_KEY = os.environ.get('FMP_API_KEY', '185VX9wJgwR7ZwQsLIfEbUzc066hfpLN')
POLY_KEY = os.environ.get('MASSIVE_API_KEY', 'UcBcCFsINWm3l6TnV6UW201ofbYwgI3q')

LOG_FILE = DATA_DIR / 'daily_update.log'


def log(msg, level='INFO'):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] [{level}] {msg}'
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def run_step(name, cmd, timeout=600):
    """运行子步骤，返回(success, output)"""
    log(f'▶ {name}')
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(PROJECT)
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            log(f'✅ {name} ({elapsed:.0f}s)')
            return True, result.stdout
        else:
            log(f'❌ {name} failed (exit {result.returncode})', 'ERROR')
            log(f'   stderr: {result.stderr[-300:]}', 'ERROR')
            return False, result.stderr
    except subprocess.TimeoutExpired:
        log(f'⏰ {name} timeout ({timeout}s)', 'ERROR')
        return False, 'timeout'
    except Exception as e:
        log(f'❌ {name} exception: {e}', 'ERROR')
        return False, str(e)


# ─── Step 1: 下载最新价格 ─────────────────────────────

def download_prices():
    """下载最新价格数据 (yfinance + Polygon补充)"""
    import yfinance as yf
    import requests

    log('📥 Step 1: 下载最新价格数据')

    # 加载universe
    with open(DATA_DIR / 'fmp_balance_sheet.json') as f:
        tickers = sorted(json.load(f).keys())

    # 加载已有数据，找最后日期
    prices_path = DATA_DIR / 'us_prices_daily.parquet'
    if prices_path.exists():
        existing = pd.read_parquet(prices_path)
        existing['date'] = existing['date'].astype(str)
        last_date = existing['date'].max()
        log(f'  已有数据到: {last_date}, {existing["ticker"].nunique()} tickers')
    else:
        existing = pd.DataFrame()
        last_date = '2016-01-01'

    # 增量下载 (最近5天确保覆盖)
    start = (datetime.strptime(last_date, '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')

    if start >= end:
        log('  数据已是最新，跳过下载')
        return True

    # yfinance下载
    all_frames = []
    failed_tickers = []
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, start=start, end=end,
                             auto_adjust=False, group_by='ticker',
                             threads=True, progress=False)
            if data is not None and not data.empty:
                if len(batch) == 1:
                    ticker = batch[0]
                    df = data.reset_index()
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                    df['ticker'] = ticker
                    rename = {'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
                              'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'}
                    df = df.rename(columns=rename)
                    cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                    all_frames.append(df[[c for c in cols if c in df.columns]].dropna(subset=['close']))
                else:
                    for ticker in batch:
                        try:
                            if ticker in data.columns.get_level_values(0):
                                sub = data[ticker].reset_index()
                                sub.columns = [c[0] if isinstance(c, tuple) else c for c in sub.columns]
                                sub['ticker'] = ticker
                                rename = {'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
                                          'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'}
                                sub = sub.rename(columns=rename)
                                cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                                sub = sub[[c for c in cols if c in sub.columns]].dropna(subset=['close'])
                                if not sub.empty:
                                    all_frames.append(sub)
                                else:
                                    failed_tickers.append(ticker)
                            else:
                                failed_tickers.append(ticker)
                        except:
                            failed_tickers.append(ticker)
        except:
            failed_tickers.extend(batch)

    # Polygon补充失败的
    if failed_tickers and POLY_KEY:
        log(f'  Polygon补充: {len(failed_tickers)} tickers')
        for ticker in failed_tickers:
            try:
                url = f'https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?apiKey={POLY_KEY}&limit=5000&adjusted=true&sort=asc'
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    results = r.json().get('results', [])
                    if results:
                        rows = []
                        for rec in results:
                            ts = datetime.fromtimestamp(rec['t'] / 1000)
                            rows.append({
                                'date': ts.strftime('%Y-%m-%d'), 'ticker': ticker,
                                'open': rec.get('o'), 'high': rec.get('h'),
                                'low': rec.get('l'), 'close': rec.get('c'),
                                'adj_close': rec.get('c'), 'volume': rec.get('v'),
                            })
                        all_frames.append(pd.DataFrame(rows))
                time.sleep(0.12)
            except:
                pass

    if not all_frames:
        log('  无新数据下载', 'WARN')
        return True

    new_df = pd.concat(all_frames, ignore_index=True)
    new_df['date'] = pd.to_datetime(new_df['date']).dt.strftime('%Y-%m-%d')

    # 合并
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date', 'ticker'], keep='last')
    else:
        combined = new_df

    combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)
    combined.to_parquet(prices_path, index=False)
    log(f'  ✅ 价格更新: {combined["ticker"].nunique()} tickers, {len(combined)} rows, {combined["date"].min()} to {combined["date"].max()}')
    return True


# ─── Step 2: 下载FMP基本面数据 ────────────────────────

def download_fmp():
    """增量下载FMP ratios/metrics/growth"""
    import urllib.request

    log('📥 Step 2: 下载FMP基本面数据')

    # 加载universe
    with open(DATA_DIR / 'fmp_balance_sheet.json') as f:
        tickers = sorted(json.load(f).keys())

    endpoints = {
        'fmp_ratios_historical': ('ratios', [
            'priceToEarningsRatio', 'priceToBookRatio', 'priceToSalesRatio',
            'priceToFreeCashFlowRatio', 'enterpriseValueMultiple',
            'grossProfitMargin', 'netProfitMargin', 'operatingProfitMargin',
            'ebitdaMargin', 'assetTurnover', 'inventoryTurnover',
            'receivablesTurnover', 'debtToEquityRatio', 'currentRatio',
            'quickRatio', 'financialLeverageRatio',
            'freeCashFlowOperatingCashFlowRatio', 'operatingCashFlowRatio',
            'dividendYieldPercentage', 'dividendPayoutRatio',
        ]),
        'fmp_key_metrics': ('key-metrics', [
            'earningsYield', 'evToEBITDA', 'evToFreeCashFlow', 'evToSales',
            'freeCashFlowYield', 'returnOnEquity', 'returnOnAssets',
            'returnOnCapitalEmployed', 'returnOnInvestedCapital',
            'returnOnTangibleAssets', 'incomeQuality', 'grahamNumber',
            'cashConversionCycle', 'capexToRevenue', 'capexToDepreciation',
            'researchAndDevelopementToRevenue', 'stockBasedCompensationToRevenue',
            'netDebtToEBITDA', 'operatingReturnOnAssets',
        ]),
        'fmp_financial_growth': ('financial-growth', [
            'revenueGrowth', 'grossProfitGrowth', 'ebitgrowth',
            'operatingIncomeGrowth', 'netIncomeGrowth', 'epsdilutedGrowth',
            'freeCashFlowGrowth', 'tenYRevenueGrowthPerShare',
            'fiveYRevenueGrowthPerShare', 'threeYRevenueGrowthPerShare',
            'receivablesGrowth', 'inventoryGrowth', 'assetGrowth',
            'bookValueperShareGrowth', 'debtGrowth',
        ]),
    }

    for out_name, (endpoint, fields) in endpoints.items():
        out_file = SNAPSHOTS_DIR / f'{out_name}.json'

        # 加载已有
        existing = {}
        if out_file.exists():
            with open(out_file) as f:
                existing = json.load(f)

        # 找需要更新的ticker (最近7天没更新的)
        needs_update = []
        for t in tickers:
            if t not in existing or not existing[t]:
                needs_update.append(t)
            else:
                # 检查最新日期
                latest = max((r.get('date', '') for r in existing[t] if isinstance(r, dict)), default='')
                if latest < (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'):
                    needs_update.append(t)

        if not needs_update:
            log(f'  {out_name}: 已最新')
            continue

        log(f'  {out_name}: 更新 {len(needs_update)} tickers')
        data = dict(existing)

        for ticker in needs_update:
            try:
                url = f'https://financialmodelingprep.com/stable/{endpoint}?symbol={ticker}&period=quarter&limit=40&apikey={FMP_KEY}'
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=20) as r:
                    raw = json.loads(r.read())
                if isinstance(raw, list):
                    filtered = []
                    for d in raw:
                        row = {'date': d.get('date', '')}
                        for f in fields:
                            row[f] = d.get(f)
                        filtered.append(row)
                    filtered.sort(key=lambda x: x['date'])
                    data[ticker] = filtered
            except:
                pass
            time.sleep(0.15)

        with open(out_file, 'w') as f:
            json.dump(data, f)
        has = sum(1 for v in data.values() if v)
        log(f'  ✅ {out_name}: {has}/{len(tickers)} tickers')

    return True


# ─── Step 3: 重建features_v02 (技术指标) ──────────────

def rebuild_features_v02():
    """从价格数据重建技术指标矩阵"""
    log('🔧 Step 3: 重建 features_v02.parquet')

    prices = pd.read_parquet(DATA_DIR / 'us_prices_daily.parquet')
    prices['date'] = prices['date'].astype(str)
    tickers = sorted(prices['ticker'].unique())

    features = []
    for ticker in tickers:
        td = prices[prices['ticker'] == ticker].sort_values('date').copy()
        if len(td) < 60:
            continue
        td = td.reset_index(drop=True)
        c = td['close']
        v = td['volume'].astype(float)

        td['ma5'] = c.rolling(5).mean()
        td['ma20'] = c.rolling(20).mean()
        td['ma60'] = c.rolling(60).mean()
        td['ma_bias20'] = (c - td['ma20']) / td['ma20']
        td['ma_align'] = ((td['ma5'] > td['ma20']) & (td['ma20'] > td['ma60'])).astype(float)
        td['ma_cross_5_20'] = (td['ma5'] > td['ma20']).astype(float)
        td['ma_cross_20_60'] = (td['ma20'] > td['ma60']).astype(float)
        h60 = td['high'].rolling(60).max()
        l60 = td['low'].rolling(60).min()
        td['price_position'] = (c - l60) / (h60 - l60 + 1e-10)
        for n in [1, 5, 10, 20, 30, 60, 90]:
            td[f'ret{n}'] = c.pct_change(n)
        td['momentum_6m'] = c.pct_change(126)
        td['momentum_1m'] = c.pct_change(21)
        td['mom_divergence'] = td['momentum_6m'] - td['momentum_1m']
        td['trend_accel'] = td['momentum_1m'].diff(5)
        td['vol20'] = c.pct_change().rolling(20).std() * np.sqrt(252)
        td['vol5'] = c.pct_change().rolling(5).std() * np.sqrt(252)
        td['vol_ratio'] = td['vol5'] / td['vol20'].clip(lower=1e-10)
        td['vol_change'] = td['vol20'].pct_change(5)
        td['vol_regime'] = (td['vol20'] > td['vol20'].rolling(60).mean()).astype(float)
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        td['rsi14'] = 100 - 100 / (1 + gain / loss.clip(lower=1e-10))
        td['rsi_change'] = td['rsi14'].diff(5)
        td['rsi_zone'] = pd.cut(td['rsi14'], bins=[0, 30, 70, 100], labels=[0, 1, 2]).astype(float)
        e12 = c.ewm(span=12).mean()
        e26 = c.ewm(span=26).mean()
        td['macd'] = e12 - e26
        td['macd_signal'] = td['macd'].ewm(span=9).mean()
        td['macd_hist'] = td['macd'] - td['macd_signal']
        td['macd_roc'] = td['macd_hist'].diff(3)
        bb = c.rolling(20).mean()
        bs = c.rolling(20).std()
        td['bb_std'] = bs
        td['bb_width'] = 2 * bs / bb.clip(lower=1e-10)
        td['bb_pos'] = (c - bb + bs) / (2 * bs + 1e-10)
        td['ret_quality'] = c.pct_change().rolling(20).apply(lambda x: (x > 0).mean(), raw=True)
        td['range_ratio'] = (td['high'] - td['low']) / c.clip(lower=1e-10)
        td['avg_body'] = abs(c - td['open']) / c.clip(lower=1e-10)
        td['vwap_drift'] = (c - td['ma20']) / td['ma20'].clip(lower=1e-10)
        td['dd_60'] = c / c.rolling(60).max() - 1
        td['ud_vol_ratio'] = (v * (c > c.shift(1)).astype(float)).rolling(20).sum() / \
                              (v * (c <= c.shift(1)).astype(float)).rolling(20).sum().clip(lower=1)
        td['beta'] = 1.0
        td['vwap'] = np.nan
        features.append(td)

    feat = pd.concat(features, ignore_index=True)
    cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'vwap',
            'ma5', 'ma20', 'ma60', 'ma_bias20', 'ma_align', 'ma_cross_5_20', 'ma_cross_20_60',
            'price_position', 'ret1', 'ret5', 'ret10', 'ret20', 'ret30', 'ret60', 'ret90',
            'momentum_6m', 'momentum_1m', 'mom_divergence', 'trend_accel',
            'vol20', 'vol5', 'vol_ratio', 'vol_change', 'vol_regime',
            'rsi14', 'rsi_change', 'rsi_zone', 'macd', 'macd_signal', 'macd_hist', 'macd_roc',
            'bb_std', 'bb_width', 'bb_pos', 'ret_quality', 'range_ratio', 'avg_body', 'vwap_drift',
            'dd_60', 'ud_vol_ratio', 'beta']
    feat = feat[[c for c in cols if c in feat.columns]]
    feat.to_parquet(DATA_DIR / 'features_v02.parquet', index=False)
    log(f'  ✅ features_v02: {len(feat):,} rows, {feat["ticker"].nunique()} tickers')
    return True


# ─── Step 4: 重建features_v04_1 (PIT因子) ────────────

def rebuild_features_v041():
    """运行 build_features_v041.py"""
    ok, out = run_step('Step 4: 重建 features_v04_1.parquet',
                       f'{sys.executable} {SCRIPTS_DIR}/build_features_v041.py',
                       timeout=300)
    return ok


# ─── Step 5: 计算IC权重 ───────────────────────────────

def compute_ic():
    """运行 compute_rolling_ic.py"""
    ok, out = run_step('Step 5: 计算IC权重',
                       f'{sys.executable} {SCRIPTS_DIR}/compute_rolling_ic.py',
                       timeout=120)
    return ok


# ─── Step 6: 评分 ─────────────────────────────────────

def run_scoring():
    """运行 falcon_score.py"""
    ok, out = run_step('Step 6: 评分',
                       f'{sys.executable} {SCRIPTS_DIR}/falcon_score.py --universe spx',
                       timeout=60)
    return ok


# ─── Step 7: 同步Dashboard ────────────────────────────

def sync_dashboard():
    """运行 dashboard_refresh.py"""
    ok, out = run_step('Step 7: 同步Dashboard',
                       f'{sys.executable} {SCRIPTS_DIR}/dashboard_refresh.py',
                       timeout=60)
    return ok


# ─── 主流程 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Falcon V0.4.6 每日全量更新')
    parser.add_argument('--skip-download', action='store_true', help='跳过数据下载')
    parser.add_argument('--dry-run', action='store_true', help='只检查不执行')
    args = parser.parse_args()

    log('=' * 60)
    log('🦅 Falcon V0.4.6 每日全量更新开始')
    log('=' * 60)

    t0 = time.time()
    results = {}

    if args.dry_run:
        log('DRY RUN - 只检查不执行')
        # 检查所有文件状态
        for f in ['features_v04_1.parquet', 'us_prices_daily.parquet', 'factor_ic_weights.json']:
            p = DATA_DIR / f
            log(f'  {f}: {"EXISTS" if p.exists() else "MISSING"} ({p.stat().st_size / 1024 / 1024:.1f}MB)' if p.exists() else f'  {f}: MISSING')
        return

    steps = [
        ('1. 下载价格', download_prices, not args.skip_download),
        ('2. 下载FMP', download_fmp, not args.skip_download),
        ('3. 重建技术指标', rebuild_features_v02, True),
        ('4. 重建PIT因子', rebuild_features_v041, True),
        ('5. 计算IC', compute_ic, True),
        ('6. 评分', run_scoring, True),
        ('7. 同步Dashboard', sync_dashboard, True),
    ]

    for name, func, should_run in steps:
        if not should_run:
            log(f'⏭️ 跳过: {name}')
            results[name] = 'SKIPPED'
            continue

        try:
            ok = func()
            results[name] = 'OK' if ok else 'FAIL'
            if not ok:
                log(f'⚠️ {name} 失败，继续后续步骤', 'WARN')
        except Exception as e:
            log(f'❌ {name} 异常: {e}', 'ERROR')
            results[name] = 'ERROR'

    elapsed = time.time() - t0

    # 输出摘要
    log('=' * 60)
    log('📋 更新摘要')
    for name, status in results.items():
        icon = {'OK': '✅', 'FAIL': '❌', 'SKIPPED': '⏭️', 'ERROR': '💥'}.get(status, '?')
        log(f'  {icon} {name}: {status}')

    # 读取评分结果
    score_files = sorted(DATA_DIR.glob('falcon_v046_scored_*.json'))
    if score_files:
        with open(score_files[-1]) as f:
            score = json.load(f)
        top = score.get('top_n', [])[:5]
        if top:
            log('\n🏆 Top-5:')
            for t in top:
                log(f'  {t.get("ticker", "?")} Score={t.get("score", 0):.3f}')

    log(f'\n⏱️ 总耗时: {elapsed / 60:.1f} 分钟')
    log('=' * 60)

    # 保存运行记录
    run_record = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'results': results,
        'success': all(v in ('OK', 'SKIPPED') for v in results.values()),
    }
    with open(DATA_DIR / 'daily_update_record.json', 'w') as f:
        json.dump(run_record, f, indent=2)


if __name__ == '__main__':
    main()
