#!/usr/bin/env python3
"""
🦅 Falcon Alternative Data Experiment
======================================
测试 News Sentiment / Insider Trading / 13F 加入 V0.4.6 后的 WF Sharpe 变化。

与 V0.4.6 使用完全相同的回测模式:
  - backtest_engine.py Walk-Forward
  - train_years=2, test_months=6
  - hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15

实验设计:
  1. V0.4.6 Baseline: 53因子, fund_ratio=0.45/gc=0.20/qoq=0.20/cf=0.15
  2. +News: 新增 news_sentiment 因子组 (5%~20% 权重)
  3. +Insider: 新增 insider_signal 因子组 (5%~15% 权重)
  4. +News+Insider: 两者同时加入

用法:
    python3 scripts/falcon/alt_data_experiment.py
    python3 scripts/falcon/alt_data_experiment.py --skip-ic  # 跳过IC分析，直接跑WF
"""

import sys, json, time, warnings, argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.stats import rankdata

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(WORKSPACE / "scripts" / "falcon"))

from backtest_engine import BacktestEngine, DataQualityError

# ═══════════════════════════════════════════════════
#  路径
# ═══════════════════════════════════════════════════
DATA_DIR = WORKSPACE / "data" / "falcon"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
NEWS_FEATURES_PATH = DATA_DIR / "news_features_v04.parquet"
INSIDER_CACHE_PATH = DATA_DIR / "insider_trading" / "openinsider_cache.json"
OUTPUT_PATH = DATA_DIR / "alt_data_experiment_results.json"

# ═══════════════════════════════════════════════════
#  V0.4.6 因子定义 (与 falcon_score.py 一致)
# ═══════════════════════════════════════════════════

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
}

FLIP_FACTORS = {
    'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
    'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
    'r_debtToEquityRatio', 'r_financialLeverageRatio', 'r_inventoryTurnover',
    'c_capex_intensity', 'g_debtGrowth', 'g_receivablesGrowth', 'g_inventoryGrowth',
    'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    'a_eps_revision', 'a_revenue_revision',
}

# V0.4.6 顶层权重
V046_WEIGHTS = {
    "fund_ratio": 0.45,
    "gc_baseline": 0.20,  # growth_composite
    "qoq": 0.20,
    "cashflow": 0.15,
}

# Growth Composite 子权重
GC_WEIGHTS = {"fund_growth": 0.60, "analyst": 0.25, "income": 0.15}


# ═══════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════

def load_data():
    """加载特征和价格数据。"""
    print("📂 加载数据...")
    t0 = time.time()

    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = df['date'].astype(str)
    print(f"  ✅ Features: {df.shape[0]}行 × {df.shape[1]}列, {df['ticker'].nunique()}只")

    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = prices_df['date'].astype(str)
    price_pivot = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
    print(f"  ✅ Prices: {price_pivot.shape[0]}天 × {price_pivot.shape[1]}只")
    print(f"  ⏱️ 加载耗时: {time.time()-t0:.1f}秒")
    return df, price_pivot


def load_news_features():
    """加载月频新闻情绪特征, 前填到日频。"""
    if not NEWS_FEATURES_PATH.exists():
        print("  ⚠️ news_features_v04.parquet 不存在")
        return None

    nf = pd.read_parquet(NEWS_FEATURES_PATH)
    nf['date'] = pd.to_datetime(nf['date'])
    nf = nf.sort_values(['ticker', 'date'])

    # 为每个 ticker 创建日频索引, 前填月频数据
    all_tickers = nf['ticker'].unique()
    date_range = pd.date_range(nf['date'].min(), nf['date'].max(), freq='D')
    
    expanded_frames = []
    news_cols = [c for c in nf.columns if c.startswith('news_')]
    
    for ticker in all_tickers:
        tdf = nf[nf['ticker'] == ticker].set_index('date')[news_cols]
        # 重采样到日频, 前填
        tdf = tdf.reindex(date_range).ffill()
        tdf['ticker'] = ticker
        tdf['date'] = tdf.index.strftime('%Y-%m-%d')
        expanded_frames.append(tdf.reset_index(drop=True))

    result = pd.concat(expanded_frames, ignore_index=True)
    result = result.dropna(subset=news_cols[:1])  # 至少有一个新闻特征非空
    print(f"  ✅ News features: {len(result)}行, {result['ticker'].nunique()}只, "
          f"{result['date'].min()} → {result['date'].max()}")
    return result


def load_insider_features():
    """加载 insider trading 数据, 计算月频信号。"""
    if not INSIDER_CACHE_PATH.exists():
        print("  ⚠️ openinsider_cache.json 不存在")
        return None

    with open(INSIDER_CACHE_PATH) as f:
        raw = json.load(f)

    # 转为 DataFrame
    records = []
    for ticker, trades in raw.items():
        for t in trades:
            records.append({
                'ticker': ticker,
                'filing_date': t.get('filing_date', ''),
                'trade_date': t.get('trade_date', ''),
                'is_buy': t.get('is_buy', False),
                'is_sell': t.get('is_sell', False),
                'is_ceo_cfo': t.get('is_ceo_cfo', False),
                'value': t.get('value', 0),
                'qty': t.get('qty', 0),
            })

    if not records:
        print("  ⚠️ insider 数据为空")
        return None

    idf = pd.DataFrame(records)
    idf['filing_date'] = pd.to_datetime(idf['filing_date'], errors='coerce')
    idf = idf.dropna(subset=['filing_date'])

    # 按(ticker, 月)聚合: 计算 cluster_buy, net_buy_ratio 等信号
    idf['month'] = idf['filing_date'].dt.to_period('M')

    grouped = idf.groupby(['ticker', 'month'])

    # cluster_buy: 3+ 不同内部人在同月买入
    def cluster_buy_count(g):
        buys = g[g['is_buy']]
        if len(buys) == 0:
            return 0
        n_insiders = buys['is_ceo_cfo'].count()  # 近似: 不同交易笔数
        return 1 if n_insiders >= 3 else 0

    # net_buy_ratio: (买-卖) / (买+卖)
    def net_buy_ratio(g):
        n_buy = g['is_buy'].sum()
        n_sell = g['is_sell'].sum()
        total = n_buy + n_sell
        if total == 0:
            return 0
        return (n_buy - n_sell) / total

    # ceo_cfo_buy: CEO/CFO 是否买入
    def ceo_cfo_buy(g):
        buys = g[(g['is_buy']) & (g['is_ceo_cfo'])]
        return 1 if len(buys) > 0 else 0

    # total_buy_value: 买入总金额
    def total_buy_value(g):
        buys = g[g['is_buy']]
        return buys['value'].sum()

    features = grouped.apply(lambda g: pd.Series({
        'insider_cluster_buy': cluster_buy_count(g),
        'insider_net_buy_ratio': net_buy_ratio(g),
        'insider_ceo_cfo_buy': ceo_cfo_buy(g),
        'insider_buy_value_log': np.log1p(total_buy_value(g)),
    })).reset_index()

    # 转月频日期为月末
    features['date'] = features['month'].dt.to_timestamp() + pd.offsets.MonthEnd(0)
    features['date'] = features['date'].dt.strftime('%Y-%m-%d')

    # 前填到日频 (与 news 类似)
    insider_cols = ['insider_cluster_buy', 'insider_net_buy_ratio',
                    'insider_ceo_cfo_buy', 'insider_buy_value_log']
    all_tickers = features['ticker'].unique()
    date_range = pd.date_range(features['date'].min(), features['date'].max(), freq='D')

    expanded = []
    for ticker in all_tickers:
        tdf = features[features['ticker'] == ticker].set_index('date')[insider_cols]
        tdf = tdf.reindex(date_range.strftime('%Y-%m-%d')).ffill()
        tdf['ticker'] = ticker
        tdf['date'] = tdf.index
        expanded.append(tdf.reset_index(drop=True))

    result = pd.concat(expanded, ignore_index=True)
    result = result.dropna(subset=insider_cols[:1])

    print(f"  ✅ Insider features: {len(result)}行, {result['ticker'].nunique()}只, "
          f"{result['date'].min()} → {result['date'].max()}")
    return result


# ═══════════════════════════════════════════════════
#  截面排名计算 (与 v044_final_validation.py 一致)
# ═══════════════════════════════════════════════════

def compute_cross_sectional_ranks(df, factor_cols):
    """计算截面百分位排名。"""
    print("📊 计算截面百分位排名...")
    t0 = time.time()

    dates = sorted(df['date'].unique())
    ranks = {}

    for date in dates:
        day_df = df[df['date'] == date].copy()
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

            ranks_raw = np.full_like(vals, np.nan)
            if valid.sum() > 0:
                ranks_raw[valid] = rankdata(vals[valid], method='average') / valid.sum()

            if col in FLIP_FACTORS:
                mask = ~np.isnan(ranks_raw)
                ranks_raw[mask] = 1.0 - ranks_raw[mask]

            rank_df[col] = ranks_raw

        ranks[date] = rank_df

    elapsed = time.time() - t0
    print(f"  ✅ {len(ranks)}天排名计算完成 ({elapsed:.0f}秒)")
    return ranks


def compute_group_ranks(ranks, factor_groups):
    """将因子组的排名合并为组级排名(等权平均)。"""
    print("📊 计算因子组排名...")
    for date in list(ranks.keys()):
        df = ranks[date]
        for group_name, factors in factor_groups.items():
            available = [f for f in factors if f in df.columns]
            if available:
                df[group_name] = df[available].mean(axis=1)
        ranks[date] = df
    print(f"  ✅ 因子组排名已添加: {list(factor_groups.keys())}")
    return ranks


def add_growth_composite(ranks):
    """添加 growth_composite (gc_baseline)。"""
    for date in ranks:
        df = ranks[date]
        try:
            df['gc_baseline'] = (
                df.get('fund_growth', 0) * GC_WEIGHTS['fund_growth'] +
                df.get('analyst', 0) * GC_WEIGHTS['analyst'] +
                df.get('income', 0) * GC_WEIGHTS['income']
            )
        except Exception:
            df['gc_baseline'] = np.nan
        ranks[date] = df
    print("  ✅ gc_baseline (growth_composite) 已添加")
    return ranks


# ═══════════════════════════════════════════════════
#  News 因子集成
# ═══════════════════════════════════════════════════

def add_news_to_ranks(ranks, news_df):
    """将新闻情绪因子加入 ranks_dict。

    策略: 用 news_pos_ratio - news_neg_ratio 作为净情绪信号,
    再加上 news_article_count 作为关注度信号。
    """
    if news_df is None:
        print("  ⚠️ 无新闻数据, 跳过")
        return ranks

    print("📊 添加新闻情绪因子到 ranks...")
    
    # 构建 ticker→date→features 的查找结构
    news_lookup = {}
    for _, row in news_df.iterrows():
        t = row['ticker']
        d = row['date']
        if t not in news_lookup:
            news_lookup[t] = {}
        # 净情绪 = 正面比例 - 负面比例
        net_sent = row.get('news_pos_ratio', 0.5) - row.get('news_neg_ratio', 0.5)
        article_count = row.get('news_article_count', 0)
        confidence = row.get('news_confidence_avg', 0.5)
        news_lookup[t][d] = {
            'net_sentiment': net_sent,
            'article_count': article_count,
            'confidence': confidence,
        }

    enriched = 0
    for date in sorted(ranks.keys()):
        rank_df = ranks[date]
        
        net_sents = {}
        article_counts = {}
        confidences = {}
        
        for ticker in rank_df.index:
            if ticker in news_lookup:
                # 找最近7天的新闻
                recent_sents = []
                recent_counts = []
                recent_confs = []
                for offset in range(7):
                    d = pd.Timestamp(date) - pd.Timedelta(days=offset)
                    d_str = d.strftime('%Y-%m-%d')
                    if d_str in news_lookup[ticker]:
                        info = news_lookup[ticker][d_str]
                        recent_sents.append(info['net_sentiment'])
                        recent_counts.append(info['article_count'])
                        recent_confs.append(info['confidence'])
                
                if recent_sents:
                    # 置信度加权平均
                    weights = np.array(recent_confs)
                    if weights.sum() > 0:
                        net_sents[ticker] = np.average(recent_sents, weights=weights)
                    else:
                        net_sents[ticker] = np.mean(recent_sents)
                    article_counts[ticker] = np.mean(recent_counts)
                    confidences[ticker] = np.mean(recent_confs)

        if len(net_sents) > 50:
            # 净情绪 (越高越好, 不翻转)
            sent_series = pd.Series(net_sents)
            rank_df['news_net_sentiment'] = sent_series.rank(pct=True)
            
            # 文章数量 (关注度, 越高越好)
            count_series = pd.Series(article_counts)
            rank_df['news_attention'] = count_series.rank(pct=True)
            
            # 综合新闻因子
            rank_df['news_composite'] = (
                rank_df['news_net_sentiment'] * 0.7 +
                rank_df['news_attention'] * 0.3
            )
            enriched += 1
        else:
            rank_df['news_net_sentiment'] = 0.5
            rank_df['news_attention'] = 0.5
            rank_df['news_composite'] = 0.5

        ranks[date] = rank_df

    print(f"  ✅ 新闻因子加入: {enriched}/{len(ranks)} 天有足够数据")
    return ranks


# ═══════════════════════════════════════════════════
#  Insider 因子集成
# ═══════════════════════════════════════════════════

def add_insider_to_ranks(ranks, insider_df):
    """将 insider trading 因子加入 ranks_dict。

    由于覆盖率低(19.7%), 对无数据的 ticker 设为中性(0.5)。
    """
    if insider_df is None:
        print("  ⚠️ 无 insider 数据, 跳过")
        return ranks

    print("📊 添加 insider 因子到 ranks...")

    # 构建查找结构
    insider_lookup = {}
    for _, row in insider_df.iterrows():
        t = row['ticker']
        d = row['date']
        if t not in insider_lookup:
            insider_lookup[t] = {}
        insider_lookup[t][d] = {
            'cluster_buy': row.get('insider_cluster_buy', 0),
            'net_buy_ratio': row.get('insider_net_buy_ratio', 0),
            'ceo_cfo_buy': row.get('insider_ceo_cfo_buy', 0),
            'buy_value_log': row.get('insider_buy_value_log', 0),
        }

    enriched = 0
    for date in sorted(ranks.keys()):
        rank_df = ranks[date]

        cluster_buys = {}
        net_ratios = {}
        ceo_buys = {}

        for ticker in rank_df.index:
            if ticker in insider_lookup:
                # 找最近90天的 insider 数据 (季度窗口)
                latest_cluster = 0
                latest_net = 0
                latest_ceo = 0
                found = False
                for offset in range(90):
                    d = pd.Timestamp(date) - pd.Timedelta(days=offset)
                    d_str = d.strftime('%Y-%m-%d')
                    if d_str in insider_lookup[ticker]:
                        info = insider_lookup[ticker][d_str]
                        latest_cluster = info['cluster_buy']
                        latest_net = info['net_buy_ratio']
                        latest_ceo = info['ceo_cfo_buy']
                        found = True
                        break  # 取最近一天的
                
                if found:
                    cluster_buys[ticker] = latest_cluster
                    net_ratios[ticker] = latest_net
                    ceo_buys[ticker] = latest_ceo

        # 对有数据的 ticker 排名, 无数据的设为 0.5 (中性)
        if len(cluster_buys) > 10:
            cs = pd.Series(cluster_buys)
            ns = pd.Series(net_ratios)
            cb = pd.Series(ceo_buys)
            
            # cluster_buy: 二值(0/1), rank后 0→0.5, 1→较高
            rank_df['insider_cluster'] = cs.rank(pct=True).reindex(rank_df.index).fillna(0.5)
            # net_buy_ratio: 连续值
            rank_df['insider_net_ratio'] = ns.rank(pct=True).reindex(rank_df.index).fillna(0.5)
            # ceo_cfo_buy: 二值
            rank_df['insider_ceo'] = cb.rank(pct=True).reindex(rank_df.index).fillna(0.5)
            
            # 综合 insider 因子
            rank_df['insider_composite'] = (
                rank_df['insider_cluster'] * 0.5 +
                rank_df['insider_net_ratio'] * 0.3 +
                rank_df['insider_ceo'] * 0.2
            )
            enriched += 1
        else:
            rank_df['insider_cluster'] = 0.5
            rank_df['insider_net_ratio'] = 0.5
            rank_df['insider_ceo'] = 0.5
            rank_df['insider_composite'] = 0.5

        ranks[date] = rank_df

    print(f"  ✅ Insider 因子加入: {enriched}/{len(ranks)} 天有足够数据")
    return ranks


# ═══════════════════════════════════════════════════
#  Walk-Forward 回测
# ═══════════════════════════════════════════════════

def run_walk_forward(ranks, prices, weights, label="",
                     train_years=2, test_months=6,
                     hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15):
    """运行 Walk-Forward 回测。"""
    engine = BacktestEngine(cost=cost, stop_loss=stop_loss)
    
    try:
        result = engine.walk_forward(
            ranks, prices, weights,
            hold_days=hold_days, top_n=top_n,
            train_years=train_years, test_months=test_months,
        )
        return result
    except Exception as e:
        print(f"  ❌ {label} 失败: {e}")
        return None


def compute_combined_scores(ranks, date, weights):
    """计算组合分数 (用于 IC 分析)。"""
    if date in ranks:
        r = ranks[date]
    else:
        rank_dates = sorted(ranks.keys())
        candidates = [d for d in rank_dates if d <= date]
        if not candidates:
            return None
        r = ranks[candidates[-1]]

    available = [f for f in weights if f in r.columns and weights[f] > 0]
    if not available:
        return None
    combined = pd.Series(0.0, index=r.index)
    for f in available:
        combined = combined + weights[f] * r[f]
    return combined.dropna().sort_values(ascending=False)


def compute_factor_ic(ranks, prices, factor_name, forward_days=30):
    """计算单个因子的日频 IC (Spearman rank correlation)。"""
    dates = sorted(ranks.keys())
    ics = []
    
    for i, date in enumerate(dates):
        if factor_name not in ranks[date].columns:
            continue
        
        factor_vals = ranks[date][factor_name].dropna()
        if len(factor_vals) < 30:
            continue
        
        # 前瞻收益
        price_dates = sorted(prices.index.astype(str))
        future_candidates = [d for d in price_dates if d > date]
        if len(future_candidates) < forward_days:
            continue
        future_date = future_candidates[min(forward_days-1, len(future_candidates)-1)]
        
        if date not in prices.index or future_date not in prices.index:
            continue
        
        fwd_ret = prices.loc[future_date] / prices.loc[date] - 1
        common = factor_vals.index.intersection(fwd_ret.dropna().index)
        if len(common) < 30:
            continue
        
        from scipy.stats import spearmanr
        ic, _ = spearmanr(factor_vals[common], fwd_ret[common])
        if not np.isnan(ic):
            ics.append(ic)
    
    if len(ics) < 5:
        return None
    
    return {
        'mean_ic': float(np.mean(ics)),
        'std_ic': float(np.std(ics)),
        'icir': float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else 0,
        'n_periods': len(ics),
        'positive_ratio': float(np.mean([1 for ic in ics if ic > 0])),
    }


# ═══════════════════════════════════════════════════
#  实验矩阵
# ═══════════════════════════════════════════════════

def build_experiment_configs():
    """构建实验配置矩阵。"""
    configs = []
    
    # 1. V0.4.6 Baseline
    configs.append({
        'name': 'V0.4.6 Baseline',
        'weights': dict(V046_WEIGHTS),
        'description': '原始53因子, 无替代数据',
    })
    
    # 2. +News 5%/10%/15%/20%
    for pct in [0.05, 0.10, 0.15, 0.20]:
        remaining = 1.0 - pct
        configs.append({
            'name': f'+News {pct:.0%}',
            'weights': {
                'fund_ratio': 0.45 * remaining,
                'gc_baseline': 0.20 * remaining,
                'qoq': 0.20 * remaining,
                'cashflow': 0.15 * remaining,
                'news_composite': pct,
            },
            'description': f'新闻情绪 {pct:.0%} 权重',
        })
    
    # 3. +Insider 5%/10%/15%
    for pct in [0.05, 0.10, 0.15]:
        remaining = 1.0 - pct
        configs.append({
            'name': f'+Insider {pct:.0%}',
            'weights': {
                'fund_ratio': 0.45 * remaining,
                'gc_baseline': 0.20 * remaining,
                'qoq': 0.20 * remaining,
                'cashflow': 0.15 * remaining,
                'insider_composite': pct,
            },
            'description': f'Insider交易 {pct:.0%} 权重',
        })
    
    # 4. +News +Insider 组合
    for news_pct in [0.05, 0.10]:
        for ins_pct in [0.05, 0.10]:
            total_alt = news_pct + ins_pct
            remaining = 1.0 - total_alt
            configs.append({
                'name': f'+News {news_pct:.0%} +Insider {ins_pct:.0%}',
                'weights': {
                    'fund_ratio': 0.45 * remaining,
                    'gc_baseline': 0.20 * remaining,
                    'qoq': 0.20 * remaining,
                    'cashflow': 0.15 * remaining,
                    'news_composite': news_pct,
                    'insider_composite': ins_pct,
                },
                'description': f'新闻 {news_pct:.0%} + Insider {ins_pct:.0%}',
            })
    
    # 5. News 子因子拆解 (5% 权重下哪个最强)
    for sub_factor in ['news_net_sentiment', 'news_attention']:
        pct = 0.05
        remaining = 1.0 - pct
        configs.append({
            'name': f'+{sub_factor} {pct:.0%}',
            'weights': {
                'fund_ratio': 0.45 * remaining,
                'gc_baseline': 0.20 * remaining,
                'qoq': 0.20 * remaining,
                'cashflow': 0.15 * remaining,
                sub_factor: pct,
            },
            'description': f'{sub_factor} 单独测试',
        })
    
    # 6. Insider 子因子拆解
    for sub_factor in ['insider_cluster', 'insider_net_ratio', 'insider_ceo']:
        pct = 0.05
        remaining = 1.0 - pct
        configs.append({
            'name': f'+{sub_factor} {pct:.0%}',
            'weights': {
                'fund_ratio': 0.45 * remaining,
                'gc_baseline': 0.20 * remaining,
                'qoq': 0.20 * remaining,
                'cashflow': 0.15 * remaining,
                sub_factor: pct,
            },
            'description': f'{sub_factor} 单独测试',
        })
    
    return configs


# ═══════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Falcon Alternative Data Experiment")
    parser.add_argument("--skip-ic", action="store_true", help="跳过IC分析")
    parser.add_argument("--skip-insider", action="store_true", help="跳过insider实验")
    parser.add_argument("--skip-news", action="store_true", help="跳过news实验")
    parser.add_argument("--quick", action="store_true", help="快速模式: 只跑baseline和几个关键配置")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 80)
    print("🦅 Falcon Alternative Data Experiment")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ── 加载数据 ──
    df, price_pivot = load_data()

    # ── 加载替代数据 ──
    news_df = None
    insider_df = None
    
    if not args.skip_news:
        news_df = load_news_features()
    
    if not args.skip_insider:
        insider_df = load_insider_features()

    # ── 计算 V0.4.6 基础排名 ──
    all_factor_cols = []
    for factors in FACTOR_GROUPS.values():
        all_factor_cols.extend(factors)

    ranks = compute_cross_sectional_ranks(df, all_factor_cols)
    ranks = compute_group_ranks(ranks, FACTOR_GROUPS)
    ranks = add_growth_composite(ranks)

    # ── 添加替代数据因子 ──
    if news_df is not None:
        ranks = add_news_to_ranks(ranks, news_df)
    
    if insider_df is not None:
        ranks = add_insider_to_ranks(ranks, insider_df)

    # ── 构建实验配置 ──
    configs = build_experiment_configs()
    
    if args.quick:
        # 快速模式: 只跑 baseline + News 10% + Insider 10% + News 10% + Insider 5%
        quick_names = {'V0.4.6 Baseline', '+News 10%', '+Insider 10%', '+News 10% +Insider 5%'}
        configs = [c for c in configs if c['name'] in quick_names]

    print(f"\n📊 实验配置: {len(configs)} 个")
    for c in configs:
        print(f"  • {c['name']}: {c['description']}")

    # ── 因子 IC 分析 ──
    ic_results = {}
    if not args.skip_ic:
        print("\n" + "=" * 80)
        print("📊 因子 IC 分析 (30天前瞻)")
        print("=" * 80)
        
        alt_factors = ['news_net_sentiment', 'news_attention', 'news_composite',
                       'insider_cluster', 'insider_net_ratio', 'insider_ceo', 'insider_composite']
        
        for factor in alt_factors:
            ic = compute_factor_ic(ranks, price_pivot, factor, forward_days=30)
            if ic:
                ic_results[factor] = ic
                print(f"  {factor:30} IC={ic['mean_ic']:+.4f}  ICIR={ic['icir']:+.3f}  "
                      f"n={ic['n_periods']}  +ratio={ic['positive_ratio']:.0%}")
            else:
                print(f"  {factor:30} ⚠️ 数据不足")

    # ── Walk-Forward 回测 ──
    print("\n" + "=" * 80)
    print("📊 Walk-Forward 回测")
    print("=" * 80)

    results = []
    baseline_sharpe = None

    for ci, config in enumerate(configs):
        name = config['name']
        weights = config['weights']
        
        print(f"\n[{ci+1}/{len(configs)}] {name}")
        print(f"  权重: { {k: f'{v:.2f}' for k, v in weights.items()} }")
        
        result = run_walk_forward(ranks, price_pivot, weights, label=name)
        
        if result:
            window_details = result.window_details or []
            valid_windows = [w for w in window_details if 'sharpe' in w]
            
            entry = {
                'name': name,
                'description': config['description'],
                'weights': {k: round(v, 4) for k, v in weights.items()},
                'sharpe': result.sharpe,
                'max_dd': result.max_dd,
                'cagr': result.cagr,
                'win_rate': result.win_rate,
                'total_return': result.total_return,
                'n_trades': result.n_trades,
                'n_windows': len(valid_windows),
                'window_sharpes': [w['sharpe'] for w in valid_windows],
            }
            results.append(entry)
            
            if 'Baseline' in name:
                baseline_sharpe = result.sharpe
            
            # 对比 baseline
            delta = ""
            if baseline_sharpe and baseline_sharpe > 0 and 'Baseline' not in name:
                pct_change = (result.sharpe - baseline_sharpe) / baseline_sharpe * 100
                delta = f"  {'📈' if pct_change > 0 else '📉'} {pct_change:+.1f}% vs baseline"
            
            print(f"  ✅ Sharpe={result.sharpe:.3f}  MaxDD={result.max_dd:.1%}  "
                  f"CAGR={result.cagr:.1%}  WR={result.win_rate:.0%}  "
                  f"Trades={result.n_trades}{delta}")
            
            # 窗口详情
            for w in valid_windows:
                print(f"    W{w['index']}: {w['period']} Sharpe={w['sharpe']:.2f} "
                      f"MaxDD={w['max_dd']:.1%}")
        else:
            results.append({
                'name': name,
                'description': config['description'],
                'error': '回测失败',
            })

    # ── 汇总 ──
    print("\n" + "=" * 80)
    print("📊 实验结果汇总")
    print("=" * 80)

    # 按 Sharpe 排序
    valid_results = [r for r in results if 'sharpe' in r]
    valid_results.sort(key=lambda x: x['sharpe'], reverse=True)

    print(f"\n{'排名':>4} {'配置':35} {'Sharpe':>8} {'MaxDD':>8} {'CAGR':>8} {'WR':>6} {'vs Base':>10}")
    print("-" * 90)
    
    for rank, r in enumerate(valid_results, 1):
        delta = ""
        if baseline_sharpe and baseline_sharpe > 0:
            pct = (r['sharpe'] - baseline_sharpe) / baseline_sharpe * 100
            delta = f"{pct:+.1f}%"
        
        print(f"{rank:>4} {r['name']:35} {r['sharpe']:8.3f} {r['max_dd']:7.1%} "
              f"{r['cagr']:7.1%} {r['win_rate']:5.0%} {delta:>10}")

    # 最佳配置
    if valid_results:
        best = valid_results[0]
        baseline = next((r for r in valid_results if 'Baseline' in r['name']), None)
        
        print(f"\n{'='*80}")
        print(f"🏆 最佳配置: {best['name']}")
        print(f"   Sharpe: {best['sharpe']:.3f}")
        print(f"   MaxDD: {best['max_dd']:.1%}")
        print(f"   CAGR: {best['cagr']:.1%}")
        
        if baseline and baseline_sharpe > 0:
            improvement = (best['sharpe'] - baseline_sharpe) / baseline_sharpe * 100
            print(f"   vs Baseline: {improvement:+.1f}%")
            
            if improvement > 5:
                print(f"   ✅ 替代数据有显著提升, 建议进一步验证")
            elif improvement > 0:
                print(f"   ⚠️ 提升微弱, 需更多数据验证")
            else:
                print(f"   ❌ 替代数据未带来提升")

    # ── 保存结果 ──
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'Falcon Alternative Data',
        'baseline_v046': V046_WEIGHTS,
        'ic_analysis': ic_results,
        'results': results,
        'best_config': valid_results[0] if valid_results else None,
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n⏱️ 总耗时: {time.time()-t0:.0f}秒")
    print(f"📁 结果: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
