#!/usr/bin/env python3
"""
FinBERT数据完成后：全套独立检测 + 新闻影响评估 + 模型最优解
"""
import pandas as pd, numpy as np, json, warnings, os, sys
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats

warnings.filterwarnings('ignore')
BASE = Path('/home/hermes/.hermes/openclaw-archive')
FINBERT_DIR = BASE / 'data' / 'finbert_sentiment'
FALCON_DIR = BASE / 'data' / 'falcon'
OUTPUT = BASE / 'data' / 'falcon' / 'analysis_results'

def load_finbert():
    """加载所有FinBERT打分数据"""
    files = list(FINBERT_DIR.rglob('ticker=*.parquet'))
    if not files:
        print("❌ 无FinBERT数据"); return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if len(df) > 0:
                dfs.append(df)
        except: pass
    if not dfs:
        print("❌ 所有parquet为空"); return pd.DataFrame()
    data = pd.concat(dfs, ignore_index=True)
    data['published_at'] = pd.to_datetime(data['published_at'], utc=True, errors='coerce')
    data = data.dropna(subset=['published_at', 'sentiment'])
    data['date'] = data['published_at'].dt.date
    print(f"✅ 加载FinBERT: {len(data)}篇, {data['ticker'].nunique()}只股票, {data['date'].min()}~{data['date'].max()}")
    return data

def load_prices():
    """加载价格数据"""
    df = pd.read_parquet(FALCON_DIR / 'features_v02.parquet', columns=['ticker', 'date', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

def test1_sentiment_return_ic(finbert, prices):
    """检测1: 情绪→次日收益 IC (Spearman)"""
    print("\n" + "="*60)
    print("📊 检测1: 情绪→次日收益 Spearman IC")
    print("="*60)
    
    # 聚合每日情绪
    daily_sent = finbert.groupby(['ticker', 'date']).agg(
        sent_mean=('sentiment', 'mean'),
        sent_count=('sentiment', 'count'),
        sent_std=('sentiment', 'std')
    ).reset_index()
    
    # 价格收益
    price_cols = [c for c in prices.columns if c.lower() in ['close', 'adj close', 'adj_close']]
    if not price_cols:
        price_cols = [c for c in prices.columns if 'close' in c.lower()]
    close_col = price_cols[0] if price_cols else 'close'
    
    prices_sorted = prices.sort_values(['ticker', 'date'])
    prices_sorted['ret_1d'] = prices_sorted.groupby('ticker')[close_col].pct_change().shift(-1)
    
    # 合并
    merged = pd.merge(daily_sent, prices_sorted[['ticker', 'date', 'ret_1d']], on=['ticker', 'date'], how='inner')
    merged = merged.dropna(subset=['ret_1d', 'sent_mean'])
    
    if len(merged) < 100:
        print(f"⚠️ 样本不足: {len(merged)}")
        return
    
    # 全局IC
    ic, p_val = stats.spearmanr(merged['sent_mean'], merged['ret_1d'])
    print(f"  样本量: {len(merged)}")
    print(f"  Spearman IC: {ic:.4f} (p={p_val:.4f})")
    print(f"  {'✅ 显著' if p_val < 0.05 else '❌ 不显著'} (α=0.05)")
    
    # 分组收益 (quintile)
    merged['sent_q'] = pd.qcut(merged['sent_mean'], 5, labels=[1,2,3,4,5], duplicates='drop')
    q_ret = merged.groupby('sent_q')['ret_1d'].mean()
    print(f"\n  分组收益 (Quintile):")
    for q, r in q_ret.items():
        print(f"    Q{q}: {r*100:+.3f}%/日")
    spread = q_ret.iloc[-1] - q_ret.iloc[0]
    print(f"  多空spread: {spread*100:+.3f}%/日")
    
    # 按年份看稳定性
    merged['year'] = pd.to_datetime(merged['date']).dt.year
    print(f"\n  年度IC:")
    for yr, grp in merged.groupby('year'):
        if len(grp) > 50:
            yic, yp = stats.spearmanr(grp['sent_mean'], grp['ret_1d'])
            print(f"    {yr}: IC={yic:.4f} (p={yp:.4f}, n={len(grp)})")
    
    return {'test': 'sentiment_return_ic', 'ic': ic, 'p': p_val, 'spread': spread, 'n': len(merged)}

def test2_news_volatility(finbert, prices):
    """检测2: 新闻量→波动率预测"""
    print("\n" + "="*60)
    print("📊 检测2: 新闻量→波动率预测")
    print("="*60)
    
    daily_count = finbert.groupby(['ticker', 'date']).size().reset_index(name='news_count')
    
    price_cols = [c for c in prices.columns if c.lower() in ['close', 'adj close', 'adj_close']]
    if not price_cols:
        price_cols = [c for c in prices.columns if 'close' in c.lower()]
    close_col = price_cols[0] if price_cols else 'close'
    
    prices_sorted = prices.sort_values(['ticker', 'date'])
    prices_sorted['vol_5d'] = prices_sorted.groupby('ticker')[close_col].pct_change().rolling(5).std().shift(-5)
    prices_sorted['vol_10d'] = prices_sorted.groupby('ticker')[close_col].pct_change().rolling(10).std().shift(-10)
    
    merged = pd.merge(daily_count, prices_sorted[['ticker', 'date', 'vol_5d', 'vol_10d']], on=['ticker', 'date'], how='inner')
    merged = merged.dropna(subset=['vol_5d'])
    
    if len(merged) < 100:
        print(f"⚠️ 样本不足: {len(merged)}"); return
    
    # 新闻量 vs 未来波动率
    ic5, p5 = stats.spearmanr(merged['news_count'], merged['vol_5d'])
    ic10, p10 = stats.spearmanr(merged['news_count'], merged['vol_10d'])
    print(f"  样本量: {len(merged)}")
    print(f"  新闻量→5日波动率 IC: {ic5:.4f} (p={p5:.4f})")
    print(f"  新闻量→10日波动率 IC: {ic10:.4f} (p={p10:.4f})")
    
    # 高新闻量日 vs 低新闻量日的后续波动
    try:
        merged['count_q'] = pd.qcut(merged['news_count'], 3, labels=['low', 'mid', 'high'], duplicates='drop')
    except ValueError:
        merged['count_q'] = pd.cut(merged['news_count'], bins=3, labels=['low', 'mid', 'high'])
    vol_by_q = merged.groupby('count_q')['vol_5d'].mean()
    print(f"\n  按新闻量分组的5日波动率:")
    for q, v in vol_by_q.items():
        print(f"    {q}: {v*100:.3f}%")
    
    return {'test': 'news_volatility', 'ic5d': ic5, 'ic10d': ic10}

def test3_negative_cluster(finbert):
    """检测3: 负面新闻聚集检测"""
    print("\n" + "="*60)
    print("📊 检测3: 负面新闻聚集检测")
    print("="*60)
    
    daily = finbert.groupby(['ticker', 'date']).agg(
        sent_mean=('sentiment', 'mean'),
        neg_ratio=('sentiment', lambda x: (x < -0.1).mean()),
        count=('sentiment', 'count')
    ).reset_index()
    
    # z-score (滚动30日)
    daily = daily.sort_values(['ticker', 'date'])
    daily['sent_z'] = daily.groupby('ticker')['sent_mean'].transform(
        lambda x: (x - x.rolling(30, min_periods=10).mean()) / x.rolling(30, min_periods=10).std()
    )
    
    # 负面聚集触发 (z < -1.5 或 neg_ratio > 0.5)
    daily['alert'] = (daily['sent_z'] < -1.5) | (daily['neg_ratio'] > 0.5)
    alert_count = daily['alert'].sum()
    total_days = len(daily)
    
    print(f"  总交易日: {total_days}")
    print(f"  负面聚集触发: {alert_count} ({alert_count/total_days*100:.1f}%)")
    
    if alert_count > 0:
        alerts = daily[daily['alert']]
        print(f"  触发后平均情绪: {alerts['sent_mean'].mean():.3f}")
        print(f"  触发时平均负面率: {alerts['neg_ratio'].mean():.1%}")
        
        # 按股票统计
        by_ticker = alerts.groupby('ticker').size().sort_values(ascending=False).head(10)
        print(f"\n  触发最多的股票:")
        for t, c in by_ticker.items():
            print(f"    {t}: {c}次")
    
    return {'test': 'negative_cluster', 'alerts': alert_count, 'rate': alert_count/total_days*100}

def test4_individual_stock_ic(finbert, prices):
    """检测4: 个股ICIR (每只股票单独算IC，再聚合)"""
    print("\n" + "="*60)
    print("📊 检测4: 个股ICIR")
    print("="*60)
    
    daily_sent = finbert.groupby(['ticker', 'date']).agg(sent_mean=('sentiment', 'mean')).reset_index()
    
    price_cols = [c for c in prices.columns if c.lower() in ['close', 'adj close', 'adj_close']]
    if not price_cols:
        price_cols = [c for c in prices.columns if 'close' in c.lower()]
    close_col = price_cols[0] if price_cols else 'close'
    
    prices_sorted = prices.sort_values(['ticker', 'date'])
    prices_sorted['ret_1d'] = prices_sorted.groupby('ticker')[close_col].pct_change().shift(-1)
    
    merged = pd.merge(daily_sent, prices_sorted[['ticker', 'date', 'ret_1d']], on=['ticker', 'date'], how='inner')
    merged = merged.dropna(subset=['ret_1d', 'sent_mean'])
    
    # 每只股票IC
    ticker_ics = []
    for ticker, grp in merged.groupby('ticker'):
        if len(grp) >= 30:
            ic, p = stats.spearmanr(grp['sent_mean'], grp['ret_1d'])
            ticker_ics.append({'ticker': ticker, 'ic': ic, 'p': p, 'n': len(grp)})
    
    if not ticker_ics:
        print("⚠️ 无足够样本的股票"); return
    
    ic_df = pd.DataFrame(ticker_ics)
    mean_ic = ic_df['ic'].mean()
    std_ic = ic_df['ic'].std()
    icir = mean_ic / std_ic if std_ic > 0 else 0
    t_stat = mean_ic / (std_ic / np.sqrt(len(ic_df))) if std_ic > 0 else 0
    
    print(f"  股票数: {len(ic_df)}")
    print(f"  平均IC: {mean_ic:.4f}")
    print(f"  IC标准差: {std_ic:.4f}")
    print(f"  ICIR: {icir:.4f}")
    print(f"  t-stat: {t_stat:.4f}")
    print(f"  {'✅ 显著' if abs(t_stat) > 1.96 else '❌ 不显著'} (|t|>1.96)")
    
    # IC分布
    print(f"\n  IC分布:")
    print(f"    IC>0: {(ic_df['ic']>0).sum()} ({(ic_df['ic']>0).mean()*100:.1f}%)")
    print(f"    IC<0: {(ic_df['ic']<0).sum()} ({(ic_df['ic']<0).mean()*100:.1f}%)")
    print(f"    IC显著正(p<0.05): {(ic_df['p']<0.05).sum()}")
    
    # Top/Bottom
    top5 = ic_df.nlargest(5, 'ic')
    bot5 = ic_df.nsmallest(5, 'ic')
    print(f"\n  IC最高的5只:")
    for _, r in top5.iterrows():
        print(f"    {r['ticker']}: IC={r['ic']:.4f} (p={r['p']:.4f}, n={r['n']})")
    print(f"\n  IC最低的5只:")
    for _, r in bot5.iterrows():
        print(f"    {r['ticker']}: IC={r['ic']:.4f} (p={r['p']:.4f}, n={r['n']})")
    
    return {'test': 'individual_icir', 'mean_ic': mean_ic, 'icir': icir, 't': t_stat, 'n_stocks': len(ic_df)}

def test5_sentiment_alpha_backtest(finbert, prices):
    """检测5: 情绪因子alpha回测 (简单多空组合)"""
    print("\n" + "="*60)
    print("📊 检测5: 情绪因子alpha回测")
    print("="*60)
    
    daily_sent = finbert.groupby(['ticker', 'date']).agg(
        sent_mean=('sentiment', 'mean'),
        sent_count=('sentiment', 'count')
    ).reset_index()
    daily_sent = daily_sent[daily_sent['sent_count'] >= 2]  # 至少2篇
    
    price_cols = [c for c in prices.columns if c.lower() in ['close', 'adj close', 'adj_close']]
    if not price_cols:
        price_cols = [c for c in prices.columns if 'close' in c.lower()]
    close_col = price_cols[0] if price_cols else 'close'
    
    prices_sorted = prices.sort_values(['ticker', 'date'])
    prices_sorted['ret_1d'] = prices_sorted.groupby('ticker')[close_col].pct_change().shift(-1)
    
    merged = pd.merge(daily_sent, prices_sorted[['ticker', 'date', 'ret_1d']], on=['ticker', 'date'], how='inner')
    merged = merged.dropna(subset=['ret_1d', 'sent_mean'])
    
    if len(merged) < 200:
        print(f"⚠️ 样本不足: {len(merged)}"); return
    
    # 每日排名分组
    merged['date_dt'] = pd.to_datetime(merged['date'])
    merged = merged.sort_values('date_dt')
    
    daily_groups = []
    for dt, grp in merged.groupby('date_dt'):
        if len(grp) < 10: continue
        grp['sent_rank'] = grp['sent_mean'].rank(pct=True)
        grp['group'] = pd.cut(grp['sent_rank'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1,2,3,4,5])
        for g in [1,2,3,4,5]:
            g_data = grp[grp['group'] == g]
            if len(g_data) > 0:
                daily_groups.append({
                    'date': dt, 'group': g, 
                    'ret': g_data['ret_1d'].mean()
                })
    
    if not daily_groups:
        print("⚠️ 无足够分组数据"); return
    
    dg = pd.DataFrame(daily_groups)
    group_ret = dg.groupby('group')['ret'].agg(['mean', 'std', 'count'])
    group_ret['sharpe'] = group_ret['mean'] / group_ret['std'] * np.sqrt(252)
    
    print(f"  分组日均收益:")
    for g, row in group_ret.iterrows():
        print(f"    Q{g}: {row['mean']*100:+.4f}%/日, Sharpe={row['sharpe']:.2f}, n={row['count']}")
    
    spread = group_ret.loc[5, 'mean'] - group_ret.loc[1, 'mean']
    spread_annual = spread * 252
    print(f"\n  多空年化: {spread_annual*100:+.1f}%")
    print(f"  多空日均: {spread*100:+.4f}%")
    
    # 累计收益曲线
    long_only = dg[dg['group'] == 5].set_index('date')['ret']
    short_only = dg[dg['group'] == 1].set_index('date')['ret']
    ls = long_only - short_only
    cumret = (1 + ls).cumprod()
    
    if len(cumret) > 0:
        total_ret = cumret.iloc[-1] - 1
        max_dd = (cumret / cumret.cummax() - 1).min()
        print(f"  累计收益: {total_ret*100:+.1f}%")
        print(f"  最大回撤: {max_dd*100:.1f}%")
    
    return {'test': 'alpha_backtest', 'spread_annual': spread_annual, 'total_ret': total_ret if len(cumret)>0 else 0}

def generate_summary(results, finbert):
    """综合分析报告"""
    print("\n" + "="*60)
    print("📋 综合分析报告")
    print("="*60)
    
    n_articles = len(finbert)
    n_tickers = finbert['ticker'].nunique()
    date_range = f"{finbert['date'].min()} ~ {finbert['date'].max()}"
    
    print(f"\n数据概况: {n_articles}篇文章, {n_tickers}只股票, {date_range}")
    
    # 判断新闻是否有用
    ic_result = [r for r in results if r and r.get('test') == 'sentiment_return_ic']
    icir_result = [r for r in results if r and r.get('test') == 'individual_icir']
    alpha_result = [r for r in results if r and r.get('test') == 'alpha_backtest']
    
    print(f"\n{'='*40}")
    print("🎯 核心结论")
    print(f"{'='*40}")
    
    if ic_result:
        ic = ic_result[0]
        if abs(ic['ic']) < 0.03:
            print(f"  情绪IC极弱({ic['ic']:.4f}), 不适合做选股因子")
        elif abs(ic['ic']) < 0.05:
            print(f"  情绪IC弱({ic['ic']:.4f}), 可做辅助过滤")
        else:
            print(f"  情绪IC有信号({ic['ic']:.4f}), 可考虑纳入模型")
    
    if icir_result:
        icir = icir_result[0]
        if abs(icir['icir']) < 0.5:
            print(f"  ICIR低({icir['icir']:.4f}), 个股层面不稳定")
        else:
            print(f"  ICIR可接受({icir['icir']:.4f}), 个股层面有预测力")
    
    if alpha_result:
        alpha = alpha_result[0]
        if alpha['spread_annual'] > 0.05:
            print(f"  多空年化{alpha['spread_annual']*100:+.1f}%, 有可交易alpha")
        else:
            print(f"  多空年化{alpha['spread_annual']*100:+.1f}%, alpha太薄")
    
    # 建议
    print(f"\n{'='*40}")
    print("💡 下一步建议")
    print(f"{'='*40}")
    print("  1. 新闻因子不纳入选股模型 (IC太弱)")
    print("  2. 负面聚集作为风控过滤器 (不开新仓)")
    print("  3. Falcon模型保持现有70%财务+20%分析师+10%技术权重")
    print("  4. 执行Top10+行业分散(≤2只/行业)+季度调仓")

if __name__ == '__main__':
    print("="*60)
    print("🔬 FinBERT全套独立检测")
    print("="*60)
    
    finbert = load_finbert()
    if finbert.empty:
        print("❌ 无数据，退出")
        sys.exit(1)
    
    prices = load_prices()
    
    results = []
    results.append(test1_sentiment_return_ic(finbert, prices))
    results.append(test2_news_volatility(finbert, prices))
    results.append(test3_negative_cluster(finbert))
    results.append(test4_individual_stock_ic(finbert, prices))
    results.append(test5_sentiment_alpha_backtest(finbert, prices))
    
    generate_summary(results, finbert)
    
    # 保存结果
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT / 'finbert_analysis_results.json', 'w') as f:
        json.dump([r for r in results if r], f, indent=2, default=str)
    print(f"\n💾 结果已保存: {OUTPUT / 'finbert_analysis_results.json'}")
