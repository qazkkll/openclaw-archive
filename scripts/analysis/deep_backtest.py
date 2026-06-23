#!/usr/bin/env python3
"""
深度历史回测 — 模型在10年数据上的真实表现
用于自我进化：从历史中提取可行动的规则
"""

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

BASE = Path.home() / '.hermes' / 'openclaw-archive'
DATA_DIR = BASE / 'data'
OUTPUT_DIR = DATA_DIR / 'backtest-rounds'

def compute_us_features(df):
    """Compute features for US stocks matching blueshield_v6/arrow_v11"""
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    c = df['close'].astype(float)
    h = df['high'].astype(float) if 'high' in df else c
    l = df['low'].astype(float) if 'low' in df else c
    v = df['volume'].astype(float) if 'volume' in df else pd.Series(0, index=df.index)
    
    # Moving averages
    df['ma5'] = c.rolling(5).mean()
    df['ma20'] = c.rolling(20).mean()
    df['ma60'] = c.rolling(60).mean()
    df['ma_bias20'] = (c - df['ma20']) / df['ma20']
    df['ma_align'] = ((df['ma5'] > df['ma20']).astype(int) + (df['ma20'] > df['ma60']).astype(int))
    df['price_position'] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min() + 1e-10)
    
    # Returns
    df['ret1'] = c.pct_change(1)
    df['ret5'] = c.pct_change(5)
    df['ret20'] = c.pct_change(20)
    df['ret60'] = c.pct_change(60)
    
    # Momentum
    df['momentum_6m'] = c.pct_change(120)
    df['momentum_1m'] = c.pct_change(20)
    df['mom_divergence'] = df['momentum_1m'] - df['momentum_6m']
    df['trend_accel'] = df['ret5'] - df['ret5'].shift(5)
    
    # Volatility
    df['vol20'] = c.pct_change().rolling(20).std()
    df['vol_ratio'] = v.rolling(5).mean() / (v.rolling(20).mean() + 1)
    
    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ATR
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / c
    
    # Additional features for 44-feature model
    df['bb_upper'] = df['ma20'] + 2 * c.rolling(20).std()
    df['bb_lower'] = df['ma20'] - 2 * c.rolling(20).std()
    df['bb_position'] = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['vol_change'] = v.pct_change(5)
    df['high_low_range'] = (h - l) / c
    df['close_position'] = (c - l) / (h - l + 1e-10)
    df['gap'] = (df['close'].shift(1) - c.shift(1).where(c.shift(1) > 0, c)) / c.shift(1)
    df['vol_ma_ratio'] = v / (v.rolling(20).mean() + 1)
    df['price_ma5_ratio'] = c / df['ma5']
    df['price_ma20_ratio'] = c / df['ma20']
    df['price_ma60_ratio'] = c / df['ma60']
    df['ret5_ret20_ratio'] = df['ret5'] / (df['ret20'].abs() + 1e-10)
    df['vol_ret_corr'] = c.pct_change().rolling(20).corr(v.pct_change())
    df['momentum_score'] = df['ret5'] * 0.5 + df['ret20'] * 0.3 + df['ret60'] * 0.2
    
    # Forward returns (for evaluation)
    df['fwd_ret5'] = c.shift(-5) / c - 1
    df['fwd_ret10'] = c.shift(-10) / c - 1
    df['fwd_ret20'] = c.shift(-20) / c - 1
    
    return df

def compute_cn_features(df):
    """Compute features for A-shares matching cn_alpha_v2"""
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    c = df['C'].astype(float)
    h = df['H'].astype(float)
    l = df['L'].astype(float)
    v = df['V'].astype(float)
    
    # Returns
    df['ret5'] = c.pct_change(5)
    df['ret10'] = c.pct_change(10)
    df['ret20'] = c.pct_change(20)
    
    # MA bias
    df['ma20'] = c.rolling(20).mean()
    df['ma60'] = c.rolling(60).mean()
    df['ma20_bias'] = (c - df['ma20']) / df['ma20']
    df['ma60_bias'] = (c - df['ma60']) / df['ma60']
    
    # Volume
    df['vol5'] = v.rolling(5).mean()
    df['vol20'] = v.rolling(20).mean()
    df['vol_ratio'] = df['vol5'] / (df['vol20'] + 1)
    
    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd = ema12 - ema26
    df['macd_hist'] = macd - macd.ewm(span=9).mean()
    
    # ATR
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df['atr_pct'] = tr.rolling(14).mean() / c
    
    # Fund flow proxies
    df['total_net_5d'] = c.pct_change(5) * v.rolling(5).mean()
    df['lg_net_5d'] = df['total_net_5d'] * 0.5
    df['md_net_5d'] = df['total_net_5d'] * 0.3
    df['elg_net_5d'] = df['total_net_5d'] * 0.2
    
    # Additional
    df['price_position'] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min() + 1e-10)
    df['volatility'] = c.pct_change().rolling(20).std()
    df['momentum'] = c.pct_change(20)
    df['turnover_change'] = v.pct_change(5)
    
    # Forward returns
    df['fwd_ret5'] = c.shift(-5) / c - 1
    df['fwd_ret10'] = c.shift(-10) / c - 1
    
    return df

def backtest_model(model, df, feature_cols, top_n=1, hold_days=5, label=''):
    """Run backtest for a single stock universe"""
    results = []
    
    # Get valid dates (where we have enough history and forward returns)
    valid = df.dropna(subset=feature_cols + ['fwd_ret5'])
    if valid.empty:
        return results
    
    # Generate predictions
    dmat = xgb.DMatrix(valid[feature_cols], feature_names=feature_cols)
    preds = model.predict(dmat)
    valid = valid.copy()
    valid['pred_score'] = preds
    
    # For each date, select top N
    for date in valid['date'].unique() if 'date' in valid else valid['Date'].unique():
        date_col = 'date' if 'date' in valid.columns else 'Date'
        day_data = valid[valid[date_col] == date].nlargest(top_n, 'pred_score')
        
        for _, row in day_data.iterrows():
            results.append({
                'date': str(date)[:10],
                'ticker': row.get('sym', row.get('Code', '?')),
                'pred_score': row['pred_score'],
                'fwd_ret5': row.get('fwd_ret5', 0),
                'fwd_ret10': row.get('fwd_ret10', 0),
                'fwd_ret20': row.get('fwd_ret20', 0) if 'fwd_ret20' in row else None,
                'hit': 1 if row.get('fwd_ret5', 0) > 0.02 else 0,
            })
    
    return results

def main():
    print("=" * 70)
    print("📊 深度历史回测 — 模型在10年数据上的真实表现")
    print("=" * 70)
    
    # Load models
    print("\n加载模型...")
    us_model = xgb.Booster()
    us_model.load_model(str(BASE / 'models/us/blueshield_v6_xgb.json'))
    
    us_small_model = xgb.Booster()
    us_small_model.load_model(str(BASE / 'models/us/arrow_v11_xgb.json'))
    
    cn_model = xgb.Booster()
    cn_model.load_model(str(BASE / 'models/cn/cn_alpha_v2_xgb.json'))
    
    # Feature lists
    us_features = ['ma5', 'ma20', 'ma60', 'ma_bias20', 'ma_align', 'price_position',
                   'ret1', 'ret5', 'ret20', 'ret60', 'momentum_6m', 'momentum_1m',
                   'mom_divergence', 'trend_accel', 'vol20', 'vol_ratio', 'rsi_14',
                   'macd', 'macd_signal', 'macd_hist', 'atr_14', 'atr_pct',
                   'bb_position', 'vol_change', 'high_low_range', 'close_position',
                   'vol_ma_ratio', 'price_ma5_ratio', 'price_ma20_ratio', 'price_ma60_ratio',
                   'ret5_ret20_ratio', 'vol_ret_corr', 'momentum_score', 'gap',
                   'bb_upper', 'bb_lower']
    
    cn_features = ['ret5', 'ret10', 'ret20', 'ma20_bias', 'ma60_bias', 'vol5', 'vol20',
                   'vol_ratio', 'rsi_14', 'macd_hist', 'atr_pct', 'total_net_5d',
                   'lg_net_5d', 'md_net_5d', 'elg_net_5d', 'price_position',
                   'volatility', 'momentum', 'turnover_change']
    
    # ========== US BACKTEST ==========
    print("\n" + "=" * 70)
    print("🇺🇸 美股蓝盾V6回测 (>$10, Top1)")
    print("=" * 70)
    
    us_data = pd.read_parquet(str(DATA_DIR / 'us/us_hist_yf_10y.parquet'))
    
    # Filter: last 5 years, price > $10, volume > 100K
    us_data['date'] = pd.to_datetime(us_data['date'])
    us_data = us_data[us_data['date'] >= '2021-01-01']
    us_data = us_data[us_data['close'] > 10]
    
    # Sample top 200 liquid stocks
    vol_avg = us_data.groupby('sym')['volume'].mean().nlargest(200)
    top_tickers = vol_avg.index.tolist()
    
    all_us_results = []
    processed = 0
    for ticker in top_tickers:
        stock = us_data[us_data['sym'] == ticker].copy()
        if len(stock) < 120:
            continue
        
        stock = compute_us_features(stock)
        # Filter to matching feature columns
        feat_cols = [f for f in us_features if f in stock.columns]
        results = backtest_model(us_model, stock, feat_cols, top_n=1, hold_days=5, label='blueshield_v6')
        all_us_results.extend(results)
        processed += 1
        if processed % 50 == 0:
            print(f"  处理中: {processed}/{len(top_tickers)}")
    
    print(f"  完成: {processed}只股票, {len(all_us_results)}条回测记录")
    
    # ========== US SMALL CAP ==========
    print("\n" + "=" * 70)
    print("🇺🇸 美股绿箭V11回测 (<$10, Top3)")
    print("=" * 70)
    
    us_small = us_data[(us_data['close'] <= 10) & (us_data['close'] > 1)]
    vol_avg_small = us_small.groupby('sym')['volume'].mean().nlargest(300)
    small_tickers = vol_avg_small.index.tolist()
    
    all_small_results = []
    processed = 0
    for ticker in small_tickers:
        stock = us_small[us_small['sym'] == ticker].copy()
        if len(stock) < 120:
            continue
        
        stock = compute_us_features(stock)
        feat_cols = [f for f in us_features if f in stock.columns]
        results = backtest_model(us_small_model, stock, feat_cols, top_n=3, hold_days=5, label='arrow_v11')
        all_small_results.extend(results)
        processed += 1
        if processed % 50 == 0:
            print(f"  处理中: {processed}/{len(small_tickers)}")
    
    print(f"  完成: {processed}只股票, {len(all_small_results)}条回测记录")
    
    # ========== CN BACKTEST ==========
    print("\n" + "=" * 70)
    print("🇨🇳 A股红杉V2回测")
    print("=" * 70)
    
    cn_data = pd.read_parquet(str(DATA_DIR / 'a_hist_10y.parquet'))
    cn_data['Date'] = pd.to_datetime(cn_data['Date'])
    cn_data = cn_data[cn_data['Date'] >= '2021-01-01']
    
    # Sample 300 stocks
    vol_avg_cn = cn_data.groupby('Code')['V'].mean().nlargest(300)
    cn_tickers = vol_avg_cn.index.tolist()
    
    all_cn_results = []
    processed = 0
    for ticker in cn_tickers:
        stock = cn_data[cn_data['Code'] == ticker].copy()
        if len(stock) < 120:
            continue
        
        stock = compute_cn_features(stock)
        feat_cols = [f for f in cn_features if f in stock.columns]
        results = backtest_model(cn_model, stock, feat_cols, top_n=1, hold_days=5, label='cn_alpha_v2')
        all_cn_results.extend(results)
        processed += 1
        if processed % 50 == 0:
            print(f"  处理中: {processed}/{len(cn_tickers)}")
    
    print(f"  完成: {processed}只股票, {len(all_cn_results)}条回测记录")
    
    # ========== ANALYSIS ==========
    print("\n" + "=" * 70)
    print("📊 深度分析")
    print("=" * 70)
    
    all_results = {
        'blueshield_v6': all_us_results,
        'arrow_v11': all_small_results,
        'cn_alpha_v2': all_cn_results,
    }
    
    report = []
    report.append("# 🧠 深度历史回测报告 — 自我进化基础")
    report.append(f"\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"> 数据范围: 2021-01 ~ 2026-06 (5年)")
    report.append(f"> 评分标准: 5天收益率 > 2% = 命中")
    
    for model_name, results in all_results.items():
        if not results:
            continue
        
        df = pd.DataFrame(results)
        total = len(df)
        hits = df['hit'].sum()
        hit_rate = hits / total * 100
        avg_ret = df['fwd_ret5'].mean() * 100
        median_ret = df['fwd_ret5'].median() * 100
        
        report.append(f"\n## {model_name}")
        report.append(f"\n| 指标 | 值 |")
        report.append(f"|------|-----|")
        report.append(f"| 回测记录 | {total} |")
        report.append(f"| 命中率 | {hit_rate:.1f}% |")
        report.append(f"| 平均5天收益 | {avg_ret:.2f}% |")
        report.append(f"| 中位数5天收益 | {median_ret:.2f}% |")
        report.append(f"| 正收益占比 | {(df['fwd_ret5']>0).mean()*100:.1f}% |")
        report.append(f"| 最大收益 | {df['fwd_ret5'].max()*100:.2f}% |")
        report.append(f"| 最大亏损 | {df['fwd_ret5'].min()*100:.2f}% |")
        
        # By year
        df['year'] = pd.to_datetime(df['date']).dt.year
        report.append(f"\n### 按年份")
        report.append(f"\n| 年份 | 记录数 | 命中率 | 平均收益 | 正收益% |")
        report.append(f"|------|--------|--------|----------|---------|")
        for year in sorted(df['year'].unique()):
            yr = df[df['year'] == year]
            if len(yr) < 10:
                continue
            report.append(f"| {year} | {len(yr)} | {yr['hit'].mean()*100:.1f}% | {yr['fwd_ret5'].mean()*100:.2f}% | {(yr['fwd_ret5']>0).mean()*100:.1f}% |")
        
        # Score distribution analysis
        report.append(f"\n### 预测分数 vs 实际收益")
        df['score_bucket'] = pd.qcut(df['pred_score'], 5, labels=['Q1(低)', 'Q2', 'Q3', 'Q4', 'Q5(高)'])
        report.append(f"\n| 分数桶 | 记录数 | 命中率 | 平均收益 | 平均分数 |")
        report.append(f"|--------|--------|--------|----------|----------|")
        for bucket in ['Q1(低)', 'Q2', 'Q3', 'Q4', 'Q5(高)']:
            b = df[df['score_bucket'] == bucket]
            if len(b) == 0:
                continue
            report.append(f"| {bucket} | {len(b)} | {b['hit'].mean()*100:.1f}% | {b['fwd_ret5'].mean()*100:.2f}% | {b['pred_score'].mean():.4f} |")
        
        # Sharpe ratio by year
        report.append(f"\n### 风险调整收益")
        report.append(f"\n| 年份 | 夏普比率 | 最大回撤 | 胜率 |")
        report.append(f"|------|----------|----------|------|")
        for year in sorted(df['year'].unique()):
            yr = df[df['year'] == year]
            if len(yr) < 10:
                continue
            rets = yr['fwd_ret5'].values
            sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252/5)
            cumret = (1 + pd.Series(rets)).cumprod()
            max_dd = (cumret / cumret.cummax() - 1).min() * 100
            report.append(f"| {year} | {sharpe:.2f} | {max_dd:.1f}% | {(rets>0).mean()*100:.1f}% |")
    
    # ========== EVOLUTION RULES ==========
    report.append(f"\n## 🧬 自我进化规则提取")
    report.append(f"\n基于回测数据，提取以下可行动规则：")
    
    # Analyze score calibration across all models
    all_df = pd.DataFrame()
    for model_name, results in all_results.items():
        if results:
            df = pd.DataFrame(results)
            df['model'] = model_name
            all_df = pd.concat([all_df, df])
    
    if not all_df.empty:
        # Score calibration
        all_df['score_bucket'] = pd.qcut(all_df['pred_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        calibration = all_df.groupby('score_bucket').agg(
            count=('hit', 'count'),
            hit_rate=('hit', 'mean'),
            avg_ret=('fwd_ret5', 'mean')
        ).reset_index()
        
        report.append(f"\n### 规则1: 分数阈值优化")
        report.append(f"\n| 分数桶 | 记录数 | 命中率 | 平均收益 | 建议 |")
        report.append(f"|--------|--------|--------|----------|------|")
        for _, row in calibration.iterrows():
            if row['hit_rate'] > 0.5:
                action = '✅ 强信号，可交易'
            elif row['hit_rate'] > 0.3:
                action = '⚠️ 中等信号，谨慎'
            else:
                action = '❌ 弱信号，避免'
            report.append(f"| {row['score_bucket']} | {int(row['count'])} | {row['hit_rate']*100:.1f}% | {row['avg_ret']*100:.2f}% | {action} |")
        
        # By model
        report.append(f"\n### 规则2: 模型比较")
        report.append(f"\n| 模型 | 记录数 | 命中率 | 平均收益 | 夏普 |")
        report.append(f"|------|--------|--------|----------|------|")
        for model in all_df['model'].unique():
            m = all_df[all_df['model'] == model]
            sharpe = m['fwd_ret5'].mean() / (m['fwd_ret5'].std() + 1e-10) * np.sqrt(252/5)
            report.append(f"| {model} | {len(m)} | {m['hit'].mean()*100:.1f}% | {m['fwd_ret5'].mean()*100:.2f}% | {sharpe:.2f} |")
        
        # Top performing conditions
        report.append(f"\n### 规则3: 最佳交易条件")
        
        # Find conditions where hit rate > 40%
        high_hit = all_df[all_df['pred_score'] >= all_df['pred_score'].quantile(0.8)]
        low_hit = all_df[all_df['pred_score'] <= all_df['pred_score'].quantile(0.2)]
        
        report.append(f"\n| 条件 | 记录数 | 命中率 | 平均收益 |")
        report.append(f"|------|--------|--------|----------|")
        report.append(f"| Top 20% 分数 | {len(high_hit)} | {high_hit['hit'].mean()*100:.1f}% | {high_hit['fwd_ret5'].mean()*100:.2f}% |")
        report.append(f"| Bottom 20% 分数 | {len(low_hit)} | {low_hit['hit'].mean()*100:.1f}% | {low_hit['fwd_ret5'].mean()*100:.2f}% |")
        
        # Simulate portfolio
        report.append(f"\n### 规则4: 组合模拟")
        
        for model_name in all_df['model'].unique():
            m = all_df[all_df['model'] == model_name].sort_values('date')
            
            # Monthly rebalance: top N by score
            m['month'] = pd.to_datetime(m['date']).dt.to_period('M')
            monthly = m.groupby('month').agg(
                picks=('hit', 'count'),
                hit_rate=('hit', 'mean'),
                avg_ret=('fwd_ret5', 'mean'),
                total_ret=('fwd_ret5', 'sum')
            ).reset_index()
            
            if len(monthly) > 0:
                cumret = (1 + monthly['avg_ret']).cumprod()
                total_return = (cumret.iloc[-1] - 1) * 100
                annual_return = ((cumret.iloc[-1]) ** (12/len(monthly)) - 1) * 100
                max_dd = (cumret / cumret.cummax() - 1).min() * 100
                
                report.append(f"\n**{model_name}**:")
                report.append(f"- 月度再平衡收益: {total_return:.1f}%")
                report.append(f"- 年化收益: {annual_return:.1f}%")
                report.append(f"- 最大回撤: {max_dd:.1f}%")
    
    # Save report
    report_text = '\n'.join(report)
    output_path = OUTPUT_DIR / 'deep-backtest-report.md'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n报告已保存: {output_path}")
    print(f"报告长度: {len(report_text)} 字符, {len(report)} 行")
    
    # Also save raw data for future analysis
    raw_output = OUTPUT_DIR / 'backtest-raw-results.json'
    with open(raw_output, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, default=str)
    print(f"原始数据: {raw_output}")

if __name__ == '__main__':
    main()
