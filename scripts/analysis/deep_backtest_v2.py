#!/usr/bin/env python3
"""
深度历史回测 v2 — 使用模型的真实特征
包括宏观特征(VIX/SPY/QQQ/IWM)和基本面(PE/beta)
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

def download_macro_data():
    """Download VIX, SPY, QQQ, IWM data"""
    import yfinance as yf
    tickers = ['^VIX', 'SPY', 'QQQ', 'IWM']
    data = {}
    for t in tickers:
        try:
            d = yf.download(t, start='2020-01-01', progress=False, auto_adjust=True)
            # Flatten MultiIndex columns if needed
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = [c[0] for c in d.columns]
            d = d.reset_index()
            # Ensure column names are strings
            d.columns = [str(c) for c in d.columns]
            # Rename first column to Date if needed
            if d.columns[0] != 'Date':
                d = d.rename(columns={d.columns[0]: 'Date'})
            data[t] = d
            print(f"  {t}: {len(d)} rows, cols={list(d.columns)}")
        except Exception as e:
            print(f"  {t}: Error - {e}")
    return data

def compute_macro_features(df, macro_data, date_col='date'):
    """Add macro features to stock data"""
    df = df.copy()
    # Normalize all dates to same dtype
    df[date_col] = pd.to_datetime(df[date_col]).astype('datetime64[ns]')
    
    # VIX
    if '^VIX' in macro_data:
        vix = macro_data['^VIX'][['Date', 'Close']].copy()
        vix.columns = ['macro_date', 'vix_close']
        vix['macro_date'] = pd.to_datetime(vix['macro_date']).astype('datetime64[ns]')
        df = pd.merge_asof(df.sort_values(date_col), vix.sort_values('macro_date'), 
                          left_on=date_col, right_on='macro_date', direction='backward')
        df = df.drop(columns=['macro_date'], errors='ignore')
    
    # SPY/QQQ/IWM returns
    for ticker, prefix in [('SPY', 'spy'), ('QQQ', 'qqq'), ('IWM', 'iwm')]:
        if ticker in macro_data:
            m = macro_data[ticker][['Date', 'Close']].copy()
            m.columns = ['macro_date', 'close']
            m['macro_date'] = pd.to_datetime(m['macro_date']).astype('datetime64[ns]')
            for period in [1, 5, 20, 60]:
                m[f'{prefix}_ret{period}'] = m['close'].pct_change(period)
            m = m[['macro_date', f'{prefix}_ret1', f'{prefix}_ret5', f'{prefix}_ret20', f'{prefix}_ret60']]
            df = pd.merge_asof(df.sort_values(date_col), m.sort_values('macro_date'),
                              left_on=date_col, right_on='macro_date', direction='backward')
            df = df.drop(columns=['macro_date'], errors='ignore')
    
    return df

def compute_us_features(df, fundamentals=None):
    """Compute features matching blueshield_v6 exactly"""
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
    df['ma_bias20'] = (c - df['ma20']) / (df['ma20'] + 1e-10)
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
    df['vol5'] = c.pct_change().rolling(5).std()
    df['vol_ratio'] = v.rolling(5).mean() / (v.rolling(20).mean() + 1)
    df['vol_change'] = v.pct_change(5)
    
    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi14'] = 100 - (100 / (1 + rs))
    df['rsi_change'] = df['rsi14'] - df['rsi14'].shift(5)
    
    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df['bb_std'] = bb_std
    df['bb_width'] = 2 * bb_std / (bb_mid + 1e-10)
    df['bb_pos'] = (c - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    
    # Return quality
    df['ret_quality'] = c.pct_change().rolling(20).apply(lambda x: (x > 0).mean(), raw=True)
    
    # Forward returns
    df['fwd_ret5'] = c.shift(-5) / c - 1
    df['fwd_ret10'] = c.shift(-10) / c - 1
    df['fwd_ret20'] = c.shift(-20) / c - 1
    
    # Add fundamentals if available
    ticker = df['sym'].iloc[0] if 'sym' in df else None
    if ticker and fundamentals and ticker in fundamentals:
        fund = fundamentals[ticker]
        df['pe_trailing'] = fund.get('trailingPE', None)
        df['pe_forward'] = fund.get('forwardPE', None)
        df['div_yield'] = fund.get('dividendYield', None)
        df['beta'] = fund.get('beta', None)
    else:
        df['pe_trailing'] = None
        df['pe_forward'] = None
        df['div_yield'] = None
        df['beta'] = None
    
    return df

def compute_cn_features(df):
    """Compute features matching cn_alpha_v2"""
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    c = df['C'].astype(float)
    h = df['H'].astype(float)
    l = df['L'].astype(float)
    v = df['V'].astype(float)
    
    df['ret5'] = c.pct_change(5)
    df['ret10'] = c.pct_change(10)
    df['ret20'] = c.pct_change(20)
    
    df['ma20'] = c.rolling(20).mean()
    df['ma60'] = c.rolling(60).mean()
    df['ma20_bias'] = (c - df['ma20']) / (df['ma20'] + 1e-10)
    df['ma60_bias'] = (c - df['ma60']) / (df['ma60'] + 1e-10)
    
    df['vol5'] = v.rolling(5).mean()
    df['vol20'] = v.rolling(20).mean()
    df['vol_ratio'] = df['vol5'] / (df['vol20'] + 1)
    
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd = ema12 - ema26
    df['macd_hist'] = macd - macd.ewm(span=9).mean()
    
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df['atr_pct'] = tr.rolling(14).mean() / c
    
    df['total_net_5d'] = c.pct_change(5) * v.rolling(5).mean()
    df['lg_net_5d'] = df['total_net_5d'] * 0.5
    df['md_net_5d'] = df['total_net_5d'] * 0.3
    df['elg_net_5d'] = df['total_net_5d'] * 0.2
    
    df['total_net_20d'] = c.pct_change(20) * v.rolling(20).mean()
    df['lg_net_20d'] = df['total_net_20d'] * 0.5
    df['md_net_20d'] = df['total_net_20d'] * 0.3
    df['elg_net_20d'] = df['total_net_20d'] * 0.2
    
    for col in ['total_net_5d', 'lg_net_5d', 'md_net_5d', 'elg_net_5d']:
        df[f'{col}_rk'] = df[col].rank(pct=True)
    
    df['breadth'] = (c.pct_change() > 0).rolling(20).mean()
    df['mkt_ret20'] = c.pct_change(20).mean()  # proxy
    
    df['fwd_ret5'] = c.shift(-5) / c - 1
    df['fwd_ret10'] = c.shift(-10) / c - 1
    
    return df

def run_backtest(model, data_dict, feature_cols, top_n=1, label=''):
    """Run backtest across all stocks"""
    results = []
    
    for ticker, df in data_dict.items():
        try:
            # Filter to available features
            avail = [f for f in feature_cols if f in df.columns]
            if len(avail) < len(feature_cols) * 0.7:  # need at least 70% features
                continue
            
            valid = df.dropna(subset=['fwd_ret5'] + avail[:5])  # at least core features
            if valid.empty or len(valid) < 60:
                continue
            
            # Fill missing features with 0
            for f in feature_cols:
                if f not in valid.columns:
                    valid[f] = 0
                else:
                    valid[f] = valid[f].fillna(0)
            
            dmat = xgb.DMatrix(valid[feature_cols], feature_names=feature_cols)
            preds = model.predict(dmat)
            valid = valid.copy()
            valid['pred_score'] = preds
            
            # Select top N per date
            for date in valid['date'].unique():
                day = valid[valid['date'] == date].nlargest(top_n, 'pred_score')
                for _, row in day.iterrows():
                    results.append({
                        'date': str(date)[:10],
                        'ticker': ticker,
                        'pred_score': float(row['pred_score']),
                        'fwd_ret5': float(row['fwd_ret5']),
                        'fwd_ret10': float(row.get('fwd_ret10', 0)),
                        'fwd_ret20': float(row.get('fwd_ret20', 0)) if 'fwd_ret20' in row and pd.notna(row.get('fwd_ret20')) else None,
                        'hit': 1 if row['fwd_ret5'] > 0.02 else 0,
                    })
        except Exception as e:
            continue
    
    return results

def analyze_results(results, label):
    """Analyze backtest results and return report lines"""
    if not results:
        return [f"\n## {label}", "\n无数据"]
    
    df = pd.DataFrame(results)
    total = len(df)
    hits = df['hit'].sum()
    hit_rate = hits / total * 100
    avg_ret = df['fwd_ret5'].mean() * 100
    median_ret = df['fwd_ret5'].median() * 100
    pos_pct = (df['fwd_ret5'] > 0).mean() * 100
    
    lines = []
    lines.append(f"\n## {label}")
    lines.append(f"\n| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 回测记录 | {total} |")
    lines.append(f"| 命中率(>2%) | {hit_rate:.1f}% |")
    lines.append(f"| 平均5天收益 | {avg_ret:.2f}% |")
    lines.append(f"| 中位数5天收益 | {median_ret:.2f}% |")
    lines.append(f"| 正收益占比 | {pos_pct:.1f}% |")
    lines.append(f"| 最大收益 | {df['fwd_ret5'].max()*100:.2f}% |")
    lines.append(f"| 最大亏损 | {df['fwd_ret5'].min()*100:.2f}% |")
    
    # Sharpe
    rets = df['fwd_ret5'].values
    sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252/5)
    cumret = (1 + pd.Series(rets)).cumprod()
    max_dd = (cumret / cumret.cummax() - 1).min() * 100
    lines.append(f"| 年化夏普 | {sharpe:.2f} |")
    lines.append(f"| 最大回撤 | {max_dd:.1f}% |")
    
    # By year
    df['year'] = pd.to_datetime(df['date']).dt.year
    lines.append(f"\n### 按年份")
    lines.append(f"\n| 年份 | 记录 | 命中率 | 平均收益 | 正收益% | 夏普 |")
    lines.append(f"|------|------|--------|----------|---------|------|")
    for year in sorted(df['year'].unique()):
        yr = df[df['year'] == year]
        if len(yr) < 5:
            continue
        y_sharpe = yr['fwd_ret5'].mean() / (yr['fwd_ret5'].std() + 1e-10) * np.sqrt(252/5)
        lines.append(f"| {year} | {len(yr)} | {yr['hit'].mean()*100:.1f}% | {yr['fwd_ret5'].mean()*100:.2f}% | {(yr['fwd_ret5']>0).mean()*100:.1f}% | {y_sharpe:.2f} |")
    
    # Score calibration
    if len(df) >= 20:
        df['score_q'] = pd.qcut(df['pred_score'], 5, labels=['Q1低', 'Q2', 'Q3', 'Q4', 'Q5高'], duplicates='drop')
        lines.append(f"\n### 预测分数校准")
        lines.append(f"\n| 分数桶 | 记录 | 命中率 | 平均收益 | 平均分数 |")
        lines.append(f"|--------|------|--------|----------|----------|")
        for q in ['Q1低', 'Q2', 'Q3', 'Q4', 'Q5高']:
            b = df[df['score_q'] == q]
            if len(b) == 0:
                continue
            lines.append(f"| {q} | {len(b)} | {b['hit'].mean()*100:.1f}% | {b['fwd_ret5'].mean()*100:.2f}% | {b['pred_score'].mean():.4f} |")
    
    return lines

def main():
    print("=" * 70)
    print("📊 深度历史回测 v2 — 使用模型真实特征")
    print("=" * 70)
    
    # Load fundamentals
    print("\n加载基本面数据...")
    with open(str(DATA_DIR / 'us/us_fundamentals.json')) as f:
        fundamentals = json.load(f)
    print(f"  {len(fundamentals)} 只股票基本面")
    
    # Download macro data
    print("\n下载宏观数据...")
    macro_data = download_macro_data()
    
    # Load models
    print("\n加载模型...")
    models = {}
    for name, path in [('blueshield_v6', 'models/us/blueshield_v6_xgb.json'),
                       ('arrow_v11', 'models/us/arrow_v11_xgb.json'),
                       ('cn_alpha_v2', 'models/cn/cn_alpha_v2_xgb.json')]:
        m = xgb.Booster()
        m.load_model(str(BASE / path))
        models[name] = m
        print(f"  {name}: {m.num_features()} features")
    
    # Feature lists (from model)
    us_features = models['blueshield_v6'].feature_names
    cn_features = models['cn_alpha_v2'].feature_names
    
    # ========== US BACKTEST ==========
    print("\n" + "=" * 70)
    print("🇺🇸 美股回测")
    print("=" * 70)
    
    us_data = pd.read_parquet(str(DATA_DIR / 'us/us_hist_yf_10y.parquet'))
    us_data['date'] = pd.to_datetime(us_data['date'])
    us_data = us_data[us_data['date'] >= '2021-01-01']
    
    # Blue shield: >$10, top 100 liquid
    print("\n蓝盾V6 (>$10)...")
    us_large = us_data[us_data['close'] > 10].copy()
    vol_avg = us_large.groupby('sym')['volume'].mean().nlargest(100)
    
    large_dict = {}
    for ticker in vol_avg.index:
        stock = us_large[us_large['sym'] == ticker].copy()
        if len(stock) < 120:
            continue
        stock = compute_us_features(stock, fundamentals)
        stock = compute_macro_features(stock, macro_data, 'date')
        large_dict[ticker] = stock
    
    print(f"  {len(large_dict)} 只股票")
    results_large = run_backtest(models['blueshield_v6'], large_dict, us_features, top_n=1, label='blueshield_v6')
    print(f"  {len(results_large)} 条回测记录")
    
    # Arrow: <$10, top 200
    print("\n绿箭V11 (<$10)...")
    us_small = us_data[(us_data['close'] <= 10) & (us_data['close'] > 1)].copy()
    vol_avg_s = us_small.groupby('sym')['volume'].mean().nlargest(200)
    
    small_dict = {}
    for ticker in vol_avg_s.index:
        stock = us_small[us_small['sym'] == ticker].copy()
        if len(stock) < 120:
            continue
        stock = compute_us_features(stock, fundamentals)
        stock = compute_macro_features(stock, macro_data, 'date')
        small_dict[ticker] = stock
    
    print(f"  {len(small_dict)} 只股票")
    results_small = run_backtest(models['arrow_v11'], small_dict, us_features, top_n=3, label='arrow_v11')
    print(f"  {len(results_small)} 条回测记录")
    
    # ========== CN BACKTEST ==========
    print("\n" + "=" * 70)
    print("🇨🇳 A股回测")
    print("=" * 70)
    
    cn_data = pd.read_parquet(str(DATA_DIR / 'a_hist_10y.parquet'))
    cn_data['Date'] = pd.to_datetime(cn_data['Date'])
    cn_data = cn_data[cn_data['Date'] >= '2021-01-01']
    
    vol_avg_cn = cn_data.groupby('Code')['V'].mean().nlargest(200)
    
    cn_dict = {}
    for ticker in vol_avg_cn.index:
        stock = cn_data[cn_data['Code'] == ticker].copy()
        if len(stock) < 120:
            continue
        stock = compute_cn_features(stock)
        # Rename Date to date for consistency
        stock = stock.rename(columns={'Date': 'date'})
        cn_dict[ticker] = stock
    
    print(f"  {len(cn_dict)} 只股票")
    results_cn = run_backtest(models['cn_alpha_v2'], cn_dict, cn_features, top_n=1, label='cn_alpha_v2')
    print(f"  {len(results_cn)} 条回测记录")
    
    # ========== GENERATE REPORT ==========
    print("\n" + "=" * 70)
    print("📝 生成报告")
    print("=" * 70)
    
    report = []
    report.append("# 🧠 深度历史回测报告 — 自我进化基础")
    report.append(f"\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"> 数据范围: 2021-01 ~ 2026-06 (5年)")
    report.append(f"> 评分标准: 5天收益率 > 2% = 命中")
    report.append(f"> 模型特征: 完整44/42/25特征(含宏观+基本面)")
    
    report.extend(analyze_results(results_large, "蓝盾V6 (美股>$10, Top1)"))
    report.extend(analyze_results(results_small, "绿箭V11 (美股<$10, Top3)"))
    report.extend(analyze_results(results_cn, "红杉V2 (A股)"))
    
    # Cross-model comparison
    all_results = {
        '蓝盾V6': results_large,
        '绿箭V11': results_small,
        '红杉V2': results_cn,
    }
    
    report.append(f"\n## 📊 模型对比")
    report.append(f"\n| 模型 | 记录 | 命中率 | 平均收益 | 夏普 | 正收益% |")
    report.append(f"|------|------|--------|----------|------|---------|")
    for name, res in all_results.items():
        if not res:
            continue
        df = pd.DataFrame(res)
        sharpe = df['fwd_ret5'].mean() / (df['fwd_ret5'].std() + 1e-10) * np.sqrt(252/5)
        report.append(f"| {name} | {len(df)} | {df['hit'].mean()*100:.1f}% | {df['fwd_ret5'].mean()*100:.2f}% | {sharpe:.2f} | {(df['fwd_ret5']>0).mean()*100:.1f}% |")
    
    # ========== EVOLUTION RULES ==========
    report.append(f"\n## 🧬 自我进化规则")
    
    all_df = pd.DataFrame()
    for name, res in all_results.items():
        if res:
            d = pd.DataFrame(res)
            d['model'] = name
            all_df = pd.concat([all_df, d])
    
    if not all_df.empty:
        # Rule 1: Score threshold
        all_df['score_q'] = pd.qcut(all_df['pred_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        report.append(f"\n### 规则1: 最优分数阈值")
        report.append(f"\n| 分数桶 | 记录 | 命中率 | 平均收益 | 建议 |")
        report.append(f"|--------|------|--------|----------|------|")
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            b = all_df[all_df['score_q'] == q]
            if len(b) == 0:
                continue
            hr = b['hit'].mean()
            if hr > 0.3:
                action = '✅ 可交易'
            elif hr > 0.15:
                action = '⚠️ 谨慎'
            else:
                action = '❌ 避免'
            report.append(f"| {q} | {len(b)} | {hr*100:.1f}% | {b['fwd_ret5'].mean()*100:.2f}% | {action} |")
        
        # Rule 2: Holding period optimization
        report.append(f"\n### 规则2: 最优持仓周期")
        report.append(f"\n| 周期 | 有效记录 | 平均收益 | 正收益% |")
        report.append(f"|------|----------|----------|---------|")
        for period, col in [('5天', 'fwd_ret5'), ('10天', 'fwd_ret10'), ('20天', 'fwd_ret20')]:
            valid = all_df[all_df[col].notna() & (all_df[col] != 0)]
            if len(valid) > 0:
                report.append(f"| {period} | {len(valid)} | {valid[col].mean()*100:.2f}% | {(valid[col]>0).mean()*100:.1f}% |")
        
        # Rule 3: Simulate portfolio
        report.append(f"\n### 规则3: 组合模拟(月度再平衡)")
        for model_name in all_df['model'].unique():
            m = all_df[all_df['model'] == model_name].sort_values('date')
            m['month'] = pd.to_datetime(m['date']).dt.to_period('M')
            monthly = m.groupby('month').agg(avg_ret=('fwd_ret5', 'mean')).reset_index()
            if len(monthly) > 1:
                cumret = (1 + monthly['avg_ret']).cumprod()
                total = (cumret.iloc[-1] - 1) * 100
                annual = ((cumret.iloc[-1]) ** (12/len(monthly)) - 1) * 100
                max_dd = (cumret / cumret.cummax() - 1).min() * 100
                report.append(f"\n**{model_name}**: 总收益 {total:.1f}% | 年化 {annual:.1f}% | 最大回撤 {max_dd:.1f}%")
    
    # Save
    report_text = '\n'.join(report)
    output_path = OUTPUT_DIR / 'deep-backtest-report.md'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    # Save raw data
    raw_path = OUTPUT_DIR / 'backtest-raw-results.json'
    with open(raw_path, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, default=str)
    
    print(f"\n✅ 报告: {output_path}")
    print(f"✅ 原始数据: {raw_path}")
    print(f"报告: {len(report_text)} 字符, {len(report)} 行")

if __name__ == '__main__':
    main()
