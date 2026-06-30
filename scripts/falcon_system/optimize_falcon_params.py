#!/usr/bin/env python3
"""
Falcon 全参数优化 — 16维参数空间 + Walk-Forward
=================================================
优化: 8因子权重 + 信号阈值 + 仓位/调仓/风控/Pricer
评估: Val Sharpe为主, 过拟合比(Val/Train)为辅
保存: 每找到better立即写入JSON, 不丢结果

用法: python3 optimize_falcon_params.py [n_search=200]
"""
import sys, json, os, gc, time, warnings, random, copy
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = Path("/home/hermes/.hermes/openclaw-archive")
OUT_PATH = ROOT / "data/falcon/falcon_best_params.json"
CHECKPOINT_PATH = ROOT / "data/falcon/falcon_opt_checkpoint.json"

# ─── 因子组 → 特征列映射 ───
FACTOR_GROUPS = {
    'fund_growth': ['grossProfitMargin_qoq', 'netProfitMargin_qoq', 'operatingProfitMargin_qoq', 'ebitdaMargin_qoq'],
    'cashflow_balance': ['freeCashFlowOperatingCashFlowRatio', 'operatingCashFlowRatio',
                         'debtToEquityRatio', 'currentRatio', 'quickRatio', 'financialLeverageRatio'],
    'analyst': ['eps_revision', 'revenue_revision', 'num_analysts_eps', 'num_analysts_rev', 'eps_dispersion'],
    'grade_sentiment': ['priceToEarningsRatio', 'priceToBookRatio', 'priceToSalesRatio', 'priceToFreeCashFlowRatio', 'enterpriseValueMultiple'],
    'earnings': ['netProfitMargin', 'grossProfitMargin', 'operatingProfitMargin', 'ebitdaMargin'],
    'fund_metric': ['assetTurnover', 'inventoryTurnover', 'receivablesTurnover', 'dividendYieldPercentage', 'dividendPayoutRatio'],
    'insider': ['beta'],
}

INVERT_COLS = {'debtToEquityRatio', 'financialLeverageRatio'}

ALL_COLS = list(set(c for cols in FACTOR_GROUPS.values() for c in cols))

# ─── 参数搜索空间 ───
PARAM_SPACE = {
    # 因子权重 (7个, 会归一化) — cashflow+balance已合并(r=0.94)
    'w_fund_growth':      (0.02, 0.35),
    'w_cashflow_balance': (0.02, 0.35),
    'w_analyst':          (0.02, 0.30),
    'w_grade':            (0.02, 0.30),
    'w_earnings':         (0.01, 0.25),
    'w_fund_metric':      (0.01, 0.15),
    'w_insider':          (0.01, 0.12),
    # 信号阈值
    'buy_threshold':  (0.45, 0.65),
    'gg_rank':        (0.90, 0.98),  # 🟢🟢 rank阈值
    'g_rank':         (0.70, 0.85),  # 🟢 rank阈值
    # 仓位
    'top_n':          [5, 8, 10, 12, 15, 20, 25],
    'max_pos_pct':    (0.05, 0.15),
    'max_exposure':   (0.60, 0.95),
    # 调仓
    'hold_days':      [20, 30, 40, 60, 90],
    # 风控
    'stop_loss':      (-0.25, -0.08),
    'vix_threshold':  (18, 35),
    # Pricer
    'atr_mult':       (1.0, 3.0),
    'max_drop':       (0.03, 0.08),
}


def random_param():
    """随机一组参数"""
    p = {}
    for k, v in PARAM_SPACE.items():
        if isinstance(v, list):
            p[k] = random.choice(v)
        else:
            lo, hi = v
            p[k] = random.uniform(lo, hi)
    
    # 归一化因子权重
    w_keys = ['w_fund_growth', 'w_cashflow_balance', 'w_analyst', 'w_grade',
              'w_earnings', 'w_fund_metric', 'w_insider']
    total = sum(p[k] for k in w_keys)
    for k in w_keys:
        p[k] /= total
    
    # 约束: gg_rank > g_rank
    if p['gg_rank'] <= p['g_rank']:
        p['gg_rank'] = p['g_rank'] + 0.05
    
    return p


def load_data():
    """加载数据"""
    print("Loading features_v02.parquet...", flush=True)
    cols = ['ticker', 'date', 'close', 'volume'] + ALL_COLS
    df = pd.read_parquet(ROOT / "data/falcon/features_v02.parquet", columns=cols)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close', 'volume'])
    df = df[df['volume'] > 0]
    
    # 反向因子取反
    for c in INVERT_COLS:
        if c in df.columns:
            df[c] = -df[c]
    
    # ─── 数据完整性检查 ───
    print("\n  📊 数据完整性检查:", flush=True)
    fund_cols = [c for cols in FACTOR_GROUPS.values() for c in cols
                 if c not in ('beta',) and c in df.columns]
    tech_cols = ['beta']
    
    df['year'] = df['date'].dt.year
    gap_years = []
    for yr in sorted(df['year'].unique()):
        yr_data = df[df['year'] == yr]
        fund_cov = yr_data[fund_cols].notna().mean().mean() * 100 if fund_cols else 0
        tech_cov = yr_data[tech_cols].notna().mean().mean() * 100 if tech_cols else 0
        if fund_cov < 10:
            gap_years.append(yr)
            print(f"    {yr}: ⚠️ 基本面={fund_cov:.0f}% Beta={tech_cov:.0f}%", flush=True)
        else:
            print(f"    {yr}: ✅ 基本面={fund_cov:.0f}% Beta={tech_cov:.0f}%", flush=True)
    
    if gap_years:
        print(f"\n  🔴 警告: {len(gap_years)}年基本面数据缺失 ({gap_years[0]}-{gap_years[-1]})", flush=True)
        print(f"  影响: Train期因子组大部分为fillna(0.5), 实际只有beta在区分股票", flush=True)
        print(f"  建议: 补全基本面数据后重跑优化", flush=True)
    
    df = df.drop(columns=['year'])
    print(f"\n  {len(df):,} rows, {df['ticker'].nunique()} tickers", flush=True)
    return df


def compute_factor_scores(df, weights):
    """百分位排名 → 加权求和"""
    scores = pd.Series(0.0, index=df.index, dtype=np.float32)
    for group_name, cols in FACTOR_GROUPS.items():
        w = weights.get(group_name, 0)
        if w <= 0:
            continue
        avail = [c for c in cols if c in df.columns]
        if not avail:
            continue
        group_mean = df[avail].mean(axis=1)
        ranked = group_mean.groupby(df['date']).rank(pct=True)
        scores += w * ranked.fillna(0.5).astype(np.float32)
    return scores


def backtest(df, scores, hold_days, top_n, stop_loss, max_pos_pct, max_exposure):
    """回测 — 逐日追踪权益曲线, 计算真实盘中回撤"""
    d = df.copy()
    d['score'] = scores
    
    dates = sorted(d['date'].unique())
    rebal_dates = set(dates[::hold_days])
    
    # 当前持仓
    holdings = {}  # ticker -> (buy_price, weight)
    pv = 100000.0
    daily_curve = []
    stopped_out = set()  # 本期已止损的ticker
    
    for date in dates:
        day = d[d['date'] == date]
        if len(day) == 0:
            continue
        
        # 调仓日: 重新选股
        if date in rebal_dates:
            if len(day) < top_n:
                continue
            top = day.nlargest(top_n, 'score')
            n_actual = min(top_n, int(max_exposure / max_pos_pct))
            top = top.head(max(n_actual, 1))
            
            holdings = {}
            weight = 1.0 / len(top)
            for _, row in top.iterrows():
                holdings[row['ticker']] = {'buy_price': row['close'], 'weight': weight}
            stopped_out = set()
        
        if not holdings:
            continue
        
        # 逐日计算持仓价值
        total_ret = 0.0
        active_count = 0
        for ticker, h in holdings.items():
            if ticker in stopped_out:
                total_ret += h['weight'] * stop_loss
                continue
            ticker_row = day[day['ticker'] == ticker]
            if len(ticker_row) == 0:
                total_ret += h['weight'] * 0  # 无数据假设0收益
                continue
            current_price = ticker_row['close'].values[0]
            ret = current_price / h['buy_price'] - 1
            
            # 日内止损检查
            if ret < stop_loss:
                ret = stop_loss
                stopped_out.add(ticker)
            
            total_ret += h['weight'] * ret
            active_count += 1
        
        # 扣除交易成本(调仓日)
        if date in rebal_dates:
            total_ret -= 0.002
        
        daily_value = pv * (1 + total_ret)
        daily_curve.append({'date': date, 'value': daily_value})
    
    if len(daily_curve) < 20:
        return None
    
    e = np.array([c['value'] for c in daily_curve])
    days = len(e)
    years = days / 252  # 用交易日而非日历日
    cagr = (e[-1] / e[0]) ** (1 / max(years, 0.1)) - 1
    
    # 日度Sharpe (年化)
    daily_rets = np.diff(e) / e[:-1]
    sh = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
    
    # 真实盘中最大回撤
    dd = (e / np.maximum.accumulate(e) - 1).min()
    
    # 年度
    cdf = pd.DataFrame(daily_curve)
    cdf['year'] = cdf['date'].dt.year
    yearly = {}
    for yr, g in cdf.groupby('year'):
        if len(g) >= 5:
            yearly[int(yr)] = round((g['value'].iloc[-1] / g['value'].iloc[0] - 1) * 100, 1)
    
    return {'sharpe': float(sh), 'cagr': float(cagr * 100), 'dd': float(dd * 100), 'yearly': yearly}


def eval_cfg(df, p):
    """评估一个配置"""
    weights = {
        'fund_growth': p['w_fund_growth'],
        'cashflow_balance': p['w_cashflow_balance'],
        'analyst': p['w_analyst'],
        'grade_sentiment': p['w_grade'],
        'earnings': p['w_earnings'],
        'fund_metric': p['w_fund_metric'],
        'insider': p['w_insider'],
    }
    scores = compute_factor_scores(df, weights)
    return backtest(df, scores, p['hold_days'], p['top_n'], p['stop_loss'], p['max_pos_pct'], p['max_exposure'])


def save_checkpoint(best, n_done, n_total):
    """增量保存"""
    out = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M'),
        'n_searched': n_done,
        'n_total': n_total,
        'best_val_sharpe': best['val']['sharpe'],
        'best_config': best['config'],
        'best_train': best['train'],
        'best_val': best['val'],
        'best_test': best.get('test'),
        'all_top10': best.get('top10', []),
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(out, f, indent=2, default=str)


def main():
    n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print("=" * 70, flush=True)
    print(f"Falcon 全参数优化 — {n_total}次搜索, 16维参数空间", flush=True)
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M')}", flush=True)
    print("=" * 70, flush=True)
    
    t0 = time.time()
    df = load_data()
    
    # Walk-Forward splits (扩大Test集)
    # Train: 2016-2021 (6年, 含2020疫情)
    # Val:   2022-2023 (2年, 含加息熊市+AI牛市, 避免只看单一牛市)
    # Test:  2024-2026 (2.5年, 大样本OOS)
    train_end = pd.Timestamp('2021-12-31')
    val_end = pd.Timestamp('2023-12-31')
    
    df_train = df[df['date'] <= train_end].copy()
    df_val = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
    df_test = df[df['date'] > val_end].copy()
    
    print(f"Train: {df_train['date'].min().date()} ~ {df_train['date'].max().date()} ({len(df_train):,})", flush=True)
    print(f"Val:   {df_val['date'].min().date()} ~ {df_val['date'].max().date()} ({len(df_val):,})", flush=True)
    print(f"Test:  {df_test['date'].min().date()} ~ {df_test['date'].max().date()} ({len(df_test):,})", flush=True)
    
    # 基线
    baseline_p = {
        'w_fund_growth': 0.1875, 'w_cashflow_balance': 0.25, 'w_analyst': 0.15, 'w_grade': 0.15,
        'w_earnings': 0.125, 'w_fund_metric': 0.075, 'w_insider': 0.0625,
        'buy_threshold': 0.55, 'gg_rank': 0.95, 'g_rank': 0.80,
        'top_n': 10, 'max_pos_pct': 0.10, 'max_exposure': 0.80,
        'hold_days': 60, 'stop_loss': -0.15, 'vix_threshold': 25,
        'atr_mult': 1.5, 'max_drop': 0.05,
    }
    
    print("\n▶ 基线 (V0.3.2)...", flush=True)
    bl_tr = eval_cfg(df_train, baseline_p)
    bl_va = eval_cfg(df_val, baseline_p)
    bl_te = eval_cfg(df_test, baseline_p)
    print(f"  Train: SH={bl_tr['sharpe']:.2f} CAGR={bl_tr['cagr']:.1f}% DD={bl_tr['dd']:.1f}%")
    print(f"  Val:   SH={bl_va['sharpe']:.2f} CAGR={bl_va['cagr']:.1f}% DD={bl_va['dd']:.1f}%")
    print(f"  Test:  SH={bl_te['sharpe']:.2f} CAGR={bl_te['cagr']:.1f}% DD={bl_te['dd']:.1f}%", flush=True)
    
    # 最佳追踪
    best = {
        'val_sharpe': -999,
        'config': baseline_p,
        'train': bl_tr,
        'val': bl_va,
        'test': bl_te,
        'top10': []
    }
    
    # 搜索
    print(f"\n▶ 搜索 {n_total} 个配置...", flush=True)
    for i in range(n_total):
        p = random_param()
        
        tr = eval_cfg(df_train, p)
        va = eval_cfg(df_val, p)
        
        if tr is None or va is None:
            continue
        
        # 综合评分: val_sharpe为主, 过拟合惩罚
        # ratio = val/train, 偏离1越远说明越过拟合
        ratio = va['sharpe'] / tr['sharpe'] if tr['sharpe'] > 0 else 0
        consistency = 1 - min(abs(ratio - 1), 1)
        # 惩罚Val与Train偏离大的情况(过拟合信号)
        # 必须用abs: val>train是过拟合, val<train是欠拟合, 都要惩罚
        penalty = max(0, 1 - abs(va['sharpe'] - tr['sharpe']) / max(tr['sharpe'], 0.5))
        composite = va['sharpe'] * 0.5 + consistency * 0.25 + penalty * 0.25
        
        # Top10追踪
        entry = {
            'idx': i,
            'composite': float(composite),
            'val_sh': va['sharpe'],
            'train_sh': tr['sharpe'],
            'val_cagr': va['cagr'],
            'val_dd': va['dd'],
            'config': p
        }
        
        if len(best['top10']) < 10:
            best['top10'].append(entry)
            best['top10'].sort(key=lambda x: -x['composite'])
        elif composite > best['top10'][-1]['composite']:
            best['top10'][-1] = entry
            best['top10'].sort(key=lambda x: -x['composite'])
        
        if va['sharpe'] > best['val_sharpe']:
            best['val_sharpe'] = va['sharpe']
            best['config'] = p
            best['train'] = tr
            best['val'] = va
            
            # 测试集验证
            te = eval_cfg(df_test, p)
            best['test'] = te
            
            # 立即保存
            save_checkpoint(best, i + 1, n_total)
            
            te_sh = te['sharpe'] if te else 0
            print(f"  [{i+1}/{n_total}] ★ Val SH={va['sharpe']:.2f} CAGR={va['cagr']:.1f}% "
                  f"DD={va['dd']:.1f}% | Train SH={tr['sharpe']:.2f} | Test SH={te_sh:.2f} "
                  f"| top_n={p['top_n']} hold={p['hold_days']}d", flush=True)
        
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_total}] best_val_sh={best['val_sharpe']:.2f}", flush=True)
    
    # 最终输出
    elapsed = time.time() - t0
    print(f"\n{'='*70}", flush=True)
    print(f"完成! 耗时{elapsed/60:.1f}分钟, 搜索{n_total}个配置", flush=True)
    print(f"{'='*70}", flush=True)
    
    print(f"\n📊 基线 vs 最优:", flush=True)
    print(f"  {'指标':15s} {'基线':>10s} {'最优':>10s} {'变化':>10s}", flush=True)
    print(f"  {'Val Sharpe':15s} {bl_va['sharpe']:>10.2f} {best['val']['sharpe']:>10.2f} {best['val']['sharpe']-bl_va['sharpe']:>+10.2f}", flush=True)
    print(f"  {'Val CAGR':15s} {bl_va['cagr']:>10.1f}% {best['val']['cagr']:>10.1f}% {best['val']['cagr']-bl_va['cagr']:>+10.1f}%", flush=True)
    print(f"  {'Val DD':15s} {bl_va['dd']:>10.1f}% {best['val']['dd']:>10.1f}%", flush=True)
    if best['test']:
        print(f"  {'Test Sharpe':15s} {bl_te['sharpe']:>10.2f} {best['test']['sharpe']:>10.2f} {best['test']['sharpe']-bl_te['sharpe']:>+10.2f}", flush=True)
        print(f"  {'Test CAGR':15s} {bl_te['cagr']:>10.1f}% {best['test']['cagr']:>10.1f}%", flush=True)
    
    p = best['config']
    print(f"\n🔧 最优参数:", flush=True)
    w_names = ['fund_growth', 'cashflow_balance', 'analyst', 'grade_sentiment', 'earnings', 'fund_metric', 'insider']
    print(f"  因子权重:", flush=True)
    for n in w_names:
        k = f'w_{n}' if n != 'grade_sentiment' else 'w_grade'
        cur = baseline_p.get(k, 0)
        print(f"    {n:20s}: {p[k]:.4f} (基线{cur:.4f}, {p[k]-cur:+.4f})", flush=True)
    
    print(f"\n  信号阈值:", flush=True)
    print(f"    buy_threshold:  {p['buy_threshold']:.3f} (基线0.550)", flush=True)
    print(f"    🟢🟢 rank:      {p['gg_rank']:.3f} (基线0.950)", flush=True)
    print(f"    🟢 rank:        {p['g_rank']:.3f} (基线0.800)", flush=True)
    
    print(f"\n  仓位/调仓:", flush=True)
    print(f"    top_n:          {p['top_n']} (基线10)", flush=True)
    print(f"    max_pos_pct:    {p['max_pos_pct']*100:.1f}% (基线10%)", flush=True)
    print(f"    max_exposure:   {p['max_exposure']*100:.1f}% (基线80%)", flush=True)
    print(f"    hold_days:      {p['hold_days']}天 (基线60)", flush=True)
    
    print(f"\n  风控:", flush=True)
    print(f"    stop_loss:      {p['stop_loss']*100:.1f}% (基线-15%)", flush=True)
    print(f"    vix_threshold:  {p['vix_threshold']:.1f} (基线25)", flush=True)
    
    print(f"\n  Pricer:", flush=True)
    print(f"    atr_mult:       {p['atr_mult']:.2f} (基线1.50)", flush=True)
    print(f"    max_drop:       {p['max_drop']*100:.1f}% (基线5%)", flush=True)
    
    if best['val']['yearly']:
        print(f"\n📅 Val年度收益:", flush=True)
        for yr, ret in sorted(best['val']['yearly'].items()):
            print(f"    {yr}: {ret:+.1f}%", flush=True)
    
    if best.get('test') and best['test'].get('yearly'):
        print(f"\n📅 Test年度收益:", flush=True)
        for yr, ret in sorted(best['test']['yearly'].items()):
            print(f"    {yr}: {ret:+.1f}%", flush=True)
    
    # Top10概览
    print(f"\n🏆 Top 10 配置:", flush=True)
    for rank, entry in enumerate(best['top10'], 1):
        c = entry['config']
        print(f"  #{rank} Val SH={entry['val_sh']:.2f} Train SH={entry['train_sh']:.2f} "
              f"CAGR={entry['val_cagr']:.1f}% top_n={c['top_n']} hold={c['hold_days']}d "
              f"stop={c['stop_loss']*100:.0f}%", flush=True)
    
    print(f"\n✅ 结果保存到 {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
