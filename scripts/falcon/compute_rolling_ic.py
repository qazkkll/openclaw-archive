#!/usr/bin/env python3
"""
🦅 Falcon V0.4.6 滚动IC计算
==============================
每日运行，计算每个因子的滚动IC（lookback=126天），输出JSON供falcon_score.py使用。

原理:
  IC = Spearman(因子截面排名, 前瞻30天收益)
  对过去126天的IC取均值 → 该因子的当前权重依据
  
输出: data/falcon/factor_ic_weights.json
{
  "computed_at": "2026-07-03",
  "lookback": 126,
  "power": 0.5,
  "weights": {
    "fund_ratio": {"r_inventoryTurnover": 0.279, ...},
    "fund_growth": {...},
    ...
  },
  "ic_values": {
    "fund_ratio": {"r_inventoryTurnover": 0.080, ...},
    ...
  }
}
"""

import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
OUTPUT_PATH = DATA_DIR / "factor_ic_weights.json"

# V0.4.6 因子组 (与V0.4.4相同，53因子)
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

IC_LOOKBACK = 126
IC_POWER = 0.5


def load_data():
    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = df['date'].astype(str)
    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = prices_df['date'].astype(str)
    prices = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
    return df, prices


def compute_today_ic(df, prices, factor_cols):
    """计算所有历史日期的单日IC，然后取最近126天的均值。
    
    这是生产环境版本，每天运行一次。
    """
    all_dates = sorted(df['date'].unique())
    price_dates = sorted(prices.index.astype(str))
    
    # 只取最近126+30+5天的数据（节省计算量）
    lookback_needed = IC_LOOKBACK + 30 + 5
    recent_dates = all_dates[-lookback_needed:]
    
    # 预计算前瞻30天收益
    fwd_cache = {}
    for date in recent_dates:
        future_candidates = [d for d in price_dates if d > date]
        if len(future_candidates) < 20:
            continue
        future_date = future_candidates[min(29, len(future_candidates)-1)]
        if future_date not in prices.index or date not in prices.index:
            continue
        fwd_cache[date] = ((prices.loc[future_date] / prices.loc[date]) - 1).dropna()
    
    # 计算每个日期每个因子的截面IC
    daily_ic = {}
    for date in recent_dates:
        if date not in fwd_cache:
            continue
        day_df = df[df['date'] == date]
        if len(day_df) < 10:
            continue
        
        tickers = day_df['ticker'].values
        fwd = fwd_cache[date]
        common = [t for t in tickers if t in fwd.index]
        if len(common) < 30:
            continue
        
        fwd_vals = fwd[common].values
        daily_ic[date] = {}
        
        for col in factor_cols:
            if col not in day_df.columns:
                continue
            # 获取该因子在这些股票上的值
            vals = day_df.set_index('ticker').loc[common, col].values.astype(float)
            valid = ~(np.isnan(vals) | np.isnan(fwd_vals))
            if valid.sum() < 30:
                continue
            
            # 截面排名
            r = rankdata(vals[valid], method='average') / valid.sum()
            f = fwd_vals[valid]
            
            # 翻转
            if col in FLIP_FACTORS:
                r = 1.0 - r
            
            ic, _ = spearmanr(r, f)
            if not np.isnan(ic):
                daily_ic[date][col] = ic
    
    # 取最近126天的IC均值
    ic_dates = sorted(daily_ic.keys())[-IC_LOOKBACK:]
    
    factor_ics = {}
    for col in factor_cols:
        vals = [daily_ic[d].get(col, np.nan) for d in ic_dates if col in daily_ic.get(d, {})]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) >= 10:
            factor_ics[col] = float(np.mean(vals))
    
    return factor_ics


def main():
    t0 = time.time()
    print("🦅 Falcon V0.4.6 IC权重计算")
    print("=" * 50)
    
    # 收集所有因子
    all_factors = []
    for factors in FACTOR_GROUPS.values():
        all_factors.extend(factors)
    all_factors = list(set(all_factors))
    
    # 加载数据
    print("📂 加载数据...")
    df, prices = load_data()
    latest_date = sorted(df['date'].unique())[-1]
    print(f"  最新日期: {latest_date}")
    
    # 计算IC
    print(f"📊 计算滚动IC (lookback={IC_LOOKBACK}天, power={IC_POWER})...")
    factor_ics = compute_today_ic(df, prices, all_factors)
    print(f"  ✅ {len(factor_ics)}/{len(all_factors)} 个因子有有效IC")
    
    # 按组计算IC加权权重
    group_weights = {}
    group_ics = {}
    for group_name, factors in FACTOR_GROUPS.items():
        available = [f for f in factors if f in factor_ics]
        ic_vals = {f: factor_ics[f] for f in available}
        
        # 正IC的因子用IC^power加权
        positive = {f: max(0, v) ** IC_POWER for f, v in ic_vals.items()}
        total = sum(positive.values())
        
        if total > 0:
            weights = {f: round(positive[f] / total, 4) for f in positive}
        else:
            # 所有IC都是负的，回退等权
            weights = {f: round(1.0 / len(available), 4) for f in available}
        
        group_weights[group_name] = weights
        group_ics[group_name] = {f: round(v, 5) for f, v in ic_vals.items()}
    
    # 打印摘要
    print(f"\n📊 IC权重分布:")
    for group_name, weights in group_weights.items():
        n_positive = sum(1 for f, w in weights.items() if w > 0)
        n_total = len(weights)
        top = sorted(weights.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{f}({w:.1%})" for f, w in top)
        print(f"  {group_name}: {n_positive}/{n_total}正IC, Top3: {top_str}")
    
    # 保存
    output = {
        'computed_at': latest_date,
        'timestamp': datetime.now().isoformat(),
        'lookback': IC_LOOKBACK,
        'power': IC_POWER,
        'weights': group_weights,
        'ic_values': group_ics,
        'total_factors': len(all_factors),
        'factors_with_positive_ic': sum(1 for v in factor_ics.values() if v > 0),
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - t0
    print(f"\n✅ 完成 ({elapsed:.0f}s)")
    print(f"📁 输出: {OUTPUT_PATH}")
    
    return output


if __name__ == '__main__':
    main()
