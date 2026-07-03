#!/usr/bin/env python3
"""
🦅 Falcon V0.4.5 完整5层门禁审计
================================
5层审计：
  L1 数据完整性 - 覆盖率、日期范围、缺失值
  L2 特征一致性 - FLIP_FACTORS方向、因子组完整性
  L3 模型有效性 - IC/ICIR、t-stat、IC>0比例
  L4 信号生产 - 评分分布、信号等级
  L5 跨层一致性 - 代码与配置一致、版本号一致

用法: python3 scripts/falcon/falcon_v045_audit.py
输出: data/falcon/v045_audit_results.json
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
from pathlib import Path
from datetime import datetime

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
FEATURES_PATH = WORKSPACE / "data/falcon/features_v04_1.parquet"
PRICES_PATH = WORKSPACE / "data/falcon/us_prices_daily.parquet"
OUTPUT_PATH = WORKSPACE / "data/falcon/v045_audit_results.json"

# V0.4.5 配置
V045_CONFIG = {
    "version": "V0.4.5",
    "total_factors": 44,  # V0.4.5: fund_ratio(16)+fund_growth(12)+analyst(4)+income(4)+qoq(4)+cashflow(4)
    "main_weights": {
        "fund_ratio": 0.40,
        "growth_composite": 0.30,
        "qoq": 0.15,
        "cashflow": 0.15,
    },
    "gc_weights": {
        "fund_growth": 0.60,
        "analyst": 0.40,
        "income": 0.00,
    },
    "factor_groups": {
        "fund_ratio": [
            'r_priceToEarningsRatio', 'r_priceToSalesRatio',
            'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
            'r_grossProfitMargin', 'r_netProfitMargin', 'r_operatingProfitMargin', 'r_ebitdaMargin',
            'r_assetTurnover',
            'r_debtToEquityRatio', 'r_currentRatio', 'r_quickRatio',
            'r_freeCashFlowOperatingCashFlowRatio', 'r_operatingCashFlowRatio',
            'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
        ],
        "fund_growth": [
            'g_revenueGrowth', 'g_grossProfitGrowth', 'g_ebitgrowth',
            'g_operatingIncomeGrowth', 'g_netIncomeGrowth', 'g_epsdilutedGrowth',
            'g_freeCashFlowGrowth', 'g_tenYRevenueGrowthPerShare',
            'g_fiveYRevenueGrowthPerShare', 'g_threeYRevenueGrowthPerShare',
            'g_assetGrowth', 'g_bookValueperShareGrowth',
        ],
        "analyst": [
            'a_eps_revision', 'a_revenue_revision', 'a_eps_dispersion', 'a_num_analysts_eps',
        ],
        "income": [
            'i_gross_margin', 'i_ebitda_margin',
            'i_revenue_growth_yoy', 'i_gross_margin_delta',
        ],
        "qoq": [
            'r_grossProfitMargin_qoq', 'r_netProfitMargin_qoq',
            'r_operatingProfitMargin_qoq', 'r_ebitdaMargin_qoq',
        ],
        "cashflow": [
            'c_fcf_margin', 'c_capex_intensity', 'c_fcf_to_income', 'c_buyback_yield',
        ],
    },
    "flip_factors": {
        'r_priceToEarningsRatio', 'r_priceToSalesRatio',
        'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
        'r_debtToEquityRatio',
        'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
        'c_capex_intensity',
        'a_eps_revision', 'a_revenue_revision',
    },
    "removed_factors": [
        'r_priceToBookRatio', 'r_inventoryTurnover', 'r_financialLeverageRatio',
        'r_receivablesTurnover',
        'i_net_margin', 'i_operating_margin',
        'g_receivablesGrowth', 'g_inventoryGrowth', 'g_debtGrowth',
        'c_operating_cashflow_ratio',
    ],
}


def load_data():
    """加载特征和价格数据。"""
    print("📊 加载数据...")
    features = pd.read_parquet(FEATURES_PATH)
    features['date'] = pd.to_datetime(features['date'])
    
    prices = pd.read_parquet(PRICES_PATH)
    prices['date'] = pd.to_datetime(prices['date'])
    
    return features, prices


def audit_layer_1_data_integrity(features, prices):
    """L1 数据完整性审计。"""
    print("\n" + "="*60)
    print("L1 数据完整性审计")
    print("="*60)
    
    results = {
        "layer": "L1_data_integrity",
        "status": "PASS",
        "blockers": [],
        "warnings": [],
        "details": {},
    }
    
    # 1.1 日期范围
    date_min = features['date'].min()
    date_max = features['date'].max()
    date_range_days = (date_max - date_min).days
    results["details"]["date_range"] = {
        "min": str(date_min.date()),
        "max": str(date_max.date()),
        "days": date_range_days,
    }
    print(f"  📅 日期范围: {date_min.date()} → {date_max.date()} ({date_range_days}天)")
    
    # 1.2 股票覆盖
    n_tickers = features['ticker'].nunique()
    results["details"]["n_tickers"] = n_tickers
    print(f"  📊 股票数量: {n_tickers}")
    
    # 1.3 因子覆盖率
    all_factor_cols = []
    for group_name, group_cols in V045_CONFIG["factor_groups"].items():
        all_factor_cols.extend(group_cols)
    
    coverage = {}
    low_coverage_factors = []
    for col in all_factor_cols:
        if col in features.columns:
            cov = features[col].notna().mean()
            coverage[col] = round(cov, 4)
            if cov < 0.80:
                low_coverage_factors.append((col, cov))
                results["warnings"].append(f"因子 {col} 覆盖率={cov:.1%}<80%")
        else:
            coverage[col] = 0.0
            results["blockers"].append(f"因子 {col} 不存在于特征文件")
            results["status"] = "FAIL"
    
    results["details"]["coverage"] = coverage
    results["details"]["low_coverage_count"] = len(low_coverage_factors)
    
    if low_coverage_factors:
        print(f"  ⚠️ 低覆盖率因子: {len(low_coverage_factors)}")
        for col, cov in sorted(low_coverage_factors, key=lambda x: x[1])[:5]:
            print(f"    - {col}: {cov:.1%}")
    
    # 1.4 特征文件完整性
    expected_factors = V045_CONFIG["total_factors"]
    actual_factors = len([c for c in all_factor_cols if c in features.columns])
    results["details"]["factor_count"] = {
        "expected": expected_factors,
        "actual": actual_factors,
        "match": expected_factors == actual_factors,
    }
    
    if actual_factors != expected_factors:
        results["warnings"].append(f"因子数量不匹配: 期望{expected_factors}, 实际{actual_factors}")
        print(f"  ⚠️ 因子数量: 期望{expected_factors}, 实际{actual_factors}")
    else:
        print(f"  ✅ 因子数量: {actual_factors}/{expected_factors}")
    
    # 1.5 价格数据
    price_tickers = prices['ticker'].nunique()
    results["details"]["price_tickers"] = price_tickers
    print(f"  📈 价格数据: {price_tickers} 只股票")
    
    if results["blockers"]:
        results["status"] = "FAIL"
    
    return results


def audit_layer_2_feature_consistency(features):
    """L2 特征一致性审计。"""
    print("\n" + "="*60)
    print("L2 特征一致性审计")
    print("="*60)
    
    results = {
        "layer": "L2_feature_consistency",
        "status": "PASS",
        "blockers": [],
        "warnings": [],
        "details": {},
    }
    
    # 2.1 FLIP_FACTORS方向验证
    print("  🔍 验证FLIP_FACTORS方向...")
    flip_validation = {}
    
    # 计算30天前瞻收益
    features_sorted = features.sort_values(['ticker', 'date'])
    features_sorted['fwd_ret_30d'] = features_sorted.groupby('ticker')['close'].transform(
        lambda x: x.shift(-30) / x - 1
    )
    
    for factor in V045_CONFIG["flip_factors"]:
        if factor in features.columns:
            # 计算Spearman相关
            valid_data = features_sorted[[factor, 'fwd_ret_30d']].dropna()
            if len(valid_data) > 100:
                corr, p_value = stats.spearmanr(valid_data[factor], valid_data['fwd_ret_30d'])
                # 相关系数解释：
                # corr < -threshold = 数值越高收益越差 = 应该翻转 (is_flipped=True正确)
                # corr > threshold = 数值越高收益越好 = 不应该翻转 (is_flipped=False正确)
                # |corr| < threshold = 噪音，不判断方向
                threshold = 0.01  # 相关系数阈值，低于此值视为噪音
                is_flipped = factor in V045_CONFIG["flip_factors"]
                if abs(corr) < threshold:
                    # 相关系数接近0，视为噪音，跳过方向验证
                    flip_validation[factor] = {
                        "spearman_corr": round(float(corr), 4),
                        "p_value": round(float(p_value), 4),
                        "should_flip": "noise",
                        "is_flipped": is_flipped,
                        "correct": True,  # 噪音不判断
                    }
                    continue
                should_flip = corr < -threshold
                correct = (should_flip and is_flipped) or (not should_flip and not is_flipped)
                
                flip_validation[factor] = {
                    "spearman_corr": round(corr, 4),
                    "p_value": round(p_value, 4),
                    "should_flip": should_flip,
                    "is_flipped": is_flipped,
                    "correct": correct,
                }
                
                if not correct:
                    results["blockers"].append(
                        f"FLIP_FACTORS方向错误: {factor} 相关={corr:.4f}, "
                        f"应{'翻转' if should_flip else '不翻转'}但{'翻转了' if is_flipped else '没翻转'}"
                    )
                    results["status"] = "FAIL"
    
    results["details"]["flip_validation"] = flip_validation
    
    correct_count = sum(1 for v in flip_validation.values() if v["correct"])
    total_count = len(flip_validation)
    print(f"  ✅ FLIP_FACTORS方向: {correct_count}/{total_count} 正确")
    
    # 2.2 因子组完整性
    print("  🔍 验证因子组完整性...")
    group_integrity = {}
    for group_name, expected_cols in V045_CONFIG["factor_groups"].items():
        actual_cols = [c for c in expected_cols if c in features.columns]
        missing_cols = [c for c in expected_cols if c not in features.columns]
        
        group_integrity[group_name] = {
            "expected": len(expected_cols),
            "actual": len(actual_cols),
            "missing": missing_cols,
            "complete": len(missing_cols) == 0,
        }
        
        if missing_cols:
            results["warnings"].append(f"因子组 {group_name} 缺少因子: {missing_cols}")
            print(f"  ⚠️ {group_name}: 缺少 {missing_cols}")
        else:
            print(f"  ✅ {group_name}: {len(actual_cols)}/{len(expected_cols)} 因子")
    
    results["details"]["group_integrity"] = group_integrity
    
    # 2.3 移除因子验证
    print("  🔍 验证移除因子...")
    removed_still_present = []
    for factor in V045_CONFIG["removed_factors"]:
        if factor in features.columns:
            # 检查是否仍在FACTOR_GROUPS中
            for group_name, group_cols in V045_CONFIG["factor_groups"].items():
                if factor in group_cols:
                    removed_still_present.append(factor)
                    results["blockers"].append(f"已移除因子 {factor} 仍在因子组 {group_name} 中")
                    results["status"] = "FAIL"
    
    if removed_still_present:
        print(f"  ❌ 已移除因子仍在使用: {removed_still_present}")
    else:
        print(f"  ✅ 已移除因子未在因子组中")
    
    results["details"]["removed_still_present"] = removed_still_present
    
    return results


def audit_layer_3_model_effectiveness(features, prices):
    """L3 模型有效性审计。"""
    print("\n" + "="*60)
    print("L3 模型有效性审计")
    print("="*60)
    
    results = {
        "layer": "L3_model_effectiveness",
        "status": "PASS",
        "blockers": [],
        "warnings": [],
        "details": {},
    }
    
    # 计算复合因子的IC/ICIR
    print("  📊 计算复合因子IC/ICIR...")
    
    # 合并价格数据计算前瞻收益
    features_dated = features.copy()
    features_dated['date'] = pd.to_datetime(features_dated['date'])
    
    prices_dated = prices.copy()
    prices_dated['date'] = pd.to_datetime(prices_dated['date'])
    prices_dated = prices_dated.sort_values(['ticker', 'date'])
    prices_dated['fwd_ret_30d'] = prices_dated.groupby('ticker')['close'].transform(
        lambda x: x.shift(-30) / x - 1
    )
    
    merged = features_dated.merge(
        prices_dated[['ticker', 'date', 'fwd_ret_30d']],
        on=['ticker', 'date'],
        how='inner'
    )
    
    # 计算各因子组的IC
    group_ics = {}
    for group_name, group_cols in V045_CONFIG["factor_groups"].items():
        available_cols = [c for c in group_cols if c in merged.columns]
        if not available_cols:
            continue
        
        # 计算因子组平均排名
        daily_ics = []
        for date_val, group in merged.groupby('date'):
            if len(group) < 30:
                continue
            
            # 计算每个因子的排名
            rank_df = pd.DataFrame(index=group.index)
            for col in available_cols:
                if col in V045_CONFIG["flip_factors"]:
                    rank_df[col] = 1 - group[col].rank(pct=True)
                else:
                    rank_df[col] = group[col].rank(pct=True)
            
            # 平均排名作为因子组得分
            group_score = rank_df.mean(axis=1)
            
            valid_mask = group_score.notna() & group['fwd_ret_30d'].notna()
            if valid_mask.sum() >= 30:
                corr, _ = stats.spearmanr(group_score[valid_mask], group['fwd_ret_30d'][valid_mask])
                if np.isfinite(corr):
                    daily_ics.append(corr)
        
        if daily_ics:
            ic_mean = np.mean(daily_ics)
            ic_std = np.std(daily_ics)
            icir = ic_mean / ic_std if ic_std > 0 else 0
            ic_positive_pct = sum(1 for ic in daily_ics if ic > 0) / len(daily_ics)
            
            group_ics[group_name] = {
                "ic_mean": round(ic_mean, 4),
                "ic_std": round(ic_std, 4),
                "icir": round(icir, 4),
                "ic_positive_pct": round(ic_positive_pct, 4),
                "n_days": len(daily_ics),
                "t_stat": round(ic_mean / (ic_std / np.sqrt(len(daily_ics))), 2) if ic_std > 0 else 0,
            }
            
            print(f"  📊 {group_name}: IC={ic_mean:.4f}, ICIR={icir:.4f}, t={group_ics[group_name]['t_stat']:.2f}")
    
    # 计算复合因子IC
    print("  📊 计算复合因子IC/ICIR...")
    
    daily_composite_ics = []
    for date_val, group in merged.groupby('date'):
        if len(group) < 30:
            continue
        
        # 计算各因子组得分
        scores = {}
        
        # fund_ratio
        fund_ratio_cols = [c for c in V045_CONFIG["factor_groups"]["fund_ratio"] if c in group.columns]
        if fund_ratio_cols:
            rank_df = pd.DataFrame(index=group.index)
            for col in fund_ratio_cols:
                if col in V045_CONFIG["flip_factors"]:
                    rank_df[col] = 1 - group[col].rank(pct=True)
                else:
                    rank_df[col] = group[col].rank(pct=True)
            scores["fund_ratio"] = rank_df.mean(axis=1)
        
        # growth_composite
        fund_growth_cols = [c for c in V045_CONFIG["factor_groups"]["fund_growth"] if c in group.columns]
        analyst_cols = [c for c in V045_CONFIG["factor_groups"]["analyst"] if c in group.columns]
        income_cols = [c for c in V045_CONFIG["factor_groups"]["income"] if c in group.columns]
        
        fg_score = pd.Series(0.5, index=group.index)
        an_score = pd.Series(0.5, index=group.index)
        inc_score = pd.Series(0.5, index=group.index)
        
        if fund_growth_cols:
            rank_df = pd.DataFrame(index=group.index)
            for col in fund_growth_cols:
                if col in V045_CONFIG["flip_factors"]:
                    rank_df[col] = 1 - group[col].rank(pct=True)
                else:
                    rank_df[col] = group[col].rank(pct=True)
            fg_score = rank_df.mean(axis=1)
        
        if analyst_cols:
            rank_df = pd.DataFrame(index=group.index)
            for col in analyst_cols:
                if col in V045_CONFIG["flip_factors"]:
                    rank_df[col] = 1 - group[col].rank(pct=True)
                else:
                    rank_df[col] = group[col].rank(pct=True)
            an_score = rank_df.mean(axis=1)
        
        if income_cols:
            rank_df = pd.DataFrame(index=group.index)
            for col in income_cols:
                if col in V045_CONFIG["flip_factors"]:
                    rank_df[col] = 1 - group[col].rank(pct=True)
                else:
                    rank_df[col] = group[col].rank(pct=True)
            inc_score = rank_df.mean(axis=1)
        
        scores["growth_composite"] = (
            V045_CONFIG["gc_weights"]["fund_growth"] * fg_score +
            V045_CONFIG["gc_weights"]["analyst"] * an_score +
            V045_CONFIG["gc_weights"]["income"] * inc_score
        )
        
        # qoq
        qoq_cols = [c for c in V045_CONFIG["factor_groups"]["qoq"] if c in group.columns]
        if qoq_cols:
            rank_df = pd.DataFrame(index=group.index)
            for col in qoq_cols:
                if col in V045_CONFIG["flip_factors"]:
                    rank_df[col] = 1 - group[col].rank(pct=True)
                else:
                    rank_df[col] = group[col].rank(pct=True)
            scores["qoq"] = rank_df.mean(axis=1)
        
        # cashflow
        cashflow_cols = [c for c in V045_CONFIG["factor_groups"]["cashflow"] if c in group.columns]
        if cashflow_cols:
            rank_df = pd.DataFrame(index=group.index)
            for col in cashflow_cols:
                if col in V045_CONFIG["flip_factors"]:
                    rank_df[col] = 1 - group[col].rank(pct=True)
                else:
                    rank_df[col] = group[col].rank(pct=True)
            scores["cashflow"] = rank_df.mean(axis=1)
        
        # 复合得分
        composite_score = sum(
            V045_CONFIG["main_weights"][f] * scores[f]
            for f in V045_CONFIG["main_weights"]
            if f in scores
        )
        
        valid_mask = composite_score.notna() & group['fwd_ret_30d'].notna()
        if valid_mask.sum() >= 30:
            corr, _ = stats.spearmanr(composite_score[valid_mask], group['fwd_ret_30d'][valid_mask])
            if np.isfinite(corr):
                daily_composite_ics.append(corr)
    
    if daily_composite_ics:
        composite_ic = np.mean(daily_composite_ics)
        composite_ic_std = np.std(daily_composite_ics)
        composite_icir = composite_ic / composite_ic_std if composite_ic_std > 0 else 0
        composite_t = composite_ic / (composite_ic_std / np.sqrt(len(daily_composite_ics))) if composite_ic_std > 0 else 0
        composite_ic_positive = sum(1 for ic in daily_composite_ics if ic > 0) / len(daily_composite_ics)
        
        results["details"]["composite_ic"] = {
            "ic_mean": round(composite_ic, 4),
            "ic_std": round(composite_ic_std, 4),
            "icir": round(composite_icir, 4),
            "t_stat": round(composite_t, 2),
            "ic_positive_pct": round(composite_ic_positive, 4),
            "n_days": len(daily_composite_ics),
        }
        
        print(f"  📊 复合因子: IC={composite_ic:.4f}, ICIR={composite_icir:.4f}, t={composite_t:.2f}")
        
        # 检查阈值
        if composite_icir < 0.3:
            results["warnings"].append(f"复合因子ICIR={composite_icir:.4f}<0.3，处于边缘")
        
        if composite_t < 1.96:
            results["warnings"].append(f"复合因子t-stat={composite_t:.2f}<1.96，统计不显著")
    
    results["details"]["group_ics"] = group_ics
    
    return results


def audit_layer_4_signal_production(features):
    """L4 信号生产审计。"""
    print("\n" + "="*60)
    print("L4 信号生产审计")
    print("="*60)
    
    results = {
        "layer": "L4_signal_production",
        "status": "PASS",
        "blockers": [],
        "warnings": [],
        "details": {},
    }
    
    # 4.1 评分分布检查
    print("  📊 检查评分分布...")
    
    # 计算最新日期的评分
    latest_date = features['date'].max()
    latest_data = features[features['date'] == latest_date].copy()
    
    if len(latest_data) < 10:
        results["blockers"].append(f"最新日期 {latest_date} 只有 {len(latest_data)} 只股票")
        results["status"] = "FAIL"
        return results
    
    # 计算各因子组得分
    scores = {}
    
    # fund_ratio
    fund_ratio_cols = [c for c in V045_CONFIG["factor_groups"]["fund_ratio"] if c in latest_data.columns]
    if fund_ratio_cols:
        rank_df = pd.DataFrame(index=latest_data.index)
        for col in fund_ratio_cols:
            if col in V045_CONFIG["flip_factors"]:
                rank_df[col] = 1 - latest_data[col].rank(pct=True)
            else:
                rank_df[col] = latest_data[col].rank(pct=True)
        scores["fund_ratio"] = rank_df.mean(axis=1)
    
    # growth_composite
    fund_growth_cols = [c for c in V045_CONFIG["factor_groups"]["fund_growth"] if c in latest_data.columns]
    analyst_cols = [c for c in V045_CONFIG["factor_groups"]["analyst"] if c in latest_data.columns]
    income_cols = [c for c in V045_CONFIG["factor_groups"]["income"] if c in latest_data.columns]
    
    fg_score = pd.Series(0.5, index=latest_data.index)
    an_score = pd.Series(0.5, index=latest_data.index)
    inc_score = pd.Series(0.5, index=latest_data.index)
    
    if fund_growth_cols:
        rank_df = pd.DataFrame(index=latest_data.index)
        for col in fund_growth_cols:
            if col in V045_CONFIG["flip_factors"]:
                rank_df[col] = 1 - latest_data[col].rank(pct=True)
            else:
                rank_df[col] = latest_data[col].rank(pct=True)
        fg_score = rank_df.mean(axis=1)
    
    if analyst_cols:
        rank_df = pd.DataFrame(index=latest_data.index)
        for col in analyst_cols:
            if col in V045_CONFIG["flip_factors"]:
                rank_df[col] = 1 - latest_data[col].rank(pct=True)
            else:
                rank_df[col] = latest_data[col].rank(pct=True)
        an_score = rank_df.mean(axis=1)
    
    if income_cols:
        rank_df = pd.DataFrame(index=latest_data.index)
        for col in income_cols:
            if col in V045_CONFIG["flip_factors"]:
                rank_df[col] = 1 - latest_data[col].rank(pct=True)
            else:
                rank_df[col] = latest_data[col].rank(pct=True)
        inc_score = rank_df.mean(axis=1)
    
    scores["growth_composite"] = (
        V045_CONFIG["gc_weights"]["fund_growth"] * fg_score +
        V045_CONFIG["gc_weights"]["analyst"] * an_score +
        V045_CONFIG["gc_weights"]["income"] * inc_score
    )
    
    # qoq
    qoq_cols = [c for c in V045_CONFIG["factor_groups"]["qoq"] if c in latest_data.columns]
    if qoq_cols:
        rank_df = pd.DataFrame(index=latest_data.index)
        for col in qoq_cols:
            if col in V045_CONFIG["flip_factors"]:
                rank_df[col] = 1 - latest_data[col].rank(pct=True)
            else:
                rank_df[col] = latest_data[col].rank(pct=True)
        scores["qoq"] = rank_df.mean(axis=1)
    
    # cashflow
    cashflow_cols = [c for c in V045_CONFIG["factor_groups"]["cashflow"] if c in latest_data.columns]
    if cashflow_cols:
        rank_df = pd.DataFrame(index=latest_data.index)
        for col in cashflow_cols:
            if col in V045_CONFIG["flip_factors"]:
                rank_df[col] = 1 - latest_data[col].rank(pct=True)
            else:
                rank_df[col] = latest_data[col].rank(pct=True)
        scores["cashflow"] = rank_df.mean(axis=1)
    
    # 复合得分
    falcon_score = sum(
        V045_CONFIG["main_weights"][f] * scores[f]
        for f in V045_CONFIG["main_weights"]
        if f in scores
    )
    
    # 评分分布
    score_stats = {
        "mean": round(float(falcon_score.mean()), 4),
        "std": round(float(falcon_score.std()), 4),
        "min": round(float(falcon_score.min()), 4),
        "max": round(float(falcon_score.max()), 4),
        "median": round(float(falcon_score.median()), 4),
        "q25": round(float(falcon_score.quantile(0.25)), 4),
        "q75": round(float(falcon_score.quantile(0.75)), 4),
    }
    
    results["details"]["score_distribution"] = score_stats
    print(f"  📊 评分分布: mean={score_stats['mean']:.4f}, std={score_stats['std']:.4f}")
    print(f"  📊 范围: [{score_stats['min']:.4f}, {score_stats['max']:.4f}]")
    
    # 4.2 信号等级分布
    print("  📊 检查信号等级...")
    
    rank_pct = falcon_score.rank(pct=True)
    
    def score_to_signal(score, pct):
        if score >= 0.55 and pct >= 0.95:
            return "🟢🟢"
        elif score >= 0.55 and pct >= 0.80:
            return "🟢"
        elif score >= 0.50:
            return "🟡"
        else:
            return "🔴"
    
    signals = [score_to_signal(s, p) for s, p in zip(falcon_score, rank_pct)]
    signal_dist = {
        "🟢🟢": signals.count("🟢🟢"),
        "🟢": signals.count("🟢"),
        "🟡": signals.count("🟡"),
        "🔴": signals.count("🔴"),
    }
    
    results["details"]["signal_distribution"] = signal_dist
    print(f"  📊 信号分布: 🟢🟢={signal_dist['🟢🟢']}, 🟢={signal_dist['🟢']}, 🟡={signal_dist['🟡']}, 🔴={signal_dist['🔴']}")
    
    # 4.3 Top-10检查
    print("  📊 检查Top-10...")
    top10_idx = falcon_score.nlargest(10).index
    top10_tickers = latest_data.loc[top10_idx, 'ticker'].tolist()
    top10_scores = falcon_score.loc[top10_idx].tolist()
    
    results["details"]["top10"] = {
        "tickers": top10_tickers,
        "scores": [round(s, 4) for s in top10_scores],
    }
    print(f"  📊 Top-10: {top10_tickers[:5]}...")
    
    # 4.4 NaN检查
    nan_count = falcon_score.isna().sum()
    if nan_count > 0:
        results["warnings"].append(f"评分有 {nan_count} 个NaN值")
        print(f"  ⚠️ NaN值: {nan_count}")
    else:
        print(f"  ✅ 无NaN值")
    
    results["details"]["nan_count"] = int(nan_count)
    
    return results


def audit_layer_5_cross_layer_consistency():
    """L5 跨层一致性审计。"""
    print("\n" + "="*60)
    print("L5 跨层一致性审计")
    print("="*60)
    
    results = {
        "layer": "L5_cross_layer_consistency",
        "status": "PASS",
        "blockers": [],
        "warnings": [],
        "details": {},
    }
    
    # 5.1 代码与配置一致
    print("  🔍 验证代码与配置一致性...")
    
    falcon_score_path = WORKSPACE / "scripts/falcon/falcon_score.py"
    if falcon_score_path.exists():
        with open(falcon_score_path) as f:
            code_content = f.read()
        
        # 检查版本号
        if "V0.4.5" in code_content:
            results["details"]["version_in_code"] = True
            print(f"  ✅ 代码版本号: V0.4.5")
        else:
            results["details"]["version_in_code"] = False
            results["blockers"].append("代码中未找到V0.4.5版本号")
            results["status"] = "FAIL"
            print(f"  ❌ 代码版本号: 未找到V0.4.5")
        
        # 检查主权重
        if "fund_ratio\": 0.40" in code_content:
            results["details"]["main_weights_correct"] = True
            print(f"  ✅ 主权重: fund_ratio=0.40")
        else:
            results["details"]["main_weights_correct"] = False
            results["blockers"].append("主权重配置不正确")
            results["status"] = "FAIL"
            print(f"  ❌ 主权重: 配置不正确")
        
        # 检查子组权重
        if "\"income\": 0.00" in code_content:
            results["details"]["gc_weights_correct"] = True
            print(f"  ✅ 子组权重: income=0.00")
        else:
            results["details"]["gc_weights_correct"] = False
            results["warnings"].append("子组权重income可能不为0")
            print(f"  ⚠️ 子组权重: income可能不为0")
    
    # 5.2 特征文件与配置一致
    print("  🔍 验证特征文件与配置一致性...")
    
    features_path = WORKSPACE / "data/falcon/features_v04_1.parquet"
    if features_path.exists():
        features = pd.read_parquet(features_path)
        
        # 检查移除因子是否仍在特征文件中（但不在因子组中）
        removed_in_features = []
        for factor in V045_CONFIG["removed_factors"]:
            if factor in features.columns:
                # 检查是否在因子组中
                in_group = False
                for group_name, group_cols in V045_CONFIG["factor_groups"].items():
                    if factor in group_cols:
                        in_group = True
                        break
                
                if not in_group:
                    removed_in_features.append(factor)
        
        if removed_in_features:
            results["details"]["removed_in_features"] = removed_in_features
            print(f"  ℹ️ 已移除因子仍在特征文件中: {len(removed_in_features)}个（不影响评分）")
        else:
            print(f"  ✅ 已移除因子不在特征文件中")
    
    # 5.3 评分文件一致性
    print("  🔍 验证评分文件一致性...")
    
    scored_files = list(WORKSPACE.glob("data/falcon/falcon_v045_scored_*.json"))
    if scored_files:
        latest_scored = max(scored_files, key=lambda f: f.stem)
        with open(latest_scored) as f:
            scored_data = json.load(f)
        
        if scored_data.get("version") == "V0.4.5":
            results["details"]["scored_version_correct"] = True
            print(f"  ✅ 评分文件版本: V0.4.5")
        else:
            results["details"]["scored_version_correct"] = False
            results["warnings"].append(f"评分文件版本不正确: {scored_data.get('version')}")
            print(f"  ⚠️ 评分文件版本: {scored_data.get('version')}")
        
        # 检查权重
        scored_weights = scored_data.get("config", {}).get("weights", {})
        if scored_weights == V045_CONFIG["main_weights"]:
            results["details"]["scored_weights_correct"] = True
            print(f"  ✅ 评分文件权重正确")
        else:
            results["details"]["scored_weights_correct"] = False
            results["warnings"].append("评分文件权重不匹配")
            print(f"  ⚠️ 评分文件权重不匹配")
    else:
        print(f"  ⚠️ 未找到V0.4.5评分文件")
    
    return results


def main():
    """运行完整5层审计。"""
    print("🦅 Falcon V0.4.5 完整5层门禁审计")
    print("="*60)
    print(f"时间: {datetime.now().isoformat()}")
    print(f"配置: {V045_CONFIG['total_factors']}因子, {len(V045_CONFIG['factor_groups'])}组")
    print(f"权重: {V045_CONFIG['main_weights']}")
    print(f"GC权重: {V045_CONFIG['gc_weights']}")
    
    # 加载数据
    features, prices = load_data()
    
    # 运行5层审计
    audit_results = {
        "timestamp": datetime.now().isoformat(),
        "config": V045_CONFIG,
        "layers": {},
        "overall_status": "PASS",
        "blockers": [],
        "warnings": [],
    }
    
    # L1
    l1 = audit_layer_1_data_integrity(features, prices)
    audit_results["layers"]["L1"] = l1
    audit_results["blockers"].extend(l1.get("blockers", []))
    audit_results["warnings"].extend(l1.get("warnings", []))
    
    # L2
    l2 = audit_layer_2_feature_consistency(features)
    audit_results["layers"]["L2"] = l2
    audit_results["blockers"].extend(l2.get("blockers", []))
    audit_results["warnings"].extend(l2.get("warnings", []))
    
    # L3
    l3 = audit_layer_3_model_effectiveness(features, prices)
    audit_results["layers"]["L3"] = l3
    audit_results["blockers"].extend(l3.get("blockers", []))
    audit_results["warnings"].extend(l3.get("warnings", []))
    
    # L4
    l4 = audit_layer_4_signal_production(features)
    audit_results["layers"]["L4"] = l4
    audit_results["blockers"].extend(l4.get("blockers", []))
    audit_results["warnings"].extend(l4.get("warnings", []))
    
    # L5
    l5 = audit_layer_5_cross_layer_consistency()
    audit_results["layers"]["L5"] = l5
    audit_results["blockers"].extend(l5.get("blockers", []))
    audit_results["warnings"].extend(l5.get("warnings", []))
    
    # 总结
    if audit_results["blockers"]:
        audit_results["overall_status"] = "FAIL"
    
    print("\n" + "="*60)
    print("审计总结")
    print("="*60)
    print(f"状态: {audit_results['overall_status']}")
    print(f"阻断项: {len(audit_results['blockers'])}")
    print(f"警告项: {len(audit_results['warnings'])}")
    
    if audit_results["blockers"]:
        print("\n❌ 阻断项:")
        for blocker in audit_results["blockers"]:
            print(f"  - {blocker}")
    
    if audit_results["warnings"]:
        print("\n⚠️ 警告项:")
        for warning in audit_results["warnings"][:5]:
            print(f"  - {warning}")
        if len(audit_results["warnings"]) > 5:
            print(f"  ... 还有 {len(audit_results['warnings']) - 5} 项")
    
    # 保存结果
    # 转换set和bool为JSON可序列化类型
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, (int, np.integer)):
            return int(obj)
        elif isinstance(obj, (float, np.floating)):
            return float(obj)
        return obj
    
    audit_results_serializable = convert_for_json(audit_results)
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(audit_results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 审计结果已保存: {OUTPUT_PATH}")
    
    return audit_results


if __name__ == "__main__":
    main()
