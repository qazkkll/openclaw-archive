#!/usr/bin/env python3
"""
T2.2 Feature Pruning for Falcon V0.4.0
Selects final feature set from IC analysis + correlation pruning.
"""
import json
import pandas as pd
import numpy as np
from collections import defaultdict

PROJECT = "/home/hermes/.hermes/openclaw-archive"

# Domain classification
DOMAIN_MAP = {
    # Technical factors
    "ma5": "technical", "ma20": "technical", "ma60": "technical",
    "ma_bias20": "technical", "ma_align": "technical",
    "ma_cross_5_20": "technical", "ma_cross_20_60": "technical",
    "price_position": "technical",
    "ret1": "technical", "ret5": "technical", "ret10": "technical",
    "ret20": "technical", "ret30": "technical", "ret60": "technical", "ret90": "technical",
    "momentum_6m": "technical", "momentum_1m": "technical",
    "mom_divergence": "technical", "trend_accel": "technical",
    "vol20": "technical", "vol5": "technical", "vol_ratio": "technical",
    "vol_change": "technical", "vol_regime": "technical",
    "rsi14": "technical", "rsi_change": "technical", "rsi_zone": "technical",
    "macd": "technical", "macd_signal": "technical",
    "macd_hist": "technical", "macd_roc": "technical",
    "bb_std": "technical", "bb_width": "technical", "bb_pos": "technical",
    "ret_quality": "technical", "range_ratio": "technical", "avg_body": "technical",
    "vwap_drift": "technical", "dd_60": "technical", "ud_vol_ratio": "technical",
    "beta": "technical",
    # Fundamental factors
    "priceToEarningsRatio": "fundamental", "priceToBookRatio": "fundamental",
    "priceToSalesRatio": "fundamental", "priceToFreeCashFlowRatio": "fundamental",
    "enterpriseValueMultiple": "fundamental",
    "grossProfitMargin": "fundamental", "netProfitMargin": "fundamental",
    "operatingProfitMargin": "fundamental", "ebitdaMargin": "fundamental",
    "assetTurnover": "fundamental", "inventoryTurnover": "fundamental",
    "receivablesTurnover": "fundamental",
    "debtToEquityRatio": "fundamental", "currentRatio": "fundamental",
    "quickRatio": "fundamental", "financialLeverageRatio": "fundamental",
    "freeCashFlowOperatingCashFlowRatio": "fundamental",
    "operatingCashFlowRatio": "fundamental",
    "dividendYieldPercentage": "fundamental", "dividendPayoutRatio": "fundamental",
    "grossProfitMargin_qoq": "fundamental", "netProfitMargin_qoq": "fundamental",
    "operatingProfitMargin_qoq": "fundamental", "ebitdaMargin_qoq": "fundamental",
    # Analyst factors
    "eps_revision": "analyst", "revenue_revision": "analyst",
    "num_analysts_eps": "analyst", "num_analysts_rev": "analyst",
    "eps_dispersion": "analyst", "fmp_covered": "analyst",
    "analyst_covered": "analyst",
    # News factors
    "news_avg_sentiment": "news", "news_sentiment_vol": "news",
    "news_neg_ratio": "news", "news_pos_ratio": "news",
    "news_article_count": "news", "news_confidence_avg": "news",
}


def get_domain(name):
    if name in DOMAIN_MAP:
        return DOMAIN_MAP[name]
    if name.startswith("news_"):
        return "news"
    if "analyst" in name or "eps" in name or "rev" in name:
        return "analyst"
    if any(k in name for k in ["margin", "ratio", "turnover", "dividend", "debt", "leverage",
                                "cashFlow", "current", "quick", "ebitda", "operating",
                                "priceTo", "enterprise"]):
        return "fundamental"
    return "technical"


def main():
    # Load IC analysis
    with open(f"{PROJECT}/data/falcon/v04_ic_analysis.json") as f:
        ic_data = json.load(f)

    factors = ic_data["factors"]
    print(f"Total factors in IC analysis: {len(factors)}")

    # Load parquet
    df = pd.read_parquet(f"{PROJECT}/data/falcon/training_data_v04.parquet")
    print(f"Training data shape: {df.shape}")

    exclude_cols = {"date", "ticker", "open", "high", "low", "close", "volume",
                    "fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d", "fwd_ret_30d"}

    # Build IC lookup
    ic_lookup = {f["name"]: f for f in factors}

    # ============================================================
    # STEP 1: Remove weak factors (|ICIR| < 0.05)
    # ============================================================
    print("\n=== STEP 1: Remove weak factors (|ICIR| < 0.05) ===")
    weak_threshold = 0.05
    strong_factors = [f for f in factors if abs(f["icir"]) >= weak_threshold]
    weak_factors = [f for f in factors if abs(f["icir"]) < weak_threshold]

    removed_weak = []
    for f in weak_factors:
        removed_weak.append({
            "name": f["name"],
            "reason": f"|ICIR|={abs(f['icir']):.4f} < {weak_threshold} threshold",
            "icir": f["icir"],
            "domain": get_domain(f["name"]),
        })
        print(f"  REMOVED: {f['name']} (ICIR={f['icir']:.4f}) [{get_domain(f['name'])}]")

    print(f"Strong: {len(strong_factors)}, Weak removed: {len(weak_factors)}")

    # ============================================================
    # STEP 2: Compute correlation matrix
    # ============================================================
    print("\n=== STEP 2: Compute pairwise correlation matrix ===")
    remaining_names = [f["name"] for f in strong_factors]
    avail_cols = [c for c in remaining_names if c in df.columns]
    missing = [c for c in remaining_names if c not in df.columns]
    if missing:
        print(f"  Not in parquet: {missing}")

    # Sample for speed
    sample_size = min(100_000, len(df))
    sample_idx = np.random.RandomState(42).choice(len(df), size=sample_size, replace=False)
    corr_data = df.iloc[sample_idx][avail_cols].copy()
    corr_matrix = corr_data.corr(method="spearman")
    print(f"  Correlation matrix computed for {len(avail_cols)} factors")

    # ============================================================
    # STEP 3: Correlation-based pruning (|r| > 0.8)
    # ============================================================
    print("\n=== STEP 3: Remove highly correlated factors (|r| > 0.8) ===")
    corr_threshold = 0.8

    icir_lookup = {f["name"]: abs(f["icir"]) for f in strong_factors}

    # Group factors by correlation clusters
    # For each pair with |r| > threshold, keep the one with higher |ICIR|
    sorted_factors = sorted(strong_factors, key=lambda x: abs(x["icir"]), reverse=True)

    selected_names = []
    selected_set = set()
    removed_corr = []

    # Track domain counts to avoid removing last representative
    domain_counts = defaultdict(int)
    for f in strong_factors:
        domain_counts[get_domain(f["name"])] += 1

    for f in sorted_factors:
        name = f["name"]
        if name not in avail_cols:
            selected_names.append(name)
            selected_set.add(name)
            continue

        domain = get_domain(name)

        # Check correlation with already-selected factors
        dominated = False
        for sel_name in list(selected_set):
            if sel_name not in avail_cols:
                continue
            if sel_name not in corr_matrix.columns or name not in corr_matrix.index:
                continue
            r = abs(corr_matrix.loc[name, sel_name])
            if r > corr_threshold:
                sel_icir = icir_lookup.get(sel_name, 0)
                cur_icir = abs(f["icir"])

                # If this is the last representative of its domain, don't remove it
                remaining_in_domain = sum(1 for sn in selected_names
                                          if get_domain(sn) == domain and sn != name)
                if remaining_in_domain == 0 and domain_counts[domain] == 1:
                    print(f"  KEEP: {name} (ICIR={f['icir']:.4f}) - last {domain} representative")
                    continue

                if cur_icir <= sel_icir:
                    reason = f"|r|={r:.4f} > {corr_threshold} with {sel_name} (ICIR {sel_icir:.4f} > {cur_icir:.4f})"
                    removed_corr.append({
                        "name": name,
                        "reason": reason,
                        "icir": f["icir"],
                        "domain": domain,
                        "correlated_with": sel_name,
                        "r": r,
                    })
                    print(f"  REMOVED: {name} (ICIR={f['icir']:.4f}) - {reason}")
                    dominated = True
                    break

        if not dominated:
            selected_names.append(name)
            selected_set.add(name)

    print(f"After correlation pruning: {len(selected_names)} factors")

    # ============================================================
    # STEP 4: Ensure domain representation
    # ============================================================
    print("\n=== STEP 4: Ensure domain representation ===")
    selected_domains = set(get_domain(n) for n in selected_names)
    print(f"  Current domains: {selected_domains}")

    # News domain - not in IC analysis, add manually
    if "news" not in selected_domains:
        for nf in ["news_avg_sentiment", "news_article_count"]:
            if nf in df.columns and nf not in selected_set:
                selected_names.append(nf)
                selected_set.add(nf)
                print(f"  ADDED: {nf} (news domain coverage)")

    # Analyst domain - ensure at least 1 representative
    if "analyst" not in selected_domains:
        # Try num_analysts_eps first (highest ICIR among analyst factors)
        for af in ["num_analysts_eps", "eps_dispersion", "num_analysts_rev", "eps_revision"]:
            if af in df.columns and af not in selected_set:
                selected_names.append(af)
                selected_set.add(af)
                print(f"  ADDED: {af} (analyst domain coverage)")
                break

    # Re-check domains
    selected_domains = set(get_domain(n) for n in selected_names)
    print(f"  Final domains: {selected_domains}")

    # ============================================================
    # STEP 5: Final count check (30-50 range)
    # ============================================================
    print(f"\n=== STEP 5: Final count check ===")
    print(f"  Current count: {len(selected_names)}")

    if len(selected_names) < 30:
        # Add more factors from remaining pool (weak but not removed from IC analysis)
        # Re-include some weak factors for domain coverage
        print(f"  Only {len(selected_names)} - adding more factors to reach 30+")

        # Get factors from parquet that aren't yet selected
        news_factors = [c for c in df.columns if c.startswith("news_") and c not in exclude_cols and c not in selected_set]
        analyst_extra = [c for c in ["fmp_covered", "analyst_covered"] if c in df.columns and c not in selected_set]

        for nf in news_factors[:3]:
            selected_names.append(nf)
            selected_set.add(nf)
            print(f"  ADDED: {nf} (news enrichment)")

        for af in analyst_extra[:2]:
            selected_names.append(af)
            selected_set.add(af)
            print(f"  ADDED: {af} (analyst enrichment)")

        # Add some moderate-strength factors that survived pruning
        for f in sorted_factors:
            if f["name"] not in selected_set and f["name"] in df.columns:
                domain = get_domain(f["name"])
                # Prefer diverse domains
                if len(selected_names) >= 35:
                    break
                selected_names.append(f["name"])
                selected_set.add(f["name"])
                print(f"  ADDED: {f['name']} (ICIR={f['icir']:.4f}) [{domain}]")

    print(f"  Final count: {len(selected_names)}")

    # ============================================================
    # STEP 6: Build output
    # ============================================================
    print("\n=== STEP 6: Build final feature set ===")
    selected_features = []
    for name in selected_names:
        ic_info = ic_lookup.get(name, {})
        icir_val = ic_info.get("icir", None)
        domain = get_domain(name)

        if icir_val is not None:
            abs_icir = abs(icir_val)
            if abs_icir >= 0.2:
                reason = "Strong predictive signal (|ICIR| >= 0.2)"
            elif abs_icir >= 0.1:
                reason = "Moderate predictive signal (|ICIR| >= 0.1)"
            elif abs_icir >= 0.05:
                reason = "Weak but consistent signal (0.05 <= |ICIR| < 0.1)"
            else:
                reason = "Domain coverage requirement"
        else:
            if name.startswith("news_"):
                reason = "News domain coverage (not in IC analysis)"
            elif name in ["fmp_covered", "analyst_covered"]:
                reason = "Analyst domain coverage (not in IC analysis)"
            else:
                reason = "Included for domain coverage"

        selected_features.append({
            "name": name,
            "reason": reason,
            "icir": icir_val,
            "domain": domain,
        })

    all_removed = removed_weak + removed_corr

    # Summary
    by_domain_selected = defaultdict(int)
    by_domain_removed = defaultdict(int)
    for f in selected_features:
        by_domain_selected[f["domain"]] += 1
    for f in all_removed:
        by_domain_removed[f["domain"]] += 1

    output = {
        "selected_features": selected_features,
        "removed_features": all_removed,
        "summary": {
            "total_selected": len(selected_features),
            "total_removed": len(all_removed),
            "total_in_ic_analysis": len(factors),
            "by_domain": dict(by_domain_selected),
            "by_domain_removed": dict(by_domain_removed),
            "pruning_parameters": {
                "weak_threshold": weak_threshold,
                "correlation_threshold": corr_threshold,
                "ic_method": "Spearman rank correlation",
                "target": "fwd_ret_30d",
            },
        },
    }

    output_path = f"{PROJECT}/data/falcon/v04_feature_set.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"FINAL FEATURE SET: {len(selected_features)} features")
    print(f"{'='*60}")
    print(f"\nBy domain:")
    for domain, count in sorted(by_domain_selected.items()):
        print(f"  {domain}: {count}")
    print(f"\nRemoved: {len(all_removed)}")
    for domain, count in sorted(by_domain_removed.items()):
        print(f"  {domain}: {count}")
    print(f"\nSaved to: {output_path}")

    # Print selected features
    print(f"\n{'='*60}")
    print("SELECTED FEATURES:")
    print(f"{'='*60}")
    for i, f in enumerate(selected_features, 1):
        icir_str = f"{f['icir']:.4f}" if f['icir'] is not None else "N/A"
        print(f"  {i:2d}. {f['name']:45s} ICIR={icir_str:>8s}  [{f['domain']:12s}] {f['reason']}")


if __name__ == "__main__":
    main()
