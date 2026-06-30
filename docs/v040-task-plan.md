# Falcon V0.4.0 Task Plan — XGBoost Model

> Created: 2026-07-01
> Status: PLANNING
> Goal: Replace V0.3.1 linear scoring (Sharpe=1.431) with XGBoost regression model (target: Sharpe > 1.5)

## Current State

| Item | Value |
|------|-------|
| Production model | V0.3.1: fund_ratio(0.70)+analyst(0.20)+fund_metric(0.10) |
| WF Sharpe (baseline) | 1.431 |
| MaxDD | -22.0% |
| Data | features_v02.parquet: 80 cols, 1.2M rows, 476 tickers, 2016-2026 |
| FinBERT news | 24,771 articles, 476 tickers, 2024-01→2026-02 |
| FMP Premium | 10 files, 75MB manifest |
| Backtest engine | backtest_engine.py (BacktestEngine class, expanding window WF) |
| Key pitfall | Classification has NEGATIVE EV at all thresholds → must use RANKING |

## Dependencies (Task IDs)

```
T1.1 → T1.2 → T1.3 → T1.4 → T2.1 → T2.2 → T3.1 → T3.2 → T3.3 → T4.1 → T4.2 → T4.3 → T5.1 → T5.2 → T5.3
T1.3 → T2.1 (feature selection needs merged data)
T2.2 → T3.1 (feature set needed for training)
T3.3 → T4.1 (trained model needed for backtest)
T4.3 → T5.1 (validated model for deployment)
```

---

## Phase 1: Data Preparation

### T1.1 — Feature Audit & Gap Analysis

| Field | Detail |
|-------|--------|
| **Task ID** | T1.1 |
| **Goal** | Audit all 78 existing factors + identify gaps for XGBoost |
| **Input** | `data/falcon/features_v02.parquet` (80 cols), `falcon_v03_engine.py` (PIT logic) |
| **Output** | `v04_feature_audit.json`: factor list with IC/coverage/missing rate per year |
| **Est. time** | 2 hours |
| **Dependencies** | None |
| **Risk** | parquet基本面字段是假数据(beta=1.0, PE单值跨所有日期) — 不影响(PIT从FMP JSON取) |
| **Success criteria** | Complete factor inventory: 78 technical + fundamental factors documented, coverage rates by year computed, gaps identified |

**Details:**
- Extract all 80 column names from features_v02.parquet
- Classify: technical (ret1-ret90, momentum, vol, rsi, macd, bb_*) vs fundamental (PE, PB, PS, margins, turnover, leverage, FCF ratios)
- Compute annual coverage rate per factor (2016-2026)
- Identify factors with <80% coverage in any year
- Check for data staleness (latest date per ticker)
- Document which factors are PIT-safe (from FMP JSON) vs potentially look-ahead

---

### T1.2 — Target Variable Definition

| Field | Detail |
|-------|--------|
| **Task ID** | T1.2 |
| **Goal** | Define and compute forward return targets for XGBoost regression |
| **Input** | `data/falcon/us_prices_daily.parquet` (OHLCV) |
| **Output** | `data/falcon/targets_v04.parquet`: (ticker, date, fwd_ret_5d, fwd_ret_10d, fwd_ret_20d, fwd_ret_30d) |
| **Est. time** | 1 hour |
| **Dependencies** | T1.1 (need to understand feature dates) |
| **Risk** | Target definition affects model — too short = noise, too long = stale. V0.3.1 uses 30d hold. |
| **Success criteria** | Target variables computed for all (ticker, date) pairs, forward returns aligned correctly, no look-ahead |

**Details:**
- Compute 5d, 10d, 20d, 30d forward returns for each ticker-date
- Primary target: `fwd_ret_30d` (matches V0.3.1 hold period)
- Secondary: `fwd_ret_5d` and `fwd_ret_10d` (for faster-turnover experiments)
- Validate: no NaN in first 30 days of each ticker's history (expected)
- Save to parquet with date/ticker index
- **PIT safety**: target computed from close prices, naturally PIT-safe (forward-looking target is what we predict)

---

### T1.3 — FinBERT News Feature Engineering

| Field | Detail |
|-------|--------|
| **Task ID** | T1.3 |
| **Goal** | Convert raw FinBERT sentiment scores into time-series features per ticker |
| **Input** | `data/finbert_sentiment/all_scored.parquet` (24,771 articles, 9 cols) |
| **Output** | `data/falcon/news_features_v04.parquet`: per-ticker monthly aggregates |
| **Est. time** | 3 hours |
| **Dependencies** | T1.1 |
| **Risk** | FinBERT data only covers 2024-01→2026-02 (2 years). Earlier data unavailable. Coverage gap 2016-2023. |
| **Success criteria** | Monthly sentiment features: avg_sentiment, sentiment_vol, neg_article_ratio, article_count, confidence_avg. All aligned to month-end for PIT. |

**Details:**
- Group by (ticker, month)
- Compute per-month:
  - `news_avg_sentiment`: mean(sentiment)
  - `news_sentiment_vol`: std(sentiment) — disagreement
  - `news_neg_ratio`: count(sentiment < -0.3) / count(all)
  - `news_pos_ratio`: count(sentiment > 0.3) / count(all)
  - `news_article_count`: total articles
  - `news_confidence_avg`: mean(confidence)
- **PIT alignment**: month-end date, published_at ensures only past articles included
- Handle ticker-months with 0 articles → fill with NaN (XGBoost handles NaN natively)
- **Critical gap**: No FinBERT data before 2024-01. For 2016-2023, these features will be NaN.

---

### T1.4 — Data Merge & Quality Gate

| Field | Detail |
|-------|--------|
| **Task ID** | T1.4 |
| **Goal** | Merge all features into a single training dataset with quality checks |
| **Input** | features_v02.parquet + news_features_v04.parquet + targets_v04.parquet + FMP JSON (PIT) |
| **Output** | `data/falcon/training_data_v04.parquet` |
| **Est. time** | 2 hours |
| **Dependencies** | T1.1, T1.2, T1.3 |
| **Risk** | Date alignment errors, FMP PIT delay (filing date + 33 days), factor count explosion |
| **Success criteria** | Unified dataset with all features + targets, coverage report passes gate (>80% per year), no look-ahead |

**Details:**
- Join features_v02 (technical + fundamental parquet) with news_features (FinBERT)
- Attach target variables (forward returns)
- Validate: column count = 78 + 6 news + 4 targets = ~88
- Data quality gate:
  - Each year must have >80% coverage of active features
  - At least 400 tickers per year
  - No date gaps > 5 trading days
- Save training dataset + data quality report

---

## Phase 2: Feature Selection

### T2.1 — IC/ICIR Analysis

| Field | Detail |
|-------|--------|
| **Task ID** | T2.1 |
| **Goal** | Compute Information Coefficient (IC) and ICIR for all factors |
| **Input** | `training_data_v04.parquet` |
| **Output** | `v04_ic_analysis.json`: per-factor IC, ICIR, t-stat, p-value, coverage |
| **Est. time** | 2 hours |
| **Dependencies** | T1.4 |
| **Risk** | ICIR can be misleading (fund_ratio ICIR=-0.045 but WF strongest). Always validate with WF. |
| **Success criteria** | All factors ranked by |ICIR|, factors with ICIR<0.05 flagged for removal |

**Details:**
- Compute rank IC: Spearman correlation between factor rank and fwd_ret_30d rank, per date
- Aggregate: mean IC → IC, std(IC) → ICIR = IC/std(IC)
- Per-year IC stability check
- Key insight from V0.3.1: ICIR analysis can be WRONG (fund_ratio ICIR=-0.045 but WF strongest). IC analysis is screening, not selection.

---

### T2.2 — Feature Pruning & Final Selection

| Field | Detail |
|-------|--------|
| **Task ID** | T2.2 |
| **Goal** | Select final feature set for XGBoost training |
| **Input** | `v04_ic_analysis.json` + correlation matrix + domain knowledge |
| **Output** | `v04_feature_set.json`: list of ~30-50 features, reasoning for each |
| **Est. time** | 2 hours |
| **Dependencies** | T2.1 |
| **Risk** | Over-pruning removes signal. Under-pruning adds noise. XGBoost handles some noise well. |
| **Success criteria** | Feature set with <50% pairwise correlation, ICIR>0.05 for all selected, covers each domain |

**Details:**
- Remove factors with ICIR < 0.05
- Compute pairwise correlation matrix, remove one of each pair with |r| > 0.8
- Keep at least 1 representative from each domain (tech, fundamental, analyst, news)
- XGBoost handles NaN natively → don't need to impute FinBERT gaps
- Document final selection with reasoning
- Expected: ~35-50 features (from 78+6 = 84)

---

## Phase 3: Model Training

### T3.1 — XGBoost Regression Baseline

| Field | Detail |
|-------|--------|
| **Task ID** | T3.1 |
| **Goal** | Train initial XGBoost regression model with default hyperparameters |
| **Input** | `training_data_v04.parquet` + `v04_feature_set.json` |
| **Output** | `models/falcon_v04_xgb_baseline.joblib`: initial model |
| **Est. time** | 3 hours |
| **Dependencies** | T2.2 |
| **Risk** | XGBoost regression with Sharpe 1.69 but -68% DD (from quant-model-development). Need risk control. |
| **Success criteria** | Model trains successfully, in-sample IC>0, rank correlation positive, no overfitting signals |

**Details:**
```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

# XGBoost regression (NOT classification)
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=20,
    objective='reg:squarederror',  # Regression, not classification
    tree_method='hist',  # Fast
    random_state=42,
    n_jobs=-1
)
```
- Target: `fwd_ret_30d` (regression)
- In-sample evaluation: IC, Sharpe on train set
- Feature importance from model
- **No classification**: quant-model-development proved classification has NEGATIVE EV

---

### T3.2 — Walk-Forward Training Framework

| Field | Detail |
|-------|--------|
| **Task ID** | T3.2 |
| **Goal** | Implement proper Walk-Forward validation with expanding windows |
| **Input** | `training_data_v04.parquet` + `backtest_engine.py` |
| **Output** | `v04_walk_forward_results.json`: per-window results |
| **Est. time** | 4 hours |
| **Dependencies** | T3.1 |
| **Risk** | WF with 3yr train produces test periods ALL in bull markets (from quant-model-development). Must use 5yr train. |
| **Success criteria** | WF Sharpe > 1.0, no window with Sharpe < -5, recent windows stable |

**Details:**
- Use expanding window: train from start → train_end, test = next 6 months
- **Critical**: Use 5yr training windows (not 3yr) to cover bear markets
  - From quant-model-development: "3yr-WF showed 100% win rate/Sharpe 3.7. 5yr-WF showed realistic"
- Walk-Forward schedule:
  - Window 1: Train 2016-01→2020-12, Test 2021-01→2021-06
  - Window 2: Train 2016-01→2021-06, Test 2021-07→2021-12
  - ... expanding 6mo at a time
  - Until test end reaches 2026-06
- Each window: retrain model on train period, predict test period
- Aggregate: mean window Sharpe, per-window Sharpe
- Must cover bear markets: at least one test period must have SPY < 0%

---

### T3.3 — Hyperparameter Tuning

| Field | Detail |
|-------|--------|
| **Task ID** | T3.3 |
| **Goal** | Optimize XGBoost hyperparameters via Walk-Forward |
| **Input** | `v04_walk_forward_results.json` |
| **Output** | `models/falcon_v04_xgb_final.joblib` + `v04_hyperparams.json` |
| **Est. time** | 6 hours |
| **Dependencies** | T3.2 |
| **Risk** | Grid search on WF windows is expensive. Use random search or Bayesian optimization. |
| **Success criteria** | WF Sharpe improves over baseline, no overfitting (IS-OOS gap < 50%) |

**Details:**
- Parameter grid:
  - n_estimators: [300, 500, 800, 1000]
  - max_depth: [4, 5, 6, 7, 8]
  - learning_rate: [0.01, 0.03, 0.05, 0.1]
  - subsample: [0.7, 0.8, 0.9]
  - colsample_bytree: [0.6, 0.7, 0.8]
  - min_child_weight: [10, 20, 30]
  - reg_alpha: [0, 0.1, 0.5]
  - reg_lambda: [1, 5, 10]
- Search method: Random search (200 iterations) with WF as evaluation
- Save best hyperparameters + final model
- Overfitting check: IS Sharpe vs OOS Sharpe ratio < 1.5

---

## Phase 4: Backtest Validation

### T4.1 — Walk-Forward Backtest via backtest_engine.py

| Field | Detail |
|-------|--------|
| **Task ID** | T4.1 |
| **Goal** | Run full Walk-Forward backtest using the unified backtest engine |
| **Input** | `models/falcon_v04_xgb_final.joblib` + `backtest_engine.py` |
| **Output** | `v04_backtest_results.json`: full WF results with per-window detail |
| **Est. time** | 4 hours |
| **Dependencies** | T3.3 |
| **Risk** | backtest_engine uses factor-weighted scoring, not direct model predictions. Need adapter. |
| **Success criteria** | Walk-Forward Sharpe > 1.0, MaxDD < -50%, no anomalous windows |

**Details:**
- **Adapter needed**: backtest_engine expects `ranks[date]` (DataFrame of factor scores). XGBoost outputs raw predicted returns.
- Solution: Create `xgb_ranks` dict where `ranks[date]` = DataFrame with single column `'xgb_score'` containing model predictions, normalized to [0,1] via percentile rank.
- Run: `engine.walk_forward(xgb_ranks, prices, weights={'xgb_score': 1.0}, hold_days=30, top_n=10, train_years=5, test_months=6)`
- Validate per backtest-protocol:
  - Data gate: coverage >80%
  - Function verification: known-answer test
  - Baseline comparison: must beat equal-weight
  - Window review: no window Sharpe >10 or < -5

---

### T4.2 — Rank Inversion Check

| Field | Detail |
|-------|--------|
| **Task ID** | T4.2 |
| **Goal** | Verify model picks outperform — Top 5% beats Bottom 20% |
| **Input** | `v04_backtest_results.json` + predictions |
| **Output** | `v04_rank_inversion_report.json` |
| **Est. time** | 2 hours |
| **Dependencies** | T4.1 |
| **Risk** | XGBoost regression historically showed rank inversion (quant-model-development: CatBoost 1.74→0.62 on WF). |
| **Success criteria** | Top 5% return > 0, Bottom 20% return < Top 5%, no inversion in any window |

**Details:**
- For each WF window, compute:
  - Top 5% avg return
  - Top 10% avg return
  - Bottom 20% avg return
  - Bottom 5% avg return
- Check: Top 5% > Top 10% > Mid > Bottom 20% > Bottom 5%
- If rank inversion detected → check if it's in specific windows or globally
- Compare with V0.3.1 rank quality

---

### T4.3 — Stability Analysis

| Field | Detail |
|-------|--------|
| **Task ID** | T4.3 |
| **Goal** | Analyze per-window stability and regime sensitivity |
| **Input** | `v04_backtest_results.json` + VIX data + SPY returns |
| **Output** | `v04_stability_report.json` |
| **Est. time** | 2 hours |
| **Dependencies** | T4.2 |
| **Risk** | Model may overfit to bull markets, collapse in bear (CatBoost pattern) |
| **Success criteria** | Sharpe > 0 in ≥70% of windows, no consecutive 3+ negative windows, bear market Sharpe > -2 |

**Details:**
- Per-window analysis:
  - Sharpe, MaxDD, Win Rate, Annual Return
  - SPY return during window (regime detection)
  - VIX avg during window
- Regime breakdown:
  - Bull windows (SPY > 0%): Sharpe target > 1.5
  - Bear windows (SPY < 0%): Sharpe target > -1.0
  - High VIX windows (VIX > 20): Sharpe target > -1.5
- Recent window stability: last 3 windows must all have Sharpe > 0
- Compare with V0.3.1 per-window Sharpe

---

## Phase 5: Integration & Deployment

### T5.1 — Update falcon_score.py

| Field | Detail |
|-------|--------|
| **Task ID** | T5.1 |
| **Goal** | Integrate XGBoost model into production scoring pipeline |
| **Input** | `scripts/falcon/falcon_score.py` + `models/falcon_v04_xgb_final.joblib` |
| **Output** | `scripts/falcon/falcon_score_v04.py` (new file, not overwrite V0.3.3) |
| **Est. time** | 3 hours |
| **Dependencies** | T4.3 |
| **Risk** | Score range changes (XGBoost outputs raw returns, not normalized 0-1). Signal thresholds need recalibration. |
| **Success criteria** | falcon_score_v04.py produces valid scored JSON, Top-10 picks make sense, no crashes |

**Details:**
- New file: `falcon_score_v04.py` (parallel to V0.3.3, not replacing)
- Load XGBoost model
- For each ticker-date: extract features → predict 30d return → percentile rank → score
- Output format compatible with `falcon_trade_exec.py`:
  ```json
  {
    "model": "falcon_v040",
    "version": "V0.4.0",
    "date": "2026-06-26",
    "universe_size": 476,
    "picks": [{"sym": "AAPL", "score": 0.035, "close": 220.5, "rank_pct": 0.97, "signal": "🟢🟢"}]
  }
  ```
- Signal thresholds: recalibrate based on XGBoost score distribution
  - XGBoost outputs raw 30d returns (e.g., -0.1 to +0.3)
  - Percentile rank → Top 5% = 🟢🟢, Top 10% = 🟢, Top 20% = 🟡
- VIX filter: keep existing logic (VIX > 25 → skip)
- Test: run on latest date, verify output format

---

### T5.2 — Gatekeeper & Trade Exec Integration

| Field | Detail |
|-------|--------|
| **Task ID** | T5.2 |
| **Goal** | Ensure falcon_trade_exec.py reads V0.4.0 signals correctly |
| **Input** | `falcon_score_v04.py` output + `falcon_trade_exec.py` + `falcon_gatekeeper.py` |
| **Output** | Integration test results |
| **Est. time** | 2 hours |
| **Dependencies** | T5.1 |
| **Risk** | Scored JSON filename format, signal field names, file path conventions |
| **Success criteria** | Full pipeline runs: score → gatekeeper → trade exec (dry run), no format errors |

**Details:**
- Scored JSON filename: `falcon_v040_scored_YYYYMMDD.json`
- Gatekeeper: no changes needed (reads picks from JSON)
- Trade exec: no changes needed (reads picks from JSON)
- Integration test: run full pipeline in dry-run mode
- Verify: output JSON has all required fields for trade_exec

---

### T5.3 — Production Validation & Rollout Plan

| Field | Detail |
|-------|--------|
| **Task ID** | T5.3 |
| **Goal** | Final validation and staged rollout plan |
| **Input** | All previous outputs |
| **Output** | `v04_deployment_plan.md` + production config |
| **Est. time** | 2 hours |
| **Dependencies** | T5.2 |
| **Risk** | Premature deployment before sufficient OOS evidence |
| **Success criteria** | Staged rollout plan approved, rollback mechanism in place |

**Details:**
- Staged rollout:
  1. **Shadow mode** (1 week): Run V0.4.0 alongside V0.3.1, compare picks
  2. **Paper trading** (2 weeks): V0.4.0 scores → no real execution
  3. **Live** (if shadow+paper OK): Replace V0.3.1
- Rollback: Keep V0.3.1 code and weights, instant revert if V0.4.0 fails
- Config update:
  - `config/falcon.yaml`: add `model_version: v0.4.0`
  - `falcon_daily_run.sh`: switch to `falcon_score_v04.py`
- Monitor: First 2 weeks, daily comparison V0.4.0 vs V0.3.1 picks

---

## Timeline Summary

| Phase | Tasks | Est. Time | Prerequisites |
|-------|-------|-----------|---------------|
| Phase 1: Data Prep | T1.1→T1.2→T1.3→T1.4 | 8 hours | Data files exist |
| Phase 2: Feature Selection | T2.1→T2.2 | 4 hours | Phase 1 complete |
| Phase 3: Model Training | T3.1→T3.2→T3.3 | 13 hours | Phase 2 complete |
| Phase 4: Backtest | T4.1→T4.2→T4.3 | 8 hours | Phase 3 complete |
| Phase 5: Deployment | T5.1→T5.2→T5.3 | 7 hours | Phase 4 complete |
| **Total** | **13 tasks** | **~40 hours** | — |

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| XGBoost -68% DD (historical) | 🔴 High | Position sizing + VIX filter + max 10 stocks |
| Walk-Forward 30-65% overfitting | 🔴 High | 5yr train windows, IS-OOS gap <50% |
| FinBERT 2016-2023 gap | 🟡 Medium | Model learns to ignore NaN features; don't force-impute |
| backtest_engine adapter | 🟡 Medium | Create xgb_ranks dict with percentile-normalized scores |
| Score range mismatch | 🟢 Low | Percentile rank normalization in scoring pipeline |

## Success Criteria (Overall V0.4.0)

1. **Sharpe > 1.431** (beat V0.3.1)
2. **MaxDD < -50%** (acceptable for 10yr backtest)
3. **No rank inversion** in any WF window
4. **Recent 3 windows all Sharpe > 0**
5. **IS/OOS gap < 50%** (not overfit)
6. **Production scoring < 30 seconds** for 476 tickers
7. **Zero look-ahead bias** verified by PIT audit

---

## Appendix: Key References

- `quant-model-development` skill: Classification vs Ranking, Walk-Forward pitfalls, model selection
- `backtest-protocol` skill: 3-step verification, baseline comparison
- `backtest_engine.py`: BacktestEngine class, expanding window WF
- `falcon_score.py`: Current V0.3.3 scoring logic
- `falcon_v03_engine.py`: PIT rank computation (precompute_pit_ranks_fast)
- V0.3.1 baseline: fund_ratio(0.70)+analyst(0.20)+fund_metric(0.10), WF Sharpe=1.431
