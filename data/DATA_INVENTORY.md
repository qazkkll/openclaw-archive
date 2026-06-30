# OpenClaw/Falcon 数据目录清单

> 扫描时间: 2026-07-01
> 总大小: 28GB | 总文件数: 164,689

---

## 1. data/ 根目录文件

| 文件 | 大小 | 类型 | 内容概要 | 状态 |
|------|------|------|----------|------|
| `a_hist_10y.parquet` | 135.4MB | Parquet | A股10年日K, 9,451,023行×7列, 5530只股票, 2016-01-04~2026-06-30 | ✅ 已处理 |
| `us_all_ohlcv.json` | 486.4MB | JSON | 美股全量OHLCV数据 | ✅ 已处理 |
| `moneyflow_full.json` | 1568.8MB | JSON | A股资金流向, 3811只股票 | ✅ 已处理 |
| `moneyflow_pool.json` | 390.4MB | JSON | 资金流向池, 400只股票 | ✅ 已处理 |
| `training_data.json` | 448KB | JSON | ML训练数据, 11个顶层key | ✅ 已处理 |
| `predictions.json` | 10KB | JSON | 模型预测结果, 1个顶层key | ✅ 已处理 |
| `recommendations.json` | 45KB | JSON | 推荐结果, 3个顶层key | ✅ 已处理 |
| `scored_v9_lottery_2026-06-19.json` | 78KB | JSON | V9评分结果 | ✅ 已处理 |
| `ld3_scored_2026-06-17.json` | 61KB | JSON | LD3层评分 | ✅ 已处理 |
| `model-optimization-results.json` | 19.9KB | JSON | 模型优化结果, 6个key | ✅ 已处理 |
| `lessons.json` | 1.2KB | JSON | 经验教训 | ✅ 已处理 |
| `positions.json` | 782B | JSON | 当前持仓 | ✅ 已处理 |
| `us_quality_pool.json` | 68.7KB | JSON | 美股质量池, 3个key (updated, us_stocks, holdings) | ✅ 已处理 |
| `mf_pool_codes.json` | 4KB | JSON | 资金流向池代码, 400只 | ✅ 已处理 |
| `active_mission.json` | 982B | JSON | 当前任务状态, 8个key | ✅ 已处理 |
| `attribution.json` | 17.9KB | JSON | 归因分析, 2个key | ✅ 已处理 |
| `compliance_snapshots.json` | 1.7KB | JSON | 合规快照 | ✅ 已处理 |
| `improvement-roadmap.json` | 3.8KB | JSON | 改进路线图, 8个key | ✅ 已处理 |
| `param_search_results.json` | 1.4KB | JSON | 参数搜索结果 | ✅ 已处理 |
| `paper_trade_arrow_v12.json` | 2KB | JSON | Arrow V12模拟交易 | ✅ 已处理 |
| `paper_trade_blueshield_v10.json` | 5.4KB | JSON | Blueshield V10模拟交易 | ✅ 已处理 |

**符号链接:**
- `a3_moneyflow_factors.parquet` → `cn/a3_moneyflow_factors.parquet`
- `moneyflow_core.parquet` → `cn/moneyflow_core.parquet`
- `quality_pool.json` → `config/quality_pool.json`
- `sp500_symbols.json` → `config/sp500_symbols.json`
- `stock_info.json` → `config/stock_info.json`
- `us_hist_clean.parquet` → `us/us_hist_clean.parquet`

---

## 2. data/falcon/ (Falcon交易系统核心数据)

### 2.1 falcon/ 根目录 - 评分与验证数据

| 文件 | 大小 | 内容概要 | 状态 |
|------|------|----------|------|
| `features_master.parquet` | 4.3MB | 358,284行×6列, 476只SPX股票, 2022-01-03~2024-12-31 | ✅ 已处理 |
| `falcon_v03_results.csv` | 17KB | V0.3回测结果 | ✅ 已处理 |
| `falcon_v03_unified.csv` | 10KB | V0.3统一结果 | ✅ 已处理 |
| `falcon_v03_russell_results.csv` | 5KB | V0.3 Russell结果 | ✅ 已处理 |
| `falcon_best_params.json` | 2KB | 最佳参数 | ✅ 已处理 |
| `falcon_verdict.json` | 1KB | Falcon最终判定 | ✅ 已处理 |
| `falcon_opt_checkpoint.json` | 2KB | 优化检查点 | ✅ 已处理 |
| `falcon_optimization_result.json` | 44KB | 优化结果V1 | ✅ 已处理 |
| `falcon_optimization_v2_result.json` | 103KB | 优化结果V2 | ✅ 已处理 |
| `v031_full_validation.json` | 1KB | V0.3.1验证 | ✅ 已处理 |
| `v032_validation.json` | 1KB | V0.3.2验证 | ✅ 已处理 |
| `v032_final_result.json` | 3KB | V0.3.2最终 | ✅ 已处理 |
| `v032_hybrid_validation.json` | 1KB | V0.3.2混合验证 | ✅ 已处理 |
| `v033_walk_forward_comparison.json` | 8KB | V0.3.3前推比较 | ✅ 已处理 |
| `pit_walkforward_result.json` | 2KB | PIT前推 | ✅ 已处理 |
| `timing_backtest_result.json` | 226KB | 时间回测 | ✅ 已处理 |
| `backtest_2026_h1.json` | 2KB | 2026 H1回测 | ✅ 已处理 |
| `backtest_pit_corrected.json` | 11KB | PIT修正回测 | ✅ 已处理 |
| `oos_validation.json` | 2KB | 样本外验证 | ✅ 已处理 |
| `event_vs_fixed_v2.json` | 28KB | 事件vs固定V2 | ✅ 已处理 |
| `vix_filter_results.json` | 5KB | VIX过滤结果 | ✅ 已处理 |
| `vwap_backtest_result.json` | 2KB | VWAP回测 | ✅ 已处理 |
| `data_quality_report.json` | 12KB | 数据质量报告 | ✅ 已处理 |
| `fundamentals.json` | 49KB | 基本面数据 | ✅ 已处理 |
| `gatekeeper_verdict.json` | 1KB | 门禁判定 | ✅ 已处理 |
| `macro_snapshot.json` | 1KB | 宏观快照 | ✅ 已处理 |
| `monitor_state.json` | 1KB | 监控状态 | ✅ 已处理 |
| `observer_state.json` | 1KB | Observer状态 | ✅ 已处理 |
| `ref_cache.json` | 1KB | 参考缓存 | ✅ 已处理 |
| `today_plan.json` | 3KB | 今日计划 | ✅ 已处理 |

**falcon/ 中的FMP数据副本 (快照):**

| 文件 | 大小 | 说明 |
|------|------|------|
| `fmp_analyst_russell.json` | 2.4MB | 691只Russell股票分析师数据 |
| `fmp_balance_sheet.json` | 4.9MB | 476只SPX资产负债表 |
| `fmp_cashflow.json` | 3.2MB | 476只SPX现金流 |
| `fmp_financial_growth.json` | 14.7MB | 476只SPX财务增长 |
| `fmp_growth_russell.json` | 7.8MB | 691只Russell增长 |
| `fmp_income_stmt.json` | 4.7MB | 476只SPX利润表 |
| `fmp_insider.json` | 43MB | 476只SPX内部人交易 |
| `fmp_key_metrics.json` | 19.7MB | 476只SPX关键指标 |
| `fmp_metrics_russell.json` | 12.2MB | 691只Russell指标 |
| `fmp_price_target.json` | 51KB | 476只SPX目标价 |
| `fmp_ratios_historical.json` | 21.6MB | 476只SPX历史比率 |
| `fmp_ratios_russell.json` | 13.4MB | 691只Russell比率 |
| `russell_prices.json` | 48.8MB | 691只Russell价格 |
| `sp500_price_targets.json` | 82KB | S&P500目标价 |
| `fmp_full.json` | 478KB | FMP综合数据 |
| `fmp_dcf.json` | 35KB | DCF估值 |
| `analyst.json` | 80KB | 分析师数据 |
| `analyst_historical.json` | 4.9MB | 历史分析师评级 |

### 2.2 falcon/klines/ (K线数据)

| 属性 | 值 |
|------|------|
| **总大小** | 18.2MB |
| **文件数** | 494个Parquet |
| **覆盖股票** | 494只 (476 SPX + 18额外) |
| **每文件典型行数** | ~753行 |
| **列** | `ticker, date, open, high, low, close, volume, vwap` |
| **日期范围** | ~2年滚动窗口 |
| **状态** | ✅ 已处理, 每日更新 |

### 2.3 falcon/features/ (特征数据)

| 属性 | 值 |
|------|------|
| **总大小** | 45.7MB |
| **文件数** | 476个Parquet |
| **覆盖股票** | 476只SPX |
| **每文件典型行数** | ~753行 |
| **列 (20)** | `ticker, date, open, high, low, close, volume, vwap, ma20, ma60, rsi, macd, macd_signal, volatility, momentum, vol_ratio, pe, roe, revenue_growth, fund_score` |
| **状态** | ✅ 已处理, 每日更新 |

### 2.4 falcon/其他子目录

| 子目录 | 大小 | 文件数 | 说明 | 状态 |
|--------|------|--------|------|------|
| `alerts/` | 12KB | 2 | pending.json + checker_state.json | ✅ 运行中 |
| `analysis/` | 9.1MB | 2 | 每日分析报告 (6/28, 6/29) | ✅ 已处理 |
| `analysis_results/` | 24KB | 3 | FinBERT分析 + 全管道报告 | ✅ 已处理 |
| `backtest_results/` | 28KB | 4 | 10年OOS验证 + 回测 | ✅ 已处理 |
| `logs/` | 16KB | 3 | 评分/交易日志 | ✅ 已处理 |
| `reviews/` | 20KB | 4 | 周度复盘 (JSON+MD) | ✅ 已处理 |
| `trades/` | 20KB | 5 | 交易记录 + 持仓 | ✅ 已处理 |

---

## 3. data/fmp_premium/ (FMP Premium原始数据 - 闲鱼购买)

### 3.1 fmp_premium/snapshots/ (处理后的快照)

| 文件 | 大小 | 覆盖股票 | 说明 | 状态 |
|------|------|----------|------|------|
| `analyst_historical.json` | 5.1MB | 476 | 历史分析师评级 | ✅ 已处理 |
| `fmp_analyst_russell.json` | 2.4MB | 691 | Russell分析师 | ✅ 已处理 |
| `fmp_balance_sheet.json` | 5.0MB | 476 | 资产负债表 | ✅ 已处理 |
| `fmp_cashflow.json` | 3.3MB | 476 | 现金流 | ✅ 已处理 |
| `fmp_financial_growth.json` | 15.4MB | 476 | 财务增长 | ✅ 已处理 |
| `fmp_growth_russell.json` | 8.1MB | 691 | Russell增长 | ✅ 已处理 |
| `fmp_income_stmt.json` | 5.0MB | 476 | 利润表 | ✅ 已处理 |
| `fmp_insider.json` | 45.1MB | 476 | 内部人交易 | ✅ 已处理 |
| `fmp_key_metrics.json` | 20.7MB | 476 | 关键指标 | ✅ 已处理 |
| `fmp_metrics_russell.json` | 12.8MB | 691 | Russell指标 | ✅ 已处理 |
| `fmp_price_target.json` | 0.1MB | 476 | 目标价 | ✅ 已处理 |
| `fmp_ratios_historical.json` | 22.6MB | 476 | 历史财务比率 | ✅ 已处理 |
| `fmp_ratios_russell.json` | 14.0MB | 691 | Russell比率 | ✅ 已处理 |
| `russell_prices.json` | 51.1MB | 691 | Russell价格 | ✅ 已处理 |
| `sp500_price_targets.json` | 0.1MB | 473 | SPX目标价 | ✅ 已处理 |

**snapshots/ 总大小: 202MB**

### 3.2 fmp_premium/data/raw/ (原始JSON - 139,059个文件, 5.64GB)

| 数据类型 | 文件数 | 大小 | 说明 |
|----------|--------|------|------|
| `historical_price_eod_full` | 2,621 | 2,151.2MB | 历史收盘价 (1996-01-01~2026-05-15) |
| `ratios_quarter` | 2,621 | 593.6MB | 季度财务比率 |
| `balance_sheet_statement_quarter` | 2,621 | 420.8MB | 季度资产负债表 |
| `cash_flow_statement_quarter` | 2,621 | 336.5MB | 季度现金流 |
| `income_statement_quarter` | 2,621 | 276.0MB | 季度利润表 |
| `key_metrics_quarter` | 2,621 | 397.9MB | 季度关键指标 |
| `ratios_annual` | 2,621 | 158.0MB | 年度财务比率 |
| `financial_growth_annual` | 2,621 | 115.6MB | 年度财务增长 |
| `balance_sheet_statement_annual` | 2,621 | 113.2MB | 年度资产负债表 |
| `cash_flow_statement_growth` | 2,621 | 107.0MB | 现金流增长 |
| `key_metrics_annual` | 2,621 | 106.0MB | 年度关键指标 |
| `cash_flow_statement_annual` | 2,621 | 92.5MB | 年度现金流 |
| `income_statement_growth` | 2,621 | 83.9MB | 利润表增长 |
| `grades` | 2,621 | 67.2MB | 评级历史 |
| `enterprise_values_quarter` | 2,621 | 54.7MB | 季度企业价值 |
| `earnings` | 2,621 | 47.6MB | 盈利数据 |
| `grades_historical` | 2,621 | 38.4MB | 历史评级 |
| `custom_discounted_cash_flow` | 2,621 | 35.7MB | 自定义DCF |
| `financial_reports_dates` | 2,621 | 33.1MB | 财报日期 |
| `dividends` | 2,621 | 36.2MB | 股息数据 |
| `custom_levered_dcf` | 2,621 | 26.3MB | 自定义杠杆DCF |
| `enterprise_values_annual` | 2,621 | 14.5MB | 年度企业价值 |
| `historical_market_cap` | 2,621 | 13.9MB | 历史市值 |
| `historical_employee_count` | 2,621 | 18.0MB | 历史员工数 |
| `employee_count` | 2,621 | 18.0MB | 当前员工数 |
| `analyst_estimates` | 2,621 | 15.2MB | 分析师预估 |
| `profile` | 2,621 | 5.8MB | 公司简介 |
| `key_executives` | 2,621 | 6.5MB | 高管信息 |
| `revenue_geographic` | 2,621 | 3.8MB | 地理收入分布 |
| `revenue_product` | 2,621 | 5.5MB | 产品收入分布 |
| `owner_earnings` | 2,621 | 3.7MB | 所有者收益 |
| `stock_peers` | 2,621 | 3.1MB | 同行对比 |
| `ratios_ttm` | 2,621 | 7.7MB | TTM比率 |
| `key_metrics_ttm` | 2,621 | 5.1MB | TTM关键指标 |
| `stock_price_change` | 2,621 | 0.7MB | 股价变动 |
| `splits` | 2,621 | 0.7MB | 拆股数据 |
| `shares_float` | 2,621 | 0.7MB | 流通股 |
| `financial_scores` | 2,621 | 0.9MB | 财务评分 |
| `price_target_summary` | 2,621 | 0.8MB | 目标价汇总 |
| `ratings_historical` | 2,621 | 0.7MB | 历史评级 |
| `ratings_snapshot` | 2,621 | 0.7MB | 评级快照 |
| `grades_consensus` | 2,621 | 0.3MB | 共识评级 |
| `price_target_consensus` | 2,621 | 0.3MB | 目标价共识 |
| `aftermarket_quote` | 2,621 | 0.4MB | 盘后报价 |
| `aftermarket_trade` | 2,621 | 0.3MB | 盘后交易 |
| `levered_dcf` | 2,621 | 0.3MB | 杠杆DCF |
| `discounted_dcf` | 2,621 | 0.3MB | DCF |
| `quote` | 2,621 | 1.2MB | 实时报价 |
| `quote_short` | 2,621 | 0.2MB | 简要报价 |
| `market_capitalization` | 2,621 | 0.2MB | 市值 |
| `company_notes` | 2,621 | 0.4MB | 公司备注 |
| **batch/系统文件** | ~100 | ~20MB | 批量查询结果 |
| **其他** | ~10 | ~5MB | CIK列表/股票列表/交易所等 |

**每种类型覆盖2,621只股票 (universe完整覆盖)**

### 3.3 fmp_premium/universe/

| 文件 | 大小 | 内容 |
|------|------|------|
| `current_universe.json` | 1.32MB | 2,621只股票的完整universe |

### 3.4 fmp_premium/ 元数据

| 文件 | 说明 |
|------|------|
| `manifest.json` | 数据包元信息 (创建时间、SHA256、任务名) |
| `package_meta.json` | 包元数据 |
| `merge_policy.json` | 合并策略 |
| `watermarks.json` | 数据水印 (from: 1996-01-01, to: 2026-05-15) |
| `checksums/sha256.json` | 文件校验和 |
| `AI_USAGE_GUIDE.md` | AI使用指南 |
| `README_中文说明.md` | 中文说明 |

---

## 4. data/finbert_sentiment/ (FinBERT情绪分析)

### 4.1 all_scored.parquet (处理后的结果)

| 属性 | 值 |
|------|------|
| **路径** | `finbert_sentiment/all_scored.parquet` |
| **大小** | 5.0MB |
| **行数** | 24,771 |
| **列 (9)** | `ticker, published_at, title, text, source, publisher, sentiment, confidence, month` |
| **覆盖股票** | 476只SPX |
| **状态** | ✅ 已处理 |

### 4.2 partitioned parquets (分区处理结果)

| 属性 | 值 |
|------|------|
| **总大小** | ~397MB |
| **文件数** | 12,741个Parquet |
| **总行数** | 268,736 |
| **每文件列 (8)** | `ticker, published_at, title, text, source, publisher, sentiment, confidence` |
| **分区结构** | `year=YYYY/month=MM/ticker=TICKER.parquet` |

**时间覆盖:**

| 年份 | 月份 |
|------|------|
| 2022 | 01-12 (全年) |
| 2023 | 01-12 (全年) |
| 2024 | 01, 06 (不完整) |
| 2025 | 12 (仅12月) |
| 2026 | 01, 02 |

### 4.3 raw_cache/ (未处理的新闻原始数据)

| 属性 | 值 |
|------|------|
| **路径** | `finbert_sentiment/raw_cache/` |
| **文件数** | 11,292个JSON |
| **文件命名** | `{ticker}:{YYYY-MM}.json` |
| **日期范围** | 2022-01 ~ 2023-12 |
| **每文件结构** | list of `{ticker, title, text, published_at, publisher, source}` |
| **每文件记录数** | ~6条 (平均) |
| **状态** | ⚠️ **未处理** (已缓存但未送入FinBERT) |

### 4.4 回填进度

| 文件 | 说明 |
|------|------|
| `backfill_progress.json` | 回填进度V1 |
| `backfill_progress_v2.json` | 回填进度V2 |

---

## 5. data/raw/ (旧版原始数据)

### 5.1 raw/massive/daily/ (价格数据)

| 属性 | 值 |
|------|------|
| **总大小** | 1.0MB |
| **文件数** | 42个Parquet |
| **列 (9)** | `ticker, date, open, high, low, close, volume, vwap, transactions` |
| **日期范围** | 2023-01-03 ~ 2024-12-33 |
| **状态** | ✅ 已处理, 但已被`falcon/klines/`替代 |

### 5.2 raw/massive/news/ (新闻数据)

| 属性 | 值 |
|------|------|
| **总大小** | 1.4MB |
| **文件数** | 37个Parquet |
| **列 (6)** | `ticker, title, text, published_at, publisher, source` |
| **状态** | ✅ 已处理, 但已被`finbert_sentiment/`替代 |

### 5.3 raw/fmp/ (旧版FMP数据)

| 子目录 | 文件数 | 大小 | 说明 | 状态 |
|--------|--------|------|------|------|
| `ratios/` | 36 | 1.7MB | 财务比率 (12行×65列/文件) | ⚠️ 旧版, 已被fmp_premium替代 |
| `analyst/` | 35 | 0.5MB | 分析师预估 (12行×22列/文件) | ⚠️ 旧版, 已被fmp_premium替代 |
| `news/` | 41 | 1.2MB | 新闻 (457行×6列/文件) | ⚠️ 旧版, 已被finbert_sentiment替代 |

---

## 6. data/us/ (美股数据)

### 6.1 Parquet价格文件

| 文件 | 大小 | 行数 | 列 | 股票数 | 日期范围 | 用途 | 状态 |
|------|------|------|-----|--------|----------|------|------|
| `us_hist_full_10y.parquet` | 493.7MB | 29,813,975 | 7 (sym,date,OHLCV) | 11,864 | 2016-06-24~2026-06-24 | 全量美股10年价格 | ✅ 已处理 |
| `us_hist_sp500_10y.parquet` | 43.7MB | 1,249,749 | 7 | S&P500 | 2016-06-13~2026-06-17 | S&P500 10年 | ✅ 已处理 |
| `us_hist_megacap_10y.parquet` | 4.1MB | 102,336 | 7 | 超大盘 | 2016-06-13~2026-06-10 | 超大盘10年 | ✅ 已处理 |
| `us_hist_yf_5y.parquet` | 41.5MB | 3,729,860 | 7 (ticker,OHLCV) | 2,436 | 2021-06-10~2026-06-09 | yfinance 5年 | ✅ 已处理 |
| `us_hist_yf_10y.parquet.archived` | 179.1MB | 5,916,472 | 9 (含dividends,splits) | - | 2016~2026 | yfinance 10年 (已归档) | ⚠️ 归档 |
| `us_hist_clean.parquet` | 150.6MB | 2,436 | 5 (ticker,c,h,l,v数组) | 2,436 | - | 清洗后的聚合数据 | ⚠️ 特殊格式 |
| `vix_10y.parquet` | 0.1MB | 2,511 | 6 (date,OHLCV) | 1 | 2016-07-05~2026-06-29 | VIX指数10年 | ✅ 已处理 |
| `spx_daily.parquet` | 0.1MB | 2,510 | 7 (date,ticker,OHLCV) | 1 | 2016-07-05~2026-06-29 | SPX指数日线 | ✅ 已处理 |
| `sector_etf_daily.parquet` | 0.1MB | 2,761 | 7 | 11 | 2025-06-30~2026-06-29 | 板块ETF日线 | ✅ 已处理 |

### 6.2 特征与ML文件

| 文件 | 大小 | 行数 | 列数 | 股票数 | 日期范围 | 状态 |
|------|------|------|------|--------|----------|------|
| `features/us_ml_feats_v75.parquet` | 1525.9MB | 5,486,276 | 55 | - | 2016-10-18~2026-06-10 | ✅ 已处理 |
| `features/us_ml_feats_v75_filtered.parquet` | 144.9MB | 505,306 | 62 | - | 2016-10-19~2026-06-30 | ✅ 已处理 |
| `features/us_ml_feats_v71_v19.parquet` | 175.9MB | 1,097,848 | 41 | - | 2021-12-01~2026-06-10 | ✅ 已处理 |
| `ml_training_data.parquet` | 977.7MB | 9,332,786 | 23 | - | - | ✅ 已处理 |
| `fundamentals_latest.parquet` | 0.3MB | 11,864 | 5 (sym,pe_trailing,pe_forward,div_yield,beta) | 11,864 | 最新 | ✅ 已处理 |

### 6.3 JSON/Meta文件

| 文件 | 大小 | 内容 | 状态 |
|------|------|------|------|
| `us_all_tickers.json` | - | 12,673只全量股票代码 | ✅ 已处理 |
| `us_all_tickers_filtered.json` | - | 11,864只过滤后代码 | ✅ 已处理 |
| `us_fundamentals.json` | 0.7MB | 2,435只基本面数据 | ✅ 已处理 |
| `us_feature_cols.json` | - | 18个特征列名 | ✅ 已处理 |
| `ml_feature_cols.json` | - | 19个ML特征列名 | ✅ 已处理 |
| `us_download_errors.json` | - | 下载错误记录 | ✅ 已处理 |
| `fund_fetch_progress.json` | - | 基本面获取进度 | ✅ 已处理 |
| `market_snapshot.json` | - | 市场快照 | ✅ 已处理 |
| `arrow_v12_scored_20260624.json` | - | Arrow V12评分 | ✅ 已处理 |
| `blueshield_v10_scored_20260624.json` | - | Blueshield V10评分 | ✅ 已处理 |
| `blueshield_v10_quantile_scored_20260624.json` | - | Blueshield V10分位数评分 | ✅ 已处理 |

---

## 7. data/cn/ (A股数据)

| 文件 | 大小 | 行数 | 列 | 股票数 | 日期范围 | 用途 | 状态 |
|------|------|------|-----|--------|----------|------|------|
| `a3_moneyflow_factors.parquet` | 1014.2MB | 10,373,601 | 27 | 5,440 | 20160104~20260603 | 资金流向因子 | ✅ 已处理 |
| `a_hist_10y.parquet` | 134.6MB | 9,422,070 | 7 | 5,528 | 20160104~20260630 | A股10年日K | ✅ 已处理 |
| `moneyflow_full.parquet` | 865.1MB | 10,484,793 | 20 | 5,643 | 20160104~20260617 | 资金流向全量 | ✅ 已处理 |
| `moneyflow_core.parquet` | 349.9MB | 7,265,106 | 20 | 5,402 | 20160104~20260630 | 资金流向核心 | ✅ 已处理 |
| `daily_basic.parquet` | 388.4MB | 10,768,098 | 9 | 5,851 | 20160104~20260630 | 每日基础指标 | ✅ 已处理 |
| `features_v2.parquet` | 1707.3MB | 6,785,247 | 42 | - | 2016-02-01~2026-05-07 | 特征V2 | ✅ 已处理 |
| `north_money.parquet` | 0.0MB | 700 | 7 | - | 20230703~20260629 | 北向资金 | ✅ 已处理 |
| `top_list.parquet` | 1.9MB | 18,766 | 19 | 3,636 | 20250630~20260629 | 龙虎榜 | ✅ 已处理 |
| `trade_cal.parquet` | 0.0MB | 2,010 | 4 | - | - | 交易日历 | ✅ 已处理 |
| `stock_info.json` | 977KB | - | - | 5,532 | - | 股票信息 | ✅ 已处理 |
| `stock_names.json` | 270KB | - | - | - | - | 股票名称 | ✅ 已处理 |
| `daily_basic_state.json` | 30KB | - | - | - | - | 每日基础状态 | ✅ 已处理 |
| `daily_basic_pull.log` | - | - | - | - | - | 数据拉取日志 | ✅ 已处理 |

**⚠️ 注意: `a1_daily.parquet` 文件已损坏 (Parquet magic bytes not found)**

---

## 8. data/features/ (旧版特征数据)

| 子目录 | 文件数 | 总大小 | 每文件 | 说明 | 状态 |
|--------|--------|--------|--------|------|------|
| `fundamental/` | 58 | 120KB | 12行×2列 | 基本面特征 | ⚠️ 大部分为空或极少数据 |
| `sentiment/` | 58 | 299KB | 多数为0行 | 情绪特征 | ⚠️ 大部分为空 |
| `technical/` | 41 | 3.6MB | ~502行×24列 | 技术特征 | ✅ 有数据 |

---

## 9. data/moneyflow_checkpoints/ (资金流向检查点)

| 文件 | 大小 | 说明 |
|------|------|------|
| `batch_0.json` | 502MB | 批次0数据 |
| `batch_1.json` | 1054MB | 批次1数据 |
| `batch_2.json` | 1555MB | 批次2数据 |
| `batch_3.json` | 2116MB | 批次3数据 |
| `state.json` | 37KB | 状态 (completed/failed/last_index) |

**总大小: 11GB** - 这是A股资金流向处理过程中的中间检查点

---

## 10. data/其他目录

### 10.1 data/backtest/ (36KB)
- `attribution_report.json` - 归因报告
- `arrow_v12_oos.json` - Arrow V12样本外
- `blueshield_v10_oos.json` - Blueshield V10样本外
- `arrow_v12_comprehensive.json` - Arrow V12综合

### 10.2 data/backtest-rounds/ (48MB, 16文件)
- 多轮回测报告和脚本

### 10.3 data/layer3_checkpoints/ (17MB, 10文件)
- Layer3处理检查点 (大部分为空)

### 10.4 data/models/ (112KB)
- `us_v9_lottery.json` - V9彩票模型

### 10.5 data/rl/ (820KB, 4文件)
- `model_scores_us.parquet` - 52,280行×4列 (sym,date,bs_score,ga_score)
- `eval_results_us.json` - 26KB
- `eval_results.json` - 4KB
- `ROADMAP.md` - RL路线图

### 10.6 data/config/ (696KB, 12文件)
- `stock_info.json` - 585KB, 股票信息
- `strategy.json` - 4KB, 策略配置
- `news_monitor.json` - 2KB, 新闻监控配置
- 其他配置文件

### 10.7 data/experiments/ (152KB, 27文件)
- 实验结果 JSON

### 10.8 data/handoffs/ (8KB)
- `20260626_145942.json` - 交接记录

### 10.9 data/news/ (空)
- 目录存在但无文件

### 10.10 data/predictions/ (8KB)
- `latest.json` - 最新预测

### 10.11 data/research/ (8KB)
- `fmp-data-integration-plan.md` - FMP数据集成计划

### 10.12 data/us_market/ (8KB)
- `us_market_latest.csv` - 空文件

### 10.13 data/checkpoints/ (空)
- 目录存在但无文件

---

## 数据流总结

```
FMP Premium (闲鱼)                    yfinance
    ↓                                      ↓
fmp_premium/data/raw/ (139K JSON)    us/us_hist_*_10y.parquet
    ↓                                      ↓
fmp_premium/snapshots/ (15 JSON)     falcon/klines/ (494 ticker parquets)
    ↓                                      ↓
falcon/*.json (评分用)               falcon/features/ (476 ticker parquets)
                                             ↓
                                     falcon_score.py → 评分
                                             ↓
                                     falcon_trade_exec.py → 交易

FinBERT Sentiment (raw_cache/)       A股数据 (cn/)
    ↓                                      ↓
finbert_sentiment/partitioned/       cn/moneyflow_*.parquet
    ↓                                      ↓
finbert_sentiment/all_scored.parquet cn/a3_moneyflow_factors.parquet
    ↓                                      ↓
falcon_score.py (sentiment因子)      moneyflow_checkpoints/ (11GB)
```

## 关键发现

1. **FMP Premium是最大数据源**: 5.64GB原始JSON, 139,059个文件, 覆盖2,621只股票, 从1996年到2026年5月
2. **FinBERT raw_cache有11,292个未处理的新闻缓存**: 仅覆盖2022-01~2023-12, 每文件~6条记录
3. **FinBERT分区覆盖不完整**: 2024年仅1月和6月, 2025年仅12月
4. **moneyflow_checkpoints占用11GB**: 这是A股资金流向处理的中间状态, 可能可清理
5. **cn/a1_daily.parquet已损坏**: 无法读取
6. **data/features/ (旧版) 大部分为空**: fundamental和sentiment目录的parquet文件多数为0行
7. **data/news/为空**: 已被finbert_sentiment替代
8. **data/us_market/us_market_latest.csv为空**: 未使用
9. **data/checkpoints/为空**: 未使用
10. **raw/ (旧版) 已被fmp_premium/和finbert_sentiment/替代**: 但仍保留
