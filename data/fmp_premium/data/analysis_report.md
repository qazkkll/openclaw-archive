# FMP Premium Data Quality Analysis — 量化特征价值评估

分析日期: 2026-06-29
样例股票: AAPL, NVDA, TSLA
数据目录: ~/.hermes/openclaw-archive/data/fmp_premium/data/raw/

---

## 总览评分表

| # | 数据类型 | 文件数 | 价值评分 | 时间跨度 | 频率 | 推荐特征数 |
|---|---------|--------|---------|---------|------|-----------|
| 1 | grades_historical | 2621 | ⭐⭐⭐⭐⭐ 5/5 | 2018-2026 | 月度 | 6 |
| 2 | analyst_estimates | 2621 | ⭐⭐⭐ 3/5 | 未来预测 | 年度 | 4 |
| 3 | financial_scores | 2621 | ⭐⭐ 2/5 | 单点快照 | 单次 | 2 |
| 4 | price (EOD full) | 2621 | ⭐⭐⭐⭐⭐ 5/5 | 1996-2026 | 日频 | 10+ |
| 5 | enterprise_values | 5242 | ⭐⭐⭐⭐ 4/5 | 1996-2026 | 季度 | 4 |
| 6 | historical_market_cap | 2621 | ⭐⭐⭐ 3/5 | 近2月 | 周度 | 2 |
| 7 | income_statement_quarter | 2621 | ⭐⭐⭐⭐⭐ 5/5 | 1996-2026 | 季度 | 12 |
| 8 | balance_sheet_quarter | 2621 | ⭐⭐⭐⭐ 4/5 | 1996-2026 | 季度 | 10 |
| 9 | earnings | 2621 | ⭐⭐⭐⭐ 4/5 | 多年 | 季度 | 3 |

**额外发现**: 还有 `income_statement_growth`、`balance_sheet_growth`、`enterprise_values_annual` 等衍生数据文件（共7863+个文件）

---

## 详细分析

### 1. grades_historical — 分析师历史评级 ⭐⭐⭐⭐⭐

**数据质量**: 极高
- 字段: symbol, date, analystRatingsStrongBuy/Buy/Hold/Sell/StrongSell
- AAPL: 89条 (2018-12 ~ 2026-05), NVDA: 87条 (2019-01 ~ 2026-05), TSLA: 85条 (2019-01 ~ 2026-05)
- 频率: 月度，几乎无缺失（AAPL仅缺1个月）
- 完整性: 5个评级级别完整覆盖，零null值

**可提取特征**:
1. `consensus_score` — 加权共识评分 (SB*5+B*4+H*3+S*2+SS*1)/总数, 范围1-5
2. `bull_ratio` — 多头比例 (SB+B)/总数
3. `bear_ratio` — 空头比例 (S+SS)/总数
4. `analyst_momentum` — 评级动量: 本月consensus - 上月consensus
5. `analyst_dispersion` — 评级分歧度: 标准差(各评级占比)
6. `num_analysts` — 覆盖分析师数量（覆盖度代理）

**对量化模型的价值**: 🔥🔥🔥 月度更新、时间序列完整、直接编码分析师情绪变化。可以与价格信号形成正交特征。

---

### 2. analyst_estimates — 分析师盈利预测 ⭐⭐⭐

**数据质量**: 中等
- 字段: symbol, date, revenueLow/High/Avg, ebitdaLow/High/Avg, ebitLow/High/Avg, netIncomeLow/High/Avg, sgaExpenseLow/High/Avg, epsAvg/High/Low, numAnalystsRevenue, numAnalystsEps
- 每个ticker只有10条年度预测记录
- 无季度预测（quarterly文件不存在）
- 是未来预测值（如2030年），非历史

**可提取特征**:
1. `eps_est` — 近期EPS预测值
2. `eps_range` — EPS预测范围 (High-Low)/Avg (不确定性)
3. `revenue_est` — 近期收入预测值
4. `num_est_analysts` — 参与预测的分析师数量

**局限**: 只有10个年度预测期，无法形成时间序列。但可用于cross-sectional排序（同一时间点不同股票的预测比较）。

---

### 3. financial_scores — 财务健康评分 ⭐⭐

**数据质量**: 低（单点快照）
- 字段: symbol, reportedCurrency, altmanZScore, piotroskiScore, workingCapital, totalAssets, retainedEarnings, ebit, marketCap, totalLiabilities, revenue
- 每只股票仅1条记录（最新时点）
- 无时间序列，无法构建趋势

**可提取特征**:
1. `altman_z_score` — 破产风险指标（直接使用）
2. `piotroski_score` — 财务健康评分0-9（直接使用）

**局限**: 单点快照，无法做时间序列分析。但如果用于cross-sectional排序（所有股票同一时点比较），有一定价值。

---

### 4. price — 30年日频OHLCV ⭐⭐⭐⭐⭐

**数据质量**: 极高
- 字段: symbol, date, open, high, low, close, volume, change, changePercent, vwap
- AAPL: 7643天 (1996-01 ~ 2026-05), NVDA: 6871天 (1999-01 ~ 2026-05), TSLA: 3995天 (2010-06 ~ 2026-05)
- 零null值，无缺失交易日
- 包含VWAP（成交量加权均价）

**可提取特征** (直接或计算):
1. 所有标准技术指标: MA, EMA, RSI, MACD, Bollinger, ATR等
2. `vwap_ratio` — 收盘价/VWAP比值
3. `daily_return` — 日收益率
4. `volatility_20d/60d` — 波动率
5. `volume_profile` — 成交量特征
6. `price_position` — 价格在历史分位数的位置
7. `momentum_1m/3m/6m/12m` — 各期动量
8. 与现有Falcon模型的27个技术特征重叠，但FMP的price可能更干净

**对量化模型的价值**: 这是量化模型的基础数据。与yfinance/parquet数据互补（FMP有vwap，yfinance可能有不同来源）。

---

### 5. enterprise_values — 企业价值估值 ⭐⭐⭐⭐

**数据质量**: 高
- 字段: symbol, date, stockPrice, numberOfShares, marketCapitalization, minusCashAndCashEquivalents, addTotalDebt, enterpriseValue
- 季度数据: AAPL 120条 (1996-2026), NVDA 108条, TSLA 73条
- 同时有季度和年度版本
- 零null值

**可提取特征**:
1. `ev_ebitda` — 企业价值/EBITDA（需结合income数据）
2. `ev_revenue` — 企业价值/收入（需结合income数据）
3. `ev_per_share` — 每股企业价值
4. `cash_to_ev_ratio` — 现金/企业价值比
5. `debt_to_ev_ratio` — 负债/企业价值比
6. `shares_outstanding_trend` — 股份数变化趋势

**对量化模型的价值**: 估值因子的核心数据源。配合income/balance sheet可计算完整估值比率。

---

### 6. historical_market_capitalization — 市值历史 ⭐⭐⭐

**数据质量**: 中等
- 字段: symbol, date, marketCap
- 每只股票仅63条记录（近2个月，周频）
- 数据范围太短，无法构建长期趋势

**可提取特征**:
1. `market_cap` — 市值（绝对值，用于cross-sectional排序）
2. `market_cap_pctile` — 市值百分位（同截面排名）

**局限**: 仅63天历史，远不如enterprise_values和price数据。与price的close * shares_outstanding计算的市值相比，价值有限。

---

### 7. income_statement_quarter — 季度利润表 ⭐⭐⭐⭐⭐

**数据质量**: 极高
- 字段数: 38个（极其丰富）
- AAPL: 120条 (1996-2026), NVDA: 108条, TSLA: 73条
- 含 filingDate（报告发布日，可用于避免前视偏差）
- 含 reportedCurrency, fiscalYear, period

**核心字段**: revenue, costOfRevenue, grossProfit, R&D, SG&A, operatingExpenses, EBITDA, EBIT, operatingIncome, netIncome, EPS, EPS_diluted, weightedAverageSharesOutstanding

**可提取特征**:
1. `gross_margin` — 毛利率
2. `operating_margin` — 营业利润率
3. `net_margin` — 净利率
4. `ebitda_margin` — EBITDA利润率
5. `rd_intensity` — R&D/Revenue
6. `sga_intensity` — SG&A/Revenue
7. `revenue_growth_yoy` — 收入同比增长（可用growth文件）
8. `eps_growth_yoy` — EPS同比增长
9. `operating_leverage` — 营业利润变化/收入变化
10. `earnings_quality` — 应计利润比 (netIncome - cashflow) / totalAssets
11. `revenue_surprise` — 收入超预期程度
12. `cost_structure` — 成本结构变化 (costOfRevenue/revenue)

**对量化模型的价值**: 🔥🔥🔥 基本面因子的核心数据源。30年季度数据+filingDate是金矿。

---

### 8. balance_sheet_quarter — 季度资产负债表 ⭐⭐⭐⭐

**数据质量**: 高
- 字段数: 65个（极其丰富）
- AAPL: 120条 (1996-2026), NVDA: 108条, TSLA: 72条
- 零null值

**核心字段**: cashAndCashEquivalents, shortTermInvestments, totalCurrentAssets, totalAssets, inventory, netReceivables, accountPayables, shortTermDebt, longTermDebt, totalDebt, netDebt, totalEquity, retainedEarnings

**可提取特征**:
1. `current_ratio` — 流动比率
2. `quick_ratio` — 速动比率 (currentAssets-inventory)/currentLiabilities
3. `debt_to_equity` — 负债权益比
4. `net_debt_to_ebitda` — 净负债/EBITDA
5. `cash_ratio` — 现金比率
6. `working_capital` — 营运资本
7. `inventory_turnover` — 存货周转率（结合income COGS）
8. `receivable_turnover` — 应收周转率（结合income revenue）
9. `asset_efficiency` — 资产效率 revenue/totalAssets
10. `capital_structure` — 资本结构变化

**对量化模型的价值**: 财务杠杆和流动性因子的数据源。与income statement配合可计算Piotroski F-score等复合指标。

---

### 9. earnings — 财报数据（含实际vs预测） ⭐⭐⭐⭐

**数据质量**: 高
- 字段: symbol, date, epsActual, epsEstimated, revenueActual, revenueEstimated, lastUpdated
- AAPL: 164条, NVDA: 108条, TSLA: 74条
- 有actual和estimated对照，可计算surprise
- 含未来预测记录（actual=null）

**可提取特征**:
1. `eps_surprise` — EPS超预期 (actual-estimated)/|estimated|
2. `revenue_surprise` — 收入超预期
3. `earnings_momentum` — 连续surprise的方向性（连续超预期=正动量）

**局限**: 
- AAPL: 163/164有actual, 110有estimated
- NVDA: 107/108有actual, 全部有estimated  
- TSLA: 71/74有actual, 64有estimated
- 缺口：有些历史记录没有estimated值

---

## 额外发现: growth 文件

还存在大量预计算的YoY增长数据文件：
- `income_statement_growth` (7863+ 文件) — 直接提供growthRevenue, growthNetIncome, growthEPS等
- `balance_sheet_growth` (7863+ 文件) — 直接提供growthTotalAssets, growthInventory等
- 这些是**极其宝贵的特征**，免去了自己计算YoY的步骤

---

## 对Falcon模型的集成建议

### 最高优先级（直接集成）
1. **grades_historical** → analyst_sentiment_score (月频，2018-2026)
2. **income_statement_quarter** + **growth** → 盈利质量因子 (季频，1996-2026)
3. **earnings** → EPS surprise因子 (季频)

### 高优先级
4. **enterprise_values** → 估值因子 (季频，1996-2026)
5. **balance_sheet_quarter** + **growth** → 财务健康因子 (季频)
6. **price** → VWAP信号 (日频，与现有技术特征互补)

### 低优先级
7. **analyst_estimates** → 仅cross-sectional可用，时间序列太短
8. **financial_scores** → 单点快照，需定期更新才有用
9. **market_cap** → 仅63天历史，信息量太少

### 潜在新特征列表（约20个）
```
# analyst sentiment (月频)
analyst_consensus_score
analyst_bull_ratio
analyst_bear_ratio
analyst_momentum (月度变化)
analyst_dispersion (分歧度)
analyst_coverage (覆盖分析师数)

# fundamental (季频)
gross_margin
operating_margin
net_margin
ebitda_margin
revenue_growth_yoy (直接用growth文件)
eps_growth_yoy
rd_intensity
current_ratio
debt_to_equity
net_debt_to_ebitda

# earnings (季频)
eps_surprise
revenue_surprise
earnings_momentum

# valuation (季频)
ev_per_share
cash_to_ev_ratio

# price (日频)
vwap_ratio
```

---

## 数据质量总结

| 维度 | 评级 |
|------|------|
| 字段丰富度 | ⭐⭐⭐⭐⭐ (65+字段/表) |
| 时间跨度 | ⭐⭐⭐⭐⭐ (30年+) |
| 数据频率 | ⭐⭐⭐⭐ (季度为主，价格日频) |
| 零值/缺失 | ⭐⭐⭐⭐⭐ (几乎无null) |
| 覆盖股票数 | ⭐⭐⭐⭐ (2621只) |
| 前视偏差控制 | ⭐⭐⭐⭐ (有filingDate) |
| 与现有模型互补性 | ⭐⭐⭐⭐⭐ (基本面+分析师，与技术面正交) |

**结论**: FMP Premium数据质量极高，尤其是grades_historical和income_statement_quarter对量化模型有重大增量价值。建议优先集成分析师情绪因子和盈利质量因子。
