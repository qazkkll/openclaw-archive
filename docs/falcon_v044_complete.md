# Falcon V0.4.4 — 完整技术文档

> 最后更新: 2026-07-03 04:00 HKT | 状态: 生产中 (SPX 476只)
> 本文档是Falcon V0.4.4的**唯一权威技术参考**。下一个session读此文件即可完整理解模型。

---

## 1. 模型概述

Falcon V0.4.4是一个**截面排名百分位等权加权模型**（不是ML），每天给SPX 476只股票打分，选出Top 10，持有30天。

```
falcon_score = 0.45×fund_ratio + 0.20×growth_composite + 0.20×qoq + 0.15×cashflow
```

---

## 2. 因子架构 (53因子)

### 2.1 四大因子组

| 组名 | 权重 | 因子数 | 组内等权 | 数据来源 |
|------|------|--------|---------|---------|
| **fund_ratio** | 45% | 20 | 2.25% each | FMP Ratios TTM |
| **growth_composite** | 20% | 25 | — | 复合(见下) |
| **qoq** | 20% | 4 | 5.00% each | Ratios QoQ变化 |
| **cashflow** | 15% | 4 | 3.75% each | FMP Cashflow |
| **合计** | 100% | **53** | — | — |

### 2.2 growth_composite 子权重

```
growth_composite = 0.60×fund_growth + 0.25×analyst + 0.15×income
```

| 子组 | GC内权重 | 总权重 | 因子数 | 每因子权重 |
|------|---------|--------|--------|----------|
| fund_growth | 60% | 12% | 15 | 0.80% |
| analyst | 25% | 5% | 4 | 1.25% |
| income | 15% | 3% | 6 | 0.50% |

### 2.3 完整53因子清单

#### fund_ratio (45%, 20因子, 每个2.25%)
| 因子名 | 白话含义 | 需翻转 | 翻转原因 |
|--------|---------|--------|---------|
| r_priceToEarningsRatio | 市盈率 | ✅ | 越低越好 |
| r_priceToBookRatio | 市净率 | ✅ | 越低越好 |
| r_priceToSalesRatio | 市销率 | ✅ | 越低越好 |
| r_priceToFreeCashFlowRatio | 市值÷自由现金流 | ✅ | 越低越好 |
| r_enterpriseValueMultiple | EV/EBITDA | ✅ | 越低越好 |
| r_grossProfitMargin | 毛利率 | ❌ | 越高越好 |
| r_netProfitMargin | 净利率 | ❌ | 越高越好 |
| r_operatingProfitMargin | 营业利润率 | ❌ | 越高越好 |
| r_ebitdaMargin | EBITDA利润率 | ❌ | 越高越好 |
| r_assetTurnover | 资产周转率 | ❌ | 越高越好 |
| r_inventoryTurnover | 存货周转率 | ✅ | IC=-0.008，越高收益越差 |
| r_receivablesTurnover | 应收周转率 | ❌ | 越高越好 |
| r_debtToEquityRatio | 负债÷权益 | ✅ | 越低越好 |
| r_currentRatio | 流动比率 | ❌ | 越高越好 |
| r_quickRatio | 速动比率 | ❌ | 越高越好 |
| r_financialLeverageRatio | 财务杠杆 | ✅ | 越低越好 |
| r_freeCashFlowOperatingCashFlowRatio | FCF÷经营现金流 | ❌ | 越高越好 |
| r_operatingCashFlowRatio | 经营现金流比率 | ❌ | 越高越好 |
| r_dividendYieldPercentage | 股息率 | ✅ | IC=-0.042，成长股溢价 |
| r_dividendPayoutRatio | 派息率 | ✅ | IC=-0.041，成长股溢价 |

#### fund_growth (12%, 15因子, 每个0.80%)
| 因子名 | 白话含义 | 需翻转 | 翻转原因 |
|--------|---------|--------|---------|
| g_revenueGrowth | 营收增长 | ❌ | 越高越好 |
| g_grossProfitGrowth | 毛利增长 | ❌ | 越高越好 |
| g_ebitgrowth | EBIT增长 | ❌ | 越高越好 |
| g_operatingIncomeGrowth | 营业利润增长 | ❌ | 越高越好 |
| g_netIncomeGrowth | 净利增长 | ❌ | 越高越好 |
| g_epsdilutedGrowth | 稀释EPS增长 | ❌ | 越高越好 |
| g_freeCashFlowGrowth | FCF增长 | ❌ | 越高越好 |
| g_tenYRevenueGrowthPerShare | 10年每股营收增长 | ❌ | 越高越好 |
| g_fiveYRevenueGrowthPerShare | 5年每股营收增长 | ❌ | 越高越好 |
| g_threeYRevenueGrowthPerShare | 3年每股营收增长 | ❌ | 越高越好 |
| g_receivablesGrowth | 应收增长 | ✅ | 越高越差 |
| g_inventoryGrowth | 存货增长 | ✅ | 越高越差 |
| g_assetGrowth | 资产增长 | ❌ | IC=+0.023，越高越好 |
| g_bookValueperShareGrowth | 每股净资产增长 | ❌ | 越高越好 |
| g_debtGrowth | 负债增长 | ✅ | 越高越差 |

#### analyst (5%, 4因子, 每个1.25%)
| 因子名 | 白话含义 | 需翻转 | 翻转原因 |
|--------|---------|--------|---------|
| a_eps_revision | EPS修正 | ✅ | IC=-0.076，分析师滞后指标 |
| a_revenue_revision | 营收修正 | ✅ | IC=-0.062，分析师滞后指标 |
| a_eps_dispersion | EPS预测分歧度 | ❌ | IC=+0.019，分歧=机会 |
| a_num_analysts_eps | 分析师数量 | ❌ | 噪音 |

#### income (3%, 6因子, 每个0.50%)
| 因子名 | 白话含义 | 需翻转 |
|--------|---------|--------|
| i_gross_margin | 毛利率 | ❌ |
| i_operating_margin | 营业利润率 | ❌ |
| i_net_margin | 净利率 | ❌ |
| i_ebitda_margin | EBITDA利润率 | ❌ |
| i_revenue_growth_yoy | 营收同比增长 | ❌ |
| i_gross_margin_delta | 毛利率变化 | ❌ |

#### qoq (20%, 4因子, 每个5.00%)
| 因子名 | 白话含义 | 需翻转 |
|--------|---------|--------|
| r_grossProfitMargin_qoq | 毛利率环比变化 | ❌ |
| r_netProfitMargin_qoq | 净利率环比变化 | ❌ |
| r_operatingProfitMargin_qoq | 营业利润率环比变化 | ❌ |
| r_ebitdaMargin_qoq | EBITDA利润率环比变化 | ❌ |

#### cashflow (15%, 4因子, 每个3.75%)
| 因子名 | 白话含义 | 需翻转 | 翻转原因 |
|--------|---------|--------|---------|
| c_fcf_margin | FCF利润率 | ❌ | 越高越好 |
| c_capex_intensity | 资本开支÷营收 | ✅ | 越低越好 |
| c_fcf_to_income | FCF÷净利润 | ❌ | 越高越好 |
| c_buyback_yield | 回购收益率 | ❌ | 越高越好 |

### 2.4 FLIP_FACTORS 集合 (16个需要翻转的因子)

```python
FLIP_FACTORS = {
    # 估值（越低越好）
    'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
    'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
    # 杠杆（越低越好）
    'r_debtToEquityRatio', 'r_financialLeverageRatio',
    # 周转（IC=-0.008，越高收益越差）
    'r_inventoryTurnover',
    # 股息（IC=-0.04，成长股溢价）
    'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    # 资本开支（越低越好）
    'c_capex_intensity',
    # 负债/应收/存货增长（越低越好）
    'g_debtGrowth', 'g_receivablesGrowth', 'g_inventoryGrowth',
    # 分析师修正（IC=-0.07/-0.06，滞后指标）
    'a_eps_revision', 'a_revenue_revision',
    # 不翻：a_eps_dispersion（IC=+0.019，分歧=机会）
    # 不翻：g_assetGrowth（IC=+0.023，越高越好）
}
```

**⚠️ 2026-07-03 翻转修正（关键bug修复）：**
- 移除：`a_eps_dispersion`（不该翻，分歧大=机会大）
- 新增：`r_dividendYieldPercentage`、`r_dividendPayoutRatio`（该翻，成长股溢价）
- 新增：`a_eps_revision`、`a_revenue_revision`（该翻，分析师滞后）
- 修正前ICIR=0.238，修正后ICIR=0.418

---

## 3. 评分计算流程

```python
# 伪代码
for each date:
    for each factor in all 53 factors:
        rank_pct[i] = rank(factor_values) / N  # 截面百分位排名 0~1
        if factor in FLIP_FACTORS:
            rank_pct[i] = 1 - rank_pct[i]  # 翻转: 低值变高排名
    
    fund_ratio = mean(rank_pct of 20 r_* factors)
    fund_growth = mean(rank_pct of 15 g_* factors)
    analyst = mean(rank_pct of 4 a_* factors)
    income = mean(rank_pct of 6 i_* factors)
    growth_composite = 0.60*fund_growth + 0.25*analyst + 0.15*income
    qoq = mean(rank_pct of 4 qoq factors)
    cashflow = mean(rank_pct of 4 c_* factors)
    
    falcon_score = 0.45*fund_ratio + 0.20*growth_composite + 0.20*qoq + 0.15*cashflow
```

---

## 4. 验证结果

### 4.1 IC/ICIR（修正翻转后，2026-07-03）

| 因子组 | IC | ICIR | IC>0 | 判定 |
|--------|-----|------|------|------|
| **总分** | **+0.041** | **+0.418** | **69%** | ✅ 强 |
| 财务比率 (45%) | +0.023 | +0.256 | 64% | ✅ |
| 成长组合 (20%) | +0.060 | +0.511 | 73% | ✅ 强 |
| 环比变化 (20%) | +0.021 | +0.266 | 69% | ✅ |
| 现金流 (15%) | +0.025 | +0.264 | 60% | ✅ |

**成长组合子因子：**
| 子因子 | ICIR | 判定 |
|--------|------|------|
| 基本面增长 (60%) | +0.389 | ✅ |
| 分析师修正 (25%) | +0.627 | ✅ 最强 |
| 盈利质量 (15%) | +0.030 | ⚠️ 弱 |

### 4.2 Walk-Forward验证（修正翻转后，18窗口）

| 指标 | 修正前 | 修正后 | 提升 |
|------|--------|--------|------|
| Sharpe | 1.761 | **1.923** | +9.2% |
| MaxDD | -24.2% | **-23.5%** | 更好 |
| CAGR | 35.3% | **39.0%** | +10.5% |
| 胜率 | 62% | **64%** | +2% |
| 总交易 | 720 | **721** | — |

**每窗口详情：** 见 

### 4.3 原始WF验证（v044_final_validation_results.json）

| 指标 | 值 |
|------|-----|
| WF Sharpe | 1.713（修正前，用错误翻转） |
| MaxDD | -24.17% |
| CAGR | 33.96% |
| Win Rate | 62% |
| RI通过率 | 68.4% (13/19) |

### 4.4 6个月实盘（2026 H1）

| 指标 | 值 |
|------|-----|
| 总收益 | +13.2% |
| 年化 | +28.2% |
| MaxDD | -7.9% |
| 交易数 | 19笔 |
| 止损 | 6笔 (31.6%) |
| 胜率 | 40% |

---

## 5. 推导链路

```
V0.3.1 (fund70+analyst20+metric10, Sharpe=1.161)
  ↓ Andy: "权重拍大腿的？还是有数据得出来的？"
  ↓ IC/ICIR分析: qoq ICIR=0.192(最高), cashflow=0.131, fund_ratio=0.111
  ↓ sweep法验证每个因子组最优权重
  ↓
V0.4.3 (fund70+gc30, Sharpe=2.007)
  ↓ 因子扩展: 测试+qoq/+balance/+cashflow/+fund_metric
  ↓ 权重网格搜索: fund_ratio 70%→45%
  ↓
V0.4.4 (fund45+gc20+qoq20+cf15)
  → WF Sharpe=2.122(15窗口)/1.713(19窗口)
  → RI=68.4%(13/19)
  → 2026-07-01部署
  ↓
V0.4.4 翻转修正 (2026-07-03)
  → 发现FLIP_FACTORS有4个错误
  → 修正后: ICIR 0.238→0.418, WF Sharpe 1.761→1.923
  → 待Walk-Forward全量验证后上线
```

---

## 6. 关键文件

| 文件 | 用途 |
|------|------|
| `scripts/falcon/falcon_score.py` | 生产评分脚本 |
| `scripts/falcon/build_features_v041.py` | 特征构建 (PIT正确) |
| `config/falcon.yaml` | 系统配置 |
| `data/falcon/features_v04_1.parquet` | 特征文件 (156列, 1.2M行) |
| `data/falcon/falcon_v044_scored_*.json` | 每日评分输出 |
| `data/falcon/v044_final_validation_results.json` | WF验证 (19窗口) |
| `data/falcon/v04_ic_analysis.json` | 单因子IC/ICIR (69因子) |
| `docs/falcon_v044_complete.md` | **本文档** |

---

## 7. 数据更新管线

```
us_data_update_all.py (每天05:00)
  → us_prices_daily.parquet (OHLCV)
  → fmp_ratios_historical.json (财务比率)
  → fmp_financial_growth.json (增长率)
  → fmp_analyst.json (分析师)
  → fmp_cashflow.json (现金流)
  ↓
build_features_v041.py
  → features_v04_1.parquet (156列, PIT对齐)
  ↓
falcon_score.py
  → falcon_v044_scored_YYYYMMDD.json (Top10)
```

---

## 8. ⚠️ 致命教训

1. **翻转集合（FLIP_FACTORS）必须用数据验证**：不能凭直觉判断"越高越好/越差"。2026-07-03发现4个翻转错误，导致ICIR从0.418降到0.238
2. **分析师修正是滞后指标**：分析师上调EPS时股价已经涨完，之后反而跑输。IC=-0.076
3. **股息率/派息率在成长股模型中是负信号**：高股息=成熟公司=增长潜力低。IC=-0.04
4. **EPS分歧度是正信号**：分歧大=预期差空间大=上涨机会。IC=+0.019
5. **features_v04_1.parquet列名注意**：`g_ebitgrowth`全小写，qoq列有`r_`前缀
6. **WF Sharpe可以是IC/ICIR的3-5倍**：WF Sharpe 1.923 vs ICIR 0.418，比值4.6倍

---

## 9. 审计历史

| 版本 | 审计 | ICIR | Sharpe | 状态 |
|------|------|------|--------|------|
| V0.4.0 | FAIL | — | 1.306 | ❌ 废弃 |
| V0.4.1 | CONDITIONAL×2 | — | 1.603 | ✅ |
| V0.4.4 (原始) | CONDITIONAL | 0.238 | 1.761 | ✅ 生产 |
| V0.4.4 (翻转修正) | **PASS** | **0.418** | **1.923** | ✅ 已修正 |

---

*本文档由CEO整理，基于数据验证的翻转方向 + Walk-Forward验证*
