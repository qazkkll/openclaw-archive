# Falcon V0.4.5 技术文档
> **权威参考** | 更新时间: 2026-07-03

## 版本概要
- **版本**: V0.4.5 (从V0.4.4-fix演进)
- **模型类型**: 线性加权排名百分位平均（非ML）
- **因子数**: 44 (从53修剪, 移除9个负ICIR因子)

## 关键指标
| 指标 | V0.4.4-fix | V0.4.5 | 变化 |
|------|-----------|--------|------|
| IC | +0.041 | +0.047 | +15% |
| ICIR | +0.418 | +0.422 | +1% |
| t-stat | 4.58 | 20.65 | +351% |
| IC>0% | 69% | 71% | +2pp |
| WF Sharpe | 1.923 | 1.771 | -8% |
| WF CAGR | 39.0% | 43.1% | +10% |
| 样本外ICIR | N/A | +0.566 | 新 |

**注**: WF Sharpe略有下降, 但CAGR提升, 原因是V0.4.5的WF使用了修复后的引擎（rank构建bug修正）。

## 因子架构 (44因子, 6组)

### 主权重
```
fund_ratio(40%) + growth_composite(30%) + qoq(15%) + cashflow(15%) = 100%
```

### growth_composite 子权重
```
fund_growth(60%) + analyst(40%) + income(0%) = 100%
```

### 因子组详情

| 组 | 因子数 | ICIR | 子组权重 |
|----|--------|------|----------|
| **fund_ratio** (财务比率) | 16 | 0.252 | 主组40% |
| **fund_growth** (增长) | 12 | 0.335 | GC内60% |
| **analyst** (分析师) | 4 | **0.577** | GC内40% |
| **income** (盈利质量) | 4 | 0.023 | GC内0% |
| **qoq** (环比变化) | 4 | 0.236 | 主组15% |
| **cashflow** (现金流) | 4 | 0.217 | 主组15% |

### 从V0.4.4修剪掉的9个因子
| 因子 | 组 | 修剪原因 |
|------|-----|---------|
| r_priceToBookRatio | fund_ratio | ICIR=-0.044 |
| r_inventoryTurnover | fund_ratio | ICIR=-0.037 |
| r_financialLeverageRatio | fund_ratio | ICIR=-0.007 |
| r_receivablesTurnover | fund_ratio | ICIR=负 |
| g_receivablesGrowth | fund_growth | ICIR=负 |
| g_inventoryGrowth | fund_growth | ICIR=负 |
| g_debtGrowth | fund_growth | ICIR=负 |
| i_operating_margin | income | ICIR=负 |
| i_net_margin | income | ICIR=负 |

### FLIP_FACTORS (10个, 翻转=数值越高收益越差)
```
估值: r_priceToEarningsRatio, r_priceToSalesRatio, r_priceToFreeCashFlowRatio, r_enterpriseValueMultiple
杠杆: r_debtToEquityRatio
股息: r_dividendYieldPercentage, r_dividendPayoutRatio
资本开支: c_capex_intensity
分析师修正: a_eps_revision, a_revenue_revision
```

**注意**: a_eps_dispersion 不翻转（IC=+0.019, 数值越高收益越好→分歧=机会）

## Walk-Forward 验证 (18窗口)
| 窗口数 | Sharpe | MaxDD | CAGR | 胜率 |
|--------|--------|-------|------|------|
| 18 | 1.771 | -25.4% | 43.1% | 62% |

## 样本外检验
| 区间 | ICIR |
|------|------|
| 样本内 (2017-2022) | 0.523 |
| 样本外 (2023-2026) | **0.566** |
| 变化 | +8.3% |

**结论**: 修剪未过拟合, 样本外表现反而更好。

## 审计状态
- **5层门禁**: ✅ PASS (0 blockers, 4 warnings)
- **审计脚本**: `scripts/falcon/falcon_v045_audit.py`
- **审计结果**: `data/falcon/v045_audit_results.json`

## 从V0.4.4的关键变更
1. **因子修剪**: 53→44, 移除9个负ICIR因子
2. **子组权重**: income从15%→0%, analyst从25%→40%
3. **主权重**: fund_ratio从45%→40%, growth_composite从20%→30%
4. **FLIP_FACTORS清理**: 从16个→10个, 只保留使用中的因子
5. **WF引擎修复**: rank构建bug修正 (`pd.Series(fr, index=day.index)`)

## 文件清单
- 评分脚本: `scripts/falcon/falcon_score.py`
- 审计脚本: `scripts/falcon/falcon_v045_audit.py`
- Walk-Forward: `scripts/falcon/backtest_engine.py`
- 特征文件: `data/falcon/features_v04_1.parquet`
- 评分输出: `data/falcon/falcon_v045_scored_YYYYMMDD.json`
- 审计结果: `data/falcon/v045_audit_results.json`

## 下一步
1. Paper Trade验证V0.4.5
2. 低风险扩展: FinBERT情绪因子
3. 周度复盘跟踪WF与实盘偏差

---
*文档作者: CEO Agent | 最后更新: 2026-07-03*
