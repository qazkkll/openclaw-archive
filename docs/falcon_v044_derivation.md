# Falcon V0.4.4 — 模型推导文档 & 审计报告

> 最后更新: 2026-07-02 | 状态: 生产中 (SPX 476只)

## 一句话

Falcon V0.4.4 是一个**截面排名百分位等权加权模型**（不是ML），用53个基本面因子给SPX 476只股票打分，选出Top 10。

---

## 1. 模型架构

### 评分公式

```
falcon_score = 0.45 × fund_ratio + 0.20 × growth_composite + 0.20 × qoq + 0.15 × cashflow
```

### 因子组定义（与 falcon_score.py 完全一致）

| 组名 | 权重 | 因子数 | 组内等权 | 计算方式 |
|------|------|--------|---------|---------|
| **fund_ratio** | 45% | 20 | 2.25% | 20个FMP财务比率的截面rank百分位均值 |
| **growth_composite** | 20% | 25 | — | 0.60×fund_growth + 0.25×analyst + 0.15×income |
| ↳ fund_growth | (12%) | 15 | 0.80% | 15个增长指标截面rank百分位均值 |
| ↳ analyst | (5%) | 4 | 1.25% | 4个分析师指标截面rank百分位均值 |
| ↳ income | (3%) | 6 | 0.50% | 6个收入指标截面rank百分位均值 |
| **qoq** | 20% | 4 | 5.00% | 4个季度环比利润率变化 |
| **cashflow** | 15% | 4 | 3.75% | 4个现金流指标 |
| **合计** | 100% | **53** | — | — |

### 具体因子清单

#### ① fund_ratio (45%, 20因子)
PE/PB/PS/PFCF/EV倍数/毛利率/净利率/营业利润率/EBITDA利润率/资产周转/存货周转/应收周转/D&E/流动比率/速动比率/财务杠杆/FCF&OCF/营业现金流/股息率/派息率

#### ②a fund_growth (12%, 15因子)
营收增长/毛利增长/EBIT增长/营业利润增长/净利增长/EPS增长/FCF增长/10Y营收/5Y营收/3Y营收/应收增长/存货增长/资产增长/BV增长/债务增长

#### ②b analyst (5%, 4因子)
EPS修正/营收修正/EPS分歧度/分析师数量

#### ②c income (3%, 6因子)
毛利率/营业利润率/净利率/EBITDA利润率/营收增长YoY/毛利率变化

#### ③ qoq (20%, 4因子)
毛利率环比/净利率环比/营业利润率环比/EBITDA利润率环比

#### ④ cashflow (15%, 4因子)
FCF利润率/资本开支强度/FCF&净利润/回购收益率

### 需要翻转的因子（越高越差）
PE/PB/PS/PFCF/EV倍数/D&E/财务杠杆/存货周转/资本开支强度/债务增长/应收增长/存货增长/EPS分歧度

---

## 2. 推导过程

### 2.1 起点: v043_baseline
- 权重: fund_ratio(70%) + gc_baseline(30%)
- Sharpe: 2.007
- 来源: `v044_factor_expansion_results.json`

### 2.2 因子扩展实验
测试了在baseline基础上添加不同因子组:

| 配置 | Sharpe | MaxDD | CAGR | RI通过 |
|------|--------|-------|------|--------|
| v043_baseline (fund70+gc30) | 2.007 | -15.8% | 40.4% | ✅ |
| +gc_qoq_20 | 1.611 | -30.4% | 30.4% | ✅ |
| +gc_balance_25 | 1.833 | -18.5% | 38.9% | ✅ |
| +gc_cashflow_15 | 1.775 | -18.6% | 35.6% | ✅ |
| +gc_fund_metric_15 | 1.935 | -16.3% | 38.8% | ✅ |

结论: 单独添加任何因子组都无法超越baseline → 需要重新分配权重

### 2.3 权重搜索
通过网格搜索确定最优4因子组合:
- fund_ratio: 45% (从70%降低)
- growth_composite: 20% (从30%降低)
- qoq: 20% (新增)
- cashflow: 15% (新增)

### 2.4 Walk-Forward验证 (v044_final_validation_results.json)
- 19个6个月窗口, 每窗口40笔交易
- 总交易: 760笔
- **Sharpe: 1.713**
- **CAGR: 33.96%**
- **MaxDD: -24.17%**
- **Win Rate: 62%**

### 2.5 Rank Inversion验证
- 方法: Top5% vs Bottom20% 前瞻30天收益
- **通过率: 68.4% (13/19窗口)**
- 平均Top5%收益: 11.34%
- 平均Bottom20%收益: 9.38%
- 平均Spread: 1.96%

---

## 3. 独立审计结果 (2026-07-02)

### L1: 数据完整性 ✅
- 特征文件: 1,236,008行, 156列, 476只
- 日期: 2016-01-04 → 2026-07-01 (2638天)
- PIT覆盖率: ratios 100%, metrics 99.3%, growth 100%, analyst 99.8%

### L2: 特征一致性 ✅
- 评分用53因子, 全部存在于特征文件
- 23个可用但未用因子 (key_metrics等)
- 权重: config = score.py = yaml ✅

### L3: 模型有效性 🔴 存疑
独立计算IC (每50天采样, 51个窗口):
- **IC均值: 0.0067** (接近零)
- **ICIR: 0.0864** (远低于0.3阈值)
- **RI通过率: 54.9%** (28/51, 低于60%阈值)
- **排名反转: 45.1%** (23/51窗口)

⚠️ 独立验证RI=54.9% vs config声称68.4%, 差异原因:
- config验证: 19个固定6个月窗口, 每窗口独立评分
- 独立验证: 每50天采样, 51个窗口, 覆盖更广
- 4个窗口因数据质量被跳过 (2018-2019年gc_baseline覆盖率<80%)

按年IC:
```
2017: +0.055 ✅ | 2018: -0.016 | 2019: +0.017
2020: -0.049 ❌ | 2021: +0.049 | 2022: -0.020
2023: +0.041 | 2024: +0.017 | 2025: -0.019 ❌
```

### L4: 信号生产 ✅
- 评分脚本正常, JSON格式完整
- Top10全🟢🟢 (阈值可能太松)

### L5: 跨层一致性 ⚠️
- Falcon是独立评分系统, 不在production.json中
- production.json指向arrow_v12 + blueshield_v10 (LightGBM)

---

## 4. 6个月实盘回测 (2026 H1)

| 指标 | 值 |
|------|-----|
| 总收益 | +13.2% |
| 年化 | +28.2% |
| MaxDD | -7.9% |
| 交易数 | 19笔 |
| 止损 | 6笔 (31.6%) |
| 胜率 | 40% |

---

## 5. 关键文件

| 文件 | 作用 |
|------|------|
| `scripts/falcon/falcon_score.py` | 生产评分脚本 |
| `scripts/falcon/build_features_v041.py` | 特征构建 (PIT正确) |
| `config/falcon.yaml` | 系统配置 |
| `data/falcon/features_v04_1.parquet` | 特征文件 (1.2M行) |
| `data/falcon/falcon_v044_scored_*.json` | 每日评分输出 |
| `data/falcon/v044_final_validation_results.json` | WF验证结果 |
| `data/falcon/v044_factor_expansion_results.json` | 因子扩展实验 |
| `data/falcon/v044_fixed_ri_results.json` | 修复RI验证 |

---

## 6. 已知问题 & 待改进

1. **IC接近零**: 模型预测能力弱, Top5%和Bot20%收益差距仅1.96%
2. **2025-2026年IC为负**: 模型在最近1.5年选出来的股票反而跑输
3. **53因子等权**: 20个fund_ratio因子等权(2.25% each), 未做因子筛选
4. **线性加权**: 无非线性交互, 无法捕捉因子组合效应
5. **未用key_metrics**: 19个高质量指标(ROE/ROA/FCF Yield等)完全未使用
6. **与production系统分离**: production用LightGBM(arrow_v12/blueshield_v10), Falcon是独立的简单模型

---

## 7. 与Production模型对比

| 指标 | Falcon V0.4.4 | BlueShield V10 | Arrow V12 |
|------|---------------|----------------|-----------|
| 算法 | 线性加权 | LightGBM | LambdaMART |
| ICIR | 0.086 | **0.734** | **0.386** |
| 因子数 | 53 | 43 | 17 |
| Universe | SPX | >$10 | $1-$10 |
| Hold | 30天 | 10天 | 20天 |
| 审计 | 本报告 | PASSED 5/5 | PASSED 5/5 |

---

*本文档由独立审计自动生成, 基于v044_final_validation_results.json + 独立IC/RI计算*
