# Falcon V0.4.4 — 项目README

> 最后更新: 2026-07-03 04:00 HKT | 状态: 生产中 (SPX 476只)

## 模型概述

Falcon V0.4.4是一个**截面排名百分位等权加权模型**（不是ML），每天给SPX 476只股票打分，选出Top 10，持有30天。

```
falcon_score = 0.45×fund_ratio + 0.20×growth_composite + 0.20×qoq + 0.15×cashflow
```

## 核心指标（翻转修正后）

| 指标 | 值 | 阈值 | 判定 |
|------|-----|------|------|
| IC | +0.041 | >0.02 | ✅ |
| ICIR | +0.418 | >0.3 | ✅ |
| t-stat | +4.58 | >1.96 | ✅ |
| IC>0 | 69% | >60% | ✅ |
| WF Sharpe | 1.923 | >1.5 | ✅ |
| MaxDD | -23.5% | <30% | ✅ |
| CAGR | 39.0% | — | ✅ |

## 文档结构

```
docs/
├── falcon_v044_complete.md      # 完整技术文档（权威参考）
├── falcon_v044_derivation.md    # 推导链路
├── falcon_v044_bugfix_log.md    # Bug修正日志
└── README.md                    # 本文件

data/falcon/
├── v044_final_validation_results.json   # WF验证 (19窗口)
├── v044_factor_expansion_results.json   # 因子扩展实验
├── v044_fixed_ri_results.json           # RI验证 (15窗口)
├── v04_ic_analysis.json                 # 单因子IC/ICIR
└── falcon_v044_scored_*.json            # 每日评分输出

scripts/falcon/
├── falcon_score.py              # 生产评分脚本
├── build_features_v041.py       # 特征构建
├── backtest_engine.py           # 回测引擎
└── dashboard_refresh.py         # Dashboard数据刷新
```

## 快速上手

```bash
# 评分最新交易日
python3 scripts/falcon/falcon_score.py

# 评分指定日期
python3 scripts/falcon/falcon_score.py --date 2024-12-31

# 取Top-10
python3 scripts/falcon/falcon_score.py --top-n 10

# 更新数据
bash scripts/us_data_update_all.sh
```

## 因子架构

53个因子，4大组：

| 组名 | 权重 | 因子数 | ICIR |
|------|------|--------|------|
| 财务比率（fund_ratio） | 45% | 20 | +0.256 |
| 成长组合（growth_composite） | 20% | 25 | +0.511 |
| 环比变化（qoq） | 20% | 4 | +0.266 |
| 现金流（cashflow） | 15% | 4 | +0.264 |

详见 `docs/falcon_v044_complete.md`

## 关键教训

1. **翻转集合必须用数据验证**：不能凭直觉判断方向，必须用Spearman相关系数
2. **分析师修正是滞后指标**：上调EPS时股价已经涨完
3. **股息率在成长股模型中是负信号**
4. **WF Sharpe可以是ICIR的3-5倍**

## 更新历史

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-07-01 | V0.4.4 | 初始部署 |
| 2026-07-03 | V0.4.4-fix | 翻转集合修正，ICIR 0.238→0.418 |

---

*本README由CEO维护*
