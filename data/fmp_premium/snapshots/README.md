# FMP Premium 快照数据（已整合）

所有从闲鱼购买的FMP Premium数据，已统一到此目录。

## 目录结构

```
data/fmp_premium/
├── data/raw/           ← 原始JSON（2621只股票×50+种数据）
├── snapshots/          ← 已提取的合并JSON（SPX/R2K子集）
│   ├── analyst_historical.json      ← 分析师EPS/收入预估历史（季度，2018-2028）
│   ├── fmp_price_target.json        ← 价格目标快照（476只）
│   ├── sp500_price_targets.json     ← 价格目标+upside%（473只）
│   ├── fmp_ratios_historical.json   ← 财务比率历史（年度）
│   ├── fmp_key_metrics.json         ← 关键指标历史
│   ├── fmp_financial_growth.json    ← 财务增长率历史
│   ├── fmp_balance_sheet.json       ← 资产负债表历史
│   ├── fmp_cashflow.json            ← 现金流历史
│   ├── fmp_income_stmt.json         ← 利润表历史
│   ├── fmp_insider.json             ← 内部人交易（44MB）
│   ├── fmp_analyst_russell.json     ← R2K分析师数据
│   ├── fmp_growth_russell.json      ← R2K增长率数据
│   ├── fmp_metrics_russell.json     ← R2K指标数据
│   ├── fmp_ratios_russell.json      ← R2K比率数据
│   └── russell_prices.json          ← R2K价格数据（49MB）
├── universe/           ← 股票池定义
└── AI_USAGE_GUIDE.md   ← 数据使用指南
```

## 数据来源
- 闲鱼购买的FMP Premium数据包
- 覆盖: 2621只股票（NASDAQ+NYSE+AMEX，市值>10亿美元）
- 历史: 最早到1996年
- 更新频率: 原始数据是快照（购买时的最新值）

## 使用规则
- **原始JSON**（data/raw/）：用于因子IC验证、历史回测
- **合并JSON**（snapshots/）：用于V0.3.1评分、Gatekeeper、日常分析
- 不要在data/falcon/里再存快照数据，统一从这里读取
