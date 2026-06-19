# OpenClaw Archive — Hermes量化投资系统

> 从OpenClaw（小钳）复刻并重构，2026-06-17
> 运行环境：WSL2 Linux, Python 3.12, RTX 3080 Ti (XGBoost CUDA)

## 快速开始

```bash
cd ~/.hermes/openclaw-archive

# 美股蓝盾V6评分
python3 scripts/us/blueshield_v6_score.py

# 美股绿箭V11评分
python3 scripts/us/arrow_v11_score.py

# A股评分（A2模型）
python3 scripts/score/a2_score_only.py

# 刷新索引
python3 scripts/utils/update_index.py
```

## 项目结构

```
openclaw-archive/
├── config.json                  ← 全局配置（V6定版）
├── README.md                    ← 本文件
├── INDEX.md                     # 动态索引
├── soul/                        # Hermory人格
├── models/
│   ├── production.json          ← 生产模型状态
│   ├── cn/                      # A股模型
│   └── us/
│       ├── blueshield_v6_xgb.json    ← V6模型（生产）
│       ├── blueshield_v6_meta.json    ← V6元数据
│       ├── blueshield_v6_README.md    ← V6技术文档
│       └── archive/                   ← V3/V4/V5归档
├── data/
│   ├── cn/                      # A股数据
│   └── us/
│       ├── us_hist_yf_10y.parquet     # 全市场OHLCV（2436只, 10年）
│       └── features/                  # 特征数据
├── scripts/
│   ├── us/
│   │   ├── blueshield_v6_score.py     ← V6每日评分
│   │   └── us_v9_daily_score.py       # V9每日评分
│   ├── cn/
│   ├── train/
│   ├── score/
│   ├── backtest/
│   ├── data/
│   └── utils/
├── analysis/                    # 回测分析结果
├── docs/                        # 技术文档
└── output/                      # 评分输出
```

## 模型清单

### 美股

| 模型 | 状态 | 描述 |
|:--|:--:|:--|
| **蓝盾V6** | 🟢生产 | XGBoost排名, 44维特征(技术27+宏观13+基本面4), 20天Top-15, 夏普1.44, 年化+30%, DD-11.1% |
|| **绿箭V11** | 🟢生产 | XGBoost排名, 41维特征(技术28+宏观13), 5天Top-5, $1-$10彩票股 ||
| 蓝盾V5 | ⚪归档 | 120天持有, S&P500, 夏普1.36（有幸存者偏差） |
| 蓝盾V4 | ⚪归档 | 5天持有, 全市场, 夏普0.949 |
| 蓝盾V3 | ⚪归档 | 纯公式110分制, 夏普0.77 |

### A股

| 模型 | 状态 | 描述 |
|:--|:--:|:--|
| **A2 Layer3** | 🟢生产 | XGBoost回归, 37特征, 预测10日涨幅 |
| A2分类 | 🟡测试 | 二分类版本 |
| A3 LightGBM | 🟡测试 | 技术+资金流融合 |

## 蓝盾V6核心数据

```
数据: 全市场2359只(>$10), 2016-2026, 10年
特征: 44维 = 技术27 + 宏观13 + 基本面4
模型: XGBoost排名, 500棵树, max_depth=6, lr=0.03
持有: 20天轮换, Top-15等权
验证: Walk-Forward 5折 + 样本外2024-2026

样本外结果 (2024-2026):
  年化: +30.1%
  夏普: 1.440
  Sortino: 2.825
  最大DD: -11.1%
  胜率: 60%
  交易: 30轮

信号分级:
  三层过滤信号系统:
  L1: VIX > 30 → 关闭所有信号
  L2: 分数必须 > 当日中位数
  L3: Top 5% → 🟢🟢精品买入
       Top 10% → 🟢强信号
       Top 20% → 🟡观察
```

## 环境要求

- Python 3.12+
- xgboost 3.2.0+ (CUDA可用)
- pandas, numpy, yfinance

## 维护

```bash
# 刷新INDEX
python3 scripts/utils/update_index.py

# 蓝盾V6评分
python3 scripts/us/blueshield_v6_score.py

# 绿箭V11评分
python3 scripts/us/arrow_v11_score.py

# 模型重训练（每6个月）
python3 /tmp/v6_train_production.py
```

## 版本历史

| 版本 | 日期 | 关键变化 |
|------|------|---------|
| V6 | 2026-06-19 | 排名模型+宏观特征+44维, 夏普1.44, 年化+30% |
| V5 | 2026-06-19 | 120天持有, S&P500, 夏普1.36（有幸存者偏差） |
| V4 | 2026-06-18 | ML排序, 全市场, 夏普0.949 |
| V3 | 2026-06前 | 纯公式, 夏普0.77 |
