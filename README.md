# OpenClaw Archive — Hermory量化投资系统

> 从OpenClaw（小钳）复刻并重构，2026-06-17
> 运行环境：WSL2 Linux, Python 3.11, RTX 3080 Ti (XGBoost CUDA)

## 快速开始

```bash
cd ~/.hermes/openclaw-archive

# A股评分（A2模型）
python3 scripts/score/a2_score_only.py

# A股训练
python3 scripts/train/a1_layer3_xgb.py

# 刷新索引
python3 scripts/utils/update_index.py
```

## 项目结构

```
openclaw-archive/
├── INDEX.md           # 动态索引（自动更新，含数据资产+文件目录）
├── STRUCTURE.md       # 命名规则 + 目录结构设计
├── soul/              # Hermory人格（合并自小钳+原OpenClaw）
├── models/
│   ├── cn/            # A股模型（production.json标记生产模型）
│   └── us/            # 美股模型
├── data/
│   ├── cn/            # A股数据（K线、资金流、每日指标）
│   ├── us/            # 美股数据（SP500、YF、特征）
│   └── config/        # 配置文件（tushare、quality_pool等）
├── scripts/
│   ├── cn/            # A股原始脚本（待清理）
│   ├── us/            # 美股脚本
│   ├── train/         # 模型训练
│   ├── score/         # 评分/选股
│   ├── backtest/      # 回测
│   ├── data/          # 数据拉取/更新
│   ├── utils/         # 系统工具
│   └── _tmp/          # 临时/测试文件
├── output/            # 输出（评分结果、回测报告）
└── docs/              # 文档
```

## 模型清单

### A股

| 模型 | 状态 | 描述 |
|:--|:--:|:--|
| **A2 Layer3** | 🟢生产 | XGBoost回归，37特征，预测10日涨幅 |
| A2分类 | 🟡测试 | 二分类版本，预测5日涨跌概率 |
| A3 LightGBM | 🟡测试 | 技术+资金流融合 |

### 美股

| 模型 | 状态 | 描述 |
|:--|:--:|:--|
| **绿箭V19** | 🟢生产 | XGBoost，$1-10彩票策略 |
| **蓝盾V3** | 🟢生产 | 纯公式评分，110分制 |
| V8 Lottery | ⚪旧版 | 绿箭前身 |
| V9 Lottery | 🟡测试 | V8改进版 |

## 数据资产

| 数据 | 市场 | 大小 | 状态 |
|:--|:--:|:--|:--|
| a_hist_10y.parquet | A股 | 400MB | ✅ |
| moneyflow_full.json | A股 | ~1GB | 🔄拉取中 |
| us_hist_sp500_10y.parquet | 美股 | 44MB | ✅ |
| us_hist_yf_10y.parquet | 美股 | 163MB | ✅ |
| us_ml_feats_v71_v19.parquet | 美股 | 176MB | ✅ |

## 环境要求

- Python 3.11+
- xgboost (CUDA可用)
- pandas, numpy, scikit-learn
- tushare (API token在 data/config/tushare.json)

## 维护

```bash
# 刷新INDEX
python3 scripts/utils/update_index.py

# 数据更新
python3 scripts/data/pull_moneyflow.py

# 模型训练
python3 scripts/train/a1_layer3_xgb.py
```
