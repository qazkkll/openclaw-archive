# OpenClaw Archive — 项目结构 & 命名规则

> 2026-06-17 由 Hermory 设计，替代原有混乱结构

## 一、目录结构

```
openclaw-archive/
├── README.md                    # 项目总览（静态）
├── INDEX.md                     # 动态索引（自动更新，含数据资产+文件目录）
│
├── soul/                        # Hermory人格（合并自小钳+原OpenClaw）
│   ├── AGENTS.md                # 行为规则
│   ├── IDENTITY.md              # 我是谁
│   ├── MEMORY.md                # 长期记忆
│   ├── USER.md                  # Andy档案
│   └── PROTOCOLS.md             # 通信协议（合并小钳GUIDE）
│
├── models/                      # 模型文件（按市场分）
│   ├── cn/                      # A股模型
│   │   ├── production.json      # 当前生产模型声明
│   │   ├── a2_l3_xgb_10d.json  # A2 Layer3 XGBoost 10日回归
│   │   ├── a2_l3_xgb_10d_meta.json
│   │   ├── a3_lgb_full.txt     # A3 LightGBM全量
│   │   └── ...
│   ├── us/                      # 美股模型
│   │   ├── production.json      # 当前生产模型声明
│   │   ├── greenshaft_v19.json # 绿箭V19
│   │   ├── blueshield_v3.model # 蓝盾V3
│   │   └── ...
│   └── _legacy/                 # 归档模型（不再使用但保留）
│
├── data/                        # 数据资产
│   ├── cn/                      # A股数据
│   │   ├── kline_10y.parquet   # K线（原a_hist_10y）
│   │   ├── moneyflow_full.json # 资金流（全量，正在拉取）
│   │   ├── a1_daily.parquet    # 每日指标
│   │   └── a1_factors/         # 预计算因子
│   ├── us/                      # 美股数据
│   │   ├── hist_sp500_10y.parquet
│   │   ├── hist_yf_10y.parquet
│   │   ├── fundamentals.json
│   │   └── features/           # 预计算特征
│   ├── config/                  # 配置文件
│   │   ├── tushare.json
│   │   ├── quality_pool.json
│   │   └── ...
│   └── checkpoints/             # 训练检查点
│
├── scripts/                     # 脚本（按功能分，不按市场分）
│   ├── data/                    # 数据拉取/更新
│   │   ├── pull_moneyflow.py
│   │   ├── pull_kline.py
│   │   └── refresh_us_data.py
│   ├── train/                   # 模型训练
│   │   ├── cn_a2_train.py
│   │   ├── cn_a3_train.py
│   │   └── us_v19_train.py
│   ├── score/                   # 评分/选股
│   │   ├── cn_a2_score.py
│   │   ├── us_score.py
│   │   └── morning_scan.py
│   ├── backtest/                # 回测
│   │   ├── cn_a2_backtest.py
│   │   └── us_backtest.py
│   └── utils/                   # 工具脚本
│       ├── adapt_paths.py
│       └── feature_engineering.py
│
├── output/                      # 输出（评分结果、回测报告）
│   ├── cn/
│   └── us/
│
└── docs/                        # 文档
    ├── a2_architecture.md
    └── ...
```

## 二、命名规则

### 文件命名

| 类型 | 格式 | 示例 |
|:--|:--|:--|
| 模型文件 | `{市场}_{模型名}_v{版本}.{ext}` | `cn_a2_l3_xgb_10d.json` |
| 模型元数据 | 同名 + `_meta.json` | `cn_a2_l3_xgb_10d_meta.json` |
| 数据文件 | `{市场}_{数据类型}.{ext}` | `cn_kline_10y.parquet` |
| 配置文件 | `{用途}.json` | `tushare.json`, `quality_pool.json` |
| 临时文件 | `tmp_{描述}.{ext}` | `tmp_debug.json` |
| 回测结果 | `bt_{模型}_{日期}.json` | `bt_cn_a2_20260617.json` |
| 评分结果 | `score_{模型}_{日期}.json` | `score_cn_a2_20260617.json` |

### 模型版本命名

```
{市场}_{模型族}_v{大版本}.{小版本}
```

**A股模型族：**
- `a2` = A2多因子（XGBoost回归，预测10日涨幅）
- `a3` = A3融合模型（LightGBM，技术+资金流）

**美股模型族：**
- `greenshaft` = 绿箭（XGBoost，$1-10彩票策略）
- `blueshield` = 蓝盾（纯公式，110分制防御）

**版本规则：**
- v1-v9：实验版本（可废弃）
- v10+：准生产版本
- `_final`：生产版本（锁定，不可修改）
- `_cal`：带校准器的版本
- `_binary`：二分类版本

### 脚本命名

```
{市场}_{功能}_{描述}.py
```

**市场前缀：** `cn_` = A股, `us_` = 美股, `sys_` = 系统通用

**功能标签：**
- `train` = 训练
- `score` = 评分
- `bt` = 回测
- `pull` = 数据拉取
- `refresh` = 数据更新
- `scan` = 扫描/筛选

**临时文件：** 以 `tmp_` 开头，不纳入INDEX

## 三、数据资产登记

每份数据在INDEX.md中登记：

```markdown
### cn_kline_10y.parquet
- **用途**: A股K线历史数据
- **覆盖**: 4581只, 2016-2026
- **更新频率**: 每日
- **来源**: tushare daily
- **大小**: ~400MB
- **状态**: ✅ 已本地化
```

## 四、模型生产声明

`models/cn/production.json` 示例：

```json
{
  "market": "cn",
  "active_models": {
    "a2": {
      "name": "A2 Layer3 XGBoost 10日回归",
      "file": "a2_l3_xgb_10d.json",
      "meta": "a2_l3_xgb_10d_meta.json",
      "version": "v1.0",
      "trained": "2026-06-17",
      "status": "production",
      "features": 37,
      "description": "多因子评分，技术+资金流，预测10日涨幅"
    },
    "a3": {
      "name": "A3 LightGBM融合",
      "file": "a3_lgb_full.txt",
      "version": "v3.0",
      "status": "testing",
      "description": "LightGBM分类+回归融合"
    }
  }
}
```
