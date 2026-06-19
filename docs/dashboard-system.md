# Hermes Trading Intelligence — 看板系统文档

## 概述

三Tab智能交易看板，Ethereal Glass设计风格，双语界面。

- **Tab 1 Trading 操盘** — 日常主界面：宏观指标 + 持仓管理 + 模型推荐
- **Tab 2 Tracking 追踪** — 推荐历史追踪：按信号等级分组的累计收益
- **Tab 3 Models 模型** — 模型数据展示：特征重要性 + 验证指标 + 超参数

## 快速启动

```bash
# 生成看板（含追踪数据更新）
cd ~/.hermes/openclaw-archive
python3 scripts/us/build_tracking.py
python3 scripts/us/generate_pro_dashboard_v7.py

# 启动服务器+隧道
./scripts/us/start_dashboard.sh
```

## 文件结构

```
~/.hermes/openclaw-archive/
├── dashboard.html                          # 生成的看板HTML（入口）
├── config.json                             # 系统配置（阈值、模型参数）
│
├── scripts/us/
│   ├── generate_pro_dashboard_v7.py        # 看板生成器（主脚本）
│   ├── build_tracking.py                   # 追踪数据构建器
│   ├── generate_tracking_data.py           # 合成追踪数据生成
│   ├── start_dashboard.sh                  # 启动脚本（HTTP+隧道）
│   ├── blueshield_v6_score.py              # 蓝盾V6评分脚本
│   ├── arrow_v11_score.py                  # 绿箭V11评分脚本
│   ├── futu_positions.py                   # Futu持仓查询
│   └── track_recommendations.py            # 推荐追踪器
│
├── models/us/
│   ├── blueshield_v6_xgb.json              # 蓝盾V6 XGBoost模型
│   ├── blueshield_v6_meta.json             # 蓝盾V6元数据（特征、验证指标）
│   ├── arrow_v11_xgb.json                  # 绿箭V11 XGBoost模型
│   └── arrow_v11_meta.json                 # 绿箭V11元数据
│
├── output/
│   ├── futu_positions.json                 # Futu持仓数据（实时）
│   ├── v6_latest.json                      # 蓝盾最新评分结果
│   ├── v11_latest.json                     # 绿箭最新评分结果
│   ├── held_scores.json                    # 持仓评分（排名百分位）
│   ├── tracking_history.json               # 推荐追踪历史
│   ├── snapshots/                          # 每日评分快照
│   │   └── YYYY-MM-DD.json
│   └── recommendations.json                # 合成回测追踪数据
│
└── archive/dashboard/                      # 旧版看板归档
```

## 数据流

```
[Futu OpenD] → futu_positions.py → output/futu_positions.json
[yfinance]   → blueshield_v6_score.py → output/v6_latest.json
[yfinance]   → arrow_v11_score.py → output/v11_latest.json
[两者]       → build_tracking.py → output/tracking_history.json
[所有数据]   → generate_pro_dashboard_v7.py → dashboard.html
```

## 自动刷新

- **盘中**（9:30-16:00 ET / 21:30-04:00 北京时间）：每5分钟自动刷新
- **盘外**：倒计时暂停，不刷新
- 底部有蓝色进度条显示倒计时

## 信号系统

| 信号 | 颜色 | 含义 | 阈值 |
|------|------|------|------|
| STRONG 强 | 🟢🟢 | 精品买入 | Top 5% 百分位 |
| BUY 买 | 🟢 | 主力信号 | Top 10% 百分位 |
| WATCH 看 | 🟡 | 观察池 | Top 20% 百分位 |
| Skip 跳 | 🔴 | 不推荐 | Below 20% |

**三层过滤**：L1 VIX>30关闭 → L2 分数>中位数 → L3 百分位分级

## 模型规格

### Shield V6 蓝盾（大盘股 >$10）
- XGBoost排名模型，500棵树，44维特征
- 持有20天，Top 15，止损-15%
- OOS夏普1.44，年化+30%，最大回撤-11.1%

### Arrow V11 绿箭（小盘股 $1-$10）
- XGBoost排名模型，500棵树，42维特征
- 持有5天，Top 5，止损-10%
- OOS夏普2.18，5天净收+5.56%

## Cron任务

| 任务 | 时间 | Job ID |
|------|------|--------|
| 蓝盾V6评分 | 04:30 | fb1723fce4f6 |
| 绿箭V11评分 | 04:30 | 419b588b962b |
| 看板更新 | 05:00 | 999c99516eab |

## ngrok访问

固定URL：`https://struggle-amends-satiable.ngrok-free.dev/dashboard.html`

首次打开需点"Visit Site"，之后不再提示。

## 设计规范

- **风格**：Ethereal Glass（OLED黑底 #050508 + 玻璃态卡片）
- **字体**：Inter正文 + JetBrains Mono数字（Google Fonts CDN）
- **颜色系统**：Indigo #6366f1（蓝盾）/ Green #10b981（绿箭）/ Amber #f59e0b（警告）/ Red #ef4444（止损）
- **图表**：纯SVG（无外部依赖），带Y轴网格线+刻度标签
- **响应式**：900px以下单列，480px以下紧凑布局

## 更新日志

- **v7 (2026-06-20)**: Triple-Tab Intelligence Platform, 双语, 自动刷新, 信号区分, 真实追踪
- **v6 (2026-06-20)**: Triple-Tab初版
- **v5 (2026-06-20)**: Ethereal Glass风格
- **v4 (2026-06-19)**: 深色底+数据密度
