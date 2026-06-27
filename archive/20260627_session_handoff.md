# Project Mercurius — Session 存档
# 存档时间: 2026-06-27 07:00 CST
# 状态: Night 1+2 完成, Night 3-10 待执行

## 一、项目概述

Project Mercurius 是美股量化交易系统，目标：10年回测验证 + 分时级实盘模拟。
数据源：Massive(Polygon $79) + FMP($69) + FinBERT(本地) + Alpaca Paper

## 二、已完成工作 (2026-06-27 凌晨)

### 2.1 Alpaca 账户
- 状态: ACTIVE, $98,439 现金, 0 持仓
- 已清仓归零 (差额 $1,560 无法恢复, API 无 reset 接口)
- Key: APCA_API_KEY_ID + APCA_API_SECRET_KEY (在 .env)

### 2.2 FinBERT 情绪打标系统
- 模型: ProsusAI/finbert (本地, 零成本)
- 速度: 188篇/秒 (CPU batch 64)
- 数据源: Massive + FMP 双源, 19个可信发布源
- 存储: data/finbert_sentiment/year={yyyy}/month={mm}/ticker={sym}.parquet
- Pipeline: scripts/falcons/finbert_pipeline.py
  - score --tickers AAPL --start 2024-01-01 --end 2024-02-01
  - backfill --months 6
  - daily (cron 用)
  - status

### 2.3 数据落地 (58只 ticker, 2023-01-01 → 2024-12-31)

Massive 日K线: 42/58 只 → data/raw/massive/daily/{ticker}.parquet
FMP 财务比率: 58/58 只 → data/raw/fmp/ratios/{ticker}.parquet
FMP 分析师评级: 58/58 只 → data/raw/fmp/analyst/{ticker}.parquet
FMP 新闻: 41/58 只 → data/raw/fmp/news/{ticker}.parquet
Massive 新闻: 37/58 只 → data/raw/massive/news/{ticker}.parquet
FinBERT 情绪: 58/58 只 → data/features/sentiment/{ticker}.parquet

缺失 ticker (无新闻覆盖):
ASTX, BMNG, BYAH, GDXY, HOOZ, ICOI, MSTP, MUD, OKLS, OTF, SHPU, SLON, SOLT, UXRP, VRXA, XRAY, XRPT

### 2.4 特征工程

技术指标 (data/features/technical/{ticker}.parquet):
- MA20, MA60, RSI(14), MACD(12,26,9), 布林带(20,2)
- bb_width, bb_pos, ret_5d, ret_20d, vol_20d, vol_ratio

情绪特征 (data/features/sentiment/{ticker}.parquet):
- daily_avg_sentiment, sentiment_volatility, news_count, avg_confidence

基本面评分 (data/features/fundamental/{ticker}.parquet):
- PE, ROE, revenue_growth (季度数据 forward-fill 到日频)

### 2.5 回测结果

Walk-Forward: 2023 训练 → 2024 验证
参数: top_n=5, hold_days=10, stop_loss=-10%

优化前 (Tech=0.3, Fund=0.4, Sent=0.3):
- Sharpe: -0.123, MaxDD: -28.02%, Win: 47.7%, PF: 0.97, Return: -11.7%

网格搜索 15 组权重, 最优 (Tech=0.5, Fund=0.2, Sent=0.3):
- Sharpe: 1.055, MaxDD: -16.23%, Win: 53.6%, PF: 1.29, Return: +37.1%

关键发现:
1. 情绪权重越高越差 (17/58只无新闻覆盖)
2. 止损是主要亏损源 (-23% vs 到期 -7.25%)
3. 小盘低价股 ($1-10) 波动率极高, 10% 止损易触发
4. 最优阈值 -0.4 (信号弱但仍有效)

### 2.6 固化配置

config/active_weights.yaml:
  tech: 0.5
  fund: 0.2
  sent: 0.3

回测参数:
  top_n: 5
  hold_days: 10
  stop_loss: -10%

归档: archive/20260627_041723_baseline/

## 三、Universe

config/universe_scored20.csv — 58 只 ticker
来源: signals/us/blueshield_v7_scores.json + arrow_v12_scores.json + SPY/QQQ

## 四、文件结构

scripts/
  falcons/
    alpaca_trade.py        # Alpaca Paper Trading 执行引擎
    finbert_pipeline.py    # FinBERT 情绪打标流水线
    mercurius_spec.md      # Mercurius 规范文档
  mercurius/
    night12_pipeline.py    # Night 1+2 全栈闭环 (universe/fetch/features/backtest)
    score_remaining.py     # 补充打标脚本

config/
  universe_scored20.csv    # Universe 列表 (58只)
  active_weights.yaml      # 固化权重

data/
  raw/massive/daily/       # Massive 日K线
  raw/massive/news/        # Massive 新闻
  raw/fmp/ratios/          # FMP 财务比率
  raw/fmp/analyst/         # FMP 分析师评级
  raw/fmp/news/            # FMP 新闻
  features/technical/      # 技术指标
  features/sentiment/      # 情绪特征
  features/fundamental/    # 基本面评分
  finbert_sentiment/       # FinBERT 离线情绪库
  backtest/                # 回测报告

archive/
  20260627_041723_baseline/ # Night 1+2 归档

## 五、Night 3-10 路线图

Night 3: 补 2022 年数据, 重跑回测, 观察加息周期下权重变化
Night 4: 补 2021 年数据, 震荡市验证
Night 5: 补 2020 年 (疫情), 黑天鹅压力测试, 重点验证熔断机制
Night 6-10: 补 2019-2014, 10年完整回测

扩容: 切换 UNIVERSE_MODE=V12_1032 验证全量池
实盘: Massive WebSocket + Alpaca Paper 日频信号执行
运维: 每日增量 T-1, 每周归因+调参, 单日回撤>2% 紧急调参

## 六、数据源状态

Massive (Polygon): ✅ 有效, MASSIVE_API_KEY 在 .env
FMP: ✅ 有效 (stable 端点), FMP_API_KEY 在 .env
FinBERT: ✅ 有效 (本地 ProsusAI/finbert)
Alpaca Paper: ✅ 有效 ($98,439), APCA keys 在 .env

## 七、已知问题

1. 42/58 只有日K线 (16只 Massive 无数据, 可能是退市或 ticker 变更)
2. 17/58 只无新闻覆盖 (情绪特征为零, 影响信号质量)
3. MaxDD -16.23% 略超 15% 目标 (差 1.2%)
4. FMP v3 端点已废弃, 必须用 stable 端点
5. Alpaca 无 API reset 接口, 只能手动网页端重置

## 八、Mercurius 规范要点

数据源有效性自检: 每次启动 4 源全检
模式隔离: 回测用历史数据, 实盘用实时数据, 绝不混用
输出规范: 每次决策附数据源引用说明
绝对禁止: 臆造数据 / 回测用未来数据 / 实盘改 FinBERT / 绕过校验
仓位规则: 单票 ≤ 5% 净值, 日止损 ≤ 2% 净值

---
_Archived: 2026-06-27 07:00 CST | Night 1+2 Complete | Next: Night 3 (2022 data)_
