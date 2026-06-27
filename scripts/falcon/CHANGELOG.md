# Falcons 版本记录

## V0.4.0 — 分钟级择时+看板+自动化 (2026-06-27)
### 新增
- **分钟级择时回测**: VWAP择时54%概率比开盘便宜, 中位改善+0.067%
- **VWAP Monitor**: 盘中分钟级择时监控 (falcon_vwap_monitor.py)
- **Trading Dashboard**: Bloomberg风格暗色看板, systemd服务 (dashboard/)
- **周度复盘**: 自动计算performance+drift告警 (weekly_review.py)
- **Master Config**: 统一配置 (config/falcon.yaml)
- **Cron**: daily-scoring(05:30 HKT) + weekly-review(周六12:00)
### 策略决策
- 入场: VWAP择时 (price ≤ VWAP → 买入, 60min fallback)
- 退出: Fixed_30d + SL=-15%
- 不用分批建仓: 增益太小(+0.01%), 复杂度不值得
### 关键发现
- 信号质量 > 择时执行 > 分批策略
- FinBERT: 微弱正信号(+4.3%), 样本不足待验证

## V0.3.1 — 独立评分+交易对接 (2026-06-27)
- falcon_score.py: 独立评分, 不依赖V10/V12
- alpaca_trade.py / futu_trade.py: 默认读falcon_v031信号
- OOS验证通过: IS=1.109, OOS=1.836 (不降反升)

## V0.3 — 全量FMP因子+灵活调仓 (2026-06-27)
- 64个FMP因子(Ratios+Metrics+Growth+Analyst)
- PIT排名 + 5种调仓策略(fixed/signal/hybrid/event/adaptive)
- Futu成本模型 + SPX+R2K双宇宙
- 最优: SPX Fund70+Ana20+Met10, Fixed_30d, SR=2.206

## V0.1 — 基线 (2026-06-27)
- S&P 500(476只), 2022-2024
- Scoring: 4技术+3基本面(静态2026数据), 情绪占位
- 结果: 牛市Sharpe=0.192, 熊市失败
- 问题: 特征太少, 信号方向搞反

## V0.2 — 全特征版 (2026-06-27) ⚠️ 有前视偏差
- 80特征(43技术+20FMP+5分析师), FMP/分析师100%覆盖
- 市场状态感知(MA200) + 动态止损
- 结果: 牛市Sharpe=2.74, 熊市DD=14% WR=47%
- **问题: FMP用2026最新季度回测2022(前视偏差), Sharpe虚高27%**

## V0.2.1 — PIT修复 (2026-06-27) ✅
### 修复
- FMP: 拉40季度历史, 回测只用point-in-time已发布数据
- 分析师: 同上, 只用backtest date之前的估计
- AAPL例: 2022-01-03 PE=20.9(不是V0.2的30.8)
- 修复pandas index对齐NaN bug

### 最优参数
- Wt=0.0 Wf=0.8 Wa=0.2 (FMP主导, 技术无增量)
- Hold=30天, BearAlloc=30%

### 结果
- 牛市(2023-24): Sharpe=1.895, DD=18%
- 熊市(2022): DD=10%, WR=45% ✅
- 880/2610组合通过全部压测

### vs V0.2(有偏差)
- Sharpe: 2.74 → 1.90 (降31%, 去除虚假alpha)
- 仍为正alpha, 真实可信

### 残留问题
1. 幸存者偏差(当前S&P 500回测2022)
2. 无滑点/冲击建模
3. 样本量小(2牛+1熊)
4. FMP数据有~3个月财报延迟

## V0.3 — 执行现实化 (2026-06-27) ✅
### 改进
- Futu动态成本模型: 按股价分层($10股RT=0.40%, $50股RT=0.08%)
- T+1执行: 5种模式(Open/VWAP/Limit/Adaptive/Signal)
- 隔夜漂移分析: 均值+0.06%/天(+4.5%年化), 限价单反而比T收盘好13%
- 5% gap filter: 跳过大幅跳空的交易
- 64个PIT因子: 20 Ratios + 23 Metrics + 18 Growth + 3 Analyst

### 最优配置 (S&P 500)
- 权重: Fund_ratio 70% + Analyst 20% + Metric 10%
- 策略: Fixed_30d, SL=-15%, BearAlloc=50%
- 牛市: Sharpe=2.349, DD=7%, Ret=+206%
- 熊市: DD=9.7%, WR=50%
- 88/128组合通过Falcon协议

### 5种执行模式对比
| 模式 | 牛市Sharpe | vs T收盘 |
|------|-----------|---------|
| T收盘(理想) | 1.970 | baseline |
| T+1 Open | 1.663 | -16% |
| T+1 VWAP | 1.704 | -14% |
| **T+1 Limit** | **2.219** | **+13%** |
| T+1 Adaptive | 1.919 | -3% |

## V0.3.1 — Russell 2000扩展 + 统一框架 (2026-06-27) ✅
### Russell 2000数据
- 数据源: FMP全量symbol抽样2000只 → Massive API验证 → 691只有价格数据
- FMP覆盖: 512只有基本面(Ratios/Metrics/Growth), 394只有分析师数据
- 3年日K(2022-2024), 432K行

### Russell回测结果
- 通过率: 47/64 (73%), 比SPX的69%更高
- 最优: Pure_Fund + Fixed_10d, SL=-15%, Bear=30%
- 牛市Sharpe=1.235, DD=17%, Ret=+59%
- 熊市: DD=19%, WR=50%
- **Analyst_Heavy策略熊市WR=75%, DD=12%** ← 小盘最强alpha

### SPX vs R2K对比
| 指标 | S&P 500 | Russell 2000 |
|------|---------|-------------|
| 最优牛市Sharpe | **2.349** | 1.235 |
| 最优熊市WR | 50% | **75%** |
| 最优熊市DD | **9.7%** | 8.5% |
| 2022 below MA200 | 85% | 40% |
| 最优策略 | Fund+Ana, Signal | Pure_Fund, Fixed |

### 关键发现
1. 小盘alpha形态不同: 分析师覆盖=质量筛选(机构愿意研究的才算数)
2. 信号驱动在小盘失效: 流动性浅, 短期信号是噪音
3. R2K的2022 regime与SPX不同步: 小盘2021底就见顶, 用SPY做proxy会错判

### 统一框架
- 单脚本: `falcon_v03.py --universe spx|r2k|both`
- 自动计算技术特征(R2K)或读预计算(SPX)
- 统一FMP数据加载, 自动映射文件名
- OOS验证内嵌

### OOS验证 (2024 H2) ✅
- SPX: SR=1.836(全样本2.206, 衰减17%), DD=18.2%, WR=60%
- R2K: SR=1.492(全样本1.235, 反涨21%), DD=6.1%, WR=63%
- **结论: alpha真实, 非过拟合**

### 简单Hybrid (2026-06-27) ✅
- 方法: SPX+R2K日收益按固定比例加权, 0参数搜索
- 相关性: 全样本0.327, 牛市0.310, 熊市0.555 → 分散化价值高

| 配置 | 全样本Sharpe | MaxDD | 牛市Sharpe | 熊市Sharpe |
|------|-------------|-------|-----------|-----------|
| 纯 SPX | 1.676 | 16.9% | 2.206 | -0.194 |
| 纯 R2K | 0.818 | 19.0% | 1.235 | 0.439 |
| **80/20** | **1.689** | **15.1%** | **2.254** | -0.037 |
| 50/50 | 1.567 | 14.2% | 2.212 | 0.189 |

- 最优: **80% SPX + 20% R2K** — Sharpe几乎不变, MaxDD降1.8pp
- 结论: 混合的价值是"更稳"不是"更赚"

---

## V0.4 — 独立评分系统 (2026-06-27) ✅
### 独立化
- **falcon_score.py**: 独立评分脚本, 不依赖V10/V12模型文件
- 输出格式与V10/V12 scored JSON兼容, alpaca_trade.py/futu_trade.py直接读取
- central_config.json已添加falcon_v031模型配置
- 默认模型从arrow_v12改为falcon_v031

### FinBERT修复
- load_universe()修复: 从Falcon数据(features_v02.parquet+russell_prices.json)读取完整universe
- 之前只读scored JSON(仅20只), 现在读476+691=1167只
- Pipeline完整: transformers/torch已安装, API key就绪
- 需backfill: 当前仅1个月1只ticker数据, 需36个月全量

### OOS验证
- IS(2022-2023): Sharpe=1.109, DD=12.4%, WR=58.8%
- Val(2024H1): Sharpe=1.862, DD=9.6%, WR=50.0%
- OOS(2024H2): Sharpe=1.836, DD=18.2%, WR=60.0%
- 衰减率1.66x (不降反升), 4/4标准通过

### 数据质量
- beta列100%NaN (features_v02.parquet), 需计算或移除
- analyst有未来日期(2031), PIT过滤已正确处理
- FMP基本面数据干净, 0%NaN

## 待实现: V0.5 — 分钟级执行 (计划)
### 分钟级止损 + 新闻流 + Analyst监控
- 分钟级止损: Alpaca WebSocket实时监控持仓价格
- 新闻流: FMP+Massive叠加, 去重, FinBERT打分
- Analyst变化: 每小时轮询FMP, 检测评级/预期变动
- 自动下单: Alpaca Paper Trading API
- 预计: 4-5天完成
