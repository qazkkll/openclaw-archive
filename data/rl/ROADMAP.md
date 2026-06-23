# RL交易系统 — 完全体路线图

> 最后更新: 2026-06-23
> 跟踪文件: openclaw-archive/data/rl/ROADMAP.md

## 架构图（目标状态）

```
┌─────────────────────────────────────────────────────┐
│                    信号层 (已有)                       │
│  Blue Shield XGBoost → "买AAPL" (score=0.12)        │
│  Green Arrow XGBoost → "买SOFI" (score=0.09)        │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                    RL执行层 (在建)                     │
│  观察: model_score + RSI + MACD + VIX + position     │
│  动作: buy_25/50/75/100, sell_25/50/75/100, hold    │
│  输出: "现在买25%仓位" / "止损50%" / "加仓到75%"     │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                    执行层 (已有)                       │
│  Futu OpenD → 实际下单                               │
└─────────────────────────────────────────────────────┘
```

## Checklist

### Phase 1: 数据与环境 ✅
- [x] 美股OHLCV数据（10年，2474只）`us_hist_yf_10y.parquet`
- [x] gymnasium TradingEnv（9种动作，技术指标，手续费+滑点）
- [x] 7种基准策略评估（B&H/Random/RSI/Model/Momentum/Anti/Turtle）
- [x] 多股票回测（10只mega-cap）
- [x] Walk-Forward回测框架
- [x] 日期兼容（A股int/美股datetime64）

### Phase 2: 模型评分接入 ✅ (2026-06-23)
- [x] 预计算脚本 `precompute_scores.py`（蓝盾V6 + 绿箭V11全历史评分）
- [x] 评分数据 `data/rl/model_scores_us.parquet`（52280行，10股票）
- [x] TradingEnv支持bs_score/ga_score作为obs[12]/obs[13]
- [x] XGBoost Score策略实现（真实评分+技术确认）
- [x] 对比验证：XGBoost Score vs 模拟Model Score

### Phase 3: 多股票组合环境 ❌
- [ ] TradingPortfolioEnv：同时持有多只股票
- [ ] 组合层观察：持仓相关性、行业暴露、总仓位比例
- [ ] 组合层动作：单只仓位调整 + 整体仓位调整
- [ ] 交易成本：含美股PDT规则（5天内≤3次日内交易）

### Phase 4: RL Agent训练 ❌
- [ ] 安装stable-baselines3
- [ ] PPO训练（连续动作空间版本）
- [ ] DQN训练（离散动作空间）
- [ ] SAC训练（自动调整探索率）
- [ ] Walk-Forward训练（训练窗口2年，测试6个月，滚动）
- [ ] 超参搜索（学习率、网络结构、batch_size）
- [ ] Reward shaping：Sharpe-adjusted vs Return vs 复合reward

### Phase 5: 评估与验证 ❌
- [ ] Out-of-sample测试（2024-2026数据hold out）
- [ ] 多市场regime测试（牛市/熊市/震荡分别评估）
- [ ] 风险指标：Sortino、Calmar、Omega、最大回撤持续时间
- [ ] 蒙特卡洛模拟：1000次随机起始点评估策略稳定性
- [ ] 与Buy & Hold的统计显著性检验（DM test）
- [ ] Paper Trading模拟（用最近60天数据假装在线交易）

### Phase 6: 与现有系统集成 ❌
- [ ] RL层包装器：`rl_position_sizer(model_signal, market_state) → action`
- [ ] 接入每日评分流程（blueshield/arrow_score.py → RL → 最终推荐）
- [ ] Dashboard展示RL建议仓位
- [ ] Cron job：每日收盘后用RL评估次日仓位建议

### Phase 7: 高级特性 ❌
- [ ] 多资产RL（股票+ETF+现金动态配置）
- [ ] Regime detection（HMM或简单规则识别牛/熊/震荡）
- [ ] Meta-learning：根据最近表现自动切换策略
- [ ] 对抗测试：2008/2020/2022暴跌场景压力测试

## 当前进度
- Phase 1: ✅ 完成
- Phase 2: ✅ 完成（XGBoost Score策略在NVDA/GOOGL/AMZN上Sharpe最优）
- Phase 3: ❌ 未开始（多股票组合环境）
- Phase 4: ❌ 未开始（RL Agent训练）

## 关键文件
- 环境: `scripts/rl/trading_env.py`
- 评估: `scripts/rl/run_eval.py`
- 数据: `data/us/us_hist_yf_10y.parquet`
- 结果: `data/rl/eval_results_us.json`
- 本文档: `data/rl/ROADMAP.md`
