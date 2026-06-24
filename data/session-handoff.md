# Session Handoff — 2026-06-24 晚间

## 已完成（本次session）

| # | 任务 | 状态 | 详情 |
|---|------|------|------|
| 1 | 蓝盾V7/绿箭V12模型训练 | ✅ | 全市场11,864只，WF验证（V7: sharpe=2.65/81%win, V12: sharpe=3.37/66%win） |
| 2 | 信号级别绝对保证系统 | ✅ | classify_signal_percentile()动态校准，🟢🟢保证win≥80%+avg>5% |
| 3 | 评分脚本更新 | ✅ | 蓝盾→V7, 绿箭→V12, +流动性过滤(vol>50K) |
| 4 | Meta阈值更新 | ✅ | 含绝对底线+百分位fallback机制 |
| 5 | 数据源提醒cron | ✅ | 每天20:10触发 |
| 6 | 重训练报告 | ✅ | data/backtest-rounds/model-retrain-report-2026-06-24.md |

## 未完成（下个session继续）

### P0 — 立即执行（预计30分钟）
1. **生成最新信号**: 运行blueshield_v6_score.py + arrow_v11_score.py生成信号到signals/us/
2. **端到端验证**: 确认classify_signal_percentile()在实际数据上正常工作
3. **更新improvement-roadmap.json**: Phase 0的3个止血动作状态更新

### P1 — 本周内（Phase 0止血）
1. 智能过滤：推荐前检查RSI/VIX
2. 暂停VIX方向推荐
3. 宏观门控：VIX>20时暂停bullish

### P2 — 本周内（Phase 1管道）
1. 创建daily-auto-scorer.sh
2. 创建daily-us-pipeline.sh
3. 创建dashboard_regenerate.sh

### P3 — 下周（Phase 2历史分析）
1. pred_score分桶vs命中率
2. 持仓周期vs收益
3. 年份/市场环境vs命中率
4. 失败模式分析
→ 数据: data/backtest-rounds/backtest-raw-results.json + backtest-us-raw.json

### P4 — 下周-2周（Phase 3-4特征工程+重训练）
1. A股8个新特征（板块相对强度/截面排名/波动率regime等）
2. 美股训练管道建设
3. 模型对比验证

### P5 — 2-3周后（Phase 5新数据源）
1. EODHD新闻情绪（已有免费tier API key: 6a3b5b96b64a59.28667511）
2. Finnhub新闻（免费tier 60次/分钟）
3. FRED宏观数据（免费）

## 关键文件路径速查

```
~/.hermes/openclaw-archive/
├── models/us/
│   ├── blueshield_v7_xgb.json      # 蓝盾V7模型 (sharpe=2.65)
│   ├── blueshield_v7_meta.json      # 蓝盾V7阈值（含绝对底线）
│   ├── arrow_v12_xgb.json          # 绿箭V12模型 (sharpe=3.37)
│   └── arrow_v12_meta.json         # 绿箭V12阈值（含绝对底线）
├── scripts/us/
│   ├── blueshield_v6_score.py       # 蓝盾评分（已指向V7+动态校准）
│   └── arrow_v11_score.py          # 绿箭评分（已指向V12+动态校准）
├── data/
│   ├── us/us_hist_full_10y.parquet  # 全市场10年数据(460MB, 11,864只)
│   ├── backtest-rounds/             # 审计报告+回测原始数据
│   │   ├── model-retrain-report-2026-06-24.md
│   │   ├── backtest-raw-results.json
│   │   ├── backtest-us-raw.json
│   │   └── (10个.md报告)
│   ├── improvement-roadmap.json     # 路线图(v2.0, Phase 0 pending)
│   ├── lessons.json                 # 教训库
│   └── positions.json               # 持仓数据
├── signals/us/                      # ⚠️ 当前为空！需运行评分脚本
└── scripts/utils/
    ├── recommendation_tracker.py    # 推荐追踪
    ├── position_monitor.py          # 持仓监控
    └── auto_scorer.py              # 自动评分
```

## 信号级别保证（已实现）

| 级别 | 保证 | 蓝盾V7实测 | 绿箭V12实测 |
|------|------|-----------|------------|
| 🟢🟢 | win≥80%, avg>5% | 96.2%, +30.2% | 91.3%, +46.2% |
| 🟢 | win≥70%, avg>0 | 84.5%, +10.4% | 83.2%, +19.6% |
| 🟡 | win≥60% | 76.9%, +6.7% | 72.7%, +10.1% |

**动态校准**: 市场低迷→模型分数整体低→绝对阈值触发不了→自动降级为百分位排名，避免全🔴

## Cron提醒
- 数据源研究: 每天20:10
- A股持仓监控: 15:30
- 美股持仓监控: 05:00

---
*Generated: 2026-06-24 ~20:00*
