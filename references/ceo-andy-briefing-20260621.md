# A股量化模型CEO报告 — 2026-06-21

## 给Andy的简报

### 一句话总结
**cn-alpha-v2.0 (XGBoost) 已经是最优模型，无需改动。WF Sharpe 3.89，远超1.0阈值。**

### 今天做了什么

| 实验 | 结论 | 影响 |
|------|------|------|
| 集成(XGB+Rule) | XGB单独更好 | 无需改动 |
| 基本面(PE/PB/PS) | IC<0.08, 无价值 | 无需改动 |
| 持有期扫描 | 5d>10d>20d | 可考虑缩短 |
| 止损扫描 | -1%>-3%>-8% | 可考虑收紧 |
| 市场环境分析 | 2022-2024弱 | 需关注 |
| 特征重要性 | 资金流36.7%+市场18.8% | 确认alpha来源 |

### 当前生产配置

```
模型: XGBoost回归, 25特征
持有期: 10天
Top N: 15
止损: -3%
WF Sharpe: 3.89+-2.18 (16折)
```

### 特征重要性 (Top 5)

| 特征 | 重要性 | 含义 |
|------|--------|------|
| mkt_ret20 | 10.1% | 市场20日收益 |
| breadth | 8.8% | 市场宽度 |
| ma60_bias | 8.0% | 60日均线偏离 |
| total_net_20d | 8.0% | 20日总资金流 |
| lg_net_20d | 7.0% | 20日大单资金流 |

### 风险提示

1. **2022-2024连续3年Sharpe < 1.0** — 模型不是万能的
2. **胜率偏低(33-40%)** — 依赖少数大赢
3. **-1%止损Sharpe 6.68但可能不现实** — 只检查收盘价

### 可选升级

| 方案 | Sharpe | 改动 |
|------|--------|------|
| 当前H10_SL3 | 3.89 | 无需改动 |
| H5_SL3 (更高频) | 5.27 | 缩短持有期 |
| H5_SL1 (盘中止损) | 6.68 | 需要券商API |

### 详细报告

- `references/ceo-session2-summary-20260621.md`
- `research/ceo_ensemble_results.json`
- `research/ceo_xgb_param_sweep.json`
- `research/ceo_xgb_regime_analysis.json`
- `research/ceo_xgb_feature_importance_v2.json`
