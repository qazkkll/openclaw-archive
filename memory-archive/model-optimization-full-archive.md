# 蓝盾+绿箭 模型优化完整存档
> 2026-06-19 | 下个session从这里接续

---

## 一、优化历程

### 蓝盾V4 优化轨迹

| 轮次 | 方案 | 数据 | 夏普 | DD | 关键突破 |
|------|------|------|------|-----|---------|
| 基线 | LGB 41维 5d | S&P500 514只 | 1.13 | -56.8% | 生产模型 |
| R1 | 基础特征工程 | S&P500 | 1.154 | -82.4% | 截面特征方向 |
| R2 | 截面特征加法 | S&P500 | 1.203 | -80.4% | 截面排名核心 |
| R3 | 聚焦优化 | S&P500 | 1.203 | -80.4% | 确认 |
| R4 | 3天持有期 | S&P500 | **1.233** | **-64.9%** | 持有期突破 |
| R5 | 集成方案 | S&P500 | 1.191 | -65.5% | 不如单模型 |
| **R6** | **全市场+风控** | **全市场2413只** | **0.949** | **-22.6%** | **最终方案** |

### 绿箭V10 优化轨迹

| 版本 | 方案 | 数据 | 夏普 | 命中率 | DD |
|------|------|------|------|--------|-----|
| V9 | 50特征 5d | 全市场 | ~0.85 | 6% | -45% |
| V10 | CS12截面 | S&P500 | 2.129 | 4% | -86.8% |
| **V10最终** | **base_28+cs8+风控** | **全市场2413只** | **0.739** | **1.1%** | **-26.7%** |

---

## 二、最终方案详情

### 蓝盾V4 — 大盘趋势跟踪

**配置：**
```yaml
数据: us_ml_feats_v3_dated.parquet (2413只, 2021-12~2026-06)
特征: 36维 (base_28 + cs8截面排名)
模型: XGBoost CUDA (max_depth=6, lr=0.03, subsample=0.8)
Top-N: 20
持有期: 5天
风控: 波动率目标0.15, 回撤缩放(DD>5%→70%仓位, DD>10%→50%仓位)
```

**Walk-Forward结果：**
- 夏普: 0.949
- 年化: +28.1%
- 最大DD: -22.6%
- 胜率: 26.5%
- 平均仓位: 70%

**36维特征列表：**
```
base_28: price, volume, ma5, ma10, ma20, ma60, rsi14, vol20, p52,
         ret1, ret5, ret20, ret60, macd, macd_signal, macd_hist,
         vol_ratio, ma_bias20, vol5, trend_accel, short_ratio, short_pct,
         market_cap, pe_trailing, pe_forward, div_yield, beta, fund_price

cs8: cs_ret5, cs_ret20, cs_vol20, cs_rsi14, cs_macd_hist, cs_ma_bias20,
     cs_vol_ratio, cs_trend_accel
```

**风控逻辑：**
```python
# 波动率目标调整
port_vol = std(past_20_days_returns) * sqrt(252/5)
vol_scale = 0.15 / max(port_vol, 0.05)
vol_scale = clip(vol_scale, 0.3, 1.5)

# 回撤缩放
dd = (equity - peak) / peak
dd_scale = 0.5 if dd < -0.10 else (0.7 if dd < -0.05 else 1.0)

# 最终仓位
pos_size = min(vol_scale * dd_scale, 1.0)
```

### 绿箭V10 — 小盘彩票

**配置：**
```yaml
数据: us_ml_feats_v3_dated.parquet (2413只)
特征: 36维 (同蓝盾)
模型: XGBoost CUDA (max_depth=4, lr=0.03, subsample=0.5, scale_pos_weight=10)
Top-N: 15
持有期: 5天
风控: 波动率目标0.15, DD>8%减仓50%
梯度止盈: +20%→卖1/3, +40%→卖1/3, +60%→清仓
```

**Walk-Forward结果：**
- 夏普: 0.739
- 命中率: 1.1% (5天涨50%)
- 年化: +25.8%
- 最大DD: -26.7%

---

## 三、关键发现

### 1. 幸存者偏差严重
```
S&P500 (514只): 蓝盾夏普1.233, 绿箭夏普2.129 ← 虚高!
全市场 (2413只): 蓝盾夏普0.949, 绿箭夏普0.739 ← 真实
```
S&P500只包含成功公司，漏掉失败者，导致结果虚高40-70%。

### 2. 风控是核心突破
```
无风控: DD -50%
有风控: DD -22% (降55%)
```
波动率目标+回撤缩放是DD控制的关键。

### 3. 截面排名特征有效
```
仅基础28维: 夏普0.521
基础28+cs8: 夏普0.949 (+82%)
```
"今天谁比谁强"比"这只股票好不好"更有预测力。

### 4. 资金流特征无效
CMF/MFI/OBV反而拖累表现，已删除。

### 5. Top-N选择
- 蓝盾: Top-20更分散，DD更低
- 绿箭: Top-15更集中，命中率更高

---

## 四、已知问题（待解决）

### P0: 交易成本未扣除
```
蓝盾 Top-20, 5天持有:
- 年交易量: ~2000笔
- 每笔成本 0.05%
- 年化成本: ~100% ← 可能吃掉所有收益!
```
**需要：延长持有期到10-20天，或减少Top-N**

### P1: 风控是向后看的
用过去20天波动率调整仓位，真实崩盘时会滞后。

### P2: 参数优化可能过拟合
Top-N和vol_target是在同一批Walk-Forward折上选的。

### P3: 特征可能有前视偏差
`pe_forward`用的是分析师未来盈利预测，需要验证。

### P4: 绿箭收益不稳定
年化25.8%可能来自少数几次大赚，需要更多样本验证。

---

## 五、下个session待办

### 优先级1: 解决交易成本问题
- [ ] 延长持有期到10天，重新回测
- [ ] 延长持有期到20天，重新回测
- [ ] 加入真实交易成本（0.05%/笔）
- [ ] 计算扣除成本后的真实夏普

### 优先级2: 验证特征前视偏差
- [ ] 检查pe_forward的计算方式
- [ ] 检查fund_price的计算方式
- [ ] 如有前视偏差，删除相关特征重新训练

### 优先级3: 改进风控
- [ ] 用VIX作为市场环境信号（VIX>30不交易）
- [ ] 用前瞻波动率（GARCH）替代历史波动率
- [ ] 加入最大持仓限制

### 优先级4: 正式验证
- [ ] 3段验证（训练2021-2023，验证2024，测试2025-2026）
- [ ] 计算扣除成本后的最终夏普
- [ ] 生成生产部署方案

---

## 六、文件索引

### 分析结果
```
analysis/
├── full_market_optimization.json    # 全市场8特征集对比
├── dd_optimization_results.json     # DD优化（止损方案）
├── dd_advanced_results.json         # 高级风控结果（最终）
├── v4_final_report_20260619.md      # 蓝盾最终报告
├── v4_3d_optimization.json          # 3d持有期优化
├── v4_feature_engineering_v2.json   # 特征工程结果
└── arrow_v10_results.json           # 绿箭V10结果
```

### 脚本
```
scripts/us/
├── blueshield_v4_ultimate.py        # 蓝盾终极优化
├── arrow_v10_train.py               # 绿箭V10训练
├── dashboard_engine.py              # 看板引擎v3
└── dashboard_renderer.py            # 看板渲染v3

/tmp/
├── full_market_optimization.py      # 全市场优化脚本
├── dd_optimization.py               # DD优化脚本
└── dd_advanced.py                   # 高级风控脚本
```

### 数据
```
/mnt/d/openclaw/ml/
└── us_ml_feats_v3_dated.parquet     # 全市场数据（2413只, 36列）

data/us/
└── us_hist_sp500_10y.parquet        # S&P500数据（514只）
```

### 模型文件
```
models/us/
├── blueshield_v4_lgb_best.txt       # 蓝盾LGB基线
├── blueshield_v4_lgb_best_meta.json # 蓝盾元数据
└── us_v9_lottery.json               # 绿箭V9基线
```

---

## 七、费用追踪

```
总Token: 333M
总费用: $28.94 (¥209)
日均: ~¥15/天
缓存命中: 98%
```

---

*存档时间: 2026-06-19*
*下个session: 从"五、下个session待办"开始*
