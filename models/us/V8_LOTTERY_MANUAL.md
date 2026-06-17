# 🟢 绿箭 V8-Lottery 使用说明书

## 文件总览

| 文件 | 路径 | 说明 |
|---|---|---|
| 模型文件 | `D:/openclaw/data/models/us_v7_5_l50.json` | XGBoost 43特征 |
| 模型报告 | `D:/openclaw/data/models/us_v7_5_l50_report.json` | 特征重要性/参数/AUC |
| 心路历程 | `D:/openclaw/data/models/V8_LOTTERY_README.md` | **[见本文]** |
| 使用说明书 | `D:/openclaw/data/models/V8_LOTTERY_MANUAL.md` | **[本文]** |
| 旧V7.5模型 | `D:/openclaw/data/models/us_v7_5.json` | 存档备用 |
| V8执行计划 | `workspace/docs/EXECUTION_PLAN_V75.md` | 策略说明书 |

---

## 一、每日评分流程

### 方法A：一键跑（推荐）

```bash
# 工作区根目录
cd C:/Users/admin/.openclaw/workspace

# 步骤1：绿箭V8评分（耗时约3-5分钟）
python scripts/us_v7_5_daily_score.py

# 步骤2：蓝盾3.0评分
python scripts/us_ld3_daily_score.py

# 步骤3：融合推荐
python scripts/us_daily_recommend.py
```

### 方法B：单独跑绿箭

```bash
python scripts/us_v7_5_daily_score.py
```

输出文件：
- `D:/openclaw/data/us_scored_v8.json` — V8评分结果（score 0-100）
- `D:/openclaw/data/us_scored_ld3.json` — 蓝盾3.0评分

### 方法C：盘中实时评分

```python
# 用minishare拉实时价，然后拼到本地历史尾部评分
# 参见 TOOLS.md 盘中评分流程
```

---

## 二、评分解读

评分输出是 `score`（0-100），**不是概率**，是排序分数。

| 评分区间 | 含义 | 操作建议 |
|---|---|---|
| ≥90 | 极强信号 | Top5首选 |
| 80-89 | 强信号 | Top5候选 |
| 70-79 | 中等 | 候选但不优先 |
| 50-69 | 弱信号 | 一般不考虑 |
| <50 | 噪音 | 忽略 |

**关键规则：**
- **只看Top5！** Top5命中率17.1%，Top10降到8.6%
- 不需要额外过滤（评分阈值、RSI过滤等都试过，只会降命中率）
- 不同日期的score绝对值不可比（评分归一化会漂移）

---

## 三、实时执行策略

### 买入规则
```python
# 每天运行评分后
top5 = scored_df.nlargest(5, 'score')
# Top5等权买入
per_position = total_cash * 0.02  # 2%/只
for sym in top5['symbol']:
    buy(sym, per_position)  # 次日开盘价执行
```

### 持有期管理
```python
# 持有5个交易日
hold_days = 5

# 每日检查止损
stop_loss = -0.15  # -15%
for position in positions:
    if position.current_return < stop_loss:
        sell(position.symbol)  # 立即市价卖出
```

### 卖出触发条件
1. ✅ 持有期满5日 → 卖出
2. ✅ 触发-15%止损 → 卖出
3. ✅ 评分下降严重 → 可提前卖出（酌情）
4. ❌ 不因为"涨太多"提前卖（彩票股要么不涨，要涨就是30%+）

---

## 四、数据依赖

### 核心数据源
- `D:/openclaw/ml/us_ml_feats_v75.parquet` — 特征数据集（yfinance下载）
  - 仅OHLCV特征，无基本面/宏观因子
  - 覆盖约600只$1-10股票
  - 需要**每日更新**（盘后yfinance拉最新K线）

### 特征列表（43个）
```
基础技术(36): ma5/ma5_ratio/ma20_ratio/ma60_ratio
               vol5/vol20/vol_ratio/ema12/ema26
               macd/macd_signal/macd_hist
               rsi14/k/d/j/bb_upper/bb_lower
               bb_width/bb_position
               vol_ratio_ma5/vol_ratio_ma20
               adx/plus_di/minus_di
               price_position/price_position_60/cmf/vix_close

交叉特征(7): close_log/close_x_vol/plus_di_x_low_vol
               adx_x_rsi/bb_x_vol/rsi_x_kdj/low_price
```

### 评分脚本特征生成
评分脚本 `us_v7_5_daily_score.py` 会从OHLCV数据动态生成这些特征，无需预计算。

---

## 五、局限性与故障处理

### 常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| 评分全在0-20 | 特征数据有问题（比如日期对不上） | 检查parquet最新数据 |
| Top5里盘>$10股 | 候选池过滤出异常 | 检查`ma5`价格过滤 |
| 评分几个月不变 | 模型权重没更新 | 重新训练（3-6个月一次） |
| 连续3个月无命中 | 市场结构变了 | 跑前瞻验证，考虑重训 |

### 何时重训模型
- ⏰ **常规**：每3-6个月重训一次
- 🚨 **紧急**：连续3个月前瞻命中率<10%
- 📈 **数据源升级后**（如加了期权数据）

重训脚本：
```bash
python scripts/us_v7_5_train_xgb.py  # 训练新V8-Lottery
```

---

## 六、性能测试环境

- **硬件**: RTX 3080 Ti (GPU), i7-13700K (CPU)
- **Python**: 3.12
- **XGBoost**: GPU加速 (tree_method=gpu_hist)
- **数据规模**: ~600只 × 5年 = ~750K样本
- **训练时间**: ~2分钟 (GPU)
- **单日评分**: <3秒 (CPU)

---

*文档版本: 1.0 (2026-06-12)*
*伴随V8-Lottery模型发布*
