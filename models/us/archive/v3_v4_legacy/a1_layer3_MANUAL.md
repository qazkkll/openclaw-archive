# A1 Layer 3 — 使用说明书

## 模型版本对比

| 模型 | 类型 | 特征 | 目标 | 推荐用途 |
|:---|:---|:---|---:|:---|
| `a1_layer3_xgb_10d` | XGBoost回归 | 37 | 预测10日涨幅 | **⭐⭐ 主力评分模型** |
| `a1_l3_clf_5d` | XGBoost分类 | 37 | 5日涨/跌分类 | ❌ 信号弱，不推荐 |
| `a1_layer1_xgb` | XGBoost 3分类 | 15 | 市场状态(防/中/激) | 参考用，用于宏观判断 |

## 评分流程

### 生产评分（每日盘后）

```bash
cd C:/Users/admin/.openclaw/workspace
python scripts/a1_layer3_5_scoring.py
```

输出：从质量池（30只）中按评分排序，输出Top10推荐。
评分数据源：检查点 `layer3_checkpoints/model_batch_5.json`

### 训练/重新训练

```bash
python scripts/a1_layer3_xgb.py
```

训练数据源：`a_hist_10y.parquet` + `moneyflow_data.parquet`
训练周期划分：train ≤2020, val 2021-23H1, test ≥23H2
输出：检查点保存在 `D:/openclaw/data/layer3_checkpoints/`
最终模型保存到 `D:/openclaw/data/models/a1_layer3_xgb_10d.json`

### 无前视回测验证

```bash
# A2纯L3评分回测（已做完）
python scripts/tmp_a2_fusion_bt.py  # 已删除(临时脚本不过夜)
```

回测结论：CAGR +59.71%，需加-15%止损

### 质量控制

| 检查项 | 标准 | 检测方法 |
|:---|---|:---|
| 特征完整性 | 全部37个非NaN | 调参前检查行数 |
| 资金流对齐 | 内链接K线+资金流 | 检查join后行数 |
| 无前视偏差 | 用历史数据验证 | 回测使用v6公平对比 |
| Q5区分度 | Q5 - Q1 > 5% | 检查分类五分位层差 |

## 故障处理

| 现象 | 原因 | 修复 |
|:---|:---|:---|
| 评分全为0 | 资金流数据未加载 | 检查moneyflow_core.parquet文件完整性 |
| 评分波动剧烈 | 当天资金流异常（如停牌） | 跳过当日，检查是否停牌日 |
| 模型加载失败 | 检查点文件损坏 | 重新训练或从models复制 |
| 回测收益异常高 | 前视偏差 | 检查日期是否使用train时间之外的数据 |
| 资金流特征为0 | moneyflow_core.parquet不完整 | bug-003曾出现，合并两个part |

## 依赖

- Python 3.12
- xgboost (已装)
- pandas, numpy
- 数据文件：a_hist_10y.parquet, moneyflow_data.parquet
