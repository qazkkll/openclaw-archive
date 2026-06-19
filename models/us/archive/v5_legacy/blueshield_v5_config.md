# 蓝盾V5 — 生产配置

## 版本信息
- **版本**: V5 (2026-06-19)
- **前身**: V3(公式) → V4(5天ML) → V5(120天ML)
- **Logo**: 🐱 猫猫贴纸

## 模型配置
```yaml
model:
  name: "蓝盾V5"
  algorithm: XGBoost
  device: CUDA (3080 Ti)
  params:
    max_depth: 6
    learning_rate: 0.03
    subsample: 0.8
    colsample_bytree: 0.8
    min_child_weight: 10
    n_estimators: 500
    early_stopping: 50

strategy:
  holding_period: 120  # 天
  top_n: 15
  universe: S&P500 (514只)
  rebalance: 每天评估，有变动就调
  
features:
  count: 28
  list:
    # 趋势
    - ma5              # 5日均线
    - ma20             # 20日均线
    - ma60             # 60日均线
    - ma_bias20        # MA偏离度
    - ma_align         # 均线排列
    - price_position   # 60日价格位置
    
    # 动量
    - ret1             # 1日收益
    - ret5             # 5日收益
    - ret20            # 20日收益
    - ret60            # 60日收益
    - momentum_6m      # 6个月动量
    - momentum_1m      # 1个月动量
    - mom_divergence   # 动量背离
    - trend_accel      # 趋势加速
    
    # 波动率
    - vol20            # 20日波动率
    - vol5             # 5日波动率
    - vol_ratio        # 量比
    - vol_change       # 波动率变化
    
    # 技术指标
    - rsi14            # RSI
    - rsi_change       # RSI变化
    - macd             # MACD
    - macd_signal      # MACD信号
    - macd_hist        # MACD柱
    - bb_std           # 布林标准差
    - bb_width         # 布林带宽
    - bb_pos           # 布林位置
    
    # 质量
    - ret_quality      # 收益/风险比
```

## Walk-Forward验证
- 训练窗口: 3年
- 重训频率: 每6个月
- 测试期: 2019-06 ~ 2025-12 (6.5年)

## 回测指标（已验证）
| 指标 | 数值 |
|------|------|
| 年化收益 | +55% |
| 夏普比率 | 1.36 |
| Sortino | 4.08 |
| 胜率 | 86% |
| SPY同期 | +15.8% |
| 超额年化 | +39% |
| 2022熊市 | +4.5% (SPY -17.5%) |

## 与V4的关键区别
| 维度 | V4 | V5 |
|------|-----|-----|
| 持有期 | 5天 | **120天** |
| 交易频率 | 50次/年 | **~12次/年** |
| 特征数 | 14-36 | **28** |
| 年交易笔数 | 900+ | **~100** |
| 交易成本 | ~27% | **~3%** |
| 验证方法 | 单次WF | **滚动WF** |

## 文件位置
- 模型脚本: `/tmp/decade_v2.py` (基础) / `/tmp/shield_opt_r1.py` (优化)
- 回测结果: `analysis/decade_backtest.json`, `analysis/shield_10yr_full.json`
- S&P500数据: `data/us/us_hist_sp500_10y.parquet`
