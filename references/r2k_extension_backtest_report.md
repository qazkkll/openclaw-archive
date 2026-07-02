# R2K扩展回测报告

**日期**: 2026-07-02  
**模型**: Falcon V0.4.4 (fund_ratio + growth_composite + qoq + cashflow)  
**回测引擎**: backtest_engine.py (Walk-Forward, 2yr train, 6mo test)

---

## 1. 数据概况

| 数据集 | Tickers | 日期范围 | 交易日数 |
|--------|---------|----------|----------|
| SPX | 476 | 2016-01-04 → 2026-07-01 | 2,638 |
| Russell 2000 | 691 | 2022-01-03 → 2024-12-31 | 753 |

**关键发现**: SPX和R2K**零重叠**（0 tickers），是完全不同的universe。

## 2. 回测结果对比

### 2.1 同期对比（2022-2024，公平比较）

| 配置 | Sharpe | MaxDD | CAGR | Win Rate | Trades |
|------|--------|-------|------|----------|--------|
| **SPX Only (Adjusted)** | **2.230** | **-10.4%** | **44.1%** | **62%** | 40 |
| SPX+R2K (Adjusted) | 0.734 | -17.0% | 30.8% | 51% | 41 |

### 2.2 全周期对比（SPX 2016-2026）

| 配置 | Sharpe | MaxDD | CAGR | Win Rate | Trades |
|------|--------|-------|------|----------|--------|
| SPX Only (V0.4.4) | 1.121 | -31.1% | 28.6% | 57% | 640 |
| SPX Only (Adjusted) | 1.153 | -29.9% | 29.9% | 58% | 640 |
| SPX+R2K (Adjusted) | 0.734 | -17.0% | 30.8% | 51% | 41 |
| SPX+R2K (Equal Weight) | 0.555 | -17.5% | 15.6% | 44% | 41 |

### 2.3 Top-10选股质量

| 指标 | SPX Only | SPX+R2K |
|------|----------|---------|
| 样本数 | 870 | 232 |
| 平均30日收益 | +3.53% | +2.06% |
| 中位数30日收益 | +2.58% | +0.65% |
| 胜率 | 59.1% | 53.0% |
| Sharpe (daily) | 0.780 | 0.415 |
| 最大收益 | +104.88% | +62.83% |
| 最大亏损 | -37.13% | -40.23% |

## 3. 分析

### 3.1 为什么R2K表现差？

1. **Universe差异**: SPX是大盘股（平均市值$200B+），R2K是小盘股（平均市值$2-5B），财务特征完全不同
2. **因子失效**: 模型的fund_ratio/growth_composite权重是为SPX大盘股校准的，对小盘股不适用
3. **数据质量**: R2K公司分析师覆盖少，财务数据噪音大，因子信号弱
4. **流动性**: 小盘股流动性差，实际交易成本更高

### 3.2 R2K的唯一优势

- **MaxDD改善**: -10.4% → -17.0%（SPX+R2K）vs -29.9%（SPX全周期）
- 但这主要是因为R2K数据只覆盖2022-2024（牛市为主），不是真正的分散化收益

### 3.3 Top-10选股示例（SPX）

- 2024-01: NVDA, ABNB, META, MRK, PODD
- 2024-06: INTU, MRK, HSY, BX, NVDA
- 2024-12: VRTX, MPWR, VST, APP, META

这些是高质量大盘股，模型在SPX上选股能力强。

## 4. 结论

### ❌ 不建议扩展到R2K

**理由**:
1. Sharpe下降67%（2.230 → 0.734）
2. 胜率下降11个百分点（62% → 51%）
3. Top-10选股质量全面下降
4. 模型因子权重是为SPX校准的，对小盘股不适用

### 建议方向

1. **保持SPX only**: 当前模型在SPX上表现优异（Sharpe > 2.0）
2. **如果要扩展**: 需要为小盘股单独训练模型（不同因子权重）
3. **数据改进**: 需要更长周期的R2K价格数据（当前只有3年）
4. **因子扩展**: 为小盘股添加流动性、动量、波动率等额外因子

---

## 附录：技术细节

### 模型权重

- **V0.4.4 (SPX)**: fund_ratio=0.45, growth_composite=0.20, qoq=0.20, cashflow=0.15
- **Adjusted (无cashflow)**: fund_ratio=0.55, growth_composite=0.20, qoq=0.25
- **Equal Weight**: fund_ratio=0.33, growth_composite=0.33, qoq=0.33

### Walk-Forward配置

- 训练窗口: 2年
- 测试窗口: 6个月
- 持有期: 30天
- Top-N: 10只
- 交易成本: 0.1% (单边)
- 止损: -15%

### 数据来源

- SPX: features_v04_1.parquet (156列, 476只)
- R2K: fmp_ratios_russell.json, fmp_metrics_russell.json, fmp_growth_russell.json, fmp_analyst_russell.json, russell_prices.json

---

**生成脚本**: `scripts/falcon/r2k_extension_backtest.py`
**生成时间**: 2026-07-02
