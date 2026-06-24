# 全市场模型重训练报告
**日期**: 2026-06-24
**数据**: us_hist_full_10y.parquet (29.8M行, 11,864只股票, 2016-2026)

---

## 蓝盾V7 (>$10, 20天持有期, Top15)

| 指标 | V6(旧) | V7(新) | 变化 |
|------|--------|--------|------|
| 训练样本 | 2,911,122 | 1,916,030 | -34% |
| 股票池 | 2,359只 | 8,133只 | +245% |
| WF avg return | 30.1%* | 2.79% | ↓ |
| WF win rate | 60% | 61.5% | +1.5% |
| WF sharpe | 1.44* | 1.05 | ↓ |
| OOS avg return | N/A | 4.55% | 新增 |
| OOS win rate | N/A | 81.0% | 新增 |
| OOS sharpe | N/A | 2.65 | 新增 |
| OOS max_dd | N/A | -2.0% | 新增 |

*V6的WF数据来自S&P500子集，不可直接比较

### 信号阈值
- 🟢🟢 green2: ≥0.166 → avg=30.2%, win=96.2%, count=7604
- 🟢 green1: ≥0.047 → avg=10.4%, win=84.5%, count=179,254
- 🟡 observe: ≥0.022 → avg=6.7%, win=76.9%, count=495,626

### Top5特征重要性
1. spy_ret20 (宏观主导)
2. iwm_ret20
3. spy_ret5
4. iwm_ret60
5. vix_close

**结论**: V7覆盖全市场8133只（vs V6的2359只），OOS sharpe 2.65很强。特征转向宏观驱动(spy/iwm/vix)。

---

## 绿箭V12 ($1-$10, 5天持有期, Top5)

| 指标 | V11(旧) | V12(新) | 变化 |
|------|---------|---------|------|
| 训练样本 | 687,081 | 2,884,153 | +320% |
| 股票池 | 1,442只 | 1,986只 | +38% |
| WF avg return | 4.56% | 6.14% | +35% |
| WF win rate | 51.5% | 63.6% | +12.1% |
| OOS avg return | 5.56% | 6.36% | +14% |
| OOS win rate | 50% | 66.0% | +16% |

### 信号阈值
- 🟢🟢 green2: ≥0.218 → avg=46.2%, win=91.3%, count=823
- 🟢 green1: ≥0.082 → avg=19.6%, win=83.2%, count=14,214
- 🟡 observe: ≥0.039 → avg=10.1%, win=72.7%, count=88,507

### Top5特征重要性
1. price (价格本身)
2. qqq_ret20 (宏观)
3. iwm_ret60
4. iwm_ret20
5. spy_ret20

**关键修复**: 极端收益率clip(±100%)解决了penny stock噪声问题。V11 win rate只有50%, V12提升到66%。

---

## 风险提示

1. **V6 WF数据不可比**: V6的"30.1%年化"来自S&P500子集WF，V7是全市场WF，计算方式不同
2. **蓝盾特征偏向宏观**: Top5全是ETF收益率，个股技术面信号弱。市场风格切换时可能失效
3. **绿箭仍有长尾风险**: penny stock流动性差，实际滑点可能高于回测
4. **200棵树**: early stopping后只有200轮（原500轮），模型可能欠拟合
5. **OOS期间(2024-2026)是牛市**: win rate可能高于长期平均

## 文件位置
- 模型: `models/us/blueshield_v7_xgb.json` (1.4MB)
- 模型: `models/us/arrow_v12_xgb.json` (1.4MB)
- 元数据: `models/us/blueshield_v7_meta.json`
- 元数据: `models/us/arrow_v12_meta.json`
- 训练脚本: `scripts/us/retrain_full_market_v3.py`
