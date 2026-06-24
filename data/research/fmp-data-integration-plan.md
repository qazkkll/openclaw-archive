# FMP数据接入预期分析
> 生成时间: 2026-06-25
> 当前模型: V8蓝盾 ICIR=0.271, V12绿箭 ICIR=0.092

## 一、数据源到特征的量化映射

### 1. 分析师评级 (FMP Premium)
```
analyst_signal = (upgrade_count - downgrade_count) / (upgrade_count + downgrade_count + 1)
pt_change = (latest_pt - 30d_ago_pt) / 30d_ago_pt
pt_upside = (target_price - current_price) / current_price
buy_pct_change = this_month_buy_pct - last_month_buy_pct
rating_std = analyst_ratings_std
```
学术IC范围: 0.03-0.08 | 预期ICIR增量: +0.10~0.20

### 2. 新闻情绪 (FMP Premium + FinBERT)
```
news_sentiment_7d = weighted_avg(FinBERT.positive - FinBERT.negative, 7d)
news_volume = count(news, 7d)
sentiment_shift = this_week_sentiment - last_week_sentiment
neg_news_count = count(negative_news, 7d)
```
学术IC范围: 0.02-0.06 | 预期ICIR增量: +0.10~0.20

### 3. 机构持仓 (SEC 13F / FMP Ultimate)
```
inst_hold_change = this_quarter_pct - last_quarter_pct
top10_conc = top10_institutional_holdings_pct
new_inst_count = new_institutions_count
```
学术IC范围: 0.02-0.05 | 预期ICIR增量: +0.05~0.10

### 4. 内部人交易 (FMP Ultimate)
```
insider_net = (buy_count - sell_count) / (buy_count + sell_count + 1)
insider_buy_value = sum(buy_value, 90d)
```
学术IC范围: 0.02-0.04 | 预期ICIR增量: +0.05~0.10

### 5. 资金流代理 (OHLCV推导, $0)
```
cmf = Chaikin_Money_Flow(close, high, low, volume, 20d)
obv_trend = slope(OBV, 20d)
vol_anomaly = volume / volume_ma20
large_order_proxy = vol_anomaly * abs(ret1)
```
学术IC范围: 0.01-0.03 | 预期ICIR增量: +0.03~0.05

## 二、预期提升估算

### 当前状态
| 模型 | IC | ICIR | 月IC>0 | 状态 |
|------|------|------|--------|------|
| V8蓝盾 | 0.055 | 0.271 | 71% | 不可用(<0.5) |
| V12绿箭 | 0.013 | 0.092 | 56% | 不可用(<0.5) |

### 逐步接入预期 (保守/乐观)

| 阶段 | 新增因子 | V8 ICIR预期 | V12 ICIR预期 | 成本 |
|------|---------|------------|------------|------|
| 当前 | 29技术+宏观 | 0.271 | 0.092 | $0 |
| +1 资金流代理 | +4个CMF/OBV特征 | 0.30~0.33 | 0.12~0.15 | $0 |
| +2 分析师评级 | +5个analyst特征 | 0.40~0.50 | 0.20~0.30 | FMP Premium |
| +3 新闻情绪 | +4个sentiment特征 | **0.50~0.65** | **0.30~0.45** | FMP+FinBERT |
| +4 机构持仓 | +3个13F特征 | 0.55~0.70 | 0.35~0.50 | SEC免费 |

### 核心预期

**保守估计**: ICIR从0.271提升到0.45~0.55（接近可用线）
**乐观估计**: ICIR从0.271提升到0.55~0.70（明确可用）

### 因子相关性折扣

分析师评级和新闻情绪存在相关性（好消息→分析师上调），实际叠加效果约打7折：
- 独立叠加: 0.271 + 0.15 + 0.15 = 0.571
- 相关性折扣后: 0.271 + 0.10 + 0.10 = **0.471**

## 三、关键假设和风险

1. **因子独立性假设**: 分析师评级和新闻情绪有~30%相关性，不是完全独立
2. **样本外衰减**: 回测IC通常是真实OOS的1.5-2倍
3. **数据质量**: FMP新闻覆盖度和及时性需验证
4. **FinBERT准确率**: 英文~95%，中文需翻译(75%)
5. **过拟合风险**: 特征越多越容易过拟合，需严格WF验证

## 四、实施路径

1. **Phase 0 ($0)**: OHLCV推导CMF/OBV/量比 → 加入V8重训 → 验证ICIR提升
2. **Phase 1 (FMP Premium)**: 接入新闻+分析师评级 → FinBERT打分 → 加入V8重训
3. **Phase 2 (FMP Premium)**: 接入13F机构持仓(SEC免费补充) → 加入V8重训
4. **Phase 3**: 全因子V9模型 → Walk-Forward验证 → 全市场回测

每个Phase独立验证，不跳步。如果Phase 0的ICIR提升<0.02，说明特征工程方向需调整。
