# Project Mercurius — 交易系统规范

## 项目概述
- **名称**: Project Mercurius
- **目标**: 10年回测 + 分时级(5min)实盘模拟交易
- **市场**: 美股
- **初始资金**: $100,000 (Alpaca Paper)

## 数据源清单

### 1. Massive (Polygon.io) — $79/月
- **Key**: MASSIVE_API_KEY (32字符)
- **✅ 行情**: 1993起全市场SIP(Tick/Quote/1s/1min/5min/日K)，盘前盘后
- **✅ WebSocket**: T.*(逐笔), Q.*(报价), NEWS(新闻)实时推送
- **✅ 新闻**: 2018起Benzinga+RSS，2024-09后有LLM情绪分(-1~+1)
- **❌ REST当日K线**: 有15分钟延迟，禁止实盘用REST拉当日
- **❌ 期权**: 未采购
- **❌ Quotes/Financials**: 该档位不支持
- **限流**: REST 800次/min, WebSocket 1500订阅

### 2. FMP Premium — $69/月
- **Key**: FMP_API_KEY (32字符)
- **✅ 基本面**: 30年财报(收入/资产/现金流), PE/PB/ROE/ROIC, DCF
- **✅ 分析师**: 评级, 目标价共识, 上调/下调, Price Target历史
- **✅ 另类数据**: 13F机构持仓, 内幕交易, 国会交易, ETF持仓
- **✅ 新闻**: 2019起(标题+摘要+Source+时间)，无原生情绪
- **✅ 宏观**: 经济日历(CPI/Fed/非农), 财报日历
- **❌ sentiment字段**: 是评论数统计，禁止当情绪用
- **❌ Earnings Transcripts**: 该档位不支持
- **限流**: 750次/min, stable端点(v3/v4已废弃)

### 3. FinBERT 离线情绪库 — 本地, 零成本
- **路径**: data/finbert_sentiment/year={yyyy}/month={mm}/ticker={sym}.parquet
- **状态**: ❌ 未部署（数据不存在）
- **✅ 覆盖**: 2019-2026全市场新闻情绪(-1~+1) + 置信度(0~1)
- **✅ 过滤**: 仅Benzinga/PRNewswire/BusinessWire/SeekingAlpha
- **❌ 禁止**: 实时调用外部API / 修改预打标分

### 4. Alpaca Paper Trading — 免费
- **Key**: APCA_API_KEY_ID + APCA_API_SECRET_KEY
- **✅ 模拟下单**: 市价/限价/止损, 盘前盘后(默认关)
- **✅ 自动算**: 滑点, 手续费, 分红, 拆股
- **❌ 仅Paper**: 禁止实盘
- **⚠️ PDT**: 净值<$25K时5日内最多4次日内round-trip
- **⚠️ 做空**: easy_to_borrow=false会被拒
- **限流**: 下单200次/min

## 模式切换

### 回测模式
- 行情: Massive历史K线(1993-2026)
- 基本面: FMP历史财报/评级
- 情绪: FinBERT本地预打标
- 执行: 按K线收盘价/开盘价模拟(不调API)
- ❌ 禁止: 调实时API / 用未来数据

### 实盘模拟模式
- 行情: Massive历史(REST)+当日(WebSocket聚合5min)
- 基本面: FMP实时(盘前更新)
- 情绪: 2024-09后用Massive insights / 其余用FMP+Hermes实时
- 执行: Alpaca Paper API真实下单
- ❌ 禁止: REST拉当日 / 用历史情绪做当前决策

## 仓位规则
- 单票 ≤ 5% 净值
- 日止损 ≤ 2% 净值
- 🟢🟢≥0.08 / 🟢≥0.06 / 🟡≥0.04

## 数据源校验
- 每次启动: 4源全检 → 输出状态报告
- 每小时: 自动重检 → 失效告警+备用逻辑
- 每次调用前: 检查状态标记
- 2+源失效: 平仓+休眠+等人工

## 输出规范
每次决策必须附数据源引用说明：
```
【决策依据数据源】
1. 行情：Massive 5min K线（时间段，已验证有效）
2. 基本面：FMP Q3财报（PE=XX，已验证有效）
3. 新闻情绪：FinBERT 0.XX（时间，置信度X，已验证有效）
4. 交易执行：Alpaca Paper（净值$XXK，购买力$XXK，已验证有效）
```

## 绝对禁止
- 臆造数据/数据源
- 回测用实时/未来数据
- 实盘改FinBERT预打标分
- 调用未授权付费API
- 绕过有效性校验
