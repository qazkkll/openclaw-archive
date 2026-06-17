# 🍤 小钳云系统完整理解文档

> 生成时间: 2026-06-01 11:38 GMT+8
> 源端: ☁️ 阿里云香港 (8.217.51.136)
> 目标: 📡 Windows本地同步学习
> 阅读范围: 全部 ~140+ 个脚本 + 全部配置/记忆文件

---

## 目录

1. [系统概览](#1-系统概览)
2. [用户画像与规则](#2-用户画像与规则)
3. [交易策略体系](#3-交易策略体系)
4. [评分引擎详解 (V1)](#4-评分引擎详解-v1)
5. [美股评分引擎详解 (V4.2)](#5-美股评分引擎详解-v42)
6. [数据源架构](#6-数据源架构)
7. [定时任务体系](#7-定时任务体系)
8. [核心脚本详解](#8-核心脚本详解)
9. [风控/合规/审计体系](#9-风控合规审计体系)
10. [回测框架详解](#10-回测框架详解)
11. [记忆系统](#11-记忆系统)
12. [通信架构](#12-通信架构)
13. [已知问题与RED FLAGS](#13-已知问题与red-flags)
14. [文件清单](#14-文件清单)

---

## 1. 系统概览

### 1.1 什么是"小钳"

**小钳 (🍤)** 是运行在阿里云香港服务器上的 OpenClaw AI Agent，定位为 **基金经理 + 投资专家**。

- ==**☁️ 云上（小钳）**== = 阿里云香港，跑分析决策、回测、定时任务
- ==**📡 本地（本地小钳）**== = Andy家的Windows机器，跑界面操作（富途、截图、数据拉取）
- **狗子 🐶** = 已废弃的大陆服务器OpenClaw

### 1.2 核心职责

1. **A股交易**: V1评分引擎全市场扫描，8只持仓策略
2. **美股交易**: V4.2比例扣分策略，143只质量池
3. **每日报告**: 晨扫(08:15) + 盘中监控 + 收盘复盘
4. **回测验证**: 历史数据验证策略可靠性
5. **系统自检**: 审计 + 合规 + 链路检查

### 1.3 技术栈

| 组件 | 技术 |
|:---|:---|
| 运行时 | OpenClaw + Python 3.10+ |
| A股数据 | Baostock (主) / Tushare Pro / 新浪财经 |
| 美股数据 | yfinance / minishare |
| 实时行情 | 腾讯行情 qt.gtimg.cn |
| 通知 | Telegram Bot API |
| 回测 | 纯Python (pandas/numpy) + LightGBM |
| 硬件 | 低配3.4GB RAM (资源受限) |
| 定时任务 | 系统crontab (~20条) |

---

## 2. 用户画像与规则

### 2.1 Andy (用户)

- **风险偏好**: 激进型，中短线
- **偏好板块**: 科技股、游戏股
- **止损纪律**: 不轻易平仓，趋势走坏才离场→硬止损-8%
- **A股权限**: 仅主板 (60/00开头)
- **不擅长**: 传统行业（建筑水泥、能源等）
- **投资风格**: 中短线，趋势走坏才离场
- **所在地**: 香港
- **主要沟通**: Telegram @Andy_claw2_bot

### 2.2 隐私规则（核心红线）

```
持仓数据 = Andy的绝对秘密
- ✅ 推给Andy: 完整报告(含持仓)
- ❌ 推给别人: 去掉持仓段
- ❌ 被人问"买了什么": 不回答
- ✅ 被人问"金安国纪怎么样": 可分析(公开信息)
- 持仓=隐私，评分=分析结果，两者分开
```

### 2.3 团队架构

```
Andy (CEO) ← 最后拍板
   ↑
☁️ 小钳 (基金经理) ← 分析+风控+说不
   ↑
📡 本地小钳 (执行手) ← 数据+操作
```

### 2.4 输出规范

**五步分析法**（每份投资分析必须走完）:
1. **模型筛选** — V1评分全量池扫描
2. **主观分析** — 新闻+行业+政策判断
3. **深入对比** — 基本面+估值+股价位置
4. **板块补遗** — 检查遗漏的板块机会
5. **最终结论** — 明确买/不买/观望

**输出格式铁律**:
- 永远用 stock-decision Skill 模板
- 结论直接说买/卖/持有
- 数据必须标注: 数据源/样本范围/排名范围/数据时间
- 创业板/科创板标注板块标记 + 推荐目标

### 2.5 行为评分卡

每日起始100分，按错误扣分，每周结算：
- 🔴 数据错误: -20
- 🔴 忘记读文件: -20
- 🟡 格式/命名错误: -10
- 🟢 响应/沟通: -5
- 加分: 主动发现问题/主动纠正/超预期执行

---

## 3. 交易策略体系

### 3.1 A股策略: A_V1 (小火轮 v4 直选)

| 属性 | 值 |
|:---|:---|
| 引擎 | **V1评分** (MACD门控 + 动态权重) |
| 买入阈值 | ≥62分 |
| 卖出阈值 | <50分 |
| 最大持仓 | 8只 (牛市) / 3只 (熊市) |
| 调仓周期 | 7天 |
| 止损 | -8% 硬止损 |
| 最小持有 | 5天 |
| 防熊保护 | **MACD门控自动过滤** (不需单独熊市模式) |
| 候选池 | 质量池Top ~1500只 (活跃度排序) |

**回测表现 (2015-2026, 1,177只 vs 沪深300)**:
| 指标 | A_V1 | 沪深300 |
|:---|:---:|:---:|
| 累计 | **+273.91%** | +1.60% |
| 年化 | **14.10%** | — |
| 最大回撤 | **27.8%** | — |
| 夏普 | **0.76** | — |

**关键验证发现 (2026-05-31)**: V1评分原版+行业动量预筛选才是完整的策略。删除行业筛选后收益-88%。因此A_V1保持V1评分+行业动量两层过滤。

### 3.2 美股策略: U_V4.2 比例扣分

| 属性 | 值 |
|:---|:---|
| 候选池 | **143只质量池** (S&P 500: ROE>15% + 毛利率>40% + 市值>100亿) |
| 牛市模式 | **V4.2比例扣分**: 30日动量 - (距52周高位 × 0.7扣分) |
| 熊市模式 | V2逆向防守 (因子: MACD 15 + ADX 20 + 均线 15 + RSI 20 + 52周位置 30) |
| 模式切换 | SPY-MA200: SPY>MA200=牛市, <MA200=熊市 |
| 持仓 | 5只 |
| 调仓 | 20天 |

**回测表现 (2014-2025, 143只池)**:
| 指标 | V4.2 | QQQ | SPY |
|:---|:---:|:---:|:---:|
| 累计 | **+315.3%** | +255.5% | +174.2% |
| 年化 | **12.6%** | — | — |
| 夏普 | **3.73** | — | — |
| 最大回撤 | **19.2%** | — | — |

**牛市模式下美股不评分，只用30日动量排序**。评分只用于熊市模式的V2逆向防守。

### 3.3 市场模式切换

```
全市场V1评分≥50的占比:
  >25% 连续2次检查 → 牛市模式
  <15% 连续2次检查 → 熊市模式
  15-25% → 死区，维持不变

检查频率: 每7天（与调仓同步）
防震荡: 死区+连续确认双保险
```

状态文件: `data/market_mode.json`

### 3.4 防守配置 (无买入信号时)

```
牛市但无票: 银行ETF 60% + 黄金ETF 30% + 现金10%
熊市确认:    国债ETF 50% + 黄金ETF 30% + 现金20%
方向不明:    黄金ETF 40% + 国债ETF 35% + 银行ETF 15% + 现金10%
```

---

## 4. 评分引擎详解 (V1)

**文件**: `scripts/score_engine.py`

### 4.1 核心技术指标

V1评分从OHLC数据计算全部技术指标：

| 指标 | 参数 | 说明 |
|:---|:---:|:---|
| MA(5/20/60) | SMA | 简单移动平均线 |
| MACD | 12/26/9 | EMA计算，含柱线/信号线/趋势线 |
| RSI | 14日 | 相对强弱指标 |
| ADX | 14日周期 | 趋势强度指标 |
| 52周百分位 | 252天 | 股价在52周高低点中的位置 |
| 动量(20日/60日) | 20天/60天 | 短期+中期动量百分比 |
| 量比 | volume/20日均量 | 成交量相对放大 |

### 4.2 V1评分算法 (5因子)

**MACD门控（第一道闸）**:
- MACD柱 <= 0 → 评分直接归零
- MACD刚翻正(上穿零轴) → 20分
- MACD柱扩大 → 12分
- MACD柱仍在正区但缩小 → 6分

**5因子权重**（根据趋势状态动态切换）:

| 因子 | 趋势模式(ADX≥22) | 非趋势模式(ADX<22) |
|:---|:---:|:---:|
| MACD门控(ms) | 25 | 10 |
| 52周位置(ws) | 15 | 30 |
| 均线排列(mas) | 15 | 15 |
| ADX强度(ads) | 25 | 10 |
| RSI超卖(rsi) | 20 | 35 |

**动量奖金**（+8分上限）:
- mom20 > 0: 加 min(mom20×0.5, 8)
- mom60 > 0: 加 min(mom60×0.3, 6)
- 量比≥1.5: 加 min((量比-1)×3, 6)
- mom20<-10: 减5; mom60<-15: 减5

### 4.3 运行流程

1. `compute_indicators(close, high, low, volume)` → 计算全部指标
2. `v1_score(ind, di)` → 对当日数据索引应用评分算法
3. 返回 float 评分 (0~100)

---

## 5. 美股评分引擎详解 (V4.2)

**文件**: `scripts/scoring.py`

### 5.1 V4.2 比例扣分

```python
v42_score = 30日动量 - max(0, 距52周高位 - ds) × dc

参数:
  ds (deduct_start) = 40  → 距52周高位40%起扣
  dc (deduct_coeff)  = 0.7 → 扣分系数
  md (momentum_days) = 30  → 30日动量
```

### 5.2 信号灯 (美股)

| 信号 | 分数 |
|:---:|:---:|
| 🟢 加仓 | ≥60分 |
| 🔵 关注 | 50-59分 |
| 🟡 持有 | 35-49分 |
| 🟠 警惕 | 25-34分 |
| 🔴 卖出 | <25分 |

### 5.3 评分路由

`scoring.py`提供统一的`score(code)`入口，自动判断：
- A股代码(6位数字 / .SH/.SZ) → 走V1评分
- 美股代码(字母2-5位) → 走V4.2比例扣分
- **禁止直接调score_engine.py或V4.2逻辑**，必须走评分路由

---

## 6. 数据源架构

**文件**: `config/data_sources.json`, `scripts/data_source.py`

### 6.1 A股日K线

| 源 | 优先级 | 说明 |
|:---|:---:|:---|
| **Baostock** | **primary** | 免费、数据准确、香港可直达、~0.3s/只 |
| 新浪财经 | fallback (备用) | 免费、最快120天、速度~0.2s/只 |
| Tushare Pro | fallback (备用) | Andy账号2000积分、全市场一次拉完5506只 |

**当前primary**: Baostock (2026-06-01起切到)

### 6.2 A股实时行情

| 源 | 说明 |
|:---|:---|
| **腾讯行情** qt.gtimg.cn | 免费实时、五档涨跌量能 |

### 6.3 美股数据

| 源 | 说明 |
|:---|:---|
| **yfinance** (主) | 免费、港股直达、速度一般 |
| minishare (备) | 实时美股行情、有Token |

### 6.4 代码转换规则

```python
代码前缀:
  60/68 → sh (上证主板+科创板)
  00/30 → sz (深证主板+创业板)
  
转Tushare:
  60xx/68xx → 600xxx.SH
  00xx/30xx → 000xxx.SZ
```

### 6.5 数据获取流程

```
A股K线: AShareKline().get_best(code)  → primary→secondary
A股实时: AShareRealtime().get_quote(code) → 腾讯行情
美股K线: yfinance.download(code, period='1y')
美股行情: yfinance.Ticker(code).history(period='5d')
```

---

## 7. 定时任务体系

**系统crontab** (~20条), 全部走系统cron。无OpenClaw cron。

### 7.1 A股 (周一至周五)

| 时间 | 任务 | 脚本 | 关键数据 |
|:---:|:---|:---:|:---|
| 08:00 | 市场模式检查 | market_mode_check.py | 质量池Top200采样 |
| 08:00 | 审计日报 | audit_engine.py | 昨日审计事件汇总 |
| 08:15 | **晨扫** | morning_scan.py | 质量池全量扫描→推TG |
| 09:00 | 报告新鲜度检查 | check_stale_reports.py | 过期告警 |
| 09:30-14:50 每10分 | **盘中扫描** | unified_check.py | 评分+排名+异动 |
| 11:00,14:00 | 加仓信号 | unified_check.py --signal | 盘中加仓检查 |
| 15:30 | 收盘复盘 | daily_compare.py a | 持仓变化+资金流 |
| 17:00 | 质量池刷新 | refresh_pool.py | Tushare daily_basic |
| */15 | 新闻监控 | news_monitor.py | Finnhub+Google News |

### 7.2 美股 (周一至周五)

| 时间 | 任务 | 脚本 | 说明 |
|:---:|:---|:---:|:---|
| 21:00 | 美股持仓日报 | defensive.py --us | 市值/浮盈/距开盘 |
| 21:30-23:50 每10分 | 盘中扫描(上半场) | unified_check.py | 美股监控 |
| 00:00-03:50 (周二-周六) | 盘中扫描(下半场) | unified_check.py | 美股凌晨监控 |
| 04:00 (周二-周六) | 美股收盘复盘 | daily_compare.py us | 隔夜检查 |
| 05:00 (周二-周六) | 美股质量池刷新 | refresh_us_pool.py | yfinance数据 |

### 7.3 周/一次性任务

| 时间 | 任务 | 说明 |
|:---:|:---|:---:|
| 周日10:00 | 策略周检提醒 | 手动触发market_mode_check |
| 每天08:00,20:00 | GitHub备份 | backup.sh commit |
| 每周日03:00 | 备份归档 | backup.sh archive |
| 2026-06-10 | minishare续费提醒 | 一次性通知 |
| 2026-06-11 | 月复盘提醒 | 一次性通知 |

---

## 8. 核心脚本详解

### 8.1 `score_engine.py` (评分引擎)
- 提供V1评分算法 `v1_score(ind, di)`
- 入口: `v1_score_from_data(close, high, low[, volume[, idx]])`
- 调试: `get_raw_scores(ind, di)` 返回各子项分数明细
- **不要直接调用**，必须通过 `scoring.py` 路由

### 8.2 `scoring.py` (评分路由)
- 统一入口: `score(code, close, high, low, details)`
- 自动识别A股(→V1) / 美股(→V4.2)
- 自动拉取数据（如果没传）

### 8.3 `data_source.py` (数据源层)
- 提供 `AShareKline` 和 `AShareRealtime` 两个类
- 支持 sina/tushare/baostock 三个源
- `get_best(code)` → primary→secondary 降级
- `code_to_prefix()` / `code_to_board()` / `code_to_tscode()`

### 8.4 `morning_scan.py` (晨扫)
- **全量扫描质量池**（已修复：不止前100）
- 对每只股计算V1评分
- 分主板/创业板/科创板输出
- 持仓判断: ≥62分→持有, ≥50分→观望, <50分→卖出
- 无买入信号时推防守配置
- 内置自我校验 (verify_report) + 审计记录 + 合规检查

### 8.5 `unified_check.py` (盘中监控)
- 自动判断当前是否A股/美股交易时间
- 扫描质量池(前300只)/美股持仓+关注
- 三种预警: 趋势变化(进前10) / 价格触达(TP/止损) / 异动
- 带基金经理分析判断

### 8.6 `sector_engine.py` (行业轮动引擎)
- 三层分析: 行业动量 → 政策新闻 → 拥挤度
- 动量计算: 5/10/20/60日
- 趋势判断: 均线上方占比+加速信号
- 拥挤度惩罚: 短期涨太快+高波动率
- 输出: 强势/反转/规避行业列表

### 8.7 `market_mode_check.py` (市场模式)
- 全市场V1评分≥50占比
- 牛市>25% / 熊市<15% / 死区15-25%
- 连续2次确认才切换 (防震荡)
- 切换时推Telegram通知

### 8.8 `refresh_pool.py` (质量池刷新)
- 从Tushare daily_basic拉全市场基本面
- 过滤: 排除ST/退市/市值<15亿/换手率<0.3%/价格<2
- 活跃度排序: 换手率×0.7 + 量比×0.3
- 取Top 1500

### 8.9 `daily_compare.py` (收盘对比)
- 记录推荐vs实盘表现
- A股+美股资金流向(通过Tushare moneyflow)
- 更新recommendations.json

### 8.10 `analyst_oversight.py` (监督流程)
- 强制检查: 证据/风险/价格位置/替代方案
- 评分≥30(美股) / 距MA10不超过15% / 亏损不超过8%
- 不满足→抛 `OversightBlockedError`
- 出投资建议前必须过这个流程

### 8.11 `compliance.py` (合规监管)
- 每次分析后自检
- 检查: 数据源正确/扫描范围完整/评分路由正确/不用scoring禁止
- 不合规→TG告警推Andy

### 8.12 `audit_engine.py` (审计引擎)
- 每条链路执行后记录事件 (JSONL格式)
- 级别: success/warning/error/critical
- error/critical级别→推TG告警
- 每天发送审计日报
- 含Gateway断联监控

### 8.13 `notify.py` (通知层)
- 统一发送入口: `send(text, chat_id)`
- 走Telegram Bot API
- Bot: 7792764974:AAFrFrZ3JAjdhkCsphy2N-gd99U5puRywUI
- Andy Chat ID: 7908145929
- 支持群聊发送

### 8.14 `defensive.py` (防守配置)
- 三层判断: 牛市/震荡/熊市
- 输出ETF配置建议
- `--us` 参数输出美股持仓日报
- 个股成本/浮盈/距开盘时间

---

## 9. 风控/合规/审计体系

### 9.1 三层防护

```
Layer 1: MACD门控 (评分引擎自带)
  → MACD柱≤0 → 评分归零 → 自动过滤熊市

Layer 2: 分析师监督 (analyst_oversight)
  → 出推荐前强制检查
  → 检查证据/风险/价格位置/替代方案
  → 不通过→抛异常→不准出推荐

Layer 3: 合规检查 (compliance.py)
  → 分析后自检
  → 数据源/评分路由/扫描范围
  → 不合规→推Andy
```

### 9.2 审计体系

```
数据流:
  每条链路执行 → audit()写入JSONL
                    ↓
               daily_audit.py 检查链路
                    ↓
               audit_engine.py 汇总日报
                    ↓
               推Telegram给Andy
```

### 9.3 止损规则

- **硬止损**: -8% (单票) / -50% (最大单票损失)
- **软止损**: 持仓评分<50分 → 卖
- **趋势止损**: MA20/MA10破位
- **美股**: 硬止损价格在alerts.json设置

### 9.4 换仓规则 (A股)

```
同时满足:
  ① 持仓评分从买入价持续下跌≥15分
  ② 已持有≥5天
  ③ 全市场有评分≥62的新标的
```

### 9.5 换仓规则 (美股)

```
同时满足:
  ① 持仓评分从买入价持续下跌≥15分
  ② 已持有≥7天
  ③ 全市场有评分更高的新标的
```

---

## 10. 回测框架详解

### 10.1 回测脚本分类

| 类型 | 脚本 | 说明 |
|:---:|:---|:---|
| 主回测框架 | `bt_framework.py` | V1完整回测框架，参数全配置化 |
| 对比回测 | `bt_compare_strategies.py` | V1 vs V1改进版对比 |
| 暴力扫描 | `bt_v41_sweep.py` | 美股960组合参数扫描 |
| 验证 | `bt_model_validator.py` | 统一策略验证 |
| Walk-forward | `bt_walkforward.py` | 9段窗口交叉验证 |
| LightGBM对比 | `bt_v1_vs_lightgbm.py` | 因子预测力分析 |
| 美股回测 | `us_backtest_v3.py` | 美股V4.2验证 |
| 合计 | ~40+个bt_*.py | 全部在scripts/目录 |

### 10.2 主回测框架 (A_V1)

**文件**: `scripts/A_framework.py` / `scripts/bt_framework.py`

**流程**:
1. 加载Yahoo数据 + V1评分缓存
2. 按日期遍历:
   - 行业动量预筛选(前4行业)
   - V1评分排序取候选
   - 卖出条件检查(评分<50 / -8%止损 / 已到最低持有期)
   - 再平衡(7天): 15%剩余现金法买入
3. 计算指标: 总收益率/年化/最大回撤/夏普/交易次数

**关键参数**:
```python
max_positions = 8
buy_threshold = 62
sell_threshold = 50
rebalance_days = 7
stop_loss_pct = -0.08
position_pct = 15%  # 15%剩余现金法
min_hold_days = 5
transaction_cost = 0.003
```

### 10.3 LightGBM因子分析结果 (2026-05-31)

| 因子 | 预测力贡献 |
|:---|:---:|
| **20日动量** | **~73%** |
| **60日动量** | **~73%** (合计与20日高度相关) |
| MACD | <13% |
| ADX | <13% |
| RSI | <13% |

**结论**: 动量因子主导A股短期预测力，MACD/ADX/RSI只是辅助验证。

### 10.4 关键回测结果对照

| 策略版本 | 累计收益 | 年化 | 说明 |
|:---|:---:|:---:|:---|
| V4直选 (V1+行业动量) | **+273.91%** | **14.10%** | ✅ 生产版本 |
| V4直选 (纯评分, 无行业) | -88.08% | — | ❌ Bug版本 |
| V3 (行业筛选5只) | +63.25% | +4.81% | 旧版本 |
| V4.2美股 (143只池) | +315.3% | +12.6% | ✅ 美股生产版本 |

---

## 11. 记忆系统

### 11.1 文件结构

```
memory/
├── YYYY-MM-DD.md        ← 每日原始日志
├── rolling_7day.md       ← 7天滚动摘要 (启动必读)
├── strategy_A.md         ← A股策略文档
├── strategy_US.md        ← 美股策略文档
├── errors.md             ← 错误教训 (RED FLAGS)
├── org_chart.md          ← 组织架构
├── records.md            ← 推荐记录追踪
├── cron_A.md             ← A股cron排程表
├── cron_US.md            ← 美股cron排程表
├── daily_schedule.md     ← 每日完整排程
├── daily_summary.md      ← 每日总结
├── analysis_logic_chain.md ← 五步分析法
├── token_reminders.md    ← 到期提醒
├── index.json            ← 知识索引
├── _wiki.md              ← Wiki页面
├── chains/               ← 决策链
├── daily/                ← 每日摘要（压缩版）
├── feedback/             ← 反馈记录
├── knowledge/            ← 知识库
├── learnings/            ← 学习教训
├── procedures/           ← SOP
├── project/              ← 项目跟踪
├── reference/            ← 参考资料
├── user/                 ← 用户偏好
└── .dreams/              ← 梦境（辅助思考）
```

### 11.2 工作流

```
启动时自动加载:
  1. rolling_7day.md → 昨天到今天发生了什么
  2. 最近的YYYY-MM-DD.md → 具体细节
  3. index.json → 知识库索引

自动整理 (每天03:00 + 17:00):
  1. 更新rolling_7day.md
  2. 压缩7天前日志(去废话留精华)
  3. 检查待办项
```

### 11.3 对话归档

- 对话日志保存在 `conversation_archive/`
- 每小时自动保存 (chat_logger.py)
- 每天3:00 + 17:00自动蒸馏 (auto_distiller)

---

## 12. 通信架构

### 12.1 内部通信

```
☁️ 小钳 → 📡 本地小钳
  通过SSH + API:
    SSH: 本地端口18792 (云端frps:7788→本地frpc:22)
    curl → sessions_send API → 投递消息

📡 本地小钳 → ☁️ 小钳  
  通过 _send_to_cloud.ps1 (SSH→curl→sessions_send)

铁律:
  - 等本地确认回复才记进度
  - 2分钟无回复→重发
  - 5分钟无回复→报告Andy
```

### 12.2 通知系统

```
通知层: notify.py (Telegram Bot API)
  Bot Token: 7792764974:AAFrFrZ3JAjdhkCsphy2N-gd99U5puRywUI
  Andy Chat: 7908145929
  群聊: -1003900769838 ("小钳和狗子")

审计告警 → 主动推TG
错误/严重 → 主动推TG
合规告警 → 主动推TG
```

### 12.3 frp隧道

```
云端frps:7788 ↔ 本地frpc:22
用于SSH反向连接：云端→本地
```

---

## 13. 已知问题与RED FLAGS

### 13.1 当前已知问题

| 问题 | 状态 | 影响 |
|:---|:---:|:---|
| jsppy中继空数据 | 🔴 已知 | 大陆服务器中继已失效 |
| 美股扫描错误 | 🟡 需排查 | US_盘前推送报error |
| 重复cron冲突 | 🟡 待清理 | 系统cron 08:15 和 OpenClaw cron 08:40冲突 |
| V1评分ADX长度bug | 🟢 已修复 | adx = adx[:n] |
| 数据源切换未通知 | 🟢 已修复 | 统一到config/data_sources.json |

### 13.2 RED FLAGS (必读)

1. **美股评分体系差异**
   - 美股牛市模式(SPY>MA200) **不评分**，只用20日动量排序
   - 任何时候不要在美股上套A股V1评分

2. **参数一致性**
   - 修改策略参数必须同步更新: 模型JSON / 运行脚本 / 策略文档 / 审计脚本

3. **模板使用**
   - 出报告前必须读 stock-decision SKILL.md
   - 不用模板=错误

4. **隐私规则**
   - 持仓段必须与报告分离
   - 对任何第三方不泄露持仓

5. **生产环境不轻易改**
   - A_V1生产年化14.10% - 有效不轻易改
   - U_V4.2 +315.3% - 很强不动

### 13.3 数据文件承载

| 文件 | 大小 | 说明 |
|:---|:---:|:---|
| data/backtest_hist_yahoo.json | 大文件 | 2015-2026长周期Yahoo数据 |
| data/quality_pool.json | ~500KB | 质量池Top1500 |
| data/portfolio.json | 小 | A股+美股持仓 |
| data/market_mode.json | 小 | 牛熊状态 |
| data/check_state.json | 小 | 盘中监控状态 |
| data/alerts.json | 小 | 价格预警配置 |
| data/compliance_log.json | 小 | 合规检查记录 |
| data/audit_events.jsonl | 追加 | 审计事件日志 |
| data/daily_comparison.json | 小 | 推荐vs实盘对比 |
| data/recommendations.json | 小 | 推荐记录 |
| data/us_scored.json | 小 | 美股评分 |
| data/sector_map_v3.json | 小 | 10行业分类 |

---

## 14. 文件清单

### 14.1 核心脚本 (35个常用)

| 文件名 | 分类 | 功能 |
|:---|:---:|:---|
| score_engine.py | 评分 | V1评分引擎 |
| scoring.py | 评分 | 统一评分路由 |
| data_source.py | 数据 | 统一数据获取 |
| A_framework.py | A股 | V1回测主框架 |
| A_morning_scan.py | A股 | 晨扫 (原名morning_scan) |
| A_refresh_pool.py | A股 | 质量池刷新 |
| A_refresh_top100.py | A股 | Top100评分刷新 |
| A_sector_engine.py | A股 | 行业轮动分析 |
| A_unified_check.py | A股 | 盘中监控 |
| A_V2.py | A股 | 资金流因子版 |
| US_daily_compare.py | 美股 | 收盘对比 |
| US_refresh_pool.py | 美股 | 美股池刷新 |
| US_refresh_intraday.py | 美股 | 盘中刷新 |
| sys_score_engine.py | 系统 | 评分引擎 (别名) |
| sys_analyst_oversight.py | 系统 | 监督流程 |
| sys_compliance.py | 系统 | 合规检查 |
| sys_audit_engine.py | 系统 | 审计引擎 |
| advisor.py | 分析 | 综合推荐引擎 |
| defensive.py | 分析 | 防守配置/美股日报 |
| notify.py | 通知 | Telegram推送 |
| unified_check.py | 监控 | 统一盘中监控 |
| morning_scan.py | A股 | 晨扫 (旧名) |
| market_mode_check.py | 系统 | 市场模式判断 |
| daily_compare.py | 分析 | 每日对比 |
| daily_audit.py | 审计 | 链路审计 |
| refresh_pool.py | A股 | 质量池刷新(旧名) |
| compliance.py | 合规 | 合规监管 |
| audit_engine.py | 审计 | 审计引擎 |
| preflight.py | 检查 | 飞行前检查 |
| sector_engine.py | 分析 | 行业轮动 |
| news_monitor.py | 新闻 | 新闻监控 |
| price_alert.py | 预警 | 价格预警 |
| moneyflow_tracker.py | 资金 | 资金流追踪 |
| backup.sh | 运维 | GitHub备份 |
| upload_server.py | 通信 | 接收本地数据 |

### 14.2 回测脚本 (~40+)

```
bt_framework.py, bt_v41.py, bt_v41_sweep.py, bt_v41_best_yearly.py
bt_v42_dual.py, bt_v4_12year.py, bt_v4_out_of_sample.py, bt_v4_yearly.py
bt_v4_us.py, bt_verify.py, bt_walkforward.py, bt_combined.py
bt_compare_strategies.py, bt_model_validator.py, bt_light_verify.py
bt_momentum_bull.py, bt_optimized.py, bt_sector_final.py
bt_signal_test.py, bt_signal_test2.py, bt_us_fast.py
bt_us_models_test.py, bt_us_signal_test.py, bt_v1_correct.py
bt_v1_efficient.py, bt_v1_etf_dual.py, bt_v1_real.py
bt_v1_upgrade.py, bt_v1_vs_lightgbm.py, bt_v3_comparison.py
bt_v3_optimize.py, bt_v3_round2.py, bt_v3_round3.py
bt_bear_asean.py, bt_bear_sweep.py, bt_bear_validate.py
bt_lowcpu.py, bt_us_fast.py, us_backtest_v3.py
capital_backtest.py, cn_dual_10y.py, cn_dual_mode.py
us_dual_mode.py, us_dual_test.py, us_fixed_bt.py
us_grid_search.py, us_long_bt.py, us_macro_backtest.py
us_qvm_backtest.py, us_stock_backtest.py
```

### 14.3 审计/运维脚本

```
audit/audit_controller.sh     - 审计控制器
audit/protect.sh              - 资源保护(内存/CPU)
audit/recovery_daemon.sh      - 恢复看门狗
audit/auto_recovery.js        - 自动恢复
audit/task_wrapper.sh         - 任务包装器
audit/intraday_watch.js       - 盘中监控(NodeJS)
audit/intra_pulse.js          - 盘中脉冲
audit/intra_pulse_us.py       - 美股脉冲
audit/lhb_watch.js            - 龙虎榜监控
audit/post_close_correction.js - 收盘修正
audit/new_user_watch.js       - 新用户监控
audit/data_source.js           - 数据源校验
audit/model_validation.js      - 模型验证
audit/cron_check.js            - Cron检查
audit/read_image.py            - 图片读取(智谱GLM)
```

---

## 附录A: Config文件一览

| 文件 | 说明 |
|:---|:---|
| config/strategy.json | 策略主配置 (买卖阈值/持仓/止损) |
| config/data_sources.json | 数据源切换 |
| config/alerts.json | 价格预警 (持仓+TP/止损) |
| config/output_templates.json | 输出模板 |
| config/news_monitor.json | 新闻监控 |
| config/api_keys_reference.md | API密钥参考 |
| config/tushare.json | Tushare配置 |
| config/telegram_token.txt | TG Bot Token |

## 附录B: 内存开销 (低配3.4GB RAM注意事项)

- 避免并行执行重操作 (回测+扫描同时跑)
- RAM≥75%清理老session, ≥90%紧急清理
- 后台任务默认4核, 空闲12核
- Python线程数控制: 回测用subprocess隔离

---

## 附录C: 数据源切换记录

| 日期 | 原因 | 从→到 |
|:---|:---|:---:|
| 2026-06-01 | 去掉Tushare 200次/分钟限流 | Tushare→Baostock |
| 2026-05-25 | Tushare恢复 | Sina→Tushare |
| 2026-05-20 | jsppy中继空数据 | jsppy→Sina/Yahoo |
| 2026-05-15 | Sina API方向修复 | 修复reverse bug |

---

*文档完 · 共阅读 ~15,000+ 行代码/配置/文档*
