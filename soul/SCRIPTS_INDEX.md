# 🍤 小钳脚本索引

> 2026-05-30 v2 — 统一命名规则
> A_ = A股  US_ = 美股  sys_ = 系统通用

---

## ⏰ 自动运行（crontab定时任务）

| 脚本 | 原名 | 频率 | 功能 |
|:---|:---|:---:|:---|
| `A_morning_scan.py` | morning_scan | 工作日8:15 | 每日A股晨扫，出评分排名 |
| `A_unified_check.py` | unified_check | 盘中每10-15分 | A股+美股盘中监控，有异动推你 |
| `A_refresh_top100.py` | refresh_top100 | 每10分钟 | 刷新A股Top100评分 |
| `A_refresh_pool.py` | refresh_pool | 工作日17:00 | 收盘后刷新全市场质量池 |
| `US_refresh_pool.py` | refresh_us_pool | 工作日20:30 | 美股质量池刷新 |
| `US_daily_compare.py` | daily_compare | 15:30/4:00 | 每日收盘对比分析 |
| `sys_audit_engine.py` | audit_engine | 每天 | 审计引擎，记录每次操作 |
| `chat_logger.py` | — | 每小时 | 保存最近3天聊天记录 |
| `auto_distiller.py` | — | 每天3:00+17:00 | 整理记忆，更新7天滚动档案 |

## 📊 日常分析（随时可调用）

| 脚本 | 原名 | 功能 |
|:---|:---|:---|
| `sys_score_engine.py` | score_engine | **核心** - V1评分引擎 |
| `sys_analyst_oversight.py` | analyst_oversight | **核心** - 推荐前强制监督流程 |
| `sys_compliance.py` | compliance | 输出前合规自检 |
| `preflight.py` | — | 集成oversight的飞行前检查 |
| `A_sector_engine.py` | sector_engine | A股行业轮动分析 |
| `advisor.py` | — | 综合推荐引擎 |
| `notify.py` | — | 推送通知到Telegram |

## 📈 投资分析工具

| 脚本 | 功能 |
|:---|:---|
| `fund_manager_screen.py` | 基金经理三层筛选 |
| `news_monitor.py` | 新闻热点监控 |
| `price_alert.py` | 价格/止盈止损预警检查 |
| `fund_query.py` | 基本面查询(PE/PB/ROE) |
| `market_mode_check.py` | 市场牛熊判断 |

## 🧪 回测/验证（主要在本地32G跑）

| 脚本 | 功能 |
|:---|:---|
| `bt_v1_vs_lightgbm.py` | LightGBM vs V1权重对比 |
| `bt_model_validator.py` | 统一策略验证框架 |
| `bt_walkforward.py` | Walk-forward交叉验证 |
| `bt_compare_strategies.py` | InStock等多策略对比 |

## 🔧 数据获取

| 脚本 | 功能 |
|:---|:---|
| `data_source.py` | A股/美股数据获取（新浪/Baostock/腾讯） |
| `fetch_backtest_data.py` | 历史数据下载 |
| `finnhub.py` | 美股新闻查询 |

## 🏠 本地Worker

| 脚本 | 功能 |
|:---|:---|
| `dispatch.py` | 派发任务给本地电脑 |
| `upload_server.py` | 接收本地计算结果 |
| `futu_bridge.py` | 富途OpenD管理 |
| `query_pc.py` | 远程查询本地电脑数据 |

---

> 全部144个脚本保留在原位，以上为35个核心常用脚本。
> 旧版回测脚本在 `scripts/archive/` 和 `scripts/bt_*.py` 中可追溯。
