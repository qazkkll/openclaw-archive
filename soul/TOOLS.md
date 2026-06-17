# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## What Goes Here

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Device nicknames
- Anything environment-specific

---

Add whatever helps you do your job. This is your cheat sheet.

## API Keys
- Finnhub: `d87hklhr01qmhakfrh0gd87hklhr01qmhakfrh10` — 新闻/基本面/分析师评级/财报日历
- NewsAPI: `7d8e0ca352664b6d9ccd96405949b5ea` — 备用（Finnhub优先）
- ~~Tushare teajoin: `f1354a1ae78f67c8dd529829551f74a48817db17771d6ea14704ffd2`~~ (已失效，见config/tushare.json中的完整token)

## 📋 输出格式铁律（2026-05-22）
- 永远用 stock-decision 模板（skills/stock-decision/SKILL.md）
- A股早报: Header → ❤️大势 → 💰总账 → ❤️评分红绿灯(全列) → 📦持仓 → 🎯今天该做的事
- 评分过买入线的**全部列出**，不能只挑几个
- 结论直接说买/卖/持有，不堆RSI/量比等数字
- 理由用一句话：趋势、信号、风险，不说具体分数

## 📊 板块标记与推荐范围（2026-05-25）
- 主板(60/00开头) → 前缀使用 📊 (如「📊 春秋电子 603890」)
- 创业板(30开头) → 前缀使用 📈 (如「📈 绿通科技 301322」)
- 科创板(688开头) → 前缀使用 📡 (如「📡 金宏气体 688106」)
- 所有A股（主板+非主板）都纳入每日推荐扫描
  - 主板 → Andy自己买
  - 创业板/科创板 → 妈妈可以买
  - 推荐时标注板块标记 + 谁可以买

## 💡 推荐流程铁律（2026-05-25）
- 评分红绿灯按买入线(62)划线，达标的全列
- **50-61分区间的**（接近买线但没到）单独加一行「👀 接近买入线」说明，但不能混进推荐里
- 每次综合推荐前：完整拉数据→思考→再回复，不着急抢答

## 📊 数据标注铁律（2026-05-25）
**所有数据输出必须标注：**
1. 📡 **数据源** — 哪个API/数据层（如腾讯Qt、新浪日线、Tushare、yfinance）
2. 🔢 **样本范围** — 从多少只票中筛选（如「从质量池1500只中筛主板Top 100」）
3. 🏆 **排名范围** — 展示的是Top几（如「Top 10排名」）
4. ⏰ **数据时间** — 数据截止时间点（如「基础数据截至05-22收盘」）

**示例：**
```
❤️ 评分红绿灯 · Top 10
#1 🟡 川润股份 57分
   数据源：V1评分(新浪日线) | 从1500质量池筛主板可买→评前30取Top10 | 数据截至05-22收盘
```

## 🔍 Cron验收标准（2026-05-25记）

检查定时任务三步走：
1. 脚本在crontab里？
2. 脚本语法通过？
3. **跑一次看实际输出内容是否符合预期？** ← 最容易漏的

之前踩过的坑：
- `refresh_pool.py` 跑完0只但日志说正常（Tushare返回str类型未转换）
- `defensive.py --us` 发持仓但不发推荐（功能不完整，只做了持仓未集成推荐）
