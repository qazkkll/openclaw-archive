# FMP Premium 美股核心股票数据包使用说明

## 1. 数据包是什么

本数据包是基于 FMP Premium 接口下载整理的美股核心股票原始 JSON 数据包，适合用于：

- 美股历史数据研究
- 基本面分析
- 量化回测前的数据准备
- AI 投研系统搭建
- 本地数据库导入
- 财务指标、估值、评级、目标价等数据分析

本数据包保存的是 FMP 接口返回的原始结构化 JSON 数据，不是人工二次加工后的 Excel 报表。不同接口的字段名、层级结构和日期字段会保持 FMP 原始返回格式。

## 2. 数据范围

本说明适用于本项目生成的 FMP Premium 美股核心股票数据包，包括 10 年标准包和 30 年完整包。具体数据范围以压缩包内 `package_meta.json`、`watermarks.json` 和文件名为准。

- 数据来源：Financial Modeling Prep, FMP Premium
- 股票市场：美国主要交易所股票
- 交易所：NASDAQ / NYSE / AMEX
- 股票状态：active
- 类型过滤：排除 ETF / Fund
- 市值过滤：marketCap >= 1,000,000,000 USD
- 股价过滤：price >= 1 USD
- 股票数量：以 `universe/current_universe.json` 为准
- 日线价格范围：以 `watermarks.json` 和日线文件名为准
- 10 年包：日线价格窗口约 10 年
- 30 年包：日线价格窗口最长约 30 年
- 实际最早交易日：如果起始日为非交易日，或股票上市时间较晚，则该股票从之后第一个可用交易日开始
- 年度财务：最多 30 个 annual period
- 季度财务：最多 120 个 quarterly period

股票池文件位于：

```text
universe/current_universe.json
```

## 3. 数据包含哪些内容

本数据包包含 69 个核心 FMP endpoint 的数据，主要分为以下几类：

| 分类 | 主要内容 |
| --- | --- |
| 股票池与基础列表 | stock list、financial statement symbol list、当前核心股票池 |
| 公司基础资料 | profile、key executives、company notes、peers |
| 行情与价格 | quote、quote short、historical daily price、price change、aftermarket quote/trade |
| 市值与股本 | market capitalization、market capitalization batch、shares float |
| 三大财务报表 | income statement、balance sheet、cash flow，包含年度和季度 |
| 财务指标 | ratios、key metrics、financial growth、financial scores |
| 估值与企业价值 | enterprise values、DCF、levered DCF、custom DCF |
| 股东回报与公司行动 | dividends、splits、earnings、symbol change、delisted 相关记录 |
| 分析师数据 | analyst estimates、price target summary、price target consensus |
| 评级数据 | ratings、ratings snapshot、ratings historical、grades、grades consensus、grades historical |
| 分拆与参考数据 | revenue geographic/product segmentation、sectors、industries、countries、exchanges |

本核心包不包含 forex、crypto、分钟级 K 线、小时线、新闻全文、SEC 全文、insider trading、ETF/Fund 持仓等非核心数据。

## 4. 压缩包结构

收到 zip 文件后，建议先完整解压到本地文件夹，再读取数据。不要直接在压缩包里批量打开大量 JSON 文件。

典型目录结构如下：

```text
README_中文说明.md
AI_USAGE_GUIDE.md
package_meta.json
manifest.json
checksums/sha256.json
merge_policy.json
watermarks.json
universe/current_universe.json
data/raw/*.json
```

各文件含义：

- `README_中文说明.md`：给用户阅读的中文说明。
- `AI_USAGE_GUIDE.md`：给 AI、程序员、数据工程师使用的处理指南。
- `package_meta.json`：数据包元信息，包括包 ID、文件数、原始大小、时间水印。
- `manifest.json`：下载任务清单，记录每个接口文件的 task name、URL、状态、大小、校验值等。
- `checksums/sha256.json`：包内文件 SHA-256 校验值。
- `merge_policy.json`：未来全量包、滚动包、补丁包合并时的主键策略。
- `watermarks.json`：数据包时间范围。
- `universe/current_universe.json`：本包股票池。
- `data/raw/*.json`：实际 FMP 原始接口数据。

## 5. 如何找到某只股票的数据

大多数单股票文件名都包含 `symbol-股票代码`。

例如 AAPL 的常见文件：

```text
data/raw/profile_symbol-AAPL.json
data/raw/quote_symbol-AAPL.json
data/raw/historical_price_eod_full_from-起始日期_to-截止日期_symbol-AAPL.json
data/raw/income_statement_annual_period-annual_symbol-AAPL_limit-30.json
data/raw/income_statement_quarter_period-quarter_symbol-AAPL_limit-120.json
data/raw/balance_sheet_statement_annual_period-annual_symbol-AAPL_limit-30.json
data/raw/cash_flow_statement_annual_period-annual_symbol-AAPL_limit-30.json
data/raw/ratios_annual_period-annual_symbol-AAPL_limit-30.json
data/raw/key_metrics_annual_period-annual_symbol-AAPL_limit-30.json
```

如果客户只想先测试数据结构，可以先使用 AAPL 样例包。样例包只包含 AAPL 单标的数据和少量全局参考文件，适合快速试用。

## 6. 推荐使用方式

推荐使用 Python、DuckDB、PostgreSQL、ClickHouse、Polars、Pandas 等工具读取和整理 JSON 数据。

不建议直接用 Excel 打开整个数据包，因为文件数量较多，且 JSON 结构不适合直接人工浏览。若需要 CSV 或 Excel，可以先用程序按股票、接口或年份转换。

### Python 简单读取示例

```python
import json
from pathlib import Path

root = Path("解压后的数据包目录")

with open(root / "universe/current_universe.json", "r", encoding="utf-8") as f:
    universe = json.load(f)

print(len(universe["symbols"]))

price_file = root / "data/raw/historical_price_eod_full_from-2016-01-01_to-2026-05-15_symbol-AAPL.json"
with open(price_file, "r", encoding="utf-8") as f:
    prices = json.load(f)

print(prices[0])
```

## 7. 给 AI 使用时怎么提示

可以把本说明和 `AI_USAGE_GUIDE.md` 一起提供给 AI，然后这样提问：

```text
这是一个 FMP Premium 美股核心股票原始 JSON 数据包。
请先阅读 README_中文说明.md 和 AI_USAGE_GUIDE.md。
然后根据 package_meta.json、manifest.json、universe/current_universe.json 理解数据结构。
不要假设所有 JSON 文件结构相同，请按 endpoint family 分别解析。
如果我要分析某只股票，请优先通过文件名里的 symbol 定位相关 JSON。
```

常见 AI 任务示例：

- 读取 AAPL 最近 10 年日线价格。
- 提取 AAPL 年度三大报表。
- 计算某只股票最近 10 年收入 CAGR、毛利率、净利率、ROE。
- 对比多只股票的估值指标。
- 把某个 endpoint 转成 CSV。
- 把全部数据导入本地数据库。

## 8. 重要注意事项

- 本包是 FMP 原始接口数据包，不是保证字段完全统一的标准数据库。
- 每只股票的上市时间不同，历史价格和财务数据长度可能不同。
- 新上市、并购、更名、退市等情况会影响历史完整性。
- 有些接口对某些股票可能返回空数组或较少记录，这是源数据常见情况。
- 分析师预期、目标价、评级等数据不应宣传为严格 point-in-time 历史修订库。
- 如果做严肃回测，需要注意幸存者偏差、股票池时间点、公司行为和复权逻辑。
- 如果未来同时使用历史底座包和滚动更新包，最近数据应优先读取滚动包，历史更早数据读取底座包。

## 9. 本包适合和不适合的场景

适合：

- 美股价格回测，具体可回测时间范围以包内日线数据为准
- 基本面量化研究
- AI 投研系统搭建
- 美股核心股票数据入库
- 财务、估值、评级、目标价等结构化分析

不适合：

- 超出本包 `watermarks.json` 范围的价格回测
- 分钟级或高频交易研究
- ETF/Fund 数据研究
- forex/crypto 数据研究
- 要求 point-in-time 历史预期修订的专业机构级数据库

## 10. 快速结论

如果你是普通用户，先解压 zip，然后阅读本文件。

如果你使用 AI，请同时提供 `README_中文说明.md` 和 `AI_USAGE_GUIDE.md`。

如果你是程序员或数据工程师，请从 `manifest.json` 和 `data/raw/` 开始构建 endpoint-specific loader，不要假设所有 JSON 文件都能放进同一张表。
