# AI Usage Guide For The FMP Premium US Core Equity Data Package

This guide is written for AI assistants, developers, and data engineers. Use it as the first technical context document before parsing the unzipped data package.

## 1. Package Identity

This is a raw FMP Premium JSON data package for a filtered US core equity universe.

It is not a normalized SQL database, not a CSV dataset, and not an Excel workbook. Each file under `data/raw/` is a raw response from one FMP endpoint task.

Package range and symbol universe must be read from `package_meta.json`, `watermarks.json`, and `universe/current_universe.json`.

The current package family uses this general universe definition:

- Source: Financial Modeling Prep, FMP Premium
- Exchanges: NASDAQ, NYSE, AMEX
- Status: active symbols only
- Exclusions: ETF and Fund excluded
- Minimum market cap: 1,000,000,000 USD
- Minimum price: 1 USD
- Symbol count: read from `universe/current_universe.json`
- Daily price range: read from `watermarks.json` and the historical price filenames
- Annual financial limit: up to 30 periods
- Quarterly financial limit: up to 120 periods

Important interpretation:

- A `10Y` package has a daily price window of about 10 years.
- A `30Y` package has a maximum daily price window of about 30 years.
- Financial data can be longer than the daily price label because annual data uses up to 30 periods and quarterly data uses up to 120 periods.
- Not every symbol has the full maximum history. Actual history depends on listing date and FMP coverage.
- Always read package metadata instead of hardcoding package ranges.

## 2. Required First Steps

When receiving this package, do this first:

1. Unzip the package to a local folder.
2. Read `package_meta.json`.
3. Read `watermarks.json`.
4. Read `universe/current_universe.json`.
5. Read `manifest.json`.
6. Inspect several representative files under `data/raw/`.
7. Build endpoint-specific parsers.

Do not assume that all JSON files share the same schema.

## 3. Important Files

| File | Purpose |
| --- | --- |
| `README_中文说明.md` | Human-facing Chinese documentation |
| `AI_USAGE_GUIDE.md` | Technical guide for AI and data tools |
| `package_meta.json` | Package id, kind, record count, raw bytes, watermark dates |
| `watermarks.json` | Time range and base package metadata |
| `manifest.json` | Packaged task records and package paths |
| `checksums/sha256.json` | SHA-256 checksums for packaged files |
| `merge_policy.json` | Merge keys for future full/delta/patch packages |
| `universe/current_universe.json` | Current symbol universe |
| `data/raw/*.json` | Raw FMP endpoint responses |

## 4. Data Directory

Actual data files are stored under:

```text
data/raw/
```

The filename is the most important routing clue. Common patterns:

```text
profile_symbol-AAPL.json
quote_symbol-AAPL.json
historical_price_eod_full_from-YYYY-MM-DD_to-YYYY-MM-DD_symbol-AAPL.json
income_statement_annual_period-annual_symbol-AAPL_limit-30.json
income_statement_quarter_period-quarter_symbol-AAPL_limit-120.json
balance_sheet_statement_annual_period-annual_symbol-AAPL_limit-30.json
cash_flow_statement_annual_period-annual_symbol-AAPL_limit-30.json
ratios_annual_period-annual_symbol-AAPL_limit-30.json
key_metrics_annual_period-annual_symbol-AAPL_limit-30.json
enterprise_values_annual_period-annual_symbol-AAPL_limit-30.json
analyst_estimates_period-annual_symbol-AAPL_limit-10_page-0.json
ratings_snapshot_symbol-AAPL.json
price_target_summary_symbol-AAPL.json
```

Use `manifest.json` when filenames are not enough. Each manifest record includes `task_name` and `package_path`.

## 5. Endpoint Families

Normalize data by endpoint family, not by one global schema.

Recommended table families:

- `symbols`
- `reference`
- `company_profiles`
- `quotes`
- `daily_prices`
- `market_cap`
- `shares_float`
- `income_statements`
- `balance_sheets`
- `cash_flow_statements`
- `ratios`
- `key_metrics`
- `financial_growth`
- `enterprise_values`
- `dividends`
- `splits`
- `earnings`
- `analyst_estimates`
- `price_targets`
- `ratings`
- `grades`
- `dcf`
- `company_events`

## 6. Recommended Merge Keys

When converting to database tables, use endpoint-specific primary keys.

Suggested keys:

| Data family | Suggested key |
| --- | --- |
| Daily prices | `symbol + date` |
| Income statement | `symbol + period + fiscalDateEnding/date/calendarYear` |
| Balance sheet | `symbol + period + fiscalDateEnding/date/calendarYear` |
| Cash flow | `symbol + period + fiscalDateEnding/date/calendarYear` |
| Ratios | `symbol + period + date` |
| Key metrics | `symbol + period + date` |
| Enterprise values | `symbol + period + date` |
| Dividends | `symbol + date` |
| Splits | `symbol + date` |
| Analyst estimates | `symbol + period + date/fiscalDateEnding` |
| Ratings / grades | `symbol + date` where available |
| Profile / quote snapshots | latest record by package timestamp |

If a date key is missing, inspect the raw endpoint schema before choosing a key.

## 7. Python Loading Examples

Read package metadata:

```python
import json
from pathlib import Path

root = Path("UNZIPPED_PACKAGE_DIR")

meta = json.loads((root / "package_meta.json").read_text(encoding="utf-8"))
watermarks = json.loads((root / "watermarks.json").read_text(encoding="utf-8"))
universe = json.loads((root / "universe/current_universe.json").read_text(encoding="utf-8"))

print(meta["package_id"], watermarks, len(universe["symbols"]))
```

Read AAPL daily prices:

```python
import json
from pathlib import Path

root = Path("UNZIPPED_PACKAGE_DIR")
price_files = sorted((root / "data/raw").glob("historical_price_eod_full_*symbol-AAPL.json"))
path = price_files[0]

rows = json.loads(path.read_text(encoding="utf-8"))
rows = sorted(rows, key=lambda row: row["date"])

print(rows[0])
print(rows[-1])
```

Find all files for a symbol:

```python
from pathlib import Path

root = Path("UNZIPPED_PACKAGE_DIR")
symbol = "AAPL"

files = sorted((root / "data/raw").glob(f"*symbol-{symbol}*.json"))
for path in files[:20]:
    print(path.name)
```

Use `manifest.json` to map task names to packaged files:

```python
import json
from pathlib import Path

root = Path("UNZIPPED_PACKAGE_DIR")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

for record in manifest["records"][:5]:
    print(record["task_name"], record.get("package_path"))
```

## 8. AI Prompt Template

When an AI assistant is asked to process this package, use this prompt:

```text
You are processing a raw FMP Premium US equity JSON data package.
First read README_中文说明.md, AI_USAGE_GUIDE.md, package_meta.json, watermarks.json, manifest.json, and universe/current_universe.json.
Do not assume all JSON files have the same schema.
Use manifest.json and file names to locate endpoint-specific raw files.
Normalize data by endpoint family, such as daily_prices, income_statements, balance_sheets, cash_flow_statements, ratios, key_metrics, analyst_estimates, ratings, and company_profiles.
If the user asks for one symbol, locate files containing symbol-TICKER in data/raw/.
If a field is missing, inspect the raw JSON response before making assumptions.
```

## 9. Common User Tasks

An AI or program should be able to answer these after reading the package:

- List all symbols in the package.
- Read AAPL daily prices for the package date range.
- Extract annual income statements for a given symbol.
- Extract quarterly balance sheets and cash flows.
- Compute revenue growth, gross margin, net margin, ROE, ROA, debt/equity, FCF margin, and valuation ratios.
- Convert one endpoint family to CSV.
- Build a local SQL/DuckDB/Parquet database.
- Compare multiple symbols by ratios, key metrics, analyst estimates, and price targets.

## 10. Cautions

- Treat missing or empty endpoint responses as normal unless the manifest shows a failed download.
- Do not use one universal date key for all endpoints.
- Do not assume analyst estimates are strict point-in-time historical revision data.
- Do not claim every symbol has the full maximum daily price history. A 30Y package means the package requests and stores a maximum 30-year daily price window, not that every symbol has existed for 30 years.
- For serious backtests, account for survivorship bias, universe timing, splits, dividends, symbol changes, and delisting behavior.
- If future rolling update packages are used, read the rolling package first for recent years and fall back to the historical base package for earlier data.

## 11. Expected First Response

If an AI receives this package from a user, it should start with:

```text
I will first inspect package_meta.json, watermarks.json, manifest.json, universe/current_universe.json, and representative files under data/raw/. This package contains raw FMP JSON endpoint responses, so I will parse it by endpoint family instead of assuming a single table schema.
```
