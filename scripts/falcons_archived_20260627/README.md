# 🦅 Falcons/猎鹰 — 独立量化交易系统

## 架构

```
scripts/falcon/                    ← 回测+评分核心
├── falcon_score.py               ← ⭐ 独立评分 → 输出 scored JSON
├── falcon_v03_engine.py          ← 共享引擎 (PIT rank + 调仓 + Futu成本)
├── falcon_v03.py                 ← 统一回测 (--universe spx/r2k/both)
├── falcon_hybrid.py              ← Hybrid回测 (SPX+R2K固定比例)
├── falcon_oos_validation.py      ← OOS验证
└── CHANGELOG.md                  ← 版本记录

scripts/falcons/                   ← 交易执行
├── alpaca_trade.py               ← Alpaca Paper Trading (默认读Falcon信号)
├── futu_trade.py                 ← Futu OpenD 实盘 (默认读Falcon信号)
├── finbert_pipeline.py           ← FinBERT情绪打标
└── README.md                     ← 本文件
```

## 信号流

```
falcon_score.py (每日评分)
    ↓ 输出 data/falcon/falcon_scored_YYYYMMDD.json
alpaca_trade.py / futu_trade.py (读取信号)
    ↓ 检查持仓到期/止损
    ↓ 下单执行
    ↓ 记录交易日志
```

## 使用方法

```bash
# ⭐ Falcon 评分 (独立, 不依赖V10/V12)
python3 scripts/falcon/falcon_score.py                 # 评分最新交易日
python3 scripts/falcon/falcon_score.py --date 2024-12-31 --top-n 10

# 交易执行 (默认读Falcon信号)
python3 scripts/falcons/alpaca_trade.py status          # 查看账户
python3 scripts/falcons/alpaca_trade.py signals         # 查看信号
python3 scripts/falcons/alpaca_trade.py full --dry-run  # 模拟运行
python3 scripts/falcons/alpaca_trade.py full            # 实盘执行

# Futu 实盘
python3 scripts/falcons/futu_trade.py status
python3 scripts/falcons/futu_trade.py full --dry-run
```

## 模型配置

| 参数 | SPX | R2K |
|------|-----|-----|
| 权重 | Fund70+Ana20+Met10 | Pure_Fund |
| 调仓 | Fixed 30天 | Fixed 10天 |
| 止损 | -15% | -15% |
| 熊市仓位 | 50% | 30% |
| Top-N | 5 | 5 |

## OOS验证 (2024H2)

| 指标 | IS(2022-23) | Val(H1) | OOS(H2) |
|------|------------|---------|---------|
| Sharpe | 1.109 | 1.862 | **1.836** |
| MaxDD | 12.4% | 9.6% | 18.2% |
| WR | 58.8% | 50.0% | 60.0% |

**结论**: alpha真实, 衰减率1.66x (不降反升) ✅

## FinBERT情绪

```bash
# 状态检查
python3 scripts/falcons/finbert_pipeline.py status

# 回填 (需API key)
python3 scripts/falcons/finbert_pipeline.py backfill --months 6

# 每日增量
python3 scripts/falcons/finbert_pipeline.py daily
```
