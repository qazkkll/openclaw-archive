# 🦅 Falcons/猎鹰 — Paper Trading验证系统

## 目标
通过Alpaca Paper Trading验证模型信号的实盘执行效果。
验证可行后，切换到Futu OpenD做真正的美股交易。

## 架构

```
scripts/falcons/
├── README.md           ← 本文件
├── setup_env.py        ← API凭据配置
├── alpaca_trade.py     ← Alpaca Paper Trading执行引擎
├── futu_trade.py       ← (TODO) Futu OpenD实盘引擎
├── scheduler.py        ← (TODO) 定时调度器
└── report.py           ← (TODO) 绩效报告
```

## 执行流

```
评分脚本 (每日06:30)
    ↓ 输出 JSON 信号文件
alpaca_trade.py
    ↓ 读取🟢🟢信号
    ↓ 检查持仓到期/止损
    ↓ 下单执行
    ↓ 记录交易日志
```

## 使用方法

```bash
# 查看账户状态
python3 scripts/falcons/alpaca_trade.py status

# 查看今日信号
python3 scripts/falcons/alpaca_trade.py signals

# 模拟运行（不下单）
python3 scripts/falcons/alpaca_trade.py full --dry-run

# 实盘执行
python3 scripts/falcons/alpaca_trade.py full

# 仅买入（不轮换）
python3 scripts/falcons/alpaca_trade.py execute --model arrow_v12

# 仅轮换（卖出到期/止损）
python3 scripts/falcons/alpaca_trade.py rebalance --hold-days 10 --stop-loss -15

# 交易历史
python3 scripts/falcons/alpaca_trade.py history
```

## 信号来源
- 绿箭V12: `data/us/arrow_v12_scored_YYYYMMDD.json`
- 蓝盾V10: `data/us/blueshield_v10_scored_YYYYMMDD.json`

## 交易规则
- **绿箭V12**: $1-$10小盘，top-5，持有20天，固定轮换
- **蓝盾V10**: >$10大盘，top-15，持有10天，trailing-15%
- **止损**: -15%（硬止损）
- **VIX过滤**: VIX>30暂停买入

## 切换到实盘
验证Paper Trading稳定后：
1. 确认Futu OpenD已连接 (127.0.0.1:11111)
2. 运行 `python3 scripts/falcons/futu_trade.py status`
3. 切换 `--mode live`
