# MAINTENANCE.md — 日常维护方案

> 2026-06-17 设计，Hermory自运行，Andy监督

## 一、维护架构

```
自动层（Hermery Cron）
├── 每日 06:00 — 数据更新 + 质量检查
├── 每日 08:00 — 晨间评分（A股+美股）
├── 每周五 18:00 — 模型健康检查
└── 每月1号 — INDEX刷新 + 数据审计

手动层（Andy触发）
├── python3 scripts/train/xxx.py — 模型重训练
├── python3 scripts/data/pull_xxx.py — 数据补全
└── python3 scripts/utils/update_index.py — 索引刷新

监控层（实时）
└── 数据异常 → 自动报警到Telegram
```

## 二、自动维护任务

### 1. 每日数据更新 (06:00)

```bash
# 更新K线数据（增量）
python3 scripts/data/refresh_kline.py

# 检查数据新鲜度
python3 scripts/utils/check_data_fresh.py
```

**检查项：**
- K线是否更新到昨天
- 资金流是否落后>3天
- tushare API是否可用
- 数据完整性（无空值、无异常值）

**异常处理：**
- API失败 → 记录日志，下次重试
- 数据缺失 → 报警到Telegram

### 2. 晨间评分 (08:00)

```bash
# A2评分
python3 scripts/score/a2_score_only.py

# 美股评分（绿箭+蓝盾）
python3 scripts/score/us_score.py
```

**输出：** 评分结果保存到 `output/cn/` 和 `output/us/`

### 3. 每周健康检查 (周五 18:00)

```bash
# 模型健康检查
python3 scripts/utils/model_health_check.py
```

**检查项：**
- 模型文件是否完整
- 特征是否匹配
- 最近评分是否合理（无漂移）
- GPU状态

### 4. 月度审计 (每月1号)

```bash
# 刷新INDEX
python3 scripts/utils/update_index.py

# 数据审计
python3 scripts/utils/data_audit.py

# 生成月度报告
python3 scripts/utils/monthly_report.py
```

## 三、监控告警

### 告警级别

| 级别 | 条件 | 动作 |
|:--|:--|:--|
| 🔴 严重 | API连续失败>3次 | 立即Telegram通知Andy |
| 🟡 警告 | 数据落后>1天 | 记录日志，下次检查时提醒 |
| 🟢 信息 | 正常运行 | 静默 |

### 监控脚本

```python
# scripts/utils/monitor.py
# 检查项：
# 1. tushare API可用性
# 2. 数据新鲜度
# 3. 模型文件完整性
# 4. 磁盘空间
# 5. 最近评分异常检测
```

## 四、手动触发场景

| 场景 | 命令 | 说明 |
|:--|:--|:--|
| 模型重训练 | `python3 scripts/train/a1_layer3_xgb.py` | 建议每月或数据显著变化时 |
| 数据补全 | `python3 scripts/data/pull_moneyflow.py` | 新增股票或数据缺失时 |
| 索引刷新 | `python3 scripts/utils/update_index.py` | 文件变动后 |
| 紧急修复 | 直接编辑脚本 | 需Andy批准 |

## 五、维护责任矩阵

| 任务 | 负责人 | 频率 | 自动/手动 |
|:--|:--|:--|:--|
| 数据更新 | Hermory | 每日 | 自动 |
| 评分生成 | Hermory | 每日 | 自动 |
| 异常监控 | Hermory | 实时 | 自动 |
| 模型重训练 | Andy触发 | 按需 | 手动 |
| 数据补全 | Andy触发 | 按需 | 手动 |
| 系统升级 | Andy决策 | 按需 | 手动 |
| 月度审计 | Hermory | 每月 | 自动 |

## 六、备份策略

| 内容 | 位置 | 频率 |
|:--|:--|:--|
| 模型文件 | `models/` | 每次训练后 |
| 配置文件 | `data/config/` | 变更时 |
| 评分结果 | `output/` | 每日 |
| 索引 | `INDEX.md` | 每次更新后 |

**备份原则：** 模型文件小（KB级），全量备份；数据文件大（GB级），增量备份。

## 七、灾难恢复

1. **数据丢失** → 从tushare重新拉取（30-40分钟）
2. **模型丢失** → 从备份恢复或重新训练（1-2小时）
3. **脚本丢失** → 从git恢复（如果有版本控制）
4. **配置丢失** → 从 `data/config/` 恢复

**关键：** 所有重要文件都在 `~/.hermes/openclaw-archive/` 本地，不依赖D盘。
