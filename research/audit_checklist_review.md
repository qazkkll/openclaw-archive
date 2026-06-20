# 模型上线检查清单审计报告


## 1. A股生产配置现状

config.json A股模型: cn-alpha-v1.1
production.json A股模型: A2 Layer3 XGBoost 10日回归
config hold_days: 20
production target: fwd_10d

## 2. 美股生产配置现状

config.json 美股蓝盾: 蓝盾V6
  hold_days: 20
  top_n: 15
  signal_thresholds: {
  "system": "three_layer_percentile",
  "L1_vix_filter": "VIX > 30 \u2192 \u5173\u95ed\u6240\u6709\u4fe1\u53f7",
  "L2_absolute": "\u5206\u6570\u5fc5\u987b > \u5f53\u65e5\u4e2d\u4f4d\u6570",
  "L3_percentile": {
    "green2": {
      "percentile": 95,
      "label": "\ud83d\udfe2\ud83d\udfe2\u7cbe\u54c1\u4e70\u5165",
      "action": "\u9a6c\u4e0a\u4e0b\u5355"
    },
    "green1": {
      "percentile": 90,
      "label": "\ud83d\udfe2\u5f3a\u4fe1\u53f7",
      "action": "\u4e3b\u529b\u4fe1\u53f7"
    },
    "observe": {
      "percentile": 80,
      "label": "\ud83d\udfe1\u89c2\u5bdf",
      "action": "\u4e0d\u4e70\uff0c\u653ewatchlist"
    }
  }
}

美股评分脚本: ['scripts/us/us_v9_daily_score.py', 'scripts/us/us_score_engine.py', 'scripts/us/us_ld2_daily_score.py', 'scripts/us/score_engine.py', 'scripts/us/arrow_v11_score.py', 'scripts/us/audit_held_scores.py', 'scripts/us/test_score_math.py', 'scripts/us/rescore.py', 'scripts/us/comprehensive_score.py', 'scripts/us/us_ld3_daily_score.py', 'scripts/us/rescore_with_hl.py', 'scripts/us/us_v75_daily_score.py', 'scripts/us/sys_score_engine.py', 'scripts/us/blueshield_v2_score.py', 'scripts/us/us_v9_daily_score_original.py', 'scripts/us/blueshield_v6_score.py', 'scripts/us/us_v7_5_daily_score.py', 'scripts/us/green_signal_verify_v2.py', 'scripts/us/blueshield_v4_signal_levels.py', 'scripts/us/green_signal_verify.py', 'scripts/us/signal_frequency_2yr.py', 'scripts/us/bt_signal_test.py', 'scripts/us/us_v7_5_signal_gen.py', 'scripts/us/us_v75_signal_gen.py', 'scripts/us/greenarrow_extreme_sell_signal.py', 'scripts/us/us_signal_pusher.py', 'scripts/us/us_7.1_s2d_fetch_signals_fast.py', 'scripts/us/bt_signal_test2.py', 'scripts/us/bt_us_signal_test.py', 'scripts/us/us_v75_final_signal.py', 'scripts/us/us_7.1_s2c_fetch_signals.py']
美股paper trade结果: 无

## 3. 信号脚本完整性检查


**A股 (scripts/cn/gen_signal_v1_1.py):**
  ❌ 信号灯分级
  ✅ 宏观过滤器
  ❌ hold_days声明
  ✅ 输出格式化
  ✅ 结果保存
  ✅ 错误处理

## 4. 美股 vs A股流程对比

**美股配置一致性:**
  ✅ 信号灯
  ✅ hold_days一致
  ✅ top_n一致
  ✅ 止损规则
  ✅ 风险规则

## 5. 现有检查清单逐项评估

| # | 检查项 | 当前定义 | 评估 | 建议 |
|---|--------|---------|------|------|
| 1 | 特征匹配 | 模型特征 == 信号脚本特征 | ✅ 充分 | 可自动化，建议加入训练脚本 |
| 2 | 配置同步 | config.json == production.json == 信号脚本 | ⚠️ 不足 | 缺少自动校验脚本，且A股美股混在同一个config |
| 3 | hold_days确认 | 训练目标 == 配置 == 信号脚本 | ✅ 充分 | 但需要从模型meta自动提取 |
| 4 | Paper Trade验证 | 20+时点, Alpha>60%, Sharpe>0.5 | ⚠️ 不足 | 未区分A股/美股标准；未定义连续亏损上限 |
| 5 | 回测数字验算 | CAGR/年化公式手动检查 | ⚠️ 不足 | 应自动化验算，不用手动 |
| 6 | 信号脚本完整性 | 信号灯+宏观过滤+输出格式 | ⚠️ 不足 | 缺少"输出必须包含hold_days声明" |
| 7 | 交叉验证 | 新模型 vs 旧模型 head-to-head | ✅ 充分 | 建议标准化：同一时点、同一批股票、不同模型 |

## 6. 建议补充的检查项

- **8. 模型meta自动生成**: 训练完成后自动写入hold_days、特征列表、训练数据范围到_meta.json
- **9. 回测数字自动验算**: 脚本自动检查CAGR公式、异常值(>200%年化)标记警告
- **10. 分市场验证标准**: A股：Alpha>55%, Sharpe>0.5; 美股：Alpha>60%, Sharpe>0.8
- **11. 牛熊分段验证**: 必须包含至少1个熊市区间(2022/2025)，不能只在牛市验证
- **12. 上线前dry-run**: 标记生产前必须跑一次完整信号生成流程（不发给用户）
- **13. 版本回滚机制**: 如果paper trade连续2期负Alpha，自动回滚到上一个版本
- **14. 配置文件单点管理**: 每个市场一个production.json，config.json只引用不复制参数

## 7. A股 vs 美股差异处理

| 维度 | A股 | 美股 |
|------|-----|------|
| 交易成本 | ~0.15%双边 | ~0.1%双边 |
| 涨跌停 | ±10%/20% | 无限制 |
| 流动性 | 中小盘差 | 大盘好 |
| 数据源 | tushare | Yahoo/FinMind |
| 验证标准 | Alpha>55% | Alpha>60% |
| Sharpe门槛 | 0.5 | 0.8 |
| 信号灯 | 同一套P95/P90/P80 | 同一套 |
| hold_days | 10天(短线) | 20天(中线) |
| 市场过滤器 | MA60/120 + 涨跌比 | VIX > 30 |

## 8. 最终推荐：通用模型上线检查清单

```
模型上线检查清单 v1.0
========================

阶段一：训练完成后（自动）
  □ 自动生成_model_meta.json（特征列表、hold_days、训练数据范围、训练日期）
  □ 特征一致性检查：meta特征 == 信号脚本特征
  □ Walk-Forward验证：IC>5%, ICIR>0.5, 各fold无负IC

阶段二：验证阶段（半自动）
  □ Paper Trade：20+历史时点
  □ Alpha正占比 > 门槛（A股55%/美股60%）
  □ Sharpe > 门槛（A股0.5/美股0.8）
  □ 牛熊分段：至少包含1个熊市区间
  □ 新旧模型head-to-head：新模型胜率>55%
  □ 回测数字验算：CAGR公式正确，无异常值(>200%)

阶段三：部署前（手动）
  □ config.json参数同步（hold_days、top_n、信号阈值）
  □ production.json更新（指向正确模型）
  □ 信号脚本完整性（信号灯、宏观过滤、hold_days声明、输出格式）
  □ dry-run：跑一次完整信号生成，不发给用户
  □ 版本回滚机制：连续2期负Alpha自动回滚
```