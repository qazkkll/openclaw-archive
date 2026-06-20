#!/usr/bin/env python3
"""审计模型上线检查清单：是否全面且可扩展到美股蓝盾/绿箭"""
import json, os, glob
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

report = ["# 模型上线检查清单审计报告\n"]

# 1. 读取A股生产配置
report.append("\n## 1. A股生产配置现状\n")
with open("config.json") as f:
    cfg = json.load(f)

with open("models/cn/production.json") as f:
    prod_cn = json.load(f)

report.append(f"config.json A股模型: cn-alpha-v1.1")
report.append(f"production.json A股模型: {prod_cn['active_models']['a2']['name']}")
report.append(f"config hold_days: {cfg['models']['shield']['hold_days']}")
report.append(f"production target: {prod_cn['active_models']['a2']['target']}")

# 2. 读取美股生产配置
report.append("\n## 2. 美股生产配置现状\n")
report.append(f"config.json 美股蓝盾: {cfg['models']['shield']['name']}")
report.append(f"  hold_days: {cfg['models']['shield']['hold_days']}")
report.append(f"  top_n: {cfg['models']['shield']['top_n']}")
report.append(f"  signal_thresholds: {json.dumps(cfg['models']['shield']['signal_thresholds'], indent=2)}")

# 检查美股评分脚本
us_scripts = glob.glob("scripts/us/*score*") + glob.glob("scripts/us/*signal*")
report.append(f"\n美股评分脚本: {us_scripts}")

# 检查美股是否有paper trade
us_paper = glob.glob("models/us/*paper*") + glob.glob("research/*us*paper*")
report.append(f"美股paper trade结果: {us_paper if us_paper else '无'}")

# 3. 读取A股信号脚本完整性
report.append("\n## 3. 信号脚本完整性检查\n")

scripts_to_check = {
    "A股": "scripts/cn/gen_signal_v1_1.py",
}

for name, path in scripts_to_check.items():
    if os.path.exists(path):
        with open(path) as f:
            content = f.read()
        
        checks = {
            "信号灯分级": "🟢🟢" in content or "percentile" in content.lower() or "green2" in content.lower(),
            "宏观过滤器": "vix" in content.lower() or "market" in content.lower() or "regime" in content.lower(),
            "hold_days声明": "hold_days" in content.lower() or "持有" in content or "rebalance" in content.lower(),
            "输出格式化": "print" in content and "排名" in content,
            "结果保存": "json.dump" in content or "save" in content.lower(),
            "错误处理": "try" in content and "except" in content,
        }
        
        report.append(f"\n**{name} ({path}):**")
        for check, passed in checks.items():
            emoji = "✅" if passed else "❌"
            report.append(f"  {emoji} {check}")

# 4. 美股流程对比
report.append("\n## 4. 美股 vs A股流程对比\n")

# 检查美股是否有类似问题
us_config_check = {
    "信号灯": cfg.get("display", {}).get("traffic_light") is not None,
    "hold_days一致": cfg["models"]["shield"]["hold_days"] == cfg["strategy"]["shield"]["rebalance_days"],
    "top_n一致": cfg["models"]["shield"]["top_n"] == cfg["scoring"]["shield"]["top_n"],
    "止损规则": cfg["strategy"]["shield"].get("exit_rules") is not None,
    "风险规则": len(cfg.get("risk_rules", [])) > 0,
}

report.append("**美股配置一致性:**")
for check, passed in us_config_check.items():
    emoji = "✅" if passed else "❌"
    report.append(f"  {emoji} {check}")

# 5. 清单评估
report.append("\n## 5. 现有检查清单逐项评估\n")

checklist_items = [
    ("特征匹配", "模型特征 == 信号脚本特征", "充分", "可自动化，建议加入训练脚本"),
    ("配置同步", "config.json == production.json == 信号脚本", "不足", "缺少自动校验脚本，且A股美股混在同一个config"),
    ("hold_days确认", "训练目标 == 配置 == 信号脚本", "充分", "但需要从模型meta自动提取"),
    ("Paper Trade验证", "20+时点, Alpha>60%, Sharpe>0.5", "不足", "未区分A股/美股标准；未定义连续亏损上限"),
    ("回测数字验算", "CAGR/年化公式手动检查", "不足", "应自动化验算，不用手动"),
    ("信号脚本完整性", "信号灯+宏观过滤+输出格式", "不足", "缺少\"输出必须包含hold_days声明\""),
    ("交叉验证", "新模型 vs 旧模型 head-to-head", "充分", "建议标准化：同一时点、同一批股票、不同模型"),
]

report.append("| # | 检查项 | 当前定义 | 评估 | 建议 |")
report.append("|---|--------|---------|------|------|")
for i, (name, desc, status, suggestion) in enumerate(checklist_items, 1):
    emoji = "✅" if status == "充分" else "⚠️"
    report.append(f"| {i} | {name} | {desc} | {emoji} {status} | {suggestion} |")

# 6. 建议补充的检查项
report.append("\n## 6. 建议补充的检查项\n")

new_items = [
    ("8. 模型meta自动生成", "训练完成后自动写入hold_days、特征列表、训练数据范围到_meta.json"),
    ("9. 回测数字自动验算", "脚本自动检查CAGR公式、异常值(>200%年化)标记警告"),
    ("10. 分市场验证标准", "A股：Alpha>55%, Sharpe>0.5; 美股：Alpha>60%, Sharpe>0.8"),
    ("11. 牛熊分段验证", "必须包含至少1个熊市区间(2022/2025)，不能只在牛市验证"),
    ("12. 上线前dry-run", "标记生产前必须跑一次完整信号生成流程（不发给用户）"),
    ("13. 版本回滚机制", "如果paper trade连续2期负Alpha，自动回滚到上一个版本"),
    ("14. 配置文件单点管理", "每个市场一个production.json，config.json只引用不复制参数"),
]

for name, desc in new_items:
    report.append(f"- **{name}**: {desc}")

# 7. A股 vs 美股差异
report.append("\n## 7. A股 vs 美股差异处理\n")
report.append("| 维度 | A股 | 美股 |")
report.append("|------|-----|------|")
report.append("| 交易成本 | ~0.15%双边 | ~0.1%双边 |")
report.append("| 涨跌停 | ±10%/20% | 无限制 |")
report.append("| 流动性 | 中小盘差 | 大盘好 |")
report.append("| 数据源 | tushare | Yahoo/FinMind |")
report.append("| 验证标准 | Alpha>55% | Alpha>60% |")
report.append("| Sharpe门槛 | 0.5 | 0.8 |")
report.append("| 信号灯 | 同一套P95/P90/P80 | 同一套 |")
report.append("| hold_days | 10天(短线) | 20天(中线) |")
report.append("| 市场过滤器 | MA60/120 + 涨跌比 | VIX > 30 |")

# 8. 最终推荐清单
report.append("\n## 8. 最终推荐：通用模型上线检查清单\n")
report.append("```")
report.append("模型上线检查清单 v1.0")
report.append("========================")
report.append("")
report.append("阶段一：训练完成后（自动）")
report.append("  □ 自动生成_model_meta.json（特征列表、hold_days、训练数据范围、训练日期）")
report.append("  □ 特征一致性检查：meta特征 == 信号脚本特征")
report.append("  □ Walk-Forward验证：IC>5%, ICIR>0.5, 各fold无负IC")
report.append("")
report.append("阶段二：验证阶段（半自动）")
report.append("  □ Paper Trade：20+历史时点")
report.append("  □ Alpha正占比 > 门槛（A股55%/美股60%）")
report.append("  □ Sharpe > 门槛（A股0.5/美股0.8）")
report.append("  □ 牛熊分段：至少包含1个熊市区间")
report.append("  □ 新旧模型head-to-head：新模型胜率>55%")
report.append("  □ 回测数字验算：CAGR公式正确，无异常值(>200%)")
report.append("")
report.append("阶段三：部署前（手动）")
report.append("  □ config.json参数同步（hold_days、top_n、信号阈值）")
report.append("  □ production.json更新（指向正确模型）")
report.append("  □ 信号脚本完整性（信号灯、宏观过滤、hold_days声明、输出格式）")
report.append("  □ dry-run：跑一次完整信号生成，不发给用户")
report.append("  □ 版本回滚机制：连续2期负Alpha自动回滚")
report.append("```")

output = "\n".join(report)
with open("research/audit_checklist_review.md", "w") as f:
    f.write(output)
print(output)
