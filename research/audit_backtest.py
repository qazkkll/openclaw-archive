#!/usr/bin/env python3
"""回测计算审计"""
import json, os
os.chdir(os.path.expanduser("~/.hermes/openclaw-archive"))

report = ["# 回测计算审计报告\n"]

# 1. backtest_report.json
report.append("\n## 1. backtest_report.json\n")
with open("models/cn/backtest_report.json") as f:
    bt = json.load(f)

report.append(f"- final_value: {bt['final_value']:,.2f}")
report.append(f"- total_return_pct (原始): {bt['total_return_pct']}%")
report.append(f"- cagr_pct (原始): {bt['cagr_pct']}%")
report.append(f"- test_dates: {bt['test_dates']}天")
report.append(f"- sharpe: {bt['sharpe']}")
report.append(f"- max_dd: {bt['max_drawdown_pct']}%")
report.append(f"- config: {bt['config']}")

initial = 100000
final = bt['final_value']
total_ret = (final / initial - 1) * 100
days = bt['test_dates']
years = days / 365
cagr_correct = (pow(final / initial, 1 / years) - 1) * 100

report.append(f"\n**审计:**")
report.append(f"- 正确总收益: {total_ret:.2f}%")
report.append(f"- 正确CAGR: {cagr_correct:.2f}%")
report.append(f"- 原始CAGR: {bt['cagr_pct']}%")
if abs(cagr_correct - bt['cagr_pct']) > 5:
    report.append(f"- **CAGR偏差: {abs(cagr_correct - bt['cagr_pct']):.1f}个百分点**")

# 2. production_results.json
report.append("\n## 2. production_results.json\n")
with open("research/production_results.json") as f:
    prod = json.load(f)

report.append("Walk-Forward各fold:\n")
for fold in prod["walk_forward"]:
    ann = fold["ann"]
    ic = fold["ic"]
    ls = fold["ls"]
    dd = fold["dd"]
    flag = ""
    if abs(ann) > 100:
        flag = " **异常**"
    report.append(f"- Fold {fold['fold']}: IC={ic:.4f}, LS={ls:.4f}, Ann={ann:.2f}%, DD={dd:.2f}%{flag}")

report.append(f"\n原始summary年化: {prod['summary']['ann']:.2f}%")
reasonable = [f['ann'] for f in prod['walk_forward'] if -100 < f['ann'] < 200]
if reasonable:
    report.append(f"去掉异常值后均值年化: {sum(reasonable)/len(reasonable):.2f}%")

# 3. plan_ab_results.json
report.append("\n## 3. plan_ab_results.json\n")
with open("research/plan_ab_results.json") as f:
    pab = json.load(f)

for plan in ["plan_a", "plan_b"]:
    p = pab[plan]
    report.append(f"\n**{plan.upper()}:**")
    report.append(f"- IC: {p['ic']:.4f}, Rank_IC: {p['rank_ic']:.4f}")
    report.append(f"- Long-Short: {p['long_short']:.4f}")
    report.append(f"- 原始年化: {p['annual_ret']:.2f}%")
    if abs(p['annual_ret']) > 100:
        report.append(f"- **年化异常，不可信**")
    for i, d in enumerate(pab[f"{plan}_details"]):
        ann = d['annual_ret']
        flag = " **异常**" if abs(ann) > 100 else ""
        report.append(f"  - Fold {i+1}: Ann={ann:.2f}%, IC={d['ic']:.4f}{flag}")

# 4. WF stability
report.append("\n## 4. Walk-Forward稳定性\n")
with open("models/cn/a3_v1_test_report.json") as f:
    a3 = json.load(f)

wf_labels = ["WF1 2020-21", "WF2 2021-22", "WF3 2022-23", "WF4 2023-24"]
for r in a3["results"]:
    if r["label"] in wf_labels:
        corr = r["corr"]
        d1 = r["D1_in_top10"]
        spread = r["spread"]
        status = "OK" if corr > 0.12 and d1 > 35 else ("WARN" if corr > 0.08 else "FAIL")
        report.append(f"- {r['label']}: corr={corr:.3f}, D1_top10={d1:.1f}%, spread={spread:.2f}% [{status}]")

# 5. Summary
report.append("\n## 5. 总结\n")
report.append("| 来源 | 原始指标 | 可信度 |")
report.append("|------|---------|--------|")
report.append("| backtest_report | CAGR 129% | 需验证 |")
report.append("| production_results | 年化3631% | 不可信 |")
report.append("| plan_ab | A=600% B=215% | 不可信 |")
report.append("| a3 WF | WF2 corr=0.07 | 不稳定 |")
report.append("| cn-alpha-v1.0 paper | 13% Sharpe0.72 | 唯一真实 |")
report.append("| cn-alpha-v1.1 | 32.3% Sharpe1.97 | 未验证 |")

output = "\n".join(report)
with open("research/audit_backtest.md", "w") as f:
    f.write(output)
print(output)
