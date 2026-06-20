#!/usr/bin/env python3
"""生产系统审计：特征匹配 + hold_days + 信号灯"""
import json, os, re
os.chdir(os.path.expanduser("~/.hermes/openclaw-archive"))

report = ["# 生产系统审计报告\n"]

# 1. 读取cn-alpha-v1.1模型的特征
report.append("\n## 1. 特征匹配检查\n")

# 从模型文件获取特征
model_path = "models/cn/cn_alpha_v1.1.json"
try:
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(model_path)
    model_features = model.feature_names
    if model_features:
        report.append(f"**模型特征 ({len(model_features)}):**")
        for i, f in enumerate(model_features, 1):
            report.append(f"  {i}. {f}")
    else:
        report.append("**模型特征: 无名称信息（XGBoost未保存feature names）**")
        model_features = []
except Exception as e:
    report.append(f"**模型读取失败:** {e}")
    model_features = []

# 从信号脚本获取特征
report.append("\n**信号脚本特征:**")
with open("scripts/cn/gen_signal_v1_1.py") as f:
    script = f.read()

# 提取features列表
feat_match = re.search(r"features\s*=\s*\[(.*?)\]", script, re.DOTALL)
if feat_match:
    feat_str = feat_match.group(1)
    script_features = [x.strip().strip("'\"") for x in feat_str.split(",") if x.strip().strip("'\"")]
    report.append(f"脚本特征 ({len(script_features)}):")
    for i, f in enumerate(script_features, 1):
        report.append(f"  {i}. {f}")
else:
    report.append("无法从脚本提取特征列表")
    script_features = []

# 对比
if model_features and script_features:
    model_set = set(model_features)
    script_set = set(script_features)
    missing_in_script = model_set - script_set
    extra_in_script = script_set - model_set
    
    report.append(f"\n**匹配结果:**")
    report.append(f"- 模型特征数: {len(model_features)}")
    report.append(f"- 脚本特征数: {len(script_features)}")
    report.append(f"- 完全匹配: {model_set == script_set}")
    
    if missing_in_script:
        report.append(f"- **脚本缺失 ({len(missing_in_script)}):** {sorted(missing_in_script)}")
    if extra_in_script:
        report.append(f"- **脚本多余 ({len(extra_in_script)}):** {sorted(extra_in_script)}")
    
    if missing_in_script or extra_in_script:
        report.append("\n**结论: 特征不匹配！信号脚本和训练模型用的不是同一套特征。**")
    else:
        report.append("\n**结论: 特征完全匹配。**")

# 2. hold_days确认
report.append("\n## 2. hold_days确认\n")

# 从summary
for fname in ["models/cn/cn_alpha_v1.1_summary.json", "models/cn/cn_alpha_v1.0_summary.json"]:
    with open(fname) as f:
        s = json.load(f)
    hd = s.get("hold_days", "未定义")
    report.append(f"- {fname}: hold_days={hd}")

# 从config
with open("config.json") as f:
    cfg = json.load(f)
report.append(f"- config.json A股 rebalance_days: {cfg.get('strategy',{}).get('shield',{}).get('rebalance_days','N/A')}")
report.append(f"- config.json hold_days: {cfg.get('models',{}).get('shield',{}).get('hold_days','N/A')}")

# 从production.json
with open("models/cn/production.json") as f:
    prod = json.load(f)
report.append(f"- production.json a2 target: {prod['active_models']['a2'].get('target','N/A')}")
report.append(f"- production.json a2 hold_days: (未显式定义，但target是fwd_10d)")

# 从训练脚本
report.append("\n训练脚本默认值:")
report.append("- a1_layer3_xgb.py: hold_days=10 (默认)")
report.append("- cn-alpha系列训练脚本: 未找到（可能是notebook或一次性脚本生成）")

# 结论
report.append("\n**结论:** cn-alpha-v1.1的hold_days最可能是10天（基于fwd_10d目标），但需要确认训练脚本。")

# 3. config vs production对齐
report.append("\n## 3. 配置对齐检查\n")
report.append(f"- config.json生产模型: cn-alpha-v1.1")
report.append(f"- production.json生产模型: a1_layer3_xgb_10d (A2)")
report.append(f"- **两个配置指向不同模型！**")
report.append(f"- gen_signal_v1_1.py加载: cn_alpha_v1.1.json")
report.append(f"- 实际使用的应该是config.json定义的（cn-alpha-v1.1）")

# 4. 信号灯状态
report.append("\n## 4. 信号灯检查\n")
if "🟢🟢" in script or "percentile" in script.lower() or "signal_level" in script.lower():
    report.append("信号脚本包含信号灯逻辑")
else:
    report.append("**信号脚本无信号灯逻辑** — 需要添加三层百分位分级")

output = "\n".join(report)
with open("research/audit_prod_system.md", "w") as f:
    f.write(output)
print(output)
