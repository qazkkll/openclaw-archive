# 📋 黑板

## 📖 启动指令
```
1. 读取 config.json（当前配置状态）
2. 读取 memory-archive/2026-06-20-progress.md（最新存档）
3. 读取本文件（待办事项）
```

---

## 🔧 工作流规则
- 决策/执行分离，Worker最多3个并行
- Skills: quant-research-workflow, blue-shield-model-development

---

## ✅ 蓝盾V6 + 绿箭V11 — 三层过滤已部署 (2026-06-20)

### 蓝盾V6
- 44维(技术27+宏观13+基本面4), XGBoost排名, **20天Top-15**
- OOS: 年化+30.1%, 夏普1.44, DD-11.1%, 胜率60%
- 信号系统: **三层过滤** (L1:VIX>30关闭, L2:>中位数, L3:Top5/10/20%)
- 抽样验证: 20d Alpha +9.86% (88%时间正), 胜率73%
- 模型: models/us/blueshield_v6_xgb.json
- 评分: scripts/us/blueshield_v6_score.py

### 绿箭V11
- 41维(技术28+宏观13), XGBoost排名, **5天Top-5**, $1-$10
- OOS: 每5天+5.56%, 夏普2.18, 胜率50%
- 信号系统: **三层过滤** (同上)
- 抽样验证: 5d Alpha +5.36% (88%时间正), 胜率71%
- 模型: models/us/arrow_v11_xgb.json
- 评分: scripts/us/arrow_v11_score.py

### 数据
- 10y数据: 2474只（补充了38只megacap: AAPL/MSFT/ASML/ANET等）
- v75宏观数据: 231只（S&P500子集，用于VIX/SPY等宏观特征）

---

## 待处理
- [ ] Pro模型切换
- [ ] 看板v4部署到Cloudflare隧道
- [ ] ASML/ANET数据缺失原因排查（已补充，但原始数据源为何缺失？）

## 已完成
- ✅ 三层过滤系统: 回测→部署→验证
- ✅ 抽样验证: 8组随机窗口，🟢🟢信号有效
- ✅ 数据修复: 38只megacap补充
- ✅ 配置文件: production.json/config.json/README.md全部更新
- ✅ 看板v4: 深色底+反emoji+数据密度
- ✅ 深度分析: 9只持仓逐一诊断+调仓建议

---

*最后更新: 2026-06-20*
