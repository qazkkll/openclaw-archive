# Skills分类存档
> 2026-06-18 | 200个skills，按使用频率和相关性分层

## 分类原则
- **L1（核心）**：每天使用，始终加载
- **L2（常用）**：每周使用几次，按需加载
- **L3（低频）**：每月使用1-2次，仅特定任务触发
- **L4（冗余）**：与我们工作无关，但保留以防万一

---

## L1 核心层（5个）— 量化交易+记忆系统

| Skill | 作用 | 来源 |
|-------|------|------|
| **memory-management** | 两层记忆架构，解决遗忘/膨胀问题 | 自写 |
| **session-archive** | 存档口令，session无缝接续 | 自写 |
| **quant-model-development** | 量化模型方法论（分类vs回归、特征选择） | 自写 |
| **ml-experimentation** | ML实验工作流（评估、校准、诚实诊断） | 自写 |
| **quant-scoring** | 每日评分pipeline（数据→特征→推理→报告） | 自写 |

## L2 常用层（8个）— 系统增强+数据研究

| Skill | 作用 | 来源 |
|-------|------|------|
| **curator-evolution** | skill自动审计、合并、清理 | 新装 |
| **hermes-dojo** | 自我监控+弱skill修复 | 新装 |
| **using-superpowers** | TDD+子代理编排，长时间自主工作 | 新装 |
| **statistical-analysis** | 统计分析（来自K-Dense） | K-Dense |
| **scikit-learn** | ML工具（来自K-Dense） | K-Dense |
| **exploratory-data-analysis** | EDA工作流（来自K-Dense） | K-Dense |
| **polars** | 高性能数据处理（来自K-Dense） | K-Dense |
| **timesfm-forecasting** | 时序预测（来自K-Dense） | K-Dense |

## L3 低频层（15个）— 开发+研究+环境

| Skill | 作用 | 来源 |
|-------|------|------|
| wsl2-development | WSL2开发模式 | 自写 |
| github-auth | GitHub认证 | Hermes |
| github-pr-workflow | PR流程 | Hermes |
| hermes-agent | Hermes配置 | Hermes |
| hermes-agent-skill-authoring | skill编写规范 | Hermes |
| llm-provider-configuration | Provider配置 | 自写 |
| xiaomi-mimo-provider | MiMo配置 | 自写 |
| project-absorption | 吸收外部代码库 | 自写 |
| plan | 规划模式 | Hermes |
| spike | 快速实验 | Hermes |
| systematic-debugging | 根因调试 | Hermes |
| test-driven-development | TDD | Hermes |
| arxiv | 论文搜索 | Hermes |
| polymarket | 预测市场 | Hermes |
| blogwatcher | RSS监控 | Hermes |

## L4 冗余层（~170个）— K-Dense生物/化学/医学skills

这些与量化交易无关，保留原因：
1. K-Dense是打包安装的，拆开需要手动操作
2. Hermes按需加载，不触发就不占token
3. 未来如果做跨领域研究可能有用

**与我们工作无关的典型skills：**
biopython, scanpy, rdkit, pysam, pydicom, histolab, clinical-reports, treatment-plans, phylogenetics, molecular-dynamics, drug-discovery等

---

## Token占用分析

### 每个skill的token成本
- **始终加载**：skill的name+description约100 tokens
- **按需加载**：完整SKILL.md约500-2000 tokens
- **200个skills的name+description列表**：约20,000 tokens（在system prompt中）

### 优化建议
1. **不删K-Dense**（省不了多少，且Hermes按需加载）
2. **L1 skills可以写短**（压缩到100-200 tokens）
3. **L4 skills的description写一句话**（不占context）
4. **定期用curator-evolver审计**（每月一次）
