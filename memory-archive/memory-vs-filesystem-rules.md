# Memory vs. 文件系统 — 信息存储分类规则

> 决定一条信息该存在哪里的原则。

## 一、Memory（常驻，每轮注入system prompt）

**存什么：下次session启动时必须立刻知道的事实**

| 类型 | 示例 | 为什么必须在memory |
|------|------|-------------------|
| 身份/偏好 | Andy是PM，中文，要求独立思考 | 新session第一句话就要用 |
| 当前生产方案 | V4-LGB WF夏普1.13 + V3监控 | 不知道就没法继续工作 |
| 关键约束 | 夏普<1不满意，config.json唯一阈值 | 防止犯重复错误 |
| 活跃决策 | 梯度卖单价格，当前持仓概要 | 下次操作需要参考 |
| 踩坑记录 | Futu API不接受market参数 | 每次都可能遇到 |
| 追踪机制 | 投资建议追踪表的路径和格式 | 需要知道在哪里更新 |

**判断标准：如果这个信息缺失，新session会犯错或重复问用户 → 存memory**

## 二、memory-archive/（文件，按需读取）

**存什么：重要但不需要每轮都知道的详细信息**

| 类型 | 示例 | 为什么放文件 |
|------|------|------------|
| 研究存档 | v4_deep_report_20260618.md | 只在需要回顾时读 |
| 投资建议追踪表 | investment-advice-tracker.md | 每日更新，按需查历史 |
| 实验数据 | v4_10year_validation.json | 分析时才需要 |
| Session存档 | session-archive-20260619.md | 跨session接续时读 |
| Skills分类 | skills-classification-20260618.md | 审计时才需要 |
| 进度记录 | 2026-06-18-progress-final.md | 历史参考 |

**判断标准：信息有价值但频率低 → 存文件，memory里只放一行指向它的路径**

## 三、Scripts（自动化执行）

**存什么：需要定期运行的代码**

| 类型 | 示例 | 位置 |
|------|------|------|
| Cron任务 | 数据更新、评分、健康检查 | scripts/ + cron jobs |
| 回测脚本 | blueshield_v4_*.py | scripts/us/ |
| 工具脚本 | check_data_fresh.py | scripts/utils/ |

## 四、Skills（可复用流程）

**存什么：反复执行的工作流**

| 类型 | 示例 |
|------|------|
| 模型训练流程 | quant-model-development |
| 回测评估流程 | ml-experimentation |
| Debug流程 | systematic-debugging |

## 五、黑板/记事板（短期任务传递 + Checklist）

**存什么：session之间需要传递的临时任务，以及任何需要跟踪的checklist项**

| 类型 | 示例 | 生命周期 |
|------|------|---------|
| 待办任务 | "下次session跑CatBoost WF验证" | 执行后删除 |
| 中断恢复 | "刚才研究到一半，继续XX" | 恢复后删除 |
| Checklist | "检查数据是否最新" "确认模型文件存在" | 完成后删除 |
| 临时笔记 | "发现XX问题，待调查" | 3天内处理 |

**文件**: `memory-archive/blackboard.md`

**写入规则**:
- 每条标明：来源session、时间、优先级(P0/P1/P2)、相关文件、验证标准
- Checklist项用 `- [ ]` / `- [x]` 格式
- 任务完成后移到"已完成"区，标注完成时间

**读取规则**:
- Session开始时**必须读黑板**
- 有任务就执行，有checklist就逐项检查
- 执行后**立即删除**该条（移到已完成区）
- 未完成的保留，超过3天自动过期

**与Memory的区别**:
- Memory = 长期事实（永久）
- 黑板 = 临时任务+checklist（用完即弃）

## 六、User Profile（用户画像）

**存什么：Andy的个人信息、角色、长期偏好**

与memory的区别：
- Memory = 我（Hermes）的工作笔记
- User Profile = Andy是什么人

## 七、决策流程图

```
新信息到来
  ├─ 需要传给下个session执行？ → BLACKBOARD
  ├─ 缺失会导致新session犯错？ → MEMORY
  ├─ 需要定期自动执行？ → SCRIPT + CRON
  ├─ 是可复用的工作流？ → SKILL
  ├─ 是用户个人信息？ → USER PROFILE
  ├─ 重要但低频？ → memory-archive/ + memory放路径
  └─ 临时/一次性？ → 不存，用完即弃
```

## 七、Memory条目格式规范

每条memory应遵循：
- **标题**：`## 类型+关键词 (日期)` — 如 `## 蓝盾V4方案 (2026-06-18)`
- **内容**：要点式，每行一个事实，不超过2层缩进
- **长度**：单条控制在200字符以内（除非不可压缩）
- **更新**：新事实覆盖旧事实（replace），不追加
- **删除**：过时信息立即remove，不保留"以防万一"

## 八、Session启动步骤（开始时必须执行）

每次新session开始时，按顺序执行：

### Step 1: 读黑板
- 读取 `memory-archive/blackboard.md`
- 有任务 → 执行
- 有checklist → 逐项检查

### Step 2: 检查数据新鲜度
```bash
cd ~/.hermes/openclaw-archive && python3 scripts/utils/check_data_fresh.py
```
- 数据过期 → 触发更新

### Step 3: 检查Cron状态
- 确认活跃cron job正常运行
- 有失败的 → 排查原因

### Step 4: 确认OpenD连接
```bash
# 测试连接
python3 -c "from futu import *; ctx=OpenSecTradeContext(filter_trdmarket=TrdMarket.US, host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES); ret,data=ctx.position_list_query(trd_env=TrdEnv.REAL); print('OK' if ret==RET_OK else data); ctx.close()"
```
- 连接失败 → 检查OpenD进程

### Step 5: 确认模型版本
- 问Andy当前用哪个版本（V4-LGB? CatBoost?）
- 不自作主张切换模型

## 九、Session存档步骤（结束前必须执行）

每次session结束前，执行以下检查并写入 `memory-archive/session-archive-YYYYMMDD-HHMM.md`：

### Step 1: 检查黑板
- 读取 `memory-archive/blackboard.md`
- 列出所有未完成的checklist项和任务
- 标注每项的当前状态（未开始/进行中/阻塞）

### Step 2: 记录当前session未完成事项
- 事无巨细描述清楚，包括：
  - 正在做什么（具体到函数/文件/步骤）
  - 做到哪一步了（已完成/进行中/未开始）
  - 遇到什么问题（错误信息/阻塞原因）
  - 相关文件位置（绝对路径）
  - 下一步该怎么做（具体操作）
  - 上下文信息（参数/配置/数据状态）

### Step 3: 记录关键决策
- 本session做了哪些决策
- 决策的理由
- Andy是否确认

### Step 4: 写入黑板
- 将需要下个session继续的任务写入黑板
- 标明优先级、来源、相关文件、验证标准

### Step 5: 更新Memory（如需要）
- 有新的持久事实 → 写入memory
- 有过时信息 → 从memory删除

## 十、当前存储分布

| 存储层 | 容量 | 当前用量 | 用途 |
|--------|------|---------|------|
| Memory | 5,000字符 | ~1,800 | 每轮注入的核心事实 |
| User Profile | 5,000字符 | ~1,400 | Andy个人信息 |
| 黑板 | 无限制 | 空 | Session间临时任务传递 |
| memory-archive/ | 无限制 | ~50KB | 历史存档文件 |
| Skills | ~5k tokens | 68个skills | 可复用流程 |
| Cron jobs | 无限制 | 6个活跃 | 定时任务 |
