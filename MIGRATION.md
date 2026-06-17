# D盘迁移方案

> 风险评估 + 操作步骤

## ⚠️ 风险

上次迁移到D盘失败的原因：
1. **NTFS权限问题** — Linux文件权限在NTFS上不支持，venv会坏
2. **Git在NTFS上慢且易出错** — 文件锁、大小写不敏感
3. **符号链接不支持** — WSL2的symlink在NTFS上会断

## 方案A：vhdx迁移到D盘（推荐）

**原理：** 把整个WSL虚拟磁盘文件从C盘移到D盘，Linux文件系统不变。

```powershell
# Windows PowerShell (管理员)
# 1. 关闭WSL
wsl --shutdown

# 2. 移动vhdx到D盘
move "$env:LOCALAPPDATA\Packages\CanonicalGroupLimited.Ubuntu_*\LocalState\ext4.vhdx" "D:\WSL\ext4.vhdx"

# 3. 在.wslconfig中指定新位置
# 创建 %USERPROFILE%\.wslconfig
wsl --shutdown
# 编辑 C:\Users\admin\.wslconfig
```

`.wslconfig` 内容：
```ini
[wsl2]
vmMemory=8GB
vmProcessors=4
# 不直接指定vhdx路径，需要通过Hyper-V管理器
```

**优点：** 零风险，Linux系统完全不变
**缺点：** 需要Hyper-V管理器或PowerShell操作

## 方案B：符号链接（数据目录）

**原理：** 保持vhdx在C盘，但把大数据目录软链到D盘。

```bash
# WSL终端
# 1. 在D盘创建数据目录
mkdir -p /mnt/d/hermes_data

# 2. 移动大数据到D盘
mv ~/.hermes/openclaw-archive/data/cn /mnt/d/hermes_data/cn
mv ~/.hermes/openclaw-archive/data/us /mnt/d/hermes_data/us

# 3. 创建符号链接
ln -s /mnt/d/hermes_data/cn ~/.hermes/openclaw-archive/data/cn
ln -s /mnt/d/hermes_data/us ~/.hermes/openclaw-archive/data/us
```

**优点：** 立竿见影，立即释放C盘8GB
**缺点：** 跨文件系统符号链接可能有性能问题，NTFS上的文件权限受限

## 方案C：D盘只存原始数据

**原理：** D盘保持原始OpenClaw数据不动，Linux本地只存模型+脚本+配置。

```bash
# 脚本中引用D盘原始数据
D_RAW = '/mnt/d/openclaw/data'  # 只读
LOCAL = '/home/hermes/.hermes/openclaw-archive'  # 读写
```

**优点：** 最安全，零迁移
**缺点：** D盘数据不更新，需要定期从D盘同步到本地

## 推荐

**先用方案A**（vhdx迁移），最安全。需要你在Windows PowerShell里操作。

## 操作步骤

### 方案A详细步骤

1. **关闭WSL**
```powershell
wsl --shutdown
```

2. **创建D盘目录**
```powershell
mkdir D:\WSL
```

3. **移动vhdx文件**
```powershell
# 找到vhdx位置
$vhdxPath = Get-ChildItem "$env:LOCALAPPDATA\Packages\CanonicalGroupLimited.Ubuntu_*\LocalState\ext4.vhdx" | Select -First 1 -ExpandProperty FullName
# 复制到D盘（先复制，确认无误后删除原文件）
Copy-Item $vhdxPath "D:\WSL\ext4.vhdx"
```

4. **配置WSL使用新位置**
```powershell
# 编辑 %USERPROFILE%\.wslconfig
notepad "$env:USERPROFILE\.wslconfig"
```

添加：
```ini
[wsl2]
vmMemory=8GB
```

5. **重启WSL验证**
```powershell
wsl -d Ubuntu
# 验证数据完整
ls ~/.hermes/openclaw-archive/
```

6. **确认无误后删除旧vhdx**
```powershell
Remove-Item $vhdxPath
```

## 备份清单

| 内容 | 备份方式 | 位置 |
|:--|:--|:--|
| 代码/脚本/配置 | Git → GitHub | 远程仓库 |
| 模型文件 | Git + 本地副本 | models/ |
| K线数据 | 从tushare重建 | 需30-40分钟 |
| 资金流数据 | 从tushare重建 | 需30-40分钟 |
| 美股特征数据 | 从D盘原始复制 | 需5分钟 |
