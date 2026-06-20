# Blackboard — 任务传递

## 当前状态
- ✅ 蓝盾V6: 生产中（模型+评分+cron全部就位）
- ✅ 绿箭V11: 生产中（模型+评分+cron全部就位）
- ✅ 看板系统: 已部署（HTML + Cloudflare隧道）

## 已完成
- [x] V6评分脚本修复（全市场扫描）
- [x] V11生产模型训练
- [x] VIX止损逻辑实装
- [x] 自动重训cron（1月/7月）
- [x] 宏观数据更新cron（每天05:00）
- [x] 综合看板HTML生成
- [x] Cloudflare隧道配置
- [x] 每日看板更新cron

## 待办
- [ ] 生产验证（等周一开盘测试评分脚本）
- [ ] 长期隧道稳定性测试

## Cron清单
| 任务 | 时间 | Job ID |
|------|------|--------|
| 蓝盾V6评分 | 04:30 | fb1723fce4f6 |
| 绿箭V11评分 | 04:30 | 419b588b962b |
| 数据更新 | 05:00 | 2be6ea453d71 |
| 看板更新 | 05:00 | 999c99516eab |
| GitHub备份 | 03:00 | — |
| 记忆压缩 | 03:00 | — |
| 市场学习 | 02:00 | — |
| 自动重训 | 1月/7月 | d4481da86200 |

## 关键文件
- 看板: `dashboard.html`
- 生成器: `scripts/us/generate_dashboard.py`
- 启动脚本: `scripts/us/start_dashboard.sh`
- Skill: `dashboard-visual-inspector`
