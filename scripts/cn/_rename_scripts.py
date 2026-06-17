"""
scripts目录重新命名脚本
按项目分类，统一命名规范
"""
import os, shutil

sp = r'/home/hermes/.hermes/openclaw-archive\scripts'
all_files = sorted([f for f in os.listdir(sp) if f.endswith('.py')])

# 分类映射：原名 → 新名 or None(保持原名) or 'DELETE'（可删）
# 规则：
#   正式: 项目名_功能名.py
#   测试: tst_项目名_功能名.py
#   临时: tmp_功能名.py（用完即删）
#   内部: _工具名.py

mapping = {}

# ===== 1. A1资金流模型 =====
for f in all_files:
    if f == 'a1_daily_report.py':
        mapping[f] = None  # 已符合规范
    elif f.startswith('bt_a1_'):
        # bt_a1_breakthrough, bt_a1_explore, bt_a1_explore2
        mapping[f] = f.replace('bt_', 'tst_a1_')
    elif f == 'build_quality_pool.py':
        mapping[f] = 'a1_build_quality_pool.py'

# ===== 2. 美股V5三模型 =====
for f in all_files:
    if f in ['us_v5s_backtest.py', 'us_s1_scan.py']:
        mapping[f] = None  # 已符合
    elif f == 'score_engine.py':
        mapping[f] = 'us_score_engine.py'
    elif f == 'scoring.py':
        mapping[f] = 'us_scoring.py'
    elif f == 's2_candidates.py':
        mapping[f] = 'us_s2_candidates.py'
    elif f == 'layer1_daily.py':
        mapping[f] = 'us_layer1_daily.py'
    elif f == 'layer1_market_state.py':
        mapping[f] = 'us_layer1_market_state.py'
    elif f == 'layer3_4_scoring.py':
        mapping[f] = 'us_layer3_4_scoring.py'
    elif f == 'multi_factor_optimize.py':
        mapping[f] = 'us_multi_factor_optimize.py'
    elif f == 'precompute_factors.py':
        mapping[f] = 'us_precompute_factors.py'
    elif f == 'regime_weights.py':
        mapping[f] = 'us_regime_weights.py'

# ===== 3. 今日现场写的（临时/测试）=====
today_temp = [
    'run_dynamic_wgt.py', 'run_market_protect.py',
    'test_dynamic_wgt.py', 'test_small_bt.py',
    'verify_backtest.py', 'diff_check.py',
    'bt_compare_wgt.py', 'test_dynamic_wgt.py',
    'tmp_append_exp.py',
]

# run_* 自动归类
if 'run_dynamic_wgt.py' in all_files:
    mapping['run_dynamic_wgt.py'] = 'tst_us_wgt_dynamic.py'
if 'run_market_protect.py' in all_files:
    mapping['run_market_protect.py'] = 'tst_us_market_protect.py'
if 'test_dynamic_wgt.py' in all_files:
    mapping['test_dynamic_wgt.py'] = 'tst_us_wgt_small.py'
if 'test_small_bt.py' in all_files:
    mapping['test_small_bt.py'] = 'tst_us_small_bt.py'
if 'verify_backtest.py' in all_files:
    mapping['verify_backtest.py'] = 'tst_us_verify_baseline.py'
if 'diff_check.py' in all_files:
    mapping['diff_check.py'] = 'tst_us_diff_check.py'
if 'bt_compare_wgt.py' in all_files:
    mapping['bt_compare_wgt.py'] = 'tst_us_wgt_compare.py'
if 'tmp_append_exp.py' in all_files:
    mapping['tmp_append_exp.py'] = 'tst_tmp_append_exp.py'  # 临时

# ===== 4. 回测文件（bt_*）=====
bt_remap = {
    'bt_12_15_clean.py': 'tst_sl_12_15_clean.py',
    'bt_12_15_final.py': 'tst_sl_12_15_final.py',
    'bt_a2.py': 'tst_a2.py',
    'bt_audit.py': 'tst_audit.py',
    'bt_audit2.py': 'tst_audit2.py',
    'bt_beta.py': 'tst_beta.py',
    'bt_cost_shock.py': 'tst_cost_shock.py',
    'bt_d3_v2.py': 'tst_d3_v2.py',
    'bt_dd_check.py': 'tst_dd_check.py',
    'bt_final.py': 'tst_final.py',
    'bt_full_strategy.py': 'tst_full_strategy.py',
    'bt_metrics_final.py': 'tst_metrics_final.py',
    'bt_real_benchmark.py': 'tst_real_benchmark.py',
    'bt_realistic.py': 'tst_realistic.py',
    'bt_risk_validation.py': 'tst_risk_validation.py',
    'bt_s2.py': 'tst_s2.py',
    'bt_s2_v2.py': 'tst_s2_v2.py',
    'bt_sweep.py': 'tst_sweep.py',
    'bt_t1_fix.py': 'tst_t1_fix.py',
    'bt_top3_vs_top5.py': 'tst_top3_vs_top5.py',
    'bt_verify_39pct.py': 'tst_verify_39pct.py',
    'bt_walkforward.py': 'tst_walkforward.py',
}
for f, new in bt_remap.items():
    if f in all_files:
        mapping[f] = new

# ===== 5. 已归档的内部工具（_开头，不动）=====
# _check_dd.py, _compare_v5.py 等已经符合规范

# ===== 6. 数据下载 =====
for f in all_files:
    if f.startswith('dl_'):
        mapping[f] = f  # dl_已符合规范

# ===== 7. 日常事务 =====
daily_map = {
    'archive_daily.py': 'daily_archive.py',
    'backup_daily.py': 'daily_backup.py',
    'run_morning.py': 'daily_morning.py',
    'run_zhengli.py': 'daily_zhengli.py',
    'gen_morning_summary.py': 'daily_gen_morning_summary.py',
    'check_session_count.py': 'daily_session_count.py',
    'save_today_recommend.py': 'daily_save_recommend.py',
    'gen_rolling_memory.py': 'daily_rolling_memory.py',
    'gen_new_files_flag.py': 'daily_gen_flag.py',
    'seed_memory.py': 'daily_seed_memory.py',
    'extract_memories.py': 'daily_extract_memories.py',
    'dream_post_process.py': 'daily_dream_post.py',
    'startup_gate.py': 'daily_startup_gate.py',
}
for f, new in daily_map.items():
    if f in all_files:
        mapping[f] = new

# ===== 8. 系统维护/工具 =====
sys_map = {
    'safe_guard.py': 'sys_safe_guard.py',
    'gateway_anticrash.py': 'sys_anticrash.py',
    'session_watchdog.py': 'sys_watchdog.py',
    'log_tracker.py': 'sys_log_tracker.py',
    'audit_independence.py': 'sys_audit.py',
    'provider_health_check.py': 'sys_provider_health.py',
    'pull_cloud_feedback.py': 'sys_cloud_feedback.py',
    'config_keys.py': 'sys_config_keys.py',
    'dual_backup.py': 'sys_dual_backup.py',
    'trader.py': 'sys_trader.py',
}
for f, new in sys_map.items():
    if f in all_files:
        mapping[f] = new

# ===== 9. A股个股分析 =====
if 'portfolio_analyzer.py' in all_files:
    mapping['portfolio_analyzer.py'] = 'a_portfolio_analyzer.py'

# ===== 10. 早先的临时调试文件等 =====
# 测试文件 - 保持tst_前缀
# 没有匹配到的？
for f in all_files:
    if f not in mapping:
        mapping[f] = f  # 保持原名

# ===== 执行重命名 =====
renamed = []
skipped = []
errors = []
for old, new in sorted(mapping.items()):
    if new is None or old == new:
        skipped.append(old)
        continue
    oldp = os.path.join(sp, old)
    newp = os.path.join(sp, new)
    if not os.path.exists(oldp):
        errors.append(f'源文件不存在: {old}')
        continue
    if os.path.exists(newp) and old != new:
        errors.append(f'目标已存在: {new} (从{old})')
        continue
    os.rename(oldp, newp)
    renamed.append((old, new))

print('=== 重命名完成 ===')
print(f'重命名: {len(renamed)}个')
for o, n in renamed:
    print(f'  {o:40s} → {n}')
print(f'\n保持原名: {len(skipped)}个')
print(f'错误: {len(errors)}个')
for e in errors:
    print(f'  ❌ {e}')

# 打印最终列表
print(f'\n=== 最终 scripts/ 目录 ===')
final = sorted([f for f in os.listdir(sp) if f.endswith('.py')])
for f in final: print(f'  {f}')
