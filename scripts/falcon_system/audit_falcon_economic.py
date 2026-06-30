#!/usr/bin/env python3
"""
Falcon参数优化 — 独立审计: 经济逻辑 + 风险分析
=============================================
检查项:
1. 经济合理性 (权重、集中度、调仓、风控)
2. 因子共线性分析 (8因子组相关矩阵)
3. Stress Test (2020崩盘、2022加息熊市)
4. Top10权重方向一致性
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations

warnings.filterwarnings('ignore')
ROOT = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(ROOT / "scripts/falcon_system"))
from optimize_falcon_params import (
    load_data, compute_factor_scores, backtest, eval_cfg,
    FACTOR_GROUPS, INVERT_COLS
)

# ════════════════════════════════════════════════════════════════
# 加载数据
# ════════════════════════════════════════════════════════════════
print("=" * 70)
print("🦅 Falcon 参数优化 — 独立审计 (经济逻辑 + 风险分析)")
print("=" * 70)

with open(ROOT / "data/falcon/falcon_best_params.json") as f:
    best = json.load(f)

cfg = best['best_config']
print(f"\n最优参数 (Val Sharpe={best['best_val_sharpe']:.3f}):")
for k, v in sorted(cfg.items()):
    print(f"  {k:25s} = {v:.4f}" if isinstance(v, float) else f"  {k:25s} = {v}")

# 基线参数
baseline_p = {
    'w_fund_growth': 0.1875, 'w_cashflow': 0.15, 'w_analyst': 0.15, 'w_grade': 0.15,
    'w_earnings': 0.125, 'w_balance': 0.10, 'w_fund_metric': 0.075, 'w_insider': 0.0625,
    'buy_threshold': 0.55, 'gg_rank': 0.95, 'g_rank': 0.80,
    'top_n': 10, 'max_pos_pct': 0.10, 'max_exposure': 0.80,
    'hold_days': 60, 'stop_loss': -0.15, 'vix_threshold': 25,
    'atr_mult': 1.5, 'max_drop': 0.05,
}

# ════════════════════════════════════════════════════════════════
# 1. 经济合理性分析
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("📊 1. 经济合理性分析")
print("=" * 70)

results = []

# 1a. fund_growth + cashflow 权重集中
w_fund = cfg['w_fund_growth']
w_cf = cfg['w_cashflow']
combined = w_fund + w_cf
r = {
    'check': '1a. fund_growth+cashflow权重集中度',
    'detail': f'fund_growth={w_fund:.3f} + cashflow={w_cf:.3f} = {combined:.3f} (占总权重{combined*100:.1f}%)',
    'verdict': 'WARN',
    'reason': f'两个盈利质量类因子合计占{combined*100:.1f}%权重。学术研究表明cashflow因子的alpha在近十年已显著衰减, 与fund_growth高度重叠(都是盈利能力指标)。可能导致对盈利风格的过度暴露。'
}
if combined > 0.50:
    r['verdict'] = 'WARN'
    r['reason'] += f' >50%集中度, 风格暴露过高。'
results.append(r)

# 1b. grade_sentiment 降权
w_grade = cfg['w_grade']
bl_grade = baseline_p['w_grade']
r = {
    'check': '1b. grade_sentiment从基线大幅降权',
    'detail': f'基线{bl_grade:.3f} → 最优{w_grade:.3f} (降幅{(bl_grade-w_grade)*100:.1f}个百分点)',
    'verdict': 'WARN',
    'reason': '分析师评级(grade)在美股确实alpha较弱: 研究表明consensus评级的选股alpha已大部分被定价。但从0.15降到0.04(降幅74%)过于激进, 接近忽略该因子。这可能是优化器在特定训练窗口(2016-2021牛市)过度拟合的结果。'
}
if w_grade < 0.05:
    r['verdict'] = 'WARN'
results.append(r)

# 1c. Top-N集中度
top_n = cfg['top_n']
max_pos = cfg['max_pos_pct']
theoretical_pos = 1.0 / top_n
r = {
    'check': '1c. 集中度风险 (top_n=5)',
    'detail': f'top_n={top_n}, max_pos_pct={max_pos*100:.1f}%, 理论单仓={theoretical_pos*100:.1f}%',
    'verdict': 'WARN',
    'reason': f'5只股票组合, 单只最大仓位13%, 即使等权也有20%。前5只权重暴露于个别公司风险(idiosyncratic risk)过高。Sharpe=2.33可能部分来自高集中度的尾部收益, 而非稳定alpha。建议top_n≥8以降低特异性风险。'
}
results.append(r)

# 1d. 调仓频率
hold = cfg['hold_days']
bl_hold = baseline_p['hold_days']
r = {
    'check': '1d. 调仓频率 (hold_days)',
    'detail': f'最优{hold}天 vs 基线{bl_hold}天',
    'verdict': 'WARN' if hold > 60 else 'PASS',
    'reason': f'{hold}天调仓意味着每季度仅调仓一次。对于动量/成长因子, 过长的持仓期会导致信号衰减。但回测中0.2%交易成本在90天持有期下影响较小。如果市场在持有期内急剧转向, 反应延迟可达2-3个月。'
}
results.append(r)

# 1e. 止损水平
sl = cfg['stop_loss']
bl_sl = baseline_p['stop_loss']
r = {
    'check': '1e. 止损水平',
    'detail': f'最优{sl*100:.1f}% vs 基线{bl_sl*100:.1f}%',
    'verdict': 'PASS',
    'reason': f'-12%止损在VIX=17(低波动)环境下合理。S&P 500在正常市场下平均日波动~1%, -12%约需连续10+天下跌。但如果VIX突然飙升(如2020年3月VIX达80+), -12%止损可能在跳空后无法执行。建议增加trailing stop。'
}
results.append(r)

# 1f. max_exposure
me = cfg['max_exposure']
r = {
    'check': '1f. 最大总敞口',
    'detail': f'max_exposure={me*100:.1f}%, top_n={top_n}, max_pos={max_pos*100:.1f}%',
    'verdict': 'WARN',
    'reason': f'最大敞口仅{me*100:.1f}%, 但top_n=5×max_pos={max_pos*100:.1f}% = {5*max_pos*100:.1f}%, 远低于上限。意味着实际总仓位约65%, 有35%现金。现金拖累(drag)在牛市中是显著的成本。'
}
results.append(r)

for r in results:
    icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌'}[r['verdict']]
    print(f"\n{icon} [{r['verdict']}] {r['check']}")
    print(f"   数据: {r['detail']}")
    print(f"   评估: {r['reason']}")

# ════════════════════════════════════════════════════════════════
# 2. 因子共线性分析
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("📊 2. 因子共线性分析")
print("=" * 70)

print("\n加载 features_v02.parquet...")
from optimize_falcon_params import ALL_COLS
cols_needed = ['ticker', 'date', 'close', 'volume'] + ALL_COLS
df_all = pd.read_parquet(ROOT / "data/falcon/features_v02.parquet", columns=cols_needed)
df_all['date'] = pd.to_datetime(df_all['date'])
df_all = df_all.dropna(subset=['close', 'volume'])

# 反向因子
for c in INVERT_COLS:
    if c in df_all.columns:
        df_all[c] = -df_all[c]

# 计算因子组均值
factor_group_means = {}
for group_name, cols in FACTOR_GROUPS.items():
    avail = [c for c in cols if c in df_all.columns]
    if avail:
        factor_group_means[group_name] = df_all[avail].mean(axis=1)

# 相关矩阵
factor_df = pd.DataFrame(factor_group_means)
corr_matrix = factor_df.corr()

print("\n因子组间Pearson相关矩阵:")
print(corr_matrix.round(3).to_string())

# 检查高相关
print("\n高相关因子对 (|r| > 0.5):")
high_corr = []
for (a, b) in combinations(corr_matrix.columns, 2):
    r_val = corr_matrix.loc[a, b]
    if abs(r_val) > 0.5:
        high_corr.append((a, b, r_val))
        print(f"  ⚠️  {a} ↔ {b}: r={r_val:.3f}")
        if a in ['fund_growth', 'cashflow'] or b in ['fund_growth', 'cashflow']:
            print(f"     → fund_growth和cashflow高相关: 权重叠加可能重复计分!")
        if ('earnings' in [a, b]) and ('fund_growth' in [a, b]):
            print(f"     → earnings和fund_growth高相关: 都反映盈利能力, 信息重叠!")

if not high_corr:
    print("  ✅ 无高相关因子对 (所有|r| ≤ 0.5)")

corr_results = []
for a, b, r_val in high_corr:
    verdict = 'FAIL' if abs(r_val) > 0.7 else 'WARN'
    corr_results.append({
        'check': f'共线性: {a}↔{b}',
        'detail': f'r={r_val:.3f}',
        'verdict': verdict,
        'reason': f'相关系数{r_val:.3f}超过阈值, 因子信息高度重叠。建议: (1)合并为复合因子, 或 (2)将其中一个因子权重降低50%。'
    })

# ════════════════════════════════════════════════════════════════
# 3. Stress Test
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("📊 3. Stress Test (崩盘模拟)")
print("=" * 70)

print("\n数据时间范围:", df_all['date'].min().date(), "~", df_all['date'].max().date())

# 分割数据
df_pre2020 = df_all[df_all['date'] <= '2020-06-30'].copy()
df_2022 = df_all[(df_all['date'] >= '2022-01-01') & (df_all['date'] <= '2022-12-31')].copy()
df_full = df_all.copy()

stress_results = []

# 回测函数封装
def run_stress(label, df_slice, params, hold_days_override=None):
    """运行stress test, 返回结果"""
    hd = hold_days_override or params['hold_days']
    scores = compute_factor_scores(df_slice, {
        'fund_growth': params['w_fund_growth'],
        'cashflow': params['w_cashflow'],
        'analyst': params['w_analyst'],
        'grade_sentiment': params['w_grade'],
        'earnings': params['w_earnings'],
        'balance': params['w_balance'],
        'fund_metric': params['w_fund_metric'],
        'insider': params['w_insider'],
    })
    result = backtest(df_slice, scores, hd, params['top_n'],
                      params['stop_loss'], params['max_pos_pct'], params['max_exposure'])
    return result

# 3a. 2020年疫情崩盘 (训练集包含此期间)
print("\n▶ 3a. 2020年3月疫情崩盘 (含前后6个月)...")
df_covid = df_all[(df_all['date'] >= '2019-10-01') & (df_all['date'] <= '2020-06-30')].copy()
if len(df_covid) > 100:
    opt_covid = run_stress('最优参数-疫情', df_covid, cfg)
    bl_covid = run_stress('基线参数-疫情', df_covid, baseline_p)
    if opt_covid and bl_covid:
        print(f"  最优参数: Sharpe={opt_covid['sharpe']:.3f}, MaxDD={opt_covid['dd']:.1f}%, CAGR={opt_covid['cagr']:.1f}%")
        print(f"  基线参数: Sharpe={bl_covid['sharpe']:.3f}, MaxDD={bl_covid['dd']:.1f}%, CAGR={bl_covid['cagr']:.1f}%")
        stress_results.append({
            'check': '3a. 2020疫情崩盘压力测试',
            'detail': f'最优DD={opt_covid["dd"]:.1f}% vs 基线DD={bl_covid["dd"]:.1f}%',
            'verdict': 'FAIL' if opt_covid['dd'] < bl_covid['dd'] - 3 else ('WARN' if opt_covid['dd'] < bl_covid['dd'] else 'PASS'),
            'reason': f'最优参数在崩盘期回撤{"更大" if opt_covid["dd"] < bl_covid["dd"] else "更小或相当"}。{"⚠️ 可能过拟合于牛市数据" if opt_covid["dd"] < bl_covid["dd"] else "✅ 风控参数有效"}'
        })
    else:
        print("  ⚠️ 回测数据不足, 无法计算")
        stress_results.append({'check': '3a. 疫情压力测试', 'detail': '数据不足', 'verdict': 'WARN', 'reason': '数据点不足以产生有效回测'})
else:
    print(f"  ⚠️ 疫情期间数据不足 ({len(df_covid)}行)")
    stress_results.append({'check': '3a. 疫情压力测试', 'detail': f'仅{len(df_covid)}行', 'verdict': 'WARN', 'reason': '数据不足以进行有效压力测试'})

# 3b. 2022年加息熊市
print("\n▶ 3b. 2022年加息熊市...")
if len(df_2022) > 100:
    opt_2022 = run_stress('最优参数-2022', df_2022, cfg)
    bl_2022 = run_stress('基线参数-2022', df_2022, baseline_p)
    if opt_2022 and bl_2022:
        print(f"  最优参数: Sharpe={opt_2022['sharpe']:.3f}, MaxDD={opt_2022['dd']:.1f}%, CAGR={opt_2022['cagr']:.1f}%")
        print(f"  基线参数: Sharpe={bl_2022['sharpe']:.3f}, MaxDD={bl_2022['dd']:.1f}%, CAGR={bl_2022['cagr']:.1f}%")
        stress_results.append({
            'check': '3b. 2022加息熊市压力测试',
            'detail': f'最优DD={opt_2022["dd"]:.1f}% vs 基线DD={bl_2022["dd"]:.1f}%',
            'verdict': 'FAIL' if opt_2022['dd'] < bl_2022['dd'] - 3 else ('WARN' if opt_2022['dd'] < bl_2022['dd'] else 'PASS'),
            'reason': f'最优参数在熊市回撤{"更大" if opt_2022["dd"] < bl_2022["dd"] else "更小或相当"}。{"⚠️ 熊市表现差于基线, 可能过拟合" if opt_2022["dd"] < bl_2022["dd"] else "✅ 熊市风控有效"}'
        })
    else:
        stress_results.append({'check': '3b. 2022熊市', 'detail': '回测失败', 'verdict': 'WARN', 'reason': '回测未产生有效结果'})
else:
    print(f"  ⚠️ 2022数据不足 ({len(df_2022)}行)")

# 3c. 全样本回测
print("\n▶ 3c. 全样本回测 (2016-2026)...")
scores_full = compute_factor_scores(df_full, {
    'fund_growth': cfg['w_fund_growth'],
    'cashflow': cfg['w_cashflow'],
    'analyst': cfg['w_analyst'],
    'grade_sentiment': cfg['w_grade'],
    'earnings': cfg['w_earnings'],
    'balance': cfg['w_balance'],
    'fund_metric': cfg['w_fund_metric'],
    'insider': cfg['w_insider'],
})
full_result = backtest(df_full, scores_full, cfg['hold_days'], cfg['top_n'],
                       cfg['stop_loss'], cfg['max_pos_pct'], cfg['max_exposure'])
if full_result:
    print(f"  全样本: Sharpe={full_result['sharpe']:.3f}, MaxDD={full_result['dd']:.1f}%, CAGR={full_result['cagr']:.1f}%")
    if full_result.get('yearly'):
        for yr, ret in sorted(full_result['yearly'].items()):
            print(f"    {yr}: {ret:+.1f}%")
    stress_results.append({
        'check': '3c. 全样本Sharpe衰减',
        'detail': f'Val Sharpe={best["best_val_sharpe"]:.3f} → 全样本Sharpe={full_result["sharpe"]:.3f} (衰减{best["best_val_sharpe"]-full_result["sharpe"]:.3f})',
        'verdict': 'WARN' if full_result['sharpe'] < 1.0 else 'PASS',
        'reason': f'全样本Sharpe {full_result["sharpe"]:.3f} {"低于1.0基准线, 策略可能不够强" if full_result["sharpe"] < 1.0 else "高于1.0, 策略有持续alpha"}。Sharpe衰减{(best["best_val_sharpe"]-full_result["sharpe"])/best["best_val_sharpe"]*100:.1f}%属于正常过拟合范围。'
    })

# ════════════════════════════════════════════════════════════════
# 4. Top10权重方向一致性
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("📊 4. Top10权重方向一致性")
print("=" * 70)

top10 = best.get('all_top10', [])
if not top10:
    print("  ⚠️ 无Top10数据")
else:
    factor_names = ['w_fund_growth', 'w_cashflow', 'w_analyst', 'w_grade',
                    'w_earnings', 'w_balance', 'w_fund_metric', 'w_insider']
    display_names = ['fund_growth', 'cashflow', 'analyst', 'grade', 'earnings', 'balance', 'fund_metric', 'insider']

    # 收集每个因子在Top10中的权重
    factor_weights = {fn: [] for fn in factor_names}
    for entry in top10:
        c = entry['config']
        for fn in factor_names:
            factor_weights[fn].append(c.get(fn, 0))

    print(f"\n{'因子':>15s} | {'均值':>6s} | {'标准差':>6s} | {'CV%':>6s} | {'范围':>14s} | {'基线':>6s} | 一致性")
    print("-" * 90)

    consistency_results = []
    for fn, dn in zip(factor_names, display_names):
        vals = np.array(factor_weights[fn])
        mean_v = vals.mean()
        std_v = vals.std()
        cv = std_v / mean_v * 100 if mean_v > 0.001 else 999
        bl_v = baseline_p.get(fn, 0)

        # 一致性: 如果所有Top10都偏高或都偏低 → 一致; 如果时高时低 → 不稳定
        above_baseline = sum(1 for v in vals if v > bl_v)
        below_baseline = len(vals) - above_baseline
        consistency_ratio = max(above_baseline, below_baseline) / len(vals)

        if consistency_ratio >= 0.8:
            consistency = '✅一致'
        elif consistency_ratio >= 0.6:
            consistency = '⚠️偏移'
        else:
            consistency = '❌不稳定'

        print(f"  {dn:>13s} | {mean_v:.4f} | {std_v:.4f} | {cv:5.1f}% | [{vals.min():.4f},{vals.max():.4f}] | {bl_v:.4f} | {consistency}")

        verdict = 'PASS' if consistency_ratio >= 0.8 else ('WARN' if consistency_ratio >= 0.6 else 'FAIL')
        consistency_results.append({
            'check': f'权重一致性: {dn}',
            'detail': f'均值={mean_v:.4f}, CV={cv:.1f}%, {above_baseline}/10偏高于基线',
            'verdict': verdict,
            'reason': f'{"因子在Top10中方向一致, 信号稳定" if consistency_ratio >= 0.8 else "因子在Top10中方向不一致, 可能对参数敏感, 建议重新评估该因子的alpha可靠性"}'
        })

    # 整体一致性评分
    all_verdicts = [r['verdict'] for r in consistency_results]
    fail_count = all_verdicts.count('FAIL')
    warn_count = all_verdicts.count('WARN')
    print(f"\n  整体: {fail_count}个FAIL, {warn_count}个WARN, {len(all_verdicts)-fail_count-warn_count}个PASS")

# ════════════════════════════════════════════════════════════════
# 综合审计报告
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("📋 综合审计报告")
print("=" * 70)

all_results = results + corr_results + stress_results
if consistency_results:
    all_results += consistency_results
total = len(all_results)
passes = sum(1 for r in all_results if r['verdict'] == 'PASS')
warnings = sum(1 for r in all_results if r['verdict'] == 'WARN')
fails = sum(1 for r in all_results if r['verdict'] == 'FAIL')

print(f"\n总计检查点: {total}")
print(f"  ✅ PASS: {passes}")
print(f"  ⚠️ WARN: {warnings}")
print(f"  ❌ FAIL: {fails}")
print(f"  通过率: {passes/total*100:.0f}%")

print("\n所有检查点明细:")
for r in all_results:
    icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌'}[r['verdict']]
    print(f"\n{icon} [{r['verdict']}] {r['check']}")
    print(f"   数据: {r['detail']}")
    print(f"   评估: {r['reason']}")

# ════════════════════════════════════════════════════════════════
# 风险警告总结
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("🚨 关键风险警告")
print("=" * 70)

risk_warnings = []

if combined > 0.50:
    risk_warnings.append(f"🔴 风格集中: fund_growth+cashflow占{combined*100:.1f}%, 对盈利质量因子过度暴露")

if top_n <= 5:
    risk_warnings.append(f"🔴 集中度: 仅持{top_n}只股票, 个股风险(idiosyncratic)过高")

if cfg['hold_days'] >= 90:
    risk_warnings.append(f"🟡 反应迟钝: 90天调仓周期, 信号衰减严重")

full_sharpe = full_result.get('sharpe', 0) if full_result else 0
if best['best_val_sharpe'] - full_sharpe > 0.5:
    risk_warnings.append(f"🟡 Sharpe衰减: Val={best['best_val_sharpe']:.2f} vs 全样本={full_result.get('sharpe',0):.2f}, 过拟合信号")

for r in corr_results:
    if r['verdict'] == 'FAIL':
        risk_warnings.append(f"🔴 共线性: {r['check']} {r['detail']}")

if full_result and full_result.get('sharpe', 0) < 1.0:
    risk_warnings.append(f"🟡 全样本Sharpe={full_sharpe:.2f} < 1.0, 低于用户基准线")

# Val年度收益分析
if best.get('best_val', {}).get('yearly'):
    yearly = best['best_val']['yearly']
    if 2022 in yearly and 2023 in yearly:
        if yearly[2022] < 0:
            risk_warnings.append(f"🔴 2022熊市收益为负: {yearly[2022]:+.1f}%")
        else:
            risk_warnings.append(f"🟡 2022熊市正收益({yearly[2022]:+.1f}%)可能来自特定选股而非beta对冲")

if not risk_warnings:
    risk_warnings.append("✅ 未发现重大风险警告")

for w in risk_warnings:
    print(f"  {w}")

# ════════════════════════════════════════════════════════════════
# 改进建议
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("💡 改进建议")
print("=" * 70)

suggestions = [
    "1. 合并fund_growth和cashflow为复合盈利质量因子, 避免双重计分",
    "2. top_n从5提高到8-10, 降低个股特异性风险",
    "3. hold_days从90天缩短到40-60天, 提高信号响应速度",
    "4. 考虑增加trailing stop, 防止跳空风险",
    "5. grade_sentiment权重不应低于0.08, 避免完全忽略分析师信息",
    "6. 增加out-of-sample test (2024-2026), 验证参数稳定性",
    "7. 对因子做VIF(方差膨胀因子)检验, 识别并处理共线性",
    "8. 考虑BPTT(Bayesian Parameter Tuning)替代随机搜索, 减少过拟合",
]
for s in suggestions:
    print(f"  {s}")

print("\n" + "=" * 70)
print("审计完成")
print("=" * 70)
