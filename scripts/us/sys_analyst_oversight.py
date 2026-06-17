#!/usr/bin/env python3
"""
🍤 Analyst Oversight — 强制监督流程
每次出投资建议前必须调这个脚本
不通过 = 不准出推荐
"""
import os, json, sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OVERRIDE_FILE = os.path.join(ROOT, 'data', 'analyst_override.json')

CHECKLIST = [
    'evidence_checked',     # 检查证据
    'risk_checked',         # 检查风险
    'price_position_checked', # 检查价格位置
    'alternative_checked',  # 检查替代方案
]

class OversightBlockedError(Exception):
    pass

def check_holding(code, name, cost, price, score, rank, ma10, chg1m, chg1d):
    """对单只持仓或候选股执行完整监督"""
    result = {
        'code': code,
        'name': name,
        'timestamp': datetime.now().isoformat(),
        'checks': {},
        'verdict': 'PASS',
        'fail_reasons': []
    }
    
    # 1. 检查证据
    result['checks']['evidence'] = {
        'score': score,
        'rank': rank,
        'score_adequate': score >= 30 if rank else True,  # 美股30分线
    }
    
    # 2. 检查风险
    pnl = (price / cost - 1) * 100 if cost > 0 else 0
    result['checks']['risk'] = {
        'pnl_pct': round(pnl, 1),
        'capital_at_risk': price * 1,  # placeholder
    }
    
    # 3. 检查价格位置
    if ma10 and ma10 > 0:
        dist_to_ma10 = (price / ma10 - 1) * 100
    else:
        dist_to_ma10 = 0
    result['checks']['price_position'] = {
        'price': price,
        'ma10': ma10,
        'distance_to_ma10': round(dist_to_ma10, 1),
        'too_extended': dist_to_ma10 > 15,  # 高于MA10超15% = 追高
    }
    
    # 4. 检查替代方案（简化版）
    result['checks']['alternative'] = {
        'scored': score > 0,
    }
    
    # 综合判断
    fail = []
    if result['checks']['price_position']['too_extended']:
        fail.append(f'🚫 股价高于MA10 {dist_to_ma10:.0f}%，追高风险')
    if result['checks']['risk']['pnl_pct'] < -8:
        fail.append(f'🚫 已亏损{pnl:.0f}%，触及止损')
    
    if fail:
        result['verdict'] = 'BLOCKED'
        result['fail_reasons'] = fail
    
    return result

def must_pass(holding_checks):
    """强制闸门：不通过就抛异常"""
    for h in holding_checks:
        if h['verdict'] == 'BLOCKED':
            msg = f"🚨 Oversight BLOCKED: {h['name']}({h['code']})\n"
            msg += '\n'.join(h['fail_reasons'])
            raise OversightBlockedError(msg)
    return True

def format_report(holding_checks):
    """输出可读报告"""
    lines = ['🔍 Analyst Oversight Report', f'时间: {datetime.now().strftime("%H:%M")}', '']
    
    for h in holding_checks:
        lines.append(f'{"─"*40}')
        lines.append(f'{h["name"]} ({h["code"]})')
        lines.append(f'  评分: {h["checks"]["evidence"]["score"]} | 盈亏: {h["checks"]["risk"]["pnl_pct"]:+.1f}%')
        lines.append(f'  距MA10: {h["checks"]["price_position"]["distance_to_ma10"]:+.1f}%')
        verdict = '✅ PASS' if h['verdict'] == 'PASS' else '❌ BLOCKED'
        lines.append(f'  判定: {verdict}')
        if h['fail_reasons']:
            for r in h['fail_reasons']:
                lines.append(f'    {r}')
    
    return '\n'.join(lines)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', nargs=5, metavar=('CODE','NAME','COST','PRICE','SCORE'))
    args = parser.parse_args()
    
    if args.check:
        code, name, cost, price, score = args.check
        r = check_holding(code, name, float(cost), float(price), float(score), None, 0, 0, 0)
        print(json.dumps(r, indent=2, ensure_ascii=False))
