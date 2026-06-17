#!/usr/bin/env python3
"""
🍤 报告合规检查器 — 从配置读规则，不改代码也能调

检查项来自:
  config/output_templates.json → required_sections
  config/strategy.json → scoring thresholds
"""
import sys, json, re, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from notify import send

# ===== 报告路径 =====
def _get_report_path(report_name='morning_report.txt'):
    return os.path.join(ROOT, 'data', report_name)

# ===== 加载配置 =====
def _load_config():
    with open(os.path.join(ROOT, 'config', 'output_templates.json')) as f:
        return json.load(f)

def _load_strategy():
    with open(os.path.join(ROOT, 'config', 'strategy.json')) as f:
        return json.load(f)

# ===== 通用校验引擎 =====
def check_report(filepath=None):
    if filepath is None:
        filepath = _get_report_path()
    
    try:
        with open(filepath) as f:
            text = f.read()
    except FileNotFoundError:
        return {"passed": False, "failures": ["❌ 报告文件不存在"], "details": []}
    
    config = _load_config()
    strategy = _load_strategy()
    failures = []
    details = []
    lines = text.split('\n')
    full_text = text
    
    # --- 检查1: 必需章节是否存在（从模板配置读取）---
    required = config.get('required_sections', [])
    # 为每个章节构造正则做模糊匹配
    for section in required:
        # 去掉emoji，只保留关键词做匹配
        keywords = re.sub(r'[^\w\u4e00-\u9fff]', '', section)
        if len(keywords) < 2:
            keywords = section
        # 分段匹配
        found = False
        for keyword in re.findall(r'[\u4e00-\u9fff\w]+', section):
            if keyword in full_text:
                found = True
                break
        if not found:
            failures.append(f"❌ 缺少章节: {section}")
            details.append(f"  {section} — 未在报告中发现")
    
    # --- 检查2: 红绿灯emoji使用是否正确 ---
    emoji_rules = config.get('emoji_rules', {})
    buyable_count = len(re.findall(r'🟢', full_text))
    watch_count = len(re.findall(r'🟡', full_text))
    danger_count = len(re.findall(r'🔴', full_text))
    
    # 如果没有买入信号但连一个🔴都没有，可能没做判断
    if '卖出' in full_text and danger_count == 0:
        failures.append("⚠️ 说了卖出但没用🔴标记")
        details.append("  建议使用🔴标记卖出信号")
    
    # --- 检查3: 每个推荐股票有没有判断理由 ---
    in_recommend_section = False
    for i, line in enumerate(lines):
        # 检测进入推荐区域
        if any(kw in line for kw in ['推荐', '红绿灯', '候选', '关注', '过线']):
            in_recommend_section = True
            continue
        # 检测离开推荐区域
        if in_recommend_section and ('📦' in line or '🎯' in line or '👀' in line or '⏰' in line):
            in_recommend_section = False
            continue
        
        if in_recommend_section and line.strip():
            # 这是一行推荐内容吗？（包含股票名+代码或评分）
            is_stock_line = bool(re.search(r'[\u4e00-\u9fff].*\d{6}|#\d|评分|¥', line))
            if is_stock_line:
                # 检查下一行是不是判断（以→或🧠结尾，或包含判断句）
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
                has_judgment = bool(re.search(r'→|🧠|但|不过|风险|机会|趋势|建议|等|别', next_line))
                is_last = (i + 1 >= len(lines) or not next_line or '📦' in next_line or '🎯' in next_line)
                if not has_judgment and not is_last:
                    failures.append(f"❌ {line.strip()[:40]}...后面没有判断")
                    details.append(f"  第{i+1}行: {line.strip()[:30]} → 缺理由")
    
    # --- 检查4: 扫描规模标注（防止只扫前100就出报告）---
    scan_scope_lines = [l for l in lines if '质量池' in l or '扫描' in l or '只/' in l or '总扫描' in l]
    has_scan_scope = len(scan_scope_lines) > 0
    if not has_scan_scope:
        failures.append('⚠️ 报告未标注扫描范围（应含"从X只中筛选"信息）')
        details.append('  建议在报告底部添加数据源标注，说明扫描了多少只股票')
    else:
        for sl in scan_scope_lines:
            nums = re.findall(r'(\d+)', sl)
            if nums and int(max(nums, key=int)) < 500:
                failures.append(f'⚠️ 扫描范围偏小：{sl.strip()[:60]}')
                details.append(f'  扫描目标可能不足500只，建议扩大范围')
    
    # --- 检查5: 阈值一致性（读strategy.json）---
    a_cfg = strategy.get('a_stock', {})
    buy_t = a_cfg.get('buy_threshold', 62)
    sell_t = a_cfg.get('sell_threshold', 50)
    
    # 检查报告里是否出现了硬编码的阈值数字
    if str(buy_t) in full_text:
        # 只是确认有引用阈值，不做失败
        pass
    
    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "details": details,
        "line_count": len(lines),
        "stats": {
            "🟢": buyable_count,
            "🟡": watch_count,
            "🔴": danger_count
        },
        "preview": full_text[:500]
    }

if __name__ == '__main__':
    result = check_report()
    if result["passed"]:
        print('✅ 报告格式合规')
        print(f'  章节: {result["line_count"]}行')
        if result.get("stats"):
            s = result["stats"]
            print(f'  标记: 🟢{s.get("🟢",0)} 🟡{s.get("🟡",0)} 🔴{s.get("🔴",0)}')
    else:
        msg = '🚨 报告合规检查未通过\n\n'
        msg += '\n'.join(result["failures"])
        print(msg)
        send(msg)
