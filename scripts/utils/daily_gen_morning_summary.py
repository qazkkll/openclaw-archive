#!/usr/bin/env python3
"""08:00晨流第一步：扫描关键数据状态，写入 morning_summary.json

当前架构 (2026-06-13):
  🟢 绿箭 V8-Lottery — $1-10彩票爆发预测
  🛡️ 蓝盾 3.0 — 大盘技术评分
  数据存储: /home/hermes/.hermes/openclaw-archive/data\
"""
import os, json, sys
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA_DIR = r'/home/hermes/.hermes/openclaw-archive/data'
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # workspace根
TZ = timezone(timedelta(hours=8))

def file_info(path):
    try:
        mtime = os.path.getmtime(path)
        age_s = datetime.now().timestamp() - mtime
        if age_s > 86400:
            age = f'{age_s/3600:.0f}h ago'
        elif age_s > 3600:
            age = f'{age_s/3600:.1f}h ago'
        else:
            age = f'{age_s/60:.0f}m ago'
        return {
            'exists': True,
            'age': age,
            'size_kb': round(os.path.getsize(path)/1024, 1),
            'mtime': datetime.fromtimestamp(mtime, TZ).isoformat()
        }
    except FileNotFoundError:
        return {'exists': False, 'age': 'missing', 'size_kb': 0}
    except Exception as e:
        return {'exists': False, 'age': str(e), 'size_kb': 0}

now = datetime.now(TZ)
today = now.strftime('%Y-%m-%d')
yesterday = (now - timedelta(days=1)).strftime('%Y-%m-%d')

summary = {
    'generated_at': now.isoformat(),
    'date': today,
    'architecture': {
        'green_arrow': 'V8-Lottery ($1-10彩票爆发, XGBoost 43特征, top5 17.1%命中)',
        'blue_shield': '蓝盾3.0 (大盘技术评分, S&P 500 ~503只, ≥90强买, <75退出)',
        'a_share': 'A1资金流模型 (Layer1+Layer3+基本面)'
    },
    'us_stocks': {},
    'a_stocks': {},
    'pending_tasks': {},
    'data_files': {}
}

# ===== 美股检查 =====
ld3_file = os.path.join(DATA_DIR, f'ld3_scored_{today}.json')
ld3_yesterday = os.path.join(DATA_DIR, f'ld3_scored_{yesterday}.json')
v75_file = os.path.join(DATA_DIR, f'v75_scored_{today}.json')
v75_yesterday = os.path.join(DATA_DIR, f'v75_scored_{yesterday}.json')

summary['us_stocks']['ld3_today'] = file_info(ld3_file)
summary['us_stocks']['ld3_yesterday'] = file_info(ld3_yesterday)
summary['us_stocks']['v75_today'] = file_info(v75_file)
summary['us_stocks']['v75_yesterday'] = file_info(v75_yesterday)

# 读评分结果（如果存在），提取关键摘要
def _parse_ld3(data):
    """解析ld3评分文件（结构为 {date, scores: [{code, score...}], rules}）"""
    scores_list = data.get('scores', data) if isinstance(data, dict) else data
    if isinstance(scores_list, list):
        return scores_list
    return []

for label, fpath in [('ld3_today', ld3_file), ('ld3_yesterday', ld3_yesterday)]:
    if os.path.exists(fpath):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            scores = _parse_ld3(raw)
            top5 = scores[:5]
            top5_scores = [f"{s.get('code','?')}={s.get('score','?')}" for s in top5]
            high_count = sum(1 for s in scores if s.get('score', 0) >= 90) if scores else 0
            total = len(scores) if scores else 0
            summary['us_stocks'][f'{label}_summary'] = {
                'total_scored': total,
                'count_ge90': high_count,
                'top5': top5_scores,
                'top_score': max(s.get('score', 0) for s in scores) if scores else 0
            }
        except Exception as e:
            summary['us_stocks'][f'{label}_summary'] = f'read_error: {e}'

for label, fpath in [('v75_today', v75_file), ('v75_yesterday', v75_yesterday)]:
    if os.path.exists(fpath):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            top5 = data[:5] if isinstance(data, list) else []
            top5_scores = [f"{s.get('code','?')}={s.get('score','?')}" for s in top5]
            summary['us_stocks'][f'{label}_summary'] = {
                'total_scored': len(data) if data else 0,
                'top5': top5_scores,
                'top_score': max(s.get('score', 0) for s in data) if data else 0
            }
        except Exception as e:
            summary['us_stocks'][f'{label}_summary'] = f'read_error: {e}'

# ===== A股检查 =====
a1_file = os.path.join(WORK_DIR, 'data', 'a1_daily.json')
# 统一路径
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import NORTH_MONEY
north_file = NORTH_MONEY
a1_report = os.path.join(WORK_DIR, 'data', 'a1_report.json')

summary['a_stocks']['a1_daily'] = file_info(a1_file)
summary['a_stocks']['north_money'] = file_info(north_file)
summary['a_stocks']['a1_report'] = file_info(a1_report)

# ===== pending任务 =====
for fname in ['operation_plan.json', 'tomorrow_plan.json', 'active_mission.json', 'wal_log.json']:
    fpath = os.path.join(WORK_DIR, 'data', fname)
    info = file_info(fpath)
    if info['exists']:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            if isinstance(content, dict):
                status = content.get('status', content.get('mode', 'unknown'))
                total_tasks = len(content.get('tasks', content.get('steps', [])))
                pending_count = sum(1 for t in content.get('tasks', []) if t.get('status') in ('pending', 'in_progress'))
                info['_summary'] = {'status': status, 'total_tasks': total_tasks, 'pending': pending_count}
            elif isinstance(content, list):
                info['_summary'] = {'total_entries': len(content)}
        except Exception as e:
            info['_summary'] = f'read_error: {e}'
    summary['pending_tasks'][fname] = info

# ===== 关键数据文件 =====
key_files = [
    ('data/decision_history.jsonl', '决策记录'),
    ('data/experience_log.jsonl', '经验日志'),
    ('data/recommendation_tracker.json', '推荐追踪'),
    ('data/rules.json', '交易规则'),
    ('scripts/positions_opend.json', '美股持仓(OpenD)'),

]
for rel_path, desc in key_files:
    fpath = os.path.join(WORK_DIR if not rel_path.startswith('scripts') else WORK_DIR, rel_path)
    summary['data_files'][desc] = file_info(fpath)

# ===== 一句话摘要 =====
tags = []
# 美股
ld3_ok = summary['us_stocks']['ld3_yesterday']['exists']
ld3_t = summary['us_stocks']['ld3_yesterday']['exists'] and summary['us_stocks']['ld3_yesterday_summary'].get('count_ge90', 0)
v75_ok = summary['us_stocks']['v75_yesterday']['exists']
ld3_yesterday_summary = summary['us_stocks'].get('ld3_yesterday_summary', {})
if isinstance(ld3_yesterday_summary, dict):
    high_count = ld3_yesterday_summary.get('count_ge90', 0)
else:
    high_count = 0
tags.append(f"蓝盾{yesterday if ld3_ok else 'MISS:' + today}")
if ld3_ok and high_count:
    tags[-1] += f" {high_count}只≥90"
tags.append(f"V8{'OK' if v75_ok else 'MISS'}")
# A股
a1_ok = summary['a_stocks']['a1_daily']['exists']
tags.append(f"A1{'OK' if a1_ok else 'MISS'}")
# pending
pending_files = [k for k, v in summary['pending_tasks'].items() if v['exists']]
if pending_files:
    tags.append(f"有{len(pending_files)}个待处理")

summary['summary_line'] = f"📡 {today} 晨检 | " + " | ".join(tags)

# ===== 写入 =====
out_path = os.path.join(WORK_DIR, 'data', 'morning_summary.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f'morning_summary.json written to {out_path}')
print(summary['summary_line'])
sys.exit(0)
