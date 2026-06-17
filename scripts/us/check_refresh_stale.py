#!/usr/bin/env python3
"""
检查扫描刷新是否过期 — A股+美股双市场
只在对应市场交易时段检查，非交易时段静默
"""
import json, os, time, sys, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

A_TOP100 = os.path.join(ROOT, 'data', 'morning_top100.json')
US_SCORED = os.path.join(ROOT, 'data', 'us_scored.json')

A_MAX_STALE = 150   # A股Top100超过2.5小时 = 过期
US_MAX_STALE = 180  # 美股评分超3小时过期 (到2点共4.5h, 至少会触发一次)

def is_a_market_hours():
    """A股交易时段: 周一到周五 9:00-15:00 HKT"""
    now = datetime.datetime.now()
    return now.weekday() < 5 and 9 <= now.hour < 15

def is_us_market_hours():
    """美股交易时段(Andy醒着时): 周一到周五 21:30-02:00 HKT"""
    now = datetime.datetime.now()
    return now.weekday() < 5 and now.hour >= 21 and now.hour < 2

def check_file(filepath, name, max_stale, market_name, is_open):
    if not os.path.exists(filepath):
        from audit_engine import audit
        audit(f'refresh_{name}', 'critical', f'{name}文件不存在！')
        return False
    
    if not is_open:
        return True  # 非交易时段不检查
    
    mtime = os.path.getmtime(filepath)
    age_minutes = (time.time() - mtime) / 60
    
    if age_minutes > max_stale:
        from audit_engine import audit
        from notify import send
        msg = f'{name}已{age_minutes:.0f}分钟未更新（阈值{max_stale}分钟）[{market_name}交易时段]'
        audit(f'refresh_{name}', 'error', msg)
        send(f'🚨 {name}刷新过期\n最近更新: {time.ctime(mtime)}\n过期: {age_minutes:.0f}分钟')
        return False
    
    return True

if __name__ == '__main__':
    check_file(A_TOP100, 'Top100', A_MAX_STALE, 'A股', is_a_market_hours())
    check_file(US_SCORED, 'US评分', US_MAX_STALE, '美股', is_us_market_hours())
