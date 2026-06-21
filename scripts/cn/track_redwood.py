#!/usr/bin/env python3
"""红杉推荐跟踪器 — 每日保存推荐，跟踪收益"""
import json, os, time
from datetime import datetime

ROOT = os.path.expanduser('~/.hermes/openclaw-archive')
TRACK_FILE = os.path.join(ROOT, 'signals/cn/redwood_tracking.json')

def load_signal():
    with open(os.path.join(ROOT, 'signals/cn/latest_xgb.json')) as f:
        return json.load(f)

def load_tracking():
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE) as f:
            return json.load(f)
    return {'recommendations': []}

def save_tracking(data):
    with open(TRACK_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def add_today():
    signal = load_signal()
    tracking = load_tracking()
    date = signal['date']
    
    # Check if today already recorded
    if any(r['date'] == date for r in tracking['recommendations']):
        print(f"Date {date} already tracked")
        return
    
    entry = {
        'date': date,
        'regime': signal['regime'],
        'breadth': signal['market']['breadth'],
        'mkt_ret20': signal['market']['ret20'],
        'picks': []
    }
    
    for s in signal['top']:
        entry['picks'].append({
            'rank': s['rank'],
            'sym': s['sym'],
            'name': s.get('name', ''),
            'industry': s.get('industry', ''),
            'price': s['close'],
            'score': s['score'],
            'signal': s['signal']
        })
    
    tracking['recommendations'].append(entry)
    
    # Keep last 60 trading days
    if len(tracking['recommendations']) > 60:
        tracking['recommendations'] = tracking['recommendations'][-60:]
    
    save_tracking(tracking)
    print(f"Tracked {len(entry['picks'])} picks for {date}")

if __name__ == '__main__':
    add_today()
