#!/usr/bin/env python3
"""
预测跟踪器 — 记录主观预测，回看准确率
用法：
  python3 prediction_tracker.py add '{"market":"cn","date":"20260622","predictions":[...]}'
  python3 prediction_tracker.py review [days_back]
  python3 prediction_tracker.py stats
"""
import json, os, sys
from datetime import datetime, timedelta

ROOT = os.path.expanduser('~/.hermes/openclaw-archive')
TRACK_FILE = os.path.join(ROOT, 'data/predictions.json')

def load():
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE) as f:
            return json.load(f)
    return {'predictions': []}

def save(data):
    os.makedirs(os.path.dirname(TRACK_FILE), exist_ok=True)
    with open(TRACK_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def add_prediction(entry):
    """添加一条预测记录
    entry格式：
    {
        "market": "cn" | "us",
        "date": "20260622",
        "time": "09:00",
        "type": "morning" | "intraday" | "close",
        "macro_context": "简述宏观背景",
        "predictions": [
            {
                "target": "NXTC" | "上证指数" | "板块名",
                "direction": "bullish" | "bearish" | "neutral",
                "confidence": 0.7,  # 0-1
                "reasoning": "判断理由",
                "timeframe": "1d" | "3d" | "1w"
            }
        ]
    }
    """
    data = load()
    
    # 去重：同一天同类型不重复添加
    key = f"{entry['date']}_{entry.get('type','unknown')}"
    for i, p in enumerate(data['predictions']):
        if f"{p['date']}_{p.get('type','unknown')}" == key and p.get('market') == entry.get('market'):
            # 更新而不是追加
            data['predictions'][i] = entry
            save(data)
            print(f"Updated prediction for {key}")
            return
    
    data['predictions'].append(entry)
    # 保留最近120天
    if len(data['predictions']) > 120:
        data['predictions'] = data['predictions'][-120:]
    save(data)
    print(f"Added prediction for {key}")

def review(days_back=7):
    """回看最近N天的预测"""
    data = load()
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
    
    recent = [p for p in data['predictions'] if p['date'] >= cutoff]
    
    if not recent:
        print(f"最近{days_back}天没有预测记录")
        return
    
    print(f"=== 最近{days_back}天预测记录 ({len(recent)}条) ===\n")
    for p in recent:
        print(f"📅 {p['date']} [{p.get('market','?')}] {p.get('type','?')}")
        print(f"   宏观: {p.get('macro_context', '无')}")
        for pred in p.get('predictions', []):
            arrow = {'bullish': '🟢', 'bearish': '🔴', 'neutral': '⚪'}.get(pred['direction'], '?')
            print(f"   {arrow} {pred['target']}: {pred['direction']} (置信度:{pred['confidence']})")
            print(f"      理由: {pred['reasoning']}")
        print()

def stats():
    """统计预测准确率"""
    data = load()
    preds = data['predictions']
    
    if not preds:
        print("没有预测记录")
        return
    
    total = sum(len(p.get('predictions', [])) for p in preds)
    reviewed = sum(1 for p in preds if p.get('reviewed', False))
    
    print(f"=== 预测统计 ===")
    print(f"总预测天数: {len(preds)}")
    print(f"总预测条数: {total}")
    print(f"已复盘天数: {reviewed}")
    print(f"待复盘天数: {len(preds) - reviewed}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: prediction_tracker.py [add|review|stats] [args]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    if cmd == 'add':
        entry = json.loads(sys.argv[2])
        add_prediction(entry)
    elif cmd == 'review':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        review(days)
    elif cmd == 'stats':
        stats()
    else:
        print(f"未知命令: {cmd}")
