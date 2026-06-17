#!/usr/bin/env python3
"""
sys_group_rec_tracker.py — 群推荐记录器
==========================================
用途：记录我在群里的每一笔推荐，每次做新推荐前自动回溯历史。
适用场景：元宝群(493450185)、Telegram群等

数据存储: data/group_records/YYYY-MM-DD_recommendations.jsonl
回溯机制: 自动读取最近N天记录并输出关键摘要

用法:
  # 记录推荐（每次群里发推荐前/后调用）
  python scripts/sys_group_rec_tracker.py record --channel yuanbao --group 493450185 --content "推荐中兴通讯(000063)，理由：资金面..."

  # 回溯历史（做推荐前调用）
  python scripts/sys_group_rec_tracker.py review --channel yuanbao --group 493450185 --days 7

  # 查看所有群组的推荐统计
  python scripts/sys_group_rec_tracker.py stats
"""

import json
import sys
import os
import argparse
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "group_records")
os.makedirs(DATA_DIR, exist_ok=True)

def get_today_file(channel, group):
    today = datetime.now().strftime("%Y-%m-%d")
    fname = f"{today}_recommendations.jsonl"
    return os.path.join(DATA_DIR, fname)

def load_today_records(channel, group):
    """加载今天已有记录，避免重复"""
    path = get_today_file(channel, group)
    records = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records, path

def record_recommendation(channel, group, content, rec_type="buy", codes=None, model=None):
    """记录一条推荐"""
    records, path = load_today_records(channel, group)
    now = datetime.now().isoformat()
    entry = {
        "ts": now,
        "channel": channel,
        "group": str(group),
        "type": rec_type,
        "content": content,
        "codes": codes or [],
        "model": model or ""
    }
    records.append(entry)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return len(records)

def review_recommendations(channel, group, days=7):
    """回溯历史推荐，返回结构化摘要"""
    today = datetime.now()
    records = []
    for i in range(days + 1):
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        path = os.path.join(DATA_DIR, f"{d}_recommendations.jsonl")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            if rec.get("channel") == channel and str(rec.get("group")) == str(group):
                                records.append(rec)
                        except json.JSONDecodeError:
                            continue
    # 按时间排序
    records.sort(key=lambda r: r.get("ts", ""))
    # 提取关键摘要
    summary = {
        "total": len(records),
        "by_type": {},
        "by_code": {},
        "recent_5": records[-5:] if records else [],
        "date_range": f"{records[0]['ts'][:10]}" if records else "无记录",
        "all_codes": list(set(
            code for r in records for code in r.get("codes", [])
        ))
    }
    for r in records:
        t = r.get("type", "unknown")
        summary["by_type"][t] = summary["by_type"].get(t, 0) + 1
        for code in r.get("codes", []):
            if code not in summary["by_code"]:
                summary["by_code"][code] = []
            summary["by_code"][code].append(r["ts"][:10])
    return summary

def stats():
    """全量统计"""
    all_records = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith("_recommendations.jsonl"):
            path = os.path.join(DATA_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    print(f"总记录数: {len(all_records)}")
    channels = set(r.get("channel", "?") for r in all_records)
    print(f"渠道: {', '.join(channels) if channels else '无'}")
    groups = set(r.get("group", "?") for r in all_records)
    print(f"群组: {', '.join(groups) if groups else '无'}")
    if all_records:
        dates = set(r["ts"][:10] for r in all_records)
        print(f"覆盖天数: {len(dates)}")
        print(f"时间范围: {min(dates)} 至 {max(dates)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="群推荐记录器")
    subparsers = parser.add_subparsers(dest="cmd")
    
    p_record = subparsers.add_parser("record")
    p_record.add_argument("--channel", required=True)
    p_record.add_argument("--group", required=True)
    p_record.add_argument("--content", required=True)
    p_record.add_argument("--type", default="buy", choices=["buy", "sell", "hold", "watch", "analysis"])
    p_record.add_argument("--codes", nargs="*", default=[])
    p_record.add_argument("--model", default="")
    
    p_review = subparsers.add_parser("review")
    p_review.add_argument("--channel", required=True)
    p_review.add_argument("--group", required=True)
    p_review.add_argument("--days", type=int, default=7)
    
    p_stats = subparsers.add_parser("stats")
    
    args = parser.parse_args()
    
    if args.cmd == "record":
        count = record_recommendation(args.channel, args.group, args.content, args.type, args.codes, args.model)
        print(json.dumps({"status": "ok", "today_count": count}, ensure_ascii=False))
    
    elif args.cmd == "review":
        summary = review_recommendations(args.channel, args.group, args.days)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    
    elif args.cmd == "stats":
        stats()
    
    else:
        parser.print_help()
