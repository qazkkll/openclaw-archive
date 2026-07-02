#!/usr/bin/env python3
"""
数据新鲜度检查 — 日频版
每个交易日收盘后检查：收盘价、FMP基本面、技术特征是否都更新到位。

时间线：
  纽约16:00收盘 = HKT 04:00
  Massive.io收盘价通常HKT 08:00前到位
  FMP基本面季度更新
  → HKT 16:00前检查收盘价是否到位
  → HKT 21:30开盘前必须确认数据新鲜
"""
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path('/home/hermes/.hermes/openclaw-archive/data/falcon')

# ── 交易日历（简化：周一到周五，排除主要假日）──
def is_likely_trading_day(dt):
    """粗略判断是否为交易日（周一到周五）。"""
    return dt.weekday() < 5

def get_latest_expected_date():
    """计算最近应该有的交易日期。
    规则：
    - 如果现在是HKT 16:00之后，应该有今天(ET)的收盘数据
    - 如果现在是HKT 16:00之前，应该有昨天(ET)的收盘数据
    """
    from datetime import timezone
    now_hkt = datetime.now()  # WSL默认HKT/CST
    
    # 纽约时间 = HKT - 12小时
    now_et = now_hkt - timedelta(hours=12)
    
    # 如果ET已经过了16:00收盘，今天的数据应该有了
    if now_et.hour >= 16:
        target = now_et.date()
    else:
        target = (now_et - timedelta(days=1)).date()
    
    # 回退到最近的交易日
    while not is_likely_trading_day(target):
        target -= timedelta(days=1)
    
    return target


def check_price_freshness():
    """检查价格数据新鲜度。"""
    parquet = DATA_DIR / 'features_v04_1.parquet'
    if not parquet.exists():
        # fallback to v02
        parquet = DATA_DIR / 'features_v02.parquet'
    if not parquet.exists():
        return False, "❌ parquet不存在", -1, None
    
    f = pd.read_parquet(parquet)
    latest = f['date'].max()
    latest_str = str(latest)
    
    expected = get_latest_expected_date()
    expected_str = expected.strftime('%Y-%m-%d')
    
    latest_dt = datetime.strptime(latest_str, '%Y-%m-%d').date()
    gap = (expected - latest_dt).days
    
    if gap <= 0:
        return True, f"✅ 价格数据最新: {latest_str}", 0, latest_str
    elif gap == 1:
        return True, f"⚠️ 价格数据差1天: {latest_str} (应为{expected_str})，可能周末正常", gap, latest_str
    elif gap <= 3:
        return False, f"🔴 价格数据过期{gap}天: {latest_str} (应为{expected_str})", gap, latest_str
    else:
        return False, f"🚨 价格数据严重过期{gap}天: {latest_str} (应为{expected_str})", gap, latest_str


def check_fmp_freshness(as_of_date):
    """检查FMP基本面数据新鲜度。"""
    checks = {}
    files = {
        '财务比率': 'fmp_ratios_historical.json',
        '关键指标': 'fmp_key_metrics.json',
        '分析师': 'analyst_historical.json',
    }
    
    for name, fname in files.items():
        fpath = DATA_DIR / fname
        if not fpath.exists():
            checks[name] = {'status': '❌ 文件不存在', 'coverage': 0}
            continue
        
        with open(fpath) as f:
            data = json.load(f)
        
        covered = 0
        latest_dates = []
        stale = 0  # 最新数据超过90天的
        
        for ticker, records in data.items():
            if not records:
                continue
            valid = [r['date'] for r in records if r.get('date') and r['date'] <= as_of_date]
            if valid:
                covered += 1
                ld = max(valid)
                latest_dates.append(ld)
                # 季度数据：如果最新记录超过90天算过期
                days_old = (datetime.strptime(as_of_date, '%Y-%m-%d') - datetime.strptime(ld, '%Y-%m-%d')).days
                if days_old > 90:
                    stale += 1
        
        total = len(data)
        pct = round(covered/total*100, 1) if total > 0 else 0
        latest = max(latest_dates) if latest_dates else 'N/A'
        
        if stale > total * 0.3:
            status = f"⚠️ {stale}只过期(>90天)"
        else:
            status = "✅"
        
        checks[name] = {
            'status': status,
            'coverage': f"{covered}/{total} ({pct}%)",
            'latest': latest,
            'stale_count': stale,
        }
    
    return checks


def full_freshness_report():
    """完整数据新鲜度报告。"""
    lines = []
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    expected = get_latest_expected_date().strftime('%Y-%m-%d')
    
    lines.append(f"📅 **数据新鲜度检查** {now}")
    lines.append(f"   应有数据截至: {expected}")
    lines.append("")
    
    # 价格数据
    ok, msg, gap, latest = check_price_freshness()
    lines.append(f"📊 **价格数据** {msg}")
    
    if latest:
        # FMP数据
        lines.append(f"\n📈 **基本面数据** (截至 {latest}):")
        fmp = check_fmp_freshness(latest)
        for name, info in fmp.items():
            lines.append(f"  {info['status']} {name}: {info['coverage']}, 最新{info.get('latest','N/A')}")
    
    # 总结
    lines.append("")
    if ok and gap <= 1:
        lines.append("✅ **数据状态: 正常** — 可以评分")
    elif ok:
        lines.append("⚠️ **数据状态: 轻微滞后** — 建议更新后评分")
    else:
        lines.append("🔴 **数据状态: 过期** — 必须运行 update_price_data.py")
    
    return "\n".join(lines), ok


if __name__ == '__main__':
    report, ok = full_freshness_report()
    print(report)
