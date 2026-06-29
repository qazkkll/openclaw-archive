#!/usr/bin/env python3
"""
🦅 FMP Premium Feature Extractor
从FMP Premium数据中提取两组新特征: Earnings Surprise + Analyst Grade Sentiment
适配实际数据格式（epsActual/epsEstimated + analystRatings*）

数据源:
  - earnings_symbol-{TICKER}.json  → {symbol, date, epsActual, epsEstimated, revenueActual, revenueEstimated}
  - grades_historical_symbol-{TICKER}.json → {symbol, date, analystRatingsStrongBuy/Buy/Hold/Sell/StrongSell}
"""
import json
from pathlib import Path
from datetime import datetime, timedelta


def load_fmp_premium_earnings(data_dir):
    """加载FMP Premium earnings数据。
    
    返回 dict: ticker -> list of earnings records (按date降序排列)
    数据已包含 epsActual, epsEstimated, revenueActual, revenueEstimated
    """
    raw_dir = Path(data_dir) / "data" / "raw"
    earnings = {}
    
    for f in sorted(raw_dir.glob("earnings_symbol-*.json")):
        ticker = f.stem.replace("earnings_symbol-", "")
        try:
            with open(f) as fh:
                records = json.load(fh)
            if not records:
                continue
            # 按date降序排列（最新在前）
            records.sort(key=lambda r: r.get("date", ""), reverse=True)
            earnings[ticker] = records
        except Exception as e:
            print(f"⚠️ 跳过 {ticker} earnings: {e}")
    
    print(f"📥 加载 {len(earnings)} 只股票的 earnings 数据")
    return earnings


def load_fmp_premium_grades(data_dir):
    """加载FMP Premium grades历史数据。
    
    返回 dict: ticker -> list of grade records (按date降序排列)
    数据已包含 analystRatingsStrongBuy/Buy/Hold/Sell/StrongSell
    """
    raw_dir = Path(data_dir) / "data" / "raw"
    grades = {}
    
    for f in sorted(raw_dir.glob("grades_historical_symbol-*.json")):
        ticker = f.stem.replace("grades_historical_symbol-", "")
        try:
            with open(f) as fh:
                records = json.load(fh)
            if not records:
                continue
            records.sort(key=lambda r: r.get("date", ""), reverse=True)
            grades[ticker] = records
        except Exception as e:
            print(f"⚠️ 跳过 {ticker} grades: {e}")
    
    print(f"📥 加载 {len(grades)} 只股票的 grades 数据")
    return grades


def extract_earnings_features(records, date):
    """PIT安全: 用date之前已发布的earnings数据提取特征。
    
    实际数据没有surprisePercentage字段，从epsActual/epsEstimated计算。
    earnings的date字段是财报季末日，但FMP数据中lastUpdated表示发布日期。
    为安全起见：用date字段（季末日）作为PIT锚点，+33天延迟。
    
    返回 dict 或空dict
    """
    if not records:
        return {}
    
    FILING_DELAY = 33  # 与falcon_v03_engine.py一致
    
    # 筛选date之前已可用的记录
    available = []
    for r in records:
        r_date = r.get("date", "")
        if not r_date:
            continue
        try:
            avail = (datetime.strptime(r_date, "%Y-%m-%d") + timedelta(days=FILING_DELAY)).strftime("%Y-%m-%d")
        except ValueError:
            continue
        if avail <= date and r.get("epsActual") is not None:
            available.append(r)
    
    if not available:
        return {}
    
    # available已按date降序排列，取最近的
    latest = available[0]
    eps_actual = latest.get("epsActual")
    eps_est = latest.get("epsEstimated")
    rev_actual = latest.get("revenueActual")
    rev_est = latest.get("revenueEstimated")
    
    features = {}
    
    # earnings_surprise: 最近一次EPS惊喜 (实际-预期)/|预期|
    if eps_actual is not None and eps_est is not None and eps_est != 0:
        features["earnings_surprise"] = (eps_actual - eps_est) / abs(eps_est)
    
    # earnings_surprise_2q: 最近2次EPS惊喜均值
    if len(available) >= 2:
        surprises = []
        for r in available[:2]:
            ea = r.get("epsActual")
            ee = r.get("epsEstimated")
            if ea is not None and ee is not None and ee != 0:
                surprises.append((ea - ee) / abs(ee))
        if surprises:
            features["earnings_surprise_2q"] = sum(surprises) / len(surprises)
    
    # earnings_beat_count_4q: 最近4次中beat(epsActual > epsEstimated)的次数
    beats = 0
    total_4q = 0
    for r in available[:4]:
        ea = r.get("epsActual")
        ee = r.get("epsEstimated")
        if ea is not None and ee is not None:
            total_4q += 1
            if ea > ee:
                beats += 1
    if total_4q > 0:
        features["earnings_beat_count_4q"] = beats / total_4q  # 用比例，便于rank
    
    # earnings_price_reaction: 用revenue surprise代替（原数据无priceChangePercentage）
    if rev_actual is not None and rev_est is not None and rev_est != 0:
        features["earnings_price_reaction"] = (rev_actual - rev_est) / abs(rev_est)
    
    return features


def extract_grade_features(records, date):
    """PIT安全: 用date之前已发布的grades数据提取特征。
    
    grades_historical数据是月度快照，包含rating分布。
    从分布变化中计算sentiment特征。
    
    返回 dict 或空dict
    """
    if not records:
        return {}
    
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {}
    
    # 筛选90天内+date之前的记录
    cutoff = (dt - timedelta(days=90)).strftime("%Y-%m-%d")
    recent = [r for r in records if cutoff <= r.get("date", "") <= date]
    
    if len(recent) < 2:
        # 不够2个快照无法计算趋势，尝试用最近1个
        if recent:
            return _grade_snapshot_features(recent[0])
        return {}
    
    return _grade_snapshot_features_trend(recent)


def _grade_snapshot_features(record):
    """从单个快照计算基础features。"""
    sb = record.get("analystRatingsStrongBuy", 0) or 0
    b = record.get("analystRatingsBuy", 0) or 0
    h = record.get("analystRatingsHold", 0) or 0
    s = record.get("analystRatingsSell", 0) or 0
    ss = record.get("analystRatingsStrongSell", 0) or 0
    total = sb + b + h + s + ss
    
    if total == 0:
        return {}
    
    return {
        "grade_upgrade_ratio_90d": (sb + b) / total,
        "grade_downgrade_ratio_90d": (s + ss) / total,
        "grade_momentum_90d": ((sb + b) - (s + ss)) / total,
        "grade_target_raised_90d": sb / total,  # StrongBuy比例作为target raised代理
    }


def _grade_snapshot_features_trend(records):
    """从多个快照计算趋势features（按date升序处理）。"""
    # records按date降序，反转为升序
    records_sorted = sorted(records, key=lambda r: r.get("date", ""))
    
    def _bullish_ratio(r):
        sb = r.get("analystRatingsStrongBuy", 0) or 0
        b = r.get("analystRatingsBuy", 0) or 0
        h = r.get("analystRatingsHold", 0) or 0
        s = r.get("analystRatingsSell", 0) or 0
        ss = r.get("analystRatingsStrongSell", 0) or 0
        total = sb + b + h + s + ss
        return total, sb, b, h, s, ss
    
    latest = records_sorted[-1]
    earliest = records_sorted[0]
    
    lt_total, lt_sb, lt_b, lt_h, lt_s, lt_ss = _bullish_ratio(latest)
    et_total, et_sb, et_b, et_h, et_s, et_ss = _bullish_ratio(earliest)
    
    if lt_total == 0:
        return {}
    
    # 当前bullish ratio
    lt_bullish = (lt_sb + lt_b) / lt_total
    lt_bearish = (lt_s + lt_ss) / lt_total
    
    # 早期bullish ratio（如果可用）
    et_bullish = (et_sb + et_b) / et_total if et_total > 0 else lt_bullish
    
    return {
        # 过去90天中最新snapshot的upgrade ratio
        "grade_upgrade_ratio_90d": lt_bullish,
        # 过去90天中最新snapshot的downgrade ratio
        "grade_downgrade_ratio_90d": lt_bearish,
        # 90天内bullish ratio变化（正=趋势向好）
        "grade_momentum_90d": lt_bullish - et_bullish,
        # 当前StrongBuy占比（target raised的代理指标）
        "grade_target_raised_90d": lt_sb / lt_total if lt_total > 0 else 0,
    }


def extract_all_features(ticker, date, earnings_data, grades_data):
    """提取单只股票在特定日期的所有FMP Premium features。"""
    features = {}
    
    # Earnings features
    er = extract_earnings_features(earnings_data.get(ticker, []), date)
    features.update(er)
    
    # Grade features
    gr = extract_grade_features(grades_data.get(ticker, []), date)
    features.update(gr)
    
    return features


# ═══════════════════════════════════════════════════
# 独立测试
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    data_dir = Path("/home/hermes/.hermes/openclaw-archive/data/fmp_premium")
    
    earnings = load_fmp_premium_earnings(data_dir)
    grades = load_fmp_premium_grades(data_dir)
    
    # 测试几只股票
    for ticker in ["AAPL", "MSFT", "GOOG"]:
        feats = extract_all_features(ticker, "2026-06-29", earnings, grades)
        print(f"\n{ticker} @ 2026-06-29:")
        for k, v in feats.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
