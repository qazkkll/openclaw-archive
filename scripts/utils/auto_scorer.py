#!/usr/bin/env python3
"""
自动评分器 v3 — 匹配模型预测周期（5天前瞻收益率>2%）
用法：
  python3 auto_scorer.py              # 评分所有到期的pending
  python3 auto_scorer.py --date 20260622  # 评分指定日期
  python3 auto_scorer.py --dry-run    # 只看不改
  python3 auto_scorer.py --force      # 强制重新评分（覆盖已评分的）
  python3 auto_scorer.py --1day       # 同时评分1天（兼容旧数据）
"""
import json, os, sys, time
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

ROOT = os.path.expanduser('~/.hermes/openclaw-archive')
TRACK_FILE = os.path.join(ROOT, 'data/recommendations.json')

# 模型训练标签: ret_5d > 0.02 → 正类
# 评分标准必须匹配: 5天后价格 vs 预测日价格, 阈值2%
MODEL_HORIZON_DAYS = 5
MODEL_THRESHOLD = 0.02  # 2%

# === Yahoo Finance ticker mapping ===
SPECIAL_TICKERS = {
    'SPY': 'SPY', 'QQQ': 'QQQ', 'VIX': '^VIX',
    'IWM': 'IWM', 'DIA': 'DIA',
    'Semiconductor': 'SOXX',
}

def to_yahoo_ticker(target: str) -> str:
    target = target.strip()
    if target in SPECIAL_TICKERS:
        return SPECIAL_TICKERS[target]
    if ' ' not in target and not target.isdigit():
        return target
    parts = target.split()
    code = parts[0]
    if code.isdigit() and len(code) == 6:
        return f"{code}.SS" if code.startswith('6') else f"{code}.SZ"
    return code

def get_price_on_date(ticker: str, date_str: str) -> float | None:
    """获取指定日期的收盘价"""
    try:
        dt = datetime.strptime(date_str, '%Y%m%d')
        start = (dt - timedelta(days=3)).strftime('%Y-%m-%d')
        end = (dt + timedelta(days=5)).strftime('%Y-%m-%d')
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start, end=end)
        if hist.empty:
            return None
        target_date = dt.date()
        for d in sorted(hist.index, reverse=True):
            if d.date() <= target_date:
                return float(hist.loc[d, 'Close'])
        return float(hist.iloc[0]['Close'])
    except Exception as e:
        print(f"  ⚠️ 获取{ticker}价格失败: {e}")
        return None

def get_price_after_days(ticker: str, date_str: str, days: int = 5) -> tuple[float | None, str | None]:
    """获取date_str之后N个交易日的收盘价。返回 (price, actual_date)"""
    try:
        dt = datetime.strptime(date_str, '%Y%m%d')
        # 往后取足够天数确保覆盖N个交易日
        start = dt.strftime('%Y-%m-%d')
        end = (dt + timedelta(days=days + 10)).strftime('%Y-%m-%d')
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start, end=end)
        if hist.empty:
            return None, None
        target_date = dt.date()
        # 找到target_date之后的第N个交易日
        future_dates = [d for d in sorted(hist.index) if d.date() > target_date]
        if len(future_dates) >= days:
            d = future_dates[days - 1]
            return float(hist.loc[d, 'Close']), d.strftime('%Y%m%d')
        return None, None
    except Exception as e:
        print(f"  ⚠️ 获取{ticker}后{days}天价格失败: {e}")
        return None, None

def score_one(rec: dict, dry_run: bool = False, score_1day: bool = False) -> dict | None:
    """对单条recommendation评分（5天周期，匹配模型训练标签）"""
    target = rec.get('target', '')
    date = rec.get('date', '')
    direction = rec.get('direction', '')
    
    if not date or not target:
        return None
    if target in ('大盘', 'A股大盘', '美股大盘'):
        return None
    
    ticker = to_yahoo_ticker(target)
    
    # 获取预测日价格
    base_price = get_price_on_date(ticker, date)
    if base_price is None:
        print(f"  ❌ {target}({ticker}): 无法获取{date}价格")
        return None
    
    # 获取5天后价格（匹配模型预测周期）
    price_5d, date_5d = get_price_after_days(ticker, date, MODEL_HORIZON_DAYS)
    if price_5d is None:
        print(f"  ❌ {target}({ticker}): 无法获取{date}后{MODEL_HORIZON_DAYS}天价格")
        return None
    
    # 5天收益率
    ret_5d = (price_5d - base_price) / base_price
    
    # 评分逻辑（匹配模型标签: ret_5d > 0.02 → 正类）
    if direction == 'bullish':
        if ret_5d > MODEL_THRESHOLD:       # 涨>2% = 正确
            score = 1.0
        elif ret_5d > 0:                   # 涨但<2% = 部分正确
            score = 0.5
        elif ret_5d > -MODEL_THRESHOLD:    # 跌但<2% = 部分错误
            score = 0.25
        else:                              # 跌>2% = 完全错误
            score = 0.0
    elif direction == 'bearish':
        if ret_5d < -MODEL_THRESHOLD:      # 跌>2% = 正确
            score = 1.0
        elif ret_5d < 0:                   # 跌但<2% = 部分正确
            score = 0.5
        elif ret_5d < MODEL_THRESHOLD:     # 涨但<2% = 部分错误
            score = 0.25
        else:                              # 涨>2% = 完全错误
            score = 0.0
    else:  # neutral
        if abs(ret_5d) < MODEL_THRESHOLD:  # 波动<2% = 正确
            score = 1.0
        elif abs(ret_5d) < 0.05:           # 波动2-5% = 部分
            score = 0.5
        else:
            score = 0.0
    
    emoji = '✅' if score >= 0.5 else '⚠️' if score > 0 else '❌'
    print(f"  {emoji} {target}({ticker}): {direction} | ${base_price:.2f}→${price_5d:.2f} (5天) | ret={ret_5d:+.2%} | score={score}")
    
    result = {
        'score': score,
        'actual_return': f"{ret_5d:+.2%}",
        'actual_return_pct': ret_5d,
        'outcome': 'correct' if score >= 0.5 else 'wrong',
        'base_price': base_price,
        'next_price': price_5d,
        'next_date': date_5d,
        'horizon_days': MODEL_HORIZON_DAYS,
        'threshold': MODEL_THRESHOLD,
    }
    
    # 可选：同时评分1天（兼容旧数据对比）
    if score_1day:
        price_1d, date_1d = get_price_after_days(ticker, date, 1)
        if price_1d is not None:
            ret_1d = (price_1d - base_price) / base_price
            result['ret_1d'] = ret_1d
            result['ret_1d_date'] = date_1d
            print(f"    (1天: ${price_1d:.2f}, ret={ret_1d:+.2%})")
    
    return result

def main():
    dry_run = '--dry-run' in sys.argv
    force = '--force' in sys.argv
    score_1day = '--1day' in sys.argv
    date_filter = None
    
    if '--date' in sys.argv:
        idx = sys.argv.index('--date')
        if idx + 1 < len(sys.argv):
            date_filter = sys.argv[idx + 1]
    
    with open(TRACK_FILE) as f:
        data = json.load(f)
    
    recs = data.get('recommendations', [])
    now = datetime.now()
    today_str = now.strftime('%Y%m%d')
    
    pending = []
    for r in recs:
        if r.get('status') != 'pending' and not force:
            continue
        if date_filter and r.get('date') != date_filter:
            continue
        rec_date = r.get('date', '')
        if rec_date >= today_str:
            continue
        if r.get('target') in ('大盘', 'A股大盘', '美股大盘'):
            continue
        pending.append(r)
    
    if not pending:
        print("没有需要评分的建议")
        return
    
    print(f"📊 待评分: {len(pending)}条 (模型周期: {MODEL_HORIZON_DAYS}天, 阈值: {MODEL_THRESHOLD:.0%})\n")
    
    scored_count = 0
    for rec in pending:
        result = score_one(rec, dry_run, score_1day)
        if result is None:
            continue
        
        scored_count += 1
        
        if not dry_run:
            rec['status'] = 'scored'
            rec['score'] = result['score']
            rec['outcome'] = result['outcome']
            rec['outcome_date'] = now.strftime('%Y-%m-%d')
            rec['actual_return'] = result['actual_return']
            rec['scoring_detail'] = {
                'base_price': result['base_price'],
                'next_price': result['next_price'],
                'next_date': result['next_date'],
                'horizon_days': result['horizon_days'],
                'threshold': result['threshold'],
            }
            if 'ret_1d' in result:
                rec['scoring_detail']['ret_1d'] = result['ret_1d']
                rec['scoring_detail']['ret_1d_date'] = result['ret_1d_date']
        
        time.sleep(0.5)
    
    if not dry_run and scored_count > 0:
        with open(TRACK_FILE, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 已评分 {scored_count} 条，已写入 {TRACK_FILE}")
    elif dry_run:
        print(f"\n🔍 Dry run完成，共{scored_count}条可评分（未写入）")
    
    all_scored = [r for r in recs if r.get('status') == 'scored']
    if all_scored:
        correct = sum(1 for r in all_scored if (r.get('score') or 0) >= 0.5)
        print(f"\n📈 累计统计: {len(all_scored)}条已评分, 命中{correct}/{len(all_scored)} = {correct/len(all_scored)*100:.1f}%")

if __name__ == '__main__':
    main()
