#!/usr/bin/env python3
"""
防御配置推荐 — 当无买入信号时，钱不躺平
市场下跌时建议将现金转入防守型资产
"""
import sys, os, json, datetime, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_defensive_suggestion(current_mode='牛市', strength_pct=None):
    """
    根据市场模式给出防守配置建议。
    
    三层判断:
    - 牛市但无票可买: 银行ETF(慢涨+分红) > 黄金ETF(对冲)
    - 震荡/不确定: 黄金ETF(避险) > 国债ETF(保本) > 银行ETF
    - 熊市确认: 国债ETF(避险) > 现金 > 黄金ETF > 银行ETF
    """
    cfg_path = os.path.join(ROOT, 'config', 'strategy.json')
    with open(cfg_path) as f:
        cfg = json.load(f)
    
    lines = []
    lines.append('防守配置建议（无买入信号时）')
    
    if current_mode == '牛市':
        # 牛市但暂时无票=中间震荡，资金可以进银行ETF
        lines.append('判断: 牛市中继，非趋势恶化，资金可工作')
        lines.append('')
        lines.append('推荐:')
        lines.append('  512800 银行ETF  60%  慢涨+4-5%分红，牛市中间震荡首选')
        lines.append('  518880 黄金ETF  30%  避险对冲，市场切换时保护')
        lines.append('  预留现金        10%  机动')
        lines.append('')
        lines.append('判断理由: 牛市未结束，银行ETF分红收益确定性强')
        
    elif current_mode == '熊市':
        # 熊市确认，保本第一
        lines.append('判断: 熊市确认，保本优先')
        lines.append('')
        lines.append('推荐:')
        lines.append('  511010 国债ETF  50%  股灾时国债上涨，最安全')
        lines.append('  518880 黄金ETF  30%  避险对冲，但流动性危机时也会跌')
        lines.append('  预留现金        20%  等抄底')
        lines.append('')
        lines.append('判断理由: 熊市中银行也会跌，国债才是真避险')
        
    else:
        # 死区/不确定
        lines.append('判断: 市场方向不明，保守为主')
        lines.append('')
        lines.append('推荐:')
        lines.append('  518880 黄金ETF  40%  不确定性下黄金表现最好')
        lines.append('  511010 国债ETF  35%  保本+小赚')
        lines.append('  512800 银行ETF  15%  少量参与以防踏空')
        lines.append('  预留现金        10%')
        lines.append('')
        lines.append('判断理由: 方向不明时不能重仓单一品种')
    
    lines.append('')
    lines.append('参考标的:')
    lines.append('  512800 银行ETF  ETF  银行防守+分红，牛市中继首选')
    lines.append('  518880 黄金ETF  ETF  避险+抗通胀，不确定性下表现好')
    lines.append('  511010 国债ETF  ETF  最安全防守，熊市中会上涨')
    lines.append('  601939 建设银行 个股  已有持仓，牛市可加仓')
    
    return '\n'.join(lines)


def format_us_report():
    """美股持仓日报 + V4.2推荐扫描（合并推送）"""
    pf_path = os.path.join(ROOT, 'data', 'portfolio.json')
    with open(pf_path) as f:
        pf = json.load(f)
    
    us = pf.get('us_stock', [])
    
    lines = ['美股简报 · ' + time.strftime('%m/%d(%a)')]
    lines.append('')
    
    # 1. 持仓
    lines.append('📦 持仓')
    total_value = 0
    total_pl = 0
    for s in us:
        try:
            import yfinance as yf
            ticker = yf.Ticker(s['code'])
            data = ticker.history(period='5d')
            if not data.empty:
                cur = float(data['Close'].iloc[-1])
                prev = float(data['Close'].iloc[-2]) if len(data) > 1 else cur
                chg = (cur / prev - 1) * 100
                shares = s.get('shares') or 1
                cost = s.get('cost', 0)
                pl = (cur - cost) * shares
                total_value += cur * shares
                total_pl += pl
                lines.append(f'  {s["code"]:4s} ${cur:>7.2f} 日{chg:+.2f}% 浮盈${pl:+,.0f}')
        except:
            pass
    lines.append(f'  市值${total_value:,.0f} | 浮盈${total_pl:+,.0f}')
    lines.append('')
    
    # 2. V4.2推荐扫描
    lines.append('❤️ V4.2评分 · Top 5')
    try:
        scan_path = os.path.join(ROOT, 'scripts', 'quick_scan_us.py')
        # Run quick_scan_us to get latest results
        result_path = os.path.join(ROOT, 'data', 'us_scan_result.json')
        if os.path.exists(result_path):
            with open(result_path) as f:
                scan_data = json.load(f)
            top = scan_data.get('top20', [])[:5]
            for i, r in enumerate(top):
                lines.append(f'  #{i+1} {r["code"]:5s} {r["score"]:5.1f}分 ${r["price"]:.1f} {r.get("mom30",0):+.1f}% 52w{r.get("p52",0):.0f}%')
    except:
        pass
    lines.append('')
    
    # 3. 状态
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone(timedelta(hours=8)))
    us_open = now.replace(hour=21, minute=30, second=0, microsecond=0)
    if now > us_open:
        lines.append('盘中')
    else:
        remaining = (us_open - now).total_seconds() / 60
        lines.append(f'距开盘{int(remaining)}分钟')
    
    lines.append('📡 数据源: yfinance | 143只美股池 | V4.2比例扣分')
    return '\n'.join(lines)


if __name__ == '__main__':
    if '--us' in sys.argv:
        report = format_us_report()
        print(report)
        try:
            sys.path.insert(0, os.path.join(ROOT, 'scripts'))
            from notify import send
            send(report)
        except:
            pass
    elif '--defense' in sys.argv:
        print(get_defensive_suggestion())
    else:
        print("用法:")
        print("  --us       推送美股持仓日报")
        print("  --defense  查看防守配置建议")
