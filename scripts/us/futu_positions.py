#!/usr/bin/env python3
"""
Futu持仓查询标准化工具
一键获取：持仓明细+买入日期+当前盈亏+模型建议+VIX状态+持仓天数

用法:
    python3 futu_positions.py              # 完整报告
    python3 futu_positions.py --json       # JSON输出
    python3 futu_positions.py --brief      # 简要摘要
"""
import json, os, sys, time
from datetime import datetime, timedelta
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(ROOT, 'output')

# ════════════════════════════════════════════════════════════
#  Futu 接口
# ════════════════════════════════════════════════════════════

def get_futu_positions():
    """从Futu OpenD获取当前持仓"""
    from futu import OpenSecTradeContext, TrdMarket
    trd_ctx = OpenSecTradeContext(host='127.0.0.1', port=11111, filter_trdmarket=TrdMarket.US)
    ret, data = trd_ctx.position_list_query()
    trd_ctx.close()
    
    if ret != 0 or len(data) == 0:
        return []
    
    positions = []
    for _, row in data.iterrows():
        positions.append({
            'code': row['code'].replace('US.', ''),
            'name': row['stock_name'],
            'qty': int(row['qty']),
            'cost_price': round(float(row['cost_price']), 3),
            'current_price': round(float(row.get('nominal_price', row['cost_price'])), 2),
            'market_val': round(float(row['market_val']), 2),
            'pnl_pct': round(float(row['pl_ratio']), 2),
            'pnl_usd': round(float(row['pl_val']), 2),
            'today_pl': round(float(row.get('today_pl_val', 0)), 2),
        })
    return positions


def get_futu_buy_dates():
    """从Futu历史订单获取买入日期"""
    from futu import OpenSecTradeContext, TrdMarket
    trd_ctx = OpenSecTradeContext(host='127.0.0.1', port=11111, filter_trdmarket=TrdMarket.US)
    ret, data = trd_ctx.history_order_list_query()
    trd_ctx.close()
    
    if ret != 0 or len(data) == 0:
        return {}
    
    # 只看买入成交的订单
    buys = data[data['trd_side'] == 'BUY']
    
    # 每只股票取最近的买入日期
    buy_dates = {}
    for _, row in buys.iterrows():
        code = row['code'].replace('US.', '')
        create_time = row['create_time'][:10]
        
        if code not in buy_dates or create_time > buy_dates[code]:
            buy_dates[code] = create_time
    
    return buy_dates


def get_vix():
    """获取VIX"""
    try:
        import yfinance as yf
        vix = yf.Ticker('^VIX').history(period='1d')['Close'].iloc[-1]
        return float(vix)
    except:
        return None


def get_model_recommendations():
    """获取模型最新推荐"""
    recs = {}
    
    for model_file, model_name in [
        ('v6_latest.json', '🛡️ 蓝盾V6'),
        ('v11_latest.json', '🎯 绿箭V11')
    ]:
        path = os.path.join(OUTPUT_DIR, model_file)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                for pick in data.get('picks', []):
                    ticker = pick.get('ticker', '')
                    if ticker not in recs:
                        recs[ticker] = []
                    recs[ticker].append({
                        'model': model_name,
                        'score': pick.get('pred_rank', 0),
                        'signal': pick.get('signal', '🔴'),
                    })
            except:
                pass
    
    return recs


# ════════════════════════════════════════════════════════════
#  综合报告
# ════════════════════════════════════════════════════════════

def generate_full_report():
    """生成完整持仓报告"""
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')
    
    print("🔍 正在查询Futu持仓...", flush=True)
    positions = get_futu_positions()
    
    if not positions:
        print("📭 当前无持仓")
        return None
    
    print("📅 正在获取买入日期...", flush=True)
    buy_dates = get_futu_buy_dates()
    
    print("📊 正在获取模型推荐...", flush=True)
    model_recs = get_model_recommendations()
    
    vix = get_vix()
    
    # 组装报告
    report = {
        'timestamp': today_str,
        'vix': vix,
        'total_positions': len(positions),
        'total_value': 0,
        'total_cost': 0,
        'total_pnl': 0,
        'positions': []
    }
    
    for pos in positions:
        code = pos['code']
        buy_date_str = buy_dates.get(code, '未知')
        
        # 计算持仓天数
        days_held = None
        if buy_date_str != '未知':
            try:
                buy_dt = datetime.strptime(buy_date_str, '%Y-%m-%d')
                days_held = (today - buy_dt).days
            except:
                pass
        
        # 判断属于哪个模型
        model = 'manual'
        if pos['cost_price'] > 10:
            model = '🛡️ 蓝盾V6'
            hold_days = 20
            stop_loss_pct = -15
        else:
            model = '🎯 绿箭V11'
            hold_days = 5
            stop_loss_pct = -10
        
        stop_loss_price = round(pos['cost_price'] * (1 + stop_loss_pct / 100), 2)
        
        # 止损检查
        stop_triggered = pos['current_price'] <= stop_loss_price if pos['qty'] > 0 else False
        
        # 到期检查
        days_to_expiry = None
        expiry_date = None
        if days_held is not None:
            expiry_date = (datetime.strptime(buy_date_str, '%Y-%m-%d') + timedelta(days=hold_days)).strftime('%Y-%m-%d')
            days_to_expiry = max(0, hold_days - days_held)
        
        # 模型建议
        rec = model_recs.get(code, [])
        
        pos_data = {
            'code': code,
            'name': pos['name'],
            'model': model,
            'qty': pos['qty'],
            'cost_price': pos['cost_price'],
            'current_price': pos['current_price'],
            'pnl_pct': pos['pnl_pct'],
            'pnl_usd': pos['pnl_usd'],
            'today_pl': pos['today_pl'],
            'buy_date': buy_date_str,
            'days_held': days_held,
            'hold_days': hold_days,
            'expiry_date': expiry_date,
            'days_to_expiry': days_to_expiry,
            'stop_loss_pct': stop_loss_pct,
            'stop_loss_price': stop_loss_price,
            'stop_triggered': stop_triggered,
            'model_recs': rec,
        }
        
        report['positions'].append(pos_data)
        report['total_value'] += pos['market_val']
        report['total_cost'] += pos['cost_price'] * pos['qty']
        report['total_pnl'] += pos['pnl_usd']
    
    if report['total_cost'] > 0:
        report['total_pnl_pct'] = round(report['total_pnl'] / report['total_cost'] * 100, 2)
    else:
        report['total_pnl_pct'] = 0
    
    # 按盈亏排序
    report['positions'].sort(key=lambda x: x['pnl_pct'], reverse=True)
    
    return report


def print_report(report):
    """打印格式化报告"""
    if not report:
        return
    
    today_str = report['timestamp']
    vix = report['vix']
    
    # VIX状态
    if vix:
        if vix > 35: vix_str = f"🔴🔴 VIX={vix:.1f} 恐慌！"
        elif vix > 25: vix_str = f"🟠 VIX={vix:.1f} 警戒"
        elif vix > 20: vix_str = f"🟡 VIX={vix:.1f} 注意"
        else: vix_str = f"🟢 VIX={vix:.1f} 正常"
    else:
        vix_str = "⚠️ VIX获取失败"
    
    print(f"\n{'='*60}")
    print(f"📊 Hermes持仓报告 | {today_str} | {vix_str}")
    print(f"{'='*60}")
    print(f"总持仓: {report['total_positions']}只 | 市值: ${report['total_value']:,.0f} | 盈亏: ${report['total_pnl']:+,.0f} ({report['total_pnl_pct']:+.1f}%)")
    print(f"{'='*60}")
    
    for pos in report['positions']:
        pnl_emoji = '🟢' if pos['pnl_pct'] >= 0 else '🔴'
        days_info = f"已持{pos['days_held']}天" if pos['days_held'] is not None else ""
        expiry_info = f"还剩{pos['days_to_expiry']}天" if pos['days_to_expiry'] is not None else ""
        
        print(f"\n{pos['code']} ({pos['name']})")
        print(f"  {pos['model']} | {pos['qty']}股 @ ${pos['cost_price']:.2f}")
        print(f"  当前: ${pos['current_price']:.2f} | 盈亏: {pnl_emoji} {pos['pnl_pct']:+.1f}% (${pos['pnl_usd']:+.0f})")
        print(f"  买入: {pos['buy_date']} | {days_info} | {expiry_info}")
        print(f"  止损: ${pos['stop_loss_price']:.2f} ({pos['stop_loss_pct']:+.0f}%) | 到期: {pos['expiry_date']}")
        
        if pos['stop_triggered']:
            print(f"  ⚠️ 触发止损！考虑卖出")
        
        if pos['model_recs']:
            for rec in pos['model_recs']:
                print(f"  模型: {rec['model']} 评分{rec['score']:.3f} {rec['signal']}")
    
    print(f"\n{'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Futu持仓查询')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--brief', action='store_true')
    args = parser.parse_args()
    
    report = generate_full_report()
    
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    elif args.brief:
        if report:
            print(f"持仓{report['total_positions']}只 | 市值${report['total_value']:,.0f} | 盈亏${report['total_pnl']:+,.0f}({report['total_pnl_pct']:+.1f}%)")
            stops = [p for p in report['positions'] if p['stop_triggered']]
            if stops:
                print(f"⚠️ 止损预警: {', '.join(p['code'] for p in stops)}")
    else:
        print_report(report)
    
    # 保存到文件
    if report:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, 'futu_positions.json'), 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)


if __name__ == '__main__':
    main()
