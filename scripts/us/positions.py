#!/usr/bin/env python3
"""
持仓管理工具
用户手动录入实际持仓，系统自动追踪盈亏和模型建议

用法:
    python3 positions.py add TICKER PRICE QTY MODEL     # 添加持仓
    python3 positions.py list                            # 查看所有持仓
    python3 positions.py check                           # 检查持仓vs模型建议
    python3 positions.py sell TICKER [PRICE]             # 卖出（标记）
"""
import json, os, sys, argparse
from datetime import datetime
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POS_FILE = os.path.join(ROOT, 'output', 'positions.json')
DATA_DIR = os.path.join(ROOT, 'data', 'us')

def load_positions():
    if os.path.exists(POS_FILE):
        with open(POS_FILE) as f:
            return json.load(f)
    return {'positions': [], 'history': []}

def save_positions(data):
    os.makedirs(os.path.dirname(POS_FILE), exist_ok=True)
    with open(POS_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def get_current_price(ticker):
    try:
        df = pd.read_parquet(os.path.join(DATA_DIR, 'us_hist_yf_10y.parquet'))
        df = df.rename(columns={'ticker': 'sym'})
        latest = df.groupby('sym').last().reset_index()
        row = latest[latest['sym'] == ticker]
        if len(row) > 0:
            return float(row['close'].iloc[0])
    except:
        pass
    return None

def get_model_advice(ticker):
    """检查模型对该股票的最新评分"""
    advice = {}
    for model_file, model_name, hold_days in [
        ('v6_latest.json', '🛡️ 蓝盾V6', 20),
        ('v11_latest.json', '🎯 绿箭V11', 5)
    ]:
        path = os.path.join(ROOT, 'output', model_file)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                for pick in data.get('picks', []):
                    if pick.get('ticker') == ticker:
                        advice[model_name] = {
                            'score': pick.get('pred_rank', 0),
                            'signal': pick.get('signal', '🔴'),
                            'hold_days': hold_days,
                            'timestamp': data.get('timestamp', '')
                        }
            except:
                pass
    return advice

def add_position(ticker, price, qty, model='manual'):
    data = load_positions()
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 检查是否已有该股票的活跃持仓
    existing = [p for p in data['positions'] 
                if p['ticker'] == ticker and p['status'] == 'active']
    
    if existing:
        # 加仓
        pos = existing[0]
        old_qty = pos['qty']
        old_cost = pos['entry_price'] * old_qty
        new_cost = price * qty
        pos['qty'] += qty
        pos['entry_price'] = (old_cost + new_cost) / pos['qty']
        pos['avg_cost'] = pos['entry_price'] * pos['qty']
        pos['updated'] = today
        print(f"✅ 加仓 {ticker}: {old_qty}→{pos['qty']}股, 均价${pos['entry_price']:.2f}")
    else:
        # 新建持仓
        hold_days = 20 if '蓝盾' in model or 'shield' in model else 5 if '绿箭' in model or 'arrow' in model else 20
        stop_loss = -0.15 if hold_days == 20 else -0.10
        
        pos = {
            'ticker': ticker,
            'entry_date': today,
            'entry_price': price,
            'qty': qty,
            'avg_cost': price * qty,
            'model': model,
            'hold_days': hold_days,
            'stop_loss_pct': stop_loss,
            'stop_loss_price': round(price * (1 + stop_loss), 2),
            'expiry_date': (datetime.now() + __import__('datetime').timedelta(days=hold_days)).strftime('%Y-%m-%d'),
            'status': 'active',
            'current_price': price,
            'pnl_pct': 0.0,
            'pnl_usd': 0.0,
        }
        data['positions'].append(pos)
        print(f"✅ 新增持仓 {ticker}: {qty}股 @ ${price:.2f}, {model}")
        print(f"   止损: ${pos['stop_loss_price']:.2f} ({stop_loss*100:.0f}%)")
        print(f"   到期: {pos['expiry_date']} ({hold_days}天)")
    
    save_positions(data)
    return pos

def list_positions():
    data = load_positions()
    active = [p for p in data['positions'] if p['status'] == 'active']
    
    if not active:
        print("📭 当前无活跃持仓")
        return
    
    print(f"\n📊 活跃持仓 ({len(active)}只)")
    print("="*70)
    
    total_cost = 0
    total_value = 0
    
    for pos in active:
        ticker = pos['ticker']
        current = get_current_price(ticker)
        if current:
            pos['current_price'] = current
            pnl_pct = (current - pos['entry_price']) / pos['entry_price'] * 100
            pnl_usd = (current - pos['entry_price']) * pos['qty']
            pos['pnl_pct'] = round(pnl_pct, 2)
            pos['pnl_usd'] = round(pnl_usd, 2)
        
        entry = pos['entry_price']
        qty = pos['qty']
        cost = entry * qty
        value = (current or entry) * qty
        total_cost += cost
        total_value += value
        
        pnl = pos.get('pnl_pct', 0)
        pnl_emoji = '🟢' if pnl >= 0 else '🔴'
        
        print(f"\n{ticker} ({pos.get('model','manual')})")
        print(f"  入场: ${entry:.2f} × {qty}股 = ${cost:.0f}")
        print(f"  当前: ${current:.2f if current else 'N/A'} | 盈亏: {pnl_emoji} {pnl:+.1f}% (${pos.get('pnl_usd',0):+.0f})")
        print(f"  止损: ${pos['stop_loss_price']:.2f} | 到期: {pos['expiry_date']}")
        
        # 检查模型建议
        advice = get_model_advice(ticker)
        if advice:
            for model, info in advice.items():
                print(f"  模型: {model} 评分{info['score']:.3f} {info['signal']}")
        else:
            print(f"  模型: 未在今日推荐中")
    
    if total_cost > 0:
        total_pnl = (total_value - total_cost) / total_cost * 100
        print(f"\n{'='*70}")
        print(f"总成本: ${total_cost:,.0f} | 总市值: ${total_value:,.0f} | 总盈亏: {total_pnl:+.1f}%")

def sell_position(ticker, price=None):
    data = load_positions()
    active = [p for p in data['positions'] if p['ticker'] == ticker and p['status'] == 'active']
    
    if not active:
        print(f"❌ 未找到{ticker}的活跃持仓")
        return
    
    pos = active[0]
    if price is None:
        price = get_current_price(ticker) or pos['entry_price']
    
    pos['status'] = 'closed'
    pos['exit_price'] = price
    pos['exit_date'] = datetime.now().strftime('%Y-%m-%d')
    pos['pnl_pct'] = round((price - pos['entry_price']) / pos['entry_price'] * 100, 2)
    pos['pnl_usd'] = round((price - pos['entry_price']) * pos['qty'], 2)
    
    # 移到历史
    data['history'].append(pos)
    data['positions'] = [p for p in data['positions'] if not (p['ticker'] == ticker and p['status'] == 'closed')]
    
    save_positions(data)
    pnl_emoji = '🟢' if pos['pnl_pct'] >= 0 else '🔴'
    print(f"✅ 卖出 {ticker}: ${price:.2f} | 盈亏: {pnl_emoji} {pos['pnl_pct']:+.1f}% (${pos['pnl_usd']:+.0f})")

def check_all():
    """综合检查：持仓+模型+VIX"""
    data = load_positions()
    active = [p for p in data['positions'] if p['status'] == 'active']
    
    print("\n🔍 综合持仓检查")
    print("="*50)
    
    # VIX
    try:
        import yfinance as yf
        vix = yf.Ticker('^VIX').history(period='1d')['Close'].iloc[-1]
        if vix > 35: print(f"🔴🔴 VIX={vix:.1f} 恐慌！清仓")
        elif vix > 25: print(f"🟠 VIX={vix:.1f} 警戒，减仓50%")
        elif vix > 20: print(f"🟡 VIX={vix:.1f} 注意，收紧止损")
        else: print(f"🟢 VIX={vix:.1f} 正常")
    except:
        print("⚠️ VIX获取失败")
    
    if not active:
        print("\n📭 无持仓")
        return
    
    print(f"\n📊 持仓检查 ({len(active)}只):")
    for pos in active:
        ticker = pos['ticker']
        current = get_current_price(ticker)
        if current is None:
            continue
        
        entry = pos['entry_price']
        pnl = (current - entry) / entry * 100
        stop = pos['stop_loss_price']
        expiry = pos['expiry_date']
        hold_days = pos['hold_days']
        
        from datetime import datetime as dt
        today = dt.now()
        try:
            expiry_dt = dt.strptime(expiry, '%Y-%m-%d')
            days_left = (expiry_dt - today).days
        except:
            days_left = hold_days
        
        # 状态判断
        if current <= stop:
            status = f"🔴 触发止损！当前${current:.2f} ≤ 止损${stop:.2f} → 卖出"
        elif days_left <= 0:
            status = f"🟡 到期！持有{hold_days}天已满 → 考虑卖出"
        elif pnl < -10:
            status = f"🟠 亏损{pnl:.1f}%，接近止损 → 密切关注"
        elif pnl > 20:
            status = f"🟢 盈利{pnl:.1f}%，继续持有"
        else:
            status = f"⚪ 持有中，还剩{days_left}天"
        
        advice = get_model_advice(ticker)
        model_note = ""
        if advice:
            for m, info in advice.items():
                model_note = f" | {m}评分{info['score']:.3f}{info['signal']}"
        
        print(f"\n  {ticker}: ${entry:.2f}→${current:.2f} ({pnl:+.1f}%)")
        print(f"  {status}{model_note}")

def main():
    parser = argparse.ArgumentParser(description='持仓管理')
    sub = parser.add_subparsers(dest='cmd')
    
    add_p = sub.add_parser('add', help='添加持仓')
    add_p.add_argument('ticker', help='股票代码')
    add_p.add_argument('price', type=float, help='买入价格')
    add_p.add_argument('qty', type=int, help='数量')
    add_p.add_argument('--model', default='manual', help='模型来源')
    
    sub.add_parser('list', help='查看持仓')
    sub.add_parser('check', help='综合检查')
    
    sell_p = sub.add_parser('sell', help='卖出')
    sell_p.add_argument('ticker', help='股票代码')
    sell_p.add_argument('price', type=float, nargs='?', help='卖出价格')
    
    args = parser.parse_args()
    
    if args.cmd == 'add':
        add_position(args.ticker, args.price, args.qty, args.model)
    elif args.cmd == 'list':
        list_positions()
    elif args.cmd == 'check':
        check_all()
    elif args.cmd == 'sell':
        sell_position(args.ticker, args.price)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
