#!/usr/bin/env python3
"""
推荐追踪系统
1. 读取蓝盾V8和绿箭V12的最新评分快照
2. 保存为带元数据的推荐记录
3. 每日更新当前价格、计算盈亏、标记过期
4. 输出追踪数据供看板使用

用法:
    python3 track_recommendations.py --save     # 保存新推荐快照
    python3 track_recommendations.py --update   # 更新价格和盈亏
    python3 track_recommendations.py --both     # 保存+更新
"""
import json, os, sys, argparse, time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, 'data', 'us')
OUTPUT_DIR = os.path.join(ROOT, 'output')
TRACK_FILE = os.path.join(OUTPUT_DIR, 'recommendations.json')
LATEST_DIR = OUTPUT_DIR  # v6_latest.json, v11_latest.json

# 推荐配置
MODEL_CONFIG = {
    'blueshield_v8': {
        'name': '🛡️ 蓝盾V8',
        'hold_days': 20,
        'stop_loss': -0.15,  # -15% 止损
        'position_size': '按仓位比例',
        'universe': '>$10',
    },
    'arrow_v12': {
        'name': '🎯 绿箭V12',
        'hold_days': 5,
        'stop_loss': -0.10,  # -10% 止损
        'position_size': '$1000/只',
        'universe': '$1-$10',
    }
}


def load_recommendations():
    """加载已有推荐记录"""
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE) as f:
            return json.load(f)
    return {'recommendations': [], 'stats': {}}


def save_recommendations(data):
    """保存推荐记录"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TRACK_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_latest(model_key):
    """加载最新评分"""
    file_map = {
        'blueshield_v8': 'v6_latest.json',
        'arrow_v12': 'v11_latest.json'
    }
    path = os.path.join(LATEST_DIR, file_map.get(model_key, ''))
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def get_current_prices(tickers):
    """从本地数据获取最新价格"""
    try:
        df = pd.read_parquet(os.path.join(DATA_DIR, 'us_hist_yf_10y.parquet'))
        df = df.rename(columns={'ticker': 'sym'})
        # 取每只股票最新价格
        latest = df.groupby('sym').last().reset_index()
        prices = {}
        for t in tickers:
            row = latest[latest['sym'] == t]
            if len(row) > 0:
                prices[t] = float(row['close'].iloc[0])
        return prices
    except Exception as e:
        print(f"⚠️ 获取价格失败: {e}", flush=True)
        return {}


def save_snapshot(model_key):
    """保存新推荐快照"""
    data = load_recommendations()
    latest = load_latest(model_key)
    if not latest:
        print(f"⚠️ {model_key} 无最新评分", flush=True)
        return 0
    
    config = MODEL_CONFIG.get(model_key, {})
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 检查今天是否已经保存过该模型的推荐
    existing_dates = [r['date'] for r in data['recommendations'] 
                     if r['model'] == model_key]
    if today in existing_dates:
        print(f"ℹ️ {model_key} 今天已保存过推荐，跳过", flush=True)
        return 0
    
    # 获取当前价格
    tickers = [p['ticker'] for p in latest.get('picks', [])]
    prices = get_current_prices(tickers)
    
    new_count = 0
    for pick in latest.get('picks', []):
        ticker = pick['ticker']
        price = prices.get(ticker, pick.get('price', 0))
        score = pick.get('pred_rank', 0)
        signal = pick.get('signal', '⚪')
        
        # 只保存有信号的推荐（🟡及以上，三层过滤后）
        if signal in ['🔴', '⚪', '']:
            continue
        
        # 计算到期日
        entry_date = datetime.now()
        hold_days = config.get('hold_days', 20)
        expiry_date = entry_date + timedelta(days=hold_days)
        
        # 计算止损价
        stop_loss_pct = config.get('stop_loss', -0.10)
        stop_loss_price = round(price * (1 + stop_loss_pct), 2)
        
        record = {
            'id': f"{model_key}_{today}_{ticker}",
            'model': model_key,
            'model_name': config.get('name', model_key),
            'ticker': ticker,
            'entry_date': today,
            'entry_price': price,
            'current_price': price,  # 初始等于入场价
            'score': round(score, 4),
            'signal': signal,
            'hold_days': config.get('hold_days', 20),
            'stop_loss_pct': stop_loss_pct,
            'stop_loss_price': stop_loss_price,
            'expiry_date': expiry_date.strftime('%Y-%m-%d'),
            'pnl_pct': 0.0,
            'pnl_usd': 0.0,
            'status': 'active',  # active / expired / stopped
            'exit_price': None,
            'exit_date': None,
            'exit_reason': None,
            'universe': config.get('universe', ''),
            'position_size': config.get('position_size', ''),
        }
        data['recommendations'].append(record)
        new_count += 1
    
    save_recommendations(data)
    print(f"✅ {model_key}: 保存{new_count}条新推荐", flush=True)
    return new_count


def update_prices():
    """更新所有活跃推荐的当前价格和盈亏"""
    data = load_recommendations()
    active = [r for r in data['recommendations'] if r['status'] == 'active']
    
    if not active:
        print("ℹ️ 无活跃推荐需要更新", flush=True)
        return 0
    
    # 获取所有活跃ticker的最新价格
    tickers = list(set(r['ticker'] for r in active))
    prices = get_current_prices(tickers)
    
    today = datetime.now().strftime('%Y-%m-%d')
    updated = 0
    
    for rec in data['recommendations']:
        if rec['status'] != 'active':
            continue
        
        ticker = rec['ticker']
        if ticker not in prices:
            continue
        
        new_price = prices[ticker]
        rec['current_price'] = round(new_price, 2)
        
        # 计算盈亏
        entry = rec['entry_price']
        if entry > 0:
            pnl_pct = (new_price - entry) / entry
            rec['pnl_pct'] = round(pnl_pct * 100, 2)
        
        # 检查止损
        if new_price <= rec['stop_loss_price']:
            rec['status'] = 'stopped'
            rec['exit_price'] = new_price
            rec['exit_date'] = today
            rec['exit_reason'] = f"止损触发 (≤${rec['stop_loss_price']})"
        
        # 检查到期
        elif today >= rec['expiry_date']:
            rec['status'] = 'expired'
            rec['exit_price'] = new_price
            rec['exit_date'] = today
            rec['exit_reason'] = f"持有期满 ({rec['hold_days']}天)"
        
        updated += 1
    
    save_recommendations(data)
    print(f"✅ 更新{updated}条活跃推荐", flush=True)
    return updated


def compute_stats():
    """计算推荐统计"""
    data = load_recommendations()
    recs = data['recommendations']
    
    if not recs:
        return {}
    
    stats = {
        'total': len(recs),
        'active': len([r for r in recs if r['status'] == 'active']),
        'expired': len([r for r in recs if r['status'] == 'expired']),
        'stopped': len([r for r in recs if r['status'] == 'stopped']),
        'winners': len([r for r in recs if r['status'] != 'active' and r.get('pnl_pct', 0) > 0]),
        'losers': len([r for r in recs if r['status'] != 'active' and r.get('pnl_pct', 0) < 0]),
    }
    
    # 按模型统计
    for model_key in MODEL_CONFIG:
        model_recs = [r for r in recs if r['model'] == model_key]
        closed = [r for r in model_recs if r['status'] != 'active']
        if closed:
            pnls = [r.get('pnl_pct', 0) for r in closed]
            stats[model_key] = {
                'total': len(model_recs),
                'active': len([r for r in model_recs if r['status'] == 'active']),
                'avg_pnl': round(np.mean(pnls), 2),
                'win_rate': round(len([p for p in pnls if p > 0]) / len(pnls) * 100, 1),
                'best': round(max(pnls), 2),
                'worst': round(min(pnls), 2),
            }
    
    # 信号级别统计
    for sig in ['🟢🟢', '🟢', '🟡']:
        sig_recs = [r for r in recs if r.get('signal') == sig and r['status'] != 'active']
        if sig_recs:
            pnls = [r.get('pnl_pct', 0) for r in sig_recs]
            stats[f'signal_{sig}'] = {
                'count': len(sig_recs),
                'avg_pnl': round(np.mean(pnls), 2),
                'win_rate': round(len([p for p in pnls if p > 0]) / len(pnls) * 100, 1),
            }
    
    data['stats'] = stats
    save_recommendations(data)
    return stats


def get_dashboard_data():
    """获取看板展示数据"""
    data = load_recommendations()
    recs = data['recommendations']
    
    # 分组
    active = sorted([r for r in recs if r['status'] == 'active'], 
                   key=lambda x: x.get('score', 0), reverse=True)
    recent_closed = sorted([r for r in recs if r['status'] != 'active'], 
                          key=lambda x: x.get('exit_date', ''), reverse=True)[:20]
    
    return {
        'active': active,
        'recent_closed': recent_closed,
        'stats': data.get('stats', {}),
        'updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }


def main():
    parser = argparse.ArgumentParser(description='推荐追踪系统')
    parser.add_argument('--save', action='store_true', help='保存新推荐快照')
    parser.add_argument('--update', action='store_true', help='更新价格和盈亏')
    parser.add_argument('--both', action='store_true', help='保存+更新')
    parser.add_argument('--stats', action='store_true', help='显示统计')
    parser.add_argument('--json', action='store_true', help='JSON输出')
    args = parser.parse_args()
    
    if args.save or args.both:
        save_snapshot('blueshield_v8')
        save_snapshot('arrow_v12')
    
    if args.update or args.both:
        update_prices()
    
    if args.stats or args.json:
        stats = compute_stats()
        if args.json:
            print(json.dumps(get_dashboard_data(), indent=2, default=str))
        else:
            print("\n📊 推荐追踪统计")
            print("="*40)
            for k, v in stats.items():
                if isinstance(v, dict):
                    print(f"\n{k}:")
                    for kk, vv in v.items():
                        print(f"  {kk}: {vv}")
                else:
                    print(f"{k}: {v}")
    elif not (args.save or args.update or args.both):
        # 默认：保存+更新+统计
        save_snapshot('blueshield_v8')
        save_snapshot('arrow_v12')
        update_prices()
        stats = compute_stats()
        print(f"\n✅ 完成: {stats.get('total',0)}条记录, {stats.get('active',0)}条活跃")


if __name__ == '__main__':
    main()
