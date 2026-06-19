#!/usr/bin/env python3
"""Generate synthetic recommendation tracking data from model validation metrics.
Uses actual OOS Sharpe/Return/DD to create realistic equity curves.
"""
import json, os, random, math
from datetime import datetime, timedelta
from collections import defaultdict

ROOT = '/home/hermes/.hermes/openclaw-archive'
random.seed(42)

with open(os.path.join(ROOT, 'models/us/blueshield_v6_meta.json')) as f:
    shield_meta = json.load(f)
with open(os.path.join(ROOT, 'models/us/arrow_v11_meta.json')) as f:
    arrow_meta = json.load(f)

# === Stock Universes ===
SHIELD_STOCKS = ['AAPL','MSFT','NVDA','GOOGL','META','AMZN','TSLA','JPM','V','UNH',
    'JNJ','WMT','PG','MA','HD','COST','ABBV','MRK','PEP','KO','AVGO','LLY','TMO',
    'CSCO','ACN','MCD','CRM','AMD','TXN','UNP','HON','LOW','AMGN','INTC','IBM',
    'CAT','GE','BA','GS','BLK','AXP','ISRG','ANET','NET','COHR','ASML','CARR',
    'DDOG','CRWD','ZS','PANW','FTNT','NOW','SNPS','CDNS','LRCX','AMAT','KLAC']

ARROW_STOCKS = ['PPBT','NYXH','NXTC','NGEN','FATE','ATOS','CRMT','KUST','BDTX','SY',
    'NEOV','ZEPP','CHRD','SKIL','TROO','DGXX','BLDP','IPWR','CRVO','LAR',
    'CLOV','WISH','BARK','SKLZ','OPEN','SOFI','HOOD','AFRM','UPST','LMND',
    'DNA','PLTR','RBLX','COIN','MARA','RIOT','CLSK','BITF']

def generate_model_data(meta, stocks, model_name):
    """Generate realistic tracking data using actual model validation metrics."""
    val = meta.get('validation', {})
    
    # Extract actual model performance
    if model_name == 'shield':
        annual_ret = val.get('oos_annual_return', 30.1) / 100  # 0.301
        sharpe = val.get('oos_sharpe', 1.44)
        max_dd = abs(val.get('oos_max_dd', -11.1)) / 100  # 0.111
        win_rate = val.get('oos_win_rate', 60) / 100
        hold_days = 20
        top_n = 15
    else:
        # Arrow: 5.56% per 5-day trade, ~50 trades/year → annual ~55%
        per_trade = val.get('oos_avg_net_per_5d', val.get('wf_avg_net', 5.56)) / 100
        annual_ret = per_trade * 50  # ~50 trades/year
        sharpe = val.get('oos_sharpe', 2.18)
        max_dd = 0.12
        win_rate = val.get('oos_win_rate', 50) / 100
        hold_days = 5
        top_n = 5
    
    # Derive daily return and volatility from annual metrics
    trading_days = 252
    daily_ret = annual_ret / trading_days
    daily_vol = (annual_ret / sharpe) / math.sqrt(trading_days)
    
    # Generate daily equity curve (2 years)
    start = datetime(2024, 7, 1)
    end = datetime(2026, 6, 15)
    
    equity = [100.0]
    dates = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # trading days only
            dates.append(current.strftime('%Y-%m-%d'))
            daily_r = random.gauss(daily_ret, daily_vol)
            equity.append(equity[-1] * (1 + daily_r))
        current += timedelta(days=1)
    
    # Calculate actual max drawdown from equity curve
    peak = equity[0]
    max_dd_actual = 0
    for v in equity:
        if v > peak: peak = v
        dd = (peak - v) / peak
        if dd > max_dd_actual: max_dd_actual = dd
    
    # Generate individual trade recommendations
    recs = []
    current = start
    trade_count = 0
    while current <= end:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue
        
        # ~70% of days have signals
        if random.random() < 0.7:
            n_picks = random.randint(1, min(top_n, 3))
            picks = random.sample(stocks, min(n_picks, len(stocks)))
            
            for ticker in picks:
                entry_price = random.uniform(10, 400) if model_name == 'shield' else random.uniform(1, 10)
                
                is_win = random.random() < win_rate
                if model_name == 'shield':
                    if is_win:
                        ret = random.gauss(0.03, 0.06)
                    else:
                        ret = random.gauss(-0.02, 0.05)
                    ret = max(-0.15, min(0.30, ret))
                else:
                    if is_win:
                        ret = random.gauss(0.08, 0.12)
                    else:
                        ret = random.gauss(-0.04, 0.08)
                    ret = max(-0.10, min(0.50, ret))
                
                exit_price = entry_price * (1 + ret)
                score = random.uniform(0.55, 0.65) if model_name == 'shield' else random.uniform(0.58, 0.85)
                
                recs.append({
                    'model': f'{model_name}_{"v6" if model_name == "shield" else "v11"}',
                    'ticker': ticker,
                    'entry_date': current.strftime('%Y-%m-%d'),
                    'entry_price': round(entry_price, 2),
                    'exit_date': (current + timedelta(days=hold_days)).strftime('%Y-%m-%d'),
                    'exit_price': round(exit_price, 2),
                    'hold_days': hold_days,
                    'return_pct': round(ret * 100, 2),
                    'score': round(score, 4),
                    'signal': '🟢🟢' if (model_name == 'shield' and score > 0.60) or (model_name == 'arrow' and score > 0.70) else '🟢',
                    'status': 'completed'
                })
        
        current += timedelta(days=1)
    
    # Compute stats
    returns = [r['return_pct'] for r in recs]
    wins = [r for r in recs if r['return_pct'] > 0]
    losses = [r for r in recs if r['return_pct'] <= 0]
    
    monthly = defaultdict(lambda: {'returns': [], 'count': 0})
    for r in recs:
        month = r['entry_date'][:7]
        monthly[month]['returns'].append(r['return_pct'])
        monthly[month]['count'] += 1
    
    monthly_stats = {}
    for m, d in sorted(monthly.items()):
        rets = d['returns']
        monthly_stats[m] = {
            'count': d['count'],
            'avg_return': round(sum(rets)/len(rets), 2),
            'win_rate': round(len([r for r in rets if r > 0])/len(rets)*100, 1),
            'total_return': round(sum(rets), 2)
        }
    
    avg_win = round(sum(r['return_pct'] for r in wins)/len(wins), 2) if wins else 0
    avg_loss = round(sum(r['return_pct'] for r in losses)/len(losses), 2) if losses else 0
    pf = round(abs(sum(r['return_pct'] for r in wins) / sum(r['return_pct'] for r in losses)), 2) if losses and sum(r['return_pct'] for r in losses) != 0 else 999
    
    # Downsample equity curve
    step = max(1, len(equity) // 80)
    eq_sampled = equity[::step]
    if eq_sampled[-1] != equity[-1]:
        eq_sampled.append(equity[-1])
    
    stats = {
        'total_trades': len(recs),
        'win_rate': round(len(wins)/len(recs)*100, 1) if recs else 0,
        'avg_return': round(sum(returns)/len(returns), 2) if returns else 0,
        'median_return': round(sorted(returns)[len(returns)//2], 2) if returns else 0,
        'best_trade': round(max(returns), 2) if returns else 0,
        'worst_trade': round(min(returns), 2) if returns else 0,
        'total_return': round(sum(returns), 2),
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': pf,
        'max_drawdown': round(max_dd_actual * 100, 2),
        'annual_return': round(annual_ret * 100, 1),
        'sharpe': sharpe,
        'monthly': monthly_stats,
        'equity_curve': [round(v, 2) for v in eq_sampled],
    }
    
    return recs, stats

# === Generate ===
print("Generating Shield V6 tracking...")
shield_recs, shield_stats = generate_model_data(shield_meta, SHIELD_STOCKS, 'shield')
print(f"  {len(shield_recs)} trades | WR {shield_stats['win_rate']}% | Avg {shield_stats['avg_return']}%")
print(f"  Equity: {shield_stats['equity_curve'][0]:.0f} → {shield_stats['equity_curve'][-1]:.0f} ({(shield_stats['equity_curve'][-1]/100-1)*100:+.1f}%)")

print("Generating Arrow V11 tracking...")
arrow_recs, arrow_stats = generate_model_data(arrow_meta, ARROW_STOCKS, 'arrow')
print(f"  {len(arrow_recs)} trades | WR {arrow_stats['win_rate']}% | Avg {arrow_stats['avg_return']}%")
print(f"  Equity: {arrow_stats['equity_curve'][0]:.0f} → {arrow_stats['equity_curve'][-1]:.0f} ({(arrow_stats['equity_curve'][-1]/100-1)*100:+.1f}%)")

output = {
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'note': 'Synthetic tracking based on model OOS validation metrics. Real tracking starts with daily snapshots.',
    'shield': {'recommendations': shield_recs, 'stats': shield_stats},
    'arrow': {'recommendations': arrow_recs, 'stats': arrow_stats}
}

out_path = os.path.join(ROOT, 'output/recommendations.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {os.path.getsize(out_path):,} bytes")

# Initialize snapshots
snap_dir = os.path.join(ROOT, 'output/snapshots')
os.makedirs(snap_dir, exist_ok=True)
today = datetime.now().strftime('%Y-%m-%d')
v6 = json.load(open(os.path.join(ROOT, 'output/v6_latest.json')))
v11 = json.load(open(os.path.join(ROOT, 'output/v11_latest.json')))
snap = {'date': today, 'shield_picks': v6.get('picks',[]), 'arrow_picks': v11.get('picks',[])}
with open(os.path.join(snap_dir, f'{today}.json'), 'w') as f:
    json.dump(snap, f, indent=2)
print(f"Snapshot: {snap_dir}/{today}.json")
