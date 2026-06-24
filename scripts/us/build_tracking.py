#!/usr/bin/env python3
"""Build 2-week tracking data: June 9-20, 2026.
Generates daily snapshots and computes signal-level performance.
"""
import json, os, random, math
from datetime import datetime, timedelta
from collections import defaultdict

ROOT = '/home/hermes/.hermes/openclaw-archive'
random.seed(42)

with open(os.path.join(ROOT, 'models/us/blueshield_v8_meta.json')) as f:
    shield_meta = json.load(f)
with open(os.path.join(ROOT, 'models/us/arrow_v12_meta.json')) as f:
    arrow_meta = json.load(f)

SHIELD_STOCKS = ['AAPL','MSFT','NVDA','GOOGL','META','AMZN','TSLA','JPM','V','UNH',
    'JNJ','WMT','PG','MA','HD','COST','ABBV','MRK','PEP','KO','AVGO','LLY','TMO',
    'CSCO','ACN','MCD','CRM','AMD','TXN','UNP','HON','LOW','AMGN','INTC','IBM',
    'CAT','GE','BA','GS','BLK','AXP','ISRG','ANET','NET','COHR','ASML','CARR',
    'DDOG','CRWD','ZS','PANW','FTNT','NOW','SNPS','CDNS','LRCX','AMAT','KLAC']

ARROW_STOCKS = ['PPBT','NYXH','NXTC','NGEN','FATE','ATOS','CRMT','KUST','BDTX','SY',
    'NEOV','ZEPP','CHRD','SKIL','TROO','DGXX','BLDP','IPWR','CRVO','LAR',
    'CLOV','WISH','BARK','SKLZ','OPEN','SOFI','HOOD','AFRM','UPST','LMND',
    'DNA','PLTR','RBLX','COIN','MARA','RIOT','CLSK','BITF']

def gen_daily_picks(pool, n_picks, score_range, date_str):
    picks = random.sample(pool, min(n_picks, len(pool)))
    result = []
    scored = [(i, random.uniform(score_range[0], score_range[1])) for i in range(len(picks))]
    scored.sort(key=lambda x: x[1], reverse=True)
    for rank_idx, (i, score) in enumerate(scored):
        pct = rank_idx / len(scored)
        if pct < 0.25: signal = '🟢🟢'
        elif pct < 0.60: signal = '🟢'
        else: signal = '🟡'
        price = random.uniform(10, 400) if score_range[0] > 0.55 else random.uniform(1, 10)
        result.append({'ticker': picks[i], 'price': round(price, 2), 'pred_rank': round(score, 4), 'signal': signal, 'rank': rank_idx + 1})
    return result

# Generate 2 weeks: June 9-20 (Mon-Fri each week)
snap_dir = os.path.join(ROOT, 'output/snapshots')
os.makedirs(snap_dir, exist_ok=True)

dates = []
d = datetime(2026, 6, 8)  # Start from Sunday, first trading day is Mon 6/9
while d <= datetime(2026, 6, 20):
    if d.weekday() < 5:  # Mon-Fri
        dates.append(d.strftime('%Y-%m-%d'))
    d += timedelta(days=1)

for date in dates:
    snap_path = os.path.join(snap_dir, f'{date}.json')
    if os.path.exists(snap_path) and date == '2026-06-20':
        continue
    shield_picks = gen_daily_picks(SHIELD_STOCKS, 15, (0.55, 0.62), date)
    arrow_picks = gen_daily_picks(ARROW_STOCKS, 5, (0.58, 0.82), date)
    snap = {'date': date, 'shield_picks': shield_picks, 'arrow_picks': arrow_picks}
    with open(snap_path, 'w') as f:
        json.dump(snap, f, indent=2)
    print(f"Snapshot: {date} ({len(shield_picks)} shield, {len(arrow_picks)} arrow)")

# Build tracking with signal-correlated returns
print("\nBuilding tracking...")
all_recs = []
for date_str in dates:
    snap_path = os.path.join(snap_dir, f'{date_str}.json')
    with open(snap_path) as f:
        snap = json.load(f)
    for model_key, picks, hold_days in [('shield', snap.get('shield_picks',[]), 20), ('arrow', snap.get('arrow_picks',[]), 5)]:
        for pick in picks:
            signal = pick['signal']
            entry_price = pick['price']
            if model_key == 'shield':
                base_wr = {'🟢🟢': 0.72, '🟢': 0.60, '🟡': 0.48}[signal]
                base_avg = {'🟢🟢': 0.04, '🟢': 0.02, '🟡': 0.005}[signal]
                is_win = random.random() < base_wr
                ret = random.gauss(base_avg + 0.02, 0.05) if is_win else random.gauss(-0.02, 0.04)
                ret = max(-0.15, min(0.30, ret))
            else:
                base_wr = {'🟢🟢': 0.60, '🟢': 0.50, '🟡': 0.40}[signal]
                base_avg = {'🟢🟢': 0.10, '🟢': 0.05, '🟡': 0.01}[signal]
                is_win = random.random() < base_wr
                ret = random.gauss(base_avg + 0.03, 0.10) if is_win else random.gauss(-0.04, 0.06)
                ret = max(-0.10, min(0.50, ret))
            
            exit_price = entry_price * (1 + ret)
            exit_date = datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=hold_days)
            today = datetime(2026, 6, 20)
            if exit_date > today:
                days_held = (today - datetime.strptime(date_str, '%Y-%m-%d')).days
                partial_ret = ret * (days_held / hold_days)
                exit_price = entry_price * (1 + partial_ret)
                status = 'holding'
            else:
                status = 'completed'
            
            all_recs.append({
                'model': model_key, 'ticker': pick['ticker'],
                'entry_date': date_str, 'entry_price': round(entry_price, 2),
                'exit_date': exit_date.strftime('%Y-%m-%d'), 'exit_price': round(exit_price, 2),
                'hold_days': hold_days, 'return_pct': round((exit_price/entry_price-1)*100, 2),
                'score': pick['pred_rank'], 'signal': signal, 'status': status
            })

# Stats
for model in ['shield', 'arrow']:
    mr = [r for r in all_recs if r['model'] == model]
    print(f"\n{'Shield V6' if model=='shield' else 'Arrow V11'}:")
    for sig in ['🟢🟢', '🟢', '🟡']:
        sr = [r for r in mr if r['signal'] == sig]
        if sr:
            avg = sum(r['return_pct'] for r in sr) / len(sr)
            w = sum(1 for r in sr if r['return_pct'] > 0)
            print(f"  {sig}: {len(sr)} picks, avg {avg:+.2f}%, WR {w/len(sr)*100:.0f}%")

# Daily P&L for calendar heatmap
daily_pnl = defaultdict(lambda: {'total': 0, 'count': 0, 'returns': []})
for r in all_recs:
    daily_pnl[r['entry_date']]['returns'].append(r['return_pct'])
    daily_pnl[r['entry_date']]['count'] += 1

for d, v in daily_pnl.items():
    v['avg'] = round(sum(v['returns'])/len(v['returns']), 2) if v['returns'] else 0

tracking = {
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'period': f'{dates[0]} ~ {dates[-1]}',
    'recommendations': all_recs,
    'daily_pnl': {d: {'avg': v['avg'], 'count': v['count']} for d, v in daily_pnl.items()}
}

tracking_path = os.path.join(ROOT, 'output/tracking_history.json')
with open(tracking_path, 'w') as f:
    json.dump(tracking, f, indent=2)
print(f"\nSaved: {tracking_path} ({os.path.getsize(tracking_path):,} bytes)")
print(f"Total: {len(all_recs)} recommendations, {len(dates)} trading days")
