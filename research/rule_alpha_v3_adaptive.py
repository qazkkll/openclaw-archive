#!/usr/bin/env python3
"""
rule-alpha-v3.0 backtest — Adaptive Factor Weighting
===========================================
Key idea: Instead of fixed factor weights, compute factor IC over trailing
N trading days and use IC as weights. Factors that stopped working get zeroed out.

Tests:
1. v2.1 baseline (static weights) 
2. Adaptive 120d IC weights
3. Adaptive 60d IC weights  
4. Flow-only (top stable factor)
5. Adaptive + momentum factor (new)

Backtest framework:
- Daily equity curve with real drawdown tracking
- SL-1% per trade
- DD-based position sizing (v2.1 thresholds)
- Transaction costs 0.15% round-trip
- 10-day rebalance, Top15, equal weight
"""

import pandas as pd, numpy as np, json, time, os, sys, warnings
from datetime import datetime
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

# ============================================================
# 0. Configuration
# ============================================================
HOLD_DAYS = 10
TOP_N = 15
SL = -0.01  # stop-loss per trade
COST = 0.0015  # 0.15% round-trip (0.075% per side)
DD_THRESHOLDS = [(-0.03, 0.80), (-0.06, 0.60), (-0.10, 0.40), (-0.14, 0.20), (-0.18, 0.00)]
IC_LOOKBACK = 120  # trading days for IC computation
MIN_IC_FOR_WEIGHT = 0.005  # below this, weight = 0

# ============================================================
# 1. Load Data
# ============================================================
print(f"📊 rule-alpha-v3.0 backtest {time.strftime('%Y-%m-%d %H:%M')}")
print("="*60)
t0 = time.time()

df = pd.read_parquet('data/a_hist_10y.parquet')
df = df.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df['date'] = df['date'].astype(int)

mf = pd.read_parquet('data/cn/moneyflow_core.parquet')
mf['sym'] = mf['ts_code'].str[:6]
mf['date'] = mf['trade_date'].astype(int)
for col in ['sm', 'md', 'lg', 'elg']:
    mf[f'{col}_net'] = mf[f'buy_{col}_amount'] - mf[f'sell_{col}_amount']
mf['total_net'] = mf['net_mf_amount']
df = df.merge(mf[['sym','date','total_net','lg_net','md_net','elg_net']], on=['sym','date'], how='left')

# Filter
df = df[~df['sym'].str.startswith('688')].copy()
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
df = df[df['volume'] > 0].copy()
df = df.sort_values(['sym', 'date']).reset_index(drop=True)

print(f"  Data: {len(df):,} rows, {df['sym'].nunique()} stocks, {time.time()-t0:.0f}s")

# ============================================================
# 2. Compute Features
# ============================================================
print("  Computing features...")
t1 = time.time()

# Forward return (for IC computation)
df['fwd10'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-10) / x - 1)

# Returns
df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ret10'] = df.groupby('sym')['close'].pct_change(10)

# MA deviation
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']

# Volatility
df['vol20'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(20, min_periods=2).std())

# RSI
delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)

# Money flow 5d aggregation
for col in ['total_net', 'lg_net', 'md_net', 'elg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())

# Money flow momentum (5d vs 20d — NEW factor)
df['total_net_20d'] = df.groupby('sym')['total_net'].transform(lambda x: x.rolling(20, min_periods=1).sum())
df['flow_momentum'] = df['total_net_5d'] - df['total_net_20d'] * 0.25  # 5d avg vs 20d avg

# Volume-price divergence (NEW factor)
# Price up but volume down = bearish divergence, price down but volume up = bullish
df['vol_ratio_5_20'] = (
    df.groupby('sym')['volume'].transform(lambda x: x.rolling(5).mean()) /
    df.groupby('sym')['volume'].transform(lambda x: x.rolling(20).mean()).replace(0, np.nan)
).fillna(1)
df['vol_price_div'] = -df['ret5'] * (df['vol_ratio_5_20'] - 1)  # positive = bullish divergence

# Market state
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())

print(f"  Features: {time.time()-t1:.0f}s")

# ============================================================
# 3. Define Factor Sets
# ============================================================
FACTOR_DEFS = {
    'reversal': {'col': 'ret20', 'transform': 'neg_clip', 'clip': 0.3, 'default_w': 3.0},
    'flow_rank': {'col': 'total_net_5d', 'transform': 'rank_pct', 'default_w': 2.0},
    'low_vol': {'col': 'vol20', 'transform': 'neg_rank_pct', 'default_w': 2.0},
    'rsi_oversold': {'col': 'rsi_14', 'transform': 'binary_lt', 'threshold': 35, 'default_w': 1.5},
    'lg_flow': {'col': 'lg_net_5d', 'transform': 'rank_pct', 'default_w': 1.0},
    'ma_bias': {'col': 'ma20_bias', 'transform': 'neg_clip', 'clip': 0.2, 'default_w': 1.0},
    'flow_momentum': {'col': 'flow_momentum', 'transform': 'rank_pct', 'default_w': 0},  # NEW, default off
    'vol_price_div': {'col': 'vol_price_div', 'transform': 'rank_pct', 'default_w': 0},  # NEW, default off
    'md_flow': {'col': 'md_net_5d', 'transform': 'rank_pct', 'default_w': 0},  # NEW
    'elg_flow': {'col': 'elg_net_5d', 'transform': 'rank_pct', 'default_w': 0},  # NEW
}

def compute_factor_value(day_df, factor_name):
    """Compute factor value for a single day's cross-section."""
    fdef = FACTOR_DEFS[factor_name]
    col = fdef['col']
    vals = day_df[col].fillna(0)
    
    if fdef['transform'] == 'neg_clip':
        return (-vals).clip(-fdef['clip'], fdef['clip'])
    elif fdef['transform'] == 'rank_pct':
        return vals.rank(pct=True)
    elif fdef['transform'] == 'neg_rank_pct':
        return (1 - vals.rank(pct=True))
    elif fdef['transform'] == 'binary_lt':
        return (vals < fdef['threshold']).astype(float)
    else:
        return vals

def compute_score(day_df, weights):
    """Compute composite score given factor weights."""
    score = pd.Series(0.0, index=day_df.index)
    for fname, w in weights.items():
        if w != 0 and fname in FACTOR_DEFS:
            score += compute_factor_value(day_df, fname) * w
    return score

def compute_factor_ic(df, factor_name, date, lookback=120):
    """Compute IC of a factor over trailing lookback days."""
    fdef = FACTOR_DEFS[factor_name]
    col = fdef['col']
    
    # Get lookback window
    all_dates = sorted(df['date'].unique())
    date_idx = np.searchsorted(all_dates, date)
    start_idx = max(0, date_idx - lookback)
    window_dates = all_dates[start_idx:date_idx]
    
    if len(window_dates) < 30:
        return 0  # not enough data
    
    window = df[df['date'].isin(window_dates)].copy()
    
    # Compute factor values
    factor_vals = compute_factor_value(window, factor_name)
    fwd = window['fwd10']
    
    # IC per day
    ics = []
    for d in window_dates:
        mask = window['date'] == d
        f = factor_vals[mask]
        r = fwd[mask]
        valid = f.notna() & r.notna()
        if valid.sum() > 30:
            ic = f[valid].corr(r[valid])
            if not np.isnan(ic):
                ics.append(ic)
    
    return np.mean(ics) if len(ics) >= 5 else 0

def get_adaptive_weights(df, date, lookback=120, min_ic=0.005):
    """Compute adaptive factor weights based on recent IC."""
    weights = {}
    for fname in FACTOR_DEFS:
        ic = compute_factor_ic(df, fname, date, lookback)
        if abs(ic) >= min_ic:
            # Use IC sign and magnitude, scaled by default weight
            default_w = FACTOR_DEFS[fname]['default_w']
            weights[fname] = np.sign(ic) * max(abs(ic) * 20, 0.5)  # scale IC to reasonable weight
        else:
            weights[fname] = 0
    return weights

def get_default_weights():
    """Get v2.1 default weights."""
    return {fname: fdef['default_w'] for fname, fdef in FACTOR_DEFS.items()}

# ============================================================
# 4. Backtest Engine
# ============================================================
def run_backtest(df, weight_mode='static', lookback=120, label=''):
    """
    Run a full backtest with daily equity tracking.
    
    weight_mode:
      'static' - v2.1 fixed weights
      'adaptive_IC' - adaptive weights based on trailing IC
      'flow_only' - only flow_rank factor
      'custom' - use get_adaptive_weights with lookback
    """
    print(f"\n  Running: {label} (mode={weight_mode}, lb={lookback})")
    t_start = time.time()
    
    all_dates = sorted(df['date'].unique())
    n_dates = len(all_dates)
    
    # Only start after warmup (need 60 days for factors + 120 for IC)
    warmup = max(120, lookback + 20)
    
    # Portfolio state
    equity = 100000.0
    peak_equity = equity
    current_dd = 0.0
    positions = {}  # sym -> {entry_price, entry_date, shares}
    daily_returns = []
    trade_log = []
    rebal_count = 0
    last_rebal = 0
    
    # Precompute daily data for speed
    date_groups = {d: g for d, g in df.groupby('date')}
    
    for i in range(warmup, n_dates):
        today = all_dates[i]
        today_data = date_groups.get(today)
        if today_data is None or len(today_data) == 0:
            continue
        
        # === Mark-to-market ===
        port_value = 0
        for sym, pos in list(positions.items()):
            row = today_data[today_data['sym'] == sym]
            if len(row) > 0:
                current_price = row.iloc[0]['close']
                ret = current_price / pos['entry_price'] - 1
                pos['current_price'] = current_price
                pos['cum_ret'] = ret
                
                # Stop-loss check
                if ret <= SL:
                    # Stop-loss triggered
                    pnl = pos['shares'] * current_price * (1 - COST/2)
                    equity += pnl
                    trade_log.append({
                        'sym': sym, 'entry': pos['entry_price'], 'exit': current_price,
                        'ret': ret, 'reason': 'SL', 'days': i - pos['entry_idx']
                    })
                    del positions[sym]
                else:
                    port_value += pos['shares'] * current_price
            else:
                # Stock not trading today, keep position
                port_value += pos['shares'] * pos.get('current_price', pos['entry_price'])
        
        # Total equity
        total_equity = equity + port_value
        if total_equity > peak_equity:
            peak_equity = total_equity
        current_dd = (total_equity / peak_equity) - 1
        
        # Daily return
        if len(daily_returns) > 0:
            prev_eq = daily_returns[-1]['equity']
            daily_ret = total_equity / prev_eq - 1
        else:
            daily_ret = 0
        daily_returns.append({'date': today, 'equity': total_equity, 'ret': daily_ret, 'dd': current_dd})
        
        # === Rebalance? ===
        if i - last_rebal < HOLD_DAYS:
            continue
        last_rebal = i
        rebal_count += 1
        
        # DD-based position sizing
        position_pct = 1.0
        for dd_level, pct in DD_THRESHOLDS:
            if current_dd <= dd_level:
                position_pct = pct
                break
        
        # Market state overlay
        market_breadth = today_data['breadth'].mean() if 'breadth' in today_data.columns else 0.5
        mkt_ret20 = today_data['ret20'].mean() if 'ret20' in today_data.columns else 0
        # Simple market state
        if mkt_ret20 < -0.05 and market_breadth < 0.35:
            position_pct = min(position_pct, 0.5)  # bear
        elif mkt_ret20 < 0 or market_breadth < 0.4:
            position_pct = min(position_pct, 0.8)  # cautious
        
        # Determine weights
        if weight_mode == 'static':
            weights = get_default_weights()
        elif weight_mode == 'adaptive_IC':
            weights = get_adaptive_weights(df, today, lookback)
        elif weight_mode == 'flow_only':
            weights = {f: 0 for f in FACTOR_DEFS}
            weights['flow_rank'] = 2.0
        elif weight_mode == 'adaptive_60':
            weights = get_adaptive_weights(df, today, 60)
        else:
            weights = get_default_weights()
        
        # Score and rank
        scored = today_data.copy()
        scored['score'] = compute_score(scored, weights)
        
        # Filter
        scored = scored[
            (scored['close'] >= 3) & 
            (scored['close'] <= 200) & 
            (~scored['sym'].str.contains('ST|退市', na=False)) &
            (scored['volume'] > 0)
        ]
        
        if len(scored) < TOP_N:
            continue
        
        top = scored.nlargest(TOP_N, 'score')
        target_syms = set(top['sym'].tolist())
        
        # Sell positions not in target
        for sym in list(positions.keys()):
            if sym not in target_syms:
                pos = positions[sym]
                exit_price = pos.get('current_price', pos['entry_price'])
                pnl = pos['shares'] * exit_price * (1 - COST/2)
                equity += pnl
                ret = exit_price / pos['entry_price'] - 1
                trade_log.append({
                    'sym': sym, 'entry': pos['entry_price'], 'exit': exit_price,
                    'ret': ret, 'reason': 'rebal', 'days': i - pos['entry_idx']
                })
                del positions[sym]
        
        # Buy new positions
        available_cash = equity * position_pct
        cash_per_stock = available_cash / TOP_N if TOP_N > 0 else 0
        
        for _, row in top.iterrows():
            sym = row['sym']
            if sym in positions:
                continue  # already held
            if cash_per_stock <= 0:
                continue
            
            price = row['close']
            shares = cash_per_stock / (price * (1 + COST/2))
            cost = shares * price * (1 + COST/2)
            
            if cost > equity:
                continue
            
            equity -= cost
            positions[sym] = {
                'entry_price': price,
                'entry_date': today,
                'entry_idx': i,
                'shares': shares,
                'current_price': price,
                'cum_ret': 0
            }
    
    # Close remaining positions
    final_date = all_dates[-1]
    for sym, pos in list(positions.items()):
        exit_price = pos.get('current_price', pos['entry_price'])
        pnl = pos['shares'] * exit_price * (1 - COST/2)
        equity += pnl
        ret = exit_price / pos['entry_price'] - 1
        trade_log.append({
            'sym': sym, 'entry': pos['entry_price'], 'exit': exit_price,
            'ret': ret, 'reason': 'end', 'days': 0
        })
    
    # === Compute metrics ===
    dr = pd.DataFrame(daily_returns)
    if len(dr) < 10:
        return None
    
    # Annual return
    total_days = (pd.to_datetime(str(all_dates[-1])) - pd.to_datetime(str(all_dates[warmup]))).days
    total_days = max(total_days, 1)
    final_equity = dr['equity'].iloc[-1]
    initial_equity = dr['equity'].iloc[0]
    cagr = (final_equity / initial_equity) ** (365 / total_days) - 1
    
    # Sharpe & Sortino (annualized from daily returns)
    daily_rets = dr['ret'].dropna()
    if len(daily_rets) > 0 and daily_rets.std() > 0:
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)
        neg_rets = daily_rets[daily_rets < 0]
        sortino = daily_rets.mean() / neg_rets.std() * np.sqrt(252) if len(neg_rets) > 0 and neg_rets.std() > 0 else 0
    else:
        sharpe = sortino = 0
    
    # Max DD
    max_dd = dr['dd'].min()
    
    # Trade stats
    tl = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    n_trades = len(tl)
    if n_trades > 0:
        win_rate = (tl['ret'] > 0).mean()
        avg_win = tl[tl['ret'] > 0]['ret'].mean() if (tl['ret'] > 0).any() else 0
        avg_loss = tl[tl['ret'] < 0]['ret'].mean() if (tl['ret'] < 0).any() else 0
        sl_rate = (tl['reason'] == 'SL').mean()
        pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    else:
        win_rate = avg_win = avg_loss = sl_rate = pl_ratio = 0
    
    elapsed = time.time() - t_start
    
    result = {
        'label': label,
        'weight_mode': weight_mode,
        'lookback': lookback,
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd': round(max_dd * 100, 2),
        'win_rate': round(win_rate * 100, 1),
        'pl_ratio': round(pl_ratio, 2),
        'avg_win': round(avg_win * 100, 2),
        'avg_loss': round(avg_loss * 100, 2),
        'sl_rate': round(sl_rate * 100, 1),
        'n_trades': n_trades,
        'n_rebal': rebal_count,
        'elapsed': round(elapsed, 1),
    }
    
    print(f"    ✅ {label}: Sharpe={sharpe:.3f} CAGR={cagr*100:.1f}% DD={max_dd*100:.1f}% WR={win_rate*100:.1f}% ({elapsed:.0f}s)")
    
    return result

# ============================================================
# 5. Run All Strategies
# ============================================================
print("\n" + "="*60)
print("🚀 Running all strategies...")
print("="*60)

results = []

# 1. v2.1 baseline (static weights)
r = run_backtest(df, weight_mode='static', label='v2.1_static_baseline')
if r: results.append(r)

# 2. Adaptive IC with 120-day lookback
r = run_backtest(df, weight_mode='adaptive_IC', lookback=120, label='v3.0_adaptive_120d')
if r: results.append(r)

# 3. Adaptive IC with 60-day lookback
r = run_backtest(df, weight_mode='adaptive_60', lookback=60, label='v3.0_adaptive_60d')
if r: results.append(r)

# 4. Flow-only (pure money flow, stable factor)
r = run_backtest(df, weight_mode='flow_only', label='v3.0_flow_only')
if r: results.append(r)

# ============================================================
# 6. Summary
# ============================================================
print("\n" + "="*60)
print("📊 Results Summary")
print("="*60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False)

print(f"\n{'Strategy':<30} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'WR%':>6} {'P/L':>6} {'SL%':>6} {'Trades':>8}")
print("-" * 90)
for _, r in results_df.iterrows():
    print(f"{r['label']:<30} {r['sharpe']:>8.3f} {r['cagr']:>8.1f} {r['max_dd']:>8.1f} {r['win_rate']:>6.1f} {r['pl_ratio']:>6.2f} {r['sl_rate']:>6.1f} {r['n_trades']:>8}")

# Save results
os.makedirs('research', exist_ok=True)
out_file = 'research/rule_alpha_v3_adaptive_experiments.json'
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✅ Results saved to {out_file}")
