# -*- coding: utf-8 -*-
"""
A1 Layer 3 Walk-Forward Backtest
Validate L3 regression model (37 features, 10d return prediction) on rolling windows

WF design:
  Train: 3 years, Test: 1 year, Slide: 1 year
  First: train 2016-2018, test 2019
  Last:  train 2023-2025, test 2026

Key fixes (2026-06-12):
1. Correct backtest engine: real position management + daily P&L
2. Build feature matrix in batch, stock-by-stock
3. Stop-loss -15%, hold 10 days

Usage: python scripts/a1_l3_walk_forward.py
"""

import json, os, sys, time, gc
import numpy as np
from collections import defaultdict, OrderedDict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

D_DATA = r'/home/hermes/.hermes/openclaw-archive/data'
HIST_JSON = os.path.join(D_DATA, 'a_hist_10y.parquet')
MF_JSON = os.path.join(D_DATA, 'moneyflow_data.parquet')
POOL_JSON = os.path.join(D_DATA, 'a_share_top100.json')

FEATURE_KEYS = [
    'pct_ma5','pct_ma10','pct_ma20','pct_ma60','pct_ma120',
    'ma20_slope','ma60_slope','ma_align',
    'vol_10d','vol_60d','vol_ratio','atr20_pct',
    'ret_5d','ret_10d','ret_20d','ret_60d','rsi14',
    'vol_ratio_5_20',
    'net_mf','lg_net','elg_net','md_net',
    'lg_pct','elg_pct',
    'major_net','major_ratio',
    'net_mf_5d','net_mf_10d','net_mf_20d','net_mf_60d',
    'major_net_5d','major_net_10d','major_net_20d',
    'lg_net_5d','lg_net_10d','lg_net_20d',
]

CONFIG = OrderedDict([
    ('candidate_size', 400), # 质量池400只
    ('hold_days', 10),
    ('buy_threshold', 4.0),     # 预测10日涨幅 >= 4%才买入 (P75)
    ('profit_target', 10.0),    # 持有期内触发+10%止盈 (P90)
    ('stop_loss', -10.0),       # 持有期内触发-10%止损
    ('top_k', 3),
    ('max_positions', 10),
    ('train_years', 3),
    ('test_years', 1),
    ('capital', 100000),
])


def parse_date(d):
    if isinstance(d, (int, float)):
        return int(d)
    if d is None:
        return 0
    return int(str(d).replace('-', '').strip())

def get_year(d_int):
    return d_int // 10000

def ensure_list(v):
    if v is None: return []
    if isinstance(v, list): return v
    return []

def load_json(path, label=''):
    print(f"  Loading {label or os.path.basename(path)}...", end=' ', flush=True)
    t0 = time.time()
    with open(path, 'rb') as f:
        data = json.load(f)
    print(f"{len(data)} items, {time.time()-t0:.1f}s")
    return data

def load_quality_pool():
    """
    Load candidate pool:
    1. quality_pool.json/a/a_stocks (400 stocks) - primary
    2. a_share_top100.json Top500 (500 stocks) - fallback
    Trim to CONFIG['candidate_size']
    """
    # Primary: quality_pool.json
    qp_path = os.path.join(D_DATA, 'quality_pool.json')
    if os.path.exists(qp_path):
        with open(qp_path, 'rb') as f:
            data = json.load(f)
        a_section = data.get('a', {})
        raw = a_section.get('a_stocks', [])
        if raw:
            codes = []
            for item in raw:
                if isinstance(item, dict):
                    c = item.get('symbol', '').strip()
                else:
                    c = str(item).strip()
                if c:
                    codes.append(c)
            if codes:
                print(f'  Loaded {len(codes)} stocks from quality_pool.json/a_stocks')
                return codes[:CONFIG['candidate_size']]

    # Fallback: a_share_top100.json Top500
    if os.path.exists(POOL_JSON):
        with open(POOL_JSON, 'rb') as f:
            data = json.load(f)
        top500 = data.get('top500', data) if isinstance(data, dict) else data
        codes = [s.get('code', '') if isinstance(s, dict) else str(s) for s in top500]
        result = [c.strip() for c in codes if c.strip()][:CONFIG['candidate_size']]
        print(f'  Loaded {len(result)} stocks from a_share_top100 fallback')
        return result

    return None


def build_mf_lookup(mf_data):
    lookup = {}
    for code, records in mf_data.items():
        if not records:
            continue
        by_date = {}
        for r in records:
            d = parse_date(r.get('trade_date', ''))
            if not d:
                continue
            by_date[d] = {
                'net_mf': float(r.get('net_mf_amount', 0) or 0),
                'lg_net': float(r.get('buy_lg_amount', 0) or 0) - float(r.get('sell_lg_amount', 0) or 0),
                'elg_net': float(r.get('buy_elg_amount', 0) or 0) - float(r.get('sell_elg_amount', 0) or 0),
                'md_net': float(r.get('buy_md_amount', 0) or 0) - float(r.get('sell_md_amount', 0) or 0),
            }
        lookup[code] = by_date
    return lookup

def calc_mf_rollup(mf_by_date, limit_date=99999999):
    if not mf_by_date:
        return {}
    dates = sorted(d for d in mf_by_date if d <= limit_date)
    result = {}
    for i, d in enumerate(dates):
        if i < 60:
            continue
        net_5d = sum(mf_by_date[dates[j]]['net_mf'] for j in range(i-4, i+1))
        net_10d = sum(mf_by_date[dates[j]]['net_mf'] for j in range(i-9, i+1))
        net_20d = sum(mf_by_date[dates[j]]['net_mf'] for j in range(i-19, i+1))
        net_60d = sum(mf_by_date[dates[j]]['net_mf'] for j in range(i-59, i+1))
        lg_5d = sum(mf_by_date[dates[j]]['lg_net'] for j in range(i-4, i+1))
        lg_10d = sum(mf_by_date[dates[j]]['lg_net'] for j in range(i-9, i+1))
        lg_20d = sum(mf_by_date[dates[j]]['lg_net'] for j in range(i-19, i+1))
        elg_5d = sum(mf_by_date[dates[j]]['elg_net'] for j in range(i-4, i+1))
        elg_10d = sum(mf_by_date[dates[j]]['elg_net'] for j in range(i-9, i+1))
        elg_20d = sum(mf_by_date[dates[j]]['elg_net'] for j in range(i-19, i+1))

        rec = mf_by_date[d]
        day_lg = rec['lg_net']
        day_elg = rec['elg_net']
        day_md = rec['md_net']
        major_net = day_lg + day_elg
        all_abs = abs(day_lg) + abs(day_elg) + abs(day_md) + 1
        major_ratio = (day_lg + day_elg) / all_abs

        result[str(d)] = {
            'net_mf_1d': rec['net_mf'], 'lg_net_1d': day_lg,
            'elg_net_1d': day_elg, 'md_net_1d': day_md,
            'net_mf_5d': net_5d, 'net_mf_10d': net_10d,
            'net_mf_20d': net_20d, 'net_mf_60d': net_60d,
            'major_net': major_net, 'major_ratio': major_ratio,
            'major_net_5d': lg_5d + elg_5d,
            'major_net_10d': lg_10d + elg_10d,
            'major_net_20d': lg_20d + elg_20d,
            'lg_net_5d': lg_5d, 'lg_net_10d': lg_10d, 'lg_net_20d': lg_20d,
            'lg_pct': 50.0, 'elg_pct': 25.0,
        }
    return result


def calc_stock_features(code, hist, mf_rollup, cutoff_date=99999999):
    """Compute feature samples for one stock, cutoff_date controls visibility"""
    c = ensure_list(hist.get('c'))
    h = ensure_list(hist.get('h'))
    lo = ensure_list(hist.get('l'))
    op = ensure_list(hist.get('o'))
    v = ensure_list(hist.get('v'))
    dates = ensure_list(hist.get('dates'))
    dates_int = [parse_date(d) for d in dates]
    limit = parse_date(cutoff_date) if cutoff_date else 99999999
    n = len(c)
    if n < 130:
        return []

    samples = []
    for i in range(120, n - 10):
        d_int = dates_int[i]
        if d_int > limit:
            break
        d = str(d_int)
        price = c[i]
        if price <= 0:
            continue

        s = {'code': code, 'date': d, 'close': price, 'idx': i}

        # MA
        ma5 = sum(c[i-4:i+1])/5
        ma10 = sum(c[i-9:i+1])/10
        ma20 = sum(c[i-19:i+1])/20
        ma60 = sum(c[i-59:i+1])/60
        ma120 = sum(c[i-119:i+1])/120
        s['pct_ma5'] = (price/ma5 - 1)*100 if ma5 > 0 else 0
        s['pct_ma10'] = (price/ma10 - 1)*100 if ma10 > 0 else 0
        s['pct_ma20'] = (price/ma20 - 1)*100 if ma20 > 0 else 0
        s['pct_ma60'] = (price/ma60 - 1)*100 if ma60 > 0 else 0
        s['pct_ma120'] = (price/ma120 - 1)*100 if ma120 > 0 else 0
        s['ma20_slope'] = (ma20/(sum(c[i-25:i-4])/20) - 1)*100 if i >= 25 else 0
        s['ma60_slope'] = (ma60/(sum(c[i-65:i-4])/60) - 1)*100 if i >= 65 else 0
        s['ma_align'] = (ma5 > ma10) + (ma10 > ma20) + (ma20 > ma60) + (price > ma5) + (price > ma10) + (price > ma60)

        # Volatility
        ret10 = [abs(c[j]/c[j-1] - 1)*100 for j in range(max(1,i-9), i+1) if c[j-1] > 0]
        ret60 = [abs(c[j]/c[j-1] - 1)*100 for j in range(max(1,i-59), i+1) if c[j-1] > 0]
        s['vol_10d'] = sum(ret10)/len(ret10) if ret10 else 0
        s['vol_60d'] = sum(ret60)/len(ret60) if ret60 else 0
        s['vol_ratio'] = s['vol_10d']/s['vol_60d'] if s['vol_60d'] > 0 else 1

        trs = [max(h[j]-lo[j], abs(h[j]-c[j-1]), abs(lo[j]-c[j-1])) for j in range(i-19, i+1)]
        s['atr20_pct'] = sum(trs)/len(trs)/price*100 if price > 0 else 0

        # Returns
        s['ret_5d'] = (price/c[i-5] - 1)*100
        s['ret_10d'] = (price/c[i-10] - 1)*100
        s['ret_20d'] = (price/c[i-20] - 1)*100
        s['ret_60d'] = (price/c[i-60] - 1)*100

        # RSI14
        changes = [c[j] - c[j-1] for j in range(i-13, i+1)]
        gains = sum(x for x in changes if x > 0)
        losses = sum(-x for x in changes if x < 0)
        avg_gain = gains / 14
        avg_loss = losses / 14
        s['rsi14'] = 100 - 100/(1 + avg_gain/avg_loss) if avg_loss > 0 else 100

        # Volume ratio
        vol5 = sum(v[i-4:i+1])/5
        vol20 = sum(v[i-19:i+1])/20
        s['vol_ratio_5_20'] = vol5/vol20 if vol20 > 0 else 1

        # Money flow
        ru = mf_rollup.get(d, {})
        for k in ['net_mf','lg_net','elg_net','md_net','major_net','major_ratio',
                   'net_mf_5d','net_mf_10d','net_mf_20d','net_mf_60d',
                   'major_net_5d','major_net_10d','major_net_20d',
                   'lg_net_5d','lg_net_10d','lg_net_20d','lg_pct','elg_pct']:
            s[k] = ru.get(k, 0)

        # Target: 10d forward return
        s['fwd_ret'] = (c[i+10] / price - 1) * 100
        samples.append(s)

    return samples


def build_window_dataset(codes, hist_data, mf_lookup, train_end, test_end=None):
    X_train, y_train, meta_train = [], [], []
    X_test, y_test, meta_test = [], [], []
    t0 = time.time()
    done = 0

    for code in codes:
        hist = hist_data.get(code)
        if not hist or not hist.get('c') or len(hist['c']) < 150:
            continue

        mf_code = code + '.SZ' if code[:1] in ('0','3') else code + '.SH'
        mf_by_date = mf_lookup.get(mf_code, {})
        mf_rollup = calc_mf_rollup(mf_by_date, test_end or train_end)

        samples = calc_stock_features(code, hist, mf_rollup, test_end or train_end)
        train_end_i = parse_date(train_end)
        test_end_i = parse_date(test_end) if test_end else 99999999

        for s in samples:
            d = int(s['date'])
            feats = [s[k] for k in FEATURE_KEYS]
            target = s['fwd_ret']
            if d <= train_end_i:
                X_train.append(feats)
                y_train.append(target)
                meta_train.append(s)
            elif test_end_i and train_end_i < d <= test_end_i:
                X_test.append(feats)
                y_test.append(target)
                meta_test.append(s)

        done += 1
        if done % 20 == 0:
            elapsed = time.time()-t0
            rate = done/elapsed if elapsed > 0 else 0
            print(f"    ...{done}/{len(codes)} ({done/len(codes)*100:.0f}%) {rate:.1f} stk/s", flush=True)

    X_train = np.array(X_train, dtype=np.float32) if X_train else np.array([])
    y_train = np.array(y_train, dtype=np.float32) if y_train else np.array([])
    X_test = np.array(X_test, dtype=np.float32) if X_test else np.array([])
    y_test = np.array(y_test, dtype=np.float32) if y_test else np.array([])
    return X_train, y_train, meta_train, X_test, y_test, meta_test


def train_model(X_train, y_train):
    import xgboost as xgb
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 500,
        'early_stopping_rounds': 50,
        'verbosity': 0,
        'device': 'cuda',
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return model


def run_backtest(predictions, meta_test, hist_data):
    """
    Absolute-standard backtest:
    - Buy only if predicted 10d return >= buy_threshold (4% = P75)
    - Take profit at +profit_target (10% = P90) during hold
    - Stop loss at stop_loss (-10%)
    - Time out at hold_days (10 days)
    - If no buy signal, stay in cash
    """
    daily_picks = defaultdict(list)
    for i, m in enumerate(meta_test):
        daily_picks[m['date']].append((predictions[i], m))

    sorted_dates = sorted(daily_picks.keys())
    cash = CONFIG['capital']
    positions = []
    balance = [cash]
    peak = cash
    trades = []
    days_cash = 0
    days_in_market = 0

    for day in sorted_dates:
        signals = sorted(daily_picks[day], key=lambda x: x[0], reverse=True)

        # --- Sell: check each open position ---
        remaining = []
        for pos in positions:
            pos['days'] += 1
            hist = hist_data.get(pos['code'])
            cur_price = pos['entry_price']
            if hist and hist.get('c'):
                idx = pos['entry_idx'] + pos['days']
                if idx < len(hist['c']):
                    cur_price = hist['c'][idx]

            ret = (cur_price / pos['entry_price'] - 1) * 100

            # Track highest price seen for trailing stop purposes
            if ret > pos['high_ret']:
                pos['high_ret'] = ret

            # Sell conditions (first one wins):
            reason = None
            if ret >= CONFIG['profit_target']:
                reason = 'profit'   # take profit
            elif pos['days'] >= CONFIG['hold_days']:
                reason = 'time'     # max hold time
            elif ret <= CONFIG['stop_loss']:
                reason = 'stop'     # stop loss

            if reason:
                pnl = (cur_price - pos['entry_price']) / pos['entry_price']
                cash += pos['alloc'] * (1 + pnl)
                trades.append({
                    'code': pos['code'], 'entry': pos['entry_date'],
                    'exit': day, 'pnl': round(pnl*100, 2),
                    'reason': reason, 'days': pos['days'],
                })
            else:
                remaining.append(pos)
        positions = remaining

        # --- Buy: only signals with predicted return >= buy_threshold ---
        buy_count = 0
        for pred, m in signals:
            if buy_count >= CONFIG['top_k']:
                break
            if len(positions) + buy_count >= CONFIG['max_positions']:
                break
            # Absolute standard: predicted return must meet threshold
            if pred < CONFIG['buy_threshold']:
                continue
            if any(p['code'] == m['code'] for p in positions):
                continue

            alloc = cash / max(1, CONFIG['top_k'] - buy_count)
            positions.append({
                'code': m['code'], 'entry_date': day,
                'entry_price': m['close'], 'entry_idx': m.get('idx', 0),
                'days': 0, 'alloc': alloc, 'high_ret': 0.0,
            })
            cash -= alloc
            buy_count += 1

        # --- Valuation ---
        total = cash
        for pos in positions:
            hist = hist_data.get(pos['code'])
            cur = pos['entry_price']
            if hist and hist.get('c'):
                idx = pos['entry_idx'] + pos['days']
                if idx < len(hist['c']):
                    cur = hist['c'][idx]
            total += pos['alloc'] * (cur / pos['entry_price'])
        balance.append(total)
        if total > peak:
            peak = total

        if len(positions) > 0:
            days_in_market += 1
        else:
            days_cash += 1

    # Metrics
    final = balance[-1]
    total_ret = (final / CONFIG['capital'] - 1) * 100

    if len(sorted_dates) >= 2:
        d0 = parse_date(sorted_dates[0])
        d1 = parse_date(sorted_dates[-1])
        years = (d1 - d0) / 365.25 / 10000
        if years > 0.1 and CONFIG['capital'] > 0 and final > 0:
            cagr = ((final / CONFIG['capital']) ** (1 / max(years, 0.01)) - 1) * 100
        else:
            cagr = total_ret
    else:
        cagr = 0

    daily_rets = [(balance[i]/balance[i-1] - 1)*100 for i in range(1, len(balance)) if balance[i-1] > 0]
    sharpe = (np.mean(daily_rets) / (np.std(daily_rets) + 1e-10) * np.sqrt(252)) if daily_rets else 0

    max_dd = 0
    rp = balance[0]
    for b in balance:
        if b > rp:
            rp = b
        dd = (rp - b) / rp * 100
        if dd > max_dd:
            max_dd = dd

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    return {
        'cagr': round(cagr, 2),
        'total_return': round(total_ret, 2),
        'trades': len(trades),
        'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd, 2),
        'win_rate': round(len(wins)/max(1, len(trades))*100, 1),
        'avg_win': round(np.mean([t['pnl'] for t in wins]), 2) if wins else 0,
        'avg_loss': round(np.mean([t['pnl'] for t in losses]), 2) if losses else 0,
        'final_value': round(final, 0),
        'days_in_market': days_in_market,
        'days_cash': days_cash,
    }


def run_walk_forward():
    t0 = time.time()
    print('=' * 65)
    print('  A1 Layer 3 Walk-Forward Backtest (v2 correct engine)')
    print('=' * 65)
    for k, v in CONFIG.items():
        print(f'  {k}: {v}')
    print()

    seg1 = time.time()
    print('[1/4] Loading data...')
    hist_data = load_json(HIST_JSON, 'A-share 10y Kline')
    mf_raw = load_json(MF_JSON, 'Money flow')

    print('[2/4] Building MF lookup...')
    mf_lookup = build_mf_lookup(mf_raw)
    print(f'  Coverage: {len(mf_lookup)} stocks')

    print('[3/4] Candidate pool...')
    codes = load_quality_pool()
    if not codes:
        codes = list(hist_data.keys())[:CONFIG['candidate_size']]
    valid = [c for c in codes if c in hist_data]
    print(f'  {len(valid)} effective candidates')

    print('[4/4] Determining WF windows...')
    all_years = []
    for c in valid[:10]:
        hist = hist_data.get(c)
        if hist and hist.get('dates'):
            for d in hist['dates']:
                y = get_year(parse_date(d))
                if 2000 < y < 2030:
                    all_years.append(y)
    min_y = max(min(all_years), 2016) if all_years else 2016
    max_y = min(max(all_years), 2026) if all_years else 2026
    print(f'  Data range: {min_y}-{max_y}')

    win = CONFIG['train_years']
    wout = CONFIG['test_years']
    first_test = min_y + win
    windows = []
    while first_test + wout <= max_y:
        windows.append(OrderedDict([
            ('train_start', int(str(first_test - win) + '0101')),
            ('train_end', int(str(first_test - 1) + '1231')),
            ('test_start', int(str(first_test) + '0101')),
            ('test_end', int(str(first_test + wout - 1) + '1231')),
        ]))
        first_test += wout

    print(f'  {len(windows)} windows')
    for w in windows:
        print(f'    T: {w["train_start"]}-{w["train_end"]}  =>  E: {w["test_start"]}-{w["test_end"]}')

    print(f'\nPrep time: {time.time()-seg1:.1f}s')
    print()

    segment_results = []

    for seg_idx, w in enumerate(windows):
        print(f'\n{"="*65}')
        print(f'  Segment {seg_idx+1}/{len(windows)}: T={w["train_start"]}-{w["train_end"]}  E={w["test_start"]}-{w["test_end"]}')
        print(f'{"="*65}')
        tseg = time.time()

        X_train, y_train, meta_train, X_test, y_test, meta_test = build_window_dataset(
            valid, hist_data, mf_lookup, w['train_end'], w['test_end'])
        print(f'  Train: {len(X_train)} | Test: {len(X_test)}')

        if len(X_train) < 200 or len(X_test) < 100:
            print('  ! Insufficient samples, skip')
            continue

        print(f'  Training XGBoost...', end=' ', flush=True)
        model = train_model(X_train, y_train)
        print('done')

        print(f'  Predict + Backtest...', end=' ', flush=True)
        preds = model.predict(X_test)
        bt = run_backtest(preds, meta_test, hist_data)
        bt['segment'] = seg_idx + 1
        bt['train_range'] = f'{w["train_start"]}-{w["train_end"]}'
        bt['test_range'] = f'{w["test_start"]}-{w["test_end"]}'
        segment_results.append(bt)

        status = '+' if bt['cagr'] > 10 else '?' if bt['cagr'] > 0 else '-'
        print(f'done in {time.time()-tseg:.0f}s')
        print(f'  [{status}] CAGR={bt["cagr"]}%  SR={bt["sharpe"]}  DD={bt["max_dd"]}%')
        print(f'  Return={bt["total_return"]}%  WinRate={bt["win_rate"]}%  Trades={bt["trades"]}')

        del X_train, y_train, X_test, y_test, model, preds
        gc.collect()

    # Summary
    print()
    print('=' * 65)
    print('  Walk-Forward Summary')
    print('=' * 65)

    if segment_results:
        cagrs = [s['cagr'] for s in segment_results]
        sharps = [s['sharpe'] for s in segment_results]
        dds = [s['max_dd'] for s in segment_results]
        wrs = [s['win_rate'] for s in segment_results]
        trs = [s['total_return'] for s in segment_results]

        for s in segment_results:
            st = '+' if s['cagr'] > 10 else '?' if s['cagr'] > 0 else '-'
            print(f'  [{st}] S{s["segment"]} ({s["test_range"]}): '
                  f'CAGR={s["cagr"]}%  SR={s["sharpe"]}  DD={s["max_dd"]}%  '
                  f'WinRate={s["win_rate"]}%  Return={s["total_return"]}%')

        print()
        print(f'  Avg CAGR: {np.mean(cagrs):.2f}%')
        print(f'  CAGR Std: {np.std(cagrs):.2f}%')
        print(f'  Avg Sharpe: {np.mean(sharps):.2f}')
        print(f'  Avg MaxDD: {np.mean(dds):.2f}%')
        print(f'  Worst MaxDD: {max(dds):.2f}%')
        print(f'  Avg WinRate: {np.mean(wrs):.1f}%')
        print(f'  Positive segments: {sum(1 for c in cagrs if c > 0)}/{len(cagrs)}')
        print(f'  CAGR>30% segments: {sum(1 for c in cagrs if c > 30)}/{len(cagrs)}')

        print()
        print(f'  Reference: original 3-period backtest CAGR=59.71%')
        if np.mean(cagrs) > 30:
            print(f'  Result: WF PASSED, model stable (avg CAGR {np.mean(cagrs):.1f}%)')
        elif np.mean(cagrs) > 0:
            print(f'  Result: WF marginal, needs tuning (avg CAGR {np.mean(cagrs):.1f}%)')
        else:
            print(f'  Result: WF FAILED, look-ahead bias suspected')

        output = {
            'config': dict(CONFIG),
            'windows': windows,
            'segments': segment_results,
            'summary': {
                'avg_cagr': round(float(np.mean(cagrs)), 2),
                'std_cagr': round(float(np.std(cagrs)), 2),
                'avg_sharpe': round(float(np.mean(sharps)), 2),
                'avg_max_dd': round(float(np.mean(dds)), 2),
                'max_drawdown': round(float(max(dds)), 2),
                'positive_segments': sum(1 for c in cagrs if c > 0),
                'total_segments': len(segment_results),
            }
        }
        out_path = os.path.join(D_DATA, 'a1_l3_walk_forward_results.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f'\n  Results saved: {out_path}')

    tot = time.time() - t0
    print(f'  Total time: {tot:.0f}s ({tot/60:.1f}min)')
    return segment_results


if __name__ == '__main__':
    run_walk_forward()
