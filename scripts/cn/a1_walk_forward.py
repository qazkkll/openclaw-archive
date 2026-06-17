#!/usr/bin/env python3
"""
A1 Walk-Forward 回测 v4 (永不丢结果版)
========================================
核心改进 (2026-06-14):
1. 每500只股票保存一次特征缓存到磁盘 - 内存可控
2. 回测每50天写一次checkpoint - session挂了也能断点续
3. 所有输出同时写日志文件 + 终端
4. 最终报告存 /home/hermes/.hermes/openclaw-project/data/bt_walk_*.json

用法: python scripts/a1_walk_forward.py [--stop -0.15] [--top 10] [--rebal 5]
"""
import json, sys, os, time, gc, signal, atexit
import numpy as np
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

D_DATA = r'/home/hermes/.hermes/openclaw-archive/data'
MODEL_DIR = os.path.join(D_DATA, 'a1_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Config ──
TOP_K = 10; STOP_LOSS = -0.15; COST_RATE = 0.001; REBALANCE_FREQ = 5
TEST_START = '2024-01-01'; TEST_END = '2026-06-12'

if '--stop' in sys.argv: STOP_LOSS = float(sys.argv[sys.argv.index('--stop')+1])
if '--top' in sys.argv: TOP_K = int(sys.argv[sys.argv.index('--top')+1])
if '--rebal' in sys.argv: REBALANCE_FREQ = int(sys.argv[sys.argv.index('--rebal')+1])
LABEL = f'K={TOP_K} SL={STOP_LOSS*100:.0f}% RB={REBALANCE_FREQ}d'
SLUG = LABEL.replace(' ','').replace('%','pct')

MODEL_PATH = os.path.join(MODEL_DIR, 'a1_tech_v1.json')
FEATURE_KEYS = [
    'pct_ma5','pct_ma10','pct_ma20','pct_ma60','pct_ma120',
    'ma20_slope','ma60_slope','ma_align',
    'vol_10d','vol_60d','vol_ratio','atr20_pct',
    'ret_1d','ret_5d','ret_10d','ret_20d','ret_60d','rsi14',
    'vol_ratio_5_20','kdj_k','kdj_d','kdj_j',
    'macd_dif','macd_dea','macd_bar',
    'bb_width','bb_position','obv_ratio_5_20',
    'ret5_max','ret3_vs_ema12','accel_5_10','ma5_ma10_cross',
    'vol_breakout',
]

# ── File paths ──
FEAT_CACHE = os.path.join(D_DATA, f'bt_feats_{SLUG}.jsonl')
CHECKPOINT = os.path.join(D_DATA, f'bt_ckpt_{SLUG}.json')
RESULT_FILE = os.path.join(D_DATA, f'bt_walk_{SLUG}.json')
LOG_FILE = os.path.join(D_DATA, f'bt_walk_{SLUG}.log')
DAILY_LOG = os.path.join(D_DATA, f'bt_daily_{SLUG}.json')

# ── Signal / atexit handling ──
_aborted = False

def _signal_handler(signum, frame):
    global _aborted
    if _aborted: return
    _aborted = True
    msg = f"\n⚠️ 收到信号 {signum}，保存断点..."
    print(msg, file=sys.stderr)
    _emergency_save()

def _atexit_handler():
    if _aborted:
        _emergency_save()

_saved_state = None

def _emergency_save():
    """Try to save whatever we have"""
    if _saved_state is not None:
        try:
            with open(CHECKPOINT, 'w') as f:
                json.dump(_saved_state, f, ensure_ascii=False)
            print(f"  ✅ 断点保存: {CHECKPOINT}", file=sys.stderr)
        except Exception as e:
            print(f"  ❌ 断点保存失败: {e}", file=sys.stderr)

def log(msg):
    """Print + write to log file"""
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
        f.flush()

def log_json(path, data):
    """Atomic append to a JSON array file"""
    existing = []
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except:
            existing = []
    existing.append(data)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False)
        f.flush()

# ── Feature helpers ──
def _ema(arr, period):
    if len(arr) < 1: return []
    m = 2 / (period + 1); r = [arr[0]]
    for v in arr[1:]:
        r.append((v - r[-1]) * m + r[-1])
    return r

def compute_features_stock(c, h, l, o, v, dates, start_idx=120):
    n = len(c); results = {}; kd = {'k': 50, 'd': 50}
    for i in range(start_idx, n - 10):
        price = c[i]; ds = dates[i]
        if ds < TEST_START or ds > TEST_END: continue
        if price <= 0: continue
        f = {}
        ma5 = np.mean(c[i-4:i+1]); ma10 = np.mean(c[i-9:i+1])
        ma20 = np.mean(c[i-19:i+1]); ma60 = np.mean(c[i-59:i+1])
        ma120 = np.mean(c[i-119:i+1]) if i >= 119 else ma60
        f['pct_ma5'] = (price/ma5-1)*100 if ma5 > 0 else 0
        f['pct_ma10'] = (price/ma10-1)*100 if ma10 > 0 else 0
        f['pct_ma20'] = (price/ma20-1)*100 if ma20 > 0 else 0
        f['pct_ma60'] = (price/ma60-1)*100 if ma60 > 0 else 0
        f['pct_ma120'] = (price/ma120-1)*100 if ma120 > 0 else 0
        mp = np.mean(c[i-24:i-4]); f['ma20_slope'] = (ma20/mp-1)*100 if mp > 0 else 0
        mp = np.mean(c[i-64:i-4]); f['ma60_slope'] = (ma60/mp-1)*100 if mp > 0 else 0
        f['ma_align'] = (ma5/ma60-1)*100 if ma60 > 0 else 0
        f['vol_10d'] = float(np.mean(v[i-9:i+1])); f['vol_60d'] = float(np.mean(v[i-59:i+1]))
        f['vol_ratio'] = f['vol_10d']/f['vol_60d'] if f['vol_60d'] > 0 else 1.0
        tr = [max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(i-19, i+1)]
        f['atr20_pct'] = float(np.mean(tr))/price*100 if price > 0 else 0
        f['ret_1d'] = (c[i]/c[i-1]-1)*100; f['ret_5d'] = (c[i]/c[i-4]-1)*100
        f['ret_10d'] = (c[i]/c[i-9]-1)*100; f['ret_20d'] = (c[i]/c[i-19]-1)*100
        f['ret_60d'] = (c[i]/c[i-59]-1)*100
        gains = [max(c[j]-c[j-1], 0) for j in range(i-13, i+1)]
        losses = [max(c[j-1]-c[j], 0) for j in range(i-13, i+1)]
        ag = np.mean(gains); al = np.mean(losses)
        f['rsi14'] = 100-100/(1+ag/al) if al > 0 else 100
        vol5 = np.mean(v[i-4:i+1]); vol20 = np.mean(v[i-19:i+1])
        f['vol_ratio_5_20'] = vol5/vol20 if vol20 > 0 else 1.0
        low9 = min(l[i-8:i+1]); high9 = max(h[i-8:i+1])
        rsv = (c[i]-low9)/(high9-low9)*100 if high9 > low9 else 50
        kd['k'] = 2/3*kd['k']+1/3*rsv; kd['d'] = 2/3*kd['d']+1/3*kd['k']
        f['kdj_k'] = kd['k']; f['kdj_d'] = kd['d']; f['kdj_j'] = 3*kd['k']-2*kd['d']
        need = min(120, i+1); ema12 = _ema(c[i-need+1:i+1], 12)
        ema26 = _ema(c[i-need+1:i+1], 26)
        if ema12 and ema26:
            dif = ema12[-1]-ema26[-1]; dea_v = dif
            if len(ema12) > 9:
                d = _ema([ema12[j]-ema26[j] for j in range(len(ema12))], 9)
                dea_v = d[-1]
        else:
            dif = 0; dea_v = 0
        f['macd_dif'] = dif; f['macd_dea'] = dea_v; f['macd_bar'] = (dif-dea_v)*2
        std20 = np.std(c[i-19:i+1])
        f['bb_width'] = std20/ma20*100 if ma20 > 0 else 0
        f['bb_position'] = (price-(ma20-2*std20))/(4*std20)*100 if std20 > 0 else 50
        obv_vals = [0]
        for j in range(i-19, i+1):
            if j > 0:
                if c[j] > c[j-1]: obv_vals.append(obv_vals[-1]+v[j])
                elif c[j] < c[j-1]: obv_vals.append(obv_vals[-1]-v[j])
        obv5 = np.mean(obv_vals[-5:]) if len(obv_vals) >= 5 else 0
        obv20 = np.mean(obv_vals) if obv_vals else 0
        f['obv_ratio_5_20'] = obv5/obv20 if obv20 != 0 else 1.0
        f['ret5_max'] = (max(h[i-4:i+1])/price-1)*100
        ema12_v = _ema(c[i-need+1:i+1], 12)[-1]
        f['ret3_vs_ema12'] = (c[i]/ema12_v-1)*100 if ema12_v > 0 else 0
        f['accel_5_10'] = f['ret_5d']-f['ret_10d']
        f['ma5_ma10_cross'] = ma5/ma10-1 if ma10 > 0 else 0
        f['vol_breakout'] = v[i]/np.mean(v[i-39:i+1]) if np.mean(v[i-39:i+1]) > 0 else 1.0
        feat_vec = [f.get(k, 0) for k in FEATURE_KEYS]
        results[ds] = {'feat': feat_vec, 'price': price}
    return results


def main():
    global _saved_state
    t0 = time.time()

    # ── Register signal/atexit ──
    signal.signal(signal.SIGTERM, _signal_handler)
    try:
        signal.signal(signal.SIGINT, _signal_handler)
    except:
        pass
    atexit.register(_atexit_handler)

    log(f"{'━'*45}")
    log(f"A1 Walk-Forward v4 | {LABEL}")
    log(f"{'━'*45}")

    # ── Step 1: Load model ──
    log(f"\n[1] 加载模型...")
    if not os.path.exists(MODEL_PATH):
        log(f"  ❌ 模型不存在: {MODEL_PATH}")
        return None
    model = xgb.Booster(); model.load_model(MODEL_PATH)
    log(f"  ✅ {len(FEATURE_KEYS)}个特征, 模型已加载")

    # ── Step 2: Load historical data ──
    log(f"\n[2] 加载历史数据...")
    t1 = time.time()
    with open(os.path.join(D_DATA, 'a_hist_10y.parquet'), 'r', encoding='utf-8') as f:
        hist = json.load(f)
    codes_zb = sorted([c for c in hist if c.startswith(('6', '0')) and len(hist[c]['c']) >= 200])
    log(f"  {len(codes_zb)}只股票 ({len(hist)}只全量), 加载用时 {time.time()-t1:.0f}s")
    mem = __import__('psutil').Process().memory_info().rss / 1024 / 1024
    log(f"  当前内存: {mem:.0f}MB")

    # ── Step 3: Compute features (with per-500 save) ──
    log(f"\n[3] 特征计算 (每500只存盘)...")
    t1 = time.time(); all_feats = {}; date_set = set()
    batch_no = 0

    # Check for cached features — if cache exists, skip computation
    need_compute = not os.path.exists(FEAT_CACHE)
    if not need_compute:
        log(f"  发现特征缓存: {FEAT_CACHE}，跳过计算")
    else:
        log(f"  无缓存，开始计算特征...")
        for ci, code in enumerate(codes_zb):
            if _aborted:
                log("  ⛔ 用户中断")
                return None
            try:
                rec = hist[code]
                feats = compute_features_stock(rec['c'], rec['h'], rec['l'], rec['o'], rec['v'], rec['dates'])
                for ds, data in feats.items():
                    if ds not in all_feats:
                        all_feats[ds] = {}
                    all_feats[ds][code] = data
                    date_set.add(ds)
            except Exception as e:
                log(f"  ⚠️ {code} 特征计算失败: {e}")
                continue

            # Save checkpoint every 500 stocks
            if (ci + 1) % 500 == 0:
                batch_no += 1
                _save_feat_cache(all_feats, FEAT_CACHE)
                mem = __import__('psutil').Process().memory_info().rss / 1024 / 1024
                log(f"  [{ci+1}/{len(codes_zb)}] {len(all_feats)}天, {len(date_set)}唯一日, RSS {mem:.0f}MB, {time.time()-t1:.0f}s")
                sys.stdout.flush()
                # Clear + reload from cache for next batch
                all_feats.clear()
                all_feats = _load_feat_cache(FEAT_CACHE)
                gc.collect()

        # Final save after all stocks processed
        _save_feat_cache(all_feats, FEAT_CACHE)
        log(f"  特征计算完成: {len(all_feats)}天, {len(date_set)}唯一日, {time.time()-t1:.0f}s")
        log(f"  缓存文件: {FEAT_CACHE}")

    # Always load from cache after (re)computing
    all_feats = _load_feat_cache(FEAT_CACHE)
    sorted_dates = sorted(all_feats.keys())
    log(f"  已加载: {len(all_feats)}天, {len(sorted_dates)}唯一日, 范围 {sorted_dates[0]}~{sorted_dates[-1]}")

    # ── Step 4: Walk-Forward Backtest (with per-50-day checkpoint) ──
    log(f"\n[4] 回测 (每50天存checkpoint)...")
    t2 = time.time()

    # Try to resume from checkpoint
    portfolio = {}; cash = 100000; peak = 100000
    daily_vals = [(sorted_dates[0], 100000)]
    start_di = 0
    stop_loss_count = 0

    if os.path.exists(DAILY_LOG):
        try:
            with open(DAILY_LOG, 'r', encoding='utf-8') as f:
                daily_vals = json.load(f)
            if daily_vals:
                start_di = len(daily_vals) - 1
                cash = daily_vals[-1].get('cash_after', 100000)
                portfolio = {k: v for k, v in daily_vals[-1].get('portfolio', {}).items()}
                peak = daily_vals[-1].get('peak', 100000)
                log(f"  ✅ 恢复checkpoint: 第{start_di}天, cash={cash:.0f}, 持仓{len(portfolio)}只")
        except:
            log(f"  ⚠️ checkpoint损坏, 重新开始")

    for di in range(start_di, len(sorted_dates)):
        date = sorted_dates[di]
        dd = all_feats.get(date, {})
        codes = list(dd.keys())
        if not codes:
            if daily_vals:
                daily_vals.append({'date': date, 'pv': 0, 'cash': cash, 'portfolio': dict(portfolio), 'peak': peak})
            continue

        X = np.array([dd[c]['feat'] for c in codes], dtype=np.float32)
        sc = model.predict(xgb.DMatrix(X, feature_names=FEATURE_KEYS))
        rk = sorted(zip(codes, sc, [dd[c]['price'] for c in codes]), key=lambda x: -x[1])

        if di % REBALANCE_FREQ == 0:
            tc = set(r[0] for r in rk[:TOP_K])
            # Sell
            for c in list(portfolio.keys()):
                if c not in tc:
                    pp = dd.get(c, {}).get('price')
                    if pp and pp > 0:
                        cash += portfolio[c]['shares'] * pp * (1 - COST_RATE)
                    del portfolio[c]
            # Buy
            for c, sc_, pp in rk:
                if len(portfolio) >= TOP_K or c in portfolio or pp <= 0:
                    continue
                sh = min((cash / max(1, TOP_K - len(portfolio))) / pp, cash * 0.99 / pp)
                cs = sh * pp * (1 + COST_RATE)
                if cs > cash:
                    continue
                portfolio[c] = {'price': pp, 'shares': sh}
                cash -= cs

        # Stop-loss
        for c in list(portfolio.keys()):
            pp = dd.get(c, {}).get('price')
            if pp and pp > 0:
                ret_pct = (pp / portfolio[c]['price'] - 1) * 100
                if ret_pct <= STOP_LOSS * 100:
                    cash += portfolio[c]['shares'] * pp * (1 - COST_RATE)
                    del portfolio[c]
                    stop_loss_count += 1

        pv = cash
        for c, e in portfolio.items():
            pp = dd.get(c, {}).get('price')
            last_price = pp or e['price']
            pv += e['shares'] * last_price

        if pv > peak:
            peak = pv

        daily_entry = {
            'date': date, 'di': di, 'pv': round(pv, 2), 'cash': round(cash, 2),
            'cash_after': round(cash, 2),  # for resume compatibility
            'n_pos': len(portfolio),
            'portfolio': {k: {'price': v['price'], 'shares': v['shares']} for k, v in portfolio.items()},
            'peak': round(peak, 2),
        }
        daily_vals.append(daily_entry)

        # Debug: watch for deep drawdowns
        dd_pct = (pv - peak) / peak * 100
        if dd_pct < -30:
            log(f"  ⚠️ 深度回撤! 日{di}({date}): PV={pv:.0f} cash={cash:.0f} 持仓{len(portfolio)} 回撤{dd_pct:.1f}%")

        # Save checkpoint every 50 days
        if (di + 1) % 50 == 0 or di == len(sorted_dates) - 1:
            with open(DAILY_LOG, 'w', encoding='utf-8') as f:
                json.dump(daily_vals, f, ensure_ascii=False)
            mem = __import__('psutil').Process().memory_info().rss / 1024 / 1024
            drawdown = (pv - peak) / peak * 100
            log(f"  CKPT 日{di+1}: PV={pv:.0f} 现金={cash:.0f} 持仓{len(portfolio)} 回撤{drawdown:.1f}% RSS={mem:.0f}MB")
            sys.stdout.flush()

        if _aborted:
            log(f"\n⛔ 用户中断，已保存至第{di}天")
            break

    # ── Step 5: Close final positions ──
    final_val = cash
    last_date = sorted_dates[-1]
    last_dd = all_feats.get(last_date, {})
    for c, e in portfolio.items():
        p = last_dd.get(c, {}).get('price') or 0
        final_val += e['shares'] * p

    # ── Step 6: Metrics ──
    log(f"\n[5] 📊 报告: {LABEL}")
    tr = (final_val / 100000 - 1) * 100
    yr = len(sorted_dates) / 252
    cagr = (final_val / 100000) ** (1 / yr) - 1 if yr > 0 else 0

    pv_vals = [d['pv'] if isinstance(d, dict) else d[1] for d in daily_vals[1:]]
    daily_rets = [(pv_vals[i] / pv_vals[i-1] - 1) for i in range(1, len(pv_vals))]
    sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if daily_rets and np.std(daily_rets) > 0 else 0

    peak2 = 100000; max_dd = 0; max_dd_date = ''
    for d in daily_vals:
        v = d['pv'] if isinstance(d, dict) else d[1]
        dt = d['date'] if isinstance(d, dict) else d[0]
        if v > peak2: peak2 = v
        dd = (v - peak2) / peak2 * 100
        if dd < max_dd:
            max_dd = dd
            max_dd_date = dt

    win_trades = 0; total_trades = stop_loss_count
    log(f"  {'初始资金':>12} ¥100,000")
    log(f"  {'最终资金':>12} ¥{final_val:,.0f}")
    log(f"  {'总收益':>12} {tr:+.2f}%")
    log(f"  {'年化CAGR':>12} {cagr*100:+.2f}%)")
    log(f"  {'夏普':>12} {sharpe:.2f}")
    log(f"  {'最大回撤':>12} {max_dd:.2f}% ({max_dd_date})")
    log(f"  {'止损次数':>12} {stop_loss_count}")
    log(f"  {'交易日':>12} {len(sorted_dates)}")

    # ── Step 7: Save final report ──
    report = {
        'label': LABEL,
        'config': {'top_k': TOP_K, 'stop_loss': STOP_LOSS, 'rebalance': REBALANCE_FREQ, 'cost_rate': COST_RATE},
        'initial': 100000,
        'final': round(final_val, 2),
        'total_return_pct': round(tr, 2),
        'cagr_pct': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'max_dd_pct': round(max_dd, 2),
        'max_dd_date': max_dd_date,
        'stop_loss_count': stop_loss_count,
        'trading_days': len(sorted_dates),
        'test_range': f'{TEST_START}~{TEST_END}',
        'features': FEATURE_KEYS,
        'n_stocks': len(codes_zb),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log(f"\n✅ 最终报告: {RESULT_FILE}")
    log(f"✅ 每日明细: {DAILY_LOG}")
    log(f"✅ 完整日志: {LOG_FILE}")
    log(f"\n总用时: {time.time()-t0:.0f}s")

    # Cleanup: remove checkpoint on successful completion
    for f in [CHECKPOINT, FEAT_CACHE]:
        if os.path.exists(f):
            try:
                os.remove(f)
                log(f"  🧹 清理临时缓存: {f}")
            except:
                pass

    return report


def _save_feat_cache(all_feats, path):
    """Save features as per-date chunks (JSONL format)"""
    with open(path, 'w', encoding='utf-8') as f:
        for date, stocks in sorted(all_feats.items()):
            record = {'date': date, 'stocks': {}}
            for code, data in stocks.items():
                record['stocks'][code] = data
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def _load_feat_cache(path):
    """Load features from JSONL cache"""
    result = {}
    if not os.path.exists(path):
        return result
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            record = json.loads(line)
            result[record['date']] = record['stocks']
    return result


if __name__ == '__main__':
    main()
