#!/usr/bin/env python3
"""
蓝盾V9 每日评分脚本（LightGBM + 43维特征版）
扫描全市场>$10股票 → 43维特征 → LightGBM回归 → 信号分级

V9 = 27技术 + 13扩展价格 + 3基本面（与训练脚本一致）
模型: blueshield_lgb_v9_quantile_lgb.txt (ICIR=0.734)

用法:
    python3 blueshield_v9_score.py              # 标准输出
    python3 blueshield_v9_score.py --json        # JSON输出
    python3 blueshield_v9_score.py --top 10      # 只输出Top-10
"""
import json, sys, os, time, argparse, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── V9特征定义（必须与训练脚本 optimize_blueshield_v9_extended.py 完全一致） ──
TECH_FEATS = ['ma5','ma20','ma60','ma_bias20','ma_align','price_position',
    'ret1','ret5','ret20','ret60','momentum_6m','momentum_1m',
    'mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change',
    'rsi14','rsi_change','macd','macd_signal','macd_hist',
    'bb_std','bb_width','bb_pos','ret_quality']  # 27个

EXT_FEATS = ['range_ratio','avg_body','vwap_drift',
    'ret_10d','ret_30d','ret_90d',
    'vol_regime','ma_cross_5_20','ma_cross_20_60',
    'rsi_zone','macd_roc','dd_60','ud_vol_ratio']  # 13个

FUND_COLS = ['pe_log','div_yield','beta']  # 3个（pe_log=log1p(pe_trailing)）

ALL_FEATS = TECH_FEATS + EXT_FEATS + FUND_COLS  # 43个


def rmean(arr, w):
    """Rolling mean via cumsum (numpy, fast)."""
    out = np.full(len(arr), np.nan)
    cs = np.cumsum(arr)
    out[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
    return out


def rstd(arr, w):
    """Rolling std (numpy, ddof=1)."""
    out = np.full(len(arr), np.nan)
    for i in range(w-1, len(arr)):
        out[i] = np.std(arr[i-w+1:i+1], ddof=1)
    return out


def ema(arr, span):
    """Exponential moving average."""
    out = np.empty(len(arr))
    alpha = 2.0 / (span + 1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out


def pctchg(arr, p):
    """Percent change over p periods."""
    out = np.full(len(arr), np.nan)
    out[p:] = arr[p:] / (arr[:-p] + 1e-10) - 1
    return out


def compute_features_extended(group):
    """
    Compute ALL 43 V9 features matching optimize_blueshield_v9_extended.py exactly.
    Uses numpy arrays for speed (matching training script).
    """
    g = group.sort_values('date').copy()
    c = g['close'].values.astype(np.float64)
    h = g['high'].values.astype(np.float64)
    l = g['low'].values.astype(np.float64)
    o = g['open'].values.astype(np.float64)
    vol = g['volume'].values.astype(np.float64)
    n = len(c)

    # Daily returns
    dr = np.full(n, np.nan)
    dr[1:] = (c[1:] - c[:-1]) / (c[:-1] + 1e-10)

    # Moving averages
    ma5 = rmean(c, 5)
    ma20 = rmean(c, 20)
    ma60 = rmean(c, 60)
    ma_bias20 = (c - ma20) / (ma20 + 1e-10)
    ma_align = ((c > ma5).astype(np.float64) + (ma5 > ma20).astype(np.float64))

    # Price position (60-day)
    mn60 = np.full(n, np.nan)
    mx60 = np.full(n, np.nan)
    for i in range(59, n):
        mn60[i] = np.min(c[i-59:i+1])
        mx60[i] = np.max(c[i-59:i+1])
    price_position = (c - mn60) / (mx60 - mn60 + 1e-10)

    # Returns
    ret1 = pctchg(c, 1)
    ret5 = pctchg(c, 5)
    ret20 = pctchg(c, 20)
    ret60 = pctchg(c, 60)
    momentum_6m = pctchg(c, 126)
    momentum_1m = pctchg(c, 21)
    mom_divergence = momentum_1m - ret20
    trend_accel = np.full(n, np.nan)
    trend_accel[10:] = ret5[10:] - ret5[:-10]

    # Volatility
    vol20 = rstd(dr, 20)
    vol5 = rstd(dr, 5)

    # Volume features
    vol_ma20 = rmean(vol, 20)
    vol_ratio = vol / (vol_ma20 + 1e-10)
    vol_change = np.full(n, np.nan)
    vol_change[20:] = vol20[20:] / (vol20[:-20] + 1e-10)

    # RSI
    delta = np.full(n, 0.0)
    delta[1:] = c[1:] - c[:-1]
    gain = np.where(delta > 0, delta, 0.0)
    loss_arr = np.where(delta < 0, -delta, 0.0)
    gain_ma = rmean(gain, 14)
    loss_ma = rmean(loss_arr, 14)
    rsi14 = 100 - 100 / (1 + gain_ma / (loss_ma + 1e-10))
    rsi_change = np.full(n, np.nan)
    rsi_change[5:] = rsi14[5:] - rsi14[:-5]

    # MACD
    e12 = ema(c, 12)
    e26 = ema(c, 26)
    macd = e12 - e26
    macd_signal = ema(macd, 9)
    macd_hist = macd - macd_signal

    # Bollinger Bands
    bb_std_val = vol20
    bb_mid = ma20
    bb_width = 4 * bb_std_val * bb_mid / (bb_mid + 1e-10)
    price_std = rstd(c, 20)
    bb_pos = (c - (bb_mid - 2 * price_std)) / (4 * price_std + 1e-10)

    # Ret quality
    ret_pos = np.where(dr > 0, dr, 0.0)
    ret_neg = np.where(dr < 0, -dr, 0.0)
    ret_pos_ma = rmean(ret_pos, 20)
    ret_neg_ma = rmean(ret_neg, 20)
    ret_quality = ret_pos_ma / (ret_pos_ma + ret_neg_ma + 1e-10)

    # ============================================
    # EXTENDED FEATURES (13)
    # ============================================

    # 1. Range ratio (daily range / avg range)
    daily_range = (h - l) / (c + 1e-10)
    avg_range = rmean(daily_range, 20)
    range_ratio = daily_range / (avg_range + 1e-10)

    # 2. Average candle body position
    candle_body = (c - o) / (h - l + 1e-10)
    avg_body = rmean(candle_body, 10)

    # 3. VWAP drift
    vwap_drift = np.full(n, np.nan)
    for i in range(20, n):
        w = vol[i-19:i+1]
        p = c[i-19:i+1]
        tw = np.sum(w * p) / (np.sum(w) + 1e-10)
        vwap_drift[i] = (c[i] - tw) / (tw + 1e-10)

    # 4. Return momentum tiers
    ret_10d = pctchg(c, 10)
    ret_30d = pctchg(c, 30)
    ret_90d = pctchg(c, 90)

    # 5. Volatility regime (vol20/vol60)
    vol60 = rstd(dr, 60)
    vol_regime = vol20 / (vol60 + 1e-10)

    # 6. MA crossovers
    ma_cross_5_20 = np.where(ma5 > ma20, 1.0, 0.0)
    ma_cross_20_60 = np.where(ma20 > ma60, 1.0, 0.0)

    # 7. RSI zone (0-1 scale)
    rsi_zone = np.zeros(n)
    rsi_zone[~np.isnan(rsi14)] = rsi14[~np.isnan(rsi14)] / 100.0

    # 8. MACD momentum (rate of change)
    macd_roc = np.full(n, np.nan)
    macd_roc[5:] = macd[5:] - macd[:-5]

    # 9. Drawdown from 60d high
    dd_60 = np.full(n, np.nan)
    for i in range(59, n):
        peak = np.max(c[i-59:i+1])
        dd_60[i] = (c[i] - peak) / (peak + 1e-10)

    # 10. Up/down volume ratio
    up_vol = np.where(dr > 0, vol, 0.0)
    dn_vol = np.where(dr < 0, vol, 0.0)
    up_vol_ma = rmean(up_vol, 20)
    dn_vol_ma = rmean(dn_vol, 20)
    ud_vol_ratio = up_vol_ma / (dn_vol_ma + 1e-10)

    # Stack into DataFrame columns
    g['ma5'] = ma5; g['ma20'] = ma20; g['ma60'] = ma60
    g['ma_bias20'] = ma_bias20; g['ma_align'] = ma_align
    g['price_position'] = price_position
    g['ret1'] = ret1; g['ret5'] = ret5; g['ret20'] = ret20; g['ret60'] = ret60
    g['momentum_6m'] = momentum_6m; g['momentum_1m'] = momentum_1m
    g['mom_divergence'] = mom_divergence; g['trend_accel'] = trend_accel
    g['vol20'] = vol20; g['vol5'] = vol5
    g['vol_ratio'] = vol_ratio; g['vol_change'] = vol_change
    g['rsi14'] = rsi14; g['rsi_change'] = rsi_change
    g['macd'] = macd; g['macd_signal'] = macd_signal; g['macd_hist'] = macd_hist
    g['bb_std'] = bb_std_val; g['bb_width'] = bb_width; g['bb_pos'] = bb_pos
    g['ret_quality'] = ret_quality

    # Extended
    g['range_ratio'] = range_ratio; g['avg_body'] = avg_body; g['vwap_drift'] = vwap_drift
    g['ret_10d'] = ret_10d; g['ret_30d'] = ret_30d; g['ret_90d'] = ret_90d
    g['vol_regime'] = vol_regime
    g['ma_cross_5_20'] = ma_cross_5_20; g['ma_cross_20_60'] = ma_cross_20_60
    g['rsi_zone'] = rsi_zone; g['macd_roc'] = macd_roc
    g['dd_60'] = dd_60; g['ud_vol_ratio'] = ud_vol_ratio

    return g


def load_fundamentals():
    """加载基本面数据，转换为训练格式 (pe_log, div_yield, beta)"""
    fund_path = os.path.join(ROOT, 'data/us/fundamentals_latest.parquet')
    if not os.path.exists(fund_path):
        print("⚠️ 基本面数据不存在，用默认值填充", flush=True)
        return None

    fund = pd.read_parquet(fund_path)

    # pe_trailing → pe_log (winsorize + log1p, matching training)
    pe = pd.to_numeric(fund.get('pe_trailing', 20.0), errors='coerce')
    pe = pe.replace([np.inf, -np.inf], np.nan)
    pe = pe.clip(-50, 200).fillna(20.0)
    pe_log = np.log1p(np.abs(pe)) * np.sign(pe)

    dy = pd.to_numeric(fund.get('div_yield', 0.0), errors='coerce').fillna(0.0)
    be = pd.to_numeric(fund.get('beta', 1.0), errors='coerce').fillna(1.0)

    fund_out = pd.DataFrame({
        'sym': fund['sym'],
        'pe_log': pe_log,
        'div_yield': dy,
        'beta': be,
    })
    return fund_out


def score_all():
    """用本地数据批量评分"""

    print("🛡️ 蓝盾V9 全市场评分 (43维: 27技术+13扩展+3基本面, LightGBM)", flush=True)
    print("="*50, flush=True)

    # 1. 加载数据
    print("1. 加载历史数据...", flush=True)
    df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'))
    df = df.dropna(subset=['close', 'volume'])
    df = df[(df['close'] > 0.5) & (df['close'] < 10000) & (df['volume'] > 0)]
    cutoff = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
    df = df[df['date'] >= cutoff]
    df['date'] = pd.to_datetime(df['date'])
    print(f"   数据量: {len(df)}行, {df['sym'].nunique()}只 (最近250天)", flush=True)

    # 2. 计算43维特征（27技术 + 13扩展 + price基础列）
    print("2. 计算43维特征 (27技术+13扩展+3基本面)...", flush=True)
    t0 = time.time()
    parts = []
    for i, (sym, g) in enumerate(df.groupby('sym')):
        if len(g) < 80:
            continue
        f = compute_features_extended(g); f['sym'] = sym; parts.append(f)
        if (i+1) % 2000 == 0:
            print(f"   {i+1}/{df['sym'].nunique()} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.concat(parts, ignore_index=True)
    print(f"   完成: {time.time()-t0:.0f}s", flush=True)

    # 3. 取每个股票最新一行
    df = df.sort_values('date')
    latest = df.groupby('sym').tail(1).reset_index(drop=True)
    print(f"   最新行: {len(latest)}只", flush=True)
    del df

    # 4. 加载基本面 (pe_log, div_yield, beta)
    print("3. 加载基本面数据...", flush=True)
    fund = load_fundamentals()
    if fund is not None:
        latest = latest.merge(fund, on='sym', how='left')
        for c in FUND_COLS:
            if c not in latest.columns:
                latest[c] = 0
            latest[c] = latest[c].fillna(0)
        print(f"   基本面: {len(fund)}只", flush=True)
    else:
        for c in FUND_COLS:
            latest[c] = 0

    # 5. 过滤
    etfs = {'SPY','QQQ','IWM','DIA','VOO','VTI','IVV','VEA','VWO','BND','AGG','TLT','GLD','SLV','USO'}
    latest = latest[~latest['sym'].isin(etfs)]
    latest = latest[latest['close'] > 10].copy()
    latest = latest[latest['volume'] > 50000].copy()
    for c in ALL_FEATS:
        if c not in latest.columns:
            latest[c] = 0
    latest = latest.dropna(subset=ALL_FEATS)
    print(f"4. 评分股票: {len(latest)}只 (>$10, vol>50K)", flush=True)

    # 6. 加载LightGBM模型
    model_path = os.path.join(ROOT, 'models/us/blueshield_lgb_v9_quantile_lgb.txt')

    if not os.path.exists(model_path):
        print(f"❌ V9模型文件不存在: {model_path}", flush=True)
        return None, None, None

    model = lgb.Booster(model_file=model_path)
    print(f"   模型: {os.path.basename(model_path)}", flush=True)

    # 7. 预测（LightGBM直接用DataFrame/numpy，无需DMatrix）
    print("5. 预测中...", flush=True)
    X = latest[ALL_FEATS].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    preds = model.predict(X)
    latest['pred_rank'] = preds
    latest = latest.sort_values('pred_rank', ascending=False)

    # 获取VIX（用于信号分级）
    vix_val = None
    try:
        vix_path = os.path.join(ROOT, 'data/us/vix_10y.parquet')
        vix_raw = pd.read_parquet(vix_path)
        if isinstance(vix_raw.columns, pd.MultiIndex):
            vix_raw.columns = [c[0] if isinstance(c, tuple) else c for c in vix_raw.columns]
        vix_raw = vix_raw.reset_index()
        vix_col = [c for c in vix_raw.columns if 'close' in c.lower() or 'Close' in c]
        if vix_col:
            vix_val = float(vix_raw[vix_col[0]].iloc[-1])
    except Exception:
        pass

    return latest, vix_val, None


def classify_signal_percentile(score, all_scores, vix=None, meta=None):
    """三层过滤信号分级"""
    if vix is not None and vix > 30:
        return '🔴', 'VIX>30暂停'

    median = np.median(all_scores)
    if score <= median:
        return '🔴', '低于中位数'

    p95 = np.percentile(all_scores, 95)
    p90 = np.percentile(all_scores, 90)
    p80 = np.percentile(all_scores, 80)

    if score >= p95:
        return '🟢🟢', f'Top5%(≥{p95:.4f})'
    elif score >= p90:
        return '🟢', f'Top10%(≥{p90:.4f})'
    elif score >= p80:
        return '🟡', f'Top20%(≥{p80:.4f})'
    else:
        return '🔴', '不推荐'


def main():
    parser = argparse.ArgumentParser(description='蓝盾V9全市场评分 (LightGBM)')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--top', type=int, default=30)
    args = parser.parse_args()

    result = score_all()
    if result[0] is None:
        return
    latest, vix_val, meta = result

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    all_scores = latest['pred_rank'].values

    if args.json:
        picks = []
        for _, r in latest.head(args.top).iterrows():
            emoji, desc = classify_signal_percentile(r['pred_rank'], all_scores, vix_val, meta)
            picks.append({
                'ticker': r['sym'], 'price': round(float(r['close']), 2),
                'pred_rank': round(float(r['pred_rank']), 4),
                'signal': emoji, 'signal_desc': desc,
                'pe': round(float(r.get('pe_log', 0)), 2),
                'beta': round(float(r.get('beta', 1)), 2),
                'div_yield': round(float(r.get('div_yield', 0)) * 100, 2),
            })
        output = {
            'timestamp': now, 'model': 'blueshield_v9_lgb',
            'total_scanned': len(latest), 'top_n': args.top, 'picks': picks
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n🛡️ 蓝盾V9 选股报告 ({now})", flush=True)
        print(f"{'='*75}", flush=True)
        print(f"模型: V9 LightGBM (43维: 27技术+13扩展+3基本面, 20天持有)", flush=True)
        print(f"扫描: {len(latest)}只股票", flush=True)
        print(f"", flush=True)
        print(f"{'排名':<5} {'代码':<8} {'价格':>8} {'排名分':>8} {'信号'}", flush=True)
        print(f"{'-'*75}", flush=True)

        for i, (_, r) in enumerate(latest.head(args.top).iterrows()):
            emoji, desc = classify_signal_percentile(r['pred_rank'], all_scores, vix_val, meta)
            print(f"{emoji}{i+1:<3} {r['sym']:<8} ${r['close']:>7.2f} {r['pred_rank']:>8.4f} "
                  f"{desc}", flush=True)

        # 统计
        median = np.median(all_scores)
        p95 = np.percentile(all_scores, 95)
        p90 = np.percentile(all_scores, 90)
        p80 = np.percentile(all_scores, 80)
        above_median = len(latest[latest['pred_rank'] > median])
        g2 = len(latest[latest['pred_rank'] >= p95])
        g1 = len(latest[(latest['pred_rank'] >= p90) & (latest['pred_rank'] < p95)])
        obs = len(latest[(latest['pred_rank'] >= p80) & (latest['pred_rank'] < p90)])
        vix_str = f"VIX={vix_val:.1f}" if vix_val else "VIX=未知"
        l1_status = "🔴暂停" if (vix_val and vix_val > 30) else "🟢正常"
        print(f"\n📊 三层过滤: {vix_str} {l1_status} | 中位数:{median:.4f} | >中位数:{above_median}只", flush=True)
        print(f"📊 信号分级: 🟢🟢精品(Top5%):{g2}只 | 🟢强信号(Top10%):{g1}只 | 🟡观察(Top20%):{obs}只", flush=True)

    # VIX止损检查
    if vix_val:
        if vix_val > 35:
            print(f"\n🔴🔴 VIX={vix_val:.1f} > 35 恐慌！建议全部清仓不买", flush=True)
        elif vix_val > 30:
            print(f"\n🔴 VIX={vix_val:.1f} > 30 三层过滤L1触发！信号全部关闭", flush=True)
        elif vix_val > 25:
            print(f"\n🟠 VIX={vix_val:.1f} > 25 警戒，建议减仓50%", flush=True)
        elif vix_val > 20:
            print(f"\n🟡 VIX={vix_val:.1f} > 20 注意，收紧止损", flush=True)
        else:
            print(f"\n🟢 VIX={vix_val:.1f} 正常，全仓位", flush=True)

    # 保存信号文件
    os.makedirs(os.path.join(ROOT, 'signals/us'), exist_ok=True)
    save_path = os.path.join(ROOT, 'signals/us/blueshield_v9_scores.json')
    with open(save_path, 'w') as f:
        json.dump({
            'timestamp': now, 'model': 'blueshield_v9_lgb', 'version': 'v9',
            'total': len(latest),
            'picks': [{
                'ticker': r['sym'], 'price': round(float(r['close']), 2),
                'pred_rank': round(float(r['pred_rank']), 4),
                'signal': classify_signal_percentile(r['pred_rank'], all_scores, vix_val, meta)[0]
            } for _, r in latest.head(args.top).iterrows()]
        }, f, indent=2, default=str)

    # 兼容旧路径
    compat_path = os.path.join(ROOT, 'output/v6_latest.json')
    os.makedirs(os.path.dirname(compat_path), exist_ok=True)
    with open(compat_path, 'w') as f:
        json.dump({
            'timestamp': now, 'model': 'blueshield_v9_lgb', 'total': len(latest),
            'picks': [{'ticker': r['sym'], 'price': round(float(r['close']), 2),
                'pred_rank': round(float(r['pred_rank']), 4),
                'signal': classify_signal_percentile(r['pred_rank'], all_scores, vix_val, meta)[0]
            } for _, r in latest.head(args.top).iterrows()]
        }, f, indent=2, default=str)

    print(f"\n✅ 保存: signals/us/blueshield_v9_scores.json + output/v6_latest.json", flush=True)


if __name__ == '__main__':
    main()
