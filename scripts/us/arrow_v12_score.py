#!/usr/bin/env python3
"""
绿箭V12 每日评分脚本（LightGBM + 36维特征版）
扫描全市场$1-$10股票 → 36维特征 → LightGBM排名 → Top-5+信号分级

V12 = 27技术 + price + range_pct + 5宏观(VIX+SPY×4) + 2资金流(CMF+OBV)
模型: arrow_v12_lgb_v12_10d_hold.txt (ICIR=2.005)

用法:
    python3 arrow_v11_score.py              # 标准输出
    python3 arrow_v11_score.py --json        # JSON输出
    python3 arrow_v11_score.py --top 10      # 只输出Top-10
"""
import json, sys, os, time, argparse, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── V12特征定义（必须与训练脚本 retrain_arrow_v12_lgb.py 完全一致） ──
BASE_FEATS = ['ma5','ma20','ma60','ma_bias20','ma_align','price_position',
    'ret1','ret5','ret20','ret60','momentum_6m','momentum_1m',
    'mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change',
    'rsi14','rsi_change','macd','macd_signal','macd_hist',
    'bb_std','bb_width','bb_pos','ret_quality','price','range_pct',
    'vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60']  # 34个
FLOW_FEATS = ['cmf_20','obv_slope_20']  # 2个
ALL_FEATS = BASE_FEATS + FLOW_FEATS  # 36个


def compute_tech_features(group):
    """
    Compute technical + price + range features matching training (retrain_arrow_v12_lgb.py).
    """
    g = group.sort_values('date').copy()
    c = g['close']
    g['ma5'] = c.rolling(5).mean()
    g['ma20'] = c.rolling(20).mean()
    g['ma60'] = c.rolling(60).mean()
    g['ma_bias20'] = (c - g['ma20']) / g['ma20']
    g['ma_align'] = ((c > g['ma5']).astype(int) + (g['ma5'] > g['ma20']).astype(int))
    mn60 = c.rolling(60).min()
    mx60 = c.rolling(60).max()
    g['price_position'] = (c - mn60) / (mx60 - mn60 + 1e-10)
    g['ret1'] = c.pct_change(1)
    g['ret5'] = c.pct_change(5)
    g['ret20'] = c.pct_change(20)
    g['ret60'] = c.pct_change(60)
    g['momentum_6m'] = c.pct_change(126)
    g['momentum_1m'] = c.pct_change(21)
    g['mom_divergence'] = g['momentum_1m'] - g['ret20']
    g['trend_accel'] = g['ret5'] - g['ret5'].shift(5)
    dr = c.pct_change(1)
    g['vol20'] = dr.rolling(20).std()
    g['vol5'] = dr.rolling(5).std()
    g['vol_ratio'] = g['volume'] / g['volume'].rolling(20).mean()
    g['vol_change'] = g['vol20'] / g['vol20'].shift(20)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    g['rsi14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    g['rsi_change'] = g['rsi14'].diff(5)
    e12 = c.ewm(span=12).mean()
    e26 = c.ewm(span=26).mean()
    g['macd'] = e12 - e26
    g['macd_signal'] = g['macd'].ewm(span=9).mean()
    g['macd_hist'] = g['macd'] - g['macd_signal']
    g['bb_std'] = dr.rolling(20).std()
    bb_mid = c.rolling(20).mean()
    g['bb_width'] = 4 * g['bb_std'] * bb_mid / (bb_mid + 1e-10)
    std20 = c.rolling(20).std()
    g['bb_pos'] = (c - (bb_mid - 2 * std20)) / (4 * std20 + 1e-10)
    ret_pos = dr.clip(lower=0).rolling(20).mean()
    ret_neg = (-dr).clip(lower=0).rolling(20).mean()
    g['ret_quality'] = ret_pos / (ret_pos + ret_neg + 1e-10)
    g['price'] = c
    g['range_pct'] = (g['high'] - g['low']) / (c + 1e-10)
    return g


def compute_flow_features(group):
    """
    CMF-20 and OBV slope-20 (matching retrain_arrow_v12_lgb.py exactly).
    """
    g = group.copy()
    h, l, c, v = g['high'], g['low'], g['close'], g['volume']

    # Chaikin Money Flow
    mf_vol = v * (2*c - l - h) / (h - l + 1e-10)
    pos_mf = mf_vol.clip(lower=0)
    neg_mf = (-mf_vol).clip(lower=0)
    g['cmf_20'] = (pos_mf.rolling(20).sum() - neg_mf.rolling(20).sum()) / (v.rolling(20).sum() + 1e-10)

    # OBV slope
    import numpy as np
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    obv_arr = obv.values.astype(np.float64)
    n = len(obv_arr)
    W = 20
    if n >= W:
        try:
            windows = np.lib.stride_tricks.sliding_window_view(obv_arr, W)
        except AttributeError:
            shape = (n - W + 1, W)
            strides = (obv_arr.strides[0], obv_arr.strides[0])
            windows = np.lib.stride_tricks.as_strided(obv_arr, shape=shape, strides=strides)
        x = np.arange(W, dtype=np.float64)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        y_means = windows.mean(axis=1)
        cov = ((windows - y_means[:, None]) * (x - x_mean)).sum(axis=1)
        slope_full = np.zeros(n)
        slope_full[W-1:] = cov / x_var
        vol_ma20 = v.rolling(20).mean().values + 1e-10
        g['obv_slope_20'] = slope_full / vol_ma20
    else:
        g['obv_slope_20'] = 0.0

    return g


def score_all():
    """用本地数据批量评分"""

    print("🎯 绿箭V12 全市场评分 ($1-$10, 36维特征, LightGBM)", flush=True)
    print("="*50, flush=True)

    # 加载数据（只取最近250天，特征最长窗口126天+buffer）
    print("1. 加载历史数据...", flush=True)
    df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'))
    df = df.dropna(subset=['close', 'volume'])
    df = df[(df['close'] > 0.5) & (df['volume'] > 0)]
    cutoff = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
    df = df[df['date'] >= cutoff]
    print(f"   数据量: {len(df)}行, {df['sym'].nunique()}只 (最近250天)", flush=True)

    # 计算技术特征 + 流量特征
    print("2. 计算36维特征 (29技术+5宏观+2资金流)...", flush=True)
    t0 = time.time()
    parts = []
    for i, (sym, g) in enumerate(df.groupby('sym')):
        if len(g) < 80:
            continue
        f = compute_tech_features(g)
        f = compute_flow_features(f)
        f['sym'] = sym
        parts.append(f)
    df = pd.concat(parts, ignore_index=True)
    print(f"   完成: {time.time()-t0:.0f}s", flush=True)

    # 取每个股票最后一行
    df = df.sort_values('date')
    latest = df.groupby('sym').tail(1).reset_index(drop=True)
    print(f"   最新行: {len(latest)}只", flush=True)
    del df

    # 加载宏观特征（仅VIX + SPY × 4周期，匹配训练）
    print("3. 加载宏观特征 (VIX+SPY×4周期)...", flush=True)
    macro_path = os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet')
    try:
        df_spy = pd.read_parquet(macro_path, columns=['sym','date','close'])
        df_spy['date'] = pd.to_datetime(df_spy['date'])
        spy = df_spy[df_spy['sym'] == 'SPY'][['date','close']].sort_values('date')
        latest_date = latest['date'].max()
        spy = spy[spy['date'] <= latest_date]
        spy_latest = spy.tail(1)
        if len(spy_latest) > 0:
            spy_close = spy['close'].values
            spy_dates = spy['date'].values
            # Compute returns from close series
            latest['spy_ret1'] = (spy_close[-1] / (spy_close[-2] + 1e-10) - 1) if len(spy_close) >= 2 else 0
            latest['spy_ret5'] = (spy_close[-1] / (spy_close[-6] + 1e-10) - 1) if len(spy_close) >= 6 else 0
            latest['spy_ret20'] = (spy_close[-1] / (spy_close[-21] + 1e-10) - 1) if len(spy_close) >= 21 else 0
            latest['spy_ret60'] = (spy_close[-1] / (spy_close[-61] + 1e-10) - 1) if len(spy_close) >= 61 else 0
        else:
            for c in ['spy_ret1','spy_ret5','spy_ret20','spy_ret60']:
                latest[c] = 0
        del df_spy, spy
    except Exception as e:
        print(f"   ⚠️ SPY数据加载失败: {e}", flush=True)
        for c in ['spy_ret1','spy_ret5','spy_ret20','spy_ret60']:
            latest[c] = 0

    # VIX
    try:
        vix_raw = pd.read_parquet(os.path.join(ROOT, 'data/us/vix_10y.parquet'))
        if isinstance(vix_raw.columns, pd.MultiIndex):
            vix_raw.columns = [c[0] if isinstance(c, tuple) else c for c in vix_raw.columns]
        vix_raw = vix_raw.reset_index()
        vix_col = [c for c in vix_raw.columns if 'close' in c.lower() or 'Close' in c]
        date_col = [c for c in vix_raw.columns if 'date' in c.lower() or 'Date' in c]
        if vix_col:
            vix_val = float(vix_raw[vix_col[0]].iloc[-1])
            latest['vix_close'] = vix_val
        else:
            latest['vix_close'] = 20
            vix_val = 20
    except Exception:
        latest['vix_close'] = 20
        vix_val = 20
    print(f"   VIX: {latest['vix_close'].values[0]:.2f}", flush=True)

    # Fill macro NaN
    for col in ['vix_close', 'spy_ret1', 'spy_ret5', 'spy_ret20', 'spy_ret60']:
        if col in latest.columns:
            latest[col] = latest[col].fillna(0)

    # 用当前价格过滤$1-$10（不是历史价格）
    latest = latest[(latest['close'] >= 1) & (latest['close'] < 10)].copy()
    latest = latest[latest['volume'] > 50000].copy()
    for col in ALL_FEATS:
        if col not in latest.columns:
            latest[col] = 0
    latest = latest.dropna(subset=ALL_FEATS)
    print(f"4. 评分股票: {len(latest)}只 ($1-$10)", flush=True)

    # 加载LightGBM模型
    model_path = os.path.join(ROOT, 'models/us/arrow_v12_lgb_v12_10d_hold.txt')

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}", flush=True)
        return

    model = lgb.Booster(model_file=model_path)
    print(f"   模型: {os.path.basename(model_path)}", flush=True)

    # 预测（LightGBM直接用numpy array）
    X = latest[ALL_FEATS].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    preds = model.predict(X)
    latest['pred_rank'] = preds
    latest = latest.sort_values('pred_rank', ascending=False)

    return latest


def classify_signal_percentile(score, all_scores, vix=None, meta_thresholds=None):
    """三层过滤信号分级（动态校准+绝对底线）"""
    if vix is not None and vix > 30:
        return '🔴', 'VIX>30暂停'

    median = np.median(all_scores)
    if score <= median:
        return '🔴', '低于中位数'

    p99 = np.percentile(all_scores, 99)
    p95 = np.percentile(all_scores, 95)
    p90 = np.percentile(all_scores, 90)
    p80 = np.percentile(all_scores, 80)

    gg_abs = meta_thresholds.get('green2', {}).get('threshold', 0) if meta_thresholds else 0
    g_abs = meta_thresholds.get('green1', {}).get('threshold', 0) if meta_thresholds else 0
    y_abs = meta_thresholds.get('observe', {}).get('threshold', 0) if meta_thresholds else 0

    gg_use_abs = gg_abs <= p99
    g_use_abs = g_abs <= p95
    y_use_abs = y_abs <= p90

    if score >= p95:
        if gg_use_abs and score >= gg_abs:
            return '🟢🟢', f'Top5%(≥{p95:.3f}) 绝对≥{gg_abs:.3f}'
        elif not gg_use_abs:
            return '🟢🟢', f'Top5%(≥{p95:.3f})'
        else:
            return '🟢', f'Top5%但未过绝对线'
    elif score >= p90:
        if g_use_abs and score >= g_abs:
            return '🟢', f'Top10%(≥{p90:.3f}) 绝对≥{g_abs:.3f}'
        elif not g_use_abs:
            return '🟢', f'Top10%(≥{p90:.3f})'
        else:
            return '🟡', f'Top10%但未过绝对线'
    elif score >= p80:
        if y_use_abs and score >= y_abs:
            return '🟡', f'Top20%(≥{p80:.3f})'
        elif not y_use_abs:
            return '🟡', f'Top20%(≥{p80:.3f})'
        else:
            return '🔴', '不推荐'
    else:
        return '🔴', '不推荐'

def main():
    parser = argparse.ArgumentParser(description='绿箭V12全市场评分 (LightGBM)')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()

    latest = score_all()
    if latest is None:
        return

    # 加载绝对阈值
    meta_path = os.path.join(ROOT, 'models/us/arrow_v12_meta.json')
    meta_thresholds = None
    try:
        with open(meta_path) as f:
            meta_data = json.load(f)
        meta_thresholds = meta_data.get('signal_thresholds', None)
    except:
        pass

    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    # 获取VIX
    vix_val = None
    try:
        vix_raw = pd.read_parquet(os.path.join(ROOT, 'data/us/vix_10y.parquet'))
        if isinstance(vix_raw.columns, pd.MultiIndex):
            vix_raw.columns = [c[0] if isinstance(c, tuple) else c for c in vix_raw.columns]
        vix_raw = vix_raw.reset_index()
        vix_col = [c for c in vix_raw.columns if 'close' in c.lower() or 'Close' in c]
        if vix_col:
            vix_val = float(vix_raw[vix_col[0]].iloc[-1])
    except:
        pass

    all_scores = latest['pred_rank'].values

    if args.json:
        picks = []
        for _, r in latest.head(args.top).iterrows():
            emoji, desc = classify_signal_percentile(r['pred_rank'], all_scores, vix_val, meta_thresholds)
            picks.append({
                'ticker': r['sym'], 'price': round(r['close'], 2),
                'pred_rank': round(r['pred_rank'], 4),
                'signal': emoji, 'signal_desc': desc,
                'rsi': round(r.get('rsi14', 0), 1),
                'ret_5d': round(r.get('ret5', 0) * 100, 1),
                'ret_20d': round(r.get('ret20', 0) * 100, 1),
            })
        output = {
            'timestamp': now, 'model': 'arrow_v12_lgb',
            'total_scanned': len(latest), 'top_n': args.top, 'picks': picks
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n🎯 绿箭V12 选股报告 ({now})", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"模型: V12 LightGBM (36维特征, 10天Top-5, $1-$10, 全市场)", flush=True)
        print(f"扫描: {len(latest)}只股票", flush=True)
        print(f"", flush=True)
        print(f"{'排名':<5} {'代码':<8} {'价格':>8} {'排名分':>7} {'RSI':>5} {'5日':>7} {'20日':>7} {'信号'}", flush=True)
        print(f"{'-'*60}", flush=True)

        for i, (_, r) in enumerate(latest.head(args.top).iterrows()):
            emoji, desc = classify_signal_percentile(r['pred_rank'], all_scores, vix_val, meta_thresholds)
            print(f"{emoji}{i+1:<3} {r['sym']:<8} ${r['close']:>7.2f} {r['pred_rank']:>7.4f} "
                  f"{r.get('rsi14',0):>5.1f} {r.get('ret5',0)*100:>+6.1f}% {r.get('ret20',0)*100:>+6.1f}% {desc}", flush=True)

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
        print(f"\n📊 三层过滤: {vix_str} {l1_status} | 中位数:{median:.3f} | >中位数:{above_median}只", flush=True)
        print(f"📊 信号分级: 🟢🟢精品(Top5%):{g2}只 | 🟢强信号(Top10%):{g1}只 | 🟡观察(Top20%):{obs}只", flush=True)
        print(f"💰 建议: 🟢🟢每只$2000 | 🟢每只$1000 | 🟡观察不买 | 止损-10%", flush=True)

    # VIX止损检查
    if vix_val:
        if vix_val > 35:
            print(f"\n🔴🔴 VIX={vix_val:.1f} > 35 恐慌！绿箭暂停买入", flush=True)
        elif vix_val > 30:
            print(f"\n🔴 VIX={vix_val:.1f} > 30 三层过滤L1触发！绿箭信号全部关闭", flush=True)
        elif vix_val > 25:
            print(f"\n🟠 VIX={vix_val:.1f} > 25 警戒，绿箭减仓", flush=True)
        elif vix_val > 20:
            print(f"\n🟡 VIX={vix_val:.1f} > 20 注意，收紧止损到-8%", flush=True)
        else:
            print(f"\n🟢 VIX={vix_val:.1f} 正常，可正常操作", flush=True)
    else:
        print(f"\n⚠️ VIX获取失败", flush=True)

    # 保存
    output_dir = os.path.join(ROOT, 'output')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(ROOT, 'signals/us/arrow_v12_scores.json')
    with open(save_path, 'w') as f:
        json.dump({
            'timestamp': now, 'model': 'arrow_v12_lgb',
            'total': len(latest),
            'picks': [{
                'ticker': r['sym'], 'price': round(r['close'], 2),
                'pred_rank': round(r['pred_rank'], 4),
                'signal': classify_signal_percentile(r['pred_rank'], all_scores, vix_val, meta_thresholds)[0]
            } for _, r in latest.head(args.top).iterrows()]
        }, f, indent=2, default=str)
    # 兼容旧路径
    compat_path = os.path.join(output_dir, 'v11_latest.json')
    with open(compat_path, 'w') as f:
        json.dump({
            'timestamp': now, 'model': 'arrow_v12_lgb', 'total': len(latest),
            'picks': [{'ticker': r['sym'], 'price': round(r['close'], 2),
                'pred_rank': round(r['pred_rank'], 4),
                'signal': classify_signal_percentile(r['pred_rank'], all_scores, vix_val, meta_thresholds)[0]
            } for _, r in latest.head(args.top).iterrows()]
        }, f, indent=2, default=str)
    print(f"\n✅ 保存: signals/us/arrow_v12_scores.json + output/v11_latest.json", flush=True)

if __name__ == '__main__':
    main()
