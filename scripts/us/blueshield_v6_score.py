#!/usr/bin/env python3
"""
蓝盾V7 每日评分脚本（全市场版）
扫描全市场>$10股票 → 44维特征 → XGBoost排名 → Top-15+信号分级

用法:
    python3 blueshield_v6_score.py              # 标准输出
    python3 blueshield_v6_score.py --json        # JSON输出
    python3 blueshield_v6_score.py --top 10      # 只输出Top-10
"""
import json, sys, os, time, argparse, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PYTHON = '/home/hermes/.hermes/hermes-agent/venv/bin/python3'

MACRO_COLS = ['vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60',
              'qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60',
              'iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60']
FUND_COLS = ['pe_trailing','pe_forward','div_yield','beta']
TECH_FEATS = ['ma5','ma20','ma60','ma_bias20','ma_align','price_position',
    'ret1','ret5','ret20','ret60','momentum_6m','momentum_1m',
    'mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change',
    'rsi14','rsi_change','macd','macd_signal','macd_hist',
    'bb_std','bb_width','bb_pos','ret_quality']
ALL_FEATS = TECH_FEATS + MACRO_COLS + FUND_COLS

def compute_features(group):
    g = group.sort_values('date').copy()
    c = g['close']
    g['ma5'] = c.rolling(5).mean(); g['ma20'] = c.rolling(20).mean(); g['ma60'] = c.rolling(60).mean()
    g['ma_bias20'] = (c - g['ma20']) / g['ma20']
    g['ma_align'] = ((c > g['ma5']).astype(int) + (g['ma5'] > g['ma20']).astype(int))
    mn60 = c.rolling(60).min(); mx60 = c.rolling(60).max()
    g['price_position'] = (c - mn60) / (mx60 - mn60 + 1e-10)
    g['ret1'] = c.pct_change(1); g['ret5'] = c.pct_change(5)
    g['ret20'] = c.pct_change(20); g['ret60'] = c.pct_change(60)
    g['momentum_6m'] = c.pct_change(126); g['momentum_1m'] = c.pct_change(21)
    g['mom_divergence'] = g['momentum_1m'] - g['ret20']
    g['trend_accel'] = g['ret5'] - g['ret5'].shift(5)
    dr = c.pct_change(1)
    g['vol20'] = dr.rolling(20).std(); g['vol5'] = dr.rolling(5).std()
    g['vol_ratio'] = g['volume'] / g['volume'].rolling(20).mean()
    g['vol_change'] = g['vol20'] / g['vol20'].shift(20)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    g['rsi14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    g['rsi_change'] = g['rsi14'].diff(5)
    e12 = c.ewm(span=12).mean(); e26 = c.ewm(span=26).mean()
    g['macd'] = e12 - e26; g['macd_signal'] = g['macd'].ewm(span=9).mean()
    g['macd_hist'] = g['macd'] - g['macd_signal']
    g['bb_std'] = c.rolling(20).std()
    g['bb_width'] = 2 * g['bb_std'] / g['ma20']
    g['bb_pos'] = (c - g['ma20']) / (2 * g['bb_std'] + 1e-10)
    g['ret_quality'] = g['ret20'] / (g['vol20'] + 1e-10)
    return g

def get_universe():
    """从历史数据获取全市场>$10股票池"""
    try:
        df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'))
        # 最近30天有交易且价格>$10
        recent = df[df['date'] >= (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')]
        universe = recent[recent['close'] > 10]['sym'].unique().tolist()
        return universe
    except Exception as e:
        print(f"⚠️ 无法获取股票池: {e}", flush=True)
        return []

def score_all():
    """用本地数据批量评分（不需要yfinance）"""
    import xgboost as xgb
    
    print("🛡️ 蓝盾V7 全市场评分", flush=True)
    print("="*50, flush=True)
    
    # 1. 加载数据（只取最近150天，特征最长窗口126天+buffer）
    print("1. 加载历史数据...", flush=True)
    df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'))
    df = df.dropna(subset=['close', 'volume'])
    df = df[(df['close'] > 0.5) & (df['close'] < 10000) & (df['volume'] > 0)]
    cutoff = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
    df = df[df['date'] >= cutoff]
    print(f"   数据量: {len(df)}行, {df['sym'].nunique()}只 (最近250天)", flush=True)
    
    # 2. 计算特征
    print("2. 计算特征...", flush=True)
    t0 = time.time()
    parts = []
    for i, (sym, g) in enumerate(df.groupby('sym')):
        f = compute_features(g); f['sym'] = sym; parts.append(f)
        if (i+1) % 500 == 0: print(f"   {i+1}/2436 ({time.time()-t0:.0f}s)", flush=True)
    df = pd.concat(parts, ignore_index=True)
    print(f"   完成: {time.time()-t0:.0f}s", flush=True)
    
    # 3. 先取每个股票最新一行（必须先按日期排序）
    df = df.sort_values('date')
    latest = df.groupby('sym').tail(1).reset_index(drop=True)
    print(f"   最新行: {len(latest)}只", flush=True)
    
    # 4. 加宏观特征（只在latest上merge，不是全量）
    try:
        v75 = pd.read_parquet(os.path.join(ROOT, 'data/us/features/us_ml_feats_v75_filtered.parquet'))
        macro_daily = v75[['date']+MACRO_COLS].drop_duplicates(subset=['date'])
        latest = pd.merge(latest, macro_daily, on='date', how='left')
        for col in MACRO_COLS:
            if col in latest.columns: latest[col] = latest[col].ffill().fillna(0)
    except:
        for col in MACRO_COLS:
            latest[col] = 0
    
    # 5. 加基本面特征（sym-only merge，因为v75可能滞后于价格数据）
    try:
        v75 = pd.read_parquet(os.path.join(ROOT, 'data/us/features/us_ml_feats_v75_filtered.parquet'))
        fund_latest = v75.sort_values('date').groupby('sym').tail(1)[['sym']+FUND_COLS]
        latest = pd.merge(latest, fund_latest, on='sym', how='left')
        for col in FUND_COLS:
            if col in latest.columns: latest[col] = latest[col].fillna(0)
    except:
        for col in FUND_COLS:
            latest[col] = 0
    
    # 6. 过滤>$10（用当前价格，不是历史价格）
    latest = latest[latest['close'] > 10].copy()
    latest = latest[latest['volume'] > 50000].copy()  # 流动性过滤
    latest = latest.dropna(subset=ALL_FEATS)
    print(f"3. 评分股票: {len(latest)}只", flush=True)
    
    # 7. 加载模型 (V7 全市场重训练 2026-06-24)
    model_path = os.path.join(ROOT, 'models/us/blueshield_v7_xgb.json')
    meta_path = os.path.join(ROOT, 'models/us/blueshield_v7_meta.json')
    
    if not os.path.exists(model_path):
        print("❌ 模型文件不存在", flush=True)
        return
    
    model = xgb.Booster()
    model.load_model(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    feats = meta['features']
    
    # 8. 预测
    X = latest[feats].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    dtest = xgb.DMatrix(X, feature_names=feats)
    preds = model.predict(dtest)
    latest['pred_rank'] = preds
    
    # 9. 排序
    latest = latest.sort_values('pred_rank', ascending=False)
    
    return latest

def classify_signal_percentile(score, all_scores, vix=None, meta_thresholds=None):
    """三层过滤信号分级（动态校准+绝对底线）
    
    设计原则：🟢🟢必须是精品，无论市场环境。
    
    机制：
    - L1: VIX>30 → 全部🔴（熊市保护）
    - L2: 低于中位数 → 🔴（基本门槛）
    - L3: 百分位排名（Top5%/10%/20%）→ 初步分级
    - L4: 绝对底线校验 → 如果meta阈值在当前分布的合理范围内，执行双重过滤
      * 当meta阈值 > 当前P99时，说明市场低迷，只用百分位（避免全🔴）
      * 当meta阈值 <= 当前P95时，必须同时过绝对底线（精品保证）
    """
    if vix is not None and vix > 30:
        return '🔴', 'VIX>30暂停'
    
    median = np.median(all_scores)
    if score <= median:
        return '🔴', '低于中位数'
    
    p99 = np.percentile(all_scores, 99)
    p95 = np.percentile(all_scores, 95)
    p90 = np.percentile(all_scores, 90)
    p80 = np.percentile(all_scores, 80)
    
    # 从meta加载绝对底线
    gg_abs = meta_thresholds.get('green2', {}).get('threshold', 0) if meta_thresholds else 0
    g_abs = meta_thresholds.get('green1', {}).get('threshold', 0) if meta_thresholds else 0
    y_abs = meta_thresholds.get('observe', {}).get('threshold', 0) if meta_thresholds else 0
    
    # 动态校准：如果绝对阈值高于当前P99，说明市场低迷，只用百分位
    # 如果绝对阈值低于当前P95，执行双重过滤（精品保证）
    gg_use_abs = gg_abs <= p99  # green2绝对阈值是否在当前市场可触发
    g_use_abs = g_abs <= p95    # green1绝对阈值是否在当前市场可触发
    y_use_abs = y_abs <= p90    # observe绝对阈值是否在当前市场可触发
    
    # 分级判定
    if score >= p95:
        if gg_use_abs and score >= gg_abs:
            return '🟢🟢', f'Top5%(≥{p95:.3f}) 绝对≥{gg_abs:.3f}'
        elif not gg_use_abs:
            return '🟢🟢', f'Top5%(≥{p95:.3f})'  # 市场低迷，纯百分位
        else:
            return '🟢', f'Top5%但未过绝对线'  # 过百分位但没过绝对底线
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
    parser = argparse.ArgumentParser(description='蓝盾V6全市场评分')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--top', type=int, default=15)
    args = parser.parse_args()
    
    latest = score_all()
    if latest is None:
        return
    
    # 加载绝对阈值
    meta_path = os.path.join(ROOT, 'models/us/blueshield_v7_meta.json')
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
        import yfinance as yf
        vix_val = float(yf.Ticker('^VIX').history(period='1d')['Close'].iloc[-1])
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
            'timestamp': now, 'model': 'blueshield_v7',
            'total_scanned': len(latest), 'top_n': args.top, 'picks': picks
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n🛡️ 蓝盾V7 选股报告 ({now})", flush=True)
        print(f"{'='*65}", flush=True)
        print(f"模型: V7排名 (44维特征, 20天Top-15, 全市场)", flush=True)
        print(f"扫描: {len(latest)}只股票", flush=True)
        print(f"", flush=True)
        print(f"{'排名':<5} {'代码':<8} {'价格':>8} {'排名分':>7} {'RSI':>5} {'5日':>7} {'20日':>7} {'信号'}", flush=True)
        print(f"{'-'*65}", flush=True)
        
        for i, (_, r) in enumerate(latest.head(args.top).iterrows()):
            emoji, desc = classify_signal_percentile(r['pred_rank'], all_scores, vix_val, meta_thresholds)
            print(f"{emoji}{i+1:<3} {r['sym']:<8} ${r['close']:>7.2f} {r['pred_rank']:>7.4f} "
                  f"{r.get('rsi14',0):>5.1f} {r.get('ret5',0)*100:>+6.1f}% {r.get('ret20',0)*100:>+6.1f}% {desc}", flush=True)
        
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
        print(f"\n📊 三层过滤: {vix_str} {l1_status} | 中位数:{median:.3f} | >中位数:{above_median}只", flush=True)
        print(f"📊 信号分级: 🟢🟢精品(Top5%):{g2}只 | 🟢强信号(Top10%):{g1}只 | 🟡观察(Top20%):{obs}只", flush=True)
    
    # VIX止损检查
    if vix_val:
        if vix_val > 35:
            print(f"\n🔴🔴 VIX={vix_val:.1f} > 35 恐慌！建议全部清仓不买", flush=True)
        elif vix_val > 30:
            print(f"\n🔴 VIX={vix_val:.1f} > 30 三层过滤L1触发！信号全部关闭，不买任何新股票", flush=True)
        elif vix_val > 25:
            print(f"\n🟠 VIX={vix_val:.1f} > 25 警戒，建议减仓50%", flush=True)
        elif vix_val > 20:
            print(f"\n🟡 VIX={vix_val:.1f} > 20 注意，收紧止损", flush=True)
        else:
            print(f"\n🟢 VIX={vix_val:.1f} 正常，全仓位", flush=True)
    else:
        print(f"\n⚠️ VIX获取失败，请手动检查", flush=True)
    
    # 保存
    output_dir = os.path.join(ROOT, 'output')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(ROOT, 'signals/us/blueshield_v7_scores.json')
    with open(save_path, 'w') as f:
        json.dump({
            'timestamp': now, 'model': 'blueshield_v7',
            'total': len(latest),
            'picks': [{
                'ticker': r['sym'], 'price': round(r['close'], 2),
                'pred_rank': round(r['pred_rank'], 4),
                'signal': classify_signal_percentile(r['pred_rank'], all_scores, vix_val, meta_thresholds)[0]
            } for _, r in latest.head(args.top).iterrows()]
        }, f, indent=2, default=str)
    # 兼容旧路径
    compat_path = os.path.join(output_dir, 'v6_latest.json')
    with open(compat_path, 'w') as f:
        json.dump({
            'timestamp': now, 'model': 'blueshield_v7', 'total': len(latest),
            'picks': [{'ticker': r['sym'], 'price': round(r['close'], 2),
                'pred_rank': round(r['pred_rank'], 4),
                'signal': classify_signal_percentile(r['pred_rank'], all_scores, vix_val, meta_thresholds)[0]
            } for _, r in latest.head(args.top).iterrows()]
        }, f, indent=2, default=str)
    print(f"\n✅ 保存: signals/us/blueshield_v7_scores.json + output/v6_latest.json", flush=True)

if __name__ == '__main__':
    main()
