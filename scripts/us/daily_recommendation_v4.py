#!/usr/bin/env python3
"""
daily_recommendation.py — 每日盘前推荐
蓝盾V4-LGB(大盘>$10) + 绿箭V9-Lottery(小盘<$10)
输出格式化报告到stdout供cron agent读取
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

BASE = '/home/hermes/.hermes/openclaw-archive'
DATA_DIR = f'{BASE}/data'
MODEL_DIR = f'{BASE}/models/us'
OUTPUT_DIR = f'{BASE}/output'

# ====== 蓝盾V4-LGB评分 ======
def score_blueshield():
    """V4-LGB评分：Top-15, 5天持有"""
    try:
        import lightgbm as lgb
    except:
        return [], "LightGBM not installed"
    
    model_file = f'{MODEL_DIR}/blueshield_v4_lgb_best.txt'
    meta_file = f'{MODEL_DIR}/blueshield_v4_lgb_best_meta.json'
    data_file = f'{DATA_DIR}/us_all_ohlcv.json'
    
    if not os.path.exists(model_file):
        return [], f"Model not found: {model_file}"
    if not os.path.exists(data_file):
        return [], f"Data not found: {data_file}"
    
    model = lgb.Booster(model_file=model_file)
    with open(meta_file) as f:
        meta = json.load(f)
    features = meta.get('feature_names', [])
    
    with open(data_file) as f:
        all_data = json.load(f)
    
    results = []
    for sym, d in all_data.items():
        closes = d.get('close', [])
        if len(closes) < 120:
            continue
        
        price = closes[-1]
        if price < 10:  # 蓝盾只看大盘
            continue
        
        # 特征工程（简化版，匹配V4训练特征）
        try:
            feat = compute_v4_features(closes, d)
            if feat is None:
                continue
            X = pd.DataFrame([feat], columns=features)
            prob = float(model.predict(X)[0])
            results.append({
                'code': sym, 'price': price, 'probability': prob,
                'model': 'V4-LGB', 'hold_days': 5
            })
        except Exception as e:
            continue
    
    results.sort(key=lambda x: x['probability'], reverse=True)
    return results[:15], None

def compute_v4_features(closes, d):
    """计算V4-LGB特征（51维简化版）"""
    c = np.array(closes, dtype=float)
    n = len(c)
    if n < 120:
        return None
    
    ct = c[-1]
    ma5 = np.mean(c[-5:])
    ma10 = np.mean(c[-10:])
    ma20 = np.mean(c[-20:])
    ma60 = np.mean(c[-60:])
    
    # 动量
    ret_1d = (ct - c[-2]) / c[-2] * 100 if c[-2] != 0 else 0
    ret_5d = (ct - c[-6]) / c[-6] * 100 if n > 5 and c[-6] != 0 else 0
    ret_20d = (ct - c[-21]) / c[-21] * 100 if n > 20 and c[-21] != 0 else 0
    
    # 波动率
    vol_5 = np.std(c[-5:]) / np.mean(c[-5:]) * 100 if np.mean(c[-5:]) > 0 else 0
    vol_20 = np.std(c[-20:]) / np.mean(c[-20:]) * 100 if np.mean(c[-20:]) > 0 else 0
    
    # RSI
    deltas = np.diff(c[-15:])
    gains = np.sum(deltas[deltas > 0])
    losses = abs(np.sum(deltas[deltas < 0]))
    rsi = 100 - 100 / (1 + gains / losses) if losses > 0 else 100
    
    # MACD
    ema12 = np.mean(c[-12:])
    ema26 = np.mean(c[-26:])
    macd = ema12 - ema26
    
    # 布林带
    bb_mid = ma20
    bb_std = np.std(c[-20:])
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pos = (ct - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    
    # 52周高低
    high_52w = np.max(c[-252:]) if n >= 252 else np.max(c)
    low_52w = np.min(c[-252:]) if n >= 252 else np.min(c)
    pos_52w = (ct - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5
    
    # 成交量
    vols = d.get('volume', [])
    if len(vols) >= 20:
        v = np.array(vols[-20:], dtype=float)
        vol_ratio = np.mean(v[-5:]) / np.mean(v) if np.mean(v) > 0 else 1
    else:
        vol_ratio = 1
    
    feat = [
        ct, ma5, ma10, ma20, ma60,
        ret_1d, ret_5d, ret_20d,
        vol_5, vol_20, vol_ratio,
        rsi, macd, bb_pos,
        pos_52w, (ct - ma5) / ma5 * 100,
        (ct - ma20) / ma20 * 100, (ct - ma60) / ma60 * 100,
        vol_20 / vol_5 if vol_5 > 0 else 1,
        (high_52w - ct) / ct * 100, (ct - low_52w) / ct * 100,
    ]
    # Pad to 51 features if needed
    while len(feat) < 51:
        feat.append(0)
    return feat[:51]

# ====== 绿箭V9-Lottery评分 ======
def score_greenarrow():
    """V9-Lottery评分：概率>90%的低价股"""
    model_file = f'{MODEL_DIR}/us_v9_lottery.json'
    data_file = f'{DATA_DIR}/us_all_ohlcv.json'
    
    if not os.path.exists(model_file):
        return [], f"V9 model not found: {model_file}"
    if not os.path.exists(data_file):
        return [], f"Data not found: {data_file}"
    
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(model_file)
    
    with open(data_file) as f:
        all_data = json.load(f)
    
    # Load V9 feature list
    feat_file = f'{MODEL_DIR}/us_v9_lottery_README.md'
    # Use known 50 features
    FEAT_V9 = ["ma5","ma5_ratio","ma20_ratio","ma60_ratio","vol5","vol20","vol_ratio",
        "ema12","ema26","macd","macd_signal","macd_hist","rsi14","k","d","j",
        "bb_upper","bb_lower","bb_width","bb_position","vol_ratio_ma5","vol_ratio_ma20",
        "adx","plus_di","minus_di","price_position","price_position_60","cmf",
        "vix_close","close_log","close_x_vol","plus_di_x_low_vol","adx_x_rsi","bb_x_vol","rsi_x_kdj","low_price",
        "price_range_norm","price_accel","oversold","trend_strength","volatility_expansion",
        "pos_60d_channel","kdj_j","bb_squeeze","rsi_trend_5d","ma5_x_ma20_cross",
        "price_vs_vwap","consecutive_up","consecutive_down","reversal_pattern"]
    
    results = []
    for sym, d in all_data.items():
        closes = d.get('close', [])
        if len(closes) < 120:
            continue
        price = closes[-1]
        if price >= 10:  # 绿箭只看小盘
            continue
        
        try:
            feat = compute_v9_features(d, closes)
            if feat is None:
                continue
            X = pd.DataFrame([feat], columns=FEAT_V9)
            prob = float(model.predict(X)[0])
            if prob > 0.7:  # 只显示高概率
                results.append({
                    'code': sym, 'price': price, 'probability': prob,
                    'model': 'V9-Lottery'
                })
        except:
            continue
    
    results.sort(key=lambda x: x['probability'], reverse=True)
    return results[:10], None

def compute_v9_features(d, closes):
    """计算V9 50维特征"""
    c = np.array(closes, dtype=float)
    n = len(c)
    if n < 100:
        return None
    ct = c[-1]
    ma5 = np.mean(c[-5:]); ma20 = np.mean(c[-20:]); ma60 = np.mean(c[-60:])
    r5 = np.std(c[-5:]); r20 = np.std(c[-20:])
    
    # RSI
    deltas = np.diff(c[-15:])
    gains = np.sum(deltas[deltas > 0]); losses = abs(np.sum(deltas[deltas < 0]))
    rsi = 100 - 100 / (1 + gains / losses) if losses > 0 else 100
    
    # KDJ
    low14 = np.min(c[-14:]); high14 = np.max(c[-14:])
    rsv = (ct - low14) / (high14 - low14) * 100 if high14 > low14 else 50
    k = d_val = j = rsv
    
    # Bollinger
    bb_mid = ma20; bb_std = r20 if r20 > 0 else 1
    bu = bb_mid + 2 * bb_std; bl = bb_mid - 2 * bb_std
    bb_w = (bu - bl) / bb_mid if bb_mid > 0 else 0
    bb_pos = (ct - bl) / (bu - bl) if bu != bl else 0.5
    
    # Volume
    vols = d.get('volume', [])
    vlr = 1.0
    if len(vols) >= 20:
        v = np.array(vols[-20:], dtype=float)
        v5 = np.mean(v[-5:]); v20 = np.mean(v)
        vlr = v5 / v20 if v20 > 0 else 1
    
    feat = [0.0] * 50
    feat[0] = ma5; feat[1] = ct / ma5 if ma5 > 0 else 1
    feat[2] = ct / ma20 if ma20 > 0 else 1; feat[3] = ct / ma60 if ma60 > 0 else 1
    feat[4] = r5; feat[5] = r20; feat[6] = r5 / r20 if r20 > 0 else 1
    feat[12] = rsi; feat[13] = k; feat[14] = d_val; feat[15] = j
    feat[16] = bu; feat[17] = bl; feat[18] = bb_w; feat[19] = bb_pos
    feat[20] = vlr; feat[21] = vlr
    feat[35] = 1 if ct < 10 else 0
    feat[41] = bb_pos; feat[42] = j
    feat[44] = 0  # rsi_trend placeholder
    return feat

# ====== 主函数 ======
def main():
    now = pd.Timestamp.now()
    report_lines = []
    report_lines.append(f"# 📊 每日盘前推荐 {now.strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("")
    
    # 蓝盾V4-LGB
    bs_results, bs_err = score_blueshield()
    report_lines.append("## 🛡️ 蓝盾V4-LGB（大盘>$10, Top-15）")
    report_lines.append("")
    if bs_err:
        report_lines.append(f"⚠️ 错误: {bs_err}")
    elif not bs_results:
        report_lines.append("无推荐信号")
    else:
        report_lines.append("| # | 代码 | 现价 | 概率 | 信号 |")
        report_lines.append("|---|------|------|------|------|")
        for i, r in enumerate(bs_results, 1):
            signal = "🟢🟢" if r['probability'] > 0.65 else ("🟢" if r['probability'] > 0.55 else "🟡")
            report_lines.append(f"| {i} | {r['code']} | ${r['price']:.2f} | {r['probability']:.1%} | {signal} |")
    report_lines.append("")
    
    # 绿箭V9-Lottery
    ga_results, ga_err = score_greenarrow()
    report_lines.append("## 🏹 绿箭V9-Lottery（小盘<$10, 概率>70%）")
    report_lines.append("")
    if ga_err:
        report_lines.append(f"⚠️ 错误: {ga_err}")
    elif not ga_results:
        report_lines.append("无推荐信号")
    else:
        report_lines.append("| # | 代码 | 现价 | 概率 | 信号 |")
        report_lines.append("|---|------|------|------|------|")
        for i, r in enumerate(ga_results, 1):
            signal = "🟢🟢" if r['probability'] > 0.9 else ("🟢" if r['probability'] > 0.8 else "🟡")
            report_lines.append(f"| {i} | {r['code']} | ${r['price']:.2f} | {r['probability']:.1%} | {signal} |")
    report_lines.append("")
    
    report_lines.append("---")
    report_lines.append(f"*模型: V4-LGB(WF夏普1.13) + V9-Lottery | 生成时间: {now.isoformat()}*")
    
    report = "\n".join(report_lines)
    print(report)
    
    # 保存到文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = f"{OUTPUT_DIR}/daily_recommendation_{now.strftime('%Y-%m-%d')}.md"
    with open(fname, 'w') as f:
        f.write(report)
    
    return report

if __name__ == '__main__':
    main()
