#!/usr/bin/env python3
"""A2 简化评分 - 不更新数据，直接用现有数据评分"""
import json, os, sys, time
import numpy as np
import pandas as pd
import xgboost as xgb
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import DATA_DIR as D_DATA, CN_MODELS, CN_DATA, CN_OUTPUT
CHECKPOINT = os.path.join(D_DATA, 'layer3_checkpoints', 'model_batch_5.json')
BUY_THRESHOLD = 4.0

# 加载模型
print("加载模型...")
booster = xgb.Booster()
booster.load_model(CHECKPOINT)
FEAT_COLS = booster.feature_names
print(f"  模型特征: {len(FEAT_COLS)}个")

# 加载数据
print("加载数据...")
t0 = time.time()
df = pd.read_parquet(os.path.join(D_DATA, 'a_hist_10y.parquet'))
print(f"  K线: {len(df)}行, {df['Code'].nunique()}只 ({time.time()-t0:.1f}s)")

# 构建hist格式
hist = {}
for code, group in df.groupby('Code'):
    group = group.sort_values('Date')
    if len(group) < 120:
        continue
    hist[str(code)] = {
        'c': group['C'].values,
        'h': group['H'].values,
        'l': group['L'].values,
        'o': group['O'].values,
        'v': group['V'].values,
    }
print(f"  有效股票: {len(hist)}只 (≥120根K线)")

# 计算特征并评分
print("计算特征...")
results = []
for code, data in hist.items():
    c = data['c']
    n = len(c)
    if n < 120:
        continue
    
    price = c[-1]
    
    # MA特征
    ma5 = np.mean(c[-5:])
    ma10 = np.mean(c[-10:])
    ma20 = np.mean(c[-20:])
    ma60 = np.mean(c[-60:])
    ma120 = np.mean(c[-120:])
    
    pct_ma5 = (price/ma5 - 1) * 100 if ma5 > 0 else 0
    pct_ma10 = (price/ma10 - 1) * 100 if ma10 > 0 else 0
    pct_ma20 = (price/ma20 - 1) * 100 if ma20 > 0 else 0
    pct_ma60 = (price/ma60 - 1) * 100 if ma60 > 0 else 0
    pct_ma120 = (price/ma120 - 1) * 100 if ma120 > 0 else 0
    
    # 波动率
    rets = np.abs(np.diff(c[-60:])) / c[-60:-1]
    vol_10d = np.mean(np.abs(np.diff(c[-10:])) / c[-10:-1]) * 100
    vol_60d = np.mean(rets) * 100
    vol_ratio = vol_10d / vol_60d if vol_60d > 0 else 1
    
    # 收益率
    ret_5d = (price/c[-5] - 1) * 100 if n > 5 else 0
    ret_10d = (price/c[-10] - 1) * 100 if n > 10 else 0
    ret_20d = (price/c[-20] - 1) * 100 if n > 20 else 0
    
    # RSI
    changes = np.diff(c[-14:])
    gains = np.sum(changes[changes > 0])
    losses = np.abs(np.sum(changes[changes < 0]))
    rsi14 = 100 - 100/(1 + gains/losses) if losses > 0 else 100
    
    # 构建特征向量
    feat = {
        'pct_ma5': pct_ma5, 'pct_ma10': pct_ma10, 'pct_ma20': pct_ma20,
        'pct_ma60': pct_ma60, 'pct_ma120': pct_ma120,
        'ma20_slope': 0, 'ma60_slope': 0,
        'vol_10d': vol_10d, 'vol_60d': vol_60d, 'vol_ratio': vol_ratio,
        'atr20_pct': 0, 'ret_1d': (c[-1]/c[-2]-1)*100 if n > 1 else 0,
        'ret_5d': ret_5d, 'ret_10d': ret_10d, 'ret_20d': ret_20d,
        'ret_60d': (price/c[-60]-1)*100 if n > 60 else 0,
        'rsi14': rsi14, 'vol_ratio_5_20': 1,
        'kdj_k': 50, 'kdj_d': 50, 'kdj_j': 50,
        'macd_dif': 0, 'macd_dea': 0, 'macd_bar': 0,
        'bb_width': 0, 'bb_position': 0.5,
        'obv_ratio_5_20': 1, 'ret5_max': 0, 'ret3_vs_ema12': 0,
        'accel_5_10': 0, 'ma5_ma10_cross': 0, 'vol_breakout': 0,
    }
    
    # 只用模型需要的特征
    feat_vec = [feat.get(f, 0) for f in FEAT_COLS]
    
    # 预测
    dmatrix = xgb.DMatrix([feat_vec], feature_names=FEAT_COLS)
    score = booster.predict(dmatrix)[0]
    
    results.append({
        'code': code,
        'score': round(score, 2),
        'price': round(price, 2),
        'rsi': round(rsi14, 1),
    })

# 排序输出
results.sort(key=lambda x: -x['score'])
print(f"\n{'='*60}")
print(f"  A2 评分结果 ({len(results)}只)")
print(f"{'='*60}")

# Top推荐
buy_signals = [r for r in results if r['score'] > BUY_THRESHOLD]
print(f"\n🟢 买入信号 (>{BUY_THRESHOLD}%): {len(buy_signals)}只")
print(f"{'排名':>4} {'代码':>8} {'评分':>8} {'价格':>10} {'RSI':>6}")
print("-" * 40)
for i, r in enumerate(buy_signals[:10], 1):
    print(f"{i:4d} {r['code']:>8} {r['score']:>7.2f}% {r['price']:>10.2f} {r['rsi']:>6.1f}")

# 大盘情绪
avg_score = np.mean([r['score'] for r in results])
med_score = np.median([r['score'] for r in results])
print(f"\n📊 大盘情绪:")
print(f"  平均评分: {avg_score:+.2f}%")
print(f"  中位数: {med_score:+.2f}%")
print(f"  >4%买入信号: {len(buy_signals)}只 ({len(buy_signals)/len(results)*100:.1f}%)")

# 保存结果
output = {
    'date': '2026-06-17',
    'model': 'A2_L3',
    'total_scored': len(results),
    'buy_signals': buy_signals[:10],
    'avg_score': round(avg_score, 2),
    'med_score': round(med_score, 2),
}
outpath = os.path.join(CN_OUTPUT, 'a2_scored_20260617.json')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
with open(outpath, 'w') as f:
    json.dump(output, f, default=str, ensure_ascii=False, indent=2)
print(f"\n✅ 结果已保存: {outpath}")
