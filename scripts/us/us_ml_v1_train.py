#!/usr/bin/env python3
"""
美股 XGBoost ML 模型 — 今晚快速版
===================================
1. 从 us_hist_clean.parquet 构建特征（技术指标）
2. 用 minishare 拉今日实时价补到最后一天
3. 训练多分类模型（涨>5% / 涨0~5% / 跌0~5% / 跌>5%）
4. 输出分档概率

用法: python3 scripts/us_ml_v1_train.py
输出: data/models/us_xgb_v1.json + 预测结果
"""

import sys, json, os, time, math, warnings
from datetime import datetime, timezone, timedelta
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

TZ = timezone(timedelta(hours=8))
now = datetime.now(TZ)
WORKSPACE = r"/home/hermes/.hermes/openclaw-archive"
BASE = os.path.join(WORKSPACE, "data")
MODEL_DIR = os.path.join(WORKSPACE, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

T0 = time.time()

# ─── 参数 ───
MIN_DAYS = 500       # 最少需要500天数据
N_PAST = 60          # 特征用过去60根K线
MIN_SYMBOLS = 100    # 最少训练100只

# ─── 1. 加载数据 ───
print("[1/4] 加载美股数据...")
t = time.time()
with open(f"{BASE}/us_hist_clean.parquet", 'r') as f:
    all_data = json.load(f)
syms = list(all_data.keys())
print(f"  总池: {len(syms)}只 ({time.time()-t:.0f}s)")

# ─── 2. 特征工程 ───
print("[2/4] 构建特征矩阵...")
t = time.time()

def compute_indicators(arr):
    """计算技术指标，返回列表（长度同arr，前段为nan）"""
    n = len(arr)
    if n < 30:
        return {}, {}
    
    # 均线
    def sma(p):
        res = [None] * (p - 1)
        for i in range(p - 1, n):
            res.append(sum(arr[i-p+1:i+1]) / p)
        return res
    
    def ema(p):
        k = 2 / (p + 1)
        res = [arr[0]]
        for v in arr[1:]:
            res.append(v * k + res[-1] * (1 - k))
        return res
    
    ma5 = sma(5)
    ma10 = sma(10)
    ma20 = sma(20)
    ma60 = sma(60)
    
    # RSI-14
    rsi14 = [None] * n
    if n > 15:
        gains = [max(arr[i] - arr[i-1], 0) for i in range(1, n)]
        losses = [max(arr[i-1] - arr[i], 0) for i in range(1, n)]
        for i in range(14, n):
            if i == 14:
                avg_g = sum(gains[:14]) / 14
                avg_l = sum(losses[:14]) / 14
            else:
                avg_g = (avg_g * 13 + gains[i-1]) / 14
                avg_l = (avg_l * 13 + losses[i-1]) / 14
            rsi14[i] = 100 - 100 / (1 + avg_g / avg_l) if avg_l > 0 else 100
    
    # 波动率
    returns = [arr[i] / arr[i-1] - 1 for i in range(1, n)]
    vol20 = [None] * 19
    for i in range(19, n):
        vol20.append(np.std(returns[i-19:i+1]) * math.sqrt(252))
    
    # 52周高低位
    p52 = [None] * 251
    for i in range(251, n):
        lo = min(arr[i-251:i+1])
        hi = max(arr[i-251:i+1])
        p52.append((arr[i] - lo) / (hi - lo) * 100 if hi > lo else 50)
    
    # 日均换手(用volume替代, 因为没有流通股数据)
    # 跳过换手率，用price indicators
    
    # 最近N天的return
    ret1 = [None] + [(arr[i] / arr[i-1] - 1) * 100 for i in range(1, n)]
    ret5 = [None] * 4 + [(arr[i] / arr[i-5] - 1) * 100 for i in range(5, n)]
    ret20 = [None] * 19 + [(arr[i] / arr[i-20] - 1) * 100 for i in range(20, n)]
    ret60 = [None] * 59 + [(arr[i] / arr[i-60] - 1) * 100 for i in range(60, n)]
    
    # 价格相对于MA的偏离度
    ma_bias = [None] * 19
    for i in range(19, n):
        if ma20[i] and ma20[i] > 0:
            ma_bias.append((arr[i] / ma20[i] - 1) * 100)
        else:
            ma_bias.append(None)
    
    # MACD
    e12 = ema(12)
    e26 = ema(26)
    macd = [e12[i] - e26[i] for i in range(n)]
    macd_sig = sma(9)  # 用MACD做9日SMA
    # 实际macd_signal要重新算
    def sma_custom(data, p):
        res = [None] * (p - 1)
        for i in range(p - 1, len(data)):
            vals = [v for v in data[i-p+1:i+1] if v is not None]
            if len(vals) == p:
                res.append(sum(vals) / p)
            else:
                res.append(None)
        return res
    
    macd_sig = sma_custom(macd, 9)
    
    features = {
        'close': arr,
        'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma60': ma60,
        'rsi14': rsi14, 'vol20': vol20, 'p52': p52,
        'ret1': ret1, 'ret5': ret5, 'ret20': ret20, 'ret60': ret60,
        'ma_bias20': ma_bias,
        'macd': macd, 'macd_signal': macd_sig,
    }
    
    # label: 明日涨幅（原始值）
    label = [None] * n
    for i in range(n - 1):
        label[i] = (arr[i+1] / arr[i] - 1) * 100
    label[-1] = None  # 最后一天无label
    
    return features, label


# 收集所有数据
rows = []
skip_short = 0
skip_error = 0

for idx, sym in enumerate(syms):
    if idx % 500 == 0 and idx > 0:
        el = time.time() - t
        print(f"  {idx}/{len(syms)} 有效{len(rows)}行 ({el:.0f}s)", flush=True)
    
    c = all_data[sym].get('c', [])
    v = all_data[sym].get('v', [])
    
    if len(c) < MIN_DAYS or len(v) < MIN_DAYS:
        skip_short += 1
        continue
    
    try:
        feat, label = compute_indicators(c)
        if feat is None:
            skip_error += 1
            continue
        
        n = len(c)
        # 只取最后两年(L=500天)作为有效范围
        start_i = n - N_PAST  # 只看最近N_PAST天作为样本生成
        
        # 采样：每隔2天取一个（减少重叠行）
        for i in range(start_i, n - 1, 2):
            if label[i] is None:
                continue
            # 确认该位置所有特征非空
            features_i = {}
            skip_row = False
            for k in ['close', 'ma5', 'ma10', 'ma20', 'rsi14', 'vol20', 
                       'p52', 'ret5', 'ret20', 'ma_bias20', 'macd', 'macd_signal']:
                val = feat[k][i] if i < len(feat[k]) else None
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    skip_row = True
                    break
                features_i[k] = val
            
            if skip_row:
                continue
            
            features_i['sym'] = sym
            features_i['price'] = c[i]
            features_i['volume'] = v[i] / 1e6  # 百万股
            features_i['label_pct'] = round(label[i], 4)  # 明日涨跌幅%
            rows.append(features_i)
        
    except Exception as e:
        skip_error += 1
        if idx < 5:
            print(f"  ERROR {sym}: {e}")

print(f"  skip_short={skip_short}, skip_error={skip_error}")
print(f"  总行数: {len(rows)} ({time.time()-t:.0f}s)")

if len(rows) < MIN_SYMBOLS:
    print(f"❌ 数据不足: {len(rows)}行 < {MIN_SYMBOLS}最少要求")
    sys.exit(1)

# ─── 3. 生成多分类Label ───
print("\n[3/5] 生成分档Label + 训练...")
df = pd.DataFrame(rows)

# 分档：涨>5%, 涨0~5%, 跌0~5%, 跌>5%
def bucket(pct):
    if pct > 5:
        return 3   # 大涨
    elif pct > 0:
        return 2   # 小涨
    elif pct > -5:
        return 1   # 小跌
    else:
        return 0   # 大跌

df['label_bucket'] = df['label_pct'].apply(bucket)

print(f"  样本分布:")
for i, name in [(3, '大涨>5%'), (2, '小涨0~5%'), (1, '小跌-5~0%'), (0, '大跌<-5%')]:
    n = (df['label_bucket'] == i).sum()
    print(f"    {name}: {n} ({n/len(df)*100:.1f}%)")

# 特征列（不含sym/price/label）
feature_cols = ['ma5', 'ma10', 'ma20', 'rsi14', 'vol20', 'p52', 
                'ret5', 'ret20', 'ma_bias20', 'macd', 'macd_signal',
                'price', 'volume']
print(f"  特征: {feature_cols}")

# 保存索引用于最后输出sym映射
all_syms = df['sym'].values
all_prices = df['price'].values
all_labels = df['label_pct'].values

# ─── 4. 训练模型 ───
print("[4/5] 训练XGBoost多分类模型...")
t = time.time()

X = df[feature_cols].values
y = df['label_bucket'].values

# 按时间分割: 取最后20%做测试
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
test_syms = all_syms[split_idx:]
test_prices = all_prices[split_idx:]
test_labels = all_labels[split_idx:]

print(f"  训练: {len(X_train)}, 测试: {len(X_test)}")

# 计算类别权重（平衡样本）
from sklearn.utils.class_weight import compute_class_weight
classes = np.array([0, 1, 2, 3])
weights = compute_class_weight('balanced', classes=classes, y=y_train)
weight_dict = {i: w for i, w in enumerate(weights)}
sample_weight = np.array([weight_dict[yi] for yi in y_train])
print(f"  类别权重: {dict(zip(['大跌','小跌','小涨','大涨'], [f'{w:.2f}' for w in weights]))}")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    num_class=4, device='cuda')

model.fit(X_train, y_train, 
          sample_weight=sample_weight,
          eval_set=[(X_test, y_test)],
          verbose=50)

train_time = time.time() - t
print(f"  训练耗时: {train_time:.0f}s")

# ─── 5. 评估 + 预测 ───
print("\n[5/5] 评估 & 预测明日...")
t = time.time()

# 测试集评估
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n  测试集准确率: {acc:.3f}")
print(classification_report(y_test, y_pred, 
      target_names=['大跌<-5%', '小跌-5~0%', '小涨0~5%', '大涨>5%']))

# 测试集夏普比模拟
# 计算每个股票的实际收益率和预测类别
test_df = pd.DataFrame({
    'label_actual': test_labels,
    'pred_bucket': y_pred,
    'prob_bucket': list(y_proba),
    'sym': test_syms,
    'price': test_prices,
})

# 按预测的"大涨>5%"概率排序，选Top 20%做策略
proba_win = y_proba[:, 3]  # 大涨概率
test_df['prob_large_up'] = proba_win
test_df['pred_win'] = test_df['label_actual'] > 0

# 按大涨概率排序模拟回测
ranked = test_df.sort_values('prob_large_up', ascending=False)
top20 = ranked.head(int(len(ranked) * 0.1))

win_rate = (top20['label_actual'] > 0).mean()
avg_ret = top20['label_actual'].mean()
sharpe = top20['label_actual'].mean() / top20['label_actual'].std() * math.sqrt(252) if top20['label_actual'].std() > 0 else 0

print(f"\n  Top10%策略模拟:")
print(f"    样本数: {len(top20)}")
print(f"    胜率(涨): {win_rate:.1%}")
print(f"    平均收益: {avg_ret:.2f}%")
print(f"    夏普(年化): {sharpe:.3f}")

# ─── 今日预测：使用最新数据 ───
print(f"\n─── 今日预测「2026-06-09」───")
# 提取每只股票最新的特征行（对应数据最后一天）
latest_rows = df.drop_duplicates(subset='sym', keep='last').copy()
# 过滤掉缺失特征的行
latest_rows = latest_rows.dropna(subset=feature_cols)
print(f"  可评分股票: {len(latest_rows)}只")

X_latest = latest_rows[feature_cols].values
y_latest_proba = model.predict_proba(X_latest)

# 组装结果
results = []
for i, (_, row) in enumerate(latest_rows.iterrows()):
    probs = y_latest_proba[i]
    results.append({
        'sym': row['sym'],
        'price': float(row['price']),
        'prob_large_up': float(probs[3]),    # 涨>5%
        'prob_small_up': float(probs[2]),     # 涨0~5%
        'prob_small_down': float(probs[1]),   # 跌0~5%
        'prob_large_down': float(probs[0]),   # 跌>5%
        'prob_up': float(probs[2] + probs[3]), # 总上涨概率
        'expected_ret': float(sum(probs * np.array([-7, -2.5, 2.5, 7]))),  # 期望收益
    })

results.sort(key=lambda x: -x['expected_ret'])

# 输出Top30
print(f"\n{'═'*80}")
print(f"{'排名':>3} {'代码':>8} {'价格':>8} {'涨>5%':>7} {'涨0~5%':>7} {'跌0~5%':>7} {'跌>5%':>7} {'总涨':>6} {'期望%':>6}")
print(f"{'─'*80}")
for i, r in enumerate(results[:30]):
    print(f"{i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {r['prob_large_up']*100:>6.1f}% {r['prob_small_up']*100:>6.1f}% "
          f"{r['prob_small_down']*100:>6.1f}% {r['prob_large_down']*100:>6.1f}% {r['prob_up']*100:>5.1f}% {r['expected_ret']*100:>5.2f}%")

# 保存
output = {
    'timestamp': now.isoformat(),
    'model_type': 'xgb_multiclass_4bucket',
    'feature_cols': feature_cols,
    'train_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'train_time_sec': round(train_time, 1),
    'test_accuracy': round(acc, 4),
    'test_sharpe_top10pct': round(sharpe, 4),
    'test_win_rate_top10pct': round(win_rate, 4),
    'test_avg_ret_top10pct': round(avg_ret, 4),
    'predictions': [{
        'rank': i+1, **r
    } for i, r in enumerate(results[:50])],
    'all_scores': results,  # 全部股票
}

model.save_model(os.path.join(MODEL_DIR, "us_xgb_v1.json"))
with open(os.path.join(MODEL_DIR, "us_xgb_v1_prediction.json"), 'w') as f:
    json.dump(output, f, indent=2)

TOTAL = time.time() - T0
print(f"\n{'═'*80}")
print(f"✅ 美股ML已完成！模型保存到: data/models/us_xgb_v1.json")
print(f"   预测结果保存到: data/models/us_xgb_v1_prediction.json")
print(f"   总耗时: {TOTAL:.0f}s ({TOTAL/60:.1f}min)")
