"""
a_ml_train_feats.py — A股ML特征训练（基于K线+质量池）
直接从 a_hist_10y.json 加载K线，计算技术指标作为特征
不需要资金流数据（后面再加）
"""
import json, sys, gc, time, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

t0 = time.time()

# ─── 1. 加载K线 ───
print('加载K线数据...')
with open('/home/hermes/.hermes/openclaw-project/data/a_hist_10y.json', 'r', encoding='utf-8') as f:
    hist = json.load(f)
print(f'  {len(hist)}只股票')

# 只取主板（60/00开头）+ 至少500天
stocks = []
for code, h in hist.items():
    if not (code.startswith('60') or code.startswith('00')):
        continue
    if len(h.get('dates', [])) < 500:
        continue
    stocks.append(code)
print(f'  主板+500天: {len(stocks)}只')

# 先取前500只测试（避免时间太长）
stocks = stocks[:500]
print(f'  训练用: {len(stocks)}只')

# ─── 2. 特征计算 ───
print('\n计算技术指标特征...')

def calc_technical(h):
    """计算技术面特征"""
    closes = np.array(h.get('closes', []), dtype=np.float64)
    highs = np.array(h.get('highs', []), dtype=np.float64)
    lows = np.array(h.get('lows', []), dtype=np.float64)
    volumes = np.array(h.get('volumes', []), dtype=np.float64)
    dates = h.get('dates', [])
    
    n = len(closes)
    if n < 100:
        return []
    
    # 反转（日期从旧到新）
    closes = closes[::-1]
    highs = highs[::-1]
    lows = lows[::-1]
    volumes = volumes[::-1]
    dates = dates[::-1]
    
    features = []
    
    for i in range(100, n - 10):
        idx = i  # 当前是第i个交易日
        
        # 价格动量
        ret_1d = closes[i] / closes[i-1] - 1 if closes[i-1] > 0 else 0
        ret_5d = closes[i] / closes[i-5] - 1 if closes[i-5] > 0 else 0
        ret_20d = closes[i] / closes[i-20] - 1 if closes[i-20] > 0 else 0
        ret_60d = closes[i] / closes[i-60] - 1 if i >= 60 and closes[i-60] > 0 else 0
        
        # 均线
        ma5 = np.mean(closes[i-4:i+1])
        ma10 = np.mean(closes[i-9:i+1])
        ma20 = np.mean(closes[i-19:i+1])
        ma60 = np.mean(closes[i-59:i+1]) if i >= 59 else ma20
        
        # 偏离度
        dev_ma5 = closes[i] / ma5 - 1
        dev_ma20 = closes[i] / ma20 - 1
        dev_ma60 = closes[i] / ma60 - 1
        
        # 均线排列
        ma_align = 1 if ma5 > ma10 > ma20 else (-1 if ma5 < ma10 < ma20 else 0)
        
        # 波动率
        vol_5d = np.std(closes[i-4:i+1] / closes[i-5:i] - 1) if i >= 5 else 0
        vol_20d = np.std([closes[j]/closes[j-1]-1 for j in range(i-19, i+1)]) if i >= 19 else 0
        
        # RSI(14)
        gains, losses = 0, 0
        for j in range(i-13, i+1):
            chg = closes[j] - closes[j-1]
            if chg > 0: gains += chg
            else: losses -= chg
        avg_gain = gains / 14
        avg_loss = losses / 14
        rsi_14 = 50
        if avg_loss > 1e-6:
            rs = avg_gain / avg_loss
            rsi_14 = 100 - 100 / (1 + rs)
        
        # MACD
        ema12_prices = closes[max(0,i-25):i+1]
        ema26_prices = closes[max(0,i-49):i+1]
        
        def ema(arr, period):
            if len(arr) < period:
                return arr[-1] if len(arr) > 0 else 0
            result = np.mean(arr[:period])
            alpha = 2 / (period + 1)
            for v in arr[period:]:
                result = v * alpha + result * (1 - alpha)
            return result
        
        ema12 = ema(closes[max(0,i-25):i+1], 12)
        ema26 = ema(closes[max(0,i-49):i+1], 26)
        macd = ema12 - ema26
        macd_signal = ema(np.array([closes[max(0,j-25):j+1] for j in range(max(9,i-9), i+1)]), 9) if i >= 9 else 0
        
        # 成交量
        vol_ma5 = np.mean(volumes[i-4:i+1])
        vol_ratio = volumes[i] / vol_ma5 if vol_ma5 > 0 else 1
        
        # 价格位置：当前价在N天区间的位置
        high_20 = np.max(highs[i-19:i+1])
        low_20 = np.min(lows[i-19:i+1])
        pos_in_20 = (closes[i] - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
        
        # Y标签：未来5天>2%
        fut_5 = closes[i+5] if i + 5 < n else 0
        if closes[i] > 0 and fut_5 > 0:
            ret_future = (fut_5 - closes[i]) / closes[i]
            y = 1.0 if ret_future > 0.02 else 0.0
            
            feat = [
                ret_1d, ret_5d, ret_20d, ret_60d,
                ma5/ma20, ma20/ma60,
                dev_ma5, dev_ma20, dev_ma60,
                ma_align,
                vol_5d, vol_20d,
                rsi_14,
                macd, macd_signal, macd - macd_signal,
                vol_ratio,
                pos_in_20,
                closes[i] / ma60,  # 相对位置
            ]
            features.append((feat, y, dates[i]))
    
    return features

# 逐只计算
all_feats = []
stock_count = 0

for code in stocks:
    h = hist[code]
    feats = calc_technical(h)
    if feats:
        all_feats.extend(feats)
        stock_count += 1
    if stock_count % 100 == 0:
        print(f'  {stock_count}只, 累计{len(all_feats)}样本', flush=True)

print(f'\n特征计算完成: {stock_count}只, {len(all_feats)}样本')

# ─── 3. 训练 ───
print('\n训练XGBoost...')

X = np.array([f[0] for f in all_feats], dtype=np.float32)
y = np.array([f[1] for f in all_feats], dtype=np.float32)

print(f'  X shape: {X.shape}, y shape: {y.shape}')
print(f'  正例率: {y.mean():.2%}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
print(f'\n✅ 评估结果:')
print(f'  Accuracy: {acc:.4f}')
print(f'  AUC: {auc:.4f}')
print(f'  正例概率均值: {y_prob.mean():.4f}')

# 按概率分桶看校准性
bins = np.linspace(0, 1, 11)
for i in range(10):
    mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
    if mask.sum() > 0:
        actual = y_test[mask].mean()
        print(f'  prob [{bins[i]:.1f}-{bins[i+1]:.1f}] actual={actual:.3f} n={mask.sum()}')

# ─── 4. Platt校准 ───
print('\nPlatt校准...')
calibrator = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
calibrator.fit(X_test, y_test)
cal_prob = calibrator.predict_proba(X_test)[:, 1]
print(f'  校准后概率均值: {cal_prob.mean():.4f}')

# ─── 5. 保存 ───
MODEL_DIR = '/home/hermes/.hermes/openclaw-project/data/models'
os.makedirs(MODEL_DIR, exist_ok=True)
model.save_model(os.path.join(MODEL_DIR, 'a_xgb_tech_v1.json'))
print(f'\n✅ 模型: {os.path.join(MODEL_DIR, "a_xgb_tech_v1.json")}')

# 特征名
fnames = ['ret_1d','ret_5d','ret_20d','ret_60d',
          'ma5/ma20','ma20/ma60',
          'dev_ma5','dev_ma20','dev_ma60','ma_align',
          'vol_5d','vol_20d','rsi_14',
          'macd','macd_signal','macd_hist',
          'vol_ratio','pos_in_20','close/ma60']

importances = model.feature_importances_
print('\n特征重要性:')
for n, imp in sorted(zip(fnames, importances), key=lambda x: -x[1]):
    print(f'  {n}: {imp:.4f}')

t1 = time.time()
print(f'\n总耗时: {(t1-t0)/60:.1f}分钟')
