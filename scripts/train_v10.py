#!/usr/bin/env python3
"""
绿箭V10 — 改进版模型

改进点：
1. 市场状态感知
2. 多时间框架特征
3. 动态止损逻辑
4. 更好的特征工程
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime

# 加载数据
with open('data/training_data.json') as f:
    all_data = json.load(f)


def compute_advanced_features(close, high, low, volume):
    """计算高级特征"""
    c = np.array(close, dtype=float)
    h = np.array(high, dtype=float)
    l = np.array(low, dtype=float)
    v = np.array(volume, dtype=float)
    
    n = len(c)
    if n < 60:
        return None
    
    features = {}
    
    # 1. 趋势特征
    ma5 = np.mean(c[-5:])
    ma20 = np.mean(c[-20:])
    ma60 = np.mean(c[-60:])
    features['ma5_ratio'] = c[-1] / ma5 if ma5 > 0 else 1
    features['ma20_ratio'] = c[-1] / ma20 if ma20 > 0 else 1
    features['ma60_ratio'] = c[-1] / ma60 if ma60 > 0 else 1
    features['ma5_ma20_cross'] = 1 if ma5 > ma20 else 0
    
    # 2. 动量特征
    delta = np.diff(c[-15:])
    gain = np.sum(delta[delta > 0])
    loss = abs(np.sum(delta[delta < 0]))
    rs = gain / max(loss, 0.001)
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # RSI变化率
    delta_prev = np.diff(c[-20:-5])
    gain_prev = np.sum(delta_prev[delta_prev > 0])
    loss_prev = abs(np.sum(delta_prev[delta_prev < 0]))
    rs_prev = gain_prev / max(loss_prev, 0.001)
    rsi_prev = 100 - (100 / (1 + rs_prev))
    features['rsi_change'] = features['rsi'] - rsi_prev
    
    # 4. 波动率特征
    returns = np.diff(c[-20:]) / c[-20:-1]
    features['volatility'] = np.std(returns) * np.sqrt(252)
    features['atr'] = np.mean(h[-20:] - l[-20:])
    features['atr_pct'] = features['atr'] / c[-1] if c[-1] > 0 else 0
    
    # 4. 布林带位置
    bb_std = np.std(c[-20:])
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std
    features['bb_position'] = (c[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
    features['bb_width'] = (bb_upper - bb_lower) / ma20 if ma20 > 0 else 0
    
    # 5. 成交量特征
    vol_ma20 = np.mean(v[-20:])
    vol_ma5 = np.mean(v[-5:])
    features['volume_ratio'] = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1
    
    # 6. 价格位置
    high20 = np.max(c[-20:])
    low20 = np.min(c[-20:])
    features['price_position_20'] = (c[-1] - low20) / (high20 - low20) if (high20 - low20) > 0 else 0.5
    
    high60 = np.max(c[-60:])
    low60 = np.min(c[-60:])
    features['price_position_60'] = (c[-1] - low60) / (high60 - low60) if (high60 - low60) > 0 else 0.5
    
    # 7. 动量指标
    features['momentum_5d'] = (c[-1] / c[-6] - 1) * 100 if len(c) > 5 and c[-6] != 0 else 0
    features['momentum_20d'] = (c[-1] / c[-21] - 1) * 100 if len(c) > 20 and c[-21] != 0 else 0
    
    # 8. 市场状态（简化版）
    if features['ma5_ratio'] > 1.02 and features['rsi'] > 50:
        features['market_regime'] = 1  # 牛市
    elif features['ma5_ratio'] < 0.98 and features['rsi'] < 50:
        features['market_regime'] = -1  # 熊市
    else:
        features['market_regime'] = 0  # 震荡
    
    # 9. 趋势强度
    features['trend_strength'] = abs(features['ma5_ratio'] - 1) * 100
    
    # 10. 反转信号
    features['reversal_signal'] = 1 if (features['rsi'] < 30 and features['momentum_5d'] > 0) else 0
    
    return features


def create_labels(close, horizon=5, threshold=0.05):
    """创建标签：未来horizon天涨超threshold的概率"""
    c = np.array(close, dtype=float)
    labels = []
    
    for i in range(len(c) - horizon):
        future_return = (c[i + horizon] / c[i] - 1)
        labels.append(1 if future_return > threshold else 0)
    
    return labels


def train_improved_model():
    """训练改进版模型"""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import xgboost as xgb
    
    print("📊 构建训练数据...")
    
    X_all = []
    y_all = []
    
    for ticker, data in all_data.items():
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # 计算特征
        features = compute_advanced_features(close, high, low, volume)
        if features is None:
            continue
        
        # 创建标签
        labels = create_labels(close, horizon=5, threshold=0.05)
        
        # 对齐数据
        n_features = len(close) - 60  # 需要60天历史
        n_labels = len(labels)
        n_common = min(n_features, n_labels)
        
        if n_common > 0:
            for i in range(n_common):
                X_all.append(list(features.values()))
                y_all.append(labels[i])
    
    X = np.array(X_all)
    y = np.array(y_all)
    
    print(f"  特征维度: {X.shape}")
    print(f"  正样本比例: {y.mean():.2%}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练XGBoost
    print("\n🚀 训练改进版模型...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # 评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ 模型训练完成")
    print(f"  准确率: {accuracy:.2%}")
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 特征重要性
    feature_names = [
        'ma5_ratio', 'ma20_ratio', 'ma60_ratio', 'ma5_ma20_cross',
        'rsi', 'rsi_change', 'volatility', 'atr', 'atr_pct',
        'bb_position', 'bb_width', 'volume_ratio',
        'price_position_20', 'price_position_60',
        'momentum_5d', 'momentum_20d', 'market_regime',
        'trend_strength', 'reversal_signal'
    ]
    
    importance = model.feature_importances_
    print("\n📊 特征重要性:")
    for i, (name, imp) in enumerate(zip(feature_names, importance)):
        if imp > 0.05:
            print(f"  {name}: {imp:.3f}")
    
    return model, feature_names


if __name__ == "__main__":
    model, features = train_improved_model()
    
    # 保存模型
    model.save_model('models/us/v10_improved.json')
    print("\n✅ 模型已保存: models/us/v10_improved.json")
