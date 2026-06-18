#!/usr/bin/env python3
"""快速训练改进模型"""

import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# 加载数据
with open('data/training_data.json') as f:
    all_data = json.load(f)

print("📊 构建特征...")

X_all = []
y_all = []

for ticker, data in all_data.items():
    c = np.array(data['close'], dtype=float)
    if len(c) < 100:
        continue
    
    # 简化特征
    for i in range(60, len(c) - 5):
        # 特征
        ma5 = np.mean(c[i-5:i])
        ma20 = np.mean(c[i-20:i])
        ma60 = np.mean(c[i-60:i])
        rsi = 100 - 100 / (1 + np.mean(np.diff(c[i-14:i])[np.diff(c[i-14:i]) > 0]) / max(abs(np.mean(np.diff(c[i-14:i])[np.diff(c[i-14:i]) < 0])), 0.001))
        momentum = (c[i] / c[i-5] - 1) * 100
        volatility = np.std(np.diff(c[i-20:i]) / c[i-20:i-1]) * np.sqrt(252)
        
        features = [
            c[i] / ma5,  # ma5_ratio
            c[i] / ma20,  # ma20_ratio
            c[i] / ma60,  # ma60_ratio
            1 if ma5 > ma20 else 0,  # ma5_ma20_cross
            rsi,
            momentum,
            volatility,
        ]
        
        # 标签：5天后涨5%+
        label = 1 if c[i+5] / c[i] > 1.05 else 0
        
        X_all.append(features)
        y_all.append(label)

X = np.array(X_all)
y = np.array(y_all)

print(f"  样本数: {len(X)}")
print(f"  正样本: {y.mean():.2%}")

# 训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# 评估
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ 准确率: {acc:.2%}")

# 对比原模型
print("\n📊 特征重要性:")
names = ['ma5_ratio', 'ma20_ratio', 'ma60_ratio', 'ma_cross', 'rsi', 'momentum', 'volatility']
for name, imp in zip(names, model.feature_importances_):
    print(f"  {name}: {imp:.3f}")

# 保存
model.save_model('models/us/v10_improved.json')
print("\n✅ 模型已保存")
