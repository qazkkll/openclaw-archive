#!/usr/bin/env python3
"""重新训练模型一次，保存完整，然后出推荐"""
import sys, warnings; warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd, numpy as np, xgboost as xgb, json, os, time
import yfinance as yf
from sklearn.isotonic import IsotonicRegression

INPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v5.parquet'
MODEL_FILE = '/home/hermes/.hermes/openclaw-project/data/models/us/greenshaft_v5_5pct.json'
PRED_FILE = '/home/hermes/.hermes/openclaw-project/data/models/us/greenshaft_v5_prediction.json'

print("训练+校准...")
df = pd.read_parquet(INPUT).dropna(subset=['fwd_5d_ret'])
exclude = {'ticker','date','label','fwd_5d_ret'}
feat_cols = [c for c in df.columns if c not in exclude]

y = (df['fwd_5d_ret'] > 0.05).astype(int).values
X = df[feat_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
n = len(X)
te, ve = int(n*0.7), int(n*0.85)

spw = (te - y[:te].sum()) / y[:te].sum()
params = {'objective':'binary:logistic','eval_metric':'auc','tree_method':'hist',
          'device':'cuda','max_depth':6,'learning_rate':0.05,'subsample':0.8,
          'colsample_bytree':0.8,'scale_pos_weight':spw,'random_state':42}

dtrain = xgb.DMatrix(X[:te], y[:te])
dval = xgb.DMatrix(X[te:ve], y[te:ve])
model = xgb.train(params, dtrain, num_boost_round=500,
                  evals=[(dtrain,'train'),(dval,'val')],
                  early_stopping_rounds=20, verbose_eval=False)

# 校准器
val_pred = model.predict(dval)
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(val_pred, y[te:ve])

model.save_model(MODEL_FILE)
print(f"模型保存: {MODEL_FILE}")

# 全量推荐
latest = df.groupby('ticker').last().reset_index()
X_latest = latest[feat_cols].values.astype(np.float32)
X_latest = np.nan_to_num(X_latest, nan=0.0, posinf=0.0, neginf=0.0)
dlatest = xgb.DMatrix(X_latest)
preds = model.predict(dlatest)
probs = ir.transform(preds)

df_out = latest[['ticker','date']].copy()
df_out['prob'] = np.round(probs, 4)
df_out = df_out.sort_values('prob', ascending=False)

# 市场热度
top50_prob = df_out.head(50)['prob'].mean()
all_avg = df_out['prob'].mean()
n35 = (df_out['prob'] > 0.35).sum()
n40 = (df_out['prob'] > 0.40).sum()

print(f"\n{'='*60}")
print(f"📊 绿箭v5 每日推荐")
print(f"{'='*60}")
print(f"📡 市场热度: Top50平均{top50_prob:.1%}, >35%: {n35}只, >40%: {n40}只")
if top50_prob > 0.40:
    print(f"   状态: 🔥 热 (可积极)")
elif top50_prob > 0.33:
    print(f"   状态: 🌤️ 温和 (精选)")
elif top50_prob > 0.25:
    print(f"   状态: ☁️ 偏低 (谨慎)")
else:
    print(f"   状态: ❄️ 冷 (观望)")
print()

# 精选推荐
print(f"★ 精选推荐 (概率>35%):")
if n35 > 0:
    for i, (_, row) in enumerate(df_out[df_out['prob'] > 0.35].head(15).iterrows()):
        print(f"  {i+1}. {row['ticker']} — {row['prob']:.1%}")
else:
    print(f"  (当前无票达到阈值, 建议观望)")
print()

# Top5 (不管阈值)
print(f"★ Top 5 (全量):")
for i, (_, row) in enumerate(df_out.head(5).iterrows()):
    print(f"  {i+1}. {row['ticker']} — {row['prob']:.1%}")

# 保存
top100 = df_out.head(100).to_dict('records')
pred_data = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'market_heat_top50': round(float(top50_prob), 4),
    'n_above_35': int(n35),
    'n_above_40': int(n40),
    'top_100': [{k:(str(v) if isinstance(v,(pd.Timestamp,np.integer)) else float(v) if isinstance(v,np.floating) else v) for k,v in item.items() if k in ['ticker','date','prob']} for item in top100],
}
json.dump(pred_data, open(PRED_FILE, 'w'), indent=2)
print(f"\n推荐保存: {PRED_FILE}")
