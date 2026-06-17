"""今日预测：用XGBoost模型预测明天涨跌"""
import sys, json, os
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np
import xgboost as xgb

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUT = "/home/hermes/.hermes/openclaw-archive_ml"
MODEL_DIR = os.path.join(WORKSPACE, "data", "models")

print("加载最新数据...")
df = pd.read_parquet(os.path.join(DATA_OUT, "ml_training_data.parquet"))
with open(os.path.join(DATA_OUT, "ml_feature_cols.json")) as f:
    feature_cols = json.load(f)

df["date_parsed"] = pd.to_datetime(df["trade_date"])
print(f"数据范围: {df['date_parsed'].min()} ~ {df['date_parsed'].max()}")

# 最后一天的股票
last_date = df["date_parsed"].max()
print(f"最后交易日: {last_date.date()}")
today_df = df[df["date_parsed"] == last_date].dropna(subset=feature_cols).copy()
print(f"今日可评分股票数: {len(today_df)}")

# 加载模型
model = xgb.XGBClassifier(, device='cuda')
model.load_model(os.path.join(MODEL_DIR, "xgb_v1_high.json"))

# 评分
today_df["xgb_prob"] = model.predict_proba(today_df[feature_cols])[:, 1]

# 结果（高收益概率Top20）
top = today_df.sort_values("xgb_prob", ascending=False).head(20)
print(f"\n═══════ 明日预测（高收益>2%概率 Top20）═══════")
print(f"{'排名':>3s} {'股票':8s} {'涨>2%概率':>10s} {'预期涨幅':>8s} {'收盘价':>8s} {'资金流':>10s}")
print(f"{'─'*55}")
for i, (_, row) in enumerate(top.iterrows()):
    prob = row["xgb_prob"]
    # 简单映射：概率转预期涨幅
    exp_chg = prob * 5.0 - 0.5
    mf = row.get("net_mf_amount", 0) or 0
    code = row["code"]
    close = row["close"]
    print(f"{i+1:>3d} {code:8s} {prob:>9.1%} {exp_chg:>+7.2%} {close:>8.2f} {mf/1e4:>+9.0f}万")

# Tab Top 3
print(f"\n═══════ Tab推荐（高概率 Top3）═══════")
for i, (_, row) in enumerate(top.head(3).iterrows()):
    prob = row["xgb_prob"]
    code = row["code"]
    close = row["close"]
    print(f"{code} — 明日涨>2%概率 {prob:.1%} | 均价{close:.2f}")

