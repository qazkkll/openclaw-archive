"""回测v4 - 预评分+预合并（O(1)查找，不是每次都全表过滤）
"""
import sys, json, os, time, math
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np
import xgboost as xgb

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUT = "/home/hermes/.hermes/openclaw-archive_ml"
MODEL_DIR = os.path.join(WORKSPACE, "data", "models")

TOP_N = 5; HOLD_DAYS = 20; STOP_LOSS = -0.12

print("加载数据...")
df = pd.read_parquet(os.path.join(DATA_OUT, "ml_training_data.parquet"))
df["pct_chg"] = (df["close_next"] - df["close"]) / df["close"]
with open(os.path.join(DATA_OUT, "ml_feature_cols.json")) as f:
    feature_cols = json.load(f)
df = df.dropna(subset=feature_cols).copy()
df["date"] = pd.to_datetime(df["trade_date"])
print(f"  总行数: {len(df)}")

test_df = df[(df["date"] >= "2023-01-01") & (df["date"] < "2025-06-01")].copy()
dates = sorted(test_df["date"].unique())
print(f"  测试集: {len(test_df)}行, {len(dates)}天")

print("评分计算中（向量化）...")
t = time.time()
mf_v = test_df["net_mf_amount"].fillna(0).clip(lower=0).values + 1
test_df["a1_score"] = np.maximum(np.log2(mf_v) * 0.3 + np.where(test_df["big_buy_ratio"].fillna(0.5) > 0.6, 1, 0) + np.where((test_df["rsi_14"].fillna(50) > 30) & (test_df["rsi_14"].fillna(50) < 70), 0.5, 0) - np.where(test_df["pct_chg"].fillna(0) < -0.03, 1, 0), 0)
print(f"  A1评分: {time.time()-t:.0f}秒")

t = time.time()
model = xgb.XGBClassifier(, device='cuda')
model.load_model(os.path.join(MODEL_DIR, "xgb_v1_high.json"))
test_df["xgb_score"] = model.predict_proba(test_df[feature_cols])[:, 1]
print(f"  XGB评分: {time.time()-t:.0f}秒")

# 建立price_lookup: {code: {date_str: close}}
print("建立价格索引...")
t = time.time()
price_lookup = {}
for _, row in test_df[["code", "date", "close"]].iterrows():
    c = row["code"]
    if c not in price_lookup:
        price_lookup[c] = {}
    price_lookup[c][row["date"]] = row["close"]
print(f"  价格索引: {len(price_lookup)}只股票, {time.time()-t:.0f}秒")

# 回测
def bt(df, dates, score_col):
    trades = []
    for i, day in enumerate(dates):
        picks = df[df["date"] == day].sort_values(score_col, ascending=False).head(TOP_N)
        for _, pp in picks.iterrows():
            code = pp["code"]
            entry = pp["close"]
            exit_day = dates[min(i + HOLD_DAYS, len(dates) - 1)]
            
            exit_price = price_lookup.get(code, {}).get(exit_day)
            if exit_price is None:
                continue
            
            ret = (exit_price - entry) / entry
            # 止损检查
            for j in range(1, HOLD_DAYS + 1):
                if i + j >= len(dates): break
                chk = price_lookup.get(code, {}).get(dates[i + j])
                if chk and (chk - entry) / entry < STOP_LOSS:
                    exit_price = chk
                    ret = STOP_LOSS
                    break
            
            trades.append(ret)
    
    if not trades:
        return {"trades":0,"win_rate":0,"annual_return":0,"max_drawdown":0,"sharpe":0}
    
    rets = np.array(trades)
    win_rate = (rets > 0).mean()
    avg_ret = rets.mean()
    annual_ret = avg_ret * (252 / HOLD_DAYS)
    
    cum = 1.0; peak = 1.0; max_dd = 0.0
    for r in rets:
        cum *= (1 + r)
        if cum > peak: peak = cum
        dd = (peak - cum) / peak
        if dd > max_dd: max_dd = dd
    
    sharpe = avg_ret / rets.std() * np.sqrt(252 / HOLD_DAYS) if rets.std() > 0 else 0
    return {"trades":len(trades),"win_rate":round(float(win_rate),4),"annual_return":round(float(annual_ret),4),"max_drawdown":round(float(max_dd),4),"sharpe":round(float(sharpe),4)}

print("A1回测...")
t = time.time()
a1_r = bt(test_df, dates, "a1_score")
print(f"  {time.time()-t:.0f}秒")

print("XGB回测...")
t = time.time()
xgb_r = bt(test_df, dates, "xgb_score")
print(f"  {time.time()-t:.0f}秒")

print(f"\n{'指标':15s}  {'A1公式':>10s}  {'XGBoost':>10s}  {'胜负':>6s}")
print(f"{'─'*45}")
for name, key, fmt in [("交易笔数","trades","d"), ("胜率","win_rate","%"), ("年化","annual_return","%"), ("最大回撤","max_drawdown","%"), ("夏普比","sharpe","f")]:
    v1 = a1_r[key]; v2 = xgb_r[key]
    if key == "trades": winner = "ML" if v2 > v1 else "A1"; print(f"{name:15s}  {v1:>10}  {v2:>10}  {winner:>6}")
    elif key == "max_drawdown": winner = "ML" if v2 < v1 else "A1"; print(f"{name:15s}  {v1:>9.1%}  {v2:>9.1%}  {winner:>6}")
    else: winner = "ML" if v2 > v1 else "A1"; print(f"{name:15s}  {v1:>9.1%}  {v2:>9.1%}  {winner:>6}")

with open(os.path.join(MODEL_DIR, "bt_compare_v3.json"), "w") as f:
    json.dump({"period":"2023-2025","a1":a1_r,"xgb":xgb_r}, f, indent=2, default=str)
