"""回测对比v3 - 精准日频版（更快：仅评分Top100候选，非全部）
"""
import sys, json, os, time, math
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np
import xgboost as xgb

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUT = "/home/hermes/.hermes/openclaw-archive_ml"
MODEL_DIR = os.path.join(WORKSPACE, "data", "models")

TOP_N = 5
HOLD_DAYS = 20
STOP_LOSS = -0.12

print("加载数据...")
df = pd.read_parquet(os.path.join(DATA_OUT, "ml_training_data.parquet"))
df["pct_chg"] = (df["close_next"] - df["close"]) / df["close"]

with open(os.path.join(DATA_OUT, "ml_feature_cols.json")) as f:
    feature_cols = json.load(f)

df = df.dropna(subset=feature_cols).copy()
df["date_str"] = pd.to_datetime(df["trade_date"])
print(f"  总数据: {len(df)}行")

# 只用2023-2025
test_df = df[(df["date_str"] >= "2023-01-01") & (df["date_str"] < "2025-06-01")].copy()
unique_dates = sorted(test_df["date_str"].unique())
print(f"  测试集: {len(test_df)}行, {len(unique_dates)}个交易日")

# 预计算评分（避免逐日计算apply的重复劳动）
print("评分计算中...")

# A1评分（向量化优化版）
t = time.time()
mf = test_df["net_mf_amount"].fillna(0).clip(lower=0).values + 1
test_df["a1_score"] = np.maximum(np.log2(mf) * 0.3 + np.where(test_df["big_buy_ratio"].fillna(0.5) > 0.6, 1, 0) + np.where((test_df["rsi_14"].fillna(50) > 30) & (test_df["rsi_14"].fillna(50) < 70), 0.5, 0) - np.where(test_df["pct_chg"].fillna(0) < -0.03, 1, 0), 0)
print(f"  A1评分: {time.time()-t:.0f}秒")

# XGBoost评分
t = time.time()
model = xgb.XGBClassifier(, device='cuda')
model.load_model(os.path.join(MODEL_DIR, "xgb_v1_high.json"))
test_df["xgb_score"] = model.predict_proba(test_df[feature_cols])[:, 1]
print(f"  XGB评分: {time.time()-t:.0f}秒")

print(f"A1高评分(>5)占比: {(test_df["a1_score"] > 5).mean():.1%}")
print(f"XGB高评分(>0.5)占比: {(test_df["xgb_score"] > 0.5).mean():.1%}")

# 回测
print("\n回测中...")
def bt(df, dates, score_col, top_n=5):
    """最简回测: 每天选评分最高top_n只，等权买入，持有结束后关掉"""
    trades = []
    
    for i, d in enumerate(dates):
        day_df = df[df["date_str"] == d].sort_values(score_col, ascending=False).head(top_n)
        if len(day_df) == 0:
            continue
        
        for _, row in day_df.iterrows():
            code = row["code"]
            entry_price = row["close"]
            exit_date = dates[min(i + HOLD_DAYS, len(dates) - 1)]
            
            # 找退出日的价格
            exit_rows = df[(df["code"] == code) & (df["date_str"] == exit_date)]
            if len(exit_rows) == 0:
                continue
            exit_price = exit_rows.iloc[0]["close"]
            
            ret = (exit_price - entry_price) / entry_price
            if ret < STOP_LOSS:
                # 止损——没精确找止损日，用夹逼法估算
                for j in range(1, HOLD_DAYS + 1):
                    if i + j >= len(dates):
                        break
                    chk = df[(df["code"] == code) & (df["date_str"] == dates[i + j])]
                    if len(chk) and (chk.iloc[0]["close"] - entry_price) / entry_price < STOP_LOSS:
                        exit_price = chk.iloc[0]["close"]
                        ret = STOP_LOSS
                        break
            
            trades.append({"code": code, "entry": float(entry_price), "exit": float(exit_price), "ret": float(ret), "buy_date": str(d.date()), "days_held": HOLD_DAYS})
    
    if not trades:
        return {"trades": 0}
    
    # 统计
    rets = np.array([t["ret"] for t in trades])
    wins = rets > 0
    win_rate = wins.mean()
    avg_ret = rets.mean()
    annual_ret = avg_ret * (252 / HOLD_DAYS)
    
    # 回撤（简化：均匀持有）
    cumulative = 1.0
    peak = 1.0
    max_dd = 0
    for r in rets:
        cumulative *= (1 + r)
        if cumulative > peak:
            peak = cumulative
        dd = (peak - cumulative) / peak
        if dd > max_dd:
            max_dd = dd
    
    sharpe = avg_ret / rets.std() * np.sqrt(252 / HOLD_DAYS) if rets.std() > 0 else 0
    
    return {"trades": len(trades), "win_rate": round(float(win_rate),4), "annual_return": round(float(annual_ret),4), "max_drawdown": round(float(max_dd),4), "sharpe": round(float(sharpe),4)}

print("  A1...")
a1_result = bt(test_df, unique_dates, "a1_score")
print(f"    {a1_result["trades"]}笔 | 胜率{a1_result["win_rate"]:.1%} | 年化{a1_result["annual_return"]:.1%} | 回撤{a1_result["max_drawdown"]:.1%} | 夏普{a1_result["sharpe"]:.2f}")

print("  XGBoost...")
xgb_result = bt(test_df, unique_dates, "xgb_score")
print(f"    {xgb_result["trades"]}笔 | 胜率{xgb_result["win_rate"]:.1%} | 年化{xgb_result["annual_return"]:.1%} | 回撤{xgb_result["max_drawdown"]:.1%} | 夏普{xgb_result["sharpe"]:.2f}")

print("\n═══════════════════════════════════════")
print(f"{'指标':15s}  {'A1公式':>10s}  {'XGBoost':>10s}  {'胜负':>6s}")
print(f"{'─'*48}")
for name, key in [("交易笔数","trades"), ("胜率","win_rate"), ("年化","annual_return"), ("最大回撤","max_drawdown"), ("夏普比","sharpe")]:
    v1 = a1_result[key]; v2 = xgb_result[key]
    if key == "trades":
        winner = "ML" if v2 > v1 else "A1"
        print(f"{name:15s}  {v1:>10}  {v2:>10}  {winner:>6}")
    elif key == "max_drawdown":
        winner = "ML" if v2 < v1 else "A1"
        print(f"{name:15s}  {v1:>9.1%}  {v2:>9.1%}  {winner:>6}")
    else:
        winner = "ML" if v2 > v1 else "A1"
        if key == "sharpe":
            print(f"{name:15s}  {v1:>9.2f}  {v2:>9.2f}  {winner:>6}")
        else:
            print(f"{name:15s}  {v1:>9.1%}  {v2:>9.1%}  {winner:>6}")

result = {"period": "2023-2025", "a1": a1_result, "xgb": xgb_result}
with open(os.path.join(MODEL_DIR, "bt_compare_v2.json"), "w") as f:
    json.dump(result, f, indent=2, default=str)
