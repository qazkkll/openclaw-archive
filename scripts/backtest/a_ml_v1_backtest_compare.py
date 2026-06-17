"""回测对比：A1公式 vs XGBoost ML
用同样的交易规则（Top5等权、持有20天、止损-12%）
"""
import sys, json, os, time
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np
import xgboost as xgb

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUT = "/home/hermes/.hermes/openclaw-archive_ml"
MODEL_DIR = os.path.join(WORKSPACE, "data", "models")
DATA_IN = os.path.join(WORKSPACE, "data")

# 回测参数
TOP_N = 5
HOLD_DAYS = 20
STOP_LOSS = -0.12

print("加载数据...")
df = pd.read_parquet(os.path.join(DATA_OUT, "ml_training_data.parquet"))
with open(os.path.join(DATA_OUT, "ml_feature_cols.json")) as f:
    feature_cols = json.load(f)

df["pct_chg"] = (df["close_next"] - df["close"]) / df["close"]
df = df.dropna(subset=feature_cols).copy()
df["date_parsed"] = pd.to_datetime(df["trade_date"])

# 只取2023-2025作为测试期（之前没训练过）
test_df = df[(df["date_parsed"] >= "2023-01-01") & (df["date_parsed"] < "2025-06-01")].copy()
print(f"测试期行数: {len(test_df)}")

# 1. A1公式评分
print("\n计算A1评分...")
def a1_score(row):
    import math
    mf = row.get("net_mf_amount", 0) or 0
    rsi = row.get("rsi_14", 50) or 50
    pct = row.get("pct_chg", 0) or 0
    big_buy = row.get("big_buy_ratio", 0.5) or 0.5
    
    score = math.log2(abs(mf) + 1) * 0.3
    if big_buy > 0.6:
        score += 1
    if 30 < rsi < 70:
        score += 0.5
    if pct < -3:
        score -= 1
    return max(score, 0)

test_df["a1_score"] = test_df.apply(a1_score, axis=1)

# 2. XGBoost评分（加载已训练的高收益模型）
print("加载XGBoost模型...")
model = xgb.XGBClassifier(, device='cuda')
model.load_model(os.path.join(MODEL_DIR, "xgb_v1_high.json"))
test_df["xgb_score"] = model.predict_proba(test_df[feature_cols])[:, 1]

# 回测
print("\n回测进行中...")
def backtest(df, score_col, top_n=5, hold=20, stop=-0.12):
    dates = sorted(df["date_parsed"].unique())
    capital = 1.0
    positions = []
    trades = []
    equity_curve = [(dates[0], 1.0)]
    
    for i, today in enumerate(dates):
        today_df = df[df["date_parsed"] == today].sort_values(score_col, ascending=False).head(top_n)
        
        # 更新持仓
        new_positions = []
        for code, entry_date, entry_price, shares in positions:
            if i < len(today_df) and code in today_df["code"].values:
                row = today_df[today_df["code"] == code].iloc[0]
                current_price = row["close"]
                pnl = (current_price - entry_price) / entry_price
                days_held = (today - entry_date).days
                if pnl < stop or days_held >= hold:
                    capital += shares * current_price / 1.0  # 简化
                    trades.append({"code": code, "entry": float(entry_price), "exit": float(current_price), "pnl_pct": float(pnl), "days": days_held})
                    continue
            new_positions.append((code, entry_date, entry_price, shares))
        positions = new_positions
        
        # 新开仓
        for _, row in today_df.iterrows():
            if len(positions) >= top_n:
                break
            code = row["code"]
            if any(p[0] == code for p in positions):
                continue
            price = row["close"]
            entry_cost = capital * 0.2  # 等权
            shares = entry_cost / price
            positions.append((code, today, price, shares))
        
        equity = capital + sum(shares * (df[(df["code"]==c) & (df["date_parsed"]==today)]["close"].values[0] if len(df[(df["code"]==c) & (df["date_parsed"]==today)]) > 0 else 0) for c, _, _, shares in positions if len(df[(df["code"]==c) & (df["date_parsed"]==today)]) > 0)
        equity_curve.append((today, float(equity)))
    
    # 统计
    returns = []
    for i in range(1, len(equity_curve)):
        r = equity_curve[i][1] / equity_curve[i-1][1] - 1
        returns.append(r)
    
    if len(returns) == 0:
        return {"trades": 0}
    
    annual_return = float(np.mean(returns) * 252)
    max_dd = 0
    peak = equity_curve[0][1]
    for _, e in equity_curve:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd
    
    win_rate = len([t for t in trades if t["pnl_pct"] > 0]) / len(trades) if trades else 0
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    
    return {
        "trades": len(trades), "win_rate": round(win_rate, 4),
        "annual_return": round(annual_return, 4),
        "max_drawdown": round(float(max_dd), 4),
        "sharpe": round(sharpe, 4),
        "returns": returns
    }

# A1回测
print("  跑A1回测...")
a1_result = backtest(test_df, "a1_score")

# XGBoost回测
print("  跑XGBoost回测...")
xgb_result = backtest(test_df, "xgb_score")

print("\n═══════════════════════════════════════")
print("     A1公式         XGBoost ML")
print("───────────────────────────────────────")
print(f"交易:   {a1_result['trades']:>5}         {xgb_result['trades']:>5}")
print(f"胜率:   {a1_result['win_rate']:>7.1%}      {xgb_result['win_rate']:>7.1%} ")
print(f"年化:   {a1_result['annual_return']:>7.2%}    {xgb_result['annual_return']:>7.2%}")
print(f"回撤:   {a1_result['max_drawdown']:>7.2%}    {xgb_result['max_drawdown']:>7.2%}")
print(f"夏普:   {a1_result['sharpe']:>7.3f}      {xgb_result['sharpe']:>7.3f}")
print("───────────────────────────────────────")

# 保存结果
result = {
    "period": "2023-2025", "label": "high_return (>2%)",
    "a1": a1_result, "xgb": xgb_result
}
with open(os.path.join(MODEL_DIR, "bt_compare_v1.json"), "w") as f:
    json.dump(result, f, indent=2, default=str)
