"""回测对比v2 - 轻量版（不逐日循环，用向量化）
"""
import sys, json, os, time
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np
import xgboost as xgb
import math

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUT = "/home/hermes/.hermes/openclaw-archive_ml"
MODEL_DIR = os.path.join(WORKSPACE, "data", "models")

print("1/4 加载数据...")
df = pd.read_parquet(os.path.join(DATA_OUT, "ml_training_data.parquet"))
df["pct_chg_real"] = (df["close_next"] - df["close"]) / df["close"]

with open(os.path.join(DATA_OUT, "ml_feature_cols.json")) as f:
    feature_cols = json.load(f)

df = df.dropna(subset=feature_cols).copy()
df["date_parsed"] = pd.to_datetime(df["trade_date"])
test_df = df[(df["date_parsed"] >= "2023-01-01") & (df["date_parsed"] < "2025-06-01")].copy()
print(f"  测试集: {len(test_df)}行")

print("2/4 计算A1公式评分...")
t = time.time()
test_df["a1_score"] = test_df.apply(lambda r: max(math.log2(abs(r.get("net_mf_amount",0) or 0) + 1) * 0.3 + (1 if (r.get("big_buy_ratio",0.5) or 0.5) > 0.6 else 0) + (0.5 if 30 < (r.get("rsi_14",50) or 50) < 70 else 0) - (1 if (r.get("pct_chg",0) or 0) < -3 else 0), 0), axis=1)
print(f"  A1评分完成 {time.time()-t:.0f}秒")

print("3/4 加载XGBoost评分...")
t = time.time()
model = xgb.XGBClassifier(, device='cuda')
model.load_model(os.path.join(MODEL_DIR, "xgb_v1_high.json"))
test_df["xgb_score"] = model.predict_proba(test_df[feature_cols])[:, 1]
print(f"  XGB评分完成 {time.time()-t:.0f}秒")

print("4/4 回测（月频，简化版）...")
t = time.time()

# 按月份分组模拟回测
test_df["year_month"] = test_df["date_parsed"].dt.to_period("M")

results = []
for method, score_col in [("A1公式", "a1_score"), ("XGBoost", "xgb_score")]:
    all_trades = []
    monthly_returns = []
    
    for ym in sorted(test_df["year_month"].unique()):
        month_df = test_df[test_df["year_month"] == ym]
        month_end = month_df[month_df["date_parsed"] == month_df["date_parsed"].max()]
        
        # 上个月底评分最高的Top 5
        prev_month_df = test_df[test_df["year_month"] < ym]
        if len(prev_month_df) == 0:
            continue
        prev_end = prev_month_df[prev_month_df["date_parsed"] == prev_month_df["date_parsed"].max()]
        picks = prev_end.sort_values(score_col, ascending=False).head(5)
        
        if len(picks) == 0:
            continue
        
        # 等权买入这5只，持有1个月
        trade_returns = []
        for _, pick in picks.iterrows():
            code = pick["code"]
            entry_price = pick["close"]
            
            # 找这个月最后一天的价格
            month_exit = month_df[month_df["code"] == code]
            if len(month_exit) == 0:
                continue
            exit_price = month_exit.iloc[-1]["close"]
            
            ret = (exit_price - entry_price) / entry_price
            trade_returns.append(ret)
            all_trades.append({"code": code, "entry": float(entry_price), "exit": float(exit_price), "ret": float(ret)})
        
        if trade_returns:
            monthly_returns.append(np.mean(trade_returns))
    
    # 统计
    if len(monthly_returns) == 0:
        print(f"  {method}: 无交易")
        continue
    
    monthly_arr = np.array(monthly_returns)
    annual_return = float(np.mean(monthly_arr) * 12)
    max_drawdown = 0
    comp = 1.0
    peak = 1.0
    for r in monthly_arr:
        comp *= (1 + r)
        if comp > peak:
            peak = comp
        dd = (peak - comp) / peak
        if dd > max_drawdown:
            max_drawdown = dd
    
    wins = len([t for t in all_trades if t["ret"] > 0])
    win_rate = wins / len(all_trades) if all_trades else 0
    sharpe = float(np.mean(monthly_arr) / np.std(monthly_arr) * np.sqrt(12)) if np.std(monthly_arr) > 0 else 0
    n_trades = len(all_trades)
    
    results.append((method, n_trades, win_rate, annual_return, max_drawdown, sharpe))
    print(f"  {method}: {n_trades}笔 | 胜率{win_rate:.1%} | 年华{annual_return:.1%} | 回撤{max_drawdown:.2%} | 夏普{sharpe:.2f}")

print(f"\n  回测耗时 {time.time()-t:.0f}秒")
print("\n═══════════ A1 vs XGBoost ═══════════")
print(f"{' ':15s} {'A1公式':>10s} {'XGBoost':>10s}")
print(f"{'─'*40}")
for metric in ["n_trades", "win_rate", "annual_return", "max_drawdown", "sharpe"]:
    idx = 0 if metric == "n_trades" else ["n_trades","win_rate","annual_return","max_drawdown","sharpe"].index(metric)
    v1 = results[0][idx] if len(results) > 0 else 0
    v2 = results[1][idx] if len(results) > 1 else 0
    fmt = f"{'':>7.1%}" if metric in ["win_rate","annual_return","max_drawdown"] else f"{'':>7.2f}" if metric == "sharpe" else f"{'':>7}"
    print(f"{metric:15s} {fmt.format(v1):>10} {fmt.format(v2):>10}")

