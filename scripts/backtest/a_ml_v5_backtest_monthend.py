"""回测v5 - fix: 不在最后一日强平，没价格跳过继续；加入滚动资金曲线
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

print("加载...")
df = pd.read_parquet(os.path.join(DATA_OUT, "ml_training_data.parquet"))
df["pct_chg"] = (df["close_next"] - df["close"]) / df["close"]
with open(os.path.join(DATA_OUT, "ml_feature_cols.json")) as f:
    feature_cols = json.load(f)
df = df.dropna(subset=feature_cols).copy()
df["date"] = pd.to_datetime(df["trade_date"])

test_df = df[(df["date"] >= "2023-01-01") & (df["date"] < "2025-06-01")].copy()
dates = sorted(test_df["date"].unique())
print(f"  测试: {len(test_df)}行, {len(dates)}天")

# 评分
print("评分...")
t = time.time()
mf_v = test_df["net_mf_amount"].fillna(0).clip(lower=0).values + 1
test_df["a1_score"] = np.maximum(np.log2(mf_v) * 0.3 + np.where(test_df["big_buy_ratio"].fillna(0.5) > 0.6, 1, 0) + np.where((test_df["rsi_14"].fillna(50) > 30) & (test_df["rsi_14"].fillna(50) < 70), 0.5, 0) - np.where(test_df["pct_chg"].fillna(0) < -0.03, 1, 0), 0)
model = xgb.XGBClassifier(, device='cuda'); model.load_model(os.path.join(MODEL_DIR, "xgb_v1_high.json"))
test_df["xgb_score"] = model.predict_proba(test_df[feature_cols])[:, 1]
print(f"  {time.time()-t:.0f}秒")

# 按日期分组，选出评分最高的票
print("回测（月频仿真）...")
t = time.time()

def bt_monthly(df_scores, score_col):
    trades = []
    date_strs = sorted(df_scores["date"].unique())
    
    # 按月选出Top5：上个月底选出，持有1个月
    # 用月度切换
    df_scores["ym"] = df_scores["date"].dt.to_period("M")
    
    for ym in sorted(df_scores["ym"].unique()):
        month_df = df_scores[df_scores["ym"] == ym]
        if len(month_df) == 0:
            continue
        
        # 取上个月最后3天的评分均值选Top N
        prev_data = df_scores[df_scores["ym"] == ym - 1] if ym - 1 >= df_scores["ym"].min() else df_scores[df_scores["ym"] == ym]
        if len(prev_data) == 0:
            continue
        prev_end = prev_data[prev_data["date"] == prev_data["date"].max()]
        if len(prev_end) == 0:
            continue
        
        picks = prev_end.sort_values(score_col, ascending=False).head(TOP_N)
        buy_price_lookup = {r["code"]: r["close"] for _, r in picks.iterrows()}
        # 无买价的跳过
        if not buy_price_lookup:
            continue
        
        # 这个月最后一个交易日卖出
        month_end = month_df[month_df["date"] == month_df["date"].max()]
        
        for code, entry in buy_price_lookup.items():
            exit_row = month_end[month_end["code"] == code]
            if len(exit_row) == 0:
                continue
            exit_price = exit_row.iloc[0]["close"]
            if exit_price <= 0:
                continue
            ret = (exit_price - entry) / entry
            
            # 止损
            for d in sorted(month_df["date"].unique()):
                chk = month_df[(month_df["date"] == d) & (month_df["code"] == code)]
                if len(chk) and chk.iloc[0]["close"] > 0:
                    interim_pnl = (chk.iloc[0]["close"] - entry) / entry
                    if interim_pnl < STOP_LOSS:
                        ret = STOP_LOSS
                        break
            
            trades.append(ret)
    
    if not trades:
        return {"trades":0,"win_rate":0,"annual_return":0,"max_drawdown":0,"sharpe":0}
    
    rets = np.array(trades)
    win_rate = (rets > 0).mean()
    avg_ret = rets.mean()
    annual_ret = avg_ret * 12  # 月度换算
    std = rets.std()
    
    cum = 100.0; peak = 100.0; max_dd = 0.0
    for r in rets:
        cum *= (1 + r)
        if cum > peak: peak = cum
        dd = (peak - cum) / peak
        if dd > max_dd: max_dd = dd
    
    sharpe = avg_ret / std * np.sqrt(12) if std > 0 else 0
    return {"trades":len(trades),"win_rate":round(float(win_rate),4),"annual_return":round(float(annual_ret),4),"max_drawdown":round(float(max_dd),4),"sharpe":round(float(sharpe),4)}

a1_r = bt_monthly(test_df, "a1_score")
print(f"  A1: {a1_r['trades']}笔 年化{a1_r['annual_return']:.1%} 回撤{a1_r['max_drawdown']:.1%} 夏普{a1_r['sharpe']:.2f}")
xgb_r = bt_monthly(test_df, "xgb_score")
print(f"  XGB: {xgb_r['trades']}笔 年化{xgb_r['annual_return']:.1%} 回撤{xgb_r['max_drawdown']:.1%} 夏普{xgb_r['sharpe']:.2f}")

# 结果表
print(f"\n{'指标':15s}  {'A1公式':>10s}  {'XGBoost':>10s}  {'胜负':>6s}")
print(f"{'─'*45}")
for name, key, fmt in [("交易笔数","trades","d"), ("胜率","win_rate","%"), ("年化","annual_return","%"), ("最大回撤","max_drawdown","%"), ("夏普比","sharpe","f")]:
    v1 = a1_r[key]; v2 = xgb_r[key]
    if key == "trades": winner = "ML" if v2 > v1 else "A1"; print(f"{name:15s}  {v1:>10}  {v2:>10}  {winner:>6}")
    elif key == "max_drawdown": winner = "ML" if v2 < v1 else "A1"; print(f"{name:15s}  {v1:>9.1%}  {v2:>9.1%}  {winner:>6}")
    else: winner = "ML" if v2 > v1 else "A1"; print(f"{name:15s}  {v1:>9.1%}  {v2:>9.1%}  {winner:>6}")

print(f"\n耗时: {time.time()-t:.0f}秒")
with open(os.path.join(MODEL_DIR, "bt_compare_v4.json"), "w") as f:
    json.dump({"a1":a1_r,"xgb":xgb_r}, f)
