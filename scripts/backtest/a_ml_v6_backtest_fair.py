"""回测v6 - 真正公平对比：用普通涨跌Label + 同一测试期
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

print("加载数据...")
df = pd.read_parquet(os.path.join(DATA_OUT, "ml_training_data.parquet"))
with open(os.path.join(DATA_OUT, "ml_feature_cols.json")) as f:
    feature_cols = json.load(f)
df["pct_chg"] = (df["close_next"] - df["close"]) / df["close"]
df["label"] = (df["close_next"] > df["close"]).astype(int)
df = df.dropna(subset=feature_cols).copy()
df["date"] = pd.to_datetime(df["trade_date"])
print(f"  总行数: {len(df)}, 正样本: {df["label"].mean():.1%}")

# 只保留测试数据用于回测（2023-2025）
test_df = df[(df["date"] >= "2023-01-01") & (df["date"] < "2025-06-01")].copy()
# 训练数据（全部之前的）
train_df = df[df["date"] < "2023-01-01"].copy()
print(f"  训练: {len(train_df)}行, 测试: {len(test_df)}行")

# 重新训练普通涨跌XGBoost模型
print("\n训练普通涨跌XGBoost模型...")
t = time.time()
X_tr = train_df[feature_cols]; y_tr = train_df["label"]
model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1, verbosity=0, device='cuda')
model.fit(X_tr, y_tr, eval_set=[(test_df[feature_cols], test_df["label"])], verbose=False)
print(f"  耗时: {time.time()-t:.0f}秒")
model.save_model(os.path.join(MODEL_DIR, "xgb_v1_normal.json"))

# 测试集AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_df["label"], model.predict_proba(test_df[feature_cols])[:, 1])
print(f"  测试集AUC: {auc:.4f}")

# 评分
print("\n评分计算中...")
t = time.time()
# A1
mf_v = test_df["net_mf_amount"].fillna(0).clip(lower=0).values + 1
test_df["a1_score"] = np.maximum(np.log2(mf_v) * 0.3 + np.where(test_df["big_buy_ratio"].fillna(0.5) > 0.6, 1, 0) + np.where((test_df["rsi_14"].fillna(50) > 30) & (test_df["rsi_14"].fillna(50) < 70), 0.5, 0) - np.where(test_df["pct_chg"].fillna(0) < -0.03, 1, 0), 0)
# XGB普通模型
test_df["xgb_score"] = model.predict_proba(test_df[feature_cols])[:, 1]
print(f"  耗时: {time.time()-t:.0f}秒")

# 月频回测
print("\n回测（月频）...")
t = time.time()
test_df["ym"] = test_df["date"].dt.to_period("M")
ym_list = sorted(test_df["ym"].unique())

def bt(score_col):
    trades = []
    for i, ym in enumerate(ym_list[:-1]):  # 不用最后一个月
        # 当前月最后一个交易日选股
        ym_df = test_df[test_df["ym"] == ym]
        month_end = ym_df[ym_df["date"] == ym_df["date"].max()]
        picks = month_end.sort_values(score_col, ascending=False).head(TOP_N)
        buy_lookup = {r["code"]: r["close"] for _, r in picks.iterrows()}
        if not buy_lookup:
            continue
        
        # 下个月最后一个交易日卖出
        next_ym = ym_list[i+1]
        next_df = test_df[test_df["ym"] == next_ym]
        next_end = next_df[next_df["date"] == next_df["date"].max()]
        
        for code, entry in buy_lookup.items():
            exit_row = next_end[next_end["code"] == code]
            if len(exit_row) == 0 or exit_row.iloc[0]["close"] <= 0:
                continue
            ret = (exit_row.iloc[0]["close"] - entry) / entry
            # 简易止损
            for d in sorted(next_df["date"].unique()):
                chk = next_df[(next_df["date"] == d) & (next_df["code"] == code)]
                if len(chk) and chk.iloc[0]["close"] > 0 and (chk.iloc[0]["close"] - entry) / entry < -0.12:
                    ret = -0.12
                    break
            trades.append(ret)
    
    if not trades:
        return {"trades":0,"win_rate":0,"annual_return":0,"max_drawdown":0,"sharpe":0}
    rets = np.array(trades)
    wr = (rets > 0).mean()
    avg_r = rets.mean()
    ann = avg_r * 12
    std = rets.std() if rets.std() > 0 else 0.001
    cum = 100.0; peak = 100.0; maxdd = 0.0
    for r in rets:
        cum *= (1 + r)
        if cum > peak: peak = cum
        dd = (peak - cum) / peak
        if dd > maxdd: maxdd = dd
    sharpe = avg_r / std * np.sqrt(12)
    return {"trades":len(trades),"win_rate":round(float(wr),4),"annual_return":round(float(ann),4),"max_drawdown":round(float(maxdd),4),"sharpe":round(float(sharpe),4)}

a1_r = bt("a1_score")
print(f"  A1: {a1_r['trades']}笔 年化{a1_r['annual_return']:.1%} 回撤{a1_r['max_drawdown']:.1%} 夏普{a1_r['sharpe']:.2f}")
xgb_r = bt("xgb_score")
print(f"  XGB: {xgb_r['trades']}笔 年化{xgb_r['annual_return']:.1%} 回撤{xgb_r['max_drawdown']:.1%} 夏普{xgb_r['sharpe']:.2f}")

print(f"\n{'指标':15s}  {'A1公式':>10s}  {'XGBoost':>10s}  {'胜负':>6s}")
print(f"{'─'*45}")
for name, key in [("交易笔数","trades"), ("胜率","win_rate"), ("年化","annual_return"), ("最大回撤","max_drawdown"), ("夏普比","sharpe")]:
    v1 = a1_r[key]; v2 = xgb_r[key]
    if key == "trades":
        print(f"{name:15s}  {v1:>10}  {v2:>10}  {'ML' if v2>v1 else 'A1':>6}")
    elif key == "max_drawdown":
        print(f"{name:15s}  {v1:>9.1%}  {v2:>9.1%}  {'ML' if v2<v1 else 'A1':>6}")
    else:
        print(f"{name:15s}  {v1:>9.1%}  {v2:>9.1%}  {'ML' if v2>v1 else 'A1':>6}")

print(f"\n耗时: {time.time()-t:.0f}秒")
with open(os.path.join(MODEL_DIR, "bt_compare_v5.json"), "w") as f:
    json.dump({"a1":a1_r,"xgb":xgb_r}, f)
