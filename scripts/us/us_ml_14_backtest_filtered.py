#!/usr/bin/env python3
"""
正式回测v5.1：绿箭v5 + 价格过滤器(>=5美元)
"""
import sys, warnings; warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
import yfinance as yf
from datetime import datetime, timedelta
import json, time

INPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v5.parquet'
MIN_PRICE = 5.0

print("加载模型...")
df = pd.read_parquet(INPUT).dropna(subset=['fwd_5d_ret'])
# 同时保留close列，用于价格过滤
df_orig = pd.read_parquet('/home/hermes/.hermes/openclaw-project/scripts/system/us_hist_yf_5y.parquet')

exclude={'ticker','date','label','fwd_5d_ret'}
feat_cols=[c for c in df.columns if c not in exclude]
y=(df['fwd_5d_ret']>0.05).astype(int).values
X=df[feat_cols].values.astype(np.float32)
X=np.nan_to_num(X,nan=0.0,posinf=0.0,neginf=0.0)
n=len(X); te,ve=int(n*0.7),int(n*0.85)
spw=(te-y[:te].sum())/y[:te].sum()
params={'objective':'binary:logistic','eval_metric':'logloss','tree_method':'hist','device':'cuda',
        'max_depth':6,'learning_rate':0.05,'subsample':0.8,'colsample_bytree':0.8,
        'scale_pos_weight':spw,'random_state':42}
dtrain=xgb.DMatrix(X[:te],y[:te]); dval=xgb.DMatrix(X[te:ve],y[te:ve])
model=xgb.train(params,dtrain,num_boost_round=500,evals=[(dtrain,'train'),(dval,'val')],early_stopping_rounds=20,verbose_eval=False)
val_pred=model.predict(dval); ir=IsotonicRegression(out_of_bounds='clip'); ir.fit(val_pred,y[te:ve])

all_dates = sorted(df['date'].unique())
test_dates = [d for d in all_dates if d >= pd.Timestamp('2025-10-01') and d < pd.Timestamp('2026-06-05')]
print(f"回测期: 2025-10-01 ~ 2026-06-04, {len(test_dates)}个交易日")
print(f"价格过滤: >= ${MIN_PRICE}")

rebalance_interval = 10
max_positions = 5
trade_log = []
no_filter_log = []  # 不算的也记一下对比

for i in range(0, len(test_dates), rebalance_interval):
    today = test_dates[i]
    date_str = str(today)[:10]
    
    # 推荐
    mask = df['date'] <= today
    df_day = df[mask].copy()
    latest = df_day.groupby('ticker').last().reset_index()
    X_day = latest[feat_cols].values.astype(np.float32)
    X_day = np.nan_to_num(X_day, nan=0.0, posinf=0.0, neginf=0.0)
    probs = ir.transform(model.predict(xgb.DMatrix(X_day)))
    sdf = pd.DataFrame({'ticker':latest['ticker'],'prob':probs}).sort_values('prob',ascending=False)
    picks = sdf[sdf['prob']>0.35]
    if len(picks) == 0:
        picks = sdf.head(3)
    
    pick_codes = picks['ticker'].iloc[:max_positions].tolist()
    
    for ticker in pick_codes:
        # 查价格
        last_close = df_orig[(df_orig['ticker']==ticker) & (df_orig['date']<=today)]['close']
        if len(last_close) == 0:
            continue
        price = float(last_close.iloc[-1])
        
        filtered = price < MIN_PRICE
        
        try:
            yf_df = yf.download(ticker, start=date_str, end=(today+timedelta(days=15)).strftime('%Y-%m-%d'), progress=False)
            if len(yf_df) < 2:
                continue
            if hasattr(yf_df.columns,'nlevels') and yf_df.columns.nlevels>1:
                yf_df.columns=[c[0] for c in yf_df.columns.to_flat_index()]
            
            entry = float(yf_df['Close'].iloc[0])
            end_idx = min(5, len(yf_df)-1)
            exit_p = float(yf_df['Close'].iloc[end_idx])
            ret = exit_p/entry - 1
            peak = float(yf_df['High'].iloc[:end_idx+1].max())
            peak_ret = peak/entry - 1
            trough = float(yf_df['Low'].iloc[:end_idx+1].min())
            max_loss = trough/entry - 1
            
            log_entry = {
                'date': date_str,
                'ticker': ticker,
                'price_before': round(price, 2),
                'filtered': filtered,
                'entry': round(entry, 2),
                'exit': round(exit_p, 2),
                'ret_5d': round(float(ret), 4),
                'peak_ret': round(float(peak_ret), 4),
                'max_loss': round(float(max_loss), 4),
            }
            
            if not filtered:
                trade_log.append(log_entry)
            no_filter_log.append(log_entry)
        except:
            pass

print(f"总交易(未过滤): {len(no_filter_log)}笔")
print(f"总交易(>=${MIN_PRICE}): {len(trade_log)}笔")

# === 计算有过滤版本 ===
rets = np.array([t['ret_5d'] for t in trade_log])
peak_rets = np.array([t['peak_ret'] for t in trade_log])
n_trades = len(rets)

periods = max(1, len(test_dates)//5)
avg_trades_per_period = n_trades / periods
avg_ret = rets.mean()
std_ret = rets.std()

# 组合收益：直接每10天调仓周期算
# 策略是每10天调仓一次，每个周期内最多5只票等权
rebalance_windows = []
for i in range(0, len(test_dates), 10):
    w_ret = []
    for t in trade_log:
        t_d = datetime.strptime(t['date'],'%Y-%m-%d')
        r_d = datetime.strptime(str(test_dates[i])[:10],'%Y-%m-%d')
        if (t_d - r_d).days < 2:
            w_ret.append(t['ret_5d'])
    if len(w_ret) >= 2:
        # 等权，每周期平均收益
        rebalance_windows.append(np.mean(w_ret))

port_rets = np.array(rebalance_windows)
n_windows = len(port_rets)

if n_windows > 0:
    avg_port = port_rets.mean()
    std_port = port_rets.std()
    tp_year = 252/10
    annual = (1 + avg_port) ** tp_year - 1
    sharpe = (avg_port / std_port) * np.sqrt(tp_year) if std_port > 0 else 0
    cum_ret = np.cumprod(1 + port_rets)
    rm = np.maximum.accumulate(cum_ret)
    dd = (cum_ret - rm) / rm
    max_dd = abs(dd.min()) * 100
else:
    avg_port = 0; std_port = 0; annual = 0; sharpe = 0; max_dd = 0

# 胜率
wins_5 = (rets > 0.05).mean() * 100
wins_0 = (rets > 0).mean() * 100
wins_peak = (peak_rets > 0.05).mean() * 100

print(f"\n{'='*60}")
print(f"📊 绿箭v5.1 — 回测(>=${MIN_PRICE})")
print(f"{'='*60}")
print(f"回测期: 2025-10 ~ 2026-06 (8个月)")
print(f"总交易笔数: {n_trades}")
print(f"{'='*60}")
print(f"综合指标:")
print(f"  年化收益率: {annual*100:+.1f}%")
print(f"  夏普比率:   {sharpe:.2f}")
print(f"  最大回撤:   {max_dd:.1f}%")
print(f"  平均每笔收益: {avg_ret*100:+.2f}%")
print(f"  每笔标准差:   {std_ret*100:.2f}%")
print(f"{'='*60}")
print(f"胜率:")
print(f"  5天后涨>5%: {wins_5:.1f}%")
print(f"  5天后不亏:  {wins_0:.1f}%")
print(f"  期间最高>5%: {wins_peak:.1f}%")
print(f"{'='*60}")

# === 对比：未过滤 ===
nof_rets = np.array([t['ret_5d'] for t in no_filter_log])
nof_n = len(nof_rets)
nof_wins_5 = (nof_rets > 0.05).mean() * 100
nof_wins_0 = (nof_rets > 0).mean() * 100
nof_avg = nof_rets.mean()

print(f"\n📊 对比(未过滤 vs >=${MIN_PRICE}):")
print(f"{'指标':<18} {'未过滤':>12} {'>=5美元':>12}")
print("-"*42)
print(f"{'交易笔数':<18} {nof_n:>12} {n_trades:>12}")
print(f"{'平均收益':<18} {nof_avg*100:>+11.2f}% {avg_ret*100:>+11.2f}%")
print(f"{'胜率(>5%)':<18} {nof_wins_5:>11.1f}% {wins_5:>11.1f}%")
print(f"{'不亏':<18} {nof_wins_0:>11.1f}% {wins_0:>11.1f}%")

# Top/Worst
print(f"\nTop 5 (>=${MIN_PRICE}):")
for t in sorted(trade_log, key=lambda x:-x['ret_5d'])[:5]:
    print(f"  {t['date']} {t['ticker']} (${t['price_before']}): {t['ret_5d']*100:+.2f}%, 最高{t['peak_ret']*100:+.2f}%")
print(f"\nWorst 5 (>=${MIN_PRICE}):")
for t in sorted(trade_log, key=lambda x:x['ret_5d'])[:5]:
    print(f"  {t['date']} {t['ticker']} (${t['price_before']}): {t['ret_5d']*100:+.2f}%, 最低{t['max_loss']*100:+.2f}%")

# 被过滤掉的
filtered_count = len([t for t in no_filter_log if t['filtered']])
print(f"\n被过滤掉的(<${MIN_PRICE}): {filtered_count}笔")
filtered_rets = [t['ret_5d'] for t in no_filter_log if t['filtered']]
if filtered_rets:
    f_arr = np.array(filtered_rets)
    print(f"  平均收益: {f_arr.mean()*100:+.2f}%, 胜率(>5%): {(f_arr>0.05).mean()*100:.1f}%")
