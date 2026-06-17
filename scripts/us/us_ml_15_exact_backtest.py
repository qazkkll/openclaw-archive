#!/usr/bin/env python3
"""
独立精准回测：绿箭v5.1 (>=5美元)
只用yfinance真实拉价计算，52笔全量，区分filtered/unfiltered
独立算年化/夏普/回撤，不估算
"""
import sys, warnings; warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
import yfinance as yf
from datetime import datetime, timedelta
import json

INPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v5.parquet'
MIN_PRICE = 5.0

print("加载模型...")
df = pd.read_parquet(INPUT).dropna(subset=['fwd_5d_ret'])
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
print(f"回测: {len(test_dates)}个交易日, 价格>={MIN_PRICE}")

# 收集每一笔交易
all_trades = []  # [{ticker, date, price, ret_5d, peak_ret, max_loss, prob, filtered}]

for i in range(0, len(test_dates), 10):
    today = test_dates[i]
    date_str = str(today)[:10]
    
    mask = df['date'] <= today
    df_day = df[mask].copy()
    latest = df_day.groupby('ticker').last().reset_index()
    X_day = latest[feat_cols].values.astype(np.float32)
    X_day = np.nan_to_num(X_day, nan=0.0, posinf=0.0, neginf=0.0)
    probs = ir.transform(model.predict(xgb.DMatrix(X_day)))
    sdf = pd.DataFrame({'ticker':latest['ticker'],'prob':probs}).sort_values('prob',ascending=False)
    picks = sdf[sdf['prob']>0.35]
    if len(picks)==0: picks=sdf.head(3)
    
    for _, pr in picks.head(5).iterrows():
        t = pr['ticker']
        p = float(pr['prob'])
        last_close = df_orig[(df_orig['ticker']==t)&(df_orig['date']<=today)]['close']
        if len(last_close)==0: continue
        price_before = float(last_close.iloc[-1])
        filtered = price_before < MIN_PRICE
        
        try:
            yf_df = yf.download(t, start=date_str, end=(today+timedelta(days=15)).strftime('%Y-%m-%d'), progress=False)
            if len(yf_df)<2: continue
            if hasattr(yf_df.columns,'nlevels') and yf_df.columns.nlevels>1:
                yf_df.columns=[c[0] for c in yf_df.columns.to_flat_index()]
            
            entry = float(yf_df['Close'].iloc[0])
            end_idx = min(5, len(yf_df)-1)
            exit_p = float(yf_df['Close'].iloc[end_idx])
            ret = exit_p/entry-1
            peak = float(yf_df['High'].iloc[:end_idx+1].max())
            peak_ret = peak/entry-1
            trough = float(yf_df['Low'].iloc[:end_idx+1].min())
            max_loss = trough/entry-1
            
            all_trades.append({
                'date': date_str, 'ticker': t, 'prob': round(p,3),
                'price_before': round(price_before,2), 'filtered': filtered,
                'entry': round(entry,2), 'exit': round(exit_p,2),
                'ret_5d': round(ret,4), 'peak_ret': round(peak_ret,4),
                'max_loss': round(max_loss,4),
            })
        except:
            pass

print(f"总交易: {len(all_trades)}笔")
filtered_trades = [t for t in all_trades if not t['filtered']]
print(f"价格>=${MIN_PRICE}: {len(filtered_trades)}笔")

if len(filtered_trades) == 0:
    print("无数据可算")
    sys.exit(1)

# ===== 指标计算 =====
rets = np.array([t['ret_5d'] for t in filtered_trades])
peak_rets = np.array([t['peak_ret'] for t in filtered_trades])
probs = np.array([t['prob'] for t in filtered_trades])

n_trades = len(rets)
avg_ret_per_trade = float(rets.mean())
std_ret_per_trade = float(rets.std())

# 年化：用每10天调仓周期算
# 模拟：每10天买一批（最多5只等权）
# 按日期分组
from collections import defaultdict
windows = defaultdict(list)
for t in filtered_trades:
    windows[t['date']].append(t['ret_5d'])

window_returns = []
for date_str, rets_list in sorted(windows.items()):
    if len(rets_list) >= 1:
        # 等权
        window_returns.append(np.mean(rets_list))

window_rets = np.array(window_returns)
n_windows = len(window_rets)

if n_windows >= 2:
    avg_window_ret = float(window_rets.mean())
    std_window_ret = float(window_rets.std())
    
    # 10天周期，一年约25个周期
    periods_per_year = 252 / 10
    annual_return = (1 + avg_window_ret) ** periods_per_year - 1
    
    # 夏普：无风险利率假设0（美股短线）
    sharpe_ratio = (avg_window_ret / std_window_ret) * np.sqrt(periods_per_year) if std_window_ret > 0 else 0
    
    # 最大回撤
    cum_prod = np.cumprod(1 + window_rets)
    running_max = np.maximum.accumulate(cum_prod)
    drawdowns = (cum_prod - running_max) / running_max
    max_drawdown = abs(float(drawdowns.min())) * 100
    
    # 卡玛比率
    calmar = annual_return / (max_drawdown/100) if max_drawdown > 0 else 0
else:
    avg_window_ret = 0; std_window_ret = 0
    annual_return = 0; sharpe_ratio = 0; max_drawdown = 0; calmar = 0

# 胜率
win_rate_5pct = (rets > 0.05).mean() * 100
win_rate_0 = (rets > 0).mean() * 100
win_rate_peak = (peak_rets > 0.05).mean() * 100

print(f"\n{'='*60}")
print(f"📊 绿箭v5.1 回测 ({len(filtered_trades)}笔, ≥${MIN_PRICE})")
print(f"{'='*60}")
print(f"回测期: 2025-10-01 ~ 2026-06-04 (8个月)")
print(f"交易日数: {len(test_dates)}天")
print(f"交易窗口(每10天): {n_windows}个")
print(f"交易总笔数: {n_trades}")
print(f"{'='*60}")
print(f"核心指标:")
print(f"  年化收益率: {annual_return*100:+>8.1f}%")
print(f"  夏普比率:   {sharpe_ratio:>8.2f}")
print(f"  最大回撤:   {max_drawdown:>8.1f}%")
print(f"  卡玛比率:   {calmar:>8.2f}")
print(f"{'='*60}")
print(f"股票级指标:")
print(f"  平均每笔收益: {avg_ret_per_trade*100:+>8.2f}%")
print(f"  每笔标准差:   {std_ret_per_trade*100:>8.2f}%")
print(f"  胜率(涨>5%): {win_rate_5pct:>8.1f}%")
print(f"  胜率(不亏):   {win_rate_0:>8.1f}%")
print(f"  期间最高>5%: {win_rate_peak:>8.1f}%")
print(f"{'='*60}")
print(f"每窗口收益:")
for i, wr in enumerate(window_rets):
    print(f"  窗口{i+1}: {wr*100:+.2f}%")
print(f"{'='*60}")

# 输出5佳/5差
print("\nTop 5:")
for t in sorted(filtered_trades, key=lambda x:-x['ret_5d'])[:5]:
    print(f"  {t['date']} {t['ticker']}: {t['ret_5d']*100:+.2f}% (峰值{t['peak_ret']*100:+.2f}%, 入场${t['entry']})")
print("\nWorst 5:")
for t in sorted(filtered_trades, key=lambda x:x['ret_5d'])[:5]:
    print(f"  {t['date']} {t['ticker']}: {t['ret_5d']*100:+.2f}% (最低{t['max_loss']*100:+.2f}%, 入场${t['entry']})")

# 全量对比(含<5美元)
all_rets = np.array([t['ret_5d'] for t in all_trades])
print(f"\n📊 对比(含<${MIN_PRICE}妖股):")
print(f"  总笔数: {len(all_trades)}, 均值: {all_rets.mean()*100:+.2f}%, 胜率(>5%): {(all_rets>0.05).mean()*100:.1f}%")

# 保存结果
json.dump({
    'model':'greenshaft_v5.1','filter':'>=5USD',
    'n_trades':n_trades,'n_windows':n_windows,
    'annual_return_pct':round(float(annual_return*100),2),
    'sharpe_ratio':round(float(sharpe_ratio),2),
    'max_drawdown_pct':round(float(max_drawdown),2),
    'calmar_ratio':round(float(calmar),2),
    'avg_trade_ret_pct':round(float(avg_ret_per_trade*100),2),
    'std_trade_ret_pct':round(float(std_ret_per_trade*100),2),
    'win_rate_5pct':round(float(win_rate_5pct),1),
    'win_rate_any':round(float(win_rate_0),1),
    'win_rate_peak':round(float(win_rate_peak),1),
    'trades':[{'date':t['date'],'ticker':t['ticker'],'ret_5d':t['ret_5d'],
               'entry':t['entry'],'exit':t['exit'],'peak_ret':t['peak_ret']}
              for t in filtered_trades],
}, open('/home/hermes/.hermes/openclaw-project/data/models/us/greenshaft_v5.1_backtest.json','w'), indent=2)
print(f"\n回测保存: /home/hermes/.hermes/openclaw-project/data/models/us/greenshaft_v5.1_backtest.json")
