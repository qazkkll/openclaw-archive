#!/usr/bin/env python3
"""
正式回测：绿箭v5，2025-10至2026-05，8个月
策略：每2周调仓，选概率>35%等权分批建仓，5天后止盈止损
输出：年化/夏普/回撤/胜率/交易明细
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
POS_FILE = '/home/hermes/.hermes/openclaw-project/data/positions_opend.json'

print("加载模型...")
df = pd.read_parquet(INPUT).dropna(subset=['fwd_5d_ret'])
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

# 回测日期范围
all_dates = sorted(df['date'].unique())
test_dates = [d for d in all_dates if d >= pd.Timestamp('2025-10-01') and d < pd.Timestamp('2026-06-05')]
print(f"回测期: 2025-10-01 ~ 2026-06-04, {len(test_dates)}个交易日")

# 每2周(10个交易日)调仓
rebalance_interval = 10
max_positions = 5
equity_curve = [100000]
trade_log = []

for i in range(0, len(test_dates), rebalance_interval):
    today = test_dates[i]
    date_str = str(today)[:10]
    
    # 获取当日推荐
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
    
    # 用yfinance查买入价
    for ticker in pick_codes:
        try:
            yf_df = yf.download(ticker, start=date_str, end=(today+timedelta(days=15)).strftime('%Y-%m-%d'), progress=False)
            if len(yf_df) < 2:
                continue
            if hasattr(yf_df.columns,'nlevels') and yf_df.columns.nlevels>1:
                yf_df.columns=[c[0] for c in yf_df.columns.to_flat_index()]
            
            entry_price = yf_df['Close'].iloc[0]
            
            # 5天后
            end_idx = min(5, len(yf_df)-1)
            exit_price = yf_df['Close'].iloc[end_idx]
            ret = exit_price/entry_price - 1
            
            # 期间最高
            peak = yf_df['High'].iloc[:end_idx+1].max()
            peak_ret = peak/entry_price - 1
            peak_idx = yf_df['High'].iloc[:end_idx+1].idxmax()
            
            # 止损检查：期间最低跌多少
            trough = yf_df['Low'].iloc[:end_idx+1].min()
            max_loss = trough/entry_price - 1
            
            trade_log.append({
                'date': date_str, 'ticker': ticker, 'prob': float(probs[max(0,len(pick_codes)-1)]),
                'entry': float(entry_price), 'exit': float(exit_price),
                'ret_5d': float(ret), 'peak_ret': float(peak_ret), 'max_loss': float(max_loss),
                'peak_date': str(peak_idx)[:10],
            })
        except Exception as e:
            trade_log.append({'date':date_str,'ticker':ticker,'prob':0,'entry':0,'exit':0,'ret_5d':0,'peak_ret':0,'max_loss':0,'peak_date':'err'})
    
    equity_curve.append(equity_curve[-1])

print(f"总交易: {len(trade_log)}笔")

# 计算指标
rets = np.array([t['ret_5d'] for t in trade_log if t['entry'] > 0])
peak_rets = np.array([t['peak_ret'] for t in trade_log if t['entry'] > 0])
n_trades = len(rets)

# 按5日周期算，8个月约48个5天周期，每周期若干笔
periods = max(1, len(test_dates)//5)
avg_trades_per_period = n_trades / periods
avg_ret_per_trade = rets.mean()
std_ret = rets.std()

# 每5天计算组合收益
portfolio_returns = []
for i in range(0, n_trades, max(1, int(avg_trades_per_period))):
    batch = rets[i:i+int(avg_trades_per_period)]
    if len(batch) > 0:
        portfolio_returns.append(batch.mean())

port_rets = np.array(portfolio_returns)
avg_port_ret = port_rets.mean()
std_port_ret = port_rets.std()

# 年化
trading_periods_per_year = 252 / 5  # 约50个5天周期
annual_ret = (1 + avg_port_ret) ** trading_periods_per_year - 1
sharpe = (avg_port_ret / std_port_ret) * np.sqrt(trading_periods_per_year) if std_port_ret > 0 else 0

# 最大回撤
cum_ret = np.cumprod(1 + port_rets)
running_max = np.maximum.accumulate(cum_ret)
dd = (cum_ret - running_max) / running_max
max_dd = abs(dd.min()) * 100

print(f"\n{'='*60}")
print(f"📊 绿箭v5 正式回测")
print(f"{'='*60}")
print(f"回测期: 2025-10-01 ~ 2026-06-04 (8个月)")
print(f"总交易笔数: {n_trades}")
print(f"策略: 每10天调仓, >35%概率, 最多5只")
print(f"{'='*60}")
print(f"综合指标:")
print(f"  年化收益率: {annual_ret*100:+.1f}%")
print(f"  夏普比率:   {sharpe:.2f}")
print(f"  最大回撤:   {max_dd:.1f}%")
print(f"  平均每笔收益: {avg_ret_per_trade*100:+.2f}%")
print(f"  每笔标准差:   {std_ret*100:.2f}%")
print(f"{'='*60}")
print(f"胜率:")
wins_5 = (rets > 0.05).mean() * 100
wins_0 = (rets > 0).mean() * 100
wins_peak = (peak_rets > 0.05).mean() * 100
print(f"  5天后涨>5%: {wins_5:.1f}%")
print(f"  5天后不亏:  {wins_0:.1f}%")
print(f"  期间最高>5%: {wins_peak:.1f}%")
print(f"{'='*60}")

# 最好/最差
print(f"\nTop 5 交易:")
if len(trade_log) > 0:
    top5 = sorted(trade_log, key=lambda x: -x['ret_5d'])[:5]
    for t in top5:
        print(f"  {t['date']} {t['ticker']}: 入场{t['entry']:.2f}→5日{t['exit']:.2f} ({t['ret_5d']*100:+.2f}%), 期间最高{t['peak_ret']*100:+.2f}%")

print(f"\nWorst 5 交易:")
if len(trade_log) > 0:
    bot5 = sorted(trade_log, key=lambda x: x['ret_5d'])[:5]
    for t in bot5:
        print(f"  {t['date']} {t['ticker']}: 入场{t['entry']:.2f}→5日{t['exit']:.2f} ({t['ret_5d']*100:+.2f}%), 期间最低{t['max_loss']*100:+.2f}%")

# 保存回测结果
backtest_result = {
    'model': 'greenshaft_v5',
    'period': '2025-10-01~2026-06-04',
    'n_trades': n_trades,
    'annual_return_pct': round(float(annual_ret*100), 2),
    'sharpe_ratio': round(float(sharpe), 2),
    'max_drawdown_pct': round(float(max_dd), 2),
    'avg_trade_ret_pct': round(float(avg_ret_per_trade*100), 2),
    'win_rate_5pct': round(float(wins_5), 1),
    'win_rate_any': round(float(wins_0), 1),
    'peak_hit_5pct': round(float(wins_peak), 1),
}
json.dump(backtest_result, open('/home/hermes/.hermes/openclaw-project/data/models/us/greenshaft_v5_backtest.json','w'), indent=2)
print(f"\n回测结果保存: /home/hermes/.hermes/openclaw-project/data/models/us/greenshaft_v5_backtest.json")
