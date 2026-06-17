"""
绿箭v19 干净回测 — 用SPY日历对齐日期，逐日模拟交易
"""
import sys, os, json, math, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import xgboost as xgb
import yfinance as yf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _paths

T0 = time.time()
print("═"*60)
print("绿箭v19 干净回测")
print("═"*60)

# 1. 获取交易日历
spy = yf.Ticker("SPY")
spy_hist = spy.history(period='3y')
all_trading_days = spy_hist.index.strftime('%Y-%m-%d').tolist()
print(f"交易日历: {len(all_trading_days)}天, {all_trading_days[0]}~{all_trading_days[-1]}")

# 2. 加载特征 + 对齐日期
print("加载特征文件...")
df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v2.parquet")
print(f"  总行数: {len(df):,}, 股票: {df['sym'].nunique()}")

with open(_paths.ML_DIR+"/us_sector_etf.json") as f:
    etf_data = json.load(f)

s2e={'Technology':'XLK','Financial Services':'XLF','Financial':'XLF','Energy':'XLE',
     'Healthcare':'XLV','Industrials':'XLI','Consumer Defensive':'XLP',
     'Consumer Cyclical':'XLY','Utilities':'XLU','Basic Materials':'XLB',
     'Materials':'XLB','Real Estate':'XLRE','Communication Services':'XLC','Semiconductor':'SMH'}
def get_er(s):
    e=s2e.get(s)
    return etf_data[e]['ret5'] if e and e in etf_data else etf_data['SPY']['ret5']

df['sector_etf_ret5']=df['sector'].apply(get_er)
for k in ['SPY','QQQ','IWM']:
    df[f'{k.lower()}_ret5']=etf_data[k]['ret5']
df['sc']=df['sector'].astype('category').cat.codes.astype(int)

feats=['price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
       'ret1','ret5','ret20','ret60','macd','macd_signal','macd_hist',
       'vol_ratio','ma_bias20','vol5','trend_accel',
       'short_ratio','short_pct','market_cap','pe_trailing','pe_forward','beta',
       'sector_etf_ret5','spy_ret5','qqq_ret5','iwm_ret5','sc']

df = df.dropna(subset=feats+['label_5d_pct'])
print(f"  有标签行数: {len(df):,}")

# 每只股票分配日期
sym_counts = df['sym'].value_counts()
dates_per_sym = {}
for sym in df['sym'].unique():
    n = sym_counts[sym]
    if n <= len(all_trading_days):
        dates_per_sym[sym] = all_trading_days[-n:]
    else:
        dates_per_sym[sym] = ['1900-01-01']*(n-len(all_trading_days)) + all_trading_days

print("  分配日期...")
df['date'] = '1900-01-01'
for sym in dates_per_sym:
    mask = df['sym'] == sym
    dlist = dates_per_sym[sym]
    df.loc[mask, 'date'] = dlist[:mask.sum()] + ['1900-01-01']*(mask.sum()-len(dlist))

df['date'] = pd.to_datetime(df['date'])
df = df[df['date'].dt.year >= 2020].copy()
print(f"  2020年后: {len(df):,}行")

# 3. 时间切分
dates = sorted(df['date'].unique())
split_idx = int(len(dates) * 0.7)
train_dates = set(dates[:split_idx])
test_dates = dates[split_idx:]
print(f"  训练截止: {dates[split_idx-1]}")
print(f"  回测开始: {test_dates[0]}")
print(f"  回测天数: {len(test_dates)}")

# 4. 模型（已在全量数据上训练）
m = xgb.XGBClassifier(, device='cuda')
m.load_model(_paths.US_MODEL_DIR + '/greenshaft_v19_base.json')

# 5. 逐日回测
print("\n逐日回测...")
daily_results = {5:[], 10:[], 20:[], 50:[]}
dates_processed = []

for di, date in enumerate(test_dates):
    if di % 50 == 0:
        print(f"  {date} ({di}/{len(test_dates)})", flush=True)
    
    # 当天所有股票
    day_df = df[df['date'] == pd.Timestamp(date)]
    if len(day_df) < 100:
        continue
    
    X_day = day_df[feats].values
    pct_day = day_df['label_5d_pct'].values
    sym_day = day_df['sym'].values
    
    pu5_day = m.predict_proba(X_day)[:, 4]
    
    for top_n in [5, 10, 20, 50]:
        idx = np.argsort(-pu5_day)[:min(top_n, len(pu5_day))]
        if len(idx) == 0:
            continue
        
        # 等权收益
        rets = pct_day[idx]
        eq_ret = np.mean(rets)
        w = pu5_day[idx]
        w = w / w.sum()
        wgt_ret = np.dot(rets, w)
        
        daily_results[top_n].append({
            'date': date,
            'eq_ret': eq_ret,
            'wgt_ret': wgt_ret,
            'hit_up5': float((rets > 5).mean()),
            'hit_up0': float((rets > 0).mean()),
            'n': len(idx),
            'max_ret': float(rets.max()),
            'min_ret': float(rets.min()),
            'avg_prob': float(pu5_day[idx].mean()),
            'syms': ','.join(sym_day[idx][:5]),
        })
    
    dates_processed.append(date)

# 6. 结果
print("\n" + "="*60)
print("回测结果")
print("="*60)

for top_n in [5, 10, 20, 50]:
    results = daily_results[top_n]
    if not results:
        continue
    
    eq_rets = np.array([r['eq_ret'] for r in results if abs(r['eq_ret']) < 50])
    wgt_rets = np.array([r['wgt_ret'] for r in results if abs(r['wgt_ret']) < 50])
    
    cum_eq = (np.prod(1 + eq_rets/100) - 1) * 100
    avg_eq = np.mean(eq_rets)
    med_eq = np.median(eq_rets)
    std_eq = np.std(eq_rets)
    sp_eq = avg_eq / std_eq * math.sqrt(252) if std_eq > 0 else 0
    win_eq = (eq_rets > 0).mean()
    best = eq_rets.max()
    worst = eq_rets.min()
    
    cum_series = np.cumprod(1 + eq_rets/100)
    peak = np.maximum.accumulate(cum_series)
    dd = (cum_series - peak) / peak
    mdd = dd.min()
    
    avg_hit5 = np.mean([r['hit_up5'] for r in results])
    avg_hit0 = np.mean([r['hit_up0'] for r in results])
    # 累积再平衡模拟
    eq_rets_clip = np.array([r['eq_ret'] for r in results])  # 不裁剪
    eq_rets_clip = np.clip(eq_rets_clip, -50, 50)
    
    print(f"\n【每日Top{top_n}】 {len(results)}个交易日")
    print(f"  累积收益: {cum_eq:+.2f}%")
    print(f"  单笔均收益: {avg_eq:+.4f}% | 中位: {med_eq:+.4f}%")
    print(f"  胜率(涨>0): {win_eq:.1%}")
    print(f"  夏普(年化): {sp_eq:.3f}")
    print(f"  最大回撤: {mdd*100:.1f}%")
    print(f"  单日最佳: {best:+.2f}% | 单日最差: {worst:+.2f}%")
    print(f"  涨>5%平均命中: {avg_hit5:.1%}")
    print(f"  涨>0平均胜率: {avg_hit0:.1%}")
    
    # 年度分解
    from collections import defaultdict
    yearly = defaultdict(list)
    for r in results:
        y = r['date'][:4]
        yearly[y].append(r['eq_ret'])
    if len(yearly) > 1:
        print(f"  {'年份':>6} {'天数':>5} {'均收益':>9} {'胜率':>7} {'夏普':>8} {'累积':>9}")
        for yr in sorted(yearly.keys()):
            yr_rets = yearly[yr]
            y_avg = np.mean(yr_rets)
            y_std = np.std(yr_rets)
            y_sp = y_avg/y_std*math.sqrt(252) if y_std>0 else 0
            y_cum = (np.prod(1+np.array(yr_rets)/100)-1)*100
            y_win = sum(1 for r in yr_rets if r>0)/len(yr_rets)
            print(f"  {yr:>6} {len(yr_rets):>5} {y_avg:>+8.3f}% {y_win:>6.1%} {y_sp:>7.3f} {y_cum:>+8.2f}%")

print(f"\n总耗时: {time.time()-T0:.0f}s")
