"""
绿箭v19 快速回测 — 批量预测+截面交易
"""
import sys, os, math, time, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import xgboost as xgb
import _paths

T0 = time.time()
print("═══ 绿箭v19 快速回测 ═══")

df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v2.parquet")
with open(_paths.ML_DIR+"/us_sector_etf.json") as f: etf_data = json.load(f)
s2e={'Technology':'XLK','Financial Services':'XLF','Financial':'XLF','Energy':'XLE',
     'Healthcare':'XLV','Industrials':'XLI','Consumer Defensive':'XLP',
     'Consumer Cyclical':'XLY','Utilities':'XLU','Basic Materials':'XLB',
     'Materials':'XLB','Real Estate':'XLRE','Communication Services':'XLC','Semiconductor':'SMH'}
def get_er(s):
    e=s2e.get(s)
    return etf_data[e]['ret5'] if e and e in etf_data else etf_data['SPY']['ret5']
df['sector_etf_ret5']=df['sector'].apply(get_er)
for k in ['SPY','QQQ','IWM']: df[f'{k.lower()}_ret5']=etf_data[k]['ret5']
df['sc']=df['sector'].astype('category').cat.codes.astype(int)
feats=['price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
       'ret1','ret5','ret20','ret60','macd','macd_signal','macd_hist',
       'vol_ratio','ma_bias20','vol5','trend_accel',
       'short_ratio','short_pct','market_cap','pe_trailing','pe_forward','beta',
       'sector_etf_ret5','spy_ret5','qqq_ret5','iwm_ret5','sc']
df=df.dropna(subset=feats+['label_5d_pct']).copy()
n=len(df)

m=xgb.XGBClassifier(, device='cuda')
m.load_model(_paths.US_MODEL_DIR+'/greenshaft_v19_base.json')

# 取最后30%做测试
test_start = int(n * 0.7)
X_test = df[feats].values[test_start:]
y_pct = df['label_5d_pct'].values[test_start:]
print(f"测试样本: {len(X_test):,}")

# 批量预测
proba = m.predict_proba(X_test)
pu5 = proba[:, 4]

# 每2435行=1个"交易日"（近似）
sym_count = df['sym'].nunique()  # 2435
# 每sym_count行算一个截面
periods = len(X_test) // sym_count
print(f"近似交易日: {periods}天")

# 模拟交易
for top_n in [5, 10, 20, 50]:
    eq_rets = []
    wgt_rets = []
    hit5_list=[]
    hit0_list=[]
    
    for p in range(periods):
        start = p * sym_count
        end = min((p+1) * sym_count, len(X_test))
        if end - start < 50:
            continue
        
        period_pu5 = pu5[start:end]
        period_pct = y_pct[start:end]
        
        idx = np.argsort(-period_pu5)[:top_n]
        sel_pct = period_pct[idx]
        
        eq_ret = sel_pct.mean()
        w = period_pu5[idx]
        w = w / w.sum()
        wgt_ret = (sel_pct * w).sum()
        
        eq_rets.append(eq_ret)
        wgt_rets.append(wgt_ret)
        hit5_list.append((sel_pct > 5).mean())
        hit0_list.append((sel_pct > 0).mean())
    
    if not eq_rets:
        continue
    
    eq_rets = np.array(eq_rets)
    # 过滤异常值（>50%收益的极少数）
    eq_rets = eq_rets[eq_rets < 50]
    
    cum_eq = (np.prod(1 + eq_rets/100) - 1) * 100
    avg = eq_rets.mean()
    med = np.median(eq_rets)
    std = eq_rets.std()
    sp = avg / std * math.sqrt(252) if std > 0 else 0
    win = (eq_rets > 0).mean()
    best = eq_rets.max()
    worst = eq_rets.min()
    
    # 回撤
    cum_series = np.cumprod(1 + eq_rets/100)
    peak = np.maximum.accumulate(cum_series)
    dd = (cum_series - peak) / peak
    max_dd = dd.min()
    
    avg_hit5 = np.mean(hit5_list)
    avg_hit0 = np.mean(hit0_list)
    
    # 年度分解（按period估算）
    days_per_year = int(252 / (sym_count / avg) )  # 粗略
    n_years = len(eq_rets) // 252 if len(eq_rets) > 252 else 1
    
    print(f"\n{'='*55}")
    print(f"【每日Top{top_n}】 {len(eq_rets)}笔交易")
    print(f"{'='*55}")
    print(f"  累积收益: {cum_eq:+.2f}%")
    print(f"  单笔均收益: {avg:+.4f}%")
    print(f"  单笔中位: {med:+.4f}%")
    print(f"  胜率(涨>0): {win:.1%}")
    print(f"  夏普(年化): {sp:.3f}")
    print(f"  最大回撤: {max_dd*100:.1f}%")
    print(f"  涨>5%命中: {avg_hit5:.1%}")
    print(f"  涨>0平均: {avg_hit0:.1%}")

print(f"\n✅ 回测完成! ({time.time()-T0:.0f}s)")
