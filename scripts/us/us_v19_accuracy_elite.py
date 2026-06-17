"""绿箭v19 准确率分析: 涨>0胜率 + 标普100表现"""
import sys, os, json, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings; warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd, numpy as np
import xgboost as xgb
import _paths

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

df=df.dropna(subset=feats+['label_5d_5class','label_5d_pct']).copy()
X=df[feats].values; y5=df['label_5d_5class'].values; pct=df['label_5d_pct'].values; syms=df['sym'].values
n=len(df)

m=xgb.XGBClassifier(, device='cuda')
m.load_model(_paths.US_MODEL_DIR+'/greenshaft_v19_base.json')

train_end=int(n*0.85)
raw=m.predict_proba(X[train_end:])
y_act=y5[train_end:]; pct_act=pct[train_end:]; syms_act=syms[train_end:]
pu5=raw[:,4]; pu0=raw[:,0]

print("="*75)
print("【一、涨>5%区间 → 实际涨>0胜率】")
print("="*75)
top10_th=np.percentile(pu5, 90)
m10=pu5>=top10_th
n10=m10.sum()
hit_up5=(y_act[m10]==4).mean()
hit_up0=(pct_act[m10]>0).mean()
avg_r=pct_act[m10].mean()
print(f"  Top10% (n={n10})")
print(f"    涨>5%命中: {hit_up5:.1%}")
print(f"    涨>0 胜率: {hit_up0:.1%}")
print(f"    5天均收益: {avg_r:+.2f}%")
print(f"    夏普: {avg_r/pct_act[m10].std()*math.sqrt(252/5):.3f}")

top5_th=np.percentile(pu5, 95)
m5=pu5>=top5_th
n5=m5.sum()
hit5_up5=(y_act[m5]==4).mean()
hit5_up0=(pct_act[m5]>0).mean()
avg5=pct_act[m5].mean()
print(f"\n  Top5% (n={n5})")
print(f"    涨>5%命中: {hit5_up5:.1%}")
print(f"    涨>0 胜率: {hit5_up0:.1%}")
print(f"    5天均收益: {avg5:+.2f}%")
print(f"    夏普: {avg5/pct_act[m5].std()*math.sqrt(252/5):.3f}")

# 按概率档位拆分
print(f"\n【涨>5%概率分档 → 涨>0胜率】")
print(f"{'档位':>10} {'n':>7} {'涨>5%':>8} {'涨>0%':>8} {'均收益':>8} {'夏普':>7}")
print("-"*55)
for lo,hi in [(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),
              (0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]:
    mc=(pu5>=lo)&(pu5<hi)
    nm=mc.sum()
    if nm<30: continue
    p_up5=(y_act[mc]==4).mean()
    p_up0=(pct_act[mc]>0).mean()
    avgr=pct_act[mc].mean()
    sp=avgr/pct_act[mc].std()*math.sqrt(252/5) if pct_act[mc].std()>0 else 0
    print(f"{lo:.0%}-{hi:.0%}: {nm:>7} {p_up5:>7.1%} {p_up0:>7.1%} {avgr:>+7.2f}% {sp:>6.3f}")

print(f"\n{'='*75}")
print("【二、跌>5%区间 → 实际跌<0胜率】")
print(f"{'='*75}")
for lo,hi in [(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),
              (0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]:
    mc=(pu0>=lo)&(pu0<hi)
    nm=mc.sum()
    if nm<30: continue
    p_dn5=(y_act[mc]==0).mean()
    p_dn0=(pct_act[mc]<0).mean()
    avgr=pct_act[mc].mean()
    sp=avgr/pct_act[mc].std()*math.sqrt(252/5) if pct_act[mc].std()>0 else 0
    print(f"{lo:.0%}-{hi:.0%}: n={nm:>7} 跌>5%={p_dn5:>7.1%} 跌>0%={p_dn0:>7.1%} 均收益={avgr:>+7.2f}% 夏普={sp:>6.3f}")

print(f"\n{'='*75}")
print("【三、标普100成分股表现】")
print(f"{'='*75}")

# 拉标普100成分股
try:
    sp100 = yf.Ticker("^OEX").history(period='1d')
    # 或用已知列表
    sp_tickers = ['AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','BRK-B','JPM','V',
                  'UNH','XOM','LLY','PG','MA','HD','CVX','MRK','ABBV','PEP',
                  'KO','BAC','WMT','COST','AVGO','MCD','TMO','ABT','CRM','NFLX',
                  'ADBE','WFC','CSCO','ACN','CMCSA','NKE','PFE','ORCL','DIS','PM',
                  'QCOM','TXN','RTX','NEE','HON','INTC','IBM','AMGN','MDT','SPGI',
                  'UPS','CAT','LOW','GE','MS','GS','DHR','ISRG','BLK','PLD',
                  'AMAT','AMD','BA','C','CB','DE','FIS','GM','JPM','LMT',
                  'MMM','MO','BAX','APD','BK','CAT','DOW','DVN','EBAY','EMR',
                  'F','FCX','FDX','GD','GM','GPS','HAL','HON','HPQ','IBM',
                  'IP','JCI','JNPR','KMB','LNC','LYB','MAS','MHK','NOC','NUE']
    # 筛选在数据中的
    test_syms = set(syms_act)
    sp_found = [t for t in sp_tickers if t in test_syms]
    print(f"  标普100中找到: {len(sp_found)}只")
    
    sp_mask = np.isin(syms_act, sp_found)
    sp_pu5 = pu5[sp_mask]
    sp_y = y_act[sp_mask]
    sp_pct = pct_act[sp_mask]
    sp_s = syms_act[sp_mask]
    print(f"  测试集中标普100样本: {len(sp_pu5)}条")
    
    # 标普100整体表现
    print(f"\n  标普100整体:")
    print(f"    涨>5%自然占比: {(sp_y==4).mean():.1%}")
    print(f"    涨>0胜率: {(sp_pct>0).mean():.1%}")
    print(f"    均收益: {sp_pct.mean():+.2f}%")
    
    # 标普100中绿箭Top10%表现
    sp_th = np.percentile(sp_pu5, 90)
    sp_m10 = sp_pu5 >= sp_th
    n_sp10 = sp_m10.sum()
    if n_sp10 > 5:
        print(f"\n  标普100 绿箭Top10% (n={n_sp10}):")
        print(f"    涨>5%命中: {(sp_y[sp_m10]==4).mean():.1%}")
        print(f"    涨>0胜率: {(sp_pct[sp_m10]>0).mean():.1%}")
        print(f"    均收益: {sp_pct[sp_m10].mean():+.2f}%")
        sp_syms_show = sp_s[sp_m10]
        print(f"    股票: {','.join(sp_syms_show[:15])}")
    
    # 标普100中按绿箭概率排序
    order = np.argsort(-sp_pu5)
    print(f"\n  标普100 绿箭看好Top15:")
    print(f"{'#':>3} {'代码':>8} {'涨>5%概率':>10} {'实际收益':>10}")
    for i in range(min(15, len(order))):
        idx = order[i]
        print(f"{i+1:>3} {sp_s[idx]:>8} {sp_pu5[idx]:>9.1%} {sp_pct[idx]:>+9.2f}%")
    
    # 标普100中绿箭看跌Top10
    sp_order = np.argsort(-pu0[sp_mask])
    print(f"\n  标普100 绿箭看跌Top10:")
    print(f"{'#':>3} {'代码':>8} {'跌>5%概率':>10} {'实际收益':>10}")
    sp_s0 = sp_s[sp_mask]
    sp_p0 = pu0[sp_mask]
    sp_pct0 = sp_pct
    for i in range(min(10, len(sp_order))):
        idx = sp_order[i]
        print(f"{i+1:>3} {sp_s0[idx]:>8} {sp_p0[idx]:>9.1%} {sp_pct0[idx]:>+9.2f}%")
    
except Exception as e:
    print(f"  标普100获取失败: {e}")
    # 用market_cap最大的100只代替
    print(f"\n  [代替方案] 按市值筛选Top100精英池")
    if 'market_cap' in df.columns:
        latest_df = df.dropna(subset=feats).drop_duplicates(subset='sym', keep='last')
        top100 = latest_df.nlargest(100, 'market_cap')['sym'].values
        test_set = set(syms_act)
        elite = [s for s in top100 if s in test_set]
        elite_mask = np.isin(syms_act, elite)
        
        print(f"  精英池(市值Top100)测试集样本: {elite_mask.sum()}条")
        el_pu5 = pu5[elite_mask]
        el_y = y_act[elite_mask]
        el_pct = pct_act[elite_mask]
        el_s = syms_act[elite_mask]
        
        # 精英池绿箭Top表现
        el_th = np.percentile(el_pu5, 90)
        el_m10 = el_pu5 >= el_th
        n_el10 = el_m10.sum()
        if n_el10 > 3:
            print(f"\n  精英池 绿箭Top10% (n={n_el10}):")
            print(f"    涨>5%命中: {(el_y[el_m10]==4).mean():.1%}")
            print(f"    涨>0胜率: {(el_pct[el_m10]>0).mean():.1%}")
            print(f"    均收益: {el_pct[el_m10].mean():+.2f}%")
        
        order = np.argsort(-el_pu5)
        print(f"\n  精英池Top15:")
        for i in range(min(15, len(order))):
            idx = order[i]
            print(f"  {i+1:>2}. {el_s[idx]:>8} 涨>5%={el_pu5[idx]:>7.1%} 实际={el_pct[idx]:>+7.2f}%")

print()
