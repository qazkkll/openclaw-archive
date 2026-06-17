"""绿箭v19 完整5档命中率分析"""
import sys, os, json, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings; warnings.filterwarnings('ignore')
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
X=df[feats].values; y5=df['label_5d_5class'].values; ap=df['label_5d_pct'].values
n=len(df)

m=xgb.XGBClassifier(, device='cuda')
m.load_model(_paths.US_MODEL_DIR+'/greenshaft_v19_base.json')
import joblib
cal=joblib.load(_paths.US_MODEL_DIR+'/greenshaft_v19_calib.pkl')
train_end=int(n*0.85)
raw=m.predict_proba(X[train_end:])
cal_pu5=cal.predict(raw[:,4])
y_act=y5[train_end:]; pct_act=ap[train_end:]
pu5=raw[:,4]

print("="*70)
print("绿箭v19 完整5档命中率分析")
print("="*70)

# 1. 每个类别Top10%的命中
for cl,cn in [(0,'跌>5%'),(1,'跌2-5%'),(2,'±2%'),(3,'涨2-5%'),(4,'涨>5%')]:
    prob=raw[:,cl]
    th=np.percentile(prob, 90)
    m10=prob>=th
    n10=m10.sum()
    if n10==0: continue
    hit=(y_act[m10]==cl).mean()
    avg_r=pct_act[m10].mean()
    sp=avg_r/pct_act[m10].std()*math.sqrt(252/5) if pct_act[m10].std()>0 else 0
    print(f"\n{cn}")
    print(f"  自然占比: {(y_act==cl).mean()*100:.1f}%")
    print(f"  Top10%命中: {hit:.1%} (n={n10})")
    print(f"  5天均收益: {avg_r:+.2f}%")
    print(f"  Top10%夏普: {sp:.3f}")

# 2. 涨>5%排序分组
print("\n" + "="*70)
print("涨>5%概率排序分组")
print(f"{'分组':>10} {'n':>6} {'涨>5%':>8} {'均收益':>8} {'胜率':>6} {'夏普':>6}")
print("-"*50)
for pct,lb in [(5,'Top5%'),(10,'Top10%'),(25,'Top25%'),(50,'Bottom50%')]:
    th=np.percentile(pu5, 100-pct)
    mx=pu5>=th
    nm=mx.sum()
    if nm<5: continue
    hit=(y_act[mx]==4).mean()
    avg_ret=pct_act[mx].mean()
    wr=(pct_act[mx]>0).mean()
    sp=avg_ret/pct_act[mx].std()*math.sqrt(252/5) if pct_act[mx].std()>0 else 0
    print(f"{lb:>10} {nm:>6} {hit:>7.1%} {avg_ret:>+7.2f}% {wr:>5.1%} {sp:>6.3f}")

# 3. 跌>5%排序分组
print("\n" + "="*70)
print("跌>5%概率排序分组")
print(f"{'分组':>10} {'n':>6} {'跌>5%':>8} {'均收益':>8} {'胜率':>6} {'夏普':>6}")
print("-"*50)
pu0=raw[:,0]
for pct,lb in [(5,'Top5%'),(10,'Top10%'),(25,'Top25%')]:
    th=np.percentile(pu0, 100-pct)
    mx=pu0>=th
    nm=mx.sum()
    if nm<5: continue
    hit=(y_act[mx]==0).mean()
    avg_ret=pct_act[mx].mean()
    wr=(pct_act[mx]>0).mean()
    sp=avg_ret/pct_act[mx].std()*math.sqrt(252/5) if pct_act[mx].std()>0 else 0
    print(f"{lb:>10} {nm:>6} {hit:>7.1%} {avg_ret:>+7.2f}% {wr:>5.1%} {sp:>6.3f}")

# 4. 混淆矩阵
print("\n" + "="*70)
print("预测vs实际混淆矩阵 (argmax)")
pred=raw.argmax(axis=1)
print(f"{'':>10}", end='')
for j in range(5):
    print(f"{'预'+str(j):>8}", end='')
print("  合计")
for i in range(5):
    print(f"{'实'+str(i):>10}", end='')
    for j in range(5):
        cnt=((pred==j)&(y_act==i)).sum()
        print(f"{cnt:>8}", end='')
    print(f"{(y_act==i).sum():>8}")

print("\n" + "="*70)
for i,nm in enumerate(['跌>5%','跌2-5%','±2%','涨2-5%','涨>5%']):
    tp=((pred==i)&(y_act==i)).sum()
    fp=(pred==i).sum()-tp
    fn=(y_act==i).sum()-tp
    prec=tp/(tp+fp)*100 if (tp+fp)>0 else 0
    rec=tp/(tp+fn)*100 if (tp+fn)>0 else 0
    f1=2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    print(f"{nm:>8}: 精度={prec:.1f}% 召回={rec:.1f}% F1={f1:.1f}")

# 5. 校准后涨>5% 分桶详细
print("\n" + "="*70)
print("涨>5% 校准后分桶")
print(f"{'区间':>10} {'n':>6} {'预测':>8} {'实际涨>5%':>10} {'均收益':>8} {'胜率':>6}")
print("-"*55)
for lo,hi in [(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),
              (0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]:
    mc=(cal_pu5>=lo)&(cal_pu5<hi)
    nm=mc.sum()
    if nm<10: continue
    act=(y_act[mc]==4).mean()
    avg_p=cal_pu5[mc].mean()
    avg_r=pct_act[mc].mean()
    wr=(pct_act[mc]>0).mean()
    print(f"{lo:.0%}-{hi:.0%}: {nm:>6} {avg_p:>7.1%} {act:>9.1%} {avg_r:>+7.2f}% {wr:>5.1%}")

# 6. 关键一问：Bottom50%的涨>5%概率
print("\n" + "="*70)
print("涨>5%概率<50%的股票实际表现")
bot=pu5<0.5
print(f"  样本量: {bot.sum()}")
print(f"  实际涨>5%: {(y_act[bot]==4).mean():.1%}")
print(f"  实际平盘: {(y_act[bot]==2).mean():.1%}")
print(f"  实际跌>5%: {(y_act[bot]==0).mean():.1%}")
print(f"  平均收益: {pct_act[bot].mean():+.2f}%")
print(f"  胜率(5天涨): {(pct_act[bot]>0).mean():.1%}")

top_rev=pu5>=0.5
print(f"\n涨>5%概率>=50%的股票:")
print(f"  样本量: {top_rev.sum()}")
print(f"  实际涨>5%: {(y_act[top_rev]==4).mean():.1%}")
print(f"  实际平盘: {(y_act[top_rev]==2).mean():.1%}")
print(f"  实际跌>5%: {(y_act[top_rev]==0).mean():.1%}")
print(f"  平均收益: {pct_act[top_rev].mean():+.2f}%")
print(f"  胜率(5天涨): {(pct_act[top_rev]>0).mean():.1%}")

print("\n" + "="*70)
print("对比: 如果你买所有股票(不选)")
print(f"  平均收益: {pct_act.mean():+.2f}%")
print(f"  涨>5%比例: {(y_act==4).mean():.1%}")
print(f"  胜率: {(pct_act>0).mean():.1%}")
print(f"  夏普: {pct_act.mean()/pct_act.std()*math.sqrt(252/5):.3f}")
