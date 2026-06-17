#!/usr/bin/env python3
"""
us_7.1_s6_train_v71v19.py — V7.1池子 + V19特征版训练
用110万行(2413只$5+) + 28个V19特征训练XGBoost
包含: 5分类, 行业ETF收益, sector编码, 基础基本面
"""
import sys, os, json, pickle, time, warnings, math
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

BASE='/home/hermes/.hermes/openclaw-archive'; ML_DIR=f'{BASE}/ml'; MD=f'{BASE}/data/models'
FEAT_FILE=f'{ML_DIR}/us_ml_feats_v71_v19.parquet'
VER='us_xgb_v71_v19'
T0=time.time()

print('='*60)
print(f'{VER} — V7.1池子 + V19特征版')
print('='*60)

# === 1. 数据 ===
print('\n[1/5] 加载数据...')
df=pd.read_parquet(FEAT_FILE)
print(f'  {len(df):,}行, {df.sym.nunique()}只, {df.date.nunique()}天')

# V19特征列表
FEATS=[
    'price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
    'ret1','ret5','ret20','ret60','macd','macd_signal','macd_hist',
    'vol_ratio','ma_bias20','vol5','trend_accel',
    'short_ratio','short_pct','market_cap',
    'sector_etf_ret5','spy_ret5','qqq_ret5','iwm_ret5','sc',
]

# 数据质量清理
df=df.replace([np.inf,-np.inf],np.nan)
df=df.dropna(subset=FEATS+['label_5d_5class','label_5d_pct'])
for f in FEATS:
    df[f]=pd.to_numeric(df[f],errors='coerce').fillna(0)
    df[f]=df[f].clip(-1e6,1e6)

df=df.sort_values(['sym','date']).reset_index(drop=True)
dates=sorted(df['date'].unique())
print(f'  有效: {len(df):,}行, 日期{df.date.min()}~{df.date.max()}')

# === 2. 5分类标签分布 ===
print('\n[2/5] 标签分布...')
for c in range(5):
    n=(df['label_5d_5class']==c).sum()
    print(f'  {c}: {n:,} ({n/len(df)*100:.1f}%)')

# === 3. 分训练/校准 ===
print('\n[3/5] 切分数据...')
split_idx=int(len(dates)*0.7)
train_dates=dates[:split_idx]
calib_dates=dates[split_idx:]
tr=df[df['date'].isin(train_dates)]
ca=df[df['date'].isin(calib_dates)]
print(f'  训练: {len(tr):,}行 ({tr.date.nunique()}天)')
print(f'  校准: {len(ca):,}行 ({ca.date.nunique()}天)')

# === 4. 训练 ===
print('\n[4/5] XGBoost多分类训练...')
spw=np.array([(tr['label_5d_5class']==c).sum() for c in range(5)])
spw=spw.min()/spw  # 加权
print(f'  类别权重: {[f"{w:.2f}" for w in spw]}')

model=xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=5,
    eval_metric='mlogloss', early_stopping_rounds=30,
    random_state=42, n_jobs=-1, verbosity=0, device='cuda')
sw=np.array([spw[int(y)] for y in tr['label_5d_5class'].values])
sw=sw*np.linspace(0.3,1.0,len(sw))  # 时间衰减
model.fit(tr[FEATS].values, tr['label_5d_5class'].values, sample_weight=sw,
          eval_set=[(tr[FEATS].values, tr['label_5d_5class'].values),
                    (ca[FEATS].values, ca['label_5d_5class'].values)],
          verbose=100)
print(f'  完成! best_iter={model.best_iteration}')

# === 5. Platt校准（涨>5%概率） ===
print('\n[5/5] Platt校准...')
raw_probs=model.predict_proba(ca[FEATS].values)
p_up5=raw_probs[:,4].reshape(-1,1)
calib_binary=(ca['label_5d_pct']>5).astype(int).values
calibrator=LogisticRegression(C=1.0, solver='lbfgs')
calibrator.fit(p_up5, calib_binary)
print(f'  系数: a={calibrator.coef_[0][0]:.4f}, b={calibrator.intercept_[0]:.4f}')

# 校准后预测整个数据集
all_raw=model.predict_proba(df[FEATS].values)[:,4].reshape(-1,1)
df['prob_up5']=calibrator.predict_proba(all_raw)[:,1]

# 校准矩阵
all_actuals=df['label_5d_pct'].values
print('\n校准检查:')
total_bias=0; nb=0
for i in range(5):
    lo,hi=i*0.1,(i+1)*0.1
    mask=(df['prob_up5']>=lo)&(df['prob_up5']<hi)
    if mask.sum()<30: continue
    pred_avg=df[mask]['prob_up5'].mean()*100
    actual=(all_actuals[mask]>5).mean()*100
    bias=pred_avg-actual
    total_bias+=abs(bias); nb+=1
    print(f'  {lo:.0%}-{hi:.0%} (n={mask.sum():,}): pred={pred_avg:.1f}% actual={actual:.1f}% bias={bias:+.1f}%')
print(f'  均绝对偏差: {total_bias/nb:.1f}%' if nb else '  N/A')

# === 核心验证 ===
print('\n'+'='*40)
print('核心验证')
print('='*40)

# A: 概率阈值命中率
print('\n[验证A] 概率阈值表现 (2025+):')
test=df[df['date']>=dates[int(len(dates)*0.7)]]  # 后30%
for th in [0.5,0.4,0.35,0.3,0.25]:
    sub=test[test['prob_up5']>th]
    if len(sub)>20:
        r=sub['label_5d_pct'].clip(-0.5,0.5)
        hit=(sub['label_5d_pct']>5).mean()
        loss=(sub['label_5d_pct']<-1).mean()
        print(f'  >{th:.0%} (n={len(sub):,}): 命中={hit:.1%} 均值={r.mean():+.2%} 亏损率={loss:.1%}')

# B: 截面选TopN
print('\n[验证B] 截面选TopN (2025+):')
test_dates=sorted(test['date'].unique())
for tn in [5,10,15,20]:
    hits=[]; losses=[]
    for d in test_dates:
        dy=test[test['date']==d]
        if len(dy)<30: continue
        pk=dy.nlargest(tn,'prob_up5')
        r=pk['label_5d_pct'].clip(-0.3,0.3).values
        hits.append((r>0.05).mean())
        losses.append((r<0).mean())
    print(f'  d{tn:>2}: 命中={np.mean(hits):.1%} 亏损率={np.mean(losses):.1%}')

# C: 特征重要性
print('\n特征重要性:')
imps=sorted(zip(FEATS,model.feature_importances_),key=lambda x:-x[1])
for f,imp in imps[:15]:
    print(f'  {f:>25s} {imp:.4f}')

# 持仓评分
last_date=dates[-1]
ld=df[df['date']==last_date].sort_values('prob_up5',ascending=False)
print(f'\n持仓评分 ({last_date}):')
for sym in ['NVDA','NOK','GNRC','ON','QCOM']:
    row=ld[ld['sym']==sym]
    if len(row)>0:
        r=row.iloc[0]; rank=ld['prob_up5'].ge(r['prob_up5']).sum()
        print(f'  {sym:>6} ${r["price"]:>7.2f} prob={r["prob_up5"]:>6.1%} rank={rank}')
    else:
        print(f'  {sym:>6}: 不在最新日')

print('\nTop15:')
for i,(_,r) in enumerate(ld.head(15).iterrows(),1):
    print(f'  {i:>2} {r["sym"]:>7} ${r["price"]:>7.2f} {r["prob_up5"]:>6.1%}')

# === 保存 ===
print('\n保存模型...')
model.save_model(f'{MD}/{VER}.json')
pickle.dump(calibrator,open(f'{MD}/{VER}_calibrator.pkl','wb'))

# 获取校准偏差
calib_err=round(total_bias/nb,2) if nb else 99
json.dump({
    'version':VER,'timestamp':'2026-06-11 10:15',
    'features':FEATS,'n_features':len(FEATS),
    'n_train':len(tr),'n_calib':len(ca),
    'base_data':'V3_us_ml_feats_v3_dated + V19_etf_features',
    'stock_universe':'2413只$5+股票(含46只SP100大盘)',
    'date_range':f'{df.date.min()}~{df.date.max()}',
    'n_estimators':int(model.best_iteration or 500),
    'learning_rate':0.08,'max_depth':6,
    'obj':'multi:softprob','num_class':5,'calib':'LogisticRegression(Platt)',
    'calib_a':round(calibrator.coef_[0][0],4),
    'calib_b':round(calibrator.intercept_[0],4),
    'calib_mean_abs_bias_pct':calib_err,
},open(f'{MD}/{VER}_report.json','w'),indent=2)
print(f'  => {VER}.json + _calibrator.pkl + _report.json')
print(f'\n总耗时: {time.time()-T0:.0f}s')
print('='*60)
