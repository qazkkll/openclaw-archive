#!/usr/bin/env python3
"""
us_7.1_s3_train_v2.py — V7.1 v2训练
基本面+技术指标(30 feat) + 1000轮深树 + Isotonic校准 + 严验证
输出: /home/hermes/.hermes/openclaw-project/data/models/us_xgb_v71_v2.*
"""
import sys, os, json, pickle, warnings, time, math
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

T0 = time.time()
BASE = '/home/hermes/.hermes/openclaw-archive'
ML_DIR = f'{BASE}/ml'
MD = f'{BASE}/data/models'
VER = 'us_xgb_v71_v2'

print('=' * 60)
print(f'{VER} -- 基本面+技术指标+1000轮深树')

# ===== 1. 加载 =====
print('\n[1/6] 加载特征...')
df = pd.read_parquet(f'{ML_DIR}/us_ml_feats_v3_dated.parquet')
print(f'  {len(df):,}行, {df.sym.nunique()}只, {df.date.min()}~{df.date.max()}')

# ===== 2. 特征 =====
print('\n[2/6] 特征配置...')
TECH = ['ma5','ma10','ma20','ma60','rsi14','p52','ret1','ret5','ret20','ret60',
        'macd','macd_signal','macd_hist','vol20','vol_ratio','ma_bias20','trend_accel']
FUND = ['market_cap','pb','roe','rev_growth','profit_growth','beta',
        'debt_equity','gross_margin','profit_margin','book_value',
        'pe_trailing','pe_forward','div_yield']
FEATS = [f for f in TECH + FUND if f in df.columns]
print(f'  特征数: {len(FEATS)}')
for f in FEATS:
    if df[f].dtype == 'object':
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

# ===== 3. 标签 =====
print('\n[3/6] 标签生成...')
df = df.sort_values(['sym','date']).reset_index(drop=True)
df['price_next_5'] = df.groupby('sym')['price'].shift(-5)
df['ret_5d'] = df['price_next_5'] / df['price'] - 1
df['label'] = (df['ret_5d'] > 0.05).astype(int)
v = df['label'].notna()
print(f'  有效: {v.sum():,}行, 正样本: {df[v]["label"].mean():.2%}')

# ===== 4. 切分 =====
print('\n[4/6] 时间切分...')
tr = df[(df['date'] < '2025-01-01') & df['label'].notna()].dropna(subset=FEATS + ['label'])
va = df[(df['date'] >= '2025-01-01') & (df['date'] < '2026-01-01') & df['label'].notna()].dropna(subset=FEATS + ['label'])
te = df[(df['date'] >= '2026-01-01') & df['label'].notna()].dropna(subset=FEATS + ['label'])
for n, d in [('训练', tr), ('验证', va), ('测试', te)]:
    print(f'  {n}: {len(d):,}行, 正样本{d.label.mean():.2%}')

Xtr = np.nan_to_num(tr[FEATS].values.astype(np.float32), nan=0.0); ytr = tr['label'].values.astype(float)
Xva = np.nan_to_num(va[FEATS].values.astype(np.float32), nan=0.0); yva = va['label'].values.astype(float)
Xte = np.nan_to_num(te[FEATS].values.astype(np.float32), nan=0.0); yte = te['label'].values.astype(float)

# ===== 5. 训练 =====
spw = max(1, (1 - ytr.mean()) / max(ytr.mean(), 0.01))
print(f'\n[5/6] XGBoost训练 (scale_pos_weight={spw:.1f})...')

dt = xgb.DMatrix(Xtr, label=ytr, feature_names=FEATS)
dv = xgb.DMatrix(Xva, label=yva, feature_names=FEATS)

model = xgb.train({
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 8, 'eta': 0.03, 'subsample': 0.8,
    'colsample_bytree': 0.7, 'min_child_weight': 3,
    'scale_pos_weight': spw, 'seed': 42,, 'device':'cuda'
}, dt, num_boost_round=1000,
   evals=[(dt, 'tr'), (dv, 'va')],
   early_stopping_rounds=200, verbose_eval=0)

print(f'  最佳迭代: {model.best_iteration}, AUC={model.best_score:.4f}')

# ===== 6. 评估 =====
print('\n[6/6] 评估+验证...')

pte = xgb.DMatrix(Xte, label=yte, feature_names=FEATS)
pv = model.predict(dv); pt = model.predict(pte)
va_auc = roc_auc_score(yva, pv); te_auc = roc_auc_score(yte, pt)
print(f'  验证AUC(raw): {va_auc:.4f}  测试AUC(raw): {te_auc:.4f}')

# Isotonic校准
cal = IsotonicRegression(out_of_bounds='clip')
cal.fit(pv, yva); pvc = cal.transform(pv); ptc = cal.transform(pt)
va_c = roc_auc_score(yva, pvc); te_c = roc_auc_score(yte, ptc)
print(f'  验证AUC(iso): {va_c:.4f}  测试AUC(iso): {te_c:.4f}')

# 校准矩阵
print('\n校准检查:')
for lb in [x / 10 for x in range(6)]:
    msk = (ptc >= lb) & (ptc < lb + 0.1)
    if msk.sum() > 50:
        actual = yte[msk].mean()
        print(f'  {lb:.0%}-{lb+0.1:.0%} (n={msk.sum():,}): pred={lb+0.05:.0%} actual={actual:.1%}')

# 特征重要性
imp = model.get_score('gain')
print('\n特征重要性 (gain):')
for fn, wgt in sorted(imp.items(), key=lambda x: -x[1])[:15]:
    print(f'  {fn:20s} {int(wgt):>10,}')

# ===== 核心验证（全体数据） =====
print('\n' + '=' * 40)
print('核心验证')
print('=' * 40)

da = df.dropna(subset=FEATS + ['label']).copy()
Xa = np.nan_to_num(da[FEATS].values.astype(np.float32), nan=0.0)
da['prob'] = cal.transform(model.predict(xgb.DMatrix(Xa, feature_names=FEATS)))

# 验证A：概率>50%
print('\n[验证A] 预测概率>50%的样本:')
for th in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]:
    sub = da[da['prob'] > th]
    if len(sub) > 20:
        r = sub['ret_5d'].clip(-0.5, 0.5)
        loss = (sub['ret_5d'] < -0.01).mean()
        loss3 = (sub['ret_5d'] < -0.03).mean()
        print(f'  >{th:.0%} (n={len(sub):,}): 命中={sub.label.mean():.1%} '
              f'均收益={r.mean():+.2%} 亏损率={loss:.1%} 亏>3%={loss3:.1%}')

# 验证B：截面TopN（不复利，统计每期命中率）
print('\n[验证B] 截面选TopN命中率 (2025+):')
btd = sorted(da['date'].unique())
btd = [d for d in btd if str(d) >= '2025-01-02']

for tn in [5, 10, 15, 20]:
    ho = []; lo = []; ro = []; nd = 0
    for d in btd:
        dy = da[da['date'] == d]
        if len(dy) < 30:
            continue
        pk = dy.nlargest(tn, 'prob')
        r = pk['ret_5d'].clip(-0.5, 1.0).values
        ho.append((r > 0.05).mean())
        lo.append((r < 0).mean())
        ro.append(r.mean())
        nd += 1
    ha = np.mean(ho); la = np.mean(lo); ra = np.mean(ro)
    print(f'  d{tn:>2} (n={nd:,}): 命中={ha:.1%} 亏损率={la:.1%} 均值={ra:+.2%}')

# 基准
print('\n基准(全市场均值等权):')
br = []
for d in btd:
    dy = da[da['date'] == d]
    if len(dy) > 50:
        br.append(dy['ret_5d'].clip(-0.5, 1.0).mean())
ba = np.array(br)
print(f'  均值={ba.mean():+.2%} 涨>5%={(ba>0.05).mean():.1%} 亏损={(ba<0).mean():.1%}')

# ===== 持仓评分 =====
last_date = da['date'].max()
ld = da[da['date'] == last_date].sort_values('prob', ascending=False)
print(f'\n持仓评分 ({last_date}):')
for code, sym in [('NOK','NOK'),('NVDA','NVDA'),('GNRC','GNRC'),('ON','ON'),('QCOM','QCOM')]:
    row = ld[ld['sym'] == sym]
    if len(row) > 0:
        r = row.iloc[0]
        rank = ld['sym'].eq(sym).values.argmax() + 1
        print(f'  {code:>6} ${r["price"]:>7.2f} V7.2={r["prob"]:>6.1%} rank={rank}')

print('\nTop20 ($5+):')
for i, (_, r) in enumerate(ld[ld['price'] >= 5].head(20).iterrows(), 1):
    print(f'  {i:>2} {r["sym"]:>7} ${r["price"]:>7.2f} {r["prob"]:>6.1%}')

# ===== 保存 =====
print('\n保存模型...')
model.save_model(f'{MD}/{VER}.json')
pickle.dump(cal, open(f'{MD}/{VER}_calibrator.pkl', 'wb'))
json.dump({
    'version': VER, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'val_auc': round(va_c, 4), 'test_auc': round(te_c, 4),
    'features': FEATS, 'n_features': len(FEATS),
    'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
    'pos_rate': round(ytr.mean(), 4),
    'best_iteration': int(model.best_iteration),
    'param_depth': 8, 'param_rounds': 1000,
    'calibration': 'Isotonic',
    'feature_importance': {fn: int(wgt) for fn, wgt in sorted(imp.items(), key=lambda x: -x[1])[:20]},
}, open(f'{MD}/{VER}_report.json', 'w'), indent=2)
print(f'  => {VER}.json + _calibrator.pkl + _report.json')
print(f'\n总耗时: {time.time() - T0:.0f}s')
print('=' * 60)
