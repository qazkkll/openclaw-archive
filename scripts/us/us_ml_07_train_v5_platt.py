import sys, json, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

INPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v5.parquet'
MODEL_DIR = '/home/hermes/.hermes/openclaw-project/data/models/us'

print("[v5 Platt 校准对比] XGBoost + LightGBM", flush=True)
print("1. 加载数据...", flush=True)
df = pd.read_parquet(INPUT).dropna(subset=['fwd_5d_ret'])
y = (df['fwd_5d_ret'] > 0.05).astype(int).values
print(f"  {len(df):,}行  正样本率: {y.mean()*100:.2f}%", flush=True)

excl = {'ticker','date','label','fwd_5d_ret'}
feats = [c for c in df.columns if c not in excl]
print(f"  特征({len(feats)}): {feats}", flush=True)

X = np.nan_to_num(df[feats].values.astype(np.float32), nan=0.0)
del df

n = len(X)
t_end, c_end = int(n*0.75), int(n*0.85)
Xt, Xc, Xv = X[:t_end], X[t_end:c_end], X[c_end:]
yt, yc, yv = y[:t_end], y[t_end:c_end], y[c_end:]
print(f"  训练:{len(Xt):,} 校准:{len(Xc):,} 测试:{len(Xv):,}", flush=True)
del X, y

from sklearn.metrics import roc_auc_score
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

T0 = time.time()

# ======== 1. XGBoost ========
print("2. XGBoost (md6, lr0.05, 400轮)...", flush=True)
import xgboost as xgb

print("  DMatrix构建...", flush=True)
dt = xgb.DMatrix(Xt, yt)
dc = xgb.DMatrix(Xc, yc)
dv = xgb.DMatrix(Xv, yv)

params = {'objective':'binary:logistic','eval_metric':'logloss',
          'tree_method':'hist','device':'cuda','max_depth':6,
          'learning_rate':0.05,'subsample':0.8,'colsample_bytree':0.8,'random_state':42}
print("  训练中...", flush=True)
t0 = time.time()
model = xgb.train(params, dt, num_boost_round=400,
                  evals=[(dt,'train'),(dc,'val')],
                  early_stopping_rounds=15, verbose_eval=50)

print(f"  raw score 预测...", flush=True)
rc = model.predict(dc, output_margin=True)
rv = model.predict(dv, output_margin=True)

# Platt
pm = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
pm.fit(rc.reshape(-1,1), yc)
yp = expit(pm.coef_[0][0]*rv + pm.intercept_[0])
auc_p = roc_auc_score(yv, yp)
nu_p = len(set(np.round(yp, 4)))

# IR对比
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(rc, yc)
yi = ir.transform(rv)
auc_i = roc_auc_score(yv, yi)
nu_i = len(set(np.round(yi, 4)))

bias = (yp.mean() - yv.mean())*100
print(f"  XGB 结果: AUC_Platt={auc_p:.4f} AUC_IR={auc_i:.4f} | 偏{bias:+.2f}% | 离散P={nu_p} IR={nu_i} | 耗时{time.time()-t0:.0f}s", flush=True)

xgb_slope, xgb_int = float(pm.coef_[0][0]), float(pm.intercept_[0])
del model, dt, dc, dv, pm, ir
import gc; gc.collect()

# ======== 2. LightGBM ========
print("3. LightGBM (md6, lr0.05, 400轮)...", flush=True)
try:
    import lightgbm as lgb
    
    # CPU fallback 
    lgb_params = {'objective':'binary','metric':'auc','boosting_type':'gbdt',
                  'max_depth':6,'learning_rate':0.05,'subsample':0.8,
                  'feature_fraction':0.8,'random_state':42,'verbosity':-1}
    
    # Try GPU first
    for dev in ['gpu', 'cpu']:
        try:
            lgb_params['device'] = dev
            lt = lgb.Dataset(Xt, yt)
            t0 = time.time()
            print(f"  device={dev}...", flush=True)
            model = lgb.train(lgb_params, lt, num_boost_round=400,
                              valid_sets=[lt], callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)])
            break
        except Exception as e:
            print(f"  {dev} failed: {e}", flush=True)
            continue
    
    rc = model.predict(Xc, raw_score=True)
    rv = model.predict(Xv, raw_score=True)
    
    pm = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    pm.fit(rc.reshape(-1,1), yc)
    yp = expit(pm.coef_[0][0]*rv + pm.intercept_[0])
    auc_p_lgb = roc_auc_score(yv, yp)
    nu_p_lgb = len(set(np.round(yp, 4)))
    
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(rc, yc)
    yi = ir.transform(rv)
    auc_i_lgb = roc_auc_score(yv, yi)
    nu_i_lgb = len(set(np.round(yi, 4)))
    
    bias_lgb = (yp.mean() - yv.mean())*100
    lgb_slope, lgb_int = float(pm.coef_[0][0]), float(pm.intercept_[0])
    print(f"  LGB 结果: AUC_Platt={auc_p_lgb:.4f} AUC_IR={auc_i_lgb:.4f} | 偏{bias_lgb:+.2f}% | 离散P={nu_p_lgb} IR={nu_i_lgb} | 耗时{time.time()-t0:.0f}s", flush=True)
    
    del model, lt, pm, ir
    gc.collect()
    
except Exception as e:
    print(f"  LightGBM failed: {e}", flush=True)
    auc_p_lgb = 0

# ======== 最佳模型 ========
use_lgb = auc_p_lgb > 0 and auc_p_lgb > auc_p
best = 'XGBoost' if not use_lgb else 'LightGBM'
print(f"4. 最佳: {best} (AUC={max(auc_p_lgb if use_lgb else 0, auc_p):.4f})", flush=True)

if use_lgb:
    calib = {'method':'platt','framework':'lightgbm',
             'slope':lgb_slope,'intercept':lgb_int,
             'bias_pct':round(bias_lgb,2),'auc':round(auc_p_lgb,4),
             'prob_unique':nu_p_lgb}
else:
    calib = {'method':'platt','framework':'xgboost',
             'slope':xgb_slope,'intercept':xgb_int,
             'bias_pct':round(bias,2),'auc':round(auc_p,4),
             'prob_unique':nu_p}

json.dump(calib, open(f'{MODEL_DIR}/greenshaft_v5_calib.json','w'), indent=2)

# 比较报告
cmp = {'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),
       'best': best,
       'xgb':{'auc_platt':round(auc_p,4),'auc_ir':round(auc_i,4),
              'bias_pct':round(bias,2),'prob_unique_platt':nu_p,'prob_unique_ir':nu_i},
       'lgb':{'auc_platt':round(auc_p_lgb,4),'auc_ir':round(auc_i_lgb,4),
              'bias_pct':round(bias_lgb,2),'prob_unique_platt':nu_p_lgb,'prob_unique_ir':nu_i_lgb} if auc_p_lgb > 0 else {}}
json.dump(cmp, open(f'{MODEL_DIR}/platt_v5_compare.json','w'), indent=2)

print(f"  校准参数已保存: slope={calib['slope']:.4f} intercept={calib['intercept']:.4f}", flush=True)

# ======== 生成推荐 ========
print("5. 生成推荐...", flush=True)
df = pd.read_parquet(INPUT).dropna(subset=['fwd_5d_ret'])
g = df.groupby('ticker').last().reset_index()
Xl = np.nan_to_num(g[feats].values.astype(np.float32), nan=0.0)

if use_lgb:
    raw = model.predict(Xl, raw_score=True)
else:
    model2 = xgb.Booster()
    model2.load_model(f'{MODEL_DIR}/greenshaft_v5_5pct.json')
    raw = model2.predict(xgb.DMatrix(Xl), output_margin=True)

probs = expit(calib['slope'] * raw + calib['intercept'])
g = g[['ticker','date']].copy()
g['prob'] = np.round(probs, 4)
g = g.sort_values('prob', ascending=False)

n35 = (probs > 0.35).sum()
n30 = (probs > 0.30).sum()
n25 = (probs > 0.25).sum()

print(f"  推荐: >35线={n35} >30线={n30} >25线={n25} 离散={len(set(np.round(probs,4)))}", flush=True)
print(f"  {'#':>3} {'代码':<8} {'概率':>8}", flush=True)
for i, (_, r) in enumerate(g.head(20).iterrows()):
    t = "**" if r['prob']>0.35 else "**" if r['prob']>0.30 else ""
    print(f"  {i+1:>3} {r['ticker']:<8} {r['prob']:.4f}", flush=True)

pred = {'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),
        'model':'greenshaft_v5_5pct (Platt)','auc':calib['auc'],
        'calib_method':'platt','calib_slope':calib['slope'],'calib_intercept':calib['intercept'],
        'calib_bias_pct':calib['bias_pct'],
        'n_above_35':int(n35),'n_above_30':int(n30),
        'top_50':[{'ticker':r['ticker'],'prob':r['prob']} for _,r in g.head(50).iterrows()]}
json.dump(pred, open(f'{MODEL_DIR}/greenshaft_v5_prediction.json','w'), indent=2)

print(f"完成! {(time.time()-T0)/60:.1f}分钟", flush=True)
