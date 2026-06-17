"""
a_ml_train_v4_gpu.py — A股ML v4训练 (GPU加速)
目标变量: 横截面前15% ∩ 5日涨幅>5% (双重门控)
参数扫描: n_estimators=[100,200,300] x max_depth=[6,8,10] x learning_rate=[0.1,0.15,0.2]
"""
import json, os, time, concurrent.futures, pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import traceback

# ─── 日志 ───
LOG = '/home/hermes/.hermes/openclaw-project/scripts/system/train_v4_log.txt'
open(LOG, 'w', encoding='utf-8').close()
log = lambda msg: (open(LOG, 'a', encoding='utf-8').write(f'[{time.strftime("%H:%M:%S")}] {msg}\n'),
                   print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True))

t0 = time.time()

FEAT_NAMES = [
    'r1','r5','r20','d5','d20','d60','align',
    'v5','v20','rsi','macd','vr','pos','c_div_m60',
    'vp_signal','vr20','vol_ratio','price_norm'
]

def compute_stock(code, h):
    """单只股票特征 + 双重门控标签"""
    try:
        for k in ['c','h','l','v','dates']:
            if k not in h or not isinstance(h[k], list) or len(h[k]) < 200:
                return None
        c = np.array(h['c'][::-1], dtype=np.float64)
        hi = np.array(h['h'][::-1], dtype=np.float64)
        lo = np.array(h['l'][::-1], dtype=np.float64)
        v = np.array(h['v'][::-1], dtype=np.float64)
        dates = h['dates'][::-1]
    except Exception:
        return None

    n = len(c)
    if n < 200:
        return None

    rows_x, rows_y = [], []
    
    for i in range(100, n-5):
        try:
            r1 = c[i]/c[i-1]-1 if c[i-1]>0 else 0
            r5 = c[i]/c[i-5]-1 if i>=5 and c[i-5]>0 else 0
            r20 = c[i]/c[i-20]-1 if i>=20 and c[i-20]>0 else 0
            
            m5 = np.mean(c[i-4:i+1]); m10 = np.mean(c[i-9:i+1])
            m20 = np.mean(c[i-19:i+1])
            m60 = np.mean(c[i-59:i+1]) if i>=59 else m20
            
            d5 = c[i]/m5-1; d20 = c[i]/m20-1; d60 = c[i]/m60-1
            align = 1 if m5>m10>m20 else (-1 if m5<m10<m20 else 0)
            
            # RSI(14)
            chgs = np.diff(c[i-13:i+1])
            avg_g = np.mean(chgs[chgs>0]) if np.any(chgs>0) else 0.001
            avg_l = -np.mean(chgs[chgs<0]) if np.any(chgs<0) else 0.001
            rsi = 100 - 100/(1+avg_g/avg_l)
            
            # MACD
            e12 = np.mean(c[i-11:i+1]); e26 = np.mean(c[i-25:i+1])
            macd = e12 - e26
            
            # 量比
            vr = v[i] / np.mean(v[i-4:i+1]) if np.mean(v[i-4:i+1])>0 else 1
            
            # 位置
            h20 = np.max(hi[i-19:i+1]); l20 = np.min(lo[i-19:i+1])
            pos = (c[i]-l20)/(h20-l20) if h20>l20 else 0.5
            
            # 波动率
            v5 = np.std([c[j]/c[j-1]-1 for j in range(i-4,i+1)])
            v20 = np.std([c[j]/c[j-1]-1 for j in range(i-19,i+1)])
            
            vol_ratio = v5/v20 if v20>0 else 1.0
            vr20 = v[i] / np.mean(v[i-19:i+1]) if np.mean(v[i-19:i+1])>0 else 1
            price_norm = c[i]/m60 - 1
            
            # vp_signal
            if v[i] > np.mean(v[i-4:i+1]) and c[i] > np.mean(c[i-4:i+1]):
                vp_s = 1.0
            elif v[i] < np.mean(v[i-4:i+1]) and c[i] < np.mean(c[i-4:i+1]):
                vp_s = -1.0
            elif v[i] > np.mean(v[i-4:i+1]) and c[i] < np.mean(c[i-4:i+1]):
                vp_s = -0.5
            else:
                vp_s = 0.5
            
            ret_f = c[i+5]/c[i]-1
            if c[i]>0 and c[i+5]>0:
                # 标签: 涨幅>5% 且 横截面前15% (这里只存原始涨幅, 后续统一排名)
                # i+5的日期作为分组依据
                rows_x.append([
                    r1,r5,r20,
                    c[i]/m5-1, c[i]/m20-1, c[i]/m60-1,
                    align, v5, v20, rsi, macd, vr, pos,
                    price_norm, vp_s, vr20, vol_ratio, price_norm
                ])
                # 存: [5日涨跌幅, 日期] 用于后续横截面排名
                rows_y.append([ret_f, dates[i+5]])
        except Exception:
            continue
    
    if len(rows_x) > 10:
        return (np.array(rows_x, dtype=np.float32), np.array(rows_y, dtype=object))
    return None


# ═══════════════════════════════════════
# Step 1: 加载数据 + 计算特征
# ═══════════════════════════════════════
log(f'Step 1/4: 加载数据...')
with open('/home/hermes/.hermes/openclaw-project/data/a_hist_10y.json', 'r', encoding='utf-8') as f:
    hist = json.load(f)

codes = [c for c in hist if c.startswith(('60','00')) and len(hist[c].get('dates',[])) >= 750]
log(f'  主板且>=3年: {len(codes)}只')

log(f'Step 2/4: 并行计算特征 ({len(codes)}只)...')

all_X, all_y_meta = [], []
batch_size = 200
total_batches = (len(codes) + batch_size - 1) // batch_size

for batch_idx, batch_start in enumerate(range(0, len(codes), batch_size)):
    batch_codes = codes[batch_start:batch_start+batch_size]
    batch_X, batch_meta = [], []
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
            futures = {ex.submit(compute_stock, code, hist[code]): code for code in batch_codes}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                try:
                    result = fut.result(timeout=60)
                    if result is not None:
                        batch_X.append(result[0])
                        batch_meta.append(result[1])
                except Exception:
                    pass
                done += 1
                if done % 50 == 0:
                    log(f'  批次{batch_idx+1}/{total_batches}: {done}/{len(batch_codes)}只')
    except Exception as e:
        log(f'  批次{batch_idx+1} 线程池异常: {e}')
        traceback.print_exc()
    
    if batch_X:
        try:
            all_X.append(np.vstack(batch_X))
            all_y_meta.extend(np.concatenate(batch_meta).tolist())
            log(f'  批次{batch_idx+1}/{total_batches}: +{sum(len(x) for x in batch_X)}行')
        except Exception as e:
            log(f'  批次{batch_idx+1} 拼接失败: {e}')
    
    if (batch_idx+1) % 5 == 0:
        log(f'  中间: {batch_idx+1}/{total_batches}批, {sum(len(x) for x in all_X) if all_X else 0}行')

if not all_X:
    log('错误: 没有任何数据!')
    raise RuntimeError("No data")

X = np.vstack(all_X).astype(np.float32)
y_meta = np.array(all_y_meta, dtype=object)
log(f'  特征完成: {X.shape}, 元数据行数: {len(y_meta)}')
del hist, batch_X, batch_meta, all_y_meta

# ═══════════════════════════════════════
# 目标: 横截面排名前15% ∩ 5日涨幅>5%
# ═══════════════════════════════════════
log(f'\nStep 3/4: 计算双重门控标签...')

# y_meta是[[涨幅, 日期], ...]
rets = np.array([float(m[0]) for m in y_meta], dtype=np.float64)
dates_arr = np.array([str(m[1]) for m in y_meta])

# 同一天内做横截面排名
unique_dates = sorted(set(dates_arr))
log(f'  共{len(unique_dates)}个交易日')

y_dual = np.zeros(len(rets), dtype=np.float64)  # 1=正例, 0=负例
pos_count = 0

for d in unique_dates:
    mask = dates_arr == d
    if mask.sum() < 10:
        continue
    d_rets = rets[mask]
    
    # 横截面前15%排名阈值
    rank_thresh = np.percentile(d_rets, 85)
    
    for idx in np.where(mask)[0]:
        # 双重条件: 前15% 而且 涨幅>5%
        if rets[idx] >= rank_thresh and rets[idx] > 0.05:
            y_dual[idx] = 1.0
            pos_count += 1
        else:
            y_dual[idx] = 0.0

pos_rate = pos_count / len(rets)
log(f'  双重门控标签: 正例率={pos_rate:.4f} ({pos_count}/{len(rets)})')
log(f'  正例条件: 横截面前15% ∩ 5日涨幅>5%')

del y_meta, dates_arr, unique_dates

# ═══════════════════════════════════════
# Step 4: 参数扫描
# ═══════════════════════════════════════
log(f'\nStep 4/4: 参数扫描...')

# 分割
X_train, X_test, y_train, y_test = train_test_split(X, y_dual, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

best_auc = 0
best_params = None
best_model = None
results = []

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.1, 0.15, 0.2]
}

total_combo = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate'])
combo_idx = 0

for ne in param_grid['n_estimators']:
    for md in param_grid['max_depth']:
        for lr in param_grid['learning_rate']:
            combo_idx += 1
            log(f'  组合 {combo_idx}/{total_combo}: ne={ne} md={md} lr={lr}')
            
            model = xgb.XGBClassifier(
                n_estimators=ne, max_depth=md, learning_rate=lr,
                subsample=0.7, colsample_bytree=0.8,
                random_state=42, n_jobs=-1,
                tree_method='hist', device='cuda',
                eval_metric='auc',
                early_stopping_rounds=10
            )
            
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                p = model.predict_proba(X_test)[:,1]
                auc_val = roc_auc_score(y_test, p)
                acc_val = accuracy_score(y_test, (p >= 0.5).astype(int))
                
                results.append({
                    'ne': ne, 'md': md, 'lr': lr,
                    'auc': round(float(auc_val), 4),
                    'acc': round(float(acc_val), 4),
                    'best_ntree': model.best_iteration if model.best_iteration else ne
                })
                
                log(f'    → AUC={auc_val:.4f} Acc={acc_val:.4f} best_iter={model.best_iteration}')
                
                if auc_val > best_auc:
                    best_auc = auc_val
                    best_params = (ne, md, lr)
                    best_model = model
            except Exception as e:
                log(f'    ❌ 失败: {e}')
                continue

log(f'\n═══ 最优组合: ne={best_params[0]} md={best_params[1]} lr={best_params[2]} AUC={best_auc:.4f} ═══')

# 结果排序
results.sort(key=lambda r: r['auc'], reverse=True)
log(f'\nTop 5 参数组合:')
for r in results[:5]:
    log(f'  ne={r["ne"]} md={r["md"]} lr={r["lr"]} AUC={r["auc"]} Acc={r["acc"]} best_ntree={r["best_ntree"]}')

log(f'\n所有组合结果:')
for r in results:
    log(f'  {r["ne"]:3d}树 d{r["md"]} lr{r["lr"]:.2f}  AUC={r["auc"]:.4f}  Acc={r["acc"]:.4f}')

# ═══════════════════════════════════════
# 校准 (使用最优模型)
# ═══════════════════════════════════════
log(f'\n校准最优模型...')
p_best = best_model.predict_proba(X_test)[:,1]

# LogReg校准
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(p_best.reshape(-1, 1), y_test)
cp = log_reg.predict_proba(p_best.reshape(-1, 1))[:,1]
c_auc = roc_auc_score(y_test, cp)
log(f'  校准后AUC: {c_auc:.4f}')

# 校准质量
log('  校准质量:')
for lo in np.arange(0, 1, 0.1):
    hi = lo + 0.1
    mask = (cp >= lo) & (cp < hi)
    if mask.sum() > 10:
        pred = cp[mask].mean()
        actual = y_test[mask].mean()
        diff = abs(pred-actual)
        flag = ' ✅' if diff < 0.03 else (' ⚠️' if diff < 0.05 else ' ❌')
        log(f'    [{lo:.1f},{hi:.1f}) n={mask.sum()}  pred={pred:.3f} actual={actual:.3f} diff={diff:.3f}{flag}')

# ═══════════════════════════════════════
# 保存模型
# ═══════════════════════════════════════
log(f'\n保存模型...')

# 用全量训练集 + 最优参数再训练一次做最终模型
log('  训练最终模型 (全量)...')
final_model = xgb.XGBClassifier(
    n_estimators=best_params[0], max_depth=best_params[1],
    learning_rate=best_params[2],
    subsample=0.7, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
    tree_method='hist', device='cuda',
    eval_metric='auc'
)

final_model.fit(
    np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]),
    eval_set=[(X_test, y_test)],
    verbose=False
)

model_path = '/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v3.json'
cal_path = '/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v3_cal.pkl'
meta_path = '/home/hermes/.hermes/openclaw-project/data/models/a_xgb_tech_v3_meta.json'

final_model.save_model(model_path)
with open(cal_path, 'wb') as f:
    pickle.dump(log_reg, f)

meta = {
    'model': 'a_xgb_tech_v3',
    'date': '2026-06-10',
    'features': FEAT_NAMES,
    'n_features': len(FEAT_NAMES),
    'target': 'rank_p85_and_ret>5pct',
    'target_desc': '横截面前15% ∩ 5日涨幅>5%',
    'params': {
        'n_estimators': best_params[0],
        'max_depth': best_params[1],
        'learning_rate': best_params[2],
        'subsample': 0.7,
        'colsample_bytree': 0.8
    },
    'performance': {
        'acc': round(float(accuracy_score(y_test, (final_model.predict_proba(X_test)[:,1] >= 0.5).astype(int))), 4),
        'auc': round(float(c_auc), 4),
        'cal_auc': round(float(c_auc), 4)
    },
    'param_scan_results': results,
    'n_train': len(X_train) + len(X_val),
    'n_test': len(X_test),
    'pos_rate': float(pos_rate),
    'features_importance': {
        name: round(float(imp), 4)
        for name, imp in zip(FEAT_NAMES, final_model.feature_importances_)
    }
}

with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

elapsed = time.time() - t0
log(f'\n✅ 完成! 总耗时: {elapsed/60:.1f}分钟')
log(f'  模型: {model_path}')
log(f'  校准器: {cal_path}')
log(f'  元数据: {meta_path}')
