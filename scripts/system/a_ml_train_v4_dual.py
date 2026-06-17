"""
a_ml_train_v4_dual.py — A股ML v4 双重门控训练
目标变量：横截面前15% ∩ 5日涨幅>5%
参数扫描：ne=[100,200,300], md=[6,8,10], lr=[0.1,0.15,0.2]
GPU加速 (device='cuda')
"""
import json, os, time, concurrent.futures, pickle, datetime, sys
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

# ─── 配置 ───
OUT_DIR = '/home/hermes/.hermes/openclaw-project/data/models'
LOG_FILE = f'/home/hermes/.hermes/openclaw-project/scripts/system/train_v4_dual_{datetime.date.today().strftime("%Y%m%d")}.log'
MODEL_NAME = 'a_xgb_tech_v4'
MODEL_PATH = f'{OUT_DIR}/{MODEL_NAME}.json'
CAL_PATH = f'{OUT_DIR}/{MODEL_NAME}_cal.pkl'
META_PATH = f'{OUT_DIR}/{MODEL_NAME}_meta.json'

FEAT_NAMES = [
    'r1','r5','r20','m5_div_m20',
    'd5','d20','d60','align',
    'v5','v20','rsi','macd','vr','pos','c_div_m60',
    'vp_signal','vr20','vol_ratio','price_norm'
]

# 参数网格
PARAM_GRID = [
    {'n_estimators': ne, 'max_depth': md, 'learning_rate': lr,
     'subsample': 0.7, 'colsample_bytree': 0.8, 'min_child_weight': 1,
     'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    for ne in [100, 200, 300]
    for md in [6, 8, 10]
    for lr in [0.1, 0.15, 0.2]
]  # 27组

log = lambda msg: (print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True),
                    open(LOG_FILE,'a',encoding='utf-8').write(f'[{time.strftime("%H:%M:%S")}] {msg}\n'))

open(LOG_FILE,'w',encoding='utf-8').close()
t0 = time.time()

# ─── 计算双重门控标签 ───
def compute_stock_dual(code, h):
    """
    所有股票统一计算特征 + 后处理做横截面排名。
    先返回 (features, forward_returns, codes_list, dates_list)
    """
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

    all_rows, all_returns, all_dates = [], [], []
    
    for i in range(100, n-5):
        try:
            r1 = c[i]/c[i-1]-1 if c[i-1]>0 else 0
            r5 = c[i]/c[i-5]-1 if i>=5 and c[i-5]>0 else 0
            r20 = c[i]/c[i-20]-1 if i>=20 and c[i-20]>0 else 0
            
            m5 = np.mean(c[i-4:i+1])
            m10 = np.mean(c[i-9:i+1])
            m20 = np.mean(c[i-19:i+1])
            m60 = np.mean(c[i-59:i+1]) if i>=59 else m20
            
            d5 = c[i]/m5-1; d20 = c[i]/m20-1; d60 = c[i]/m60-1
            align = 1 if m5>m10>m20 else (-1 if m5<m10<m20 else 0)
            m5_div_m20 = m5/m20 - 1
            
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
            
            # 波动率
            v5 = np.std([c[j]/c[j-1]-1 for j in range(i-4,i+1)])
            v20 = np.std([c[j]/c[j-1]-1 for j in range(i-19,i+1)])
            
            # 位置
            h20 = np.max(hi[i-19:i+1]); l20 = np.min(lo[i-19:i+1])
            pos = (c[i]-l20)/(h20-l20) if h20>l20 else 0.5
            
            # 波动率比
            vol_ratio = v5/v20 if v20>0 else 1.0
            vr20 = v[i] / np.mean(v[i-19:i+1]) if np.mean(v[i-19:i+1])>0 else 1
            price_norm = c[i]/m60 - 1
            
            # vp_signal
            vol_mean_5 = np.mean(v[i-4:i+1])
            price_mean_5 = np.mean(c[i-4:i+1])
            if v[i] > vol_mean_5 and c[i] > price_mean_5:
                vp_s = 1.0
            elif v[i] < vol_mean_5 and c[i] < price_mean_5:
                vp_s = -1.0
            elif v[i] > vol_mean_5 and c[i] < price_mean_5:
                vp_s = -0.5
            else:
                vp_s = 0.5
            
            # c_div_m60 = price/m60 - 1
            c_div_m60 = c[i]/m60 - 1
            
            ret_f = c[i+5]/c[i]-1
            if c[i] > 0 and c[i+5] > 0 and not np.isnan(ret_f) and not np.isinf(ret_f):
                feat = [r1, r5, r20, m5_div_m20,
                        d5, d20, d60, align,
                        v5, v20, rsi, macd, vr, pos, c_div_m60,
                        vp_s, vr20, vol_ratio, price_norm]
                all_rows.append(feat)
                all_returns.append(ret_f)
                all_dates.append(dates[i])
        except Exception:
            continue
    
    if len(all_rows) > 10:
        return (np.array(all_rows, dtype=np.float32),
                np.array(all_returns, dtype=np.float32),
                all_dates)
    return None


# ─── Step 1: 加载数据 ───
log('='*60)
log('A股ML v4 双重门控训练启动')
log('目标: 横截面前15% ∩ 5日涨幅>5%')
log(f'参数网格: 27组')
log('='*60)

log('Step 1/4: 加载K线数据...')
with open('/home/hermes/.hermes/openclaw-project/data/a_hist_10y.json', 'r', encoding='utf-8') as f:
    hist = json.load(f)

codes = [c for c in hist 
         if c.startswith(('60','00')) 
         and len(hist[c].get('dates',[])) >= 750]
log(f'  主板且>=3年: {len(codes)}只')

# ─── Step 2: 并行特征计算 ───
log(f'Step 2/4: 并行计算特征 ({len(codes)}只)...')

all_X, all_returns, all_dates = [], [], []
batch_size = 200
total_batches = (len(codes) + batch_size - 1) // batch_size

for batch_idx, batch_start in enumerate(range(0, len(codes), batch_size)):
    batch_codes = codes[batch_start:batch_start+batch_size]
    batch_X, batch_ret, batch_dt = [], [], []
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
            futures = {ex.submit(compute_stock_dual, code, hist[code]): code for code in batch_codes}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                try:
                    result = fut.result(timeout=60)
                    if result is not None:
                        batch_X.append(result[0])
                        batch_ret.append(result[1])
                        batch_dt.extend(result[2])
                except Exception:
                    pass
                done += 1
                if done % 50 == 0:
                    log(f'  批次{batch_idx+1}/{total_batches}: {done}/{len(batch_codes)}只')
    except Exception as e:
        log(f'  批次{batch_idx+1} 异常: {e}')
    
    if batch_X:
        all_X.append(np.vstack(batch_X))
        all_returns.append(np.concatenate(batch_ret))
        all_dates.extend(batch_dt)
        n_rows = sum(len(x) for x in batch_X)
        log(f'  批次{batch_idx+1}/{total_batches}: +{n_rows}行')
    
    if (batch_idx+1) % 5 == 0:
        log(f'  累计: {sum(len(x) for x in all_X)}行')

if not all_X:
    log('错误: 无数据生成')
    sys.exit(1)

X = np.vstack(all_X); returns = np.concatenate(all_returns)
log(f'  总行数: {X.shape}, 特征数: {X.shape[1]}')
log(f'  5日涨幅统计: mean={returns.mean():.4f}, std={returns.std():.4f}')
log(f'  涨幅>5%: {(returns>0.05).mean()*100:.1f}%')

del hist  # 释放内存

# ─── Step 2b: 构建双重门控标签 ───
# 按日期分组做横截面排名
log('Step 2b/4: 构建双重门控标签...')

# 收集唯一的日期
unique_dates = sorted(set(all_dates))
date_to_indices = {}
for idx, dt in enumerate(all_dates):
    date_to_indices.setdefault(dt, []).append(idx)

# 每个日期内做横截面排名
y_dual = np.zeros(len(returns), dtype=np.float32)
daily_stats = []  # 记录每日信息

for dt in unique_dates:
    idxs = date_to_indices[dt]
    n_day = len(idxs)
    day_rets = returns[idxs]
    
    # 横截面排名百分位（高到低）
    ranks = np.argsort(np.argsort(-day_rets))  # 0=最好
    pct = ranks / max(n_day - 1, 1)  # 0.0=最好, 1.0=最差
    
    # 双重门控: 横截面前15% AND 涨幅>5%
    for j, idx in enumerate(idxs):
        if pct[j] < 0.15 and day_rets[j] > 0.05:
            y_dual[idx] = 1.0
    
    # 记录当日统计
    n_pos = int(np.sum(y_dual[idxs]))
    daily_stats.append({'date': dt, 'n_stocks': n_day, 'n_dual_pos': n_pos,
                        'top15_mean_ret': np.mean(day_rets[pct<0.15])})

pos_rate = y_dual.mean()
log(f'  双重门控标签: 正例率 = {pos_rate:.4f} ({int(pos_rate*len(y_dual))}/{len(y_dual)})')
log(f'  日均正例: {np.mean([d["n_dual_pos"] for d in daily_stats]):.1f}')
log(f'  前15%平均涨幅: {np.mean([d["top15_mean_ret"] for d in daily_stats])*100:.2f}%')

# 保存日期分组信息供特征重要性分析
log(f'  日期数: {len(unique_dates)}, 日均样本数: {len(returns)/len(unique_dates):.0f}')

del all_dates, date_to_indices, all_returns  # 释放内存

# ─── Step 3: 参数扫描 ───
log(f'Step 3/4: GPU参数扫描 ({len(PARAM_GRID)}组)...')

X_train, X_test, y_train, y_test = train_test_split(
    X, y_dual, test_size=0.2, random_state=42)

best_score = 0
best_params = None
best_model = None
best_p = None
results = []

for idx, params in enumerate(PARAM_GRID):
    t_start = time.time()
    log(f'  [{idx+1}/{len(PARAM_GRID)}] ne={params["n_estimators"]} md={params["max_depth"]} lr={params["learning_rate"]}')
    
    try:
        m = xgb.XGBClassifier(
            **params,
            random_state=42, n_jobs=-1,
            tree_method='hist', device='cuda',
            eval_metric='auc', early_stopping_rounds=10,
            verbosity=0
        )
        m.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
        
        p = m.predict_proba(X_test)[:, 1]
        auc_val = roc_auc_score(y_test, p)
        acc_val = accuracy_score(y_test, (p > 0.5).astype(float))
        
        # 检查校准后AUC
        from sklearn.linear_model import LogisticRegression
        lr_cal = LogisticRegression(random_state=42, max_iter=1000)
        lr_cal.fit(p.reshape(-1,1), y_test)
        cp = lr_cal.predict_proba(p.reshape(-1,1))[:,1]
        cal_auc = roc_auc_score(y_test, cp)
        
        elapsed = time.time() - t_start
        log(f'    -> AUC={auc_val:.4f} calAUC={cal_auc:.4f} Acc={acc_val:.4f} ({elapsed:.0f}s)')
        
        results.append({
            'params': params,
            'auc': float(auc_val),
            'cal_auc': float(cal_auc),
            'acc': float(acc_val),
            'elapsed': round(elapsed, 1)
        })
        
        if cal_auc > best_score:
            best_score = cal_auc
            best_params = params.copy()
            best_model = m
            best_p = cp
            log(f'    ⭐ 新最优!')
            
    except Exception as e:
        log(f'    ❌ 失败: {e}')
        results.append({'params': params, 'auc': 0, 'error': str(e)})
    
    # 每9组保存一次中间结果
    if (idx+1) % 9 == 0:
        tmp = {'model': MODEL_NAME, 'best_auc': best_score, 'best_params': best_params,
               'results': results, 'partial': True}
        with open(f'{OUT_DIR}/{MODEL_NAME}_scan_tmp.json', 'w') as f:
            json.dump(tmp, f, indent=2, ensure_ascii=False, default=str)
        log(f'  中间保存: best calAUC={best_score:.4f}')

if best_model is None:
    log('错误: 所有参数组合全部失败!')
    sys.exit(1)

log(f'\n⭐ 最优参数: ne={best_params["n_estimators"]} md={best_params["max_depth"]} lr={best_params["learning_rate"]}')
log(f'⭐ 最优校准AUC: {best_score:.4f}')

# ─── Step 4: 完整校准 + 保存 ───
log('Step 4/4: 校准 + 保存...')

log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(best_p.reshape(-1,1), y_test)
cp = log_reg.predict_proba(best_p.reshape(-1,1))[:,1]

# 校准质量检查
log('  校准质量:')
cal_report = []
for lo in np.arange(0, 1, 0.1):
    hi = lo + 0.1
    mask = (cp >= lo) & (cp < hi)
    if mask.sum() > 10:
        pred = cp[mask].mean()
        actual = y_test[mask].mean()
        diff = abs(pred - actual)
        flag = '✅' if diff < 0.03 else ('⚠️' if diff < 0.05 else '❌')
        line = f'    [{lo:.1f},{hi:.1f}) n={mask.sum():5d}  pred={pred:.3f} actual={actual:.3f} diff={diff:.3f} {flag}'
        log(line)
        cal_report.append({'bin': f'{lo:.1f}-{hi:.1f}', 'n': int(mask.sum()),
                           'pred': round(float(pred),3), 'actual': round(float(actual),3)})

# 特征重要性
log('\n--- 特征重要性 ---')
imp = best_model.feature_importances_
imp_pairs = sorted(zip(imp, FEAT_NAMES), reverse=True)
for v, n in imp_pairs:
    log(f'  {n}: {v:.4f}')

# 保存校准器
cal = {
    'intercept': float(log_reg.intercept_[0]),
    'coef': float(log_reg.coef_[0][0]),
    'method': 'platt_logistic'
}

# 保存模型
best_model.save_model(MODEL_PATH)
with open(CAL_PATH, 'wb') as f:
    pickle.dump(log_reg, f)

# 元数据
meta = {
    'model': MODEL_NAME,
    'date': str(datetime.date.today()),
    'features': FEAT_NAMES,
    'n_features': len(FEAT_NAMES),
    'label_desc': '横截面前15% ∩ 5日涨幅>5% (双重门控)',
    'params': best_params,
    'performance': {
        'acc': float(accuracy_score(y_test, (cp > 0.5).astype(float))),
        'auc': float(roc_auc_score(y_test, cp)),
        'pos_rate': float(y_train.mean()),
    },
    'calibration': cal,
    'calibration_report': cal_report,
    'feature_importance': {n: round(float(v),4) for v, n in imp_pairs},
    'n_train': int(len(y_train)),
    'n_test': int(len(y_test)),
    'param_scan_results': sorted(results, key=lambda r: r.get('cal_auc', 0), reverse=True)[:5],
}

with open(META_PATH, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

log(f'\n✅ 完成! 总耗时: {(time.time()-t0)/60:.1f}分钟')
log(f'模型: {MODEL_PATH}')
log(f'校准器: {CAL_PATH}')
log(f'元数据: {META_PATH}')

# 输出对比摘要
v2_path = f'{OUT_DIR}/a_xgb_tech_v2_meta.json'
if os.path.exists(v2_path):
    with open(v2_path) as f:
        v2_meta = json.load(f)
    v2_auc = v2_meta['performance'].get('cal_auc', v2_meta['performance'].get('auc', 0))
    log(f'\n📊 对比 v2:')
    log(f'  v2 AUC (old 目标>2%): {v2_auc:.4f}')
    log(f'  v4 AUC (双重门控): {meta["performance"]["auc"]:.4f}')
    delta = meta['performance']['auc'] - v2_auc
    log(f'  差异: {delta:+.4f} {"(更好) ✅" if delta >= 0 else "(退步) ⚠️"}')
