"""
a_ml_scan01_params.py — A股ML参数扫描优化夏普比率
命名规范: a_ = A股, ml = ML, scan01 = 第一版参数扫描, params = 内容

流程:
1. 加载特征缓存 (a_ml_feats_cache.json) 
2. 遍历不同参数组合，训练XGBoost
3. 用回测模拟计算夏普比率
4. 输出最优参数推荐

存储: /home/hermes/.hermes/openclaw-project/data/scan_params_v1.json (中间结果)
       /home/hermes/.hermes/openclaw-project/data/scan_params_v1_best.json (最优)
"""
import json, sys, os, time, gc
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')

t0 = time.time()

# ─── 路径 ───
CACHE_PATH = '/home/hermes/.hermes/openclaw-project/data/a_ml_feats_cache.json'
RESULT_PATH = '/home/hermes/.hermes/openclaw-project/data/scan_params_v1.json'
BEST_PATH   = '/home/hermes/.hermes/openclaw-project/data/scan_params_v1_best.json'
LOG_PATH    = '/home/hermes/.hermes/openclaw-project/scripts/system/scan_params_v1_log.txt'

def log(msg):
    msg = f'{time.strftime("%H:%M:%S")} | {msg}'
    print(msg, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# ─── Step 1: 加载数据 ───
log('Step 1/4: 加载特征缓存...')
with open(CACHE_PATH, 'rb') as f:
    d = json.load(f)
X = np.array(d['X'], dtype=np.float32)
y = np.array(d['y'], dtype=np.float32)
log(f'  X: {X.shape}, y: {y.shape}, 正例率: {y.mean():.2%}')
del d
gc.collect()

# ─── Step 2: 参数空间 ───
# 说明：特征已预计算，所以只扫模型超参数
# 回测用时间序列split模拟调仓cycle
PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth':    [3, 4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample':    [0.7, 0.8, 1.0],
}

# 计算组合数
total_combos = 1
for k, v in PARAM_GRID.items():
    total_combos *= len(v)
log(f'参数组合: {total_combos}')

# ─── Step 3: 回测函数 ───
def compute_sharpe(y_true, y_prob, top_pct=30):
    """从模型预测计算等价夏普比率
    
    策略: 每5天选top_pct%概率最高的标的, 预期收益由命中率决定
    y_true=1 → 目标收益+2%
    简化夏普 = (命中率*0.02 - (1-命中率)*0.02) / 0.04
    = (2*命中率 - 1) * 0.5
    """
    threshold = np.percentile(y_prob, 100 - top_pct)
    pred_buy = y_prob > threshold
    if pred_buy.sum() == 0:
        return 0.0, 0.0
    hit_rate = (y_true[pred_buy] == 1).mean()
    avg_ret = hit_rate * 0.02 - (1 - hit_rate) * 0.02
    sharpe = (2 * hit_rate - 1) * 0.5
    return float(sharpe), float(hit_rate)

def train_backtest(X, y, params, combo_idx):
    """训练+回测一个参数组合"""
    m = xgb.XGBClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        device='cuda'
    )
    
    # Train/Test split
    split_pt = int(len(y) * 0.8)
    X_tr, X_te = X[:split_pt], X[split_pt:]
    y_tr, y_te = y[:split_pt], y[split_pt:]
    
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    
    acc = float(accuracy_score(y_te, m.predict(X_te)))
    auc = float(roc_auc_score(y_te, p))
    
    # Compute Sharpe from test set
    sharpe, hit_rate = compute_sharpe(y_te, p)
    
    # Sharpe at different thresholds for robustness
    sharpe20, _ = compute_sharpe(y_te, p, top_pct=20)
    sharpe40, _ = compute_sharpe(y_te, p, top_pct=40)
    avg_sharpe = (sharpe + sharpe20 + sharpe40) / 3
    
    result = {
        'params': params,
        'combo_idx': combo_idx,
        'acc': round(acc, 4),
        'auc': round(auc, 4),
        'sharpe_top20': round(sharpe20, 4),
        'sharpe_top30': round(sharpe, 4),
        'sharpe_top40': round(sharpe40, 4),
        'avg_sharpe': round(avg_sharpe, 4),
        'hit_rate_top30': round(hit_rate, 4),
    }
    
    return result

# ─── Step 4: 遍历参数 ───
log(f'Step 2/4: 参数扫描开始 ({total_combos} combos)...')

results = []
combo_idx = 0
best_sharpe = -999
best_result = None

for ne in PARAM_GRID['n_estimators']:
    for md in PARAM_GRID['max_depth']:
        for lr in PARAM_GRID['learning_rate']:
            for ss in PARAM_GRID['subsample']:
                combo_idx += 1
                params = {
                    'n_estimators': ne,
                    'max_depth': md,
                    'learning_rate': lr,
                    'subsample': ss
                }
                
                log(f'  [{combo_idx}/{total_combos}] ne={ne} md={md} lr={lr} ss={ss}')
                try:
                    r = train_backtest(X, y, params, combo_idx)
                    results.append(r)
                    
                    log(f'    → Acc={r["acc"]:.4f} AUC={r["auc"]:.4f} '
                        f'Sharpe30={r["sharpe_top30"]:.4f} Sharpe_avg={r["avg_sharpe"]:.4f} Hit={r["hit_rate_top30"]:.4f}')
                    
                    if r['avg_sharpe'] > best_sharpe:
                        best_sharpe = r['avg_sharpe']
                        best_result = r
                        log(f'    ★ 新最优 Sharpe={best_sharpe:.4f}')
                    
                except Exception as e:
                    log(f'    ✗ 失败: {e}')
                
                # 增量存盘，每10个组合存一次
                if combo_idx % 10 == 0:
                    with open(RESULT_PATH + '.tmp', 'w') as f:
                        json.dump({'results': results, 'total': len(results)}, f, ensure_ascii=False)
                    log(f'  [增量存盘: {len(results)}个结果]')

# 最终存盘
with open(RESULT_PATH, 'w') as f:
    json.dump({'results': results, 'total': len(results)}, f, indent=2, ensure_ascii=False)
log(f'参数扫描全部完成, 共{len(results)}个结果')

# ─── Step 5: 最优参数 ───
log(f'\n{"="*60}')
log(f'最优参数 (基于夏普比率):')
log(f'  n_estimators: {best_result["params"]["n_estimators"]}')
log(f'  max_depth: {best_result["params"]["max_depth"]}')
log(f'  learning_rate: {best_result["params"]["learning_rate"]}')
log(f'  subsample: {best_result["params"]["subsample"]}')
log(f'  Acc: {best_result["acc"]:.4f}')
log(f'  AUC: {best_result["auc"]:.4f}')
log(f'  Sharpe (top30): {best_result["sharpe_top30"]:.4f}')
log(f'  Avg Sharpe: {best_result["avg_sharpe"]:.4f}')
log(f'  Hit Rate (top30): {best_result["hit_rate_top30"]:.4f}')

# 按夏普排序Top 5
results_sorted = sorted(results, key=lambda x: x['avg_sharpe'], reverse=True)
log(f'\nTop 5 参数组合:')
for i, r in enumerate(results_sorted[:5]):
    log(f'  #{i+1}: ne={r["params"]["n_estimators"]} md={r["params"]["max_depth"]} '
        f'lr={r["params"]["learning_rate"]} ss={r["params"]["subsample"]} '
        f'Sharpe={r["avg_sharpe"]:.4f} Acc={r["acc"]:.4f} AUC={r["auc"]:.4f}')

# 按AUC排序Top 5
results_sorted_auc = sorted(results, key=lambda x: x['auc'], reverse=True)
log(f'\nTop 5 (AUC):')
for i, r in enumerate(results_sorted_auc[:5]):
    log(f'  #{i+1}: ne={r["params"]["n_estimators"]} md={r["params"]["max_depth"]} '
        f'lr={r["params"]["learning_rate"]} ss={r["params"]["subsample"]} '
        f'AUC={r["auc"]:.4f} Acc={r["acc"]:.4f} Sharpe={r["avg_sharpe"]:.4f}')

# 与当前模型对比
log(f'\n当前模型 (a_xgb_tech_v1):')
log(f'  n_estimators=200 max_depth=5 learning_rate=0.05 subsample=0.8')
log(f'  Acc=67.73% AUC=0.6388')

# 最优vs当前对比
current_params = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8}
current_in_results = [r for r in results if 
                      r['params']['n_estimators']==200 and 
                      r['params']['max_depth']==5 and
                      r['params']['learning_rate']==0.05 and
                      r['params']['subsample']==0.8]
if current_in_results:
    cr = current_in_results[0]
    log(f'\n当前参数对比:')
    log(f'  当前: Acc={cr["acc"]:.4f} AUC={cr["auc"]:.4f} Sharpe={cr["avg_sharpe"]:.4f}')
    log(f'  最优: Acc={best_result["acc"]:.4f} AUC={best_result["auc"]:.4f} Sharpe={best_result["avg_sharpe"]:.4f}')

# 保存最优
with open(BEST_PATH, 'w') as f:
    json.dump({'best_sharpe': best_result, 'top5_sharpe': results_sorted[:5], 
               'top5_auc': results_sorted_auc[:5], 'total_tested': len(results)}, 
              f, indent=2, ensure_ascii=False)

log(f'\n结果保存: {RESULT_PATH}')
log(f'最优保存: {BEST_PATH}')
log(f'总耗时: {(time.time()-t0)/60:.1f}分钟')
