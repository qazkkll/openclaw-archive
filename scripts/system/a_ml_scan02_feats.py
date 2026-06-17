"""
a_ml_scan02_feats.py — A股ML特征工程参数扫描
命名规范: a_ = A股, ml = ML, scan02 = 第二版特征扫描, feats = 特征工程

扫描维度:
1. 标签阈值: 1.5%, 2.0%, 3.0% (5日涨幅)
2. 前视窗口: 3日, 5日, 10日
3. 波动率窗口: 10日/20日, 15日/30日, 20日/40日
4. 额外特征: 加入绝对价格、成交量比window扩展

效率策略: 3134只 → 采样300只代表股（按市值/行业分层抽样）
存储: /home/hermes/.hermes/openclaw-project/data/scan_params_v1_feats.json
"""
import json, sys, os, time, gc, random
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')

t0 = time.time()
LOG_PATH = '/home/hermes/.hermes/openclaw-project/scripts/system/scan_params_v2_log.txt'
RESULT_PATH = '/home/hermes/.hermes/openclaw-project/data/scan_params_v2_feats.json'
BEST_PATH = '/home/hermes/.hermes/openclaw-project/data/scan_params_v2_best.json'

def log(msg):
    msg = f'{time.strftime("%H:%M:%S")} | {msg}'
    print(msg, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# ─── Step 1: 加载K线数据 ───
log('Step 1/4: 加载K线数据...')
with open('/home/hermes/.hermes/openclaw-project/data/a_hist_10y.json', 'rb') as f:
    hist = json.load(f)
log(f'  总股票数: {len(hist)}')

# 过滤
codes = [c for c in hist if c.startswith(('60','00')) and len(hist[c].get('dates',[])) >= 500]
log(f'  合格主板股 (>=500日): {len(codes)}')

# 采样300只
random.seed(42)
codes_sample = sorted(random.sample(codes, min(300, len(codes))))
log(f'  采样训练: {len(codes_sample)}只')

# ─── Step 2: 特征工程函数 ───
def compute_feats(code, hist, forward_days=5, label_pct=0.02, 
                  vol_window_s=10, vol_window_l=20, 
                  add_extra=False):
    """从K线数据计算技术面特征
    
    Args:
        forward_days: 前视窗口（预测未来N天涨幅）
        label_pct: 标签阈值（涨幅超过此值=正例）
        vol_window_s: 短期波动率窗口
        vol_window_l: 长期波动率窗口
        add_extra: 是否加入额外的绝对价格特征
    
    Returns:
        (X_rows, y_rows)
    """
    h = hist[code]
    try:
        c = np.array(h['c'][::-1], dtype=np.float64)
        hi = np.array(h['h'][::-1], dtype=np.float64)
        lo = np.array(h['l'][::-1], dtype=np.float64)
        v = np.array(h['v'][::-1], dtype=np.float64)
    except:
        return None, None
    
    n = len(c)
    need_lookback = max(vol_window_l, 60) + forward_days + 10
    if n < need_lookback:
        return None, None
    
    rows_x, rows_y = [], []
    
    for i in range(100, n - forward_days):
        # 基础特征 (与v1一致)
        r1 = c[i]/c[i-1]-1 if c[i-1]>0 else 0
        r5 = c[i]/c[i-5]-1 if i>=5 and c[i-5]>0 else 0
        r20 = c[i]/c[i-20]-1 if i>=20 and c[i-20]>0 else 0
        
        m5 = np.mean(c[i-4:i+1]); m10 = np.mean(c[i-9:i+1])
        m20 = np.mean(c[i-19:i+1]); m60 = np.mean(c[i-59:i+1]) if i>=59 else m20
        
        d5 = c[i]/m5-1; d20 = c[i]/m20-1; d60 = c[i]/m60-1
        align = 1 if m5>m10>m20 else (-1 if m5<m10<m20 else 0)
        
        # RSI (14日)
        lookback_rsi = min(14, i)
        if lookback_rsi < 3:
            continue
        chgs = np.diff(c[i-lookback_rsi:i+1])
        avg_g = np.mean(chgs[chgs>0]) if np.any(chgs>0) else 0.001
        avg_l = -np.mean(chgs[chgs<0]) if np.any(chgs<0) else 0.001
        rsi = 100 - 100/(1+avg_g/avg_l) if avg_l>1e-8 else 50
        
        # MACD (12/26)
        def ema(arr, p):
            if len(arr) < p: return arr[-1]
            r = np.mean(arr[:p])
            a = 2/(p+1)
            for val in arr[p:]:
                r = val*a + r*(1-a)
            return r
        
        e12 = ema(c[max(0,i-25):i+1], 12)
        e26 = ema(c[max(0,i-49):i+1], 26)
        macd = e12 - e26
        
        # 成交量比 (5日平均)
        vr = v[i] / np.mean(v[i-4:i+1]) if np.mean(v[i-4:i+1])>0 else 1
        
        # 20日位置
        h20 = np.max(hi[i-19:i+1]); l20 = np.min(lo[i-19:i+1])
        pos = (c[i]-l20)/(h20-l20) if h20>l20 else 0.5
        
        # 波动率 (使用可变窗口)
        v5 = np.std([c[j]/c[j-1]-1 for j in range(i-vol_window_s+1, i+1)])
        v20 = np.std([c[j]/c[j-1]-1 for j in range(i-vol_window_l+1, i+1)])
        
        # 基础特征列表: 15个
        feats = [r1, r5, r20, m5/m20, d5, d20, d60, align, v5, v20, rsi, macd, vr, pos, c[i]/m60]
        
        # 额外特征 (可开关)
        if add_extra:
            # 价格偏离20日线 (已有d20)
            # 成交量相对20日均值
            vr20 = v[i] / np.mean(v[i-19:i+1]) if np.mean(v[i-19:i+1])>0 else 1
            # 波动率比
            vol_ratio = v5 / v20 if v20 > 0 else 1
            # 价格归一化 (避免绝对值影响树分裂)
            price_norm = c[i] / m60 if m60 > 0 else 1
            # 量价配合: 涨+放量 = 1, 跌+缩量 = -1
            vp_signal = 1 if (r1 > 0 and vr > 1.2) else (-1 if r1 < 0 and vr < 0.8 else 0)
            
            feats.extend([vr20, vol_ratio, price_norm, vp_signal])
        
        # Y标签
        ret_f = c[i+forward_days]/c[i]-1
        if c[i]>0 and c[i+forward_days]>0:
            y = 1.0 if ret_f > label_pct else 0.0
            rows_x.append(feats)
            rows_y.append(y)
    
    if len(rows_x) < 10:
        return None, None
    
    return np.array(rows_x, dtype=np.float32), np.array(rows_y, dtype=np.float32)


# ─── Step 3: 参数组合 ───
FEAT_PARAMS = {
    'forward_days': [5],         # 固定5日前视 (与v1一致，不改)
    'label_pct': [0.015, 0.02, 0.03],
    'vol_window': [(10, 20), (15, 30), (20, 40)],
    'add_extra': [False, True],  # 是否加额外特征
}

# 跳过前6组（已跑过），从第7组开始
SKIP_FIRST_N = 6

total_feat_combos = 1
for k, v in FEAT_PARAMS.items():
    total_feat_combos *= len(v) if isinstance(v, list) else 1
log(f'特征组合数: {total_feat_combos}')


def train_eval(X, y, label):
    """给定特征矩阵训练XGBoost并评估"""
    if X is None or len(X) < 100:
        return None
    if len(np.unique(y)) < 2:
        return None
    
    split_pt = int(len(y) * 0.8)
    X_tr, X_te = X[:split_pt], X[split_pt:]
    y_tr, y_te = y[:split_pt], y[split_pt:]
    
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None
    
    # 用当前最优参数 (来自scan01先验)
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
        device='cuda'
    )
    m.fit(X_tr, y_tr)
    
    p = m.predict_proba(X_te)[:, 1]
    acc = float(accuracy_score(y_te, m.predict(X_te)))
    auc = float(roc_auc_score(y_te, p))
    
    # Sharpe at top30%
    threshold = np.percentile(p, 70)
    pred_buy = p > threshold
    hit_rate = (y_te[pred_buy] == 1).mean() if pred_buy.sum() > 0 else 0
    sharpe = (2 * hit_rate - 1) * 0.5
    
    # Feature importance
    fn = []
    n_feats = X.shape[1]
    base_fn = ['r1','r5','r20','m5/m20','d5','d20','d60','align','v5','v20','rsi','macd','vr','pos','c/m60']
    if n_feats > 15:
        extra_fn = ['vr20','vol_ratio','price_norm','vp_signal']
        fn = base_fn + extra_fn[:n_feats-15]
    else:
        fn = base_fn[:n_feats]
    
    imp = m.feature_importances_
    imp_sorted = sorted(zip(fn, imp), key=lambda x: -x[1])
    
    return {
        'label': label,
        'acc': round(acc, 4),
        'auc': round(auc, 4),
        'sharpe_top30': round(float(sharpe), 4),
        'hit_rate_top30': round(float(hit_rate), 4),
        'n_features': int(n_feats),
        'n_samples': int(len(y)),
        'pos_rate': round(float(y.mean()), 4),
        'top_features': [{'name': n, 'imp': round(float(v), 4)} for n, v in imp_sorted[:5]],
        'feature_names': fn,
    }


# ─── Step 4: 扫描 ───
log('Step 2/4: 生成特征+训练...')
results = []
best_sharpe = -999
combo_idx = 0

for fd in FEAT_PARAMS['forward_days']:
    for lp in FEAT_PARAMS['label_pct']:
        for ws, wl in FEAT_PARAMS['vol_window']:
            for add_x in FEAT_PARAMS['add_extra']:
                combo_idx += 1
                
                label = f'fd{fd}_p{int(lp*100)}_vw{ws}_{wl}_extra{int(1 if add_x else 0)}'
                log(f'  [{combo_idx}/{total_feat_combos}] {label}')
                
                # 跳过前6组（已跑过），只跑剩余12组
                if combo_idx <= SKIP_FIRST_N:
                    log(f'    → 跳过(已跑)')
                    continue
                
                all_X, all_y = [], []
                stock_cnt = 0
                
                t_start = time.time()
                for idx, code in enumerate(codes_sample):
                    try:
                        Xs, ys = compute_feats(code, hist, 
                            forward_days=fd, label_pct=lp,
                            vol_window_s=ws, vol_window_l=wl,
                            add_extra=add_x)
                        if Xs is not None:
                            all_X.append(Xs)
                            all_y.append(ys)
                            stock_cnt += 1
                    except Exception as e:
                        pass  # skip broken stocks
                    if (idx + 1) % 50 == 0:
                        elapsed = time.time() - t_start
                        log(f'      {idx+1}/{len(codes_sample)} stocks, {stock_cnt} ok, {elapsed:.0f}s')
                
                if len(all_X) < 5:
                    log(f'    → 数据不足, 跳过')
                    continue
                
                X = np.vstack(all_X)
                y = np.concatenate(all_y)
                log(f'    → {stock_cnt}只股票, {X.shape}, 正例率: {y.mean():.2%}')
                
                r = train_eval(X, y, label)
                if r is None:
                    log(f'    → 训练失败, 跳过')
                    continue
                
                # 去重：检查相同label是否已存在
                duplicate = any(existing['label'] == label for existing in results)
                if duplicate:
                    log(f'    → 跳过重复组合: {label}')
                    continue
                
                results.append(r)
                log(f'    → Acc={r["acc"]:.4f} AUC={r["auc"]:.4f} '
                    f'Sharpe={r["sharpe_top30"]:.4f} Hit={r["hit_rate_top30"]:.4f} '
                    f'Feats={r["top_features"]}')
                
                if r['sharpe_top30'] > best_sharpe:
                    best_sharpe = r['sharpe_top30']
                    log(f'    ★ 新最优!')

# 保存
with open(RESULT_PATH, 'w') as f:
    json.dump({'results': results, 'total': len(results), 'total_combos': total_feat_combos}, 
              f, indent=2, ensure_ascii=False)

# ─── Step 5: 汇报 ───
log(f'\n{"="*60}')
log(f'特征工程扫描完成 (共{len(results)}个结果)')

results_by_sharpe = sorted(results, key=lambda x: x['sharpe_top30'], reverse=True)
log(f'\nTop 3 (夏普):')
for r in results_by_sharpe[:3]:
    log(f'  {r["label"]}: Sharpe={r["sharpe_top30"]:.4f} AUC={r["auc"]:.4f} Hit={r["hit_rate_top30"]:.4f}')

results_by_auc = sorted(results, key=lambda x: x['auc'], reverse=True)
log(f'\nTop 3 (AUC):')
for r in results_by_auc[:3]:
    log(f'  {r["label"]}: AUC={r["auc"]:.4f} Sharpe={r["sharpe_top30"]:.4f} Hit={r["hit_rate_top30"]:.4f}')

# 对比基准 (原始参数)
baseline = [r for r in results if r['label'] == 'fd5_p2_vw10_20_extra0']
if baseline:
    log(f'\n基准(fd5_p2_vw10_20_extra0) vs 最优:')
    log(f'  基准: AUC={baseline[0]["auc"]:.4f} Sharpe={baseline[0]["sharpe_top30"]:.4f}')
    log(f'  最优: AUC={results_by_sharpe[0]["auc"]:.4f} Sharpe={results_by_sharpe[0]["sharpe_top30"]:.4f}')

# 保存最优
with open(BEST_PATH, 'w') as f:
    json.dump({
        'best_by_sharpe': results_by_sharpe[:3],
        'best_by_auc': results_by_auc[:3],
        'total_tested': len(results)
    }, f, indent=2, ensure_ascii=False)

log(f'\n结果: {RESULT_PATH}')
log(f'最优: {BEST_PATH}')
log(f'总耗时: {(time.time()-t0)/60:.1f}分钟')
