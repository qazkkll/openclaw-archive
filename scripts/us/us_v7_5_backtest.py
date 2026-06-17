# 优化版本 v3 - 用groupby替代逐天过滤
import sys, os, json, pickle, time, itertools, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, xgboost as xgb

# ⚡ Session 预清理（防卡死）
sys.path.insert(0, os.path.dirname(__file__))
try:
    from sys_session_cleanup import clean_sessions
    clean_sessions()
except Exception:
    pass

BASE = '/home/hermes/.hermes/openclaw-archive'; ML = f'{BASE}/ml'; MD = f'{BASE}/data/models'
VER = 'us_v7_5'
print('='*70, flush=True); print('V7.5 快速回测 v3', flush=True); print('='*70, flush=True)
T0 = time.time()

model = xgb.Booster(); model.load_model(f'{MD}/{VER}.json')
cal = pickle.load(open(f'{MD}/{VER}_calibrator.pkl', 'rb'))
report = json.load(open(f'{MD}/{VER}_report.json'))
FEATS = report['features']
print(f'[1] {len(FEATS)}特征', flush=True)

df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str'] = df['date'].astype(str).str[:10]
df['target'] = (df['fwd_5d_ret'] > 0.05).astype(int)
for f in FEATS:
    if f in df.columns:
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0).clip(-1e6, 1e6)
df = df.replace([np.inf, -np.inf], np.nan)
del df['date']  # 释放原始date列
print(f'  特征: {len(df):,}行, {df.sym.nunique()}只', flush=True)

BTD = [ds for ds in sorted(df['date_str'].unique()) if ds >= '2022-01-01']
print(f'  回测: {len(BTD)}天 ({BTD[0]}~{BTD[-1]})', flush=True)

# 价格索引（缓存）
idx_path = f'{ML}/us_v75_close_idx.pkl'
if os.path.exists(idx_path) and os.path.getsize(idx_path) > 1e6:
    close_idx = pickle.load(open(idx_path, 'rb'))
    print(f'  价格缓存: {len(close_idx)}只', flush=True)
else:
    main = pd.read_parquet(f'{ML}/us_hist_yf_10y.parquet', columns=['ticker', 'date', 'close'])
    main.rename(columns={'ticker': 'sym'}, inplace=True)
    mega = pd.read_parquet(f'{ML}/us_hist_megacap_10y.parquet', columns=['sym', 'date', 'close'])
    all_v = pd.concat([main, mega], ignore_index=True).drop_duplicates(subset=['sym', 'date'])
    all_v['ds'] = all_v['date'].astype(str).str[:10]
    all_v = all_v[all_v['ds'].isin(BTD)]
    close_idx = {}
    for s, g in all_v.groupby('sym'):
        close_idx[s] = dict(zip(g['ds'].values, g['close'].values.astype(float)))
    pickle.dump(close_idx, open(idx_path, 'wb'))
    del main, mega, all_v
    print(f'  价格索引: {len(close_idx)}只, {time.time()-T0:.0f}s', flush=True)

# 分批概率+候选股（在同一个分批里完成，不产生中间大DataFrame）
print(f'\n[2] 概率计算+候选索引（分批）...', flush=True)
day_cands = {d: [] for d in BTD}
n_batch = 10000
n_total = len(df)
for i in range(0, n_total, n_batch):
    pct = 100 * i // n_total if n_total else 0
    if pct % 20 == 0: print(f'  {pct}%...', flush=True)
    chunk = df.iloc[i:i+n_batch]
    chunk = chunk[chunk['date_str'].isin(BTD)]
    if len(chunk) == 0: continue
    X = np.nan_to_num(chunk[FEATS].values.astype(np.float32), nan=0)
    raw = model.predict(xgb.DMatrix(X, feature_names=FEATS))
    calib = cal.predict_proba(raw.reshape(-1, 1))[:, 1]
    for j, (_, r) in enumerate(chunk.iterrows()):
        d = r['date_str']
        p = float(calib[j])
        if p <= 0: continue
        price = close_idx.get(r['sym'], {}).get(d)
        if price is None or np.isnan(price): continue
        day_cands[d].append((r['sym'], p, float(price)))

# 按概率排序
for d in BTD:
    day_cands[d].sort(key=lambda x: -x[1])
# 统计
total_cands = sum(len(v) for v in day_cands.values())
print(f'  候选股: {total_cands:,}条, {sum(1 for v in day_cands.values() if len(v)>=30)}天>=30只', flush=True)

# 回测
print(f'\n[3] 参数回测...', flush=True)
PM_TOP = [5, 10, 15]; PM_HOLD = [5, 10]; PM_STOP = [5, 10, 15]; PM_REB = [5, 10]
results = []

for top_n, hold, stop, rebal in itertools.product(PM_TOP, PM_HOLD, PM_STOP, PM_REB):
    if hold < rebal: continue
    cap = 10000.0; cash = cap; portfolio = {}; trades = 0; wins = 0; curve = [cap]
    sl = stop / 100.0
    
    for day_idx, d in enumerate(BTD):
        for sym in list(portfolio.keys()):
            pos = portfolio[sym]
            cp = close_idx.get(sym, {}).get(d)
            if cp is None: continue
            ret = (cp - pos['bp']) / pos['bp']
            if ret <= -sl or (day_idx - pos['di']) >= hold:
                cash += pos['qty'] * cp
                trades += 1
                if cp >= pos['bp']: wins += 1
                del portfolio[sym]
        
        if day_idx % rebal == 0:
            cands = [c for c in day_cands.get(d, []) if c[0] not in portfolio]
            buys = cands[:top_n]
            for sym, prob, price in buys:
                qty = cash / max(top_n, 1) / max(price, 0.01)
                if qty < 1: continue
                portfolio[sym] = {'bp': price, 'qty': qty, 'di': day_idx}
                cash -= qty * price
        
        pv = sum(p['qty'] * close_idx.get(s, {}).get(d, p['bp']) for s, p in portfolio.items())
        curve.append(cash + pv)
    
    final = cash + sum(p['qty'] * p['bp'] for p in portfolio.values())
    ec = np.array(curve)
    tr = (final / 10000 - 1) * 100
    yrs = len(BTD) / 252
    an = ((final / 10000) ** (1 / max(yrs, 0.01)) - 1) * 100
    peak = np.maximum.accumulate(ec)
    mdd = (ec - peak).min() / peak.max() * 100 if peak.max() > 0 else 0
    dr = np.diff(ec) / (ec[:-1] + 1e-10)
    sh = (dr.mean() / max(dr.std(), 1e-6)) * np.sqrt(252) if len(dr) > 20 else 0
    wr = wins / max(trades, 1)
    tag = f'T{top_n}_H{hold}_S{stop}_R{rebal}'
    results.append({'tag': tag, 'tr': round(tr, 1), 'an': round(an, 1),
        'sh': round(sh, 2), 'mdd': round(mdd, 1), 'wr': round(wr, 3), 'trades': trades})
    print(f'  {tag}: 年化{an:.1f}% 夏普{sh:.2f} 回撤{mdd:.1f}%', flush=True)

# 结果
print('\n[4] 结果', flush=True)
rdf = pd.DataFrame(results).sort_values('sh', ascending=False)
print(f'{"参数":20s} {"收益":>7s} {"年化":>7s} {"夏普":>6s} {"回撤":>7s} {"胜率":>6s} {"交易":>6s}')
print('-' * 60)
for _, r in rdf.iterrows():
    print(f'{r["tag"]:20s} {r["tr"]:>6.1f}% {r["an"]:>6.1f}% {r["sh"]:>6.2f} {r["mdd"]:>6.1f}% {r["wr"]:>5.1%} {r["trades"]:>6}')

print('\n=== 夏普Top5 ===')
for _, r in rdf.head(5).iterrows():
    print(f'  {r["tag"]:20s} 年化{r["an"]:>5.1f}% 夏普{r["sh"]:>5.2f} 回撤{r["mdd"]:>5.1f}%')
print('\n=== 年化Top5 ===')
for _, r in rdf.sort_values('an', ascending=False).head(5).iterrows():
    print(f'  {r["tag"]:20s} 年化{r["an"]:>5.1f}% 夏普{r["sh"]:>5.2f} 回撤{r["mdd"]:>5.1f}%')

json.dump({
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'model': VER,
    'range': f'{BTD[0]}~{BTD[-1]}', 'days': len(BTD),
    'all': rdf.to_dict('records'),
    'sharpe_top': rdf.head(5).to_dict('records'),
}, open(f'{MD}/us_v7_5_backtest.json', 'w'), indent=2)
print(f'\n[5] 保存: us_v7_5_backtest.json')
print(f'耗时: {(time.time()-T0)/60:.1f}分钟')
print('='*70)
