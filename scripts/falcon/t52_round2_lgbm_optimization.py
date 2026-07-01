"""
T5.2 Round 2: LightGBM Deep Optimization (5 more configs)
Focus on LightGBM variants since it showed best results.
"""
import sys, os, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime
warnings.filterwarnings('ignore')

DATA_PATH = '/home/hermes/.hermes/openclaw-archive/data/falcon/training_data_v04.parquet'
OUTPUT_PATH = '/home/hermes/.hermes/openclaw-archive/data/falcon/v04_xgboost_final_results.json'
ENGINE_PATH = '/home/hermes/.hermes/openclaw-archive/scripts/falcon'

sys.path.insert(0, ENGINE_PATH)
from backtest_engine import BacktestEngine
import xgboost as xgb_lib
import lightgbm as lgb_lib
from sklearn.linear_model import Ridge

t_total = time.time()
print("=" * 70)
print("  T5.2 Round 2: LightGBM Deep Optimization")
print("=" * 70)

df = pd.read_parquet(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])

all_exclude = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'vwap',
               'fwd_ret_5d', 'fwd_ret_10d', 'fwd_ret_20d', 'fwd_ret_30d']
all_feature_cols = [c for c in df.columns if c not in all_exclude]

strong_features = [
    'news_avg_sentiment', 'news_neg_ratio', 'news_sentiment_vol', 'news_pos_ratio',
    'ebitdaMargin_qoq', 'operatingProfitMargin_qoq', 'eps_dispersion', 'ret1',
    'news_confidence_avg', 'receivablesTurnover', 'news_article_count'
]

# Load previous results
with open(OUTPUT_PATH) as f:
    prev = json.load(f)
prev_results = prev.get('results', [])

# Add engineered features
df_w = df.copy()
df_w['vol20_x_momentum'] = df_w['vol20'] * df_w['momentum_1m']
df_w['PE_x_growth'] = df_w['priceToEarningsRatio'] * df_w['ebitdaMargin_qoq']
df_w['vol20_x_ret1'] = df_w['vol20'] * df_w['ret1']
df_w['news_sent_x_count'] = df_w['news_avg_sentiment'] * df_w['news_article_count']
vol20_s = df_w['vol20'].replace(0, np.nan)
df_w['ret1_voladj'] = df_w['ret1'] / vol20_s
df_w['rsi14_voladj'] = df_w['rsi14'] / vol20_s
for c in ['ret1', 'vol20']:
    df_w[f'{c}_lag1'] = df_w.groupby('ticker')[c].shift(1)
    df_w[f'{c}_lag5'] = df_w.groupby('ticker')[c].shift(5)

feat_eng = strong_features + [
    'vol20_x_momentum', 'PE_x_growth', 'vol20_x_ret1', 'news_sent_x_count',
    'ret1_voladj', 'rsi14_voladj', 'ret1_lag1', 'ret1_lag5', 'vol20_lag1', 'vol20_lag5'
]

# All features without news (for broader model)
feat_all_nonews = [f for f in all_feature_cols if 'news' not in f][:25]

prices = df_w.pivot_table(index='date', columns='ticker', values='close')
prices.index = prices.index.astype(str).str[:10]
prices = prices.sort_index()

print(f"  Data: {df_w.shape[0]} rows, {len(all_feature_cols)} features, {df_w['ticker'].nunique()} tickers")

TRAIN_YEARS = 5
TEST_MONTHS = 6

def run_wf(feat_cols, target, hold_days, model_type, model_params, name):
    df_t = df_w.dropna(subset=[target]).copy()
    all_dates = sorted(df_t['date'].unique())
    first_d, last_d = pd.Timestamp(all_dates[0]), pd.Timestamp(all_dates[-1])
    
    ranks = {}
    ts = first_d
    wi = 0
    
    while True:
        te = ts + pd.DateOffset(years=TRAIN_YEARS)
        tend = te + pd.DateOffset(months=TEST_MONTHS)
        if tend > last_d:
            break
        
        t0 = time.time()
        trn = df_t[(df_t['date'] >= ts) & (df_t['date'] < te)]
        tst = df_t[(df_t['date'] >= te) & (df_t['date'] < tend)]
        
        if len(trn) < 100 or len(tst) < 10:
            wi += 1
            ts += pd.DateOffset(months=TEST_MONTHS)
            continue
        
        af = [f for f in feat_cols if f in trn.columns]
        Xtr = np.ascontiguousarray(trn[af].values, dtype=np.float32)
        ytr = trn[target].values
        Xte = np.ascontiguousarray(tst[af].values, dtype=np.float32)
        
        Xtr = np.nan_to_num(np.where(np.isinf(Xtr), np.nan, Xtr), nan=0.0)
        Xte = np.nan_to_num(np.where(np.isinf(Xte), np.nan, Xte), nan=0.0)
        
        si = int(len(Xtr) * 0.8)
        
        if model_type == 'lgbm':
            dtr = lgb_lib.Dataset(Xtr[:si], label=ytr[:si])
            dvl = lgb_lib.Dataset(Xtr[si:], label=ytr[si:], reference=dtr)
            m = lgb_lib.train(model_params, dtr, valid_sets=[dvl],
                              callbacks=[lgb_lib.early_stopping(50), lgb_lib.log_evaluation(0)])
            yp = m.predict(Xte)
        elif model_type == 'xgb':
            m = xgb_lib.XGBRegressor(**model_params)
            m.fit(Xtr[:si], ytr[:si], eval_set=[(Xtr[si:], ytr[si:])], verbose=False)
            yp = m.predict(Xte)
        elif model_type == 'ridge':
            m = Ridge(**model_params)
            m.fit(Xtr, ytr)
            yp = m.predict(Xte)
        
        for i in range(len(tst)):
            tk = tst['ticker'].iloc[i]
            ds = str(tst['date'].iloc[i])[:10]
            if ds not in ranks:
                ranks[ds] = {}
            ranks[ds][tk] = float(yp[i])
        
        print(f"    W{wi}: {str(te)[:10]}→{str(tend)[:10]} | {time.time()-t0:.0f}s")
        wi += 1
        ts += pd.DateOffset(months=TEST_MONTHS)
    
    return ranks

def evaluate(ranks, target, hold_days, name):
    eng = BacktestEngine(cost=0.001, stop_loss=-0.15)
    er = {d: pd.DataFrame({'score': pd.Series(s)}) for d, s in ranks.items()}
    
    res = None
    try:
        res, _ = eng.run(er, prices, {'score': 1.0}, hold_days=hold_days, top_n=10, run_baseline=False)
        print(f"  → Sharpe={res.sharpe:.3f} MaxDD={res.max_dd:.1%} CAGR={res.cagr:.1%} WR={res.win_rate:.0%}")
    except Exception as e:
        print(f"  ⚠️ Backtest error: {e}")
    
    cd = sorted(set(er.keys()) & set(prices.index))
    t5, b20 = [], []
    dft = df_w.dropna(subset=[target]).copy()
    for d in cd:
        if d in er and d in prices.index:
            rd = er[d]
            ti = rd.dropna().index.tolist()
            if len(ti) < 20:
                continue
            act = dft[(dft['date'].astype(str).str[:10] == d) & (dft['ticker'].isin(ti))]
            if len(act) < 20:
                continue
            act = act.set_index('ticker')[target]
            sc = rd.loc[act.index, 'score'].dropna()
            n = len(sc)
            t5.append(act.loc[sc.nlargest(max(1, int(n*0.05))).index].mean())
            b20.append(act.loc[sc.nsmallest(max(1, int(n*0.20))).index].mean())
    
    spread = None
    if t5 and b20:
        a5, b2 = float(np.mean(t5)), float(np.mean(b20))
        spread = a5 - b2
        print(f"  → Rank: Top5={a5:.4f} Bot20={b2:.4f} {'✅' if spread > 0 else '❌'}")
    
    ri = {
        'top5_avg_ret': round(float(np.mean(t5)), 6) if t5 else None,
        'bot20_avg_ret': round(float(np.mean(b20)), 6) if b20 else None,
        'spread': round(float(spread), 6) if spread is not None else None,
        'direction_correct': bool(spread > 0) if spread is not None else None,
    }
    return res, ri

# ═══════════════════════════════════════════════════════════════════
#  5 LIGHTGBM FOCUSED CONFIGS
# ═══════════════════════════════════════════════════════════════════
LGBM_P200 = dict(objective='regression', metric='mse', boosting_type='gbdt',
                 num_leaves=15, learning_rate=0.05, feature_fraction=0.7,
                 bagging_fraction=0.7, bagging_freq=5, min_child_samples=50,
                 reg_alpha=1.0, reg_lambda=10.0, n_estimators=200,
                 random_state=42, n_jobs=-1, verbose=-1)

LGBM_P300 = dict(objective='regression', metric='mse', boosting_type='gbdt',
                 num_leaves=20, learning_rate=0.03, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=5, min_child_samples=30,
                 reg_alpha=0.5, reg_lambda=5.0, n_estimators=300,
                 random_state=42, n_jobs=-1, verbose=-1)

LGBM_P_DEEP = dict(objective='regression', metric='mse', boosting_type='gbdt',
                    num_leaves=10, learning_rate=0.03, feature_fraction=0.6,
                    bagging_fraction=0.6, bagging_freq=5, min_child_samples=100,
                    reg_alpha=5.0, reg_lambda=50.0, n_estimators=200,
                    random_state=42, n_jobs=-1, verbose=-1)

configs = [
    ('G_lgbm200',  strong_features, 'fwd_ret_10d', 8, 'lgbm', LGBM_P200, 'LightGBM 200 trees, strong11'),
    ('H_lgbm300',  strong_features, 'fwd_ret_10d', 8, 'lgbm', LGBM_P300, 'LightGBM 300 trees, strong11'),
    ('I_lgbm_eng',  feat_eng, 'fwd_ret_10d', 8, 'lgbm', LGBM_P200, 'LightGBM 200 + engineered features'),
    ('J_lgbm_deep', strong_features, 'fwd_ret_10d', 8, 'lgbm', LGBM_P_DEEP, 'LightGBM deep reg, strong11'),
    ('K_lgbm5d',   strong_features, 'fwd_ret_5d', 5, 'lgbm', LGBM_P200, 'LightGBM 200, ret5d hold5'),
]

new_results = []

for name, fc, tgt, hd, mt, mp, desc in configs:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  {name}: {desc}")
    print(f"  Model={mt} Feats={len(fc)} Target={tgt} Hold={hd}d")
    print(f"{'='*60}")
    
    try:
        ranks = run_wf(fc, tgt, hd, mt, mp, name)
        res, ri = evaluate(ranks, tgt, hd, name)
        
        entry = {
            'config_name': name, 'description': desc, 'model_type': mt,
            'n_features': len([f for f in fc if f in df_w.columns]),
            'target': tgt, 'hold_days': hd,
            'sharpe': res.sharpe if res else None,
            'maxdd': res.max_dd if res else None,
            'cagr': res.cagr if res else None,
            'win_rate': res.win_rate if res else None,
            'n_trades': res.n_trades if res else 0,
            'total_return': res.total_return if res else None,
            'rank_inversion': ri,
            'elapsed_sec': round(time.time() - t0, 1),
        }
        new_results.append(entry)
        
        # Save incrementally
        all_results = prev_results + new_results
        with open(OUTPUT_PATH, 'w') as f:
            json.dump({
                'task': 'T5.2 XGBoost Deep Optimization',
                'version': 'v0.4.0',
                'timestamp': datetime.now().isoformat(),
                'status': 'in_progress',
                'round': 2,
                'results': [{k: v for k, v in r.items() if k != 'window_details'} for r in all_results],
            }, f, indent=2, default=str)
        
        print(f"  Done ({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback; traceback.print_exc()
        new_results.append({'config_name': name, 'error': str(e)})

# ═══════════════════════════════════════════════════════════════════
#  ENSEMBLE: LightGBM + XGB
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  Ensemble: LightGBM + XGB average")
print(f"{'='*60}")

try:
    lgbm_r = run_wf(strong_features, 'fwd_ret_10d', 8, 'lgbm', LGBM_P200, 'ens_lgbm')
    xgb_r = run_wf(strong_features, 'fwd_ret_10d', 8, 'xgb', 
                    dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                         subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=10.0,
                         min_child_weight=50, objective='reg:squarederror', tree_method='hist',
                         random_state=42, n_jobs=-1), 'ens_xgb')
    
    all_d = sorted(set(lgbm_r.keys()) & set(xgb_r.keys()))
    ens = {}
    for d in all_d:
        ens[d] = {}
        ts = set(lgbm_r.get(d, {}).keys()) | set(xgb_r.get(d, {}).keys())
        for t in ts:
            ens[d][t] = 0.5 * lgbm_r.get(d, {}).get(t, 0) + 0.5 * xgb_r.get(d, {}).get(t, 0)
    
    re, ri = evaluate(ens, 'fwd_ret_10d', 8, 'L_ens_lgbm_xgb')
    new_results.append({
        'config_name': 'L_ens_lgbm_xgb', 'description': 'LightGBM+XGB 50/50 avg, ret10d hold8',
        'model_type': 'ensemble', 'n_features': len(strong_features),
        'target': 'fwd_ret_10d', 'hold_days': 8,
        'sharpe': re.sharpe if re else None, 'maxdd': re.max_dd if re else None,
        'cagr': re.cagr if re else None, 'win_rate': re.win_rate if re else None,
        'n_trades': re.n_trades if re else 0, 'total_return': re.total_return if re else None,
        'rank_inversion': ri, 'elapsed_sec': 0,
    })
except Exception as e:
    print(f"  ⚠️ Ensemble error: {e}")

# ═══════════════════════════════════════════════════════════════════
#  FINAL
# ═══════════════════════════════════════════════════════════════════
all_results = prev_results + new_results

print(f"\n{'='*70}")
print("  ALL RESULTS (Round 1 + Round 2)")
print(f"{'='*70}")
print(f"{'Config':<20} {'Model':<10} {'Sharpe':>8} {'MaxDD':>8} {'CAGR':>8} {'WR':>6} {'Rank':>6}")
print("-" * 70)

valid = [r for r in all_results if r.get('sharpe') is not None]
for r in sorted(valid, key=lambda x: -(x['sharpe'] or 0)):
    ri = '✅' if r.get('rank_inversion', {}).get('direction_correct') else '❌'
    print(f"{r['config_name']:<20} {r.get('model_type','?'):<10} {r['sharpe']:>8.3f} {r['maxdd']:>8.1%} {r['cagr']:>8.1%} {r['win_rate']:>6.0%} {ri:>6}")

v031 = 1.161
pv = [r for r in valid if r.get('sharpe', 0) > v031 and r.get('rank_inversion', {}).get('direction_correct')]
pr = [r for r in valid if r.get('rank_inversion', {}).get('direction_correct')]

if pv:
    best = max(pv, key=lambda r: r['sharpe'])
    print(f"\n🏆 BEST (beats V0.3.1!): {best['config_name']} Sharpe={best['sharpe']:.3f}")
elif pr:
    best = max(pr, key=lambda r: r['sharpe'])
    print(f"\n⚠️ Best (rank OK): {best['config_name']} Sharpe={best['sharpe']:.3f} (V0.3.1={v031})")
else:
    best = max(valid, key=lambda r: r.get('sharpe', -999)) if valid else None
    print(f"\n❌ No configs passed rank inversion")

final = {
    'task': 'T5.2 XGBoost Deep Optimization',
    'version': 'v0.4.0',
    'timestamp': datetime.now().isoformat(),
    'status': 'complete',
    'best_config': {k: best.get(k) for k in ['config_name','description','model_type','n_features','target','hold_days','sharpe','maxdd','cagr','win_rate','n_trades']} if best else None,
    'v031_baseline_sharpe': v031,
    'beats_v031': best['sharpe'] > v031 if best else False,
    'results': [{k: v for k, v in r.items() if k != 'window_details'} for r in all_results],
    'configs_completed': len(all_results),
    'configs_passed_ri': len(pr),
    'configs_beat_v031': len(pv),
    'total_time_sec': round(time.time() - t_total, 1),
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(final, f, indent=2, default=str)

print(f"\n✅ Saved: {OUTPUT_PATH}")
print(f"Round 2 time: {time.time()-t_total:.0f}s")
