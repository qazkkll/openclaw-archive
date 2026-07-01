"""
T3.4: XGBoost Optimization — Faster version (6 configs, early stopping)
"""
import sys, os, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime
warnings.filterwarnings('ignore')

DATA_PATH = '/home/hermes/.hermes/openclaw-archive/data/falcon/training_data_v04.parquet'
OUTPUT_PATH = '/home/hermes/.hermes/openclaw-archive/data/falcon/v04_xgboost_optimized_results.json'
ENGINE_PATH = '/home/hermes/.hermes/openclaw-archive/scripts/falcon'

sys.path.insert(0, ENGINE_PATH)
from backtest_engine import BacktestEngine, DataQualityError
import xgboost as xgb_lib

t_total = time.time()
print("=" * 70)
print("  T3.4 XGBoost Optimization (Fast)")
print("=" * 70)

# Load data
print("Loading data...")
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

top32_features = [
    'news_avg_sentiment', 'news_neg_ratio', 'news_sentiment_vol', 'news_pos_ratio',
    'ebitdaMargin_qoq', 'operatingProfitMargin_qoq', 'eps_dispersion', 'ret1',
    'news_confidence_avg', 'receivablesTurnover', 'news_article_count',
    'macd_roc', 'eps_revision', 'ret5', 'num_analysts_eps', 'vol_regime',
    'ma20', 'vol5', 'ma5', 'macd_hist', 'bb_pos', 'ma_align', 'quickRatio',
    'vol20', 'ma60', 'ret_quality', 'vwap_drift', 'operatingProfitMargin',
    'ma_bias20', 'macd', 'priceToBookRatio', 'dd_60'
]

prices = df.pivot_table(index='date', columns='ticker', values='close')
prices.index = prices.index.astype(str).str[:10]
prices = prices.sort_index()

print(f"  Rows: {df.shape[0]}, Features: {len(all_feature_cols)}, Tickers: {df['ticker'].nunique()}")

# Configs (6 most promising)
CONSERVATIVE_PARAMS = dict(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=5.0,
    min_child_weight=50, objective='reg:squarederror', tree_method='hist',
    random_state=42, n_jobs=-1,
)

BASE_PARAMS = dict(
    n_estimators=100, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=20, objective='reg:squarederror', tree_method='hist',
    random_state=42, n_jobs=-1,
)

configs = [
    ('A_baseline_repro', all_feature_cols, 'fwd_ret_30d', 30, BASE_PARAMS, 'Baseline: 84feats, ret30d, hold30'),
    ('B_strong_30d_30', strong_features, 'fwd_ret_30d', 30, CONSERVATIVE_PARAMS, 'Strong11, ret30d, hold30'),
    ('C_top32_30d_30', top32_features, 'fwd_ret_30d', 30, CONSERVATIVE_PARAMS, 'Top32, ret30d, hold30'),
    ('D_strong_10d_10', strong_features, 'fwd_ret_10d', 10, CONSERVATIVE_PARAMS, 'Strong11, ret10d, hold10'),
    ('E_strong_5d_5', strong_features, 'fwd_ret_5d', 5, CONSERVATIVE_PARAMS, 'Strong11, ret5d, hold5'),
    ('F_top32_20d_20', top32_features, 'fwd_ret_20d', 20, CONSERVATIVE_PARAMS, 'Top32, ret20d, hold20'),
]

TRAIN_YEARS = 5
TEST_MONTHS = 6
all_results = []
engine = BacktestEngine(cost=0.001, stop_loss=-0.15)

for name, feat_cols, target, hold_days, params, desc in configs:
    cfg_t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  {name}: {desc}")
    print(f"  Features: {len(feat_cols)}, Target: {target}, Hold: {hold_days}d")
    print(f"{'='*60}")
    
    df_target = df.dropna(subset=[target]).copy()
    all_dates = sorted(df_target['date'].unique())
    first_date, last_date = pd.Timestamp(all_dates[0]), pd.Timestamp(all_dates[-1])
    
    ranks_dict = {}
    train_start = first_date
    window_idx = 0
    window_details = []
    
    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)
        if test_end > last_date:
            break
        
        t0 = time.time()
        train_df = df_target[(df_target['date'] >= train_start) & (df_target['date'] < train_end)]
        test_df = df_target[(df_target['date'] >= train_end) & (df_target['date'] < test_end)]
        
        if len(train_df) < 100 or len(test_df) < 10:
            window_idx += 1
            train_start += pd.DateOffset(months=TEST_MONTHS)
            continue
        
        avail_feats = [f for f in feat_cols if f in train_df.columns]
        X_train = np.ascontiguousarray(train_df[avail_feats].values, dtype=np.float32)
        y_train = train_df[target].values
        X_test = np.ascontiguousarray(test_df[avail_feats].values, dtype=np.float32)
        
        X_train = np.nan_to_num(np.where(np.isinf(X_train), np.nan, X_train), nan=0.0)
        X_test = np.nan_to_num(np.where(np.isinf(X_test), np.nan, X_test), nan=0.0)
        
        # Split train for early stopping (last 20% as eval)
        split_idx = int(len(X_train) * 0.8)
        X_tr, X_eval = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_eval = y_train[:split_idx], y_train[split_idx:]
        
        model = xgb_lib.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_eval, y_eval)], verbose=False)
        y_pred = model.predict(X_test)
        
        for i in range(len(test_df)):
            ticker = test_df['ticker'].iloc[i]
            date_str = str(test_df['date'].iloc[i])[:10]
            if date_str not in ranks_dict:
                ranks_dict[date_str] = {}
            ranks_dict[date_str][ticker] = float(y_pred[i])
        
        elapsed = time.time() - t0
        top_feat = sorted(zip(avail_feats, model.feature_importances_), key=lambda x: -x[1])[:3]
        
        window_details.append({
            'window': window_idx,
            'test_period': f"{str(train_end)[:10]} → {str(test_end)[:10]}",
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'elapsed_sec': round(elapsed, 1),
            'top_features': [{'name': n, 'importance': round(float(v), 4)} for n, v in top_feat],
        })
        print(f"  W{window_idx}: {str(train_end)[:10]}→{str(test_end)[:10]} | {elapsed:.0f}s | top: {top_feat[0][0]}({top_feat[0][1]:.3f})")
        
        window_idx += 1
        train_start += pd.DateOffset(months=TEST_MONTHS)
    
    print(f"  Windows: {window_idx}, Predictions: {len(ranks_dict)}")
    
    # Build engine ranks
    engine_ranks = {}
    for date_str, ticker_scores in ranks_dict.items():
        engine_ranks[date_str] = pd.DataFrame({'xgb_score': pd.Series(ticker_scores)})
    
    common_dates = sorted(set(engine_ranks.keys()) & set(prices.index))
    
    # Backtest
    full_result = None
    try:
        full_result, _ = engine.run(engine_ranks, prices, {'xgb_score': 1.0},
                                     hold_days=hold_days, top_n=10, run_baseline=False)
        print(f"  Sharpe={full_result.sharpe:.3f} MaxDD={full_result.max_dd:.1%} CAGR={full_result.cagr:.1%} WR={full_result.win_rate:.0%}")
    except Exception as e:
        print(f"  ⚠️ Backtest error: {e}")
    
    # Rank Inversion
    top5_rets, bot20_rets = [], []
    for d in common_dates:
        if d in engine_ranks and d in prices.index:
            rank_df = engine_ranks[d]
            tickers_in = rank_df.dropna().index.tolist()
            if len(tickers_in) < 20:
                continue
            actual = df_target[(df_target['date'].astype(str).str[:10] == d) & (df_target['ticker'].isin(tickers_in))]
            if len(actual) < 20:
                continue
            actual = actual.set_index('ticker')[target]
            scores = rank_df.loc[actual.index, 'xgb_score'].dropna()
            n = len(scores)
            top5_rets.append(actual.loc[scores.nlargest(max(1, int(n*0.05))).index].mean())
            bot20_rets.append(actual.loc[scores.nsmallest(max(1, int(n*0.20))).index].mean())
    
    spread = None
    if top5_rets and bot20_rets:
        avg_top5, avg_bot20 = float(np.mean(top5_rets)), float(np.mean(bot20_rets))
        spread = avg_top5 - avg_bot20
        print(f"  Rank: Top5={avg_top5:.4f} Bot20={avg_bot20:.4f} {'✅' if spread > 0 else '⚠️ INVERSION'}")
    
    result_entry = {
        'config_name': name,
        'description': desc,
        'n_features': len(feat_cols),
        'target': target,
        'hold_days': hold_days,
        'sharpe': full_result.sharpe if full_result else None,
        'maxdd': full_result.max_dd if full_result else None,
        'cagr': full_result.cagr if full_result else None,
        'win_rate': full_result.win_rate if full_result else None,
        'n_trades': full_result.n_trades if full_result else 0,
        'total_return': full_result.total_return if full_result else None,
        'rank_inversion': {
            'top5_avg_ret': round(avg_top5, 6) if top5_rets else None,
            'bot20_avg_ret': round(avg_bot20, 6) if bot20_rets else None,
            'spread': round(spread, 6) if spread is not None else None,
            'direction_correct': bool(spread > 0) if spread is not None else None,
        },
        'window_details': window_details,
    }
    all_results.append(result_entry)
    
    # Incremental save after each config
    partial_output = {
        'task': 'T3.4 XGBoost Optimization',
        'timestamp': datetime.now().isoformat(),
        'status': 'in_progress',
        'configs_completed': len(all_results),
        'results': [{k: v for k, v in r.items()} for r in all_results],
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(partial_output, f, indent=2, default=str)
    
    print(f"  Config {name} done ({time.time()-cfg_t0:.0f}s)")

# ═══════════════════════════════════════════════════════════════════
#  Final Summary
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  RESULTS SUMMARY")
print(f"{'='*70}")
print(f"{'Config':<25} {'Sharpe':>8} {'MaxDD':>8} {'CAGR':>8} {'WR':>6} {'Rank':>5}")
print("-" * 65)

valid = [r for r in all_results if r['sharpe'] is not None]
for r in valid:
    ri = '✅' if r['rank_inversion']['direction_correct'] else '❌'
    print(f"{r['config_name']:<25} {r['sharpe']:>8.3f} {r['maxdd']:>8.1%} {r['cagr']:>8.1%} {r['win_rate']:>6.0%} {ri:>5}")

correct_dir = [r for r in valid if r['rank_inversion']['direction_correct']]
if correct_dir:
    best = max(correct_dir, key=lambda r: r['sharpe'])
else:
    best = max(valid, key=lambda r: r['sharpe'] if r['sharpe'] else -999)

print(f"\n🏆 Best: {best['config_name']} — Sharpe={best['sharpe']:.3f}")

comparison = {
    'baseline': {'sharpe': 0.852, 'max_dd': -0.5154, 'cagr': 0.1796, 'win_rate': 0.566},
    'best_optimized': {
        'sharpe': best['sharpe'], 'max_dd': best['maxdd'],
        'cagr': best['cagr'], 'win_rate': best['win_rate'],
        'config_name': best['config_name'],
    },
    'sharpe_improvement': round(best['sharpe'] - 0.852, 3) if best['sharpe'] else None,
    'meets_target': best['sharpe'] > 1.0 if best['sharpe'] else False,
}

print(f"\n📊 vs Baseline: {0.852:.3f} → {best['sharpe']:.3f} ({comparison['sharpe_improvement']:+.3f})")
print(f"   Target (>1.0): {'✅' if comparison['meets_target'] else '❌'}")

# Final save
final_output = {
    'task': 'T3.4 XGBoost Optimization',
    'version': 'v0.4.0',
    'timestamp': datetime.now().isoformat(),
    'status': 'complete',
    'best_config': {
        'name': best['config_name'],
        'description': best['description'],
        'n_features': best['n_features'],
        'target': best['target'],
        'hold_days': best['hold_days'],
        'sharpe': best['sharpe'],
        'maxdd': best['maxdd'],
        'cagr': best['cagr'],
        'win_rate': best['win_rate'],
    },
    'results': all_results,
    'comparison_with_baseline': comparison,
    'total_time_sec': round(time.time() - t_total, 1),
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(final_output, f, indent=2, default=str)

print(f"\n✅ Saved: {OUTPUT_PATH}")
print(f"Total time: {time.time()-t_total:.0f}s")
print("T3.4 COMPLETE")
