"""
T3.2: XGBoost Walk-Forward Baseline (Optimized for speed)
"""
import sys, os, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime
warnings.filterwarnings('ignore')

DATA_PATH = '/home/hermes/.hermes/openclaw-archive/data/falcon/training_data_v04.parquet'
OUTPUT_PATH = '/home/hermes/.hermes/openclaw-archive/data/falcon/v04_xgboost_baseline_results.json'
ENGINE_PATH = '/home/hermes/.hermes/openclaw-archive/scripts/falcon'

XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=20, objective='reg:squarederror', tree_method='hist',
    random_state=42, n_jobs=-1,
)
TRAIN_YEARS = 5; TEST_MONTHS = 6
HOLD_DAYS = 30; TOP_N = 10; COST = 0.001; STOP_LOSS = -0.15

t_total = time.time()
print("=== T3.2 XGBoost Walk-Forward Baseline ===")
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.dropna(subset=['fwd_ret_30d'])
exclude_cols = ['date', 'ticker', 'fwd_ret_5d', 'fwd_ret_10d', 'fwd_ret_20d', 'fwd_ret_30d']
feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"  Rows: {df.shape[0]}, Features: {len(feature_cols)}, Tickers: {df['ticker'].nunique()}")

# Prices matrix
prices = df.pivot_table(index='date', columns='ticker', values='close')
prices.index = prices.index.astype(str).str[:10]
prices = prices.sort_index()

# ═══════════════════════════════════════════════════════════════════
# Walk-Forward XGBoost (OOS predictions only)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Walk-Forward XGBoost Training ===")
import xgboost as xgb_lib

all_dates = sorted(df['date'].unique())
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
    train_df = df[(df['date'] >= train_start) & (df['date'] < train_end)]
    test_df = df[(df['date'] >= train_end) & (df['date'] < test_end)]

    if len(train_df) < 100 or len(test_df) < 10:
        window_idx += 1; train_start += pd.DateOffset(months=TEST_MONTHS)
        continue

    X_train = np.ascontiguousarray(train_df[feature_cols].values, dtype=np.float32)
    y_train = train_df['fwd_ret_30d'].values
    X_test = np.ascontiguousarray(test_df[feature_cols].values, dtype=np.float32)
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)

    model = xgb_lib.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_test)

    for i in range(len(test_df)):
        ticker = test_df['ticker'].iloc[i]
        date_str = str(test_df['date'].iloc[i])[:10]
        if date_str not in ranks_dict:
            ranks_dict[date_str] = {}
        ranks_dict[date_str][ticker] = float(y_pred[i])

    elapsed = time.time() - t0
    top_feat = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])[:3]
    print(f"  W{window_idx}: {str(train_end)[:10]}→{str(test_end)[:10]} | "
          f"tr={len(train_df):,} te={len(test_df):,} | "
          f"μ={np.mean(y_pred):.4f} σ={np.std(y_pred):.4f} | {elapsed:.0f}s | "
          f"{', '.join(f'{n}({v:.3f})' for n,v in top_feat)}")

    window_details.append({
        'window': window_idx,
        'train_period': f"{str(train_start)[:10]} → {str(train_end)[:10]}",
        'test_period': f"{str(train_end)[:10]} → {str(test_end)[:10]}",
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'pred_mean': float(np.mean(y_pred)),
        'pred_std': float(np.std(y_pred)),
        'elapsed_sec': round(elapsed, 1),
        'top_features': [{'name': n, 'importance': round(float(v), 4)} for n,v in top_feat],
    })

    window_idx += 1
    train_start += pd.DateOffset(months=TEST_MONTHS)

print(f"\nTotal windows: {window_idx}, Dates with predictions: {len(ranks_dict)}")

# ═══════════════════════════════════════════════════════════════════
# Build Engine Ranks
# ═══════════════════════════════════════════════════════════════════
engine_ranks = {}
for date_str, ticker_scores in ranks_dict.items():
    engine_ranks[date_str] = pd.DataFrame({'xgb_score': pd.Series(ticker_scores)})

rank_dates = sorted(engine_ranks.keys())
price_dates = sorted(prices.index)
common_dates = sorted(set(rank_dates) & set(price_dates))
print(f"Common dates for backtest: {len(common_dates)}")

# ═══════════════════════════════════════════════════════════════════
# Backtest
# ═══════════════════════════════════════════════════════════════════
print("\n=== Backtest ===")
sys.path.insert(0, ENGINE_PATH)
from backtest_engine import BacktestEngine, DataQualityError

engine = BacktestEngine(cost=COST, stop_loss=STOP_LOSS)
weights = {'xgb_score': 1.0}

t0 = time.time()
full_result, baseline_result = None, None
try:
    full_result, baseline_result = engine.run(
        engine_ranks, prices, weights,
        hold_days=HOLD_DAYS, top_n=TOP_N, run_baseline=True
    )
    print(f"Full Period: {full_result.summary()}")
    if baseline_result:
        print(f"Baseline:    {baseline_result.summary()}")
except Exception as e:
    print(f"Error: {e}")
    import traceback; traceback.print_exc()
print(f"Backtest time: {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════
# Rank Inversion Check
# ═══════════════════════════════════════════════════════════════════
print("\n=== Rank Inversion Check ===")
top5_rets, bot20_rets = [], []
for d in common_dates:
    if d in engine_ranks and d in prices.index:
        rank_df = engine_ranks[d]
        tickers_in = rank_df.dropna().index.tolist()
        if len(tickers_in) < 20:
            continue
        actual = df[(df['date'].astype(str).str[:10] == d) & (df['ticker'].isin(tickers_in))]
        if len(actual) < 20:
            continue
        actual = actual.set_index('ticker')['fwd_ret_30d']
        scores = rank_df.loc[actual.index, 'xgb_score'].dropna()
        n = len(scores)
        top5_tickers = scores.nlargest(max(1, int(n*0.05))).index
        bot20_tickers = scores.nsmallest(max(1, int(n*0.20))).index
        top5_rets.append(actual.loc[top5_tickers].mean())
        bot20_rets.append(actual.loc[bot20_tickers].mean())

spread = None
if top5_rets and bot20_rets:
    avg_top5, avg_bot20 = float(np.mean(top5_rets)), float(np.mean(bot20_rets))
    spread = avg_top5 - avg_bot20
    print(f"  Top 5% avg fwd_ret_30d: {avg_top5:.4f}")
    print(f"  Bottom 20% avg fwd_ret_30d: {avg_bot20:.4f}")
    print(f"  Spread: {spread:.4f}")
    print(f"  {'✅ CORRECT' if spread > 0 else '⚠️ INVERSION'}")
else:
    print("  Insufficient data")

# ═══════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════
v031_sharpe = 1.161
wf_sharpe = full_result.sharpe if full_result else None

print(f"\n=== V0.3.1 Comparison ===")
print(f"  V0.3.1 Sharpe: {v031_sharpe}")
print(f"  XGBoost Sharpe: {wf_sharpe}")

results = {
    'task': 'T3.2 XGBoost Baseline',
    'version': 'v0.4.0',
    'timestamp': datetime.now().isoformat(),
    'config': {
        'model': 'XGBRegressor', 'params': XGB_PARAMS,
        'train_years': TRAIN_YEARS, 'test_months': TEST_MONTHS,
        'hold_days': HOLD_DAYS, 'top_n': TOP_N,
        'cost': COST, 'stop_loss': STOP_LOSS,
        'n_features': len(feature_cols),
    },
    'data': {
        'total_rows': int(df.shape[0]),
        'n_tickers': int(df['ticker'].nunique()),
        'date_range': [str(df['date'].min())[:10], str(df['date'].max())[:10]],
        'n_dates_with_predictions': len(ranks_dict),
    },
    'full_backtest': {
        'sharpe': full_result.sharpe if full_result else None,
        'max_dd': full_result.max_dd if full_result else None,
        'cagr': full_result.cagr if full_result else None,
        'win_rate': full_result.win_rate if full_result else None,
        'n_trades': full_result.n_trades if full_result else None,
        'n_rebalances': full_result.n_rebalances if full_result else None,
        'total_return': full_result.total_return if full_result else None,
        'warnings': full_result.warnings if full_result else [],
    },
    'baseline_comparison': {
        'v031_sharpe': v031_sharpe,
        'baseline_sharpe': baseline_result.sharpe if baseline_result else None,
        'xgboost_vs_v031': round(wf_sharpe - v031_sharpe, 3) if wf_sharpe else None,
    },
    'rank_inversion': {
        'top5_avg_ret': round(avg_top5, 6) if top5_rets else None,
        'bot20_avg_ret': round(avg_bot20, 6) if bot20_rets else None,
        'spread': round(spread, 6) if spread is not None else None,
        'direction_correct': bool(spread > 0) if spread is not None else None,
    },
    'xgb_training_windows': window_details,
    'total_time_sec': round(time.time() - t_total, 1),
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n✅ Saved: {OUTPUT_PATH}")
print(f"Total time: {time.time()-t_total:.0f}s")
print("T3.2 COMPLETE")
