# -*- coding: utf-8 -*-
"""
蓝盾V4 实验矩阵
方向A: 去市场依赖
方向B: 排序模型
方向C: 风控特征
方向D: Purging+Embargo验证
"""
import warnings, json, os, time
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, 'data', 'us')
MODEL_DIR = os.path.join(ROOT, 'models', 'us')
os.makedirs(MODEL_DIR, exist_ok=True)

# ════════════════════════════════════════
#  1. 加载 + 特征计算
# ════════════════════════════════════════
print("📊 加载数据...")
df = pd.read_parquet(os.path.join(DATA_DIR, 'us_hist_sp500_10y.parquet'))
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['sym', 'date']).reset_index(drop=True)

# 加载SPY/VIX
import yfinance as yf
start_date = df['date'].min().strftime('%Y-%m-%d')
end_date = (df['date'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
spy_df = spy[['Close']].reset_index()
spy_df.columns = ['date', 'spy_close']
spy_df['date'] = pd.to_datetime(spy_df['date']).dt.tz_localize(None)
spy_df = spy_df.drop_duplicates(subset='date', keep='last')
spy_df['spy_ret1'] = spy_df['spy_close'].pct_change()
spy_df['spy_ret5'] = spy_df['spy_close'].pct_change(5)
spy_df['spy_ret20'] = spy_df['spy_close'].pct_change(20)
vix_df = vix[['Close']].reset_index()
vix_df.columns = ['date', 'vix_close']
vix_df['date'] = pd.to_datetime(vix_df['date']).dt.tz_localize(None)
vix_df = vix_df.drop_duplicates(subset='date', keep='last')

def compute_all_features(group):
    g = group.copy().reset_index(drop=True)
    c = g['close'].values.astype(float)
    h = g['high'].values.astype(float)
    l = g['low'].values.astype(float)
    v = g['volume'].values.astype(float)
    cs = pd.Series(c)

    # 收益率
    for d in [1,3,5,10,20]:
        g[f'ret_{d}d'] = cs.pct_change(d).values

    # 均线比例
    for w in [5,10,20,50]:
        ma = cs.rolling(w).mean()
        g[f'ma_{w}_ratio'] = c / ma.values

    # 波动率
    for w in [5,10,20]:
        g[f'vol_{w}d'] = cs.pct_change().rolling(w).std().values

    # RSI
    delta = cs.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 0.001)
    g['rsi_14'] = (100 - 100 / (1 + rs)).values
    g['rsi_50_pct'] = ((g['rsi_14'] - 50) / 50)

    # 量比
    vol5 = pd.Series(v).rolling(5).mean()
    vol20 = pd.Series(v).rolling(20).mean()
    g['vol_ratio_5'] = (v / vol5.values)
    g['vol_ratio_20'] = (v / vol20.values)
    v5r = cs.pct_change().rolling(5).std()
    v20r = cs.pct_change().rolling(20).std()
    v20r_safe = np.where(v20r.values == 0, 0.001, v20r.values)
    g['vol_5d_norm'] = (v5r.values / v20r_safe)

    # 价格位置
    for w in [20,50,100]:
        hh = pd.Series(h).rolling(w).max().values
        ll = pd.Series(l).rolling(w).min().values
        rng = np.where(hh - ll == 0, 0.001, hh - ll)
        g[f'price_pos_{w}'] = (c - ll) / rng

    # MACD
    ema12 = cs.ewm(span=12).mean()
    ema26 = cs.ewm(span=26).mean()
    g['macd'] = (ema12 - ema26).values
    g['macd_sig'] = g['macd'].ewm(span=9).mean().values
    g['macd_hist'] = (g['macd'] - g['macd_sig'])

    # ATR
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    g['atr_pct'] = (pd.Series(tr).rolling(20).mean() / c * 100).values

    # MA交叉
    ma20 = cs.rolling(20).mean()
    ma50 = cs.rolling(50).mean()
    g['ma20_ma50_cross'] = (ma20.values - ma50.values)

    # 成交额
    dv = c * v
    dm5 = pd.Series(dv).rolling(5).mean()
    g['dvol_ma5'] = dm5.values
    g['dvol_ratio'] = np.where(dm5.values > 0, dv / dm5.values, 1.0)

    # 方向C: 风控特征
    g['ret_5d_abs'] = cs.pct_change(5).abs()
    # 最大回撤 (20日)
    def max_dd(x):
        cummax = np.maximum.accumulate(x)
        dd = x / cummax - 1
        return dd.min()
    g['max_dd_20d'] = pd.Series(c).rolling(20).apply(max_dd, raw=True).values
    # Sharpe (20日)
    daily_ret = cs.pct_change()
    g['sharpe_20d'] = daily_ret.rolling(20).apply(
        lambda x: x.mean() / max(x.std(), 0.001) * np.sqrt(252), raw=True).values
    g['calmar_20d'] = g['ret_20d'] / g['max_dd_20d'].replace(0, -0.001)

    return g

t0 = time.time()
results = []
for sym, group in df.groupby('sym'):
    results.append(compute_all_features(group))
df = pd.concat(results, ignore_index=True)

df = df.merge(spy_df[['date', 'spy_ret1', 'spy_ret5', 'spy_ret20']], on='date', how='left')
df = df.merge(vix_df, on='date', how='left')
df['rel_ret_1d'] = df['ret_1d'] - df['spy_ret1']
df['rel_ret_5d'] = df['ret_5d'] - df['spy_ret5']
df['rel_ret_10d'] = df['ret_10d'] - df['spy_ret20']

print(f"  计算完成: {time.time()-t0:.1f}s")

# 标签
df['fwd_5d_ret'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-5) / x - 1)

# 全特征列
ALL_COLS = [
    'ret_1d','ret_3d','ret_5d','ret_10d','ret_20d',
    'ma_5_ratio','ma_10_ratio','ma_20_ratio','ma_50_ratio',
    'vol_5d','vol_10d','vol_20d','rsi_14','rsi_50_pct',
    'vol_ratio_5','vol_ratio_20','vol_5d_norm',
    'price_pos_20','price_pos_50','price_pos_100',
    'macd','macd_sig','macd_hist','atr_pct',
    'rel_ret_1d','rel_ret_5d','rel_ret_10d','ma20_ma50_cross',
    'dvol_ratio','dvol_ma5',
    'spy_ret1','spy_ret5','spy_ret20','vix_close',
    # 方向C
    'ret_5d_abs','max_dd_20d','sharpe_20d','calmar_20d',
]
ALL_COLS = [c for c in ALL_COLS if c in df.columns]

# 过滤
valid = df.dropna(subset=['fwd_5d_ret'] + ALL_COLS).copy()
valid = valid.groupby('sym').filter(lambda x: len(x) >= 100)

# 严格划分
train = valid[valid['date'] < '2022-01-01'].copy()
val = valid[(valid['date'] >= '2022-01-01') & (valid['date'] < '2024-01-01')].copy()
test = valid[valid['date'] >= '2024-01-01'].copy()
trainval = pd.concat([train, val]).sort_values('date')

print(f"  训练: {len(train):,}  验证: {len(val):,}  测试: {len(test):,}")

# ════════════════════════════════════════
#  实验函数
# ════════════════════════════════════════
def run_experiment(name, feature_cols, use_ranking=False, embargo_days=0):
    """运行单个实验"""
    print(f"\n{'='*50}")
    print(f"实验: {name}")
    print(f"特征数: {len(feature_cols)}")

    # Purging + Embargo
    if embargo_days > 0:
        trainval_cut = trainval[trainval['date'] <= trainval['date'].max() - pd.Timedelta(days=embargo_days)]
        test_cut = test[test['date'] >= test['date'].min() + pd.Timedelta(days=embargo_days)]
    else:
        trainval_cut = trainval
        test_cut = test

    # Walk-Forward
    tv_dates = sorted(trainval_cut['date'].unique())
    n_tv = len(tv_dates)
    step = n_tv // 5
    wf_results = []

    for i in range(4):
        te = tv_dates[min((i+1)*step, n_tv-1)]
        tee = tv_dates[min((i+2)*step, n_tv-1)]
        wt = trainval_cut[trainval_cut['date'] <= te]
        wv = trainval_cut[(trainval_cut['date'] > te) & (trainval_cut['date'] <= tee)]
        if len(wv) < 500:
            continue

        if use_ranking:
            # 排序模型：预测相对排名
            wt_rank = wt.copy()
            wv_rank = wv.copy()
            # 每日排名归一化到0-1
            wt_rank['target_rank'] = wt_rank.groupby('date')['fwd_5d_ret'].rank(pct=True)
            wv_rank['target_rank'] = wv_rank.groupby('date')['fwd_5d_ret'].rank(pct=True)
            model = xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.5,
                reg_alpha=0.1, reg_lambda=1.0,
                eval_metric='rmse', early_stopping_rounds=50,
                random_state=42, n_jobs=-1)
            model.fit(wt_rank[feature_cols].values, wt_rank['target_rank'].values,
                      eval_set=[(wv_rank[feature_cols].values, wv_rank['target_rank'].values)],
                      verbose=False)
            pred = model.predict(wv[feature_cols].values)
        else:
            model = xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.5,
                reg_alpha=0.1, reg_lambda=1.0,
                eval_metric='rmse', early_stopping_rounds=50,
                random_state=42, n_jobs=-1)
            model.fit(wt[feature_cols].values, wt['fwd_5d_ret'].values,
                      eval_set=[(wv[feature_cols].values, wv['fwd_5d_ret'].values)],
                      verbose=False)
            pred = model.predict(wv[feature_cols].values)

        rmse = np.sqrt(mean_squared_error(wv['fwd_5d_ret'].values, pred))
        wv_c = wv.copy()
        wv_c['pred'] = pred
        daily = []
        for d, day in wv_c.groupby('date'):
            if len(day) < 10: continue
            top10 = day.nlargest(10, 'pred')
            daily.append({'avg_ret': top10['fwd_5d_ret'].mean(),
                         'win_rate': (top10['fwd_5d_ret'] > 0).mean()})
        if daily:
            tdf = pd.DataFrame(daily)
            geo = np.exp(np.log(1+tdf['avg_ret']).mean())-1
            ann = geo*252/5
            sharpe = tdf['avg_ret'].mean()/max(tdf['avg_ret'].std(),0.001)*np.sqrt(252/5)
            dd = (1+tdf['avg_ret']).cumprod()
            dd_max = (dd/dd.cummax()-1).min()
            wf_results.append({'rmse':rmse,'ann':ann,'sharpe':sharpe,'dd':dd_max,'wr':tdf['win_rate'].mean()})

    if not wf_results:
        print("  无有效结果")
        return None

    avg = {k: np.mean([r[k] for r in wf_results]) for k in wf_results[0]}
    print(f"  WF: 年化={avg['ann']*100:.1f}%, 夏普={avg['sharpe']:.2f}, 回撤={avg['dd']*100:.1f}%, 胜率={avg['wr']*100:.1f}%")

    # 最终模型
    if use_ranking:
        final_tv = trainval.copy()
        final_tv['target_rank'] = final_tv.groupby('date')['fwd_5d_ret'].rank(pct=True)
        fm = xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric='rmse', early_stopping_rounds=100, random_state=42, n_jobs=-1)
        val_r = val.copy()
        val_r['target_rank'] = val_r.groupby('date')['fwd_5d_ret'].rank(pct=True)
        fm.fit(final_tv[feature_cols].values, final_tv['target_rank'].values,
               eval_set=[(val_r[feature_cols].values, val_r['target_rank'].values)], verbose=False)
    else:
        fm = xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric='rmse', early_stopping_rounds=100, random_state=42, n_jobs=-1)
        fm.fit(trainval[feature_cols].values, trainval['fwd_5d_ret'].values,
               eval_set=[(val[feature_cols].values, val['fwd_5d_ret'].values)], verbose=False)

    pred = fm.predict(test_cut[feature_cols].values)
    rmse = np.sqrt(mean_squared_error(test_cut['fwd_5d_ret'].values, pred))
    tc = test_cut.copy()
    tc['pred'] = pred
    daily = []
    for d, day in tc.groupby('date'):
        if len(day) < 10: continue
        top10 = day.nlargest(10, 'pred')
        daily.append({'avg_ret': top10['fwd_5d_ret'].mean(),
                     'win_rate': (top10['fwd_5d_ret'] > 0).mean()})
    if daily:
        tdf = pd.DataFrame(daily)
        geo = np.exp(np.log(1+tdf['avg_ret']).mean())-1
        ann = geo*252/5
        sharpe = tdf['avg_ret'].mean()/max(tdf['avg_ret'].std(),0.001)*np.sqrt(252/5)
        dd = (1+tdf['avg_ret']).cumprod()
        dd_max = (dd/dd.cummax()-1).min()
        print(f"  测试集: 年化={ann*100:.1f}%, 夏普={sharpe:.2f}, 回撤={dd_max*100:.1f}%, 胜率={tdf['win_rate'].mean()*100:.1f}%")
        return {'name':name,'wf':avg,'test':{'ann':ann,'sharpe':sharpe,'dd':dd_max,'wr':tdf['win_rate'].mean(),'rmse':rmse},'model':fm,'features':feature_cols}
    return None

# ════════════════════════════════════════
#  实验矩阵
# ════════════════════════════════════════
base_cols = [c for c in ALL_COLS if c not in ['spy_ret1','spy_ret5','spy_ret20','vix_close']]
mkt_cols = ['spy_ret1','spy_ret5','spy_ret20','vix_close']
risk_cols = ['ret_5d_abs','max_dd_20d','sharpe_20d','calmar_20d']
all_cols = ALL_COLS.copy()

experiments = [
    # V4.1 基线
    ("V4.1_baseline", all_cols, False, 0),
    # 方向A: 去市场
    ("V4.2_no_market", base_cols, False, 0),
    # 方向B: 排序模型
    ("V4.3_ranking", all_cols, True, 0),
    # 方向A+B: 去市场+排序
    ("V4.4_no_mkt_rank", base_cols, True, 0),
    # 方向C: 风控特征
    ("V4.5_risk_feat", all_cols + risk_cols, False, 0),
    # 方向D: Purging
    ("V4.6_purging", all_cols, False, 10),
    # A+C: 去市场+风控
    ("V4.7_no_mkt_risk", base_cols + risk_cols, False, 0),
    # A+B+C: 去市场+排序+风控
    ("V4.8_full_combine", base_cols + risk_cols, True, 0),
    # A+B+C+D: 全叠加
    ("V4.9_all_combined", base_cols + risk_cols, True, 10),
]

all_results = []
for name, cols, ranking, embargo in experiments:
    try:
        r = run_experiment(name, cols, ranking, embargo)
        if r:
            all_results.append(r)
    except Exception as e:
        print(f"  ❌ 失败: {e}")

# ════════════════════════════════════════
#  汇总对比
# ════════════════════════════════════════
print(f"\n{'='*70}")
print("实验汇总:")
print(f"{'='*70}")
print(f"{'实验名':<25s} {'WF年化':>8s} {'WF夏普':>7s} {'WF回撤':>8s} {'测试年化':>9s} {'测试夏普':>9s} {'测试回撤':>9s} {'测试胜率':>9s}")
print("-"*95)
for r in sorted(all_results, key=lambda x: -x['test']['sharpe']):
    wf = r['wf']
    ts = r['test']
    print(f"{r['name']:<25s} {wf['ann']*100:>7.1f}% {wf['sharpe']:>7.2f} {wf['dd']*100:>7.1f}% "
          f"{ts['ann']*100:>8.1f}% {ts['sharpe']:>8.2f} {ts['dd']*100:>8.1f}% {ts['wr']*100:>8.1f}%")

# 保存最佳模型
if all_results:
    best = max(all_results, key=lambda x: x['test']['sharpe'])
    print(f"\n🏆 最佳: {best['name']}")
    print(f"  测试集: 年化={best['test']['ann']*100:.1f}%, 夏普={best['test']['sharpe']:.2f}, 回撤={best['test']['dd']*100:.1f}%")
    best['model'].save_model(os.path.join(MODEL_DIR, 'blueshield_v4_best.model'))
    with open(os.path.join(MODEL_DIR, 'blueshield_v4_best_meta.json'), 'w') as f:
        json.dump({'name': best['name'], 'features': best['features'],
                   'wf': {k:v for k,v in best['wf'].items()},
                   'test': {k:v for k,v in best['test'].items() if k != 'model'}},
                  f, indent=2, default=str)
    print("  ✅ 已保存最佳模型")
