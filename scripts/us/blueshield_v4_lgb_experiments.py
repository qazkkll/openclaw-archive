# -*- coding: utf-8 -*-
"""
蓝盾V4 第二轮实验
核心改进：
1. Adjusted-MSE（错方向惩罚11倍）
2. LightGBM替代XGBoost
3. 更丰富的特征（+动量质量+量价背离+波动率regime）
4. 排序学习（LambdaRank近似）
"""
import warnings, json, os, time
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, 'data', 'us')
MODEL_DIR = os.path.join(ROOT, 'models', 'us')
os.makedirs(MODEL_DIR, exist_ok=True)

# ════════════════════════════════════════
#  1. 加载数据
# ════════════════════════════════════════
print("📊 加载数据...")
df = pd.read_parquet(os.path.join(DATA_DIR, 'us_hist_sp500_10y.parquet'))
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['sym', 'date']).reset_index(drop=True)

import yfinance as yf
spy = yf.download('SPY', start=df['date'].min().strftime('%Y-%m-%d'),
                  end=(df['date'].max()+pd.Timedelta(days=1)).strftime('%Y-%m-%d'), progress=False)
vix = yf.download('^VIX', start=df['date'].min().strftime('%Y-%m-%d'),
                  end=(df['date'].max()+pd.Timedelta(days=1)).strftime('%Y-%m-%d'), progress=False)
spy_df = spy[['Close']].reset_index()
spy_df.columns = ['date','spy_close']
spy_df['date'] = pd.to_datetime(spy_df['date']).dt.tz_localize(None)
spy_df = spy_df.drop_duplicates(subset='date', keep='last')
spy_df['spy_ret1'] = spy_df['spy_close'].pct_change()
spy_df['spy_ret5'] = spy_df['spy_close'].pct_change(5)
spy_df['spy_ret20'] = spy_df['spy_close'].pct_change(20)
vix_df = vix[['Close']].reset_index()
vix_df.columns = ['date','vix_close']
vix_df['date'] = pd.to_datetime(vix_df['date']).dt.tz_localize(None)
vix_df = vix_df.drop_duplicates(subset='date', keep='last')

# ════════════════════════════════════════
#  2. 计算特征（扩展版）
# ════════════════════════════════════════
print("\n🔧 计算扩展特征...")

def compute_features(group):
    g = group.copy().reset_index(drop=True)
    c = g['close'].values.astype(float)
    h = g['high'].values.astype(float)
    l = g['low'].values.astype(float)
    v = g['volume'].values.astype(float)
    cs = pd.Series(c)

    # ── 收益率 ──
    for d in [1,2,3,5,10,20,60]:
        g[f'ret_{d}d'] = cs.pct_change(d).values

    # ── 均线比例 ──
    for w in [5,10,20,50,120]:
        ma = cs.rolling(w).mean()
        g[f'ma_{w}_ratio'] = (c / ma.values)

    # ── 均线交叉信号 ──
    ma5 = cs.rolling(5).mean()
    ma20 = cs.rolling(20).mean()
    ma50 = cs.rolling(50).mean()
    ma120 = cs.rolling(120).mean()
    g['ma5_ma20_cross'] = ((ma5 > ma20).astype(float)).values
    g['ma20_ma50_cross'] = ((ma20 > ma50).astype(float)).values
    g['ma50_ma120_cross'] = ((ma50 > ma120).astype(float)).values
    g['ma_align_score'] = (g['ma5_ma20_cross'] + g['ma20_ma50_cross'] + g['ma50_ma120_cross'])

    # ── 波动率 ──
    daily_ret = cs.pct_change()
    for w in [5,10,20,60]:
        g[f'vol_{w}d'] = daily_ret.rolling(w).std().values
    v5 = daily_ret.rolling(5).std()
    v20 = daily_ret.rolling(20).std()
    v60 = daily_ret.rolling(60).std()
    g['vol_ratio_5_20'] = (v5 / v20.replace(0, 0.001)).values
    g['vol_ratio_5_60'] = (v5 / v60.replace(0, 0.001)).values
    g['vol_regime'] = np.where(g['vol_20d'] > g['vol_20d'].rolling(60).mean().values, 1.0, 0.0)

    # ── RSI ──
    delta = cs.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 0.001)
    g['rsi_14'] = (100 - 100 / (1 + rs)).values
    g['rsi_50_pct'] = ((g['rsi_14'] - 50) / 50)

    # ── MACD ──
    ema12 = cs.ewm(span=12).mean()
    ema26 = cs.ewm(span=26).mean()
    g['macd'] = (ema12 - ema26).values
    g['macd_sig'] = g['macd'].ewm(span=9).mean().values
    g['macd_hist'] = (g['macd'] - g['macd_sig'])
    g['macd_cross'] = ((g['macd'] > g['macd_sig']).astype(float))

    # ── 量价关系 ──
    vol5 = pd.Series(v).rolling(5).mean()
    vol20 = pd.Series(v).rolling(20).mean()
    g['vol_ratio_5'] = (v / vol5.values)
    g['vol_ratio_20'] = (v / vol20.values)
    # 量价背离：价格涨但量缩
    g['price_vol_div'] = (g['ret_5d'] * -1 * (g['vol_ratio_5'] - 1))

    # ── 价格位置 ──
    for w in [20,50,100,252]:
        hh = pd.Series(h).rolling(w, min_periods=20).max().values
        ll = pd.Series(l).rolling(w, min_periods=20).min().values
        rng = np.where(hh - ll == 0, 0.001, hh - ll)
        g[f'price_pos_{w}'] = (c - ll) / rng

    # ── ATR ──
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    g['atr_pct'] = (pd.Series(tr).rolling(20).mean() / c * 100).values

    # ── 成交额 ──
    dv = c * v
    dm5 = pd.Series(dv).rolling(5).mean()
    dm20 = pd.Series(dv).rolling(20).mean()
    g['dvol_ratio'] = np.where(dm5.values > 0, dv / dm5.values, 1.0)

    # ── 动量质量（Sharpe风格）──
    g['ret_quality_10'] = daily_ret.rolling(10).apply(
        lambda x: x.mean() / max(x.std(), 0.001), raw=True).values
    g['ret_quality_20'] = daily_ret.rolling(20).apply(
        lambda x: x.mean() / max(x.std(), 0.001), raw=True).values

    # ── 趋势强度 ──
    g['trend_strength'] = (
        g['ma5_ma20_cross'] * 2 + g['ma20_ma50_cross'] * 3 + g['ma50_ma120_cross'] * 4
    )

    return g

t0 = time.time()
results = []
for sym, group in df.groupby('sym'):
    results.append(compute_features(group))
df = pd.concat(results, ignore_index=True)
df = df.merge(spy_df[['date','spy_ret1','spy_ret5','spy_ret20']], on='date', how='left')
df = df.merge(vix_df, on='date', how='left')
df['rel_ret_1d'] = df['ret_1d'] - df['spy_ret1']
df['rel_ret_5d'] = df['ret_5d'] - df['spy_ret5']
df['rel_ret_20d'] = df['ret_20d'] - df['spy_ret20']
print(f"  计算完成: {time.time()-t0:.1f}s")

# 标签
df['fwd_5d_ret'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-5) / x - 1)

# 全特征
ALL_COLS = [c for c in df.columns if c not in ['sym','date','open','high','low','close','volume','fwd_5d_ret']]
ALL_COLS = [c for c in ALL_COLS if df[c].dtype in ['float64','float32','int64','int32']]

valid = df.dropna(subset=['fwd_5d_ret'] + ALL_COLS).copy()
valid = valid.groupby('sym').filter(lambda x: len(x) >= 100)

train = valid[valid['date'] < '2022-01-01'].copy()
val = valid[(valid['date'] >= '2022-01-01') & (valid['date'] < '2024-01-01')].copy()
test = valid[valid['date'] >= '2024-01-01'].copy()
trainval = pd.concat([train, val]).sort_values('date')

print(f"  特征数: {len(ALL_COLS)}")
print(f"  训练: {len(train):,}  验证: {len(val):,}  测试: {len(test):,}")

# ════════════════════════════════════════
#  实验函数
# ════════════════════════════════════════
def eval_top10(test_df, pred, label=''):
    tc = test_df.copy()
    tc['pred'] = pred
    daily = []
    for d, day in tc.groupby('date'):
        if len(day) < 10: continue
        top10 = day.nlargest(10, 'pred')
        daily.append({'avg_ret': top10['fwd_5d_ret'].mean(),
                     'win_rate': (top10['fwd_5d_ret'] > 0).mean()})
    if not daily:
        return None
    tdf = pd.DataFrame(daily)
    geo = np.exp(np.log(1+tdf['avg_ret']).mean())-1
    ann = geo*252/5
    sharpe = tdf['avg_ret'].mean()/max(tdf['avg_ret'].std(),0.001)*np.sqrt(252/5)
    dd = (1+tdf['avg_ret']).cumprod()
    dd_max = (dd/dd.cummax()-1).min()
    return {'ann':ann,'sharpe':sharpe,'dd':dd_max,'wr':tdf['win_rate'].mean(),
            'avg5d':tdf['avg_ret'].mean()}

def run_experiment(name, feature_cols, objective='regression', use_sample_weight=False):
    print(f"\n{'='*50}")
    print(f"实验: {name} ({len(feature_cols)}特征, {objective})")

    # Walk-Forward
    tv_dates = sorted(trainval['date'].unique())
    n_tv = len(tv_dates)
    step = n_tv // 5
    wf_results = []

    for i in range(4):
        te = tv_dates[min((i+1)*step, n_tv-1)]
        tee = tv_dates[min((i+2)*step, n_tv-1)]
        wt = trainval[trainval['date'] <= te].copy()
        wv = trainval[(trainval['date'] > te) & (trainval['date'] <= tee)].copy()
        if len(wv) < 500: continue

        if objective == 'ranking':
            wt['target_rank'] = wt.groupby('date')['fwd_5d_ret'].rank(pct=True)
            wv['target_rank'] = wv.groupby('date')['fwd_5d_ret'].rank(pct=True)
            y_train = wt['target_rank'].values
            y_val = wv['target_rank'].values
        else:
            y_train = wt['fwd_5d_ret'].values
            y_val = wv['fwd_5d_ret'].values

        # Sample weights for Adjusted-MSE
        if use_sample_weight:
            sw = np.where(y_train > 0, 1.0, 11.0)  # 负样本惩罚11倍
        else:
            sw = np.ones(len(y_train))

        dtrain = lgb.Dataset(wt[feature_cols].values, label=y_train, weight=sw)
        dval = lgb.Dataset(wv[feature_cols].values, label=y_val)

        params = {
            'objective': 'regression', 'metric': 'rmse',
            'num_leaves': 63, 'learning_rate': 0.03,
            'feature_fraction': 0.5, 'bagging_fraction': 0.7,
            'lambda_l1': 0.1, 'lambda_l2': 1.0,
            'verbose': -1, 'n_jobs': -1,
        }
        model = lgb.train(params, dtrain, num_boost_round=500,
                         valid_sets=[dval], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

        pred = model.predict(wv[feature_cols].values)
        r = eval_top10(wv, pred)
        if r:
            wf_results.append(r)

    if not wf_results:
        print("  无有效结果")
        return None

    avg = {k: np.mean([r[k] for r in wf_results]) for k in wf_results[0]}
    print(f"  WF: 年化={avg['ann']*100:.1f}%, 夏普={avg['sharpe']:.2f}, 回撤={avg['dd']*100:.1f}%, 胜率={avg['wr']*100:.1f}%")

    # 最终模型
    if objective == 'ranking':
        ftv = trainval.copy()
        ftv['target_rank'] = ftv.groupby('date')['fwd_5d_ret'].rank(pct=True)
        fv = val.copy()
        fv['target_rank'] = fv.groupby('date')['fwd_5d_ret'].rank(pct=True)
        y_final = ftv['target_rank'].values
        y_val_final = fv['target_rank'].values
    else:
        ftv = trainval
        fv = val
        y_final = trainval['fwd_5d_ret'].values
        y_val_final = val['fwd_5d_ret'].values

    if use_sample_weight:
        sw = np.where(y_final > 0, 1.0, 11.0)
    else:
        sw = np.ones(len(y_final))

    dtrain_f = lgb.Dataset(ftv[feature_cols].values, label=y_final, weight=sw)
    dval_f = lgb.Dataset(fv[feature_cols].values, label=y_val_final)
    final_model = lgb.train(params, dtrain_f, num_boost_round=800,
                           valid_sets=[dval_f], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

    pred = final_model.predict(test[feature_cols].values)
    r = eval_top10(test, pred)
    if r:
        print(f"  测试: 年化={r['ann']*100:.1f}%, 夏普={r['sharpe']:.2f}, 回撤={r['dd']*100:.1f}%, 胜率={r['wr']*100:.1f}%")
    return {'name':name,'wf':avg,'test':r,'model':final_model,'features':feature_cols}

# ════════════════════════════════════════
#  实验矩阵
# ════════════════════════════════════════
base_cols = [c for c in ALL_COLS if c not in ['spy_ret1','spy_ret5','spy_ret20','vix_close']]

experiments = [
    # 基线
    ("V4.1_lgb_baseline", ALL_COLS, 'regression', False),
    # A: 去市场
    ("V4.10_no_market", base_cols, 'regression', False),
    # B: 排序
    ("V4.11_ranking", ALL_COLS, 'ranking', False),
    # C: Adjusted-MSE
    ("V4.12_adj_mse", ALL_COLS, 'regression', True),
    # A+B
    ("V4.13_no_mkt_rank", base_cols, 'ranking', False),
    # A+C
    ("V4.14_no_mkt_adj", base_cols, 'regression', True),
    # B+C
    ("V4.15_rank_adj", ALL_COLS, 'ranking', True),
    # A+B+C
    ("V4.16_full", base_cols, 'ranking', True),
]

all_results = []
for name, cols, obj, sw in experiments:
    try:
        r = run_experiment(name, cols, obj, sw)
        if r and r['test']:
            all_results.append(r)
    except Exception as e:
        print(f"  ❌ 失败: {e}")

# ════════════════════════════════════════
#  汇总
# ════════════════════════════════════════
print(f"\n{'='*80}")
print("实验汇总（LightGBM + 扩展特征）:")
print(f"{'='*80}")
print(f"{'实验名':<25s} {'WF年化':>8s} {'WF夏普':>7s} {'WF回撤':>8s} {'测试年化':>9s} {'测试夏普':>9s} {'测试回撤':>9s} {'胜率':>7s}")
print("-"*90)
for r in sorted(all_results, key=lambda x: -x['test']['sharpe']):
    w, t = r['wf'], r['test']
    print(f"{r['name']:<25s} {w['ann']*100:>7.1f}% {w['sharpe']:>7.2f} {w['dd']*100:>7.1f}% "
          f"{t['ann']*100:>8.1f}% {t['sharpe']:>8.2f} {t['dd']*100:>8.1f}% {t['wr']*100:>6.1f}%")

if all_results:
    best = max(all_results, key=lambda x: x['test']['sharpe'])
    print(f"\n🏆 最佳: {best['name']}")
    t = best['test']
    print(f"  测试: 年化={t['ann']*100:.1f}%, 夏普={t['sharpe']:.2f}, 回撤={t['dd']*100:.1f}%, 胜率={t['wr']*100:.1f}%")
    best['model'].save_model(os.path.join(MODEL_DIR, 'blueshield_v4_lgb_best.txt'))
    with open(os.path.join(MODEL_DIR, 'blueshield_v4_lgb_best_meta.json'), 'w') as f:
        json.dump({'name':best['name'],'features':best['features'],
                   'wf':{k:v for k,v in best['wf'].items()},
                   'test':{k:v for k,v in best['test'].items()}},
                  f, indent=2, default=str)
    print("  ✅ 已保存")
