# -*- coding: utf-8 -*-
"""
蓝盾V4 — 纯Raw数据版
从S&P 500 OHLCV直接计算V2的30个特征
严格划分：训练(2016-2021) / 验证(2022-2023) / 测试(2024-2026.6)
Walk-Forward 5折（训练+验证集）
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
#  1. 加载数据
# ════════════════════════════════════════
print("📊 加载S&P 500数据...")
df = pd.read_parquet(os.path.join(DATA_DIR, 'us_hist_sp500_10y.parquet'))
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['sym', 'date']).reset_index(drop=True)
print(f"  {len(df):,}行, {df['sym'].nunique()}只, {df['date'].min().date()}~{df['date'].max().date()}")

# ════════════════════════════════════════
#  2. 计算V2完整30特征
# ════════════════════════════════════════
print("\n🔧 计算V2特征（30维）...")

def compute_v2_features(group):
    g = group.copy().reset_index(drop=True)
    c = g['close'].values.astype(float)
    h = g['high'].values.astype(float)
    l = g['low'].values.astype(float)
    v = g['volume'].values.astype(float)
    cs = pd.Series(c)

    # ── 收益率 (5个) ──
    g['ret_1d'] = cs.pct_change()
    g['ret_3d'] = cs.pct_change(3)
    g['ret_5d'] = cs.pct_change(5)
    g['ret_10d'] = cs.pct_change(10)
    g['ret_20d'] = cs.pct_change(20)

    # ── 均线比例 (4个) ──
    ma5 = cs.rolling(5).mean()
    ma10 = cs.rolling(10).mean()
    ma20 = cs.rolling(20).mean()
    ma50 = cs.rolling(50).mean()
    g['ma_5_ratio'] = c / ma5.values
    g['ma_10_ratio'] = c / ma10.values
    g['ma_20_ratio'] = c / ma20.values
    g['ma_50_ratio'] = c / ma50.values

    # ── 波动率 (3个) ──
    g['vol_5d'] = cs.pct_change().rolling(5).std()
    g['vol_10d'] = cs.pct_change().rolling(10).std()
    g['vol_20d'] = cs.pct_change().rolling(20).std()

    # ── RSI (2个) ──
    delta = cs.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 0.001)
    g['rsi_14'] = 100 - 100 / (1 + rs)
    g['rsi_50_pct'] = (g['rsi_14'] - 50) / 50

    # ── 量比 (3个) ──
    vol5 = pd.Series(v).rolling(5).mean()
    vol20 = pd.Series(v).rolling(20).mean()
    g['vol_ratio_5'] = v / vol5.values
    g['vol_ratio_20'] = v / vol20.values
    vol5_ret = cs.pct_change().rolling(5).std()
    vol20_ret = cs.pct_change().rolling(20).std()
    v20r_safe = np.where(vol20_ret.values == 0, 0.001, vol20_ret.values)
    g['vol_5d_norm'] = vol5_ret.values / v20r_safe

    # ── 价格位置 (3个) ──
    for period, name in [(20, 'price_pos_20'), (50, 'price_pos_50'), (100, 'price_pos_100')]:
        hh = pd.Series(h).rolling(period).max().values
        ll = pd.Series(l).rolling(period).min().values
        rng = hh - ll
        rng = np.where(rng == 0, 0.001, rng)
        g[name] = (c - ll) / rng

    # ── MACD (3个) ──
    ema12 = cs.ewm(span=12).mean()
    ema26 = cs.ewm(span=26).mean()
    g['macd'] = ema12 - ema26
    g['macd_sig'] = g['macd'].ewm(span=9).mean()
    g['macd_hist'] = g['macd'] - g['macd_sig']

    # ── ATR (1个) ──
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    g['atr_pct'] = pd.Series(tr).rolling(20).mean() / c * 100

    # ── MA交叉 (1个) ──
    g['ma20_ma50_cross'] = ma20.values - ma50.values

    # ── 成交额 (2个) ──
    dollar_vol = c * v
    dvol_ma5 = pd.Series(dollar_vol).rolling(5).mean()
    g['dvol_ma5'] = dvol_ma5.values
    g['dvol_ratio'] = np.where(dvol_ma5.values > 0, dollar_vol / dvol_ma5.values, 1.0)

    return g

t0 = time.time()
results = []
for sym, group in df.groupby('sym'):
    results.append(compute_v2_features(group))
df = pd.concat(results, ignore_index=True)
print(f"  计算完成: {time.time()-t0:.1f}s")

# ════════════════════════════════════════
#  3. 添加市场特征
# ════════════════════════════════════════
print("\n📈 添加市场特征...")
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
df = df.merge(spy_df[['date', 'spy_ret1', 'spy_ret5', 'spy_ret20']], on='date', how='left')

vix_df = vix[['Close']].reset_index()
vix_df.columns = ['date', 'vix_close']
vix_df['date'] = pd.to_datetime(vix_df['date']).dt.tz_localize(None)
vix_df = vix_df.drop_duplicates(subset='date', keep='last')
df = df.merge(vix_df, on='date', how='left')

# 相对强度
df['rel_ret_1d'] = df['ret_1d'] - df['spy_ret1']
df['rel_ret_5d'] = df['ret_5d'] - df['spy_ret5']
df['rel_ret_10d'] = df['ret_10d'] - df['spy_ret20']  # 近似

print("  完成")

# ════════════════════════════════════════
#  4. 标签 + 严格划分
# ════════════════════════════════════════
print("\n🎯 标签 + 严格划分...")

df['fwd_5d_ret'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-5) / x - 1)

feature_cols = [
    'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d',
    'ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio', 'ma_50_ratio',
    'vol_5d', 'vol_10d', 'vol_20d', 'rsi_14', 'rsi_50_pct',
    'vol_ratio_5', 'vol_ratio_20', 'vol_5d_norm',
    'price_pos_20', 'price_pos_50', 'price_pos_100',
    'macd', 'macd_sig', 'macd_hist', 'atr_pct',
    'rel_ret_1d', 'rel_ret_5d', 'rel_ret_10d', 'ma20_ma50_cross',
    'dvol_ratio', 'dvol_ma5',
    'spy_ret1', 'spy_ret5', 'spy_ret20', 'vix_close',
]

valid = df.dropna(subset=['fwd_5d_ret'] + feature_cols).copy()
valid = valid.groupby('sym').filter(lambda x: len(x) >= 100)

# 严格划分
train = valid[valid['date'] < '2022-01-01'].copy()
val = valid[(valid['date'] >= '2022-01-01') & (valid['date'] < '2024-01-01')].copy()
test = valid[valid['date'] >= '2024-01-01'].copy()

print(f"  特征数: {len(feature_cols)}")
print(f"  训练集: {len(train):,}行 ({train['date'].min().date()}~{train['date'].max().date()})")
print(f"  验证集: {len(val):,}行 ({val['date'].min().date()}~{val['date'].max().date()})")
print(f"  测试集: {len(test):,}行 ({test['date'].min().date()}~{test['date'].max().date()})")

# ════════════════════════════════════════
#  5. Walk-Forward（训练+验证集）
# ════════════════════════════════════════
print("\n🔄 Walk-Forward验证...")

trainval = pd.concat([train, val]).sort_values('date')
tv_dates = sorted(trainval['date'].unique())
n_tv = len(tv_dates)
n_folds = 5
step = n_tv // n_folds

wf_results = []
for i in range(n_folds - 1):
    train_end = tv_dates[min((i + 1) * step, n_tv - 1)]
    test_end = tv_dates[min((i + 2) * step, n_tv - 1)]

    wf_train = trainval[trainval['date'] <= train_end].copy()
    wf_test = trainval[(trainval['date'] > train_end) & (trainval['date'] <= test_end)].copy()

    if len(wf_test) < 500:
        continue

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.5,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='rmse', early_stopping_rounds=50,
        random_state=42, n_jobs=-1
    )
    model.fit(
        wf_train[feature_cols].values, wf_train['fwd_5d_ret'].values,
        eval_set=[(wf_test[feature_cols].values, wf_test['fwd_5d_ret'].values)],
        verbose=False
    )

    pred = model.predict(wf_test[feature_cols].values)
    rmse = np.sqrt(mean_squared_error(wf_test['fwd_5d_ret'].values, pred))

    wf_test_copy = wf_test.copy()
    wf_test_copy['pred'] = pred
    top10_daily = []
    for d, day in wf_test_copy.groupby('date'):
        if len(day) < 10:
            continue
        top10 = day.nlargest(10, 'pred')
        avg_ret = top10['fwd_5d_ret'].mean()
        win_rate = (top10['fwd_5d_ret'] > 0).mean()
        top10_daily.append({'date': d, 'avg_ret': avg_ret, 'win_rate': win_rate})

    if top10_daily:
        tdf = pd.DataFrame(top10_daily)
        geo = np.exp(np.log(1 + tdf['avg_ret']).mean()) - 1
        ann = geo * 252 / 5
        sharpe = tdf['avg_ret'].mean() / max(tdf['avg_ret'].std(), 0.001) * np.sqrt(252/5)
        dd = (1 + tdf['avg_ret']).cumprod()
        dd_max = (dd / dd.cummax() - 1).min()
        wr = tdf['win_rate'].mean()
    else:
        ann, sharpe, dd_max, wr = 0, 0, 0, 0

    wf_results.append({'fold': i+1, 'rmse': rmse, 'annual_return': ann,
                       'sharpe': sharpe, 'max_drawdown': dd_max, 'win_rate': wr})
    print(f"  Fold {i+1}: RMSE={rmse*100:.2f}%, 年化={ann*100:.1f}%, "
          f"夏普={sharpe:.2f}, 回撤={dd_max*100:.1f}%, 胜率={wr*100:.1f}%")

print(f"\n  Walk-Forward平均:")
for k in ['rmse', 'annual_return', 'sharpe', 'max_drawdown', 'win_rate']:
    v = np.mean([r[k] for r in wf_results])
    label = {'rmse': 'RMSE', 'annual_return': '年化', 'sharpe': '夏普',
             'max_drawdown': '回撤', 'win_rate': '胜率'}[k]
    fmt = f"{v*100:.2f}%" if k != 'sharpe' else f"{v:.2f}"
    print(f"    {label}: {fmt}")

# ════════════════════════════════════════
#  6. 最终模型 + 测试集评估
# ════════════════════════════════════════
print(f"\n🎯 训练最终模型...")

final_model = xgb.XGBRegressor(
    n_estimators=800, max_depth=6, learning_rate=0.02,
    subsample=0.7, colsample_bytree=0.5,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric='rmse', early_stopping_rounds=100,
    random_state=42, n_jobs=-1
)
final_model.fit(
    trainval[feature_cols].values, trainval['fwd_5d_ret'].values,
    eval_set=[(val[feature_cols].values, val['fwd_5d_ret'].values)],
    verbose=200
)

print(f"\n📊 测试集最终评估（完全隔离）...")
pred = final_model.predict(test[feature_cols].values)
rmse = np.sqrt(mean_squared_error(test['fwd_5d_ret'].values, pred))

test_copy = test.copy()
test_copy['pred'] = pred
top10_daily = []
for d, day in test_copy.groupby('date'):
    if len(day) < 10:
        continue
    top10 = day.nlargest(10, 'pred')
    avg_ret = top10['fwd_5d_ret'].mean()
    win_rate = (top10['fwd_5d_ret'] > 0).mean()
    top10_daily.append({'date': d, 'avg_ret': avg_ret, 'win_rate': win_rate})

if top10_daily:
    tdf = pd.DataFrame(top10_daily)
    geo = np.exp(np.log(1 + tdf['avg_ret']).mean()) - 1
    ann = geo * 252 / 5
    sharpe = tdf['avg_ret'].mean() / max(tdf['avg_ret'].std(), 0.001) * np.sqrt(252/5)
    dd = (1 + tdf['avg_ret']).cumprod()
    dd_max = (dd / dd.cummax() - 1).min()
    wr = tdf['win_rate'].mean()

    print(f"\n  测试集结果:")
    print(f"    RMSE: {rmse*100:.2f}%")
    print(f"    年化: {ann*100:.1f}%")
    print(f"    夏普: {sharpe:.2f}")
    print(f"    最大回撤: {dd_max*100:.1f}%")
    print(f"    平均5日收益: {tdf['avg_ret'].mean()*100:.2f}%")
    print(f"    胜率: {wr*100:.1f}%")

    print(f"\n  对比蓝盾V3:")
    print(f"    V3: 年化+39.6%, 回撤~4%, 胜率48.4% (50只池70天)")
    print(f"    V4: 年化{ann*100:.1f}%, 回撤{dd_max*100:.1f}%, 胜率{wr*100:.1f}% (S&P500 2.5年)")

# ════════════════════════════════════════
#  7. 特征重要性
# ════════════════════════════════════════
print(f"\n📊 特征重要性 Top 15:")
importances = final_model.feature_importances_
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:15]:
    bar = '█' * int(imp * 200)
    print(f"  {name:25s} {imp:.4f} {bar}")

# ════════════════════════════════════════
#  8. 保存
# ════════════════════════════════════════
print(f"\n💾 保存模型...")
model_path = os.path.join(MODEL_DIR, 'blueshield_v4.model')
final_model.save_model(model_path)

meta = {
    'model': 'blueshield_v4',
    'version': 'V4',
    'strategy': 'V2_V3_fusion_regression',
    'features': feature_cols,
    'n_features': len(feature_cols),
    'data_split': {
        'train': f"2016-2021 ({len(train):,}行)",
        'val': f"2022-2023 ({len(val):,}行)",
        'test': f"2024-2026.6 ({len(test):,}行)",
    },
    'walk_forward': {
        'n_folds': 5,
        'avg_rmse': float(np.mean([r['rmse'] for r in wf_results])),
        'avg_annual_return': float(np.mean([r['annual_return'] for r in wf_results])),
        'avg_sharpe': float(np.mean([r['sharpe'] for r in wf_results])),
        'avg_max_drawdown': float(np.mean([r['max_drawdown'] for r in wf_results])),
        'avg_win_rate': float(np.mean([r['win_rate'] for r in wf_results])),
    },
    'test_set': {
        'rmse': float(rmse),
        'annual_return': float(ann),
        'sharpe': float(sharpe),
        'max_drawdown': float(dd_max),
        'win_rate': float(wr),
    },
    'date': '2026-06-18',
    'data_latest': str(valid['date'].max().date()),
}
meta_path = os.path.join(MODEL_DIR, 'blueshield_v4_meta.json')
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"  模型: {model_path}")
print(f"  元数据: {meta_path}")
print(f"\n✅ 蓝盾V4训练完成")
