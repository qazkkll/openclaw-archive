#!/usr/bin/env python3
"""
预计算XGBoost模型历史评分
===========================
为RL环境生成每只股票每天的模型评分。

输出: data/rl/model_scores_us.parquet
  columns: sym, date, bs_score (蓝盾), ga_score (绿箭)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.expanduser("~/.hermes/openclaw-archive")

# 从blueshield_v6_score.py复制的特征计算（保持一致）
MACRO_COLS = ['vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60',
              'qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60',
              'iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60']
FUND_COLS = ['pe_trailing','pe_forward','div_yield','beta']
TECH_FEATS = ['ma5','ma20','ma60','ma_bias20','ma_align','price_position',
    'ret1','ret5','ret20','ret60','momentum_6m','momentum_1m',
    'mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change',
    'rsi14','rsi_change','macd','macd_signal','macd_hist',
    'bb_std','bb_width','bb_pos','ret_quality']
ALL_FEATS = TECH_FEATS + MACRO_COLS + FUND_COLS


def compute_features(group):
    """计算技术特征（与blueshield_v6_score.py完全一致）"""
    g = group.sort_values('date').copy()
    c = g['close']
    g['ma5'] = c.rolling(5).mean()
    g['ma20'] = c.rolling(20).mean()
    g['ma60'] = c.rolling(60).mean()
    g['ma_bias20'] = (c - g['ma20']) / g['ma20']
    g['ma_align'] = ((c > g['ma5']).astype(int) + (g['ma5'] > g['ma20']).astype(int))
    mn60 = c.rolling(60).min()
    mx60 = c.rolling(60).max()
    g['price_position'] = (c - mn60) / (mx60 - mn60 + 1e-10)
    g['ret1'] = c.pct_change(1)
    g['ret5'] = c.pct_change(5)
    g['ret20'] = c.pct_change(20)
    g['ret60'] = c.pct_change(60)
    g['momentum_6m'] = c.pct_change(126)
    g['momentum_1m'] = c.pct_change(21)
    g['mom_divergence'] = g['momentum_1m'] - g['ret20']
    g['trend_accel'] = g['ret5'] - g['ret5'].shift(5)
    dr = c.pct_change(1)
    g['vol20'] = dr.rolling(20).std()
    g['vol5'] = dr.rolling(5).std()
    g['vol_ratio'] = g['volume'] / g['volume'].rolling(20).mean()
    g['vol_change'] = g['vol20'] / g['vol20'].shift(20)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    g['rsi14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    g['rsi_change'] = g['rsi14'].diff(5)
    e12 = c.ewm(span=12).mean()
    e26 = c.ewm(span=26).mean()
    g['macd'] = e12 - e26
    g['macd_signal'] = g['macd'].ewm(span=9).mean()
    g['macd_hist'] = g['macd'] - g['macd_signal']
    g['bb_std'] = c.rolling(20).std()
    g['bb_width'] = 2 * g['bb_std'] / g['ma20']
    g['bb_pos'] = (c - g['ma20']) / (2 * g['bb_std'] + 1e-10)
    g['ret_quality'] = g['ret20'] / (g['vol20'] + 1e-10)
    # 绿箭额外特征
    g['price'] = c
    g['range_pct'] = (g['high'] - g['low']) / (c + 1e-10)
    return g


def main():
    # 1. 加载美股数据
    print("📊 加载数据...")
    hist_path = os.path.join(ROOT, "data/us/us_hist_yf_10y.parquet")
    df = pd.read_parquet(hist_path)
    # 兼容列名（parquet里是ticker不是sym）
    if 'ticker' in df.columns and 'sym' not in df.columns:
        df = df.rename(columns={'ticker': 'sym'})
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['close'] > 0.5) & (df['close'] < 10000) & (df['volume'] > 0)]
    print(f"  原始: {len(df)} rows, {df['sym'].nunique()} stocks")

    # 只保留RL测试的10只股票（加速）
    target_stocks = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "JNJ", "XOM", "UNH"]
    df = df[df['sym'].isin(target_stocks)].copy()
    print(f"  目标: {len(df)} rows, {df['sym'].nunique()} stocks")

    # 2. 计算技术特征
    print("🔧 计算特征...")
    t0 = time.time()
    parts = []
    for sym, g in df.groupby('sym'):
        f = compute_features(g)
        f['sym'] = sym
        parts.append(f)
    df = pd.concat(parts, ignore_index=True)
    print(f"  完成: {time.time()-t0:.0f}s")

    # 3. 加宏观特征
    print("📈 加载宏观特征...")
    try:
        v75 = pd.read_parquet(os.path.join(ROOT, "data/us/features/us_ml_feats_v75_filtered.parquet"))
        v75['date'] = pd.to_datetime(v75['date'])
        macro_daily = v75[['date'] + MACRO_COLS].drop_duplicates(subset=['date'])
        df = pd.merge(df, macro_daily, on='date', how='left')
        for col in MACRO_COLS:
            if col in df.columns:
                df[col] = df[col].ffill().fillna(0)
        print(f"  宏观特征已合并")
    except Exception as e:
        print(f"  ⚠️ 宏观特征加载失败: {e}, 填充0")
        for col in MACRO_COLS:
            df[col] = 0

    # 4. 加基本面特征
    print("📊 加载基本面特征...")
    try:
        v75 = pd.read_parquet(os.path.join(ROOT, "data/us/features/us_ml_feats_v75_filtered.parquet"))
        v75['date'] = pd.to_datetime(v75['date'])
        fund_daily = v75[['sym', 'date'] + FUND_COLS]
        df = pd.merge(df, fund_daily, on=['sym', 'date'], how='left')
        for col in FUND_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        print(f"  基本面特征已合并")
    except Exception as e:
        print(f"  ⚠️ 基本面特征加载失败: {e}, 填充0")
        for col in FUND_COLS:
            df[col] = 0

    # 5. 加载蓝盾模型
    print("🛡️ 加载蓝盾V6模型...")
    bs_model_path = os.path.join(ROOT, "models/us/blueshield_v6_xgb.json")
    bs_meta_path = os.path.join(ROOT, "models/us/blueshield_v6_meta.json")

    if os.path.exists(bs_model_path) and os.path.exists(bs_meta_path):
        bs_model = xgb.Booster()
        bs_model.load_model(bs_model_path)
        with open(bs_meta_path) as f:
            bs_meta = json.load(f)
        bs_features = bs_meta['features']

        # 为每行生成评分
        print(f"  生成评分 ({len(df)} rows)...")
        # 按股票分组预测（避免内存问题）
        scores = []
        for sym in df['sym'].unique():
            mask = df['sym'] == sym
            X = df.loc[mask, bs_features].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
            dtest = xgb.DMatrix(X, feature_names=bs_features)
            pred = bs_model.predict(dtest)
            scores.extend(pred)
        df['bs_score'] = scores
        print(f"  蓝盾评分完成: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
    else:
        print(f"  ⚠️ 蓝盾模型不存在，跳过")
        df['bs_score'] = 0.0

    # 6. 加载绿箭模型
    print("🏹 加载绿箭V11模型...")
    ga_model_path = os.path.join(ROOT, "models/us/arrow_v11_xgb.json")
    ga_meta_path = os.path.join(ROOT, "models/us/arrow_v11_meta.json")

    if os.path.exists(ga_model_path) and os.path.exists(ga_meta_path):
        ga_model = xgb.Booster()
        ga_model.load_model(ga_model_path)
        with open(ga_meta_path) as f:
            ga_meta = json.load(f)
        ga_features = ga_meta['features']

        scores = []
        for sym in df['sym'].unique():
            mask = df['sym'] == sym
            X = df.loc[mask, ga_features].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
            dtest = xgb.DMatrix(X, feature_names=ga_features)
            pred = ga_model.predict(dtest)
            scores.extend(pred)
        df['ga_score'] = scores
        print(f"  绿箭评分完成: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
    else:
        print(f"  ⚠️ 绿箭模型不存在，跳过")
        df['ga_score'] = 0.0

    # 7. 保存
    output_path = os.path.join(ROOT, "data/rl/model_scores_us.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_df = df[['sym', 'date', 'bs_score', 'ga_score']].copy()
    save_df = save_df.sort_values(['sym', 'date']).reset_index(drop=True)
    save_df.to_parquet(output_path, index=False)

    print(f"\n✅ 保存: {output_path}")
    print(f"   行数: {len(save_df)}")
    print(f"   股票: {save_df['sym'].nunique()}")
    print(f"   日期: {save_df['date'].min().date()} ~ {save_df['date'].max().date()}")
    print(f"\n蓝盾评分分布:")
    print(save_df['bs_score'].describe())
    print(f"\n绿箭评分分布:")
    print(save_df['ga_score'].describe())


if __name__ == "__main__":
    main()
