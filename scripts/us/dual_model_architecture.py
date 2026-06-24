#!/usr/bin/env python3
"""
Regime-Adaptive Dual Model Architecture
解决核心问题：当前模型62%靠宏观、Top10全是大盘方向，无法区分个股。

新架构：分离「市场方向判断」和「个股选择」为两个独立模型。

┌─────────────────────────────────────────────────────┐
│                    输入数据层                          │
│  价格数据(11,864只) + 基本面 + 新闻情绪 + 板块ETF     │
└────────────┬────────────────────────────┬────────────┘
             │                            │
    ┌────────▼────────┐          ┌────────▼────────┐
    │  Regime Detector │          │  Feature Layer  │
    │  (市场方向判断)    │          │  (特征工程)      │
    │                  │          │                  │
    │  输入:            │          │  输入:            │
    │  - SPY/QQQ/IWM   │          │  - 价格技术指标   │
    │  - VIX           │          │  - 板块相对强度   │
    │  - 利率/国债      │          │  - 截面排名       │
    │  - 板块轮动       │          │  - 新闻情绪      │
    │                  │          │  - 基本面(如有)   │
    │  输出:            │          │                  │
    │  - bull/bear/    │          │  输出:            │
    │    sideways      │          │  - 股票排名分数   │
    │  - 置信度        │          │                  │
    └────────┬────────┘          └────────┬────────┘
             │                            │
    ┌────────▼────────────────────────────▼────────┐
    │              Signal Generator                  │
    │  根据regime调整选股策略:                        │
    │                                                │
    │  BULL(大盘向上):                               │
    │    - 买动量强+板块领先的股票                    │
    │    - 止损宽松(-15%)，持有期长(20天)             │
    │                                                │
    │  BEAR(大盘向下):                               │
    │    - 买相对强度高(跌得少)的股票                 │
    │    - 买估值低(PE低)的防御股                     │
    │    - 止损严格(-8%)，持有期短(10天)              │
    │                                                │
    │  SIDEWAYS(横盘):                               │
    │    - 买均值回归(超跌反弹)的股票                 │
    │    - 小仓位，快进快出(5天)                      │
    └───────────────────────────────────────────────┘

关键设计原则:
1. Regime detector只用宏观特征(13个) → 判断大盘方向
2. Stock selector不用宏观特征 → 只用个股/板块/情绪特征
3. 两者完全分离，避免宏观信号污染选股
4. 每月重训练，自动适应市场变化

用法:
    python3 dual_model_architecture.py --train    # 训练双模型
    python3 dual_model_architecture.py --score    # 用双模型评分
    python3 dual_model_architecture.py --validate # 验证架构有效性
"""
import json, os, sys, time, argparse, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 特征定义
# ============================================================

# Regime detector特征：只用宏观指标
REGIME_FEATURES = [
    'vix_close', 'vix_ma20', 'vix_change',
    'spy_ret1', 'spy_ret5', 'spy_ret20', 'spy_ret60',
    'qqq_ret1', 'qqq_ret5', 'qqq_ret20', 'qqq_ret60',
    'iwm_ret1', 'iwm_ret5', 'iwm_ret20', 'iwm_ret60',
    'spy_vs_ma20', 'spy_vs_ma60',  # 大盘相对均线位置
    'sector_rotation',              # 板块轮动指标
]

# Stock selector特征：不用宏观，只用个股/板块/情绪
STOCK_FEATURES_TECH = [
    'ma5', 'ma20', 'ma60', 'ma_bias20', 'ma_align', 'price_position',
    'ret1', 'ret5', 'ret20', 'ret60',
    'momentum_6m', 'momentum_1m', 'mom_divergence', 'trend_accel',
    'vol20', 'vol5', 'vol_ratio', 'vol_change',
    'rsi14', 'rsi_change',
    'macd', 'macd_signal', 'macd_hist',
    'bb_std', 'bb_width', 'bb_pos', 'ret_quality',
]

STOCK_FEATURES_CROSS = [
    'sector_ret5', 'sector_ret20',     # 板块相对强度
    'cs_rank_ret20', 'cs_rank_vol',    # 截面排名
    'vol_anomaly', 'vol_regime',        # 成交量异常/波动率regime
    'beat_mkt_5d', 'beat_mkt_10d',     # 连续跑赢市场天数
]

STOCK_FEATURES_FUND = [
    'pe_trailing', 'pe_forward', 'div_yield', 'beta',
]

# 所有stock selector特征（不含宏观）
STOCK_FEATURES = STOCK_FEATURES_TECH + STOCK_FEATURES_CROSS + STOCK_FEATURES_FUND


def compute_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算增强版特征（在原始compute_features基础上加截面特征）"""
    df = df.sort_values(['sym', 'date']).copy()
    
    # 原始技术特征（从blueshield_v6_score.py复制）
    c = df['close']
    df['ma5'] = c.rolling(5).mean()
    df['ma20'] = c.rolling(20).mean()
    df['ma60'] = c.rolling(60).mean()
    df['ma_bias20'] = (c - df['ma20']) / df['ma20']
    df['ma_align'] = (df['ma5'] > df['ma20']).astype(int) + (df['ma20'] > df['ma60']).astype(int)
    df['price_position'] = (c - df['ma60']) / (df['ma60'] + 1e-10)
    df['ret1'] = c.pct_change(1)
    df['ret5'] = c.pct_change(5)
    df['ret20'] = c.pct_change(20)
    df['ret60'] = c.pct_change(60)
    df['momentum_6m'] = c.pct_change(126)
    df['momentum_1m'] = c.pct_change(21)
    df['mom_divergence'] = df['momentum_6m'] - df['momentum_1m']
    df['trend_accel'] = df['ret5'] - df['ret20'] / 4
    
    v = df['volume']
    df['vol20'] = c.pct_change(1).rolling(20).std()
    df['vol5'] = c.pct_change(1).rolling(5).std()
    df['vol_ratio'] = v / v.rolling(20).mean()
    df['vol_change'] = df['vol_ratio'].diff(5)
    
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi14'] = 100 - 100 / (1 + rs)
    df['rsi_change'] = df['rsi14'].diff(5)
    
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    df['bb_std'] = c.pct_change(1).rolling(20).std()
    df['bb_width'] = 4 * df['bb_std']
    sma20 = c.rolling(20).mean()
    df['bb_pos'] = (c - sma20) / (2 * c.rolling(20).std() + 1e-10)
    
    df['ret_quality'] = df['ret5'].rolling(20).apply(lambda x: (x > 0).sum() / len(x), raw=True)
    
    return df


def compute_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算截面特征：板块相对强度、截面排名、连续跑赢天数"""
    df = df.sort_values(['sym', 'date']).copy()
    
    # 日收益
    df['daily_ret'] = df.groupby('sym')['close'].pct_change(1)
    
    # 市场平均日收益
    df['mkt_daily_ret'] = df.groupby('date')['daily_ret'].transform('mean')
    df['excess_daily'] = df['daily_ret'] - df['mkt_daily_ret']
    
    # 板块相对强度：个股收益 - 市场平均收益
    # 用SPY作为市场基准（因为板块ETF映射需要额外数据）
    spy = df[df['sym'] == 'SPY'][['date', 'daily_ret']].rename(columns={'daily_ret': 'spy_ret'})
    df = pd.merge(df, spy, on='date', how='left')
    df['sector_ret5'] = df['ret5'] - df['spy_ret'].rolling(5).sum()
    df['sector_ret20'] = df['ret20'] - df['spy_ret'].rolling(20).sum()
    
    # 截面排名
    df['cs_rank_ret20'] = df.groupby('date')['ret20'].rank(pct=True)
    df['cs_rank_vol'] = df.groupby('date')['vol_ratio'].rank(pct=True)
    
    # 成交量异常
    df['vol_anomaly'] = df['vol_ratio']
    df['vol_regime'] = df['vol5'] / (df['vol20'] + 1e-10)
    
    # 连续跑赢市场天数
    df['beat_mkt_5d'] = df.groupby('sym')['excess_daily'].transform(
        lambda x: x.rolling(5).apply(lambda y: (y > 0).sum() / len(y), raw=True))
    df['beat_mkt_10d'] = df.groupby('sym')['excess_daily'].transform(
        lambda x: x.rolling(10).apply(lambda y: (y > 0).sum() / len(y), raw=True))
    
    return df


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算regime detector特征"""
    spy = df[df['sym'] == 'SPY'].copy()
    if len(spy) == 0:
        return df
    
    # VIX相关（如果有）
    if 'vix_close' not in df.columns:
        # 用SPY波动率作为VIX代理
        spy_vol = spy['close'].pct_change(1).rolling(20).std() * np.sqrt(252) * 100
        df['vix_close'] = np.nan  # placeholder
    
    # SPY相对均线
    spy_c = spy['close']
    spy_ma20 = spy_c.rolling(20).mean()
    spy_ma60 = spy_c.rolling(60).mean()
    spy_latest = spy.sort_values('date').iloc[-1]
    
    # 板块轮动：用SPY和IWM的相对强弱
    iwm = df[df['sym'] == 'IWM']
    if len(iwm) > 0:
        sector_rotation = spy_c.pct_change(20).iloc[-1] - iwm['close'].pct_change(20).iloc[-1]
    else:
        sector_rotation = 0
    
    return df


def train_regime_model(df: pd.DataFrame, labels: pd.Series) -> xgb.Booster:
    """训练regime判断模型"""
    params = {
        'max_depth': 4,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 50,
    }
    dtrain = xgb.DMatrix(df, label=labels)
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model


def train_stock_model(df: pd.DataFrame, labels: pd.Series, features: list) -> xgb.Booster:
    """训练个股选择模型（不用宏观特征）"""
    params = {
        'max_depth': 6,
        'eta': 0.05,
        'objective': 'rank:pairwise',
        'eval_metric': 'auc',
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'min_child_weight': 20,
        'lambda': 1.0,
    }
    dtrain = xgb.DMatrix(df[features], label=labels)
    model = xgb.train(params, dtrain, num_boost_round=200)
    return model


def validate_architecture():
    """验证双模型架构是否比单模型更好"""
    print('=== 双模型架构验证 ===')
    print('加载数据...')
    
    df = pd.read_parquet(os.path.join(ROOT, 'data/us/us_hist_full_10y.parquet'))
    df = df.dropna(subset=['close', 'volume'])
    df = df[(df['close'] > 0.5) & (df['close'] < 10000) & (df['volume'] > 0)]
    cutoff = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
    df = df[df['date'] >= cutoff]
    
    # 计算特征
    print('计算特征...')
    parts = []
    for sym, g in df.groupby('sym'):
        f = compute_enhanced_features(g)
        f['sym'] = sym
        parts.append(f)
    df = pd.concat(parts, ignore_index=True)
    
    # 计算截面特征
    df = compute_cross_sectional_features(df)
    
    # 前瞻收益
    df = df.sort_values(['sym', 'date'])
    df['fwd_5d'] = df.groupby('sym')['close'].shift(-5) / df['close'] - 1
    
    # 过滤
    df = df[(df['close'] > 10) & (df['volume'] > 50000)]
    
    # 只用有完整特征的行
    available_stock_features = [f for f in STOCK_FEATURES if f in df.columns]
    df = df.dropna(subset=available_stock_features + ['fwd_5d'])
    
    print(f'数据: {len(df)}行, {df["sym"].nunique()}只股票')
    
    # Walk-forward验证
    dates = sorted(df['date'].unique())
    n_folds = 6
    fold_size = len(dates) // n_folds
    
    results = []
    
    for fold in range(n_folds - 1):  # 最后一个fold只做测试
        train_end = dates[(fold + 1) * fold_size]
        test_start = train_end
        test_end = dates[min((fold + 2) * fold_size, len(dates) - 1)]
        
        train_df = df[df['date'] < train_end]
        test_df = df[(df['date'] >= test_start) & (df['date'] < test_end)]
        
        if len(train_df) < 1000 or len(test_df) < 100:
            continue
        
        # 训练stock selector（不用宏观特征）
        X_train = train_df[available_stock_features].values
        y_train = train_df['fwd_5d'].values
        dtrain = xgb.DMatrix(X_train, feature_names=available_stock_features)
        
        # 创建排名标签（每天内按fwd_5d排名）
        train_df = train_df.copy()
        train_df['rank_label'] = train_df.groupby('date')['fwd_5d'].rank(pct=True)
        
        params = {
            'max_depth': 6, 'eta': 0.05,
            'objective': 'rank:pairwise',
            'eval_metric': 'auc',
            'subsample': 0.8, 'colsample_bytree': 0.6,
            'min_child_weight': 20, 'lambda': 1.0,
        }
        dtrain = xgb.DMatrix(X_train, feature_names=available_stock_features, 
                            label=train_df['rank_label'].values)
        model = xgb.train(params, dtrain, num_boost_round=100)
        
        # 预测
        X_test = test_df[available_stock_features].values
        dtest = xgb.DMatrix(X_test, feature_names=available_stock_features)
        test_df = test_df.copy()
        test_df['pred'] = model.predict(dtest)
        
        # 评估Top5%
        test_df['pred_rank'] = test_df.groupby('date')['pred'].rank(pct=True)
        top5 = test_df[test_df['pred_rank'] > 0.95]
        rest = test_df[test_df['pred_rank'] <= 0.95]
        
        r5_top = top5['fwd_5d'].mean() * 100
        wr_top = (top5['fwd_5d'] > 0).mean() * 100
        r5_rest = rest['fwd_5d'].mean() * 100
        
        # 特征重要性
        importance = model.get_score(importance_type='gain')
        
        results.append({
            'fold': fold + 1,
            'train_end': str(train_end)[:10],
            'test_period': f'{str(test_start)[:10]} ~ {str(test_end)[:10]}',
            'top5_return': r5_top,
            'top5_winrate': wr_top,
            'rest_return': r5_rest,
            'excess': r5_top - r5_rest,
            'top5_n': len(top5),
        })
        
        print(f'\\nFold {fold+1} ({results[-1]["test_period"]}):')
        print(f'  Top5%: {r5_top:+.2f}% win={wr_top:.1f}% (n={len(top5)})')
        print(f'  其余: {r5_rest:+.2f}%')
        print(f'  超额: {r5_top-r5_rest:+.2f}%')
        
        # 显示feature importance
        top_fi = sorted(importance.items(), key=lambda x: -x[1])[:5]
        print(f'  Top5特征: {", ".join(k for k,v in top_fi)}')
    
    # 汇总
    if results:
        avg_excess = np.mean([r['excess'] for r in results])
        avg_winrate = np.mean([r['top5_winrate'] for r in results])
        print(f'\\n=== 汇总 ===')
        print(f'平均超额收益: {avg_excess:+.2f}%')
        print(f'平均Top5%胜率: {avg_winrate:.1f}%')
        print(f'vs 原模型Top5%胜率: 49.6%')
        print(f'提升: {avg_winrate - 49.6:+.1f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--score', action='store_true')
    args = parser.parse_args()
    
    if args.validate:
        validate_architecture()
    else:
        print('用法: python3 dual_model_architecture.py --validate')
