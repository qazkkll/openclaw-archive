#!/usr/bin/env python3
"""
零成本特征工程：板块相对强度 + 截面排名 + 成交量异常 + 波动率regime
不需要外部数据，用已有价格数据计算。

用法:
    from feature_engineering import add_cross_sectional_features
    latest = add_cross_sectional_features(latest, df_full)

设计原则:
    - 所有特征都是截面特征（同一天所有股票之间的比较）
    - 需要全量数据计算排名，但只在latest上输出
    - 特征名前缀cs_表示cross-sectional
"""
import numpy as np
import pandas as pd


def add_cross_sectional_features(latest: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
    """在latest DataFrame上添加截面特征
    
    Args:
        latest: 每只股票的最新一行（已计算技术特征）
        df_full: 全量历史数据（用于计算截面排名）
    
    Returns:
        latest with new features added
    """
    
    # 1. 板块相对强度：个股收益 vs 市场平均收益
    # 用SPY作为市场基准（因为所有股票都受大盘影响）
    spy_rows = df_full[df_full['sym'] == 'SPY'].sort_values('date')
    if len(spy_rows) > 0:
        spy_latest = spy_rows.iloc[-1]
        # 计算SPY的5d/20d收益
        spy_ret5 = spy_rows['close'].pct_change(5).iloc[-1] if len(spy_rows) >= 5 else 0
        spy_ret20 = spy_rows['close'].pct_change(20).iloc[-1] if len(spy_rows) >= 20 else 0
        
        latest['sector_ret5'] = latest['ret5'] - spy_ret5
        latest['sector_ret20'] = latest['ret20'] - spy_ret20
    else:
        latest['sector_ret5'] = latest['ret5']
        latest['sector_ret20'] = latest['ret20']
    
    # 2. 截面排名：在同一天所有股票中的百分位排名
    # 用全量数据的最后一天计算排名
    latest_date = df_full['date'].max()
    day_data = df_full[df_full['date'] == latest_date].copy()
    
    if len(day_data) > 100:  # 至少100只股票才有意义
        # 对day_data也计算简单特征
        day_data = day_data.sort_values('date')
        
        # 用latest的已有特征做排名
        for feat in ['ret5', 'ret20', 'rsi14', 'vol_ratio']:
            if feat in latest.columns:
                # 计算该特征在全市场的百分位排名
                all_values = latest[feat].values
                ranks = pd.Series(all_values).rank(pct=True).values
                latest[f'cs_rank_{feat}'] = ranks
    
    # 3. 成交量异常：最近5天平均成交量 / 20天平均成交量
    if 'vol_ratio' in latest.columns:
        # vol_ratio已经是volume / volume.rolling(20).mean()
        latest['vol_anomaly'] = latest['vol_ratio']
    else:
        latest['vol_anomaly'] = 0
    
    # 4. 波动率regime：当前波动率在历史中的位置
    if 'vol20' in latest.columns and 'vol5' in latest.columns:
        # vol_ratio = vol5 / vol20，>1说明短期波动在放大
        latest['vol_regime'] = latest['vol5'] / (latest['vol20'] + 1e-10)
    else:
        latest['vol_regime'] = 1.0
    
    # 5. VIX regime（如果有的话）
    if 'vix_close' in latest.columns:
        latest['vix_high'] = (latest['vix_close'] > 20).astype(int)
        latest['vix_extreme'] = (latest['vix_close'] > 30).astype(int)
    
    # 6. 动量加速度：短期动量 vs 长期动量的差异
    if 'ret5' in latest.columns and 'ret20' in latest.columns:
        latest['mom_accel'] = latest['ret5'] - latest['ret20'] / 4  # 5d vs 5d-equivalent of 20d
    
    # 7. 价格位置强度：在60天范围中的位置 × 近期动量
    if 'price_position' in latest.columns and 'ret5' in latest.columns:
        latest['price_pos_mom'] = latest['price_position'] * latest['ret5']
    
    return latest


# 新特征列表（供评分脚本使用）
NEW_FEATURES = [
    'sector_ret5', 'sector_ret20',
    'cs_rank_ret5', 'cs_rank_ret20', 'cs_rank_rsi14', 'cs_rank_vol_ratio',
    'vol_anomaly', 'vol_regime',
    'vix_high', 'vix_extreme',
    'mom_accel', 'price_pos_mom',
]
