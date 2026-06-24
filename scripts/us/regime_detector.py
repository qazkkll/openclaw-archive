#!/usr/bin/env python3
"""
市场Regime检测器
判断当前是「动量市」还是「反转市」，决定模型信号方向。

核心发现（2026-06-24验证）：
- 过去7个月中，反转效应出现4次，动量出现3次
- 反转市：买近期输家（Bottom20%），5d收益+0.90%，胜率56%
- 动量市：买近期赢家（Top20%），5d收益+0.40%，胜率50.5%
- 两种regime交替出现，不能固定一个方向

用法:
    from regime_detector import detect_regime, get_signal_direction
    regime = detect_regime(df_full)
    direction = get_signal_direction(regime)  # 'momentum' or 'reversal'
"""
import numpy as np
import pandas as pd


def detect_regime(df: pd.DataFrame, lookback_days: int = 30) -> str:
    """检测当前市场regime
    
    方法：计算最近N天的动量-反转收益差。
    如果近期输家表现 > 近期赢家 → reversal
    如果近期赢家表现 > 近期输家 → momentum
    
    Args:
        df: 全量数据（需包含sym, date, close, volume列）
        lookback_days: 回看天数
    
    Returns:
        'momentum' 或 'reversal'
    """
    df = df.copy()
    df = df.sort_values(['sym', 'date'])
    
    # 计算20日收益
    df['ret20'] = df.groupby('sym')['close'].pct_change(20)
    df['fwd_5d'] = df.groupby('sym')['close'].shift(-5) / df['close'] - 1
    
    # 只看最近N天的数据
    recent = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=lookback_days))]
    recent = recent.dropna(subset=['ret20', 'fwd_5d'])
    recent = recent[recent['close'] > 10]
    
    if len(recent) < 1000:
        return 'unknown'
    
    # 按ret20分桶
    recent['q'] = recent.groupby('date')['ret20'].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
    )
    
    bottom = recent[recent['q'] == 0]['fwd_5d'].mean()
    top = recent[recent['q'] == 4]['fwd_5d'].mean()
    
    if bottom > top:
        return 'reversal'
    else:
        return 'momentum'


def get_signal_direction(regime: str) -> str:
    """根据regime返回信号方向
    
    Returns:
        'momentum' = 买近期赢家（模型当前行为）
        'reversal' = 买近期输家（反向操作）
    """
    return regime


def get_regime_score_adjustment(pred_scores: np.ndarray, ret20: np.ndarray, 
                                 regime: str) -> np.ndarray:
    """根据regime调整模型分数
    
    在reversal regime下，反转分数：高动量→降分，低动量→加分
    在momentum regime下，保持原始分数
    
    Args:
        pred_scores: 模型原始预测分数
        ret20: 20日收益
        regime: 'momentum' 或 'reversal'
    
    Returns:
        调整后的分数
    """
    if regime != 'reversal':
        return pred_scores
    
    # 在reversal regime下，反转分数
    # ret20高的降分，ret20低的加分
    ret20_rank = pd.Series(ret20).rank(pct=True).values
    
    # 调整幅度：原始分数的rank + 反转的ret20_rank
    # 反转权重：0.3（不要太激进）
    original_rank = pd.Series(pred_scores).rank(pct=True).values
    adjusted_rank = original_rank * 0.7 + (1 - ret20_rank) * 0.3
    
    # 转换回分数空间
    adjusted_scores = np.percentile(pred_scores, adjusted_rank * 100)
    
    return adjusted_scores
