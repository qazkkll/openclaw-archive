#!/usr/bin/env python3
"""
A_V2评分引擎 — V1评分 + 资金流因子

架构:
  V1评分 (原始5因子) + 资金流修正层
  
资金流因子使用:
  Tushare moneyflow API (大单/超大单净买入)
  盘后可用，作为评分修正(+/- 5分)

用法:
  from scripts.A_V2 import v2_score
  score = v2_score(ind, di, fund_flow=fund_flow_dict)

注意:
  A_V2不替代A_V1，是A_V1的扩展。
  资金流数据需要每日Tushare调用获取，回测不可用。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.score_engine import compute_indicators, v1_score


def compute_fund_flow_score(fund_flow):
    """
    根据资金流数据计算修正分。
    
    fund_flow: dict with keys from Tushare moneyflow API:
      - buy_lg_amount: 大单买入额(万)
      - sell_lg_amount: 大单卖出额(万)
      - buy_elg_amount: 超大单买入额(万)
      - sell_elg_amount: 超大单卖出额(万)
      - net_mf_amount: 净流入额(万)
    
    返回: 修正分(-5 ~ +5)
    """
    if not fund_flow:
        return 0.0
    
    # 主力净流入 = 大单净额 + 超大单净额
    lg_net = float(fund_flow.get('buy_lg_amount', 0) or 0) - float(fund_flow.get('sell_lg_amount', 0) or 0)
    elg_net = float(fund_flow.get('buy_elg_amount', 0) or 0) - float(fund_flow.get('sell_elg_amount', 0) or 0)
    total_net = lg_net + elg_net
    
    # 净流入率 (相对买卖总额)
    total_buy = float(fund_flow.get('buy_lg_amount', 0) or 0) + float(fund_flow.get('buy_elg_amount', 0) or 0)
    total_sell = float(fund_flow.get('sell_lg_amount', 0) or 0) + float(fund_flow.get('sell_elg_amount', 0) or 0)
    total = total_buy + total_sell
    
    if total == 0:
        return 0.0
    
    net_rate = total_net / total * 100  # 百分比
    
    # 评分修正
    if net_rate > 20:
        return 5.0   # 强主力买入
    elif net_rate > 10:
        return 3.0
    elif net_rate > 5:
        return 2.0
    elif net_rate > 0:
        return 1.0   # 微弱买入
    elif net_rate > -5:
        return -1.0  # 微弱卖出
    elif net_rate > -10:
        return -2.0
    elif net_rate > -20:
        return -3.0
    else:
        return -5.0  # 强主力卖出


def v2_score(ind, di, fund_flow=None):
    """
    A_V2评分: V1基础分 + 资金流修正。
    
    Args:
        ind: compute_indicators() 返回的dict
        di: 数据索引
        fund_flow: Tushare moneyflow数据 (可选)
    
    Returns:
        float 评分(0~100)
    """
    base = v1_score(ind, di)
    if base <= 0:
        return 0.0
    
    # 资金流修正
    flow_adj = compute_fund_flow_score(fund_flow) if fund_flow else 0.0
    
    # 最终评分 (不超过100)
    final_score = min(base + flow_adj, 100.0)
    return max(final_score, 0.0)


def v2_score_from_data(close, high, low, fund_flow=None, idx=-1):
    """从原始K线数据直接评分 (便捷入口)"""
    ind = compute_indicators(close, high, low)
    if ind is None:
        return 0.0
    di = idx if idx >= 0 else len(close) - 1
    return v2_score(ind, di, fund_flow)


if __name__ == '__main__':
    # 测试
    import random
    n = 300
    close = [100 + sum(random.uniform(-2,2) for _ in range(i)) for i in range(n)]
    high = [c + random.uniform(0,3) for c in close]
    low = [c - random.uniform(0,3) for c in close]
    
    # 测试1: 无资金流 (等于V1)
    ind = compute_indicators(close, high, low)
    s1 = v2_score(ind, -1)
    print(f'V2 (无资金流) = {s1:.1f}')
    
    # 测试2: 主力净买入
    fund_buy = {'buy_lg_amount': 50000, 'sell_lg_amount': 10000,
                'buy_elg_amount': 30000, 'sell_elg_amount': 5000}
    s2 = v2_score(ind, -1, fund_buy)
    print(f'V2 (主力买入) = {s2:.1f}  (+{s2-s1:.1f})')
    
    # 测试3: 主力净卖出
    fund_sell = {'buy_lg_amount': 10000, 'sell_lg_amount': 50000,
                 'buy_elg_amount': 5000, 'sell_elg_amount': 30000}
    s3 = v2_score(ind, -1, fund_sell)
    print(f'V2 (主力卖出) = {s3:.1f}  ({s3-s1:+.1f})')
    
    print('A_V2 OK')
