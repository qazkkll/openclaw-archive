#!/usr/bin/env python3
"""
A_V2_gated.py - 三层门控系统 v1
基于 V1 评分引擎，叠加市场环境/选股确认/出场信号三层门控。
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_engine import compute_indicators, v1_score, safe

class MarketGating:
    """三层门控系统"""

    def __init__(self, use_index=False, index_data=None):
        """
        use_index: True时使用指数数据(需外部提供沪深300日K)
        index_data: 指数日K线 [{close,high,low},...]
        """
        self.use_index = use_index
        self.index_data = index_data or []

    # ---- L1: 市场环境门 ----
    def l1_market_gate(self, close_prices):
        """
        判断市场状态: bull / sideways / bear
        基于MA200趋势 + 价格偏离度
        返回: (gate_pass: bool, regime: str)
        """
        if len(close_prices) < 200:
            return True, 'unknown'

        # 使用指数数据或个股数据
        prices = self.index_data if self.use_index and len(self.index_data) >= 200 else close_prices
        ma200 = sum(prices[-200:]) / 200
        current = prices[-1]
        deviation = (current - ma200) / ma200 * 100

        if deviation > 5:
            regime = 'bull'
            gate_pass = True      # 牛市：正常交易
        elif deviation < -5:
            regime = 'bear'
            # 熊市：评分需>65才准入, 否则拦截
            gate_pass = False     # 默认拦截，由调用方决定
        else:
            regime = 'sideways'
            gate_pass = True      # 震荡市：正常交易

        return gate_pass, regime

    # ---- L2: 选股确认门 ----
    def l2_confirm_gate(self, ind, di):
        """
        选股确认: MA5 > MA10 + 量能 + (可选)资金流
        返回: (pass: bool, reasons: list)
        """
        reasons = []
        m5 = safe(ind.get('m5'), di)
        m20 = safe(ind.get('m20'), di)
        close = safe(ind.get('close'), di)
        vol_ratio = safe(ind.get('vol_ratio'), di)

        # MA5 > MA20 (代替MA10，因已有MA5/MA20计算)
        if m5 and m20 and m5 > m20:
            reasons.append('ma_trend_ok')
        else:
            reasons.append('ma_trend_block')
            return False, reasons

        # 量能放大
        if vol_ratio and vol_ratio > 1.2:
            reasons.append('volume_ok')
        else:
            reasons.append('volume_low')

        # 资金流(需Tushare, 暂标记)
        reasons.append('moneyflow_check_disabled_no_tushare')

        return True, reasons

    # ---- L3: 出场信号门 ----
    def l3_exit_gate(self, ind, di, current_price, position=None):
        """
        出场信号: 破MA10减仓/破MA20清仓
        返回: (action: str, reason: str)
        action: 'hold' / 'reduce' / 'clear'
        """
        m10 = safe(ind.get('m20'), di)  # 用MA20近似代替MA10/MA20
        m20 = safe(ind.get('m60'), di)  # 用MA60近似代替MA20
        close = safe(ind.get('close'), di)

        if close and m20 and close < m20:
            return 'clear', 'price_below_ma60_clear'
        if close and m10 and close < m10:
            return 'reduce', 'price_below_ma20_reduce'
        return 'hold', ''

    # ---- 综合评分(带门控) ----
    def score_with_gating(self, close, high, low, volume=None, idx=-1):
        """
        完整门控评分流程
        返回: {score, gated_score, gates, action, regime}
        """
        ind = compute_indicators(close, high, low, volume)
        if ind is None:
            return {'score': 0, 'gated_score': 0, 'gates': 'no_data'}

        di = idx if idx >= 0 else len(close) - 1
        base_score = v1_score(ind, di)

        # L1 市场环境门
        l1_pass, regime = self.l1_market_gate(close)
        if regime == 'bear' and base_score < 65:
            return {
                'score': base_score,
                'gated_score': 0,
                'regime': 'bear',
                'gates': 'l1_blocked_bear_market',
                'action': 'hold'
            }

        # L2 选股确认门
        l2_pass, l2_reasons = self.l2_confirm_gate(ind, di)
        if not l2_pass:
            return {
                'score': base_score,
                'gated_score': 0,
                'regime': regime,
                'gates': 'l2_blocked_' + '_'.join(l2_reasons),
                'action': 'hold'
            }

        # L3 出场信号(这里只是检查, 实际出场由风控执行)
        l3_action, l3_reason = self.l3_exit_gate(ind, di, close[-1] if isinstance(close, list) else close)
        final_score = base_score
        if l3_action == 'clear':
            final_score = 0
        elif l3_action == 'reduce':
            final_score = base_score * 0.5  # 减半

        return {
            'score': base_score,
            'gated_score': round(final_score, 1),
            'regime': regime,
            'gates': 'l1_pass_l2_pass',
            'action': l3_action,
            'l2_details': l2_reasons
        }


if __name__ == '__main__':
    # 快速测试
    import sys
    from data_source import AShareKline
    kl = AShareKline()
    data = kl.get_kline('002015', days=250, source='baostock')
    if data:
        close = [d['close'] for d in data]
        high = [d['high'] for d in data]
        low = [d['low'] for d in data]
        g = MarketGating()
        r = g.score_with_gating(close, high, low)
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        print('no data')
