#!/usr/bin/env python3
"""
门控系统实现 + 测试
三阶层:
 L1: 市场环境门 - 牛市/震荡/熊市感知
 L2: 选股确认门 - V1分数+资金流+趋势验证
 L3: 出场信号门 - 趋势破坏/资金撤退/动量衰竭
"""
import os, sys, json, time, numpy as np
import pandas as pd
from datetime import datetime

WORKDIR = r'C:\workspace\av2' if sys.platform == 'win32' else '/home/admin/.openclaw/workspace/av2_data'
os.makedirs(WORKDIR, exist_ok=True)

def log(m): print(f'[{datetime.now():%H:%M:%S}] {m}', flush=True)

# ========================
# L1: 市场环境门
# ========================
class MarketRegimeGate:
    """检测当前市场处于什么阶段"""
    
    def __init__(self, config=None):
        self.config = config or {
            'bull_ma200_ratio': 1.0,      # 指数>MA200
            'volatility_pct': 80,          # 波动率分位
            'breadth_pct': 60,             # 上涨家数占比
        }
    
    def detect(self, index_df):
        """
        index_df: 大盘指数(000001.SH)的OHLCV数据
        Returns: 'bull' | 'oscillation' | 'bear'
        """
        if index_df is None or len(index_df) < 60:
            return 'oscillation'  # 默认震荡
        
        df = index_df.copy()
        df['ma200'] = df['close'].rolling(200).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        
        # 计算20日ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_ratio'] = df['tr'].rolling(20).mean() / df['close']
        
        latest = df.iloc[-1]
        price_ma200 = latest['close'] / latest['ma200'] if not np.isnan(latest['ma200']) else 1.0
        
        # 判断依据
        is_bull = price_ma200 >= self.config['bull_ma200_ratio']
        
        # 波动率检测
        atr_hist_pct = df['atr_ratio'].rank(pct=True).iloc[-1]
        is_high_vol = atr_hist_pct > (self.config['volatility_pct'] / 100)
        
        # MA60趋势
        ma60_slope = df['ma60'].diff(5).iloc[-1] / df['ma60'].iloc[-5]
        
        if is_bull and not is_high_vol and ma60_slope > 0:
            return 'bull'
        elif is_bull and is_high_vol:
            return 'oscillation'  # 高位震荡
        else:
            return 'oscillation' if is_bull else 'bear'
    
    def get_multipliers(self, regime):
        """根据市场状态返回评分和持仓乘数"""
        mults = {
            'bull': {'score_mult': 1.0, 'position_mult': 1.0, 'threshold_offset': 0.0},
            'oscillation': {'score_mult': 0.8, 'position_mult': 0.5, 'threshold_offset': 0.1},
            'bear': {'score_mult': 0.5, 'position_mult': 0.25, 'threshold_offset': 0.2},
        }
        return mults.get(regime, mults['oscillation'])


# ========================
# L2: 选股确认门
# ========================
class StockSelectionGate:
    """候选股验证层"""
    
    def __init__(self, config=None):
        self.config = config or {
            'min_turnover_ratio': 0.5,     # 换手率 > 5日均值的一半
            'require_ma_trend': True,       # 需要均线多头
            'require_double_vol': False,    # 不需要倍量
            'min_moneyflow': 0,             # 主力净流入为正
        }
    
    def verify(self, candidates_df, moneyflow_data=None):
        """
        验证候选股是否通过门控
        Returns: passed_df (通过列表), scores (每只分数)
        """
        if candidates_df is None or len(candidates_df) == 0:
            return pd.DataFrame(), {}
        
        df = candidates_df.copy()
        passed = []
        scores = {}
        
        for _, row in df.iterrows():
            gate_score = 0
            reasons = []
            
            # 1. 均线多头确认
            if self.config['require_ma_trend']:
                if row.get('ma5', 0) > row.get('ma10', 0) > row.get('ma20', 0):
                    gate_score += 30
                    reasons.append('ma_bull')
                elif row.get('close', 0) > row.get('ma10', 0):
                    gate_score += 15
                    reasons.append('above_ma10')
            
            # 2. 量能确认
            if row.get('turnover_rate', 0) > row.get('turnover_ma5', 0) * self.config['min_turnover_ratio']:
                gate_score += 20
                reasons.append('vol_ok')
            
            # 3. 倍量加分
            if self.config.get('require_double_vol', False) and row.get('double_vol', False):
                gate_score += 20
                reasons.append('double_vol')
            
            # 4. 资金流验证
            if moneyflow_data and row['ts_code'] in moneyflow_data:
                mf = moneyflow_data[row['ts_code']]
                if mf.get('mf_signal', 0) > 0:
                    gate_score += 30
                    reasons.append('mf_inflow')
                elif mf.get('mf_signal', 0) < 0:
                    gate_score -= 20
                    reasons.append('mf_outflow')
            
            # 通过条件
            tc = row.get('threshold', 50)
            if gate_score >= tc:
                passed.append(row['ts_code'])
            
            scores[row['ts_code']] = {
                'gate_score': gate_score,
                'pass': gate_score >= tc,
                'reasons': reasons
            }
        
        result_df = df[df['ts_code'].isin(passed)] if passed else pd.DataFrame()
        return result_df, scores


# ========================
# L3: 出场信号门
# ========================
class ExitSignalGate:
    """持仓卖出决策"""
    
    def __init__(self, config=None):
        self.config = config or {
            'trend_break_ma': 20,          # 跌破MA20清仓
            'trend_reduce_ma': 10,         # 跌破MA10减仓
            'moneyflow_exit_days': 3,      # 连续N天主力流出
            'momentum_exit': -0.02,        # 5日动量< -2%
        }
    
    def should_exit(self, stock_df, moneyflow_data=None):
        """
        判断是否应该卖出
        Returns: ('hold'|'reduce'|'clear', reasons)
        """
        if stock_df is None or len(stock_df) < 20:
            return 'hold', []
        
        df = stock_df.copy()
        latest = df.iloc[-1]
        reasons = []
        
        close = latest['close']
        ma10 = latest.get('ma10', None)
        ma20 = latest.get('ma20', None)
        ma5 = latest.get('ma5', None)
        
        # 1. 趋势破坏
        if ma10 and close < ma10:
            reasons.append('below_ma10')
        if ma20 and close < ma20:
            reasons.append('below_ma20')
        
        # 2. 动量衰竭
        mom5 = latest.get('mom5', 0)
        if mom5 < self.config['momentum_exit']:
            reasons.append('momentum_fail')
        
        # 3. 资金撤退
        if moneyflow_data:
            mf_3d = moneyflow_data.get('net_3d', 0)
            if mf_3d < 0:
                reasons.append('mf_outflow_3d')
        
        # 决策
        if 'below_ma20' in reasons or 'momentum_fail' in reasons:
            return 'clear', reasons
        elif 'below_ma10' in reasons or 'mf_outflow_3d' in reasons:
            return 'reduce', reasons
        else:
            return 'hold', reasons


# ========================
# 完整门控系统测试
# ========================
def gate_system_test():
    """跑一个小测试验证门控逻辑"""
    log('=' * 50)
    log('门控系统测试')
    log('=' * 50)
    
    # Test L1
    log('\n--- L1 市场环境门 ---')
    mg = MarketRegimeGate()
    regimes = ['bull', 'oscillation', 'bear']
    for r in regimes:
        m = mg.get_multipliers(r)
        log(f'  {r}: score_mult={m["score_mult"]} position_mult={m["position_mult"]}')
    
    # Test L2
    log('\n--- L2 选股确认门 ---')
    sg = StockSelectionGate()
    test_data = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ'],
        'close': [15.0, 8.0, 12.0],
        'ma5': [14.8, 7.5, 11.5],
        'ma10': [14.5, 8.0, 11.0],
        'ma20': [14.0, 8.5, 10.5],
        'turnover_rate': [3.0, 1.0, 5.0],
        'turnover_ma5': [2.0, 2.0, 2.0],
        'double_vol': [True, False, True],
        'threshold': 50
    })
    
    moneyflow_test = {
        '000001.SZ': {'mf_signal': 1},
        '000002.SZ': {'mf_signal': -1},
    }
    
    result, scores = sg.verify(test_data, moneyflow_test)
    log(f'  Passed: {result["ts_code"].tolist() if len(result) > 0 else "none"}')
    for code, score in scores.items():
        log(f'  {code}: score={score["gate_score"]} pass={score["pass"]} reasons={score["reasons"]}')
    
    # Test L3
    log('\n--- L3 出场信号门 ---')
    eg = ExitSignalGate()
    
    # Test: going below MA20
    hold_df = pd.DataFrame({
        'close': [15.0, 15.2, 15.5, 15.3, 15.0],
        'ma10': [14.8, 14.9, 15.0, 15.0, 14.9],
        'ma20': [14.5, 14.6, 14.6, 14.7, 14.7],
        'ma5': [15.0, 15.1, 15.3, 15.2, 15.0],
        'mom5': [0.02, 0.01, 0.03, 0.01, 0.0]
    })
    decision, reasons = eg.should_exit(hold_df)
    log(f'  MA20以上: {decision} reasons={reasons}')
    
    # Test: below MA10
    break_df = hold_df.copy()
    break_df.loc[break_df.index[-1], 'close'] = 14.6
    break_df.loc[break_df.index[-1], 'mom5'] = -0.03
    decision, reasons = eg.should_exit(break_df)
    log(f'  跌破MA10: {decision} reasons={reasons}')
    
    # Test: below MA20
    clear_df = hold_df.copy()
    clear_df.loc[clear_df.index[-1], 'close'] = 14.0
    decision, reasons = eg.should_exit(clear_df)
    log(f'  跌破MA20: {decision} reasons={reasons}')
    
    log('\n✅ 门控系统测试完成')


# ========================
# 门控参数组合生成 (Bruteforce 用)
# ========================
def generate_gate_params():
    """生成所有门控参数组合"""
    params = []
    for l1_bull_ratio in [0.98, 1.0, 1.02, 1.05]:
        for l1_vol_pct in [70, 80, 90]:
            for l1_breadth in [50, 55, 60]:
                for l2_require_ma in [True, False]:
                    for l2_min_turn in [0.3, 0.5, 0.7]:
                        for l3_stop_ma in [10, 20]:
                            for l3_mom_exit in [-0.01, -0.02, -0.03]:
                                params.append({
                                    'L1_bull_ratio': l1_bull_ratio,
                                    'L1_vol_pct': l1_vol_pct,
                                    'L1_breadth': l1_breadth,
                                    'L2_require_ma': l2_require_ma,
                                    'L2_min_turn': l2_min_turn,
                                    'L3_stop_ma': l3_stop_ma,
                                    'L3_mom_exit': l3_mom_exit,
                                })
    return params


if __name__ == '__main__':
    gate_system_test()
    
    gate_params = generate_gate_params()
    log(f'\n门控参数组合数: {len(gate_params)}')
    
    # Save gate params for brute force
    with open(os.path.join(WORKDIR, 'gate_params.json'), 'w') as f:
        json.dump(gate_params[:100], f, indent=2, default=str)  # Just save a sample
    
    log('门控参数已保存')
