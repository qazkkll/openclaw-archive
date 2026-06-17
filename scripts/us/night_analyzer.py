#!/usr/bin/env python3
"""
夜间因子分析引擎 v1
测试路径：
A: V1因子 + 门控 + 资金流 (改良路线)
B: 全新因子组合 (激进路线)
"""
import os, sys, json, time, datetime as dt, numpy as np
import pandas as pd

WORKDIR = r'C:\workspace\av2' if sys.platform == 'win32' else '/home/admin/.openclaw/workspace/av2_data'
os.makedirs(WORKDIR, exist_ok=True)

def log(msg):
    print(f'[{dt.datetime.now():%H:%M:%S}] {msg}', flush=True)

# ======================
# Factor 1: 资金流因子
# ======================
def compute_moneyflow_signal(df_mf):
    """计算个股资金流信号"""
    if df_mf is None or len(df_mf) == 0:
        return None
    df = df_mf.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values(['ts_code', 'trade_date'])
    
    # 主力净流入 (超大单+大单)
    df['net_big'] = (df['buy_elg_vol'].fillna(0) + df['buy_lg_vol'].fillna(0) 
                    - df['sell_elg_vol'].fillna(0) - df['sell_lg_vol'].fillna(0))
    # 净流入占比
    total_vol = (df['buy_sm_vol'].fillna(0) + df['sell_sm_vol'].fillna(0)
                + df['buy_md_vol'].fillna(0) + df['sell_md_vol'].fillna(0)
                + df['buy_lg_vol'].fillna(0) + df['sell_lg_vol'].fillna(0)
                + df['buy_elg_vol'].fillna(0) + df['sell_elg_vol'].fillna(0))
    df['net_big_ratio'] = df['net_big'] / (total_vol + 1)
    
    # 3日、5日、10日主力累积
    signals = {}
    for code in df['ts_code'].unique():
        cdf = df[df['ts_code'] == code].copy()
        cdf['net_3d'] = cdf['net_big'].rolling(3).sum()
        cdf['net_5d'] = cdf['net_big'].rolling(5).sum()
        cdf['net_10d'] = cdf['net_big'].rolling(10).sum()
        cdf['ratio_3d'] = cdf['net_big_ratio'].rolling(3).mean()
        
        # 信号
        cdf['mf_signal'] = 0
        cdf.loc[(cdf['net_3d'] > 0) & (cdf['ratio_3d'] > 0.1), 'mf_signal'] = 1  # 主力连续流入
        cdf.loc[(cdf['net_3d'] < 0) & (cdf['ratio_3d'] < -0.1), 'mf_signal'] = -1  # 主力连续流出
        
        latest = cdf.iloc[-1] if len(cdf) > 0 else None
        if latest is not None:
            signals[code] = {
                'mf_signal': int(latest['mf_signal']),
                'net_3d': float(latest['net_3d']),
                'net_5d': float(latest['net_5d']),
                'net_10d': float(latest['net_10d']),
                'ratio_3d': float(latest['ratio_3d'])
            }
    return signals


# ======================
# Factor 2: 北向资金因子
# ======================
def compute_northbound_signal(df_hsgt):
    """北向资金整体信号"""
    if df_hsgt is None or len(df_hsgt) == 0:
        return None
    df = df_hsgt.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date')
    
    # 几个维度
    result = {}
    for _, row in df.iterrows():
        d = row['trade_date']
        result[str(d.date())] = {
            'north_net': float(row.get('north_money', 0)),  # 北向净买入
            'south_net': float(row.get('south_money', 0)),
            'total_net': float(row.get('north_money', 0)) - float(row.get('south_money', 0))
        }
    
    # 5日累积
    dates = sorted(result.keys())
    if len(dates) >= 5:
        last5 = dates[-5:]
        cum = sum(result[d]['total_net'] for d in last5)
        result['north_5d_cum'] = cum
        result['north_trend'] = 1 if cum > 0 else (-1 if cum < 0 else 0)
    
    return result


# ======================
# Factor 3: PE/PB/换手率/量比 分析
# ======================
def analyze_fundamental_factors(df_db):
    """分析PE/PB/换手率/量比的预测力"""
    if df_db is None or len(df_db) == 0:
        return None
    
    df = df_db.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    # 1. PE分段收益率
    results = {
        'pe_percentiles': {},
        'pb_percentiles': {},
        'turnover_percentiles': {},
        'vol_ratio_percentiles': {}
    }
    
    # 按日期分组，每天分5组看后续收益
    for percentile in [20, 40, 60, 80]:
        name = f'p{percentile}'
        # 后续收益统计需要价格数据配合
        pass
    
    # 2. 换手率异常信号
    # 倍量信号：当日换手率是5日均值的2倍以上
    df_sorted = df.sort_values(['ts_code', 'trade_date'])
    df_sorted['turnover_ma5'] = df_sorted.groupby('ts_code')['turnover_rate'].transform(
        lambda x: x.rolling(5, min_periods=3).mean())
    df_sorted['turnover_ratio'] = df_sorted['turnover_rate'] / (df_sorted['turnover_ma5'] + 0.001)
    
    # 倍量标准
    df_sorted['double_vol'] = (df_sorted['turnover_ratio'] > 2.0) & (df_sorted['turnover_ma5'] > 0.1)
    
    # 3. PE/PB分位数
    for code in df_sorted['ts_code'].unique()[:100]:  # Sample 100 stocks
        cdf = df_sorted[df_sorted['ts_code'] == code].copy()
        if len(cdf) < 60:
            continue
        cdf['pe_pct'] = cdf['pe'].rank(pct=True)
        cdf['pb_pct'] = cdf['pb'].rank(pct=True)
        # 低PE+低PB信号
        cdf['value_signal'] = ((cdf['pe_pct'] < 0.3) & (cdf['pb_pct'] < 0.3)).astype(int)
    
    return df_sorted


# ======================
# 路线A: V1改良 + 门控 + 资金流
# ======================
def route_A_v1_improved(df_db=None, df_mf=None, df_hsgt=None, df_daily=None):
    """路线A: 在V1基础上加门控系统"""
    log('=== 路线A: V1改良 + 门控 + 资金流 ===')
    
    results = {}
    
    # Gate 1: 市场环境感知
    if df_daily is not None:
        # 计算市场宽基指数状态 (用000001.SH或全市场均值)
        results['gate_market'] = {
            'type': 'market_regime',
            'description': '牛市/震荡/熊市门控',
            'params_tested': ['MA200_position', 'volatility_regime', 'breadth']
        }
    
    # Gate 2: 入场信号验证
    if df_mf is not None:
        signals = compute_moneyflow_signal(df_mf)
        results['gate_entry'] = {
            'type': 'entry_validation',
            'description': '资金流+龙虎榜确认入场',
            'signals': signals
        }
    
    # Gate 3: 趋势确认 (5日/10日均线)
    results['gate_trend'] = {
        'type': 'trend_confirmation',
        'description': '5日/10日均线趋势确认',
        'candidates': ['MA5_above_MA10', 'MA10_slope', 'price_above_MA10']
    }
    
    return results


# ======================
# 路线B: 全新因子组合
# ======================
def route_B_new_factors(df_db=None, df_daily=None):
    """路线B: 抛弃旧框架，从零组因子"""
    log('=== 路线B: 全新因子组合 ===')
    
    factors = {
        # 动量类 (占重要权重)
        'momentum': {
            '5d_mom': '5日动量',
            '10d_mom': '10日动量',
            '20d_mom': '20日动量',
            '60d_mom': '60日动量 (LightGBM确认最强)',
            '120d_mom': '120日动量'
        },
        # 资金流类
        'moneyflow': {
            'mf_3d_net': '3日主力净流入',
            'mf_5d_net': '5日主力净流入',
            'mf_ratio': '主力净流入占比'
        },
        # 价值类
        'value': {
            'pe_pct': 'PE历史分位',
            'pb_pct': 'PB历史分位',
            'pe_rank': 'PE行业内排名'
        },
        # 量能类
        'volume': {
            'double_vol': '倍量信号',
            'turnover_ma20': '20日平均换手率',
            'vol_ratio': '量比'
        },
        # 市场类
        'market': {
            'industry_mom': '行业动量 (V3证明有效)',
            'north_flow': '北向资金趋势',
            'lhb_buy': '龙虎榜主力买入'
        },
        # 技术类 (精简)
        'technical': {
            'rsi_30': 'RSI超卖(<30)',
            'ma5_ma10': '5日上穿10日',
            'adx_25': '趋势强度>25'
        }
    }
    
    return {'type': 'new_approach', 'factor_candidates': factors}


# ======================
# Gate Design: 门控机制
# ======================
def design_gate_mechanisms():
    """设计三阶层门控系统"""
    log('=== 门控系统设计 ===')
    
    gates = {
        'level1': {
            'name': '市场环境门 (Market Regime Gate)',
            'purpose': '判断当前市场阶段，切换策略权重',
            'components': [
                {'name': 'MA200门', 'logic': '指数在MA200之上↔牛市激进；之下↔震荡防守'},
                {'name': '波动率门', 'logic': '20日ATR/价格 > 历史80%分位↔高波动防守'},
                {'name': '宽度门', 'logic': '全市场上涨家数占比 > 60%↔强势可买'},
            ],
            'output': ['bull_mode', 'oscillation_mode', 'bear_mode']
        },
        'level2': {
            'name': '选股确认门 (Stock Selection Gate)',
            'purpose': 'V1评分候选股进入前，加一层验证',
            'components': [
                {'name': '资金流验证', 'logic': '候选股3日主力净流入为正↔通过'},
                {'name': '趋势验证', 'logic': '候选股价格在5日/10日均线之上↔通过'},
                {'name': '量能验证', 'logic': '换手率大于5日均值↔通过'},
            ],
            'output': ['passed', 'rejected']
        },
        'level3': {
            'name': '出场信号门 (Exit Signal Gate)',
            'purpose': '何时止盈止损',
            'components': [
                {'name': '趋势破坏', 'logic': '收盘跌破10日线↔减仓；跌破20日线↔清仓'},
                {'name': '资金撤退', 'logic': '主力3日内净流出↔减仓'},
                {'name': '动量衰竭', 'logic': '5日动量为负且10日动量转负↔清仓'},
            ],
            'output': ['hold', 'reduce', 'clear']
        }
    }
    
    return gates


# ======================
# Backtest Gate Configs
# ======================
def generate_gate_configs():
    """生成门控参数组合，供bruteforce"""
    configs = []
    
    # Level 1: Market regime thresholds
    for ma200_bull_threshold in [0.98, 1.0, 1.02, 1.05]:  # price/MA200 ratio
        for vol_threshold_pct in [70, 80, 90]:  # volatility percentile
            for breadth_min in [50, 55, 60]:  # min % stocks up
                configs.append({
                    'id': f'L1_M{ma200_bull_threshold}_V{vol_threshold_pct}_B{breadth_min}',
                    'level1': {
                        'ma200_bull_ratio': ma200_bull_threshold,
                        'volatility_pct': vol_threshold_pct,
                        'breadth_min_pct': breadth_min
                    },
                    'level2': {},  # Will fill default
                    'level3': {}
                })
    
    return configs


if __name__ == '__main__':
    log('=' * 50)
    log('夜间因子分析引擎启动')
    log('=' * 50)
    
    # Check if data files exist
    data_available = {}
    for fname in ['daily_basic.parquet', 'moneyflow_hsgt.parquet', 'daily_ohlcv.parquet', 'top_list.parquet']:
        fpath = os.path.join(WORKDIR, fname)
        data_available[fname] = os.path.exists(fpath)
        if data_available[fname]:
            sz = os.path.getsize(fpath) // (1024*1024)
            log(f'  ✅ {fname} ({sz} MB)')
        else:
            log(f'  ⏳ {fname} 等待中...')
    
    # Generate gate configs
    gates = design_gate_mechanisms()
    gate_configs = generate_gate_configs()
    log(f'门控组合数: {len(gate_configs)}')
    
    # Route A analysis
    route_a = route_A_v1_improved()
    
    # Route B factor candidates
    route_b = route_B_new_factors()
    
    # Save analysis structure
    output = {
        'timestamp': dt.datetime.now().isoformat(),
        'gates': gates,
        'gate_configs': gate_configs[:20],  # Sample
        'route_a': route_a,
        'route_b': route_b
    }
    
    with open(os.path.join(WORKDIR, 'analysis_framework.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    log('分析框架已保存')
