#!/usr/bin/env python3
"""
🍤 统一策略验证框架 — 用LightGBM检验任何策略的真实预测力
周日给本地小钳在32G上跑全量

用法:
  python3 scripts/bt_model_validator.py --list    # 列出可测策略
  python3 scripts/bt_model_validator.py --strategy turtle  # 测单一策略
  python3 scripts/bt_model_validator.py --all     # 测全部
"""
import json, numpy as np, sys, os
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from score_engine import compute_indicators, get_raw_scores

STRATEGIES = {
    'v1_factors': {
        'desc': 'V1评分5因子: MACD+位置+均线+ADX+RSI',
        'features': ['macd','pos52','ma','adx','rsi']
    },
    'momentum': {
        'desc': '动量因子: 5/10/20/60日收益率',
        'features': ['mom5','mom10','mom20','mom60']
    },
    'instock_turtle': {
        'desc': '海龟60日新高(布尔信号)',
        'features': ['turtle60']
    },
    'instock_ma60': {
        'desc': '站上MA60(布尔信号)',
        'features': ['above_ma60']
    },
    'instock_ma30_trend': {
        'desc': '均线多头MA30持续向上',
        'features': ['ma30_trend']
    },
    'volume': {
        'desc': '成交量因子: 量比/5日均量比',
        'features': ['vol_ratio','vol_ma5_ratio']
    },
    'all': {
        'desc': '全部因子组合',
        'features': ['macd','pos52','ma','adx','rsi',
                     'mom5','mom10','mom20','mom60',
                     'turtle60','above_ma60','ma30_trend',
                     'vol_ratio','vol_ma5_ratio']
    }
}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', choices=list(STRATEGIES.keys()) + ['all'])
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--market', choices=['a','us'], default='us')
    args = parser.parse_args()
    
    if args.list:
        for k,v in STRATEGIES.items():
            print(f'  {k:25} {v["desc"]}')
        sys.exit(0)
    
    print('🍤 统一策略验证框架')
    print('  → 周日用本地32G跑全量')
    print('  → 用法: python3 bt_model_validator.py --strategy v1_factors')
