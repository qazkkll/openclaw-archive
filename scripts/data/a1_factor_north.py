"""
北向资金因子提取 (第一梯队)
从north_money.json提取5个北向资金因子

因子设计:
1. north_ma5: 北向资金5日均值
2. north_ma20: 北向资金20日均值
3. north_trend: 趋势斜率 (近5日 vs 前5日)
4. consecutive_inflow_days: 连续净流入天数
5. north_zscore: 北向资金相对历史均值的偏离度

输出: a1_north_factors.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 统一路径管理
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import NORTH_MONEY
INPUT_PATH = Path(NORTH_MONEY)
OUTPUT_PATH = Path('/home/hermes/.hermes/openclaw-project/data/a1_factors/north_factors.parquet')
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_north_data():
    """加载北向资金数据"""
    print("Loading north money data...")
    with open(INPUT_PATH) as f:
        raw = json.load(f)
    
    # records是每日总数据
    records = raw['records']
    df = pd.DataFrame(records)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    # 选择北向资金列（使用north_money）
    # north_money = 当日北向净流入
    df['north_money'] = pd.to_numeric(df['north_money'], errors='coerce')
    
    print(f"Loaded: {len(df):,} rows")
    print(f"Date range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    return df

def extract_north_factors(df):
    """提取北向资金因子"""
    
    # === 因子1-2: 滚动均值 ===
    df['north_ma5'] = df['north_money'].rolling(5, min_periods=1).mean()
    df['north_ma20'] = df['north_money'].rolling(20, min_periods=1).mean()
    df['north_ma60'] = df['north_money'].rolling(60, min_periods=1).mean()
    
    # === 因子3: 趋势斜率 ===
    # 近5日均值 vs 之前5日均值
    north_recent5 = df['north_money'].rolling(5, min_periods=1).mean()
    north_prev5 = df['north_money'].shift(5).rolling(5, min_periods=1).mean()
    df['north_trend'] = (north_recent5 - north_prev5) / north_prev5.abs().replace(0, np.nan)
    
    # 另一视角：近20日 vs 前20日
    north_recent20 = df['north_money'].rolling(20, min_periods=1).mean()
    north_prev20 = df['north_money'].shift(20).rolling(20, min_periods=1).mean()
    df['north_trend_20v20'] = (north_recent20 - north_prev20) / north_prev20.abs().replace(0, np.nan)
    
    # === 因子4: 连续净流入天数 ===
    df['north_positive'] = (df['north_money'] > 0).astype(int)
    df['consecutive_inflow_days'] = df['north_positive'].groupby(
        (df['north_positive'] != df['north_positive'].shift()).cumsum()
    ).cumsum()
    # 如果当天净流出，连续天数为0
    df.loc[df['north_money'] <= 0, 'consecutive_inflow_days'] = 0
    
    # === 因子5: 相对历史偏离度 (z-score) ===
    df['north_mean_60'] = df['north_money'].rolling(60, min_periods=1).mean()
    df['north_std_60'] = df['north_money'].rolling(60, min_periods=1).std()
    df['north_zscore'] = (df['north_money'] - df['north_mean_60']) / df['north_std_60'].replace(0, np.nan)
    
    # 额外: 累计北向资金 (模拟指数，含趋势信息)
    df['north_cumsum'] = df['north_money'].cumsum()
    
    return df

def align_to_individual_stocks(north_factors):
    """
    北向资金是市场总体指标，需要对齐到单只股票
    对每只股票，回填当日的北向资金状态
    """
    # 这里我们是市场级因子，直接返回即可
    # 实际使用时，需要join到股票日线数据上
    return north_factors

def main():
    df = load_north_data()
    df = extract_north_factors(df)
    
    # 选择因子列
    factor_cols = [
        'trade_date',
        'north_ma5', 'north_ma20', 'north_ma60',
        'north_trend', 'north_trend_20v20',
        'consecutive_inflow_days',
        'north_zscore',
        'north_cumsum'
    ]
    
    result = df[factor_cols].copy()
    
    print(f"\nSaving to {OUTPUT_PATH}...")
    result.to_parquet(OUTPUT_PATH, index=False)
    
    print(f"\n=== Done ===")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Shape: {result.shape}")
    print(f"Date range: {result['trade_date'].min()} ~ {result['trade_date'].max()}")
    
    # 打印统计
    print(f"\n=== Factor Statistics ===")
    stat_cols = [c for c in result.columns if c != 'trade_date']
    print(result[stat_cols].describe().to_string())
    
    print(f"\nSample data:")
    print(result.tail(5).to_string())
    
    return result

if __name__ == '__main__':
    main()
