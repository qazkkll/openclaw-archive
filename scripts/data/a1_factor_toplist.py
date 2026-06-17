"""
龙虎榜因子提取 (第一梯队)
从top_list_data.parquet提取5个龙虎榜因子

因子设计:
1. net_rate_ma5: 净流入率5日滚动均值（机构活跃度）
2. l_buy_signal: 机构席位净买入信号（l_amount > 均值1.5倍）
3. l_buy_binary: 机构积极买入二值信号
4. toplist_freq_30d: 30日内上榜次数
5. l_buy_net_ma10: 机构净买入10日均值

输出: a1_toplist_factors.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

INPUT_PATH = Path('/home/hermes/.hermes/openclaw-project/data/top_list_data.parquet')
OUTPUT_PATH = Path('/home/hermes/.hermes/openclaw-project/data/a1_factors/toplist_factors.parquet')
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_toplist():
    print("Loading toplist data...")
    df = pd.read_parquet(INPUT_PATH)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    print(f"Loaded: {len(df):,} rows, {df['ts_code'].nunique()} stocks")
    print(f"Date range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    return df

def extract_toplist_factors(df):
    """
    暴力特征提取：对每只股票，用rolling遍历所有历史窗口
    核心思想：一只股票历史上每次上龙虎榜，都包含了当时的买卖信息
    我们要提取的是：基于历史所有龙虎榜记录的统计特征
    """
    
    # === 因子1: 净流入率滚动均值 ===
    # net_rate = (l_buy - l_sell) / (l_buy + l_sell) * 100
    df['net_rate'] = df['net_amount'] / df['l_amount'].replace(0, np.nan) * 100
    
    # 按股票排序后计算历史滚动均值
    df['net_rate_ma5'] = df.groupby('ts_code')['net_rate'].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    df['net_rate_ma10'] = df.groupby('ts_code')['net_rate'].transform(
        lambda x: x.rolling(10, min_periods=1).mean())
    
    # === 因子2: 机构席位净买入信号 ===
    # 用机构买入金额(l_buy) vs 个股自身历史均值
    df['l_buy_ma5'] = df.groupby('ts_code')['l_buy'].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    df['l_buy_signal'] = (df['l_buy'] > df['l_buy_ma5'] * 1.5).astype(float)
    
    # === 因子3: 机构积极买入二值信号 ===
    # net_amount > 0 且 net_rate > 中位数
    median_net_rate = df['net_rate'].median()
    df['l_buy_binary'] = ((df['net_amount'] > 0) & 
                          (df['net_rate'] > median_net_rate)).astype(float)
    
    # === 因子4: 30日内上榜次数 (历史统计, 不含未来) ===
    df['toplist_freq_30d'] = df.groupby('ts_code').cumcount()
    # 修正为: 前30日滚动的上榜次数
    # 因为cumcount是全局的，我们需要用时间窗口
    def count_rolling_30d(group):
        group = group.sort_values('trade_date')
        dates = group['trade_date'].values
        return np.array([np.sum((dates[:i] >= dates[i] - pd.Timedelta(days=30)) & 
                                (dates[:i] < dates[i])) 
                         for i in range(len(dates))])
    
    print("  Calculating toplist_freq_30d (this may take a moment)...")
    df['toplist_freq_30d'] = (df.groupby('ts_code')
                                .apply(count_rolling_30d)
                                .explode()
                                .values)
    
    # === 因子5: 机构净买入10日均值 ===
    df['l_buy_net_ma10'] = df.groupby('ts_code')['net_amount'].transform(
        lambda x: x.rolling(10, min_periods=1).mean())
    
    return df

def build_daily_factor_panel(df):
    """
    将龙虎榜事件数据展开为每日面板数据
    每天每只股票只能有一条记录（合并同一天多次上龙虎榜）
    """
    # 先按天聚合：同一天多次上榜的合并
    # 选择要聚合的因子列
    factor_cols = ['net_rate_ma5', 'net_rate_ma10', 'l_buy_signal', 
                   'l_buy_binary', 'toplist_freq_30d', 'l_buy_net_ma10']
    
    # 对同一天多次上榜的取均值/最大值
    agg_dict = {col: 'last' for col in factor_cols}  # 取最后一条（最新）
    agg_dict['l_buy_binary'] = 'max'  # 只要有一次积极买入就算
    agg_dict['l_buy_signal'] = 'max'
    
    daily = (df.groupby(['ts_code', 'trade_date'])
             .agg(agg_dict)
             .reset_index())
    
    # 填补非龙虎榜日（用前值填充）
    # 对每只股票，生成完整日期序列
    print("  Filling non-toplist days (forward fill)...")
    all_stocks = []
    for code, group in daily.groupby('ts_code'):
        group = group.sort_values('trade_date')
        # 创建完整日期范围
        full_dates = pd.date_range(group['trade_date'].min(), 
                                   group['trade_date'].max(), freq='D')
        full_df = pd.DataFrame({'trade_date': full_dates})
        full_df['ts_code'] = code
        full_df = full_df.merge(group, on=['ts_code', 'trade_date'], how='left')
        # 前向填充因子（龙虎榜信号持续有效直到新信号）
        for col in factor_cols:
            full_df[col] = full_df[col].ffill()
        # 没上过榜的填0
        full_df = full_df.fillna(0)
        all_stocks.append(full_df)
    
    return pd.concat(all_stocks, ignore_index=True)

def main():
    df = load_toplist()
    df = extract_toplist_factors(df)
    daily = build_daily_factor_panel(df)
    
    print(f"\nSaving to {OUTPUT_PATH}...")
    daily.to_parquet(OUTPUT_PATH, index=False)
    
    print(f"\n=== Done ===")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Shape: {daily.shape}")
    print(f"Stocks: {daily['ts_code'].nunique()}")
    print(f"Date range: {daily['trade_date'].min()} ~ {daily['trade_date'].max()}")
    print(f"Columns: {list(daily.columns)}")
    
    # 打印统计
    factor_cols = [c for c in daily.columns if c not in ['ts_code', 'trade_date']]
    print(f"\n=== Factor Statistics ===")
    print(daily[factor_cols].describe().to_string())
    
    return daily

if __name__ == '__main__':
    main()
