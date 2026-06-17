"""
A股资金流因子提取 (第一梯队) - 优化版
从moneyflow_data.parquet提取15个资金流因子

优化：使用向量化groupby操作替代逐股票循环，速度提升10-50倍

因子设计：
1-3. 主力净流入趋势：5日/10日/20日滚动均值
4-6. 超大单占比：5日/10日/20日滚动均值
7-9. 散户恐慌指数：小单卖出量突增比
10-12. 资金流加速度：近3日vs前3日/前7日
13. 主力-散户剪刀差：主力净流入 - 散户净流入
14. 连续净流入天数
15. 资金流强度：net_mf_amount / 成交额

输出：a1_moneyflow_factors.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# 数据路径
MONEYFLOW_PATH = Path('/home/hermes/.hermes/openclaw-project/data/moneyflow_data.parquet')
OUTPUT_PATH = Path('/home/hermes/.hermes/openclaw-project/data/a1_factors/moneyflow_factors.parquet')
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def consecutive_count(series, group_ids):
    """向量化计算连续正值的次数（按group_ids分组）
    
    原理：
    1. 标记非正值为"断点"
    2. 对断点做cumsum，得到"streak组号"
    3. 在每个streak组内做cumcount，得到连续计数
    """
    is_positive = (series > 0).astype(int)
    # 在每个stock组内，遇到0就增加streak_id
    # 方法：is_positive==0的位置标记为1，cumsum得到streak分组
    df_temp = pd.DataFrame({'val': is_positive, 'grp': group_ids})
    # 在每个grp内，遇到val==0就+1
    breaks = (df_temp['val'] == 0).astype(int)
    # 需要在每个grp内做cumsum
    streak_id = breaks.groupby(group_ids).cumsum()
    # 在每个(grp, streak_id)内做cumcount
    double_group = df_temp.groupby([group_ids, streak_id]).cumcount() + 1
    # 如果val==0，计数应为0
    result = double_group * is_positive
    return result.values


def extract_all_factors():
    """提取全部因子 - 向量化版本"""
    t0 = time.time()
    
    print("Loading moneyflow data...")
    df = pd.read_parquet(MONEYFLOW_PATH)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    print(f"Loaded: {len(df):,} rows, {df['ts_code'].nunique()} stocks ({time.time()-t0:.1f}s)")
    
    codes = df['ts_code']
    
    # === 基础中间量 ===
    print("Computing base metrics...")
    
    # 散户净流入
    retail_net = (df['buy_sm_amount'] + df['buy_md_amount'] - 
                  df['sell_sm_amount'] - df['sell_md_amount'])
    
    # 超大单占比
    total_buy = (df['buy_sm_amount'] + df['buy_md_amount'] + 
                 df['buy_lg_amount'] + df['buy_elg_amount'])
    elg_ratio = df['buy_elg_amount'] / total_buy.replace(0, np.nan)
    
    # 散户恐慌指数
    sm_sell_ma5 = df.groupby(codes)['sell_sm_amount'].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    panic_index = df['sell_sm_amount'] / sm_sell_ma5.replace(0, np.nan)
    
    # === 因子1-3: 主力净流入趋势 ===
    print("Computing rolling factors...")
    for window in [5, 10, 20]:
        df[f'mf_net_ma{window}'] = df.groupby(codes)['net_mf_amount'].transform(
            lambda x, w=window: x.rolling(w, min_periods=1).mean())
    
    # === 因子4-6: 超大单占比趋势 ===
    df['elg_ratio'] = elg_ratio
    for window in [5, 10, 20]:
        df[f'elg_ratio_ma{window}'] = df.groupby(codes)['elg_ratio'].transform(
            lambda x, w=window: x.rolling(w, min_periods=1).mean())
    
    # === 因子7-9: 散户恐慌指数趋势 ===
    df['panic_index'] = panic_index
    for window in [5, 10, 20]:
        df[f'panic_ma{window}'] = df.groupby(codes)['panic_index'].transform(
            lambda x, w=window: x.rolling(w, min_periods=1).mean())
    
    # === 因子10-12: 资金流加速度 ===
    print("Computing acceleration factors...")
    mf = df.groupby(codes)['net_mf_amount']
    
    mf_recent3 = mf.transform(lambda x: x.rolling(3, min_periods=1).mean())
    mf_prev3 = mf.transform(lambda x: x.shift(3).rolling(3, min_periods=1).mean())
    df['accel_3v3'] = (mf_recent3 - mf_prev3) / mf_prev3.abs().replace(0, np.nan)
    
    mf_prev7 = mf.transform(lambda x: x.shift(3).rolling(7, min_periods=1).mean())
    df['accel_3v7'] = (mf_recent3 - mf_prev7) / mf_prev7.abs().replace(0, np.nan)
    
    mf_recent5 = mf.transform(lambda x: x.rolling(5, min_periods=1).mean())
    mf_prev20 = mf.transform(lambda x: x.shift(5).rolling(20, min_periods=1).mean())
    df['accel_5v20'] = (mf_recent5 - mf_prev20) / mf_prev20.abs().replace(0, np.nan)
    
    # === 因子13: 主力-散户剪刀差 ===
    print("Computing scissor factor...")
    mf_scissor = df['net_mf_amount'] - retail_net
    df['scissor_ma5'] = df.groupby(codes).apply(
        lambda g: g['net_mf_amount'].sub(retail_net.loc[g.index])
    ).reset_index(level=0, drop=True)
    # Simpler approach:
    scissor = df['net_mf_amount'] - retail_net
    df['scissor_ma5'] = scissor.groupby(codes).transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    
    # === 因子14: 连续净流入天数 ===
    print("Computing consecutive inflow days...")
    df['consecutive_inflow'] = consecutive_count(df['net_mf_amount'], codes)
    
    # === 因子15: 资金流强度 ===
    print("Computing intensity factor...")
    total_amount = (df['buy_sm_amount'] + df['sell_sm_amount'] + 
                    df['buy_md_amount'] + df['sell_md_amount'] +
                    df['buy_lg_amount'] + df['sell_lg_amount'] +
                    df['buy_elg_amount'] + df['sell_elg_amount'])
    mf_intensity = df['net_mf_amount'] / total_amount.replace(0, np.nan)
    df['mf_intensity_ma5'] = mf_intensity.groupby(codes).transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    
    # === 输出 ===
    factor_cols = [
        'mf_net_ma5', 'mf_net_ma10', 'mf_net_ma20',
        'elg_ratio_ma5', 'elg_ratio_ma10', 'elg_ratio_ma20',
        'panic_ma5', 'panic_ma10', 'panic_ma20',
        'accel_3v3', 'accel_3v7', 'accel_5v20',
        'scissor_ma5',
        'consecutive_inflow',
        'mf_intensity_ma5',
    ]
    
    output_cols = ['ts_code', 'trade_date'] + factor_cols
    final = df[output_cols].copy()
    
    # 清理临时列
    for col in ['elg_ratio', 'panic_index']:
        if col in final.columns:
            final = final.drop(columns=[col])
    
    print(f"Saving to {OUTPUT_PATH}...")
    final.to_parquet(OUTPUT_PATH, index=False)
    
    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed:.1f}s ===")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Shape: {final.shape}")
    print(f"Stocks: {final['ts_code'].nunique()}")
    print(f"Date range: {final['trade_date'].min()} ~ {final['trade_date'].max()}")
    print(f"Factor columns ({len(factor_cols)}): {factor_cols}")
    
    # 样本数据
    print(f"\n=== Sample (last 5 rows) ===")
    print(final.tail(5).to_string())
    
    return final


if __name__ == '__main__':
    extract_all_factors()
