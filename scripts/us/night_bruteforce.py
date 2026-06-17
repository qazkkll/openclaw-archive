#!/usr/bin/env python3
"""
夜间暴力破解引擎 v1
路线A: V1+门控+资金流 (改良)
路线B: 全新因子 (激进)

运行在本地Windows，并行20线程
"""
import os, sys, json, time, datetime as dt, itertools, multiprocessing as mp
import numpy as np
import pandas as pd

WORKDIR = r'C:\workspace\av2'
os.makedirs(WORKDIR, exist_ok=True)

def log(msg):
    print(f'[{dt.datetime.now():%H:%M:%S}] {msg}', flush=True)

# ======================
# 行情数据加载
# ======================
def load_data():
    log('Loading data...')
    data = {}
    
    try:
        data['daily_ohlcv'] = pd.read_parquet(os.path.join(WORKDIR, 'daily_ohlcv.parquet'))
        log(f'  OHLCV: {len(data["daily_ohlcv"])} rows')
    except:
        log('  ⚠️ No OHLCV data yet')
    
    try:
        data['daily_basic'] = pd.read_parquet(os.path.join(WORKDIR, 'daily_basic.parquet'))
        log(f'  daily_basic: {len(data["daily_basic"])} rows')
    except:
        log('  ⚠️ No daily_basic data yet')
    
    return data


# ======================
# 技术指标计算 (精简版)
# ======================
def compute_indicators(df):
    """基于OHLCV计算基本技术指标"""
    if df is None or len(df) == 0:
        return None
    
    df = df.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values(['ts_code', 'trade_date'])
    
    # 保证每个股票数据完整
    # 均线
    df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    df['ma10'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(10, min_periods=5).mean())
    df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df['ma60'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(60, min_periods=30).mean())
    
    # 动量
    df['mom5'] = df.groupby('ts_code')['close'].transform(lambda x: x / x.shift(5) - 1)
    df['mom10'] = df.groupby('ts_code')['close'].transform(lambda x: x / x.shift(10) - 1)
    df['mom20'] = df.groupby('ts_code')['close'].transform(lambda x: x / x.shift(20) - 1)
    df['mom60'] = df.groupby('ts_code')['close'].transform(lambda x: x / x.shift(60) - 1)
    
    return df


# ======================
# 单一策略回测 (单个CPU)
# ======================
def backtest_single_strategy(args):
    """回测一组因子组合，返回绩效指标"""
    combo_id, params, df_data = args
    
    try:
        # 解参数
        momentums = params.get('momentum_weights', [0.4, 0.5, 0.6])  # 20d, 60d, 10d
        moneyflow_on = params.get('moneyflow_on', False)
        value_on = params.get('value_on', False)
        double_vol_on = params.get('double_vol_on', False)
        industry_mom_on = params.get('industry_mom_on', True)
        gate_level1 = params.get('gate_level1', None)  # 市场环境门
        gate_level2 = params.get('gate_level2', None)  # 选股确认门
        
        # 市场参数
        buy_threshold = params.get('buy_threshold', 0.6)
        max_holdings = params.get('max_holdings', 8)
        stop_loss = params.get('stop_loss', -0.08)
        
        # ====== 模拟回测 ======
        df_sorted = df_data.sort_values(['ts_code', 'trade_date'])
        
        # 计算综合评分
        # 动量评分 (LightGBM确认占73%预测力)
        df_sorted['score_mom'] = (
            df_sorted['mom20'].fillna(0) * momentums[0] +
            df_sorted['mom60'].fillna(0) * momentums[1] +
            df_sorted['mom10'].fillna(0) * momentums[2]
        )
        
        # 标准化
        df_sorted['score_mom'] = df_sorted.groupby('trade_date')['score_mom'].transform(
            lambda x: (x - x.mean()) / (x.std() + 0.001)
        )
        
        # V1基础分值 (模拟简化版)
        # MA5 > MA10 加分
        df_sorted['score_ma'] = ((df_sorted['ma5'] > df_sorted['ma10']) & 
                                  (df_sorted['close'] > df_sorted['ma5'])).astype(float)
        
        # 综合评分
        df_sorted['score_total'] = df_sorted['score_mom'] * 0.7 + df_sorted['score_ma'] * 0.3
        
        # 门控1: 市场环境
        if gate_level1 == 'conservative':
            # 震荡市: 提高买入门槛, 降低持仓
            buy_threshold = buy_threshold * 1.2
            max_holdings = max(1, max_holdings // 2)
        
        # 选择每天评分前N的股票
        selected = df_sorted[df_sorted['score_total'] > buy_threshold].copy()
        selected = selected.sort_values('score_total', ascending=False)
        selected = selected.groupby('trade_date').head(max_holdings)
        
        # 计算持仓收益 (简化: 买入后持有5日)
        selected = selected.copy()
        
        # 模拟5日持有收益
        def calc_forward_return(grp):
            grp = grp.sort_values('trade_date')
            grp['fwd_5d_return'] = grp.groupby('ts_code')['close'].transform(
                lambda x: x.shift(-5) / x - 1
            )
            return grp
        
        selected = calc_forward_return(selected)
        
        # 绩效统计
        if len(selected) == 0 or selected['fwd_5d_return'].isna().all():
            return {'combo_id': combo_id, 'win_rate': 0, 'avg_return': 0, 'total_return': 0, 'trades': 0}
        
        trades = selected.dropna(subset=['fwd_5d_return'])
        if len(trades) == 0:
            return {'combo_id': combo_id, 'win_rate': 0, 'avg_return': 0, 'total_return': 0, 'trades': 0}
        
        win_rate = (trades['fwd_5d_return'] > 0).mean()
        avg_return = trades['fwd_5d_return'].mean()
        
        # 累计收益 (几何累加)
        total_return = (1 + trades['fwd_5d_return']).prod() - 1
        
        return {
            'combo_id': combo_id,
            'win_rate': round(float(win_rate), 4),
            'avg_return': round(float(avg_return), 4),
            'total_return': round(float(total_return), 4),
            'trades': len(trades),
            'params': params
        }
    
    except Exception as e:
        return {'combo_id': combo_id, 'error': str(e)}


# ======================
# 路线A: 门控参数暴力搜索
# ======================
def sweep_route_A(df_data):
    """路线A: V1+门控 - 暴力搜索最优门控参数"""
    log('=== 路线A 暴力搜索 ===')
    
    # 参数网格
    param_grid = {
        'momentum_weights': [[0.3, 0.5, 0.2], [0.4, 0.4, 0.2], [0.2, 0.6, 0.2], [0.5, 0.3, 0.2]],
        'moneyflow_on': [False, True],
        'industry_mom_on': [True],
        'buy_threshold': [0.5, 0.6, 0.7, 0.8],
        'max_holdings': [5, 8, 10],
        'stop_loss': [-0.05, -0.08, -0.10],
        'gate_level1': [None, 'conservative'],  # None=牛市激进, conservative=震荡防守
        'gate_level2': [None, 'trend_confirm']
    }
    
    # 生成所有组合
    keys = ['momentum_weights', 'moneyflow_on', 'industry_mom_on', 
            'buy_threshold', 'max_holdings', 'stop_loss', 'gate_level1', 'gate_level2']
    values = [param_grid[k] for k in keys]
    
    all_configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        all_configs.append(config)
    
    log(f'  路线A组合数: {len(all_configs)}')
    
    # 给每个组合编号
    jobs = [(i, cfg, df_data) for i, cfg in enumerate(all_configs)]
    
    # 并行执行
    n_workers = min(20, mp.cpu_count())
    log(f'  并行 {n_workers} 核')
    
    with mp.Pool(n_workers) as pool:
        results = pool.map(backtest_single_strategy, jobs)
    
    # 排序 - 按total_return
    valid = [r for r in results if 'error' not in r and r['trades'] > 0]
    valid.sort(key=lambda x: x['total_return'], reverse=True)
    
    # Top 10
    log(f'\n  路线A TOP 10 参数组合:')
    for i, r in enumerate(valid[:10]):
        log(f'  #{i+1} 收益:{r["total_return"]:+.2%} 胜率:{r["win_rate"]:.1%} 交易:{r["trades"]}次')
    
    return valid[:20]


# ======================
# 路线B: 全新因子暴力搜索
# ======================
def sweep_route_B(df_data):
    """路线B: 新因子组合 - 暴力搜索"""
    log('=== 路线B 暴力搜索 ===')
    
    # 因子权重组合
    param_grid = {
        'mom20_weight': [0.2, 0.3, 0.4, 0.5],
        'mom60_weight': [0.1, 0.2, 0.3, 0.4],
        'mom10_weight': [0.1, 0.2, 0.3],
        'ma_trend_weight': [0.1, 0.2, 0.3],
        'volume_weight': [0, 0.1, 0.2],
        'buy_threshold': [0.4, 0.5, 0.6, 0.7],
        'max_holdings': [5, 8, 10],
        'stop_loss': [-0.05, -0.08, -0.10]
    }
    
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    
    # 生组合 (部分采样，否则太多)
    all_configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        all_configs.append(config)
    
    log(f'  路线B组合数: {len(all_configs)}')
    
    # 并行
    jobs = [(i, cfg, df_data) for i, cfg in enumerate(all_configs)]
    n_workers = min(20, mp.cpu_count())
    log(f'  并行 {n_workers} 核')
    
    with mp.Pool(n_workers) as pool:
        results = pool.map(backtest_single_strategy, jobs)
    
    valid = [r for r in results if 'error' not in r and r['trades'] > 0]
    valid.sort(key=lambda x: x['total_return'], reverse=True)
    
    log(f'\n  路线B TOP 10:')
    for i, r in enumerate(valid[:10]):
        log(f'  #{i+1} 收益:{r["total_return"]:+.2%} 胜率:{r["win_rate"]:.1%} 交易:{r["trades"]}次')
    
    return valid[:20]


# ======================
# 主流程
# ======================
if __name__ == '__main__':
    log('=' * 60)
    log('暴力破解引擎启动')
    log('=' * 60)
    
    mp.set_start_method('spawn', force=True)
    
    # 1. 加载数据
    data = load_data()
    if not data.get('daily_ohlcv') is not None and len(data.get('daily_ohlcv', [])) == 0:
        log('等待数据拉取完成...')
        sys.exit(1)
    
    # 2. 计算指标
    df_indicators = compute_indicators(data.get('daily_ohlcv'))
    if df_indicators is None:
        log('数据不完整，退出')
        sys.exit(1)
    
    # 3. 路线A暴力搜索
    log('开始路线A搜索...')
    result_a = sweep_route_A(df_indicators)
    
    # 4. 路线B暴力搜索
    log('开始路线B搜索...')
    result_b = sweep_route_B(df_indicators)
    
    # 5. 保存结果
    output = {
        'timestamp': dt.datetime.now().isoformat(),
        'route_a_top': result_a,
        'route_b_top': result_b
    }
    with open(os.path.join(WORKDIR, 'sweep_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    log(f'\n✅ 暴力搜索完成！结果已保存')
    log(f'  路线A TOP1: {result_a[0]["total_return"]:+.2%}' if result_a else '  路线A: 无结果')
    log(f'  路线B TOP1: {result_b[0]["total_return"]:+.2%}' if result_b else '  路线B: 无结果')
