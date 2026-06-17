"""
Layer 3 + 4 — 个股评分 + 买卖决策

用法：
  from a1_layer3_4_scoring import score_stock, make_decision
  
  # 单只股票评分
  result = score_stock('600032', market_score=46)
  # result = {score, v4_score, mf_score, val_score, decision, position_suggestion}
  
  # 批量扫描
  from a1_layer3_4_scoring import scan_top
  top = scan_top(score_engine, market_score=46, top_n=5)
"""

import json, os, sys, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 统一路径管理
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import NORTH_MONEY, DATA_DIR
WORKSPACE = DATA_DIR.replace("/data", "/workspace")

# 缓存（避免重复加载大文件）
_mf_cache = None
_db_cache = None

def _load_mf():
    """延迟加载资金流（只在需要时）"""
    global _mf_cache
    if _mf_cache is None:
        path = os.path.join(WORKSPACE, 'data', 'moneyflow_data.parquet')
        # 只读文件头部索引，需要时再seek具体股票
        _mf_cache = json.load(open(path, 'rb'))
    return _mf_cache

def _load_db():
    """延迟加载每日指标"""
    global _db_cache
    if _db_cache is None:
        path = os.path.join(WORKSPACE, 'data', 'daily_basic_data.parquet')
        _db_cache = json.load(open(path, 'rb'))
    return _db_cache


# ─── Layer 1 接口 ───

def get_market_score():
    """从 Layer 1 读取当前市场系数"""
    try:
        # 直接计算北向动量百分位，不走子进程
        north_data = json.load(open(NORTH_MONEY, 'rb'))
        records = north_data.get('records', north_data)
        ndates = [r['trade_date'] for r in records]
        nvals = []
        for r in records:
            try: v = float(r.get('north_money', 0) or 0)
            except: v = 0
            nvals.append(v)
        
        if len(nvals) < 60:
            return 50
        
        recent = nvals[-60:]
        sum20 = sum(recent[-20:])
        sum60 = sum(recent)
        momentum = sum20 / sum60 if sum60 != 0 else 1.0
        
        # 用全量历史算百分位
        all_mom = []
        for i in range(59, len(nvals)):
            s20 = sum(nvals[i-19:i+1])
            s60 = sum(nvals[i-59:i+1])
            all_mom.append(s20 / s60 if s60 != 0 else 1.0)
        
        pct = sum(1 for m in all_mom if m < momentum) / len(all_mom) * 100
        return round(pct)
    except Exception as e:
        return 50


# ─── Layer 2: 权重调整（活性因子驱动）───

# 活性因子来源：Layer 1 北向动量百分位
# 百分位越高(北向越强)：动量/技术因子权重越大
# 百分位越低(北向越弱)：估值/防御因子权重越大
# 连续平滑调整，不分段

def get_weights(market_score):
    """
    根据 Layer 1 北向动量百分位动态调整因子权重
    
    参数：
        market_score: 0-100, 来自 Layer 1 北向动量百分位
        (62% = 中性偏多，80% = 强烈看多，20% = 偏空)
    
    返回：
        {'v4': 0-1, 'moneyflow': 0-1, 'valuation': 0-1, 'momentum': 0-1}
    """
    # 基础权重（震荡市下的中性配置）
    base = {'v4': 0.30, 'moneyflow': 0.30, 'valuation': 0.30, 'momentum': 0.10}
    
    # 市场系数归一化到-1~1
    # 0%分位=-1(极空), 50%分位=0(中性), 100%分位=1(极多)
    t = (market_score - 50) / 50.0
    
    # 动量/技术因子权重：随t线性增加
    momentum_weight = base['momentum'] + t * 0.15
    v4_weight = base['v4'] + t * 0.15
    
    # 估值/防御因子权重：随t线性减少
    valuation_weight = base['valuation'] - t * 0.20
    
    # 资金流权重：中性，偏多时稍增(北向流入+资金流入=共振)
    moneyflow_weight = base['moneyflow'] + t * 0.05
    
    # 限制在合理范围
    momentum_weight = max(0.05, min(0.25, momentum_weight))
    v4_weight = max(0.15, min(0.50, v4_weight))
    valuation_weight = max(0.10, min(0.55, valuation_weight))
    moneyflow_weight = max(0.15, min(0.40, moneyflow_weight))
    
    # 归一化保证总和=1
    total = momentum_weight + v4_weight + valuation_weight + moneyflow_weight
    
    return {
        'v4': round(v4_weight / total, 3),
        'moneyflow': round(moneyflow_weight / total, 3),
        'valuation': round(valuation_weight / total, 3),
        'momentum': round(momentum_weight / total, 3)
    }


# ─── Layer 3: 因子计算 ───

def _v4_score(close, high, low, volume=None):
    """V4技术分 (0-100)，调用score_engine"""
    try:
        from us_score_engine import v1_score_from_data
        s = v1_score_from_data(close, high, low, volume)
        return float(s) if s is not None else 50
    except:
        return 50


def _moneyflow_score(code):
    """
    资金流分 (0-100)
    moneyflow_data.parquet 4.7GB, 需流式读取
    当前返回默认50
    """
    return 50


def _valuation_score(code, close_price=None):
    """
    估值分 (0-100)
    daily_basic_data.parquet 3.9GB, 需流式读取
    当前返回默认50
    """
    return 50


def _momentum_score(close):
    """动量修正 (0-100)"""
    if not close or len(close) < 20:
        return 50
    
    ret_20d = (close[-1] / close[-21] - 1) * 100 if close[-21] != 0 else 0
    ret_60d = (close[-1] / close[-61] - 1) * 100 if len(close) >= 61 and close[-61] != 0 else 0
    
    score = 50
    if ret_20d > 5: score += 15
    elif ret_20d > 0: score += 5
    elif ret_20d < -10: score -= 15
    elif ret_20d < -5: score -= 5
    
    if ret_60d > 15: score += 10
    elif ret_60d > 5: score += 5
    elif ret_60d < -15: score -= 10
    elif ret_60d < -5: score -= 5
    
    return max(0, min(100, score))


# ─── 综合评分 ───

def score_stock(code, close=None, high=None, low=None, volume=None, 
               market_score=None, 
               weights=None):
    """
    单只股票综合评分
    
    参数:
        code: 股票代码
        close/high/low/volume: K线数据
        market_score: 市场系数 (0-100)，None则自动读取
        weights: 因子权重，None则根据market_score自动计算
    
    返回:
        {'code': code, 'score': 0-100, 'v4': ..., 'mf': ..., 'val': ..., 
         'mom': ..., 'decision': 'buy/hold/watch/avoid', 
         'position': '重仓/轻仓/不参与'}
    """
    if market_score is None:
        market_score = get_market_score()
    
    if weights is None:
        weights = get_weights(market_score)
    
    # 计算各因子分
    v4 = _v4_score(close, high, low, volume) if close else 50
    mf = _moneyflow_score(code)
    val = _valuation_score(code, close[-1] if close else None)
    mom = _momentum_score(close) if close else 50
    
    # 加权综合
    raw = (v4 * weights['v4'] + 
           mf * weights['moneyflow'] + 
           val * weights['valuation'] + 
           mom * weights['momentum'])
    
    score = round(raw, 1)
    
    # ─── Layer 4: 决策 ───
    decision, position = _make_decision(score, market_score)
    
    return {
        'code': code,
        'score': score,
        'v4': round(v4, 1),
        'moneyflow': round(mf, 1),
        'valuation': round(val, 1),
        'momentum': round(mom, 1),
        'market_score': market_score,
        'weights': weights,
        'decision': decision,
        'position': position
    }


def _make_decision(score, market_score):
    """Layer 4: 买卖决策"""
    # 动态门槛
    if market_score > 65:
        buy_threshold = 60
        max_positions = 5
        position_size = 0.20
    elif market_score < 35:
        buy_threshold = 75
        max_positions = 2
        position_size = 0.30
    else:
        buy_threshold = 65
        max_positions = 3
        position_size = 0.30
    
    if score >= buy_threshold:
        decision = 'buy'
        position = f'{position_size*100:.0f}%仓位'
    elif score >= buy_threshold - 10:
        decision = 'watch'
        position = '观察'
    else:
        decision = 'avoid'
        position = '不参与'
    
    return decision, position


# ─── 批量扫描 ───

def scan_top(scored_stocks, market_score=None, top_n=5):
    """
    从已评分的股票列表中选出最优
    
    参数:
        scored_stocks: [{'code': ..., 'score': ...}, ...]
        market_score: 市场系数
        top_n: 返回数量
    
    返回: 排序后的Top N
    """
    if market_score is None:
        market_score = get_market_score()
    
    # 过滤掉不建议买入的
    buyable = [s for s in scored_stocks if s.get('decision') in ['buy', 'watch']]
    
    # 按分排序
    buyable.sort(key=lambda x: -x['score'])
    
    return {
        'market_score': market_score,
        'timestamp': __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M'),
        'top': buyable[:top_n],
        'total_scored': len(scored_stocks),
        'total_buyable': len(buyable)
    }


def demo_layer2():
    """演示Layer 2在不同市场状态下的权重变化"""
    print("Layer 2: 活性因子(北向动量)驱动权重调整")
    print(f"{'市场状态':<10} {'北向百分位':>10} {'V4':>6} {'资金流':>6} {'估值':>6} {'动量':>6}")
    print("-" * 50)
    
    for label, ms in [("极空", 15), ("偏空", 30), ("中性", 50), ("偏多", 70), ("极多", 85)]:
        w = get_weights(ms)
        print(f"{label:<10} {ms:>6}%      {w['v4']*100:>3.0f}%  {w['moneyflow']*100:>3.0f}%  {w['valuation']*100:>3.0f}%  {w['momentum']*100:>3.0f}%")


def main():
    """测试入口：对几个已知股票评分"""
    import random
    
    print("=" * 65)
    print("Layer 2+3+4 — 综合评分 + 买卖决策（测试）")
    print("=" * 65)
    
    # 演示Layer 2
    demo_layer2()
    
    print()
    print("=" * 65)
    
    # 读当前市场系数
    ms = get_market_score()
    print(f"\n当前市场系数: {ms}/100 ({'偏多' if ms>65 else '偏空' if ms<35 else '中性'})")
    
    weights = get_weights(ms)
    print(f"当前因子权重: V4={weights['v4']*100:.0f}% 资金流={weights['moneyflow']*100:.0f}% 估值={weights['valuation']*100:.0f}% 动量={weights['momentum']*100:.0f}%")
    
    # 测试几个已知股票
    test_codes = ['600032', '603373', '605018', '000668', '601869', '600519']
    
    print(f"\n{'代码':>8} {'综合分':>6} {'V4':>6} {'资金':>6} {'估值':>6} {'动量':>6} {'决策':>8} {'仓位':>8}")
    print("-" * 65)
    
    for code in test_codes:
        result = score_stock(code, market_score=ms)
        print(f"{result['code']:>8} {result['score']:>5.1f}  {result['v4']:>5.1f} {result['moneyflow']:>5.1f} {result['valuation']:>5.1f} {result['momentum']:>5.1f} {result['decision']:>8} {result['position']:>8}")
    
    print(f"\n完成")


if __name__ == "__main__":
    main()
