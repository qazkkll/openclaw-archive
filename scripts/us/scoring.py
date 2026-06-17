"""
🍤 统一评分路由 — 自动判断A股/美股，调用正确模型

用法:
    from scoring import score
    result = score('600519')  # → 自动走V1
    result = score('NVDA')    # → 自动走V4.2

无需手动选择模型，系统根据股票代码路由。
"""
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def is_a_stock(code):
    """判断是不是A股代码"""
    if not code:
        return False
    # A股代码特征：全数字、6位
    if code.isdigit() and len(code) == 6:
        return True
    # 带后缀的A股
    if code.endswith('.SZ') or code.endswith('.SH') or code.endswith('.SS'):
        return True
    return False

def is_us_stock(code):
    """判断是不是美股代码"""
    if not code:
        return False
    # A股的排除掉
    if is_a_stock(code):
        return False
    # 美股代码特征：字母、2-5位
    if code.isalpha() and 1 <= len(code) <= 5:
        return True
    return False

def resolve_market(code):
    """解析市场类型"""
    if is_a_stock(code):
        return 'a_stock'
    elif is_us_stock(code):
        return 'us_stock'
    return 'unknown'

# ===== A股评分 =====
def _a_score(close, high, low):
    """A股评分 = V1评分（MACD门+动态权重），熊市保护靠MACD门控"""
    from score_engine import v1_score_from_data
    return v1_score_from_data(close, high, low)

# ===== V4.2比例扣分（美股）=====
def _v42_score(close, high, low):
    """V4.2 比例扣分: 30日动量 - (距52周高位扣分)"""
    import yfinance as yf
    
    # 从config读参数
    try:
        with open(os.path.join(ROOT, 'config', 'strategy.json')) as f:
            import json
            strat = json.load(f)
            us = strat.get('us_stock', {})
    except:
        us = {}
    
    ds = us['deduct_start']            # 从strategy.json读取
    dc = us['deduct_coeff']            # 从strategy.json读取
    md = us.get('momentum_days', 30)  # 30日动量
    
    if len(close) < md:
        return 0.0
    
    cur = close[-1]
    past = close[-1-md] if len(close) > md else close[0]
    momentum = (cur - past) / past * 100
    
    high52 = max(close[-252:]) if len(close) >= 252 else max(close)
    dist_from_high = (high52 - cur) / high52 * 100
    
    deduction = 0
    if dist_from_high > ds:
        deduction = (dist_from_high - ds) * dc
    
    net = momentum - deduction
    return round(net, 1)


def score(code, close=None, high=None, low=None, details=False):
    """
    统一评分入口。
    
    参数:
        code: 股票代码 (如 '600519', 'NVDA')
        close/high/low: 可选，已获取的数据。没传则自动拉取。
        details: 是否返回详细信息
    
    返回:
        details=False → 评分(float)
        details=True  → {score, market, model, ...}
    """
    market = resolve_market(code)
    
    # 如果没有传数据，自动拉取
    volume = None
    if close is None:
        if market == 'a_stock':
            from data_source import AShareKline
            kl = AShareKline()
            data = kl.get_best(code)
            if data and len(data) >= 60:
                close = [d['close'] for d in data]
                high = [d['high'] for d in data]
                low = [d['low'] for d in data]
                if 'volume' in data[0]:
                    volume = [d['volume'] for d in data]
        elif market == 'us_stock':
            import yfinance as yf
            suffix = ''
            data = yf.download(code, period='1y', interval='1d', progress=False)
            if len(data) >= 60:
                close = list(data['Close'].values.flatten())
                high = list(data['High'].values.flatten())
                low = list(data['Low'].values.flatten())
    
    if not close or len(close) < 60:
        result = {'score': 0, 'market': market, 'model': 'none', 'error': '数据不足'}
        return result if details else 0
    
    # 路由到正确模型
    if market == 'a_stock':
        s = _a_score(close, high, low) or 0
        result = {
            'score': round(float(s), 1),
            'market': 'A股',
            'model': 'V1评分',
            'buy_threshold': 62,
            'sell_threshold': 50
        }
    elif market == 'us_stock':
        s = _v42_score(close, high, low)
        import json
        _cfg = json.load(open('config/strategy.json', encoding='utf-8'))
        _us = _cfg.get('us_stock', {})
        result = {
            'score': s,
            'market': '美股',
            'model': 'V4.2比例扣分',
            'momentum_days': _us.get('momentum_days', 30),
            'deduct_start': _us.get('deduct_start', 40),
            'deduct_coeff': _us.get('deduct_coeff', 0.7)
        }
    else:
        result = {'score': 0, 'market': market, 'model': 'unknown', 'warning': '无法识别市场'}
    
    # 信号灯
    if market == 'a_stock':
        if result['score'] >= 62:
            result['signal'] = '🟢 买入'
        elif result['score'] >= 50:
            result['signal'] = '🟡 观望'
        else:
            result['signal'] = '🔴 卖出/不推荐'
    else:
        if result['score'] >= 20:
            result['signal'] = '🟢 推荐'
        elif result['score'] >= 0:
            result['signal'] = '🟡 中性'
        else:
            result['signal'] = '🔴 不推荐'
    
    return result if details else result['score']


# ===== 快速验证 =====
if __name__ == '__main__':
    print('🍤 统一评分路由 — 自检')
    print()
    
    for code in ['600519', 'NVDA', 'MSFT', '000997']:
        r = score(code, details=True)
        print(f'  {code} ({r["market"]}): {r["score"]}分 → {r["signal"]} [{r["model"]}]')
    
    print()
    print('✅ 路由正常 — A股走V1，美股走V4.2')
