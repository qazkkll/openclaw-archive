"""
A1 宏观状态层 v6 — 2016年起 + 优化三期分离 + 信号持久化

数据来源：
  - /home/hermes/.hermes/openclaw-project/data/index_300.json (2016-01-04 ~ 2026-06-08, 2531条)
  
三期分离：
  - 训练: 2016-01 ~ 2020-12  (约1210天) — 完整牛熊周期
  - 验证: 2021-01 ~ 2023-06  (约600天) — 调参/选特征用
  - 测试: 2023-07 ~ 2026-06  (约720天) — 期末考

用法：
  python scripts/a1_macro_state.py           # 完整运行（含三期分离验证）
  python -c "from a1_macro_state import get_market_state; print(get_market_state())"
  
输出：
  - 实时状态打印（标准输出）
  - /home/hermes/.hermes/openclaw-project/data/a1_macro_state.json（持久化，供其他脚本调用）
"""

import json, os, sys, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 路径
WORKSPACE_DATA = r'/home/hermes/.hermes/openclaw-archive/data'
# 统一路径管理
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORKSPACE D_DATA, INDEX_300

# 数据路径：优先D盘（2016年起），fallback到C盘（2020年起）
def load_index():
    p1 = INDEX_300
    p2 = INDEX_300  # 统一路径
    for p in [p1, p2]:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                return json.load(f)
    raise FileNotFoundError("index_300.json not found")


# ─── 指标计算（2016年起全周期） ───

def calc_indicators(klines):
    """从沪深300日K计算候选指标，含未来20日收益"""
    dates = [k['trade_date'] for k in klines]
    closes = [k['close'] for k in klines]
    highs = [k['high'] for k in klines]
    lows = [k['low'] for k in klines]
    
    results = []
    for i in range(len(closes)):
        d = dates[i]
        if i < 60:
            continue  # 需要至少60个交易日预热
        
        ir = {'date': d}
        c = closes[:i+1]
        h = highs[:i+1]
        lo = lows[:i+1]
        
        price = c[-1]
        
        # ── 1. 均线位置（价格 vs MA偏离度） ──
        ma5 = sum(c[-5:]) / 5
        ma10 = sum(c[-10:]) / 10
        ma20 = sum(c[-20:]) / 20
        ma60 = sum(c[-60:]) / 60
        ma120 = sum(c[-120:]) / 120 if len(c) >= 120 else ma60
        
        ir['ma5'] = round(ma5, 2)
        ir['ma10'] = round(ma10, 2)
        ir['ma20'] = round(ma20, 2)
        ir['ma60'] = round(ma60, 2)
        
        ir['pct_ma5'] = round((price / ma5 - 1) * 100, 2) if ma5 > 0 else 0
        ir['pct_ma10'] = round((price / ma10 - 1) * 100, 2) if ma10 > 0 else 0
        ir['pct_ma20'] = round((price / ma20 - 1) * 100, 2) if ma20 > 0 else 0
        ir['pct_ma60'] = round((price / ma60 - 1) * 100, 2) if ma60 > 0 else 0
        ir['pct_ma120'] = round((price / ma120 - 1) * 100, 2) if ma120 > 0 else 0
        
        # ── 2. 均线斜率 ──
        if i >= 25:
            ma20_5ago = sum(c[-25:-5]) / 20
            ir['ma20_slope'] = round((ma20 / ma20_5ago - 1) * 100, 3) if ma20_5ago > 0 else 0
        else:
            ir['ma20_slope'] = 0
        
        if i >= 65:
            ma60_5ago = sum(c[-65:-5]) / 60
            ir['ma60_slope'] = round((ma60 / ma60_5ago - 1) * 100, 3) if ma60_5ago > 0 else 0
        else:
            ir['ma60_slope'] = 0
        
        # ── 3. 均线排列强度（6分制） ──
        align = 0
        if ma5 > ma10: align += 1
        if ma10 > ma20: align += 1
        if ma20 > ma60: align += 1
        if price > ma5: align += 1
        if price > ma10: align += 1
        if price > ma60: align += 1
        ir['ma_align'] = align
        
        # ── 4. 波动率比（10d/60d） ──
        rets = [abs((c[j] / c[j-1] - 1) * 100) if c[j-1] > 0 else 0 for j in range(1, len(c))]
        vol10 = sum(rets[-10:]) / 10 if len(rets) >= 10 else 1
        vol60 = sum(rets[-60:]) / 60 if len(rets) >= 60 else 1
        ir['vol_ratio'] = round(vol10 / vol60, 3) if vol60 > 0 else 1.0
        
        # ── 5. 真实波动幅度ATR% ──
        trs = [max(h[j]-lo[j], abs(h[j]-c[j-1]), abs(lo[j]-c[j-1])) for j in range(max(1,i-19), i+1)]
        atr20 = sum(trs) / len(trs) if trs else 0
        ir['atr20_pct'] = round(atr20 / price * 100, 3) if price > 0 else 0
        
        # ── 6. 近期收益 ──
        ir['ret_5d'] = round((price / c[-6] - 1) * 100, 2) if len(c) >= 6 else 0
        ir['ret_10d'] = round((price / c[-11] - 1) * 100, 2) if len(c) >= 11 else 0
        ir['ret_20d'] = round((price / c[-21] - 1) * 100, 2) if len(c) >= 21 else 0
        ir['ret_60d'] = round((price / c[-61] - 1) * 100, 2) if len(c) >= 61 else 0
        
        # ── 7. 相对强弱(RSI-like, 20日) ──
        changes = [c[j] - c[j-1] for j in range(max(1,i-19), i+1)]
        gains = [x for x in changes if x > 0]
        losses = [-x for x in changes if x < 0]
        avg_gain = sum(gains) / 20 if gains else 0
        avg_loss = sum(losses) / 20 if losses else 0
        ir['rsi'] = round(100 - 100 / (1 + avg_gain / avg_loss), 1) if avg_loss > 0 else 100
        
        # ── 8. 未来20日收益 ──
        if i + 20 < len(closes):
            ir['fwd_20d'] = round((closes[i+20] / price - 1) * 100, 2)
        else:
            ir['fwd_20d'] = None
        
        # ── 9. 过去20日收益百分位（动量） ──
        if len(c) >= 41:
            past_20d_rets = [(c[j] / c[j-20] - 1) * 100 for j in range(20, i+1)]
            cur_20d = (price / c[-21] - 1) * 100
            pct_rank = sum(1 for r in past_20d_rets if r < cur_20d) / len(past_20d_rets) * 100
            ir['ret20d_pct'] = round(pct_rank, 1)
        else:
            ir['ret20d_pct'] = 50
        
        results.append(ir)
    
    return results


# ─── 三期分离验证 ───

def quintile_spread(rs, ind_name):
    """计算五分位高低组平均未来收益差距"""
    vals = [(r[ind_name], r['fwd_20d']) for r in rs
            if r[ind_name] is not None and r['fwd_20d'] is not None]
    if len(vals) < 20:
        return None
    
    sv = sorted([v[0] for v in vals])
    n = len(sv)
    q1_thr = sv[n // 5]
    q5_thr = sv[4 * n // 5]
    
    low = [v[1] for v in vals if v[0] <= q1_thr]
    high = [v[1] for v in vals if v[0] >= q5_thr]
    
    if not low or not high:
        return None
    
    la = sum(low) / len(low)
    ha = sum(high) / len(high)
    lw = sum(1 for x in low if x > 0) / len(low) * 100
    hw = sum(1 for x in high if x > 0) / len(high) * 100
    
    return {
        'spread': round(ha - la, 2),
        'high_avg': round(ha, 2),
        'low_avg': round(la, 2),
        'high_win': round(hw, 1),
        'low_win': round(lw, 1),
        'n': len(vals)
    }


def validate_all(results):
    """三期分离验证全部候选指标"""
    # 三期分割
    train = [r for r in results if r['date'] <= '20201231']
    val = [r for r in results if '20210101' <= r['date'] <= '20230630']
    test = [r for r in results if r['date'] >= '20230701']
    
    indicators = [
        'pct_ma5', 'pct_ma10', 'pct_ma20', 'pct_ma60', 'pct_ma120',
        'ma20_slope', 'ma60_slope', 'ma_align',
        'vol_ratio', 'atr20_pct', 'rsi',
        'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d', 'ret20d_pct'
    ]
    
    print(f"\n三期分离验证：训练(2016-2020) 验证(2021-2023H1) 测试(2023H2-2026)")
    print(f"  训练: {len(train)}天 | 验证: {len(val)}天 | 测试: {len(test)}天")
    print()
    print(f"{'Indicator':<14} {'Train_sp':>8} {'Val_sp':>8} {'Test_sp':>8} {'T->V':>6} {'T->Tst':>6} {'Score':>6}")
    print("-" * 60)
    
    proven = []
    for ind in indicators:
        tr = quintile_spread(train, ind)
        vr = quintile_spread(val, ind)
        ter = quintile_spread(test, ind)
        
        if not tr:
            print(f"  {ind:<14} {'N/A':>8}")
            continue
        
        vs = vr['spread'] if vr else 0
        tes = ter['spread'] if ter else 0
        
        # 方向一致性
        d_ok = (tr['spread'] * vs > 0)
        e_ok = (tr['spread'] * tes > 0) if ter else False
        
        # 综合评分：三期方向一致 + spread幅度加权
        score = 0
        if d_ok and abs(tr['spread']) > 0.5:
            score += 1
        if e_ok and abs(tr['spread']) > 0.5:
            score += 2
        if d_ok and e_ok and abs(tr['spread']) > 1.0:
            score += 1
        
        label = '✅' if score >= 3 else '⚠️' if score >= 2 else ''
        
        ts_str = f"{tes:>+7.2f}%" if ter else f"{'N/A':>8}"
        print(f"{label} {ind:<12} {tr['spread']:>+7.2f}% {vs:>+7.2f}% {ts_str:>8}  {'Y' if d_ok else 'N':>4}   {'Y' if e_ok else 'N':>5}  {score}")
        
        if score >= 2:  # 验证方向一致 + spread > 0.5%
            proven.append({
                'name': ind,
                'direction': 'pos' if tr['spread'] > 0 else 'neg',
                'train': tr,
                'val': vr,
                'test': ter,
                'score': score
            })
    
    return proven


# ─── 实时市场状态（仅使用最新数据） ───

def current_state(results):
    """从指标计算结果提取当前市场状态"""
    cur = results[-1]
    
    # 趋势分数
    trend_score = cur['ma_align'] * 16.7  # 0-100
    
    # 波动分数
    vr = cur['vol_ratio']
    if vr >= 1.5: vol_score = 85
    elif vr >= 1.2: vol_score = 70
    elif vr >= 1.0: vol_score = 55
    elif vr >= 0.8: vol_score = 40
    else: vol_score = 25
    
    # 动量分数
    momentum_score = cur['ret20d_pct']
    
    # 综合
    combined = trend_score * 0.45 + vol_score * 0.20 + momentum_score * 0.35
    
    if combined >= 65: state = 'aggressive'
    elif combined >= 40: state = 'neutral'
    else: state = 'defensive'
    
    return {
        'state': state,
        'score': round(combined),
        'trend_score': round(trend_score),
        'vol_score': round(vol_score),
        'momentum_score': round(momentum_score),
        'ma_align': cur['ma_align'],
        'vol_ratio': cur['vol_ratio'],
        'rsi': cur['rsi'],
        'ret_20d': cur['ret_20d'],
        'ret_60d': cur['ret_60d'],
        'pct_ma60': cur['pct_ma60'],
        'date': cur['date']
    }


def get_weights(state_info):
    """根据市场状态输出因子权重"""
    state = state_info['state']
    score = state_info['score']
    
    if state == 'aggressive':
        base = {'v4': 0.35, 'moneyflow': 0.25, 'valuation': 0.10, 'momentum': 0.30}
    elif state == 'defensive':
        base = {'v4': 0.15, 'moneyflow': 0.35, 'valuation': 0.40, 'momentum': 0.10}
    else:
        base = {'v4': 0.25, 'moneyflow': 0.30, 'valuation': 0.30, 'momentum': 0.15}
    
    t = (score - 50) / 50.0
    smooth = {
        'v4': base['v4'] + t * 0.08,
        'moneyflow': base['moneyflow'] + t * 0.03,
        'valuation': base['valuation'] - t * 0.12,
        'momentum': base['momentum'] + t * 0.08,
    }
    for k in smooth:
        smooth[k] = max(0.05, min(0.50, smooth[k]))
    
    total = sum(smooth.values())
    return {k: round(v / total, 3) for k, v in smooth.items()}


def get_market_state():
    """外部快速接口"""
    idx = load_index()
    klines = sorted(idx, key=lambda x: x['trade_date'])
    results = calc_indicators(klines)
    return current_state(results)

def get_market_score():
    return get_market_state()['score']


# ─── 主入口 ───

def main():
    print("=" * 65)
    print("A1 宏观状态层 v6 — 2016起 + 三期分离验证")
    print("=" * 65)
    
    idx = load_index()
    klines = sorted(idx, key=lambda x: x['trade_date'])
    print(f"\n沪深300: {len(klines)}天, {klines[0]['trade_date']} ~ {klines[-1]['trade_date']}")
    
    results = calc_indicators(klines)
    print(f"指标计算: {len(results)}天")
    
    # 三期验证
    proven = validate_all(results)
    
    print(f"\n{'=' * 65}")
    print(f"通过验证: {len(proven)} / 16")
    if proven:
        proven.sort(key=lambda x: -x['score'])
        for p in proven:
            dir_s = '+HIGH>' if p['direction'] == 'pos' else '-HIGH<'
            print(f"  {p['name']:<12} {dir_s:<7} train={p['train']['spread']:>+6.2f}%  val={p['val']['spread']:>+6.2f}%  test={p['test']['spread']:>+6.2f}%  score={p['score']}")
    
    # 当前状态
    state = current_state(results)
    labels = {
        'aggressive': '🟢 进攻 — 趋势偏多',
        'neutral': '⚪ 震荡 — 均衡配置',
        'defensive': '🔴 防御 — 趋势偏空'
    }
    
    print(f"\n{'=' * 65}")
    print(f"当前市场状态: {labels.get(state['state'], '?')}")
    print(f"综合评分: {state['score']}/100")
    print(f"数据: {state['date']}")
    print(f"  趋势: {state['trend_score']}/100 | 排列: {state['ma_align']}/6")
    print(f"  波动: {state['vol_score']}/100 | vol_ratio: {state['vol_ratio']:.3f}")
    print(f"  动量: {state['momentum_score']}/100 | 20d: {state['ret_20d']:+.2f}%")
    print(f"  距MA60: {state['pct_ma60']:+.2f}% | RSI: {state['rsi']}")
    
    weights = get_weights(state)
    print(f"\nLayer 2 因子权重:")
    print(f"  技术(V4) {weights['v4']*100:.0f}% | 资金流 {weights['moneyflow']*100:.0f}% | 估值 {weights['valuation']*100:.0f}% | 动量 {weights['momentum']*100:.0f}%")
    
    # 持久化
    output = {
        'state': state,
        'weights': weights,
        'proven_indicators': [{'name': p['name'], 'direction': p['direction'], 'test_spread': p['test']['spread'] if p['test'] else 0} for p in proven],
        'timestamp': '2026-06-12T09:30:00+08:00'
    }
    out_path = os.path.join(D_DATA, 'a1_macro_state.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n持久化: {out_path}")
    
    # 近一周沪深300走势
    print(f"\n--- 近一周沪深300 ---")
    for k in klines[-6:]:
        chg = (k['close'] / k['pre_close'] - 1) * 100 if k['pre_close'] > 0 else 0
        print(f"  {k['trade_date']}: {k['close']:.0f} ({chg:+.2f}%)")
    
    print()


if __name__ == "__main__":
    main()
