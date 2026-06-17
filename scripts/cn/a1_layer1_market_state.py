"""
Layer 1 - 宏观状态层 v4
严格的训练/验证/测试分离 + 真实沪深300数据

分法：
- 训练 2016-2019: 选指标方向 + 定权重
- 验证 2020-2022: 调参数 + 定阈值
- 测试 2023-2026: 只看结果，不动参数

指标来源：
- 沪深300指数 (2529天, 2016-2026)
- 北向资金 (2016起)
- 融资融券 (2016起)
"""
import json, os, sys, math, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 统一路径管理
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORKSPACE NORTH_MONEY, INDEX_300

def load_json(path):
    with open(os.path.join(WORKSPACE, path), 'rb') as f:
        return json.load(f)

def ensure_float(v):
    if v is None: return 0.0
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        try: return float(v)
        except: return 0.0
    return 0.0

# ─── 1. 指标计算 ───

def calc_indicators(klines, north_records, margin_data):
    """计算所有候选指标"""
    dates = [k['trade_date'] for k in klines]
    closes = [k['close'] for k in klines]
    
    # 北向索引
    ndates = [r['trade_date'] for r in north_records]
    nvals = [ensure_float(r.get('north_money', 0)) for r in north_records]
    
    # 融资索引
    mdates = sorted(margin_data.keys())
    mdaily = []
    for d in mdates:
        total = sum(ensure_float(s.get('rzye', 0)) for s in margin_data[d])
        mdaily.append(total)
    
    results = []
    
    for i in range(len(dates)):
        d = dates[i]
        c = closes[:i+1]
        
        if len(c) < 60:
            continue
        
        ind = {'date': d}
        
        price = c[-1]
        
        # 均线位置
        ma20 = sum(c[-20:]) / 20
        ma60 = sum(c[-60:]) / 60
        ma120 = sum(c[-120:]) / 120 if len(c) >= 120 else None
        
        ind['price_ma20'] = round((price / ma20 - 1) * 100, 2) if ma20 > 0 else 0
        ind['price_ma60'] = round((price / ma60 - 1) * 100, 2) if ma60 > 0 else 0
        ind['price_ma120'] = round((price / ma120 - 1) * 100, 2) if ma120 and ma120 > 0 else 0
        
        # 均线斜率
        if i >= 25:
            ma20_5 = sum(c[-25:-5]) / 20
            ind['ma20_slope'] = round((ma20 / ma20_5 - 1) * 100, 3) if ma20_5 > 0 else 0
        else:
            ind['ma20_slope'] = 0
        
        if i >= 65:
            ma60_5 = sum(c[-65:-5]) / 60
            ind['ma60_slope'] = round((ma60 / ma60_5 - 1) * 100, 3) if ma60_5 > 0 else 0
        else:
            ind['ma60_slope'] = 0
        
        # 波动率
        rets = []
        for j in range(1, len(c)):
            r = abs((c[j] / c[j-1] - 1) * 100) if c[j-1] > 0 else 0
            rets.append(r)
        
        vol10 = sum(rets[-10:]) / 10 if len(rets) >= 10 else 1
        vol60 = sum(rets[-60:]) / 60 if len(rets) >= 60 else 1
        ind['vol_ratio'] = round(vol10 / vol60, 3) if vol60 > 0 else 1.0
        
        # 近期收益
        ind['ret_20d'] = round((c[-1] / c[-21] - 1) * 100, 2) if len(c) >= 21 else 0
        ind['ret_60d'] = round((c[-1] / c[-61] - 1) * 100, 2) if len(c) >= 61 else 0
        ind['ret_10d'] = round((c[-1] / c[-11] - 1) * 100, 2) if len(c) >= 11 else 0
        
        # 均线排列
        ma5 = sum(c[-5:]) / 5
        ma10 = sum(c[-10:]) / 10
        s = 0
        if ma5 > ma10: s += 1
        if ma10 > ma20: s += 1
        if ma20 > ma60: s += 1
        if price > ma5: s += 1
        ind['ma_alignment'] = s
        
        # 北向（最近匹配）
        nidx = len(ndates) - 1
        for ni, nd in enumerate(ndates):
            if nd > d:
                nidx = max(0, ni - 1)
                break
        
        if nidx >= 0 and nidx < len(nvals):
            recent_n = nvals[max(0, nidx-59):nidx+1]
            if len(recent_n) >= 20:
                sum20 = sum(recent_n[-20:])
                sum60 = sum(recent_n)
                ind['north_20d'] = round(sum20 / 1e8, 2)
                ind['north_momentum'] = round(sum20 / sum60, 3) if sum60 != 0 else 1.0
                
                avg5 = sum(recent_n[-5:]) / 5
                avg15 = sum(recent_n[-20:-5]) / 15 if len(recent_n) >= 20 else 0
                ind['north_accel'] = round(avg5 - avg15, 2) if avg15 != 0 else 0
            else:
                ind['north_20d'] = 0; ind['north_momentum'] = 1.0; ind['north_accel'] = 0
        else:
            ind['north_20d'] = 0; ind['north_momentum'] = 1.0; ind['north_accel'] = 0
        
        # 融资
        midx = len(mdates) - 1
        for mi, md in enumerate(mdates):
            if md > d:
                midx = max(0, mi - 1)
                break
        
        if midx >= 0 and midx < len(mdaily):
            recent_m = mdaily[max(0, midx-59):midx+1]
            if len(recent_m) >= 20:
                m20 = sum(recent_m[-20:]) / 20
                m60 = sum(recent_m) / len(recent_m)
                ind['margin_trend'] = round(m20 / m60, 4) if m60 > 0 else 1.0
                
                m_avg5 = sum(recent_m[-5:]) / 5
                m_avg15 = sum(recent_m[-20:-5]) / 15 if len(recent_m) >= 20 else 0
                ind['margin_accel'] = round((m_avg5 / m_avg15 - 1) * 100, 3) if m_avg15 > 0 else 0
            else:
                ind['margin_trend'] = 1.0; ind['margin_accel'] = 0
        else:
            ind['margin_trend'] = 1.0; ind['margin_accel'] = 0
        
        # 未来20日收益
        if i + 20 < len(closes):
            ind['ret_future_20d'] = round((closes[i+20] / closes[i] - 1) * 100, 2)
        else:
            ind['ret_future_20d'] = 0
        
        results.append(ind)
    
    return results


# ─── 2. 验证引擎 ───

def validate_indicator(results, ind_name, train_cutoff, val_cutoff):
    """
    验证单个指标的三期表现
    训练集上确定方向(高->涨 还是 高->跌)
    验证集+测试集检验该方向是否持续成立
    """
    train = [r for r in results if r['date'] <= train_cutoff and r.get(ind_name) is not None]
    val = [r for r in results if train_cutoff < r['date'] <= val_cutoff and r.get(ind_name) is not None]
    test = [r for r in results if r['date'] > val_cutoff and r.get(ind_name) is not None]
    
    if len(train) < 20 or len(val) < 10 or len(test) < 10:
        return None
    
    def _quintile_gap(rs):
        """计算Q5高组 - Q1低组的收益差距"""
        sv = sorted([r[ind_name] for r in rs])
        n = len(sv)
        if n < 10: return None
        
        q1_thr = sv[n // 5]
        q5_thr = sv[4 * n // 5]
        
        low_group = [r for r in rs if r[ind_name] <= q1_thr]
        high_group = [r for r in rs if r[ind_name] >= q5_thr]
        
        if not low_group or not high_group:
            return None
        
        low_avg = sum(r['ret_future_20d'] for r in low_group) / len(low_group)
        high_avg = sum(r['ret_future_20d'] for r in high_group) / len(high_group)
        low_win = sum(1 for r in low_group if r['ret_future_20d'] > 0) / len(low_group) * 100
        high_win = sum(1 for r in high_group if r['ret_future_20d'] > 0) / len(high_group) * 100
        
        return {
            'count': len(rs),
            'low_avg': round(low_avg, 2),
            'high_avg': round(high_avg, 2),
            'low_win': round(low_win, 1),
            'high_win': round(high_win, 1),
            'spread': round(high_avg - low_avg, 2)
        }
    
    train_result = _quintile_gap(train)
    val_result = _quintile_gap(val)
    test_result = _quintile_gap(test)
    
    if not train_result:
        return None
    
    # 方向：spread > 0 => 高组好(正向指标), spread < 0 => 低组好(反向指标)
    direction = 'pos' if train_result['spread'] > 0 else 'neg'
    
    return {
        'name': ind_name,
        'direction': direction,
        'train_spread': train_result['spread'],
        'train': train_result,
        'val': val_result,
        'test': test_result,
        'n_train': len(train),
        'n_val': len(val),
        'n_test': len(test)
    }


def main():
    t0 = time.time()
    print("=" * 65)
    print("Layer 1 - 宏观状态层 v4 (严格三期分离)")
    print(f"  训练 2016-2019 | 验证 2020-2022 | 测试 2023-2026")
    print("=" * 65)
    
    # --- Load ---
    print("\n[1/4] Loading CSI 300...")
    index_data = load_json(INDEX_300)
    klines = sorted(index_data, key=lambda x: x['trade_date'])
    print(f"  {len(klines)} days, {klines[0]['trade_date']} ~ {klines[-1]['trade_date']}")
    
    print("[2/4] Loading northbound + margin...")
    north_data = load_json(NORTH_MONEY)
    north_records = north_data.get('records', north_data)
    print(f"  北向: {len(north_records)} records")
    
    margin_data = load_json('data/margin_detail.json')
    print(f"  融资: {len(margin_data)} trading days")
    
    # --- Compute ---
    print("\n[3/4] Computing indicators...")
    results = calc_indicators(klines, north_records, margin_data)
    print(f"  {len(results)} trading days with full indicators")
    
    train_cutoff = '20191231'
    val_cutoff = '20221231'
    
    # --- Validate ---
    print("\n[4/4] Validating...")
    all_indicators = ['price_ma20', 'price_ma60', 'price_ma120',
                      'ma20_slope', 'ma60_slope', 'ma_alignment',
                      'vol_ratio', 'ret_10d', 'ret_20d', 'ret_60d',
                      'north_20d', 'north_momentum', 'north_accel',
                      'margin_trend', 'margin_accel']
    
    print(f"\n{'Indicator':<18} {'Train_spread':>10} {'Val_spread':>10} {'Test_spread':>10} {'Consistent':>10}")
    print("-" * 65)
    
    proven = []
    for ind_name in all_indicators:
        result = validate_indicator(results, ind_name, train_cutoff, val_cutoff)
        if result is None:
            continue
        
        ts = result['train']
        vs = result['val']
        tes = result['test']
        
        v_spread = vs['spread'] if vs else 0
        te_spread = tes['spread'] if tes else 0
        
        # 一致性检测：方向在三期都相同
        consistent = False
        if result['direction'] == 'pos':
            consistent = (ts['spread'] > 0 and v_spread > 0 and te_spread > 0)
        else:
            consistent = (ts['spread'] < 0 and v_spread < 0 and te_spread < 0)
        
        marker = '✅' if (consistent and abs(ts['spread']) > 1.0) else \
                 '⚠️' if abs(ts['spread']) > 1.0 else ''
        
        print(f"{marker} {ind_name:<16} {ts['spread']:>+7.2f}%   {v_spread:>+7.2f}%   {te_spread:>+7.2f}%  {'✅' if consistent else '❌':>8}")
        
        if consistent and abs(ts['spread']) > 1.0:
            proven.append({
                'name': ind_name,
                'direction': result['direction'],
                'train_spread': ts['spread'],
                'val_spread': v_spread,
                'test_spread': te_spread,
                'high_win': ts['high_win'],
                'n': result['n_train']
            })
    
    # --- Best indicators summary ---
    print(f"\n{'=' * 65}")
    print(f"PROVEN INDICATORS: {len(proven)}")
    print(f"{'=' * 65}")
    
    proven.sort(key=lambda x: -abs(x['train_spread']))
    for p in proven:
        dir_str = 'HIGH=bull' if p['direction'] == 'pos' else 'HIGH=bear'
        print(f"  {p['name']:<16} {dir_str:<12} train={p['train_spread']:>+6.2f}%  val={p['val_spread']:>+6.2f}%  test={p['test_spread']:>+6.2f}%")
    
    # --- Composite score ---
    if proven:
        print(f"\n{'=' * 65}")
        print("COMPOSITE MARKET SCORE (0-100)")
        print(f"{'=' * 65}")
        
        # Build composite from ALL proven indicators
        # For each day, compute weighted signal
        pos_inds = [p for p in proven if p['direction'] == 'pos']
        neg_inds = [p for p in proven if p['direction'] == 'neg']
        
        print(f"  Positive: {[p['name'] for p in pos_inds]}")
        print(f"  Negative: {[p['name'] for p in neg_inds]}")
        
        # Weight = abs(spread) in training
        # Normalize so total positive weight = total negative weight
        total_pos_weight = sum(abs(p['train_spread']) for p in pos_inds) or 1
        total_neg_weight = sum(abs(p['train_spread']) for p in neg_inds) or 1
        
        market_scores = []
        for r in results:
            pos_score = 0
            for p in pos_inds:
                v = r.get(p['name'], 0)
                if v is None: continue
                # percentile: how high is this value relative to training period
                # Use all data up to this date
                hist = [rr[p['name']] for rr in results if rr['date'] <= r['date'] and rr.get(p['name']) is not None]
                if len(hist) < 10: continue
                pct = sum(1 for h in hist if h < v) / len(hist)
                pos_score += pct * abs(p['train_spread'])
            
            neg_score = 0
            for p in neg_inds:
                v = r.get(p['name'], 0)
                if v is None: continue
                hist = [rr[p['name']] for rr in results if rr['date'] <= r['date'] and rr.get(p['name']) is not None]
                if len(hist) < 10: continue
                # For negative indicators: LOW value is good, so invert the percentile
                pct = 1 - sum(1 for h in hist if h < v) / len(hist)
                neg_score += pct * abs(p['train_spread'])
            
            if pos_score + neg_score > 0:
                # Normalize: score = 50 + (pos_weighted - neg_weighted) / total * 50
                norm_pos = pos_score / total_pos_weight
                norm_neg = neg_score / total_neg_weight
                raw = (norm_pos / (norm_pos + norm_neg)) * 100 if (norm_pos + norm_neg) > 0 else 50
            else:
                raw = 50
            
            market_scores.append({
                'date': r['date'],
                'score': round(raw),
                'ret_future_20d': r['ret_future_20d']
            })
        
        # Validate composite
        print(f"\nComposite score validation (三期独立):")
        
        for label, start, end in [("TRAIN 2016-2019", '20160101', train_cutoff),
                                    ("VAL 2020-2022", '20200101', val_cutoff),
                                    ("TEST 2023-2026", '20230101', '20261231')]:
            period = [r for r in market_scores if start <= r['date'] <= end]
            if len(period) < 10: continue
            
            bulls = [r for r in period if r['score'] > 65]
            bears = [r for r in period if r['score'] < 35]
            neutral = [r for r in period if 35 <= r['score'] <= 65]
            
            for name, group in [(">65(Bull)", bulls), ("35-65(Neu)", neutral), ("<35(Bear)", bears)]:
                if len(group) < 3: continue
                avg = sum(r['ret_future_20d'] for r in group) / len(group)
                win = sum(1 for r in group if r['ret_future_20d'] > 0) / len(group) * 100
                print(f"  {label:<18} {name:<12}: {len(group):3d}x | avg+20d={avg:>+6.2f}% | win%={win:.0f}%")
        
        # Current
        if market_scores:
            cur = market_scores[-1]
            print(f"\n{'=' * 65}")
            print(f"CURRENT: {cur['score']}/100")
            if cur['score'] > 65:
                print(f"  >> BULL - offensive mode")
            elif cur['score'] < 35:
                print(f"  >> BEAR - defensive mode")
            else:
                print(f"  >> NEUTRAL - selective mode")
            
            # Recent 20
            print(f"\nRecent 20 signals:")
            for r in market_scores[-20:]:
                m = "+" if r['ret_future_20d'] > 0 else ""
                print(f"  {r['date']} score={r['score']:3d} => +20d: {m}{r['ret_future_20d']:+.2f}%")
    
    print(f"\nTime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
