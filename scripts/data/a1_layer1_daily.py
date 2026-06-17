"""
Layer 1 - 每日宏观信号输出
用法: python scripts/layer1_daily.py
输出: 北向动量百分位 + 历史对比 + 市场状态判断
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 统一路径管理
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import NORTH_MONEY, INDEX_300

def load_json(path):
    with open(path, 'rb') as f:
        return json.load(f)

def ensure_float(v):
    if v is None: return 0.0
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        try: return float(v)
        except: return 0.0
    return 0.0

def main():
    t0 = time.time()
    
    # 加载数据
    north_data = load_json(NORTH_MONEY)
    records = north_data.get('records', north_data)
    
    klines = sorted(load_json(INDEX_300), key=lambda x: x['trade_date'])
    closes = [k['close'] for k in klines]
    dates = [k['trade_date'] for k in klines]
    
    # 计算北向动量 + 未来20日收益
    ndates = [r['trade_date'] for r in records]
    nvals = [ensure_float(r.get('north_money', 0)) for r in records]
    
    signals = []
    for i in range(59, len(ndates)):
        d = ndates[i]
        recent = nvals[max(0, i-59):i+1]
        sum20 = sum(recent[-20:])
        sum60 = sum(recent)
        momentum = round(sum20 / sum60, 3) if sum60 != 0 else 1.0
        
        # 未来20日沪深300收益
        future_ret = 0
        for j, kd in enumerate(dates):
            if kd == d and j + 20 < len(closes):
                future_ret = round((closes[j+20] / closes[j] - 1) * 100, 2)
                break
        
        signals.append({
            'date': d,
            'north_20d_sum': round(sum20 / 10000, 2),  # 万元→亿元
            'momentum': momentum,
            'future_ret_20d': future_ret
        })
    
    print("=" * 65)
    print("Layer 1 北向信号监控")
    print("=" * 65)
    print(f"数据: 北向资金 {len(signals)} 个交易日")
    print(f"      {signals[0]['date']} ~ {signals[-1]['date']}")
    print()
    
    # 当前
    cur = signals[-1]
    cur_mom = cur['momentum']
    
    # 历史百分位
    all_mom = sorted([s['momentum'] for s in signals])
    pct = sum(1 for m in all_mom if m < cur_mom) / len(all_mom) * 100
    print(f"当前信号")
    print(f"  北向20日累计: {cur['north_20d_sum']:+.0f} 亿元")
    print(f"  动量比(20d/60d): {cur_mom:.3f}")
    print(f"  历史百分位: {pct:.0f}%")
    
    if pct > 80:
        print(f"  状态: 🟢 北向加速流入 — 可进攻")
    elif pct > 60:
        print(f"  状态: 🟡 北向温和流入 — 中性偏多")
    elif pct > 40:
        print(f"  状态: ⚪ 北向中性 — 观望")
    elif pct > 20:
        print(f"  状态: 🟠 北向流出 — 偏防守")
    else:
        print(f"  状态: 🔴 北向大幅流出 — 防守")
    
    print()
    
    # 类似信号的历史表现
    print(f"类似动量水平的信号历史表现 (百分位{pct:.0f}%区间):")
    similar = [s for s in signals[:-20] if abs(s['momentum'] - cur_mom) < 0.05]
    if similar:
        avg_ret = sum(s['future_ret_20d'] for s in similar) / len(similar)
        win = sum(1 for s in similar if s['future_ret_20d'] > 0) / len(similar) * 100
        pos_avg = sum(s['future_ret_20d'] for s in similar if s['future_ret_20d'] > 0) / max(1, sum(1 for s in similar if s['future_ret_20d'] > 0))
        neg_avg = sum(s['future_ret_20d'] for s in similar if s['future_ret_20d'] <= 0) / max(1, sum(1 for s in similar if s['future_ret_20d'] <= 0))
        print(f"  出现次数: {len(similar)}")
        print(f"  平均+20d收益: {avg_ret:+.2f}%")
        print(f"  胜率: {win:.0f}%")
        print(f"  平均赢: {pos_avg:+.2f}% / 平均输: {neg_avg:+.2f}%")
        print(f"  最近一次类似信号: {similar[-1]['date']} (后续+20d: {similar[-1]['future_ret_20d']:+.2f}%)")
    
    print()
    
    # 近期北向趋势
    print(f"最近 20 个交易日北向:")
    for s in signals[-20:]:
        flow_str = '流入' if s['north_20d_sum'] > 0 else '流出'
        pace_str = '加速' if s['momentum'] > 0.38 else '减速'
        ret_str = f"后续+20d: {s['future_ret_20d']:+.2f}%" if s['future_ret_20d'] != 0 else "待定"
        print(f"  {s['date']} 累计{s['north_20d_sum']:>+7.0f}亿 {flow_str} {pace_str} 动量{s['momentum']:.3f} | {ret_str}")
    
    # 北向 vs 沪深300收益
    print()
    print("北向动量分位数 vs 后续20日收益:")
    for thr in [80, 60, 40, 20]:
        above = [s for s in signals if s['future_ret_20d'] != 0]
        mom_sorted = sorted([s['momentum'] for s in above])
        cutoff = len(mom_sorted) * thr // 100
        if cutoff >= len(mom_sorted): continue
        thr_val = mom_sorted[cutoff]
        group = [s for s in above if s['momentum'] >= thr_val]
        if group:
            avg = sum(s['future_ret_20d'] for s in group) / len(group)
            win = sum(1 for s in group if s['future_ret_20d'] > 0) / len(group) * 100
            print(f"  >{thr:>2}%分位({thr_val:.3f}): {len(group):3d}次 | avg+20d: {avg:+.2f}% | 胜率: {win:.0f}%")
    
    print()
    print(f"耗时: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
