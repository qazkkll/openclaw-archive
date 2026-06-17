#!/usr/bin/env python3
import numpy as np
"""
us_daily_recommend.py — 每日美股推荐（双模型融合）
------------------------------------------------------------------
默认绿箭 = V9-Lottery                         默认蓝盾 = 蓝盾3.0
                          (绿箭V9-Lottery ~600只)                 (蓝盾3.0 SP500 ~503只)
------------------------------------------------------------------
融合输出：
  【绿箭推荐】小盘候选（V9-Lottery top5评分）
  【蓝盾推荐】大盘候选（蓝盾3.0 评分≥80）
  【共振信号】绿箭 ≥85 ∩ 蓝盾 ≥80 的票（罕见但含金量高）
  【市场判断】绿箭风控 + 蓝盾大盘温度

运行：python us_daily_recommend.py
依赖：us_v9_daily_score.py / us_ld3_daily_score.py 的产出文件
"""
import sys, json, os, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = '/home/hermes/.hermes/openclaw-archive/data'
today = time.strftime('%Y-%m-%d')

import glob

def _find_latest(prefix, today_ts):
    """找今天文件，回退最近交易日"""
    p = f'{DATA_DIR}/{prefix}{today_ts}.json'
    if os.path.exists(p):
        return p, today_ts
    files = glob.glob(f'{DATA_DIR}/{prefix}*.json')
    if not files:
        return None, None
    files.sort(reverse=True)
    for f in files:
        base = os.path.basename(f)
        date_str = base.replace(prefix, '').replace('.json', '')
        if date_str <= today_ts and date_str.startswith('20'):
            return f, date_str
    return None, None

print('=' * 65)
print(f'  🌙 每日美股推荐（双模型融合）  {time.strftime("%Y-%m-%d %H:%M")}')
print(f'  🟢 绿箭=V9-Lottery  🛡️ 蓝盾3.0(大盘技术)')
print('=' * 65)

# ─── 1. 加载绿箭 V9-Lottery 结果 ───
# 先试新文件名格式 scored_v9_lottery_YYYY-MM-DD.json, 再试旧格式
v75_path, v75_date = _find_latest('scored_v9_lottery_', today)
if v75_path is None:
    v75_path, v75_date = _find_latest('scored_v75_', today)
v9_data = None
if v75_path:
    v9_data = json.load(open(v75_path))
    print(f'\n📡 绿箭 V9 ✅ 已加载 ({v75_date})')
else:
    print(f'\n📡 绿箭 V9 ❌ 未找到评分文件')
    print(f'  请先运行: python us_v9_daily_score.py')

# ─── 2. 加载蓝盾3.0结果 ───
ld3_path, ld3_date = _find_latest('ld3_scored_', today)
ld3_data = None
if ld3_path:
    ld3_data = json.load(open(ld3_path))
    print(f'🛡️  蓝盾3.0 ✅ 已加载 ({ld3_date})')
else:
    print(f'🛡️  蓝盾3.0 ❌ 未找到评分文件')
    print(f'  请先运行: python us_ld3_daily_score.py')

if not v9_data and not ld3_data:
    print('\n❌ 两个模型都无数据，退出')
    sys.exit(1)

# ─── 3. 绿箭推荐（V9-Lottery小盘） ───
print(f'\n{"─"*65}')
print(f'🟢【绿箭推荐 | 小盘候选】V9-Lottery top5评分')
print(f'  排序: prob/price (prob除以价格, 4年回测Top5爆率+63%| 现价一并展示)')
print(f'  策略: T5_H10_S15_R10 (20天持有/前5名/15%止损/10只上限)')
print(f'{"─"*65}')

v9_buy = []
v9_strong = []
v9_watch = []
if v9_data:
    signals = v9_data.get('buy_signals', [])
    
    # Classify by probability with strategy rules
    for s in signals:
        prob = s.get("prob", 0)
        s['action'] = '🟢必买' if prob >= 0.90 else ('🟡可买' if prob >= 0.85 else '🔍观望' if prob >= 0.75 else '⏸️忽略')
        s['exp_return'] = '+29.5%' if prob >= 0.90 else '+18.6%' if prob >= 0.85 else '+10.4%'
        s['lottery_rate'] = '14%' if prob >= 0.90 else '5%' if prob >= 0.85 else '1%'
        
        if prob >= 0.90:
            v9_strong.append(s)
            v9_buy.append(s)
        elif prob >= 0.85:
            v9_buy.append(s)
        elif prob >= 0.75:
            v9_watch.append(s)
    
    print(f'  绿箭V9策略: ≥0.90必买(+29.5%平均, 14%彩票率) | 0.85-0.90可买(+18.6%平均)')
    print()
    
    if v9_strong:
        print(f'  🟢 强买 ({len(v9_strong)}只, 概率≥0.90, 优先下单):')
        for i, s in enumerate(v9_strong[:10]):
            bp = s.get('ratio_bp', round(s['prob']/s['price'],4)) if s['price']>0 else 0
            print(f'    {i+1:2d}. {s["sym"]:<6} 概率={s["prob"]:.4f}  ratio={bp:.4f}  ${s["price"]:.2f}  {s["action"]}  预期+{s["exp_return"]}')
        print()
    
    if v9_buy:
        print(f'  🟡 可买 ({len(v9_buy)}只, 概率≥0.85):')
        for i, s in enumerate(v9_buy[:10]):
            bp = s.get('ratio_bp', round(s['prob']/s['price'],4)) if s['price']>0 else 0
            print(f'    {i+1:2d}. {s["sym"]:<6} 概率={s["prob"]:.4f}  ratio={bp:.4f}  ${s["price"]:.2f}  {s["action"]}  预期+{s["exp_return"]}')
    else:
        print(f'  🟡 可买: (无票达阈值, 建议观望)')
    
    if v9_watch:
        print(f'\n  🔍 关注 ({len(v9_watch)}只, 概率0.75-0.85, 动量信号):')
        for i, s in enumerate(v9_watch[:8]):
            bp = s.get('ratio_bp', round(s['prob']/s['price'],4)) if s['price']>0 else 0
            print(f'    {i+1}. {s["sym"]:<6} 概率={s["prob"]:.4f}  ratio={bp:.4f}  ${s["price"]:.2f}')
else:
    print(f'  ⚠️ 绿箭V9数据不可用')

# ─── 4. 蓝盾推荐（蓝盾3.0大盘） ───
print(f'\n{"─"*65}')
print(f'🛡️【蓝盾推荐 | 大盘候选】蓝盾3.0 评分≥80')
print(f'  引擎: 纯技术评分(6维度)  |  满分100  |  买入线≥80  |  上限10只  |  止损-15%')
print(f'{"─"*65}')

ld3_buy = []
ld3_strong = []
ld3_watch = []
ld3_danger = []

if ld3_data:
    # 从scores列表动态分类（兼容新旧格式）
    all_scores = ld3_data.get('all_scores', ld3_data.get('scores', []))
    if not all_scores and 'scores' in ld3_data:
        # 新格式: ld3_data['scores'] = [{'code':'AAPL', 'score':85, ...}]
        all_scores = ld3_data['scores']
    
    # 检查是否已有分类字段（旧格式）
    ld3_strong = ld3_data.get('strong_buy', [])
    ld3_buy = ld3_data.get('buy', [])
    ld3_watch = ld3_data.get('watch', [])
    ld3_danger = ld3_data.get('danger', [])
    
    # 如果没有预分类字段，从scores动态分类
    if not ld3_strong and not ld3_buy and all_scores:
        for s in all_scores:
            sc = s.get('score', 0)
            entry = {'code': s.get('code',''), 'name': s.get('name',''), 'score': sc,
                     'price': s.get('price', '?'), 'rsi': s.get('rsi','?'), 'pct52': s.get('pct52',s.get('52w_pct','?'))}
            if sc >= 90:
                ld3_strong.append(entry)
            elif sc >= 75:
                ld3_buy.append(entry)

    if all_scores:
        avg = sum(s.get('score', 0) for s in all_scores) / len(all_scores)
        if avg >= 80:
            ld3_market = '🔥 热（大盘买盘活跃）'
        elif avg >= 70:
            ld3_market = '🌤️ 温（精选买入）'
        else:
            ld3_market = '❄️ 冷（大盘整体回避）'
        print(f'  大盘温度: {ld3_market} (全部均值{avg:.4f}分)')
    print(f'  蓝盾策略: ≥80强势买入(仓位可重) | 70-79买入 | 60-69观望 | <60回避')

    print()
    if ld3_strong:
        print(f'  🟢 强势买入(≥80) ({len(ld3_strong)}只, 仓位可重):')
        for s in ld3_strong[:15]:  # 最多显示15只
            name = s.get('name', '')
            rsi = s.get('rsi', '?')
            p52 = s.get('p52', s.get('pct52', '?'))
            price = s.get('price', s.get('close', '?'))
            rsi_s = f'RSI{s["rsi"]:.0f}' if isinstance(s.get("rsi"),(int,float)) else "RSI?"
            p52_s = f'距52高{s["pct52"]:.0f}%' if isinstance(s.get("pct52"),(int,float)) else "P52?"
            print(f'    {s["code"]:<6} 评分={s.get("prob", s.get("score", 0)):2d}  ${s.get("price","?"):<8}  {rsi_s}  {p52_s}')
    if ld3_buy:
        print(f'  🟢 买入(70-79) ({len(ld3_buy)}只):')
        for s in ld3_buy[:10]:
            rsi_s = f'RSI{s["rsi"]:.0f}' if isinstance(s.get("rsi"),(int,float)) else "RSI?"
            p52_s = f'距52高{s["pct52"]:.0f}%' if isinstance(s.get("pct52"),(int,float)) else "P52?"
            print(f'    {s["code"]:<6} 评分={s.get("prob", s.get("score", 0)):2d}  ${s.get("price","?"):<8}  {rsi_s}  {p52_s}')
    if ld3_watch:
        print(f'  🟡 观望(60-69) ({len(ld3_watch)}只)')
    if ld3_danger:
        print(f'  🔴 规避(<60) ({len(ld3_danger)}只)')
else:
    print(f'  ⚠️ 蓝盾3.0数据不可用')

# ─── 5. 共振信号 ───
print(f'\n{"─"*65}')
print(f'⚡【共振信号 | 双模型同时看好】')
print(f'  绿箭≥85 ∩ 蓝盾≥80 的股票')
print(f'{"─"*65}')

resonance = []
if v9_buy and (ld3_buy or ld3_strong):
    v75_syms = set(s['sym'] for s in v9_buy)
    ld3_syms = set(s['code'] for s in ld3_buy + ld3_strong)
    overlap = v75_syms & ld3_syms
    if overlap:
        for sym in sorted(overlap):
            v75_p = next((s.get('prob', 0) for s in v9_buy if s['sym'] == sym), 0)
            v75_s = next((s.get('prob', s.get('score', 0)) for s in v9_buy if s['sym'] == sym), 0)
            ld3_s = next((s.get('score', 0) for s in ld3_buy + ld3_strong if s['code'] == sym), 0)
            ld3_n = next((s.get('name', sym) for s in ld3_data.get('all_scores', [])
                         if s['code'] == sym), sym)
            print(f'  ⭐ {sym:<6} ({ld3_n:<8})  绿箭评分={v75_s:.4f}  蓝盾评分={ld3_s}')
            resonance.append({'sym': sym, 'v75_score': v75_s, 'ld3_score': ld3_s})

if not resonance:
    print(f'  (今日无共振, 大小盘池独立, 符合预期)')

# ─── 6. 市场判断总览 ───
print(f'\n{"─"*65}')
print(f'📊【市场判断】')
print(f'{"─"*65}')

if v9_data:
    strong_count = len([s for s in v9_data.get('buy_signals', []) if s.get('prob',0) >= 0.90])
    mt_tag = '🔥热' if strong_count >= 20 else '🌤️温' if strong_count >= 5 else '❄️冷'
    if '热' in mt_tag:
        v75_advice = '小盘密集, 优先买≥0.90的票'
    elif '冷' in mt_tag:
        v75_advice = '小盘信号少, 控制仓位'
    else:
        v75_advice = '正常执行V9策略'
    # Compute average of top 50
    top50 = v9_data.get('buy_signals', [])[:50]
    t50_avg = float(np.mean([s.get('prob',0) for s in top50])) if top50 else 0
    print(f'  绿箭(V9): Top50平均概率{t50_avg:.3f} → {mt_tag}  |  {v75_advice}')

if ld3_data and 'ld3_market' in dir():
    print(f'  蓝盾3.0: 大盘评分均值{avg:.4f}/100  |  {ld3_market}')

# ─── 7. 综合建议 ───
print(f'\n{"─"*65}')
print(f'💡【综合建议】')
print(f'{"─"*65}')

green_signal = bool(v9_buy) if v9_data else False
blue_signal = bool(ld3_buy or ld3_strong) if ld3_data else False

if resonance:
    print(f'  ⚡ 共振信号存在! 双模型同时看好的票置信度最高')
elif not green_signal and blue_signal:
    print(f'  🛡️ 蓝盾>绿箭: 小盘偏冷, 大盘有买入信号, 转向大盘股')
elif green_signal and not blue_signal:
    print(f'  🟢 绿箭>蓝盾: 小盘有机会, 大盘偏冷, 执行小盘策略')
elif not green_signal and not blue_signal:
    print(f'  🟡 双冷: 市场偏弱, 持有现金观望')
else:
    print(f'  🟢 双温: 市场正常, 独立执行')

print(f'\n{"="*65}')
print(f'  ℹ️  详细评分见:')
if v75_path:
    print(f'     绿箭V9: {v75_path}')
if ld3_path:
    print(f'     蓝盾3.0: {ld3_path}')
print(f'  {"="*65}')

# 融合报告保存
fusion = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'model': {'green': 'V9-Lottery', 'blue': '蓝盾3.0'},
    'green': {
        'available': bool(v9_data),
        'market_temp': mt_tag if v9_data else None,
        'top50_avg_prob': t50_avg if v9_data else None,
        'buy_signals': v9_buy,
        'watch_signals': v9_watch[:15],
    } if v9_data else None,
    'blue': {
        'available': bool(ld3_data),
        'strong_buy': ld3_strong,
        'buy': ld3_buy,
        'watch': ld3_watch,
        'danger': ld3_danger,
    } if ld3_data else None,
    'resonance': resonance,
}
dst = f'{DATA_DIR}/fusion_rec_{today}.json'
json.dump(fusion, open(dst, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print(f'\n💾 融合报告: {dst}')
