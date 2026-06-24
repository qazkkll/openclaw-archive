#!/usr/bin/env python3
"""
美股综合评分报告
蓝盾V8(>$10) + 绿箭V12($1-$10) + 持仓分类 + 信号说明
"""
import json, os
from datetime import datetime

ROOT = '/home/hermes/.hermes/openclaw-archive'

def load(p):
    try:
        with open(os.path.join(ROOT, p)) as f: return json.load(f)
    except: return {}

def main():
    futu = load('output/futu_positions.json')
    v6 = load('output/v6_latest.json')
    v11 = load('output/v11_latest.json')
    
    positions = futu.get('positions', [])
    vix = futu.get('vix')
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # 分类持仓
    shield_pos = [p for p in positions if p.get('cost_price', 0) > 10]
    arrow_pos = [p for p in positions if p.get('cost_price', 0) <= 10]
    
    # VIX状态
    if vix:
        if vix > 35: vix_s = '🔴🔴 VIX={:.1f} 恐慌！清仓'.format(vix)
        elif vix > 25: vix_s = '🟠 VIX={:.1f} 警戒，减仓50%'.format(vix)
        elif vix > 20: vix_s = '🟡 VIX={:.1f} 注意，收紧止损'.format(vix)
        else: vix_s = '🟢 VIX={:.1f} 正常，全仓位'.format(vix)
    else:
        vix_s = '⚠️ VIX获取失败'
    
    # ============ 打印报告 ============
    print(f'\n{"="*60}')
    print(f'📊 美股综合评分 | {now}')
    print(f'{"="*60}')
    print(f'{vix_s}')
    print(f'持仓: {len(positions)}只 | 市值: ${futu.get("total_value",0):,.0f} | 盈亏: ${futu.get("total_pnl",0):+,.0f}({futu.get("total_pnl_pct",0):+.1f}%)')
    
    # ── 蓝盾V8持仓 ──
    print(f'\n{"─"*60}')
    print(f'🛡️ 蓝盾V8持仓 ({len(shield_pos)}只, 持有20天, 止损-15%)')
    print(f'{"─"*60}')
    if shield_pos:
        for p in sorted(shield_pos, key=lambda x: x['pnl_pct'], reverse=True):
            held = p.get('days_held') or 0
            exp = p.get('days_to_expiry') or 0
            stop = '⚠️止损!' if p.get('stop_triggered') else ''
            model_score = ''
            # 检查模型是否推荐
            for pick in v6.get('picks', []):
                if pick['ticker'] == p['code']:
                    model_score = f'模型评分{pick["pred_rank"]:.3f} {pick["signal"]}'
            if not model_score:
                model_score = '未在今日推荐中'
            
            pnl_e = '🟢' if p['pnl_pct'] >= 0 else '🔴'
            print(f'  {p["code"]:8} {p["name"][:15]:15} {p["qty"]}股 @${p["cost_price"]:.2f}→${p["current_price"]:.2f} {pnl_e}{p["pnl_pct"]:+.1f}%(${p["pnl_usd"]:+.0f}) 持{held}天/剩{exp}天 {model_score} {stop}')
    else:
        print('  (无蓝盾持仓)')
    
    # ── 绿箭V12持仓 ──
    print(f'\n{"─"*60}')
    print(f'🎯 绿箭V12持仓 ({len(arrow_pos)}只, 持有5天, 止损-10%)')
    print(f'{"─"*60}')
    if arrow_pos:
        for p in sorted(arrow_pos, key=lambda x: x['pnl_pct'], reverse=True):
            held = p.get('days_held') or 0
            exp = p.get('days_to_expiry') or 0
            stop = '⚠️止损!' if p.get('stop_triggered') else ''
            model_score = ''
            for pick in v11.get('picks', []):
                if pick['ticker'] == p['code']:
                    model_score = f'模型评分{pick["pred_rank"]:.3f} {pick["signal"]}'
            if not model_score:
                model_score = '未在今日推荐中'
            
            pnl_e = '🟢' if p['pnl_pct'] >= 0 else '🔴'
            print(f'  {p["code"]:8} {p["name"][:15]:15} {p["qty"]}股 @${p["cost_price"]:.2f}→${p["current_price"]:.2f} {pnl_e}{p["pnl_pct"]:+.1f}%(${p["pnl_usd"]:+.0f}) 持{held}天/剩{exp}天 {model_score} {stop}')
    else:
        print('  (无绿箭持仓)')
    
    # ── 蓝盾V8今日推荐 ──
    print(f'\n{"─"*60}')
    print(f'🛡️ 蓝盾V8今日推荐 (全市场>{len(v6.get("picks",[]))}只, Top-15)')
    print(f'{"─"*60}')
    v6_picks = v6.get('picks', [])
    if v6_picks:
        print(f'  {"排名":<4} {"代码":<8} {"价格":>8} {"排名分":>7} {"信号":<4} {"建议"}')
        for i, p in enumerate(v6_picks):
            sig = p.get('signal', '🔴')
            if sig == '🟢🟢': advice = '🟢🟢马上下单'
            elif sig == '🟢': advice = '🟢主力买入'
            elif sig == '🟡': advice = '🟡观察'
            else: advice = '🔴不推荐'
            # 是否已持有
            held = '✓持有' if any(x['code'] == p['ticker'] for x in shield_pos) else ''
            print(f'  {i+1:<4} {p["ticker"]:<8} ${p["price"]:>7.2f} {p["pred_rank"]:>7.4f} {sig:<4} {advice} {held}')
    else:
        print('  (无推荐)')
    
    # ── 绿箭V12今日推荐 ──
    print(f'\n{"─"*60}')
    print(f'🎯 绿箭V12今日推荐 ($1-$10, Top-5)')
    print(f'{"─"*60}')
    v11_picks = v11.get('picks', [])
    if v11_picks:
        print(f'  {"排名":<4} {"代码":<8} {"价格":>8} {"排名分":>7} {"信号":<4} {"建议"}')
        for i, p in enumerate(v11_picks):
            sig = p.get('signal', '🔴')
            if sig == '🟢🟢': advice = '🟢🟢马上下单'
            elif sig == '🟢': advice = '🟢主力买入'
            elif sig == '🟡': advice = '🟡观察'
            else: advice = '🔴不推荐'
            held = '✓持有' if any(x['code'] == p['ticker'] for x in arrow_pos) else ''
            print(f'  {i+1:<4} {p["ticker"]:<8} ${p["price"]:>7.2f} {p["pred_rank"]:>7.4f} {sig:<4} {advice} {held}')
    else:
        print('  (无推荐)')
    
    # ── 信号说明 ──
    print(f'\n{"─"*60}')
    print(f'🚦 信号说明')
    print(f'{"─"*60}')
    print(f'  🟢🟢 Top5% 精品买入 — 马上下单')
    print(f'  🟢  Top10% 强信号 — 主力买入')
    print(f'  🟡  Top20% 观察 — 放watchlist不买')
    print(f'  🔴  低于中位数/VIX>30 — 不推荐')
    
    print(f'\n  蓝盾: 持有20天 | 止损-15% | 未到期不卖')
    print(f'  绿箭: 持有5天  | 止损-10% | 未到期不卖')
    
    # ── 操作建议 ──
    stops = [p for p in positions if p.get('stop_triggered')]
    if stops:
        print(f'\n{"─"*60}')
        print(f'⚠️ 操作建议')
        print(f'{"─"*60}')
        for s in stops:
            print(f'  🔴 {s["code"]} 触发止损! ${s["current_price"]:.2f} ≤ ${s["stop_loss_price"]:.2f} → 考虑卖出')
    
    print(f'\n{"="*60}\n')

if __name__ == '__main__':
    main()
