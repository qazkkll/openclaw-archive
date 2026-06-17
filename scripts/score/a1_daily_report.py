#!/usr/bin/env python3
"""
A1 资金流模型 1.0 — 每日推荐
================================
数据源策略（A1_MODEL_MANUAL.md 第四章）:
  16:00-22:00 → tushare 今日完整资金流（精确）
  09:30-16:00 → tushare pro 实时资金流
  其他时段   → 最近交易日的tushare数据

价格门控（第五章）:
  1. 盘中实时跌幅 > 3%? → ❌ 屏蔽
  2. 3日累计涨幅 > 10%? → ❌ 屏蔽
  3. ST/*ST?  → ❌ 屏蔽
  4. 停牌?    → ❌ 屏蔽

用法:
  python a1_daily_report.py
"""
import json, os, sys, time, traceback
from datetime import datetime, timezone, timedelta
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TZ = timezone(timedelta(hours=8))
WORKSPACE = "/home/hermes/.hermes/openclaw-archive"
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db")

now = datetime.now(TZ)
now_str = now.strftime("%H:%M")
today = now.strftime("%Y%m%d")
today_str = now.strftime("%Y-%m-%d")

# ─── 数据源选择 ─────────────────────────────────────────────
hour = now.hour + now.minute / 60
data_source = ""
trade_date = ""

# 统一使用 tushare pro 数据源
if 16 <= hour <= 22:
    data_source = "tushare_today"
    trade_date = today
else:
    data_source = "tushare_latest"

print(f"=== A1 资金流模型 1.0 每日推荐 ===")
print(f"时间: {today_str} {now_str}")
print(f"数据源: {data_source}")

# ─── 加载股票信息 ────────────────────────────────────────────
info = {}
try:
    with open(os.path.join(WORKSPACE, "data", "stock_info.json"), 'r', encoding='utf-8') as f:
        info = json.load(f)
    print(f"股票信息: {len(info)}只")
except Exception as e:
    print(f"⚠️ stock_info.json加载失败: {e}")

# ─── 加载历史K线（用于价格门控）────────────────────────────
hist_10y = {}
try:
    with open(os.path.join(WORKSPACE, "data", "a_hist_10y.parquet"), 'r', encoding='utf-8') as f:
        hist_10y = json.load(f)
    print(f"历史K线: {len(hist_10y)}只")
except Exception as e:
    print(f"⚠️ a_hist_10y.parquet加载失败: {e}")

# ─── 辅助函数 ────────────────────────────────────────────────

def get_code(code_raw):
    """从ts_code或code里提取干净的6位代码"""
    c = code_raw.replace('.SH','').replace('.SZ','').replace('.BJ','')
    return c

def is_main_board(code):
    """只处理主板60/00开头"""
    return code.startswith('60') or code.startswith('00')

def price_gate(code, price_today, chg_today_percent):
    """价格门控检查，返回(通过, 原因)"""
    # 1. 盘中跌幅 > 3%
    if chg_today_percent is not None and chg_today_percent < -3:
        return False, f"今日跌幅{chg_today_percent:.2f}% > 3%"
    
    # 2. 3日涨幅 > 10% (从历史K线算)
    if code in hist_10y:
        d = hist_10y[code]
        dates = d.get('dates', d.get('date', []))
        closes = d.get('c', d.get('close', []))
        
        # 找最近的非今日收盘
        last_close = None
        third_close = None
        count = 0
        for i in range(len(dates)-1, -1, -1):
            if dates[i] != today:
                if last_close is None:
                    last_close = closes[i]
                    count += 1
                elif count < 3:
                    third_close = closes[i]
                    count += 1
                else:
                    break
        
        if last_close and third_close and third_close > 0:
            rise_3d = (last_close - third_close) / third_close * 100
            if rise_3d > 10:
                return False, f"3日涨幅{rise_3d:.2f}% > 10%"
    
    return True, "通过"

def get_name_industry(code_clean):
    """从info里取名称和行业"""
    si = info.get(code_clean, {})
    return si.get('name', '?'), si.get('industry', '?')

# ─── 方案A: tushare收盘数据 ───────────────────────────────
def run_tushare(td):
    """用tushare moneyflow拉某交易日的数据"""
    import tushare as ts
    pro = ts.pro_api(TUSHARE_TOKEN)
    
    df = pro.moneyflow(trade_date=td)
    if df is None or len(df) == 0:
        print(f"⚠️ tushare {td} 无数据")
        return []
    
    print(f"tushare原始数据: {len(df)}条")
    
    rows = []
    for _, row in df.iterrows():
        code = row['ts_code']
        clean = get_code(code)
        
        # 只主板
        if not is_main_board(clean):
            continue
        
        # 过滤ST
        name, industry = get_name_industry(clean)
        if 'ST' in name or '*' in name:
            continue
        
        net_mf = row['net_mf_amount']
        buy_lg = row['buy_lg_amount'] + row['buy_elg_amount']
        sell_lg = row['sell_lg_amount'] + row['sell_elg_amount']
        total_vol = (row['buy_sm_amount'] + row['sell_sm_amount'] +
                     row['buy_md_amount'] + row['sell_md_amount'] +
                     row['buy_lg_amount'] + row['sell_lg_amount'] +
                     row['buy_elg_amount'] + row['sell_elg_amount'])
        big_net_ratio = (buy_lg - sell_lg) / total_vol if total_vol > 0 else 0
        score = net_mf / 10000 * 0.4 + max(big_net_ratio, 0) * 0.6
        
        rows.append({
            'code': clean,
            'name': name,
            'industry': industry,
            'score': round(score, 4),
            'net_mf': net_mf,
            'big_net_ratio': round(big_net_ratio * 100, 2),
            'price': row.get('amount', 0) / row.get('vol', 1) * 100 if row.get('vol', 0) > 0 else 0,
            'chg_today': None,  # tushare moneyflow无涨跌幅字段
        })
    
    rows.sort(key=lambda x: -x['score'])
    return rows

# ─── [废弃] 东方财富push2 — 2026-06-16起全部用tushare pro ──────────
# 旧函数run_eastmoney_push2已废弃，保留代码存档
#         br = s.get('f69', 0) or 0  # 超大单占比%
#         mid = s.get('f70', 0) or 0  # 大单净额（元），换算为万元
#         mid_wan = mid / 10000
#         
        # 计算大单占比（超大单+大单）/主力总额
#         total_big = bf + mid
#         big_net_ratio = total_big / abs(mf) if mf != 0 else 0
#         
        # A1评分（mf_wan是万元，取对数归一化避免亿元级分数爆炸）
#         import math
#         mf_norm = math.log2(mf_wan + 1) * 0.3  # log2(1万)≈13, log2(10亿)≈30
#         score = mf_norm + max(big_net_ratio, 0) * 0.7
#         
        # 存万元版本供输出
#         net_mf = mf_wan
#         
        # 价格门控
#         gate_pass, gate_reason = price_gate(code, price, chg_pct)
#         
#         rows.append({
#             'code': code,
#             'name': name,
#             'industry': next((info.get(code, {}).get('industry', '?') for _ in [1]), '?'),
#             'score': round(score, 4),
#             'net_mf': mf_wan,  # 万元
#             'big_net_ratio': round(big_net_ratio * 100, 2),
#             'price': price,
#             'chg_today': chg_pct,
#             'gate_pass': gate_pass,
#             'gate_reason': gate_reason,
#             'mf_ratio': mr,
#             'super_net': bf,
#             'super_ratio': br,
#         })
#     
    # 补行业
#     for r in rows:
#         _, ind = get_name_industry(r['code'])
#         r['industry'] = ind
#     
#     rows.sort(key=lambda x: -x['score'])
#     return rows
# 
# ─── [废弃] akshare — 2026-06-16起全部用tushare pro ──────────
# 旧函数run_akshare_spot已废弃，保留代码存档
# ─── 主流程 ──────────────────────────────────────────────────

all_rows = []

if data_source in ("tushare_today", "tushare_latest"):
    if data_source == "tushare_latest":
        # 找最近有数据的交易日
        import tushare as ts
        pro = ts.pro_api(TUSHARE_TOKEN)
        found_td = None
        for i in range(1, 8):
            d = (now - timedelta(days=i)).strftime("%Y%m%d")
            cal = pro.trade_cal(exchange='SSE', start_date=d, end_date=d)
            if len(cal) > 0 and cal.iloc[0]['is_open'] == 1:
                found_td = d
                break
        if found_td:
            trade_date = found_td
            print(f"回退到最近交易日: {trade_date}")
        else:
            print("❌ 找不到最近交易日")
    all_rows = run_tushare(trade_date)

# 盘中统一用tushare pro，不再用东财push2

# ─── 输出 ────────────────────────────────────────────────────

if not all_rows:
    print("\n❌ 没有选出任何标的")
    sys.exit(1)

# 输出Top 10
top_n = min(10, len(all_rows))
is_spot = False  # 已全面切换至tushare pro

print(f"\n{'='*70}")
print(f" A1 资金流模型 1.0 — 今日推荐")
print(f" 数据: {trade_date or '实时'} ({data_source})")
if is_spot:
    print(f" ⚠️ 盘中实时数据，收盘后请重新确认")
print(f"{'='*70}")

# 表头
print(f"\n{'#':<3} {'代码':<7} {'名称':<10} {'行业':<10} {'评分':<8} {'净流入':<12} {'大单%':<7} {'涨跌':<6} {'门控'}")
print(f"{'-'*70}")

top_5 = all_rows[:5]
top_rest = all_rows[5:top_n]

for i, r in enumerate(top_5 + top_rest):
    chg_str = f"{r.get('chg_today', 'N/A'):+.2f}%" if r.get('chg_today') is not None else "N/A"
    gate_str = "✅" if r.get('gate_pass', True) else f"❌{r.get('gate_reason','')[:15]}"
    mf_str = f"{r['net_mf']/10000:.1f}亿" if abs(r['net_mf']) >= 10000 else f"{r['net_mf']:.0f}万"
    
    # 给Top5每行加上主观评价标记
    flag = ""
    if i < 5:
        if r.get('chg_today') is not None and r['chg_today'] < -3:
            flag = "⚠️今日暴跌"
        elif r['big_net_ratio'] < 0:
            flag = "大单在出"
        elif r['big_net_ratio'] > 3:
            flag = "大单强推"
    
    print(f"{i+1:<3} {r['code']:<7} {r['name']:<10} {r['industry']:<10} {r['score']:<8.4f} {mf_str:<12} {r['big_net_ratio']:<7.2f} {chg_str:<6} {gate_str}")
    if flag:
        print(f"   → {flag}")

# 主观评价
print(f"\n{'='*70}")
print(" 📡 主观评价")
print(f"{'='*70}")

passed = [r for r in top_5 if r.get('gate_pass', True)]
blocked = [r for r in top_5 if not r.get('gate_pass', True)]

if is_spot:
    print(" ⚡ 盘中实时数据，结论仅供参考。收盘后16:00再确认一遍。")
    
if blocked:
    for r in blocked:
        reason = r.get('gate_reason', '门控未通过')
        print(f" ❌ {r['name']}({r['code']}): 门控拦截 — {reason}")
    
if passed:
    best = passed[0]
    if best['chg_today'] is not None and best['chg_today'] < -2:
        print(f" ⚠️ Top1 {best['name']}今日跌{best['chg_today']:.1f}%，即使评分高也不建议追")
        if len(passed) > 1:
            print(f" → 建议观望或选评分更高的防守标的")
    elif best['big_net_ratio'] > 3:
        print(f" ✅ Top1 {best['name']}: 信号最纯（大单净+{best['big_net_ratio']}%）")
    elif best['big_net_ratio'] < 0:
        print(f" ⚠️ Top1 {best['name']}: 净流入高但大单在出，可能是散户接盘")

# ─── 写入决策历史（强制存档） ────────────────────────────────
# 收集主观判断
subjective_notes = []
if is_spot:
    subjective_notes.append("盘中实时数据，收盘后需确认")
if blocked:
    subjective_notes.append(f"门控拦截{len(blocked)}只: {', '.join([r['name'] for r in blocked])}")
if passed:
    best = passed[0]
    if best.get('chg_today') is not None and best['chg_today'] < -2:
        subjective_notes.append(f"Top1 {best['name']}今日跌{best['chg_today']:.1f}%，不建议追")
    elif best['big_net_ratio'] > 3:
        subjective_notes.append(f"Top1 {best['name']}: 信号纯（大单净+{best['big_net_ratio']}%）")
    elif best['big_net_ratio'] < 0:
        subjective_notes.append(f"Top1 {best['name']}: 净流入高但大单在出")
    else:
        subjective_notes.append(f"Top1 {best['name']}: 信号中性")

record = {
    'type': 'daily_recommendation',
    'time': now.isoformat(),
    'model': 'A1 1.0',
    'data_source': data_source,
    'trade_date': trade_date,
    'is_spot': is_spot,
    'top5': [{'code': r['code'], 'name': r['name'], 'score': r['score'], 'net_mf': r['net_mf'], 'gate_pass': r.get('gate_pass', True), 'gate_reason': r.get('gate_reason', ''), 'chg_today': r.get('chg_today'), 'big_net_ratio': r.get('big_net_ratio', 0)} for r in top_5],
    'top10': [{'code': r['code'], 'name': r['name'], 'score': r['score']} for r in top_5 + top_rest],
    'total_candidates': len(all_rows),
    'subjective': '; '.join(subjective_notes) if subjective_notes else '无特殊意见',
    'sender_id': 'Andi Yang',
    'sender_name': 'Andi Yang',
    'source_channel': 'telegram',
}
try:
    dec_path = os.path.join(WORKSPACE, 'data', 'decision_history.jsonl')
    with open(dec_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"\n[存档完成 ✅ 写入 decision_history.jsonl]")
except Exception as e:
    print(f"⚠️ 存档失败: {e}")
