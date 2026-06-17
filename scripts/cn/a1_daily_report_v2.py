#!/usr/bin/env python3
"""
A1_2_每日推荐.py — A股每日推荐 v2（替代原有A1 1.0资金流模型）
================================================================
模型: Layer 3 XGBoost (38特征, 10日滚动预测) + 今日资金流增强
候选: 全市场 A股 (4581只, 价格>1+剔除ST)
策略: 评分经超跌衰减+特大单信号催化剂; 5-10日持有; -15%止损

输出: 
  📊 市场热度层(沪深300距20日线+Top50正评分率)
  🏆 精选Top榜 (分数打散, ⚡特大单标记)
  🟢 强势买入 (score>8, ma60>-10%)
  🟡 抄底关注 (score>8, ma60<-10%但rsi<20)
  🔴 规避列表
  🛡️ 仓位建议

用法: python scripts/a1_daily_report_v2.py
"""

import json, os, sys, time, math, warnings
import numpy as np
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─── 配置 ───
D_DATA = r'/home/hermes/.hermes/openclaw-archive/data'
WORKSPACE = r'/home/hermes/.hermes/openclaw-archive'
CHECKPOINT_MODEL = os.path.join(D_DATA, 'layer3_checkpoints', 'model_batch_5.json')
TUSHARE_TOKEN = 'e563b75a4d0fd88f4e08a2084313b22fba0b98c7b6580557bcc6fc8a'

print("=" * 65)
print("A1 v2 每日推荐 — Layer 3 XGBoost + 今日资金流")
print("=" * 65)
t0_all = time.time()

# ─── 1. 加载模型 ───
import xgboost as xgb
m = xgb.Booster()
m.load_model(CHECKPOINT_MODEL)
FEAT_COLS = m.feature_names
TECH_FEATS = ['pct_ma5','pct_ma10','pct_ma20','pct_ma60','ma20_slope','ma60_slope','ma_align',
    'vol_10d','vol_60d','vol_ratio','atr20_pct','ret_5d','ret_10d','ret_20d','ret_60d',
    'rsi14','vol_ratio_5_20','ret20d_pct']
MF_FEATS = [c for c in FEAT_COLS if c not in TECH_FEATS]
print(f"✅ 模型: {len(FEAT_COLS)}特征 ({len(TECH_FEATS)}技术 + {len(MF_FEATS)}资金流)")

# ─── 2. 加载数据 ───
print(f"[{time.strftime('%H:%M:%S')}] 加载历史数据...")
t = time.time()
with open(os.path.join(D_DATA, 'a_hist_10y.parquet'), 'rb') as f: hist = json.load(f)
with open(os.path.join(D_DATA, 'moneyflow_data.parquet'), 'rb') as f: mf_big = json.load(f)
print(f"  K线: {len(hist)}只 | 资金流: {len(mf_big)}只 ({time.time()-t:.0f}s)")

# ─── 3. 今日资金流（tushare） ───
today_str = datetime.now().strftime("%Y%m%d")
yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

print(f"[{time.strftime('%H:%M:%S')}] 今日资金流...")
mf_lookup = {}
HAS_TODAY = False
mf_date_used = ""
try:
    import tushare as ts
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    for attempt_date in [today_str, yesterday_str]:
        mf_df = pro.moneyflow(trade_date=attempt_date)
        if mf_df is not None and len(mf_df) > 0:
            mf_date_used = attempt_date
            HAS_TODAY = True
            break
    if HAS_TODAY:
        for _, r in mf_df.iterrows():
            c6 = str(r['ts_code'])[:6]
            net_mf = (r['net_mf_amount'] or 0) / 1e4  # 万元→亿
            buy_elg = (r['buy_elg_amount'] or 0) / 1e4
            sell_elg = (r['sell_elg_amount'] or 0) / 1e4
            buy_lg = (r['buy_lg_amount'] or 0) / 1e4
            sell_lg = (r['sell_lg_amount'] or 0) / 1e4
            # 综合信号: 特大单净买入为主轴
            elg_net = buy_elg - sell_elg
            lg_net = buy_lg - sell_lg
            signal = round(elg_net + 0.5 * lg_net + 0.3 * net_mf, 2)
            mf_lookup[c6] = {
                'signal': signal,
                'elg_net_yi': round(elg_net, 2),
                'lg_net_yi': round(lg_net, 2),
                'net_mf_yi': round(net_mf, 2),
            }
        print(f"  ✅ {mf_date_used}: {len(mf_lookup)}只有资金数据")
    else:
        print(f"  ⚠️ tushare两日均无数据")
except Exception as e:
    print(f"  ⚠️ tushare出错: {e}")

# ─── 4. 指数状态（沪深300） ───
print(f"[{time.strftime('%H:%M:%S')}] 市场状态...")
try:
    idf = pro.index_daily(ts_code='000300.SH',
        start_date=(datetime.now()-timedelta(days=35)).strftime("%Y%m%d"), end_date=today_str)
    if idf is not None and len(idf) > 0:
        idf = idf.sort_values('trade_date')
        idx_close = float(idf.iloc[-1]['close'])
        idx_pct = float(idf.iloc[-1]['pct_chg'])
        idx_ma20 = idf['close'].tail(20).mean()
        idx_ma20_pct = (idx_close / idx_ma20 - 1) * 100
        # 60日线判断中期趋势
        idx_ma60 = idf['close'].tail(60).mean() if len(idf) >= 60 else idx_ma20
        idx_ma60_pct = (idx_close / idx_ma60 - 1) * 100 if idx_ma60 > 0 else 0
    else:
        idx_close = 0; idx_pct = 0; idx_ma20_pct = 0; idx_ma60_pct = 0
except Exception as e:
    idx_close = 0; idx_pct = 0; idx_ma20_pct = 0; idx_ma60_pct = 0
    print(f"  ⚠️ 指数数据错误: {e}")

print(f"  沪深300: {idx_close:.0f} ({idx_pct:+.2f}%)")
print(f"  距20日线: {idx_ma20_pct:+.1f}% | 距60日线: {idx_ma60_pct:+.1f}%")

# ─── 5. Layer 3 全量评分 ───
print(f"[{time.strftime('%H:%M:%S')}] Layer 3 评分...")
codes = sorted(hist.keys())
results = []
sk_no_mf = sk_short = 0

for idx, code in enumerate(codes):
    hrec = hist[code]
    c = hrec['c']
    if len(c) < 120: sk_short += 1; continue
    h = hrec['h']; l = hrec['l']; v = hrec['v']; da = hrec['dates']

    mfc = code + ('.SZ' if code[:1] in ('0', '3') else '.SH')
    mr = mf_big.get(mfc, [])
    if not mr: sk_no_mf += 1; continue

    # ── 资金流rollup ──
    nm = len(mr)
    na = np.array([(x.get('net_mf_amount', 0) or 0) for x in mr], dtype=np.float32)
    bl = np.array([(x.get('buy_lg_amount', 0) or 0) for x in mr], dtype=np.float32)
    sl = np.array([(x.get('sell_lg_amount', 0) or 0) for x in mr], dtype=np.float32)
    be = np.array([(x.get('buy_elg_amount', 0) or 0) for x in mr], dtype=np.float32)
    se = np.array([(x.get('sell_elg_amount', 0) or 0) for x in mr], dtype=np.float32)
    ble = bl + be; sle = sl + se

    cs_net = np.zeros(nm + 1, dtype=np.float64)
    cs_maj = np.zeros(nm + 1, dtype=np.float64)
    cs_lg = np.zeros(nm + 1, dtype=np.float64)
    for i in range(nm):
        cs_net[i + 1] = cs_net[i] + na[i]
        cs_maj[i + 1] = cs_maj[i] + (ble[i] - sle[i])
        cs_lg[i + 1] = cs_lg[i] + (bl[i] - sl[i])

    ru = {}
    for i in range(1, nm):
        d = mr[i]['trade_date']
        t = ble[i] + sle[i]
        ru[d] = {
            'net_mf_1d': na[i], 'lg_net_1d': bl[i] - sl[i],
            'elg_net_1d': be[i] - se[i], 'major_net_1d': ble[i] - sle[i],
            'lg_pct': (ble[i] / t * 100) if t > 0 else 50.0,
        }
        for lb in [5, 10, 20, 60]:
            s = max(0, i - lb + 1)
            ru[d][f'net_mf_{lb}d'] = cs_net[i + 1] - cs_net[s]
            ru[d][f'major_net_{lb}d'] = cs_maj[i + 1] - cs_maj[s]
            ru[d][f'lg_net_{lb}d'] = cs_lg[i + 1] - cs_lg[s]

    # ── 技术指标 ──
    i = len(c) - 1
    price = c[i]
    if price <= 0: continue

    ma5 = sum(c[i - 4:i + 1]) / 5
    ma10 = sum(c[i - 9:i + 1]) / 10
    ma20 = sum(c[i - 19:i + 1]) / 20
    ma60 = sum(c[i - 59:i + 1]) / 60

    f = {}
    f['pct_ma5'] = (price / ma5 - 1) * 100 if ma5 > 0 else 0
    f['pct_ma10'] = (price / ma10 - 1) * 100 if ma10 > 0 else 0
    f['pct_ma20'] = (price / ma20 - 1) * 100 if ma20 > 0 else 0
    f['pct_ma60'] = (price / ma60 - 1) * 100 if ma60 > 0 else 0

    f['ma20_slope'] = 0
    if i >= 25:
        mb = sum(c[i - 25:i - 4]) / 20
        f['ma20_slope'] = (ma20 / mb - 1) * 100 if mb > 0 else 0
    f['ma60_slope'] = 0
    if i >= 65:
        mb = sum(c[i - 65:i - 4]) / 60
        f['ma60_slope'] = (ma60 / mb - 1) * 100 if mb > 0 else 0

    f['ma_align'] = (ma5 > ma10) + (ma10 > ma20) + (ma20 > ma60) + (price > ma5) + (price > ma10) + (price > ma60)

    r10 = [abs(c[j] / c[j - 1] - 1) * 100 if c[j - 1] > 0 else 0 for j in range(i - 9, i + 1)]
    f['vol_10d'] = sum(r10) / 10
    r60 = [abs(c[j] / c[j - 1] - 1) * 100 if c[j - 1] > 0 else 0 for j in range(i - 59, i + 1)]
    f['vol_60d'] = sum(r60) / 60
    f['vol_ratio'] = f['vol_10d'] / f['vol_60d'] if f['vol_60d'] > 0 else 1.0

    trs = [max(h[j] - l[j], abs(h[j] - c[j - 1]), abs(l[j] - c[j - 1])) for j in range(i - 19, i + 1)]
    f['atr20_pct'] = sum(trs) / 20 / price * 100 if price > 0 else 0

    ret5d = (price / c[i - 5] - 1) * 100 if i >= 5 else 0
    ret10d = (price / c[i - 10] - 1) * 100 if i >= 10 else 0
    ret20d = (price / c[i - 20] - 1) * 100 if i >= 20 else 0
    f['ret_5d'] = ret5d
    f['ret_10d'] = ret10d
    f['ret_20d'] = ret20d
    f['ret_60d'] = (price / c[i - 60] - 1) * 100 if i >= 60 else 0

    # RSI(14)
    chgs = [c[j] - c[j - 1] for j in range(i - 13, i + 1)]
    rg = sum(x for x in chgs if x > 0)
    rl = sum(-x for x in chgs if x < 0)
    f['rsi14'] = 100 - 100 / (1 + rg / rl / 14) if rl > 0 else 100.0

    v5 = sum(v[i - 4:i + 1]) / 5
    v20 = sum(v[i - 19:i + 1]) / 20
    f['vol_ratio_5_20'] = v5 / v20 if v20 > 0 else 1.0

    f['ret20d_pct'] = 50.0
    if i >= 40:
        p20 = [c[j] / c[j - 20] - 1 for j in range(20, i + 1) if c[j - 20] > 0]
        cr = price / c[i - 20] - 1 if c[i - 20] > 0 else 0
        f['ret20d_pct'] = sum(1 for r in p20 if r < cr) / len(p20) * 100 if p20 else 50.0

    # 资金流特征
    ru_row = ru.get(da[i], {})
    for k in MF_FEATS:
        f[k] = ru_row.get(k, 0.0)

    feats = [f[k] for k in FEAT_COLS]
    if any(np.isnan(x) or np.isinf(x) for x in feats):
        continue

    dm = xgb.DMatrix(np.array([feats], dtype=np.float32), feature_names=FEAT_COLS)
    l3_raw = float(m.predict(dm)[0])

    # ── 今日资金流叠加信号（催化剂，不修改原始评分） ──
    c6 = code[:6]
    mfi = mf_lookup.get(c6, None)
    today_elg_net = mfi['elg_net_yi'] if mfi else 0.0
    today_signal = mfi['signal'] if mfi else 0.0

    # ── 评分衰减（分数打散核心） ──
    decay = 1.0
    death_crosses = (ma5 <= ma10) + (ma10 <= ma20) + (ma20 <= ma60)  # 死叉个数
    # 每个死叉打折: 0死叉=0%, 1个=-10%, 2个=-25%, 3个=-50%
    if death_crosses >= 3: decay *= 0.50
    elif death_crosses == 2: decay *= 0.75
    elif death_crosses == 1: decay *= 0.90
    
    # 超跌额外折扣: 深度超跌的票即使评分高也只给"抄底关注"
    if ret10d < -30: decay *= 0.65
    if ret20d < -40: decay *= 0.55
    if f['pct_ma60'] < -50: decay *= 0.70

    final_score = l3_raw * decay

    # ── 今日资金流修正（只有特大单净买入>0.5亿的才加分） ──
    if today_elg_net > 0.5:
        final_score += min(today_elg_net * 0.03, 2.0)  # 最多加2分
    elif today_elg_net < -0.5:
        final_score += max(today_elg_net * 0.02, -1.0)  # 最多减1分

    results.append({
        'code': code,
        'name': hrec.get('name', ''),
        'close': round(price, 2),
        'score': round(final_score, 1),
        'raw': round(l3_raw, 1),
        'today_elg': round(today_elg_net, 2),
        'today_signal': round(today_signal, 2),
        'pct_ma5': round(f['pct_ma5'], 1),
        'pct_ma20': round(f['pct_ma20'], 1),
        'pct_ma60': round(f['pct_ma60'], 1),
        'rsi14': round(f['rsi14'], 1),
        'ma_align': f['ma_align'],
        'death_crosses': death_crosses,
        'ret_5d': round(ret5d, 1),
        'ret_10d': round(ret10d, 1),
        'ret_20d': round(ret20d, 1),
        'vol_ratio_5_20': round(f['vol_ratio_5_20'], 2),
        'net_mf_5d': round(f.get('net_mf_5d', 0) / 1e4, 2),
    })

    if (idx + 1) % 1000 == 0:
        print(f"  {idx + 1}/{len(codes)} ({len(results)} scored)", flush=True)

print(f"  完成: {len(results)}/{len(codes)}只 (跳过:无MF={sk_no_mf} 短={sk_short})")

# ─── 6. 过滤 + 排序 ───
all_scored = [r for r in results
              if r['close'] >= 2.0  # 2元以上
              and 'ST' not in r.get('name', '').upper()
              and '退' not in r.get('name', '')]
all_scored.sort(key=lambda x: -x['score'])

# ─── 7. 市场热度分析 ───
# 指数层
if abs(idx_ma20_pct) < 2:
    idx_heat = "🌤️ 中性"
elif idx_ma20_pct > 3:
    idx_heat = "🔥 过热"
elif idx_ma20_pct < -3:
    idx_heat = "❄️ 过冷"
else:
    idx_heat = "🌤️ 微偏" + ("热" if idx_ma20_pct > 0 else "冷")

# 模型层
top50_all = [r['score'] for r in all_scored[:50]]
top50_pos = sum(1 for s in top50_all if s > 0)
l3_heat = "☀️ 偏热" if top50_pos > 35 else ("🌤️ 中性" if top50_pos > 20 else "🌧️ 偏冷")

# 综合判断
if idx_ma20_pct > 3 and top50_pos > 30:
    heat_desc = ("🔥 指数过热 + 模型积极 → 建议低仓位(<40%), 仅选特大单流入为正的票, "
                 "警惕冲高回落")
elif idx_ma20_pct < -3 and top50_pos < 15:
    heat_desc = ("❄️ 指数过冷 + 模型谨慎 → 恐慌末期特征, 可小仓试水抄底关注票, "
                 "但不到全面进场时机")
elif idx_ma20_pct < -2 and top50_pos > 25:
    heat_desc = ("📉 指数回调中 + 模型仍有信心 → 底部区域概率大, 可渐进建仓, "
                 "评分>8的票优先入场")
elif idx_ma20_pct > 2 and top50_pos < 20:
    heat_desc = ("📈 指数上涨中但模型偏冷静 → 结构市特征, 跟随模型精选, "
                 "不追热门板块")
elif abs(idx_ma20_pct) < 2 and 20 <= top50_pos <= 35:
    heat_desc = ("🌤️ 指数+模型双中性 → 正常操作, 按评分分级入场, "
                 "强势票优先, 抄底票控仓")
else:
    heat_desc = "🌤️ 中性状态, 按系统信号操作"

# 评分分布
n_total = len(all_scored)
max_s = all_scored[0]['score'] if n_total > 0 else 0
p50_s = all_scored[n_total // 2]['score'] if n_total > 0 else 0
gt10 = sum(1 for r in all_scored if r['score'] > 10)
gt8 = sum(1 for r in all_scored if r['score'] > 8)
gt5 = sum(1 for r in all_scored if r['score'] > 5)
gt2 = sum(1 for r in all_scored if r['score'] > 2)

# ─── 8. 筛选分级 ───
# 🟢 强势买入: score>8, ma60>-10%, rsi<70, 10日不暴跌
strong_buy = [
    r for r in all_scored
    if r['score'] > 8 and r['pct_ma60'] > -10 and r['rsi14'] < 70
    and r['ret_10d'] > -15 and r['ret_10d'] < 25
]

# 🟡 抄底关注: score>8, ma60<=-10%但rsi<20 (深度超跌但模型认可)
catch_up = [
    r for r in all_scored
    if r['score'] > 8 and r['pct_ma60'] <= -10
    and r['rsi14'] < 25 and r['ret_10d'] > -30
]

# 🔴 规避: 评分高但技术面极差(死叉3个+ma60<-40%)
avoid = [
    r for r in all_scored[:50]
    if r['death_crosses'] >= 2 and r['pct_ma60'] < -25
]

# 板块拥挤度检测（简单版: 看Top20中重复行业多的）
top20 = all_scored[:20]

# ═══════════════════════════════════════════
# 输出报告
# ═══════════════════════════════════════════
print(f"\n{'=' * 65}")
print(f"A股每日推荐 v2 — {mf_date_used if HAS_TODAY else '近交易日'}数据")
print(f"{'=' * 65}")

# ── 市场热度 ──
print(f"\n📊 市场热度")
print(f"  {'─' * 40}")
print(f"  沪深300: {idx_close:.0f} ({idx_pct:+.2f}%)")
print(f"  距20日线: {idx_ma20_pct:+.1f}% | 距60日线: {idx_ma60_pct:+.1f}%")
print(f"  指数判定: {idx_heat} | Layer3 Top50: {l3_heat}")
print(f"  Top50正评分率: {top50_pos}/{len(top50_all)} ({top50_pos / len(top50_all) * 100:.0f}%)")
print(f"  📌 {heat_desc}")

# ── Layer 3 精选 Top 20 ──
print(f"\n🏆 Layer 3 精选 Top 20  🏆")
print(f"  评分 = 10日XGBoost预测 + 超跌折扣 + 特大单修正")
print(f"  ⚡ = 今日特大单净买入>1亿 | 💀 = 死叉≥2(注意风险)")
print(f"  {'─' * 62}")
print(f"  {'#':>3} {'代码':>8} {'评分':>8} {'特大单':>8} {'ma60':>7} {'rsi':>5} {'10日':>7} {'死叉':>4}")
print(f"  {'─' * 62}")
for i, r in enumerate(top20):
    tag = ""
    if r['today_elg'] > 1: tag = "⚡"
    if r['death_crosses'] >= 2: tag += "💀"
    if r['pct_ma60'] > -10: tag = "✅ " + tag if tag else "✅"
    if tag == "": tag = "   "
    name_short = r.get('name', '')[:6]
    print(f"  {tag}{i + 1:>2}. {r['code']:>8} {r['score']:>+7.1f} "
          f"{r['today_elg']:>+7.1f} {r['pct_ma60']:>+6.1f}% {r['rsi14']:>5.1f} "
          f"{r['ret_10d']:>+6.1f}% {r['death_crosses']:>4}")

# ── 强势买入 ──
print(f"\n🟢 强势买入 (score>8, ma60>-10%, rsi<70, ret10d>-15%/+25%)")
print(f"  {'─' * 62}")
if strong_buy:
    for i, r in enumerate(strong_buy[:10]):
        name_short = r.get('name', '')[:8]
        elg_tag = "⚡" if r['today_elg'] > 1 else " "
        print(f"  {i + 1:>2}. {r['code']:>8} ({name_short:8s}) "
              f"评分={r['score']:>+5.1f} 特大单={r['today_elg']:>+5.1f}亿{elg_tag} "
              f"ma60={r['pct_ma60']:+.1f}% rsi={r['rsi14']:5.1f} "
              f"10日={r['ret_10d']:+.1f}% 死叉={r['death_crosses']}")
else:
    print(f"  ❌ 今日无符合条件的强势买入股")

# ── 抄底关注 ──
print(f"\n🟡 抄底关注 (score>8, ma60<=-10%, rsi<25, ret10d>-30%)")
print(f"  {'─' * 62}")
if catch_up:
    for i, r in enumerate(catch_up[:10]):
        name_short = r.get('name', '')[:8]
        print(f"  {i + 1:>2}. {r['code']:>8} ({name_short:8s}) "
              f"评分={r['score']:>+5.1f} 特大单={r['today_elg']:>+5.1f}亿 "
              f"ma60={r['pct_ma60']:+.1f}% rsi={r['rsi14']:5.1f} "
              f"10日={r['ret_10d']:+.1f}%")
    print(f"  ⚠️ 抄底候选为深度超跌, 建议小仓位(5-15%)左侧布局, "
          f"等待ma60拐头确认")
else:
    print(f"  ❌ 今日无符合条件抄底股")

# ⚠️ 规避
print(f"\n🔴 Top50中技术面极差(死叉≥2 + ma60<-25%)")
print(f"  {'─' * 62}")
if avoid:
    for i, r in enumerate(avoid[:5]):
        print(f"  {i + 1:>2}. {r['code']:>8} ({r.get('name', '')[:8]:8s}) "
              f"评分={r['score']:>+5.1f} ma60={r['pct_ma60']:+.1f}% "
              f"死叉={r['death_crosses']}个 ⚠️ 不宜追入")
else:
    print(f"  ✅ Top50中无极端技术面风险股")

# ── 评分分布 ──
print(f"\n📊 全市场评分分布 (共{n_total}只, 价格>2, 剔除ST)")
print(f"  {'─' * 40}")
print(f"  最高: {max_s:+.1f} | 中位: {p50_s:+.1f}")
print(f"  >10分: {gt10}只 >8分: {gt8}只 >5分: {gt5}只 >2分: {gt2}只")

# ── 仓位建议 ──
print(f"\n🛡️ 仓位建议")
print(f"  {'─' * 40}")
if idx_ma20_pct > 3:
    print(f"  建议仓位: 30-40% (市场过热)")
elif idx_ma20_pct < -3:
    print(f"  建议仓位: 40-60% (市场过冷, 逢低入场)")
else:
    print(f"  建议仓位: 50-70% (中性)")
print(f"  持仓上限: 10-20只 | 单只上限: 50% | 止损: -15% | 持有: 5-10日")
print(f"  退出: 评分从>8跌破5→退出 | 评分<2→强退 | 连续2日特大单隔夜负→减仓")

# ── 主观归纳 ──
print(f"\n💡 主观总结")
print(f"  {'─' * 40}")
subjective_notes = []
if len(strong_buy) == 0 and len(catch_up) == 0:
    subjective_notes.append("当前无买入信号, 建议观望或减仓")
elif len(strong_buy) >= 3:
    subjective_notes.append(f"强势买入{len(strong_buy)}只, 市场存在明确机会")
else:
    subjective_notes.append(f"机会有限({len(strong_buy)}强买+{len(catch_up)}抄底), 精选为主")
if top50_pos < 20:
    subjective_notes.append("Top50评分偏冷, 警惕系统性风险")
elif top50_pos > 35:
    subjective_notes.append("市场温度偏高, 注意止盈节奏")

for note in subjective_notes:
    print(f"  · {note}")

# ═══════════════════════════════════════════
# 存档
# ═══════════════════════════════════════════
print(f"\n{'=' * 65}")
print(f"⏱️ 耗时: {time.time() - t0_all:.0f}s")

# 保存评分数据
out_path = os.path.join(D_DATA, f'a1_layer3_scored_{today_str}.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(all_scored[:50], f, indent=2, ensure_ascii=False)
print(f"📁 评分已保存: {out_path}")

# 写入decision_history
record = {
    'type': 'a1_daily_v2',
    'time': datetime.now().isoformat(),
    'trade_date': mf_date_used if HAS_TODAY else 'N/A',
    'market': {
        'hs300_close': idx_close,
        'hs300_pct': idx_pct,
        'hs300_ma20_dev': idx_ma20_pct,
        'idx_heat': idx_heat,
        'l3_top50_pos': f'{top50_pos}/{len(top50_all)}',
        'heat_desc': heat_desc,
    },
    'strong_buy': [{'code': r['code'], 'name': r.get('name',''), 'score': r['score'],
                     'today_elg': r['today_elg']} for r in strong_buy[:10]],
    'catch_up': [{'code': r['code'], 'name': r.get('name',''), 'score': r['score']}
                  for r in catch_up[:10]],
    'top20_codes': [r['code'] for r in top20],
    'total_scored': n_total,
    'sender_id': '7908145929',
    'sender_name': 'Andi Yang',
}
try:
    dec_path = os.path.join(WORKSPACE, 'data', 'decision_history.jsonl')
    with open(dec_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"[存档完成 ✅]")
except Exception as e:
    print(f"⚠️ 存档失败: {e}")
