"""
绿箭V7.5 深度分析 - 止损后大涨的规律挖掘
"""
import sys, json, os, time
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

print('加载数据...', flush=True)

# 加载特征数据
df = pd.read_parquet('/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v75.parquet')
may = df[df['date'].astype(str).str[:7] == '2026-05'].copy()
may_dates = sorted(may['date'].astype(str).str[:10].unique())

# 加载价格
with open('/home/hermes/.hermes/openclaw-project/data/us_hist_clean.parquet', 'r', encoding='utf-8', errors='replace') as f:
    hist = json.load(f)

# 加载缓存（评分数据）
with open('_green_arrow_cache.json', 'r') as f:
    cache = json.load(f)

all_day_scores = cache['scores']
price_db_raw = cache['prices']

# 重新构建价格DB（含完整日期）
print('构建完整价格数据库...', flush=True)
df_sym_dates = df[['sym','date']].drop_duplicates().copy()
df_sym_dates.loc[:, 'datestr'] = df_sym_dates['date'].astype(str).str[:10]
all_sym_dates = df_sym_dates.groupby('sym')['datestr'].apply(lambda x: sorted(x.values)).to_dict()

price_db_full = {}
syms_in_price = set(may['sym'].unique())
for sym in syms_in_price:
    if sym not in hist:
        continue
    h = hist[sym]
    c, hi, lo = h.get('c',[]), h.get('h',[]), h.get('l',[])
    if not (c and hi and lo):
        continue
    fdates = all_sym_dates.get(sym, [])
    if not fdates:
        continue
    n_f = len(fdates)
    n_h = len(c)
    if n_f > n_h:
        continue
    offset = n_h - n_f
    pmap = {}
    for j, d in enumerate(fdates):
        idx = offset + j
        if idx < n_h:
            pmap[d] = {'close':float(c[idx]), 'high':float(hi[idx]), 'low':float(lo[idx]), 'open':float(hi[idx])}
    if pmap:
        price_db_full[sym] = pmap

print(f'价格DB: {len(price_db_full)}只股票', flush=True)

# 特征列
report_path = '/home/hermes/.hermes/openclaw-project/data/models/us_v7_5_report.json'
if os.path.exists(report_path):
    with open(report_path, 'r') as f:
        report = json.load(f)
    FEATS = report['features']
    
print(f'特征列: {len(FEATS)}', flush=True)

# ======== 1. 找到"止损后大涨"的买入 ========
print('\n=== 分析1: 止损后大涨的买入特征 ===', flush=True)

BUY_SCORE = 85
MAX_PER_DAY = 5
HOLD_DAYS = 4
STOP_LOSS = -0.10
TRIGGER_UP = 0.30
WIN_SMALL = 0.30
WIN_BIG = 1.00

# 重跑模拟，收集每笔买入的完整数据
class Collector:
    def __init__(self):
        self.all_positions = {}  # key -> pos
        self.day_trades = {}  # date -> [(sym, action, price, reason)]
    
    def run(self):
        holdings = {}
        for date in may_dates:
            records = all_day_scores.get(date, [])
            qualified = [r for r in records if r['score'] >= BUY_SCORE]
            candidates = [r for r in qualified]
            candidates = candidates[:MAX_PER_DAY]
            
            for r in candidates:
                sym, score = r['sym'], r['score']
                key = f'{sym}_{date}'
                if key in holdings:
                    continue
                dp = price_db_full.get(sym, {}).get(date)
                if dp and dp['close'] > 0:
                    holdings[key] = {
                        'sym': sym, 'buy_date': date, 'buy_price': dp['close'],
                        'score': score,
                        'daily_features': [],  # 每天的特征
                        'daily_prices': [],    # 每天的价格
                        'daily_scores': [],    # 当天的模型评分（次新）
                    }
                    self.day_trades.setdefault(date, []).append((sym, 'buy', dp['close'], score))
            
            # 更新&退出
            exits = []
            for key, pos in list(holdings.items()):
                if pos.get('exit_date'): continue
                dp = price_db_full.get(pos['sym'], {}).get(date)
                if not dp: continue
                
                # 记录每日状态
                pos['daily_prices'].append({'date': date, **dp})
                
                # 检查条件
                buy_p = pos['buy_price']
                if dp['high'] >= buy_p * (1 + TRIGGER_UP):
                    pos['exit_date'] = date
                    pos['exit_price'] = dp['high']
                    pos['exit_reason'] = 'daily_pop'
                    pos['ret'] = (dp['high'] - buy_p) / buy_p
                    exits.append(key)
                elif dp['low'] <= buy_p * (1 + STOP_LOSS):
                    pos['exit_date'] = date
                    pos['exit_price'] = dp['close']
                    pos['exit_reason'] = 'stop_loss'
                    pos['ret'] = (dp['close'] - buy_p) / buy_p
                    exits.append(key)
                else:
                    hold_idx = may_dates.index(pos['buy_date']) + HOLD_DAYS
                    hold_until = may_dates[hold_idx] if hold_idx < len(may_dates) else may_dates[-1]
                    if date >= hold_until:
                        pos['exit_date'] = date
                        pos['exit_price'] = dp['close']
                        pos['exit_reason'] = 'hold_expiry'
                        pos['ret'] = (dp['close'] - buy_p) / buy_p
                        exits.append(key)
            
            for key in exits:
                self.all_positions[key] = holdings.pop(key)
            
            # 记录每天的交易
            for key in exits:
                pos = self.all_positions[key]
                self.day_trades.setdefault(date, []).append((pos['sym'], pos['exit_reason'], pos['exit_price'], 0))
        
        # 强制平仓
        last_date = may_dates[-1]
        for key in list(holdings.keys()):
            pos = holdings[key]
            dp = price_db_full.get(pos['sym'], {}).get(last_date)
            if dp:
                pos['exit_date'] = last_date
                pos['exit_price'] = dp['close']
                pos['exit_reason'] = 'sim_end'
                pos['ret'] = (dp['close'] - pos['buy_price']) / pos['buy_price']
                self.all_positions[key] = holdings.pop(key)
        
        return self.all_positions

collector = Collector()
all_positions = collector.run()

# 找到止损后大涨的：exit_reason='stop_loss' 且后续峰值>=30%
print('计算止损后峰值...', flush=True)

def get_peak_after(sym, stop_date, buy_price, days=10):
    """止损后N天内最高涨幅"""
    stop_idx = may_dates.index(stop_date) if stop_date in may_dates else -1
    if stop_idx < 0:
        return 0
    end_idx = min(stop_idx + days, len(may_dates) - 1)
    pmap = price_db_full.get(sym, {})
    max_ret = 0
    for d in may_dates[stop_idx:end_idx+1]:
        dp = pmap.get(d)
        if dp and dp['high'] > buy_price:
            ret = (dp['high'] - buy_price) / buy_price
            if ret > max_ret:
                max_ret = ret
    return max_ret

# 收集止损案例
stop_loss_cases = []
for key, pos in all_positions.items():
    if pos.get('exit_reason') == 'stop_loss':
        peak_10d = get_peak_after(pos['sym'], pos['exit_date'], pos['buy_price'], days=10)
        peak_full = get_peak_after(pos['sym'], pos['exit_date'], pos['buy_price'], days=30)
        stop_loss_cases.append({
            'sym': pos['sym'], 'buy_date': pos['buy_date'], 'sell_date': pos['exit_date'],
            'buy_price': pos['buy_price'], 'sell_price': pos['exit_price'],
            'ret': pos['ret'], 'peak_10d': peak_10d, 'peak_full': peak_full,
            'score': pos['score'], 'key': key
        })

# 分类
bad_stop = [c for c in stop_loss_cases if c['peak_10d'] >= 0.30]  # 止损后10天内涨超30%
good_stop = [c for c in stop_loss_cases if c['peak_10d'] < 0.30]  # 止损是对的

print(f'\n止损总数: {len(stop_loss_cases)}')
print(f'止损后10天内涨超30%: {len(bad_stop)} (止损错了)')
print(f'止损后10天也没涨: {len(good_stop)} (止损对了)')
print(f'止损后全时段涨超30%: {len([c for c in stop_loss_cases if c["peak_full"] >= 0.30])}')

print('\n止损错了的（TOP 15，按买入日排序）:')
for c in sorted(bad_stop, key=lambda x: x['buy_date'])[:15]:
    print(f'  {c["sym"]:<8s} 买入{c["buy_date"]} 止损{c["sell_date"]} 成本${c["buy_price"]:.2f} '
          f'止损{c["ret"]*100:.1f}% → 10天峰值+{c["peak_10d"]*100:.1f}% 全时段+{c["peak_full"]*100:.1f}%')

# ======== 2. 止损后大涨的票在买入当天的特征 ========
print('\n\n=== 分析2: "止损后大涨"组的买入日特征 vs "止损对了"组 ===', flush=True)

# 提取买入当天的特征
def get_buy_day_features(pos, may_df, FEATS):
    """提取买入当天的特征值"""
    sym = pos['sym']
    buy_date = pos['buy_date']
    row = may_df[(may_df['sym']==sym) & (may_df['date'].astype(str).str[:10]==buy_date)]
    if len(row) == 0:
        return {}
    row = row.iloc[0]
    feats = {}
    for f in FEATS:
        if f in row.index:
            val = row[f]
            if pd.notna(val) and np.isfinite(val):
                feats[f] = float(val)
    return feats

# 止损错了的组
bad_features = []
for c in bad_stop:
    pos = all_positions.get(c['key'])
    if pos:
        feats = get_buy_day_features(pos, may, FEATS)
        bad_features.append(feats)

# 止损对了的组
good_features = []
for c in good_stop:
    pos = all_positions.get(c['key'])
    if pos:
        feats = get_buy_day_features(pos, may, FEATS)
        good_features.append(feats)

print(f'止损错了组（后续涨）: {len(bad_features)}笔')
print(f'止损对了组（没涨）: {len(good_features)}笔')

if len(bad_features) > 3 and len(good_features) > 3:
    bad_df = pd.DataFrame(bad_features)
    good_df = pd.DataFrame(good_features)
    
    # 对比两组均值差异最大的特征
    diff = {}
    for f in FEATS:
        if f in bad_df.columns and f in good_df.columns:
            b_mean = bad_df[f].dropna().mean()
            g_mean = good_df[f].dropna().mean()
            if pd.notna(b_mean) and pd.notna(g_mean):
                diff[f] = abs(b_mean - g_mean)
    
    top_diff = sorted(diff.items(), key=lambda x: -x[1])[:20]
    
    print('\n两组差异最大的TOP20特征:')
    for feat, d in top_diff:
        bm = bad_df[feat].dropna().mean()
        gm = good_df[feat].dropna().mean()
        print(f'  {feat:<25s}  止损错了(均值):{bm:.4f}  止损对了(均值):{gm:.4f}  差异:{d:.4f}')

# ======== 3. 量能分析 ========
print('\n\n=== 分析3: 起飞前是否量能增大 ===', flush=True)

# 对每只止损后大涨的票，看止损日前后几天的成交量变化
vol_analysis = []
for c in bad_stop:
    sym = c['sym']
    stop_date = c['sell_date']
    stop_idx = may_dates.index(stop_date)
    
    # 前3天、当天、后3天的价格和特征
    days_range = []
    for offset in range(-3, 4):
        idx = stop_idx + offset
        if 0 <= idx < len(may_dates):
            d = may_dates[idx]
            dp = price_db_full.get(sym, {}).get(d)
            row = may[(may['sym']==sym) & (may['date'].astype(str).str[:10]==d)]
            
            entry = {'date': d, 'offset': offset}
            if dp:
                entry['close'] = dp['close']
                entry['high'] = dp['high']
                entry['low'] = dp['low']
            if len(row) > 0:
                r = row.iloc[0]
                for col in ['vol5','vol20','vol_ratio','vol_ratio_ma5','vol_ratio_ma20',
                           'ma5','ma20','ma60','ma5_ratio','ma20_ratio','ma60_ratio',
                           'macd','rsi14','k','d','j','bb_position','bb_width',
                           'adx','plus_di','minus_di','price_position','cmf']:
                    if col in r.index and pd.notna(r[col]):
                        entry[col] = float(r[col])
            days_range.append(entry)
    
    if len(days_range) >= 5:
        vol_analysis.append({'sym': sym, 'buy_date': c['buy_date'], 'stop_date': stop_date, 'days': days_range})

# 分析成交量变化模式
print(f'止损后大涨案例数: {len(vol_analysis)}')

# 检查vol_ratio和vol_ratio_ma5的变化趋势
if vol_analysis:
    vol_trends = {'pre_increase': 0, 'no_change': 0, 'post_increase': 0}
    for va in vol_analysis:
        days = va['days']
        pre = [d.get('vol_ratio_ma5', 0) for d in days if d['offset'] <= 0 and 'vol_ratio_ma5' in d]
        post = [d.get('vol_ratio_ma5', 0) for d in days if d['offset'] > 0 and 'vol_ratio_ma5' in d]
        if pre and post:
            pre_avg = np.mean(pre)
            post_avg = np.mean(post)
            if post_avg > pre_avg * 1.3:
                vol_trends['post_increase'] += 1
            elif pre_avg > post_avg * 1.3:
                vol_trends['pre_increase'] += 1
            else:
                vol_trends['no_change'] += 1
    
    print(f'  成交量(vol_ratio_ma5)趋势:')
    print(f'    起飞前量能增大: {vol_trends["pre_increase"]}')
    print(f'    起飞后量能增大: {vol_trends["post_increase"]}')
    print(f'    无明显变化: {vol_trends["no_change"]}')

# 打印几个典型案例的详细量能数据
print('\n典型案例量能变化:')
for va in vol_analysis[:5]:
    print(f'\n  {va["sym"]} 止损{va["stop_date"]}:')
    for d in va['days']:
        vr5 = d.get('vol_ratio_ma5', 0)
        vr = d.get('vol_ratio', 0)
        close = d.get('close', 0)
        rsi = d.get('rsi14', 0)
        macd = d.get('macd', 0)
        print(f'    {d["offset"]:+d} {d["date"]} close=${close:.2f} vol_ratio={vr:.2f} vol/ma5={vr5:.2f} rsi14={rsi:.1f} macd={macd:.4f}')

# ======== 4. 评分分位数分析：好票的评分分布 ========
print('\n\n=== 分析4: "被选中且涨了" vs "被选中但跌了" 的评分分布 ===', flush=True)

win_positions = [p for p in all_positions.values() if p.get('ret', 0) >= WIN_SMALL]
lose_positions = [p for p in all_positions.values() if p.get('ret', 0) < 0]

print(f'盈利(>=30%): {len(win_positions)}笔')
print(f'亏损: {len(lose_positions)}笔')

if win_positions:
    win_scores = [p['score'] for p in win_positions]
    print(f'  盈利组评分: min={min(win_scores):.0f} max={max(win_scores):.0f} mean={np.mean(win_scores):.1f}')

if lose_positions:
    lose_scores = [p['score'] for p in lose_positions]
    print(f'  亏损组评分: min={min(lose_scores):.0f} max={max(lose_scores):.0f} mean={np.mean(lose_scores):.1f}')

# ======== 5. 被遗漏的起飞票特征 ========
print('\n\n=== 分析5: "被前5挤掉"的漏网鱼特征 ===', flush=True)

# 用price_db_full里的数据判断
all_bought_keys = {f'{p["sym"]}_{p["buy_date"]}' for p in all_positions.values()}

missed_peaks = []  # (sym, date, score, peak_ret)
for date in may_dates:
    records = all_day_scores.get(date, [])
    for r in records:
        sym, score = r['sym'], r['score']
        key = f'{sym}_{date}'
        if key in all_bought_keys:
            continue
        if score < 85:
            continue
        dp = price_db_full.get(sym, {}).get(date)
        if not dp or dp['close'] <= 0:
            continue
        buy_price = dp['close']
        
        # 买入日后30天内最高涨幅
        buy_idx = may_dates.index(date)
        pmap = price_db_full.get(sym, {})
        peak_ret = 0
        for d in may_dates[buy_idx:]:
            dp2 = pmap.get(d)
            if dp2 and dp2['high'] > buy_price:
                ret = (dp2['high'] - buy_price) / buy_price
                if ret > peak_ret:
                    peak_ret = ret
        
        if peak_ret >= WIN_SMALL:
            missed_peaks.append((sym, date, score, peak_ret))

# 这些遗漏的票，当天的特征里有没有什么共性？
missed_syms = set(m[0] for m in missed_peaks)
print(f'漏网鱼总数: {len(missed_peaks)}只次')
print(f'独立股票数: {len(missed_syms)}只')

if missed_peaks:
    # 按峰值排序，打印前10
    sorted_missed = sorted(missed_peaks, key=lambda x: -x[3])
    print('\nTOP10漏网大鱼（按峰值）:')
    for sym, date, score, peak in sorted_missed[:10]:
        print(f'  {sym:<8s} {date} score={score:.0f} 峰值+{peak*100:.1f}%')

# ======== 6. 单一票连续多日被选中的情况 ========
print('\n\n=== 分析6: 同一只票被连续买入的情况 ===', flush=True)

sym_buys = {}  # sym -> [(date, score, ret)]
for p in all_positions.values():
    s = p['sym']
    sym_buys.setdefault(s, []).append((p['buy_date'], p['score'], p.get('ret',0)))

# 找连续买入>=3次的
for s, buys in sorted(sym_buys.items()):
    if len(buys) >= 3:
        dates = [b[0] for b in buys]
        scores = [b[1] for b in buys]
        rets = [b[2]*100 for b in buys]
        print(f'  {s:<8s} {len(buys)}次: {", ".join(dates)}')
        print(f'          scores: {scores}')
        print(f'          rets(%): {[round(r,1) for r in rets]}')
