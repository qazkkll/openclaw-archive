#!/usr/bin/env python3
"""
us_v75_signal_gen.py — V7.5 极致版调仓信号
极致参数: T7_H10_S20_R5
读取评分结果 + 当前持仓 → 输出具体买入/卖出/持有指令
"""
import sys, os, json, pickle, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, xgboost as xgb

BASE = '/home/hermes/.hermes/openclaw-archive'; ML = f'{BASE}/ml'; MD = f'{BASE}/data/models'; VER = 'us_v7_5'
print('='*70); print(f'V7.5 极致版调仓信号 {time.strftime("%Y-%m-%d %H:%M")}'); print('='*70)
T0 = time.time()

# ========== 1. 加载模型 & 特征 ==========
model = xgb.Booster(); model.load_model(f'{MD}/{VER}.json')
cal = pickle.load(open(f'{MD}/{VER}_calibrator.pkl', 'rb'))
report = json.load(open(f'{MD}/{VER}_report.json'))
FEATS = report['features']

df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str'] = df['date'].astype(str).str[:10]

feat_cols = [f for f in FEATS if f in df.columns]
for f in feat_cols:
    df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0).clip(-1e6, 1e6)
df = df.replace([np.inf, -np.inf], 0)

latest_date = sorted(df['date_str'].unique())[-1]
latest = df[df['date_str'] == latest_date].copy()

# 基本面过滤
FILTER_PATH = f'{ML}/us_filtered_syms.json'
if os.path.exists(FILTER_PATH):
    flist = json.load(open(FILTER_PATH))
    valid_syms = set(flist['syms'])
    latest = latest[latest['sym'].isin(valid_syms)].copy()

# 收盘价
open_idx, close_idx = pickle.load(open(f'{ML}/us_v75_close_idx_v4.pkl', 'rb'))
latest['close'] = latest['sym'].map(lambda s: close_idx.get(s, {}).get(latest_date, 0)).fillna(0)
latest['close'] = latest['close'].replace(0, np.nan)
latest['close'] = latest['close'].fillna(
    latest['sym'].map(lambda s: open_idx.get(s, {}).get(latest_date, 0))).fillna(0)

# 评分
X = np.nan_to_num(latest[feat_cols].values.astype(np.float32), nan=0)
raw = model.predict(xgb.DMatrix(X, feature_names=feat_cols))
calib = cal.predict_proba(raw.reshape(-1, 1))[:, 1]
latest['prob_5pct'] = calib
latest = latest.sort_values('prob_5pct', ascending=False)

# ========== 2. 读取持仓 ==========
PORTFOLIO_FILE = f'{BASE}/data/portfolio_v75_extreme.json'
if os.path.exists(PORTFOLIO_FILE):
    portfolio = json.load(open(PORTFOLIO_FILE))
    print(f'读取持仓: {len(portfolio)}只')
else:
    portfolio = []
    print('无持仓, 全新启动')

# ========== 3. 持仓检查 ==========
T = 7; H = 10; S_PCT = 20; R = 5
STOP = -S_PCT / 100.0

buy_cands = [r['sym'] for _, r in latest.head(T * 2).iterrows()]
hold_syms = {p['sym'] for p in portfolio if p['action'] == 'hold'}

# 检查每只持仓
sell_signals = []
keep_signals = []
for p in portfolio:
    sym = p['sym']; bp = p['bp']; days_held = p.get('days_held', 1)
    cp = latest.loc[latest['sym'] == sym, 'close'].values
    cp = float(cp[0]) if len(cp) > 0 else p.get('last_price', bp)
    ret = (cp - bp) / bp
    # 找到当前评分
    prob_row = latest[latest['sym'] == sym]
    prob = float(prob_row.iloc[0]['prob_5pct']) if len(prob_row) > 0 else 0

    if ret <= STOP:
        sell_signals.append({'sym': sym, 'reason': f'止损(S>{S_PCT}%)',
            'days_held': days_held, 'ret_pct': round(ret * 100, 1),
            'bp': round(bp, 2), 'cp': round(cp, 2), 'prob': round(prob, 4)})
    elif days_held >= H:
        sell_signals.append({'sym': sym, 'reason': f'到期(H={H}天)',
            'days_held': days_held, 'ret_pct': round(ret * 100, 1),
            'bp': round(bp, 2), 'cp': round(cp, 2), 'prob': round(prob, 4)})
    else:
        # 如果评分掉到极低(<-0.20)，也建议卖出换仓
        if prob < 0.05:
            sell_signals.append({'sym': sym, 'reason': f'评分过低({prob:.1%})',
                'days_held': days_held, 'ret_pct': round(ret * 100, 1),
                'bp': round(bp, 2), 'cp': round(cp, 2), 'prob': round(prob, 4)})
        else:
            keep_signals.append({'sym': sym, 'action': 'hold', 'days_held': days_held,
                                 'ret_pct': round(ret * 100, 1), 'prob': round(prob, 4),
                                 'bp': round(bp, 2), 'cp': round(cp, 2)})

# ========== 4. 买入候选 ==========
sell_syms = {s['sym'] for s in sell_signals}
keep_syms = {s['sym'] for s in keep_signals}
existing_syms = keep_syms | sell_syms

# 需要补充的仓位数
slots_needed = T - len(keep_signals)
slots_needed = max(0, min(slots_needed, T))

new_buys = [s for s in buy_cands if s not in existing_syms]
buy_targets = new_buys[:slots_needed]

buy_signals = []
for sym in buy_targets:
    prob_row = latest[latest['sym'] == sym].iloc[0]
    buy_signals.append({'sym': sym, 'action': 'buy',
        'prob': round(float(prob_row['prob_5pct']), 4),
        'est_price': round(float(prob_row['close']), 2) if prob_row['close'] > 0 else '次日开盘'})

# ========== 5. 市场状态 ==========
top50 = latest.head(50)
avg_prob = top50['prob_5pct'].mean()

if avg_prob < 0.25:
    market_temp = '冷'
    position_limit = 0.3
elif avg_prob < 0.33:
    market_temp = '温'
    position_limit = 0.7
else:
    market_temp = '热'
    position_limit = 0.6

# 现金建议
hold_count = len(keep_signals)
total_pos = hold_count / T
if total_pos > position_limit:
    reduce_note = f'超仓: 当前{hold_count}只/{T}只, 建议减至{max(1, int(T*position_limit))}只'
else:
    reduce_note = f'仓位正常: {hold_count}只/{T}只'

# ========== 6. 输出 ==========
print(f'\n评分日: {latest_date} | 市场: {market_temp} | Top50概率: {avg_prob:.3f}')
print(f'当前持仓: {len(portfolio)}只 | 持有中: {len(keep_signals)}只')
print(f'\n━━━ 策略参数: T={T} H={H} S={S_PCT}% R={R} ━━━')

print(f'\n=== 持有 ({len(keep_signals)}只) ===')
if keep_signals:
    print(f'{"代码":>8s} {"天数":>4s} {"收益":>7s} {"概率":>7s} {"买入价":>8s} {"现价":>8s}')
    print('-' * 45)
    for s in keep_signals:
        print(f'{s["sym"]:>8s} {s["days_held"]:>4d} {s["ret_pct"]:>+6.1f}% {s["prob"]:>6.1%} {s["bp"]:>8.2f} {s["cp"]:>8.2f}')
else:
    print('  无')

print(f'\n=== 卖出 ({len(sell_signals)}只) ===')
if sell_signals:
    print(f'{"代码":>8s} {"原因":<16s} {"天数":>4s} {"收益":>7s} {"买入价":>8s} {"现价":>8s}')
    print('-' * 55)
    for s in sell_signals:
        print(f'{s["sym"]:>8s} {s["reason"]:<16s} {s["days_held"]:>4d} {s["ret_pct"]:>+6.1f}% {s["bp"]:>8.2f} {s["cp"]:>8.2f}')
else:
    print('  无')

print(f'\n=== 买入 ({len(buy_signals)}只 / 最多{slots_needed}只) ===')
if buy_signals:
    print(f'{"代码":>8s} {"概率":>7s} {"参考价":>10s}')
    print('-' * 28)
    for s in buy_signals:
        est = f'{s["est_price"]:.2f}' if isinstance(s['est_price'], float) else s['est_price']
        print(f'{s["sym"]:>8s} {s["prob"]:>6.1%} {est:>10s}')
else:
    print('  无需买入')

print(f'\n━━━ 风控 ━━━')
print(f'  市场{market_temp}, {reduce_note}')
print(f'  单只止损: {(S_PCT)}% | 持有期: {H}天')

# ========== 7. 更新持仓文件 ==========
new_portfolio = []
for s in keep_signals:
    new_portfolio.append({
        'sym': s['sym'], 'action': 'hold', 'days_held': s['days_held'] + 1,
        'bp': s['bp'], 'last_price': s['cp'],
        'prob': s['prob'], 'ret_pct': s['ret_pct']
    })
# 已卖出的不加入
for s in buy_signals:
    est_price = float(s['est_price']) if isinstance(s['est_price'], (int, float)) else 0
    new_portfolio.append({
        'sym': s['sym'], 'action': 'hold', 'days_held': 1,
        'bp': est_price, 'last_price': est_price,
        'prob': s['prob'], 'ret_pct': 0.0
    })

json.dump(new_portfolio, open(PORTFOLIO_FILE, 'w'), indent=2)
print(f'\n持仓已更新: {PORTFOLIO_FILE} ({len(new_portfolio)}只)')
print(f'耗时: {time.time()-T0:.1f}s')
