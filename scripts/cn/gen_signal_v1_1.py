#!/usr/bin/env python3
"""
cn-alpha-v1.1 生产信号生成
每日盘后运行，输出：市场状态 + Top30信号 + 仓位建议 + 信号灯分级
持有期: 10天 (fwd_10d)
"""
import pandas as pd, numpy as np, xgboost as xgb, json, time, os, sys, warnings
import tushare as ts
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

HOLD_DAYS = 10  # 模型训练目标: fwd_10d

# ============================================================
# 1. 拉取最新数据
# ============================================================
ts.set_token('ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db')
pro = ts.pro_api()

today = time.strftime('%Y%m%d')
print(f"cn-alpha-v1.1 信号生成 {today}\n")

# 最新日线+资金流+基本面
try:
    daily = pro.daily(trade_date=today)
    mf = pro.moneyflow(trade_date=today)
    basic = pro.daily_basic(trade_date=today, 
        fields='ts_code,trade_date,pe_ttm,pb,ps_ttm,dv_ratio,total_mv,circ_mv,turnover_rate')
except Exception as e:
    # 如果今天不是交易日，取上一个交易日
    cal = pro.trade_cal(exchange='SSE', is_open='1', 
        start_date=(pd.Timestamp.now() - pd.Timedelta(days=10)).strftime('%Y%m%d'),
        end_date=today)
    trade_dates = sorted(cal[cal['is_open']==1]['cal_date'].tolist())
    if trade_dates:
        today = trade_dates[-1]
        print(f"  今天非交易日，使用最近交易日: {today}")
        daily = pro.daily(trade_date=today)
        mf = pro.moneyflow(trade_date=today)
        basic = pro.daily_basic(trade_date=today,
            fields='ts_code,trade_date,pe_ttm,pb,ps_ttm,dv_ratio,total_mv,circ_mv,turnover_rate')
    else:
        print("无法获取交易日"); sys.exit(1)

print(f"  日线: {len(daily)}只, 资金流: {len(mf)}只, 基本面: {len(basic)}只")

# 合并
daily['sym'] = daily['ts_code'].str[:6]
mf['sym'] = mf['ts_code'].str[:6]
basic['sym'] = basic['ts_code'].str[:6]

df = daily[['sym','open','high','low','close','vol','amount']].copy()
df = df.merge(mf[['sym','buy_sm_amount','sell_sm_amount','buy_md_amount','sell_md_amount',
    'buy_lg_amount','sell_lg_amount','buy_elg_amount','sell_elg_amount','net_mf_amount']], on='sym', how='left')
df = df.merge(basic[['sym','pe_ttm','pb','ps_ttm','dv_ratio','circ_mv','turnover_rate']], on='sym', how='left')

# ============================================================
# 2. 计算特征（需要历史数据）
# ============================================================
print("  加载历史数据计算特征...")

# 加载历史特征数据
hist = pd.read_parquet('data/cn/features_v2.parquet')
hist['date'] = pd.to_datetime(hist['date'])
hist['date_int'] = hist['date'].dt.strftime('%Y%m%d').astype(int)

# 最近60天数据用于计算滚动特征
recent_dates = sorted(hist['date_int'].unique())[-60:]
recent = hist[hist['date_int'].isin(recent_dates)].copy()

# 合并今日数据到历史
today_int = int(today)
df['date_int'] = today_int
df['date'] = pd.to_datetime(today)

# 计算今日特征
# 反转
recent_by_sym = recent.groupby('sym')
for sym in df['sym']:
    sym_hist = recent[recent['sym'] == sym].sort_values('date_int')
    if len(sym_hist) < 5:
        continue
    
    row = df[df['sym'] == sym]
    if len(row) == 0:
        continue
    
    close = row.iloc[0]['close']
    
    # 5/10/20日收益
    if len(sym_hist) >= 5:
        r5 = (close - sym_hist.iloc[-5]['close']) / sym_hist.iloc[-5]['close']
    else: r5 = 0
    if len(sym_hist) >= 10:
        r10 = (close - sym_hist.iloc[-10]['close']) / sym_hist.iloc[-10]['close']
    else: r10 = 0
    if len(sym_hist) >= 20:
        r20 = (close - sym_hist.iloc[-20]['close']) / sym_hist.iloc[-20]['close']
    else: r20 = 0
    
    idx = df[df['sym'] == sym].index[0]
    df.loc[idx, 'r5'] = r5
    df.loc[idx, 'r10'] = r10
    df.loc[idx, 'r20'] = r20

# 用历史特征的最后一天作为基础，更新今日变化
last_day = hist[hist['date_int'] == recent_dates[-1]].set_index('sym')

# 对于今日有数据的股票，计算增量特征
for sym in df['sym']:
    if sym not in last_day.index:
        continue
    
    idx = df[df['sym'] == sym].index
    if len(idx) == 0:
        continue
    idx = idx[0]
    
    # 资金流
    row = df.loc[idx]
    for col in ['buy_sm_amount','sell_sm_amount','buy_md_amount','sell_md_amount',
                'buy_lg_amount','sell_lg_amount','buy_elg_amount','sell_elg_amount']:
        if pd.isna(row.get(col)):
            df.loc[idx, col] = 0
    
    df.loc[idx, 'sm_net'] = (row.get('buy_sm_amount',0) or 0) - (row.get('sell_sm_amount',0) or 0)
    df.loc[idx, 'md_net'] = (row.get('buy_md_amount',0) or 0) - (row.get('sell_md_amount',0) or 0)
    df.loc[idx, 'lg_net'] = (row.get('buy_lg_amount',0) or 0) - (row.get('sell_lg_amount',0) or 0)
    df.loc[idx, 'elg_net'] = (row.get('buy_elg_amount',0) or 0) - (row.get('sell_elg_amount',0) or 0)
    df.loc[idx, 'total_net'] = row.get('net_mf_amount', 0) or 0
    
    # 滚动特征用历史近似
    hist_sym = recent[recent['sym'] == sym].sort_values('date_int')
    if len(hist_sym) >= 5:
        df.loc[idx, 'sm_net_5'] = hist_sym['sm_net'].tail(5).sum() + df.loc[idx, 'sm_net']
        df.loc[idx, 'md_net_5'] = hist_sym['md_net'].tail(5).sum() + df.loc[idx, 'md_net']
        df.loc[idx, 'lg_net_5'] = hist_sym['lg_net'].tail(5).sum() + df.loc[idx, 'lg_net']
        df.loc[idx, 'elg_net_5'] = hist_sym['elg_net'].tail(5).sum() + df.loc[idx, 'elg_net']
        df.loc[idx, 'total_net_5'] = hist_sym['total_net'].tail(5).sum() + df.loc[idx, 'total_net']
    if len(hist_sym) >= 20:
        df.loc[idx, 'sm_net_20'] = hist_sym['sm_net'].tail(20).sum() + df.loc[idx, 'sm_net']
        df.loc[idx, 'md_net_20'] = hist_sym['md_net'].tail(20).sum() + df.loc[idx, 'md_net']
        df.loc[idx, 'lg_net_20'] = hist_sym['lg_net'].tail(20).sum() + df.loc[idx, 'lg_net']
        df.loc[idx, 'elg_net_20'] = hist_sym['elg_net'].tail(20).sum() + df.loc[idx, 'elg_net']
        df.loc[idx, 'total_net_20'] = hist_sym['total_net'].tail(20).sum() + df.loc[idx, 'total_net']
    
    # 技术指标
    if len(hist_sym) >= 20:
        closes = list(hist_sym['close'].tail(19)) + [close]
        df.loc[idx, 'vol5'] = np.std(closes[-5:]) / np.mean(closes[-5:]) if len(closes) >= 5 else 0
        df.loc[idx, 'vol20'] = np.std(closes) / np.mean(closes)
        df.loc[idx, 'atr_pct'] = df.loc[idx, 'vol20']  # 近似
    
    # 市值
    if not pd.isna(row.get('circ_mv')):
        df.loc[idx, 'log_circ_mv'] = np.log(row['circ_mv']) if row['circ_mv'] > 0 else 0
        df.loc[idx, 'circ_mv'] = row['circ_mv']

# ============================================================
# 3. 构建特征向量
# ============================================================
print("  构建特征...")

# 反转
df['rev_5d'] = -df.get('r5', 0)
df['rev_10d'] = -df.get('r10', 0)
df['rev_20d'] = -df.get('r20', 0)
df['rsi_reversal'] = -50  # 近似，无RSI历史
df['macd_reversal'] = 0   # 近似
df['macd_hist'] = 0
df['low_vol_5d'] = -df.get('vol5', 0)
df['low_vol_20d'] = -df.get('vol20', 0)
df['low_atr'] = -df.get('atr_pct', 0)
df['small_cap'] = -df.get('log_circ_mv', 0)
df['residual_mom_5d'] = 0  # 需要截面数据
df['residual_mom_20d'] = 0
df['lg_flow_momentum'] = df.get('lg_net_5', 0) - df.get('lg_net_20', 0) / 4
df['total_flow_momentum'] = df.get('total_net_5', 0) - df.get('total_net_20', 0) / 4

# 排名特征
for col in ['lg_net_20', 'md_net_20', 'total_net_20']:
    if col in df.columns:
        df[f'{col}_rank'] = df[col].rank(pct=True)
    else:
        df[f'{col}_rank'] = 0.5

df['rev_flow_interaction'] = df['rev_20d'] * df['lg_net_20_rank']
df['vol_r'] = df.get('turnover_rate', 0) / 5 if 'turnover_rate' in df.columns else 0.5
df['turnover_rank'] = df['vol_r'].rank(pct=True) if df['vol_r'].notna().any() else 0.5

# 基本面
pe = df.get('pe_ttm', pd.Series(dtype=float))
df['pe_clean'] = pe.where((pe > 0) & (pe < 500), np.nan)
df['pe_rank'] = df['pe_clean'].rank(pct=True, ascending=True)
df['pe_inverse'] = 1.0 / df['pe_clean'].clip(lower=1)
pb = df.get('pb', pd.Series(dtype=float))
df['pb_clean'] = pb.where((pb > 0) & (pb < 100), np.nan)
df['pb_rank'] = df['pb_clean'].rank(pct=True, ascending=True)
df['pb_inverse'] = 1.0 / df['pb_clean'].clip(lower=0.1)
df['div_rank'] = df.get('dv_ratio', pd.Series(dtype=float)).rank(pct=True, ascending=False)
ps = df.get('ps_ttm', pd.Series(dtype=float))
df['ps_clean'] = ps.where((ps > 0) & (ps < 200), np.nan)
df['ps_rank'] = df['ps_clean'].rank(pct=True, ascending=True)

# 残差动量（用截面均值近似）
r5_mean = df['rev_5d'].mean() if 'rev_5d' in df.columns else 0
df['residual_mom_5d'] = -df['rev_5d'] - (-r5_mean)
r20_mean = df['rev_20d'].mean() if 'rev_20d' in df.columns else 0
df['residual_mom_20d'] = -df['rev_20d'] - (-r20_mean)

features = [
    'rev_5d','rev_10d','rev_20d','rsi_reversal','macd_reversal','macd_hist',
    'low_vol_5d','low_vol_20d','low_atr',
    'md_net_5','md_net_20','lg_net_5','lg_net_20','total_net_5','total_net_20',
    'small_cap','residual_mom_5d','residual_mom_20d',
    'lg_flow_momentum','total_flow_momentum',
    'lg_net_20_rank','md_net_20_rank','total_net_20_rank',
    'rev_flow_interaction','turnover_rank',
    'pe_rank','pe_inverse','pb_rank','pb_inverse','div_rank','ps_rank',
    'vol_r','sm_net_5','sm_net_20','elg_net_5','elg_net_20',
]

# 填充缺失特征
for f in features:
    if f not in df.columns:
        df[f] = 0
    df[f] = df[f].fillna(0)

# ============================================================
# 4. 市场状态
# ============================================================
market_daily = hist.groupby('date_int')['r1'].mean()
market_ma60 = market_daily.rolling(60).mean().iloc[-1]
market_ma120 = market_daily.rolling(120).mean().iloc[-1]
market_ret20 = market_daily.rolling(20).sum().iloc[-1]

adv_dec_today = (df.get('r5', pd.Series([0]*len(df))) > 0).sum() / max((df.get('r5', pd.Series([0]*len(df))) < 0).sum(), 1)

ma_bull = market_ma60 > market_ma120
mom_pos = market_ret20 > 0
breadth = adv_dec_today > 0.4

if not ma_bull and not mom_pos:
    regime = 'bear'
elif not ma_bull or not mom_pos:
    regime = 'cautious'
elif not breadth:
    regime = 'weak'
else:
    regime = 'bull'

position_map = {'bull': 1.0, 'cautious': 0.5, 'weak': 0.5, 'bear': 0}
position_pct = position_map[regime]

# ============================================================
# 5. 模型预测 + 信号
# ============================================================
model = xgb.Booster()
model.load_model('models/cn/cn_alpha_v1.1.json')

X = df[features]
df['score'] = model.predict(xgb.DMatrix(X))

# 过滤
df_signal = df[df['close'] > 3].copy()
df_signal = df_signal[df_signal['close'] < 1000]

# Top30
top30 = df_signal.nlargest(30, 'score')[['sym','close','score','pe_ttm','pb','dv_ratio','circ_mv']].copy()
top30['rank'] = range(1, 31)
top30['weight'] = position_pct / 30

# ============================================================
# 6. 输出
# ============================================================
print(f"\n{'='*60}")
print(f"📊 cn-alpha-v1.1 信号 {today}")
print(f"{'='*60}")

print(f"\n🌐 市场状态: {regime.upper()}")
print(f"  MA60: {market_ma60:.4f} {'>' if ma_bull else '<'} MA120: {market_ma120:.4f}")
print(f"  20日动量: {market_ret20:.4f} ({'正' if mom_pos else '负'})")
print(f"  涨跌比: {adv_dec_today:.2f} ({'OK' if breadth else '弱'})")
print(f"  ➜ 仓位: {position_pct*100:.0f}%")

if position_pct > 0:
    # 信号灯分级（百分位）
    scores = df_signal['score']
    p95 = scores.quantile(0.95)
    p90 = scores.quantile(0.90)
    p80 = scores.quantile(0.80)
    
    def signal_label(score):
        if score >= p95: return '🟢🟢'
        elif score >= p90: return '🟢'
        elif score >= p80: return '🟡'
        else: return '⚪'
    
    top30['signal'] = top30['score'].apply(signal_label)
    
    print(f"\n📈 Top30 信号 (持有{HOLD_DAYS}天):")
    print(f"  百分位门槛: P95={p95:.4f} P90={p90:.4f} P80={p80:.4f}")
    print(f"{'排名':>4} {'信号':>4} {'代码':>8} {'价格':>8} {'评分':>8} {'PE':>8} {'PB':>6} {'股息%':>6}")
    print("-" * 60)
    for _, r in top30.iterrows():
        pe_str = f"{r['pe_ttm']:.1f}" if not pd.isna(r['pe_ttm']) else "N/A"
        pb_str = f"{r['pb']:.1f}" if not pd.isna(r['pb']) else "N/A"
        div_str = f"{r['dv_ratio']:.2f}" if not pd.isna(r['dv_ratio']) else "0"
        print(f"{int(r['rank']):>4} {r['signal']:>4} {r['sym']:>8} ¥{r['close']:>7.2f} {r['score']:>8.4f} {pe_str:>8} {pb_str:>6} {div_str:>6}")
    
    # 统计
    g2 = (top30['signal'] == '🟢🟢').sum()
    g1 = (top30['signal'] == '🟢').sum()
    y = (top30['signal'] == '🟡').sum()
    print(f"\n  🟢🟢精品: {g2}只  🟢强信号: {g1}只  🟡观察: {y}只")
else:
    print(f"\n⚠️ 市场状态BEAR，建议空仓观望")

# 保存
signal_output = {
    'date': today,
    'regime': regime,
    'position_pct': position_pct,
    'market': {
        'ma60': round(market_ma60, 6),
        'ma120': round(market_ma120, 6),
        'ret20': round(market_ret20, 6),
        'adv_dec': round(adv_dec_today, 3)
    },
    'top30': top30[['sym','close','score']].to_dict('records') if position_pct > 0 else []
}

os.makedirs('signals/cn', exist_ok=True)
with open(f'signals/cn/cn_alpha_v1.1_{today}.json', 'w') as f:
    json.dump(signal_output, f, indent=2)
with open('signals/cn/latest.json', 'w') as f:
    json.dump(signal_output, f, indent=2)

print(f"\n✅ 信号已保存 signals/cn/cn_alpha_v1.1_{today}.json")
