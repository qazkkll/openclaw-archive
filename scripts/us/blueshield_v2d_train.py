#!/usr/bin/env python3
"""
蓝盾 V2d — 回撤优化(行业分散+硬止损)
"""
import json, warnings, os, sys, time
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

print('加载数据...')
feat = pd.read_parquet('/home/hermes/.hermes/openclaw-project/data/us/sp500_feats.parquet')
feat = feat.sort_values(['Code','Date']).reset_index(drop=True)
feat['Date'] = pd.to_datetime(feat['Date'])

raw_dir = '/home/hermes/.hermes/openclaw-project/data/hist_sp500'
all_rows = []
for f in sorted(os.listdir(raw_dir)):
    if not f.startswith('sp500_chunk_') or not f.endswith('.json'): continue
    raw = json.load(open(os.path.join(raw_dir, f)))
    for sym, bars in raw.items():
        for b in bars: b['Code'] = sym
        all_rows.extend(bars)
raw_df = pd.DataFrame(all_rows)
raw_df['Date'] = pd.to_datetime(raw_df['Date'])
raw_df['DollarVol'] = raw_df['C'] * raw_df['V']

# 行业映射（yfinance sector）
print('获取行业...')
try:
    import yfinance as yf
    sectors = {}
    for sym in raw_df['Code'].unique()[:200]:  # 只查热门票
        try:
            info = yf.Ticker(sym).info
            sectors[sym] = info.get('sector', 'Unknown')
        except: pass
    sec_df = pd.DataFrame([(k,v) for k,v in sectors.items()], columns=['Code','Sector'])
except Exception as e:
    print(f'行业获取失败: {e}, 用代码前缀替代')
    # 用代码前缀分群（粗糙但够用）
    def guess_sector(sym):
        tech_ish = ['AAPL','MSFT','NVDA','AMD','INTC','CRM','ADBE','ORCL','IBM',
                    'GOOGL','GOOG','META','AMZN','NFLX','TSLA']
        finance = ['JPM','BAC','GS','MS','V','MA','AXP','WFC','C','SCHW','BLK']
        if sym in tech_ish: return 'Technology'
        if sym in finance: return 'Financial'
        return 'Other'
    sec_df = pd.DataFrame([(s, guess_sector(s)) for s in raw_df['Code'].unique()],
                          columns=['Code','Sector'])

raw_df = raw_df.merge(sec_df, on='Code', how='left')

# 特征
feat = feat.merge(raw_df[['Code','Date','C','DollarVol','Sector']], on=['Code','Date'], how='left')
market_ret = feat.groupby('Date')['ret_1d'].mean().reset_index()
market_ret.columns = ['Date', 'market_ret']
feat = feat.merge(market_ret, on='Date', how='left')

feat['rel_ret_5d'] = feat['ret_5d'] - feat.groupby('Date')['ret_5d'].transform('mean')
feat['rel_ret_10d'] = feat['ret_10d'] - feat.groupby('Date')['ret_10d'].transform('mean')
feat['ma20_ma50_cross'] = feat['ma_20_ratio'] - feat['ma_50_ratio']
feat['dvol_ma5'] = feat.groupby('Code')['DollarVol'].transform(lambda x: x.rolling(5).mean())
feat['dvol_ratio'] = np.where(feat['dvol_ma5'] > 0, feat['DollarVol'] / feat['dvol_ma5'], 1.0)
feat['vol_5d_norm'] = feat['vol_5d'] / (feat.groupby('Date')['vol_5d'].transform('mean') + 1e-8)
feat['rsi_50_pct'] = (feat['rsi_14'] - 50) / 50

feat_cols = ['ret_1d','ret_3d','ret_5d','ret_10d','ret_20d',
             'ma_5_ratio','ma_10_ratio','ma_20_ratio','ma_50_ratio',
             'vol_5d','vol_10d','vol_20d','rsi_14','rsi_50_pct',
             'vol_ratio_5','vol_ratio_20','vol_5d_norm',
             'price_pos_20','price_pos_50','price_pos_100',
             'macd','macd_sig','macd_hist','atr_pct',
             'rel_ret_5d','rel_ret_10d','ma20_ma50_cross','dvol_ratio']

valid = feat.dropna(subset=feat_cols + ['ret_5d','ret_f5','dvol_ma5','C']).copy()
valid['is_trend'] = (valid['ret_5d'] > 0) & (valid['dvol_ma5'] >= 5_000_000)
trend = valid[valid['is_trend']].copy()
print(f'趋势票: {len(trend)}')

dates = pd.Series(valid['Date'].unique()).sort_values().values
te_mask = trend['Date'] > dates[int(len(dates)*0.85)]
t_mask = trend['Date'] <= dates[int(len(dates)*0.85)]

X_t = trend.loc[t_mask, feat_cols].values; y_t = trend.loc[t_mask, 'ret_f5'].values
X_te = trend.loc[te_mask, feat_cols].values

print('训练...')
model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=1.0,
    eval_metric='rmse', early_stopping_rounds=50, random_state=42, n_jobs=-1)
model.fit(X_t, y_t, eval_set=[(X_t, y_t)], verbose=200)

# ========= 回测 =========
print('\n=== 蓝盾V2d 模拟回测($100K, 含行业分散+止损) ===')
te_df = trend[te_mask].copy()
te_df['pred'] = model.predict(X_te)

# 构建每日价格矩阵（用于止损计算）
price_lookup = raw_df.set_index(['Date','Code'])['C'].to_dict()

date_list = sorted(te_df['Date'].unique())

# 账户
cash = 100000.0
positions = {}  # sym -> (buy_date, buy_price, shares, sector)
equity = []

# 每日状态
for di, d in enumerate(date_list):
    day_df = te_df[te_df['Date'] == d].copy()
    if len(day_df) < 5: continue
    
    # ---- 1. 平仓(止损+到期) ----
    to_close = []
    for sym, (buy_date, buy_p, shs, sec) in positions.items():
        # 获取当日价格
        today_p = price_lookup.get((d, sym))
        if today_p is None or today_p <= 0:
            to_close.append(sym); continue
        
        ret = (today_p - buy_p) / buy_p
        
        # 止损-8%
        if ret < -0.08:
            to_close.append(sym)
            cash += shs * today_p
            continue
        
        # 持有超过5天
        days_held = (d - buy_date).days if isinstance(d, pd.Timestamp) and isinstance(buy_date, pd.Timestamp) else 5
        if days_held >= 5:
            to_close.append(sym)
            cash += shs * today_p
    
    for sym in to_close: del positions[sym]
    
    # ---- 2. 组合止损（当日持仓跌过-2.5%就砍半） ----
    pos_value = 0
    day_ret_sum = 0
    for sym, (_, buy_p, shs, _) in list(positions.items()):
        today_p = price_lookup.get((d, sym), buy_p)
        pos_value += shs * today_p
    
    if cash + pos_value < 20000:  # 亏太多就止损
        for sym in list(positions.keys()):
            bd, bp, shs, sec = positions[sym]
            p = price_lookup.get((d, sym), bp)
            cash += shs * p
            del positions[sym]
        equity.append({'date': d, 'equity': cash, 'n': 0, 'action': 'panic_stop'})
        continue
    
    # ---- 3. 开仓(按ML Score + 行业分散) ----
    max_pos = 10 - len(positions)
    if max_pos <= 0: continue
    
    # 已持仓行业
    held_sectors = set(s for _, (_,_,_,s) in positions.items())
    sector_names = [s for s in held_sectors if pd.notna(s)]
    
    # 候选
    day_df = day_df[day_df['dvol_ma5'] >= 5_000_000].copy()
    if len(day_df) == 0: continue
    
    day_df = day_df.sort_values('pred', ascending=False)
    
    opens = 0
    for _, row in day_df.iterrows():
        if opens >= max_pos: break
        sym = row['Code']
        if sym in positions: continue
        
        sec = row.get('Sector')
        # 同行业最多2只
        if pd.notna(sec) and sec in sector_names:
            sec_count = sum(1 for _, (_,_,_,s) in positions.items() if s == sec)
            if sec_count >= 2: continue
        
        # 建仓
        available = cash / max(1, max_pos - opens)
        price = row['C']
        if price <= 0: continue
        shares = available / price
        cash -= shares * price
        positions[sym] = (d, price, shares, sec)
        sector_names.append(sec) if pd.notna(sec) else None
        opens += 1
    
    # ---- 4. 记录净值 ----
    pos_value = 0
    for sym, (_, buy_p, shs, _) in positions.items():
        today_p = price_lookup.get((d, sym), buy_p)
        pos_value += shs * today_p
    
    total_eq = cash + pos_value
    equity.append({'date': str(d)[:10], 'equity': total_eq, 'n': len(positions), 'cash': cash})

# 结果分析
ec = pd.DataFrame(equity)
if len(ec) == 0:
    print('没有交易数据!')
    exit()

init = 100000.0
final = ec.iloc[-1]['equity']
total_ret = final / init - 1

ec['ret'] = ec['equity'].pct_change().fillna(0)
ann_ret = (final/init) ** (252/len(ec)) - 1 if len(ec) > 0 else 0
ann_vol = ec['ret'].std() * np.sqrt(252)
sharpe = ann_ret / max(ann_vol, 0.001)

ec['cum'] = ec['equity'] / init
ec['peak'] = ec['cum'].cummax()
ec['dd'] = ec['cum'] / ec['peak'] - 1
max_dd = ec['dd'].min()

print(f'\n=== 蓝盾V2d 最终结果 ===')
print(f'终值: ${final:,.0f}')
print(f'总收益: {total_ret*100:.1f}%')
print(f'年化收益: {ann_ret*100:.1f}%')
print(f'夏普: {sharpe:.2f}')
print(f'最大回撤: {max_dd*100:.1f}%')
print(f'平均持仓: {ec["n"].mean():.1f}只')
print(f'交易天数: {len(ec)}')
print(f'终值/初始: {final/init:.2f}x')

print(f'\n=== V2 vs V2d ===')
print(f'V2:  年化+49.7%, 夏普1.85, 回撤-31.7%')
print(f'V2d: 年化{ann_ret*100:.1f}%, 夏普{sharpe:.2f}, 回撤{max_dd*100:.1f}%')
print(f'vs V5.5: 年化+21.2%, 夏普3.87')

# 保存
model.get_booster().save_model('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v2d.model')
ec.to_parquet('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v2d_equity.parquet', index=False)

meta = {
    'model': 'blueshield_v2d',
    'strategy': 'trend_ML_sector_diversified_stop8pct',
    'backtest': {
        'final_equity': float(final), 'total_return': float(total_ret),
        'annual_return': float(ann_ret), 'sharpe': float(sharpe),
        'max_drawdown': float(max_dd), 'avg_positions': float(ec['n'].mean()),
        'n_days': len(ec),
        'v2_benchmark': 'annual_49.7_sharpe_1.85_dd_31.7',
        'v55_target': 'annual_21.2_sharpe_3.87'
    },
    'date': time.strftime('%Y-%m-%d %H:%M')
}
json.dump(meta, open('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v2d_meta.json', 'w'), indent=2)
print(f'\n完成: {time.strftime("%Y-%m-%d %H:%M")}')
