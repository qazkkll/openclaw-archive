#!/usr/bin/env python3
"""
us_ml_02_feat_calc.py — 股票技术指标计算 + 5日标签
分批处理，控制内存。GPU无需参与(特征计算CPU足够快)

Out: /home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v4.parquet
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np

INPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_hist_yf_5y.parquet'
OUTPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v4.parquet'
CKPT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_feat_checkpoint.json'
BATCH = 300  # 300只一批

print("us_ml_02: 计算技术指标 + 5日标签...")

src = pd.read_parquet(INPUT)
tickers = sorted(src['ticker'].unique())
print(f"  {len(tickers)}只, {len(src):,}行")
del src

start = 0
if os.path.exists(CKPT):
    start = json.load(open(CKPT)).get('completed_to', 0)
    print(f"  断点: {start}/{len(tickers)}")

TOTAL_T0 = time.time()

for bs in range(start, len(tickers), BATCH):
    be = min(bs + BATCH, len(tickers))
    t0 = time.time()
    batch = tickers[bs:be]
    
    df = pd.read_parquet(INPUT, filters=[('ticker', 'in', batch)])
    
    # 每只股票算指标
    results = []
    for t, grp in df.groupby('ticker', sort=False):
        grp = grp.sort_values('date').reset_index(drop=True)
        c = grp['close'].values.astype(float)
        h = grp['high'].values.astype(float)
        l = grp['low'].values.astype(float)
        v = grp['volume'].values.astype(float)
        dates = grp['date'].values
        n = len(c)
        if n < 120:
            continue
        
        # 用pandas rolling一次性算所有指标
        s_close = pd.Series(c); s_high = pd.Series(h); s_low = pd.Series(l); s_vol = pd.Series(v)
        
        # 均线
        ma5 = s_close.rolling(5).mean().values
        ma10 = s_close.rolling(10).mean().values
        ma20 = s_close.rolling(20).mean().values
        ma30 = s_close.rolling(30).mean().values
        ma60 = s_close.rolling(60).mean().values
        
        # 波动率
        vol20 = s_close.rolling(20).std().fillna(0).values
        vol5 = s_close.rolling(5).std().fillna(0).values
        
        # EMA
        ema12 = s_close.ewm(span=12).mean().values
        ema26 = s_close.ewm(span=26).mean().values
        
        # MACD
        macd_line = ema12 - ema26
        macd_signal = pd.Series(macd_line).ewm(span=9).mean().values
        macd_hist = macd_line - macd_signal
        
        # RSI
        delta = np.diff(c, prepend=c[0])
        gain = np.where(delta>0, delta, 0)
        loss = np.where(delta<0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(span=14).mean().values
        avg_loss = pd.Series(loss).ewm(span=14).mean().values
        rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss>0.001)
        rsi14 = 100 - 100/(1+rs)
        
        # KDJ
        hh9 = s_high.rolling(9).max().values
        ll9 = s_low.rolling(9).min().values
        rsv = np.where((hh9-ll9)>0.01, (c-ll9)/(hh9-ll9)*100, 50)
        k_arr, d_arr = np.zeros(n), np.zeros(n)
        for i in range(n):
            k_arr[i] = 2/3*(k_arr[i-1] if i>0 else 50) + 1/3*rsv[i]
            d_arr[i] = 2/3*(d_arr[i-1] if i>0 else 50) + 1/3*k_arr[i]
        j_arr = 3*k_arr - 2*d_arr
        
        # Bollinger
        bb_std = s_close.rolling(20).std().fillna(0).values
        bb_upper = ma20 + 2*bb_std
        bb_lower = ma20 - 2*bb_std
        
        # 成交量比
        vol_ma5 = s_vol.rolling(5).mean().values
        vol_ma20 = s_vol.rolling(20).mean().values
        
        # ADX
        up_move = np.append(0, np.diff(h))
        down_move = np.append(0, -np.diff(l))
        p_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        m_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr = np.maximum(h-l, np.abs(h - np.append(c[0], c[:-1])))
        tr = np.maximum(tr, np.abs(l - np.append(c[0], c[:-1])))
        atr = pd.Series(tr).ewm(span=14).mean().values + 1e-8
        p_di = 100 * pd.Series(p_dm).ewm(span=14).mean().values / atr
        m_di = 100 * pd.Series(m_dm).ewm(span=14).mean().values / atr
        adx = 100 * np.abs(p_di - m_di) / np.maximum(p_di + m_di, 1e-8)
        adx = pd.Series(adx).ewm(span=14).mean().values
        
        # 价格位置(20天)
        hh20 = s_high.rolling(20).max().values
        ll20 = s_low.rolling(20).min().values
        price_pos = np.where((hh20-ll20)>0.01, (c-ll20)/(hh20-ll20), 0.5)
        
        # 未来5日收益
        fwd_ret = np.full(n, np.nan)
        fwd_ret[:-5] = c[5:] / c[:-5] - 1
        
        # 只取60天后（指标预热期）
        for i in range(60, n):
            results.append({
                'ticker': t, 'date': dates[i],
                'ma5': ma5[i], 'ma10': ma10[i], 'ma20': ma20[i],
                'ma30': ma30[i], 'ma60': ma60[i],
                'ma5_ratio': c[i]/ma5[i]-1 if ma5[i]>0 else 0,
                'ma20_ratio': c[i]/ma20[i]-1 if ma20[i]>0 else 0,
                'ma60_ratio': c[i]/ma60[i]-1 if ma60[i]>0 else 0,
                'vol5': vol5[i], 'vol20': vol20[i],
                'vol_ratio': vol5[i]/vol20[i] if vol20[i]>0.001 else 1,
                'ema12': ema12[i], 'ema26': ema26[i],
                'macd': macd_line[i], 'macd_signal': macd_signal[i],
                'macd_hist': macd_hist[i],
                'rsi14': rsi14[i],
                'k': k_arr[i], 'd': d_arr[i], 'j': j_arr[i],
                'bb_upper': bb_upper[i], 'bb_lower': bb_lower[i],
                'price_position': price_pos[i],
                'vol_ratio_ma5': v[i]/vol_ma5[i] if vol_ma5[i]>0 else 1,
                'vol_ratio_ma20': v[i]/vol_ma20[i] if vol_ma20[i]>0 else 1,
                'adx': adx[i], 'plus_di': p_di[i], 'minus_di': m_di[i],
                'fwd_5d_ret': fwd_ret[i],
            })
    
    if not results:
        json.dump({'completed_to': be}, open(CKPT, 'w'))
        continue
    
    df_out = pd.DataFrame(results)
    del results
    
    # 标签（>2%上涨=1, >2%下跌=-1, 其他=0）
    conds = [df_out['fwd_5d_ret'] > 0.02, df_out['fwd_5d_ret'] < -0.02]
    df_out['label'] = np.select(conds, [1, -1], default=0)
    
    # 写盘
    if bs == 0 or not os.path.exists(OUTPUT):
        df_out.to_parquet(OUTPUT, index=False)
    else:
        old = pd.read_parquet(OUTPUT)
        pd.concat([old, df_out], ignore_index=True).to_parquet(OUTPUT, index=False)
        del old
    
    del df_out, df
    json.dump({'completed_to': be}, open(CKPT, 'w'))
    
    sec = time.time() - t0
    pct = (be/len(tickers))*100
    print(f"  {bs}~{be}: {sec:.0f}s ({pct:.0f}%), {(time.time()-TOTAL_T0)/60:.0f}min总", flush=True)

# 最终统计
df = pd.read_parquet(OUTPUT)
feat_count = len([c for c in df.columns if c not in ['ticker','date','fwd_5d_ret','label']])
print(f"\nDONE! 行数:{len(df):,} 股票:{df['ticker'].nunique()} 特征:{feat_count}")
print(f"  标签: {df['label'].value_counts().to_dict()}")
print(f"  总耗时: {(time.time()-TOTAL_T0)/60:.0f}分钟")
if os.path.exists(CKPT):
    os.remove(CKPT)
