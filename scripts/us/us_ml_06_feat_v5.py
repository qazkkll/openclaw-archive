#!/usr/bin/env python3
"""
us_ml_06_feat_v5.py — v5特征（扩展版）：在v4基础上增加
- 大盘相对强度 (vs SPY)
- 价格动量因子 (5d/10d/20d ret)
- 波动率突变
- Sector编码(从现有数据算行业分组)
- 量价背离指标
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
import yfinance as yf

INPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_hist_yf_5y.parquet'
OUTPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v5.parquet'
CKPT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_feat_v5_ckpt.json'
BATCH = 300

print("us_ml_06: v5特征(大盘相对强度+动量+波动率突变)")

# 1. 拉SPY做大盘基准
print("拉SPY 5年...")
spy = yf.download('SPY', period='5y', progress=False)
if hasattr(spy.columns, 'nlevels') and spy.columns.nlevels > 1:
    spy.columns = [c[0] for c in spy.columns.to_flat_index()]
spy = spy.reset_index()
spy_dates = spy['index'].values
spy_close = spy['Close'].values
print(f"  SPY: {len(spy_dates)}天, {spy_dates[0]}~{spy_dates[-1]}")

# 建立日期→SPY收益查询
spy_series = pd.Series(spy_close, index=pd.to_datetime(spy_dates))
spy_ret_5d = spy_series.pct_change(5).values
spy_ret_10d = spy_series.pct_change(10).values
spy_ret_20d = spy_series.pct_change(20).values
spy_vol20 = spy_series.rolling(20).std().values

src = pd.read_parquet(INPUT)
tickers = sorted(src['ticker'].unique())
print(f"  {len(tickers)}只, {len(src):,}行")
del src

start = 0
if os.path.exists(CKPT):
    start = json.load(open(CKPT)).get('completed_to', 0)
    print(f"  断点: {start}/{len(tickers)}")

T0 = time.time()

for bs in range(start, len(tickers), BATCH):
    be = min(bs + BATCH, len(tickers))
    t0 = time.time()
    batch = tickers[bs:be]
    
    df = pd.read_parquet(INPUT, filters=[('ticker', 'in', batch)])
    df = df.sort_values(['ticker', 'date'])
    
    results = []
    for t, grp in df.groupby('ticker', sort=False):
        grp = grp.reset_index(drop=True)
        c = grp['close'].values.astype(float)
        h = grp['high'].values.astype(float)
        l = grp['low'].values.astype(float)
        v = grp['volume'].values.astype(float)
        dates = grp['date'].values.astype('datetime64[ns]')
        n = len(c)
        if n < 120:
            continue
        
        s_c = pd.Series(c); s_h = pd.Series(h); s_l = pd.Series(l); s_v = pd.Series(v)
        
        # === v4基础特征（精简版, 只保留最高贡献的）===
        ma5 = s_c.rolling(5).mean().values
        ma20 = s_c.rolling(20).mean().values
        ma60 = s_c.rolling(60).mean().values
        vol5 = s_c.rolling(5).std().fillna(0).values
        vol20 = s_c.rolling(20).std().fillna(0).values
        ema12 = s_c.ewm(span=12).mean().values
        ema26 = s_c.ewm(span=26).mean().values
        macd_line = ema12 - ema26
        macd_signal = pd.Series(macd_line).ewm(span=9).mean().values
        delta = np.diff(c, prepend=c[0])
        gain = np.where(delta>0, delta, 0)
        loss = np.where(delta<0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(span=14).mean().values
        avg_loss = pd.Series(loss).ewm(span=14).mean().values
        rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss>0.001)
        rsi14 = 100 - 100/(1+rs)
        
        hh20 = s_h.rolling(20).max().values
        ll20 = s_l.rolling(20).min().values
        price_pos = np.where((hh20-ll20)>0.01, (c-ll20)/(hh20-ll20), 0.5)
        
        bb_std = s_c.rolling(20).std().fillna(0).values
        bb_upper = ma20 + 2*bb_std
        bb_lower = ma20 - 2*bb_std
        
        # === 新特征 ===
        # 1. 动量因子
        ret_5d = s_c.pct_change(5).values
        ret_10d = s_c.pct_change(10).values
        ret_20d = s_c.pct_change(20).values
        
        # 2. 大盘相对强度 (alpha)
        # 对齐日期查SPY收益
        idx_map = {str(d)[:10]: i for i, d in enumerate(spy_dates)}
        alpha_5d = np.full(n, np.nan)
        alpha_10d = np.full(n, np.nan)
        
        for i in range(n):
            dstr = str(dates[i])[:10]
            if dstr in idx_map:
                si = idx_map[dstr]
                if si >= 5:
                    alpha_5d[i] = ret_5d[i] - spy_ret_5d[si]
                if si >= 10:
                    alpha_10d[i] = ret_10d[i] - spy_ret_10d[si]
        
        # 3. 波动率突变 (当前vol vs 60天平均vol)
        vol60 = s_c.rolling(60).std().fillna(0).values
        vol_shock = np.where(vol60>0.001, vol5 / vol60, 1)
        vol_regime = np.where(vol20>0.001, vol5 / vol20, 1)
        
        # 4. 量价背离
        # 价格跌但成交量涨 = bearish divergence
        price_down = (ret_5d < -0.02).astype(float)
        vol_up = (s_v > s_v.rolling(20).mean().values).astype(float)
        bearish_div = price_down * vol_up
        
        # 价格涨但成交量缩 = 假突破
        price_up = (ret_5d > 0.02).astype(float)
        vol_down = (s_v < s_v.rolling(20).mean().values * 0.7).astype(float)
        fake_break = price_up * vol_down
        
        # 5. 波动率趋势
        vol_ratio_5d_20d = np.where(vol20>0.001, vol5/vol20, 1)
        vol_ratio_20d_60d = np.where(vol60>0.01, vol20/vol60, 1)
        
        # 6. BB宽度(波动率指标)
        bb_width = bb_upper - bb_lower
        bb_pct = np.where(bb_width>0.001, (c-bb_lower)/bb_width, 0.5)
        
        # 7. 价格vs均线综合(均值回归强度)
        mean_reversion = (c/ma60 - 1) + (c/ma20 - 1) + (c/ma5 - 1)
        
        # 8. 累计收益斜率(趋势强度)
        slope_20d = np.full(n, 0.0)
        for i in range(20, n):
            x = np.arange(20)
            y = np.log(c[i-19:i+1] + 1e-10)
            slope_20d[i] = np.polyfit(x, y, 1)[0]
        
        # 9. 5日收益
        fwd_ret = np.full(n, np.nan)
        fwd_ret[:-5] = c[5:] / c[:-5] - 1
        
        for i in range(60, n):
            results.append({
                'ticker': t, 'date': dates[i],
                # v4 base
                'ma5_ratio': c[i]/ma5[i]-1, 'ma20_ratio': c[i]/ma20[i]-1,
                'ma60_ratio': c[i]/ma60[i]-1,
                'vol5': vol5[i], 'vol20': vol20[i],
                'macd': macd_line[i], 'macd_signal': macd_signal[i],
                'macd_hist': macd_line[i] - macd_signal[i],
                'rsi14': rsi14[i],
                'price_position': price_pos[i],
                'bb_upper': bb_upper[i], 'bb_lower': bb_lower[i],
                # v5 new
                'ret_5d': ret_5d[i], 'ret_10d': ret_10d[i], 'ret_20d': ret_20d[i],
                'alpha_5d': alpha_5d[i] if not np.isnan(alpha_5d[i]) else 0,
                'alpha_10d': alpha_10d[i] if not np.isnan(alpha_10d[i]) else 0,
                'vol_shock': vol_shock[i],
                'vol_regime': vol_regime[i],
                'bearish_div': bearish_div[i],
                'fake_break': fake_break[i],
                'vol_ratio_5_20': vol_ratio_5d_20d[i],
                'vol_ratio_20_60': vol_ratio_20d_60d[i],
                'bb_width': bb_width[i],
                'bb_pct': bb_pct[i],
                'mean_rev': mean_reversion[i],
                'slope_20d': slope_20d[i],
                'fwd_5d_ret': fwd_ret[i],
            })
    
    if not results:
        json.dump({'completed_to': be}, open(CKPT, 'w'))
        continue
    
    df_out = pd.DataFrame(results)
    del results
    
    # 标签 (>5% = 1)
    df_out['label'] = (df_out['fwd_5d_ret'] > 0.05).astype(int)
    
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
    print(f"  {bs}~{be}: {sec:.0f}s ({pct:.0f}%), {(time.time()-T0)/60:.0f}min总", flush=True)

df = pd.read_parquet(OUTPUT)
feat_count = len([c for c in df.columns if c not in ['ticker','date','fwd_5d_ret','label']])
print(f"\nDONE! 行数:{len(df):,} 股票:{df['ticker'].nunique()} 特征:{feat_count}")
print(f"  标签: {df['label'].value_counts().to_dict()}")
if os.path.exists(CKPT):
    os.remove(CKPT)
