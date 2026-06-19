#!/usr/bin/env python3
"""
宏观数据每日更新脚本
从yfinance拉取最新VIX/SPY/QQQ/IWM数据，更新v75文件
"""
import os, time, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = lambda *a, **k: print(*a, **k, flush=True)

P("🔄 宏观数据更新")
P("=" * 40)

# 1. 加载现有v75
v75_path = os.path.join(ROOT, 'data/us/features/us_ml_feats_v75_filtered.parquet')
if os.path.exists(v75_path):
    v75 = pd.read_parquet(v75_path)
    last_date = v75['date'].max()
    P(f"   现有数据截止: {last_date}")
else:
    v75 = pd.DataFrame()
    last_date = None

# 2. 拉取最新宏观数据
P("   拉取宏观数据...")
today = datetime.now().strftime('%Y-%m-%d')
start = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')

macro_data = {}
for sym, prefix in [('^VIX', 'vix'), ('SPY', 'spy'), ('QQQ', 'qqq'), ('IWM', 'iwm')]:
    try:
        ticker = yf.Ticker(sym)
        hist = ticker.history(start=start, end=today)
        if len(hist) > 0:
            close = hist['Close']
            macro_data[f'{prefix}_close'] = float(close.iloc[-1])
            macro_data[f'{prefix}_ret1'] = float(close.pct_change(1).iloc[-1]) if len(close) > 1 else 0
            macro_data[f'{prefix}_ret5'] = float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0
            macro_data[f'{prefix}_ret20'] = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0
            macro_data[f'{prefix}_ret60'] = float(close.pct_change(60).iloc[-1]) if len(close) > 60 else 0
            P(f"   {sym}: {macro_data[f'{prefix}_close']:.2f}")
    except Exception as e:
        P(f"   ⚠️ {sym} 失败: {e}")

# 3. 添加到v75
if macro_data:
    new_row = {'date': pd.Timestamp(today)}
    new_row.update(macro_data)
    
    # 找到需要更新的列
    macro_cols = [c for c in v75.columns if any(c.startswith(p) for p in ['vix_', 'spy_', 'qqq_', 'iwm_'])]
    
    if len(v75) > 0:
        # 检查今天是否已有数据
        today_rows = v75[v75['date'] == pd.Timestamp(today)]
        if len(today_rows) > 0:
            # 更新今天的行
            for col, val in macro_data.items():
                if col in v75.columns:
                    v75.loc[v75['date'] == pd.Timestamp(today), col] = val
            P(f"   更新今天的数据")
        else:
            # 添加新行（只更新宏观列，其他为NaN）
            new_df = pd.DataFrame([new_row])
            v75 = pd.concat([v75, new_df], ignore_index=True)
            P(f"   添加新行")
    else:
        v75 = pd.DataFrame([new_row])
    
    # 保存
    v75.to_parquet(v75_path, index=False)
    P(f"   ✅ 保存: {v75_path}")
    P(f"   数据范围: {v75['date'].min()} ~ {v75['date'].max()}")
else:
    P("   ⚠️ 无新数据")

# 4. 同时更新OHLCV主数据
P("\n   更新OHLCV主数据...")
ohlcv_path = os.path.join(ROOT, 'data/us/us_hist_yf_10y.parquet')
if os.path.exists(ohlcv_path):
    df = pd.read_parquet(ohlcv_path)
    df = df.rename(columns={'ticker': 'sym'})
    last_ohlcv = df['date'].max()
    P(f"   OHLCV截止: {last_ohlcv}")
    
    if str(last_ohlcv)[:10] < today:
        # 拉取最新数据（只拉活跃股票）
        recent_syms = df[df['date'] >= (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')]['sym'].unique()
        P(f"   需更新: {len(recent_syms)}只股票")
        
        new_data = []
        for i, sym in enumerate(recent_syms[:200]):  # 限制200只避免yfinance限流
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(start=last_ohlcv.strftime('%Y-%m-%d'), end=today)
                if len(hist) > 0:
                    hist = hist.reset_index()
                    hist.columns = [c.lower() for c in hist.columns]
                    hist['ticker'] = sym
                    new_data.append(hist)
            except:
                pass
            if (i+1) % 50 == 0:
                P(f"   进度: {i+1}/{min(len(recent_syms), 200)}")
                time.sleep(1)  # 避免限流
        
        if new_data:
            new_df = pd.concat(new_data, ignore_index=True)
            # 合并到主数据
            old_df = df.rename(columns={'sym': 'ticker'})
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['ticker', 'date'], keep='last')
            combined = combined.rename(columns={'ticker': 'sym'})
            combined.to_parquet(ohlcv_path, index=False)
            P(f"   ✅ OHLCV更新完成: {len(combined):,}行")
        else:
            P(f"   ⚠️ 无新OHLCV数据")
    else:
        P(f"   OHLCV已是最新")

P(f"\n✅ 宏观数据更新完成")
