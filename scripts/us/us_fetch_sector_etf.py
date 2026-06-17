"""拉取美股行业ETF的5天收益率，对齐到sym的sector上"""
import sys, json, os, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _paths

T0 = time.time()
print("═══ 拉取行业ETF数据 ═══")

# 主要行业ETF
sector_etfs = {
    'XLK': 'Technology',
    'XLF': 'Financial', 
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'XLI': 'Industrials',
    'XLP': 'Consumer Defensive',
    'XLY': 'Consumer Cyclical',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services',
    'SMH': 'Semiconductor',
    'IBB': 'Biotech',
    'ARKK': 'Innovation Tech',
    'QQQ': 'Tech Growth',
    'SPY': 'Market (S&P500)',
    'IWM': 'Small Cap',
}

print(f"ETF数量: {len(sector_etfs)}")

# 拉取近60天日K + 计算5天收益率
all_data = {}
etfs = list(sector_etfs.keys())

for etf in etfs:
    try:
        tk = yf.Ticker(etf)
        hist = tk.history(period='3mo')
        if len(hist) < 10:
            print(f"  {etf}: 数据不足({len(hist)})")
            continue
        
        # 计算1天/5天收益率
        close = hist['Close'].values
        ret1 = np.full(len(close), np.nan)
        ret5 = np.full(len(close), np.nan)
        ret1[1:] = np.diff(close) / close[:-1] * 100
        ret5[5:] = (close[5:] - close[:-5]) / close[:-5] * 100
        
        # 最新值
        all_data[etf] = {
            'ret1': float(ret1[-1]) if not np.isnan(ret1[-1]) else 0,
            'ret5': float(ret5[-1]) if not np.isnan(ret5[-1]) else 0,
            'ret5_3d_ago': float(ret5[-3]) if not np.isnan(ret5[-3]) else 0,
            'ret5_5d_ago': float(ret5[-5]) if not np.isnan(ret5[-5]) else 0,
            'last_close': float(close[-1]),
            'sector_name': sector_etfs[etf],
            'hist_ret5': ret5[~np.isnan(ret5)].tolist()[-20:]  # 最近20个5天收益
        }
        print(f"  {etf}: ret5={all_data[etf]['ret5']:.2f}%", flush=True)
        time.sleep(0.3)
    except Exception as e:
        print(f"  {etf}: {e}")

# 保存
with open(_paths.ML_DIR + "/us_sector_etf.json", 'w') as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f"\nETF数据保存成功: {len(all_data)}只")
print(f"用时: {time.time()-T0:.0f}s")
