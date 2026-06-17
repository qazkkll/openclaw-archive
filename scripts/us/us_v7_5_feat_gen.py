#!/usr/bin/env python3
"""
us_v7_5_feat_gen.py — V7.5特征工程
基于10年yfinance数据，重算技术指标 + 大盘风控特征
合并主池+大盘池，统一特征空间

输入:
  - /home/hermes/.hermes/openclaw-project/scripts/system/us_hist_yf_10y.parquet (主池2436只x10年)
  - /home/hermes/.hermes/openclaw-project/scripts/system/us_hist_megacap_10y.parquet (大盘46只x10年)
输出:
  - /home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v75.parquet (~600万行, 50+列)

依赖: pip install yfinance
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd, numpy as np, yfinance as yf

BASE = '/home/hermes/.hermes/openclaw-archive'
ML_DIR = f'{BASE}/ml'
MAIN_INPUT = f'{ML_DIR}/us_hist_yf_10y.parquet'
MEGA_INPUT = f'{ML_DIR}/us_hist_megacap_10y.parquet'
OUTPUT = f'{ML_DIR}/us_ml_feats_v75.parquet'
CKPT = f'{ML_DIR}/us_v75_feat_checkpoint.json'
BATCH = 300  # 300只一批，控制内存

# 行业→ETF映射（从s5复制）
S2E = {
    'Technology':'XLK','Financial Services':'XLF','Financial':'XLF',
    'Energy':'XLE','Healthcare':'XLV','Industrials':'XLI',
    'Consumer Defensive':'XLP','Consumer Cyclical':'XLY','Utilities':'XLU',
    'Basic Materials':'XLB','Materials':'XLB','Real Estate':'XLRE',
    'Communication Services':'XLC','Semiconductors':'SMH',
}
ETF_SYMS = list(set(['SPY','QQQ','IWM','VXX','^VIX'] + list(S2E.values())))
WARMUP = 90  # 预热天数（修改：从60→90，因为macd/ema/bb都需要60天预热，但adx需要更久）

T0 = time.time()
print('='*60)
print('V7.5 特征工程 — 10年数据重算')
print('='*60)

# ============================================================
# 1. 加载源数据 合并主池+大盘
# ============================================================
print('\n[1/7] 加载源数据...')
main = pd.read_parquet(MAIN_INPUT)
main.rename(columns={'ticker': 'sym'}, inplace=True)  # 统一列名
print(f'  主池: {len(main):,}行, {main.sym.nunique()}只')

mega = pd.read_parquet(MEGA_INPUT)
print(f'  大盘: {len(mega):,}行, {mega.sym.nunique()}只')

# 合并，去重（有些ticker两边都有）
all_syms = set(main.sym.unique()) | set(mega.sym.unique())
print(f'  合并后: {len(all_syms)}只')

# 合并成一个df，统一列名 (sym, date, open, high, low, close, volume)
# megacap已经有sym列，主池需要rename
df_all = pd.concat([main, mega], ignore_index=True)
del main, mega
# 去重（同名同日期）
df_all = df_all.drop_duplicates(subset=['sym', 'date']).sort_values(['sym', 'date']).reset_index(drop=True)
print(f'  去重后: {len(df_all):,}行')

tickers = sorted(df_all.sym.unique())
print(f'  共{len(tickers)}只股票')

# ============================================================
# 2. 下载ETF/指数数据（SPY/QQQ/IWM/VIX）
# ============================================================
print('\n[2/7] 下载ETF/指数日K线...')
etf_data = {}
for e in ETF_SYMS:
    try:
        t = yf.Ticker(e)
        h = t.history(period='10y')  # 和股票同一时间窗口
        if len(h) == 0:
            print(f'  ⚠️ {e}: 无数据')
            continue
        if hasattr(h.columns, 'nlevels') and h.columns.nlevels > 1:
            h.columns = h.columns.droplevel(1)
        h = h.reset_index()
        # 统一date列名
        date_col = 'Date' if 'Date' in h.columns else ('date' if 'date' in h.columns else h.columns[0])
        h.rename(columns={date_col: 'date'}, inplace=True)
        h['date'] = pd.to_datetime(h['date']).dt.date
        # 存成 {date: close} dict
        closes = {str(r['date']): float(r['Close']) for _, r in h.iterrows()}
        etf_data[e] = closes
        print(f'  ✅ {e}: {len(closes)}行', flush=True)
    except Exception as ex:
        print(f'  ❌ {e}: {ex}')
print(f'  成功: {len(etf_data)}/{len(ETF_SYMS)}')

# VIX特殊处理（指数代码是^VIX，yfinance用^VIX或VIX）
if '^VIX' in etf_data and 'VIX' not in etf_data:
    etf_data['VIX'] = etf_data['^VIX']

# ============================================================
# 3. 计算ETF日收益率（适配股票的date格式）
# ============================================================
print('\n[3/7] 计算ETF日收益率...')

def compute_returns(closes, windows=[1, 5, 20, 60]):
    """计算各窗口收益率，返回 {date: {win: ret}}"""
    dates = sorted(closes.keys())
    results = {}
    for i, d in enumerate(dates):
        rets = {}
        for w in windows:
            if i >= w:
                rets[w] = (closes[d] - closes[dates[i-w]]) / closes[dates[i-w]]
            else:
                rets[w] = 0.0
        results[d] = rets
    return results

etf_rets = {}
for e, closes in etf_data.items():
    etf_rets[e] = compute_returns(closes)

print(f'  ETF收益率计算完成: {len(etf_rets)}只')

# ============================================================
# 4. 加载基本面数据（PE/市值/行业等，从v3_dated获取映射）
# ============================================================
print('\n[4/7] 加载基本面数据...')
try:
    v3_src = pd.read_parquet(f'{ML_DIR}/us_ml_feats_v3_dated.parquet')[['sym','pe_trailing','pe_forward','div_yield','beta','market_cap','sector','industry']]
    # 去重（每只取最新的）
    v3_src = v3_src.drop_duplicates(subset=['sym'], keep='last')
    # 存成dict
    fund_data = {}
    for _, r in v3_src.iterrows():
        fund_data[r['sym']] = {
            'pe_trailing': r.get('pe_trailing', np.nan),
            'pe_forward': r.get('pe_forward', np.nan),
            'div_yield': r.get('div_yield', np.nan),
            'beta': r.get('beta', np.nan),
            'sector': r.get('sector', 'Unknown'),
            'industry': r.get('industry', 'Unknown'),
        }
    print(f'  V3基本面加载: {len(fund_data)}只')
except Exception as ex:
    print(f'  ⚠️ 无法加载V3基本面: {ex}')
    fund_data = {}

# ============================================================
# 5. 分批处理：技术指标 + 大盘因子
# ============================================================
print(f'\n[5/7] 计算技术指标 + 大盘因子（{len(tickers)}只, 每次{BATCH}只）...')

start = 0
if os.path.exists(CKPT):
    start = json.load(open(CKPT)).get('completed_to', 0)
    print(f'  断点: {start}/{len(tickers)}')

for bs in range(start, len(tickers), BATCH):
    be = min(bs + BATCH, len(tickers))
    batch = tickers[bs:be]
    t1 = time.time()
    
    # 读取这批股票的全部数据
    df_batch = df_all[df_all['sym'].isin(batch)].copy()
    
    results = []
    for sym, grp in df_batch.groupby('sym', sort=False):
        grp = grp.sort_values('date').reset_index(drop=True)
        c = grp['close'].values.astype(float)
        h = grp['high'].values.astype(float)
        l = grp['low'].values.astype(float)
        v = grp['volume'].values.astype(float)
        dates = grp['date'].values
        n = len(c)
        if n < WARMUP:
            continue
        
        s_close = pd.Series(c); s_high = pd.Series(h)
        s_low = pd.Series(l); s_vol = pd.Series(v)
        
        # === 均线 ===
        ma5 = s_close.rolling(5).mean().values
        ma10 = s_close.rolling(10).mean().values
        ma20 = s_close.rolling(20).mean().values
        ma30 = s_close.rolling(30).mean().values
        ma60 = s_close.rolling(60).mean().values
        
        # === 波动率 ===
        vol5 = s_close.rolling(5).std().fillna(0).values
        vol20 = s_close.rolling(20).std().fillna(0).values
        
        # === MACD ===
        ema12 = s_close.ewm(span=12).mean().values
        ema26 = s_close.ewm(span=26).mean().values
        macd_line = ema12 - ema26
        macd_signal = pd.Series(macd_line).ewm(span=9).mean().values
        macd_hist = macd_line - macd_signal
        
        # === RSI(14) ===
        delta = np.diff(c, prepend=c[0])
        gain = np.where(delta>0, delta, 0)
        loss = np.where(delta<0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(span=14).mean().values
        avg_loss = pd.Series(loss).ewm(span=14).mean().values
        rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss>0.001)
        rsi14 = 100 - 100/(1+rs)
        
        # === KDJ ===
        hh9 = s_high.rolling(9).max().values
        ll9 = s_low.rolling(9).min().values
        rsv = np.where((hh9-ll9)>0.01, (c-ll9)/(hh9-ll9)*100, 50)
        k_arr, d_arr = np.zeros(n), np.zeros(n)
        for i in range(n):
            k_arr[i] = 2/3*(k_arr[i-1] if i>0 else 50) + 1/3*rsv[i]
            d_arr[i] = 2/3*(d_arr[i-1] if i>0 else 50) + 1/3*k_arr[i]
        j_arr = 3*k_arr - 2*d_arr
        
        # === Bollinger ===
        bb_std = s_close.rolling(20).std().fillna(0).values
        bb_upper = ma20 + 2*bb_std
        bb_lower = ma20 - 2*bb_std
        bb_width = np.where(ma20>0.01, (bb_upper - bb_lower)/ma20, 0)
        bb_pos = np.where((bb_upper-bb_lower)>0.01, (c-bb_lower)/(bb_upper-bb_lower), 0.5)
        
        # === 成交量 ===
        vol_ma5 = s_vol.rolling(5).mean().values
        vol_ma20 = s_vol.rolling(20).mean().values
        vol_ratio5 = np.divide(v, vol_ma5, out=np.ones_like(v, dtype=float), where=vol_ma5>0.001)
        vol_ratio20 = np.divide(v, vol_ma20, out=np.ones_like(v, dtype=float), where=vol_ma20>0.001)
        
        # === ADX ===
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
        
        # === 价格位置(20天) ===
        hh20 = s_high.rolling(20).max().values
        ll20 = s_low.rolling(20).min().values
        price_pos = np.where((hh20-ll20)>0.01, (c-ll20)/(hh20-ll20), 0.5)
        
        # === 价格位置(60天) ===
        hh60 = s_high.rolling(60).max().values
        ll60 = s_low.rolling(60).min().values
        price_pos60 = np.where((hh60-ll60)>0.01, (c-ll60)/(hh60-ll60), 0.5)
        
        # === CMF (Chaikin Money Flow, 成交量加权价格位置) ===
        mfv = ((c - l) - (h - c)) / np.maximum(h - l, 0.001) * v
        cmf = pd.Series(mfv).rolling(20).sum().values / np.maximum(vol_ma20 * 20, 0.001)
        
        # === 未来5日收益（标签） ===
        fwd_ret = np.full(n, np.nan)
        fwd_ret[:-5] = c[5:] / c[:-5] - 1
        
        # === 基本面（从缓存获取） ===
        fund = fund_data.get(sym, {})
        sector = fund.get('sector', 'Unknown')
        industry = fund.get('industry', 'Unknown')
        pe_t = fund.get('pe_trailing', np.nan)
        pe_f = fund.get('pe_forward', np.nan)
        dy = fund.get('div_yield', np.nan)
        beta_v = fund.get('beta', np.nan)
        mc = fund.get('market_cap', np.nan)
        
        # 行业编码（用于ETF因子映射）
        etf_key = S2E.get(sector, 'SPY')
        
        # === 构建特征行 ===
        for i in range(WARMUP, n):
            date_str = str(dates[i])[:10]
            
            # 大盘因子（从ETF数据获取该日的各窗口收益率）
            spy_r = etf_rets.get('SPY', {}).get(date_str, {})
            qqq_r = etf_rets.get('QQQ', {}).get(date_str, {})
            iwm_r = etf_rets.get('IWM', {}).get(date_str, {})
            vix_closes = etf_data.get('VIX', etf_data.get('^VIX', {}))
            
            # VIX值（取当日或最近）
            vix_vals = sorted(vix_closes.keys())
            vix_c = vix_closes.get(date_str, vix_closes.get(vix_vals[-1], 25) if vix_vals else 25)
            
            results.append({
                'sym': sym, 'date': dates[i],
                # 均线
                'ma5': ma5[i], 'ma10': ma10[i], 'ma20': ma20[i],
                'ma30': ma30[i], 'ma60': ma60[i],
                'ma5_ratio': c[i]/ma5[i]-1 if ma5[i]>0 else 0,
                'ma20_ratio': c[i]/ma20[i]-1 if ma20[i]>0 else 0,
                'ma60_ratio': c[i]/ma60[i]-1 if ma60[i]>0 else 0,
                # 波动率
                'vol5': vol5[i], 'vol20': vol20[i],
                'vol_ratio': vol5[i]/vol20[i] if vol20[i]>0.001 else 1,
                # MACD
                'ema12': ema12[i], 'ema26': ema26[i],
                'macd': macd_line[i], 'macd_signal': macd_signal[i],
                'macd_hist': macd_hist[i],
                # RSI
                'rsi14': rsi14[i],
                # KDJ
                'k': k_arr[i], 'd': d_arr[i], 'j': j_arr[i],
                # Bollinger
                'bb_upper': bb_upper[i], 'bb_lower': bb_lower[i],
                'bb_width': bb_width[i], 'bb_position': bb_pos[i],
                # 成交量
                'vol_ratio_ma5': vol_ratio5[i],
                'vol_ratio_ma20': vol_ratio20[i],
                # 趋势
                'adx': adx[i], 'plus_di': p_di[i], 'minus_di': m_di[i],
                # 价格位置
                'price_position': price_pos[i],
                'price_position_60': price_pos60[i],
                # 资金流
                'cmf': cmf[i],
                # 大盘因子（ETF窗口收益率）
                'spy_ret1': spy_r.get(1, 0.0),
                'spy_ret5': spy_r.get(5, 0.0),
                'spy_ret20': spy_r.get(20, 0.0),
                'spy_ret60': spy_r.get(60, 0.0),
                'qqq_ret1': qqq_r.get(1, 0.0),
                'qqq_ret5': qqq_r.get(5, 0.0),
                'qqq_ret20': qqq_r.get(20, 0.0),
                'qqq_ret60': qqq_r.get(60, 0.0),
                'iwm_ret1': iwm_r.get(1, 0.0),
                'iwm_ret5': iwm_r.get(5, 0.0),
                'iwm_ret20': iwm_r.get(20, 0.0),
                'iwm_ret60': iwm_r.get(60, 0.0),
                # 恐慌因子
                'vix_close': float(vix_c),
                # 基本面
                'pe_trailing': pe_t, 'pe_forward': pe_f,
                'div_yield': dy, 'beta': beta_v,
                'sector': sector, 'industry': industry,
                # 标签
                'fwd_5d_ret': fwd_ret[i],
            })
    
    if not results:
        json.dump({'completed_to': be}, open(CKPT, 'w'))
        print(f'  {bs}~{be}: 0行（全部预热不足跳过)', flush=True)
        continue
    
    df_out = pd.DataFrame(results)
    del results
    
    # === 标签（>2%涨=1, >2%跌=-1, 其他=0）===
    conds = [df_out['fwd_5d_ret'] > 0.02, df_out['fwd_5d_ret'] < -0.02]
    df_out['label'] = np.select(conds, [1, -1], default=0)
    
    # 行业ETF因子（sector_etf_ret5：从该股票sector对应的ETF取5日收益）
    sector_etf_rets = etf_rets.get('SPY', {})  # fallback
    etf_5d_map = {}
    for sk in etf_rets:
        etf_5d_map[sk] = {d: r.get(5, 0.0) for d, r in etf_rets[sk].items()}
    
    def get_sector_etf_ret(sector, date_str):
        sym = S2E.get(sector, 'SPY')
        dret = etf_5d_map.get(sym, {})
        return dret.get(date_str, 0.0)
    
    df_out['sector_etf_ret5'] = df_out.apply(
        lambda r: get_sector_etf_ret(r['sector'], str(r['date'])[:10]), axis=1)
    
    # sector编码
    df_out['sc'] = df_out['sector'].astype('category').cat.codes.astype(int)
    
    # === 写入磁盘 ===
    # parquet不支持混合类型，去掉文本列再写（sector/industry存元数据文件）
    df_numeric = df_out.drop(columns=['sector', 'industry'], errors='ignore')
    # sc是整数编码，保留
    if bs == 0 or not os.path.exists(OUTPUT):
        df_numeric.to_parquet(OUTPUT, index=False)
        print(f'  创建文件: {len(df_numeric)}行', flush=True)
    else:
        old = pd.read_parquet(OUTPUT)
        pd.concat([old, df_numeric], ignore_index=True).to_parquet(OUTPUT, index=False)
        del old
        print(f'  追加完成: {len(df_numeric)}行', flush=True)
    
    del df_out, df_batch
    json.dump({'completed_to': be}, open(CKPT, 'w'))
    
    sec = time.time() - t1
    pct = (be/len(tickers))*100
    elapsed = (time.time()-T0)/60
    # ETA估算
    per_batch = (time.time()-T0) / (be/BATCH) if be > 0 else 0
    remain_batches = (len(tickers) - be) / BATCH
    eta = remain_batches * per_batch / 60
    print(f'  [{pct:.0f}%] {bs}~{be}: {sec:.0f}s, {elapsed:.0f}min总, ETA~{eta:.0f}min', flush=True)

# ============================================================
# 6. 最终统计
# ============================================================
print(f'\n[6/7] 最终统计...')
df_final = pd.read_parquet(OUTPUT)
feature_cols = [c for c in df_final.columns if c not in ['sym','date','fwd_5d_ret','label','sector','industry']]
print(f'  行数: {len(df_final):,}')
print(f'  股票: {df_final.sym.nunique()}只')
print(f'  特征: {len(feature_cols)}列')
print(f'  日期: {df_final.date.min()} ~ {df_final.date.max()}')
print(f'  标签分布: {df_final.label.value_counts().to_dict()}')

# 每只股票平均年数
sym_years = df_final.groupby('sym')['date'].agg(['min','max'])
sym_years['years'] = (pd.to_datetime(sym_years['max']) - pd.to_datetime(sym_years['min'])).dt.days / 365.25
for lo, hi in [(0,1),(1,2),(2,3),(3,5),(5,10),(10,100)]:
    cnt = ((sym_years['years'] >= lo) & (sym_years['years'] < hi)).sum()
    print(f'  {lo}-{hi}y: {cnt}只 ({cnt/len(sym_years)*100:.1f}%)')
print(f'  中位数: {sym_years.years.median():.1f}年')

# ============================================================
# 7. 清理
# ============================================================
if os.path.exists(CKPT):
    os.remove(CKPT)
    print(f'\n[7/7] 断点已清理')

print(f'\n🎉 完成! 总耗时: {(time.time()-T0)/60:.0f}分钟')
print(f'✅ {OUTPUT}')
print('='*60)
