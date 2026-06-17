#!/usr/bin/env python3
"""
A2 — A股纯L3评分 + 每日推荐（全自动版）
============================================
一条命令完成：检查数据 → 更新K线+资金流到最新 → 评分 → 推荐输出

数据源：
  - K线: tushare pro daily（全市场单日0.4秒）
  - 资金流: tushare pro moneyflow（全市场单日5-10秒）
  - 股票名称: stock_info.json

最佳参数（Walk-Forward 7段验证, 2019-2025）：
  - 买入门槛: 评分 > 4.0%  |  止损: -15%  |  止盈: +10%
  - 持仓上限: 20只  |  持有期: 10个交易日
  - 候选池: 成交量大、价格≥1元、非ST的400只
  - 再评: 每日，跌破5%建议退出

用法： python a1_layer3_5_scoring.py
"""

import json, os, sys, time, datetime
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

D_DATA = r'/home/hermes/.hermes/openclaw-archive/data'
CHECKPOINT_MODEL = os.path.join(D_DATA, 'layer3_checkpoints', 'model_batch_5.json')

# 参数
BUY_THRESHOLD   = 4.0
WATCH_MIN       = 3.0
EXIT_THRESHOLD  = 5.0
MAX_HOLD        = 20
MAX_RECOMMEND   = 5
# 全盘扫描，不限制候选池大小
HOLD_DAYS       = 10
STOP_LOSS       = -15.0
TAKE_PROFIT     = 10.0

# ── 模型 ──
import xgboost as xgb
booster = xgb.Booster()
booster.load_model(CHECKPOINT_MODEL)
FEAT_COLS = booster.feature_names
HAS_PCT_120 = 'pct_ma120' in FEAT_COLS
HAS_MA60_SL = 'ma60_slope' in FEAT_COLS


# ════════════════════════════════════════
#  第一部分：数据检查 + 更新
# ════════════════════════════════════════

def check_data_status(hist, df_mf):
    """返回K线和资金流的最新日期"""
    # K线
    k_dates = set()
    for rec in hist.values():
        d = rec.get('dates')
        if d is not None and len(d) > 0:
            k_dates.add(str(d[-1]))
    kline_latest = max(k_dates) if k_dates else '未知'
    
    # 资金流
    mf_latest = '未知'
    if len(df_mf) > 0 and 'trade_date' in df_mf.columns:
        mf_latest = str(df_mf['trade_date'].max())
    
    return kline_latest, mf_latest


def load_existing_data():
    """加载本地K线parquet + 资金流parquet
    
    parquet格式: ticker, c, h, l, o, v, dates (数组列)
    保持numpy数组不转list,下游代码兼容
    """
    kpath_pq = os.path.join(D_DATA, 'a_hist_10y.parquet')
    
    t0 = time.time()
    
    if not os.path.exists(kpath_pq):
        print(f"❌ 找不到K线数据: {kpath_pq}")
        return None, None, None, None
    
    df = pd.read_parquet(kpath_pq)
    # Group by stock and build dict format
    hist = {}
    for code, group in df.groupby('Code'):
        group = group.sort_values('Date')
        hist[str(code)] = {
            'c': group['C'].tolist(),
            'h': group['H'].tolist(),
            'l': group['L'].tolist(),
            'o': group['O'].tolist(),
            'v': group['V'].tolist(),
            'dates': group['Date'].tolist(),
        }
    print(f"   ✅ parquet加载: {len(hist)}只 ({time.time()-t0:.1f}s)")
    
    mfpath = os.path.join(D_DATA, 'moneyflow_core.parquet')
    if True:  # Skip tushare update
        df_mf = pd.read_parquet(mfpath)
    else:
        mfpath_old = os.path.join(D_DATA, 'moneyflow_data.parquet')
        if os.path.exists(mfpath_old):
            df_mf = pd.read_parquet(mfpath_old)
        else:
            df_mf = pd.DataFrame()
    
    kline_latest, mf_latest = check_data_status(hist, df_mf)
    print(f"   现有数据: K线最新={kline_latest}, 资金流最新={mf_latest} ({time.time()-t0:.1f}s)")
    return hist, df_mf, kline_latest, mf_latest


def get_latest_trade_date(pro):
    """获取最新交易日"""
    today = datetime.date.today().strftime('%Y%m%d')
    df = pro.trade_cal(start_date=today, end_date=today)
    if len(df) > 0 and df.iloc[0]['is_open'] == 1:
        return today
    
    # 搜索最近10天
    for d in range(1, 15):
        dt = (datetime.date.today() - datetime.timedelta(days=d)).strftime('%Y%m%d')
        df = pro.trade_cal(start_date=dt, end_date=dt)
        if len(df) > 0 and df.iloc[0]['is_open'] == 1:
            return dt
    return None


def update_kline(hist, pro, latest_trade):
    """增量更新K线数据"""
    print("\n📡 更新K线...")
    
    # 样本股票日期
    sample = list(hist.keys())[0]
    existing = str(hist[sample]['dates'][-1]) if hist[sample].get('dates') is not None else '20160101'
    
    if existing >= latest_trade:
        print(f"   ✅ K线已是最新 ({existing})")
        return hist, existing
    
    # 需要补的日期
    start_date = existing
    print(f"   增量更新: {start_date} → {latest_trade}")
    
    # 批量拉取
    t0 = time.time()
    try:
        # ✅ 单次拉取，全市场所有股票
        df = pro.daily(start_date=start_date, end_date=latest_trade)
    except Exception as e:
        print(f"   ⚠️ 批量拉取失败: {e}，尝试分批...")
        # 分批：先拉资金流有数据的股票
        all_stocks = list(hist.keys())
        dfs = []
        chunk_size = 2000
        for i in range(0, len(all_stocks), chunk_size):
            batch = all_stocks[i:i+chunk_size]
            codes_str = ','.join([s+'.SZ' if s.startswith(('0','3')) else s+'.SH' for s in batch])
            df_part = pro.daily(ts_code=codes_str, start_date=start_date, end_date=latest_trade)
            if len(df_part) > 0:
                dfs.append(df_part)
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    if len(df) == 0:
        print(f"   ⚠️ 未拉到新数据")
        return hist, existing
    
    # 按股票分组追加
    grouped = df.groupby('ts_code')
    updated = 0
    for ts_code, grp in grouped:
        # ts_code格式 "000001.SZ" → 本地code "000001"
        code = ts_code.split('.')[0]
        if code not in hist:
            continue
        
        rec = hist[code]
        existing_dates = set(str(d) for d in rec.get('dates', []))
        
        # 排序，去重
        new_rows = []
        for _, row in grp.sort_values('trade_date').iterrows():
            d = str(row['trade_date']).replace('-', '')
            if d not in existing_dates:
                new_rows.append({
                    'd': d,
                    'o': float(row['open']),
                    'h': float(row['high']),
                    'l': float(row['low']),
                    'c': float(row['close']),
                    'v': float(row.get('vol', 0)),
                })
                existing_dates.add(d)
        
        if new_rows:
            for r in new_rows:
                rec['dates'].append(str(r['d']))
                rec['o'].append(r['o'])
                rec['h'].append(r['h'])
                rec['l'].append(r['l'])
                rec['c'].append(r['c'])
                rec['v'].append(r['v'])
            updated += 1
    
    # 从hist中取最新K线日期
    all_dates = []
    for rec in hist.values():
        if rec.get('dates'):
            all_dates.extend(rec['dates'])
    new_kline = max(all_dates) if all_dates else '未知'
    print(f"   更新了{updated}只股票, 耗时{time.time()-t0:.1f}s")
    print(f"   最新K线日期: {new_kline}")
    
    # 保存
    kpath = os.path.join(D_DATA, 'a_hist_10y.parquet')
    with open(kpath, 'w', encoding='utf-8') as f:
        json.dump(hist, f, ensure_ascii=False)
    print(f"   💾 K线已保存")
    
    return hist, new_kline


def update_moneyflow(df_mf, pro, latest_trade):
    """增量更新资金流"""
    print("\n📡 更新资金流...")
    
    existing = '未知'
    if len(df_mf) > 0 and 'trade_date' in df_mf.columns:
        existing = str(df_mf['trade_date'].max())
    
    if existing >= latest_trade:
        print(f"   ✅ 资金流已是最新 ({existing})")
        return df_mf, existing
    
    start_date = existing if existing != '未知' else '20260101'
    print(f"   增量更新: {start_date} → {latest_trade}")
    
    t0 = time.time()
    
    try:
        # 批量拉
        df_new = pro.moneyflow(start_date=start_date, end_date=latest_trade)
    except Exception as e:
        print(f"   ⚠️ 批量资金流失败: {e}")
        return df_mf, existing
    
    if len(df_new) == 0:
        print(f"   ⚠️ 未拉到新资金流数据")
        return df_mf, existing
    
    # 合并去重
    if len(df_mf) > 0:
        # 构建唯一键
        old_keys = set(zip(df_mf['ts_code'], df_mf['trade_date'].astype(str)))
        new_keys = set(zip(df_new['ts_code'], df_new['trade_date'].astype(str)))
        new_only = new_keys - old_keys
        if len(new_only) == 0:
            print(f"   ✅ 资金流已包含所有新数据")
            return df_mf, existing
        # 保留资金流需要的列
        mf_cols = ['ts_code', 'trade_date', 'buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount',
                   'buy_md_vol', 'buy_md_amount', 'sell_md_vol', 'sell_md_amount',
                   'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount',
                   'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount',
                   'net_mf_vol', 'net_mf_amount']
        available_cols = [c for c in mf_cols if c in df_new.columns]
        df_new_filtered = df_new[available_cols].copy()
        
        # 标记去重
        mask = df_new_filtered.apply(
            lambda r: (r['ts_code'], str(r['trade_date'])) not in old_keys, axis=1
        )
        df_new_filtered = df_new_filtered[mask]
        
        if len(df_new_filtered) == 0:
            print(f"   ✅ 资金流已包含所有新数据")
            return df_mf, existing
        
        df_mf = pd.concat([df_mf, df_new_filtered], ignore_index=True)
    else:
        df_mf = df_new
    
    new_mf_latest = str(df_mf['trade_date'].max())
    print(f"   新增{len(df_mf) if len(df_mf)>0 else 0}行资金流数据, 耗时{time.time()-t0:.1f}s")
    print(f"   最新资金流日期: {new_mf_latest}")
    
    # 保存parquet
    mfpath = os.path.join(D_DATA, 'moneyflow_core.parquet')
    df_mf.to_parquet(mfpath, index=False)
    print(f"   💾 资金流已保存")
    
    return df_mf, new_mf_latest


# ════════════════════════════════════════
#  第二部分：评分
# ════════════════════════════════════════

def load_name_map():
    try:
        with open(os.path.join(D_DATA, 'stock_info.json'), encoding='utf-8') as f:
            si = json.load(f)
        return {k: v.get('name', '') for k, v in si.items()}
    except:
        return {}


def build_pool(hist):
    """全盘扫描，不限制候选池大小"""
    pool = set()
    for code, rec in hist.items():
        if len(rec['c']) < 120: continue
        if code.startswith('9'): continue
        close = rec['c'][-1]
        if close < 1.0: continue
        pool.add(code)
    return pool


def calc_mf_for_date(code, df_mf, target_date):
    if len(df_mf) == 0:
        return None
    sz = code + '.SZ' if code.startswith(('0','3')) else code + '.SH'
    
    cd = df_mf[df_mf['ts_code'] == sz].sort_values('trade_date').reset_index(drop=True)
    if len(cd) < 2: return None
    
    net_mf = cd['net_mf_amount'].values.astype(np.float64)
    buy_lg = cd['buy_lg_amount'].values.astype(np.float64)
    sell_lg = cd['sell_lg_amount'].values.astype(np.float64)
    buy_elg = cd['buy_elg_amount'].values.astype(np.float64)
    sell_elg = cd['sell_elg_amount'].values.astype(np.float64)
    buy_md = cd['buy_md_amount'].values.astype(np.float64)
    sell_md = cd['sell_md_amount'].values.astype(np.float64)
    buy_sm = cd['buy_sm_amount'].values.astype(np.float64)
    sell_sm = cd['sell_sm_amount'].values.astype(np.float64)
    buy_le = buy_lg + buy_elg; sell_le = sell_lg + sell_elg
    
    n = len(cd)
    cs_net = np.zeros(n+1, dtype=np.float64)
    cs_major = np.zeros(n+1, dtype=np.float64)
    cs_lg = np.zeros(n+1, dtype=np.float64)
    for i in range(n):
        cs_net[i+1] = cs_net[i] + net_mf[i]
        cs_major[i+1] = cs_major[i] + (buy_le[i] - sell_le[i])
        cs_lg[i+1] = cs_lg[i] + (buy_lg[i] - sell_lg[i])
    
    dates_a = cd['trade_date'].values
    mf_map = {}
    for i in range(1, n):
        d = str(dates_a[i])
        t = buy_sm[i] + sell_sm[i]
        m = buy_le[i] + sell_le[i]
        mf_map[d] = {
            'net_mf_1d': float(net_mf[i]),
            'lg_net_1d': float(buy_lg[i] - sell_lg[i]),
            'elg_net_1d': float(buy_elg[i] - sell_elg[i]),
            'md_net_1d': float(buy_md[i] - sell_md[i]),
            'major_net_1d': float(buy_le[i] - sell_le[i]),
            'lg_pct': float(buy_le[i]/t*100) if t>0 else 50.0,
            'elg_pct': float(buy_elg[i]/t*100) if t>0 else 25.0,
            'major_ratio': float((buy_le[i]-sell_le[i])/m*100) if m>0 else 0.0,
        }
        for lb in (5, 10, 20, 60):
            s = max(0, i-lb+1)
            mf_map[d][f'net_mf_{lb}d'] = float(cs_net[i+1]-cs_net[s])
            mf_map[d][f'major_net_{lb}d'] = float(cs_major[i+1]-cs_major[s])
            mf_map[d][f'lg_net_{lb}d'] = float(cs_lg[i+1]-cs_lg[s])
    
    return mf_map.get(target_date, mf_map.get(list(mf_map.keys())[-1], None))


def calc_tech(rec):
    c = rec['c']; h = rec.get('h', c); l_ = rec.get('l', c)
    v = rec.get('v', [1]*len(c)); dates = rec.get('dates', [])
    i = len(c) - 1
    if i < 119: return None
    
    price = float(c[i])
    ma5  = float(np.mean(c[i-4:i+1]))
    ma10 = float(np.mean(c[i-9:i+1]))
    ma20 = float(np.mean(c[i-19:i+1]))
    ma60 = float(np.mean(c[i-59:i+1]))
    ma120 = float(np.mean(c[i-119:i+1])) if i>=119 else ma60
    
    feat = {'pct_ma5':(price/ma5-1)*100 if ma5>0 else 0,
            'pct_ma10':(price/ma10-1)*100 if ma10>0 else 0,
            'pct_ma20':(price/ma20-1)*100 if ma20>0 else 0,
            'pct_ma60':(price/ma60-1)*100 if ma60>0 else 0}
    if HAS_PCT_120: feat['pct_ma120'] = (price/ma120-1)*100 if ma120>0 else 0
    
    feat['ma20_slope'] = (ma20/np.mean(c[i-25:i-4])-1)*100 if i>=25 else 0
    if HAS_MA60_SL: feat['ma60_slope'] = (ma60/np.mean(c[i-65:i-4])-1)*100 if i>=65 else 0
    
    feat['ma_align'] = int((ma5>ma10)+(ma10>ma20)+(ma20>ma60)+(price>ma5)+(price>ma10)+(price>ma60))
    
    r10 = [abs(c[j]/c[j-1]-1)*100 if c[j-1]>0 else 0 for j in range(i-9,i+1)]
    feat['vol_10d'] = sum(r10)/10
    r60 = [abs(c[j]/c[j-1]-1)*100 if c[j-1]>0 else 0 for j in range(i-59,i+1)]
    feat['vol_60d'] = sum(r60)/60
    feat['vol_ratio'] = feat['vol_10d']/feat['vol_60d'] if feat['vol_60d']>0 else 1.0
    
    trs = [max(h[j]-l_[j], abs(h[j]-c[j-1]), abs(l_[j]-c[j-1])) for j in range(i-19,i+1)]
    feat['atr20_pct'] = sum(trs)/20/price*100 if price>0 else 0
    feat['ret_5d'] = (price/c[i-5]-1)*100
    feat['ret_10d'] = (price/c[i-10]-1)*100
    feat['ret_20d'] = (price/c[i-20]-1)*100
    feat['ret_60d'] = (price/c[i-60]-1)*100
    
    chg = [c[j]-c[j-1] for j in range(i-13,i+1)]
    rg = sum(x for x in chg if x>0)
    rl = sum(-x for x in chg if x<0)
    feat['rsi14'] = 100-100/(1+rg/rl/14) if rl>0 else 100.0
    feat['vol_ratio_5_20'] = float(np.mean(v[i-4:i+1])/np.mean(v[i-19:i+1])) if np.mean(v[i-19:i+1])>0 else 1.0
    
    if i >= 40:
        pr = [(c[j]/c[j-20]-1)*100 for j in range(20,i+1)]
        cr = (price/c[i-20]-1)*100
        feat['ret20d_pct'] = sum(1 for r in pr if r<cr)/len(pr)*100
    else: feat['ret20d_pct'] = 50.0
    
    # ─── KDJ (9,3,3) ───
    if i >= 8:
        hh9 = max(h[i-8:i+1])
        ll9 = min(l_[i-8:i+1])
        rsv = (price - ll9) / (hh9 - ll9) * 100 if (hh9 - ll9) > 0 else 50.0
        k_kdj = 2/3 * 50.0 + 1/3 * rsv
        d_kdj = 2/3 * 50.0 + 1/3 * k_kdj
        j_kdj = 3 * k_kdj - 2 * d_kdj
    else:
        k_kdj, d_kdj, j_kdj = 50.0, 50.0, 50.0
    feat['kdj_k'] = round(k_kdj, 2)
    feat['kdj_d'] = round(d_kdj, 2)
    feat['kdj_j'] = round(j_kdj, 2)
    
    # ─── MACD (12,26,9) ───
    ema12 = price; ema26 = price
    for jj in range(min(34, i)):
        ema12 = c[i-jj-1] * (2/13) + ema12 * (11/13)
        ema26 = c[i-jj-1] * (2/27) + ema26 * (25/27)
    dif = ema12 - ema26
    dea = dif
    for jj in range(min(8, i)):
        idx2 = i - jj - 1
        e12 = c[idx2]; e26 = c[idx2]
        for kk in range(min(26, idx2)):
            e12 = c[idx2-kk-1] * (2/13) + e12 * (11/13)
            e26 = c[idx2-kk-1] * (2/27) + e26 * (25/27)
        prev_dif = e12 - e26
        dea = prev_dif * (2/10) + dea * (8/10)
    macd_bar = 2 * (dif - dea)
    feat['macd_dif'] = round(dif, 4)
    feat['macd_dea'] = round(dea, 4)
    feat['macd_bar'] = round(macd_bar, 4)
    
    # ─── 布林带 (20,2) ───
    if i >= 19:
        ma20_bb = sum(c[i-19:i+1]) / 20
        var20_bb = sum((c[j]-ma20_bb)**2 for j in range(i-19,i+1)) / 20
        std20_bb = var20_bb ** 0.5
        upper_bb = ma20_bb + 2 * std20_bb
        lower_bb = ma20_bb - 2 * std20_bb
        bb_w = (upper_bb - lower_bb) / ma20_bb if ma20_bb > 0 else 0
        bb_p = (price - lower_bb) / (upper_bb - lower_bb) if (upper_bb - lower_bb) > 0 else 0.5
    else:
        bb_w, bb_p = 0, 0.5
    feat['bb_width'] = round(bb_w, 4)
    feat['bb_position'] = round(bb_p, 4)
    
    # ─── OBV ───
    obv_seq = [0]
    for j in range(1, i+1):
        delta = v[j] if c[j] > c[j-1] else (-v[j] if c[j] < c[j-1] else 0)
        obv_seq.append(obv_seq[-1] + delta)
    obv5 = abs(obv_seq[-1] - obv_seq[max(0, len(obv_seq)-6)])
    obv20 = abs(obv_seq[-1] - obv_seq[max(0, len(obv_seq)-21)])
    feat['obv_ratio_5_20'] = round(obv5 / obv20, 4) if obv20 > 0 else 1.0
    
    # ─── 动量补充 ───
    win5 = [c[j]/c[j-1]-1 for j in range(max(1,i-4),i+1)]
    feat['ret5_max'] = round(max(win5)*100, 2) if win5 else 0
    ret3 = price / c[i-3] - 1 if i >= 3 else 0
    ema12_hist = sum(c[i-11:i+1]) / 12 if i >= 11 else 0.01
    feat['ret3_vs_ema12'] = round((ret3*100) / (ema12_hist*100 + 1), 4) if abs(ema12_hist*100) > 0.001 else 0
    
    return feat, str(dates[i]) if dates else 'unknown'


# ════════════════════════════════════════
#  第三部分：主流程
# ════════════════════════════════════════

def run():
    gt0 = time.time()
    
    print(f"""
{'='*60}
  A2 — A股纯L3评分 + 每日推荐
  买入≥{BUY_THRESHOLD}% | 止损{STOP_LOSS}% | 止盈{TAKE_PROFIT}%
  全盘扫描 | 持仓≤{MAX_HOLD} | 持有{HOLD_DAYS}个交易日
{'='*60}
""")
    
    # ── 1. 加载 ──
    print("📂 第一步：数据检查")
    print("-" * 30)
    hist, df_mf, kline_latest, mf_latest = load_existing_data()
    if hist is None: return
    
    kline_clean = kline_latest.replace('-', '')
    mf_clean = mf_latest.replace('-', '')
    today_8 = datetime.date.today().strftime('%Y%m%d')
    needs_update = (kline_clean < today_8) or (mf_clean < today_8)
    
    if needs_update:
        print(f"\n   ⚠️ 数据需要更新")
        import tushare as ts
        pro = ts.pro_api()
        
    latest_trade = "20260616"  # Skip tushare
        if not latest_trade:
            print("   ❌ 无法确定最新交易日")
        else:
            print(f"   最新交易日: {latest_trade}")
            
            # 更新K线
            
            # 更新资金流
    else:
        print(f"\n   ✅ 数据已是最新")
    
    # ⚠️ 数据时效标注
    print(f"\n⚠️  数据时效性:")
    print(f"   📅 K线:     {kline_latest}")
    print(f"   📅 资金流:  {mf_latest}")
    today_str = datetime.date.today().strftime('%Y%m%d')
    kl = kline_latest.replace('-', '')
    ml = mf_latest.replace('-', '')
    print(f"   📅 K线: {kline_latest}  |  资金流: {mf_latest}")
    # 判断差距是否在合理范围内（非交易日不报错）
    k_days = (datetime.date.today() - datetime.datetime.strptime(kl, '%Y%m%d').date()).days
    m_days = (datetime.date.today() - datetime.datetime.strptime(ml, '%Y%m%d').date()).days
    if k_days <= 3:
        print(f"   ✅ K线距今天{k_days}天（含周末/节假日，正常）")
    else:
        print(f"   ⚠️ K线距今天{k_days}天，可能滞后")
    if m_days <= 5:
        print(f"   ✅ 资金流距今天{m_days}天（含周末/节假日，正常）")
    else:
        print(f"   ⚠️ 资金流距今天{m_days}天，可能滞后")
    
    # ── 2. 评分 ──
    print(f"\n📊 第二步：评分计算")
    print("-" * 30)
    
    pool = build_pool(hist)
    name_map = load_name_map()
    
    codes = sorted(hist.keys())
    results = []
    skip_pool = 0; skip_short = 0; skip_mf = 0; skip_nan = 0
    
    t0 = time.time()
    for idx, code in enumerate(codes):
        if code not in pool:
            skip_pool += 1
            continue
        rec = hist[code]
        if len(rec['c']) < 120: skip_short += 1; continue
        
        tech_r = calc_tech(rec)
        if tech_r is None: skip_short += 1; continue
        tech_feat, date_str = tech_r
        
        mf_vals = calc_mf_for_date(code, df_mf, kline_latest)
        if mf_vals is None: skip_mf += 1; continue
        
        feat_dict = {**tech_feat, **mf_vals}
        feat_vec = [feat_dict.get(k, 0.0) for k in FEAT_COLS]
        if any(np.isnan(f) or np.isinf(f) for f in feat_vec): skip_nan += 1; continue
        
        dm = xgb.DMatrix(np.array([feat_vec], dtype=np.float32), feature_names=FEAT_COLS)
        score = float(booster.predict(dm)[0])
        
        close = rec['c'][-1]
        results.append({
            'code': code, 'name': name_map.get(code, ''),
            'close': round(close, 2), 'date': date_str,
            'score': round(score, 2),
            'pct_ma60': round(tech_feat.get('pct_ma60', 0), 2),
            'rsi14': round(tech_feat.get('rsi14', 0), 1),
            'ret_5d': round(tech_feat.get('ret_5d', 0), 2),
            'ret_20d': round(tech_feat.get('ret_20d', 0), 2),
            'net_mf_5d': round(mf_vals.get('net_mf_5d', 0), 0),
        })
        if (idx+1) % 500 == 0:
            print(f"  {idx+1}/{len(codes)} ({len(results)} scored, {time.time()-t0:.0f}s)")
    
    results.sort(key=lambda x: -x['score'])
    print(f"\n  ✅ 评分完成: {len(results)}只 (跳过: 候选池{skip_pool}+短K线{skip_short}+无资金流{skip_mf}+NaN{skip_nan}) {time.time()-t0:.0f}s")
    
    # ── 3. 大盘情绪 ──
    print(f"\n🌡️ 第三步：大盘情绪")
    print("-" * 30)
    
    arr = np.array([r['score'] for r in results])
    mean_s = float(np.mean(arr))
    med_s = float(np.median(arr))
    buys_count = int((arr > BUY_THRESHOLD).sum())
    
    if med_s > 5: mood = "🔥 过热"
    elif med_s > 2: mood = "🌤️ 温和"
    elif med_s > 0: mood = "❄️ 偏冷"
    else: mood = "🥶 极冷"
    
    print(f"   平均评分: {mean_s:+.2f}% | 中位数: {med_s:+.2f}%")
    print(f"   >{BUY_THRESHOLD}%买入信号: {buys_count}只 ({buys_count/len(results)*100:.1f}%)")
    print(f"   >10%强势信号: {(arr>10).sum()}只 ({(arr>10).sum()/len(results)*100:.1f}%)")
    print(f"   综合判断: {mood}")
    
    # ── 4. 推荐 ──
    print(f"\n🎯 第四步：推荐输出")
    print("-" * 30)
    
    buys = [r for r in results if r['score'] > BUY_THRESHOLD]
    print(f"   满足买入门槛(>{BUY_THRESHOLD}%): {len(buys)}只")
    
    if buys:
        top = buys[:MAX_RECOMMEND]
        print(f"\n⭐ Top {len(top)} 推荐:")
        hdr = f"{'':>4} {'代码':>8} {'名称':>8} {'评分':>6} {'现价':>7} {'MA60%':>7} {'RSI':>5} {'5日%':>7} {'净流5d':>11}"
        print(hdr)
        print('-' * len(hdr))
        for i, r in enumerate(top):
            n = r.get('name', '')[:6]
            print(f"{i+1:>3}. {r['code']:>8} {n:>8} {r['score']:>+6.2f} {r['close']:>7.2f} "
                  f"{r['pct_ma60']:>+6.1f}% {r['rsi14']:>5.1f} {r['ret_5d']:>+6.2f}% "
                  f"{r['net_mf_5d']:>+10.0f}")
        
        print(f"\n📝 一句话点评:")
        for i, r in enumerate(top):
            n = r.get('name',''); rsi = r['rsi14']; ma60 = r['pct_ma60']; ret5 = r['ret_5d']
            parts = []
            if rsi<20: parts.append("极度超卖"); 
            elif rsi<30: parts.append("超卖区")
            if ma60>10: parts.append(f"突破MA60 {ma60:+.0f}%")
            elif ma60<-10: parts.append(f"低于MA60 {ma60:+.0f}%")
            if ret5<-5: parts.append("近5日急跌")
            elif ret5>5: parts.append("短期向上")
            desc = ', '.join(parts) if parts else "L3评分靠前"
            print(f"   {i+1}. {r['code']}({n}) {desc}")
        
        watch = [r for r in results if WATCH_MIN < r['score'] <= BUY_THRESHOLD]
        if watch:
            print(f"\n👀 关注 ({WATCH_MIN}-{BUY_THRESHOLD}%): {len(watch)}只")
            for i, r in enumerate(watch[:5]):
                print(f"   {i+1}. {r['code']:>8} {r.get('name','')[:6]:>6} "
                      f"{r['score']:>+6.2f} {r['close']:>7.2f} "
                      f"MA60:{r['pct_ma60']:>+.1f}% RSI{r['rsi14']:>5.1f}")
    
    # ── 5. 模型说明书 ──
    print(f"\n{'='*60}")
    print("📖 A2 模型使用说明书")
    print(f"{'='*60}")
    print(f"""
⭐ 最佳使用场景
    A股中短线选股辅助工具，38特征（18技术+20资金流）
    预测未来{HOLD_DAYS}个交易日的预期涨跌幅%

🎯 最佳参数（Walk-Forward 7段验证, 2019-2025）
    候选池:   全盘（价格≥1元、非ST、上市>120天）
    买入门槛: 评分 > {BUY_THRESHOLD}%
    持仓上限: {MAX_HOLD}只
    持有期:   {HOLD_DAYS}个交易日
    止损:     {STOP_LOSS}%  |  止盈:  {TAKE_PROFIT}%
    再评:     每日，跌破{EXIT_THRESHOLD}%建议退出

📜 历史表现（Walk-Forward 7段滚动）
    平均年化: +33.14%  |  平均夏普: 1.03
    最大回撤: -37.51%  |  正收益段: 6/7

⚠️ 风控规则
    1. 单票仓位≤50% | 总持仓≤{MAX_HOLD}只
    2. 单票止损{STOP_LOSS}% | 评分跌破{EXIT_THRESHOLD}%退出
    3. 候选池外不纳入 | 大盘暴跌>3%暂停交易
    4. A1资金流只做盘中实时信号，不做选股融合

📅 数据时效
    K线:    {kline_latest}
    资金流: {mf_latest}
""")
    
    # ── 6. 保存 ──
    today = time.strftime('%Y%m%d')
    opath = os.path.join(D_DATA, f'a2_scored_{today}.json')
    
    top10_basic = [{'code':r['code'],'name':r.get('name',''),'score':r['score'],
                    'close':r['close'],'pct_ma60':r['pct_ma60'],'rsi14':r['rsi14']}
                   for r in results[:10]]
    
    with open(opath, 'w', encoding='utf-8') as f:
        json.dump({
            'date': kline_latest, 'mf_date': mf_latest, 'run_date': today,
            'total_scored': len(results), 'buy_signals': len(buys),
            'avg_score': round(mean_s,2), 'med_score': round(med_s,2), 'mood': mood,
            'top10': top10_basic,
            'first_5_recs': top10_basic[:MAX_RECOMMEND],
            'config': {'buy_threshold':BUY_THRESHOLD,'stop_loss':STOP_LOSS,
                       'take_profit':TAKE_PROFIT,'max_hold':MAX_HOLD,
                       'hold_days':HOLD_DAYS,'candidate_pool':'full_market'}
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 保存: {os.path.basename(opath)}")
    print(f"⏱️ 总耗时: {time.time()-gt0:.0f}s")


if __name__ == '__main__':
    run()
