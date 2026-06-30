#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股全量数据更新脚本
====================
一条命令更新所有A股数据源（增量模式）。

数据源:
  1. K线数据      → data/cn/a_hist_10y.parquet
  2. 资金流       → data/cn/moneyflow_core.parquet
  3. daily_basic  → data/cn/daily_basic.parquet
  4. 北向资金     → data/cn/north_money.parquet
  5. 涨跌停       → data/cn/limit_list.parquet
  6. 龙虎榜       → data/cn/top_list.parquet
  7. 板块数据     → data/cn/sector_daily.parquet
  8. 股票信息     → data/cn/stock_names.json + data/cn/stock_info.json
  9. 交易日历     → data/cn/trade_cal.parquet

用法:
    python3 cn_data_update_all.py              # 全部更新 (--all)
    python3 cn_data_update_all.py --kline      # 只更新K线
    python3 cn_data_update_all.py --moneyflow  # 只更新资金流
    python3 cn_data_update_all.py --basic      # 只更新daily_basic
    python3 cn_data_update_all.py --north      # 只更新北向资金
    python3 cn_data_update_all.py --limit      # 只更新涨跌停
    python3 cn_data_update_all.py --toplist    # 只更新龙虎榜
    python3 cn_data_update_all.py --sector     # 只更新板块
    python3 cn_data_update_all.py --info       # 只更新股票信息
    python3 cn_data_update_all.py --cal        # 只更新交易日历
    python3 cn_data_update_all.py --status     # 查看状态
"""

import json, os, sys, time, argparse
from datetime import datetime, timedelta
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path("/home/hermes/.hermes/openclaw-archive")
DATA_CN = PROJECT_ROOT / "data" / "cn"
TUSHARE_TOKEN = os.environ.get(
    "TUSHARE_TOKEN",
    "ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db"
)

# Tushare 调用间隔（秒）
API_SLEEP = 0.35
# 重试参数
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # 秒，指数退避基数


# ════════════════════════════════════════════════════════════════
# 通用工具
# ════════════════════════════════════════════════════════════════

def get_tushare():
    """获取tushare pro接口"""
    import tushare as ts
    return ts.pro_api(TUSHARE_TOKEN)


def api_call(fn, **kwargs):
    """
    带重试和指数退避的 tushare API 调用。
    返回 DataFrame 或 None。
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = fn(**kwargs)
            time.sleep(API_SLEEP)
            return df
        except Exception as e:
            err_msg = str(e)
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                print(f"    ⚠️ API调用失败 (第{attempt}次): {err_msg[:80]}")
                print(f"       等待 {wait:.0f}s 后重试...")
                time.sleep(wait)
            else:
                print(f"    ❌ API调用失败 ({MAX_RETRIES}次重试后): {err_msg[:120]}")
                return None
    return None


def get_latest_trade_date(pro):
    """获取最新交易日（从今天往前找）"""
    today = datetime.now().strftime('%Y%m%d')
    df = api_call(pro.trade_cal, start_date=today, end_date=today)
    if df is not None and len(df) > 0 and df.iloc[0]['is_open'] == 1:
        return today
    for d in range(1, 15):
        dt = (datetime.now() - timedelta(days=d)).strftime('%Y%m%d')
        df = api_call(pro.trade_cal, start_date=dt, end_date=dt)
        if df is not None and len(df) > 0 and df.iloc[0]['is_open'] == 1:
            return dt
    return None


def read_parquet_safe(path):
    """安全读取parquet，返回 (DataFrame, True) 或 (None, False)"""
    import pandas as pd
    if not path.exists():
        return None, False
    try:
        df = pd.read_parquet(path)
        return df, True
    except Exception as e:
        print(f"    ⚠️ 读取 {path.name} 失败: {e}")
        return None, False


def get_max_date(df, date_col='trade_date'):
    """从DataFrame获取某日期列的最大值，返回str(YYYYMMDD)"""
    if df is None or len(df) == 0:
        return None
    if date_col not in df.columns:
        return None
    val = df[date_col].max()
    return str(val).replace('-', '').split('T')[0][:8]


def dedup_merge(df_old, df_new, key_cols):
    """按key_cols去重合并：保留旧数据，追加新数据中不重复的行。
    使用向量化操作避免iterrows的性能问题。"""
    if df_old is None or len(df_old) == 0:
        return df_new
    if df_new is None or len(df_new) == 0:
        return df_old
    import pandas as pd

    # 构造key字符串用于高效set查找
    def _make_keys(df):
        parts = []
        for c in key_cols:
            parts.append(df[c].astype(str).str.replace('-', '', regex=False))
        return parts[0].str.cat(parts[1:], sep='|') if len(parts) > 1 else parts[0]

    old_key_set = set(_make_keys(df_old))
    new_keys_series = _make_keys(df_new)
    mask = ~new_keys_series.isin(old_key_set)
    df_new_filtered = df_new[mask].copy()

    if len(df_new_filtered) == 0:
        return df_old
    return pd.concat([df_old, df_new_filtered], ignore_index=True)


# ════════════════════════════════════════════════════════════════
# 1. K线数据更新
# ════════════════════════════════════════════════════════════════

def update_kline(pro, latest_trade):
    """增量更新K线数据 → a_hist_10y.parquet"""
    import pandas as pd

    kpath = DATA_CN / "a_hist_10y.parquet"
    df_old, exists = read_parquet_safe(kpath)

    if exists and df_old is not None and len(df_old) > 0:
        existing_latest = get_max_date(df_old, 'Date')
        if existing_latest and existing_latest >= latest_trade:
            print(f"  ✅ K线已是最新 ({existing_latest})")
            return {"status": "ok", "rows_added": 0, "latest": existing_latest}
        start_date = existing_latest
        print(f"  增量更新: {start_date} → {latest_trade}")
    else:
        print("  ⚠️ 无现有K线数据，需全量拉取（此模式不支持，请先用旧脚本初始化）")
        return {"status": "skip", "rows_added": 0, "reason": "no existing data"}

    t0 = time.time()
    df_new = api_call(pro.daily, start_date=start_date, end_date=latest_trade)

    if df_new is None or len(df_new) == 0:
        print(f"  ✅ 无新K线数据")
        return {"status": "ok", "rows_added": 0, "latest": existing_latest}

    # 转为内部格式
    rows = []
    for _, row in df_new.iterrows():
        code = str(row['ts_code']).split('.')[0]
        d = str(row['trade_date']).replace('-', '')[:8]
        rows.append({
            'Code': code,
            'Date': d,
            'O': float(row.get('open', 0)),
            'H': float(row.get('high', 0)),
            'L': float(row.get('low', 0)),
            'C': float(row.get('close', 0)),
            'V': float(row.get('vol', 0)),
        })
    df_new_formatted = pd.DataFrame(rows)

    # 去重合并
    df_merged = dedup_merge(df_old, df_new_formatted, ['Code', 'Date'])
    df_merged.to_parquet(kpath, index=False)

    rows_added = len(df_merged) - len(df_old)
    elapsed = time.time() - t0
    latest_out = get_max_date(df_merged, 'Date')
    print(f"  💾 K线: +{rows_added}行 → {len(df_merged)}行, 最新: {latest_out}, {elapsed:.1f}s")
    return {"status": "ok", "rows_added": rows_added, "latest": latest_out}


# ════════════════════════════════════════════════════════════════
# 2. 资金流更新
# ════════════════════════════════════════════════════════════════

def update_moneyflow(pro, latest_trade):
    """增量更新资金流 → moneyflow_core.parquet"""
    import pandas as pd

    mfpath = DATA_CN / "moneyflow_core.parquet"
    df_old, exists = read_parquet_safe(mfpath)

    if exists and df_old is not None and len(df_old) > 0:
        existing_latest = get_max_date(df_old, 'trade_date')
        if existing_latest and existing_latest >= latest_trade:
            print(f"  ✅ 资金流已是最新 ({existing_latest})")
            return {"status": "ok", "rows_added": 0, "latest": existing_latest}
        start_date = existing_latest
        print(f"  增量更新: {start_date} → {latest_trade}")
    else:
        print("  ⚠️ 资金流文件不存在，跳过（需先初始化）")
        return {"status": "skip", "rows_added": 0, "reason": "no existing file"}

    t0 = time.time()
    df_new = api_call(pro.moneyflow, start_date=start_date, end_date=latest_trade)

    if df_new is None or len(df_new) == 0:
        print(f"  ✅ 无新资金流数据")
        return {"status": "ok", "rows_added": 0, "latest": existing_latest}

    # 只保留核心列
    mf_cols = ['ts_code', 'trade_date', 'buy_sm_vol', 'buy_sm_amount',
               'sell_sm_vol', 'sell_sm_amount', 'buy_md_vol', 'buy_md_amount',
               'sell_md_vol', 'sell_md_amount', 'buy_lg_vol', 'buy_lg_amount',
               'sell_lg_vol', 'sell_lg_amount', 'buy_elg_vol', 'buy_elg_amount',
               'sell_elg_vol', 'sell_elg_amount', 'net_mf_vol', 'net_mf_amount']
    available = [c for c in mf_cols if c in df_new.columns]
    df_new_filtered = df_new[available].copy()

    df_merged = dedup_merge(df_old, df_new_filtered, ['ts_code', 'trade_date'])
    df_merged.to_parquet(mfpath, index=False)

    rows_added = len(df_merged) - len(df_old)
    elapsed = time.time() - t0
    latest_out = get_max_date(df_merged, 'trade_date')
    print(f"  💾 资金流: +{rows_added}行 → {len(df_merged)}行, 最新: {latest_out}, {elapsed:.1f}s")
    return {"status": "ok", "rows_added": rows_added, "latest": latest_out}


# ════════════════════════════════════════════════════════════════
# 3. daily_basic 更新
# ════════════════════════════════════════════════════════════════

def update_daily_basic(pro, latest_trade):
    """增量更新daily_basic (PE/PB/换手率/市值) → daily_basic.parquet"""
    import pandas as pd

    dbpath = DATA_CN / "daily_basic.parquet"
    df_old, exists = read_parquet_safe(dbpath)

    if exists and df_old is not None and len(df_old) > 0:
        existing_latest = get_max_date(df_old, 'trade_date')
        if existing_latest and existing_latest >= latest_trade:
            print(f"  ✅ daily_basic已是最新 ({existing_latest})")
            return {"status": "ok", "rows_added": 0, "latest": existing_latest}
        start_date = existing_latest
        print(f"  增量更新: {start_date} → {latest_trade}")
    else:
        print("  ⚠️ daily_basic文件不存在，跳过（需先初始化）")
        return {"status": "skip", "rows_added": 0, "reason": "no existing file"}

    t0 = time.time()

    # 逐日拉取（tushare daily_basic限流严格）
    dates_to_pull = []
    if start_date:
        current = datetime.strptime(start_date, '%Y%m%d') + timedelta(days=1)
    else:
        current = datetime.now() - timedelta(days=30)  # 无历史数据时拉30天
    latest_dt = datetime.strptime(latest_trade, '%Y%m%d')
    while current <= latest_dt:
        dates_to_pull.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)

    if not dates_to_pull:
        print(f"  ✅ 无需更新")
        return {"status": "ok", "rows_added": 0, "latest": existing_latest}

    new_rows = []
    for dt in dates_to_pull:
        df_day = api_call(
            pro.daily_basic, trade_date=dt,
            fields='ts_code,trade_date,pe_ttm,pb,ps_ttm,dv_ratio,total_mv,circ_mv,turnover_rate'
        )
        if df_day is not None and len(df_day) > 0:
            new_rows.append(df_day)
            print(f"    {dt}: {len(df_day)}条")

    if not new_rows:
        print(f"  ✅ 无新数据")
        return {"status": "ok", "rows_added": 0, "latest": existing_latest}

    df_new = pd.concat(new_rows, ignore_index=True)
    df_merged = dedup_merge(df_old, df_new, ['ts_code', 'trade_date'])
    df_merged.to_parquet(dbpath, index=False)

    rows_added = len(df_merged) - len(df_old)
    elapsed = time.time() - t0
    latest_out = get_max_date(df_merged, 'trade_date')
    print(f"  💾 daily_basic: +{rows_added}行 → {len(df_merged)}行, 最新: {latest_out}, {elapsed:.1f}s")
    return {"status": "ok", "rows_added": rows_added, "latest": latest_out}


# ════════════════════════════════════════════════════════════════
# 4. 北向资金更新
# ════════════════════════════════════════════════════════════════

def update_north_money(pro, latest_trade):
    """增量更新北向资金 → north_money.parquet
    使用 pro.moneyflow_hsgt() 获取沪深港通资金流向
    """
    import pandas as pd

    nm_path = DATA_CN / "north_money.parquet"
    df_old, exists = read_parquet_safe(nm_path)

    if exists and df_old is not None and len(df_old) > 0:
        existing_latest = get_max_date(df_old, 'trade_date')
        if existing_latest and existing_latest >= latest_trade:
            print(f"  ✅ 北向资金已是最新 ({existing_latest})")
            return {"status": "ok", "rows_added": 0, "latest": existing_latest}
        start_date = existing_latest
        print(f"  增量更新: {start_date} → {latest_trade}")
    else:
        # 首次拉取：拉最近3年
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y%m%d')
        print(f"  全量拉取: {start_date} → {latest_trade}")

    t0 = time.time()

    # moneyflow_hsgt 按月拉取（API限制单次最大约1000条）
    all_rows = []
    current = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(latest_trade, '%Y%m%d')

    while current <= end:
        # 每次拉一个月
        month_start = current.strftime('%Y%m%d')
        next_month = (current + timedelta(days=32)).replace(day=1)
        month_end = min(next_month - timedelta(days=1), end).strftime('%Y%m%d')

        df_chunk = api_call(
            pro.moneyflow_hsgt,
            start_date=month_start,
            end_date=month_end,
            fields='trade_date,ggt_ss,ggt_sz,hgt,sgt,north_money,south_money'
        )
        if df_chunk is not None and len(df_chunk) > 0:
            all_rows.append(df_chunk)
            print(f"    {month_start}~{month_end}: {len(df_chunk)}条")

        current = next_month

    if not all_rows:
        print(f"  ✅ 无新北向资金数据")
        return {"status": "ok", "rows_added": 0, "latest": get_max_date(df_old, 'trade_date') if df_old is not None else None}

    df_new = pd.concat(all_rows, ignore_index=True)
    df_new['trade_date'] = df_new['trade_date'].astype(str).str.replace('-', '').str[:8]

    if df_old is not None and len(df_old) > 0:
        df_merged = dedup_merge(df_old, df_new, ['trade_date'])
    else:
        df_merged = df_new

    df_merged.to_parquet(nm_path, index=False)

    rows_added = len(df_merged) - (len(df_old) if df_old is not None else 0)
    elapsed = time.time() - t0
    latest_out = get_max_date(df_merged, 'trade_date')
    print(f"  💾 北向资金: +{rows_added}行 → {len(df_merged)}行, 最新: {latest_out}, {elapsed:.1f}s")
    return {"status": "ok", "rows_added": rows_added, "latest": latest_out}


# ════════════════════════════════════════════════════════════════
# 5. 涨跌停更新
# ════════════════════════════════════════════════════════════════

def update_limit_list(pro, latest_trade):
    """增量更新涨跌停 → limit_list.parquet
    使用 pro.limit_list_d()
    """
    import pandas as pd

    ll_path = DATA_CN / "limit_list.parquet"
    df_old, exists = read_parquet_safe(ll_path)

    if exists and df_old is not None and len(df_old) > 0:
        existing_latest = get_max_date(df_old, 'trade_date')
        if existing_latest and existing_latest >= latest_trade:
            print(f"  ✅ 涨跌停已是最新 ({existing_latest})")
            return {"status": "ok", "rows_added": 0, "latest": existing_latest}
        start_date = existing_latest
        print(f"  增量更新: {start_date} → {latest_trade}")
    else:
        # 首次拉取：拉最近1年
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        print(f"  全量拉取: {start_date} → {latest_trade}")

    t0 = time.time()

    # limit_list_d 按日拉取
    dates_to_pull = []
    current = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(latest_trade, '%Y%m%d')
    while current <= end:
        dates_to_pull.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)

    new_rows = []
    for dt in dates_to_pull:
        df_day = api_call(
            pro.limit_list_d, trade_date=dt,
            fields='trade_date,ts_code,industry,name,close,pct_chg,amount,limit_amount,float_mv,total_mv,turnover_ratio,fd_amount,first_time,last_time,open_times,up_stat,limit_times,limit'
        )
        if df_day is not None and len(df_day) > 0:
            new_rows.append(df_day)
            print(f"    {dt}: {len(df_day)}条")

    if not new_rows:
        print(f"  ✅ 无新涨跌停数据")
        return {"status": "ok", "rows_added": 0, "latest": get_max_date(df_old, 'trade_date') if df_old is not None else None}

    df_new = pd.concat(new_rows, ignore_index=True)
    df_new['trade_date'] = df_new['trade_date'].astype(str).str.replace('-', '').str[:8]

    if df_old is not None and len(df_old) > 0:
        df_merged = dedup_merge(df_old, df_new, ['trade_date', 'ts_code'])
    else:
        df_merged = df_new

    df_merged.to_parquet(ll_path, index=False)

    rows_added = len(df_merged) - (len(df_old) if df_old is not None else 0)
    elapsed = time.time() - t0
    latest_out = get_max_date(df_merged, 'trade_date')
    print(f"  💾 涨跌停: +{rows_added}行 → {len(df_merged)}行, 最新: {latest_out}, {elapsed:.1f}s")
    return {"status": "ok", "rows_added": rows_added, "latest": latest_out}


# ════════════════════════════════════════════════════════════════
# 6. 龙虎榜更新
# ════════════════════════════════════════════════════════════════

def update_top_list(pro, latest_trade):
    """增量更新龙虎榜 → top_list.parquet
    使用 pro.top_list() + pro.top_inst()
    """
    import pandas as pd

    tl_path = DATA_CN / "top_list.parquet"
    df_old, exists = read_parquet_safe(tl_path)

    if exists and df_old is not None and len(df_old) > 0:
        existing_latest = get_max_date(df_old, 'trade_date')
        if existing_latest and existing_latest >= latest_trade:
            print(f"  ✅ 龙虎榜已是最新 ({existing_latest})")
            return {"status": "ok", "rows_added": 0, "latest": existing_latest}
        start_date = existing_latest
        print(f"  增量更新: {start_date} → {latest_trade}")
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        print(f"  全量拉取: {start_date} → {latest_trade}")

    t0 = time.time()

    # top_list 按日拉取
    dates_to_pull = []
    current = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(latest_trade, '%Y%m%d')
    while current <= end:
        dates_to_pull.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)

    new_rows = []
    for dt in dates_to_pull:
        # 拉取龙虎榜个股明细
        df_day = api_call(
            pro.top_list, trade_date=dt,
            fields='trade_date,ts_code,name,close,pct_change,turnover_rate,amount,l_sell,l_buy,l_amount,net_amount,net_rate,amount_rate,reason'
        )
        if df_day is not None and len(df_day) > 0:
            # 同时拉取龙虎榜机构明细
            df_inst = api_call(
                pro.top_inst, trade_date=dt,
                fields='trade_date,ts_code,exalter,buy,buy_rate,sell,sell_rate,net_buy'
            )
            if df_inst is not None and len(df_inst) > 0:
                # 合并：每只股票的机构买卖汇总
                inst_summary = df_inst.groupby('ts_code').agg({
                    'buy': 'sum', 'sell': 'sum', 'net_buy': 'sum',
                    'buy_rate': 'sum', 'sell_rate': 'sum'
                }).reset_index()
                inst_summary.columns = ['ts_code', 'inst_buy', 'inst_sell', 'inst_net',
                                        'inst_buy_rate', 'inst_sell_rate']
                df_day = df_day.merge(inst_summary, on='ts_code', how='left')
            else:
                for c in ['inst_buy', 'inst_sell', 'inst_net', 'inst_buy_rate', 'inst_sell_rate']:
                    df_day[c] = 0.0

            new_rows.append(df_day)
            print(f"    {dt}: {len(df_day)}条 (含机构明细)")

    if not new_rows:
        print(f"  ✅ 无新龙虎榜数据")
        return {"status": "ok", "rows_added": 0, "latest": get_max_date(df_old, 'trade_date') if df_old is not None else None}

    df_new = pd.concat(new_rows, ignore_index=True)
    df_new['trade_date'] = df_new['trade_date'].astype(str).str.replace('-', '').str[:8]

    if df_old is not None and len(df_old) > 0:
        df_merged = dedup_merge(df_old, df_new, ['trade_date', 'ts_code'])
    else:
        df_merged = df_new

    df_merged.to_parquet(tl_path, index=False)

    rows_added = len(df_merged) - (len(df_old) if df_old is not None else 0)
    elapsed = time.time() - t0
    latest_out = get_max_date(df_merged, 'trade_date')
    print(f"  💾 龙虎榜: +{rows_added}行 → {len(df_merged)}行, 最新: {latest_out}, {elapsed:.1f}s")
    return {"status": "ok", "rows_added": rows_added, "latest": latest_out}


# ════════════════════════════════════════════════════════════════
# 7. 板块数据更新
# ════════════════════════════════════════════════════════════════

def update_sector_daily(pro, latest_trade):
    """增量更新板块指数日线 → sector_daily.parquet
    拉取主要指数的日线数据（上证指数、深证成指、创业板指、沪深300等）
    使用 pro.index_daily()
    """
    import pandas as pd

    sd_path = DATA_CN / "sector_daily.parquet"
    df_old, exists = read_parquet_safe(sd_path)

    # 主要指数代码
    INDEX_CODES = [
        '000001.SH',  # 上证指数
        '399001.SZ',  # 深证成指
        '399006.SZ',  # 创业板指
        '000300.SH',  # 沪深300
        '000905.SH',  # 中证500
        '000852.SH',  # 中证1000
        '399005.SZ',  # 中小100
        '000688.SH',  # 科创50
    ]

    if exists and df_old is not None and len(df_old) > 0:
        existing_latest = get_max_date(df_old, 'trade_date')
        if existing_latest and existing_latest >= latest_trade:
            print(f"  ✅ 板块数据已是最新 ({existing_latest})")
            return {"status": "ok", "rows_added": 0, "latest": existing_latest}
        start_date = existing_latest
        print(f"  增量更新: {start_date} → {latest_trade}")
    else:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y%m%d')
        print(f"  全量拉取: {start_date} → {latest_trade}")

    t0 = time.time()

    new_rows = []
    for idx_code in INDEX_CODES:
        df_chunk = api_call(
            pro.index_daily, ts_code=idx_code,
            start_date=start_date, end_date=latest_trade,
            fields='ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount'
        )
        if df_chunk is not None and len(df_chunk) > 0:
            new_rows.append(df_chunk)
            print(f"    {idx_code}: {len(df_chunk)}条")

    # 补充：主要行业板块ETF
    SECTOR_ETFS = [
        '510050.SH', '510300.SH', '510500.SH', '512010.SH',
        '512880.SH', '515030.SH', '516160.SH', '159915.SZ',
    ]
    for etf_code in SECTOR_ETFS:
        df_chunk = api_call(
            pro.fund_daily, ts_code=etf_code,
            start_date=start_date, end_date=latest_trade,
            fields='ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount'
        )
        if df_chunk is not None and len(df_chunk) > 0:
            df_chunk['type'] = 'etf'
            new_rows.append(df_chunk)
            print(f"    {etf_code}: {len(df_chunk)}条")

    if not new_rows:
        print(f"  ✅ 无新板块数据")
        return {"status": "ok", "rows_added": 0, "latest": get_max_date(df_old, 'trade_date') if df_old is not None else None}

    df_new = pd.concat(new_rows, ignore_index=True)
    df_new['trade_date'] = df_new['trade_date'].astype(str).str.replace('-', '').str[:8]

    if df_old is not None and len(df_old) > 0:
        df_merged = dedup_merge(df_old, df_new, ['ts_code', 'trade_date'])
    else:
        df_merged = df_new

    df_merged.to_parquet(sd_path, index=False)

    rows_added = len(df_merged) - (len(df_old) if df_old is not None else 0)
    elapsed = time.time() - t0
    latest_out = get_max_date(df_merged, 'trade_date')
    print(f"  💾 板块数据: +{rows_added}行 → {len(df_merged)}行, 最新: {latest_out}, {elapsed:.1f}s")
    return {"status": "ok", "rows_added": rows_added, "latest": latest_out}


# ════════════════════════════════════════════════════════════════
# 8. 股票信息更新
# ════════════════════════════════════════════════════════════════

def update_stock_info(pro):
    """更新股票名称和行业信息 → stock_names.json + stock_info.json"""
    t0 = time.time()

    df = api_call(
        pro.stock_basic, exchange='', list_status='L',
        fields='ts_code,name,industry,market,list_date,fullname,enname,cnspell,area'
    )

    if df is None or len(df) == 0:
        print(f"  ⚠️ 无股票信息数据")
        return {"status": "error", "rows_added": 0}

    # stock_names.json
    names = {}
    industries = {}
    for _, row in df.iterrows():
        code = str(row['ts_code']).split('.')[0]
        names[code] = row['name']
        industries[code] = row.get('industry', '')

    names_path = DATA_CN / "stock_names.json"
    with open(names_path, 'w', encoding='utf-8') as f:
        json.dump({'names': names, 'industries': industries}, f, ensure_ascii=False)

    # stock_info.json
    info = {}
    for _, row in df.iterrows():
        code = str(row['ts_code']).split('.')[0]
        info[code] = {
            'name': row['name'],
            'industry': row.get('industry', ''),
            'market': row.get('market', ''),
            'list_date': str(row.get('list_date', '')),
            'fullname': row.get('fullname', ''),
            'area': row.get('area', ''),
        }

    info_path = DATA_CN / "stock_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"  💾 stock_names: {len(names)}只, stock_info: {len(info)}只, {elapsed:.1f}s")
    return {"status": "ok", "rows_added": len(names)}


# ════════════════════════════════════════════════════════════════
# 9. 交易日历更新
# ════════════════════════════════════════════════════════════════

def update_trade_cal(pro):
    """更新交易日历 → trade_cal.parquet"""
    import pandas as pd

    tc_path = DATA_CN / "trade_cal.parquet"
    df_old, exists = read_parquet_safe(tc_path)

    # 拉取最近5年日历（覆盖大部分使用场景）
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y%m%d')
    end_date = (datetime.now() + timedelta(days=365)).strftime('%Y%m%d')

    t0 = time.time()
    df_new = api_call(
        pro.trade_cal, start_date=start_date, end_date=end_date,
        fields='exchange,cal_date,is_open,pretrade_date'
    )

    if df_new is None or len(df_new) == 0:
        print(f"  ⚠️ 无交易日历数据")
        return {"status": "error", "rows_added": 0}

    df_new['cal_date'] = df_new['cal_date'].astype(str).str.replace('-', '').str[:8]
    if 'pretrade_date' in df_new.columns:
        df_new['pretrade_date'] = df_new['pretrade_date'].astype(str).str.replace('-', '').str[:8]

    df_new.to_parquet(tc_path, index=False)

    elapsed = time.time() - t0
    print(f"  💾 交易日历: {len(df_new)}行, 范围: {df_new['cal_date'].min()}~{df_new['cal_date'].max()}, {elapsed:.1f}s")
    return {"status": "ok", "rows_added": len(df_new)}


# ════════════════════════════════════════════════════════════════
# 状态检查
# ════════════════════════════════════════════════════════════════

def check_status():
    """检查所有A股数据文件状态"""
    import pandas as pd

    print("📊 A股数据文件状态")
    print("=" * 60)

    files = [
        ("K线 (a_hist_10y)", "a_hist_10y.parquet", "Date"),
        ("资金流 (moneyflow_core)", "moneyflow_core.parquet", "trade_date"),
        ("daily_basic (PE/PB/换手)", "daily_basic.parquet", "trade_date"),
        ("北向资金 (north_money)", "north_money.parquet", "trade_date"),
        ("涨跌停 (limit_list)", "limit_list.parquet", "trade_date"),
        ("龙虎榜 (top_list)", "top_list.parquet", "trade_date"),
        ("板块数据 (sector_daily)", "sector_daily.parquet", "trade_date"),
        ("交易日历 (trade_cal)", "trade_cal.parquet", "cal_date"),
    ]

    for label, fname, date_col in files:
        fpath = DATA_CN / fname
        if not fpath.exists():
            print(f"  ❌ {label}: 不存在")
            continue

        mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        size_mb = fpath.stat().st_size / 1024 / 1024

        latest = "?"
        rows = 0
        try:
            df = pd.read_parquet(fpath)
            rows = len(df)
            if date_col in df.columns:
                latest = str(df[date_col].max()).replace('-', '')[:8]
        except Exception as e:
            latest = f"读取失败: {str(e)[:50]}"

        status = "✅" if age_hours < 48 else "⚠️" if age_hours < 120 else "❌"
        print(f"  {status} {label}: {latest} ({rows:,}行, {age_hours/24:.1f}天前, {size_mb:.0f}MB)")

    # JSON文件
    for label, fname in [("stock_names", "stock_names.json"), ("stock_info", "stock_info.json")]:
        fpath = DATA_CN / fname
        if not fpath.exists():
            print(f"  ❌ {label}: 不存在")
            continue
        try:
            with open(fpath) as f:
                d = json.load(f)
            count = len(d) if isinstance(d, dict) else 0
            if 'names' in d:
                count = len(d['names'])
            print(f"  ✅ {label}: {count}只")
        except:
            print(f"  ⚠️ {label}: 读取失败")

    print()


# ════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="A股全量数据更新")
    parser.add_argument("--kline", action="store_true", help="只更新K线")
    parser.add_argument("--moneyflow", action="store_true", help="只更新资金流")
    parser.add_argument("--basic", action="store_true", help="只更新daily_basic")
    parser.add_argument("--north", action="store_true", help="只更新北向资金")
    parser.add_argument("--limit", action="store_true", help="只更新涨跌停")
    parser.add_argument("--toplist", action="store_true", help="只更新龙虎榜")
    parser.add_argument("--sector", action="store_true", help="只更新板块数据")
    parser.add_argument("--info", action="store_true", help="只更新股票信息")
    parser.add_argument("--cal", action="store_true", help="只更新交易日历")
    parser.add_argument("--all", action="store_true", help="全部更新（默认）")
    parser.add_argument("--status", action="store_true", help="查看状态")
    args = parser.parse_args()

    print(f"🔄 A股全量数据更新 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    if args.status:
        check_status()
        return

    # 获取tushare
    pro = get_tushare()
    latest_trade = get_latest_trade_date(pro)
    if not latest_trade:
        print("❌ 无法获取最新交易日，请检查网络或token")
        return
    print(f"📅 最新交易日: {latest_trade}\n")

    # 判断运行哪些模块
    run_all = args.all or not any([
        args.kline, args.moneyflow, args.basic, args.north,
        args.limit, args.toplist, args.sector, args.info, args.cal
    ])

    t_total = time.time()
    results = {}

    # 交易日历优先更新（其他模块可能依赖）
    if args.cal or run_all:
        print("📡 [9/9] 交易日历:")
        try:
            results['cal'] = update_trade_cal(pro)
        except Exception as e:
            print(f"  ❌ 交易日历更新失败: {e}")
            results['cal'] = {"status": "error", "rows_added": 0}
        print()

    # 股票信息（优先，其他模块可能需要）
    if args.info or run_all:
        print("📡 [8/9] 股票信息:")
        try:
            results['info'] = update_stock_info(pro)
        except Exception as e:
            print(f"  ❌ 股票信息更新失败: {e}")
            results['info'] = {"status": "error", "rows_added": 0}
        print()

    if args.kline or run_all:
        print("📡 [1/9] K线数据:")
        try:
            results['kline'] = update_kline(pro, latest_trade)
        except Exception as e:
            print(f"  ❌ K线更新失败: {e}")
            results['kline'] = {"status": "error", "rows_added": 0}
        print()

    if args.moneyflow or run_all:
        print("📡 [2/9] 资金流:")
        try:
            results['moneyflow'] = update_moneyflow(pro, latest_trade)
        except Exception as e:
            print(f"  ❌ 资金流更新失败: {e}")
            results['moneyflow'] = {"status": "error", "rows_added": 0}
        print()

    if args.basic or run_all:
        print("📡 [3/9] daily_basic:")
        try:
            results['basic'] = update_daily_basic(pro, latest_trade)
        except Exception as e:
            print(f"  ❌ daily_basic更新失败: {e}")
            results['basic'] = {"status": "error", "rows_added": 0}
        print()

    if args.north or run_all:
        print("📡 [4/9] 北向资金:")
        try:
            results['north'] = update_north_money(pro, latest_trade)
        except Exception as e:
            print(f"  ❌ 北向资金更新失败: {e}")
            results['north'] = {"status": "error", "rows_added": 0}
        print()

    if args.limit or run_all:
        print("📡 [5/9] 涨跌停:")
        try:
            results['limit'] = update_limit_list(pro, latest_trade)
        except Exception as e:
            print(f"  ❌ 涨跌停更新失败: {e}")
            results['limit'] = {"status": "error", "rows_added": 0}
        print()

    if args.toplist or run_all:
        print("📡 [6/9] 龙虎榜:")
        try:
            results['toplist'] = update_top_list(pro, latest_trade)
        except Exception as e:
            print(f"  ❌ 龙虎榜更新失败: {e}")
            results['toplist'] = {"status": "error", "rows_added": 0}
        print()

    if args.sector or run_all:
        print("📡 [7/9] 板块数据:")
        try:
            results['sector'] = update_sector_daily(pro, latest_trade)
        except Exception as e:
            print(f"  ❌ 板块数据更新失败: {e}")
            results['sector'] = {"status": "error", "rows_added": 0}
        print()

    # 汇总
    total_elapsed = time.time() - t_total
    total_rows = sum(r.get('rows_added', 0) for r in results.values())
    success = sum(1 for r in results.values() if r.get('status') == 'ok')
    errors = sum(1 for r in results.values() if r.get('status') == 'error')
    skipped = sum(1 for r in results.values() if r.get('status') == 'skip')

    print("=" * 60)
    print(f"✅ 更新完成")
    print(f"   成功: {success}  跳过: {skipped}  失败: {errors}")
    print(f"   总新增行数: {total_rows:,}")
    print(f"   总耗时: {total_elapsed:.1f}s")
    print()

    # 各源详情
    print("📋 各数据源详情:")
    name_map = {
        'kline': 'K线', 'moneyflow': '资金流', 'basic': 'daily_basic',
        'north': '北向资金', 'limit': '涨跌停', 'toplist': '龙虎榜',
        'sector': '板块', 'info': '股票信息', 'cal': '交易日历',
    }
    for key, name in name_map.items():
        if key in results:
            r = results[key]
            status_icon = {"ok": "✅", "error": "❌", "skip": "⏭️"}.get(r.get('status', ''), "❓")
            extra = f" → {r.get('latest', '?')}" if r.get('latest') else ""
            reason = f" ({r.get('reason', '')})" if r.get('reason') else ""
            print(f"  {status_icon} {name}: +{r.get('rows_added', 0):,}行{extra}{reason}")

    print()
    check_status()


if __name__ == "__main__":
    main()
