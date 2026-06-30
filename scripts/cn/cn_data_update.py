#!/usr/bin/env python3
"""
A股数据更新管线
================
一条命令完成：K线增量 + 资金流增量 + daily_basic增量 + K线JSON→parquet转换

数据源: tushare pro
用法:
    python3 cn_data_update.py              # 全部更新
    python3 cn_data_update.py --kline      # 只更新K线
    python3 cn_data_update.py --moneyflow  # 只更新资金流
    python3 cn_data_update.py --basic      # 只更新daily_basic
    python3 cn_data_update.py --convert    # 只做JSON→parquet转换
    python3 cn_data_update.py --status     # 只看状态
"""

import json, os, sys, time, argparse
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
DATA_CN = WORKSPACE / "data" / "cn"
TUSHARE_TOKEN=os.environ.get("TUSHARE_TOKEN", "ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db")


def get_tushare():
    """获取tushare pro接口"""
    import tushare as ts
    return ts.pro_api(TUSHARE_TOKEN)


def get_latest_trade_date(pro):
    """获取最新交易日"""
    today = datetime.now().strftime('%Y%m%d')
    df = pro.trade_cal(start_date=today, end_date=today)
    if len(df) > 0 and df.iloc[0]['is_open'] == 1:
        return today
    for d in range(1, 15):
        dt = (datetime.now() - timedelta(days=d)).strftime('%Y%m%d')
        df = pro.trade_cal(start_date=dt, end_date=dt)
        if len(df) > 0 and df.iloc[0]['is_open'] == 1:
            return dt
    return None


# ════════════════════════════════════════════════════════════════
# 1. K线更新
# ════════════════════════════════════════════════════════════════

def update_kline(pro, latest_trade):
    """增量更新K线数据"""
    import pandas as pd
    
    kpath = DATA_CN / "a_hist_10y.parquet"
    
    # 加载现有数据 (可能是JSON格式)
    hist = {}
    if kpath.exists():
        try:
            # 先尝试parquet
            df = pd.read_parquet(kpath)
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
            print(f"  加载parquet格式: {len(hist)}只")
        except:
            # JSON格式
            with open(kpath, 'r') as f:
                hist = json.load(f)
            print(f"  加载JSON格式: {len(hist)}只")
    
    if not hist:
        print("  ❌ 无现有K线数据，需要全量拉取")
        return False
    
    # 检查现有数据最新日期
    sample_code = list(hist.keys())[0]
    existing_dates = hist[sample_code].get('dates', [])
    existing_latest = str(existing_dates[-1]) if existing_dates else '20160101'
    
    if existing_latest >= latest_trade:
        print(f"  ✅ K线已是最新 ({existing_latest})")
        return True
    
    print(f"  增量更新: {existing_latest} → {latest_trade}")
    
    # 批量拉取
    t0 = time.time()
    try:
        df_new = pro.daily(start_date=existing_latest, end_date=latest_trade)
    except Exception as e:
        print(f"  ⚠️ 批量拉取失败: {e}")
        return False
    
    if len(df_new) == 0:
        print(f"  ⚠️ 无新数据")
        return True
    
    # 按股票分组追加
    grouped = df_new.groupby('ts_code')
    updated = 0
    for ts_code, grp in grouped:
        code = ts_code.split('.')[0]
        if code not in hist:
            hist[code] = {'c': [], 'h': [], 'l': [], 'o': [], 'v': [], 'dates': []}
        
        rec = hist[code]
        existing_date_set = set(str(d) for d in rec.get('dates', []))
        
        for _, row in grp.sort_values('trade_date').iterrows():
            d = str(row['trade_date']).replace('-', '')
            if d not in existing_date_set:
                rec['dates'].append(d)
                rec['o'].append(float(row['open']))
                rec['h'].append(float(row['high']))
                rec['l'].append(float(row['low']))
                rec['c'].append(float(row['close']))
                rec['v'].append(float(row.get('vol', 0)))
                existing_date_set.add(d)
        updated += 1
    
    # 保存为真parquet
    _save_kline_as_parquet(hist)
    
    elapsed = time.time() - t0
    print(f"  更新了{updated}只股票, 耗时{elapsed:.1f}s")
    
    return True


def _save_kline_as_parquet(hist):
    """将K线dict保存为真parquet格式"""
    import pandas as pd
    
    rows = []
    for code, rec in hist.items():
        dates = rec.get('dates', [])
        c = rec.get('c', [])
        h = rec.get('h', [])
        l = rec.get('l', [])
        o = rec.get('o', [])
        v = rec.get('v', [])
        
        for i in range(len(dates)):
            rows.append({
                'Code': code,
                'Date': str(dates[i]),
                'O': o[i] if i < len(o) else 0,
                'H': h[i] if i < len(h) else 0,
                'L': l[i] if i < len(l) else 0,
                'C': c[i] if i < len(c) else 0,
                'V': v[i] if i < len(v) else 0,
            })
    
    df = pd.DataFrame(rows)
    kpath = DATA_CN / "a_hist_10y.parquet"
    df.to_parquet(kpath, index=False)
    print(f"  💾 K线已保存为parquet: {len(df)}行, {df['Code'].nunique()}只")


# ════════════════════════════════════════════════════════════════
# 2. 资金流更新
# ════════════════════════════════════════════════════════════════

def update_moneyflow(pro, latest_trade):
    """增量更新资金流数据"""
    import pandas as pd
    
    mfpath = DATA_CN / "moneyflow_core.parquet"
    
    if not mfpath.exists():
        print("  ❌ moneyflow_core.parquet不存在")
        return False
    
    df_mf = pd.read_parquet(mfpath)
    existing_latest = str(df_mf['trade_date'].max())
    
    if existing_latest >= latest_trade:
        print(f"  ✅ 资金流已是最新 ({existing_latest})")
        return True
    
    print(f"  增量更新: {existing_latest} → {latest_trade}")
    
    t0 = time.time()
    try:
        df_new = pro.moneyflow(start_date=existing_latest, end_date=latest_trade)
    except Exception as e:
        print(f"  ⚠️ 拉取失败: {e}")
        return False
    
    if len(df_new) == 0:
        print(f"  ⚠️ 无新数据")
        return True
    
    # 去重合并
    mf_cols = ['ts_code', 'trade_date', 'buy_sm_vol', 'buy_sm_amount', 
               'sell_sm_vol', 'sell_sm_amount', 'buy_md_vol', 'buy_md_amount', 
               'sell_md_vol', 'sell_md_amount', 'buy_lg_vol', 'buy_lg_amount', 
               'sell_lg_vol', 'sell_lg_amount', 'buy_elg_vol', 'buy_elg_amount', 
               'sell_elg_vol', 'sell_elg_amount', 'net_mf_vol', 'net_mf_amount']
    
    available_cols = [c for c in mf_cols if c in df_new.columns]
    df_new_filtered = df_new[available_cols].copy()
    
    old_keys = set(zip(df_mf['ts_code'], df_mf['trade_date'].astype(str)))
    new_keys = set(zip(df_new_filtered['ts_code'], df_new_filtered['trade_date'].astype(str)))
    new_only = new_keys - old_keys
    
    if len(new_only) == 0:
        print(f"  ✅ 已包含所有新数据")
        return True
    
    df_new_filtered = df_new_filtered[
        df_new_filtered.apply(lambda r: (r['ts_code'], str(r['trade_date'])) not in old_keys, axis=1)
    ]
    
    df_mf = pd.concat([df_mf, df_new_filtered], ignore_index=True)
    df_mf.to_parquet(mfpath, index=False)
    
    elapsed = time.time() - t0
    print(f"  新增{len(df_new_filtered)}行, 最新: {df_mf['trade_date'].max()}, 耗时{elapsed:.1f}s")
    print(f"  💾 已保存")
    
    return True


# ════════════════════════════════════════════════════════════════
# 3. daily_basic更新
# ════════════════════════════════════════════════════════════════

def update_daily_basic(pro, latest_trade):
    """增量更新daily_basic (PE/PB/换手率/市值)"""
    import pandas as pd
    
    dbpath = DATA_CN / "daily_basic.parquet"
    
    if not dbpath.exists():
        print("  ❌ daily_basic.parquet不存在")
        return False
    
    df_db = pd.read_parquet(dbpath)
    existing_latest = str(df_db['trade_date'].max())
    
    if existing_latest >= latest_trade:
        print(f"  ✅ daily_basic已是最新 ({existing_latest})")
        return True
    
    print(f"  增量更新: {existing_latest} → {latest_trade}")
    
    t0 = time.time()
    
    # tushare daily_basic有频率限制，分批拉取
    dates_to_pull = []
    current = datetime.strptime(existing_latest, '%Y%m%d') + timedelta(days=1)
    latest_dt = datetime.strptime(latest_trade, '%Y%m%d')
    
    while current <= latest_dt:
        # 检查是否交易日
        dates_to_pull.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    
    if not dates_to_pull:
        print(f"  无需更新")
        return True
    
    # 逐日拉取（避免频率限制）
    new_rows = []
    for dt in dates_to_pull:
        try:
            df_day = pro.daily_basic(trade_date=dt, 
                                     fields='ts_code,trade_date,pe_ttm,pb,ps_ttm,dv_ratio,total_mv,circ_mv,turnover_rate')
            if len(df_day) > 0:
                new_rows.append(df_day)
                print(f"    {dt}: {len(df_day)}条")
            time.sleep(0.3)  # 频率限制
        except Exception as e:
            print(f"    {dt}: ⚠️ {e}")
            time.sleep(1)
    
    if not new_rows:
        print(f"  无新数据")
        return True
    
    df_new = pd.concat(new_rows, ignore_index=True)
    
    # 去重合并
    old_keys = set(zip(df_db['ts_code'], df_db['trade_date'].astype(str)))
    df_new_filtered = df_new[
        df_new.apply(lambda r: (r['ts_code'], str(r['trade_date'])) not in old_keys, axis=1)
    ]
    
    if len(df_new_filtered) == 0:
        print(f"  ✅ 已包含所有新数据")
        return True
    
    df_db = pd.concat([df_db, df_new_filtered], ignore_index=True)
    df_db.to_parquet(dbpath, index=False)
    
    elapsed = time.time() - t0
    print(f"  新增{len(df_new_filtered)}行, 最新: {df_db['trade_date'].max()}, 耗时{elapsed:.1f}s")
    print(f"  💾 已保存")
    
    return True


# ════════════════════════════════════════════════════════════════
# 4. stock_names更新
# ════════════════════════════════════════════════════════════════

def update_stock_names(pro):
    """更新股票名称和行业信息"""
    import pandas as pd
    
    names_path = DATA_CN / "stock_names.json"
    info_path = WORKSPACE / "data" / "stock_info.json"
    
    try:
        df = pro.stock_basic(exchange='', list_status='L', 
                            fields='ts_code,name,industry,market,list_date')
    except Exception as e:
        print(f"  ⚠️ 拉取失败: {e}")
        return False
    
    if len(df) == 0:
        print(f"  ⚠️ 无数据")
        return False
    
    # stock_names.json格式
    names = {}
    industries = {}
    for _, row in df.iterrows():
        code = row['ts_code'].split('.')[0]
        names[code] = row['name']
        industries[code] = row.get('industry', '')
    
    names_data = {'names': names, 'industries': industries}
    with open(names_path, 'w', encoding='utf-8') as f:
        json.dump(names_data, f, ensure_ascii=False)
    
    # stock_info.json格式
    info = {}
    for _, row in df.iterrows():
        code = row['ts_code'].split('.')[0]
        info[code] = {
            'name': row['name'],
            'industry': row.get('industry', ''),
            'market': row.get('market', ''),
            'list_date': str(row.get('list_date', '')),
        }
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False)
    
    print(f"  💾 stock_names: {len(names)}只")
    return True


# ════════════════════════════════════════════════════════════════
# 5. 状态检查
# ════════════════════════════════════════════════════════════════

def check_status():
    """检查所有A股数据文件状态"""
    import pandas as pd
    
    print("📊 A股数据文件状态")
    print("=" * 60)
    
    files = [
        ("K线 (a_hist_10y.parquet)", "a_hist_10y.parquet"),
        ("资金流 (moneyflow_core.parquet)", "moneyflow_core.parquet"),
        ("资金流全量 (moneyflow_full.parquet)", "moneyflow_full.parquet"),
        ("daily_basic (PE/PB/换手)", "daily_basic.parquet"),
        ("特征 (features_v2.parquet)", "features_v2.parquet"),
        ("a1_daily", "a1_daily.parquet"),
    ]
    
    for label, fname in files:
        fpath = DATA_CN / fname
        if not fpath.exists():
            print(f"  ❌ {label}: 不存在")
            continue
        
        mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        size_mb = fpath.stat().st_size / 1024 / 1024
        
        # 尝试读取最新日期
        latest = "?"
        try:
            df = pd.read_parquet(fpath)
            # 找日期列(兼容不同列名)
            for col in ['trade_date', 'Date', 'date', 'dt']:
                if col in df.columns:
                    latest = str(df[col].max())
                    break
            if latest == "?":
                latest = f"无日期列({list(df.columns)[:5]})"
        except Exception as e:
            # 可能是JSON格式
            try:
                with open(fpath) as f:
                    d = json.load(f)
                sample = list(d.keys())[0]
                dates = d[sample].get('dates', [])
                latest = str(dates[-1]) if dates else "JSON无日期"
            except:
                latest = f"读取失败: {str(e)[:50]}"
        
        status = "✅" if age_hours < 48 else "⚠️" if age_hours < 120 else "❌"
        print(f"  {status} {label}: {latest} ({age_hours/24:.1f}天前) {size_mb:.0f}MB")
    
    # 名称文件
    names_path = DATA_CN / "stock_names.json"
    if names_path.exists():
        with open(names_path) as f:
            d = json.load(f)
        print(f"  ✅ stock_names: {len(d.get('names', {}))}只")
    else:
        print(f"  ❌ stock_names: 不存在")


# ════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="A股数据更新管线")
    parser.add_argument("--kline", action="store_true", help="只更新K线")
    parser.add_argument("--moneyflow", action="store_true", help="只更新资金流")
    parser.add_argument("--basic", action="store_true", help="只更新daily_basic")
    parser.add_argument("--names", action="store_true", help="只更新股票名称")
    parser.add_argument("--convert", action="store_true", help="只做JSON→parquet转换")
    parser.add_argument("--status", action="store_true", help="只看状态")
    args = parser.parse_args()
    
    print(f"🔄 A股数据更新 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    if args.status:
        check_status()
        return
    
    if args.convert:
        print("🔄 K线 JSON→parquet 转换")
        with open(DATA_CN / "a_hist_10y.parquet") as f:
            hist = json.load(f)
        _save_kline_as_parquet(hist)
        return
    
    # 获取tushare
    pro = get_tushare()
    latest_trade = get_latest_trade_date(pro)
    print(f"最新交易日: {latest_trade}")
    
    run_all = not any([args.kline, args.moneyflow, args.basic, args.names])
    
    if args.kline or run_all:
        print("\n📡 K线更新:")
        update_kline(pro, latest_trade)
    
    if args.moneyflow or run_all:
        print("\n📡 资金流更新:")
        update_moneyflow(pro, latest_trade)
    
    if args.basic or run_all:
        print("\n📡 daily_basic更新:")
        update_daily_basic(pro, latest_trade)
    
    if args.names or run_all:
        print("\n📡 股票名称更新:")
        update_stock_names(pro)
    
    print("\n✅ 更新完成")
    check_status()


if __name__ == "__main__":
    main()
