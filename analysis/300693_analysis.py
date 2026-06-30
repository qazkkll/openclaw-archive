#!/usr/bin/env python3
"""深度分析300693盛弘股份"""
import tushare as ts
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

# 读取token
with open('/home/hermes/.hermes/openclaw-archive/data/config/tushare.json') as f:
    cfg = json.load(f)
ts.set_token(cfg['token'])
pro = ts.pro_api()

ts_code = '300693.SZ'
print("=" * 80)
print("📊 300693 盛弘股份 深度分析报告")
print("=" * 80)

# 1. 近30天日K线数据
print("\n📈 【1】近30天K线数据")
try:
    df_daily = pro.daily(ts_code=ts_code, start_date='20260528', end_date='20260630')
    if df_daily is not None and len(df_daily) > 0:
        df_daily = df_daily.sort_values('trade_date').reset_index(drop=True)
        print(df_daily[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg']].to_string(index=False))
        
        # 计算均线
        df_daily['ma5'] = df_daily['close'].rolling(5).mean()
        df_daily['ma10'] = df_daily['close'].rolling(10).mean()
        df_daily['ma20'] = df_daily['close'].rolling(20).mean()
        print(f"\n最新收盘价: {df_daily['close'].iloc[-1]}")
        print(f"MA5: {df_daily['ma5'].iloc[-1]:.2f}")
        print(f"MA10: {df_daily['ma10'].iloc[-1]:.2f}")
        print(f"MA20: {df_daily['ma20'].iloc[-1]:.2f}")
        
        # 近期高低点
        recent = df_daily.tail(20)
        print(f"\n20日最高价: {recent['high'].max()} ({recent.loc[recent['high'].idxmax(), 'trade_date']})")
        print(f"20日最低价: {recent['low'].min()} ({recent.loc[recent['low'].idxmin(), 'trade_date']})")
        print(f"20日振幅: {(recent['high'].max() - recent['low'].min()) / recent['close'].iloc[0] * 100:.2f}%")
        
        # 走势判断
        last_close = df_daily['close'].iloc[-1]
        ma5 = df_daily['ma5'].iloc[-1]
        ma10 = df_daily['ma10'].iloc[-1]
        ma20 = df_daily['ma20'].iloc[-1]
        
        print(f"\n走势形态分析:")
        if last_close < ma5 < ma10 < ma20:
            print("  ❌ 空头排列（价 < MA5 < MA10 < MA20），短期趋势向下")
        elif last_close > ma5 > ma10 > ma20:
            print("  ✅ 多头排列，短期趋势向上")
        elif last_close < ma5 and ma5 < ma10:
            print("  ⚠️ 空头排列但可能出现底部信号（RSI极低）")
        else:
            print(f"  ➡️ 价在均线间震荡")
        
        # 最近5日走势
        last5 = df_daily.tail(5)
        up_days = (last5['pct_chg'] > 0).sum()
        down_days = (last5['pct_chg'] < 0).sum()
        print(f"  近5日: {up_days}涨/{down_days}跌")
        
    else:
        print("  ⚠️ 未获取到日K数据")
        df_daily = None
except Exception as e:
    print(f"  ❌ 日K数据获取失败: {e}")
    df_daily = None

# 2. 每日指标（换手率、PE等）
print("\n📋 【2】每日基本面指标")
try:
    df_basic = pro.daily_basic(ts_code=ts_code, start_date='20260610', end_date='20260630')
    if df_basic is not None and len(df_basic) > 0:
        df_basic = df_basic.sort_values('trade_date').reset_index(drop=True)
        cols = ['trade_date', 'turnover_rate', 'turnover_rate_f', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'total_mv', 'circ_mv']
        available_cols = [c for c in cols if c in df_basic.columns]
        print(df_basic[available_cols].to_string(index=False))
    else:
        print("  ⚠️ 未获取到daily_basic数据")
except Exception as e:
    print(f"  ❌ daily_basic获取失败: {e}")

# 3. 个股资金流向
print("\n💰 【3】资金流向详情")
try:
    df_money = pro.moneyflow(ts_code=ts_code, start_date='20260610', end_date='20260630')
    if df_money is not None and len(df_money) > 0:
        df_money = df_money.sort_values('trade_date').reset_index(drop=True)
        print(df_money.to_string(index=False))
        
        # 最新一日资金流向
        latest = df_money.iloc[-1]
        print(f"\n最新交易日资金流向 ({latest.get('trade_date', 'N/A')}):")
        print(f"  超大单净流入: {latest.get('buy_elg_amount', 'N/A')} vs {latest.get('sell_elg_amount', 'N/A')}")
        print(f"  大单净流入:   {latest.get('buy_lg_amount', 'N/A')} vs {latest.get('sell_lg_amount', 'N/A')}")
        print(f"  中单净流入:   {latest.get('buy_md_amount', 'N/A')} vs {latest.get('sell_md_amount', 'N/A')}")
        print(f"  小单净流入:   {latest.get('buy_sm_amount', 'N/A')} vs {latest.get('sell_sm_amount', 'N/A')}")
    else:
        print("  ⚠️ 未获取到资金流向数据")
except Exception as e:
    print(f"  ❌ 资金流向获取失败: {e}")

# 4. 股东变动
print("\n👥 【4】股东信息")
try:
    # 十大流通股东
    df_holders = pro.top10_floatholders(ts_code=ts_code, start_date='20260101', end_date='20260630')
    if df_holders is not None and len(df_holders) > 0:
        print("近期十大流通股东:")
        print(df_holders[['ann_date', 'end_date', 'holder_name', 'hold_amount', 'hold_ratio']].head(20).to_string(index=False))
except Exception as e:
    print(f"  ⚠️ 股东数据获取失败: {e}")

# 5. 公司基本面
print("\n🏢 【5】公司基本面")
try:
    df_info = pro.stock_basic(ts_code=ts_code, fields='ts_code,symbol,name,area,industry,market,list_date,fullname,cnspell')
    if df_info is not None and len(df_info) > 0:
        info = df_info.iloc[0]
        print(f"  股票名称: {info.get('name', 'N/A')}")
        print(f"  全称: {info.get('fullname', 'N/A')}")
        print(f"  行业: {info.get('industry', 'N/A')}")
        print(f"  地区: {info.get('area', 'N/A')}")
        print(f"  上市日期: {info.get('list_date', 'N/A')}")
        print(f"  市场: {info.get('market', 'N/A')}")
except Exception as e:
    print(f"  ❌ 公司信息获取失败: {e}")

# 6. 财务数据
print("\n📊 【6】财务数据")
try:
    df_fina = pro.fina_indicator(ts_code=ts_code, start_date='20250101', end_date='20260630')
    if df_fina is not None and len(df_fina) > 0:
        df_fina = df_fina.sort_values('end_date', ascending=False).head(4)
        fina_cols = ['end_date', 'eps', 'roe', 'roe_waa', 'grossprofit_margin', 'netprofit_margin', 'debt_to_assets', 'current_ratio']
        available = [c for c in fina_cols if c in df_fina.columns]
        print(df_fina[available].to_string(index=False))
except Exception as e:
    print(f"  ❌ 财务数据获取失败: {e}")

# 7. 利润表
print("\n💵 【7】利润表")
try:
    df_income = pro.income(ts_code=ts_code, start_date='20250101', end_date='20260630')
    if df_income is not None and len(df_income) > 0:
        df_income = df_income.sort_values('end_date', ascending=False).head(4)
        inc_cols = ['end_date', 'revenue', 'n_income', 'total_profit', 'oper_cost']
        available = [c for c in inc_cols if c in df_income.columns]
        print(df_income[available].to_string(index=False))
except Exception as e:
    print(f"  ❌ 利润表获取失败: {e}")

# 8. 行业对比
print("\n🏭 【8】行业数据")
try:
    df_industry = pro.stock_basic(market='创业板', fields='ts_code,name,industry')
    if df_industry is not None:
        # 统计行业
        ind_counts = df_industry['industry'].value_counts().head(20)
        print("创业板行业分布(TOP20):")
        print(ind_counts.to_string())
        
        # 找同行业公司
        target_industry = df_info.iloc[0].get('industry', '') if df_info is not None and len(df_info) > 0 else ''
        if target_industry:
            same_ind = df_industry[df_industry['industry'] == target_industry]
            print(f"\n同行业({target_industry})创业板公司:")
            print(same_ind[['ts_code', 'name']].to_string(index=False))
except Exception as e:
    print(f"  ❌ 行业数据获取失败: {e}")

# 9. 周线数据
print("\n📅 【9】周线数据")
try:
    df_week = pro.weekly(ts_code=ts_code, start_date='20260301', end_date='20260630')
    if df_week is not None and len(df_week) > 0:
        df_week = df_week.sort_values('trade_date').reset_index(drop=True)
        print(df_week[['trade_date', 'open', 'high', 'low', 'close', 'pct_chg']].tail(10).to_string(index=False))
except Exception as e:
    print(f"  ❌ 周线数据获取失败: {e}")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
