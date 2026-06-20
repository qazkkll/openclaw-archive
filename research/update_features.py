#!/usr/bin/env python3
"""
更新特征数据到最新日期，生成实时信号
"""
import pandas as pd, numpy as np, xgboost as xgb, json, time, os, warnings
import tushare as ts
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

ts.set_token('ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db')
pro = ts.pro_api()

print("更新数据到最新...")
t0 = time.time()

# 加载现有特征
hist = pd.read_parquet('data/cn/features_v2.parquet')
hist['date'] = pd.to_datetime(hist['date'])
hist['date_int'] = hist['date'].dt.strftime('%Y%m%d').astype(int)
last_date = hist['date_int'].max()
print(f"  现有数据到: {last_date}")

# 获取交易日历
cal = pro.trade_cal(exchange='SSE', is_open='1', 
    start_date=str(last_date), end_date='20260620')
new_dates = sorted([d for d in cal['cal_date'].tolist() if d > str(last_date)])
print(f"  需更新: {len(new_dates)}天 ({new_dates[0]}→{new_dates[-1]})")

if not new_dates:
    print("  数据已是最新")
else:
    # 拉取新数据
    daily_dfs = []
    mf_dfs = []
    
    for i, d in enumerate(new_dates):
        try:
            dd = pro.daily(trade_date=d)
            mm = pro.moneyflow(trade_date=d)
            if dd is not None and len(dd) > 0:
                daily_dfs.append(dd)
            if mm is not None and len(mm) > 0:
                mf_dfs.append(mm)
            if (i+1) % 10 == 0:
                print(f"  {i+1}/{len(new_dates)} ({time.time()-t0:.0f}s)")
            time.sleep(0.35)
        except Exception as e:
            if 'freq' in str(e).lower() or 'limit' in str(e).lower():
                print(f"  频次限制，等60s...")
                time.sleep(60)
            else:
                print(f"  {d} 错误: {e}")
                time.sleep(1)
    
    if daily_dfs:
        new_daily = pd.concat(daily_dfs, ignore_index=True)
        new_mf = pd.concat(mf_dfs, ignore_index=True) if mf_dfs else pd.DataFrame()
        
        # 合并
        new_daily['sym'] = new_daily['ts_code'].str[:6]
        new_daily['date'] = pd.to_datetime(new_daily['trade_date'])
        new_daily['date_int'] = new_daily['trade_date'].astype(int)
        
        df = new_daily[['sym','date','date_int','open','high','low','close','vol','amount']].copy()
        df = df.rename(columns={'vol': 'volume'})
        
        if len(new_mf) > 0:
            new_mf['sym'] = new_mf['ts_code'].str[:6]
            new_mf['date_int'] = new_mf['trade_date'].astype(int)
            mf_cols = ['sym','date_int','buy_sm_amount','sell_sm_amount','buy_md_amount','sell_md_amount',
                'buy_lg_amount','sell_lg_amount','buy_elg_amount','sell_elg_amount','net_mf_amount']
            df = df.merge(new_mf[mf_cols], on=['sym','date_int'], how='left')
        
        # 计算资金流
        for prefix in ['sm','md','lg','elg']:
            buy = df.get(f'buy_{prefix}_amount', 0)
            sell = df.get(f'sell_{prefix}_amount', 0)
            df[f'{prefix}_net'] = buy.fillna(0) - sell.fillna(0)
        df['total_net'] = df.get('net_mf_amount', pd.Series(0, index=df.index)).fillna(0)
        
        # 合并daily_basic
        basic = pd.read_parquet('data/cn/daily_basic.parquet')
        basic['sym'] = basic['ts_code'].str[:6]
        basic['date_int'] = basic['trade_date'].astype(int)
        basic = basic[['sym','date_int','pe_ttm','pb','ps_ttm','dv_ratio','circ_mv','turnover_rate']].drop_duplicates(['sym','date_int'])
        df = df.merge(basic, on=['sym','date_int'], how='left')
        
        # 计算技术特征（需要合并历史数据）
        combined = pd.concat([hist[['sym','date','date_int','close','volume','sm_net','md_net','lg_net','elg_net','total_net','circ_mv','turnover_rate']].rename(columns={'volume':'volume'}), 
            df[['sym','date','date_int','close','volume','sm_net','md_net','lg_net','elg_net','total_net','circ_mv','turnover_rate']]], ignore_index=True)
        combined = combined.drop_duplicates(['sym','date_int']).sort_values(['sym','date_int'])
        
        # 计算滚动特征
        for sym, group in combined.groupby('sym'):
            if len(group) < 25:
                continue
            closes = group['close'].values
            
            for idx, row in group.iterrows():
                date_int = row['date_int']
                if date_int <= last_date:
                    continue
                
                pos = group.index.get_loc(idx)
                if pos < 20:
                    continue
                
                # 收益率
                c = closes[pos]
                r1 = (c - closes[pos-1]) / closes[pos-1] if pos >= 1 else 0
                r5 = (c - closes[pos-5]) / closes[pos-5] if pos >= 5 else 0
                r10 = (c - closes[pos-10]) / closes[pos-10] if pos >= 10 else 0
                r20 = (c - closes[pos-20]) / closes[pos-20] if pos >= 20 else 0
                
                # 波动率
                rets_5 = [(closes[pos-j] - closes[pos-j-1])/closes[pos-j-1] for j in range(min(5, pos))]
                rets_20 = [(closes[pos-j] - closes[pos-j-1])/closes[pos-j-1] for j in range(min(20, pos))]
                vol5 = np.std(rets_5) if len(rets_5) > 1 else 0
                vol20 = np.std(rets_20) if len(rets_20) > 1 else 0
                atr = vol20  # 近似
                
                # RSI
                gains = [r for r in rets_20[-14:] if r > 0]
                losses = [-r for r in rets_20[-14:] if r < 0]
                avg_gain = np.mean(gains) if gains else 0
                avg_loss = np.mean(losses) if losses else 0.001
                rsi = 100 - 100 / (1 + avg_gain / avg_loss)
                
                # MACD近似
                ema12 = np.mean(closes[max(0,pos-11):pos+1])
                ema26 = np.mean(closes[max(0,pos-25):pos+1])
                macd = ema12 - ema26
                macd_hist = macd  # 近似
                
                # 资金流滚动
                sm_vals = group['sm_net'].values[max(0,pos-19):pos+1]
                md_vals = group['md_net'].values[max(0,pos-19):pos+1]
                lg_vals = group['lg_net'].values[max(0,pos-19):pos+1]
                elg_vals = group['elg_net'].values[max(0,pos-19):pos+1]
                total_vals = group['total_net'].values[max(0,pos-19):pos+1]
                
                sm_5 = np.sum(sm_vals[-5:])
                sm_20 = np.sum(sm_vals)
                md_5 = np.sum(md_vals[-5:])
                md_20 = np.sum(md_vals)
                lg_5 = np.sum(lg_vals[-5:])
                lg_20 = np.sum(lg_vals)
                elg_5 = np.sum(elg_vals[-5:])
                elg_20 = np.sum(elg_vals)
                total_5 = np.sum(total_vals[-5:])
                total_20 = np.sum(total_vals)
                
                # 52周高低
                high_52w = np.max(closes[max(0,pos-250):pos+1]) if pos >= 50 else c * 1.5
                low_52w = np.min(closes[max(0,pos-250):pos+1]) if pos >= 50 else c * 0.5
                
                # 更新hist
                mask = (hist['sym'] == sym) & (hist['date_int'] == date_int)
                if mask.any():
                    continue  # 已存在
                
                new_row = {
                    'sym': sym, 'date': row['date'], 'date_int': date_int,
                    'close': c, 'volume': row.get('volume', 0),
                    'sm_net': row.get('sm_net', 0), 'md_net': row.get('md_net', 0),
                    'lg_net': row.get('lg_net', 0), 'elg_net': row.get('elg_net', 0),
                    'total_net': row.get('total_net', 0),
                    'circ_mv': row.get('circ_mv', 0), 'turnover_rate': row.get('turnover_rate', 0),
                    'r1': r1, 'r5': r5, 'r10': r10, 'r20': r20,
                    'vol5': vol5, 'vol20': vol20, 'atr_pct': atr,
                    'vol_r': row.get('turnover_rate', 0) / 5 if row.get('turnover_rate', 0) else 0.5,
                    'rsi14': rsi, 'macd': macd, 'macd_hist': macd_hist,
                    'log_circ_mv': np.log(row['circ_mv']) if row.get('circ_mv', 0) and row.get('circ_mv', 0) > 0 else 15,
                    'sm_net_5': sm_5, 'sm_net_20': sm_20,
                    'md_net_5': md_5, 'md_net_20': md_20,
                    'lg_net_5': lg_5, 'lg_net_20': lg_20,
                    'elg_net_5': elg_5, 'elg_net_20': elg_20,
                    'total_net_5': total_5, 'total_net_20': total_20,
                }
                hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        
        print(f"  更新完成: {len(hist):,}行, 最新日期: {hist['date_int'].max()}")
        hist.to_parquet('data/cn/features_v2.parquet', index=False)
        print(f"  已保存 features_v2.parquet")

print(f"\n耗时: {time.time()-t0:.0f}秒")
