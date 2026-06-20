#!/usr/bin/env python3
"""
拉取 daily_basic (PE/PB/PS/股息率) 按tushare官方方法
按trade_date拉全市场截面，每天一次调用
"""
import tushare as ts, pandas as pd, time, os, json

os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))
ts.set_token('ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db')
pro = ts.pro_api()

# 交易日历
cal = pro.trade_cal(exchange='SSE', is_open='1', start_date='20160101', end_date='20260620')
trade_dates = sorted(cal['cal_date'].tolist())
print(f"交易日: {len(trade_dates)}")

# 检查已有缓存
cache_path = 'data/cn/daily_basic.parquet'
state_path = 'data/cn/daily_basic_state.json'

if os.path.exists(state_path):
    with open(state_path) as f:
        state = json.load(f)
    done_dates = set(state.get('done', []))
else:
    done_dates = set()

if os.path.exists(cache_path):
    cached = pd.read_parquet(cache_path)
    print(f"缓存: {len(cached):,}行, {len(done_dates)}天")
else:
    cached = pd.DataFrame()
    print("无缓存，全量拉取")

missing = [d for d in trade_dates if d not in done_dates]
print(f"待拉取: {len(missing)}天")

if not missing:
    print("已完成！")
    exit(0)

t0 = time.time()
new_dfs = []
errors = 0

for i, d in enumerate(missing):
    for retry in range(3):
        try:
            df = pro.daily_basic(trade_date=d,
                fields='ts_code,trade_date,pe_ttm,pb,ps_ttm,dv_ratio,total_mv,circ_mv,turnover_rate')
            if df is not None and len(df) > 0:
                new_dfs.append(df)
            done_dates.add(d)
            break
        except Exception as e:
            if 'freq' in str(e).lower() or 'limit' in str(e).lower() or '每分钟' in str(e):
                print(f"  频次限制，等60s...")
                time.sleep(60)
            else:
                errors += 1
                if retry == 2:
                    print(f"  {d} 失败: {e}")
                time.sleep(2)
    
    # 保存进度（每100天）
    if (i + 1) % 100 == 0:
        if new_dfs:
            batch = pd.concat(new_dfs, ignore_index=True)
            cached = pd.concat([cached, batch], ignore_index=True) if len(cached) > 0 else batch
            cached.to_parquet(cache_path, index=False)
            new_dfs = []
        
        with open(state_path, 'w') as f:
            json.dump({'done': list(done_dates)}, f)
        
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(missing) - i - 1) / rate
        print(f"  {i+1}/{len(missing)} ({elapsed:.0f}s, ETA {eta:.0f}s)")
    
    time.sleep(0.35)

# 最终保存
if new_dfs:
    batch = pd.concat(new_dfs, ignore_index=True)
    cached = pd.concat([cached, batch], ignore_index=True) if len(cached) > 0 else batch

if len(cached) > 0:
    cached.to_parquet(cache_path, index=False)

with open(state_path, 'w') as f:
    json.dump({'done': list(done_dates)}, f)

elapsed = time.time() - t0
print(f"\n完成: {len(cached):,}行, {len(done_dates)}天, {elapsed:.0f}秒, 错误{errors}次")
