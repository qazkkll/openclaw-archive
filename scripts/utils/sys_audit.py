#!/usr/bin/env python3
"""
验证集独立性审计 — A1回测框架
检查三个致命问题：
1. 评分是否泄露了未来信息（i.e. 今天是否用到了明天才知道的资金流？）
2. 缓存(cache)是否跨窗口泄露（walk-forward各窗口间是否共享了同一天的资金流评分？）
3. 北向动量百分位是否泄露（计算n_mom时是否用到了未来数据？）
"""
import json, sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 统一路径管理
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import NORTH_MONEY

t0 = time.time()

# ─── 加载数据 ─────────────────────────────────────────────
print("加载数据...")
kline = json.load(open(f'{BASE}/a_hist_10y.parquet', 'r'))
raw = json.load(open(f'{BASE}/a1_daily.json', 'r'))
nd = json.load(open(NORTH_MONEY, 'r'))
recs = nd.get('records', nd)
print(f"K线: {len(kline)}只")
print(f"资金流原始: {len(raw)}个交易日" if isinstance(raw, dict) else f"资金流原始: {len(raw)}行")

# 所有有数据的交易日（按tushare资金流日期）
all_dates = sorted(raw.keys())
print(f"资金流交易天数: {len(all_dates)} (最早{all_dates[0]}, 最晚{all_dates[-1]})")

# 北向数据
north_dates = [r['trade_date'] for r in recs]
north_vals = [r.get('north_money', 0) for r in recs]
north_vals = [float(v) if v else 0.0 for v in north_vals]
print(f"北向天数: {len(north_dates)} (最早{north_dates[0]}, 最晚{north_dates[-1]})")

# ─── 检查1: 资金流缓存(cache)是否跨窗口泄露 ──────────
# 原代码中，cache = {} 在文件顶部定义，所有窗口共享
# 每个窗口虽然只跑 test_dates，但 cache[today] 是在第一次run_backtest时写入
# 如果先跑WF1再跑WF2，WF2用到了WF1已缓存的评分——这没问题
# 但如果WF2的test_dates包含了WF1训练期的数据？不，WF是前向滚动的
# 但有一点：cache是全局的，WF1测试期跑的评分结果不会用在WF2的测试期
# 因为WF1和WF2的测试期不重叠（前移250天）
# 
# 真正的独立性问题：评分cache对当天是否有效？
# 比如今天2024-01-01的资金流数据 → 到2024-01-01收盘后才出来
# 而我们在跑2024-01-01的回测时，用到了当天收盘时的资金流数据
# 这是用"收盘后才有的数据"作为"当天的买入信号"
# 但A1模型中，买入价也是当天收盘价...
# 所以问题是：同一天的收盘价 + 同一天的资金流 → 这是否算偷跑？

print("\n\n=== 检查1: 数据是否存在未来函数 ===")
print("场景: 今天收盘 = 30元, 资金流数据 = 净流入1亿")
print("A1用今天收盘价买入, 用今天资金流评分")
print("但实际上: 资金流数据在16:00后才能拿到")
print("             而收盘价在15:00就定了")
print("结论: 如果A1是16:00后出推荐 → 次日开盘买入 → 没问题")
print("     但回测里假设的是 '今天买入' → 这有问题！")
print("     因为15:00收盘→16:00数据出来→次日09:30才能入场")
print("     回测偷跑了1天！")

# ─── 检查2: 北向动量百分位是否用了未来数据 ──────────
print("\n\n=== 检查2: 北向动量计算是否有未来函数 ===")
print("原代码:")
print("  for i in range(59, len(north_vals)):")
print("      s20 = sum(north_vals[i-19:i+1])  ← 含当天")
print("      s60 = sum(north_vals[i-59:i+1])  ← 含当天")
print("  def nb_pct(ds):")
print("      # 找ds对应的索引i")
print("      # s20 = sum(north_vals[i-19:i+1]) ← 含当天")
print("      # 然后用这个占比跟全量n_mom比较")
print()
print("这里nb_pct(ds)在计算s20/s60时用到了当天的北向值")
print("而北向数据也是在收盘后才出的")
print("如果回测假设当天买入 → 北向数据还没出来 → 泄露")

# ─── 检查3: 评分公式中的数据时间对齐 ───────────────
print("\n\n=== 检查3: 评分cache的时间对齐 ===")
print("原代码在文件顶部跑:")
print("  for i, td in enumerate(all_dates):")
print("      td_data = raw.get(td, {})")
print("      scores = []")
print("      ... 用td_data的net_mf/buy_lg/sell_lg计算评分 ...")
print("      cache[td] = scores")
print("这个cache是用tushare moneyflow(trade_date=td)的数据")
print("moneyflow(td)的数据是td日收盘后才出的")
print("但回测里run_backtest(test_dates)中:")
print("  for today in test_dates:")
print("      cands = cache[today]  ← 用t+0的资金流数据")
print("      # 再用today的收盘价买入")
print("")
print("='这=意=味=着=用=t+0=的=资=金=流=数=据= + =t+0=的=收=盘=价=买=入='")
print("但实际上: 16:00才出资金流 → 次日09:30才能买")
print("回测里直接用了当天资金流+当天收盘价买 → 偷跑1天")

# ─── 总结 ──────────────────────────────────────────────
print("\n\n" + "="*70)
print("审计结论")
print("="*70)
print("""
1. 评分cache本身OK（每个交易日的资金流数据互不混淆）

2. 但『时间对齐』有重大缺陷：
   回测假设：当天15:00收盘 → 当天16:00出资金流 → 当天买入
   现实是： 当天15:00收盘 → 当天16:00出资金流 → 次日09:30才能买
   
   这导致：
   - 回测中如果某天资金流好股价涨，当天就买了
   - 但实际上要次日才能买，而次日可能已经跌了
   - 简单说：回测多拿了1天的信息

3. 北向百分位同理：nb_pct(today)用到了today的北向数据
   而北向也是收盘后才出的

4. 这算不算『严重泄露』？
   T+1 vs T+0的差异对高频策略影响最大
   对A1这种5天换一次的低频策略，偷跑1天影响较小但仍然是缺陷

5. 修正方案：
   - 回测中：run_backtest(today) 改用 yesterday的评分（T-1）
   - 即：today的买入信号用yesterday的资金流数据
   - 真实的交易日延后1天
   - 北向门控同理
""")

print(f"耗时: {time.time()-t0:.1f}s")
