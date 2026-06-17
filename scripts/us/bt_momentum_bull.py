#!/usr/bin/env python3
"""
检验：20日动量追涨策略在不同牛市中的表现
假设：SPY > MA200 = 牛市 → 选20日涨幅最高5只
对比：SPY 买入持有
"""
import yfinance as yf, warnings
warnings.filterwarnings('ignore')
import numpy as np

UNIVERSE = ['AAPL','MSFT','GOOGL','AMZN','NVDA','TSLA','AMD','MU','INTC',
            'AVGO','QCOM','CRM','ADBE','ORCL','PANW','CRWD','FTNT','NFLX',
            'DIS','HD','COST','WMT','JPM','V','MA','UNH','LLY','PFE','MRK','CAT','GE']

BULL_PERIODS = [
    ("2009.3-2020.2", "2009-03-09", "2020-02-19", "后金融危机大牛市"),
    ("2016.7-2018.9", "2016-07-01", "2018-09-30", "特朗普牛市"),
    ("2020.4-2022.1", "2020-04-01", "2022-01-03", "疫情后反弹"),
    ("2023.1-2026.5", "2023-01-01", "2026-05-18", "AI牛市"),
]

print("=" * 80)
print("20日动量追涨策略 · 多牛市回测")
print("策略：每20天，选过去20日涨幅最高5只，持有20天")
print("对比：同期SPY买入持有")
print("=" * 80)

for pname, start, end, desc in BULL_PERIODS:
    print(f"\n--- {pname}: {desc} ---")
    
    spy = yf.download('SPY', start=start, end=end, progress=False)
    spy_c = spy['Close']
    spy_ma200 = spy_c.rolling(200).mean()
    bull_days = int((spy_c > spy_ma200).sum())
    print(f"SPY在MA200上方天数: {bull_days}/{len(spy_c)}")

    prices = {}
    for t in UNIVERSE:
        try:
            h = yf.download(t, start=start, end=end, progress=False)
            if len(h) > 250:
                prices[t] = h['Close']
        except:
            pass

    strategy_rets = []
    spy_rets = []
    dates = spy_c.index.tolist()
    
    for i in range(20, len(dates)-20, 20):
        d_open = dates[i]
        d_close = dates[i+20]
        
        # 动量筛选
        momentum = []
        for t, p in prices.items():
            try:
                v_prev = float(p.asof(dates[i-20]))
                v_now = float(p.asof(d_open))
                if v_prev > 0:
                    momentum.append((t, (v_now / v_prev - 1) * 100))
            except:
                pass
        
        if len(momentum) < 5:
            continue
        
        momentum.sort(key=lambda x: x[1], reverse=True)
        top5 = momentum[:5]
        
        # 计算下20天表现
        rets = []
        for t, _ in top5:
            try:
                v_buy = float(prices[t].asof(d_open))
                v_sell = float(prices[t].asof(d_close))
                if v_buy > 0:
                    rets.append((v_sell / v_buy - 1) * 100)
            except:
                pass
        
        if rets:
            avg_ret = np.mean(rets)
            strategy_rets.append(avg_ret)
            
            s_buy = float(spy_c.asof(d_open))
            s_sell = float(spy_c.asof(d_close))
            spy_rets.append((s_sell / s_buy - 1) * 100)

    if strategy_rets:
        total_mom = sum(strategy_rets)
        total_spy = sum(spy_rets)
        wins = sum(1 for s, m in zip(strategy_rets, spy_rets) if s > m)
        n = len(strategy_rets)
        
        print(f"  策略总收益: {total_mom:+.1f}%")
        print(f"  SPY总收益:   {total_spy:+.1f}%")
        print(f"  超额收益:    {total_mom-total_spy:+.1f}%")
        print(f"  胜率:        {wins}/{n} ({100*wins/n:.0f}%)")
        print(f"  平均20天:    策略{np.mean(strategy_rets):+.2f}% vs SPY{np.mean(spy_rets):+.2f}%")
        print(f"  最大单期:    策略{max(strategy_rets):+.1f}% / {min(strategy_rets):+.1f}%")
        print(f"  SPY最大单期:  {max(spy_rets):+.1f}% / {min(spy_rets):+.1f}%")
    else:
        print("  数据不足")

print("\n" + "=" * 80)
print("回测完毕")
