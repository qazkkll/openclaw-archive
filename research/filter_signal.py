#!/usr/bin/env python3
"""A股V2 — 信号过滤（排除ST/垃圾股）"""
import json, warnings
import pandas as pd, numpy as np
import tushare as ts
warnings.filterwarnings('ignore')

ts.set_token('ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db')
pro = ts.pro_api()

# 加载评分
scores = pd.read_parquet('research/v2_all_scores.parquet')
print(f'Total scores: {len(scores)}')

# 股票信息
si = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
si['sym'] = si['ts_code'].str.replace(r'\.\w+$', '', regex=True)
name_map = dict(zip(si['sym'], si['name']))
ind_map = dict(zip(si['sym'], si['industry']))
scores['name'] = scores['sym'].map(name_map)
scores['industry'] = scores['sym'].map(ind_map)

# 20日跌幅
h = pd.read_parquet('data/a_hist_10y.parquet')
h = h.rename(columns={'Code':'sym','Date':'date','C':'close'})
h['date'] = pd.to_datetime(h['date'].astype(str), format='%Y%m%d')
h = h.sort_values(['sym','date'])

# 最新价格和20日前价格
latest_close = h.groupby('sym')['close'].last()
past_close = h.groupby('sym')['close'].apply(lambda x: x.iloc[-21] if len(x) > 20 else x.iloc[0])
drop_20d = (latest_close / past_close - 1).to_dict()

scores['drop_20d'] = scores['sym'].map(drop_20d)
scores['drop_20d'] = scores['drop_20d'].fillna(0)

# 过滤
mask = (
    ~scores['name'].str.contains('ST|退市', na=False) &
    (scores['close'] >= 3) &
    (scores['drop_20d'] > -0.5) &
    (scores['score'] > -0.05)
)
filtered = scores[mask].sort_values('score', ascending=False)
print(f'Filtered: {len(scores)} -> {len(filtered)}')

top15 = filtered.head(15).copy()
top15['rank'] = range(1, len(top15) + 1)
top15['expected_ret'] = top15['score'] * 100

print(f'\n  A股V2 Top 15 (2026-06-18) [Filtered]')
print(f'  {"#":<3} {"股票":<8} {"行业":<8} {"名称":<10} {"价格":>7} {"预期":>6} {"20d":>6}')
print(f'  {"-"*55}')
for _, r in top15.iterrows():
    print(f'  {r["rank"]:<3} {r["sym"]:<8} {str(r["industry"]):<8} {str(r["name"]):<10} {r["close"]:>7.2f} {r["expected_ret"]:>5.1f}% {r["drop_20d"]*100:>5.1f}%')

# 更新信号
signal = {
    'date': '2026-06-18',
    'model': 'a_stock_xgb_v2',
    'wf': {'ic': 0.0809, 'rank_ic': 0.0817, 'icir': 0.996, 'ls': 0.0262},
    'filters': 'excl ST, price>=3, 20d drop>-50%',
    'top15': top15[['sym','close','score','expected_ret','industry','name']].to_dict('records'),
}
with open('research/v2_signal.json', 'w') as f:
    json.dump(signal, f, indent=2, default=str)
print(f'\nSaved: research/v2_signal.json')
