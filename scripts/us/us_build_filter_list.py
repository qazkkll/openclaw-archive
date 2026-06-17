#!/usr/bin/env python3
"""计算过滤后的有效股票列表"""
import pandas as pd, json, warnings; warnings.filterwarnings('ignore')
ML='/home/hermes/.hermes/openclaw-archive/scripts/system'; OUT='/home/hermes/.hermes/openclaw-project/scripts/system/us_filtered_syms.json'
print('加载10年价格数据...')
m1=pd.read_parquet(ML+'/us_hist_yf_10y.parquet',columns=['ticker','date','close','volume'])
m2=pd.read_parquet(ML+'/us_hist_megacap_10y.parquet',columns=['sym','date','close','volume'])
m2.rename(columns={'sym':'ticker'},inplace=True)
all_df=pd.concat([m1,m2],ignore_index=True).drop_duplicates(subset=['ticker','date'])
print(f'全部: {all_df.ticker.nunique()}只')
print(f'取近90天计算成交额...')
all_df=all_df.sort_values(['ticker','date'])
latest=all_df.groupby('ticker').tail(90)
res=[]
for sym,g in latest.groupby('ticker'):
    if len(g)<20:
        res.append({'sym':sym,'ok':False})
        continue
    avg=float(g['close'].mean()*g['volume'].mean())
    price=float(g['close'].iloc[-1])
    ok=avg>=5000000 and price>=3.0
    res.append({'sym':sym,'avg_vol_dollar':avg,'last_price':price,'ok':ok})
df=pd.DataFrame(res)
valid=sorted(df[df['ok']]['sym'].tolist())
print(f'过滤后: {len(valid)}只 (原{all_df.ticker.nunique()}只)')
print(f'条件: 近90日日均成交额≥$500万 + 最新价≥$3')
famous=['MARA','RIOT','COIN','AMC','GME','PLTR','SOFI','RIVN','SNAP','NIO','XPEV','LI']
vs=set(valid)
for f in famous:
    print(f'  {f}: {"通过" if f in vs else "过滤掉"}')
json.dump({'syms':valid,'count':len(valid),'total':all_df.ticker.nunique(),
           'date':str(pd.Timestamp.now())},open(OUT,'w'),indent=2)
print(f'\n保存: {OUT}')
