import pandas as pd, numpy as np, json, xgboost as xgb, warnings, os
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

df = pd.read_parquet('data/a_hist_10y.parquet')
df = df.rename(columns={'Code':'sym','Date':'date','O':'open','H':'high','L':'low','C':'close','V':'volume'})
df['date'] = df['date'].astype(int)

mf = pd.read_parquet('data/cn/moneyflow_core.parquet')
mf['sym'] = mf['ts_code'].str[:6]
mf['date'] = mf['trade_date'].astype(int)
for col in ['sm','md','lg','elg']:
    mf[col+'_net'] = mf['buy_'+col+'_amount'] - mf['sell_'+col+'_amount']
mf['total_net'] = mf['net_mf_amount']
df = df.merge(mf[['sym','date','total_net','lg_net','md_net','elg_net']], on=['sym','date'], how='left')

df = df[(df['close']>=3) & (df['close']<=200) & (df['volume']>0)].copy()
df = df.sort_values(['sym','date']).reset_index(drop=True)
day_counts = df.groupby('sym')['date'].transform('count')
df = df[day_counts >= 20].copy()

# Features
df['ret5']=df.groupby('sym')['close'].pct_change(5)
df['ret10']=df.groupby('sym')['close'].pct_change(10)
df['ret20']=df.groupby('sym')['close'].pct_change(20)
df['ma20']=df.groupby('sym')['close'].transform(lambda x:x.rolling(20,min_periods=1).mean())
df['ma60']=df.groupby('sym')['close'].transform(lambda x:x.rolling(60,min_periods=1).mean())
df['ma20_bias']=(df['close']-df['ma20'])/df['ma20']
df['ma60_bias']=(df['close']-df['ma60'])/df['ma60']
df['vol5']=df.groupby('sym')['close'].transform(lambda x:x.pct_change().rolling(5,min_periods=2).std())
df['vol20']=df.groupby('sym')['close'].transform(lambda x:x.pct_change().rolling(20,min_periods=2).std())
delta=df.groupby('sym')['close'].diff()
gain=delta.clip(lower=0).groupby(df['sym']).transform(lambda x:x.rolling(14,min_periods=1).mean())
loss=(-delta).clip(lower=0).groupby(df['sym']).transform(lambda x:x.rolling(14,min_periods=1).mean())
df['rsi_14']=100-100/(1+gain/loss.replace(0,np.nan))
df['rsi_14']=df['rsi_14'].fillna(50)
ema12=df.groupby('sym')['close'].transform(lambda x:x.ewm(span=12,min_periods=1).mean())
ema26=df.groupby('sym')['close'].transform(lambda x:x.ewm(span=26,min_periods=1).mean())
df['macd']=ema12-ema26
df['macd_signal']=df.groupby('sym')['macd'].transform(lambda x:x.ewm(span=9,min_periods=1).mean())
df['macd_hist']=df['macd']-df['macd_signal']
df['tr']=np.maximum(df['high']-df['low'],np.maximum(abs(df['high']-df.groupby('sym')['close'].shift(1)),abs(df['low']-df.groupby('sym')['close'].shift(1))))
df['atr14']=df.groupby('sym')['tr'].transform(lambda x:x.rolling(14,min_periods=1).mean())
df['atr_pct']=df['atr14']/df['close']
df['vol_ratio']=df.groupby('sym')['volume'].transform(lambda x:x.rolling(5).mean())/df.groupby('sym')['volume'].transform(lambda x:x.rolling(20).mean())
for col in ['total_net','lg_net','md_net','elg_net']:
    df[col+'_5d']=df.groupby('sym')[col].transform(lambda x:x.rolling(5,min_periods=1).sum())
    df[col+'_20d']=df.groupby('sym')[col].transform(lambda x:x.rolling(20,min_periods=1).sum())
    df[col+'_5d_rk']=df.groupby('date')[col+'_5d'].rank(pct=True)
df['breadth']=df.groupby('date')['ret5'].transform(lambda x:(x>0).mean())
df['mkt_ret20']=df.groupby('date')['ret20'].transform('mean')

FEATURE_COLS=['ret5','ret10','ret20','ma20_bias','ma60_bias','vol5','vol20','rsi_14','macd_hist','atr_pct','vol_ratio','total_net_5d','lg_net_5d','md_net_5d','elg_net_5d','total_net_20d','lg_net_20d','md_net_20d','elg_net_20d','total_net_5d_rk','lg_net_5d_rk','md_net_5d_rk','elg_net_5d_rk','breadth','mkt_ret20']

model = xgb.XGBRegressor(n_estimators=150,max_depth=5,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=1.0,random_state=42,n_jobs=4,verbosity=0)
model.load_model('models/cn/cn_alpha_v2_xgb.json')

max_date = df['date'].max()
td = df[df['date']==max_date].copy()
X = td[FEATURE_COLS].fillna(0)
td['score'] = model.predict(X)

with open('data/cn/stock_names.json') as f:
    nd = json.load(f)
    name_map = nd.get('names',{})
    ind_map = nd.get('industries',{})

# Filter: 3xxx or 688xxx, RSI 25-65 (not overbought), score > 0.03
mask = (td['sym'].str.startswith('3') | td['sym'].str.startswith('688'))
mask = mask & (td['rsi_14'] >= 25) & (td['rsi_14'] <= 65)
mask = mask & (td['score'] > 0.03)
candidates = td[mask].nlargest(20, 'score')

print(f"Date: {max_date} | Filter: 3xxx/688xxx, RSI 25-65, score>0.03")
print(f"Candidates found: {len(candidates)}")
print()
for i,(_,r) in enumerate(candidates.iterrows()):
    nm = name_map.get(r['sym'],'')[:4]
    ind = ind_map.get(r['sym'],'')[:6]
    ret20 = "{:.1%}".format(r['ret20']) if not pd.isna(r['ret20']) else 'N/A'
    above_ma = "above" if r['close'] > r.get('ma20',0) else "below"
    sig = "GG" if r['score']>=0.08 else "G" if r['score']>=0.06 else "Y"
    fund_rk = r.get('lg_net_5d_rk',0)
    print(f"{i+1:>2} [{sig}] {r['sym']:>8} {nm:>6} price={r['close']:>8.2f} score={r['score']:>7.4f} RSI={r['rsi_14']:>4.0f} 20d={ret20:>8} MA20={above_ma:>5} fund_rk={fund_rk:.0%} {ind:>8}")
