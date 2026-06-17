import sys, os, json, gc, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE = r'/home/hermes/.hermes/openclaw-archive/data'
hist = pd.read_parquet(BASE + '/a_hist_10y.parquet')
mf = pd.read_parquet(BASE + '/a3_moneyflow_factors.parquet')

# 全量验证：使用所有股票（从hist里取）
pool_codes = hist['ticker'].unique().tolist()
print(f"全量验证模式: {len(pool_codes)} 只股票")

# fallback
if not pool_codes:
    pool_codes = ['000001','000002','000005','000008','000016','000027','000039',
        '000050','000059','000060','000061','000063','000069','000100',
        '000157','000166','000333','000338','000400','000401','000402',
        '000408','000413','000415','000423','000425','000426','000429',
        '000430','000488','000498','000501','000503','000513','000516',
        '000517','000519','000520','000525','000528','000531','000536',
        '000537','000538','000539','000540','000541','000543','000544',
        '000546','000547','000548','000550','000551','000552','000553',
        '000554','000555','000558','000559','000560','000561','000563',
        '000564','000565','000566','000567','000568','000569','000570',
        '000571','000572','000573','000576','000581','000582','000584',
        '000585','000586','000587','000589','000590','000591','000592',
        '000593','000594','000595','000596','000597','000598','000599',
        '000600','000601','000603','000605','000607','000608','000609',
        '000610','000612','000615','000617','000619','000620','000622',
        '000623','000625','000626','000627','000628','000629','000630',
        '000631','000632','000633','000635','000636','000637','000638',
        '000639','000650','000651','000652','000655','000656','000657',
        '000659','000661','000662','000663','000665','000666','000667',
        '000668','000669','000670','000671','000672','000676','000677',
        '000678','000679','000680','000681','000682','000683','000685',
        '000686','000687','000688','000690','000691','000692','000693',
        '000695','000697','000698','000700','000701','000702','000703',
        '000705','000707','000708','000709','000710','000711','000712',
        '000713','000715','000716','000717','000718','000719','000720',
        '000721','000722','000723','000725','000726','000727','000728',
        '000729','000731','000738','000739','000750','000751','000752',
        '000753','000756','000758','000759','000760','000761','000762',
        '000765','000766','000767','000768','000776','000777','000778',
        '000779','000782','000783','000785','000786','000787','000788',
        '000789','000790','000791','000792','000793','000795','000796',
        '000797','000798','000799','000800','000801','000802','000803',
        '000807','000809','000810','000811','000816','000819','000823',
        '000825','000826','000829','000830','000831','000833','000836',
        '000837','000838','000839','000848','000850','000851','000856',
        '000858','000860','000861','000868','000869','000876','000877',
        '000878','000880','000881','000882','000883','000885','000886',
        '000887','000888','000889','000893','000895','000897','000898',
        '000899','000900','000901','000902','000903','000905','000906',
        '000908','000909','000910','000911','000912','000913','000915',
        '000917','000918','000919','000921','000922','000923','000925',
        '000926','000927','000928','000930','000932','000933','000935',
        '000937','000938','000948','000949','000951','000952','000957',
        '000958','000959','000960','000961','000962','000963','000966',
        '000967','000968','000969','000970','000975','000976','000977',
        '000978','000979','000980','000981','000982','000983','000987',
        '000988','000989','000990','000993','000996','000997','000998',
        '000999','001203','001205','001227','001267','001269','001283',
        '001286','001287','001288','001289','001296','001299','001301',
        '001308','001309','001311','001313','001314','001316','001317',
        '001318','001319','001323','001324','001326','001328','001330',
        '001331','001332','001333','001336','001337','001338','001339',
        '001360','001366','001367','001368','001373','001378','001379',
        '001380','001382','001385','001386','001387','001389','001390',
        '001391','001392','001393','001395','001396','001397','001398',
        '001399','001896']

# 只保留hist里有的
pool_codes = [c for c in pool_codes if c in set(hist['ticker'].values)]
print("Selected", len(pool_codes), "stocks")

mf_cols = [c for c in mf.columns if c not in ['ts_code','trade_date']]
tech_cols = ['pct_ma5','pct_ma10','pct_ma20','pct_ma60','pct_ma120',
    'ma20_slope','ma60_slope','vol_atr20','vol_ratio','ret_1d','ret_5d',
    'ret_10d','ret_20d','ret_60d','rsi14','macd_dif','macd_dea','macd_bar',
    'bb_width','bb_position','vol_ratio_5_20','obv_ratio_5_20',
    'kdj_k','kdj_d','kdj_j','ma5_ma10_cross','accel_5_10',
    'vol_breakout','ma_align','ret5_max','ret3_vs_ema12']
feat_cols = tech_cols + mf_cols

def calc_tech(ticker, row):
    dates_arr = row['dates']
    c = np.array(row['c'], dtype=np.float64)
    h = np.array(row['h'], dtype=np.float64)
    l = np.array(row['l'], dtype=np.float64)
    o = np.array(row['o'], dtype=np.float64)
    v = np.array(row['v'], dtype=np.float64)
    n = len(c)
    if n < 140:
        return None
    
    def _ma(arr, w):
        cs = np.cumsum(arr)
        res = np.full(n, np.nan)
        res[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
        return res
    
    ma5 = _ma(c, 5); ma10 = _ma(c, 10); ma20 = _ma(c, 20)
    ma60 = _ma(c, 60); ma120 = _ma(c, 120)
    ma120 = np.where(np.isnan(ma120), ma60, ma120)
    
    pct_ma5 = (c/ma5-1)*100; pct_ma10 = (c/ma10-1)*100
    pct_ma20 = (c/ma20-1)*100; pct_ma60 = (c/ma60-1)*100; pct_ma120 = (c/ma120-1)*100
    
    # slope
    ma20_sl = np.full(n, np.nan); ma60_sl = np.full(n, np.nan)
    for i in range(9, n):
        if not np.isnan(ma20[i-9:i+1]).any():
            ma20_sl[i] = np.polyfit(np.arange(10), ma20[i-9:i+1], 1)[0]
    for i in range(19, n):
        if not np.isnan(ma60[i-19:i+1]).any():
            ma60_sl[i] = np.polyfit(np.arange(20), ma60[i-19:i+1], 1)[0]
    
    # ma_align
    ma_align = np.full(n, np.nan)
    for i in range(120, n):
        vs = [ma5[i], ma10[i], ma20[i], ma60[i], ma120[i]]
        if not np.isnan(vs).any():
            ma_align[i] = sum(1 for j in range(4) if vs[j] > vs[j+1])
    
    # ret
    ret_1d=np.full(n,np.nan); ret_5d=np.full(n,np.nan); ret_10d=np.full(n,np.nan)
    ret_20d=np.full(n,np.nan); ret_60d=np.full(n,np.nan)
    if n>1: ret_1d[1:]=(c[1:]/c[:-1]-1)*100
    if n>5: ret_5d[4:]=(c[4:]/c[:-4]-1)*100
    if n>10: ret_10d[9:]=(c[9:]/c[:-9]-1)*100
    if n>20: ret_20d[19:]=(c[19:]/c[:-19]-1)*100
    if n>60: ret_60d[59:]=(c[59:]/c[:-59]-1)*100
    
    # RSI
    diff = np.diff(c)
    g = np.where(diff>0,diff,0); ls = np.where(diff<0,-diff,0)
    rsi14 = np.full(n,np.nan)
    for i in range(14,n):
        ag=np.mean(g[i-13:i+1]); al=np.mean(ls[i-13:i+1])
        rsi14[i]=100-(100/(1+ag/al)) if al>1e-10 else 100
    
    # MACD
    def _ema(arr,p):
        a=2.0/(p+1); res=np.full(n,np.nan)
        res[0]=arr[0]
        for i in range(1,n): res[i]=a*arr[i]+(1-a)*res[i-1]
        return res
    ema12=_ema(c,12); ema26=_ema(c,26)
    macd_dif=ema12-ema26; macd_dea=_ema(macd_dif,9); macd_bar=(macd_dif-macd_dea)*2
    
    # BBands
    bb_std=np.full(n,np.nan)
    for i in range(19,n): bb_std[i]=np.std(c[i-19:i+1])
    bb_width=bb_std/ma20*100
    bb_pos=np.where(bb_std>1e-10,(c-ma20+2*bb_std)/(4*bb_std+1e-10),0.5)
    
    # Vol
    vol_ma5=_ma(v,5); vol_ma20=_ma(v,20)
    vol_ratio=np.where(vol_ma20>0,v/vol_ma20*100,0)
    vol_ratio_5_20=np.where(vol_ma20>0,vol_ma5/vol_ma20*100,0)
    
    tr=np.maximum(h[1:]-l[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(l[1:]-c[:-1])))
    atr=np.full(n,np.nan)
    for i in range(13,n): atr[i]=np.mean(tr[i-13:i+1])
    vol_atr20=atr/ma20*100
    
    obv=np.zeros(n)
    for j in range(1,n):
        obv[j]=v[j] if c[j]>c[j-1] else (-v[j] if c[j]<c[j-1] else 0)
    obv_cum=np.cumsum(obv)
    obv_5=np.full(n,np.nan); obv_20=np.full(n,np.nan)
    for i in range(4,n): obv_5[i]=obv_cum[i]-obv_cum[i-4]
    for i in range(19,n): obv_20[i]=obv_cum[i]-obv_cum[i-19]
    obv_r=obv_5/(np.abs(obv_20)+1e-10)*100
    
    vol_brk=np.full(n,np.nan)
    for i in range(20,n):
        m=np.max(v[i-19:i+1])
        vol_brk[i]=(v[i]-m)/m*100 if m>0 else 0
    
    # KDJ
    ll=np.full(n,np.nan); hh=np.full(n,np.nan)
    for i in range(8,n):
        ll[i]=np.min(l[i-8:i+1]); hh[i]=np.max(h[i-8:i+1])
    rsv=np.where(hh-ll>1e-10,(c-ll)/(hh-ll)*100,50)
    k_k=np.full(n,np.nan); k_d=np.full(n,np.nan); k_j=np.full(n,np.nan)
    for i in range(9,n):
        k_k[i]=rsv[i] if np.isnan(k_k[i-1]) else 2/3*k_k[i-1]+1/3*rsv[i]
        k_d[i]=rsv[i] if np.isnan(k_d[i-1]) else 2/3*k_d[i-1]+1/3*k_k[i]
        k_j[i]=3*k_k[i]-2*k_d[i]
    
    ma5_ma10_cross=(ma5-ma10)/ma10*100
    accel=np.full(n,np.nan)
    if n>10: accel[9:]=ret_5d[9:]-ret_10d[9:]
    
    ret5_max=np.full(n,np.nan)
    for i in range(4,n):
        ret5_max[i]=np.max(c[i-4:i+1])/c[i-4]*100-100 if c[i-4]>0 else 0
    
    ret3_ema=np.full(n,np.nan)
    for i in range(2,n):
        r3=(c[i]/c[i-2]-1)*100 if c[i-2]>0 else 0
        ret3_ema[i]=r3-(ema12[i]/ema12[i-2]-1)*100 if i>=11 and ema12[i-2]>0 else r3
    
    lookback = min(240, n)
    dates_str = [str(d) for d in dates_arr[-lookback:]]
    
    return {
        'trade_date': dates_str,
        'pct_ma5': pct_ma5[-lookback:].tolist(),
        'pct_ma10': pct_ma10[-lookback:].tolist(),
        'pct_ma20': pct_ma20[-lookback:].tolist(),
        'pct_ma60': pct_ma60[-lookback:].tolist(),
        'pct_ma120': pct_ma120[-lookback:].tolist(),
        'ma20_slope': ma20_sl[-lookback:].tolist(),
        'ma60_slope': ma60_sl[-lookback:].tolist(),
        'vol_atr20': vol_atr20[-lookback:].tolist(),
        'vol_ratio': vol_ratio[-lookback:].tolist(),
        'ret_1d': ret_1d[-lookback:].tolist(),
        'ret_5d': ret_5d[-lookback:].tolist(),
        'ret_10d': ret_10d[-lookback:].tolist(),
        'ret_20d': ret_20d[-lookback:].tolist(),
        'ret_60d': ret_60d[-lookback:].tolist(),
        'rsi14': rsi14[-lookback:].tolist(),
        'macd_dif': macd_dif[-lookback:].tolist(),
        'macd_dea': macd_dea[-lookback:].tolist(),
        'macd_bar': macd_bar[-lookback:].tolist(),
        'bb_width': bb_width[-lookback:].tolist(),
        'bb_position': bb_pos[-lookback:].tolist(),
        'vol_ratio_5_20': vol_ratio_5_20[-lookback:].tolist(),
        'obv_ratio_5_20': obv_r[-lookback:].tolist(),
        'kdj_k': k_k[-lookback:].tolist(),
        'kdj_d': k_d[-lookback:].tolist(),
        'kdj_j': k_j[-lookback:].tolist(),
        'ma5_ma10_cross': ma5_ma10_cross[-lookback:].tolist(),
        'accel_5_10': accel[-lookback:].tolist(),
        'vol_breakout': vol_brk[-lookback:].tolist(),
        'ma_align': ma_align[-lookback:].tolist(),
        'ret5_max': ret5_max[-lookback:].tolist(),
        'ret3_vs_ema12': ret3_ema[-lookback:].tolist(),
        'close': c[-lookback:].tolist(),
        'ticker': ticker
    }

print("Calculating technical features for", len(pool_codes), "stocks...")
all_dfs = []
for idx, ticker in enumerate(pool_codes):
    if (idx+1) % 50 == 0:
        print(f"  {idx+1}/{len(pool_codes)}")
    row = hist[hist['ticker']==ticker].iloc[0]
    sim_id = ticker + '.SZ'
    
    feats = calc_tech(ticker, row)
    if feats is None:
        continue
    
    tech_df = pd.DataFrame(feats)
    tech_df['ts_code'] = sim_id
    
    # Merge with moneyflow
    mf_stock = mf[mf['ts_code']==sim_id].copy()
    mf_stock['td_str'] = mf_stock['trade_date'].astype(str)
    
    merged = tech_df.merge(mf_stock, left_on='trade_date', right_on='td_str', how='left', suffixes=('','_y'))
    for col in mf_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill(limit=5)
    
    all_dfs.append(merged)

final = pd.concat(all_dfs, ignore_index=True)
print(f"\nFinal dataset: {final.shape} rows, {len(all_dfs)} stocks")

# Generate labels
print("Generating labels...")
final = final.sort_values(['ticker','trade_date']).reset_index(drop=True)

def make_label(grp):
    c_vals = grp['close'].values
    if len(c_vals) <= 5:
        grp['label'] = -1
        return grp
    fwd = np.full(len(c_vals), np.nan)
    fwd[:-5] = (c_vals[5:]/c_vals[:-5]-1)*100
    grp['fwd_ret'] = fwd
    grp['label'] = np.where(fwd > 5, 1, np.where(fwd < -3, 0, -1))
    return grp

final = final.groupby('ticker', group_keys=False).apply(make_label)
train = final[final['label'] >= 0].copy()
n_pos = sum(train['label']==1)
n_neg = sum(train['label']==0)
print(f"Training samples: {len(train)} (pos={n_pos}, neg={n_neg}, ratio={n_pos/max(1,n_neg):.3f})")

# Save sample
save_cols = ['ticker','trade_date','label'] + feat_cols + ['fwd_ret']
save_cols = [c for c in save_cols if c in train.columns]
train[save_cols].to_csv(BASE+'/a3_v3_sample.csv', index=False)
print(f"Sample saved: {BASE+'/a3_v3_sample.csv'}")

# LightGBM WF
if len(train) < 1000:
    print("Not enough training samples")
    sys.exit(1)

print("\n=== LightGBM Walk-Forward ===")
import lightgbm as lgb

all_dates = sorted(train['trade_date'].unique())
fold_size = len(all_dates) // 4
reports = []

for fold in range(4):
    ts = fold * fold_size
    te = min((fold+1)*fold_size, len(all_dates))
    if fold == 3:
        te = len(all_dates)
    if fold == 0:
        continue  # first fold has no prior data
    
    trn_dates = set(all_dates[:ts])
    tst_dates = set(all_dates[ts:te])
    
    tr = train[train['trade_date'].isin(trn_dates)]
    te_df = train[train['trade_date'].isin(tst_dates)]
    
    if len(tr) < 500 or len(te_df) < 100:
        continue
    
    scale_pos = sum(tr['label']==0) / max(1, sum(tr['label']==1))
    X_tr = tr[feat_cols].fillna(0).values
    y_tr = tr['label'].values
    X_te = te_df[feat_cols].fillna(0).values
    y_te = te_df['label'].values
    
    model = lgb.LGBMClassifier(
        num_leaves=31, learning_rate=0.05, n_estimators=200,
        scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, random_state=42
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], eval_metric='auc')
    
    y_p = (model.predict(X_te) > 0.5).astype(int)
    acc = np.mean(y_p == y_te)
    prec = np.sum((y_p==1)&(y_te==1)) / max(1, np.sum(y_p==1))
    rec = np.sum((y_p==1)&(y_te==1)) / max(1, np.sum(y_te==1))
    
    print(f"  Fold {fold+1}: train={len(tr):,} test={len(te_df):,} | acc={acc:.4f} prec={prec:.4f} rec={rec:.4f}")
    reports.append({'fold':fold+1, 'accuracy':round(acc,4), 'precision':round(prec,4), 'recall':round(rec,4)})

# Save final model
model.booster_.save_model(BASE+'/models/a3_v3_lightgbm.txt')

# Feature importance
imp = pd.DataFrame({'feature': feat_cols, 'importance': model.feature_importances_})
imp = imp.sort_values('importance', ascending=False).head(20)
print(f"\nTop 20 features:")
print(imp.to_string(index=False))

avg_acc = np.mean([r['accuracy'] for r in reports])
avg_prec = np.mean([r['precision'] for r in reports])
print(f"\n=== A3_v3 WF Average ===")
print(f"Accuracy: {avg_acc:.4f} (V1 baseline: 0.49, improvement: {avg_acc-0.49:+.4f})")
print(f"Precision: {avg_prec:.4f}")

report = {
    'model': 'a3_v3_lgb',
    'n_stocks': len(all_dfs),
    'n_samples': len(train),
    'n_pos': int(n_pos),
    'n_neg': int(n_neg),
    'n_features': len(feat_cols),
    'tech_features': len(tech_cols),
    'mf_features': len(mf_cols),
    'avg_accuracy': round(avg_acc, 4),
    'avg_precision': round(avg_prec, 4),
    'improvement_v1': round(avg_acc - 0.49, 4),
    'fold_reports': reports,
    'top_features': imp['feature'].tolist()[:20],
    'v1_baseline': 0.49,
}

with open(BASE+'/a3_v3_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\nReport saved: {BASE+'/a3_v3_report.json'}")
print("Done!")
