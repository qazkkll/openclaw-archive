"""
绿箭v17 最终预测输�?�?含校准表，直接输出修正后概率
"""
import sys, json, os, math, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
import _paths

T0 = time.time()
print("══�?绿箭v17 最终预�?(含校�? ══�?)

df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v2.parquet")
with open(_paths.ML_DIR+"/us_sector_etf.json") as f: etf_data = json.load(f)

s2e = {'Technology':'XLK','Financial Services':'XLF','Financial':'XLF','Energy':'XLE',
       'Healthcare':'XLV','Industrials':'XLI','Consumer Defensive':'XLP',
       'Consumer Cyclical':'XLY','Utilities':'XLU','Basic Materials':'XLB',
       'Materials':'XLB','Real Estate':'XLRE','Communication Services':'XLC','Semiconductor':'SMH'}
def get_er(s):
    e=s2e.get(s)
    return etf_data[e]['ret5'] if e and e in etf_data else etf_data['SPY']['ret5']

df['sector_etf_ret5'] = df['sector'].apply(get_er)
for k in ['SPY','QQQ','IWM']:
    df[f'{k.lower()}_ret5'] = etf_data[k]['ret5']
df['sc'] = df['sector'].astype('category').cat.codes.astype(int)

all_feats = ['price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
             'ret1','ret5','ret20','ret60','macd','macd_signal','macd_hist',
             'vol_ratio','ma_bias20','vol5','trend_accel',
             'short_ratio','short_pct','market_cap','pe_trailing','pe_forward','beta',
             'sector_etf_ret5','spy_ret5','qqq_ret5','iwm_ret5','sc']

latest = df.dropna(subset=all_feats).drop_duplicates(subset='sym', keep='last')
Xl = latest[all_feats].values

m = xgb.XGBClassifier(, device='cuda')
m.load_model(_paths.US_MODEL_DIR + "/greenshaft_v17.json")
yp = m.predict_proba(Xl)

# 校准因子表（从v17测试集算出的�?cal_factors = [
    (0.9, 0.625),  # >90% �?实际62.5%
    (0.8, 0.571),  # >80% �?57.1%
    (0.7, 0.514),  # >70% �?51.4%
    (0.6, 0.475),  # >60% �?47.5%
    (0.5, 0.424),  # >50% �?42.4%
]

def adjust_prob(prob, cal_factors):
    """查表校准"""
    for threshold, actual in cal_factors:
        if prob >= threshold:
            return actual
    return prob  # <50%不调

preds = []
for i, (_, row) in enumerate(latest.iterrows()):
    p = yp[i]
    raw = float(p[4])
    adjusted = adjust_prob(raw, cal_factors)
    preds.append({
        'sym': row['sym'],
        'price': float(row['price']),
        'raw_up5': raw,
        'adj_up5': adjusted,
        'dn5': float(p[0]),
        'flat': float(p[2]),
    })
preds.sort(key=lambda x: -x['raw_up5'])

print(f"\n{'�?*75}")
print(f"{'排名':>3} {'代码':>8} {'价格':>8} {'原涨>5%':>8} {'校准�?:>8} {'�?5%':>8} {'sector':>12}")
print(f"{'─'*75}")
for i, r in enumerate(preds[:30]):
    row = latest[latest['sym']==r['sym']].iloc[0]
    sector = row.get('sector','?')
    print(f"{i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {r['raw_up5']*100:>7.1f}% {r['adj_up5']*100:>7.1f}% {r['dn5']*100:>7.1f}% {sector:>12}"[:75])

# 保存
out = {
    'timestamp': str(__import__('datetime').datetime.now()),
    'model': 'greenshaft_v17',
    'features': all_feats,
    'calibration': {f'>{t*100:.0f}%':{'predicted':t,'actual':a} for t,a in cal_factors},
    'predictions': [{'rank':i+1,'sym':r['sym'],'price':r['price'],
                      'raw_up5':round(r['raw_up5'],4),
                      'adjusted_up5':round(r['adj_up5'],4),
                      'dn5':round(r['dn5'],4)} 
                     for i,r in enumerate(preds[:50])],
}
with open(_paths.US_MODEL_DIR + "/us_v19_v17_prediction.json", 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"\n�?绿箭v17 最终预测保�? ({time.time()-T0:.0f}s)")
print(f"  保存: {_paths.win(_paths.US_MODEL_DIR + '/us_v19_v17_prediction.json')}")
