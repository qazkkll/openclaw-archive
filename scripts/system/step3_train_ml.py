"""
step3_train_ml.py — A股ML训练脚本
直接从 a_hist_10y.json + moneyflow_data.parquet 逐批读 → 训练XGBoost
不生成中间大文件，一步到位
"""
import sys, json, gc, time, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

t0 = time.time()

# ─── 1. 加载K线数据 ───
print('加载日K线数据...')
with open('/home/hermes/.hermes/openclaw-project/data/a_hist_10y.json', 'r', encoding='utf-8') as f:
    hist = json.load(f)

print(f'  {len(hist)}只股票')
stocks = list(hist.keys())[:3000]  # 取前3000只（避免内存爆炸）

# ─── 2. 加载资金流（只取最近2年） ───
print('加载资金流数据（最近2年）...')
from pyarrow.parquet import ParquetFile
pf = ParquetFile('/home/hermes/.hermes/openclaw-project/data/moneyflow_data.parquet')

# 先构建 ts_code -> moneyflow_records 映射
mf_map = {}
total_rg = pf.metadata.num_row_groups
for rg_idx in range(total_rg):
    tbl = pf.read_row_group(rg_idx, 
        columns=['ts_code','trade_date','net_mf_amount','net_mf_ratio',
                 'buy_lg_amount','sell_lg_amount','buy_elg_amount','sell_elg_amount',
                 'buy_md_amount','sell_md_amount','buy_sm_amount','sell_sm_amount','close'])
    code = tbl.column(0).to_pylist()[0]
    if code not in stocks:
        continue
    # 提取列
    data = {}
    n = tbl.num_rows
    data['trade_date'] = [int(d) for d in tbl.column(1).to_pylist()]
    # 数值列
    col_idx = {'net_mf_amount': 2, 'net_mf_ratio': 3,
               'buy_lg_amount': 4, 'sell_lg_amount': 5,
               'buy_elg_amount': 6, 'sell_elg_amount': 7,
               'buy_md_amount': 8, 'sell_md_amount': 9,
               'buy_sm_amount': 10, 'sell_sm_amount': 11,
               'close': 12}
    for col_name, ci in col_idx.items():
        arr = tbl.column(ci).to_pylist()
        data[col_name] = [float(v) if v is not None else 0.0 for v in arr]
    
    mf_map[code] = data
    
    if (rg_idx+1) % 50 == 0:
        print(f'  行组{rg_idx+1}/{total_rg}, 已加载{len(mf_map)}只', end='\r')

print(f'\n加载完成: {len(mf_map)}只')

# ─── 3. 特征计算 + 合并 ───
print('计算特征...')

def calc_features(code):
    """对一只股票计算特征"""
    h = hist.get(code)
    mf = mf_map.get(code)
    if not h or not mf:
        return None, None
    
    h_dates = h.get('dates', [])
    h_closes = h.get('closes', [])
    
    # 转成dict方便查找
    h_dict = {}
    for i, d in enumerate(h_dates):
        h_dict[d] = h_closes[i] if i < len(h_closes) else 0
    
    # 资金流日期与K线日期对齐
    mf_dates = mf['trade_date']
    
    X_list, y_list = [], []
    
    for idx in range(len(mf_dates) - 10):
        d = mf_dates[idx]
        if d not in h_dict:
            continue
        
        # 资金流特征
        net_mf = mf['net_mf_amount'][idx]
        f = {
            'net_mf': net_mf / 1e4 if abs(net_mf) > 1 else net_mf,
            'net_mf_ratio': mf['net_mf_ratio'][idx],
            'big_net': (mf['buy_lg_amount'][idx] - mf['sell_lg_amount'][idx]) / 1e4,
            'xbig_net': (mf['buy_elg_amount'][idx] - mf['sell_elg_amount'][idx]) / 1e4,
        }
        
        # 3日趋势
        if idx >= 2:
            mf3 = sum(mf['net_mf_amount'][idx-j] for j in range(3))
            f['net_mf_3d'] = mf3 / 1e4 if abs(mf3) > 1 else mf3
            f['net_mf_trend'] = 1.0 if mf3 > 0 else 0.0
        else:
            f['net_mf_3d'] = f['net_mf']
            f['net_mf_trend'] = 1.0 if f['net_mf'] > 0 else 0.0
        
        # 主力散户背离
        big_in = mf['buy_lg_amount'][idx] + mf['buy_elg_amount'][idx]
        big_out = mf['sell_lg_amount'][idx] + mf['sell_elg_amount'][idx]
        small_net = mf['buy_sm_amount'][idx] - mf['sell_sm_amount'][idx]
        f['big_small_div'] = (big_in - big_out) - small_net
        
        # K线技术面（MA5/MA10对比）
        curr_close = h_dict[d]
        f['close'] = curr_close
        
        # 查找近5日均价
        ma5_prices = []
        for j in range(1, 10):
            # 向前找前几天的close
            td = d
            for d2 in reversed(h_dates):
                if d2 < td:
                    td = d2
                    break
            if j <= 5:
                ma5_prices.append(h_dict.get(td, 0))
        
        ma5 = np.mean(ma5_prices[:5]) if len(ma5_prices[:5]) > 0 else curr_close
        f['ma5_ratio'] = curr_close / ma5 if ma5 > 0 else 1.0
        
        # Y标签：未来5天涨跌
        # 找未来第5天的close
        future_dates = [d2 for d2 in h_dates if d2 > d]
        if len(future_dates) >= 5:
            d5 = future_dates[4]
            fut_close = h_dict.get(d5, 0)
            if curr_close > 0 and fut_close > 0:
                ret_5d = (fut_close - curr_close) / curr_close
                y = 1.0 if ret_5d > 0.02 else 0.0
                X_list.append(list(f.values()))
                y_list.append(y)
    
    if len(X_list) < 10:
        return None, None
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

# 逐只计算
all_X, all_y = [], []
stock_processed = 0

for code in stocks:
    X, y = calc_features(code)
    if X is not None and len(X) > 0:
        all_X.append(X)
        all_y.append(y)
        stock_processed += 1
    if stock_processed % 100 == 0:
        print(f'  {stock_processed}只完成, 总样本{sum(len(yy) for yy in all_y)}', flush=True)

X_all = np.vstack(all_X)
y_all = np.concatenate(all_y)
t1 = time.time()
print(f'\n特征计算完成: {stock_processed}只, {len(X_all)}样本, {X_all.shape[1]}特征')
print(f'正例率: {y_all.mean():.2%}')
print(f'耗时: {(t1-t0)/60:.1f}分钟')

# ─── 4. 训练 ───
print('\n训练XGBoost...')
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

# 评估
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
print(f'\n✅ 模型评估:')
print(f'  Accuracy: {acc:.4f}')
print(f'  AUC: {auc:.4f}')
print(f'  正例率(测试): {y_test.mean():.2%}')

# ─── 5. Platt校准 ───
print('\nPlatt校准...')
calibrator = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
calibrator.fit(X_test, y_test)
cal_prob = calibrator.predict_proba(X_test)[:, 1]
print(f'  校准后平均概率: {cal_prob.mean():.4f}')

# ─── 6. 保存模型 ───
MODEL_DIR = '/home/hermes/.hermes/openclaw-project/data/models'
os.makedirs(MODEL_DIR, exist_ok=True)

model.save_model(os.path.join(MODEL_DIR, 'a_xgb_v1.json'))
print(f'\n✅ 模型保存: {MODEL_DIR}')

# 特征名
feature_names = ['net_mf','net_mf_ratio','big_net','xbig_net','net_mf_3d','net_mf_trend','big_small_div','close','ma5_ratio']

# 输出特征重要性
importances = model.feature_importances_
print('\n特征重要性:')
for n, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f'  {n}: {imp:.4f}')

t2 = time.time()
print(f'\n总耗时: {(t2-t0)/60:.1f}分钟')
print(f'模型: {os.path.join(MODEL_DIR, "a_xgb_v1.json")}')
