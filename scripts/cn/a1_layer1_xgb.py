"""
A1 Layer 1 — XGBoost市场状态分类器

目标：用三期分离验证通过的指标做特征，把沪深300未来20日收益分成3类：
  - 0: 防御 (fwd_20d <= -3%)     ← 市场偏空
  - 1: 震荡 (-3% < fwd_20d < 3%)  ← 无方向
  - 2: 进攻 (fwd_20d >= 3%)      ← 市场偏多

三期分离：训练2016-2020 | 验证2021-2023H1 | 测试2023H2-2026

用法：
  python scripts/a1_layer1_xgb.py           # 完整训练+评估
  python scripts/a1_layer1_xgb.py --tune     # 网格搜索调参
"""

import json, os, sys, time, pickle
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 统一路径管理
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import D_DATA, INDEX_300, NORTH_MONEY
MODEL_DIR = os.path.join(D_DATA, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 数据加载 ───

def load_data():
    with open(INDEX_300, 'rb') as f:
        idx = json.load(f)
    klines = sorted(idx, key=lambda x: x['trade_date'])
    return klines


def calc_features(klines):
    """计算全部候选特征 + 标签"""
    from collections import OrderedDict
    
    closes = [k['close'] for k in klines]
    highs = [k['high'] for k in klines]
    lows = [k['low'] for k in klines]
    
    records = []
    for i in range(len(closes)):
        if i < 65:
            continue
        
        c = closes[:i+1]
        h = highs[:i+1]
        lo = lows[:i+1]
        
        r = OrderedDict()
        r['date'] = klines[i]['trade_date']
        price = c[-1]
        
        # ── 均线相对位置 ──
        ma5 = sum(c[-5:])/5
        ma10 = sum(c[-10:])/10
        ma20 = sum(c[-20:])/20
        ma60 = sum(c[-60:])/60
        
        r['pct_ma5'] = round((price/ma5 - 1)*100, 2) if ma5 > 0 else 0
        r['pct_ma10'] = round((price/ma10 - 1)*100, 2) if ma10 > 0 else 0
        r['pct_ma20'] = round((price/ma20 - 1)*100, 2) if ma20 > 0 else 0
        r['pct_ma60'] = round((price/ma60 - 1)*100, 2) if ma60 > 0 else 0
        
        # ── 均线斜率 ──
        if i >= 25:
            ma20_before = sum(c[-25:-5])/20
            r['ma20_slope'] = round((ma20/ma20_before - 1)*100, 3) if ma20_before > 0 else 0
        else:
            r['ma20_slope'] = 0
        
        if i >= 65:
            ma60_before = sum(c[-65:-5])/60
            r['ma60_slope'] = round((ma60/ma60_before - 1)*100, 3) if ma60_before > 0 else 0
        else:
            r['ma60_slope'] = 0
        
        # ── 均线排列 ──
        align = 0
        if ma5 > ma10: align += 1
        if ma10 > ma20: align += 1
        if ma20 > ma60: align += 1
        if price > ma5: align += 1
        if price > ma10: align += 1
        if price > ma60: align += 1
        r['ma_align'] = align
        
        # ── 波动 ──
        rets = [abs((c[j]/c[j-1] - 1)*100) if c[j-1] > 0 else 0 for j in range(1, len(c))]
        vol10 = sum(rets[-10:])/10 if len(rets) >= 10 else 1
        vol60 = sum(rets[-60:])/60 if len(rets) >= 60 else 1
        r['vol_ratio'] = round(vol10/vol60, 3) if vol60 > 0 else 1.0
        
        trs = [max(h[j]-lo[j], abs(h[j]-c[j-1]), abs(lo[j]-c[j-1])) for j in range(max(1,i-19), i+1)]
        atr20 = sum(trs)/len(trs) if trs else 0
        r['atr20_pct'] = round(atr20/price*100, 3) if price > 0 else 0
        
        # ── 动量 ──
        r['ret_5d'] = round((price/c[-6] - 1)*100, 2) if len(c) >= 6 else 0
        r['ret_10d'] = round((price/c[-11] - 1)*100, 2) if len(c) >= 11 else 0
        r['ret_20d'] = round((price/c[-21] - 1)*100, 2) if len(c) >= 21 else 0
        r['ret_60d'] = round((price/c[-61] - 1)*100, 2) if len(c) >= 61 else 0
        
        # ── RSI ──
        changes = [c[j] - c[j-1] for j in range(max(1,i-19), i+1)]
        gains = [x for x in changes if x > 0]
        losses = [-x for x in changes if x < 0]
        avg_gain = sum(gains)/20 if gains else 0
        avg_loss = sum(losses)/20 if losses else 0
        r['rsi'] = round(100 - 100/(1 + avg_gain/avg_loss), 1) if avg_loss > 0 else 100
        
        # ── 20日收益百分位 ──
        if len(c) >= 41:
            past_20d_rets = [(c[j]/c[j-20] - 1)*100 for j in range(20, i+1)]
            cur_20d = (price/c[-21] - 1)*100 if len(c) >= 21 else 0
            pct_rank = sum(1 for x in past_20d_rets if x < cur_20d)/len(past_20d_rets)*100
            r['ret20d_pct'] = round(pct_rank, 1)
        else:
            r['ret20d_pct'] = 50
        
        # ── 标签 ──
        if i + 20 < len(closes):
            fwd = (closes[i+20]/price - 1)*100
            r['fwd_20d'] = round(fwd, 2)
            if fwd >= 3.0:
                r['label'] = 2  # 进攻
            elif fwd <= -3.0:
                r['label'] = 0  # 防御
            else:
                r['label'] = 1  # 震荡
        else:
            r['fwd_20d'] = None
            r['label'] = None
        
        records.append(r)
    
    return records


def split_data(records):
    """三期分离"""
    train = [r for r in records if r['date'] <= '20201231']
    val = [r for r in records if '20210101' <= r['date'] <= '20230630']
    test = [r for r in records if r['date'] >= '20230701']
    return train, val, test


def to_xy(rs, feature_names):
    """转换为numpy特征矩阵 + 标签"""
    X, y = [], []
    for r in rs:
        if r['label'] is None:
            continue
        X.append([r[f] for f in feature_names])
        y.append(r['label'])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ─── 特征列表（三期验证通过的 + 辅助的） ───

BASE_FEATURES = [
    'pct_ma5', 'pct_ma10', 'pct_ma20', 'pct_ma60',
    'ma20_slope', 'ma60_slope', 'ma_align',
    'vol_ratio', 'atr20_pct', 'rsi',
    'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d', 'ret20d_pct'
]


# ─── 训练 ───

def train_xgb(X_train, y_train, X_val, y_val, params=None):
    """训练XGBoost分类器"""
    import xgboost as xgb
    
    if params is None:
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'eval_metric': 'mlogloss',
            'seed': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'device': 'cuda'
        }
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=BASE_FEATURES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=BASE_FEATURES)
    
    n_rounds = params.pop('n_estimators', 500)
    early_stop = params.pop('early_stopping_rounds', 50)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_rounds,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=early_stop,
        verbose_eval=50
    )
    
    params['n_estimators'] = n_rounds
    params['early_stopping_rounds'] = early_stop
    
    return model


def evaluate(model, X, y, name):
    """评估分类结果"""
    import xgboost as xgb
    d = xgb.DMatrix(X, feature_names=BASE_FEATURES)
    preds = model.predict(d)
    pred_classes = np.argmax(preds, axis=1)
    
    acc = np.mean(pred_classes == y)
    
    # 按类别的准确率
    per_class = {}
    for cls in [0, 1, 2]:
        mask = y == cls
        if mask.sum() > 0:
            acc_c = np.mean(pred_classes[mask] == cls)
            per_class[cls] = {
                'n': int(mask.sum()),
                'accuracy': round(float(acc_c), 3)
            }
    
    return {
        'accuracy': round(float(acc), 3),
        'per_class': per_class,
        'n': len(y)
    }


def grid_search(X_train, y_train, X_val, y_val):
    """网格搜索最佳参数"""
    import xgboost as xgb
    
    param_grid = [
        {'max_depth': d, 'learning_rate': lr, 'min_child_weight': mw,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'n_estimators': 500}
        for d in [3, 5, 7]
        for lr in [0.03, 0.05, 0.1]
        for mw in [3, 5, 10]
    ]
    
    best_acc = 0
    best_params = None
    best_model = None
    
    print(f"\n网格搜索: {len(param_grid)} 组参数")
    print(f"{'depth':>5} {'lr':>5} {'mw':>5} | {'train_acc':>9} {'val_acc':>8} {'best_round':>10}")
    print("-" * 45)
    
    for i, p in enumerate(param_grid):
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': p['max_depth'],
            'learning_rate': p['learning_rate'],
            'subsample': p['subsample'],
            'colsample_bytree': p['colsample_bytree'],
            'min_child_weight': p['min_child_weight'],
            'eval_metric': 'mlogloss',
            'seed': 42,
            'n_jobs': -1
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=BASE_FEATURES)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=BASE_FEATURES)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=p['n_estimators'],
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # 验证集准确率
        preds = model.predict(dval)
        pred_cls = np.argmax(preds, axis=1)
        val_acc = np.mean(pred_cls == y_val)
        
        # 训练集准确率
        preds_train = model.predict(dtrain)
        pred_cls_train = np.argmax(preds_train, axis=1)
        train_acc = np.mean(pred_cls_train == y_train)
        
        best_iter = model.best_iteration if hasattr(model, 'best_iteration') else p['n_estimators']
        
        marker = '← BEST' if val_acc > best_acc else ''
        print(f"{p['max_depth']:>5} {p['learning_rate']:>5.3f} {p['min_child_weight']:>5} | {train_acc:>8.3f} {val_acc:>8.3f} {best_iter:>10} {marker}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = p
            best_model = model
    
    return best_model, best_params, best_acc


# ─── 主流程 ───

def main():
    tune = '--tune' in sys.argv
    
    print("=" * 65)
    print("A1 Layer 1 — XGBoost市场状态分类器")
    print(f"  特征: {len(BASE_FEATURES)}个 | 分类: 防御/震荡/进攻")
    print(f"  数据: index_300 (2016-01 ~ 2026-06)")
    print("=" * 65)
    
    t0 = time.time()
    
    # 加载+特征
    klines = load_data()
    records = calc_features(klines)
    train, val, test = split_data(records)
    
    print(f"\n数据分布:")
    print(f"  训练: {len(train)}天")
    print(f"  验证: {len(val)}天")
    print(f"  测试: {len(test)}天")
    
    X_train, y_train = to_xy(train, BASE_FEATURES)
    X_val, y_val = to_xy(val, BASE_FEATURES)
    X_test, y_test = to_xy(test, BASE_FEATURES)
    
    # 标签分布
    for name, y in [('训练', y_train), ('验证', y_val), ('测试', y_test)]:
        label_names = ['防御(0)', '震荡(1)', '进攻(2)']
        dist = [f"{label_names[i]}: {(y==i).sum()}" for i in [0,1,2]]
        print(f"  {name}: {', '.join(dist)}")
    
    import xgboost as xgb
    
    if tune:
        print("\n>>> 网格搜索模式 <<<")
        model, best_params, best_val_acc = grid_search(X_train, y_train, X_val, y_val)
        print(f"\n最佳参数: {best_params}")
        print(f"最佳验证准确率: {best_val_acc:.3f}")
    else:
        # 默认参数（中等树，控制过拟合）
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'eval_metric': 'mlogloss',
            'seed': 42,
            'n_jobs': -1
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=BASE_FEATURES)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=BASE_FEATURES)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=BASE_FEATURES)
        
        print(f"\n训练参数: {params}")
        print(f"  树的数量: 300 (early stopping at 50轮的验证损失不下降)")
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=50
        )
        
        best_iter = model.best_iteration
        print(f"\n最佳迭代: {best_iter}")
    
    # ── 评估 ──
    print(f"\n{'=' * 65}")
    print("评估结果:")
    print(f"{'Dataset':>8} {'Accuracy':>10} {'N':>6}  {'防御acc':>8} {'震荡acc':>8} {'进攻acc':>8}")
    print("-" * 55)
    
    for name, X, y in [('Train', X_train, y_train), ('Val', X_val, y_val), ('Test', X_test, y_test)]:
        md = xgb.DMatrix(X, feature_names=BASE_FEATURES)
        preds = model.predict(md)
        pc = np.argmax(preds, axis=1)
        acc = np.mean(pc == y)
        
        cls_acc = []
        for cls in [0, 1, 2]:
            mask = y == cls
            if mask.sum() > 0:
                ca = np.mean(pc[mask] == cls)
                cls_acc.append(f"{ca:.3f}")
            else:
                cls_acc.append("N/A")
        
        print(f"{name:>8} {acc:>10.3f} {len(y):>6}  {cls_acc[0]:>8} {cls_acc[1]:>8} {cls_acc[2]:>8}")
    
    # ── 特征重要性 ──
    importance = model.get_score(importance_type='gain')
    imp_sorted = sorted(importance.items(), key=lambda x: -x[1])
    print(f"\n特征重要性 (gain):")
    total_gain = sum(imp_sorted[i][1] for i in range(min(10, len(imp_sorted))))
    for name, gain in imp_sorted[:10]:
        print(f"  {name:<14}: {gain:>8.1f} ({gain/total_gain*100:.0f}%)")
    
    # ── 混淆矩阵（测试集） ──
    md_test = xgb.DMatrix(X_test, feature_names=BASE_FEATURES)
    preds_test = model.predict(md_test)
    pc_test = np.argmax(preds_test, axis=1)
    
    print(f"\n测试集混淆矩阵:")
    print(f"{'':>8} {'Pred_0':>8} {'Pred_1':>8} {'Pred_2':>8}")
    for true_cls in [0, 1, 2]:
        mask = y_test == true_cls
        counts = [int((pc_test[mask] == p).sum()) for p in [0, 1, 2]]
        label = ['Def', 'Neu', 'Agg'][true_cls]
        print(f"{'True_'+label:>8} {counts[0]:>8} {counts[1]:>8} {counts[2]:>8}")
    
    # ── 当前状态预测 ──
    cur_record = records[-1]
    cur_X = np.array([[cur_record[f] for f in BASE_FEATURES]], dtype=np.float32)
    cur_dm = xgb.DMatrix(cur_X, feature_names=BASE_FEATURES)
    cur_probs = model.predict(cur_dm)[0]
    
    label_map = {0: '🔴 防御', 1: '⚪ 震荡', 2: '🟢 进攻'}
    print(f"\n当前状态 ({cur_record['date']}):")
    for cls in [0, 1, 2]:
        print(f"  {label_map[cls]}: {cur_probs[cls]*100:.1f}%")
    print(f"  判断: {label_map[np.argmax(cur_probs)]}")
    
    # ── 保存模型 ──
    model_path = os.path.join(MODEL_DIR, 'a1_layer1_xgb.json')
    model.save_model(model_path)
    
    # 保存元数据
    meta = {
        'features': BASE_FEATURES,
        'n_estimators': model.best_iteration if hasattr(model, 'best_iteration') else 300,
        'params': {
            'max_depth': params['max_depth'],
            'learning_rate': params['learning_rate'],
            'min_child_weight': params['min_child_weight'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree']
        },
        'label_map': {0: 'defensive', 1: 'neutral', 2: 'aggressive'},
        'label_cutoffs': {'defensive': '<= -3%', 'neutral': '-3% ~ +3%', 'aggressive': '>= +3%'},
        'test_accuracy': float(np.mean(pc_test == y_test)),
        'feature_importance': dict(imp_sorted[:15]),
        'data_dates': {'train': '2016-2020', 'val': '2021-2023H1', 'test': '2023H2-2026'},
        'generated': '2026-06-12'
    }
    
    meta_path = os.path.join(MODEL_DIR, 'a1_layer1_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print(f"\n模型保存: {model_path}")
    print(f"元数据: {meta_path}")
    print(f"\n总耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
