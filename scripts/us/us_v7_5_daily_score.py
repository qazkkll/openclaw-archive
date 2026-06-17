#!/usr/bin/env python3
"""
绿箭V8-Lottery — 彩票股专属评分
底层重构：只对$1-10低价股评分，标签=fwd_5d_ret>50%
3倍于旧模型的彩票捕捉率
取代原绿箭V8全市场模型
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import xgboost as xgb

MD = '/home/hermes/.hermes/openclaw-project/data/models'
ML = '/home/hermes/.hermes/openclaw-archive/scripts/system'
DATA_DIR = '/home/hermes/.hermes/openclaw-archive/data'

def gen_lottery_feats(df):
    """生成彩票模型需要的7个交叉特征"""
    d = df.copy()
    d['close_log'] = np.log1p(d['ma5'].clip(lower=0.01))
    d['close_x_vol'] = d['ma5'] * d['vol_ratio']
    d['plus_di_x_low_vol'] = d['plus_di'] * (1 / (1 + d['vol_ratio']))
    d['adx_x_rsi'] = d['adx'] * d['rsi14']
    d['bb_x_vol'] = d['bb_width'] * d['vol_ratio']
    d['rsi_x_kdj'] = d['rsi14'] * (d['k'] + d['d']) / 100
    d['low_price'] = (d['ma5'] < 3.0).astype(float)
    return d

def main():
    t0 = time.time()
    today = time.strftime('%Y-%m-%d')
    print(f'绿箭 V8-Lottery (L50) — {today}')
    print('=' * 45)
    
    # 加载L50模型
    model = xgb.Booster()
    model.load_model(f'{MD}/us_v7_5_l50.json')
    with open(f'{MD}/us_v7_5_l50_report.json') as f:
        report = json.load(f)
    FEATS = report['features']
    
    print(f'模型特征: {len(FEATS)}个 (原v7.5=51)')
    
    # 加载数据
    df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
    df['date_str'] = df['date'].astype(str).str[:10]
    
    # 找最新日 — 先生成特征再dropna
    all_dates = sorted(df['date_str'].unique())
    latest_date = None
    for d in reversed(all_dates):
        subset = df[df['date_str'] == d]
        lt = gen_lottery_feats(subset)
        lt = lt[(lt['ma5'] >= 1.0) & (lt['ma5'] <= 10.0)].dropna(subset=FEATS)
        if len(lt) >= 20:
            latest_date = d
            break
    
    if latest_date is None:
        print('❌ 没有可用数据')
        return
    
    latest = df[df['date_str'] == latest_date].copy()
    latest = gen_lottery_feats(latest)
    
    # 只保留$1-10的票
    pool = latest[(latest['ma5'] >= 1.0) & (latest['ma5'] <= 10.0)].copy()
    pool = pool.dropna(subset=FEATS).reset_index(drop=True)
    
    print(f'候选池: {len(pool)}只 ($1-10低价股)')
    print(f'全市场共: {len(latest)}只')
    
    if len(pool) == 0:
        print('❌ 候选池为空')
        return
    
    # 评分
    X = pool[FEATS].values.astype(np.float32)
    prob = model.predict(xgb.DMatrix(X, feature_names=FEATS))
    
    # 排序
    results = []
    for i in range(len(pool)):
        results.append({
            'sym': pool.iloc[i]['sym'],
            'score': round(float(prob[i] * 100), 1),  # 0-1 概率映射到0-100
            'prob': round(float(prob[i]), 4),
            'price': float(pool.iloc[i]['ma5']),
        })
    results.sort(key=lambda x: -x['score'])
    
    # 统计
    scores_arr = np.array([r['score'] for r in results])
    print(f'\n评分范围: {scores_arr.min():.1f} - {scores_arr.max():.1f}')
    print(f'评分均值: {scores_arr.mean():.1f}')
    print(f'评分中位数: {np.median(scores_arr):.1f}')
    
    # 分位数
    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        idx = int(len(results) * (100-p) / 100)
        if idx >= len(results): idx = len(results)-1
        print(f'  Top {100-p}%: {results[idx]["score"]:.0f}分')
    
    # 买入信号 (彩票模型概率低, 阈值相应调低)
    buy_threshold = 30  # prob>0.30 = score>30
    watch_threshold = 15
    
    buys = [r for r in results if r['score'] >= buy_threshold]
    watches = [r for r in results if watch_threshold <= r['score'] < buy_threshold]
    
    print(f'\n🟢💪 彩票买入候选 (≥{buy_threshold}): {len(buys)}只')
    for r in buys[:20]:
        print(f'      {r["sym"]:6s} 评分{r["score"]:5.1f}  概率{r["prob"]:.3f}  ${r["price"]:>7.2f}')
    if len(buys) > 20:
        print(f'      ... 还有{len(buys)-20}只')
    
    print(f'\n🟡 关注 (≥{watch_threshold}, <{buy_threshold}): {len(watches)}只')
    for r in watches[:10]:
        print(f'      {r["sym"]:6s} 评分{r["score"]:5.1f}  概率{r["prob"]:.3f}  ${r["price"]:>7.2f}')
    
    # 市场热度
    top20_score = np.mean([r['score'] for r in results[:20]])
    print(f'\n彩票市场热度(Top20平均评分): {top20_score:.1f}')
    
    # 价格分层统计
    price_tiers = {'<$3': 0, '$3-5': 0, '$5-10': 0}
    for r in buys:
        p = r['price']
        if p < 3: price_tiers['<$3'] += 1
        elif p < 5: price_tiers['$3-5'] += 1
        else: price_tiers['$5-10'] += 1
    print(f'\n买入候选价格分布:')
    for tier, n in price_tiers.items():
        print(f'  {tier}: {n}只')
    
    # 保存
    output = {
        'date': today,
        'model': 'v7_5_lottery_l50',
        'model_desc': '$1-10低价彩票预测, 标签fwd_5d_ret>50%, 8月跨周期验证捕获率22.5%',
        'total_candidates': len(results),
        'top20_avg_score': round(float(top20_score), 1),
        'buy_signals': buys[:30],
        'watch_signals': watches[:20],
        'rules': {
            'buy_threshold': buy_threshold,
            'watch_threshold': watch_threshold,
            'price_filter': '$1-$10',
            'label': 'fwd_5d_ret > 50%',
            'features': len(FEATS),
        }
    }
    out_path = f'{DATA_DIR}/scored_v75_lottery_{today}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n已保存: {out_path}')
    print(f'⏱️ 耗时: {time.time()-t0:.1f}s')

if __name__ == '__main__':
    main()
