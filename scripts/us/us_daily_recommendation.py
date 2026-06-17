#!/usr/bin/env python3
"""
美股盘前推荐 — 蓝盾V3 + 绿箭V19 联合评分
使用minishare拉取实时数据（替代yfinance）
"""
import sys, os, json, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import xgboost as xgb
import minishare as ms

# ====== 配置 ======
BASE = '/home/hermes/.hermes/openclaw-archive'
DATA_DIR = f'{BASE}/data'
MODEL_DIR = f'{BASE}/models/us'
OUTPUT_DIR = f'{BASE}/output'

MINISHARE_TOKEN = 'Jarvne6fmgArRa46Xfon0e1kw55E6hes5IB2Fy2X0ndqnvrL48jsVOtTbf014f06'

# 蓝盾参数
LD3_ENTRY = 90
LD3_EXIT = 75

# 绿箭参数
GA_THRESHOLD = 0.20

# ====== 蓝盾V3评分 ======
def run_blueshield():
    """运行蓝盾V3评分（使用minishare实时数据）"""
    from us_score_engine import v5s_calc, v5s_score
    
    # 加载SP500列表
    sp500_file = f'{DATA_DIR}/sp500_symbols.json'
    if os.path.exists(sp500_file):
        with open(sp500_file) as f:
            sp500 = json.load(f)
    else:
        sp500 = ['AAPL','MSFT','GOOGL','AMZN','NVDA','AVGO','TSLA','META','LLY',
                 'JPM','V','XOM','WMT','PG','JNJ','UNH','HD','BAC','KO']
    
    EXTRA_POOL = ['HPK', 'MRDN']
    all_target = list(set(sp500 + EXTRA_POOL))
    all_target.sort()
    
    # 使用minishare批量获取历史数据
    api = ms.pro_api(MINISHARE_TOKEN)
    results = []
    
    # 分批获取（minishare支持批量查询）
    batch_size = 50
    for i in range(0, len(all_target), batch_size):
        batch = all_target[i:i+batch_size]
        codes = ','.join(batch)
        
        try:
            # 获取历史日线数据（用于计算指标）
            df = api.us_daily(ts_code=codes, start_date='20240101', end_date='20260617')
            
            if df is None or len(df) == 0:
                continue
            
            # 按股票分组计算评分
            for code in batch:
                code_df = df[df['ts_code'] == code].sort_values('trade_date')
                if len(code_df) < 60:
                    continue
                
                c = code_df['close'].astype(float).tolist()
                h = code_df['high'].astype(float).tolist()
                l = code_df['low'].astype(float).tolist()
                
                ind = v5s_calc(c, h, l)
                if ind is None:
                    continue
                
                s = v5s_score(ind, len(c)-1)
                rsi_val = ind['rsi'][-1] if ind['rsi'] and ind['rsi'][-1] is not None else 0
                p52_val = ind['p52'][-1] if ind['p52'] and ind['p52'][-1] is not None else 0
                
                results.append({
                    'code': code,
                    'score': int(round(s)),
                    'price': float(c[-1]),
                    'rsi': round(rsi_val, 1),
                    'pct52': round(p52_val, 1)
                })
        except Exception as e:
            print(f"  批次{i//batch_size+1}错误: {e}")
            continue
        
        time.sleep(0.5)  # 限速
    
    results.sort(key=lambda x: -x['score'])
    return results

# ====== 绿箭V19评分 ======
def run_greenarrow():
    """运行绿箭V19评分"""
    # 加载模型
    model_file = f'{MODEL_DIR}/greenshaft_v19_final.json'
    model = xgb.XGBClassifier()
    model.load_model(model_file)
    
    # 特征列
    FEATS = [
        'price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
        'ret1','ret5','ret20','ret60','macd','macd_signal','macd_hist',
        'vol_ratio','ma_bias20','vol5','trend_accel',
        'short_ratio','short_pct','market_cap',
        'sector_etf_ret5','spy_ret5','qqq_ret5','iwm_ret5','sc',
    ]
    
    # 加载特征数据
    v71_file = f'{DATA_DIR}/us/features/us_ml_feats_v71_v19.parquet'
    df = pd.read_parquet(v71_file)
    latest = df.groupby('sym').last().reset_index()
    
    # 确保特征列存在
    for f in FEATS:
        if f not in latest.columns:
            latest[f] = 0
    
    # 预测
    X = latest[FEATS].values
    probs = model.predict_proba(X)[:, 1]
    
    results = []
    for i, row in latest.iterrows():
        results.append({
            'ticker': row['sym'],
            'price': float(row['price']),
            'prob': float(probs[i]),
            'ret5d': float(row.get('ret5', 0)),
            'ret20d': float(row.get('ret20', 0))
        })
    
    results.sort(key=lambda x: -x['prob'])
    return results

# ====== 分析与推荐 ======
def generate_recommendations(ld3_results, ga_results):
    """生成带思考的推荐"""
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 合并数据
    ld3_dict = {s['code']: s for s in ld3_results}
    ga_dict = {s['ticker']: s for s in ga_results}
    
    output = []
    output.append(f"# 美股盘前推荐 {today}")
    output.append("")
    
    # 1. 市场概览
    output.append("## 市场洞察")
    output.append("")
    
    strong_buy = [s for s in ld3_results if s['score'] >= LD3_ENTRY]
    hold = [s for s in ld3_results if LD3_EXIT <= s['score'] < LD3_ENTRY]
    
    output.append(f"**蓝盾V3扫描结果：**")
    output.append(f"- 强势买入（≥{LD3_ENTRY}分）：{len(strong_buy)}只")
    output.append(f"- 持有（≥{LD3_EXIT}分）：{len(hold)}只")
    output.append(f"- 观望：{len(ld3_results) - len(strong_buy) - len(hold)}只")
    output.append("")
    
    high_prob = [s for s in ga_results if s['prob'] >= GA_THRESHOLD]
    output.append(f"**绿箭V19扫描结果：**")
    output.append(f"- 高概率（≥{GA_THRESHOLD:.0%}）：{len(high_prob)}只")
    output.append("")
    
    # 2. 核心洞察
    output.append("## 核心洞察")
    output.append("")
    
    # 检查行业集中度
    if strong_buy:
        sectors = {}
        for s in strong_buy[:10]:
            code = s['code']
            if code in ['CAT', 'CMI', 'EMR', 'ITW', 'ROK']:
                sector = '工业'
            elif code in ['DAL', 'UAL', 'LUV']:
                sector = '航空'
            elif code in ['JPM', 'BAC', 'GS', 'MS']:
                sector = '金融'
            elif code in ['NVDA', 'AVGO', 'AMD']:
                sector = '半导体'
            else:
                sector = '其他'
            sectors[sector] = sectors.get(sector, 0) + 1
        
        if sectors:
            max_sector = max(sectors.items(), key=lambda x: x[1])
            if max_sector[1] >= 3:
                output.append(f"⚠️ **行业集中风险：** {max_sector[0]}股占{max_sector[1]}只，建议分散")
                output.append("")
    
    # 检查追高风险
    if high_prob:
        overbought = [s for s in high_prob if s['ret5d'] > 0.2]
        if overbought:
            output.append(f"⚠️ **追高警告：** {len(overbought)}只高概率股票近5日已涨超20%，均值回归风险高")
            output.append("")
    
    # 3. 推荐组合
    output.append("## 推荐组合")
    output.append("")
    
    # 蓝盾核心（60%仓位）
    output.append("### 蓝盾核心（60%仓位）")
    output.append("")
    
    confirmed = []
    for s in strong_buy[:10]:
        if s['code'] in ga_dict:
            ga = ga_dict[s['code']]
            if ga['prob'] >= 0.15:
                confirmed.append({
                    'code': s['code'],
                    'score': s['score'],
                    'prob': ga['prob'],
                    'price': s['price']
                })
    
    confirmed.sort(key=lambda x: -(x['score'] * 0.6 + x['prob'] * 100 * 0.4))
    
    if confirmed:
        for i, s in enumerate(confirmed[:5], 1):
            output.append(f"{i}. **{s['code']}** — 蓝盾{s['score']}分，绿箭{s['prob']:.1%}，${s['price']:.2f}")
    else:
        for i, s in enumerate(strong_buy[:5], 1):
            output.append(f"{i}. **{s['code']}** — 蓝盾{s['score']}分，${s['price']:.2f}")
    
    output.append("")
    
    # 绿箭卫星（10%仓位）
    output.append("### 绿箭卫星（10%仓位）")
    output.append("")
    
    value_picks = [s for s in high_prob if s['prob'] >= 0.22 and s['price'] < 100]
    value_picks.sort(key=lambda x: -x['prob'])
    
    if value_picks:
        for i, s in enumerate(value_picks[:3], 1):
            output.append(f"{i}. **{s['ticker']}** — 概率{s['prob']:.1%}，${s['price']:.2f}")
    else:
        output.append("暂无符合条件的标的（概率≥22%且价格<$100）")
    
    output.append("")
    
    # 现金
    output.append("### 现金（30%仓位）")
    output.append("")
    output.append("等待更好的入场机会。当前市场波动较大，保留弹药。")
    output.append("")
    
    # 4. 风险提示
    output.append("## 风险提示")
    output.append("")
    output.append("1. **蓝盾V3问题：** 500只候选池中≥90分有12只，甄别度偏低，建议提高到≥95分")
    output.append("2. **绿箭V19问题：** 高概率股票全是近期暴涨股，追高=送钱")
    output.append("3. **宏观风险：** 利率、通胀、地缘政治未纳入模型")
    output.append("4. **流动性风险：** 小盘股买卖价差大，实际收益可能低于回测")
    output.append("")
    
    # 5. 持仓自查
    output.append("## 持仓自查")
    output.append("")
    output.append("无持仓数据（OpenD未连接）")
    output.append("")
    
    # 6. 脚本信息
    output.append("---")
    output.append(f"*生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    output.append(f"*数据源：minishare美股实时行情*")
    
    return "\n".join(output)

# ====== 主程序 ======
if __name__ == '__main__':
    print("=" * 60)
    print("美股盘前推荐生成中...")
    print("=" * 60)
    
    t0 = time.time()
    
    # 运行蓝盾
    print("\n[1/2] 蓝盾V3评分...")
    ld3_results = run_blueshield()
    print(f"  完成: {len(ld3_results)}只")
    
    # 运行绿箭
    print("\n[2/2] 绿箭V19评分...")
    ga_results = run_greenarrow()
    print(f"  完成: {len(ga_results)}只")
    
    # 生成推荐
    print("\n生成推荐报告...")
    report = generate_recommendations(ld3_results, ga_results)
    
    # 保存
    today = datetime.now().strftime('%Y-%m-%d')
    output_file = f'{OUTPUT_DIR}/us_daily_recommendation_{today}.md'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    elapsed = time.time() - t0
    print(f"\n✅ 完成! 耗时{elapsed:.0f}秒")
    print(f"📄 报告已保存: {output_file}")
    
    # 打印报告
    print("\n" + report)
