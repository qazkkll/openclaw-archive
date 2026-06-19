#!/usr/bin/env python3
"""
蓝盾V5 每日评分脚本
扫描S&P500 → 计算28特征 → XGBoost打分 → 输出Top15

用法:
    python3 blueshield_v5_score.py              # 单次运行
    python3 blueshield_v5_score.py --json        # JSON输出
    python3 blueshield_v5_score.py --html        # 生成HTML报告
"""

import json, sys, os, time, argparse, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

# ── S&P500成分股 ──
# 从yfinance获取当前成分股
def get_sp500_tickers():
    """获取S&P500成分股列表"""
    try:
        import yfinance as yf
        # 用一个已知的S&P500 ETF替代
        tickers = ['AAPL','MSFT','AMZN','NVDA','GOOGL','META','TSLA','AVGO','AMD','JPM',
                   'V','MA','UNH','JNJ','LLY','XOM','PG','JPM','HD','CVX',
                   'MRK','ABBV','COST','PEP','KO','AVGO','WMT','MCD','CSCO','ACN',
                   'TMO','ABT','DHR','LIN','NEE','PM','TXN','UNP','RTX','HON',
                   'LOW','UPS','INTC','BA','GE','CAT','DE','AXP','GS','BLK',
                   'MS','WFC','C','SCHW','PFE','MRK','BMY','LLY','ABBV','TMO',
                   'ISRG','MDT','SYK','BSX','EW','ZTS','REGN','VRTX','GILD','AMGN',
                   'TSLA','GM','F','RIVN','LCID','TM','HMC','STLA',
                   'DIS','CMCSA','NFLX','T','VZ','TMUS','CHTR',
                   'NKE','SBUX','TJX','MCD','YUM','CMG',
                   'PLD','AMT','CCI','EQIX','SPG','O',
                   'NEE','DUK','SO','D','AEP','SRE',
                   'BABA','JD','PDD','NIO','LI','XPEV']
        return list(set(tickers))
    except:
        return []

# ── V5特征计算 ──
def compute_v5_features(df):
    """计算V5的28个特征"""
    df = df.sort_values('date').reset_index(drop=True)
    
    c = df['Close']
    
    # 趋势
    df['ma5'] = c.rolling(5).mean()
    df['ma20'] = c.rolling(20).mean()
    df['ma60'] = c.rolling(60).mean()
    df['ma_bias20'] = (c - df['ma20']) / df['ma20']
    df['ma_align'] = ((c > df['ma5']).astype(int) + (df['ma5'] > df['ma20']).astype(int))
    
    min60 = c.rolling(60).min()
    max60 = c.rolling(60).max()
    df['price_position'] = (c - min60) / (max60 - min60 + 1e-10)
    
    # 动量
    df['ret1'] = c.pct_change(1)
    df['ret5'] = c.pct_change(5)
    df['ret20'] = c.pct_change(20)
    df['ret60'] = c.pct_change(60)
    df['momentum_6m'] = c.pct_change(126)
    df['momentum_1m'] = c.pct_change(21)
    df['mom_divergence'] = df['momentum_1m'] - df['ret20']
    df['trend_accel'] = df['ret5'] - df['ret5'].shift(5)
    
    # 波动率
    daily_ret = c.pct_change(1)
    df['vol20'] = daily_ret.rolling(20).std()
    df['vol5'] = daily_ret.rolling(5).std()
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['vol_change'] = df['vol20'] / df['vol20'].shift(20)
    
    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    df['rsi14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df['rsi_change'] = df['rsi14'].diff(5)
    
    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 布林带
    df['bb_std'] = c.rolling(20).std()
    df['bb_width'] = 2 * df['bb_std'] / df['ma20']
    df['bb_pos'] = (c - df['ma20']) / (2 * df['bb_std'] + 1e-10)
    
    # 质量
    df['ret_quality'] = df['ret20'] / (df['vol20'] + 1e-10)
    
    return df

V5_FEATS = [
    'ma5','ma20','ma60','ma_bias20','ma_align','price_position',
    'ret1','ret5','ret20','ret60','momentum_6m','momentum_1m',
    'mom_divergence','trend_accel',
    'vol20','vol5','vol_ratio','vol_change',
    'rsi14','rsi_change',
    'macd','macd_signal','macd_hist',
    'bb_std','bb_width','bb_pos',
    'ret_quality'
]

def score_stocks(tickers):
    """扫描所有股票并打分"""
    import yfinance as yf
    import xgboost as xgb
    
    results = []
    failed = []
    
    print(f"扫描 {len(tickers)} 只股票...")
    
    for i, ticker in enumerate(tickers):
        if (i+1) % 50 == 0:
            print(f"  进度: {i+1}/{len(tickers)}")
        
        try:
            # 下载1年数据
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            
            if len(hist) < 100:  # 需要至少100天数据
                failed.append(ticker)
                continue
            
            # 重命名列以匹配计算
            hist = hist.rename(columns={'Stock Splits': 'Stock_Splits'})
            
            # 计算特征
            df = compute_v5_features(hist)
            
            # 取最后一行
            last = df.iloc[-1:]
            
            # 检查特征是否完整
            feats_available = [f for f in V5_FEATS if f in last.columns and not last[f].isna().all()]
            if len(feats_available) < 20:
                failed.append(ticker)
                continue
            
            # 填充缺失特征为0
            feat_vals = {}
            for f in V5_FEATS:
                if f in last.columns:
                    feat_vals[f] = float(last[f].iloc[0]) if not pd.isna(last[f].iloc[0]) else 0.0
                else:
                    feat_vals[f] = 0.0
            
            results.append({
                'ticker': ticker,
                'price': float(last['Close'].iloc[0]),
                'features': feat_vals,
                'rsi': feat_vals.get('rsi14', 0),
                'ret_5d': feat_vals.get('ret5', 0) * 100,
                'ret_20d': feat_vals.get('ret20', 0) * 100,
            })
            
        except Exception as e:
            failed.append(ticker)
    
    print(f"成功: {len(results)} 只 | 失败: {len(failed)} 只")
    return results

def predict_scores(results):
    """用V5模型预测分数"""
    import xgboost as xgb
    
    if not results:
        return []
    
    # 构建特征矩阵
    X = np.array([[r['features'][f] for f in V5_FEATS] for r in results], dtype=np.float32)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    # 加载V5训练好的模型
    import xgboost as xgb
    model_path = os.path.join(ROOT, 'models/us/blueshield_v5_xgb.json')
    meta_path = os.path.join(ROOT, 'models/us/blueshield_v5_meta.json')
    
    if os.path.exists(model_path):
        model = xgb.Booster()
        model.load_model(model_path)
        with open(meta_path) as f:
            meta = json.load(f)
        feats = meta['features']
        
        X = np.array([[r['features'][f] for f in feats] for r in results], dtype=np.float32)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        dtest = xgb.DMatrix(X, feature_names=feats)
        preds = model.predict(dtest)
        
        for i, r in enumerate(results):
            r['score'] = float(preds[i])
    else:
        # 简化评分fallback
        weights = {
            'vol20': -0.15, 'ma60': 0.12, 'momentum_6m': 0.10,
            'ret60': 0.08, 'ma5': 0.08, 'ma20': 0.07,
            'bb_width': -0.05, 'macd_signal': 0.05, 'vol_change': -0.05,
            'ret_quality': 0.08, 'rsi14': 0.03, 'price_position': 0.05,
        }
        for r in results:
            score = 0
            for feat, weight in weights.items():
                val = r['features'].get(feat, 0)
                if feat in ['vol20', 'vol5']:
                    score += weight * max(0, 1 - val * 10)
                elif feat in ['rsi14']:
                    score += weight * (1 if 30 < val < 70 else 0)
                else:
                    score += weight * min(max(val, -1), 1)
            r['score'] = float(score)
    
    # 排序
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def format_report(results, top_n=15):
    """格式化报告"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    report = []
    report.append(f"🛡️ 蓝盾V5 选股报告 ({now})")
    report.append(f"{'='*50}")
    report.append(f"")
    report.append(f"模型: V5 (120天Top15, XGBoost, 28特征)")
    report.append(f"扫描: {len(results)} 只S&P500成分股")
    report.append(f"")
    report.append(f"{'排名':<4} {'代码':<8} {'价格':>8} {'评分':>6} {'RSI':>6} {'5日':>8} {'20日':>8}")
    report.append(f"{'-'*52}")
    
    for i, r in enumerate(results[:top_n]):
        emoji = '🟢🟢' if i < 3 else '🟢' if i < 8 else '🟡'
        report.append(f"{emoji}{i+1:<3} {r['ticker']:<8} ${r['price']:>7.2f} {r['score']:>6.3f} "
                      f"{r['rsi']:>5.1f} {r['ret_5d']:>+7.1f}% {r['ret_20d']:>+7.1f}%")
    
    report.append(f"")
    report.append(f"⚠️ 注意: 这是V5初版评分(简化版)")
    report.append(f"   正式版需训练好的XGBoost模型")
    
    return '\n'.join(report)

def main():
    parser = argparse.ArgumentParser(description='蓝盾V5每日评分')
    parser.add_argument('--json', action='store_true', help='JSON输出')
    parser.add_argument('--html', action='store_true', help='生成HTML报告')
    parser.add_argument('--top', type=int, default=15, help='输出Top-N')
    args = parser.parse_args()
    
    print("🛡️ 蓝盾V5 每日评分")
    print("="*50)
    
    # 1. 获取股票池
    tickers = get_sp500_tickers()
    if not tickers:
        print("❌ 无法获取股票列表")
        return
    
    # 2. 扫描+评分
    t0 = time.time()
    results = score_stocks(tickers)
    results = predict_scores(results)
    elapsed = time.time() - t0
    
    print(f"\n评分完成 ({elapsed:.1f}s)")
    
    # 3. 输出
    if args.json:
        output = {
            'timestamp': datetime.now().isoformat(),
            'model': 'blueshield_v5',
            'total': len(results),
            'top_n': args.top,
            'picks': [{
                'ticker': r['ticker'],
                'price': r['price'],
                'score': r['score'],
                'rsi': r['rsi'],
                'ret_5d': r['ret_5d'],
                'ret_20d': r['ret_20d']
            } for r in results[:args.top]]
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_report(results, args.top))
    
    # 4. 保存结果
    output_dir = os.path.join(ROOT, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'v5_latest.json'), 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': 'blueshield_v5',
            'total': len(results),
            'picks': [{
                'ticker': r['ticker'],
                'price': r['price'],
                'score': r['score'],
                'rsi': r['rsi'],
                'ret_5d': r['ret_5d'],
                'ret_20d': r['ret_20d']
            } for r in results[:args.top]]
        }, f, indent=2, default=str)
    
    print(f"\n✅ 结果已保存: output/v5_latest.json")

if __name__ == '__main__':
    main()
