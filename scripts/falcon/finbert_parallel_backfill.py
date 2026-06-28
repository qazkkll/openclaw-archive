#!/usr/bin/env python3
"""
FinBERT 并行回填 — 优化版
- ThreadPoolExecutor 并行API调用 (10线程)
- 只用FMP新闻源 (Massive/Polygon无数据)
- FinBERT批量打标
- 断点续传
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts/falcon')
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd, numpy as np
from dotenv import load_dotenv

load_dotenv(Path('/home/hermes/.hermes/openclaw-archive/.env'))

PROJECT_ROOT = Path('/home/hermes/.hermes/openclaw-archive')
DATA_DIR = PROJECT_ROOT / 'data' / 'finbert_sentiment'
FALCON_DIR = PROJECT_ROOT / 'data' / 'falcon'
PROGRESS_FILE = DATA_DIR / 'backfill_progress.json'

FMP_KEY = os.getenv('FMP_API_KEY', '')

# FinBERT (lazy load)
_tokenizer = None
_model = None

def load_finbert():
    global _tokenizer, _model
    if _tokenizer is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("  🧠 加载 FinBERT...", flush=True)
        _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _model.eval()
        print("  ✅ FinBERT 就绪", flush=True)
    return _tokenizer, _model

def score_texts(texts):
    """批量打标。"""
    import torch
    tokenizer, model = load_finbert()
    results = []
    for i in range(0, len(texts), 64):
        batch = texts[i:i+64]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        for j in range(len(batch)):
            pos, neg, neu = probs[j].tolist()
            results.append((round(pos - neg, 4), round(max(probs[j]).item(), 4)))
    return results

def fetch_fmp(ticker, start, end):
    """FMP新闻API。"""
    import urllib.request
    url = f"https://financialmodelingprep.com/stable/news/stock?symbols={ticker}&from={start}&to={end}&limit=100&apikey={FMP_KEY}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        if not isinstance(data, list): return []
        articles = []
        for a in data:
            title = a.get('title', '')
            if not title: continue
            articles.append({
                'ticker': ticker, 'title': title,
                'text': a.get('text', '') or title,
                'published_at': a.get('publishedDate', ''),
                'publisher': a.get('publisher', ''),
                'source': 'fmp',
            })
        return articles
    except:
        return []

def load_progress():
    if PROGRESS_FILE.exists():
        return set(json.load(open(PROGRESS_FILE)))
    return set()

def save_progress(done):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(sorted(done), f)

def save_parquet(df):
    """按year/month/ticker保存。"""
    if df.empty: return
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True, errors='coerce')
    df = df.dropna(subset=['published_at'])
    df['year'] = df['published_at'].dt.year.astype(int)
    df['month'] = df['published_at'].dt.month.astype(int)
    for (year, month, ticker), group in df.groupby(['year', 'month', 'ticker']):
        out_dir = DATA_DIR / f"year={year}" / f"month={month:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"ticker={ticker}.parquet"
        group = group[['ticker','published_at','title','text','source','publisher','sentiment','confidence']]
        if out_file.exists():
            existing = pd.read_parquet(out_file)
            group = pd.concat([existing, group], ignore_index=True)
            group = group.drop_duplicates(subset=['title','published_at'])
        group.to_parquet(out_file, index=False)

def process_ticker_month(ticker, start, end, key):
    """处理单个ticker-month: 拉新闻→打标→保存。"""
    articles = fetch_fmp(ticker, start, end)
    if not articles:
        return key, 0
    
    df = pd.DataFrame(articles)
    df['title_norm'] = df['title'].str.lower().str.strip()
    df = df.drop_duplicates(subset=['ticker', 'title_norm'])
    df = df.drop(columns=['title_norm'])
    
    if df.empty:
        return key, 0
    
    # FinBERT打标
    texts = (df['title'].fillna('') + '. ' + df['text'].fillna('')).tolist()
    scores = score_texts(texts)
    df['sentiment'] = [s[0] for s in scores]
    df['confidence'] = [s[1] for s in scores]
    
    save_parquet(df)
    return key, len(df)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2016-01-01')
    parser.add_argument('--end', default='2023-12-31')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    
    # 加载tickers
    tickers = sorted(pd.read_parquet(FALCON_DIR / 'features_v02.parquet', columns=['ticker'])['ticker'].unique())
    done = load_progress()
    
    # 生成月份列表
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta
    
    start = datetime.strptime(args.start, '%Y-%m-%d')
    end = datetime.strptime(args.end, '%Y-%m-%d')
    months = []
    cursor = start
    while cursor < end:
        m_end = min(cursor + relativedelta(months=1) - timedelta(days=1), end)
        months.append((cursor.strftime('%Y-%m-%d'), m_end.strftime('%Y-%m-%d')))
        cursor = m_end + timedelta(days=1)
    
    # 生成任务列表(跳过已完成的)
    tasks = []
    for m_start, m_end in months:
        month_key = m_start[:7]
        for t in tickers:
            key = f"{t}:{month_key}"
            if key not in done:
                tasks.append((t, m_start, m_end, key))
    
    print(f"📊 FinBERT并行回填", flush=True)
    print(f"   Tickers: {len(tickers)}", flush=True)
    print(f"   Months: {len(months)} ({args.start} → {args.end})", flush=True)
    print(f"   已完成: {len(done)}", flush=True)
    print(f"   待处理: {len(tasks)}", flush=True)
    print(f"   Workers: {args.workers}", flush=True)
    print(flush=True)
    
    # 预加载FinBERT
    load_finbert()
    
    t0 = time.time()
    completed = 0
    total_articles = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        batch_size = args.workers * 2
        
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start:batch_start + batch_size]
            
            for t, m_start, m_end, key in batch:
                f = executor.submit(fetch_fmp, t, m_start, m_end)
                futures[f] = (t, m_start, m_end, key)
            
            for future in as_completed(futures):
                t, m_start, m_end, key = futures[future]
                try:
                    articles = future.result()
                    if articles:
                        df = pd.DataFrame(articles)
                        df['title_norm'] = df['title'].str.lower().str.strip()
                        df = df.drop_duplicates(subset=['ticker', 'title_norm'])
                        df = df.drop(columns=['title_norm'])
                        
                        if not df.empty:
                            texts = (df['title'].fillna('') + '. ' + df['text'].fillna('')).tolist()
                            scores = score_texts(texts)
                            df['sentiment'] = [s[0] for s in scores]
                            df['confidence'] = [s[1] for s in scores]
                            save_parquet(df)
                            total_articles += len(df)
                except:
                    pass
                
                done.add(key)
                completed += 1
                
                if completed % 100 == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    remaining = (len(tasks) - completed) / rate if rate > 0 else 0
                    save_progress(done)
                    print(f"  [{completed}/{len(tasks)}] {total_articles}篇 ({elapsed/60:.0f}m, ETA {remaining/60:.0f}m)", flush=True)
            
            futures.clear()
    
    save_progress(done)
    elapsed = time.time() - t0
    print(f"\n✅ 完成: {completed} ticker-months, {total_articles}篇, {elapsed/60:.1f}分钟", flush=True)

if __name__ == '__main__':
    main()
