#!/usr/bin/env python3
"""
FinBERT 两步回填
Step 1: 并行拉FMP新闻 → 存原始JSON
Step 2: 批量FinBERT打标 → 存Parquet
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import pandas as pd, numpy as np
from dotenv import load_dotenv

load_dotenv(Path('/home/hermes/.hermes/openclaw-archive/.env'))

DATA_DIR = Path('/home/hermes/.hermes/openclaw-archive/data/finbert_sentiment')
FALCON_DIR = Path('/home/hermes/.hermes/openclaw-archive/data/falcon')
RAW_DIR = DATA_DIR / 'raw_cache'
RAW_DIR.mkdir(parents=True, exist_ok=True)
FMP_KEY = os.getenv('FMP_API_KEY', '')

def fetch_fmp(ticker, start, end):
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

def step1_fetch(start_date, end_date, workers=10):
    """Step 1: 并行拉FMP新闻，存原始JSON。"""
    tickers = sorted(pd.read_parquet(FALCON_DIR / 'features_v02.parquet', columns=['ticker'])['ticker'].unique())
    
    # 生成月份
    from dateutil.relativedelta import relativedelta
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    months = []
    cursor = start
    while cursor < end:
        m_end = min(cursor + relativedelta(months=1) - timedelta(days=1), end)
        months.append((cursor.strftime('%Y-%m-%d'), m_end.strftime('%Y-%m-%d')))
        cursor = m_end + timedelta(days=1)
    
    # 检查已完成的
    done = set()
    for f in RAW_DIR.glob('*.json'):
        done.add(f.stem)  # ticker:YYYY-MM
    
    tasks = []
    for m_start, m_end in months:
        mk = m_start[:7]
        for t in tickers:
            key = f"{t}:{mk}"
            if key not in done:
                tasks.append((t, m_start, m_end, key))
    
    print(f"📊 Step 1: 拉FMP新闻", flush=True)
    print(f"   Tickers: {len(tickers)}, Months: {len(months)}", flush=True)
    print(f"   已完成: {len(done)}, 待拉: {len(tasks)}", flush=True)
    print(f"   Workers: {workers}", flush=True)
    print(flush=True)
    
    t0 = time.time()
    completed = 0
    total_articles = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for t, ms, me, key in tasks:
            f = executor.submit(fetch_fmp, t, ms, me)
            futures[f] = key
        
        for future in as_completed(futures):
            key = futures[future]
            try:
                articles = future.result()
                if articles:
                    with open(RAW_DIR / f'{key}.json', 'w') as f:
                        json.dump(articles, f)
                    total_articles += len(articles)
            except:
                pass
            completed += 1
            if completed % 500 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                remaining = (len(tasks) - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{len(tasks)}] {total_articles}篇 ({elapsed/60:.0f}m, ETA {remaining/60:.0f}m)", flush=True)
    
    elapsed = time.time() - t0
    print(f"\n✅ Step 1完成: {completed} ticker-months, {total_articles}篇, {elapsed/60:.1f}分钟", flush=True)

def step2_score():
    """Step 2: 读原始JSON，批量FinBERT打标，存Parquet。"""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    print("📊 Step 2: FinBERT批量打标", flush=True)
    
    # 加载模型到GPU
    print("  🧠 加载FinBERT到GPU...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model = model.to(device)
    model.eval()
    print(f"  ✅ FinBERT就绪 ({device}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    
    # 读所有raw JSON
    raw_files = sorted(RAW_DIR.glob('*.json'))
    print(f"  文件: {len(raw_files)}", flush=True)
    
    # 加载已有进度
    progress_file = DATA_DIR / 'backfill_progress.json'
    done = set(json.load(open(progress_file))) if progress_file.exists() else set()
    
    t0 = time.time()
    total_scored = 0
    
    # 批量处理: 读所有文件，收集所有文章，批量打标
    all_articles = []
    file_map = {}  # file_idx -> [article_indices]
    
    for fi, f in enumerate(raw_files):
        key = f.stem
        if key in done:
            continue
        try:
            articles = json.load(open(f))
        except:
            continue
        if not articles:
            done.add(key)
            continue
        
        start_idx = len(all_articles)
        all_articles.extend(articles)
        end_idx = len(all_articles)
        file_map[fi] = (key, start_idx, end_idx)
    
    if not all_articles:
        print("  无需打标", flush=True)
        return
    
    print(f"  文章总数: {len(all_articles)}", flush=True)
    
    # 批量打标
    texts = [(a.get('title','') + '. ' + a.get('text','')).strip() for a in all_articles]
    batch_size = 512
    sentiments = []
    confidences = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu()
        for j in range(len(batch)):
            pos, neg, neu = probs[j].tolist()
            sentiments.append(round(pos - neg, 4))
            confidences.append(round(max(probs[j]).item(), 4))
        
        if (i + batch_size) % 5000 == 0:
            print(f"  打标: {i+len(batch)}/{len(texts)}", flush=True)
    
    print(f"  ✅ 打标完成: {len(sentiments)}篇", flush=True)
    
    # 按文件分组保存Parquet
    for fi, (key, si, ei) in file_map.items():
        articles = all_articles[si:ei]
        sents = sentiments[si:ei]
        confs = confidences[si:ei]
        
        df = pd.DataFrame(articles)
        df['sentiment'] = sents
        df['confidence'] = confs
        
        # 解析日期保存
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
        
        done.add(key)
        total_scored += len(articles)
    
    # 保存进度
    with open(progress_file, 'w') as f:
        json.dump(sorted(done), f)
    
    elapsed = time.time() - t0
    print(f"\n✅ Step 2完成: {total_scored}篇, {elapsed/60:.1f}分钟", flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['1', '2', 'both'], default='both')
    parser.add_argument('--start', default='2022-01-01')
    parser.add_argument('--end', default='2023-12-31')
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()
    
    if args.step in ('1', 'both'):
        step1_fetch(args.start, args.end, args.workers)
    if args.step in ('2', 'both'):
        step2_score()
