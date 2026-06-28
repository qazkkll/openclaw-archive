#!/usr/bin/env python3
"""
FinBERT Step 2 — 高效逐文件打标版
每读一个raw_cache JSON → 打标 → 立即存parquet，不全量加载。
"""
import json, sys, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd, numpy as np

DATA_DIR = Path('/home/hermes/.hermes/openclaw-archive/data/finbert_sentiment')
RAW_DIR = DATA_DIR / 'raw_cache'
PROGRESS_FILE = DATA_DIR / 'backfill_progress.json'

# 加载进度
done = set(json.load(open(PROGRESS_FILE))) if PROGRESS_FILE.exists() else set()

# 找未完成的文件
raw_files = sorted(RAW_DIR.glob('*.json'))
todo = [f for f in raw_files if f.stem not in done]
print(f"📊 FinBERT Step 2 (高效版)", flush=True)
print(f"  总文件: {len(raw_files)}, 已完成: {len(done)}, 待处理: {len(todo)}", flush=True)

if not todo:
    print("  无需处理", flush=True)
    sys.exit(0)

# 加载模型
print("  🧠 加载FinBERT...", flush=True)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
    print("  ✅ GPU加速", flush=True)
else:
    print("  ✅ CPU模式", flush=True)

# 逐文件处理
t0 = time.time()
total_scored = 0
total_files = 0
errors = 0

for fi, fpath in enumerate(todo):
    key = fpath.stem  # e.g. "AAPL:2022-01"
    try:
        articles = json.load(open(fpath))
    except Exception:
        errors += 1
        continue

    if not articles:
        done.add(key)
        total_files += 1
        continue

    # 提取文本
    texts = []
    for a in articles:
        title = a.get('title', '')
        text = a.get('text', '') or title
        combined = (title + '. ' + text).strip()
        if combined:
            texts.append(combined)
        else:
            texts.append(title)

    if not texts:
        done.add(key)
        total_files += 1
        continue

    # FinBERT打标 (batch=128, 内存友好)
    sentiments = []
    confidences = []
    batch_size = 128

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=512, padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        for j in range(len(batch)):
            pos, neg, neu = probs[j].tolist()
            sentiments.append(round(pos - neg, 4))
            confidences.append(round(max(probs[j]).item(), 4))

    # 构建DataFrame并保存
    df = pd.DataFrame(articles)
    df['sentiment'] = sentiments[:len(articles)]
    df['confidence'] = confidences[:len(articles)]

    # 解析日期
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True, errors='coerce')
    df = df.dropna(subset=['published_at'])
    df['year'] = df['published_at'].dt.year.astype(int)
    df['month'] = df['published_at'].dt.month.astype(int)

    # 按year/month/ticker分组保存
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
    total_files += 1

    # 每100个文件保存进度+打印
    if total_files % 100 == 0:
        with open(PROGRESS_FILE, 'w') as pf:
            json.dump(sorted(done), pf)
        elapsed = time.time() - t0
        rate = total_files / elapsed
        remaining = (len(todo) - total_files) / rate if rate > 0 else 0
        print(f"  [{total_files}/{len(todo)}] {total_scored}篇 | {elapsed/60:.1f}m | ETA {remaining/60:.1f}m", flush=True)

# 最终保存
with open(PROGRESS_FILE, 'w') as pf:
    json.dump(sorted(done), pf)

elapsed = time.time() - t0
print(f"\n✅ 完成: {total_files}文件, {total_scored}篇, {errors}错误, {elapsed/60:.1f}分钟", flush=True)
