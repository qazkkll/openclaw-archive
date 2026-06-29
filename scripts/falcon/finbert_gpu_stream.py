#!/usr/bin/env python3
"""
FinBERT 流式GPU打标 v2
逐文件加载→打标→存parquet，不全量加载到内存
GPU推理，batch_size=512
"""
import sys, os, json, time, warnings, gc
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd, numpy as np

BASE = Path('/home/hermes/.hermes/openclaw-archive')
DATA_DIR = BASE / 'data' / 'finbert_sentiment'
RAW_DIR = DATA_DIR / 'raw_cache'

def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    print("📊 FinBERT 流式GPU打标 v2", flush=True)
    
    # Load model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  🧠 加载模型到 {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model = model.to(device)
    model.eval()
    if torch.cuda.is_available():
        print(f"  ✅ GPU就绪: {torch.cuda.get_device_name(0)}, 显存: {torch.cuda.memory_allocated()/1024**2:.0f}MB", flush=True)
    else:
        print(f"  ⚠️ CPU模式", flush=True)
    
    # Load progress
    progress_file = DATA_DIR / 'backfill_progress_v2.json'
    done = set(json.load(open(progress_file))) if progress_file.exists() else set()
    print(f"  已完成: {len(done)}个文件", flush=True)
    
    # Get all raw files
    raw_files = sorted(RAW_DIR.glob('*.json'))
    todo = [f for f in raw_files if f.stem not in done]
    print(f"  待处理: {len(todo)}/{len(raw_files)}", flush=True)
    
    if not todo:
        print("  ✅ 全部完成！")
        return
    
    t0 = time.time()
    total_scored = 0
    batch_size = 512
    
    for fi, f in enumerate(todo):
        try:
            articles = json.load(open(f))
        except:
            done.add(f.stem)
            continue
        
        if not articles:
            done.add(f.stem)
            continue
        
        # Extract texts
        texts = []
        for a in articles:
            title = a.get('title', '') or ''
            text = a.get('text', '') or ''
            t = (title + '. ' + text).strip()
            if t:
                texts.append(t)
            else:
                texts.append('neutral')  # placeholder
        
        # Batch inference on GPU
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
        
        # Build DataFrame and save as parquet
        df = pd.DataFrame(articles)
        df['sentiment'] = sentiments[:len(df)]
        df['confidence'] = confidences[:len(df)]
        
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
        
        done.add(f.stem)
        total_scored += len(articles)
        
        # Progress report
        if (fi + 1) % 50 == 0 or (fi + 1) == len(todo):
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed * 60  # files/min
            remaining = (len(todo) - fi - 1) / rate if rate > 0 else 0
            gpu_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            print(f"  [{fi+1}/{len(todo)}] {total_scored}篇 | {rate:.0f}文件/分 | 剩{remaining:.0f}分 | GPU {gpu_mem:.0f}MB", flush=True)
        
        # Save progress every 100 files
        if (fi + 1) % 100 == 0:
            with open(progress_file, 'w') as pf:
                json.dump(sorted(done), pf)
            gc.collect()
    
    # Final save
    with open(progress_file, 'w') as pf:
        json.dump(sorted(done), pf)
    
    elapsed = time.time() - t0
    print(f"\n✅ 完成: {total_scored}篇, {elapsed/60:.1f}分钟", flush=True)

if __name__ == '__main__':
    main()
