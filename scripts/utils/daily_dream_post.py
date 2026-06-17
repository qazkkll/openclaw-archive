#!/usr/bin/env python3
"""
Dream Post-Process — extract lessons/insights from today's memory/YYYY-MM-DD.md
and append new entries to data/experience_log.jsonl.

Trigger: after Dream daily (12:30 cron) — run as a chained agentTurn or standalone cron.

Logic:
1. Read today's memory/YYYY-MM-DD.md (source of truth for what happened today)
2. Parse for sections containing: lesson / 教训 / 经验 / 心得 / 纠正 / correction / mistake / insight / 注意 / 优化 / 改进
3. Cross-reference with existing experience_log.jsonl to avoid duplicates (by summary text)
4. Append only genuinely new entries
5. Write a brief report to stdout for cron audit

Usage:
    python scripts/dream_post_process.py [--date YYYY-MM-DD]
    (--date defaults to today if omitted)

No dependencies beyond stdlib.
"""
import json, os, re, sys
# GBK-safe stdout
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from datetime import datetime, timedelta, timezone

TZ = timezone(timedelta(hours=8))
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
EXPERIENCE_LOG = os.path.join(WORKSPACE, "data", "experience_log.jsonl")

# Keywords that signal a lesson/insight worth extracting
SIGNAL_PATTERNS = [
    r"(?i)(lesson|教训|经验|心得|纠正|correction|mistake|insight|心得|注意|优化|改进|发现|结论|根因|后果|教训|经验教训)",
    r"(?i)(prevention|避免|下次|以后|应该|不应该|需要|必须|不要)",
]

# Sections to skip (not about learnings)
SKIP_SECTIONS = [
    "今日焦点", "待办", "下步方向", "决策记录", "Todo", "In Progress", "Done",
]

def load_existing_entries():
    """Load all existing experience entries for dedup."""
    entries = []
    if os.path.exists(EXPERIENCE_LOG):
        with open(EXPERIENCE_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return entries

def load_memory_diary(date_str):
    """Load today's memory file."""
    path = os.path.join(MEMORY_DIR, f"{date_str}.md")
    if not os.path.exists(path):
        print(f"[dream_post_process] ❌ No memory file for {date_str}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_learnings(content, date_str):
    """
    Extract structured learning entries from the diary.
    Returns list of dicts with keys: summary, detail, prevention, tags.
    """
    results = []
    lines = content.split("\n")
    
    current_section = ""
    current_tags = []
    buffer_lines = []
    in_learning = False
    
    for line in lines:
        stripped = line.strip()
        
        # Track section headers
        if stripped.startswith("## "):
            current_section = stripped.lstrip("## ").strip()
            # Reset on section boundary
            if in_learning and buffer_lines:
                entry = _build_entry(buffer_lines, current_tags, date_str)
                if entry:
                    results.append(entry)
                buffer_lines = []
                current_tags = []
            in_learning = False
            continue
        
        # Detect signal lines
        has_signal = any(re.search(p, stripped) for p in SIGNAL_PATTERNS)
        
        if has_signal and not any(skip in current_section for skip in SKIP_SECTIONS):
            in_learning = True
            buffer_lines.append(stripped)
        elif in_learning:
            # Continue collecting context (next few lines after signal)
            if stripped and not stripped.startswith("##") and not stripped.startswith("---"):
                buffer_lines.append(stripped)
            else:
                # End of this learning block
                entry = _build_entry(buffer_lines, current_tags, date_str)
                if entry:
                    results.append(entry)
                buffer_lines = []
                current_tags = []
                in_learning = False
    
    # Flush remaining buffer
    if in_learning and buffer_lines:
        entry = _build_entry(buffer_lines, current_tags, date_str)
        if entry:
            results.append(entry)
    
    return results

def _build_entry(lines, tags, date_str):
    """Build a structured entry from signal lines."""
    if not lines:
        return None
    
    text = " ".join(lines)
    
    # Skip very short entries (noise)
    if len(text) < 20:
        return None
    
    # Determine type
    entry_type = "lesson"
    if re.search(r"(?i)(mistake|错误|bug|不该|不应该|别|别做)", text):
        entry_type = "mistake"
    elif re.search(r"(?i)(correction|纠正|修正|修复)", text):
        entry_type = "correction"
    elif re.search(r"(?i)(bug|缺陷|问题)", text):
        entry_type = "bug"
    
    # Extract prevention hint
    prevention = ""
    prev_match = re.search(r"(?i)(prevention|避免|下次|以后.*应该|必须|不要)[：:\s]*(.{10,200})", text)
    if prev_match:
        prevention = prev_match.group(2).strip()
    
    # Extract a concise summary (first sentence or up to 120 chars)
    summary = lines[-1] if len(lines) <= 3 else lines[-1]  # The "conclusion" line
    # Try to get the most meaningful sentence
    for line in reversed(lines):
        if len(line) > 30:
            summary = line
            break
    
    # Clean summary
    summary = re.sub(r"^- ", "", summary).strip()
    summary = re.sub(r"\s+", " ", summary)
    if len(summary) > 150:
        summary = summary[:147] + "..."
    
    return {
        "ts": datetime.now(TZ).strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "type": entry_type,
        "sender_id": "Andi Yang",
        "sender_name": "Andy Yang",
        "source_channel": "telegram",
        "summary": summary,
        "detail": text[:500] if len(text) > 500 else text,
        "prevention": prevention or "",
        "tags": tags or [],
        "source": f"dream_post_process@{date_str}",
    }

def deduplicate(new_entries, existing_entries):
    """Remove entries that are already in the log (by summary fuzzy match)."""
    if not existing_entries:
        return new_entries
    
    existing_summaries = set()
    for e in existing_entries:
        # Normalize for comparison
        norm = e.get("summary", "").strip().lower()
        norm = re.sub(r"\s+", " ", norm)
        existing_summaries.add(norm)
    
    result = []
    for entry in new_entries:
        norm = entry.get("summary", "").strip().lower()
        norm = re.sub(r"\s+", " ", norm)
        
        # Check for duplicates (exact or near-exact match)
        is_dup = False
        for existing in existing_summaries:
            # If one is a substring of the other, likely duplicate
            if norm in existing or existing in norm:
                is_dup = True
                break
            # Check if they share >70% of words
            norm_words = set(norm.split())
            existing_words = set(existing.split())
            if norm_words and existing_words:
                overlap = len(norm_words & existing_words)
                min_len = min(len(norm_words), len(existing_words))
                if min_len > 0 and overlap / min_len > 0.7:
                    is_dup = True
                    break
        
        if not is_dup:
            result.append(entry)
    
    return result

def append_to_log(entries):
    """Append new entries to experience_log.jsonl."""
    if not entries:
        print("[dream_post_process] ✏️ No new entries to add")
        return 0
    
    os.makedirs(os.path.dirname(EXPERIENCE_LOG), exist_ok=True)
    with open(EXPERIENCE_LOG, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"[dream_post_process] ✅ Appended {len(entries)} entries to {EXPERIENCE_LOG}")
    return len(entries)

def main():
    # Determine target date
    if "--date" in sys.argv:
        idx = sys.argv.index("--date")
        if idx + 1 < len(sys.argv):
            date_str = sys.argv[idx + 1]
        else:
            print("[dream_post_process] ❌ --date requires a value (YYYY-MM-DD)")
            return 1
    else:
        date_str = datetime.now(TZ).strftime("%Y-%m-%d")
    
    print(f"[dream_post_process] 🎯 Target date: {date_str}")
    
    # Load diary
    content = load_memory_diary(date_str)
    if content is None:
        return 1
    
    # Extract learnings
    learnings = extract_learnings(content, date_str)
    print(f"[dream_post_process] 📖 Extracted {len(learnings)} raw learning candidates")
    
    # Load existing entries for dedup
    existing = load_existing_entries()
    print(f"[dream_post_process] 📚 Existing experience log: {len(existing)} entries")
    
    # Deduplicate
    new_entries = deduplicate(learnings, existing)
    print(f"[dream_post_process] 🔍 After dedup: {len(new_entries)} new entries")
    
    # Append
    count = append_to_log(new_entries)
    
    # Summarize
    if count > 0:
        print("\n--- New entries ---")
        for e in new_entries:
            print(f"  [{e['type']}] {e['summary']}")
        print(f"--- {count} entries added ---")
    
    print(f"[dream_post_process] ✅ Done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
