"""
跨 Session Token 用量汇总
读取 .usage-cost-cache.json，输出全局统计
"""
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CACHE_PATH = r"C:\Users\admin\.openclaw\agents\main\sessions\.usage-cost-cache.json"

def main():
    try:
        data = json.load(open(CACHE_PATH, encoding="utf-8"))
    except FileNotFoundError:
        print("❌ 找不到 usage-cost-cache.json")
        sys.exit(1)

    files = data.get("files", {})
    if not files:
        print("❌ 缓存为空，没有历史用量记录")
        sys.exit(0)

    total_cost = 0.0
    total_input = 0
    total_output = 0
    by_model = {}
    by_provider = {}
    sessions = []

    for fp, info in files.items():
        entries = info.get("usageEntries", [])
        session_cost = 0.0
        session_tokens = 0
        for e in entries:
            cost = e.get("costUsd", 0) or 0
            inp = e.get("inputTokens", 0) or 0
            out = e.get("outputTokens", 0) or 0
            model = e.get("model", "unknown")
            provider = e.get("provider", "unknown")
            total_cost += cost
            total_input += inp
            total_output += out
            session_cost += cost
            session_tokens += inp + out
            by_model[model] = by_model.get(model, 0) + cost
            by_provider[provider] = by_provider.get(provider, 0) + cost
        if session_cost > 0:
            sid = fp.rsplit("\\", 1)[-1].split(".")[0][:12]
            sessions.append((sid, session_cost, session_tokens, len(entries)))

    print("=" * 50)
    print("[Token Usage Summary]")
    print("=" * 50)
    print(f"🔢 总 Session 数: {len(sessions)}")
    print(f"📥 总输入 tokens: {total_input:,}")
    print(f"📤 总输出 tokens: {total_output:,}")
    print(f"💰 总费用: ${total_cost:.4f}")
    print()
    print("\n[By Provider]")
    for p, c in sorted(by_provider.items(), key=lambda x: -x[1]):
        print(f"   {p}: ${c:.4f}")
    print()
    print("\n[By Model]")
    for m, c in sorted(by_model.items(), key=lambda x: -x[1]):
        print(f"   {m}: ${c:.4f}")
    print()
    print("\n[Top 10 Costliest Sessions]")
    sessions.sort(key=lambda x: -x[1])
    for sid, cost, tokens, n in sessions[:10]:
        print(f"   {sid}… | ${cost:.4f} | {tokens:,} tok | {n} calls")

if __name__ == "__main__":
    main()
