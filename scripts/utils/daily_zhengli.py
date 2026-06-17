#!/usr/bin/env python3
# zhengli 强制整理检查脚本
import os, json, datetime, sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def today_str():
    return datetime.date.today().isoformat()

def log(msg):
    print(msg)

def check_step1():
    path = os.path.join(WS, "memory", "%s.md" % today_str())
    exists = os.path.exists(path)
    lines = len(open(path,'r',encoding='utf-8',errors='replace').read().strip().split('\n')) if exists else 0
    return {"passed": exists, "detail": "memory/%s.md %s (%d lines)" % (today_str(), "EXISTS" if exists else "MISSING", lines)}

def check_step2():
    path = os.path.join(WS, "data", "decision_history.jsonl")
    if not os.path.exists(path):
        return {"passed": False, "detail": "decision_history.jsonl NOT FOUND"}
    lines = open(path,'r',encoding='utf-8',errors='replace').read().strip().split('\n')
    today_entries = 0
    no_subj = []
    for i, line in enumerate(lines):
        if not line.strip(): continue
        try:
            e = json.loads(line)
        except:
            no_subj.append("line %d: parse error" % (i+1))
            continue
        d = e.get("date", e.get("ts", ""))[:10]
        if d == today_str():
            today_entries += 1
            if not e.get("subjective", e.get("opinion", e.get("comment", ""))):
                no_subj.append("line %d: %s - no subjective" % (i+1, e.get("symbol","?")))
    return {"passed": True, "detail": "today %d entries, %d missing subjective" % (today_entries, len(no_subj)),
            "today_entries": today_entries, "no_subjective": no_subj}

def check_step3():
    path = os.path.join(WS, "data", "experience_log.jsonl")
    if not os.path.exists(path):
        return {"passed": False, "detail": "experience_log.jsonl NOT FOUND"}
    lines = open(path,'r',encoding='utf-8',errors='replace').read().strip().split('\n')
    total = len([l for l in lines if l.strip()])
    today_count = 0
    for l in reversed(lines):
        if not l.strip(): continue
        try:
            e = json.loads(l)
        except:
            continue
        if e.get("ts","")[:10] == today_str():
            today_count += 1
    return {"passed": today_count > 0, "detail": "total %d entries, today +%d" % (total, today_count),
            "today_entries": today_count}

def check_step4():
    path = os.path.join(WS, "data", "bug_log.md")
    if not os.path.exists(path):
        return {"passed": False, "detail": "bug_log.md NOT FOUND"}
    content = open(path,'r',encoding='utf-8',errors='replace').read()
    if today_str() in content:
        return {"passed": True, "detail": "bug_log.md contains %s update" % today_str()}
    return {"passed": False, "detail": "bug_log.md missing %s update" % today_str()}

def check_step5():
    path = os.path.join(WS, "docs", "A1_CHECKLIST.md")
    if not os.path.exists(path):
        return {"passed": "skip", "detail": "A1_CHECKLIST.md NOT FOUND"}
    content = open(path,'r',encoding='utf-8',errors='replace').read()
    if today_str() in content:
        return {"passed": True, "detail": "A1_CHECKLIST.md updated"}
    return {"passed": "manual", "detail": "A1_CHECKLIST.md not updated (skip if A1 unrelated today)"}

def check_step6():
    path = os.path.join(WS, "data", "experience_log.jsonl")
    issues = []
    if os.path.exists(path):
        lines = open(path,'r',encoding='utf-8',errors='replace').read().strip().split('\n')
        seen = set()
        for l in lines:
            if not l.strip(): continue
            try:
                e = json.loads(l)
                s = e.get("summary","")
                if s in seen:
                    issues.append("duplicate: %s" % s[:40])
                seen.add(s)
            except:
                pass
    return {"passed": "manual", "detail": "cleanup needs manual review. %s" % ("%d issues found" % len(issues) if issues else "no anomalies detected")}

def check_step7():
    return {"passed": "manual", "detail": "skill usage check needs manual review"}

def check_step8():
    return {"passed": "manual", "detail": "SKILL.md format check needs manual review"}

def main():
    today = today_str()
    log("="*60)
    log("  zhengli check report - %s" % today)
    log("="*60)

    checks = [
        ("1. diary", check_step1()),
        ("2. decision_history", check_step2()),
        ("3. experience_log", check_step3()),
        ("4. bug_log", check_step4()),
        ("5. A1_CHECKLIST", check_step5()),
        ("6. data cleanup", check_step6()),
        ("7. skill usage", check_step7()),
        ("8. SKILL.md format", check_step8()),
    ]

    all_ok = True
    results = []
    for name, r in checks:
        s = r["passed"]
        if s is True:
            icon = "[OK]"
        elif s == "skip":
            icon = "[SKIP]"
        elif s == "manual":
            icon = "[MANUAL]"
        else:
            icon = "[FAIL]"
            all_ok = False
        log("%s %s" % (icon.ljust(10), name))
        log("       %s" % r["detail"])
        results.append((name, r))

    log("")
    log("="*60)
    if all_ok:
        log("[OK] zhengli complete (%s)" % today)
    else:
        fails = [n for n,r in checks if r["passed"] is False]
        log("[WARN] unfinished: %s" % ", ".join(fails))
    log("="*60)

    # write summary
    summary_lines = ["# zhengli summary - %s\n" % today]
    for name, r in results:
        icon = "[OK]" if r["passed"] is True else ("[SKIP]" if r["passed"] == "skip" else ("[MANUAL]" if r["passed"] == "manual" else "[FAIL]"))
        summary_lines.append("%s %s: %s" % (icon, name, r["detail"]))
    summary_lines.append("\n" + "="*30)
    summary_lines.append("status: %s" % ("all complete" if all_ok else "incomplete"))
    summary = '\n'.join(summary_lines)

    log_path = os.path.join(WS, "data", "zhengli_%s.md" % today)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    log("[DONE] summary saved: data/zhengli_%s.md" % today)

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
