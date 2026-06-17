"""
Provider Health Check & Recovery Automation
Based on error-recovery-automation skill patterns.

Checks:
1. DeepSeek balance (low balance alert)
2. Model availability (quick API call)
3. Gateway status

Recovery:
- Gateway restart if model unavailable but gateway running

Usage:
  python scripts/provider_health_check.py
  python scripts/provider_health_check.py --recover   # attempt recovery on failure

Returns exit code 0 = healthy, 1 = degraded, 2 = failed
"""

import os
import sys
import json
import time
import requests
import subprocess
from datetime import datetime

# --- Config ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BALANCE_WARN_THRESHOLD_CNY = 50.0  # Alert when balance below this
GATEWAY_URL = "http://127.0.0.1:18789"
LOG_FILE = os.path.expanduser("~/.openclaw/provider_health.log")
ATTEMPT_RECOVERY = "--recover" in sys.argv

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def check_balance():
    """Check DeepSeek account balance"""
    if not DEEPSEEK_API_KEY:
        log("No DEEPSEEK_API_KEY found", "WARN")
        return True  # skip check if no key
    
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Accept": "application/json"}
    try:
        r = requests.get("https://api.deepseek.com/user/balance", headers=headers, timeout=10)
        data = r.json()
        available = data.get("is_available", False)
        for bi in data.get("balance_infos", []):
            bal = float(bi["total_balance"])
            if bi["currency"] == "CNY" and bal > 0:
                log(f"Balance: CNY {bal:.2f}")
                if bal < BALANCE_WARN_THRESHOLD_CNY:
                    log(f"LOW BALANCE WARNING: Only CNY {bal:.2f} remaining!", "ALERT")
                    return False
        if not available:
            log("DeepSeek account is NOT available", "ERROR")
            return False
        log("Balance check: OK")
        return True
    except requests.RequestException as e:
        log(f"Balance check failed: {e}", "ERROR")
        return False

def check_model():
    """Test model availability with a minimal API call"""
    if not DEEPSEEK_API_KEY:
        log("No DEEPSEEK_API_KEY found", "WARN")
        return True
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1
    }
    try:
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        if r.status_code == 200:
            log("Model test: OK")
            return True
        elif r.status_code == 402:
            log("Model test: FAILED - Insufficient balance (402)", "ERROR")
            return False
        elif r.status_code == 429:
            log("Model test: FAILED - Rate limited (429)", "ERROR")
            return False
        elif r.status_code == 503:
            log("Model test: FAILED - Service unavailable (503) - possibly in cooldown", "ERROR")
            return False
        else:
            log(f"Model test: FAILED - HTTP {r.status_code}: {r.text[:100]}", "ERROR")
            return False
    except requests.RequestException as e:
        log(f"Model test: FAILED - Connection error: {e}", "ERROR")
        return False

def _run_openclaw(args):
    """Run openclaw CLI, works on Windows with PowerShell/cmd"""
    cmd = " ".join(["openclaw"] + args)
    try:
        result = subprocess.run(
            ["powershell", "-Command", cmd],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_gateway():
    """Check if gateway is running"""
    ok, out, err = _run_openclaw(["gateway", "status"])
    if ok:
        log("Gateway: running")
        return True
    else:
        log(f"Gateway: NOT running ({err[:100]})", "ERROR")
        return False

def recover_gateway():
    """Restart gateway to clear cooldown states"""
    log("Attempting gateway restart...", "RECOVERY")
    ok, out, err = _run_openclaw(["gateway", "restart"])
    if ok:
        log("Gateway restart command succeeded, waiting 10s...", "RECOVERY")
        time.sleep(10)
        return True
    else:
        log(f"Gateway restart failed: {err[:200]}", "ERROR")
        return False

def main():
    log("=== Provider Health Check Started ===")
    
    checks = {
        "balance": False,
        "model": False,
        "gateway": False
    }
    
    checks["balance"] = check_balance()
    checks["gateway"] = check_gateway()
    checks["model"] = check_model()
    
    healthy = all(checks.values())
    
    # Report results
    status = "HEALTHY" if healthy else "DEGRADED"
    log(f"Summary: {status}")
    for name, ok in checks.items():
        log(f"  {name}: {'OK' if ok else 'FAIL'}")
    
    # Attempt recovery if flagged and not healthy
    if not healthy and ATTEMPT_RECOVERY:
        log("Recovery mode enabled, attempting fixes...", "RECOVERY")
        
        if not checks["gateway"]:
            recover_gateway()
            checks["gateway"] = check_gateway()
        
        if not checks["model"] and checks["gateway"]:
            log("Model failing but gateway OK - restarting gateway to clear cooldown", "RECOVERY")
            recover_gateway()
            time.sleep(5)
            checks["model"] = check_model()
            checks["balance"] = check_balance()
        
        healthy = all(checks.values())
        log(f"After recovery: {'HEALTHY' if healthy else 'STILL FAILING'}")
    
    # Write status file for cron to read
    status_data = {
        "timestamp": datetime.now().isoformat(),
        "checks": {k: v for k, v in checks.items()},
        "healthy": healthy,
        "attempted_recovery": ATTEMPT_RECOVERY
    }
    status_path = os.path.expanduser("~/.openclaw/provider_health_status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)
    
    log(f"=== Provider Health Check Completed: {status} ===")
    
    return 0 if healthy else (1 if any(checks.values()) else 2)

if __name__ == "__main__":
    sys.exit(main())
