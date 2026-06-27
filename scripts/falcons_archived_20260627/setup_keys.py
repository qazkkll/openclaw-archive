#!/usr/bin/env python3
"""
Falcons API Key Setup
Usage: python3 setup_keys.py <ALPACA_KEY> <ALPACA_SECRET> [MASSIVE_KEY]
"""
import sys
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent.parent / ".env"

if len(sys.argv) < 3:
    print("Usage: python3 setup_keys.py <ALPACA_KEY> <ALPACA_SECRET> [MASSIVE_KEY]")
    sys.exit(1)

ak = sys.argv[1]
asec = sys.argv[2]
mkey = sys.argv[3] if len(sys.argv) > 3 else ""

with open(env_path, "w") as f:
    f.write("APCA_API_KEY_ID=" + ak + "\n")
    f.write("APCA_API_SECRET_KEY=" + asec + "\n")
    f.write("APCA_API_BASE_URL=https://paper-api.alpaca.markets\n")
    if mkey:
        f.write("MASSIVE_API_KEY=" + mkey + "\n")

# Verify
import requests
headers = {"APCA-API-KEY-ID": ak, "APCA-API-SECRET-KEY": asec}
r = requests.get("https://paper-api.alpaca.markets/v2/account", headers=headers)
if r.status_code == 200:
    d = r.json()
    print("OK Alpaca connected")
    print("Cash:", d.get("cash"))
    print("Equity:", d.get("equity"))
    print("Status:", d.get("status"))
else:
    print("FAIL", r.status_code, r.text)
