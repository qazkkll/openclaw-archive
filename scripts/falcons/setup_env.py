#!/usr/bin/env python3
"""Setup .env file for Falcons project."""
import os
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent.parent / ".env"

# Read credentials from user input or environment
api_key = os.environ.get("ALPACA_KEY", "")
secret = os.environ.get("ALPACA_SECRET", "")

if not api_key:
    api_key = input("Enter Alpaca API Key: ").strip()
if not secret:
    secret = input("Enter Alpaca Secret Key: ").strip()

with open(env_path, "w") as f:
    f.write(f"APCA_API_KEY_ID={api_key}\n")
    f.write(f"APCA_API_SECRET_KEY={secret}\n")
    f.write("APCA_API_BASE_URL=https://paper-api.alpaca.markets\n")

print(f"✅ .env written to {env_path}")
print(f"   Key: {api_key[:8]}...{api_key[-4:]}")
