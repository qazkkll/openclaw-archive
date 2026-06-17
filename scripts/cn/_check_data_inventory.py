import json, sys
sys.stdout.reconfigure(encoding='utf-8')
try:
    d = json.load(open('data/data_inventory.json', 'r', encoding='utf-8'))
    print(json.dumps(d, indent=2, ensure_ascii=False)[:3000])
except Exception as e:
    print(f"ERROR: {e}")
