# 更新data_inventory.json中的美股特征文件描述
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')

path = r'/home/hermes/.hermes/openclaw-archive/data\data_inventory.json'
d = json.load(open(path, 'r', encoding='utf-8'))

# 看看结构
print(json.dumps(d, indent=2, ensure_ascii=False)[:2000])
