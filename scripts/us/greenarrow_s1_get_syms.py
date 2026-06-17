#!/usr/bin/env python3
"""绿箭 S1：获取SP500成分股 — 纯python, 无pandas read_html"""
import json, time, warnings, sys
warnings.filterwarnings('ignore')

# 直接从 Wikipedia API 获取 json 格式的成分股列表
import requests

url = 'https://en.wikipedia.org/w/api.php?action=parse&page=List_of_S%26P_500_companies&prop=text&format=json'
resp = requests.get(url, headers={'User-Agent': 'GreenArrow/1.0'}, timeout=30)
data = resp.json()

# 从返回的HTML中提取表格
html = data['parse']['text']['*']

# 找到第一个表格
table_start = html.find('<table')
table_end = html.find('</table>', table_start) + len('</table>')
table_html = html[table_start:table_end]

# 手动解析表格中的Symbol
# 格式: <tr><td>...<a href="...">AAPL</a>...</td>...
import re

# 找所有 `<tr>` 内的 `<td>` 
rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL)

syms = []
for row in rows[1:]:  # 跳过表头
    cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
    if cells:
        # 第一个cell是symbol
        cell = cells[0]
        # 找 <a href> 内的文本
        m = re.search(r'<a[^>]*>(.*?)</a>', cell)
        if m:
            sym = m.group(1).strip()
            if sym and not sym.startswith('^'):
                syms.append(sym)

syms = sorted(set(syms))
print(f'S1 解析: {len(syms)}只')

if len(syms) < 400:
    print(f'只拿到{len(syms)}只，改用备选方案')
    # 直接从yfinance获取S&P500 tickers
    try:
        syms = sorted(download_yf_sp500())
    except:
        pass

json.dump({'syms': syms, 'count': len(syms),
           'date': time.strftime('%Y-%m-%d')},
          open('/home/hermes/.hermes/openclaw-project/data/sp500_list.json', 'w'), indent=2)
print(f'\n保存: {len(syms)}只')
print(f'前10: {syms[:10]}')
