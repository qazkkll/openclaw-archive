#!/usr/bin/env python3
"""
INDEX自动更新器 — 扫描项目生成 INDEX.md
用法: python3 scripts/utils/update_index.py
"""
import os, json, time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX_FILE = os.path.join(PROJECT_ROOT, 'INDEX.md')

def get_file_size(path):
    """返回人类可读的文件大小"""
    try:
        size = os.path.getsize(path)
        if size > 1024*1024*1024:
            return f"{size/1024/1024/1024:.1f}GB"
        elif size > 1024*1024:
            return f"{size/1024/1024:.1f}MB"
        elif size > 1024:
            return f"{size/1024:.0f}KB"
        else:
            return f"{size}B"
    except:
        return "?"

def get_mtime(path):
    """返回修改时间"""
    try:
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
    except:
        return "?"

def scan_data_assets():
    """扫描数据资产"""
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    assets = []
    
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.startswith('.') or f.endswith('.pyc'):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, PROJECT_ROOT)
            size = get_file_size(path)
            mtime = get_mtime(path)
            
            # 判断市场
            if '/cn/' in rel or 'a_hist' in f or 'moneyflow' in f:
                market = 'cn'
            elif '/us/' in rel or 'sp500' in f or 'yf_' in f:
                market = 'us'
            else:
                market = 'shared'
            
            assets.append({
                'path': rel,
                'name': f,
                'market': market,
                'size': size,
                'mtime': mtime
            })
    
    return sorted(assets, key=lambda x: x['market'] + x['name'])

def scan_models():
    """扫描模型文件"""
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    models = []
    
    for market in ['cn', 'us']:
        market_dir = os.path.join(models_dir, market)
        if not os.path.exists(market_dir):
            continue
        
        # 读取production.json
        prod_file = os.path.join(market_dir, 'production.json')
        production = {}
        if os.path.exists(prod_file):
            with open(prod_file) as f:
                production = json.load(f).get('active_models', {})
        
        for root, dirs, files in os.walk(market_dir):
            for f in files:
                if f.startswith('.') or f == 'production.json':
                    continue
                path = os.path.join(root, f)
                rel = os.path.relpath(path, PROJECT_ROOT)
                size = get_file_size(path)
                
                # 判断是否生产模型
                status = 'legacy'
                for model_id, model_info in production.items():
                    if model_info.get('file') == f or model_info.get('calibrator') == f:
                        status = model_info.get('status', 'production')
                        break
                
                models.append({
                    'path': rel,
                    'name': f,
                    'market': market,
                    'size': size,
                    'status': status
                })
    
    return sorted(models, key=lambda x: (x['market'], x['status'], x['name']))

def scan_scripts():
    """扫描脚本目录"""
    scripts_dir = os.path.join(PROJECT_ROOT, 'scripts')
    dirs = {}
    
    for item in sorted(os.listdir(scripts_dir)):
        item_path = os.path.join(scripts_dir, item)
        if os.path.isdir(item_path):
            count = len([f for f in os.listdir(item_path) if f.endswith('.py')])
            dirs[item] = count
    
    return dirs

def generate_index():
    """生成INDEX.md"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    data_assets = scan_data_assets()
    models = scan_models()
    scripts = scan_scripts()
    
    lines = []
    lines.append(f'# INDEX.md — 项目资产索引')
    lines.append(f'')
    lines.append(f'> 自动生成于 {now}，运行 `python3 scripts/utils/update_index.py` 刷新')
    lines.append(f'')
    
    # 项目概览
    lines.append(f'## 📊 项目概览')
    lines.append(f'')
    total_data_size = sum(1 for _ in data_assets)  # 简化
    lines.append(f'| 项目 | 数量 |')
    lines.append(f'|:--|:--|')
    lines.append(f'| 数据文件 | {len(data_assets)} |')
    lines.append(f'| 模型文件 | {len(models)} |')
    lines.append(f'| 脚本目录 | {len(scripts)} |')
    lines.append(f'| 生产模型 | {sum(1 for m in models if m["status"] == "production")} |')
    lines.append(f'')
    
    # 数据资产
    lines.append(f'## 📁 数据资产')
    lines.append(f'')
    current_market = None
    for asset in data_assets:
        if asset['market'] != current_market:
            current_market = asset['market']
            market_label = {'cn': 'A股', 'us': '美股', 'shared': '通用'}.get(current_market, current_market)
            lines.append(f'### {market_label}')
            lines.append(f'')
            lines.append(f'| 文件 | 大小 | 更新时间 |')
            lines.append(f'|:--|:--|:--|')
        lines.append(f'| `{asset["name"]}` | {asset["size"]} | {asset["mtime"]} |')
    lines.append(f'')
    
    # 模型清单
    lines.append(f'## 🤖 模型清单')
    lines.append(f'')
    current_market = None
    for model in models:
        if model['market'] != current_market:
            current_market = model['market']
            market_label = {'cn': 'A股', 'us': '美股'}.get(current_market, current_market)
            lines.append(f'### {market_label}')
            lines.append(f'')
            lines.append(f'| 模型 | 状态 | 大小 |')
            lines.append(f'|:--|:--|:--|')
        status_emoji = {'production': '🟢', 'testing': '🟡', 'legacy': '⚪'}.get(model['status'], '❓')
        lines.append(f'| `{model["name"]}` | {status_emoji} {model["status"]} | {model["size"]} |')
    lines.append(f'')
    
    # 脚本目录
    lines.append(f'## 📜 脚本目录')
    lines.append(f'')
    lines.append(f'| 目录 | 用途 | 文件数 |')
    lines.append(f'|:--|:--|:--|')
    dir_desc = {
        'cn': 'A股原始脚本',
        'us': '美股脚本',
        '_tmp': '临时/测试文件',
        'utils': '系统工具',
        'data': '数据拉取/更新',
        'train': '模型训练',
        'score': '评分/选股',
        'backtest': '回测',
    }
    for d, count in scripts.items():
        desc = dir_desc.get(d, '')
        lines.append(f'| `scripts/{d}/` | {desc} | {count} |')
    lines.append(f'')
    
    # 快速导航
    lines.append(f'## 🧭 快速导航')
    lines.append(f'')
    lines.append(f'### A股评分')
    lines.append(f'```bash')
    lines.append(f'cd ~/.hermes/openclaw-archive')
    lines.append(f'python3 scripts/score/a2_score_only.py  # A2评分')
    lines.append(f'```')
    lines.append(f'')
    lines.append(f'### A股训练')
    lines.append(f'```bash')
    lines.append(f'python3 scripts/train/a1_layer3_xgb.py  # A2训练')
    lines.append(f'```')
    lines.append(f'')
    lines.append(f'### 数据更新')
    lines.append(f'```bash')
    lines.append(f'python3 scripts/data/pull_moneyflow.py  # 资金流全量拉取')
    lines.append(f'```')
    lines.append(f'')
    lines.append(f'### 刷新索引')
    lines.append(f'```bash')
    lines.append(f'python3 scripts/utils/update_index.py  # 重建此文件')
    lines.append(f'```')
    
    # 写入文件
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ INDEX.md 已更新: {INDEX_FILE}")
    print(f"   数据: {len(data_assets)} 文件")
    print(f"   模型: {len(models)} 文件")
    print(f"   脚本: {sum(scripts.values())} 文件")

if __name__ == '__main__':
    generate_index()
