#!/usr/bin/env python3
"""
数据新鲜度检查 — 检查所有数据源是否更新
用法: python3 scripts/utils/check_data_fresh.py
"""
import os, json, time
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def check_file_freshness(filepath, max_age_days=2):
    """检查文件修改时间是否在允许范围内"""
    if not os.path.exists(filepath):
        return {'status': 'missing', 'message': f'文件不存在: {filepath}'}
    
    mtime = os.path.getmtime(filepath)
    age_days = (time.time() - mtime) / 86400
    
    if age_days > max_age_days:
        return {'status': 'stale', 'message': f'文件已过期 {age_days:.1f} 天 (阈值: {max_age_days}天)'}
    else:
        return {'status': 'ok', 'message': f'文件新鲜 ({age_days:.1f} 天前)'}

def check_tushare_api():
    """检查tushare API是否可用"""
    try:
        import tushare as ts
        with open(os.path.join(DATA_DIR, 'config', 'tushare.json')) as f:
            config = json.load(f)
        ts.set_token(config['token'])
        pro = ts.pro_api()
        df = pro.query('trade_cal', exchange='SSE', start_date='20260101', end_date='20260131')
        return {'status': 'ok', 'message': f'Tushare API可用, 返回{len(df)}行'}
    except Exception as e:
        return {'status': 'error', 'message': f'Tushare API失败: {str(e)[:100]}'}

def check_data_completeness():
    """检查数据完整性"""
    issues = []
    
    # 检查K线数据（注意：a_hist_10y.parquet实际是JSON格式）
    kline = os.path.join(DATA_DIR, 'cn', 'a_hist_10y.parquet')
    if os.path.exists(kline):
        try:
            with open(kline, 'r') as f:
                data = json.load(f)
            stock_count = len(data)
            if stock_count < 1000:
                issues.append(f'K线数据只有{stock_count}只股票')
        except:
            issues.append('K线数据格式异常')
    
    # 检查模型文件
    model = os.path.join(PROJECT_ROOT, 'models', 'cn', 'a1_layer3_xgb_10d.json')
    if not os.path.exists(model):
        issues.append('A2模型文件缺失')
    
    return issues

def main():
    print("="*60)
    print("🔍 数据新鲜度检查")
    print("="*60)
    
    results = []
    
    # 1. 检查关键文件
    print("\n📁 关键文件:")
    files_to_check = [
        ('K线数据', os.path.join(DATA_DIR, 'cn', 'a_hist_10y.parquet'), 3),
        ('资金流数据', os.path.join(DATA_DIR, 'moneyflow_pool.json'), 30),
        ('Tushare配置', os.path.join(DATA_DIR, 'config', 'tushare.json'), 365),
        ('美股SP500', os.path.join(DATA_DIR, 'us', 'us_hist_sp500_10y.parquet'), 7),
        ('美股YF', os.path.join(DATA_DIR, 'us', 'us_hist_yf_10y.parquet'), 7),
    ]
    
    for name, path, max_age in files_to_check:
        result = check_file_freshness(path, max_age)
        status_emoji = {'ok': '✅', 'stale': '⚠️', 'missing': '❌', 'error': '❌'}
        print(f"  {status_emoji.get(result['status'], '❓')} {name}: {result['message']}")
        results.append({'name': name, **result})
    
    # 2. 检查API
    print("\n🔗 API检查:")
    api_result = check_tushare_api()
    status_emoji = {'ok': '✅', 'error': '❌'}
    print(f"  {status_emoji.get(api_result['status'], '❓')} {api_result['message']}")
    results.append({'name': 'Tushare API', **api_result})
    
    # 3. 检查数据完整性
    print("\n📊 数据完整性:")
    issues = check_data_completeness()
    if issues:
        for issue in issues:
            print(f"  ⚠️ {issue}")
    else:
        print(f"  ✅ 数据完整")
    
    # 4. 检查moneyflow进度
    state_file = os.path.join(DATA_DIR, 'moneyflow_checkpoints', 'state.json')
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        completed = len(state.get('completed', []))
        print(f"\n🔄 资金流拉取: {completed}/5529 ({completed/5529*100:.1f}%)")
    
    # 总结
    print(f"\n{'='*60}")
    warnings = sum(1 for r in results if r['status'] in ('stale', 'missing', 'error'))
    if warnings:
        print(f"⚠️ 发现 {warnings} 个问题，需要关注")
    else:
        print(f"✅ 所有检查通过")
    
    return results

if __name__ == '__main__':
    main()
