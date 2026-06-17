"""
V7 一键跑通：合并特征 → 训练模型
在后台下载完成后执行，不用cron
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

fund_path = r'/home/hermes/.hermes/openclaw-archive/data\us_fundamentals_v7_raw.json'
v3_path = r'/home/hermes/.hermes/openclaw-archive/scripts/system\us_ml_feats_v3_dated.parquet'
v7_path = r'/home/hermes/.hermes/openclaw-archive/scripts/system\us_ml_feats_v7_full.parquet'

# 先等下载完成（最多等10分钟）
max_wait = 600
waited = 0
while waited < max_wait:
    if os.path.exists(fund_path):
        with open(fund_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        if len(d) >= 2300:
            has_pb = sum(1 for v in d.values() if v and v.get('pb') is not None)
            if has_pb >= 2200:
                print(f'下载完成: {len(d)} 只, pb有值: {has_pb}')
                break
    print(f'等待下载... ({len(d) if os.path.exists(fund_path) else 0}/2435)')
    time.sleep(30)
    waited += 30
else:
    print('超时等待，用当前数据继续')
    with open(fund_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    print(f'当前: {len(d)} 只')

# 1. 合并特征
print('\n=== Step 1: 合并特征 ===')
ret = os.system(f'python scripts/us_v7_s3_merge_features.py')
if ret != 0:
    print('合并失败，退出')
    sys.exit(1)

# 2. 训练
print('\n=== Step 2: 训练XGBoost ===')
ret = os.system(f'python scripts/us_v7_s4_train_xgb.py')
if ret != 0:
    print('训练失败，退出')
    sys.exit(1)

# 3. 检查结果
report_path = r'/home/hermes/.hermes/openclaw-archive/data\models\us_xgb_v7_report.json'
if os.path.exists(report_path):
    with open(report_path, 'r', encoding='utf-8') as f:
        rpt = json.load(f)
    print(f'\n✅ V7 训练完成')
    print(f'  验证集 AUC: {rpt.get(\"val_auc\",\"?\")}')
    print(f'  验证集 Acc: {rpt.get(\"val_accuracy\",\"?\")}')
    print(f'  特征数: {rpt.get(\"num_features\",\"?\")}')
    print(f'  模型: {rpt.get(\"model_path\",\"?\")}')

print('\n=== 全部完成 ===')
