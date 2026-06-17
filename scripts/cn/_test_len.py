import sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '/home/hermes/.hermes/openclaw-project/scripts')
import score_engine

data = json.load(open('/home/hermes/.hermes/openclaw-project/data\\us_hist_clean.parquet', 'r'))

ibkr = data['IBKR']
print(f'IBKR 总天数: {len(ibkr["c"])}')
ind_full = score_engine.v5s_calc(ibkr['c'], ibkr['h'], ibkr['l'])
print(f'全量 v5s_score: {score_engine.v5s_score(ind_full, -1)}')

# 截断测试
for ndays in [250, 500, 1000, 2000]:
    ind = score_engine.v5s_calc(ibkr['c'][:ndays], ibkr['h'][:ndays], ibkr['l'][:ndays])
    sc = score_engine.v5s_score(ind, -1)
    print(f'  {ndays:>5}天: v5s_score={sc}')
