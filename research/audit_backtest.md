# 回测计算审计报告


## 1. backtest_report.json

- final_value: 675,828.71
- total_return_pct (原始): 575.83%
- cagr_pct (原始): 129.38%
- test_dates: 580天
- sharpe: 1.57
- max_dd: -53.22%
- config: {'top_k': 10, 'stop_loss': -0.15, 'rebalance': 5, 'max_hold': 60}

**审计:**
- 正确总收益: 575.83%
- 正确CAGR: 232.83%
- 原始CAGR: 129.38%
- **CAGR偏差: 103.5个百分点**

## 2. production_results.json

Walk-Forward各fold:

- Fold 1: IC=0.0664, LS=0.0210, Ann=-1.00%, DD=-0.98%
- Fold 2: IC=0.1055, LS=0.0463, Ann=5.66%, DD=-0.96%
- Fold 3: IC=0.0793, LS=0.0328, Ann=-0.43%, DD=-0.60%
- Fold 4: IC=0.1207, LS=0.0536, Ann=150.03%, DD=-0.82% **异常**
- Fold 5: IC=0.0916, LS=0.0283, Ann=-1.00%, DD=-0.99%
- Fold 6: IC=0.0911, LS=0.0366, Ann=21633.60%, DD=-0.64% **异常**

原始summary年化: 3631.14%
去掉异常值后均值年化: 30.65%

## 3. plan_ab_results.json


**PLAN_A:**
- IC: 0.0638, Rank_IC: 0.0635
- Long-Short: 0.0201
- 原始年化: 600.92%
- **年化异常，不可信**
  - Fold 1: Ann=-1.00%, IC=0.0639
  - Fold 2: Ann=122.43%, IC=0.0514 **异常**
  - Fold 3: Ann=24.07%, IC=0.0653
  - Fold 4: Ann=-1.00%, IC=0.0886
  - Fold 5: Ann=2860.07%, IC=0.0497 **异常**

**PLAN_B:**
- IC: 0.0375, Rank_IC: 0.0587
- Long-Short: 0.0154
- 原始年化: 214.98%
- **年化异常，不可信**
  - Fold 1: Ann=-0.96%, IC=0.0767
  - Fold 2: Ann=824.79%, IC=0.0614 **异常**
  - Fold 3: Ann=-0.45%, IC=-0.0318
  - Fold 4: Ann=-1.00%, IC=0.0624
  - Fold 5: Ann=252.53%, IC=0.0188 **异常**

## 4. Walk-Forward稳定性

- WF1 2020-21: corr=0.169, D1_top10=38.1%, spread=6.09% [OK]
- WF2 2021-22: corr=0.070, D1_top10=27.6%, spread=2.03% [FAIL]
- WF3 2022-23: corr=0.140, D1_top10=36.6%, spread=5.09% [OK]
- WF4 2023-24: corr=0.096, D1_top10=19.6%, spread=2.30% [WARN]

## 5. 总结

| 来源 | 原始指标 | 可信度 |
|------|---------|--------|
| backtest_report | CAGR 129% | 需验证 |
| production_results | 年化3631% | 不可信 |
| plan_ab | A=600% B=215% | 不可信 |
| a3 WF | WF2 corr=0.07 | 不稳定 |
| cn-alpha-v1.0 paper | 13% Sharpe0.72 | 唯一真实 |
| cn-alpha-v1.1 | 32.3% Sharpe1.97 | 未验证 |