# 生产系统审计报告


## 1. 特征匹配检查

**模型特征 (36):**
  1. rev_5d
  2. rev_10d
  3. rev_20d
  4. rsi_reversal
  5. macd_reversal
  6. macd_hist
  7. low_vol_5d
  8. low_vol_20d
  9. low_atr
  10. md_net_5
  11. md_net_20
  12. lg_net_5
  13. lg_net_20
  14. total_net_5
  15. total_net_20
  16. small_cap
  17. residual_mom_5d
  18. residual_mom_20d
  19. lg_flow_momentum
  20. total_flow_momentum
  21. lg_net_20_rank
  22. md_net_20_rank
  23. total_net_20_rank
  24. rev_flow_interaction
  25. turnover_rank
  26. pe_rank
  27. pe_inverse
  28. pb_rank
  29. pb_inverse
  30. div_rank
  31. ps_rank
  32. vol_r
  33. sm_net_5
  34. sm_net_20
  35. elg_net_5
  36. elg_net_20

**信号脚本特征:**
脚本特征 (36):
  1. rev_5d
  2. rev_10d
  3. rev_20d
  4. rsi_reversal
  5. macd_reversal
  6. macd_hist
  7. low_vol_5d
  8. low_vol_20d
  9. low_atr
  10. md_net_5
  11. md_net_20
  12. lg_net_5
  13. lg_net_20
  14. total_net_5
  15. total_net_20
  16. small_cap
  17. residual_mom_5d
  18. residual_mom_20d
  19. lg_flow_momentum
  20. total_flow_momentum
  21. lg_net_20_rank
  22. md_net_20_rank
  23. total_net_20_rank
  24. rev_flow_interaction
  25. turnover_rank
  26. pe_rank
  27. pe_inverse
  28. pb_rank
  29. pb_inverse
  30. div_rank
  31. ps_rank
  32. vol_r
  33. sm_net_5
  34. sm_net_20
  35. elg_net_5
  36. elg_net_20

**匹配结果:**
- 模型特征数: 36
- 脚本特征数: 36
- 完全匹配: True

**结论: 特征完全匹配。**

## 2. hold_days确认

- models/cn/cn_alpha_v1.1_summary.json: hold_days=未定义
- models/cn/cn_alpha_v1.0_summary.json: hold_days=未定义
- config.json A股 rebalance_days: 20
- config.json hold_days: 20
- production.json a2 target: fwd_10d
- production.json a2 hold_days: (未显式定义，但target是fwd_10d)

训练脚本默认值:
- a1_layer3_xgb.py: hold_days=10 (默认)
- cn-alpha系列训练脚本: 未找到（可能是notebook或一次性脚本生成）

**结论:** cn-alpha-v1.1的hold_days最可能是10天（基于fwd_10d目标），但需要确认训练脚本。

## 3. 配置对齐检查

- config.json生产模型: cn-alpha-v1.1
- production.json生产模型: a1_layer3_xgb_10d (A2)
- **两个配置指向不同模型！**
- gen_signal_v1_1.py加载: cn_alpha_v1.1.json
- 实际使用的应该是config.json定义的（cn-alpha-v1.1）

## 4. 信号灯检查

**信号脚本无信号灯逻辑** — 需要添加三层百分位分级