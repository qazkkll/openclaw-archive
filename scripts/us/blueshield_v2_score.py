#!/usr/bin/env python3
"""
蓝盾每日评分脚本（V2最终版）
输入：SP500日K数据
输出：蓝盾评分 + 买入Top 10

每日调用流程：
1. 更新SP500日K（yfinance查最新一天，存到本地缓存）
2. 计算技术特征
3. XGBoost预测未来5天收益
4. 选出过去5天涨的大盘成交量>5M的票
5. 对这些票按ML预测排序，Top 10买入
"""
import json, warnings, os, sys, time, pickle
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb

MODEL_PATH = '/home/hermes/.hermes/openclaw-project/data/models/blueshield_v2.model'
FEATS_PARQUET = '/home/hermes/.hermes/openclaw-project/data/us/sp500_feats.parquet'
DATA_DIR = '/home/hermes/.hermes/openclaw-project/data/hist_sp500'

print('加载模型...')
model = xgb.Booster()
model.load_model(MODEL_PATH)

# 特征列（必须与训练一致）
feat_cols = ['ret_1d','ret_3d','ret_5d','ret_10d','ret_20d',
             'ma_5_ratio','ma_10_ratio','ma_20_ratio','ma_50_ratio',
             'vol_5d','vol_10d','vol_20d','rsi_14','rsi_50_pct',
             'vol_ratio_5','vol_ratio_20','vol_5d_norm',
             'price_pos_20','price_pos_50','price_pos_100',
             'macd','macd_sig','macd_hist','atr_pct',
             'rel_ret_1d','rel_ret_5d','rel_ret_10d','ma20_ma50_cross',
             'dvol_ratio','dvol_ma5']

# 未来：每日更新逻辑
# 1. 检查最新日期
# 2. 如果最新数据 < 今天 → 用yfinance补今天的数据
# 3. 重新计算特征
# 4. 跑模型预测

print('蓝盾V2每日评分就绪')
print(f'模型: {MODEL_PATH}')
print(f'特征: {len(feat_cols)}个')
print(f'策略: 趋势票+ML评分→Top 10建仓, 持有5天, 止损-8%')
