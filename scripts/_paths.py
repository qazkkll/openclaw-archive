#!/usr/bin/env python3
"""
统一路径管理 — 所有脚本从此文件导入路径
基于项目根目录自动推导，不依赖绝对路径
"""
import os, sys

# 项目根目录 = scripts/ 的上级目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ======= 核心路径 =======
DATA_DIR      = os.path.join(PROJECT_ROOT, 'data')
CN_DATA       = os.path.join(DATA_DIR, 'cn')
US_DATA       = os.path.join(DATA_DIR, 'us')
CONFIG_DIR    = os.path.join(DATA_DIR, 'config')
MODELS_DIR    = os.path.join(PROJECT_ROOT, 'models')
CN_MODELS     = os.path.join(MODELS_DIR, 'cn')
US_MODELS     = os.path.join(MODELS_DIR, 'us')
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, 'output')
CN_OUTPUT     = os.path.join(OUTPUT_DIR, 'cn')
US_OUTPUT     = os.path.join(OUTPUT_DIR, 'us')

# ======= 数据文件 =======
CN_KLINE      = os.path.join(CN_DATA, 'a_hist_10y.parquet')
CN_MONEYFLOW  = os.path.join(CN_DATA, 'moneyflow_full.json')
CN_MF_POOL    = os.path.join(DATA_DIR, 'moneyflow_pool.json')
CN_A1_DAILY   = os.path.join(CN_DATA, 'a1_daily.parquet')
US_HIST_SP500 = os.path.join(US_DATA, 'us_hist_sp500_10y.parquet')
US_HIST_YF    = os.path.join(US_DATA, 'us_hist_yf_10y.parquet')

# ======= 配置文件 =======
TUSHARE_CFG   = os.path.join(CONFIG_DIR, 'tushare.json')
QUALITY_POOL  = os.path.join(CONFIG_DIR, 'quality_pool.json')
STRATEGY_CFG  = os.path.join(CONFIG_DIR, 'strategy.json')

# ======= 模型文件 =======
CN_A2_MODEL   = os.path.join(CN_MODELS, 'a1_layer3_xgb_10d.json')
CN_A2_META    = os.path.join(CN_MODELS, 'a1_layer3_xgb_10d_meta.json')

# ======= 兼容旧脚本（D盘路径 → 本地路径映射） =======
D_DATA = CN_DATA  # 兼容
D_MODELS = CN_MODELS  # 兼容
ML_DIR = os.path.join(DATA_DIR, 'us')  # 美股ML数据目录（兼容旧脚本）

def ensure_dirs():
    """确保所有必要目录存在"""
    for d in [DATA_DIR, CN_DATA, US_DATA, CONFIG_DIR, 
              CN_MODELS, US_MODELS, OUTPUT_DIR, CN_OUTPUT, US_OUTPUT]:
        os.makedirs(d, exist_ok=True)

# 初始化时确保目录存在
ensure_dirs()
