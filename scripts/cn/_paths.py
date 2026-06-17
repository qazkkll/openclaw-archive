#!/usr/bin/env python3
"""
统一路径管理 — 从 scripts/_paths.py 导入
此文件为兼容旧脚本保留，新脚本直接用 from _paths import *
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import *
