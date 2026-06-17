#!/usr/bin/env python3
"""2:00自动执行 - OpenD全量回测"""
import os, sys, time, json, warnings, numpy as np
warnings.filterwarnings('ignore')

SP500_FILE = '/home/admin/.openclaw/workspace/data/sp500_universe.json'
RESULTS_FILE = '/home/admin/.openclaw/workspace/data/bt_opend_final.json'
OPEND_HOST = '127.0.0.1'
OPEND_PORT = 11111
LOG_FILE = '/home/admin/.openclaw/workspace/logs/bt_opend.log'

def log(msg):
    t = time.strftime('%H:%M:%S')
    with open(LOG_FILE, 'a') as f:
        f.write('[%s] %s\n' % (t, msg))
    print(msg, flush=True)

log('=== 2:00 全量回测启动 ===')
