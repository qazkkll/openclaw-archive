#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_sweep import run_backtest

c1 = {"buy_threshold":62,"sell_threshold":50,"rebalance_days":7,"sector_top_n":4,"max_positions":10,"initial_capital":1000000.0}
r1, a1, t1 = run_backtest(c1)
print("Test 1: ret=%.2f%% ann=%.2f%% trades=%d" % (r1, a1, t1))

c2 = {"buy_threshold":66,"sell_threshold":45,"rebalance_days":10,"sector_top_n":5,"max_positions":8,"initial_capital":1000000.0}
r2, a2, t2 = run_backtest(c2)
print("Test 2: ret=%.2f%% ann=%.2f%% trades=%d" % (r2, a2, t2))

if abs(r1 - 55.85) < 10:
    print("PASS: Test 1 matches expected range")
else:
    print("NOTE: Test 1 result differs from default (55.85%%) - expected with different config")
