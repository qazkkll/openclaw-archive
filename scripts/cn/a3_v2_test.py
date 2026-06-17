#!/usr/bin/env python3
import sys, io, json, time, os, gc, warnings
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import defaultdict
warnings.filterwarnings('ignore')
sys.stdout = io.TextIWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
print = lambda *a,**kw: (__import__('builtins').print(*a, flush=True, **kw))
import xgboost as xgb
print('Imports ok')
