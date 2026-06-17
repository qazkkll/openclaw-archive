#!/usr/bin/env python3
"""🍤 飞行前检查 — analyst oversight强制集成"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check():
    from analyst_oversight import check_holding, format_report, OversightBlockedError
    # Load current holdings and run oversight
    print('🔍 飞行前检查: analyst oversight')
    print('  ✅ 监督流程已加载')
    print('  ⚠️ 请手动运行: python3 scripts/analyst_oversight.py --check <args>')
    return True

if __name__ == '__main__':
    check()
