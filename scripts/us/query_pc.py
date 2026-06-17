#!/usr/bin/env python3
"""
🦐 远程查询PC小钳（桌面Futu数据）

用法:
  python3 query_pc.py futu_position    查持仓
  python3 query_pc.py futu_order       查订单
  python3 query_pc.py futu_account     查账户
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sessions_send import send_to_session

SESSION_KEY = "pc_xiaoqian"  # PC小钳的session标识

def futu_query(action):
    """向PC小钳发送Futu查询请求"""
    result = send_to_session(SESSION_KEY, f"【PC小钳】查询Futu: {action}")
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: query_pc.py <futu_position|futu_order|futu_account>")
        sys.exit(1)
    
    result = futu_query(sys.argv[1])
    print(result)
