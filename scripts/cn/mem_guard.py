#!/usr/bin/env python3
"""内存守卫：训练前检查可用内存，不够就自动降级"""
import os, sys

def get_free_memory_gb():
    """获取可用内存(GB)"""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        # fallback to wmic on Windows
        import subprocess
        result = subprocess.run(['wmic', 'os', 'get', 'FreePhysicalMemory'], 
                              capture_output=True, text=True)
        for line in result.stdout.strip().split('\n'):
            if line.strip() and line.strip().isdigit():
                return int(line.strip()) / (1024**2)
        return 0

def check_memory(threshold_gb=8, light_mode_stocks=100):
    """
    检查内存，返回 (mode, free_gb)
    mode: 'full' 或 'light'
    """
    free_gb = get_free_memory_gb()
    
    if free_gb < threshold_gb:
        print(f"⚠️ 内存不足: {free_gb:.1f}GB < {threshold_gb}GB")
        print(f"🔄 切换到轻量模式（{light_mode_stocks}只股票）")
        return 'light', free_gb
    else:
        print(f"✅ 内存充足: {free_gb:.1f}GB")
        return 'full', free_gb

if __name__ == '__main__':
    mode, free = check_memory()
    print(f"\n模式: {mode}")
    print(f"可用内存: {free:.1f}GB")
