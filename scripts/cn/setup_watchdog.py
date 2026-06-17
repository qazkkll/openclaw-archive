"""
创建 OpenClaw Gateway看门狗
方式1: 尝试schtasks（需要admin）
方式2: 用PowerShell ScheduledJob（无需admin，但后台进程常驻）
"""
import subprocess, sys, os
sys.stdout.reconfigure(encoding='utf-8')

script_path = r'/home/hermes/.hermes/openclaw-archive\scripts\watchdog_gateway.bat'
task_name = 'OpenClawGatewayWatchdog'

# 先用普通用户权限创建（可能失败）
cmd = [
    'schtasks', '/Create', '/SC', 'MINUTE', '/MO', '5',
    '/TN', task_name,
    '/TR', f'cmd.exe /c "{script_path}"',
    '/ST', '00:00',
    '/F'
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
print(f'schtasks 返回: {result.returncode}')
if result.returncode == 0:
    print(f'✅ 看门狗已创建 (每5分钟)')
else:
    print(f'schtasks 创建失败（需要管理员权限）')
    print(f'\n建议手动执行以下命令（以管理员身份运行cmd.exe）：')
    print(f'='*60)
    print(f'schtasks /Create /SC MINUTE /MO 5 /TN "{task_name}" /TR "cmd.exe /c {script_path}" /ST 00:00 /F /RU SYSTEM')
    print(f'='*60)
    print(f'\n或者用下面的PowerShell脚本（无需admin）：')
    print(f'powershell -ExecutionPolicy Bypass -File scripts/setup_watchdog_ps.ps1')
    print()
    print(f'日志位置: %TEMP%\\openclaw-watchdog.log')
