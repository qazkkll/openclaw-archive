#!/usr/bin/env python3
"""
交易日历 — 美股交易日判断
============================
周末 + 美国联邦假日自动跳过。
用法:
    from trading_calendar import is_trading_day, next_trading_day, sleep_until_next_trading_day
"""

from datetime import datetime, date, timedelta
from typing import Optional

# 美国联邦假日 (固定日期 + 浮动日期)
# 每年年初更新一次即可

def _easter(year: int) -> date:
    """计算复活节日期 (Anonymous Gregorian algorithm)"""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """某月第n个weekday (0=Mon, 6=Sun)"""
    first = date(year, month, 1)
    # 找到该月第一个weekday
    offset = (weekday - first.weekday()) % 7
    first_target = first + timedelta(days=offset)
    return first_target + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """某月最后一个weekday"""
    if month == 12:
        last = date(year, 12, 31)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)


def get_us_holidays(year: int) -> set:
    """获取美国股市假日 (NYSE/NASDAQ共用)"""
    holidays = set()

    # 固定日期假日
    holidays.add(date(year, 1, 1))   # New Year's Day
    holidays.add(date(year, 6, 19))  # Juneteenth
    holidays.add(date(year, 7, 4))   # Independence Day
    holidays.add(date(year, 12, 25)) # Christmas

    # 固定日期遇周末 → 观察日 (前周五或下周一)
    jan1 = date(year, 1, 1)
    if jan1.weekday() == 5:  # Saturday → 前周五
        holidays.add(date(year - 1, 12, 31))
    elif jan1.weekday() == 6:  # Sunday → 下周一
        holidays.add(date(year, 1, 2))

    jun19 = date(year, 6, 19)
    if jun19.weekday() == 5:
        holidays.add(date(year, 6, 18))
    elif jun19.weekday() == 6:
        holidays.add(date(year, 6, 20))

    jul4 = date(year, 7, 4)
    if jul4.weekday() == 5:
        holidays.add(date(year, 7, 3))
    elif jul4.weekday() == 6:
        holidays.add(date(year, 7, 5))

    dec25 = date(year, 12, 25)
    if dec25.weekday() == 5:
        holidays.add(date(year, 12, 24))
    elif dec25.weekday() == 6:
        holidays.add(date(year, 12, 26))

    # 浮动日期假日
    holidays.add(_nth_weekday(year, 1, 0, 3))   # MLK Day (3rd Monday Jan)
    holidays.add(_nth_weekday(year, 2, 0, 3))   # Presidents Day (3rd Monday Feb)
    easter = _easter(year)
    holidays.add(easter - timedelta(days=2))     # Good Friday
    holidays.add(_last_weekday(year, 5, 0))      # Memorial Day (last Monday May)
    holidays.add(_nth_weekday(year, 9, 0, 1))   # Labor Day (1st Monday Sep)
    holidays.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving (4th Thursday Nov)

    return holidays


def is_trading_day(d: Optional[date] = None) -> bool:
    """判断是否为美股交易日"""
    if d is None:
        d = date.today()
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return d not in get_us_holidays(d.year)


def next_trading_day(from_date: Optional[date] = None) -> date:
    """下一个交易日 (含当天如果今天是交易日)"""
    if from_date is None:
        from_date = date.today()
    d = from_date
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def sleep_until_next_trading_day(log=print):
    """睡眠到下一个交易日的盘前开始时间。返回下一个交易日。"""
    today = date.today()
    if is_trading_day(today):
        return today

    nxt = next_trading_day(today)
    days_ahead = (nxt - today).days
    log(f"  📅 非交易日 ({today}, {'周六' if today.weekday()==5 else '周日' if today.weekday()==6 else '假日'})，"
        f"休市{days_ahead}天，下一个交易日: {nxt}")
    return nxt


if __name__ == "__main__":
    # 打印本月日历
    today = date.today()
    year, month = today.year, today.month
    holidays = get_us_holidays(year)
    print(f"\n📅 {year}年{month}月 美股交易日历")
    print("=" * 40)
    for d in range(1, 32):
        try:
            dt = date(year, month, d)
        except ValueError:
            break
        status = "✅" if is_trading_day(dt) else "❌"
        day_name = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][dt.weekday()]
        holiday = " (假日)" if dt in holidays else ""
        print(f"  {dt} {day_name} {status}{holiday}")

    print(f"\n今天: {'✅ 交易日' if is_trading_day() else '❌ 非交易日'}")
    print(f"下一个交易日: {next_trading_day()}")
