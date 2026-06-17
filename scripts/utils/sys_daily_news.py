#!/usr/bin/env python3
"""
sys_daily_news.py — 每日财经新闻整理（cron用）
==============================================
用途：每天早上8:30被cron调用，拉前一天新闻，整理出不超过1500字的精简摘要
输出：打印到stdout，cron delivery=announce会自动发到已配渠道

用法：python scripts/sys_daily_news.py
"""

import minishare as ms
from datetime import datetime, timedelta
import sys
import os

# ── 配置 ──
TOKEN = 'Jarvne6fmgArRa46Xfon0e1kw55E6hes5IB2Fy2X0ndqnvrL48jsVOtTbf014f06'
MAX_NEWS = 80
MAX_OUTPUT_LEN = 1700  # 消息体长度限制，尽量塞但别超

# ── 日期计算 ──
now = datetime.now()
# 拉前一天的新闻（从今天0点往前推24h的范围）
today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
yesterday = today_start - timedelta(days=1)
yesterday_end = today_start - timedelta(seconds=1)

date_label_yesterday = yesterday.strftime('%m/%d')
date_today_str = now.strftime('%Y-%m-%d')

y_start_str = yesterday.strftime('%Y-%m-%d %H:%M:%S')
y_end_str = yesterday_end.strftime('%Y-%m-%d %H:%M:%S')

# ── 拉新闻 ──
try:
    api = ms.pro_api(TOKEN)
    df = api.query('news', start_date=y_start_str, end_date=y_end_str, limit=MAX_NEWS)
except Exception as e:
    print(f"📡 新闻抓取失败: {e}")
    sys.exit(0)

if len(df) == 0:
    print(f"📅 {now.strftime('%Y-%m-%d %H:%M')} 日报\n昨日无重大新闻记录。")
    sys.exit(0)

# ── 去重（same content appears multiple times with slight format diff） ──
seen = set()
unique_rows = []
for _, row in df.iterrows():
    content = row['content']
    # 去空格/标点做简易去重
    key = content.strip()[:80]
    if key not in seen:
        seen.add(key)
        unique_rows.append(row)

# ── 按时间排序 ──
unique_rows.sort(key=lambda r: r['datetime'])

# ── 分类（很粗糙的关键词组匹配，够用） ──
categories = {
    '🏛️ 宏观/政策': ['央行', '国务院', '发改委', '商务部', '财政部', '海关', '统计局', '政治局',
                     '国常会', '降准', '降息', '加息', 'LPR', '关税', '贸易', 'GDP', 'CPI', 'PMI',
                     '逆回购', 'MLF', '美联储', '欧洲央行', '美联储', '通胀', '就业数据',
                     '外交', '声明', '沙伊', '中美', '中欧', '俄乌', '巴以', '制裁'],
    '📊 市场/行情': ['收评', '开盘', '收盘', '成交', '涨停', '跌停', '板块', '指数', '沪深',
                     'A股', '港股', '美股', '涨幅', '跌幅', '创业板', '科创板', '北向',
                     '主力资金', 'ETF', '期货', '原油', '黄金', '汇率', '人民币',
                     '日经', '恒生', '标普', '纳指', '道指', '牛市', '熊市'],
    '🏭 行业/公司': ['半导体', '芯片', 'AI', '人工智能', '光伏', '新能源', '锂电', '汽车',
                     '医药', '创新药', '消费', '地产', '银行', '保险', '券商',
                     '互联网', '电商', '游戏', '华为', '腾讯', '阿里', '字节',
                     '比亚迪', '宁德时代', '特斯拉', '苹果', '英伟达', '微软'],
    '🌍 国际': ['特朗普', '拜登', '欧盟', '北约', 'OPEC', '俄罗斯', '乌克兰',
                '韩国', '日本', '菲律宾', '印尼', '台湾', '朝鲜',
                '地震', '自然灾害', '埃博拉', '疫情'],
    '📈 数据发布': ['海关总署', '统计局', '数据', '同比增长', '环比', '进出口',
                   '社融', 'M2', '工业', '投资', '消费', '用电量',
                   'shibor', 'SHIBOR', '收益率', '国债', '财报', '营收',
                   '利润', '业绩预告', '评级', '目标价']
}

def classify_news(content):
    """简单关键词分类"""
    scores = {}
    for cat, keywords in categories.items():
        score = sum(1 for kw in keywords if kw in content)
        if score > 0:
            scores[cat] = score
    if not scores:
        return '📌 其他'
    return max(scores, key=scores.get)

# ── 分类整理 ──
classified = {}
for row in unique_rows:
    content = row['content']
    cat = classify_news(content)
    time_str = row['datetime'][11:16]
    if cat not in classified:
        classified[cat] = []
    classified[cat].append(f"[{time_str}] {content}")

# ── 输出 ──
lines = []
lines.append(f"📰 早安财经日报 | {date_label_yesterday}")
lines.append(f"来源: minishare · 共{len(unique_rows)}条摘要")
lines.append("")

# 按类别优先级排序
cat_order = ['🏛️ 宏观/政策', '📊 市场/行情', '📈 数据发布', '🏭 行业/公司', '🌍 国际', '📌 其他']
for cat in cat_order:
    if cat not in classified:
        continue
    items = classified[cat]
    lines.append(f"━━━ {cat} ({len(items)}条) ━━━")
    for item in items:
        lines.append(item)
    
    # 检查长度，防止超限
    total = sum(len(l) + 1 for l in lines)
    if total > MAX_OUTPUT_LEN:
        lines.append("")
        lines.append(f"... 更多内容因篇幅限制已省略")
        break

output = '\n'.join(lines)
try:
    print(output)
except UnicodeEncodeError:
    print(output.encode('utf-8', errors='replace').decode('gbk', errors='replace'))
