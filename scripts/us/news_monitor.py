#!/usr/bin/env python3
"""
📰 热点新闻监控 🍤
从 Google News RSS 拉新闻 → 重要性评分 → 超过阈值才推送
支持中英文，按分类/关键词过滤
"""

import json, os, re, time, hashlib, sys
from datetime import datetime, timezone
import urllib.request
import xml.etree.ElementTree as ET
import urllib.parse

WORKSPACE = "/home/admin/.openclaw/workspace"
CONFIG_FILE = f"{WORKSPACE}/config/news_monitor.json"
STATE_FILE = f"{WORKSPACE}/data/news_pushed.json"
LOG_FILE = f"{WORKSPACE}/logs/news_monitor.log"

# Telegram Bot 配置
BOT_TOKEN = ""
CHAT_ID = "7908145929"
try:
    c = json.load(open("/home/admin/.openclaw/openclaw.json"))
    BOT_TOKEN = c["channels"]["telegram"]["botToken"]
except: pass

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def load_config():
    """加载监控配置"""
    default = {
        "categories": [
            {
                "name": "美股持仓",
                "rss_query": "NVDA OR ADBE OR Nvidia OR Adobe OR semiconductor stock",
                "hl": "zh-CN",
                "gl": "HK",
                "score_threshold": 60
            },
            {
                "name": "A股持仓",
                "rss_query": "新大陆 OR 科大讯飞 OR A股",
                "hl": "zh-CN",
                "gl": "CN",
                "score_threshold": 70
            },
            {
                "name": "科技趋势",
                "rss_query": "AI OR artificial intelligence OR semiconductor",
                "hl": "en-US",
                "gl": "US",
                "score_threshold": 75
            },
            {
                "name": "CS 反恐精英",
                "rss_query": "Counter-Strike OR CS2 OR ESL OR Major",
                "hl": "en-US",
                "gl": "US",
                "score_threshold": 80
            },
            {
                "name": "足球",
                "rss_query": "Premier League OR Champions League OR 英超 OR 欧冠",
                "hl": "zh-CN",
                "gl": "HK",
                "score_threshold": 75
            }
        ],
        "priority_keywords": {
            "holdings": ["NVDA", "ADBE", "新大陆", "科大讯飞", "英伟达", "Adobe"],
            "high_impact": ["earnings", "财报", "acquisition", "收购", "SEC", "investigation",
                          "FDA", "approval", "breakup", "split", "crash", "rally",
                          "加息", "降息", "利率", "关税", "tariff", "sanctions",
                          "layoffs", "裁员", "CEO", "resign", "CEO辞职"],
            "medium_impact": ["analyst", "upgrade", "downgrade", "target price",
                            "评级", "目标价", "partnership", "合作"]
        },
        "interval_minutes": 10,
        "max_articles_per_run": 30
    }
    
    # 从持仓文件自动加载holdings关键词
    try:
        pf = json.load(open(f"{WORKSPACE}/data/portfolio.json"))
        holdings = []
        for s in pf.get("us_stock", []) + pf.get("a_stock", []):
            name = s.get("name", "")
            ticker = s.get("code", "")
            if ticker: holdings.append(ticker)
            if name: holdings.append(name)
        if holdings:
            default["priority_keywords"]["holdings"] = list(set(holdings))
    except:
        pass

    if os.path.exists(CONFIG_FILE):
        loaded = json.load(open(CONFIG_FILE))
        # 更新配置文件中的持仓关键词
        if holdings:
            loaded["priority_keywords"]["holdings"] = list(set(
                loaded["priority_keywords"].get("holdings", []) + holdings
            ))
        return {**default, **loaded}
    
    # 首次运行写入默认配置
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(default, f, indent=2, ensure_ascii=False)
    return default

def load_pushed():
    """加载已推送记录（去重用）"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"pushed_ids": [], "last_run": ""}

def save_pushed(pushed):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    # 只保留最近 500 条去重记录
    pushed["pushed_ids"] = pushed["pushed_ids"][-500:]
    pushed["last_run"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(pushed, f, indent=2)

def fetch_google_rss(query, hl="zh-CN", gl="HK", max_results=30):
    """拉取 Google News RSS"""
    from urllib.parse import quote
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl={hl}&gl={gl}"
    
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml_data = resp.read()
            root = ET.fromstring(xml_data)
            
            articles = []
            for item in root.findall(".//item"):
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                pub_date_str = item.findtext("pubDate", "")
                source = item.findtext("source", "")
                
                # 解析日期
                pub_date = ""
                try:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(pub_date_str)
                    pub_date = dt.strftime("%m-%d %H:%M")
                except:
                    pub_date = pub_date_str[:16]
                
                articles.append({
                    "title": title,
                    "link": link,
                    "pub_date": pub_date,
                    "source": source.strip() if source else "",
                    "raw_date": pub_date_str
                })
            
            return articles[:max_results]
    except Exception as e:
        log(f"⚠️ RSS请求失败 ({query[:30]}...): {e}")
        return []

def score_article(article, config):
    """重要性评分 0-100"""
    title = article["title"].lower()
    full_text = title + " " + article.get("source", "").lower()
    score = 0
    
    # 1. 基础分（有消息就算10分）
    score += 10
    
    # 2. 持仓关键词命中（最重要）
    for kw in config["priority_keywords"]["holdings"]:
        if kw.lower() in full_text:
            score += 40
            break
    
    # 3. 高影响力关键词
    for kw in config["priority_keywords"]["high_impact"]:
        if kw.lower() in full_text:
            score += 25
            break
    
    # 4. 中等影响力关键词
    for kw in config["priority_keywords"]["medium_impact"]:
        if kw.lower() in full_text:
            score += 10
            break
    
    # 5. 来源质量加分
    quality_sources = ["reuters", "bloomberg", "cnbc", "wsj", "financial times",
                       "bbc", "路透", "彭博", "华尔街", "财新", "第一财经"]
    for src in quality_sources:
        if src in full_text:
            score += 10
            break
    
    return min(score, 100)

def fetch_article_summary(url, max_chars=200):
    """获取文章摘要（追踪重定向，提取meta描述或正文第一段）"""
    import re as regex
    try:
        # 先请求RSS的跳转链接（不自动跳转），拿到真实URL
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        try:
            resp = urllib.request.urlopen(req, timeout=8)
        except urllib.error.HTTPError as e:
            # Google News 用302跳转，httplib自动跟。不行就原地放弃
            return ""
        
        html = resp.read()
        # 尝试多种编码
        for enc in ['utf-8', 'gbk', 'gb2312', 'big5', 'latin-1']:
            try:
                text = html.decode(enc)
                break
            except:
                continue
        else:
            text = html.decode('utf-8', errors='replace')
        
        # 提取meta description
        meta_patterns = [
            r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+property=["\']twitter:description["\'][^>]+content=["\']([^"\']+)["\']',
        ]
        for pat in meta_patterns:
            m = regex.search(pat, text, regex.IGNORECASE)
            if m:
                summary = m.group(1).strip()
                if len(summary) > 20:
                    return summary[:max_chars]
        
        # 回退：找p标签的第一个内容
        p_match = regex.search(r'<p[^>]*>([^<]+)</p>', text)
        if p_match:
            content = p_match.group(1).strip()
            if len(content) > 30:
                return content[:max_chars]
        
        return ""
    except Exception as e:
        return ""

def main():
    config = load_config()
    pushed = load_pushed()
    pushed_ids = set(pushed.get("pushed_ids", []))
    
    log(f"📰 热点监控启动 — {len(config['categories'])}个分类")
    
    new_pushes = 0
    push_texts = []
    
    for cat in config["categories"]:
        name = cat["name"]
        threshold = cat["score_threshold"]
        
        articles = fetch_google_rss(
            cat["rss_query"],
            hl=cat.get("hl", "zh-CN"),
            gl=cat.get("gl", "HK"),
            max_results=config.get("max_articles_per_run", 30)
        )
        
        if not articles:
            continue
        
        for article in articles:
            # 去重
            article_id = hashlib.md5(article["link"].encode()).hexdigest()
            if article_id in pushed_ids:
                continue
            
            # 评分
            score = score_article(article, config)
            
            if score >= threshold:
                pushed["pushed_ids"].append(article_id)
                pushed_ids.add(article_id)
                new_pushes += 1
                
                # 重要性等级
                if score >= 85:
                    level = "🔴 重要"
                elif score >= 70:
                    level = "🟠 关注"
                else:
                    level = "🟡 参考"
                
                # 获取正文摘要
                summary = fetch_article_summary(article['link'])
                
                # 格式化推送文本
                line = f"{level} [{name}]\n{article['title']}"
                if summary:
                    line += f"\n{summary}"
                if article.get("source"):
                    line += f"\n_{article['source']} | {article['pub_date']}_"
                push_texts.append(line)
                
                # 也输出到stdout（供日志查看）
                print(f"\n[{level}] [{name}] ({score}分)")
                print(f"{article['title']}")
                if article.get("source"):
                    print(f"来源: {article['source']} | {article['pub_date']}")
    
    if new_pushes == 0:
        print("\n✅ 无新消息推送")
    else:
        push_to_telegram(push_texts)
    
    save_pushed(pushed)
    log(f"完成: 检查{len(config['categories'])}类, 推送{new_pushes}条")


def push_to_telegram(messages):
    """通过Bot API直接推送"""
    if not BOT_TOKEN or not messages:
        return
    
    header = "📰 *热点速递*"
    text = header + "\n" + "\n\n".join(messages[:5])
    text += "\n\n_每15分钟自动扫描_ 🍤"
    
    if len(text) > 4000:
        text = text[:3997] + "..."
    
    try:
        data = urllib.parse.urlencode({
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": "true"
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            log("✅ 已推送到Telegram")
    except Exception as e:
        log(f"⚠️ 推送失败: {e}")

if __name__ == "__main__":
    main()
