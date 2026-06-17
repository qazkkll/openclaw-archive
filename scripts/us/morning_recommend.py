#!/usr/bin/env python3
"""
A股开盘前推荐 (8:45) 🍤 v3
数据源: 腾讯实时行情 + daily_cache.json (RSI/MA20)

改进:
1. 从全市场股票池扫描(主板+全A=3029只)
2. 腾讯批量实时行情获取价格
3. 多因子评分
4. 新闻过滤利空
5. 主板/全A分别推荐
"""

import json
import os
import re
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta

BOT_TOKEN = '7792764974:AAFrFrZ3JAjdhkCsphy2N-gd99U5puRywUI'
CHAT_ID = '7908145929'
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_json(filepath):
    try:
        with open(filepath) as f:
            return json.load(f)
    except:
        return None

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, ensure_ascii=False)

def send_telegram(text):
    url = "https://api.telegram.org/bot7792764974:AAFrFrZ3JAjdhkCsphy2N-gd99U5puRywUI/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}
    data = urllib.parse.urlencode(payload).encode()
    try:
        req = urllib.request.Request(url, data=data)
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"Telegram send error: {e}")
        return False

# ===== 腾讯实时行情 (批量) =====

def fetch_tencent_realtime(codes):
    """批量获取实时行情，codes = ['000001','600519',...]"""
    if not codes:
        return {}
    
    results = {}
    for i in range(0, len(codes), 50):
        batch = codes[i:i+50]
        # Build qt.gtimg.cn query string
        q_parts = []
        for c in batch:
            if c.startswith(('6', '5')):
                q_parts.append(f'sh{c}')
            elif c.startswith(('0', '3')):
                q_parts.append(f'sz{c}')
            elif c.startswith(('4', '8')):
                q_parts.append(f'bj{c}')
            else:
                q_parts.append(f'sz{c}')
        
        qs = ','.join(q_parts)
        url = f'http://qt.gtimg.cn/q={qs}'
        
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()
                try:
                    text = raw.decode('gbk')
                except:
                    text = raw.decode('utf-8', errors='replace')
                
                for line in text.split('\n'):
                    m = re.match(r'v_(\w+)="(.+)"', line.strip())
                    if not m:
                        continue
                    parts = m.group(2).split('~')
                    code = parts[2] if len(parts) > 2 else ''
                    if not code:
                        continue
                    try:
                        name = parts[1].strip()
                        price = float(parts[3]) if parts[3] else 0
                        prev_close = float(parts[4]) if parts[4] else 0
                        vol = int(parts[7]) if len(parts) > 7 and parts[7] else 0
                        change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0
                        high = float(parts[33]) if len(parts) > 33 and parts[33] else 0
                        low = float(parts[34]) if len(parts) > 34 and parts[34] else 0
                        amount = float(parts[38]) if len(parts) > 38 and parts[38] else 0  # 成交额
                        turnover_ratio = float(parts[39]) if len(parts) > 39 and parts[39] else 0  # 换手率
                        
                        results[code] = {
                            'name': name,
                            'price': price,
                            'prev_close': prev_close,
                            'vol': vol,
                            'change_pct': round(change_pct, 2),
                            'high': high,
                            'low': low,
                            'amount': amount,
                            'turnover_ratio': turnover_ratio
                        }
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            print(f"  Tencent batch {i//50+1} error: {e}")
        
        time.sleep(0.3)
    
    return results

# ===== 评分 =====

def calc_score(price, rsi, ma20, change_pct):
    """多因子评分 (0-100)"""
    if not price or price <= 0 or not ma20 or ma20 <= 0:
        return 0
    
    score = 0
    
    # RSI (40%)
    if rsi is not None:
        if rsi < 30:
            score += 35
        elif rsi < 45:
            score += 30
        elif rsi < 60:
            score += 25
        elif rsi < 70:
            score += 15
        elif rsi >= 70:
            score -= 5
    
    # 均线位置 (25%)
    ratio = price / ma20
    if 0.99 <= ratio <= 1.03:
        score += 25  # 紧贴均线，蓄势待发
    elif 1.03 < ratio <= 1.08:
        score += 20  # 站上均线
    elif ratio > 1.08 and ratio <= 1.15:
        score += 12  # 远离均线
    elif 0.95 <= ratio < 0.99:
        score += 10  # 略破均线
    elif ratio < 0.95:
        score += 3   # 弱势
    
    # 价格位置 (15%) 
    if 5 <= price <= 30:
        score += 15
    elif 30 < price <= 100:
        score += 12
    elif 100 < price <= 300:
        score += 8
    elif 3 <= price < 5:
        score += 5
    elif price > 300:
        score += 3
    else:
        score -= 5
    
    # 涨幅 (10%) - 低开或平开更好
    if change_pct is not None:
        if -2 <= change_pct <= 1:
            score += 10  # 低开/平开，有入场空间
        elif 1 < change_pct <= 3:
            score += 6
        elif -5 <= change_pct < -2:
            score += 5  # 低开较多但可能是机会
        elif change_pct > 3:
            score += 2   # 高开太多
        else:
            score -= 5   # 大幅低开
    
    # 成交量 (10%) - 用amount
    score += 5
    
    return max(0, min(100, score))

# ===== 新闻 =====

def get_news(stock_name, stock_code):
    """获取新闻并判断情绪"""
    negative_kw = ['减持', '立案', '亏损扩大', '监管函', '警示函', '退市',
                   '调查', '处罚', '利空', '爆雷', '造假', '违规', 'ST', '*ST',
                   '违约', '业绩变脸', '亏损', '问询函', '关注函']
    positive_kw = ['涨停', '大涨', '利好', '增长', '突破', '回购', '增持', 
                   '盈利', '中标', '签约', '新高', '扭亏']
    
    # Search Baidu news
    query = f"{stock_name} {stock_code}"
    search_url = f"https://www.baidu.com/s?wd={urllib.parse.quote(query)}&tn=news"
    
    titles = []
    try:
        req = urllib.request.Request(search_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0'
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode('utf-8', errors='replace')
            titles = re.findall(r'<h3[^>]*>.*?<a[^>]*>(.*?)</a>', html, re.DOTALL)
            titles = [re.sub(r'<[^>]+>', '', t).strip() for t in titles[:5]]
            titles = [t for t in titles if t]
    except:
        pass
    
    if not titles:
        return "暂无相关新闻", "中性"
    
    # Check negative
    for t in titles:
        for kw in negative_kw:
            if kw in t:
                return t, "负面"
    
    # Determine sentiment
    all_text = ' '.join(titles)
    pos_count = sum(1 for kw in positive_kw if kw in all_text)
    
    if pos_count >= 2:
        return titles[0], "正面"
    elif pos_count == 1:
        return titles[0], "正面"
    else:
        return titles[0], "中性"

def quick_negative_check(stock_name, stock_code):
    """快速利空检查"""
    negative_kw = ['减持', '立案', '监管函', '警示函', '退市', '调查', '处罚', '爆雷', '造假', '违规', '*ST', 'ST']
    query = f"{stock_name} {stock_code}"
    search_url = f"https://www.baidu.com/s?wd={urllib.parse.quote(query)}&tn=news"
    
    try:
        req = urllib.request.Request(search_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0'
        })
        with urllib.request.urlopen(req, timeout=6) as resp:
            html = resp.read().decode('utf-8', errors='replace')
            titles = re.findall(r'<h3[^>]*>.*?<a[^>]*>(.*?)</a>', html, re.DOTALL)
            for t in titles[:5]:
                clean = re.sub(r'<[^>]+>', '', t)
                for kw in negative_kw:
                    if kw in clean:
                        return True, clean
    except:
        pass
    return False, ""

# ===== 技术面分析 =====

def tech_analysis(price, rsi, ma20):
    chars = []
    if rsi is not None:
        if rsi < 30: chars.append(f"RSI{rsi:.0f}超卖")
        elif rsi < 45: chars.append(f"RSI{rsi:.0f}偏弱")
        elif rsi < 60: chars.append(f"RSI{rsi:.0f}健康")
        elif rsi < 70: chars.append(f"RSI{rsi:.0f}强势")
        else: chars.append(f"RSI{rsi:.0f}⚠️超买")
    
    if price and ma20:
        r = price / ma20
        if r > 1.05: chars.append("价>20日线(多头)")
        elif r > 0.98: chars.append("紧贴20日线(蓄势)")
        elif r > 0.95: chars.append("略破20日线")
        else: chars.append("跌破20日线(弱)")
    
    return " | ".join(chars) if chars else "数据不足"

def buy_params(price, ma20):
    if not price or not ma20:
        return "开盘观察", "待定", "待定"
    
    now_hk = datetime.now(timezone(timedelta(hours=8)))
    if now_hk.hour < 9 or (now_hk.hour == 9 and now_hk.minute < 30):
        bt = "开盘30分钟内（9:30-10:00）低吸"
    else:
        bt = "盘中回踩MA20附近低吸"
    
    buy_low = round(max(ma20 * 0.97, price * 0.95), 2)
    buy_high = round(price * 1.02, 2)
    if buy_low > buy_high:
        buy_low, buy_high = buy_high, buy_low
    
    target = round(price * 1.10, 2)
    return bt, f"{buy_low} - {buy_high}", f"~{target}"

# ===== 主函数 =====

def main():
    print("🍤 A股开盘前推荐 v3")
    now_hk = datetime.now(timezone(timedelta(hours=8)))
    print(f"  时间: {now_hk.strftime('%Y-%m-%d %H:%M')}")
    today = now_hk.strftime('%Y-%m-%d')
    
    # Load data
    cache = load_json(os.path.join(WORKSPACE, 'data', 'daily_cache.json')) or {}
    stocks_cache = {k: v for k, v in cache.items() if not k.startswith('_')}
    
    stock_list = load_json(os.path.join(WORKSPACE, 'data', 'a_stock_mainboard.json')) or []
    stock_names = {s['code']: s['name'] for s in stock_list}
    
    recommend_record = load_json(os.path.join(WORKSPACE, 'data', 'recommend_history.json')) or {}
    if recommend_record.get('_last_date') != today:
        recommend_record = {'_last_date': today, 'today_codes': []}
    
    print(f"📊 股票池: {len(stock_list)} 只 | 缓存RSI/MA20: {len(stocks_cache)} 只")
    
    # === Phase 1: Get all real-time prices ===
    # We need to process 3029 stocks through Tencent API in batches
    # But that's too many. Let's first get the 50 candidates + a batch of top stocks
    
    # Strategy: Get all real-time in batches of 50 (~121 batches in a loop)
    print("📡 获取全市场实时行情 (3029 stocks)...")
    
    all_codes = [s['code'] for s in stock_list]
    all_rt = {}
    for batch_idx in range(0, len(all_codes), 50):
        batch = all_codes[batch_idx:batch_idx+50]
        batch_rt = fetch_tencent_realtime(batch)
        all_rt.update(batch_rt)
        
        if (batch_idx // 50) % 10 == 0:
            print(f"  📡 进度: {batch_idx+50}/{len(all_codes)} ({len(all_rt)} only) ")
    
    print(f"  ✅ 获取到 {len(all_rt)} 只实时行情")
    
    # === Phase 2: Score all stocks ===
    scored = []
    for code, rt in all_rt.items():
        name = rt['name']
        price = rt['price']
        change_pct = rt['change_pct']
        vol = rt['vol']
        
        if not name or not price or price <= 0:
            continue
        if price < 3:  # Skip penny stocks
            continue
        if 'ST' in name or '退' in name:
            continue
        
        # Check cache for RSI/MA20
        cached = stocks_cache.get(code, {})
        rsi = cached.get('rsi')
        ma20 = cached.get('ma20')
        
        # For stocks without cache, estimate MA20 from price (rough)
        # We'll give lower scores for uncached stocks
        if not ma20:
            continue  # Skip stocks without MA20 data
        
        # Market: mainboard (60/00) or gem (30/688)
        is_mainboard = code.startswith(('60', '00', '001'))
        is_sme = code.startswith('002')  # 中小板 counted as mainboard
        is_gem = code.startswith(('30', '688'))  # 创业板/科创板
        
        score = calc_score(price, rsi, ma20, change_pct)
        if score >= 30:
            scored.append({
                'code': code,
                'name': name,
                'price': price,
                'change_pct': change_pct,
                'score': round(score),
                'rsi': round(rsi, 1) if rsi else None,
                'ma20': round(ma20, 2) if ma20 else None,
                'vol': vol,
                'turnover_ratio': rt.get('turnover_ratio', 0),
                'is_mainboard': is_mainboard or is_sme,
                'is_gem': is_gem,
                'cache': cached
            })
    
    scored.sort(key=lambda x: x['score'], reverse=True)
    print(f"📈 评分完成: {len(scored)} 只合格")
    
    # Split into mainboard and all-A
    mb = [s for s in scored if s['is_mainboard']]
    all_a = scored  # All stocks including gem
    
    print(f"  主板: {len(mb)} 只 | 全A(含创业/科创): {len(all_a)} 只")
    
    # Show top 10
    for label, lst in [("主板Top10", mb[:10]), ("全ATop10", all_a[:10])]:
        names = [f"{s['name']}({s['code']}){s['score']}" for s in lst]
        print(f"  {label}: {', '.join(names)}")
    
    # === Phase 3: News filter ===
    def filter_news(candidates, target=10):
        result = []
        for s in candidates:
            if len(result) >= target:
                break
            
            code = s['code']
            name = s['name']
            
            if code in recommend_record.get('today_codes', []):
                continue
            
            print(f"  📰 {name}({code})...", end=' ', flush=True)
            
            # Quick negative check
            has_neg, neg_detail = quick_negative_check(name, code)
            if has_neg:
                print(f"❌ {neg_detail[:50]}")
                time.sleep(0.8)
                continue
            
            # Full news
            summary, sentiment = get_news(name, code)
            print(f"✅ {sentiment}")
            
            if sentiment == "负面":
                continue
            
            s['news'] = summary
            s['sentiment'] = sentiment
            result.append(s)
            time.sleep(1)
        
        result.sort(key=lambda x: x['score'], reverse=True)
        return result
    
    print("\n🔍 主板新闻过滤...")
    mb_filtered = filter_news(mb, 10)
    print(f"  通过: {len(mb_filtered)}")
    
    print("\n🔍 全A新闻过滤...")
    all_filtered = filter_news(all_a, 10)
    print(f"  通过: {len(all_filtered)}")
    
    # === Phase 4: Generate messages ===
    def fmt_stock(s, idx):
        return (
            f"*#{idx} {s['name']} ({s['code']})* — 评分 {s['score']}/100\n"
            f"  现价 {s['price']} | {'+' if s['change_pct'] >= 0 else ''}{s['change_pct']:.2f}%\n"
            f"  📊 {tech_analysis(s['price'], s['rsi'], s['ma20'])}\n"
            f"  ⏰ {buy_params(s['price'], s['ma20'])[0]}\n"
            f"  💰 买入区间: {buy_params(s['price'], s['ma20'])[1]}\n"
            f"  🎯 目标价: {buy_params(s['price'], s['ma20'])[2]}\n"
            f"  📰 {'🟢' if s.get('sentiment')=='正面' else '⚪'}{s.get('news', '暂无')[:80]}\n"
        )
    
    # Message 1: 主板 (top 5)
    mb_top5 = mb_filtered[:5]
    if mb_top5:
        msg1 = f"🏛 *主板推荐 · 评分前5*\n📅 {today} 开盘前\n\n" + "\n".join(fmt_stock(s, i) for i, s in enumerate(mb_top5, 1))
        msg1 += "\n⚠️ 量化评分仅供参考，不构成投资建议。"
    else:
        msg1 = f"🏛 *主板推荐*\n\n{today} 暂无符合条件的标的。"
    
    # Message 2: 全A (top 5)
    all_top5 = all_filtered[:5]
    if all_top5:
        msg2 = f"📊 *全A推荐 · 评分前5（含创业板/科创板）*\n📅 {today} 开盘前\n\n" + "\n".join(fmt_stock(s, i) for i, s in enumerate(all_top5, 1))
        msg2 += "\n⚠️ 量化评分仅供参考，不构成投资建议。"
    else:
        msg2 = f"📊 *全A推荐*\n\n{today} 暂无符合条件的标的。"
    
    # Send
    print("\n📤 发送主板推荐...")
    r1 = send_telegram(msg1)
    print(f"  {'✅' if r1 else '❌'}")
    time.sleep(1.5)
    
    print("📤 发送全A推荐...")
    r2 = send_telegram(msg2)
    print(f"  {'✅' if r2 else '❌'}")
    
    # Update record
    for s in mb_top5 + all_top5:
        if s['code'] not in recommend_record['today_codes']:
            recommend_record['today_codes'].append(s['code'])
    save_json(os.path.join(WORKSPACE, 'data', 'recommend_history.json'), recommend_record)
    
    # Summary
    seen = set()
    all_printed = []
    for s in mb_top5 + all_top5:
        if s['code'] not in seen:
            seen.add(s['code'])
            all_printed.append(s)
    
    print(f"\n{'='*50}")
    print("今日推荐:")
    for s in all_printed:
        label = "主板" if s['is_mainboard'] else "全A"
        sent = s.get('sentiment', '')
        print(f"  {label} | {s['name']}({s['code']}) 评分{s['score']} {sent}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
