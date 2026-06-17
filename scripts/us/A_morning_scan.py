#!/usr/bin/env python3
"""
🍤 小钳晨扫 — 质量池Top 500

数据源: 质量池(data/quality_pool.json) — 基本面达标+换手率排名前500
覆盖: 主板 + 创业板 + 科创板
数据源: 统一走 data_source 层，切源改 config/data_sources.json
输出: TG推送 + data/morning_report.txt
"""
import sys, json, time, os, urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_engine import v1_score_from_data
from data_source import AShareKline, AShareRealtime, code_to_board

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_MARKET = os.path.join(ROOT, 'data', 'full_market_stocks.json')
PORTFOLIO = os.path.join(ROOT, 'data', 'portfolio.json')
QUALITY_POOL = os.path.join(ROOT, 'data', 'quality_pool.json')

kl = AShareKline()
rt = AShareRealtime()

# 加载策略参数
with open(os.path.join(ROOT, 'config', 'strategy.json'), encoding='utf-8') as f:
    STRATEGY = json.load(f)
A_CFG = STRATEGY['a_stock']
BUY_THRESHOLD = A_CFG['buy_threshold']
SELL_THRESHOLD = A_CFG['sell_threshold']

def get_index():
    data = kl.get_kline('000001', 120, source='sina')
    if data and len(data) >= 60:
        close = [d['close'] for d in data]
        last = close[-1]
        prev = close[-2] if len(close) > 1 else last
        ma20 = sum(close[-20:]) / 20
        return {
            'index': last,
            'change_pct': (last - prev) / prev * 100,
            'ma20': ma20,
            'is_bear': last < ma20
        }
    return None

def main():
    t0 = time.time()
    
    # 读取质量池 → 全量扫描（避免只取Top 100错过高分股）
    with open(QUALITY_POOL, encoding='utf-8') as f:
        pool = json.load(f)
    
    all_stocks = pool.get('stocks', [])
    
    # 全量扫描（不限Top 100，按活跃度排完扫全部）
    scan_stocks = all_stocks
    # 记录总池量用于审计
    _audit_total = len(all_stocks)
    scan_codes = [s['code'] for s in scan_stocks]
    stock_map = {s['code']: s for s in scan_stocks}
    
    positions = {}
    try:
        with open(PORTFOLIO, encoding='utf-8') as f:
            pf = json.load(f)
        for p in pf.get('a_stock', []):
            positions[p['code']] = {'name': p['name'], 'cost': p['cost']}
    except:
        pass
    
    index = get_index()
    results = []
    errors = 0
    
    for i, code in enumerate(scan_codes):
        try:
            data = kl.get_best(code)
            if not data or len(data) < 60:
                errors += 1
                continue
            close = [d['close'] for d in data]
            high = [d['high'] for d in data]
            low = [d['low'] for d in data]
            score = v1_score_from_data(close, high, low)
            if score is None:
                errors += 1
                continue
            if (i + 1) % 300 == 0:
                print(f'  [{time.strftime("%H:%M")}] {i+1}/{len(scan_codes)} | 有效: {len(results)} | 跳过: {errors}', file=sys.stderr)
            
            board = stock_map.get(code, {}).get('board', code_to_board(code))
            results.append({
                'code': code,
                'name': stock_map.get(code, {}).get('name', code),
                'board': board,
                'score': round(float(score), 0),
                'price': close[-1],
                'change_pct': 0
            })
        except:
            errors += 1
            continue
    
    results.sort(key=lambda x: x['score'], reverse=True)
    elapsed = time.time() - t0
    
    # ===== 报告（新格式：只说结论，不堆数字）=====
    src_name = kl.sources[kl.primary]['name']
    
    # 获取当前市场模式
    mode_state = {'mode': '牛市', 'last_check_pct': '?'}
    try:
        with open(os.path.join(ROOT, 'data', 'market_mode.json'), encoding='utf-8') as f:
            mode_state = json.load(f)
    except:
        pass
    market_mode = mode_state['mode']
    market_strength = mode_state.get('last_check_pct', '?')
    market_icon = '🚀' if market_mode == '牛市' else '🛡️'
    
    lines = []
    lines.append('📊 A股早报')
    lines.append(f'{market_icon} 市场模式: {mode_state["mode"]} (全市场强度: {mode_state["last_check_pct"]}%)')
    lines.append('')
    
    # 大盘判断
    if index:
        if index['is_bear']:
            lines.append('🔴 大盘偏弱，上证在MA20下方，建议谨慎')
        else:
            lines.append('🟢 大盘健康，上证在MA20上方')
        lines.append('')
    
    # 持仓判断
    lines.append('📋 持仓')
    for code, info in positions.items():
        for r in results:
            if r['code'] == code:
                score = r['score']
                if score >= BUY_THRESHOLD:
                    lines.append(f'{info["name"]} → ✅ 持有（技术面偏多）')
                elif score >= SELL_THRESHOLD:
                    lines.append(f'{info["name"]} → ⚠️ 观望（趋势中性，等方向）')
                else:
                    lines.append(f'{info["name"]} → ❌ 卖出（趋势走弱，不建议留）')
                break
    lines.append('')
    
    # 分类取Top 10
    main_board = [r for r in results if r['board'] in ('上证主板', '深证主板')]
    other_board = [r for r in results if r['board'] in ('创业板', '科创板')]
    
    # 对Top补实时行情（盘前会失败，fallback到K线收盘价）
    now_hour = int(time.strftime('%H'))
    is_market_hours = 9 <= now_hour <= 14
    _price_source = '实时' if is_market_hours else '前收盘'
    
    for r in main_board[:10] + other_board[:10]:
        try:
            q = rt.get_quote(r['code'])
            if q and q.get('price', 0) > 0:
                r['price'] = q['price']
                r['change_pct'] = q['change_pct']
                r['name'] = q['name']
        except:
            pass
    for r in results:
        if r['code'] in positions:
            try:
                q = rt.get_quote(r['code'])
                if q and q.get('price', 0) > 0:
                    r['price'] = q['price']
            except:
                pass
    
    # 主板 - 评分达标(>=62)的
    buyable_m = [r for r in main_board if r['score'] >= BUY_THRESHOLD]
    watch_m = [r for r in main_board if SELL_THRESHOLD <= r['score'] < BUY_THRESHOLD]
    top_m = (buyable_m or watch_m)[:10]
    
    # 非主板
    buyable_o = [r for r in other_board if r['score'] >= BUY_THRESHOLD]
    watch_o = [r for r in other_board if SELL_THRESHOLD <= r['score'] < BUY_THRESHOLD]
    top_o = (buyable_o or watch_o)[:10]
    
    # 接近买入线(50-61)
    near_m = [r for r in main_board if BUY_THRESHOLD > r['score'] >= 50 and r not in top_m][:5]
    near_o = [r for r in other_board if BUY_THRESHOLD > r['score'] >= 50 and r not in top_o][:5]
    
    lines.append('──────────────────────────────────')
    lines.append('❤️ 评分红绿灯 · 今日推荐（主板买入线62，非主板62）')
    lines.append('')
    
    # 主板推荐
    if buyable_m:
        lines.append('📊 主板 → Andy买')
        for r in buyable_m[:8]:
            p = f' ¥{r.get("price","?")}' if r.get('price') else ''
            cp = f'({r.get("change_pct",0):+.2f}%)' if r.get('change_pct') else ''
            lines.append(f'  🟢 {r["name"]} ({r["code"]}) {r["score"]}分{p} {cp}')
    else:
        lines.append('📊 主板 — 暂无达标买入信号（需≥62分）')
        if watch_m:
            for r in watch_m[:5]:
                p = f' ¥{r.get("price","?")}' if r.get('price') else ''
                cp = f'({r.get("change_pct",0):+.2f}%)' if r.get('change_pct') else ''
                lines.append(f'  🟡 {r["name"]} ({r["code"]}) {r["score"]}分{p} {cp}')
    lines.append('')
    
    # 非主板推荐
    if buyable_o:
        lines.append('📈📡 创业板/科创板 → 妈妈买')
        for r in buyable_o[:5]:
            marker = '📈' if '创业' in r.get('board','') else '📡'
            p = f' ¥{r.get("price","?")}' if r.get('price') else ''
            lines.append(f'  🟢 {marker} {r["name"]} ({r["code"]}) {r["score"]}分{p}')
    else:
        lines.append('📈📡 创业板/科创板 — 暂无达标买入信号')
        if watch_o:
            for r in watch_o[:5]:
                marker = '📈' if '创业' in r.get('board','') else '📡'
                p = f' ¥{r.get("price","?")}' if r.get('price') else ''
                lines.append(f'  🟡 {marker} {r["name"]} ({r["code"]}) {r["score"]}分{p}')
    lines.append('')
    
    # 接近买入线
    if near_m or near_o:
        lines.append('👀 接近买入线（50-61分，未达标但值得关注）')
        for r in near_m:
            p = f' ¥{r.get("price","?")}' if r.get('price') else ''
            lines.append(f'  📊 {r["name"]} ({r["code"]}) {r["score"]}分{p} → Andy关注')
        for r in near_o:
            marker = '📈' if '创业' in r.get('board','') else '📡'
            p = f' ¥{r.get("price","?")}' if r.get('price') else ''
            lines.append(f'  {marker} {r["name"]} ({r["code"]}) {r["score"]}分{p} → 妈妈关注')
        lines.append('')
    
    # 数据源标注
    today_str = time.strftime('%m-%d')
    time_note = '前交易日收盘' if int(time.strftime('%H')) < 9 or int(time.strftime('%H')) >= 15 else '盘中'  # 盘前和盘后都标前收盘
    lines.append(f'📡 数据源: V1评分(新浪K线) | 从质量池{len(all_stocks)}只→全量扫描={len(scan_stocks)}只 | 价格来源:{_price_source} | 基础数据截至{today_str} {time_note}')
    
    # 统计达标数（用于审计上报）
    _qualified_cnt = len(buyable_m) + len(buyable_o)
    
    # 如果无买入信号，追加防守配置建议
    if not buyable_m and not buyable_o:
        lines.append('')
        lines.append('──────────────────────────────────')
        try:
            from defensive import get_defensive_suggestion
            def_text = get_defensive_suggestion(market_mode, market_strength)
            def_lines = def_text.split('\n')
            for dl in def_lines[:12]:
                lines.append(dl)
        except Exception as e:
            try:
                lines.append(f'防守配置加载失败: {e}')
            except:
                pass
    
    # ===== 今日要闻 =====
    try:
        news_lines = []
        for source_url in [
            'https://finnhub.io/api/v1/news?category=general&minId=10&token=d87hklhr01qmhakfrh10',
            'https://newsapi.org/v2/everything?q=stock+market&pageSize=3&apiKey=7d8e0ca352664b6d9ccd96405949b5ea'
        ]:
            try:
                req = urllib.request.Request(source_url)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                items = data if isinstance(data, list) else data.get('articles', data.get('items', []))
                for item in items[:3]:
                    h = item.get('headline', item.get('title', ''))
                    if h and all(h not in nl for nl in news_lines):
                        news_lines.append(f'  📰 {h[:120]}')
                if news_lines:
                    break
            except:
                continue
        if news_lines:
            lines.append('')
            lines.append('📰 今日要闻')
            lines.extend(news_lines)
    except:
        pass
    
    report = '\n'.join(lines)
    print(report)
    
    out_path = os.path.join(ROOT, 'data', 'morning_report.txt')
    with open(out_path, 'w') as f:
        f.write(report)
    
    return report, len(results), errors, len(scan_stocks), _qualified_cnt

if __name__ == '__main__':
    from notify import send
    try:
        report, valid_cnt, err_cnt, total_cnt, qualified_cnt = main()
        send(report)
        
        # 自我校验：报告发出后检查格式合规
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from verify_report import check_report
        result = check_report(os.path.join(ROOT, 'data', 'morning_report.txt'))
        if not result['passed']:
            alert = '🚨 晨扫报告格式不合规⚠️\n'
            alert += '\n'.join(result['failures'])
            send(alert)
        
        # 审计记录 — 带质量参数
        err_rate = err_cnt / max(total_cnt, 1) * 100
        level = 'success'
        if err_rate > 50:
            level = 'error'
        elif err_rate > 30:
            level = 'warning'
        
        from audit_engine import audit
        audit('morning_scan', level, f'晨扫: {valid_cnt}仅有分/{total_cnt}总扫描, 错误率{err_rate:.0f}%, 达标{qualified_cnt}只')
        
        # 合规检查
        try:
            from compliance import check_compliance
            check_compliance('A股晨扫', stocks_count=valid_cnt, scoring='V1', source='sina', universe_max=total_cnt)
        except ImportError:
            pass
        except Exception as ce:
            print(f'[compliance] 合规检查失败: {ce}', file=sys.stderr)
    except Exception as e:
        # 捕获任何异常 → 记录审计错误 + 推送Andy
        err_msg = f'晨扫执行失败: {e}'
        try:
            from audit_engine import audit
            audit('morning_scan', 'error', err_msg, traceback.format_exc())
        except:
            pass
        try:
            send(f'🚨 晨扫异常: {err_msg}')
        except:
            pass
        raise  # 让cron日志也能看见
