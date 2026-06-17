#!/usr/bin/env python3
"""生成因子卡片图片 — 小钳策略 v4"""
from PIL import Image, ImageDraw, ImageFont
import os

FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
FONT_REG = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

def font_sz(sz):
    try:
        return ImageFont.truetype(FONT_REG, sz)
    except:
        return ImageFont.load_default()

def font_b(sz):
    try:
        return ImageFont.truetype(FONT_BOLD, sz)
    except:
        return ImageFont.load_default()

def font_m(sz):
    try:
        return ImageFont.truetype(FONT_MONO, sz)
    except:
        return ImageFont.load_default()

def rounded_rect(draw, xy, r, fill):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, r, fill=fill)

def draw_asean_card():
    W, H = 900, 1200
    img = Image.new("RGB", (W, H), "#0f1419")
    d = ImageDraw.Draw(img)

    # ── Title ──
    d.text((40, 30), "📊 A股 — V1 原始评分", fill="#e2e8f0", font=font_b(26))

    d.text((40, 75), "MACD门控：MACD柱 > 0 → 进入评分 / ≤ 0 → 直接0分（熊市自动过滤）", fill="#94a3b8", font=font_sz(14))

    # ── 5 Factors ──
    factors = [
        ("MACD 状态", "ms", [
            "刚金叉（柱刚转正）= 20分",
            "MACD线 > 信号线持续上涨 = 12分",
            "仅柱为正 = 6分",
        ], "#6366f1", "判断趋势启动时机"),
        ("52周位置", "ws", [
            "价格 < 52周高点20%以内 = 20分",
            "< 35% = 15分  /  < 50% = 10分",
            "< 65% = 6分   /  < 80% = 3分",
        ], "#8b5cf6", "越接近52周低点越值"),
        ("均线系统", "mas", [
            "站上MA20（价>MA20）= +7分",
            "MA5 > MA20（短期向上）= +7分",
            "MA20 > MA60（中期走好）= +6分",
        ], "#06b6d4", "均线多头排列加分"),
        ("ADX 趋势", "ads", [
            "ADX ≥ 35（极强趋势）= 20分",
            "≥ 28 = 15分  /  ≥ 22 = 10分",
            "≥ 18 = 5分  /  < 18 = -5分",
        ], "#f59e0b", "趋势越强分越高"),
        ("RSI 超卖", "rs", [
            "RSI < 25（严重超卖）= 20分",
            "< 35 = 14分  /  < 50 = 10分",
            "< 65 = 6分  /  < 75 = 2分",
            "> 75（超买）= -5分",
        ], "#ef4444", "超卖加分，超买扣分"),
    ]

    y = 110
    for name, sym, descs, clr, note in factors:
        # Card bg
        rounded_rect(d, (30, y, W-30, y+100), 10, "#1a1f2e")
        # Left color bar
        d.rectangle([30, y, 38, y+100], fill=clr)
        # Name + symbol
        d.text((52, y+8), f"{name}  ({sym})", fill="#e2e8f0", font=font_b(18))
        d.text((52, y+33), note, fill="#94a3b8", font=font_sz(13))
        # Score rules
        for i, line in enumerate(descs):
            d.text((52, y+55+i*15), line, fill="#cbd5e1", font=font_sz(13))
        y += 110

    # ── Dynamic Weights ──
    y += 10
    rounded_rect(d, (30, y, W-30, y+90), 10, "#1a1f2e")
    d.rectangle([30, y, 38, y+90], fill="#22c55e")
    d.text((52, y+10), "⚖️ 动态权重（按ADX自动切换）", fill="#e2e8f0", font=font_b(17))
    d.text((52, y+38), "趋势股（ADX≥22）:  [25  +  15  +  15  +  25  +  20]", fill="#86efac", font=font_m(14))
    d.text((52, y+55), "震荡股（ADX<22）:  [10  +  30  +  15  +  10  +  35]", fill="#86efac", font=font_m(14))
    d.text((52, y+72), "  ↑MACD    ↑52W位置    ↑均线    ↑ADX    ↑RSI", fill="#64748b", font=font_sz(11))
    y += 105

    # ── Signal Lights ──
    rounded_rect(d, (30, y, W-30, y+75), 10, "#1a1f2e")
    d.rectangle([30, y, 38, y+75], fill="#f97316")
    d.text((52, y+10), "🚦 五档信号灯", fill="#e2e8f0", font=font_b(17))
    lights = [("🟢 买入", 62), ("🔵 关注", "57-61"), ("🟡 持有", "51-56"), ("🟠 警惕", "47-50"), ("🔴 卖出", "<47")]
    lx = 52
    for emoji, val in lights:
        d.text((lx, y+38), f"{emoji}  {val}分", fill="#e2e8f0", font=font_sz(13))
        lx += 155

    y += 90
    # ── Footer ──
    d.text((40, y), "数据源：新浪240分钟K线 | 候选池：全市场V4直选", fill="#475569", font=font_sz(12))
    d.text((40, y+18), "Top8等权12.5% | 最短持有5天 | 换仓条件：评分从买入跌≥15分", fill="#475569", font=font_sz(12))

    out = "/home/admin/.openclaw/workspace/data/factor_card_asean.png"
    img.save(out)
    print(f"Saved: {out}")
    return out

def draw_us_card():
    W, H = 900, 1200
    img = Image.new("RGB", (W, H), "#0f1419")
    d = ImageDraw.Draw(img)

    d.text((40, 30), "📊 美股 — V3 双模", fill="#e2e8f0", font=font_b(26))
    d.text((40, 75), "市场状态识别器：SPY vs MA200 — 上方=牛市 ｜ 下方=熊市", fill="#94a3b8", font=font_sz(14))

    # ── Bull Mode ──
    y = 110
    rounded_rect(d, (30, y, W-30, y+65), 10, "#1a1f2e")
    d.rectangle([30, y, 38, y+65], fill="#22c55e")
    d.text((52, y+10), "🟢 牛市模式（SPY > MA200）", fill="#e2e8f0", font=font_b(18))
    d.text((52, y+38), "纯动量追涨 — 选20日涨幅最高的5只，每20天调仓轮出", fill="#cbd5e1", font=font_sz(14))

    # ── Bear Mode Title ──
    y += 80
    rounded_rect(d, (30, y, W-30, y+55), 10, "#1a1f2e")
    d.rectangle([30, y, 38, y+55], fill="#ef4444")
    d.text((52, y+10), "🔴 熊市模式（SPY < MA200）", fill="#e2e8f0", font=font_b(18))
    d.text((52, y+32), "V2逆向防守评分 — 5因子加权 + MACD门控", fill="#cbd5e1", font=font_sz(14))

    # ── Bear Mode Factors ──
    y += 70
    factors = [
        ("52周位置", 30, "越低越好，偏重抄底", "#8b5cf6"),
        ("ADX趋势", 20, "判断趋势强度", "#f59e0b"),
        ("RSI超卖", 20, "超卖加分，反弹预期", "#ef4444"),
        ("MACD状态", 15, "MACD门控 + 状态判断", "#6366f1"),
        ("均线系统", 15, "站上均线加分", "#06b6d4"),
    ]

    for name, weight, note, clr in factors:
        rounded_rect(d, (30, y, W-30, y+60), 10, "#1a1f2e")
        d.rectangle([30, y, 38, y+60], fill=clr)
        # Weight badge
        rounded_rect(d, (52, y+8, 90, y+38), 6, "#1e293b")
        d.text((60, y+14), f"{weight}", fill="#e2e8f0", font=font_b(14))
        d.text((100, y+12), name, fill="#e2e8f0", font=font_b(17))
        d.text((100, y+35), note, fill="#94a3b8", font=font_sz(13))
        # Weight bar
        bw = weight * 7
        d.rectangle([420, y+15, 420+bw, y+35], fill=clr)
        d.text((430+bw, y+17), f"{weight}%", fill="#64748b", font=font_sz(11))
        y += 70

    # ── Signal Lights (US) ──
    rounded_rect(d, (30, y, W-30, y+70), 10, "#1a1f2e")
    d.rectangle([30, y, 38, y+70], fill="#f97316")
    d.text((52, y+10), "🚦 美股五档信号灯", fill="#e2e8f0", font=font_b(17))
    lights = [("🟢 买入", 60), ("🔵 关注", "50-59"), ("🟡 持有", "35-49"), ("🟠 警惕", "25-34"), ("🔴 卖出", "<25")]
    lx = 52
    for emoji, val in lights:
        d.text((lx, y+38), f"{emoji}  {val}分", fill="#e2e8f0", font=font_sz(13))
        lx += 155

    y += 85
    # ── Footer ──
    d.text((40, y), "数据源：Yahoo Finance | 识别器：每20天检查SPY-MA200位置", fill="#475569", font=font_sz(12))
    d.text((40, y+18), "最大5只持仓 | 最短持有7天 | 牛市追涨 / 熊市防守自动切换", fill="#475569", font=font_sz(12))

    out = "/home/admin/.openclaw/workspace/data/factor_card_us.png"
    img.save(out)
    print(f"Saved: {out}")
    return out

if __name__ == "__main__":
    a = draw_asean_card()
    u = draw_us_card()
    print("Done!")
