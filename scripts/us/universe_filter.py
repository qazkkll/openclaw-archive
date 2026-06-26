"""
宇宙过滤规则 — 蓝盾/绿箭共用
==============================
绿箭($1-$10): 小盘股，重在排除退市风险和垃圾
蓝盾(>$10): 中大盘股，重在排除杠杆产品和SPAC
"""

# 杠杆/反向ETF黑名单（定期更新）
LEVERAGED_ETFS = {
    # 单股杠杆
    'AAPU','AAPD','AMZU','AMZD','MSFU','MSFD','GOOX','GOOG',
    'NFLU','METU','PLTU','HOOD','NVDL','NVDX','NVDU',
    'TSLT','TSLL','TSLZ','TSLQ','CONL',
    # 行业杠杆
    'SOXL','SOXS','LABU','LABD','FAS','FAZ','TNA','TZA',
    'TECL','TECS','FNGU','FNGD','BULZ','BERZ',
    'DFEN','DUSL','DPST','RETL','CURE',
    'HIBL','HIBS','PILL','MWJ','FLYD','FLYU',
    'SVIX','UVIX','UVXY',
    # 指数杠杆
    'UPRO','SPXU','UDOW','SDOW','UMDD','SMDD','URTY','SRTY',
    'SPXS','SPXL','SQQQ','TQQQ','QLD','QID',
    'NUGT','DUST','JNUG','JDST',
    # VIX杠杆
    'VXX','UVXY','SVXY','VIXY','VIXM',
    # 其他已知
    'ARTNA','BSMC','SMCX','SMCL','GDXD','RKLX','GUSH','DRIP',
}

# 杠杆ETF名称模式
LEVERAGED_PATTERNS = [
    '2X', '3X', '4X', '5X', '-1X', '-2X', '-3X',
    'BULL', 'BEAR', 'ULTRA', 'SHORT', 'LONG',
    'DAILY', 'WEEKLY',
]


def filter_green_arrow(df, min_price=1.0, min_dollar_vol=500_000):
    """
    绿箭V12实战宇宙过滤 ($1-$10)
    ──────────────────────────────
    训练池: $1-$10全量（无流动性过滤，保留最大数据量）
    实战池: $1-$10 + 以下过滤（保证可交易性）

    规则设计逻辑：
    - $1.00底价: 保留$1-$2但需更高流动性（$200万/日），$2以上只需$50万/日
    - 排除权证(W): 权证有到期日和行权价，不是股票，模型不适用
    - 排除SPAC单位(U): 未完成合并的空壳，没有业务
    - 排除权利(R): 有行权期限，非普通股
    - 排除杠杆ETF: $1-$10区间也有(如TECS $6.99)
    - 流动性分层: 低价股($1-$2)要求$200万/日，中价股($2+)要求$50万/日
    """
    mask = (
        (df['close'] >= min_price) &
        (df['close'] <= 10)
    )
    # 排除非普通股
    sym_mask = (
        ~df['sym'].str.endswith('W') &        # 权证
        ~df['sym'].str.endswith('U') &         # SPAC单位
        ~df['sym'].str.endswith('R') &         # 权利(rights)
        ~df['sym'].isin(LEVERAGED_ETFS)        # 杠杆ETF
    )
    df = df[mask & sym_mask].copy()

    # 流动性过滤：分层要求
    # $1-$2: 日均$200万（低价股必须高流动性才可交易）
    # $2+:   日均$50万
    if 'dollar_vol_20d' in df.columns:
        low_liq = (df['close'] < 2) & (df['dollar_vol_20d'] < 2_000_000)
        high_liq = (df['close'] >= 2) & (df['dollar_vol_20d'] < min_dollar_vol)
        df = df[~(low_liq | high_liq)]

    return df


def filter_blue_shield(df, min_price=10.0, min_dollar_vol=5_000_000):
    """
    蓝盾V10宇宙过滤 (>$10)
    规则设计逻辑：
    - 排除杠杆/反向ETF: 黑名单+名称模式双保险，LambdaMART标签对杠杆产品无意义
    - 排除SPAC单位(U): 同绿箭
    - 排除权证(W): 同绿箭
    - 最低日均美元成交量>$5M: 蓝盾是中大盘，流动性应更高
    - 名称模式排除: 新上市的杠杆产品可能不在黑名单中
    """
    mask = df['close'] >= min_price
    df = df[mask].copy()

    # 排除非普通股 + 杠杆ETF
    sym_upper = df['sym'].str.upper()
    sym_mask = (
        ~df['sym'].str.endswith('W') &
        ~df['sym'].str.endswith('U') &
        ~df['sym'].str.endswith('R') &
        ~df['sym'].isin(LEVERAGED_ETFS) &
        ~sym_upper.str.contains('|'.join(LEVERAGED_PATTERNS), regex=True, na=False)
    )
    df = df[sym_mask].copy()

    # 流动性过滤
    if 'dollar_vol_20d' in df.columns:
        df = df[df['dollar_vol_20d'] >= min_dollar_vol]

    return df
