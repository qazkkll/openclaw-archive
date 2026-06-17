"""
🍤 统一数据源层
================
所有脚本通过此模块获取数据，改 config/data_sources.json 即可切换源。

支持板块: 上证主板(60xx) / 深证主板(00xx) / 创业板(30xx) / 科创板(68xx)

用法:
    from data_source import AShareKline
    ds = AShareKline()
    data = ds.get_kline('600850')  # 自动走 primary
    data = ds.get_kline('600850', source='tushare')  # 强制指定
"""
import json, urllib.request, time, os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'data_sources.json')

def _load_config():
    with open(CONFIG_PATH, encoding='utf-8') as f:
        return json.load(f)

def code_to_prefix(code):
    """根据股票代码返回交易所前缀"""
    if code == '000001':
        return 'sh'  # 上证指数
    first_two = code[:2] if len(code) >= 2 else ''
    if first_two in ('60', '68'):
        return 'sh'
    elif first_two in ('00', '30', '001'):
        return 'sz'
    elif first_two == '92':
        return 'bj'
    return 'sh'

def code_to_board(code):
    """根据股票代码返回板块名称"""
    first_two = code[:2] if len(code) >= 2 else ''
    if first_two == '60':
        return '上证主板'
    elif first_two == '68':
        return '科创板'
    elif first_two == '30':
        return '创业板'
    elif first_two == '00':
        return '深证主板'
    return '其他'

def code_to_tscode(code):
    """转Tushare ts_code格式"""
    if '.' in code:
        return code
    prefix = code[:2] if len(code) >= 2 else ''
    if prefix in ('60', '68'):
        return code + '.SH'
    else:
        return code + '.SZ'

class AShareKline:
    """A股日K线数据"""
    
    def __init__(self, days=None):
        cfg = _load_config()
        self.primary = cfg['a_share_kline']['primary']
        self.secondary = cfg['a_share_kline'].get('secondary')
        self.sources = cfg['a_share_kline']['sources']
        # 从strategy.json读默认天数
        if days is None:
            try:
                strat = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'strategy.json'), encoding='utf-8'))
                self.default_days = strat.get('a_stock', {}).get('kline_days', 120)
            except:
                self.default_days = 120
        else:
            self.default_days = days
    
    def get_kline(self, code, days=None, source=None, reuse_login=False):
        """获取日K线数据，返回 [{'close','high','low','open','volume','day'}, ...] 或 None"""
        if days is None:
            days = self.default_days
        source = source or self.primary
        if source == 'sina':
            return self._from_sina(code, days)
        elif source == 'tushare':
            return self._from_tushare(code, days)
        elif source == 'baostock':
            return self._from_baostock(code, days, reuse_login=reuse_login)
        return self._from_sina(code, days)
    
    def _from_sina(self, code, days=120):
        prefix = code_to_prefix(code)
        url = f"{self.sources['sina']['url']}?symbol={prefix}{code}&scale=240&datalen={days}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            if not isinstance(data, list) or len(data) < 20:
                return None
            return [{
                'close': float(d['close']),
                'high': float(d['high']),
                'low': float(d['low']),
                'open': float(d['open']),
                'volume': int(float(d.get('volume', 0))),
                'day': d['day']
            } for d in data]
        except:
            return None
    
    def _from_tushare(self, code, days=120):
        cfg = self.sources['tushare']
        ts_code = code_to_tscode(code)
        import datetime
        end = datetime.date.today()
        start = end - datetime.timedelta(days=days + 30)
        payload = json.dumps({
            "api_name": "daily",
            "token": cfg['token'],
            "params": {"ts_code": ts_code, "start_date": start.strftime('%Y%m%d'), "end_date": end.strftime('%Y%m%d')}
        }).encode()
        req = urllib.request.Request(cfg['url'], data=payload, headers={'Content-Type': 'application/json'})
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            result = json.loads(resp.read())
            items = result.get('data', {}).get('items', [])
            fields = result.get('data', {}).get('fields', [])
            if not items or len(items) < 20:
                return None
            idx_c = fields.index('close')
            idx_h = fields.index('high')
            idx_l = fields.index('low')
            idx_o = fields.index('open')
            idx_v = fields.index('vol')
            idx_d = fields.index('trade_date')
            items.reverse()
            return [{
                'close': float(item[idx_c]),
                'high': float(item[idx_h]),
                'low': float(item[idx_l]),
                'open': float(item[idx_o]),
                'volume': int(float(item[idx_v])),
                'day': str(item[idx_d])
            } for item in items]
        except:
            return None
    
    def _from_baostock(self, code, days=120, reuse_login=False):
        import baostock as bs
        import datetime
        prefix = code_to_prefix(code)
        bs_code = prefix + '.' + code
        end = datetime.date.today()
        start = end - datetime.timedelta(days=days + 30)
        try:
            if not reuse_login:
                lg = bs.login()
                if lg.error_code != '0':
                    return None
            rs = bs.query_history_k_data_plus(bs_code,
                "date,open,high,low,close,volume,pctChg",
                start_date=start.strftime('%Y-%m-%d'),
                end_date=end.strftime('%Y-%m-%d'),
                frequency='d', adjustflag='3')
            data = []
            while (rs.error_code == '0') & rs.next():
                row = rs.get_row_data()
                if row[4] and float(row[4]) > 0:
                    data.append({
                        'close': float(row[4]),
                        'high': float(row[2]),
                        'low': float(row[3]),
                        'open': float(row[1]),
                        'volume': int(float(row[5])),
                        'day': row[0]
                    })
            if not reuse_login:
                bs.logout()
            if len(data) < 30:
                return None
            return data
        except:
            if not reuse_login:
                try:
                    bs.logout()
                except:
                    pass
            return None

    def get_best(self, code, days=120):
        """先走primary，失败走secondary"""
        data = self.get_kline(code, days, source=self.primary)
        if data is None and self.secondary:
            data = self.get_kline(code, days, source=self.secondary)
        return data


class AShareRealtime:
    """A股实时行情"""
    
    def __init__(self):
        cfg = _load_config()
        self.source = cfg['a_share_realtime']['primary']
    
    def get_quote(self, code):
        if self.source == 'tencent':
            return self._from_tencent(code)
        return self._from_tencent(code)
    
    def _from_tencent(self, code):
        prefix = code_to_prefix(code)
        url = f"http://qt.gtimg.cn/q={prefix}{code}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            raw = urllib.request.urlopen(req, timeout=5).read()
            text = raw.decode('GBK', errors='replace')
            parts = text.split('"')[1].split('~')
            return {
                'code': code,
                'name': parts[1],
                'price': float(parts[3]) if parts[3] else 0,
                'prev_close': float(parts[4]) if parts[4] else 0,
                'open': float(parts[5]) if parts[5] else 0,
                'high': float(parts[33]) if parts[33] else 0,
                'low': float(parts[34]) if parts[34] else 0,
                'volume': int(parts[6]) if parts[6] else 0,
                'change_pct': float(parts[32]) if len(parts) > 32 and parts[32] else 0
            }
        except:
            return None


# ===== 兼容层 =====
_kl = AShareKline()
_rt = AShareRealtime()

def fetch_kline(code, days=120):
    return _kl.get_best(code, days)

def get_realtime_price(code):
    return _rt.get_quote(code)
