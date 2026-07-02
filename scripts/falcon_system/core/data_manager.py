"""
🦅 Falcon 数据管理器
====================
统一数据访问层，自动更新+新鲜度检查。
数据源可替换：只需实现对应的adapter接口。
"""

import json
import os
import sys
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from .config import CONFIG, DATA_CONFIG, DATA_DIR, PROJECT_ROOT


# ════════════════════════════════════════════════════════════════
# 数据源接口 (可替换)
# ════════════════════════════════════════════════════════════════

class PriceDataSource(ABC):
    """价格数据源接口"""
    
    @abstractmethod
    def get_latest_prices(self, tickers: List[str]) -> pd.DataFrame:
        """获取最新价格"""
        pass
    
    @abstractmethod
    def get_historical_prices(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """获取历史价格"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查数据源是否可用"""
        pass


class FundamentalDataSource(ABC):
    """基本面数据源接口"""
    
    @abstractmethod
    def get_financial_ratios(self, ticker: str) -> List[Dict]:
        """获取财务比率历史"""
        pass
    
    @abstractmethod
    def get_analyst_estimates(self, ticker: str) -> List[Dict]:
        """获取分析师预期"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查数据源是否可用"""
        pass


# ════════════════════════════════════════════════════════════════
# 具体数据源实现
# ════════════════════════════════════════════════════════════════

class YFinancePriceSource(PriceDataSource):
    """Yahoo Finance价格数据源"""
    
    def get_latest_prices(self, tickers: List[str]) -> pd.DataFrame:
        """获取最新价格"""
        try:
            import yfinance as yf
            data = yf.download(tickers, period="1d", progress=False)
            if data.empty:
                return pd.DataFrame()
            
            # 提取收盘价
            if isinstance(data.columns, pd.MultiIndex):
                prices = data["Close"].iloc[-1]
            else:
                prices = data["Close"]
            
            return pd.DataFrame({
                "ticker": prices.index,
                "close": prices.values,
                "date": datetime.now().strftime("%Y-%m-%d"),
            })
        except Exception as e:
            print(f"⚠️ YFinance获取价格失败: {e}")
            return pd.DataFrame()
    
    def get_historical_prices(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """获取历史价格"""
        try:
            import yfinance as yf
            data = yf.download(tickers, start=start, end=end, progress=False)
            if data.empty:
                return pd.DataFrame()
            
            # 转换为长格式
            if isinstance(data.columns, pd.MultiIndex):
                records = []
                for ticker in tickers:
                    if ticker in data.columns.get_level_values(1):
                        ticker_data = data.xs(ticker, level=1, axis=1)
                        for date, row in ticker_data.iterrows():
                            records.append({
                                "ticker": ticker,
                                "date": date.strftime("%Y-%m-%d"),
                                "open": row.get("Open"),
                                "high": row.get("High"),
                                "low": row.get("Low"),
                                "close": row.get("Close"),
                                "volume": row.get("Volume"),
                            })
                return pd.DataFrame(records)
            else:
                # 单个ticker
                records = []
                for date, row in data.iterrows():
                    records.append({
                        "ticker": tickers[0],
                        "date": date.strftime("%Y-%m-%d"),
                        "open": row.get("Open"),
                        "high": row.get("High"),
                        "low": row.get("Low"),
                        "close": row.get("Close"),
                        "volume": row.get("Volume"),
                    })
                return pd.DataFrame(records)
        except Exception as e:
            print(f"⚠️ YFinance获取历史价格失败: {e}")
            return pd.DataFrame()
    
    def is_available(self) -> bool:
        """检查YFinance是否可用"""
        try:
            import yfinance
            return True
        except ImportError:
            return False


class FMPPriceSource(PriceDataSource):
    """FMP价格数据源"""
    
    def __init__(self):
        self.api_key = os.environ.get("FMP_API_KEY")
        self.base_url = "https://financialmodelingprep.com/api/v3"
    
    def get_latest_prices(self, tickers: List[str]) -> pd.DataFrame:
        """获取最新价格"""
        if not self.api_key:
            return pd.DataFrame()
        
        try:
            import requests
            records = []
            for ticker in tickers[:50]:  # 限制请求数
                url = f"{self.base_url}/quote/{ticker}?apikey={self.api_key}"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        records.append({
                            "ticker": ticker,
                            "close": data[0].get("price"),
                            "date": datetime.now().strftime("%Y-%m-%d"),
                        })
            return pd.DataFrame(records)
        except Exception as e:
            print(f"⚠️ FMP获取价格失败: {e}")
            return pd.DataFrame()
    
    def get_historical_prices(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """获取历史价格"""
        # FMP历史价格需要Premium，这里用本地缓存
        return pd.DataFrame()
    
    def is_available(self) -> bool:
        """检查FMP是否可用"""
        return bool(self.api_key)


# ════════════════════════════════════════════════════════════════
# 数据管理器
# ════════════════════════════════════════════════════════════════

@dataclass
class DataFreshness:
    """数据新鲜度状态"""
    source: str
    last_update: Optional[datetime]
    age_hours: float
    is_fresh: bool
    record_count: int
    error: Optional[str] = None


class DataManager:
    """统一数据管理器"""
    
    def __init__(self):
        self.price_sources = self._init_price_sources()
        self.freshness_cache = {}
    
    def _init_price_sources(self) -> List[PriceDataSource]:
        """初始化价格数据源(按优先级)"""
        sources = []
        for source_name in DATA_CONFIG.price_sources:
            if source_name == "yfinance":
                sources.append(YFinancePriceSource())
            elif source_name == "fmp":
                sources.append(FMPPriceSource())
        return sources
    
    # ── 价格数据 ──
    
    def load_master_prices(self) -> pd.DataFrame:
        """加载主价格数据(features_v02.parquet)"""
        parquet_path = DATA_DIR / "features_v02.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"价格数据不存在: {parquet_path}")
        
        df = pd.read_parquet(parquet_path)
        df["date"] = df["date"].astype(str)
        return df
    
    def get_latest_trading_date(self) -> str:
        """获取最新交易日期"""
        df = self.load_master_prices()
        return df["date"].max()
    
    def get_price_pivot(self) -> pd.DataFrame:
        """获取价格矩阵(date x ticker)"""
        df = self.load_master_prices()
        return df.pivot_table(index="date", columns="ticker", values="close").sort_index()
    
    def update_prices(self) -> Tuple[bool, str]:
        """更新价格数据"""
        try:
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "falcon" / "update_price_data.py")],
                capture_output=True, text=True, timeout=180
            )
            if result.returncode == 0:
                return True, "价格数据更新成功"
            return False, f"更新失败: {result.stderr[:200]}"
        except Exception as e:
            return False, f"更新异常: {e}"
    
    # ── 基本面数据 ──
    
    def load_fundamentals(self) -> Dict[str, Any]:
        """加载所有基本面数据"""
        data = {}
        
        # FMP历史数据
        for name, fname in [
            ("fmp_ratios_historical", "fmp_ratios_historical.json"),
            ("analyst_historical", "analyst_historical.json"),
            ("fmp_key_metrics", "fmp_key_metrics.json"),
            ("fmp_financial_growth", "fmp_financial_growth.json"),
            ("fmp_insider", "fmp_insider.json"),
        ]:
            f = DATA_DIR / fname
            data[name] = json.load(open(f)) if f.exists() else {}
        
        # 三大报表
        for name, fname in [
            ("fmp_balance_sheet", "fmp_balance_sheet.json"),
            ("fmp_cashflow", "fmp_cashflow.json"),
            ("fmp_income_stmt", "fmp_income_stmt.json"),
        ]:
            f = DATA_DIR / fname
            data[name] = json.load(open(f)) if f.exists() else {}
        
        # FMP Premium
        premium_dir = PROJECT_ROOT / "data" / "fmp_premium"
        if premium_dir.exists():
            sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "falcon"))
            from extract_fmp_premium_features import load_fmp_premium_earnings, load_fmp_premium_grades
            data["earnings"] = load_fmp_premium_earnings(str(premium_dir))
            data["grades"] = load_fmp_premium_grades(str(premium_dir))
        else:
            data["earnings"] = {}
            data["grades"] = {}
        
        return data
    
    # ── VIX数据 ──
    
    def get_latest_vix(self) -> Tuple[Optional[float], Optional[str]]:
        """获取最新VIX值"""
        try:
            vix_path = PROJECT_ROOT / "data" / "us" / "vix_10y.parquet"
            if not vix_path.exists():
                return None, None
            
            vix_raw = pd.read_parquet(vix_path)
            # 统一列名为小写后查找，避免大小写不匹配
            col_map = {c.lower(): c for c in vix_raw.columns}
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_close = vix_raw[("Close", "^VIX")]
            elif "close" in col_map:
                vix_close = vix_raw[col_map["close"]]
            elif "Close" in col_map:
                vix_close = vix_raw[col_map["Close"]]
            else:
                vix_close = vix_raw.iloc[:, 0]
            
            latest_vix = float(vix_close.iloc[-1])
            # 从date列获取日期，而不是从index
            if "date" in col_map:
                vix_date = str(vix_raw[col_map["date"]].iloc[-1])[:10]
            elif "Date" in col_map:
                vix_date = str(vix_raw[col_map["Date"]].iloc[-1])[:10]
            else:
                vix_date = str(vix_raw.index[-1])[:10]
            return latest_vix, vix_date
        except Exception as e:
            print(f"⚠️ 获取VIX失败: {e}")
            return None, None
    
    # ── 新鲜度检查 ──
    
    def check_price_freshness(self) -> DataFreshness:
        """检查价格数据新鲜度"""
        parquet_path = DATA_DIR / "features_v02.parquet"
        if not parquet_path.exists():
            return DataFreshness(
                source="features_v02.parquet",
                last_update=None,
                age_hours=float("inf"),
                is_fresh=False,
                record_count=0,
                error="文件不存在"
            )
        
        # 检查文件修改时间
        mtime = datetime.fromtimestamp(parquet_path.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        
        # 检查数据中的最新日期
        df = pd.read_parquet(parquet_path)
        latest_date = df["date"].astype(str).max()
        record_count = len(df)
        
        # 计算数据年龄(基于交易日)
        try:
            latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
            data_age_days = (datetime.now() - latest_dt).days
            # 周末不算
            data_age_hours = data_age_days * 24
        except:
            data_age_hours = age_hours
        
        is_fresh = age_hours < DATA_CONFIG.price_max_age_hours
        
        return DataFreshness(
            source="features_v02.parquet",
            last_update=mtime,
            age_hours=age_hours,
            is_fresh=is_fresh,
            record_count=record_count,
        )
    
    def check_fundamental_freshness(self) -> Dict[str, DataFreshness]:
        """检查基本面数据新鲜度"""
        results = {}
        
        files_to_check = [
            ("fmp_ratios_historical", "fmp_ratios_historical.json"),
            ("analyst_historical", "analyst_historical.json"),
            ("fmp_financial_growth", "fmp_financial_growth.json"),
            ("fmp_balance_sheet", "fmp_balance_sheet.json"),
            ("fmp_cashflow", "fmp_cashflow.json"),
        ]
        
        for name, fname in files_to_check:
            fpath = DATA_DIR / fname
            if not fpath.exists():
                results[name] = DataFreshness(
                    source=fname,
                    last_update=None,
                    age_hours=float("inf"),
                    is_fresh=False,
                    record_count=0,
                    error="文件不存在"
                )
                continue
            
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
            age_hours = (datetime.now() - mtime).total_seconds() / 3600
            
            try:
                data = json.load(open(fpath))
                record_count = len(data) if isinstance(data, dict) else 0
            except:
                record_count = 0
            
            max_age_hours = DATA_CONFIG.fundamental_max_age_days * 24
            
            results[name] = DataFreshness(
                source=fname,
                last_update=mtime,
                age_hours=age_hours,
                is_fresh=age_hours < max_age_hours,
                record_count=record_count,
            )
        
        return results
    
    def get_all_freshness(self) -> Dict[str, DataFreshness]:
        """获取所有数据的新鲜度状态"""
        result = {}
        result["prices"] = self.check_price_freshness()
        result.update(self.check_fundamental_freshness())
        
        # VIX
        vix, vix_date = self.get_latest_vix()
        if vix is not None:
            try:
                vix_dt = datetime.strptime(vix_date, "%Y-%m-%d")
                vix_age = (datetime.now() - vix_dt).total_seconds() / 3600
            except:
                vix_age = float("inf")
            result["vix"] = DataFreshness(
                source="vix_10y.parquet",
                last_update=vix_dt if vix_date else None,
                age_hours=vix_age,
                is_fresh=vix_age < 48,  # VIX可以接受48小时延迟
                record_count=1,
            )
        
        return result
    
    def is_all_fresh(self) -> Tuple[bool, List[str]]:
        """检查所有数据是否新鲜"""
        freshness = self.get_all_freshness()
        issues = []
        
        for name, status in freshness.items():
            if not status.is_fresh:
                issues.append(f"{name}: {status.source} 已过期({status.age_hours:.1f}小时)")
        
        return len(issues) == 0, issues
    
    # ── 数据更新 ──
    
    def update_all(self) -> Tuple[bool, List[str]]:
        """更新所有数据"""
        results = []
        
        # 更新价格
        success, msg = self.update_prices()
        results.append(f"价格: {'✅' if success else '❌'} {msg}")
        
        # TODO: 更新基本面数据(需要FMP API)
        
        all_ok = all("✅" in r for r in results)
        return all_ok, results


# 全局数据管理器实例
data_manager = DataManager()
