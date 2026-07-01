#!/usr/bin/env python3
"""
Falcon 数据路径统一配置 (V0.4.1)
================================
集中管理所有数据路径，指向 FMP Premium 快照。

用法:
    from data_paths import FalconPaths
    paths = FalconPaths()
    print(paths.fmp_ratios)
"""
from pathlib import Path

# ═══════════════════════════════════════════════════
# 项目根目录 (相对于此文件)
# ═══════════════════════════════════════════════════
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent


class FalconPaths:
    """Falcon 所有数据路径的单一真相来源 (Single Source of Truth)。
    
    优先级:
      1. FMP Premium 快照 (data/fmp_premium/snapshots/) — 最新、最完整
      2. Falcon 本地数据 (data/falcon/) — 兼容旧脚本
    """
    
    def __init__(self, project_root: Path | None = None):
        self.PROJECT_ROOT = project_root or _PROJECT_ROOT
    
    # ─────────────────────────────────────────────
    # 核心目录
    # ─────────────────────────────────────────────
    @property
    def DATA_DIR(self) -> Path:
        """data/falcon/ — 特征文件、评分输出"""
        return self.PROJECT_ROOT / "data" / "falcon"
    
    @property
    def FMP_PREMIUM_DIR(self) -> Path:
        """data/fmp_premium/ — FMP Premium 数据根目录"""
        return self.PROJECT_ROOT / "data" / "fmp_premium"
    
    @property
    def SNAPSHOTS_DIR(self) -> Path:
        """data/fmp_premium/snapshots/ — FMP Premium 快照"""
        return self.FMP_PREMIUM_DIR / "snapshots"
    
    @property
    def US_DATA_DIR(self) -> Path:
        """data/us/ — 价格、VIX等"""
        return self.PROJECT_ROOT / "data" / "us"
    
    # ─────────────────────────────────────────────
    # FMP Premium 快照文件 (优先使用)
    # ─────────────────────────────────────────────
    @property
    def fmp_ratios(self) -> Path:
        """fmp_ratios_historical.json — 476 tickers, 到 2026-03-31"""
        return self.SNAPSHOTS_DIR / "fmp_ratios_historical.json"
    
    @property
    def fmp_key_metrics(self) -> Path:
        """fmp_key_metrics.json — 476 tickers, 到 2026-03-31"""
        return self.SNAPSHOTS_DIR / "fmp_key_metrics.json"
    
    @property
    def fmp_financial_growth(self) -> Path:
        """fmp_financial_growth.json — 476 tickers, 到 2026-03-31"""
        return self.SNAPSHOTS_DIR / "fmp_financial_growth.json"
    
    @property
    def analyst_historical(self) -> Path:
        """analyst_historical.json — 476 tickers, 到 2026-03-31 (99.8% 覆盖)"""
        return self.SNAPSHOTS_DIR / "analyst_historical.json"
    
    @property
    def fmp_balance_sheet(self) -> Path:
        """fmp_balance_sheet.json — 三大报表"""
        return self.SNAPSHOTS_DIR / "fmp_balance_sheet.json"
    
    @property
    def fmp_cashflow(self) -> Path:
        """fmp_cashflow.json — 现金流量表"""
        return self.SNAPSHOTS_DIR / "fmp_cashflow.json"
    
    @property
    def fmp_income_stmt(self) -> Path:
        """fmp_income_stmt.json — 收入报表"""
        return self.SNAPSHOTS_DIR / "fmp_income_stmt.json"
    
    @property
    def fmp_insider(self) -> Path:
        """fmp_insider.json — 内部人交易"""
        return self.SNAPSHOTS_DIR / "fmp_insider.json"
    
    @property
    def fmp_price_target(self) -> Path:
        """fmp_price_target.json — 分析师目标价"""
        return self.SNAPSHOTS_DIR / "fmp_price_target.json"
    
    @property
    def fmp_dcf(self) -> Path:
        """fmp_dcf.json — DCF估值"""
        return self.SNAPSHOTS_DIR / "fmp_dcf.json"
    
    # ─────────────────────────────────────────────
    # Russell 2000 数据 (快照)
    # ─────────────────────────────────────────────
    @property
    def russell_prices(self) -> Path:
        """russell_prices.json — Russell 2000 价格"""
        return self.SNAPSHOTS_DIR / "russell_prices.json"
    
    @property
    def russell_prices_updated(self) -> Path:
        """russell_prices_updated.json — 更新后的 Russell 价格"""
        return self.SNAPSHOTS_DIR / "russell_prices_updated.json"
    
    @property
    def fmp_ratios_russell(self) -> Path:
        """fmp_ratios_russell.json"""
        return self.SNAPSHOTS_DIR / "fmp_ratios_russell.json"
    
    @property
    def fmp_analyst_russell(self) -> Path:
        """fmp_analyst_russell.json"""
        return self.SNAPSHOTS_DIR / "fmp_analyst_russell.json"
    
    @property
    def fmp_metrics_russell(self) -> Path:
        """fmp_metrics_russell.json"""
        return self.SNAPSHOTS_DIR / "fmp_metrics_russell.json"
    
    @property
    def fmp_growth_russell(self) -> Path:
        """fmp_growth_russell.json"""
        return self.SNAPSHOTS_DIR / "fmp_growth_russell.json"
    
    # ─────────────────────────────────────────────
    # 特征文件 (输出)
    # �────────────────────────────────────────────
    @property
    def features_v02(self) -> Path:
        """features_v02.parquet — 原始特征 (K线+技术+基本面)"""
        return self.DATA_DIR / "features_v02.parquet"
    
    @property
    def features_v041(self) -> Path:
        """features_v04_1.parquet — V0.4.1 特征 (FMP Premium)"""
        return self.DATA_DIR / "features_v04_1.parquet"
    
    @property
    def v041_feature_audit(self) -> Path:
        """v041_feature_audit.json — 特征审计报告"""
        return self.DATA_DIR / "v041_feature_audit.json"
    
    # ─────────────────────────────────────────────
    # 其他数据文件
    # ─────────────────────────────────────────────
    @property
    def us_prices_daily(self) -> Path:
        """us_prices_daily.parquet — 美股日频价格"""
        return self.DATA_DIR / "us_prices_daily.parquet"
    
    @property
    def vix_10y(self) -> Path:
        """vix_10y.parquet — VIX 10年数据"""
        return self.US_DATA_DIR / "vix_10y.parquet"
    
    @property
    def sp500_price_targets(self) -> Path:
        """sp500_price_targets.json"""
        return self.SNAPSHOTS_DIR / "sp500_price_targets.json"
    
    # ─────────────────────────────────────────────
    # 兼容旧路径 (data/falcon/ 中的同名文件)
    # ─────────────────────────────────────────────
    def _fallback(self, snapshot_path: Path, local_name: str) -> Path:
        """如果快照不存在，回退到 data/falcon/ 中的同名文件。"""
        if snapshot_path.exists():
            return snapshot_path
        local = self.DATA_DIR / local_name
        if local.exists():
            return local
        return snapshot_path  # 返回原始路径（不存在时让调用方处理错误）
    
    @property
    def fmp_ratios_compat(self) -> Path:
        """兼容路径: 优先快照，回退 data/falcon/"""
        return self._fallback(self.fmp_ratios, "fmp_ratios_historical.json")
    
    @property
    def fmp_key_metrics_compat(self) -> Path:
        return self._fallback(self.fmp_key_metrics, "fmp_key_metrics.json")
    
    @property
    def fmp_financial_growth_compat(self) -> Path:
        return self._fallback(self.fmp_financial_growth, "fmp_financial_growth.json")
    
    @property
    def analyst_historical_compat(self) -> Path:
        return self._fallback(self.analyst_historical, "analyst_historical.json")
    
    @property
    def fmp_balance_sheet_compat(self) -> Path:
        return self._fallback(self.fmp_balance_sheet, "fmp_balance_sheet.json")
    
    @property
    def fmp_cashflow_compat(self) -> Path:
        return self._fallback(self.fmp_cashflow, "fmp_cashflow.json")
    
    @property
    def fmp_income_stmt_compat(self) -> Path:
        return self._fallback(self.fmp_income_stmt, "fmp_income_stmt.json")
    
    @property
    def fmp_insider_compat(self) -> Path:
        return self._fallback(self.fmp_insider, "fmp_insider.json")
    
    @property
    def fmp_financial_growth_r2k_compat(self) -> Path:
        return self._fallback(self.fmp_growth_russell, "fmp_growth_russell.json")
    
    def verify(self) -> dict:
        """验证所有关键路径是否存在。返回 {path: exists} 字典。"""
        results = {}
        for name in [
            'fmp_ratios', 'fmp_key_metrics', 'fmp_financial_growth',
            'analyst_historical', 'fmp_balance_sheet', 'fmp_cashflow',
            'fmp_income_stmt', 'fmp_insider', 'features_v02',
            'us_prices_daily', 'vix_10y',
        ]:
            p = getattr(self, name)
            results[name] = p.exists()
        return results


# ═══════════════════════════════════════════════════
# 全局单例 (方便直接 import)
# ═══════════════════════════════════════════════════
paths = FalconPaths()


if __name__ == "__main__":
    print("🔍 Falcon 数据路径验证:")
    print("=" * 60)
    for name, exists in paths.verify().items():
        status = "✅" if exists else "❌"
        p = getattr(paths, name)
        print(f"  {status} {name}: {p}")
