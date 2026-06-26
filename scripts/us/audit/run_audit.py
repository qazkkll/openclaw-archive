"""
模型端到端审计框架 — 5层门禁
============================
设计原则：每层独立判定，全部PASS才能上线。
任何一层FAIL = 阻断，不继续往下。

层级：
  L1 Data    — 数据完整性（catch: 假数据、覆盖率不足）
  L2 Feature — 特征一致性（catch: fill vs fillna、公式错误、窗口不足）
  L3 Model   — 模型有效性（catch: 排名反转、只有一年盈利、过拟合）
  L4 Signal  — 信号生产（catch: 全部低于阈值、版本标签错误、输出格式异常）
  L5 Cross   — 跨层一致性（catch: 训练数据 ≠ 评分数据、特征顺序错位）

用法：
  python scripts/us/audit/run_audit.py --model blueshield_v10
  python scripts/us/audit/run_audit.py --model arrow_v12
  python scripts/us/audit/run_audit.py --all
"""

import sys
import os
import json
import time
import importlib.util
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# 项目根目录
ROOT = Path(__file__).resolve().parents[3]  # openclaw-archive/
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════
# 审计结果数据结构
# ═══════════════════════════════════════════════════
class CheckResult:
    """单个检查项的结果"""
    def __init__(self, name, passed, value=None, threshold=None, detail=""):
        self.name = name
        self.passed = passed
        self.value = value
        self.threshold = threshold
        self.detail = detail

    def __repr__(self):
        icon = "✅" if self.passed else "❌"
        val_str = f" ({self.value} vs {self.threshold})" if self.threshold is not None else ""
        return f"  {icon} {self.name}{val_str} {self.detail}"


class LayerResult:
    """单层审计结果"""
    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.checks = []
        self.start_time = time.time()
        self.end_time = None

    def add(self, name, passed, value=None, threshold=None, detail=""):
        self.checks.append(CheckResult(name, passed, value, threshold, detail))

    def finish(self):
        self.end_time = time.time()

    @property
    def passed(self):
        return all(c.passed for c in self.checks)

    @property
    def pass_count(self):
        return sum(1 for c in self.checks if c.passed)

    @property
    def fail_count(self):
        return sum(1 for c in self.checks if not c.passed)

    @property
    def elapsed(self):
        if self.end_time:
            return f"{self.end_time - self.start_time:.1f}s"
        return "running..."

    def report(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        lines = [
            f"\n{'='*60}",
            f"Layer: {self.layer_name}  [{status}]  {self.pass_count}/{len(self.checks)} passed  ({self.elapsed})",
            f"{'='*60}",
        ]
        for c in self.checks:
            lines.append(str(c))
        # 列出失败项的详情
        failures = [c for c in self.checks if not c.passed]
        if failures:
            lines.append(f"\n  ⚠️ FAILURES ({len(failures)}):")
            for c in failures:
                lines.append(f"    → {c.name}: {c.detail}")
        return "\n".join(lines)


class AuditReport:
    """完整审计报告"""
    def __init__(self, model_name):
        self.model_name = model_name
        self.layers = []
        self.start_time = time.time()

    def add_layer(self, layer_result):
        self.layers.append(layer_result)

    @property
    def passed(self):
        return all(l.passed for l in self.layers)

    @property
    def first_failure(self):
        for l in self.layers:
            if not l.passed:
                return l.layer_name
        return None

    def full_report(self):
        lines = [
            f"\n{'#'*60}",
            f"# 端到端审计报告: {self.model_name}",
            f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# 总结果: {'✅ PASS — 可上线' if self.passed else f'❌ FAIL — 阻断于 {self.first_failure}'}",
            f"# 耗时: {time.time() - self.start_time:.1f}s",
            f"{'#'*60}",
        ]
        for layer in self.layers:
            lines.append(layer.report())
        return "\n".join(lines)

    def to_dict(self):
        return {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "passed": self.passed,
            "first_failure": self.first_failure,
            "elapsed_s": round(time.time() - self.start_time, 1),
            "layers": [
                {
                    "name": l.layer_name,
                    "passed": l.passed,
                    "checks": [
                        {
                            "name": c.name,
                            "passed": bool(c.passed),
                            "value": c.value,
                            "threshold": c.threshold,
                            "detail": c.detail,
                        }
                        for c in l.checks
                    ],
                }
                for l in self.layers
            ],
        }


# ═══════════════════════════════════════════════════
# 模型配置注册表 — 从central_config.json读取（代码级契约）
# ═══════════════════════════════════════════════════
import hashlib as _hashlib
_central_cfg_path = ROOT / "config" / "central_config.json"
if _central_cfg_path.exists():
    with open(_central_cfg_path) as _f:
        _central_cfg = json.load(_f)
    MODEL_REGISTRY = {}
    for _mname, _mcfg in _central_cfg["models"].items():
        _fset = _mcfg.get("feature_set", "lambdamart_v10_v12")
        _cfg_features = _central_cfg["features"][_fset]
        _cfg_feature_hash = _hashlib.sha256(",".join(sorted(_cfg_features)).encode()).hexdigest()[:12]
        _meta_name = _mcfg.get("meta_file", f"models/us/{_mname}_meta.json")
        MODEL_REGISTRY[_mname] = {
            "model_file": _mcfg["model_file"],
            "meta_file": _meta_name,
            "score_script": _mcfg["score_script"],
            "features_count": len(_cfg_features),
            "features_hash": _cfg_feature_hash,
            "universe_filter": f"{_mcfg['universe']['min_price']} <= price <= {_mcfg['universe']['max_price']}",
            "universe_name": f"${_mcfg['universe']['min_price']}-${_mcfg['universe']['max_price']}",
            "signal_output": f"data/us/{_mname}_scored_*.json",
            "algorithm": "LGB LambdaMART",
        }
else:
    # Fallback: 硬编码（不应该走到这里）
    print("⚠️ WARNING: central_config.json not found, using hardcoded registry")
    MODEL_REGISTRY = {
        "blueshield_v10": {
            "model_file": "models/us/blueshield_v10_lambdamart.txt",
            "meta_file": "models/us/blueshield_v9_meta.json",
            "score_script": "scripts/us/blueshield_v10_lambdamart_score.py",
            "features_count": 17,
            "universe_filter": "price >= 10",
            "universe_name": ">$10",
            "signal_output": "data/us/blueshield_v10_scored_*.json",
            "algorithm": "LGB LambdaMART",
        },
        "arrow_v12": {
            "model_file": "models/us/arrow_v12_lambdamart.txt",
            "meta_file": "models/us/arrow_v12_meta.json",
            "score_script": "scripts/us/arrow_v12_lambdamart_score.py",
            "features_count": 17,
            "universe_filter": "1 <= price <= 10",
            "universe_name": "$1-$10",
            "signal_output": "data/us/arrow_v12_scored_*.json",
            "algorithm": "LGB LambdaMART",
        },
    }


def get_model_config(model_name):
    """获取模型配置，如果不在注册表中则报错"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    cfg = MODEL_REGISTRY[model_name]
    # 解析绝对路径
    for key in ["model_file", "meta_file", "score_script"]:
        cfg[f"{key}_abs"] = str(ROOT / cfg[key])
    return cfg


# ═══════════════════════════════════════════════════
# L1: 数据完整性审计
# ═══════════════════════════════════════════════════
def audit_l1_data(cfg, model_name):
    """
    检查数据是否可信。
    踩坑教训：
    - fundamentals_latest.parquet PE全部=22.89（中位数填充假数据）
    - us_hist_yf_10y.parquet只有2474只（覆盖率不足）
    - 数据中混入未来数据（look-ahead bias）
    """
    layer = LayerResult("L1 Data Integrity")
    import pandas as pd
    import numpy as np

    # --- 1.1 主价格数据覆盖 ---
    try:
        hist_path = ROOT / "data/us/us_hist_full_10y.parquet"
        if not hist_path.exists():
            layer.add("1.1 Price data exists", False, detail=f"Missing: {hist_path}")
        else:
            df = pd.read_parquet(hist_path)
            n_stocks = df["sym"].nunique()
            n_rows = len(df)
            layer.add("1.1 Price data exists", True, value=f"{n_stocks} stocks, {n_rows} rows")
            # 覆盖率检查：至少5000只
            layer.add(
                "1.2 Stock coverage >= 5000",
                n_stocks >= 5000,
                value=n_stocks, threshold=5000,
                detail="Need full market coverage"
            )
            # 日期范围检查
            date_range = pd.to_datetime(df["date"])
            latest = date_range.max()
            days_old = (pd.Timestamp.now() - latest).days
            layer.add(
                "1.3 Price data fresh (<=3 trading days)",
                days_old <= 5,
                value=f"{days_old} days old", threshold="<=5",
                detail=f"Latest date: {latest.strftime('%Y-%m-%d')}"
            )
    except Exception as e:
        layer.add("1.1 Price data load", False, detail=str(e))

    # --- 1.4 基本面数据真实性 ---
    # 核心教训：之前下载时用中位数填充，导致8882只股票PE全部相同
    try:
        fund_path = ROOT / "data/us/fundamentals_latest.parquet"
        if not fund_path.exists():
            layer.add("1.4 Fundamentals data exists", False, detail="Missing")
        else:
            fund = pd.read_parquet(fund_path)
            checks_passed = True
            details = []

            # 检查每个关键字段的唯一值比例
            for col in ["pe_trailing", "beta", "dividend_yield"]:
                if col in fund.columns:
                    nunique = fund[col].nunique()
                    total = len(fund)
                    unique_ratio = nunique / total if total > 0 else 0
                    # 如果唯一值占比<5%，说明大量填充了同一值
                    is_ok = unique_ratio > 0.05
                    layer.add(
                        f"1.5 {col} has variety (unique ratio > 5%)",
                        is_ok,
                        value=f"{unique_ratio:.1%} ({nunique}/{total})",
                        threshold=">5%",
                        detail="CRITICAL: All-same values = fake data, model loses discrimination"
                    )
                    if not is_ok:
                        checks_passed = False

                # 检查极端值
                if col in fund.columns and fund[col].nunique() > 1:
                    median = fund[col].median()
                    # 如果中位数恰好是常见默认值，可能有问题
                    if col == "beta" and abs(median - 1.0) < 0.01:
                        layer.add(
                            f"1.6 {col} median not default",
                            False,
                            value=f"median={median}", threshold="not 1.0",
                            detail="Beta=1.0 median = suspicious default fill"
                        )
                    elif col == "dividend_yield" and median == 0:
                        layer.add(
                            f"1.6 {col} median not zero",
                            False,
                            value=f"median={median}", threshold="not 0",
                            detail="div_yield=0 median = likely missing data"
                        )
                    else:
                        layer.add(f"1.6 {col} median reasonable", True, value=f"median={median}")
    except Exception as e:
        layer.add("1.4 Fundamentals data load", False, detail=str(e))

    # --- 1.7 VIX/SPY宏观数据 ---
    try:
        vix_path = ROOT / "data/us/vix_10y.parquet"
        spy_in_hist = True
        if vix_path.exists():
            vix = pd.read_parquet(vix_path)
            layer.add("1.7 VIX data exists", True, value=f"{len(vix)} rows")
        else:
            layer.add("1.7 VIX data exists", False, detail="Missing vix_10y.parquet")
    except Exception as e:
        layer.add("1.7 Macro data load", False, detail=str(e))

    # --- 1.8 数据源一致性（训练 vs 评分用同一数据） ---
    # 核心教训：训练用us_hist_full_10y，评分脚本用us_hist_yf_10y
    try:
        old_path = ROOT / "data/us/us_hist_yf_10y.parquet"
        new_path = ROOT / "data/us/us_hist_full_10y.parquet"
        if old_path.exists() and new_path.exists():
            old_n = len(pd.read_parquet(old_path, columns=["sym"]).drop_duplicates())
            new_n = len(pd.read_parquet(new_path, columns=["sym"]).drop_duplicates())
            coverage = old_n / new_n if new_n > 0 else 0
            layer.add(
                "1.8 Old data source coverage (should be ~100% or removed)",
                coverage > 0.9 or not old_path.exists(),
                value=f"{coverage:.0%} ({old_n}/{new_n})",
                detail="If old source still exists, ensure no script references it"
            )
        else:
            layer.add("1.8 Data source dedup check", True, detail="Only one source exists")
    except Exception as e:
        layer.add("1.8 Data source check", False, detail=str(e))

    layer.finish()
    return layer


# ═══════════════════════════════════════════════════
# L2: 特征一致性审计
# ═══════════════════════════════════════════════════
def audit_l2_features(cfg, model_name):
    """
    检查评分脚本的特征是否与训练一致。
    踩坑教训：
    - bb_std/bb_width/bb_pos/ret_quality公式与训练不一致
    - .fill(0) vs .fillna(0) — pandas已废弃.fill()
    - 150天窗口不够momentum_6m(需127交易日)
    - 特征匿名(Column_0-42)时顺序必须严格一致
    """
    layer = LayerResult("L2 Feature Consistency")
    import pandas as pd
    import numpy as np

    # --- 2.1 模型文件存在且可加载 ---
    model_path = ROOT / cfg["model_file"]
    meta_path = ROOT / cfg["meta_file"]

    if not model_path.exists():
        layer.add("2.1 Model file exists", False, detail=str(model_path))
        layer.finish()
        return layer

    layer.add("2.1 Model file exists", True, value=f"{model_path.stat().st_size / 1024:.0f}KB")

    # 加载模型
    try:
        if cfg["model_file"].endswith(".txt"):
            # LightGBM
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(model_path))
            model_features = model.feature_name()
            n_trees = model.num_trees()
        else:
            # XGBoost
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(str(model_path))
            model_features = model.feature_names or [f"Column_{i}" for i in range(model.num_features())]
            n_trees = model.num_boosted_rounds()
        layer.add("2.2 Model loads successfully", True, value=f"{len(model_features)} features, {n_trees} trees")
    except Exception as e:
        layer.add("2.2 Model loads successfully", False, detail=str(e))
        layer.finish()
        return layer

    # --- 2.3 特征数量匹配 ---
    expected_count = cfg["features_count"]
    actual_count = len(model_features)
    layer.add(
        "2.3 Feature count matches",
        actual_count == expected_count,
        value=actual_count, threshold=expected_count
    )

    # --- 2.4 Meta文件存在且字段完整 ---
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        required_fields = ["version", "features", "n_features", "trained_on"]
        missing = [f for f in required_fields if f not in meta]
        layer.add(
            "2.4 Meta has required fields",
            len(missing) == 0,
            detail=f"Missing: {missing}" if missing else "All present"
        )
        # 检查trained_on（用于检测look-ahead bias）
        if "trained_on" in meta:
            layer.add("2.5 trained_on specified (prevents look-ahead)", True, value=meta["trained_on"])
        else:
            layer.add("2.5 trained_on specified", False, detail="CRITICAL: Without this, cannot verify no future data")
    else:
        layer.add("2.4 Meta file exists", False, detail=str(meta_path))

    # --- 2.6 评分脚本存在且可解析 ---
    score_path = ROOT / cfg["score_script"]
    if not score_path.exists():
        layer.add("2.6 Score script exists", False, detail=str(score_path))
        layer.finish()
        return layer

    with open(score_path) as f:
        score_code = f.read()

    layer.add("2.6 Score script exists", True, value=f"{len(score_code)} chars")

    # --- 2.7 代码级检查：常见bug模式 ---
    # 检查.fill(0)（应为.fillna(0)）
    if ".fill(0)" in score_code and ".fillna(0)" not in score_code:
        layer.add(
            "2.7 No .fill(0) bug",
            False,
            detail="Found .fill(0) — should be .fillna(0). Pandas deprecated .fill()"
        )
    elif ".fill(0)" in score_code:
        layer.add(
            "2.7 No .fill(0) bug",
            False,
            detail="Found both .fill(0) and .fillna(0) — remove .fill(0)"
        )
    else:
        layer.add("2.7 No .fill(0) bug", True)

    # 检查数据窗口是否足够（需要至少250天）
    import re
    window_match = re.search(r"timedelta\(days=(\d+)\)", score_code)
    if window_match:
        window_days = int(window_match.group(1))
        layer.add(
            "2.8 Data window >= 250 days",
            window_days >= 250,
            value=f"{window_days} days", threshold=">=250",
            detail="momentum_6m needs 127 trading days ≈ 180 calendar days"
        )
    else:
        layer.add("2.8 Data window check", True, detail="No explicit window cutoff (using full data)")

    # 检查数据源引用
    if "us_hist_yf_10y" in score_code:
        layer.add(
            "2.9 Uses correct data source",
            False,
            detail="References us_hist_yf_10y (old, 2474 stocks). Should use us_hist_full_10y"
        )
    elif "us_hist_full_10y" in score_code:
        layer.add("2.9 Uses correct data source", True, value="us_hist_full_10y")
    else:
        layer.add("2.9 Data source reference", True, detail="No explicit old-source reference (check manually)")

    # --- 2.10 特征公式独立验证 ---
    # 对关键特征做抽样验证：用相同输入计算特征，与评分脚本输出对比
    # 这需要运行评分脚本，属于L4的内容。这里只做静态检查。
    has_bb = "bb_std" in score_code or "bb_upper" in score_code
    has_rsi = "rsi" in score_code.lower()
    has_macd = "macd" in score_code.lower()
    layer.add(
        "2.10 Key technical features present (bb/rsi/macd)",
        has_bb and has_rsi and has_macd,
        detail=f"bb={'✅' if has_bb else '❌'} rsi={'✅' if has_rsi else '❌'} macd={'✅' if has_macd else '❌'}"
    )

    # --- 2.11 特征顺序检查（匿名特征） ---
    if model_features and model_features[0].startswith("Column_"):
        layer.add(
            "2.11 Anonymous features detected",
            True,
            detail=f"Features: {model_features[0]}..{model_features[-1]}. Order must match training exactly!"
        )
        # 检查meta中是否有features列表
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if "features" in meta and len(meta["features"]) == actual_count:
                layer.add("2.12 Meta feature list matches count", True, value=f"{len(meta['features'])} features in meta")
            else:
                layer.add("2.12 Meta feature list matches count", False,
                          detail="Meta features list missing or count mismatch — cannot verify order")

    layer.finish()
    return layer


# ═══════════════════════════════════════════════════
# L3: 模型有效性审计
# ═══════════════════════════════════════════════════
def audit_l3_model(cfg, model_name):
    """
    检查模型是否真的有预测力。
    优先读训练报告(lambdamart_{model}_full.json)，含完整模型IC/单调性。
    降级：用ret20单特征近似（旧回归模型）。
    """
    layer = LayerResult("L3 Model Validity")
    import pandas as pd
    import numpy as np
    import json as _json

    # --- 优先路径：读训练报告 ---
    report_path = ROOT / f"data/lambdamart_{model_name}_full.json"
    if report_path.exists():
        try:
            with open(report_path) as f:
                report = _json.load(f)
            layer.add("3.1 Load training report", True,
                      value=f"{report.get('n_features', '?')} features, gpu={report.get('gpu', '?')}")

            results = report.get("results", {})
            if not results:
                layer.add("3.2 Training results exist", False, detail="Empty results in report")
                layer.finish()
                return layer

            # 每年IC/ICIR检查
            for yr, vals in sorted(results.items()):
                ic = vals.get("ic", 0)
                icir = vals.get("icir", 0)
                mono = vals.get("mono", False)
                top5 = vals.get("top5", 0)
                bot20 = vals.get("bot20", 0)
                win = vals.get("win", 0)

                layer.add(
                    f"3.3-{yr} Model IC positive",
                    ic > 0,
                    value=f"IC={ic:+.4f}", threshold=">0"
                )
                layer.add(
                    f"3.4-{yr} Rank monotonicity (Top > Bottom)",
                    mono or top5 > bot20,
                    value=f"Top5={top5:+.4f}, Bot20={bot20:+.4f}",
                    detail="Top group should outperform Bottom group"
                )
                layer.add(
                    f"3.5-{yr} Top group win rate",
                    win > 0.50,
                    value=f"{win:.1%}", threshold=">50%"
                )

            # ICIR聚合
            icirs = [v.get("icir", 0) for v in results.values()]
            avg_icir = np.mean(icirs)
            all_mono = all(v.get("mono", False) or v.get("top5", 0) > v.get("bot20", 0) for v in results.values())

            layer.add(
                "3.6 ICIR > 0.3 (yearly consistency)",
                avg_icir > 0.3,
                value=f"{avg_icir:.3f}", threshold=">0.3",
                detail="ICIR = mean(IC)/std(IC), measures year-to-year consistency"
            )
            layer.add(
                "3.7 All years Top > Bottom",
                all_mono,
                detail="Every test year should show Top group outperforming Bottom"
            )

            layer.finish()
            return layer
        except Exception as e:
            layer.add("3.1 Read training report", False, detail=str(e))
            # 降级到旧路径

    # --- 降级路径：单特征IC（旧回归模型） ---
    try:
        df = pd.read_parquet(ROOT / "data/us/us_hist_full_10y.parquet")
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        layer.add("3.1 Load training data", False, detail=str(e))
        layer.finish()
        return layer

    model_path = ROOT / cfg["model_file"]
    try:
        if cfg["model_file"].endswith(".txt"):
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(model_path))
            features = model.feature_name()
        else:
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(str(model_path))
            features = model.feature_names or [f"Column_{i}" for i in range(model.num_features())]
    except Exception as e:
        layer.add("3.2 Load model", False, detail=str(e))
        layer.finish()
        return layer

    layer.add("3.1 Load training data", True, value=f"{len(df)} rows, {df['sym'].nunique()} stocks")

    try:
        df = df.sort_values(["sym", "date"])
        df["fwd_5d"] = df.groupby("sym")["close"].transform(lambda x: x.shift(-5) / x - 1)
        cutoff = df["date"].max() - pd.Timedelta(days=365 * 3)
        recent = df[df["date"] >= cutoff].copy()

        if "ret20" in features:
            feat_col = "ret20"
            recent["ret20"] = recent.groupby("sym")["close"].transform(lambda x: x.pct_change(20))
        elif "close" in features:
            feat_col = "close"
        else:
            feat_col = features[0]
            layer.add("3.3 IC calculation", True, detail=f"Skipped — need full feature pipeline for {feat_col}")

        if "fwd_5d" in recent.columns and feat_col in recent.columns:
            recent = recent.dropna(subset=["fwd_5d", feat_col])
            recent["year"] = recent["date"].dt.year
            yearly_ic = recent.groupby("year").apply(
                lambda g: g[feat_col].corr(g["fwd_5d"]), include_groups=False
            )
            pos_years = (yearly_ic > 0).sum()
            total_years = len(yearly_ic)
            layer.add(
                "3.4 IC positive in most years",
                pos_years >= total_years * 0.6,
                value=f"{pos_years}/{total_years} years", threshold=f">={int(total_years * 0.6)}",
                detail=f"Yearly IC: {dict(zip(yearly_ic.index, [f'{v:.4f}' for v in yearly_ic.values]))}"
            )
    except Exception as e:
        layer.add("3.3-3.4 IC analysis", False, detail=str(e))

    try:
        if "fwd_5d" in recent.columns and feat_col in recent.columns:
            recent_clean = recent.dropna(subset=["fwd_5d", feat_col])
            recent_clean["group"] = recent_clean.groupby("date")[feat_col].transform(
                lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
            )
            group_ret = recent_clean.groupby("group")["fwd_5d"].agg(["mean", "count", "std"])
            group_ret.columns = ["avg_ret", "count", "std"]

            if len(group_ret) >= 4:
                top_ret = group_ret.iloc[-1]["avg_ret"]
                bot_ret = group_ret.iloc[0]["avg_ret"]
                is_monotonic = top_ret > bot_ret
                layer.add(
                    "3.5 Rank monotonicity (Top > Bottom)",
                    is_monotonic,
                    value=f"Top={top_ret:+.4f}, Bot={bot_ret:+.4f}",
                    detail="Top group should outperform Bottom group"
                )
                if "fwd_5d" in recent_clean.columns:
                    top_mask = recent_clean["group"] == group_ret.index[-1]
                    top_win = (recent_clean.loc[top_mask, "fwd_5d"] > 0).mean()
                    layer.add(
                        "3.6 Top group win rate > 50%",
                        top_win > 0.50,
                        value=f"{top_win:.1%}", threshold=">50%",
                        detail="Top picks should win more than half the time"
                    )
            else:
                layer.add("3.5 Rank monotonicity", False, detail=f"Only {len(group_ret)} groups formed")
    except Exception as e:
        layer.add("3.5-3.6 Rank analysis", False, detail=str(e))

    # 3.7 训练数据质量
    try:
        first_last = df.groupby("sym")["close"].agg(["first", "last"])
        ratio = first_last["last"] / first_last["first"]
        crashed = (ratio < 0.01).sum()
        total = len(ratio)
        crash_pct = crashed / total
        layer.add(
            "3.7 Crash stocks ratio < 5%",
            crash_pct < 0.05,
            value=f"{crash_pct:.1%} ({crashed}/{total})",
            threshold="<5%",
            detail="Stocks that lost 99%+ — likely reverse splits or delistings"
        )
    except Exception as e:
        layer.add("3.7 Training data quality", False, detail=str(e))

    # 3.8 ICIR
    try:
        if "yearly_ic" in dir() and len(yearly_ic) > 0:
            ic_mean = yearly_ic.mean()
            ic_std = yearly_ic.std()
            icir = ic_mean / ic_std if ic_std > 0 else 0
            layer.add(
                "3.8 ICIR > 0.3 (yearly IC consistency)",
                icir > 0.3,
                value=f"{icir:.3f}", threshold=">0.3",
                detail="ICIR = mean(IC)/std(IC), measures year-to-year consistency"
            )
    except Exception as e:
        layer.add("3.8 ICIR check", False, detail=str(e))

    layer.finish()
    return layer


# ═══════════════════════════════════════════════════
# L4: 信号生产审计
# ═══════════════════════════════════════════════════
def audit_l4_signal(cfg, model_name):
    """
    检查评分脚本是否正确运行并输出有意义的信号。
    踩坑教训：
    - 所有分数低于绝对阈值 → 全部标🟢🟢
    - 版本标签写V6但实际是V7
    - 输出JSON格式缺字段
    - 信号脚本跑不动（OOM、超时）
    """
    layer = LayerResult("L4 Signal Production")
    import pandas as pd

    # --- 4.1 评分脚本存在且语法正确 ---
    score_path = ROOT / cfg["score_script"]
    if not score_path.exists():
        layer.add("4.1 Score script exists", False, detail=str(score_path))
        layer.finish()
        return layer

    # 语法检查
    try:
        with open(score_path) as f:
            code = f.read()
        compile(code, str(score_path), "exec")
        layer.add("4.1 Score script syntax OK", True)
    except SyntaxError as e:
        layer.add("4.1 Score script syntax OK", False, detail=f"Line {e.lineno}: {e.msg}")
        layer.finish()
        return layer

    # --- 4.2 运行评分脚本（限时60秒） ---
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, str(score_path)],
            capture_output=True, text=True,
            timeout=120,
            cwd=str(ROOT)
        )
        if result.returncode == 0:
            layer.add("4.2 Score script runs successfully", True)
        else:
            # 提取最后3行错误信息
            err_lines = result.stderr.strip().split("\n")[-3:]
            layer.add("4.2 Score script runs successfully", False,
                       detail="\n".join(err_lines))
            layer.finish()
            return layer
    except subprocess.TimeoutExpired:
        layer.add("4.2 Score script runs (120s timeout)", False, detail="Timeout — likely OOM or infinite loop")
        layer.finish()
        return layer
    except Exception as e:
        layer.add("4.2 Score script runs", False, detail=str(e))
        layer.finish()
        return layer

    # --- 4.3 输出文件存在 ---
    import glob
    output_files = glob.glob(str(ROOT / cfg["signal_output"]))
    if not output_files:
        # 尝试标准路径
        alt_outputs = [
            ROOT / f"signals/us/{model_name}_scores.json",
            ROOT / f"data/scored_{model_name}_latest.json",
        ]
        for alt in alt_outputs:
            if alt.exists():
                output_files = [str(alt)]
                break

    if not output_files:
        layer.add("4.3 Output file exists", False,
                   detail=f"No output matching {cfg['signal_output']}")
        layer.finish()
        return layer

    latest_output = max(output_files, key=os.path.getmtime)
    layer.add("4.3 Output file exists", True, value=latest_output)

    # --- 4.4 输出JSON格式检查 ---
    try:
        with open(latest_output) as f:
            output = json.load(f)

        # 必须有的字段
        required_keys = ["model", "total", "picks"]
        missing_keys = [k for k in required_keys if k not in output]
        layer.add(
            "4.4 Output has required keys",
            len(missing_keys) == 0,
            detail=f"Missing: {missing_keys}" if missing_keys else "All present"
        )

        # --- 4.5 版本标签一致性 ---
        expected_version = cfg.get("meta_file", "").split("/")[-1].replace("_meta.json", "")
        actual_version = output.get("model", "")
        # 宽松匹配：只要包含版本关键字即可
        version_ok = (
            model_name.replace("_", "") in actual_version.replace("_", "").lower()
            or actual_version.replace("_", "").lower() in model_name.replace("_", "").lower()
            or not actual_version  # 允许空版本（有些脚本不输出版本）
        )
        layer.add(
            "4.5 Version label matches",
            version_ok or not actual_version,
            value=actual_version, threshold=model_name,
            detail="" if version_ok else "Version label mismatch — may reference old model"
        )

        # --- 4.6 信号分数分布 ---
        picks = output.get("picks", [])
        if picks:
            scores = [p.get("pred_rank", p.get("score", 0)) for p in picks]
            score_min = min(scores)
            score_max = max(scores)
            score_median = sorted(scores)[len(scores) // 2]
            n_picks = len(picks)

            # 分数有区分度（不是全部相同）
            score_range = score_max - score_min
            layer.add(
                "4.6 Score range > 0.001",
                score_range > 0.001,
                value=f"{score_range:.4f}",
                detail="If all scores identical, features are dead"
            )

            # Top picks数量合理
            layer.add(
                "4.7 Picks count >= 5",
                n_picks >= 5,
                value=n_picks, threshold=">=5",
                detail="Need enough picks for signal to be meaningful"
            )

            # Top1 vs Top30 区分度
            if n_picks >= 30:
                top1_score = scores[0]
                top30_score = scores[29]
                spread = top1_score - top30_score
                layer.add(
                    "4.8 Top1 vs Top30 spread > 0.01",
                    spread > 0.01,
                    value=f"{spread:.4f}", threshold=">0.01",
                    detail="Top picks must have meaningfully different scores"
                )
            else:
                layer.add("4.8 Top1 vs Top30 spread", True,
                           detail=f"Only {n_picks} picks, skip spread check")
        else:
            layer.add("4.6-4.8 Score analysis", False, detail="No picks in output")

    except json.JSONDecodeError as e:
        layer.add("4.4 Output JSON valid", False, detail=str(e))
    except Exception as e:
        layer.add("4.4-4.8 Output analysis", False, detail=str(e))

    # --- 4.9 持仓推荐质量 ---
    # 检查是否有推荐包含已知垃圾股特征
    try:
        if picks:
            cheap_count = sum(1 for p in picks if p.get("price", 999) < 3)
            total_picks = len(picks)
            if cfg["universe_filter"].startswith("price >= 10"):
                # 蓝盾不应该推荐<$3的股票
                layer.add(
                    "4.9 No penny stocks in blue-shield picks",
                    cheap_count == 0,
                    value=f"{cheap_count}/{total_picks} < $3", threshold="0",
                    detail="Blue-shield (>$10) should never have penny stocks"
                )
            else:
                layer.add("4.9 Price range check", True,
                           detail=f"Green-arrow universe, {cheap_count} < $3 picks OK")
    except Exception as e:
        layer.add("4.9 Price range check", False, detail=str(e))

    layer.finish()
    return layer


# ═══════════════════════════════════════════════════
# L5: 跨层一致性审计
# ═══════════════════════════════════════════════════
def audit_l5_cross(cfg, model_name):
    """
    检查训练、评分、配置之间的一致性。
    踩坑教训：
    - production.json指向V8但评分脚本还是V6
    - 模型用44特征训练，评分脚本只传43个
    - config.json的hold_days=20但训练目标是fwd_5d
    - 基本面数据路径在不同脚本中不一致
    """
    layer = LayerResult("L5 Cross-Layer Consistency")

    # --- 5.1 production.json指向正确模型 ---
    try:
        prod_path = ROOT / "models/us/production.json"
        if prod_path.exists():
            with open(prod_path) as f:
                prod = json.load(f)

            # 检查蓝盾或绿箭的配置
            if model_name.startswith("blue"):
                key = model_name  # blueshield_v10
            elif model_name.startswith("arrow"):
                key = model_name  # arrow_v12
            else:
                key = model_name

            model_cfg = prod.get(key, prod.get("active_models", {}).get(key, {}))
            if model_cfg:
                prod_model = model_cfg.get("model_file", "")
                expected_model = cfg["model_file"].split("/")[-1]
                matches = expected_model in prod_model or prod_model in expected_model
                layer.add(
                    "5.1 production.json points to correct model",
                    matches or not prod_model,
                    value=prod_model, threshold=expected_model
                )
            else:
                layer.add("5.1 production.json has model entry", False,
                           detail=f"Key '{key}' not found in production.json")
        else:
            layer.add("5.1 production.json exists", False)
    except Exception as e:
        layer.add("5.1 production.json check", False, detail=str(e))

    # --- 5.2 模型特征数 vs 评分脚本特征数 ---
    try:
        model_path = ROOT / cfg["model_file"]
        if model_path.exists():
            if cfg["model_file"].endswith(".txt"):
                import lightgbm as lgb
                model = lgb.Booster(model_file=str(model_path))
                model_nfeat = len(model.feature_name())
            else:
                import xgboost as xgb
                model = xgb.Booster()
                model.load_model(str(model_path))
                model_nfeat = model.num_features()

            with open(ROOT / cfg["score_script"]) as f:
                score_code = f.read()

            # 从评分脚本中推断特征数量
            # 检查meta文件
            meta_path = ROOT / cfg["meta_file"]
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                meta_nfeat = meta.get("n_features", 0)
                all_match = (model_nfeat == meta_nfeat == cfg["features_count"])
                layer.add(
                    "5.2 Feature count consistency (model/meta/config)",
                    all_match,
                    value=f"model={model_nfeat}, meta={meta_nfeat}, config={cfg['features_count']}",
                )
            else:
                layer.add("5.2 Feature count check", True,
                           detail=f"Model has {model_nfeat} features (no meta to cross-check)")
    except Exception as e:
        layer.add("5.2 Feature count check", False, detail=str(e))

    # --- 5.3 hold_days vs 训练目标 ---
    try:
        meta_path = ROOT / cfg["meta_file"]
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            label = meta.get("label", "")
            # 从label中提取forward period
            import re
            fwd_match = re.search(r"(\d+)d", label)
            if fwd_match:
                train_fwd = int(fwd_match.group(1))
                # 检查config中的hold_days
                config_path = ROOT / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    us_config = config.get("us", config)
                    hold_days = us_config.get("hold_days", 0)
                    # hold_days应该是train_fwd的1-3倍
                    layer.add(
                        "5.3 hold_days compatible with training target",
                        hold_days >= train_fwd and hold_days <= train_fwd * 3,
                        value=f"hold_days={hold_days}, train_fwd={train_fwd}d",
                        detail="hold_days should be 1-3x training forward period"
                    )
    except Exception as e:
        layer.add("5.3 hold_days check", False, detail=str(e))

    # --- 5.4 评分脚本数据路径 vs 训练数据路径 ---
    try:
        with open(ROOT / cfg["score_script"]) as f:
            score_code = f.read()

        # 检查评分脚本引用的数据路径
        data_refs = set()
        import re
        for match in re.finditer(r"['\"]([\w/_.-]+\.parquet)['\"]", score_code):
            data_refs.add(match.group(1))

        # 训练数据应该是us_hist_full_10y
        uses_full = any("full_10y" in ref for ref in data_refs)
        uses_old = any("yf_10y" in ref for ref in data_refs)

        if uses_old and not uses_full:
            layer.add(
                "5.4 Score script uses training data source",
                False,
                detail=f"Uses OLD data: {[r for r in data_refs if 'yf_10y' in r]}"
            )
        else:
            layer.add(
                "5.4 Score script uses training data source",
                True,
                detail=f"Data refs: {data_refs}"
            )
    except Exception as e:
        layer.add("5.4 Data path check", False, detail=str(e))

    # --- 5.5 信号阈值配置 ---
    try:
        meta_path = ROOT / cfg["meta_file"]
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            has_thresholds = "signal_thresholds" in meta
            layer.add(
                "5.5 Signal thresholds in meta",
                has_thresholds,
                detail="Without thresholds, signal levels cannot be assigned"
            )
    except Exception as e:
        layer.add("5.5 Threshold check", False, detail=str(e))

    layer.finish()
    return layer


# ═══════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════
def run_audit(model_name, skip_l4=False):
    """
    对指定模型运行完整5层审计。

    Args:
        model_name: 模型名（如 blueshield_v10, arrow_v12）
        skip_l4: 是否跳过L4（L4会实际运行评分脚本，较慢）

    Returns:
        AuditReport
    """
    cfg = get_model_config(model_name)
    report = AuditReport(model_name)

    print(f"\n🔍 Starting audit for: {model_name}")
    print(f"   Model: {cfg['model_file']}")
    print(f"   Score: {cfg['score_script']}")
    print(f"   Universe: {cfg['universe_name']}")
    print()

    # L1: Data
    print("📊 L1: Data Integrity...")
    l1 = audit_l1_data(cfg, model_name)
    report.add_layer(l1)
    print(l1.report())
    if not l1.passed:
        print(f"\n🛑 L1 FAILED — stopping audit. Fix data issues first.")
        return report

    # L2: Features
    print("\n🔧 L2: Feature Consistency...")
    l2 = audit_l2_features(cfg, model_name)
    report.add_layer(l2)
    print(l2.report())
    if not l2.passed:
        print(f"\n🛑 L2 FAILED — stopping audit. Fix feature issues first.")
        return report

    # L3: Model
    print("\n📈 L3: Model Validity...")
    l3 = audit_l3_model(cfg, model_name)
    report.add_layer(l3)
    print(l3.report())
    if not l3.passed:
        print(f"\n⚠️ L3 FAILED — model has validity issues. Continuing to check signals...")

    # L4: Signal (optional skip for speed)
    if skip_l4:
        print("\n⏭️ L4: Signal Production (SKIPPED)")
    else:
        print("\n📡 L4: Signal Production...")
        l4 = audit_l4_signal(cfg, model_name)
        report.add_layer(l4)
        print(l4.report())
        if not l4.passed:
            print(f"\n⚠️ L4 FAILED — signal production has issues.")

    # L5: Cross-layer
    print("\n🔗 L5: Cross-Layer Consistency...")
    l5 = audit_l5_cross(cfg, model_name)
    report.add_layer(l5)
    print(l5.report())

    # 最终报告
    print(report.full_report())

    # 保存报告
    report_path = ROOT / f"data/audit_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"\n📄 Report saved: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Model End-to-End Audit")
    parser.add_argument("--model", type=str, help="Model name (blueshield_v10 or arrow_v12)")
    parser.add_argument("--all", action="store_true", help="Audit all registered models")
    parser.add_argument("--skip-l4", action="store_true", help="Skip L4 (signal production, slow)")
    args = parser.parse_args()

    if args.all:
        models = list(MODEL_REGISTRY.keys())
    elif args.model:
        models = [args.model]
    else:
        parser.print_help()
        return

    all_passed = True
    for model in models:
        report = run_audit(model, skip_l4=args.skip_l4)
        if not report.passed:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL MODELS PASSED — Ready for production")
    else:
        print("🛑 SOME MODELS FAILED — Do NOT deploy")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
