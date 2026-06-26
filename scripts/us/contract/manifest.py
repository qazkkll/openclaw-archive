"""
Model Manifest System — 代码级契约
===================================
训练后自动生成manifest.json，评分/审计前校验一致性。
消除session间"我记得用的是XX特征"的问题。

用法:
  # 训练后生成
  python3 manifest.py create --model arrow_v12 --icir 0.386 --rows 948136

  # 评分前校验
  python3 manifest.py validate --model arrow_v12

  # 查看manifest
  python3 manifest.py show --model arrow_v12
"""
import json, hashlib, sys, os, time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # openclaw-archive
CONFIG_PATH = ROOT / "config" / "central_config.json"
MANIFEST_DIR = ROOT / "models" / "manifests"


def load_config():
    """读中央配置"""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def feature_hash(features: list) -> str:
    """特征列表SHA256，前12位"""
    return hashlib.sha256(",".join(sorted(features)).encode()).hexdigest()[:12]


def manifest_path(model_name: str) -> Path:
    return MANIFEST_DIR / f"{model_name}_manifest.json"


def create_manifest(model_name: str, icir=None, ic=None,
                    train_rows=None, train_stocks=None,
                    session_id=None, notes: str = ""):
    """训练后生成manifest — 训练脚本调用这个"""
    config = load_config()
    # Support model-specific feature sets via _model_feature_map
    model_feature_map = config["features"].get("_model_feature_map", {})
    feature_key = model_feature_map.get(model_name, "lambdamart_v10_v12")
    features = config["features"][feature_key]
    model_cfg = config["models"].get(model_name)
    if not model_cfg:
        raise ValueError(f"Unknown model: {model_name}. Valid: {list(config['models'].keys())}")

    model_file = ROOT / model_cfg["model_file"]
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    manifest = {
        "model_name": model_name,
        "model_file": model_cfg["model_file"],
        "model_file_sha256": _file_hash(model_file),
        "features": features,
        "features_count": len(features),
        "features_hash": feature_hash(features),
        "universe": model_cfg["universe"],
        "hold_days": model_cfg["hold_days"],
        "top_n": model_cfg["top_n"],
        "stop_loss": model_cfg["stop_loss"],
        "metrics": {
            "icir": icir,
            "ic": ic,
            "train_rows": train_rows,
            "train_stocks": train_stocks,
        },
        "trained_at": datetime.now().isoformat(),
        "trained_by_session": session_id or os.environ.get("HERMES_SESSION_ID", "unknown"),
        "central_config_version": config.get("_version", "unknown"),
        "notes": notes,
    }

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = manifest_path(model_name)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"✅ Manifest created: {out}")
    print(f"   Features: {len(features)}, hash={manifest['features_hash']}")
    print(f"   ICIR: {icir}, Rows: {train_rows}")
    return manifest


def validate_manifest(model_name: str, strict: bool = True) -> dict:
    """
    评分/审计前校验 — 返回 {valid: bool, errors: [], warnings: []}
    
    校验项:
    1. manifest文件存在
    2. 特征列表与central_config一致
    3. 特征hash匹配
    4. 模型文件存在且hash匹配
    5. central_config版本一致
    """
    errors = []
    warnings = []
    config = load_config()

    # 1. manifest存在
    mp = manifest_path(model_name)
    if not mp.exists():
        return {"valid": False, "errors": [f"Manifest not found: {mp}"], "warnings": [],
                "action": "Run: python3 manifest.py create --model " + model_name}

    with open(mp) as f:
        manifest = json.load(f)

    # 2. 特征列表一致 (model-specific via _model_feature_map)
    model_feature_map = config["features"].get("_model_feature_map", {})
    feature_key = model_feature_map.get(model_name, "lambdamart_v10_v12")
    config_features = config["features"][feature_key]
    manifest_features = manifest.get("features", [])
    if set(config_features) != set(manifest_features):
        missing = set(config_features) - set(manifest_features)
        extra = set(manifest_features) - set(config_features)
        if missing:
            errors.append(f"Features in config but missing in manifest: {missing}")
        if extra:
            errors.append(f"Features in manifest but not in config: {extra}")

    # 3. 特征hash
    expected_hash = feature_hash(config_features)
    actual_hash = manifest.get("features_hash", "")
    if expected_hash != actual_hash:
        errors.append(f"Feature hash mismatch: config={expected_hash} manifest={actual_hash}")

    # 4. 模型文件
    model_file = ROOT / manifest.get("model_file", "")
    if not model_file.exists():
        errors.append(f"Model file not found: {model_file}")
    else:
        current_hash = _file_hash(model_file)
        stored_hash = manifest.get("model_file_sha256", "")
        if current_hash != stored_hash:
            warnings.append(f"Model file changed since training: {stored_hash[:12]}→{current_hash[:12]}")

    # 5. config版本
    config_version = config.get("_version", "")
    manifest_config_version = manifest.get("central_config_version", "")
    if config_version != manifest_config_version:
        warnings.append(f"Config version mismatch: config={config_version} manifest={manifest_config_version}")

    # 6. 模型配置一致性
    model_cfg = config["models"].get(model_name, {})
    for key in ["hold_days", "top_n", "stop_loss"]:
        cfg_val = model_cfg.get(key)
        manifest_val = manifest.get(key)
        if cfg_val is not None and manifest_val is not None and cfg_val != manifest_val:
            warnings.append(f"{key}: config={cfg_val} manifest={manifest_val}")

    valid = len(errors) == 0
    result = {"valid": valid, "errors": errors, "warnings": warnings, "manifest": manifest}

    if valid:
        print(f"✅ {model_name} manifest VALID")
        if warnings:
            for w in warnings:
                print(f"   ⚠️ {w}")
    else:
        print(f"❌ {model_name} manifest INVALID")
        for e in errors:
            print(f"   🔴 {e}")
        for w in warnings:
            print(f"   ⚠️ {w}")

    return result


def show_manifest(model_name: str):
    """打印manifest详情"""
    mp = manifest_path(model_name)
    if not mp.exists():
        print(f"❌ No manifest for {model_name}")
        return
    with open(mp) as f:
        m = json.load(f)
    print(json.dumps(m, indent=2, ensure_ascii=False))


def _file_hash(path: Path) -> str:
    """文件SHA256，前16位"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def import_from_training_report(model_name: str, report_path: str):
    """从训练报告JSON导入manifest（兼容已有训练结果）"""
    with open(report_path) as f:
        report = json.load(f)

    # 提取指标
    summary = report.get("summary", report)
    icir = summary.get("avg_icir") or summary.get("icir")
    ic = summary.get("avg_ic") or summary.get("ic")
    rows = summary.get("total_rows") or summary.get("train_rows")
    stocks = summary.get("total_stocks") or summary.get("train_stocks")

    return create_manifest(
        model_name=model_name,
        icir=icir, ic=ic,
        train_rows=rows, train_stocks=stocks,
        notes=f"Imported from {report_path}"
    )


# === 评分脚本内嵌的校验函数 ===
def validate_before_scoring(model_name: str):
    """
    评分脚本在import后第一行调用。
    校验失败直接sys.exit(1)，不允许带着不一致的数据跑。
    """
    result = validate_manifest(model_name, strict=True)
    if not result["valid"]:
        print(f"\n🔴 FATAL: {model_name} manifest validation failed. Scoring aborted.")
        print("   Fix: retrain model or update manifest.")
        sys.exit(1)
    if result["warnings"]:
        print(f"⚠️ {model_name} has {len(result['warnings'])} warnings, proceeding with caution.")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model Manifest System")
    sub = parser.add_subparsers(dest="cmd")

    # create
    p_create = sub.add_parser("create", help="Create manifest after training")
    p_create.add_argument("--model", required=True)
    p_create.add_argument("--icir", type=float)
    p_create.add_argument("--ic", type=float)
    p_create.add_argument("--rows", type=int)
    p_create.add_argument("--stocks", type=int)
    p_create.add_argument("--session")
    p_create.add_argument("--notes", default="")

    # validate
    p_val = sub.add_parser("validate", help="Validate manifest before scoring")
    p_val.add_argument("--model", required=True)

    # show
    p_show = sub.add_parser("show", help="Show manifest")
    p_show.add_argument("--model", required=True)

    # import
    p_imp = sub.add_parser("import", help="Import from training report JSON")
    p_imp.add_argument("--model", required=True)
    p_imp.add_argument("--report", required=True)

    args = parser.parse_args()

    if args.cmd == "create":
        create_manifest(args.model, args.icir, args.ic, args.rows, args.stocks, args.session, args.notes)
    elif args.cmd == "validate":
        result = validate_manifest(args.model)
        sys.exit(0 if result["valid"] else 1)
    elif args.cmd == "show":
        show_manifest(args.model)
    elif args.cmd == "import":
        import_from_training_report(args.model, args.report)
    else:
        parser.print_help()
