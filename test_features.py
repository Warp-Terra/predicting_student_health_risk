#!/usr/bin/env python3
"""Phase 2 incremental feature testing script."""
import sys
import importlib

TEST_ORDER = [
    ("Baseline",        {"missing_indicators": False, "cross_features": False, "target_encoding": False, "outlier_clip": False}),
    ("+ MissingInd",    {"missing_indicators": True,  "cross_features": False, "target_encoding": False, "outlier_clip": False}),
    ("+ CrossFeat",     {"missing_indicators": False, "cross_features": True,  "target_encoding": False, "outlier_clip": False}),
    ("+ TargetEnc",     {"missing_indicators": False, "cross_features": False, "target_encoding": True,  "outlier_clip": False}),
    ("+ OutlierClip",   {"missing_indicators": False, "cross_features": False, "target_encoding": False, "outlier_clip": True}),
    ("All Combined",    {"missing_indicators": True,  "cross_features": True,  "target_encoding": True,  "outlier_clip": True}),
]


def run_test(name, feature_sets):
    import src.config as cfg
    cfg.FEATURE_SETS = feature_sets

    # Force reimport of data and train to pick up new config
    importlib.reload(sys.modules.get("src.data", importlib.import_module("src.data")))
    importlib.reload(sys.modules.get("src.train", importlib.import_module("src.train")))

    from src.data import prepare_data
    from src.train import train_cv_lgb
    import glob, os

    # Clear old models
    for f in glob.glob(str(cfg.MODEL_DIR / "lgb_*.pkl")):
        os.remove(f)

    print(f"\n{'='*60}", flush=True)
    print(f"  {name}", flush=True)
    print(f"{'='*60}", flush=True)

    X, y, _, _, cat = prepare_data()
    train_cv_lgb(X, y, cat)


if __name__ == "__main__":
    # Force smoke mode
    if "--smoke" not in sys.argv:
        sys.argv.append("--smoke")

    for name, fs in TEST_ORDER:
        run_test(name, fs)
