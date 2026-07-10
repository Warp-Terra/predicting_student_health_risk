import pickle
import time

import numpy as np
import pandas as pd

from src.config import SUBMISSION_DIR, MODEL_DIR, ID_COL, TARGET, INV_LABEL_MAP, SMOKE_TEST
from src.ensemble import blend_oof, blend_ensemble


def predict_model(model_prefix, X_test):
    fold_files = sorted(MODEL_DIR.glob(f"{model_prefix}_fold*.pkl"))
    n_folds = len(fold_files)
    print(f"  Predicting {model_prefix}: {n_folds} folds × {len(X_test)} samples ...", flush=True)
    t0 = time.time()
    preds = np.zeros((len(X_test), 3))
    for model_path in fold_files:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        is_catboost = "CatBoost" in type(model).__name__
        if is_catboost:
            preds += model.predict_proba(X_test) / n_folds
        else:
            preds += model.predict(X_test.astype(float)) / n_folds
    elapsed = time.time() - t0
    print(f"    {model_prefix} done in {elapsed:.1f}s", flush=True)
    return preds


def build_submission(test_preds, test_ids, version, suffix=""):
    pred_labels = np.argmax(test_preds, axis=1)
    pred_labels_str = [INV_LABEL_MAP[label] for label in pred_labels]

    sub = pd.DataFrame({ID_COL: test_ids, TARGET: pred_labels_str})
    sub_path = SUBMISSION_DIR / f"submission_phase3_{version}{suffix}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\n  Submission saved to {sub_path}", flush=True)

    print(f"  Prediction distribution:", flush=True)
    dist = sub[TARGET].value_counts(normalize=True)
    for label, pct in dist.items():
        print(f"    {label}: {pct:.4f} ({100*pct:.1f}%)", flush=True)
    return sub


def ensemble_and_submit(X_test, test_ids, oof_preds_list, y_train, version, model_prefixes):
    suffix = "_smoke" if SMOKE_TEST else ""

    weights = blend_oof(oof_preds_list, y_train)

    test_preds_list = [predict_model(prefix, X_test) for prefix in model_prefixes]
    blended = blend_ensemble(test_preds_list, weights)
    build_submission(blended, test_ids, version, suffix)

    return blended
