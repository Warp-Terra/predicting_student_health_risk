import pickle

import numpy as np
import pandas as pd

from src.config import SUBMISSION_DIR, MODEL_DIR, ID_COL, TARGET, INV_LABEL_MAP, SMOKE_TEST


def predict_and_submit(X_test, test_ids, version="v1", use_cv=True):

    if use_cv:
        preds = np.zeros((len(X_test), 3))
        for fold in range(1, 6):
            model_path = MODEL_DIR / f"lgb_fold{fold}.pkl"
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            preds += model.predict(X_test) / 5
    else:
        model_path = MODEL_DIR / "lgb_full.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        preds = model.predict(X_test)

    pred_labels = np.argmax(preds, axis=1)
    pred_labels_str = [INV_LABEL_MAP[label] for label in pred_labels]

    sub = pd.DataFrame({ID_COL: test_ids, TARGET: pred_labels_str})

    suffix = "_smoke" if SMOKE_TEST else ""
    sub_path = SUBMISSION_DIR / f"submission_phase1_{version}{suffix}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission saved to {sub_path}", flush=True)

    print(f"\n  Prediction distribution:", flush=True)
    dist = sub[TARGET].value_counts(normalize=True)
    for label, pct in dist.items():
        print(f"    {label}: {pct:.4f} ({100*pct:.1f}%)", flush=True)
