#!/usr/bin/env python3
import time

from src.data import prepare_data
from src.train import train_cv_lgb, train_full_lgb
from src.predict import predict_model, build_submission
from src.config import LGB_PARAMS, NUM_BOOST_ROUND


def main():
    t_start = time.time()

    print("=" * 60, flush=True)
    print("  Phase 3c: Phase 1 params + 2000 rounds", flush=True)
    print("  School Student Health Risk Prediction", flush=True)
    print("=" * 60, flush=True)

    print("\n[1/3] Loading & preparing data ...", flush=True)
    X_train, y_train, X_test, test_ids, cat_features = prepare_data()

    print(f"\n[2/3] Training (5-fold CV, {NUM_BOOST_ROUND} rounds) ...", flush=True)
    train_cv_lgb(X_train, y_train, cat_features, params=dict(LGB_PARAMS))
    train_full_lgb(X_train, y_train, cat_features, params=dict(LGB_PARAMS))

    print("\n[3/3] Generating submission ...", flush=True)
    test_preds = predict_model("lgb", X_test)
    build_submission(test_preds, test_ids, "p1r2k")

    print(f"\n{'='*60}", flush=True)
    print(f"  Total elapsed: {time.time() - t_start:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
