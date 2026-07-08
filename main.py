#!/usr/bin/env python3
import sys
import time

from src.data import prepare_data
from src.train import train_cv, train_full
from src.predict import predict_and_submit


def main():
    t_start = time.time()

    print("=" * 60, flush=True)
    print("  Phase 1: LightGBM Baseline", flush=True)
    print("  School Student Health Risk Prediction", flush=True)
    print("=" * 60, flush=True)

    print("\n[1/4] Loading & preparing data ...", flush=True)
    X_train, y_train, X_test, test_ids = prepare_data()

    print("\n[2/4] Cross-validation training ...", flush=True)
    train_cv(X_train, y_train)

    print("\n[3/4] Training full model on all data ...", flush=True)
    train_full(X_train, y_train)

    print("\n[4/4] Generating submission ...", flush=True)
    predict_and_submit(X_test, test_ids, version="v1", use_cv=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  Total elapsed: {time.time() - t_start:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
