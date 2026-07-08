import numpy as np
import pandas as pd

from src.config import (
    TRAIN_PATH,
    TEST_PATH,
    TARGET,
    ID_COL,
    LABEL_MAP,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    SMOKE_TEST,
)


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def prepare_data():
    print("  Reading data ...", flush=True)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if SMOKE_TEST:
        print(f"  [SMOKE] Sampling to 10% ...", flush=True)
        train = train.sample(frac=0.1, random_state=42).reset_index(drop=True)

    print(f"  Encoding target ...", flush=True)
    train[TARGET] = train[TARGET].map(LABEL_MAP)

    features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    cat_to_int = {}
    for col in CATEGORICAL_FEATURES:
        train[col] = train[col].fillna("missing").astype("category")
        test[col] = test[col].fillna("missing").astype("category")
        categories = train[col].cat.categories
        train[col] = train[col].cat.codes
        test[col] = test[col].cat.codes
        cat_to_int[col] = dict(enumerate(categories))

    X_train = train[features].copy()
    y_train = train[TARGET].copy()
    X_test = test[features].copy()
    test_ids = test[ID_COL].copy()

    print(f"  Done. Train: {X_train.shape}, Test: {X_test.shape}", flush=True)
    return X_train, y_train, X_test, test_ids
