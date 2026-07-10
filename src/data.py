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
    FEATURE_SETS,
    BMI_THRESHOLDS,
    BMI_LABELS,
    SLEEP_QUALITY_ENCODE,
    PHYSICAL_ACTIVITY_ENCODE,
    OUTLIER_LOW,
    OUTLIER_HIGH,
    TARGET_ENCODING_SMOOTH,
    SMOKE_TEST,
)


def _add_missing_indicators(train, test):
    print("  [FE] Adding missing indicators ...", flush=True)
    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    for col in all_features:
        train[f"{col}_is_missing"] = train[col].isnull().astype("int8")
        test[f"{col}_is_missing"] = test[col].isnull().astype("int8")
    return train, test


def _add_cross_features(train, test):
    print("  [FE] Adding cross features ...", flush=True)
    new_categorical = []

    train["bmi_group"] = pd.cut(
        train["bmi"], bins=BMI_THRESHOLDS, labels=BMI_LABELS, right=False
    )
    test["bmi_group"] = pd.cut(
        test["bmi"], bins=BMI_THRESHOLDS, labels=BMI_LABELS, right=False
    )
    new_categorical.append("bmi_group")

    def _safe_intensity(cal, dur):
        mask = dur.fillna(0) > 1
        result = np.where(mask, cal / dur, np.nan)
        return pd.Series(result, index=cal.index)

    train["exercise_intensity"] = _safe_intensity(
        train["calorie_expenditure"], train["exercise_duration"]
    )
    test["exercise_intensity"] = _safe_intensity(
        test["calorie_expenditure"], test["exercise_duration"]
    )

    sq_train = train["sleep_quality"].map(SLEEP_QUALITY_ENCODE).fillna(0)
    sq_test = test["sleep_quality"].map(SLEEP_QUALITY_ENCODE).fillna(0)
    train["sleep_score"] = train["sleep_duration"] * sq_train
    test["sleep_score"] = test["sleep_duration"] * sq_test

    pa_train = train["physical_activity_level"].map(PHYSICAL_ACTIVITY_ENCODE).fillna(0)
    pa_test = test["physical_activity_level"].map(PHYSICAL_ACTIVITY_ENCODE).fillna(0)
    train["health_behavior_score"] = train["step_count"] * pa_train + train["water_intake"]
    test["health_behavior_score"] = test["step_count"] * pa_test + test["water_intake"]

    train["heart_bmi_ratio"] = train["heart_rate"] / (train["bmi"] + 1e-5)
    test["heart_bmi_ratio"] = test["heart_rate"] / (test["bmi"] + 1e-5)

    train["diet_stress"] = train["diet_type"].fillna("missing") + "_" + train["stress_level"].fillna("missing")
    test["diet_stress"] = test["diet_type"].fillna("missing") + "_" + test["stress_level"].fillna("missing")
    new_categorical.append("diet_stress")

    train["sleep_activity"] = train["sleep_quality"].fillna("missing") + "_" + train["physical_activity_level"].fillna("missing")
    test["sleep_activity"] = test["sleep_quality"].fillna("missing") + "_" + test["physical_activity_level"].fillna("missing")
    new_categorical.append("sleep_activity")

    return train, test, new_categorical


def _encode_categorical(train, test, extra_categorical):
    all_cat = CATEGORICAL_FEATURES + extra_categorical
    for col in all_cat:
        train[col] = train[col].astype(str).replace("nan", "missing")
        test[col] = test[col].astype(str).replace("nan", "missing")
        train[col] = train[col].astype("category")
        test[col] = test[col].astype("category")
        train[col] = train[col].cat.codes
        test[col] = test[col].cat.codes
    return all_cat


def _add_target_encoding(train, test, y_train):
    print("  [FE] Adding target encoding ...", flush=True)
    global_mean = y_train.mean()
    for col in CATEGORICAL_FEATURES:
        agg = train.groupby(col)[TARGET].agg(["sum", "count"])
        smoothed = (agg["sum"] + global_mean * TARGET_ENCODING_SMOOTH) / (agg["count"] + TARGET_ENCODING_SMOOTH)
        mapping = smoothed.to_dict()
        train[f"{col}_te"] = train[col].map(mapping).fillna(global_mean)
        test[f"{col}_te"] = test[col].map(mapping).fillna(global_mean)
    return train, test


def _clip_outliers(train, test):
    print("  [FE] Clipping outliers ...", flush=True)
    for col in NUMERICAL_FEATURES:
        lower = train[col].quantile(OUTLIER_LOW)
        upper = train[col].quantile(OUTLIER_HIGH)
        train[col] = train[col].clip(lower, upper)
        test[col] = test[col].clip(lower, upper)
    return train, test


def prepare_data():
    print("  Reading data ...", flush=True)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if SMOKE_TEST:
        print("  [SMOKE] Sampling to 10% ...", flush=True)
        train = train.sample(frac=0.1, random_state=42).reset_index(drop=True)

    print("  Encoding target ...", flush=True)
    train[TARGET] = train[TARGET].map(LABEL_MAP)
    y_train = train[TARGET].copy()

    if FEATURE_SETS["missing_indicators"]:
        train, test = _add_missing_indicators(train, test)

    extra_cat = []
    if FEATURE_SETS["cross_features"]:
        train, test, extra_cat = _add_cross_features(train, test)

    cat_cols = _encode_categorical(train, test, extra_cat)

    if FEATURE_SETS["target_encoding"]:
        train, test = _add_target_encoding(train, test, y_train)

    if FEATURE_SETS["outlier_clip"]:
        train, test = _clip_outliers(train, test)

    drop_cols = [TARGET, ID_COL]
    X_train = train.drop(columns=[c for c in drop_cols if c in train.columns])
    X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])
    test_ids = test[ID_COL].copy()

    print(f"  Done. Train: {X_train.shape}, Test: {X_test.shape}, Cat cols: {len(cat_cols)}", flush=True)
    return X_train, y_train, X_test, test_ids, cat_cols
