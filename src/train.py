import pickle
import time
import warnings

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    LGB_PARAMS,
    CATBOOST_PARAMS,
    SEED,
    N_FOLDS,
    MODEL_DIR,
    CATEGORICAL_FEATURES,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
    LOG_PERIOD,
    SMOKE_TEST,
)

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


def _get_sample_weight(y):
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight_map = dict(zip(classes, weights))
    return np.array([class_weight_map[label] for label in y])


def train_cv_lgb(X_train, y_train, cat_features, params=None):
    if params is None:
        params = dict(LGB_PARAMS)

    if SMOKE_TEST:
        n_folds = 2
        n_boost = 100
        early_stop = 10
    else:
        n_folds = N_FOLDS
        n_boost = NUM_BOOST_ROUND
        early_stop = EARLY_STOPPING_ROUNDS

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    oof_preds = np.zeros((len(X_train), 3))
    oof_labels = y_train.values
    scores = []
    elapsed_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        t0 = time.time()
        print(f"\n  Fold {fold + 1}/{n_folds} "
              f"| Train: {len(train_idx)} | Val: {len(val_idx)}", flush=True)

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        sample_weight = _get_sample_weight(y_tr)

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight, categorical_feature=cat_features)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, categorical_feature=cat_features)

        model = lgb.train(
            params, dtrain,
            num_boost_round=n_boost,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
                lgb.log_evaluation(period=LOG_PERIOD),
            ],
        )

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred

        val_pred_label = np.argmax(val_pred, axis=1)
        acc = accuracy_score(y_val, val_pred_label)
        scores.append(acc)
        elapsed = time.time() - t0
        elapsed_per_fold.append(elapsed)
        print(f"  Fold {fold + 1} | Acc: {acc:.5f} | {elapsed:.1f}s", flush=True)

        model_path = MODEL_DIR / f"lgb_fold{fold + 1}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    oof_acc = accuracy_score(oof_labels, np.argmax(oof_preds, axis=1))
    oof_ll = log_loss(oof_labels, oof_preds)
    print(f"\n  LGB CV: Acc {np.mean(scores):.5f} ± {np.std(scores):.5f} | OOF {oof_acc:.5f} | Avg {np.mean(elapsed_per_fold):.0f}s/fold", flush=True)

    return oof_preds


def train_cv_catboost(X_train, y_train, cat_features, params=None):
    if params is None:
        params = dict(CATBOOST_PARAMS)

    if SMOKE_TEST:
        n_folds = 2
        n_boost = 100
        early_stop = 10
    else:
        n_folds = N_FOLDS
        n_boost = NUM_BOOST_ROUND
        early_stop = EARLY_STOPPING_ROUNDS

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    oof_preds = np.zeros((len(X_train), 3))
    oof_labels = y_train.values
    scores = []
    elapsed_per_fold = []

    cb_categorical = [X_train.columns.get_loc(c) for c in cat_features]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        t0 = time.time()
        print(f"\n  Fold {fold + 1}/{n_folds} "
              f"| Train: {len(train_idx)} | Val: {len(val_idx)}", flush=True)

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        sample_weight = _get_sample_weight(y_tr)

        model = CatBoostClassifier(
            **params,
            iterations=n_boost,
            early_stopping_rounds=early_stop,
            cat_features=cb_categorical,
        )

        model.fit(
            X_tr, y_tr,
            sample_weight=sample_weight,
            eval_set=(X_val, y_val),
            verbose_eval=LOG_PERIOD,
        )

        val_pred = model.predict_proba(X_val)
        oof_preds[val_idx] = val_pred

        val_pred_label = np.argmax(val_pred, axis=1)
        acc = accuracy_score(y_val, val_pred_label)
        scores.append(acc)
        elapsed = time.time() - t0
        elapsed_per_fold.append(elapsed)
        print(f"  Fold {fold + 1} | Acc: {acc:.5f} | {elapsed:.1f}s", flush=True)

        model_path = MODEL_DIR / f"cb_fold{fold + 1}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    oof_acc = accuracy_score(oof_labels, np.argmax(oof_preds, axis=1))
    oof_ll = log_loss(oof_labels, oof_preds)
    print(f"\n  CB CV:  Acc {np.mean(scores):.5f} ± {np.std(scores):.5f} | OOF {oof_acc:.5f} | Avg {np.mean(elapsed_per_fold):.0f}s/fold", flush=True)

    return oof_preds


def train_full_lgb(X_train, y_train, cat_features, params=None):
    if params is None:
        params = dict(LGB_PARAMS)
    if SMOKE_TEST:
        n_boost = 100
    else:
        n_boost = NUM_BOOST_ROUND

    sample_weight = _get_sample_weight(y_train)
    dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weight, categorical_feature=cat_features)

    model = lgb.train(params, dtrain, num_boost_round=n_boost,
                      valid_sets=[dtrain], valid_names=["train"],
                      callbacks=[lgb.log_evaluation(period=LOG_PERIOD)])

    with open(MODEL_DIR / "lgb_full.pkl", "wb") as f:
        pickle.dump(model, f)
    return model


def train_full_catboost(X_train, y_train, cat_features, params=None):
    if params is None:
        params = dict(CATBOOST_PARAMS)
    if SMOKE_TEST:
        n_boost = 100
    else:
        n_boost = NUM_BOOST_ROUND

    sample_weight = _get_sample_weight(y_train)
    cb_categorical = [X_train.columns.get_loc(c) for c in cat_features]

    model = CatBoostClassifier(**params, iterations=n_boost, cat_features=cb_categorical)
    model.fit(X_train, y_train, sample_weight=sample_weight, verbose_eval=LOG_PERIOD)

    with open(MODEL_DIR / "cb_full.pkl", "wb") as f:
        pickle.dump(model, f)
    return model
