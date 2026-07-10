import numpy as np
import optuna
import warnings

import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    LGB_PARAMS,
    SEED,
    N_FOLDS,
    OPTUNA_N_TRIALS,
    CATEGORICAL_FEATURES,
    EARLY_STOPPING_ROUNDS,
    SMOKE_TEST,
)

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _balanced_accuracy_eval(probabilities, dataset):
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(3, -1).T
    score = balanced_accuracy_score(
        dataset.get_label(), np.argmax(probabilities, axis=1)
    )
    return "balanced_accuracy", score, True


def _get_sample_weight(y):
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight_map = dict(zip(classes, weights))
    return np.array([class_weight_map[label] for label in y])


def objective(trial, X_train, y_train):
    params = {
        **{k: v for k, v in LGB_PARAMS.items() if k not in ("verbose",)},
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "random_state": SEED,
        "n_jobs": -1,
        "metric": "None",
    }

    n_boost = 1000 if not SMOKE_TEST else 100
    n_folds = N_FOLDS
    early_stop = EARLY_STOPPING_ROUNDS

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        sw = _get_sample_weight(y_tr)

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw, categorical_feature=CATEGORICAL_FEATURES)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, categorical_feature=CATEGORICAL_FEATURES)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_boost,
            valid_sets=[dval],
            valid_names=["val"],
            feval=_balanced_accuracy_eval,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
            ],
        )

        val_pred = model.predict(X_val)
        val_pred_label = np.argmax(val_pred, axis=1)
        acc = balanced_accuracy_score(y_val, val_pred_label)
        scores.append(acc)

    return np.mean(scores)


def tune_lightgbm(X_train, y_train):
    n_trials = OPTUNA_N_TRIALS if not SMOKE_TEST else 5

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def callback(study, trial):
        print(f"    Trial {trial.number}: acc={trial.value:.5f}", flush=True)

    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[callback],
    )

    print(f"\n  Best trial ({study.best_trial.number}): value={study.best_value:.5f}", flush=True)
    print(f"  Best params: {dict(study.best_params)}", flush=True)

    return study
