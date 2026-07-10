#!/usr/bin/env python3
import argparse
import json
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from src.calibration import (
    apply_class_multipliers,
    apply_prior_gamma,
    tune_class_multipliers,
    tune_prior_gamma,
)
from src.config import (
    EARLY_STOPPING_ROUNDS,
    INV_LABEL_MAP,
    LGB_PARAMS,
    MODEL_DIR,
    N_FOLDS,
    ROOT,
    SEED,
    SUBMISSION_DIR,
)
from src.data import prepare_data


def parse_args():
    parser = argparse.ArgumentParser(description="Run a reproducible LightGBM experiment")
    parser.add_argument("--name", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--weight-power", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-data-in-leaf", type=int, default=100)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument(
        "--early-stop-metric",
        choices=("logloss", "balanced_accuracy"),
        default="balanced_accuracy",
    )
    parser.add_argument("--reuse-model-prefix")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def make_sample_weight(labels, power):
    if power == 0:
        return None
    classes = np.unique(labels)
    balanced = compute_class_weight("balanced", classes=classes, y=labels)
    mapping = dict(zip(classes, np.power(balanced, power)))
    weights = np.array([mapping[label] for label in labels])
    return weights / weights.mean()


def prediction_distribution(labels):
    counts = np.bincount(labels, minlength=3) / len(labels)
    return {INV_LABEL_MAP[index]: float(value) for index, value in enumerate(counts)}


def balanced_accuracy_eval(probabilities, dataset):
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(3, -1).T
    predictions = np.argmax(probabilities, axis=1)
    score = balanced_accuracy_score(dataset.get_label(), predictions)
    return "balanced_accuracy", score, True


def save_submission(probabilities, test_ids, path):
    labels = np.argmax(probabilities, axis=1)
    frame = pd.DataFrame(
        {
            "id": test_ids,
            "health_condition": [INV_LABEL_MAP[label] for label in labels],
        }
    )
    frame.to_csv(path, index=False)
    return prediction_distribution(labels)


def run(args):
    started_at = time.time()
    artifact_dir = ROOT / "artifacts" / args.name
    artifact_dir.mkdir(parents=True, exist_ok=False)

    X_train, y_train, X_test, test_ids, categorical_features = prepare_data()
    y = y_train.to_numpy()
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X_train), 3), dtype=np.float32)
    test_probabilities = np.zeros((len(X_test), 3), dtype=np.float64)
    fold_ids = np.empty(len(X_train), dtype=np.int8)
    fold_metrics = []

    params = dict(LGB_PARAMS)
    params.update(
        {
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "min_data_in_leaf": args.min_data_in_leaf,
            "random_state": args.model_seed,
            "bagging_seed": args.model_seed,
            "feature_fraction_seed": args.model_seed,
            "data_random_seed": args.model_seed,
            "num_threads": args.threads,
        }
    )
    if args.early_stop_metric == "balanced_accuracy":
        params["metric"] = "None"

    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(X_train, y_train), start=1
    ):
        fold_started_at = time.time()
        fold_ids[valid_indices] = fold

        if args.reuse_model_prefix:
            model_path = MODEL_DIR / f"{args.reuse_model_prefix}_fold{fold}.pkl"
            with open(model_path, "rb") as handle:
                model = pickle.load(handle)
            best_iteration = args.iterations
        else:
            train_labels = y_train.iloc[train_indices]
            train_set = lgb.Dataset(
                X_train.iloc[train_indices],
                label=train_labels,
                weight=make_sample_weight(train_labels.to_numpy(), args.weight_power),
                categorical_feature=categorical_features,
            )
            valid_set = lgb.Dataset(
                X_train.iloc[valid_indices],
                label=y_train.iloc[valid_indices],
                reference=train_set,
                categorical_feature=categorical_features,
            )
            model = lgb.train(
                params,
                train_set,
                num_boost_round=args.iterations,
                valid_sets=[valid_set],
                valid_names=["valid"],
                feval=(
                    balanced_accuracy_eval
                    if args.early_stop_metric == "balanced_accuracy"
                    else None
                ),
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=EARLY_STOPPING_ROUNDS,
                        first_metric_only=True,
                        verbose=False,
                    ),
                    lgb.log_evaluation(period=100),
                ],
            )
            best_iteration = model.best_iteration or args.iterations
            with open(artifact_dir / f"lgb_fold{fold}.pkl", "wb") as handle:
                pickle.dump(model, handle)

        valid_probabilities = model.predict(
            X_train.iloc[valid_indices],
            num_iteration=best_iteration,
            num_threads=args.threads,
        )
        oof[valid_indices] = valid_probabilities
        test_probabilities += model.predict(
            X_test,
            num_iteration=best_iteration,
            num_threads=args.threads,
        ) / N_FOLDS

        fold_accuracy = accuracy_score(y[valid_indices], np.argmax(valid_probabilities, axis=1))
        fold_balanced_accuracy = balanced_accuracy_score(
            y[valid_indices], np.argmax(valid_probabilities, axis=1)
        )
        fold_metrics.append(
            {
                "fold": fold,
                "best_iteration": int(best_iteration),
                "accuracy": float(fold_accuracy),
                "balanced_accuracy": float(fold_balanced_accuracy),
                "elapsed_seconds": round(time.time() - fold_started_at, 2),
            }
        )
        print(
            f"Fold {fold}: iteration={best_iteration}, "
            f"balanced_accuracy={fold_balanced_accuracy:.6f}, "
            f"accuracy={fold_accuracy:.6f}, "
            f"elapsed={fold_metrics[-1]['elapsed_seconds']:.1f}s",
            flush=True,
        )

    gamma_calibration = tune_prior_gamma(oof, y, fold_ids)
    multiplier_calibration = tune_class_multipliers(oof, y, fold_ids)
    gamma_test = apply_prior_gamma(
        test_probabilities,
        gamma_calibration["class_priors"],
        gamma_calibration["full_gamma"],
    )
    multiplier_test = apply_class_multipliers(
        test_probabilities,
        multiplier_calibration["full_multipliers"],
    )

    raw_path = SUBMISSION_DIR / f"submission_{args.name}_raw.csv"
    gamma_path = SUBMISSION_DIR / (
        f"submission_{args.name}_g{gamma_calibration['full_gamma']:.2f}.csv"
    )
    multiplier_path = SUBMISSION_DIR / f"submission_{args.name}_multiplier.csv"
    raw_distribution = save_submission(test_probabilities, test_ids, raw_path)
    gamma_distribution = save_submission(gamma_test, test_ids, gamma_path)
    multiplier_distribution = save_submission(
        multiplier_test, test_ids, multiplier_path
    )

    np.save(artifact_dir / "oof.npy", oof)
    np.save(artifact_dir / "test.npy", test_probabilities.astype(np.float32))
    np.save(artifact_dir / "fold_ids.npy", fold_ids)

    metrics = {
        "name": args.name,
        "arguments": vars(args),
        "parameters": params,
        "fold_metrics": fold_metrics,
        "raw_oof_accuracy": float(accuracy_score(y, np.argmax(oof, axis=1))),
        "raw_oof_balanced_accuracy": float(
            balanced_accuracy_score(y, np.argmax(oof, axis=1))
        ),
        "raw_oof_log_loss": float(log_loss(y, oof)),
        "gamma_calibration": {
            "class_priors": gamma_calibration["class_priors"].tolist(),
            "fold_gammas": gamma_calibration["fold_gammas"],
            "full_gamma": gamma_calibration["full_gamma"],
            "fitted_balanced_accuracy": gamma_calibration[
                "fitted_balanced_accuracy"
            ],
            "crossfit_balanced_accuracy": gamma_calibration[
                "crossfit_balanced_accuracy"
            ],
        },
        "multiplier_calibration": {
            "fold_multipliers": multiplier_calibration[
                "fold_multipliers"
            ],
            "full_multipliers": multiplier_calibration[
                "full_multipliers"
            ].tolist(),
            "fitted_balanced_accuracy": multiplier_calibration[
                "fitted_balanced_accuracy"
            ],
            "crossfit_balanced_accuracy": multiplier_calibration[
                "crossfit_balanced_accuracy"
            ],
        },
        "raw_test_distribution": raw_distribution,
        "gamma_test_distribution": gamma_distribution,
        "multiplier_test_distribution": multiplier_distribution,
        "raw_submission": str(raw_path),
        "gamma_submission": str(gamma_path),
        "multiplier_submission": str(multiplier_path),
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=True)

    print(json.dumps(metrics, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    run(parse_args())
