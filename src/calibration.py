import numpy as np
from sklearn.metrics import balanced_accuracy_score


def apply_prior_gamma(probabilities, class_priors, gamma):
    """Undo some or all of the decision-boundary shift from class weighting."""
    adjusted = probabilities * np.power(class_priors, gamma)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def apply_class_multipliers(probabilities, multipliers):
    adjusted = probabilities * np.asarray(multipliers)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def _best_gamma(probabilities, labels, class_priors, gamma_grid):
    scores = []
    for gamma in gamma_grid:
        adjusted = apply_prior_gamma(probabilities, class_priors, gamma)
        scores.append(
            balanced_accuracy_score(labels, np.argmax(adjusted, axis=1))
        )

    best_score = max(scores)
    candidates = [
        gamma for gamma, score in zip(gamma_grid, scores) if score == best_score
    ]
    # Prefer the least adjustment when several grid points are equivalent.
    best_gamma = min(candidates, key=lambda value: (abs(value), value))
    return float(best_gamma), float(best_score)


def tune_prior_gamma(
    oof_probabilities,
    labels,
    fold_ids,
    gamma_grid=None,
):
    if gamma_grid is None:
        gamma_grid = np.round(np.arange(-0.5, 1.501, 0.02), 10)

    labels = np.asarray(labels)
    fold_ids = np.asarray(fold_ids)
    class_priors = np.bincount(labels) / len(labels)
    crossfit_predictions = np.empty_like(labels)
    fold_gammas = []

    for fold in np.unique(fold_ids):
        fit_mask = fold_ids != fold
        holdout_mask = ~fit_mask
        gamma, _ = _best_gamma(
            oof_probabilities[fit_mask],
            labels[fit_mask],
            class_priors,
            gamma_grid,
        )
        fold_gammas.append(gamma)
        adjusted = apply_prior_gamma(
            oof_probabilities[holdout_mask], class_priors, gamma
        )
        crossfit_predictions[holdout_mask] = np.argmax(adjusted, axis=1)

    full_gamma, fitted_accuracy = _best_gamma(
        oof_probabilities,
        labels,
        class_priors,
        gamma_grid,
    )

    return {
        "class_priors": class_priors,
        "fold_gammas": fold_gammas,
        "full_gamma": full_gamma,
        "fitted_balanced_accuracy": fitted_accuracy,
        "crossfit_balanced_accuracy": float(
            balanced_accuracy_score(labels, crossfit_predictions)
        ),
    }


def _best_class_multipliers(probabilities, labels, log_grid):
    log_multipliers = np.zeros(3)

    # Fit the two minority-class thresholds; the majority multiplier is the
    # reference value. Repeating coordinate updates handles their small overlap.
    for _ in range(3):
        for class_index in (0, 1):
            scores = []
            for value in log_grid:
                candidate = log_multipliers.copy()
                candidate[class_index] = value
                predictions = np.argmax(
                    probabilities * np.exp(candidate), axis=1
                )
                scores.append(balanced_accuracy_score(labels, predictions))

            best_score = max(scores)
            candidates = [
                value
                for value, score in zip(log_grid, scores)
                if score == best_score
            ]
            log_multipliers[class_index] = min(
                candidates, key=lambda value: (abs(value), value)
            )

    multipliers = np.exp(log_multipliers)
    predictions = np.argmax(probabilities * multipliers, axis=1)
    return multipliers, float(balanced_accuracy_score(labels, predictions))


def tune_class_multipliers(
    oof_probabilities,
    labels,
    fold_ids,
    log_grid=None,
):
    if log_grid is None:
        log_grid = np.round(np.arange(-0.5, 1.501, 0.02), 10)

    labels = np.asarray(labels)
    fold_ids = np.asarray(fold_ids)
    crossfit_predictions = np.empty_like(labels)
    fold_multipliers = []

    for fold in np.unique(fold_ids):
        fit_mask = fold_ids != fold
        holdout_mask = ~fit_mask
        multipliers, _ = _best_class_multipliers(
            oof_probabilities[fit_mask], labels[fit_mask], log_grid
        )
        fold_multipliers.append(multipliers.tolist())
        adjusted = apply_class_multipliers(
            oof_probabilities[holdout_mask], multipliers
        )
        crossfit_predictions[holdout_mask] = np.argmax(adjusted, axis=1)

    full_multipliers, fitted_balanced_accuracy = _best_class_multipliers(
        oof_probabilities, labels, log_grid
    )
    return {
        "fold_multipliers": fold_multipliers,
        "full_multipliers": full_multipliers,
        "fitted_balanced_accuracy": fitted_balanced_accuracy,
        "crossfit_balanced_accuracy": float(
            balanced_accuracy_score(labels, crossfit_predictions)
        ),
    }
