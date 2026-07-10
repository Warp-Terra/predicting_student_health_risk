import numpy as np

from src.calibration import (
    apply_class_multipliers,
    apply_prior_gamma,
    tune_class_multipliers,
    tune_prior_gamma,
)


def test_calibrated_probabilities_remain_normalized():
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    priors = np.array([0.1, 0.2, 0.7])

    gamma_adjusted = apply_prior_gamma(probabilities, priors, -0.4)
    multiplier_adjusted = apply_class_multipliers(
        probabilities, [2.0, 1.5, 1.0]
    )

    assert np.allclose(gamma_adjusted.sum(axis=1), 1.0)
    assert np.allclose(multiplier_adjusted.sum(axis=1), 1.0)


def test_crossfit_tuning_returns_valid_balanced_accuracy():
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.1, 0.3],
            [0.1, 0.8, 0.1],
            [0.1, 0.6, 0.3],
            [0.1, 0.1, 0.8],
            [0.2, 0.2, 0.6],
        ]
    )
    labels = np.array([0, 0, 1, 1, 2, 2])
    fold_ids = np.array([1, 2, 1, 2, 1, 2])

    gamma = tune_prior_gamma(
        probabilities,
        labels,
        fold_ids,
        gamma_grid=np.array([-0.2, 0.0, 0.2]),
    )
    multipliers = tune_class_multipliers(
        probabilities,
        labels,
        fold_ids,
        log_grid=np.array([-0.2, 0.0, 0.2]),
    )

    assert 0 <= gamma["crossfit_balanced_accuracy"] <= 1
    assert 0 <= multipliers["crossfit_balanced_accuracy"] <= 1
