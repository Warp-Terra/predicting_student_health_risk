# Predicting Student Health Risk

> [Kaggle Playground Series S6E7](https://www.kaggle.com/competitions/playground-series-s6e7) — balanced-accuracy LightGBM pipeline.

*[中文版本](README_CN.md)*

## Problem Statement

Predict students' health condition into one of three categories based on lifestyle and physiological data:

| Label | Description |
|-------|-------------|
| `fit` | Physically fit (5.8%) |
| `unhealthy` | Unhealthy (8.4%) |
| `at-risk` | At risk (85.9%) |

Evaluation metric: **balanced accuracy** (the mean recall across the three classes).

## Project Structure

```
predicting_student_health_risk/
├── main.py                   # Entry point: full pipeline (data → train → predict)
├── run_experiment.py         # Reproducible CV, calibration, and artifact runner
├── resubmit.py               # Quick re-submit (data → predict only, skip training)
├── src/
│   ├── config.py             # Paths, hyperparameters, feature lists, smoke test flag
│   ├── data.py               # Data loading & preprocessing (label encoding, categorical encoding)
│   ├── train.py              # 5-fold StratifiedKFold CV training + full-model training
│   ├── calibration.py        # Cross-fitted decision calibration
│   └── predict.py            # Ensemble prediction & submission generation
├── specs/                    # Design docs
│   ├── phase1-lightgbm-baseline.md
│   └── phase2-feature-engineering.md
├── data/                     # Raw datasets (ignored by git)
├── models/                   # Trained pickles (ignored by git)
└── submissions/              # Output CSVs (ignored by git)
```

## Features

| Type | Features |
|------|----------|
| Numerical (7) | `sleep_duration`, `heart_rate`, `bmi`, `calorie_expenditure`, `step_count`, `exercise_duration`, `water_intake` |
| Categorical (6) | `diet_type`, `stress_level`, `sleep_quality`, `physical_activity_level`, `smoking_alcohol`, `gender` |

- Missing values handled natively by LightGBM (no imputation needed)
- Categorical features encoded as integers for LightGBM's native categorical support

## Quick Start

### Prerequisites

- Python 3.10+
- Dependencies: `pandas`, `numpy`, `lightgbm`, `scikit-learn`

### Install

```bash
pip install pandas numpy lightgbm scikit-learn
```

### Run

**Best validated pipeline** (train + predict):
```bash
python main.py
```

**Named experiment**:
```bash
python run_experiment.py --name my_run --iterations 1500 \
  --weight-power 1 --model-seed 2026 \
  --early-stop-metric balanced_accuracy
```

**Smoke test** (10% data, reduced rounds — for rapid iteration):
```bash
python run_experiment.py --name smoke_run --smoke --iterations 100
```

**Re-submit** (skip training, re-use existing models):
```bash
python resubmit.py
```

### Output

- Trained models saved to `models/lgb_fold{1..5}.pkl` and `models/lgb_full.pkl`
- Models, OOF probabilities, test probabilities, and metrics are saved under `artifacts/<name>/`
- Raw and calibrated submissions are saved under `submissions/`

## Model

| Detail | Value |
|--------|-------|
| Algorithm | LightGBM (GBDT) |
| CV Strategy | 5-fold StratifiedKFold |
| Class Weights | Balanced |
| Early Stopping | 100 rounds on validation balanced accuracy |
| Max Rounds | 1,500 |
| Learning Rate | 0.05 |

### Why LightGBM

- **Native missing value handling** — automatically routes NAs to the optimal split branch
- **Native categorical support** — no one-hot encoding needed; uses optimal binning internally
- **Fast** — ~700k rows train in minutes on CPU
- **Class weight** — directly handles the heavy class imbalance

## Validated Results

| Metric | Value |
|--------|-------|
| Original public score | 0.94868 |
| Best OOF balanced accuracy | 0.94980 |
| Best public score | **0.95011** |

See [RESULTS.md](RESULTS.md) for the experiment history and rejected variants.

## Roadmap

- [x] Phase 1: LightGBM baseline
- [x] Phase 2: Feature engineering experiments (rejected by validation)
- [x] Phase 3: Optuna and longer training experiments (rejected)
- [x] Phase 4: Correct metric, cross-fitted calibration, and metric-aligned early stopping
- [ ] Phase 5: Add a genuinely diverse model only when cross-fit improves

## License

MIT
