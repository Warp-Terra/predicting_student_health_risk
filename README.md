# Predicting Student Health Risk

> [Kaggle Playground Series S6E7](https://www.kaggle.com/competitions/playground-series-s6e7) — 3-class classification baseline with LightGBM.

*[中文版本](README_CN.md)*

## Problem Statement

Predict students' health condition into one of three categories based on lifestyle and physiological data:

| Label | Description |
|-------|-------------|
| `fit` | Physically fit (5.8%) |
| `unhealthy` | Unhealthy (8.4%) |
| `at-risk` | At risk (85.9%) |

Evaluation metric: **accuracy**.

## Project Structure

```
predicting_student_health_risk/
├── main.py                   # Entry point: full pipeline (data → train → predict)
├── resubmit.py               # Quick re-submit (data → predict only, skip training)
├── src/
│   ├── config.py             # Paths, hyperparameters, feature lists, smoke test flag
│   ├── data.py               # Data loading & preprocessing (label encoding, categorical encoding)
│   ├── train.py              # 5-fold StratifiedKFold CV training + full-model training
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

**Full pipeline** (train + predict):
```bash
python main.py
```

**Smoke test** (10% data, reduced rounds — for rapid iteration):
```bash
python main.py --smoke
```

**Re-submit** (skip training, re-use existing models):
```bash
python resubmit.py
```

### Output

- Trained models saved to `models/lgb_fold{1..5}.pkl` and `models/lgb_full.pkl`
- Submission file saved to `submissions/submission_phase1_v1.csv`

## Model

| Detail | Value |
|--------|-------|
| Algorithm | LightGBM (GBDT) |
| CV Strategy | 5-fold StratifiedKFold |
| Class Weights | Balanced |
| Early Stopping | 50 rounds on val logloss |
| Max Rounds | 1,000 |
| Learning Rate | 0.05 |

### Why LightGBM

- **Native missing value handling** — automatically routes NAs to the optimal split branch
- **Native categorical support** — no one-hot encoding needed; uses optimal binning internally
- **Fast** — ~700k rows train in minutes on CPU
- **Class weight** — directly handles the heavy class imbalance

## CV Results (Phase 1 Baseline)

| Metric | Value |
|--------|-------|
| CV Accuracy (mean ± std) | ~0.9469 |
| OOF Accuracy | ~0.9469 |
| OOF LogLoss | ~0.1643 |
| Public LB Score | ~0.9487 |

## Roadmap

- [x] Phase 1: LightGBM baseline
  - [x] Data loading & preprocessing
  - [x] 5-fold StratifiedKFold CV
  - [x] Full-model training
  - [x] Kaggle submission
- [ ] Phase 2: Feature engineering
  - [ ] Missing value indicators
  - [ ] Cross / composite features
  - [ ] Target encoding
  - [ ] Outlier clipping
- [ ] Phase 3: Hyperparameter tuning (Optuna)
- [ ] Phase 4: Model ensembling (XGBoost, CatBoost, stacking)

## License

MIT
