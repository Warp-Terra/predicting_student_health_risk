# Experiment Results

The official competition metric is **Balanced Accuracy**, not ordinary accuracy.
All scores below were read back through the Kaggle CLI.

| Experiment | Local selection | Public score | Decision |
|---|---:|---:|---|
| Phase 1 baseline, 1,000 rounds | accuracy 0.94694 | 0.94868 | Historical baseline |
| All engineered features | wrong metric | 0.94671 | Rejected |
| Optuna-tuned LightGBM | wrong metric | 0.92810 | Rejected |
| Baseline, 2,000 rounds | wrong metric | 0.93697 | Rejected |
| Accuracy prior correction, gamma 0.90 | accuracy 0.96682 | 0.86877 | Rejected: wrong metric |
| Balanced-accuracy correction, gamma -0.40 | 0.94937 cross-fit | 0.94955 | Improved |
| Per-class multipliers | 0.94942 cross-fit | 0.94921 | Rejected |
| Balanced-accuracy early stop, seed 2026 | **0.94980 OOF** | **0.95011** | Current best |
| Balanced-accuracy early stop, seed 314159 | 0.94981 OOF | 0.94986 | No improvement |

## Current Best Configuration

```text
features: 13 original features
model: LightGBM multiclass GBDT
class weighting: balanced (power 1.0)
folds: 5-fold StratifiedKFold, seed 42
model seed: 2026
learning rate: 0.05
num leaves: 63
min data in leaf: 100
maximum rounds: 1,500
early stopping: 100 rounds on validation balanced accuracy
fold best iterations: 168, 79, 7, 50, 129
```

The best submission is:

```text
submissions/submission_iter3_seed2026_balacc_es_raw.csv
```

## Main Finding

The earlier iterations optimized ordinary accuracy or log loss while Kaggle scores
balanced accuracy. This caused feature selection, Optuna, early stopping, and
ensembling to prefer the wrong models. Direct balanced-accuracy early stopping
improved the public score while using only the original features.
