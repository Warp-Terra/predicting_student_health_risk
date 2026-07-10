import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

MODEL_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submissions"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "health_condition"
ID_COL = "id"
OFFICIAL_METRIC = "balanced_accuracy"

LABEL_MAP = {"fit": 0, "unhealthy": 1, "at-risk": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

NUMERICAL_FEATURES = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]

CATEGORICAL_FEATURES = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]

FEATURE_SETS = {
    "missing_indicators": False,
    "cross_features": False,
    "target_encoding": False,
    "outlier_clip": False,
}

BMI_THRESHOLDS = [0, 18.5, 24, 28, float("inf")]
BMI_LABELS = ["underweight", "normal", "overweight", "obese"]

SLEEP_QUALITY_ENCODE = {"poor": 0, "average": 1, "good": 2}
PHYSICAL_ACTIVITY_ENCODE = {"sedentary": 0, "moderate": 1, "active": 2}

OUTLIER_LOW = 0.005
OUTLIER_HIGH = 0.995

TARGET_ENCODING_SMOOTH = 100

LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "random_state": 42,
    "n_jobs": -1,
}

CATBOOST_PARAMS = {
    "objective": "MultiClass",
    "eval_metric": "MultiClass",
    "learning_rate": 0.05,
    "depth": 6,
    "min_data_in_leaf": 100,
    "random_seed": 42,
    "thread_count": -1,
}

XGBOOST_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

OPTUNA_N_TRIALS = 30
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100
LOG_PERIOD = 100

SEED = 42
N_FOLDS = 5

SMOKE_TEST = "--smoke" in sys.argv
if SMOKE_TEST:
    print("[SMOKE TEST MODE] Using reduced data & params")
