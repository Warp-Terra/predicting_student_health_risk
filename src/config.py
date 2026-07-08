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

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50
LOG_PERIOD = 100

SEED = 42
N_FOLDS = 5

SMOKE_TEST = "--smoke" in sys.argv
if SMOKE_TEST:
    print("[SMOKE TEST MODE] Using reduced data & params")
