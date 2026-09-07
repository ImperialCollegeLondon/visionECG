#!/usr/bin/env python3
"""Configure ECG-parameter prognostic nested cross-validation."""

from typing import Dict, List

EVENT_ALIASES: Dict[str, str] = {
    "Heart_Failure": "HF",
    "MI": "MI",
    "MACE": "MACE",
}
EVENT_TYPES: List[str] = list(EVENT_ALIASES.keys())

OUTER_CV_FOLDS: int = 5
INNER_CV_FOLDS: int = 5
INNER_CV_REPEATS: int = 2

OPTUNA_STARTUP_TRIALS: int = 15
OPTUNA_MEDIAN_PRUNER_STARTUP: int = 10
OPTUNA_MEDIAN_PRUNER_WARMUP: int = 3
DEFAULT_OPTUNA_METRIC: str = "logloss"

XGB_OBJECTIVE: str = "binary:logistic"
XGB_EVAL_METRIC: str = "logloss"
XGB_TREE_METHOD: str = "hist"
XGB_DEVICE: str = "cpu"

N_ESTIMATORS_CEILING: int = 2000
N_ESTIMATORS_FLOOR: int = 50
EARLY_STOPPING_ROUNDS: int = 50
OUTER_REFIT_ES_HOLDOUT_FRAC: float = 0.20

LANDMARK_YEARS_DEFAULT: float = 5.0
DURATION_COL: str = "follow_up_days"
EVENT_COL: str = "event_status"
LABEL_COL: str = "y_landmark"
JOIN_COL: str = "patient_id"

SPARSE_EVENT_WARNING_THRESHOLD: int = 20
MIN_TRAIN_CASES_TO_FIT: int = 5
RANDOM_SEED_DEFAULT: int = 123

FEATURE_MODE_DIR: str = "ecg_parameters"
CONFIG_NAME: str = "ecg_parameters_prognostic_index"
FOLD_AUC_FILENAME: str = "fold_auc_summary.csv"
