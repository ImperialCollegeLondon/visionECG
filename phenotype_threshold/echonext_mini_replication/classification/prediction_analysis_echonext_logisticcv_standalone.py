#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TaskConfig:
    task_key: str
    display_name: str
    prediction_col: str
    reference_col: str
    positive_operator: str
    threshold: float
    method_dir: str = "logistic_cv"


EPSILON = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Logistic-CV calibration analysis for EchoNext-Mini predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Path to the training prediction CSV (used for CV).",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="Path to the external-test prediction CSV.",
    )
    parser.add_argument(
        "--reference_table",
        type=str,
        required=True,
        help="Path to the reference measurement CSV.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output root for task folders.",
    )
    parser.add_argument(
        "--task_key",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--display_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--prediction_col",
        type=str,
        required=True,
        help="Probability column in --train_csv / --test_csv.",
    )
    parser.add_argument(
        "--reference_col",
        type=str,
        required=True,
        help="Reference measurement column in --reference_table.",
    )
    parser.add_argument(
        "--positive_operator",
        type=str,
        required=True,
        choices=["lt", "gt"],
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Threshold value applied to --reference_col to derive the binary label.",
    )
    parser.add_argument(
        "--method_dir",
        type=str,
        default="logistic_cv",
        help="Method subfolder name under <output_root>/<task_key>/ (default: logistic_cv).",
    )
    parser.add_argument(
        "--prediction_id_col",
        type=str,
        default="eid",
        help="ID column in prediction CSVs (default: eid).",
    )
    parser.add_argument(
        "--reference_id_col",
        type=str,
        default="patient_id",
        help="ID column in the reference table (default: patient_id).",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for confidence intervals (default: 1000).",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of CV folds within the training CSV (default: 5).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def setup_logging(output_root: Path) -> logging.Logger:
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("EchoNextLogisticCVAnalysis")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    log_path = log_dir / f"logisticcv_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info("Log file created: %s", log_path)
    return logger


def validate_required_columns(df: pd.DataFrame, required_cols: Iterable[str], table_name: str) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {table_name}: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )


def load_reference_table(
    path: str, reference_id_col: str, reference_col: str, logger: logging.Logger,
) -> pd.DataFrame:
    logger.info("Loading reference table: %s", path)
    required_cols = [reference_id_col, reference_col]
    df = pd.read_csv(
        path,
        usecols=required_cols,
        dtype={reference_id_col: "string"},
        low_memory=False,
    )
    validate_required_columns(df, required_cols, "reference table")

    duplicate_mask = df.duplicated(subset=[reference_id_col], keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            f"Reference table contains {duplicate_count} duplicated rows for {reference_id_col}. "
            "This would create ambiguous joins."
        )

    logger.info(
        "Reference table loaded: %d rows, %d columns",
        df.shape[0],
        df.shape[1],
    )
    return df


def load_prediction_table(
    path: str, prediction_id_col: str, prediction_col: str, logger: logging.Logger,
) -> pd.DataFrame:
    logger.info("Loading prediction table: %s", path)
    required_cols = [prediction_id_col, prediction_col]
    df = pd.read_csv(
        path,
        usecols=required_cols,
        dtype={prediction_id_col: "string"},
        low_memory=False,
    )
    validate_required_columns(df, required_cols, "prediction table")

    duplicate_mask = df.duplicated(subset=[prediction_id_col], keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            f"Prediction table {path} contains {duplicate_count} duplicated rows for {prediction_id_col}."
        )

    logger.info(
        "Prediction table loaded: %d rows, %d columns",
        df.shape[0],
        df.shape[1],
    )
    return df


def create_binary_labels(values: pd.Series, task: TaskConfig) -> pd.Series:
    if task.positive_operator == "lt":
        return (values <= task.threshold).astype(int)
    if task.positive_operator == "gt":
        return (values >= task.threshold).astype(int)
    raise ValueError(f"Unsupported positive_operator: {task.positive_operator}")


def prepare_task_dataframe(
    predictions_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    task: TaskConfig,
    batch_label: str,
    prediction_id_col: str,
    reference_id_col: str,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    logger.info("Preparing merged dataframe for task=%s batch=%s", task.task_key, batch_label)

    total_prediction_rows = int(len(predictions_df))

    merged = predictions_df.merge(
        reference_df,
        left_on=prediction_id_col,
        right_on=reference_id_col,
        how="left",
        validate="one_to_one",
    )

    task_df = merged[[prediction_id_col, task.reference_col, task.prediction_col]].copy()
    task_df = task_df.rename(
        columns={
            prediction_id_col: "patient_id",
            task.reference_col: "reference_value",
            task.prediction_col: "raw_prediction_probability",
        }
    )

    task_df["reference_value"] = pd.to_numeric(task_df["reference_value"], errors="coerce")
    task_df["raw_prediction_probability"] = pd.to_numeric(task_df["raw_prediction_probability"], errors="coerce")

    matched_rows = int(task_df["reference_value"].notna().sum())
    rows_missing_reference = int(task_df["reference_value"].isna().sum())
    rows_missing_probability = int(task_df["raw_prediction_probability"].isna().sum())

    before_drop = len(task_df)
    task_df = task_df.dropna(subset=["reference_value", "raw_prediction_probability"]).copy()
    dropped_rows = before_drop - len(task_df)
    task_df["true_label"] = create_binary_labels(task_df["reference_value"], task)

    if task_df.empty:
        raise ValueError(f"No usable rows remain after filtering for task={task.task_key}, batch={batch_label}.")

    if not task_df["raw_prediction_probability"].between(0.0, 1.0).all():
        min_prob = task_df["raw_prediction_probability"].min()
        max_prob = task_df["raw_prediction_probability"].max()
        raise ValueError(
            f"Raw predicted probabilities are outside [0, 1] for task={task.task_key}, batch={batch_label}: "
            f"[{min_prob}, {max_prob}]"
        )

    unique_labels = sorted(task_df["true_label"].unique().tolist())
    if unique_labels != [0, 1]:
        raise ValueError(
            f"Task={task.task_key}, batch={batch_label} does not contain both classes after filtering. "
            f"Found labels: {unique_labels}"
        )

    join_stats = {
        "total_prediction_rows": total_prediction_rows,
        "matched_rows": matched_rows,
        "rows_missing_reference": rows_missing_reference,
        "rows_missing_probability": rows_missing_probability,
        "rows_dropped_after_filtering": dropped_rows,
    }
    logger.info("Join/filter stats for task=%s batch=%s: %s", task.task_key, batch_label, join_stats)
    return task_df.reset_index(drop=True), join_stats


def fit_logistic_calibrator(
    x_values: np.ndarray,
    y_true: np.ndarray,
    random_state: int,
) -> Tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_values.reshape(-1, 1))
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(x_scaled, y_true)
    return scaler, model


def predict_calibrated_probability(
    x_values: np.ndarray,
    scaler: StandardScaler,
    model: LogisticRegression,
) -> np.ndarray:
    x_scaled = scaler.transform(x_values.reshape(-1, 1))
    return model.predict_proba(x_scaled)[:, 1]


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_index = tpr - fpr
    optimal_idx = int(np.argmax(youden_index))
    return float(thresholds[optimal_idx])


def compute_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1 = (2.0 * ppv * tpr / (ppv + tpr)) if (ppv + tpr) > 0 else 0.0

    tp_adj = tp + EPSILON if tp == 0 else tp
    tn_adj = tn + EPSILON if tn == 0 else tn
    fp_adj = fp + EPSILON if fp == 0 else fp
    fn_adj = fn + EPSILON if fn == 0 else fn
    dor = (tp_adj * tn_adj) / (fp_adj * fn_adj)

    return {
        "threshold_value": threshold,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "accuracy": accuracy,
        "f1": f1,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "ppv": ppv,
        "npv": npv,
        "dor": dor,
    }


def _safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15), labels=[0, 1]))


def compute_point_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    metrics = {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "loss": _safe_log_loss(y_true, y_prob),
    }
    metrics.update(compute_threshold_metrics(y_true, y_prob, threshold))
    return metrics


def safe_percentile(values: List[float], percentile: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, percentile))


def bootstrap_refit_metrics(
    x_values: np.ndarray,
    y_true: np.ndarray,
    frozen_threshold: float,
    n_bootstrap: int,
    random_state: int,
) -> Dict[str, float]:
    rng = np.random.RandomState(random_state)
    bootstrap_store: Dict[str, List[float]] = {
        "auroc": [],
        "auprc": [],
        "loss": [],
        "accuracy": [],
        "f1": [],
        "tpr": [],
        "tnr": [],
        "fpr": [],
        "fnr": [],
        "ppv": [],
        "npv": [],
        "dor": [],
    }

    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_boot = y_true[indices]
        x_boot = x_values[indices]

        if len(np.unique(y_boot)) < 2:
            continue

        scaler_boot, model_boot = fit_logistic_calibrator(x_boot, y_boot, random_state=random_state)
        y_prob_boot = predict_calibrated_probability(x_boot, scaler_boot, model_boot)
        metrics = compute_point_metrics(y_boot, y_prob_boot, frozen_threshold)

        for key in bootstrap_store:
            bootstrap_store[key].append(metrics[key])

    ci_results: Dict[str, float] = {}
    for key, values in bootstrap_store.items():
        ci_results[f"{key}_ci_lower"] = safe_percentile(values, 2.5)
        ci_results[f"{key}_ci_upper"] = safe_percentile(values, 97.5)
        ci_results[f"{key}_bootstrap_n"] = float(len(values))
    return ci_results


def bootstrap_frozen_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    frozen_threshold: float,
    n_bootstrap: int,
    random_state: int,
) -> Dict[str, float]:
    rng = np.random.RandomState(random_state)
    bootstrap_store: Dict[str, List[float]] = {
        "auroc": [],
        "auprc": [],
        "loss": [],
        "accuracy": [],
        "f1": [],
        "tpr": [],
        "tnr": [],
        "fpr": [],
        "fnr": [],
        "ppv": [],
        "npv": [],
        "dor": [],
    }

    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        metrics = compute_point_metrics(y_true_boot, y_prob_boot, frozen_threshold)
        for key in bootstrap_store:
            bootstrap_store[key].append(metrics[key])

    ci_results: Dict[str, float] = {}
    for key, values in bootstrap_store.items():
        ci_results[f"{key}_ci_lower"] = safe_percentile(values, 2.5)
        ci_results[f"{key}_ci_upper"] = safe_percentile(values, 97.5)
        ci_results[f"{key}_bootstrap_n"] = float(len(values))
    return ci_results


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Dict[str, float],
    output_path: Path,
    title: str,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        fpr,
        tpr,
        color="tab:blue",
        linewidth=2,
        label=(
            f"AUROC={metrics['auroc']:.3f} "
            f"[{metrics['auroc_ci_lower']:.3f}, {metrics['auroc_ci_upper']:.3f}]"
        ),
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Dict[str, float],
    output_path: Path,
    title: str,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    prevalence = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        recall,
        precision,
        color="tab:orange",
        linewidth=2,
        label=(
            f"AUPRC={metrics['auprc']:.3f} "
            f"[{metrics['auprc_ci_lower']:.3f}, {metrics['auprc_ci_upper']:.3f}]"
        ),
    )
    ax.hlines(prevalence, xmin=0.0, xmax=1.0, linestyle="--", color="grey", linewidth=1, label="Prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix_png(
    metrics: Dict[str, float],
    output_path: Path,
    title: str,
) -> None:
    matrix = np.array(
        [
            [int(metrics["tn"]), int(metrics["fp"])],
            [int(metrics["fn"]), int(metrics["tp"])],
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)

    for (row_idx, col_idx), value in np.ndenumerate(matrix):
        ax.text(col_idx, row_idx, str(value), ha="center", va="center", color="black", fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix_csv(
    metrics: Dict[str, float],
    threshold_source: str,
    output_path: Path,
) -> None:
    report_df = pd.DataFrame(
        [
            {
                "threshold_policy": "optimal_threshold",
                "threshold_source": threshold_source,
                "threshold_value": metrics["threshold_value"],
                "true_positives": int(metrics["tp"]),
                "false_positives": int(metrics["fp"]),
                "true_negatives": int(metrics["tn"]),
                "false_negatives": int(metrics["fn"]),
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "tpr": metrics["tpr"],
                "tnr": metrics["tnr"],
                "fpr": metrics["fpr"],
                "fnr": metrics["fnr"],
                "ppv": metrics["ppv"],
                "npv": metrics["npv"],
                "dor": metrics["dor"],
            }
        ]
    )
    report_df.to_csv(output_path, index=False, float_format="%.6f")


def save_prediction_export(
    task_df: pd.DataFrame,
    calibrated_prob: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    export_df = task_df.copy()
    export_df["calibrated_probability"] = calibrated_prob
    export_df["predicted_label_at_frozen_optimal"] = (calibrated_prob >= threshold).astype(int)
    export_df.to_csv(output_path, index=False, float_format="%.6f")


def save_fold_split_predictions(
    task_df: pd.DataFrame,
    indices: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    fold_index: int,
    split: str,
    output_path: Path,
) -> None:
    """Save one fold's split predictions."""
    slice_df = task_df.iloc[indices].copy() if indices is not None else task_df.copy()
    slice_df = slice_df.rename(columns={"patient_id": "eid"})
    slice_df["predicted_probability"] = prob
    slice_df["predicted_label_at_threshold"] = (prob >= threshold).astype(int)
    slice_df["threshold_value"] = threshold
    slice_df["fold"] = fold_index
    slice_df["split"] = split

    keep_cols = [
        "eid",
        "reference_value",
        "true_label",
        "raw_prediction_probability",
        "predicted_probability",
        "predicted_label_at_threshold",
        "threshold_value",
        "fold",
        "split",
    ]
    slice_df[keep_cols].to_csv(output_path, index=False, float_format="%.6f")


def summarize_metric_lists(metric_lists: Dict[str, List[float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for key, values in metric_lists.items():
        clean = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
        if clean:
            summary[f"{key}_mean"] = float(np.mean(clean))
            summary[f"{key}_median"] = float(np.median(clean))
            summary[f"{key}_iqr"] = float(np.percentile(clean, 75) - np.percentile(clean, 25))
        else:
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_median"] = float("nan")
            summary[f"{key}_iqr"] = float("nan")
        summary[f"{key}_std"] = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
    return summary


@dataclass
class FoldData:
    """Store fold-specific data, predictions, and model."""
    fold_index: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    y_prob_train: np.ndarray
    y_prob_val: np.ndarray
    y_prob_external: np.ndarray
    train_threshold: float
    val_threshold: float
    scaler: StandardScaler
    model: LogisticRegression


def cross_validation_analysis(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_external: np.ndarray,
    y_external: np.ndarray,
    cv_folds: int,
    random_state: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], List[FoldData]]:
    """Run stratified K-fold CV on training data """
    class_counts = np.bincount(y_train)
    if class_counts.min() < cv_folds:
        raise ValueError(
            f"Cannot run {cv_folds}-fold stratified CV because the minority class has fewer than {cv_folds} samples."
        )

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    metric_names = ["auroc", "auprc", "loss", "accuracy", "f1", "tpr", "tnr", "fpr", "fnr", "ppv", "npv", "dor"]

    train_metric_lists: Dict[str, List[float]] = {k: [] for k in metric_names}
    val_metric_lists: Dict[str, List[float]] = {k: [] for k in metric_names}
    external_metric_lists: Dict[str, List[float]] = {k: [] for k in metric_names}
    fold_rows: List[Dict[str, float]] = []
    fold_data_list: List[FoldData] = []

    for fold_index, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train), start=1):
        x_tr = x_train[train_idx]
        y_tr = y_train[train_idx]
        x_va = x_train[val_idx]
        y_va = y_train[val_idx]

        scaler, model = fit_logistic_calibrator(x_tr, y_tr, random_state=random_state)
        y_prob_tr = predict_calibrated_probability(x_tr, scaler, model)
        y_prob_va = predict_calibrated_probability(x_va, scaler, model)
        y_prob_ex = predict_calibrated_probability(x_external, scaler, model)

        train_threshold = find_optimal_threshold(y_tr, y_prob_tr)
        val_threshold = find_optimal_threshold(y_va, y_prob_va)

        train_metrics = compute_point_metrics(y_tr, y_prob_tr, train_threshold)
        val_metrics = compute_point_metrics(y_va, y_prob_va, val_threshold)
        # Freeze external-test threshold from fold validation.
        external_metrics = compute_point_metrics(y_external, y_prob_ex, val_threshold)

        for key in metric_names:
            train_metric_lists[key].append(train_metrics[key])
            val_metric_lists[key].append(val_metrics[key])
            external_metric_lists[key].append(external_metrics[key])

        row: Dict[str, float] = {
            "fold": fold_index,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_external": int(len(y_external)),
            "n_train_positive": int(y_tr.sum()),
            "n_val_positive": int(y_va.sum()),
            "n_external_positive": int(y_external.sum()),
            "train_optimal_threshold": train_threshold,
            "val_optimal_threshold": val_threshold,
        }
        for key in metric_names:
            out_key = "auc" if key == "auroc" else key
            row[f"train_{out_key}"] = train_metrics[key]
            row[f"internal_val_{out_key}"] = val_metrics[key]
            row[f"external_val_{out_key}"] = external_metrics[key]
        fold_rows.append(row)

        fold_data_list.append(
            FoldData(
                fold_index=fold_index,
                train_idx=train_idx,
                val_idx=val_idx,
                y_prob_train=y_prob_tr,
                y_prob_val=y_prob_va,
                y_prob_external=y_prob_ex,
                train_threshold=train_threshold,
                val_threshold=val_threshold,
                scaler=scaler,
                model=model,
            )
        )

    def _rename_auroc(lists: Dict[str, List[float]]) -> Dict[str, List[float]]:
        return {("auc" if k == "auroc" else k): v for k, v in lists.items()}

    summary_rows = [
        {"split": "train", "cv_folds": cv_folds, **summarize_metric_lists(_rename_auroc(train_metric_lists))},
        {"split": "internal_val", "cv_folds": cv_folds, **summarize_metric_lists(_rename_auroc(val_metric_lists))},
        {"split": "external_val", "cv_folds": cv_folds, **summarize_metric_lists(_rename_auroc(external_metric_lists))},
    ]
    return fold_rows, summary_rows, fold_data_list


def build_metrics_row(
    dataset_label: str,
    dataset_role: str,
    task: TaskConfig,
    metrics: Dict[str, float],
    join_stats: Dict[str, int],
    task_df: pd.DataFrame,
    threshold_source: str,
) -> Dict[str, float]:
    positives = int(task_df["true_label"].sum())
    negatives = int(len(task_df) - positives)
    return {
        "dataset": dataset_label,
        "dataset_role": dataset_role,
        "task_key": task.task_key,
        "task_name": task.display_name,
        "reference_col": task.reference_col,
        "prediction_col": task.prediction_col,
        "threshold_policy": "optimal_threshold",
        "threshold_source": threshold_source,
        "threshold_value": metrics["threshold_value"],
        "n_samples": int(len(task_df)),
        "n_positive": positives,
        "n_negative": negatives,
        **join_stats,
        **metrics,
    }


def save_metrics_row(row: Dict[str, float], output_path: Path) -> None:
    pd.DataFrame([row]).to_csv(output_path, index=False, float_format="%.6f")


def save_calibrator_artifact(
    task: TaskConfig,
    scaler: StandardScaler,
    model: LogisticRegression,
    optimal_threshold: float,
    output_path: Path,
) -> None:
    artifact = {
        "task_key": task.task_key,
        "task_name": task.display_name,
        "prediction_col": task.prediction_col,
        "reference_col": task.reference_col,
        "training_batch": "train",
        "threshold_policy": "optimal_threshold",
        "optimal_threshold": optimal_threshold,
        "created_at": datetime.now().isoformat(),
        "scaler": scaler,
        "model": model,
    }
    joblib.dump(artifact, output_path)


def analyze_task(
    task: TaskConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    args: argparse.Namespace,
    task_output_dir: Path,
    logger: logging.Logger,
) -> None:
    development_df, dev_join_stats = prepare_task_dataframe(
        predictions_df=train_df,
        reference_df=reference_df,
        task=task,
        batch_label="train",
        prediction_id_col=args.prediction_id_col,
        reference_id_col=args.reference_id_col,
        logger=logger,
    )
    heldout_df, heldout_join_stats = prepare_task_dataframe(
        predictions_df=test_df,
        reference_df=reference_df,
        task=task,
        batch_label="test",
        prediction_id_col=args.prediction_id_col,
        reference_id_col=args.reference_id_col,
        logger=logger,
    )

    x_dev = development_df["raw_prediction_probability"].to_numpy(dtype=float)
    y_dev = development_df["true_label"].to_numpy(dtype=int)
    x_heldout = heldout_df["raw_prediction_probability"].to_numpy(dtype=float)
    y_heldout = heldout_df["true_label"].to_numpy(dtype=int)

    logger.info("Starting CV analysis for task=%s", task.task_key)
    fold_rows, summary_rows, fold_data_list = cross_validation_analysis(
        x_train=x_dev,
        y_train=y_dev,
        x_external=x_heldout,
        y_external=y_heldout,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )

    # Fit full calibrator on the whole training file
    logger.info("Fitting full calibrator on train for task=%s", task.task_key)
    scaler_full, model_full = fit_logistic_calibrator(x_dev, y_dev, random_state=args.random_state)
    y_prob_dev = predict_calibrated_probability(x_dev, scaler_full, model_full)
    frozen_threshold = find_optimal_threshold(y_dev, y_prob_dev)

    # Save cv_results_detailed / summary.
    pd.DataFrame(fold_rows).to_csv(
        task_output_dir / "cv_results_detailed.csv",
        index=False,
        float_format="%.6f",
    )
    pd.DataFrame(summary_rows).to_csv(
        task_output_dir / "cv_results_summary.csv",
        index=False,
        float_format="%.6f",
    )

    # Save per-fold train / internal_val / external_test predictions.
    for fold_data in fold_data_list:
        save_fold_split_predictions(
            task_df=development_df,
            indices=fold_data.train_idx,
            prob=fold_data.y_prob_train,
            threshold=fold_data.train_threshold,
            fold_index=fold_data.fold_index,
            split="train",
            output_path=task_output_dir / f"fold_{fold_data.fold_index}_train.csv",
        )
        save_fold_split_predictions(
            task_df=development_df,
            indices=fold_data.val_idx,
            prob=fold_data.y_prob_val,
            threshold=fold_data.val_threshold,
            fold_index=fold_data.fold_index,
            split="internal_val",
            output_path=task_output_dir / f"fold_{fold_data.fold_index}_internal_val.csv",
        )
        save_fold_split_predictions(
            task_df=heldout_df,
            indices=None,
            prob=fold_data.y_prob_external,
            threshold=fold_data.val_threshold,
            fold_index=fold_data.fold_index,
            split="external_test",
            output_path=task_output_dir / f"fold_{fold_data.fold_index}_external_test.csv",
        )
    logger.info("Saved per-fold train/internal_val/external_test predictions for task=%s", task.task_key)

    # -------- Full-fit reference block --------
    save_calibrator_artifact(
        task=task,
        scaler=scaler_full,
        model=model_full,
        optimal_threshold=frozen_threshold,
        output_path=task_output_dir / "logistic_calibrator_train.joblib",
    )

    dev_metrics = compute_point_metrics(y_dev, y_prob_dev, frozen_threshold)
    dev_metrics.update(
        bootstrap_refit_metrics(
            x_values=x_dev,
            y_true=y_dev,
            frozen_threshold=frozen_threshold,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
        )
    )
    dev_metrics_row = build_metrics_row(
        dataset_label="train",
        dataset_role="development_fullfit",
        task=task,
        metrics=dev_metrics,
        join_stats=dev_join_stats,
        task_df=development_df,
        threshold_source="train_fullfit_optimal",
    )
    save_prediction_export(
        task_df=development_df,
        calibrated_prob=y_prob_dev,
        threshold=frozen_threshold,
        output_path=task_output_dir / "predictions_train.csv",
    )
    save_confusion_matrix_csv(
        metrics=dev_metrics,
        threshold_source="train_fullfit_optimal",
        output_path=task_output_dir / "confusion_matrix_train.csv",
    )
    save_metrics_row(dev_metrics_row, task_output_dir / "metrics_train.csv")
    plot_roc_curve(
        y_true=y_dev,
        y_prob=y_prob_dev,
        metrics=dev_metrics,
        output_path=task_output_dir / "roc_curve_train.png",
        title=f"{task.display_name} | train | Logistic full fit",
    )
    plot_pr_curve(
        y_true=y_dev,
        y_prob=y_prob_dev,
        metrics=dev_metrics,
        output_path=task_output_dir / "pr_curve_train.png",
        title=f"{task.display_name} | train | Logistic full fit",
    )
    plot_confusion_matrix_png(
        metrics=dev_metrics,
        output_path=task_output_dir / "confusion_matrix_train.png",
        title=f"{task.display_name} | train | Optimal threshold",
    )

    y_prob_heldout = predict_calibrated_probability(x_heldout, scaler_full, model_full)
    heldout_metrics = compute_point_metrics(y_heldout, y_prob_heldout, frozen_threshold)
    heldout_metrics.update(
        bootstrap_frozen_metrics(
            y_true=y_heldout,
            y_prob=y_prob_heldout,
            frozen_threshold=frozen_threshold,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
        )
    )
    heldout_metrics_row = build_metrics_row(
        dataset_label="test",
        dataset_role="heldout_frozen",
        task=task,
        metrics=heldout_metrics,
        join_stats=heldout_join_stats,
        task_df=heldout_df,
        threshold_source="frozen_from_train_fullfit",
    )
    save_prediction_export(
        task_df=heldout_df,
        calibrated_prob=y_prob_heldout,
        threshold=frozen_threshold,
        output_path=task_output_dir / "predictions_test.csv",
    )
    save_confusion_matrix_csv(
        metrics=heldout_metrics,
        threshold_source="frozen_from_train_fullfit",
        output_path=task_output_dir / "confusion_matrix_test.csv",
    )
    save_metrics_row(heldout_metrics_row, task_output_dir / "metrics_test.csv")
    plot_roc_curve(
        y_true=y_heldout,
        y_prob=y_prob_heldout,
        metrics=heldout_metrics,
        output_path=task_output_dir / "roc_curve_test.png",
        title=f"{task.display_name} | test | Frozen calibrator",
    )
    plot_pr_curve(
        y_true=y_heldout,
        y_prob=y_prob_heldout,
        metrics=heldout_metrics,
        output_path=task_output_dir / "pr_curve_test.png",
        title=f"{task.display_name} | test | Frozen calibrator",
    )
    plot_confusion_matrix_png(
        metrics=heldout_metrics,
        output_path=task_output_dir / "confusion_matrix_test.png",
        title=f"{task.display_name} | test | Frozen threshold",
    )

    pd.DataFrame([dev_metrics_row, heldout_metrics_row]).to_csv(
        task_output_dir / "metrics_summary_all_batches.csv",
        index=False,
        float_format="%.6f",
    )

    logger.info(
        "Completed logistic-CV analysis for task=%s | full-fit train AUROC=%.4f | full-fit test AUROC=%.4f | "
        "frozen_threshold=%.4f",
        task.task_key,
        dev_metrics["auroc"],
        heldout_metrics["auroc"],
        frozen_threshold,
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    logger = setup_logging(output_root)

    logger.info("=" * 80)
    logger.info("ECHONEXT-MINI LOGISTIC-CV ANALYSIS STARTED")
    logger.info("=" * 80)
    logger.info("Arguments: %s", vars(args))

    task = TaskConfig(
        task_key=args.task_key,
        display_name=args.display_name,
        prediction_col=args.prediction_col,
        reference_col=args.reference_col,
        positive_operator=args.positive_operator,
        threshold=args.threshold,
        method_dir=args.method_dir,
    )

    reference_df = load_reference_table(
        args.reference_table, args.reference_id_col, task.reference_col, logger,
    )
    train_df = load_prediction_table(
        args.train_csv, args.prediction_id_col, task.prediction_col, logger,
    )
    test_df = load_prediction_table(
        args.test_csv, args.prediction_id_col, task.prediction_col, logger,
    )

    task_output_dir = output_root / task.task_key / task.method_dir
    task_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("-" * 80)
    logger.info("Running logistic-CV analysis for task: %s", task.display_name)
    logger.info("-" * 80)

    analyze_task(
        task=task,
        train_df=train_df,
        test_df=test_df,
        reference_df=reference_df,
        args=args,
        task_output_dir=task_output_dir,
        logger=logger,
    )

    logger.info("=" * 80)
    logger.info("ECHONEXT-MINI LOGISTIC-CV ANALYSIS COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
