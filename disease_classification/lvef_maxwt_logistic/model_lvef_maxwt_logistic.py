"""Leakage-safe LVEF and max-WT logistic cross-validation."""

from __future__ import annotations

import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger(__name__)
METRIC_NAMES: Tuple[str, ...] = (
    "loss",
    "auc",
    "auprc",
    "accuracy",
    "f1",
    "tpr",
    "tnr",
    "fpr",
    "fnr",
)
SPLIT_NAMES: Tuple[str, ...] = ("train", "internal_val", "external_test")


def _as_binary_labels(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty.")

    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only binary numeric values 0/1.") from exc
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} contains missing or non-finite values.")
    invalid = ~np.isin(numeric, (0.0, 1.0))
    if invalid.any():
        examples = np.unique(numeric[invalid])[:5].tolist()
        raise ValueError(f"{name} must contain only 0/1; invalid values include {examples}.")
    return numeric.astype(np.int64, copy=False)


def _as_feature_matrix(values: Any, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric two-dimensional feature matrix.") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got shape {matrix.shape}.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one feature.")
    if np.isinf(matrix).any():
        raise ValueError(f"{name} contains infinite values; only finite values and NaN are allowed.")
    return matrix


def _as_eids(values: Any, expected_length: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if len(array) != expected_length:
        raise ValueError(
            f"{name} length ({len(array)}) does not match its feature rows "
            f"({expected_length})."
        )
    series = pd.Series(array, copy=False)
    if series.isna().any():
        raise ValueError(f"{name} contains missing identifiers.")
    duplicate_mask = series.duplicated(keep=False)
    if duplicate_mask.any():
        examples = series.loc[duplicate_mask].head(5).tolist()
        raise ValueError(f"{name} contains duplicate identifiers, including {examples}.")
    return array


def _validate_inputs(
    X_train_raw: Any,
    y_train: Any,
    eids_train: Any,
    X_test_raw: Any,
    y_test: Any,
    eids_test: Any,
    feature_names: Sequence[str],
    n_folds: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    X_train = _as_feature_matrix(X_train_raw, "X_train_raw")
    X_test = _as_feature_matrix(X_test_raw, "X_test_raw")
    labels_train = _as_binary_labels(y_train, "y_train")
    labels_test = _as_binary_labels(y_test, "y_test")

    if X_train.shape[0] != len(labels_train):
        raise ValueError(
            f"X_train_raw rows ({X_train.shape[0]}) do not match y_train "
            f"({len(labels_train)})."
        )
    if X_test.shape[0] != len(labels_test):
        raise ValueError(
            f"X_test_raw rows ({X_test.shape[0]}) do not match y_test ({len(labels_test)})."
        )
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Training and external-test feature counts differ: "
            f"{X_train.shape[1]} versus {X_test.shape[1]}."
        )

    names = [str(name) for name in feature_names]
    if len(names) != X_train.shape[1]:
        raise ValueError(
            f"feature_names has {len(names)} entries but the matrices have "
            f"{X_train.shape[1]} columns."
        )
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique.")

    train_eids = _as_eids(eids_train, len(labels_train), "eids_train")
    test_eids = _as_eids(eids_test, len(labels_test), "eids_test")
    overlap = set(pd.Series(train_eids).astype(str)).intersection(
        set(pd.Series(test_eids).astype(str))
    )
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(
            "Training and external-test EIDs overlap, which would leak patients; "
            f"examples: {examples}."
        )

    if isinstance(n_folds, bool) or not isinstance(n_folds, (int, np.integer)):
        raise ValueError(f"n_folds must be an integer; got {n_folds!r}.")
    if int(n_folds) < 2:
        raise ValueError("n_folds must be at least 2.")
    class_counts = np.bincount(labels_train, minlength=2)
    if (class_counts == 0).any():
        raise ValueError(
            "y_train must contain both classes; got "
            f"negative={class_counts[0]}, positive={class_counts[1]}."
        )
    if class_counts.min() < int(n_folds):
        raise ValueError(
            "The minority training class must contain at least n_folds samples "
            "so every internal-validation fold has both classes; got "
            f"negative={class_counts[0]}, positive={class_counts[1]}, "
            f"n_folds={n_folds}."
        )

    globally_empty = np.isnan(X_train).all(axis=0)
    if globally_empty.any():
        missing_names = [names[index] for index in np.flatnonzero(globally_empty)]
        raise ValueError(
            "Training features cannot be entirely missing because a median cannot "
            f"be learned: {missing_names}."
        )

    return (
        X_train,
        labels_train,
        train_eids,
        X_test,
        labels_test,
        test_eids,
        names,
    )


def compute_binary_metrics(
    y_true: Any,
    predicted_probability: Any,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Binary metrics; undefined rates are returned as NaN."""
    labels = _as_binary_labels(y_true, "y_true")
    probabilities = np.asarray(predicted_probability, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != len(labels):
        raise ValueError(
            "predicted_probability must be one-dimensional and match y_true; "
            f"got shape {probabilities.shape} for {len(labels)} labels."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("predicted_probability contains missing or non-finite values.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("predicted_probability must lie in the closed interval [0, 1].")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must lie in the closed interval [0, 1].")

    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    loss = float(log_loss(labels, clipped, labels=[0, 1]))
    if np.unique(labels).size == 2:
        auc = float(roc_auc_score(labels, probabilities))
        auprc = float(average_precision_score(labels, probabilities))
    else:
        auc = float("nan")
        auprc = float("nan")

    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    positive_total = tp + fn
    negative_total = tn + fp

    return {
        "loss": loss,
        "auc": auc,
        "auprc": auprc,
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tpr": float(tp / positive_total) if positive_total else float("nan"),
        "tnr": float(tn / negative_total) if negative_total else float("nan"),
        "fpr": float(fp / negative_total) if negative_total else float("nan"),
        "fnr": float(fn / positive_total) if positive_total else float("nan"),
    }


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=10000,
                ),
            ),
        ]
    )


def _coefficient_table(pipeline: Pipeline, feature_names: Sequence[str]) -> pd.DataFrame:
    imputer: SimpleImputer = pipeline.named_steps["imputer"]
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    classifier: LogisticRegression = pipeline.named_steps["logistic"]

    standardized = np.asarray(classifier.coef_[0], dtype=float)
    scale = np.asarray(scaler.scale_, dtype=float)
    mean = np.asarray(scaler.mean_, dtype=float)
    imputer_median = np.asarray(imputer.statistics_, dtype=float)
    if len(standardized) != len(feature_names):
        raise RuntimeError(
            "The fitted pipeline changed feature dimensionality. This normally "
            "means a fold-training feature was entirely missing."
        )

    original = standardized / scale
    standardized_intercept = float(classifier.intercept_[0])
    original_intercept = float(standardized_intercept - np.dot(original, mean))

    table = pd.DataFrame(
        {
            "feature": list(feature_names),
            "coefficient_standardized": standardized,
            "abs_coefficient_standardized": np.abs(standardized),
            "coefficient_original_units": original,
            "abs_coefficient_original_units": np.abs(original),
            "intercept_standardized": standardized_intercept,
            "intercept_original_units": original_intercept,
            "imputer_median": imputer_median,
            "scaler_mean": mean,
            "scaler_scale": scale,
        }
    )
    return table.sort_values(
        ["abs_coefficient_standardized", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _write_predictions(
    path: Path,
    eids: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            "eid": eids,
            "true_label": labels.astype(np.int64, copy=False),
            "predicted_probability": probabilities,
        }
    ).to_csv(path, index=False)


def _write_pickle(path: Path, pipeline: Pipeline) -> None:
    with path.open("wb") as handle:
        pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _prefixed_metrics(prefix: str, metrics: Mapping[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{name}": float(metrics[name]) for name in METRIC_NAMES}


def _summary_table(
    detailed: pd.DataFrame,
    n_folds: int,
    selection_metric: str,
    use_pos_weight: bool,
    ensemble_metrics: Mapping[str, float],
) -> pd.DataFrame:
    row: Dict[str, Any] = {
        "n_folds": n_folds,
        "selection_metric": selection_metric,
        "use_pos_weight": use_pos_weight,
    }
    for split_name in SPLIT_NAMES:
        for metric_name in METRIC_NAMES:
            column = f"{split_name}_{metric_name}"
            values = detailed[column]
            row[f"mean_{column}"] = float(values.mean())
            row[f"std_{column}"] = float(values.std(ddof=1))
            row[f"min_{column}"] = float(values.min())
            row[f"max_{column}"] = float(values.max())
    for column in ("positive_class_weight", "fold_train_n", "fold_train_positive"):
        values = detailed[column]
        row[f"mean_{column}"] = float(values.mean())
        row[f"std_{column}"] = float(values.std(ddof=1))
        row[f"min_{column}"] = float(values.min())
        row[f"max_{column}"] = float(values.max())
    row.update(_prefixed_metrics("ensemble_external_test", ensemble_metrics))
    return pd.DataFrame([row])


def _rank_median_fold(
    detailed: pd.DataFrame,
    selection_metric: str,
) -> Tuple[int, str]:
    """Pick the median fold using internal-validation results only."""
    selection_column = f"internal_val_{selection_metric}"
    median_ranking = detailed.sort_values(
        [selection_column, "fold"],
        ascending=[True, True],
        kind="mergesort",
    )
    median_value = median_ranking.iloc[len(median_ranking) // 2][selection_column]
    median_fold_idx = int(
        detailed.loc[detailed[selection_column] == median_value, "fold"].min()
    )
    return median_fold_idx, selection_column


def _selected_fold_info(
    detailed: pd.DataFrame,
    fold_idx: int,
    selection_metric: str,
    selection_column: str,
    ensemble_metrics: Mapping[str, float],
) -> Dict[str, Any]:
    row = detailed.loc[detailed["fold"] == fold_idx].iloc[0].to_dict()
    row["median_fold_idx"] = fold_idx
    row["selection_metric"] = selection_metric
    row["selection_column"] = selection_column
    row["selected_internal_val_metric"] = float(row[selection_column])
    row.update(_prefixed_metrics("ensemble_external_test", ensemble_metrics))
    return row


def run_logistic_cv(
    X_train_raw,
    y_train,
    eids_train,
    X_test_raw,
    y_test,
    eids_test,
    feature_names,
    output_dir,
    n_folds=5,
    seed=42,
    selection_metric="loss",
    use_pos_weight=True,
    logger=None,
) -> dict:
    """Fit fold-local pipelines and select the median fold."""
    active_logger = logger if logger is not None else LOGGER
    metric = str(selection_metric).lower()
    if metric not in {"auc", "loss"}:
        raise ValueError("selection_metric must be either 'auc' or 'loss'.")
    if not isinstance(use_pos_weight, (bool, np.bool_)):
        raise ValueError("use_pos_weight must be a boolean.")
    use_weight = bool(use_pos_weight)

    (
        X_train,
        labels_train,
        train_eids,
        X_test,
        labels_test,
        test_eids,
        names,
    ) = _validate_inputs(
        X_train_raw,
        y_train,
        eids_train,
        X_test_raw,
        y_test,
        eids_test,
        feature_names,
        n_folds,
    )
    n_folds = int(n_folds)
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    active_logger.info("=" * 80)
    active_logger.info("LEAKAGE-SAFE STRATIFIED LOGISTIC CROSS-VALIDATION")
    active_logger.info("=" * 80)
    active_logger.info(
        "Training rows=%d, external-test rows=%d, features=%d, folds=%d, seed=%s",
        len(labels_train),
        len(labels_test),
        X_train.shape[1],
        n_folds,
        seed,
    )
    active_logger.info(
        "Median-fold selection uses only %s; external-test outcomes are reporting-only.",
        f"internal_val_{metric}",
    )
    if use_weight:
        active_logger.warning(
            "Positive-class weighting is enabled. Probabilities describe the "
            "fold-specific reweighted objective and are not prevalence-calibrated."
        )

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_pipelines: List[Pipeline] = []
    fold_indices: List[Dict[str, Any]] = []
    fold_metadata: List[Dict[str, Any]] = []
    coefficient_tables: List[pd.DataFrame] = []
    detailed_rows: List[Dict[str, Any]] = []
    external_probabilities: List[np.ndarray] = []
    oof_probability = np.full(len(labels_train), np.nan, dtype=float)
    oof_fold = np.full(len(labels_train), -1, dtype=np.int64)

    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(X_train, labels_train)
    ):
        X_fold_train = X_train[train_idx]
        y_fold_train = labels_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = labels_train[val_idx]

        fold_empty = np.isnan(X_fold_train).all(axis=0)
        if fold_empty.any():
            missing_names = [names[index] for index in np.flatnonzero(fold_empty)]
            raise ValueError(
                f"Fold {fold_idx} has feature(s) entirely missing in its training "
                f"rows, so a leakage-safe median cannot be learned: {missing_names}."
            )

        n_positive = int(np.count_nonzero(y_fold_train == 1))
        n_negative = int(np.count_nonzero(y_fold_train == 0))
        positive_weight = float(n_negative / n_positive) if use_weight else 1.0
        sample_weight = np.where(y_fold_train == 1, positive_weight, 1.0)

        pipeline = _build_pipeline()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(
                X_fold_train,
                y_fold_train,
                logistic__sample_weight=sample_weight,
            )
        convergence_messages = [
            str(item.message)
            for item in caught_warnings
            if issubclass(item.category, ConvergenceWarning)
        ]
        if convergence_messages:
            active_logger.warning(
                "Fold %d emitted a convergence warning: %s",
                fold_idx,
                " | ".join(convergence_messages),
            )

        train_probability = pipeline.predict_proba(X_fold_train)[:, 1]
        val_probability = pipeline.predict_proba(X_fold_val)[:, 1]
        test_probability = pipeline.predict_proba(X_test)[:, 1]
        train_metrics = compute_binary_metrics(y_fold_train, train_probability)
        val_metrics = compute_binary_metrics(y_fold_val, val_probability)
        test_metrics = compute_binary_metrics(labels_test, test_probability)

        if np.unique(labels_test).size < 2 and fold_idx == 0:
            active_logger.warning(
                "The external test labels contain one class; AUROC, AUPRC, and "
                "rates requiring the absent class are saved as NaN."
            )

        metadata: Dict[str, Any] = {
            "fold": fold_idx,
            "fold_train_n": int(len(train_idx)),
            "fold_train_negative": n_negative,
            "fold_train_positive": n_positive,
            "positive_class_weight": positive_weight,
            "use_pos_weight": use_weight,
            "internal_val_n": int(len(val_idx)),
            "internal_val_negative": int(np.count_nonzero(y_fold_val == 0)),
            "internal_val_positive": int(np.count_nonzero(y_fold_val == 1)),
            "external_test_n": int(len(labels_test)),
            "external_test_negative": int(np.count_nonzero(labels_test == 0)),
            "external_test_positive": int(np.count_nonzero(labels_test == 1)),
            "seed": int(seed),
            "n_folds": n_folds,
            "convergence_warning": bool(convergence_messages),
            "n_iter": int(pipeline.named_steps["logistic"].n_iter_[0]),
        }
        pipeline.training_metadata_ = metadata.copy()

        row = metadata.copy()
        row.update(_prefixed_metrics("train", train_metrics))
        row.update(_prefixed_metrics("internal_val", val_metrics))
        row.update(_prefixed_metrics("external_test", test_metrics))
        detailed_rows.append(row)

        _write_predictions(
            output_path / f"fold_{fold_idx}_train.csv",
            train_eids[train_idx],
            y_fold_train,
            train_probability,
        )
        _write_predictions(
            output_path / f"fold_{fold_idx}_internal_val.csv",
            train_eids[val_idx],
            y_fold_val,
            val_probability,
        )
        _write_predictions(
            output_path / f"fold_{fold_idx}_external_test.csv",
            test_eids,
            labels_test,
            test_probability,
        )

        pipeline_path = output_path / f"fold_{fold_idx}_pipeline.pkl"
        coefficient_path = output_path / f"fold_{fold_idx}_coefficients.csv"
        metadata_path = output_path / f"fold_{fold_idx}_model_metadata.json"
        coefficient_table = _coefficient_table(pipeline, names)
        _write_pickle(pipeline_path, pipeline)
        coefficient_table.to_csv(coefficient_path, index=False)
        _write_json(metadata_path, metadata)

        oof_probability[val_idx] = val_probability
        oof_fold[val_idx] = fold_idx
        external_probabilities.append(test_probability)
        coefficient_tables.append(coefficient_table)
        fold_pipelines.append(pipeline)
        fold_indices.append(
            {"fold": fold_idx, "train_idx": train_idx.copy(), "val_idx": val_idx.copy()}
        )
        fold_metadata.append(metadata)

        active_logger.info(
            "Fold %d: train n=%d (negative=%d, positive=%d), positive weight=%.6g; "
            "internal AUC=%.4f, loss=%.4f; external AUC=%s, loss=%.4f",
            fold_idx,
            len(train_idx),
            n_negative,
            n_positive,
            positive_weight,
            val_metrics["auc"],
            val_metrics["loss"],
            f"{test_metrics['auc']:.4f}" if np.isfinite(test_metrics["auc"]) else "NaN",
            test_metrics["loss"],
        )

    if np.isnan(oof_probability).any() or (oof_fold < 0).any():
        raise RuntimeError("OOF construction failed: at least one training row was not predicted.")
    validation_counts = np.bincount(
        np.concatenate([item["val_idx"] for item in fold_indices]),
        minlength=len(labels_train),
    )
    if not np.all(validation_counts == 1):
        raise RuntimeError("OOF construction failed: validation rows were missing or repeated.")

    oof_path = output_path / "oof_internal_val_predictions.csv"
    pd.DataFrame(
        {
            "eid": train_eids,
            "true_label": labels_train,
            "predicted_probability": oof_probability,
            "fold": oof_fold,
        }
    ).to_csv(oof_path, index=False)

    external_matrix = np.vstack(external_probabilities)
    ensemble_probability = external_matrix.mean(axis=0)
    ensemble_metrics = compute_binary_metrics(labels_test, ensemble_probability)
    ensemble_path = output_path / "ensemble_external_test.csv"
    pd.DataFrame(
        {
            "eid": test_eids,
            "true_label": labels_test,
            "predicted_probability": ensemble_probability,
            "predicted_probability_std": external_matrix.std(axis=0, ddof=0),
        }
    ).to_csv(ensemble_path, index=False)
    ensemble_metrics_path = output_path / "ensemble_external_test_metrics.csv"
    pd.DataFrame(
        [
            {
                "n": len(labels_test),
                "negative": int(np.count_nonzero(labels_test == 0)),
                "positive": int(np.count_nonzero(labels_test == 1)),
                **ensemble_metrics,
            }
        ]
    ).to_csv(ensemble_metrics_path, index=False)

    detailed = pd.DataFrame(detailed_rows).sort_values("fold").reset_index(drop=True)
    median_fold_idx, selection_column = _rank_median_fold(detailed, metric)
    median_info = _selected_fold_info(
        detailed,
        median_fold_idx,
        metric,
        selection_column,
        ensemble_metrics,
    )

    detailed_path = output_path / "cv_results_detailed.csv"
    summary_path = output_path / "cv_results_summary.csv"
    median_info_path = output_path / "median_fold_info.csv"
    detailed.to_csv(detailed_path, index=False)
    summary = _summary_table(
        detailed,
        n_folds,
        metric,
        use_weight,
        ensemble_metrics,
    )
    summary.to_csv(summary_path, index=False)
    pd.DataFrame([median_info]).to_csv(median_info_path, index=False)
    (output_path / "median_fold_idx.txt").write_text(
        f"{median_fold_idx}\n", encoding="utf-8"
    )

    median_pipeline_path = output_path / "median_pipeline.pkl"
    median_coefficients_path = output_path / "median_coefficients.csv"
    _write_pickle(median_pipeline_path, fold_pipelines[median_fold_idx])
    coefficient_tables[median_fold_idx].to_csv(median_coefficients_path, index=False)

    active_logger.info(
        "Selected median fold %d using only %s.",
        median_fold_idx,
        selection_column,
    )
    active_logger.info(
        "External-test ensemble: AUC=%s, AUPRC=%s, loss=%.4f.",
        f"{ensemble_metrics['auc']:.4f}"
        if np.isfinite(ensemble_metrics["auc"])
        else "NaN",
        f"{ensemble_metrics['auprc']:.4f}"
        if np.isfinite(ensemble_metrics["auprc"])
        else "NaN",
        ensemble_metrics["loss"],
    )

    paths = {
        "output_dir": str(output_path),
        "cv_detailed": str(detailed_path),
        "cv_summary": str(summary_path),
        "median_fold": str(median_info_path),
        "median_pipeline": str(median_pipeline_path),
        "median_coefficients": str(median_coefficients_path),
        "oof_internal_val": str(oof_path),
        "ensemble_external_test": str(ensemble_path),
        "ensemble_external_test_metrics": str(ensemble_metrics_path),
    }
    return {
        "checkpoint_dir": str(output_path),
        **paths,
        "paths": paths,
        "cv_results_detailed": detailed,
        "cv_results_summary": summary,
        "median_fold_idx": median_fold_idx,
        "median_fold_info": median_info,
        "ensemble_metrics": ensemble_metrics,
        "fold_pipelines": fold_pipelines,
        "median_pipeline_object": fold_pipelines[median_fold_idx],
        "fold_indices": fold_indices,
        "fold_metadata": fold_metadata,
    }


__all__ = ["METRIC_NAMES", "compute_binary_metrics", "run_logistic_cv"]
