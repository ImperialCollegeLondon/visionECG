#!/usr/bin/env python3
"""Data loader for the ECG-parameters prognostic-index pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config_ecg_parameters_prognostic_index import (
    DURATION_COL,
    EVENT_ALIASES,
    EVENT_COL,
    JOIN_COL,
    LABEL_COL,
    MIN_TRAIN_CASES_TO_FIT,
    SPARSE_EVENT_WARNING_THRESHOLD,
)


def resolve_prefixed_path(abs_pre: str, path: str) -> str:
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    return str(Path(abs_pre).expanduser() / path)


def detect_id_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"None of {candidates} were found. Available columns: {df.columns.tolist()}"
    )


def warn_sparse_events(split_name: str, event_count: int, logger: logging.Logger) -> None:
    if event_count < SPARSE_EVENT_WARNING_THRESHOLD:
        logger.warning(
            f"{split_name} has only {event_count} events. Estimates may be unstable."
        )


def build_landmark_labels(
    durations: np.ndarray, events: np.ndarray, landmark_days: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Censored-as-control 5-year landmark labelling."""
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    cases = (durations <= landmark_days) & (events == 1)
    y = np.zeros(len(durations), dtype=int)
    y[cases] = 1
    summary = {
        "landmark_days": int(landmark_days),
        "n_total": int(len(durations)),
        "n_cases": int(cases.sum()),
        "n_controls": int((~cases).sum()),
        "label_strategy": "censored_as_control",
    }
    return y, summary


def _resolve_event_prefix(event_alias: str) -> str:
    try:
        return EVENT_ALIASES[event_alias]
    except KeyError as exc:
        raise ValueError(
            f"Unknown event_alias={event_alias!r}. Choose one of {sorted(EVENT_ALIASES)}."
        ) from exc


def _read_main_table(main_table_file: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info("== Loading main table ==")
    logger.info(f"  Path: {main_table_file}")
    df = pd.read_csv(main_table_file)
    logger.info(f"  Rows: {len(df)}   Cols: {df.shape[1]}")
    for col in (JOIN_COL, "eid", "eid_40616"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _read_split_ids(split_file: str, split_name: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"== Loading {split_name} split IDs ==")
    logger.info(f"  Path: {split_file}")
    header = pd.read_csv(split_file, nrows=0)
    id_col = detect_id_col(header, [JOIN_COL, "eid_40616", "eid"])
    df = pd.read_csv(split_file, usecols=[id_col])
    df = df.rename(columns={id_col: JOIN_COL})
    df[JOIN_COL] = pd.to_numeric(df[JOIN_COL], errors="coerce")
    df = df.dropna(subset=[JOIN_COL]).drop_duplicates(subset=[JOIN_COL])
    df[JOIN_COL] = df[JOIN_COL].astype(np.int64)
    logger.info(f"  {split_name} IDs: {len(df)}")
    return df


def _restrict_to_event_cohort(
    df_all: pd.DataFrame, event_alias: str, landmark_days: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Create standardized follow-up and landmark columns."""
    prefix = _resolve_event_prefix(event_alias)
    status_col = f"{prefix}_event_status"
    days_col = f"{prefix}_follow_up_days"
    for c in (status_col, days_col):
        if c not in df_all.columns:
            raise ValueError(
                f"Column {c!r} not found. Available with prefix {prefix!r}: "
                f"{[x for x in df_all.columns if x.startswith(prefix)]}"
            )

    n_before = len(df_all)
    df = df_all.dropna(subset=[status_col, days_col]).copy()
    n_after = len(df)
    logger.info(
        f"  {event_alias}: dropped {n_before - n_after} pre-MRI or missing-followup "
        f"rows ({n_after} remain)."
    )

    df[EVENT_COL] = df[status_col].astype(int)
    df[DURATION_COL] = df[days_col].astype(float)
    y, summary = build_landmark_labels(
        df[DURATION_COL].values, df[EVENT_COL].values, landmark_days,
    )
    df[LABEL_COL] = y

    logger.info(
        f"  {event_alias}: landmark {landmark_days}d -> cases={summary['n_cases']}, "
        f"controls={summary['n_controls']} "
        f"({100.0 * summary['n_cases'] / max(summary['n_total'], 1):.2f}% pos)."
    )
    return df


def _extract_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in main table: {missing}")
    return df[feature_cols].to_numpy(dtype=np.float64, copy=True)


def load_event_cohort(
    main_table_file: str, train_file: str, test_file: str,
    event_alias: str, landmark_days: int,
    feature_cols: List[str],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """Prepare one event's features and dataset splits."""
    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("feature_cols must be a non-empty list of column names.")

    df_all = _read_main_table(main_table_file, logger)
    df_event = _restrict_to_event_cohort(df_all, event_alias, landmark_days, logger)

    df_train_ids = _read_split_ids(train_file, "TRAIN", logger)
    df_test_ids = _read_split_ids(test_file, "TEST", logger)

    if JOIN_COL not in df_event.columns:
        raise ValueError(
            f"Main table missing join column {JOIN_COL!r}. "
            f"Available id-like columns: "
            f"{[c for c in df_event.columns if 'eid' in c or c == JOIN_COL]}"
        )
    df_event[JOIN_COL] = pd.to_numeric(df_event[JOIN_COL], errors="coerce")
    df_event = df_event.dropna(subset=[JOIN_COL]).copy()
    df_event[JOIN_COL] = df_event[JOIN_COL].astype(np.int64)

    df_train = df_event.merge(df_train_ids, on=JOIN_COL, how="inner").reset_index(drop=True)
    df_test = df_event.merge(df_test_ids, on=JOIN_COL, how="inner").reset_index(drop=True)

    logger.info(
        f"  Join yield -> TRAIN {len(df_train)} rows, TEST {len(df_test)} rows "
        f"(from {len(df_event)} at-risk in main table)."
    )
    warn_sparse_events("TRAIN", int(df_train[LABEL_COL].sum()), logger)
    warn_sparse_events("TEST", int(df_test[LABEL_COL].sum()), logger)

    if int(df_train[LABEL_COL].sum()) < MIN_TRAIN_CASES_TO_FIT:
        raise ValueError(
            f"TRAIN has only {int(df_train[LABEL_COL].sum())} landmark cases; "
            f"below MIN_TRAIN_CASES_TO_FIT={MIN_TRAIN_CASES_TO_FIT}."
        )

    logger.info(
        f"  Feature selection: {len(feature_cols)} columns "
        f"(first 3 = {feature_cols[:3]}, last 3 = {feature_cols[-3:]})"
    )
    X_train_raw = _extract_feature_matrix(df_train, feature_cols)
    X_test_raw = _extract_feature_matrix(df_test, feature_cols)
    return df_train, df_test, X_train_raw, X_test_raw, list(feature_cols)
