#!/usr/bin/env python
"""LVEF + max-WT logistic data loader."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]
LOGGER = logging.getLogger(__name__)


class LvefMaxwtLogisticDataModule:
    """Load features and labels, then filter the fixed cohort."""

    def __init__(
        self,
        train_csv: PathLike,
        external_test_csv: PathLike,
        disease_cols: Sequence[str],
        filter_groups: Sequence[str],
        feature_cols: Sequence[str],
        virtual_label_map: Optional[Dict[str, Sequence[str]]] = None,
        eid_col: str = "eid_18545",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.train_csv = Path(train_csv)
        self.external_test_csv = Path(external_test_csv)
        self.logger = logger if logger is not None else LOGGER
        self.eid_col = eid_col

        self.filter_groups = list(filter_groups)
        self.disease_cols = list(disease_cols)
        self.virtual_label_map = {
            key: list(value) for key, value in (virtual_label_map or {}).items()
        }
        self.feature_cols = list(feature_cols)
        self.physical_filter_cols = self._expand_groups(self.filter_groups)

        self._train_df: Optional[pd.DataFrame] = None
        self._external_test_df: Optional[pd.DataFrame] = None
        self._cohort_summary: Optional[pd.DataFrame] = None

    @staticmethod
    def _deduplicate(values: Sequence[str]) -> List[str]:
        return list(dict.fromkeys(values))

    def _expand_groups(self, groups: Sequence[str]) -> List[str]:
        expanded: List[str] = []
        for group in groups:
            expanded.extend(self.virtual_label_map.get(group, [group]))
        return self._deduplicate(expanded)

    def _read_required_columns(self, csv_path: Path, dataset: str) -> pd.DataFrame:
        self.logger.info("Reading %s header: %s", dataset, csv_path)
        try:
            header = pd.read_csv(csv_path, nrows=0)
        except Exception as exc:
            raise ValueError(f"Could not read {dataset} CSV header at {csv_path}: {exc}") from exc

        header_cols = list(header.columns)
        header_set = set(header_cols)

        ordinary_required = [
            self.eid_col,
            *self.feature_cols,
            *(col for col in self.disease_cols if col != "MACE_all"),
        ]
        missing = [col for col in ordinary_required if col not in header_set]
        if "MACE_all" in self.disease_cols and "MACE_all" not in header_set and "MACE" not in header_set:
            missing.append("MACE_all (or legacy MACE)")
        if missing:
            raise ValueError(
                f"{dataset} CSV is missing required columns: {missing}. "
                f"CSV path: {csv_path}"
            )

        usecols = list(ordinary_required)
        if "MACE_all" in header_set:
            usecols.append("MACE_all")
        if "MACE" in header_set:
            usecols.append("MACE")
        usecols = self._deduplicate(usecols)

        self.logger.info(
            "Loading %s required columns only (%d of %d CSV columns)",
            dataset,
            len(usecols),
            len(header_cols),
        )
        try:
            df = pd.read_csv(csv_path, usecols=usecols)
        except Exception as exc:
            raise ValueError(f"Could not read {dataset} CSV at {csv_path}: {exc}") from exc

        df = self._canonicalize_mace_column(df, dataset)
        self._validate_dataframe(df, dataset)

        canonical_order = [self.eid_col, *self.feature_cols, *self.disease_cols]
        return df.loc[:, canonical_order]

    def _canonicalize_mace_column(
        self, df: pd.DataFrame, dataset: str
    ) -> pd.DataFrame:
        has_canonical = "MACE_all" in df.columns
        has_legacy = "MACE" in df.columns

        if has_canonical and has_legacy:
            canonical = self._validated_binary_series(
                df["MACE_all"], "MACE_all", dataset
            )
            legacy = self._validated_binary_series(df["MACE"], "MACE", dataset)
            mismatch = canonical.ne(legacy)
            if mismatch.any():
                example_rows = df.index[mismatch].tolist()[:10]
                raise ValueError(
                    f"{dataset} contains both MACE_all and legacy MACE, but they "
                    f"disagree in {int(mismatch.sum())} rows; example row indices: "
                    f"{example_rows}"
                )
            self.logger.warning(
                "%s contains both MACE_all and legacy MACE with matching values; "
                "using MACE_all and dropping MACE.",
                dataset,
            )
            df = df.drop(columns=["MACE"])
        elif has_legacy and "MACE_all" in self.disease_cols:
            self.logger.warning(
                "%s uses legacy 'MACE'; renaming in memory to canonical 'MACE_all'.",
                dataset,
            )
            df = df.rename(columns={"MACE": "MACE_all"})

        return df

    @staticmethod
    def _validated_binary_series(
        series: pd.Series, column: str, dataset: str
    ) -> pd.Series:
        if series.isna().any():
            raise ValueError(
                f"{dataset} disease column '{column}' contains "
                f"{int(series.isna().sum())} missing values."
            )
        numeric = pd.to_numeric(series, errors="coerce")
        invalid_numeric = numeric.isna()
        if invalid_numeric.any():
            examples = series.loc[invalid_numeric].astype(str).unique().tolist()[:10]
            raise ValueError(
                f"{dataset} disease column '{column}' contains non-numeric values: "
                f"{examples}"
            )
        invalid_binary = ~numeric.isin([0, 1])
        if invalid_binary.any():
            examples = numeric.loc[invalid_binary].unique().tolist()[:10]
            raise ValueError(
                f"{dataset} disease column '{column}' must contain only 0/1; "
                f"found values such as {examples}."
            )
        return numeric.astype(np.int8)

    def _validate_dataframe(self, df: pd.DataFrame, dataset: str) -> None:
        required = [self.eid_col, *self.feature_cols, *self.disease_cols]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{dataset} is missing required canonical columns: {missing}")

        if df[self.eid_col].isna().any():
            raise ValueError(
                f"{dataset} ID column '{self.eid_col}' contains "
                f"{int(df[self.eid_col].isna().sum())} missing values."
            )
        duplicated = df[self.eid_col].duplicated(keep=False)
        if duplicated.any():
            examples = df.loc[duplicated, self.eid_col].drop_duplicates().tolist()[:10]
            raise ValueError(
                f"{dataset} contains {int(duplicated.sum())} rows with duplicated "
                f"'{self.eid_col}' values; example IDs: {examples}"
            )

        for column in self.disease_cols:
            df[column] = self._validated_binary_series(df[column], column, dataset)

        for column in self.feature_cols:
            original = df[column]
            numeric = pd.to_numeric(original, errors="coerce")
            invalid = original.notna() & numeric.isna()
            if invalid.any():
                examples = original.loc[invalid].astype(str).unique().tolist()[:10]
                raise ValueError(
                    f"{dataset} feature '{column}' contains non-numeric values: "
                    f"{examples}"
                )
            infinite = numeric.notna() & ~np.isfinite(numeric)
            if infinite.any():
                raise ValueError(
                    f"{dataset} feature '{column}' contains "
                    f"{int(infinite.sum())} infinite values."
                )
            df[column] = numeric.astype(float)

    def _logical_mask(self, df: pd.DataFrame, group: str) -> pd.Series:
        physical = self._expand_groups([group])
        missing = [col for col in physical if col not in df.columns]
        if missing:
            raise ValueError(
                f"Cannot resolve logical group '{group}'; missing physical columns: {missing}"
            )
        return df[physical].eq(1).any(axis=1)

    def _group_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        return {
            group: int(self._logical_mask(df, group).sum())
            for group in self.filter_groups
        }

    def _filter_fixed_cohort(
        self, df: pd.DataFrame, dataset: str, source_csv: Path
    ) -> Tuple[pd.DataFrame, List[dict]]:
        physical = list(self.physical_filter_cols)
        missing = [col for col in physical if col not in df.columns]
        if missing:
            raise ValueError(
                f"{dataset} is missing physical cohort-filter columns: {missing}"
            )

        before_counts = self._group_distribution(df)
        before_n = len(df)

        # Keep rows with any physical filter-group label.
        keep_mask = df[physical].eq(1).any(axis=1)
        filtered = df.loc[keep_mask].copy().reset_index(drop=True)
        after_n = len(filtered)
        after_counts = self._group_distribution(filtered)

        self.logger.info("=== %s fixed-cohort filtering ===", dataset)
        self.logger.info("Source CSV: %s", source_csv)
        self.logger.info("Rows before filtering: %d", before_n)
        self.logger.info("Fixed logical filter groups: %s", self.filter_groups)
        self.logger.info("Resolved physical filter columns: %s", physical)
        self.logger.info("Logical group distribution BEFORE filtering:")
        for group in self.filter_groups:
            count = before_counts[group]
            pct = 100.0 * count / before_n if before_n else float("nan")
            self.logger.info("  %-14s %6d (%6.2f%%)", group, count, pct)
        self.logger.info("Rows after filtering: %d", after_n)
        self.logger.info("Rows removed: %d", before_n - after_n)
        self.logger.info("Logical group distribution AFTER filtering:")
        for group in self.filter_groups:
            count = after_counts[group]
            pct = 100.0 * count / after_n if after_n else float("nan")
            self.logger.info("  %-14s %6d (%6.2f%%)", group, count, pct)

        if filtered.empty:
            raise ValueError(
                f"{dataset} has no rows after filtering by filter_groups "
                f"{self.filter_groups} (physical columns {physical})."
            )

        records = []
        for group in self.filter_groups:
            records.append(
                {
                    "dataset": dataset,
                    "source_csv": str(source_csv),
                    "rows_loaded": before_n,
                    "rows_kept": after_n,
                    "rows_removed": before_n - after_n,
                    "filter_groups": "|".join(self.filter_groups),
                    "physical_filter_columns": "|".join(physical),
                    "logical_group": group,
                    "positive_before": before_counts[group],
                    "prevalence_before": (
                        before_counts[group] / before_n if before_n else np.nan
                    ),
                    "positive_after": after_counts[group],
                    "prevalence_after": (
                        after_counts[group] / after_n if after_n else np.nan
                    ),
                }
            )
        return filtered, records

    def _validate_no_train_test_overlap(
        self, train_df: pd.DataFrame, external_test_df: pd.DataFrame
    ) -> None:
        train_ids = set(train_df[self.eid_col].tolist())
        test_ids = set(external_test_df[self.eid_col].tolist())
        overlap = train_ids.intersection(test_ids)
        if overlap:
            examples = sorted(overlap, key=str)[:10]
            raise ValueError(
                f"Patient leakage detected: train and external_test share "
                f"{len(overlap)} '{self.eid_col}' values; example IDs: {examples}"
            )
        self.logger.info(
            "Train/external-test EID leakage check passed: no overlap in '%s'.",
            self.eid_col,
        )

    def prepare_cohorts(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load, validate, and filter train/test cohorts once."""
        if self._cohort_summary is not None:
            return (
                self._train_df.copy(),  # type: ignore[union-attr]
                self._external_test_df.copy(),  # type: ignore[union-attr]
                self._cohort_summary.copy(),
            )

        train_raw = self._read_required_columns(self.train_csv, "train")
        test_raw = self._read_required_columns(
            self.external_test_csv, "external_test"
        )
        self._validate_no_train_test_overlap(train_raw, test_raw)

        train_filtered, train_records = self._filter_fixed_cohort(
            train_raw, "train", self.train_csv
        )
        test_filtered, test_records = self._filter_fixed_cohort(
            test_raw, "external_test", self.external_test_csv
        )

        self._train_df = train_filtered
        self._external_test_df = test_filtered
        self._cohort_summary = pd.DataFrame(train_records + test_records)

        overview = self._cohort_summary[
            ["dataset", "source_csv", "rows_loaded", "rows_kept", "rows_removed"]
        ].drop_duplicates()
        self.logger.info("=== Consolidated cohort-filter summary ===\n%s", overview.to_string(index=False))

        return (
            self._train_df.copy(),
            self._external_test_df.copy(),
            self._cohort_summary.copy(),
        )

    def create_binary_labels(
        self,
        df: pd.DataFrame,
        target: str,
        dataset_name: str = "dataset",
        require_both_classes: bool = True,
    ) -> np.ndarray:
        """Return integer one-vs-rest labels for ``target``."""
        supported = self._deduplicate([*self.filter_groups, *self.disease_cols])
        if target not in supported:
            raise ValueError(
                f"Unknown target '{target}'. Supported targets are: {supported}"
            )

        labels = self._logical_mask(df, target).astype(np.int8).to_numpy()
        n_positive = int(labels.sum())
        n_negative = int(len(labels) - n_positive)
        if require_both_classes and (n_positive == 0 or n_negative == 0):
            raise ValueError(
                f"{dataset_name} target '{target}' requires both classes after fixed-cohort "
                f"filtering; found {n_positive} positive and {n_negative} negative rows."
            )
        if not require_both_classes and (n_positive == 0 or n_negative == 0):
            self.logger.warning(
                "%s target '%s' contains one class (%d positive, %d negative); "
                "probabilities will be saved but class-dependent metrics will be NaN.",
                dataset_name,
                target,
                n_positive,
                n_negative,
            )

        self.logger.info(
            "Created %s %s-vs-rest labels: %d positive, %d negative (%.2f%% positive).",
            dataset_name,
            target,
            n_positive,
            n_negative,
            100.0 * n_positive / len(labels),
        )
        return labels

    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing selected features: {missing}")
        return df.loc[:, self.feature_cols].copy()

    def get_feature_names(self) -> List[str]:
        return list(self.feature_cols)

    def get_eids(self, df: pd.DataFrame) -> pd.Series:
        if self.eid_col not in df.columns:
            raise ValueError(f"DataFrame is missing ID column '{self.eid_col}'.")
        return df[self.eid_col].copy()

    def get_labels(
        self, df: pd.DataFrame, target: str, dataset_name: str = "dataset"
    ) -> np.ndarray:
        return self.create_binary_labels(df, target, dataset_name=dataset_name).copy()

    @property
    def train_cohort(self) -> pd.DataFrame:
        if self._train_df is None:
            self.prepare_cohorts()
        return self._train_df.copy()  

    @property
    def external_test_cohort(self) -> pd.DataFrame:
        if self._external_test_df is None:
            self.prepare_cohorts()
        return self._external_test_df.copy()  

    @property
    def cohort_summary(self) -> pd.DataFrame:
        if self._cohort_summary is None:
            self.prepare_cohorts()
        return self._cohort_summary.copy() 
