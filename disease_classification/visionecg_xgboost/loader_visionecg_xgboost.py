#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""visionECG-XGBoost data loader for one-vs-rest disease classification."""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionECGXGBoostDataModule:
    """One-vs-rest binary classification data module."""

    def __init__(
        self,
        train_csv: str,
        target_disease: str,
        disease_cols: list,
        filter_groups: list,
        measurement_cols: list,
        external_test_csv: Optional[str] = None,
        train_ratio: float = 0.64,
        internal_val_ratio: float = 0.16,
        external_test_ratio: float = 0.20,
        seed: int = 42,
        virtual_label_map: Optional[dict] = None,
    ):
        self.train_csv = train_csv
        self.target_disease = target_disease
        self.external_test_csv = external_test_csv
        self.train_ratio = train_ratio
        self.internal_val_ratio = internal_val_ratio
        self.external_test_ratio = external_test_ratio
        self.seed = seed
        self.virtual_label_map = dict(virtual_label_map) if virtual_label_map else {}
        self.disease_cols = list(disease_cols)
        self.filter_groups = list(filter_groups)
        self.measurement_cols = list(measurement_cols)
        self.feature_cols = self.measurement_cols
        self.id_cols = ['patient_id', 'eid_40616', 'eid_18545']

        if (self.target_disease not in self.filter_groups
                and self.target_disease not in self.virtual_label_map):
            raise ValueError(
                f"target_disease '{self.target_disease}' is not in filter_groups "
                f"{self.filter_groups} and not in virtual_label_map keys "
                f"{list(self.virtual_label_map)}."
            )

        self.two_csv_mode = (external_test_csv is not None)

        if not self.two_csv_mode:
            total_ratio = train_ratio + internal_val_ratio + external_test_ratio
            if abs(total_ratio - 1.0) > 0.01:
                raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.3f}")

        self.X_train = None
        self.y_train = None
        self.X_internal_val = None
        self.y_internal_val = None
        self.X_external_test = None
        self.y_external_test = None
        self.eids_train = None
        self.eids_internal_val = None
        self.eids_external_test = None
        self.measurement_scaler = None

    def _expand_label_groups(self, groups: list) -> list:
        expanded = []
        for group in groups:
            expanded.extend(self.virtual_label_map.get(group, [group]))
        deduped = []
        for col in expanded:
            if col not in deduped:
                deduped.append(col)
        return deduped

    def _validate_label_columns(self, df: pd.DataFrame, label_cols: list, context: str):
        missing = [col for col in label_cols if col not in df.columns]
        if missing:
            available = [col for col in self.disease_cols if col in df.columns]
            raise ValueError(
                f"Missing disease label columns for {context}: {missing}. "
                f"Requested groups: {self.filter_groups}; expanded: {label_cols}. "
                f"Available: {available}"
            )

    def _filter_labeled_patients(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("\n=== Filtering Labeled Patients ===")
        logger.info(f"Original samples: {len(df)}")
        logger.info(f"Filter groups: {self.filter_groups}")

        filter_label_cols = self._expand_label_groups(self.filter_groups)
        self._validate_label_columns(df, filter_label_cols, "cohort filtering")

        has_any_label = df[filter_label_cols].sum(axis=1) > 0
        filtered = df[has_any_label].copy()

        logger.info(f"After filtering: {len(filtered)} samples "
                    f"(removed {len(df) - len(filtered)})")

        if len(filtered) == 0:
            raise ValueError("No samples left after filtering.")

        for disease in self.filter_groups:
            disease_cols_expanded = self._expand_label_groups([disease])
            self._validate_label_columns(filtered, disease_cols_expanded, f"distribution {disease}")
            count = (filtered[disease_cols_expanded].sum(axis=1) > 0).sum()
            pct = 100 * count / len(filtered)
            logger.info(f"  {disease}: {count} ({pct:.2f}%)")

        return filtered

    def _create_binary_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"\n=== Binary Labels: {self.target_disease} vs rest ===")

        target_label_cols = self._expand_label_groups([self.target_disease])
        self._validate_label_columns(df, target_label_cols, f"target {self.target_disease}")

        df['label'] = (df[target_label_cols].sum(axis=1) > 0).astype(int)

        n_positive = (df['label'] == 1).sum()
        n_negative = (df['label'] == 0).sum()

        if n_positive == 0:
            raise ValueError(f"No positive samples for '{self.target_disease}'.")
        if n_negative == 0:
            raise ValueError(f"No negative samples for '{self.target_disease}'.")

        logger.info(f"  Positive: {n_positive} ({100*n_positive/len(df):.2f}%)")
        logger.info(f"  Negative: {n_negative} ({100*n_negative/len(df):.2f}%)")

        return df

    def _three_way_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("\n=== Three-way Stratified Split ===")

        train_df, temp_df = train_test_split(
            df, train_size=self.train_ratio, stratify=df['label'],
            random_state=self.seed, shuffle=True,
        )
        temp_internal_ratio = self.internal_val_ratio / (self.internal_val_ratio + self.external_test_ratio)
        internal_val_df, external_test_df = train_test_split(
            temp_df, train_size=temp_internal_ratio, stratify=temp_df['label'],
            random_state=self.seed, shuffle=True,
        )

        for name, split_df in [("Train", train_df), ("Internal Val", internal_val_df),
                               ("External Test", external_test_df)]:
            dist = split_df['label'].value_counts().sort_index()
            logger.info(f"  {name}: n={len(split_df)}, "
                        f"neg={dist.get(0, 0)}, pos={dist.get(1, 0)}")

        return train_df, internal_val_df, external_test_df

    def _two_way_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("\n=== Two-way Stratified Split (80/20) ===")

        train_df, internal_val_df = train_test_split(
            df, train_size=0.8, stratify=df['label'],
            random_state=self.seed, shuffle=True,
        )
        for name, split_df in [("Train", train_df), ("Internal Val", internal_val_df)]:
            dist = split_df['label'].value_counts().sort_index()
            logger.info(f"  {name}: n={len(split_df)}, "
                        f"neg={dist.get(0, 0)}, pos={dist.get(1, 0)}")

        return train_df, internal_val_df

    def prepare_data(self):
        """Load, split, impute, and scale features. Returns 9-tuple."""
        logger.info("\n" + "="*80)
        logger.info(f"LOADING: {self.target_disease} vs rest")
        logger.info("="*80)

        if self.two_csv_mode:
            logger.info("Mode: TWO-CSV")
        else:
            logger.info("Mode: SINGLE-CSV (three-way split)")

        logger.info(f"Loading training CSV: {self.train_csv}")
        df_train_full = pd.read_csv(self.train_csv)
        n_train_loaded = len(df_train_full)
        logger.info(f"Loaded {n_train_loaded} rows")

        df_train_full_filtered = self._filter_labeled_patients(df_train_full)
        n_train_filtered = len(df_train_full_filtered)
        df_train_full_filtered = self._create_binary_labels(df_train_full_filtered)

        missing_cols = [c for c in self.measurement_cols if c not in df_train_full_filtered.columns]
        if missing_cols:
            raise ValueError(
                f"Missing measurement columns in training CSV "
                f"(requested {len(self.measurement_cols)} features): {missing_cols}."
            )
        logger.info(f"All {len(self.measurement_cols)} measurement features found")

        n_test_loaded = None
        n_test_filtered = None
        if self.two_csv_mode:
            train_df, internal_val_df = self._two_way_split(df_train_full_filtered)

            logger.info(f"Loading external test CSV: {self.external_test_csv}")
            df_external_test_full = pd.read_csv(self.external_test_csv)
            n_test_loaded = len(df_external_test_full)
            logger.info(f"Loaded {n_test_loaded} rows")

            missing_ext = [c for c in self.measurement_cols if c not in df_external_test_full.columns]
            if missing_ext:
                raise ValueError(f"Missing measurement columns in external test CSV: {missing_ext}")

            df_external_test_filtered = self._filter_labeled_patients(df_external_test_full)
            n_test_filtered = len(df_external_test_filtered)
            external_test_df = self._create_binary_labels(df_external_test_filtered)
        else:
            train_df, internal_val_df, external_test_df = self._three_way_split(df_train_full_filtered)

        logger.info(f"\n=== Row Filter Summary (target={self.target_disease}) ===")
        logger.info(f"  Train CSV loaded rows:      {n_train_loaded}")
        logger.info(f"  Train after row-filter:     {n_train_filtered}")
        if self.two_csv_mode:
            logger.info(f"  Test  CSV loaded rows:      {n_test_loaded}")
            logger.info(f"  Test  after row-filter:     {n_test_filtered}")
        logger.info(f"  Train split rows:           {len(train_df)}")
        logger.info(f"  Internal-val split rows:    {len(internal_val_df)}")
        logger.info(f"  External-test rows:         {len(external_test_df)}")

        if 'eid_18545' in train_df.columns:
            eid_col = 'eid_18545'
        elif 'patient_id' in train_df.columns:
            eid_col = 'patient_id'
        else:
            raise ValueError("No EID column found (eid_18545 or patient_id)")

        X_train_raw = train_df[self.measurement_cols].values.astype(np.float32)
        y_train = train_df['label'].values.astype(np.int32)
        eids_train = train_df[eid_col].values.astype(int)

        X_iv_raw = internal_val_df[self.measurement_cols].values.astype(np.float32)
        y_internal_val = internal_val_df['label'].values.astype(np.int32)
        eids_internal_val = internal_val_df[eid_col].values.astype(int)

        X_et_raw = external_test_df[self.measurement_cols].values.astype(np.float32)
        y_external_test = external_test_df['label'].values.astype(np.int32)
        eids_external_test = external_test_df[eid_col].values.astype(int)

        self.measurement_scaler = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ])
        self.X_train = self.measurement_scaler.fit_transform(X_train_raw)
        self.X_internal_val = self.measurement_scaler.transform(X_iv_raw)
        self.X_external_test = self.measurement_scaler.transform(X_et_raw)

        self.y_train = y_train
        self.y_internal_val = y_internal_val
        self.y_external_test = y_external_test
        self.eids_train = eids_train
        self.eids_internal_val = eids_internal_val
        self.eids_external_test = eids_external_test

        logger.info(f"\nX_train: {self.X_train.shape}, X_iv: {self.X_internal_val.shape}, "
                    f"X_et: {self.X_external_test.shape}")

        return (self.X_train, self.y_train, self.X_internal_val, self.y_internal_val,
                self.X_external_test, self.y_external_test,
                self.eids_train, self.eids_internal_val, self.eids_external_test)

    def get_feature_names(self):
        return self.feature_cols

    def get_scalers(self):
        return self.measurement_scaler
