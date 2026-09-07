#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""ECG-parameters loader for one-vs-rest disease classification."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ECGParametersXGBoostDataModule:
    """Load ECG parameters for one-vs-rest classification."""

    def __init__(
        self,
        train_csv: str,
        target_disease: str,
        disease_cols: list,
        filter_groups: list,
        morphology_cols: list,
        ecg_phenotypes_path: str,
        atrial_features: Optional[list] = None,
        external_test_csv: Optional[str] = None,
        train_ratio: float = 0.64,
        internal_val_ratio: float = 0.16,
        external_test_ratio: float = 0.20,
        seed: int = 42,
        virtual_label_map: Optional[dict] = None,
    ):
        self.train_csv = train_csv
        self.target_disease = target_disease
        self.ecg_phenotypes_path = ecg_phenotypes_path
        self.external_test_csv = external_test_csv
        self.train_ratio = train_ratio
        self.internal_val_ratio = internal_val_ratio
        self.external_test_ratio = external_test_ratio
        self.seed = seed
        self.virtual_label_map = dict(virtual_label_map) if virtual_label_map else {}
        self.disease_cols = list(disease_cols)
        self.filter_groups = list(filter_groups)
        self.morphology_cols = list(morphology_cols)
        self.atrial_features = list(atrial_features) if atrial_features else []
        self.feature_cols = self.morphology_cols
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
        self.morphology_scaler = None

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

    def _build_eid_instance_mapping(self, df_morphology: pd.DataFrame) -> Dict[int, Optional[int]]:
        eid_instances: Dict[int, list] = {}
        for _, row in df_morphology.iterrows():
            eid = int(row['eid_40616'])
            instance = int(row['Instance'])
            eid_instances.setdefault(eid, []).append(instance)

        eid_to_instance: Dict[int, int] = {}
        for eid, instances in eid_instances.items():
            if 2 in instances:
                eid_to_instance[eid] = 2
            elif 3 in instances:
                eid_to_instance[eid] = 3
            else:
                eid_to_instance[eid] = min(instances)
        return eid_to_instance

    def _build_morphology_lookup(self, df_morphology: pd.DataFrame) -> Dict:
        morphology_lookup: Dict = {}

        df_morph = df_morphology.copy()
        for col in self.atrial_features:
            if col in df_morph.columns:
                df_morph[col] = df_morph[col].fillna(0)

        for _, row in df_morph.iterrows():
            eid = int(row['eid_40616'])
            instance = int(row['Instance'])
            morph_values = []
            for col in self.morphology_cols:
                val = row[col]
                morph_values.append(float(val) if pd.notna(val) else np.nan)
            morphology_lookup[(eid, instance)] = np.array(morph_values, dtype=np.float32)

        logger.info(f"Built morphology lookup with {len(morphology_lookup)} (eid, instance) pairs")
        return morphology_lookup

    def _load_morphology_features(
        self,
        df: pd.DataFrame,
        eid_to_instance: Dict,
        morphology_lookup: Dict,
        dataset_name: str,
    ) -> np.ndarray:
        if 'eid_40616' not in df.columns:
            raise ValueError("'eid_40616' column not found in CSV")

        eids_40616 = df['eid_40616'].values.astype(int)
        n_features = len(self.morphology_cols)

        morphology_features = []
        available_count = 0
        for eid_40616 in eids_40616:
            instance = eid_to_instance.get(eid_40616)
            if instance is not None:
                key = (eid_40616, instance)
                morph = morphology_lookup.get(key, np.full(n_features, np.nan, dtype=np.float32))
                if not np.all(np.isnan(morph)):
                    available_count += 1
            else:
                morph = np.full(n_features, np.nan, dtype=np.float32)
            morphology_features.append(morph)

        morphology_features = np.array(morphology_features, dtype=np.float32)
        logger.info(f"{dataset_name} morphology coverage: {available_count}/{len(eids_40616)} "
                    f"({100*available_count/max(len(eids_40616),1):.1f}%)")
        return morphology_features

    def prepare_data(self):
        """Return prepared ECG features and dataset splits."""
        logger.info("\n" + "="*80)
        logger.info(f"LOADING: {self.target_disease} vs rest")
        logger.info("="*80)

        if self.two_csv_mode:
            logger.info("Mode: TWO-CSV")
        else:
            logger.info("Mode: SINGLE-CSV (three-way split)")

        logger.info(f"Loading ECG phenotypes: {self.ecg_phenotypes_path}")
        df_morphology = pd.read_csv(self.ecg_phenotypes_path)
        logger.info(f"Loaded {len(df_morphology)} ECG phenotype rows")

        missing_morph = [c for c in self.morphology_cols if c not in df_morphology.columns]
        if missing_morph:
            raise ValueError(
                f"Missing morphology columns in ECG phenotypes CSV "
                f"({len(self.morphology_cols)} requested): {missing_morph}"
            )

        eid_to_instance = self._build_eid_instance_mapping(df_morphology)
        morphology_lookup = self._build_morphology_lookup(df_morphology)

        logger.info(f"Loading training CSV: {self.train_csv}")
        df_train_full = pd.read_csv(self.train_csv)
        n_train_loaded = len(df_train_full)
        logger.info(f"Loaded {n_train_loaded} rows")

        df_train_full_filtered = self._filter_labeled_patients(df_train_full)
        n_train_filtered = len(df_train_full_filtered)
        df_train_full_filtered = self._create_binary_labels(df_train_full_filtered)

        n_test_loaded = None
        n_test_filtered = None
        if self.two_csv_mode:
            train_df, internal_val_df = self._two_way_split(df_train_full_filtered)

            logger.info(f"Loading external test CSV: {self.external_test_csv}")
            df_external_test_full = pd.read_csv(self.external_test_csv)
            n_test_loaded = len(df_external_test_full)
            logger.info(f"Loaded {n_test_loaded} rows")

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

        y_train = train_df['label'].values.astype(np.int32)
        eids_train = train_df[eid_col].values.astype(int)

        y_internal_val = internal_val_df['label'].values.astype(np.int32)
        eids_internal_val = internal_val_df[eid_col].values.astype(int)

        y_external_test = external_test_df['label'].values.astype(np.int32)
        eids_external_test = external_test_df[eid_col].values.astype(int)

        X_train_raw = self._load_morphology_features(
            train_df, eid_to_instance, morphology_lookup, 'train',
        )
        X_iv_raw = self._load_morphology_features(
            internal_val_df, eid_to_instance, morphology_lookup, 'internal_val',
        )
        X_et_raw = self._load_morphology_features(
            external_test_df, eid_to_instance, morphology_lookup, 'external_test',
        )

        self.morphology_scaler = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ])
        self.X_train = self.morphology_scaler.fit_transform(X_train_raw)
        self.X_internal_val = self.morphology_scaler.transform(X_iv_raw)
        self.X_external_test = self.morphology_scaler.transform(X_et_raw)

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
        return self.morphology_scaler
