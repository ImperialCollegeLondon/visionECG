#!/usr/bin/env python
import logging
import os
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils_label_generation import (
    create_binary_labels_batch,
    calculate_pos_weight,
    validate_labels,
    get_class_distribution,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _patch_transformer_for_sklearn_compat(transformer) -> None:
    """Restore legacy SimpleImputer compatibility."""
    if not hasattr(transformer, "named_steps"):
        return
    for step in transformer.named_steps.values():
        if isinstance(step, SimpleImputer) and not hasattr(step, "keep_empty_features"):
            step.keep_empty_features = False


class ECGFineTuneDataset(Dataset):
    """Load ECG, tabular features, and binary labels."""

    def __init__(
        self,
        csv_path: str,
        preprocessed_ecg_path: str,
        ecg_phenotypes_path: str,
        label_column: str,
        threshold: float,
        threshold_direction: str,
        tabular_transform_path: Optional[str] = None,
        trained_transformer: Optional[Pipeline] = None,
        sex_filter: Optional[int] = None,
        is_train: bool = True,
        inference_mode: bool = False,
    ):
        self.csv_path = csv_path
        self.preprocessed_ecg_path = preprocessed_ecg_path
        self.ecg_phenotypes_path = ecg_phenotypes_path
        self.label_column = label_column
        self.threshold = threshold
        self.threshold_direction = threshold_direction
        self.sex_filter = sex_filter
        self.is_train = is_train
        self.inference_mode = inference_mode

        self.ecg_num_leads = 12
        self.ecg_num_timepoints = 2500

        logger.info(f"Label column: {label_column} ({threshold_direction} {threshold})")
        if sex_filter is not None:
            sex_name = 'male' if sex_filter == 1 else 'female'
            logger.info(f"Sex filter: {sex_name} ({sex_filter})")

        if tabular_transform_path is not None:
            self.use_reference_transformer = True
            self.tabular_transformer = joblib.load(tabular_transform_path)
            _patch_transformer_for_sklearn_compat(self.tabular_transformer)
            logger.info(f"Loaded reference tabular transformer: {tabular_transform_path}")
        elif trained_transformer is not None:
            self.use_reference_transformer = False
            self.tabular_transformer = trained_transformer
        else:
            self.use_reference_transformer = False
            self.tabular_transformer = None

        logger.info(f"Loading preprocessed ECG: {preprocessed_ecg_path}")
        self.ecg_data = torch.load(preprocessed_ecg_path)
        logger.info(f"Loaded {len(self.ecg_data)} preprocessed ECG samples")

        sample_key = list(self.ecg_data.keys())[0]
        sample_ecg = self.ecg_data[sample_key]
        assert sample_ecg.shape == (self.ecg_num_leads, self.ecg_num_timepoints), \
            f"Expected ECG shape (12, 2500), got {sample_ecg.shape}"

        self.df_morphology = pd.read_csv(ecg_phenotypes_path)
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} patients")

        if sex_filter is not None:
            original_len = len(self.df)
            self.df = self.df[self.df['Sex'] == sex_filter].copy()
            logger.info(f"Applied sex filter: {original_len} -> {len(self.df)} patients")
            if len(self.df) == 0:
                raise ValueError(f"No patients remaining after sex filter (sex={sex_filter})")

        for required in ('eid_18545', 'eid_40616', label_column):
            if required not in self.df.columns:
                raise ValueError(f"Required column '{required}' not found")

        self.eids = self.df['eid_18545'].values.astype(int)
        self.eids_40616 = self.df['eid_40616'].values.astype(int)

        self.eid_to_instance = self._build_eid_instance_mapping()
        self.tabular_lookup = self._build_tabular_lookup()

        available_count = sum(1 for eid in self.eids_40616
                              if self.eid_to_instance.get(eid) is not None)
        logger.info(f"ECG coverage: {available_count}/{len(self.eids_40616)} "
                    f"({100*available_count/len(self.eids_40616):.1f}%)")

        tabular_7_raw = []
        for eid_40616 in self.eids_40616:
            instance = self.eid_to_instance.get(eid_40616)
            if instance is not None:
                key = (eid_40616, instance)
                features = self.tabular_lookup.get(key, np.full(7, np.nan, dtype=np.float32))
            else:
                features = np.full(7, np.nan, dtype=np.float32)
            tabular_7_raw.append(features)
        tabular_7_raw = np.array(tabular_7_raw, dtype=np.float32)

        self._generate_binary_labels()

        if self.use_reference_transformer:
            self.tabular_features = self._apply_reference_preprocessing(tabular_7_raw)
        else:
            self.tabular_features = self._apply_local_preprocessing(tabular_7_raw, is_train)

    def _generate_binary_labels(self):
        raw_values = self.df[self.label_column].values.astype(np.float32)

        valid_values = raw_values[~np.isnan(raw_values)]
        if len(valid_values) > 0:
            logger.info(f"Measurement range: [{np.min(valid_values):.2f}, "
                        f"{np.max(valid_values):.2f}] "
                        f"mean±std: {np.mean(valid_values):.2f}±{np.std(valid_values):.2f}")

        self.disease_labels = create_binary_labels_batch(
            raw_values, self.threshold, self.threshold_direction
        )

        nan_mask = np.isnan(self.disease_labels)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logger.warning(f"Removing {n_nan} samples with NaN labels")
            valid_mask = ~nan_mask
            self.df = self.df[valid_mask].reset_index(drop=True)
            self.eids = self.eids[valid_mask]
            self.eids_40616 = self.eids_40616[valid_mask]
            self.disease_labels = self.disease_labels[valid_mask]

        if not self.inference_mode:
            is_valid, error_msg = validate_labels(self.disease_labels, min_samples_per_class=10)
            if not is_valid:
                raise ValueError(f"Label validation failed: {error_msg}")

        num_healthy, num_diseased, imbalance_ratio = get_class_distribution(self.disease_labels)
        logger.info(f"Class distribution: healthy={num_healthy} "
                    f"({100*num_healthy/len(self.disease_labels):.1f}%), "
                    f"diseased={num_diseased} "
                    f"({100*num_diseased/len(self.disease_labels):.1f}%), "
                    f"ratio={imbalance_ratio:.2f}:1")

        pos_weight = calculate_pos_weight(self.disease_labels)
        logger.info(f"pos_weight: {pos_weight.item():.2f}")

    def _build_eid_instance_mapping(self) -> Dict[int, Optional[int]]:
        eid_to_instance = {}
        eid_instances = {}
        for eid, instance in self.ecg_data.keys():
            eid_instances.setdefault(eid, []).append(instance)

        for eid in self.eids_40616:
            if eid in eid_instances:
                instances = eid_instances[eid]
                if 2 in instances:
                    eid_to_instance[eid] = 2
                elif 3 in instances:
                    eid_to_instance[eid] = 3
                else:
                    eid_to_instance[eid] = min(instances)
            else:
                eid_to_instance[eid] = None
        return eid_to_instance

    def _build_tabular_lookup(self) -> Dict:
        tabular_lookup = {}

        for _, morph_row in self.df_morphology.iterrows():
            eid = int(morph_row['eid_40616'])
            instance = int(morph_row['Instance'])

            demo_matches = self.df[self.df['eid_40616'] == eid]
            if len(demo_matches) == 0:
                continue
            demo_row = demo_matches.iloc[0]

            sex_val = demo_row.get('Sex', demo_row.get('sex', np.nan))
            sex = 0.0 if pd.isna(sex_val) else float(sex_val)
            age = float(demo_row.get('Age', demo_row.get('age_at_MRI', np.nan)))

            ventricular_rate = float(morph_row.get('VentricularRate', np.nan))
            pp_interval = float(morph_row.get('PPInterval', np.nan))
            pr_interval = float(morph_row.get('PQInterval', np.nan))
            qrs_duration = float(morph_row.get('QRSDuration', np.nan))
            qt_corrected = float(morph_row.get('QTCInterval', np.nan))

            if pp_interval > 0 and not np.isnan(pp_interval):
                atrial_rate = 60000.0 / pp_interval
            else:
                atrial_rate = 0.0

            tabular_lookup[(eid, instance)] = np.array([
                sex, age, ventricular_rate, atrial_rate,
                pr_interval, qrs_duration, qt_corrected,
            ], dtype=np.float32)

        return tabular_lookup

    def _apply_reference_preprocessing(self, tabular_7_raw: np.ndarray) -> np.ndarray:
        sex = np.nan_to_num(tabular_7_raw[:, 0].copy(), nan=0.0)
        age_at_ecg = tabular_7_raw[:, 1].copy()
        ventricular_rate = tabular_7_raw[:, 2].copy()
        atrial_rate = np.nan_to_num(tabular_7_raw[:, 3].copy(), nan=0.0)
        pr_interval = np.nan_to_num(tabular_7_raw[:, 4].copy(), nan=0.0)
        qrs_duration = tabular_7_raw[:, 5].copy()
        qt_corrected = tabular_7_raw[:, 6].copy()

        float_features = np.stack([
            age_at_ecg, ventricular_rate, atrial_rate,
            pr_interval, qrs_duration, qt_corrected,
        ], axis=1)

        float_features_scaled = self.tabular_transformer.transform(float_features)

        tabular_7_clean = np.concatenate(
            [sex.reshape(-1, 1), float_features_scaled], axis=1
        )

        if np.isnan(tabular_7_clean).any():
            nan_count = int(np.isnan(tabular_7_clean).sum())
            raise ValueError(f"Tabular features contain {nan_count} NaN values")

        return tabular_7_clean.astype(np.float32)

    def _apply_local_preprocessing(self, tabular_7_raw: np.ndarray, is_train: bool) -> np.ndarray:
        sex = np.nan_to_num(tabular_7_raw[:, 0].copy(), nan=0.0)
        atrial_rate = np.nan_to_num(tabular_7_raw[:, 3].copy(), nan=0.0)
        pr_interval = np.nan_to_num(tabular_7_raw[:, 4].copy(), nan=0.0)

        float_features = np.stack([
            tabular_7_raw[:, 1],  
            tabular_7_raw[:, 2],  
            atrial_rate,
            pr_interval,
            tabular_7_raw[:, 5], 
            tabular_7_raw[:, 6],  
        ], axis=1)

        if is_train:
            self.tabular_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ])
            float_features_scaled = self.tabular_transformer.fit_transform(float_features)
        else:
            float_features_scaled = self.tabular_transformer.transform(float_features)

        tabular_7_clean = np.concatenate(
            [sex.reshape(-1, 1), float_features_scaled], axis=1
        )

        if np.isnan(tabular_7_clean).any():
            nan_count = int(np.isnan(tabular_7_clean).sum())
            raise ValueError(f"Tabular features contain {nan_count} NaN values")

        return tabular_7_clean.astype(np.float32)

    def _load_ecg(self, eid: int) -> torch.Tensor:
        instance = self.eid_to_instance.get(eid)
        if instance is None:
            return torch.zeros((1, self.ecg_num_timepoints, self.ecg_num_leads),
                               dtype=torch.float32)

        key = (eid, instance)
        if key in self.ecg_data:
            ecg = self.ecg_data[key]  
            return ecg.T.unsqueeze(0)  
        return torch.zeros((1, self.ecg_num_timepoints, self.ecg_num_leads),
                           dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        eid = self.eids[idx]
        eid_40616 = self.eids_40616[idx]
        ecg_preprocessed = self._load_ecg(eid_40616)
        tabular_7 = torch.FloatTensor(self.tabular_features[idx])
        label = torch.tensor(self.disease_labels[idx], dtype=torch.float32)
        return {
            "ecg_raw": ecg_preprocessed,
            "tabular_7": tabular_7,
            "label": label,
            "eid": eid,
        }

    def get_tabular_transformer(self):
        return self.tabular_transformer

    def get_all_labels(self) -> np.ndarray:
        return self.disease_labels

    def get_eids(self):
        return self.eids


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    preprocessed_ecg_path: str,
    ecg_phenotypes_path: str,
    label_column: str,
    threshold: float,
    threshold_direction: str,
    tabular_transform_path: Optional[str] = None,
    sex_filter: Optional[int] = None,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, Pipeline, torch.Tensor]:
    """Build (train_loader, val_loader, tabular_transformer, pos_weight)."""
    train_dataset = ECGFineTuneDataset(
        csv_path=train_csv,
        preprocessed_ecg_path=preprocessed_ecg_path,
        ecg_phenotypes_path=ecg_phenotypes_path,
        label_column=label_column,
        threshold=threshold,
        threshold_direction=threshold_direction,
        tabular_transform_path=tabular_transform_path,
        sex_filter=sex_filter,
        is_train=True,
        inference_mode=False,
    )

    pos_weight = calculate_pos_weight(train_dataset.get_all_labels())
    logger.info(f"Training pos_weight: {pos_weight.item():.2f}")

    trained_transformer = train_dataset.get_tabular_transformer()

    val_dataset = ECGFineTuneDataset(
        csv_path=val_csv,
        preprocessed_ecg_path=preprocessed_ecg_path,
        ecg_phenotypes_path=ecg_phenotypes_path,
        label_column=label_column,
        threshold=threshold,
        threshold_direction=threshold_direction,
        trained_transformer=trained_transformer,
        sex_filter=sex_filter,
        is_train=False,
        inference_mode=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info(f"Dataloaders ready — train={len(train_dataset)} val={len(val_dataset)} "
                f"batch_size={batch_size}")
    return train_loader, val_loader, trained_transformer, pos_weight
