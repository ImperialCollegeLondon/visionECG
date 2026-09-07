#!/usr/bin/env python

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParametersLassoDataModule:
    """Load ECG parameters and binary labels."""

    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        ecg_parameters_path: str,
        label_column: str,
        parameter_cols: List[str],
        atrial_features: List[str],
        threshold: Optional[float] = None,
        threshold_direction: str = 'less_than',
    ):
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.ecg_parameters_path = ecg_parameters_path
        self.label_column = label_column
        self.threshold = threshold
        self.threshold_direction = threshold_direction

        self.parameter_cols = list(parameter_cols)
        # Fill missing atrial features with zero for AF.
        self.atrial_features = list(atrial_features)

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.eids_train = None
        self.eids_val = None
        self.parameter_scaler = None

    def _build_eid_instance_mapping(self, df_parameters: pd.DataFrame) -> Dict[int, Optional[int]]:
        eid_instances: Dict[int, list] = {}
        for _, row in df_parameters.iterrows():
            eid = int(row['eid_40616'])
            eid_instances.setdefault(eid, []).append(int(row['Instance']))
        eid_to_instance = {}
        for eid, instances in eid_instances.items():
            if 2 in instances:
                eid_to_instance[eid] = 2
            elif 3 in instances:
                eid_to_instance[eid] = 3
            else:
                eid_to_instance[eid] = min(instances)
        return eid_to_instance

    def _build_parameters_lookup(self, df_parameters: pd.DataFrame) -> Dict:
        lookup: Dict = {}
        df_p = df_parameters.copy()
        for col in self.atrial_features:
            if col in df_p.columns:
                df_p[col] = df_p[col].fillna(0)
        for _, row in df_p.iterrows():
            eid = int(row['eid_40616'])
            instance = int(row['Instance'])
            values = [float(row[c]) if pd.notna(row[c]) else np.nan for c in self.parameter_cols]
            lookup[(eid, instance)] = np.array(values, dtype=np.float32)
        logger.info(f"Built parameters lookup with {len(lookup)} (eid, instance) pairs")
        return lookup

    def _process_labels(self, df: pd.DataFrame, dataset_name: str) -> np.ndarray:
        if self.label_column not in df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in {dataset_name} CSV.")
        raw = df[self.label_column].values.astype(np.float32)

        if self.threshold is not None:
            if self.threshold_direction == 'less_than':
                labels = (raw <= self.threshold).astype(np.float32)
            elif self.threshold_direction == 'greater_than':
                labels = (raw >= self.threshold).astype(np.float32)
            else:
                raise ValueError(f"Invalid threshold_direction: {self.threshold_direction}")
            if np.isnan(raw).any():
                logger.warning(f"{dataset_name}: {np.isnan(raw).sum()} NaN labels → 0 (healthy)")
                labels[np.isnan(raw)] = 0.0
        else:
            unique = np.unique(raw[~np.isnan(raw)])
            if not np.all(np.isin(unique, [0, 1])):
                raise ValueError(f"Non-binary labels {unique} without --threshold set")
            labels = np.nan_to_num(raw, nan=0.0)

        n_pos = int(np.sum(labels == 1))
        n_neg = int(np.sum(labels == 0))
        logger.info(f"{dataset_name}: healthy={n_neg}, diseased={n_pos} "
                    f"(pos ratio={n_pos/(n_pos+n_neg):.2%})")
        return labels

    def _load_parameters(self, df: pd.DataFrame, eid_to_instance: Dict,
                         lookup: Dict, dataset_name: str) -> np.ndarray:
        if 'eid_40616' not in df.columns:
            raise ValueError("'eid_40616' column not found in CSV")
        eids = df['eid_40616'].values.astype(int)
        n_features = len(self.parameter_cols)
        rows, available = [], 0
        for eid in eids:
            instance = eid_to_instance.get(int(eid))
            if instance is not None:
                arr = lookup.get((int(eid), instance), np.full(n_features, np.nan, dtype=np.float32))
                if not np.all(np.isnan(arr)):
                    available += 1
            else:
                arr = np.full(n_features, np.nan, dtype=np.float32)
            rows.append(arr)
        X = np.array(rows, dtype=np.float32)
        logger.info(f"{dataset_name}: parameter coverage {available}/{len(eids)} "
                    f"({100*available/len(eids):.1f}%)")
        return X

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df_parameters = pd.read_csv(self.ecg_parameters_path)
        logger.info(f"Loaded {len(df_parameters)} ECG parameter records")

        eid_to_instance = self._build_eid_instance_mapping(df_parameters)
        lookup = self._build_parameters_lookup(df_parameters)

        df_train = pd.read_csv(self.train_csv)
        if 'eid_18545' not in df_train.columns:
            raise ValueError("'eid_18545' column not found in training CSV")
        self.eids_train = df_train['eid_18545'].values.astype(int)
        self.y_train = self._process_labels(df_train, 'train')
        X_train_raw = self._load_parameters(df_train, eid_to_instance, lookup, 'train')

        df_val = pd.read_csv(self.val_csv)
        if 'eid_18545' not in df_val.columns:
            raise ValueError("'eid_18545' column not found in validation CSV")
        self.eids_val = df_val['eid_18545'].values.astype(int)
        self.y_val = self._process_labels(df_val, 'val')
        X_val_raw = self._load_parameters(df_val, eid_to_instance, lookup, 'val')

        self.parameter_scaler = Pipeline([
            ('scale', StandardScaler()),
            ('impute', SimpleImputer(strategy='median')),
        ])
        self.X_train = self.parameter_scaler.fit_transform(X_train_raw)
        self.X_val = self.parameter_scaler.transform(X_val_raw)

        logger.info(f"Shapes: X_train={self.X_train.shape}, X_val={self.X_val.shape}")
        return self.X_train, self.y_train, self.X_val, self.y_val, self.eids_train, self.eids_val

    def get_feature_names(self):
        return self.parameter_cols

    def get_scaler(self):
        return self.parameter_scaler
