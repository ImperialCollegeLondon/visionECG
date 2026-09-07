"""Load EchoNext-Mini ECG and tabular features."""

import logging
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def _patch_transformer_for_sklearn_compat(transformer) -> None:
    if not hasattr(transformer, "named_steps"):
        return
    imputer = transformer.named_steps.get("impute")
    if imputer is not None and not hasattr(imputer, "keep_empty_features"):
        imputer.keep_empty_features = False


class ECGEchoNextMiniDataset(Dataset):
    """Load each ECG and seven tabular features."""

    def __init__(
        self,
        csv_path: str,
        preprocessed_ecg_path: str,
        ecg_phenotypes_path: str,
        label_column: str = "diseased",
        threshold: Optional[float] = None,
        threshold_direction: str = "less_than",
        tabular_transform_path: Optional[str] = None,
        trained_transformer: Optional[Pipeline] = None,
        is_train: bool = True,
        inference_mode: bool = False,
    ):
        self.csv_path = csv_path
        self.preprocessed_ecg_path = preprocessed_ecg_path
        self.ecg_phenotypes_path = ecg_phenotypes_path
        self.label_column = label_column
        self.threshold_direction = threshold_direction
        self.is_train = is_train
        self.inference_mode = inference_mode

        self.ecg_num_leads = 12
        self.ecg_num_timepoints = 2500

        if tabular_transform_path is not None:
            self.use_reference_transformer = True
            self.tabular_transformer = joblib.load(tabular_transform_path)
            _patch_transformer_for_sklearn_compat(self.tabular_transformer)
        elif trained_transformer is not None:
            self.use_reference_transformer = False
            self.tabular_transformer = trained_transformer
        else:
            self.use_reference_transformer = False
            self.tabular_transformer = None

        self.ecg_data = torch.load(preprocessed_ecg_path)

        sample_key = next(iter(self.ecg_data.keys()))
        sample_ecg = self.ecg_data[sample_key]
        assert sample_ecg.shape == (self.ecg_num_leads, self.ecg_num_timepoints), \
            f"Expected shape (12, 2500), got {sample_ecg.shape}"

        self.df_morphology = pd.read_csv(ecg_phenotypes_path)
        self.df = pd.read_csv(csv_path)

        if "eid_18545" not in self.df.columns:
            raise ValueError("'eid_18545' column not found in CSV")
        if "eid_40616" not in self.df.columns:
            raise ValueError("'eid_40616' column not found in CSV")
        self.eids = self.df["eid_18545"].values.astype(int)
        self.eids_40616 = self.df["eid_40616"].values.astype(int)

        self.eid_to_instance = self._build_eid_instance_mapping()
        self.tabular_lookup = self._build_tabular_lookup()

        tabular_7_raw = np.array([
            self.tabular_lookup.get(
                (eid_40616, self.eid_to_instance.get(eid_40616)),
                np.full(7, np.nan, dtype=np.float32),
            ) if self.eid_to_instance.get(eid_40616) is not None
            else np.full(7, np.nan, dtype=np.float32)
            for eid_40616 in self.eids_40616
        ], dtype=np.float32)

        if self.inference_mode:
            self.disease_labels = np.zeros(len(self.df), dtype=np.float32)
        else:
            if label_column not in self.df.columns:
                raise ValueError(f"Label column '{label_column}' not found. Available: {list(self.df.columns)}")
            raw_labels = self.df[label_column].values.astype(np.float32)

            if threshold is not None:
                if self.threshold_direction not in ("less_than", "greater_than"):
                    raise ValueError(f"Invalid threshold_direction: {self.threshold_direction}")
                if self.threshold_direction == "less_than":
                    self.disease_labels = (raw_labels < threshold).astype(np.float32)
                else:
                    self.disease_labels = (raw_labels > threshold).astype(np.float32)
                if np.isnan(raw_labels).any():
                    self.disease_labels[np.isnan(raw_labels)] = 0.0
            else:
                self.disease_labels = raw_labels
                unique_labels = np.unique(self.disease_labels[~np.isnan(self.disease_labels)])
                if not np.all(np.isin(unique_labels, [0, 1])):
                    raise ValueError(f"Labels must be 0 or 1 without a threshold; found {unique_labels}")
                if np.isnan(self.disease_labels).any():
                    self.disease_labels = np.nan_to_num(self.disease_labels, nan=0.0)

        if self.use_reference_transformer:
            self.tabular_features = self._apply_reference_preprocessing(tabular_7_raw)
        else:
            self.tabular_features = self._apply_local_preprocessing(tabular_7_raw, is_train)

    def _build_eid_instance_mapping(self) -> Dict[int, Optional[int]]:
        eid_to_instance: Dict[int, Optional[int]] = {}
        eid_instances: Dict[int, list] = {}
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
        tabular_lookup: Dict = {}
        for _, morph_row in self.df_morphology.iterrows():
            eid = int(morph_row["eid_40616"])
            instance = int(morph_row["Instance"])
            demo_matches = self.df[self.df["eid_40616"] == eid]
            if len(demo_matches) == 0:
                continue
            demo_row = demo_matches.iloc[0]

            sex_val = demo_row.get("Sex", demo_row.get("sex", np.nan))
            sex = 0.0 if pd.isna(sex_val) else float(sex_val)
            age = float(demo_row.get("Age", demo_row.get("age_at_MRI", np.nan)))
            ventricular_rate = float(morph_row.get("VentricularRate", np.nan))
            pp_interval = float(morph_row.get("PPInterval", np.nan))
            pr_interval = float(morph_row.get("PQInterval", np.nan))
            qrs_duration = float(morph_row.get("QRSDuration", np.nan))
            qt_corrected = float(morph_row.get("QTCInterval", np.nan))
            atrial_rate = 60000.0 / pp_interval if pp_interval > 0 and not np.isnan(pp_interval) else 0.0

            tabular_lookup[(eid, instance)] = np.array(
                [sex, age, ventricular_rate, atrial_rate, pr_interval, qrs_duration, qt_corrected],
                dtype=np.float32,
            )
        return tabular_lookup

    def _apply_reference_preprocessing(self, tabular_7_raw: np.ndarray) -> np.ndarray:
        sex = np.nan_to_num(tabular_7_raw[:, 0].copy(), nan=0.0)
        age_at_ecg = tabular_7_raw[:, 1].copy()
        ventricular_rate = tabular_7_raw[:, 2].copy()
        atrial_rate = np.nan_to_num(tabular_7_raw[:, 3].copy(), nan=0.0)
        pr_interval = np.nan_to_num(tabular_7_raw[:, 4].copy(), nan=0.0)
        qrs_duration = tabular_7_raw[:, 5].copy()
        qt_corrected = tabular_7_raw[:, 6].copy()

        float_features = np.stack(
            [age_at_ecg, ventricular_rate, atrial_rate, pr_interval, qrs_duration, qt_corrected],
            axis=1,
        )
        float_features_scaled = self.tabular_transformer.transform(float_features)
        tabular_7_clean = np.concatenate([sex.reshape(-1, 1), float_features_scaled], axis=1)

        if np.isnan(tabular_7_clean).any():
            nan_cols = np.isnan(tabular_7_clean).sum(axis=0)
            raise ValueError(f"NaN in preprocessed tabular features (per-column count: {nan_cols})")
        return tabular_7_clean.astype(np.float32)

    def _apply_local_preprocessing(self, tabular_7_raw: np.ndarray, is_train: bool) -> np.ndarray:
        sex = np.nan_to_num(tabular_7_raw[:, 0].copy(), nan=0.0)
        atrial_rate = np.nan_to_num(tabular_7_raw[:, 3].copy(), nan=0.0)
        pr_interval = np.nan_to_num(tabular_7_raw[:, 4].copy(), nan=0.0)

        float_features = np.stack(
            [tabular_7_raw[:, 1], tabular_7_raw[:, 2], atrial_rate,
             pr_interval, tabular_7_raw[:, 5], tabular_7_raw[:, 6]],
            axis=1,
        )

        if is_train:
            self.tabular_transformer = Pipeline([
                ("scale", StandardScaler()),
                ("impute", SimpleImputer(strategy="median")),
            ])
            float_features_scaled = self.tabular_transformer.fit_transform(float_features)
        else:
            if self.tabular_transformer is None:
                raise ValueError("A fitted transformer must be provided for validation/test in local mode")
            float_features_scaled = self.tabular_transformer.transform(float_features)

        tabular_7_clean = np.concatenate([sex.reshape(-1, 1), float_features_scaled], axis=1)
        if np.isnan(tabular_7_clean).any():
            nan_cols = np.isnan(tabular_7_clean).sum(axis=0)
            raise ValueError(f"NaN in preprocessed tabular features (per-column count: {nan_cols})")
        return tabular_7_clean.astype(np.float32)

    def _load_ecg(self, eid: int) -> torch.Tensor:
        instance = self.eid_to_instance.get(eid)
        if instance is None:
            return torch.zeros((1, self.ecg_num_timepoints, self.ecg_num_leads), dtype=torch.float32)
        key = (eid, instance)
        if key not in self.ecg_data:
            return torch.zeros((1, self.ecg_num_timepoints, self.ecg_num_leads), dtype=torch.float32)
        ecg = self.ecg_data[key]
        return ecg.T.unsqueeze(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        eid = self.eids[idx]
        eid_40616 = self.eids_40616[idx]
        return {
            "ecg_raw": self._load_ecg(eid_40616),
            "tabular_7": torch.FloatTensor(self.tabular_features[idx]),
            "disease_label": torch.tensor(self.disease_labels[idx], dtype=torch.float32),
            "eid": eid,
        }

    def get_tabular_transformer(self):
        return self.tabular_transformer

    def get_eids(self):
        return self.eids


class ECGEchoNextMiniDataModule:
    """Train/val dataset + dataloader manager for EchoNext-Mini."""

    def __init__(
        self,
        train_csv_path: str,
        val_csv_path: str,
        preprocessed_ecg_path: str,
        ecg_phenotypes_path: str,
        label_column: str = "diseased",
        threshold: Optional[float] = None,
        threshold_direction: str = "less_than",
        tabular_transform_path: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        inference_mode: bool = False,
    ):
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.preprocessed_ecg_path = preprocessed_ecg_path
        self.ecg_phenotypes_path = ecg_phenotypes_path
        self.label_column = label_column
        self.threshold = threshold
        self.threshold_direction = threshold_direction
        self.tabular_transform_path = tabular_transform_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.inference_mode = inference_mode
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = "fit"):
        if stage == "fit" or stage is None:
            self.train_dataset = ECGEchoNextMiniDataset(
                csv_path=self.train_csv_path,
                preprocessed_ecg_path=self.preprocessed_ecg_path,
                ecg_phenotypes_path=self.ecg_phenotypes_path,
                label_column=self.label_column,
                threshold=self.threshold,
                threshold_direction=self.threshold_direction,
                tabular_transform_path=self.tabular_transform_path,
                trained_transformer=None,
                is_train=True,
                inference_mode=self.inference_mode,
            )

            trained_transformer = (
                self.train_dataset.get_tabular_transformer()
                if self.tabular_transform_path is None else None
            )

            self.val_dataset = ECGEchoNextMiniDataset(
                csv_path=self.val_csv_path,
                preprocessed_ecg_path=self.preprocessed_ecg_path,
                ecg_phenotypes_path=self.ecg_phenotypes_path,
                label_column=self.label_column,
                threshold=self.threshold,
                threshold_direction=self.threshold_direction,
                tabular_transform_path=self.tabular_transform_path,
                trained_transformer=trained_transformer,
                is_train=False,
                inference_mode=self.inference_mode,
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Training dataset not initialized. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("Validation dataset not initialized. Call setup('fit') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
