#!/usr/bin/env python
"""EchoNext-Mini retrain multilabel loader."""

import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


DERIVED_LABEL_SOURCES = {"CM": ("HCM", "DCM")}

logger = logging.getLogger(__name__)


def patch_legacy_sklearn_transformer(transformer: Pipeline) -> Pipeline:
    if hasattr(transformer, "steps"):
        for _, step in transformer.steps:
            if isinstance(step, SimpleImputer) and not hasattr(step, "keep_empty_features"):
                step.keep_empty_features = False
    return transformer


def build_multilabel_target_dataframe(
    df: pd.DataFrame,
    disease_label_columns: List[str],
) -> pd.DataFrame:
    if not disease_label_columns:
        raise ValueError("At least one disease label column must be selected.")

    target_columns: Dict[str, pd.Series] = {}
    missing_columns: List[str] = []

    for col in disease_label_columns:
        if col in DERIVED_LABEL_SOURCES:
            source_cols = list(DERIVED_LABEL_SOURCES[col])
            missing_sources = [source_col for source_col in source_cols if source_col not in df.columns]
            if missing_sources:
                missing_columns.extend([f"{col} source '{source_col}'" for source_col in missing_sources])
                continue
            target_columns[col] = df[source_cols].eq(1).any(axis=1).astype(np.float32)
            continue

        if col not in df.columns:
            missing_columns.append(col)
            continue

        target_columns[col] = (df[col] == 1).astype(np.float32)

    if missing_columns:
        raise ValueError(
            f"Missing disease label columns: {missing_columns}. Available columns: {list(df.columns)}"
        )

    return pd.DataFrame(target_columns, index=df.index, columns=disease_label_columns)


def filter_multilabel_dataframe(
    df: pd.DataFrame,
    disease_label_columns: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, int]:
    label_df = build_multilabel_target_dataframe(df, disease_label_columns)
    valid_mask = label_df.sum(axis=1) > 0
    filtered_df = df.loc[valid_mask].reset_index(drop=True).copy()
    filtered_labels = label_df.loc[valid_mask].reset_index(drop=True).values.astype(np.float32)
    num_excluded = int((~valid_mask).sum())
    return filtered_df, filtered_labels, num_excluded


def stratified_split_multilabel(
    labels: np.ndarray,
    disease_cols: List[str],
    test_size: float = 0.2,
    seed: int = 42,
    split_logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    labels_df = pd.DataFrame(labels, columns=disease_cols)
    composite_labels = labels_df.apply(
        lambda row: "_".join(str(int(row[col])) for col in disease_cols),
        axis=1,
    )
    label_counts = composite_labels.value_counts()
    rare_labels = label_counts[label_counts < 2].index

    if len(rare_labels) > 0:
        most_prevalent = labels_df.sum(axis=0).idxmax()
        stratify_labels = labels_df[most_prevalent].values.astype(int)
        if split_logger is not None:
            split_logger.warning(
                "Found %d rare multilabel combinations; falling back to '%s' stratification.",
                len(rare_labels),
                most_prevalent,
            )
    else:
        stratify_labels = composite_labels.values

    indices = np.arange(len(labels_df))
    try:
        train_indices, val_indices = train_test_split(
            indices, test_size=test_size, stratify=stratify_labels, random_state=seed,
        )
    except ValueError as exc:
        if split_logger is not None:
            split_logger.warning("Stratified split failed (%s); falling back to random split.", exc)
        train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=seed)

    return train_indices, val_indices


def prepare_train_internal_val_split(
    train_csv_path: str,
    disease_label_columns: List[str],
    internal_split_ratio: float,
    seed: int,
    output_dir: str,
    split_logger: Optional[logging.Logger] = None,
) -> Tuple[str, str, Dict[str, int]]:
    if not 0.0 < internal_split_ratio < 1.0:
        raise ValueError(f"internal_split_ratio must be between 0 and 1, got {internal_split_ratio}")

    df_full = pd.read_csv(train_csv_path)
    filtered_df, filtered_labels, num_excluded = filter_multilabel_dataframe(df_full, disease_label_columns)

    if len(df_full) != num_excluded + len(filtered_df):
        raise RuntimeError(
            "Training filtering counts are inconsistent: "
            f"input={len(df_full)}, excluded={num_excluded}, retained={len(filtered_df)}"
        )

    if split_logger is not None:
        split_logger.info("Loaded %d rows from training CSV.", len(df_full))
        split_logger.info("Filtered out %d rows without any positive disease label.", num_excluded)
        split_logger.info("Training pool after filtering: %d rows.", len(filtered_df))

    if filtered_df.empty:
        raise ValueError("No rows remain after multilabel filtering.")

    train_indices, val_indices = stratified_split_multilabel(
        labels=filtered_labels,
        disease_cols=disease_label_columns,
        test_size=1.0 - internal_split_ratio,
        seed=seed,
        split_logger=split_logger,
    )

    train_df = filtered_df.iloc[train_indices].reset_index(drop=True)
    internal_val_df = filtered_df.iloc[val_indices].reset_index(drop=True)

    if len(filtered_df) != len(train_df) + len(internal_val_df):
        raise RuntimeError(
            "Training split counts are inconsistent: "
            f"retained={len(filtered_df)}, train={len(train_df)}, internal_val={len(internal_val_df)}"
        )

    train_split_csv = os.path.join(output_dir, "temp_train_split_mini_retrain_multilabel.csv")
    internal_val_split_csv = os.path.join(output_dir, "temp_internal_val_split_mini_retrain_multilabel.csv")
    train_df.to_csv(train_split_csv, index=False)
    internal_val_df.to_csv(internal_val_split_csv, index=False)

    if split_logger is not None:
        split_logger.info("Saved train split: %s (%d rows)", train_split_csv, len(train_df))
        split_logger.info(
            "Saved internal validation split: %s (%d rows)", internal_val_split_csv, len(internal_val_df)
        )

    split_statistics = {
        "input_rows": int(len(df_full)),
        "excluded_rows": int(num_excluded),
        "retained_rows": int(len(filtered_df)),
        "train_rows": int(len(train_df)),
        "internal_val_rows": int(len(internal_val_df)),
    }
    return train_split_csv, internal_val_split_csv, split_statistics


def calculate_multilabel_pos_weight(labels: np.ndarray) -> torch.Tensor:
    if labels.ndim != 2:
        raise ValueError(f"Expected labels with shape [N, C], got {labels.shape}")

    pos_weight = np.zeros(labels.shape[1], dtype=np.float32)
    for idx in range(labels.shape[1]):
        num_positive = float(np.sum(labels[:, idx] == 1))
        num_negative = float(np.sum(labels[:, idx] == 0))
        pos_weight[idx] = num_negative / (num_positive + 1e-6)
    return torch.tensor(pos_weight, dtype=torch.float32)


class ECGMiniRetrainMultiLabelDataset(Dataset):
    """Dataset yielding preprocessed ECG + 7 tabular features + multilabel targets."""

    def __init__(
        self,
        csv_path: str,
        preprocessed_ecg_path: str,
        ecg_phenotypes_path: str,
        disease_label_columns: List[str],
        tabular_transform_path: Optional[str] = None,
        trained_transformer: Optional[Pipeline] = None,
        is_train: bool = True,
        ecg_data: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
        df_morphology: Optional[pd.DataFrame] = None,
    ):
        self.csv_path = csv_path
        self.preprocessed_ecg_path = preprocessed_ecg_path
        self.ecg_phenotypes_path = ecg_phenotypes_path
        self.is_train = is_train
        self.disease_label_columns = list(disease_label_columns)

        self.ecg_num_leads = 12
        self.ecg_num_timepoints = 2500

        if tabular_transform_path is not None and str(tabular_transform_path).lower() != "none":
            self.tabular_transformer = patch_legacy_sklearn_transformer(joblib.load(tabular_transform_path))
            self.use_reference_transformer = True
            logger.info("Loaded reference tabular transformer from %s", tabular_transform_path)
        elif trained_transformer is not None:
            self.tabular_transformer = trained_transformer
            self.use_reference_transformer = False
            logger.info("Using fitted training tabular transformer.")
        else:
            self.tabular_transformer = None
            self.use_reference_transformer = False
            logger.info("No reference transformer supplied; a local transformer will be fitted on train.")

        if ecg_data is not None:
            self.ecg_data = ecg_data
            logger.info("Using shared preloaded ECG data.")
        else:
            logger.info("Loading preprocessed ECG data from %s", preprocessed_ecg_path)
            self.ecg_data = torch.load(preprocessed_ecg_path)
        sample_key = list(self.ecg_data.keys())[0]
        sample_ecg = self.ecg_data[sample_key]
        if tuple(sample_ecg.shape) != (self.ecg_num_leads, self.ecg_num_timepoints):
            raise ValueError(
                f"Expected ECG shape {(self.ecg_num_leads, self.ecg_num_timepoints)}, got {tuple(sample_ecg.shape)}"
            )

        if df_morphology is not None:
            self.df_morphology = df_morphology
            logger.info("Using shared preloaded ECG morphology data.")
        else:
            logger.info("Loading ECG morphology data from %s", ecg_phenotypes_path)
            self.df_morphology = pd.read_csv(ecg_phenotypes_path)

        logger.info("Loading label CSV from %s", csv_path)
        df_raw = pd.read_csv(csv_path)
        self.input_row_count = int(len(df_raw))
        self.df, self.disease_labels, num_excluded = filter_multilabel_dataframe(
            df_raw, self.disease_label_columns,
        )
        self.excluded_row_count = int(num_excluded)
        self.retained_row_count = int(len(self.df))
        if self.input_row_count != self.excluded_row_count + self.retained_row_count:
            raise RuntimeError(
                "Dataset filtering counts are inconsistent: "
                f"input={self.input_row_count}, excluded={self.excluded_row_count}, "
                f"retained={self.retained_row_count}"
            )
        logger.info("Input dataset size before filtering: %d", self.input_row_count)
        logger.info("Filtered out %d rows without any positive disease label.", num_excluded)
        logger.info("Dataset size after filtering: %d", len(self.df))

        if "eid_18545" not in self.df.columns or "eid_40616" not in self.df.columns:
            raise ValueError("CSV must contain both 'eid_18545' and 'eid_40616' columns.")

        self.eids = self.df["eid_18545"].values.astype(int)
        self.eids_40616 = self.df["eid_40616"].values.astype(int)
        self.eid_to_instance = self._build_eid_instance_mapping()
        self.tabular_lookup = self._build_tabular_lookup()

        tabular_7_raw = []
        missing_features = np.full(7, np.nan, dtype=np.float32)
        for eid_40616 in self.eids_40616:
            instance = self.eid_to_instance.get(eid_40616)
            if instance is None:
                tabular_7_raw.append(missing_features.copy())
                continue
            tabular_7_raw.append(self.tabular_lookup.get((eid_40616, instance), missing_features.copy()))

        tabular_7_raw = np.asarray(tabular_7_raw, dtype=np.float32)
        if self.use_reference_transformer:
            self.tabular_features = self._apply_reference_preprocessing(tabular_7_raw)
        else:
            self.tabular_features = self._apply_local_preprocessing(tabular_7_raw, is_train=is_train)

        self._log_label_distribution()

    def _build_eid_instance_mapping(self) -> Dict[int, Optional[int]]:
        eid_instances: Dict[int, List[int]] = {}
        for eid, instance in self.ecg_data.keys():
            eid_instances.setdefault(eid, []).append(instance)

        eid_to_instance: Dict[int, Optional[int]] = {}
        for eid in self.eids_40616:
            if eid not in eid_instances:
                eid_to_instance[eid] = None
            elif 2 in eid_instances[eid]:
                eid_to_instance[eid] = 2
            elif 3 in eid_instances[eid]:
                eid_to_instance[eid] = 3
            else:
                eid_to_instance[eid] = min(eid_instances[eid])
        return eid_to_instance

    def _build_tabular_lookup(self) -> Dict[Tuple[int, int], np.ndarray]:
        tabular_lookup: Dict[Tuple[int, int], np.ndarray] = {}
        relevant_eids = set(self.eids_40616.tolist())

        demo_lookup: Dict[int, Tuple[float, float]] = {}
        demo_df = self.df[["eid_40616"]].copy()
        demo_df["Sex_value"] = self.df.get("Sex", self.df.get("sex", np.nan))
        demo_df["Age_value"] = self.df.get("Age", self.df.get("age_at_MRI", np.nan))
        demo_df = demo_df.drop_duplicates(subset=["eid_40616"])
        for row in demo_df.itertuples(index=False):
            sex = 0.0 if pd.isna(row.Sex_value) else float(row.Sex_value)
            age = float(row.Age_value) if pd.notna(row.Age_value) else np.nan
            demo_lookup[int(row.eid_40616)] = (sex, age)

        morph_df = self.df_morphology[self.df_morphology["eid_40616"].isin(relevant_eids)]
        for morph_row in morph_df.itertuples(index=False):
            eid = int(morph_row.eid_40616)
            instance = int(morph_row.Instance)
            demo_values = demo_lookup.get(eid)
            if demo_values is None:
                continue

            sex, age = demo_values
            ventricular_rate = float(morph_row.VentricularRate) if pd.notna(morph_row.VentricularRate) else np.nan
            pp_interval = float(morph_row.PPInterval) if pd.notna(morph_row.PPInterval) else np.nan
            pr_interval = float(morph_row.PQInterval) if pd.notna(morph_row.PQInterval) else np.nan
            qrs_duration = float(morph_row.QRSDuration) if pd.notna(morph_row.QRSDuration) else np.nan
            qt_corrected = float(morph_row.QTCInterval) if pd.notna(morph_row.QTCInterval) else np.nan

            if pp_interval > 0 and not np.isnan(pp_interval):
                atrial_rate = 60000.0 / pp_interval
            else:
                atrial_rate = 0.0

            tabular_lookup[(eid, instance)] = np.array(
                [sex, age, ventricular_rate, atrial_rate, pr_interval, qrs_duration, qt_corrected],
                dtype=np.float32,
            )

        logger.info("Built 7-feature tabular lookup with %d entries.", len(tabular_lookup))
        return tabular_lookup

    def _apply_reference_preprocessing(self, tabular_7_raw: np.ndarray) -> np.ndarray:
        sex = np.nan_to_num(tabular_7_raw[:, 0].copy(), nan=0.0)
        age = tabular_7_raw[:, 1].copy()
        ventricular_rate = tabular_7_raw[:, 2].copy()
        atrial_rate = np.nan_to_num(tabular_7_raw[:, 3].copy(), nan=0.0)
        pr_interval = np.nan_to_num(tabular_7_raw[:, 4].copy(), nan=0.0)
        qrs_duration = tabular_7_raw[:, 5].copy()
        qt_corrected = tabular_7_raw[:, 6].copy()

        float_features = np.stack(
            [age, ventricular_rate, atrial_rate, pr_interval, qrs_duration, qt_corrected], axis=1,
        )
        float_features_scaled = self.tabular_transformer.transform(float_features)
        tabular_clean = np.concatenate([sex.reshape(-1, 1), float_features_scaled], axis=1)
        if np.isnan(tabular_clean).any():
            raise ValueError("Reference-preprocessed tabular features still contain NaNs.")
        return tabular_clean.astype(np.float32)

    def _apply_local_preprocessing(self, tabular_7_raw: np.ndarray, is_train: bool) -> np.ndarray:
        sex = np.nan_to_num(tabular_7_raw[:, 0].copy(), nan=0.0)
        atrial_rate = np.nan_to_num(tabular_7_raw[:, 3].copy(), nan=0.0)
        pr_interval = np.nan_to_num(tabular_7_raw[:, 4].copy(), nan=0.0)
        float_features = np.stack(
            [tabular_7_raw[:, 1], tabular_7_raw[:, 2], atrial_rate, pr_interval, tabular_7_raw[:, 5], tabular_7_raw[:, 6]],
            axis=1,
        )

        if is_train:
            self.tabular_transformer = Pipeline(
                [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            )
            float_features_scaled = self.tabular_transformer.fit_transform(float_features)
        else:
            if self.tabular_transformer is None:
                raise ValueError("A fitted local transformer is required for non-training datasets.")
            float_features_scaled = self.tabular_transformer.transform(float_features)

        tabular_clean = np.concatenate([sex.reshape(-1, 1), float_features_scaled], axis=1)
        if np.isnan(tabular_clean).any():
            raise ValueError("Locally-preprocessed tabular features still contain NaNs.")
        return tabular_clean.astype(np.float32)

    def _load_ecg(self, eid: int) -> torch.Tensor:
        instance = self.eid_to_instance.get(eid)
        if instance is None:
            return torch.zeros((1, self.ecg_num_timepoints, self.ecg_num_leads), dtype=torch.float32)

        key = (eid, instance)
        if key not in self.ecg_data:
            return torch.zeros((1, self.ecg_num_timepoints, self.ecg_num_leads), dtype=torch.float32)

        ecg = self.ecg_data[key]
        return ecg.T.unsqueeze(0)

    def _log_label_distribution(self) -> None:
        logger.info("Per-label distribution for %s:", os.path.basename(self.csv_path))
        for idx, disease_name in enumerate(self.disease_label_columns):
            num_positive = int(np.sum(self.disease_labels[:, idx] == 1))
            prevalence = 100.0 * num_positive / max(len(self.disease_labels), 1)
            logger.info("  %s: %d positive (%.2f%%)", disease_name, num_positive, prevalence)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        eid = self.eids[idx]
        eid_40616 = self.eids_40616[idx]
        return {
            "ecg_raw": self._load_ecg(eid_40616),
            "tabular_7": torch.tensor(self.tabular_features[idx], dtype=torch.float32),
            "disease_labels": torch.tensor(self.disease_labels[idx], dtype=torch.float32),
            "eid": torch.tensor(eid, dtype=torch.int64),
        }

    def get_tabular_transformer(self) -> Optional[Pipeline]:
        return self.tabular_transformer

    def get_all_labels(self) -> np.ndarray:
        return self.disease_labels

    def get_disease_label_names(self) -> List[str]:
        return list(self.disease_label_columns)


def create_dataloaders(
    train_csv_path: str,
    internal_val_csv_path: str,
    external_test_csv_path: str,
    preprocessed_ecg_path: str,
    ecg_phenotypes_path: str,
    disease_label_columns: List[str],
    tabular_transform_path: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    disease_label_columns = list(disease_label_columns)

    logger.info("Preloading shared ECG tensor store once from %s", preprocessed_ecg_path)
    shared_ecg_data = torch.load(preprocessed_ecg_path)
    logger.info("Preloading shared ECG morphology table once from %s", ecg_phenotypes_path)
    shared_df_morphology = pd.read_csv(ecg_phenotypes_path)

    train_dataset = ECGMiniRetrainMultiLabelDataset(
        csv_path=train_csv_path,
        preprocessed_ecg_path=preprocessed_ecg_path,
        ecg_phenotypes_path=ecg_phenotypes_path,
        disease_label_columns=disease_label_columns,
        tabular_transform_path=tabular_transform_path,
        is_train=True,
        ecg_data=shared_ecg_data,
        df_morphology=shared_df_morphology,
    )

    pos_weight = calculate_multilabel_pos_weight(train_dataset.get_all_labels())

    trained_transformer = None
    if tabular_transform_path is None or str(tabular_transform_path).lower() == "none":
        trained_transformer = train_dataset.get_tabular_transformer()

    internal_val_dataset = ECGMiniRetrainMultiLabelDataset(
        csv_path=internal_val_csv_path,
        preprocessed_ecg_path=preprocessed_ecg_path,
        ecg_phenotypes_path=ecg_phenotypes_path,
        disease_label_columns=disease_label_columns,
        tabular_transform_path=tabular_transform_path if trained_transformer is None else None,
        trained_transformer=trained_transformer,
        is_train=False,
        ecg_data=shared_ecg_data,
        df_morphology=shared_df_morphology,
    )

    external_test_dataset = ECGMiniRetrainMultiLabelDataset(
        csv_path=external_test_csv_path,
        preprocessed_ecg_path=preprocessed_ecg_path,
        ecg_phenotypes_path=ecg_phenotypes_path,
        disease_label_columns=disease_label_columns,
        tabular_transform_path=tabular_transform_path if trained_transformer is None else None,
        trained_transformer=trained_transformer,
        is_train=False,
        ecg_data=shared_ecg_data,
        df_morphology=shared_df_morphology,
    )

    effective_num_workers = int(num_workers)
    if effective_num_workers > 0:
        logger.warning(
            "Forcing num_workers=0 because this project preloads ECG tensors in memory."
        )
        effective_num_workers = 0

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=effective_num_workers, pin_memory=torch.cuda.is_available(),
    )
    internal_val_loader = DataLoader(
        internal_val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=effective_num_workers, pin_memory=torch.cuda.is_available(),
    )
    external_test_loader = DataLoader(
        external_test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=effective_num_workers, pin_memory=torch.cuda.is_available(),
    )

    return train_loader, internal_val_loader, external_test_loader, pos_weight
