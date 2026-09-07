import logging
import os
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config_visionECG_Motion import MotionDecoderConfig


logger = logging.getLogger(__name__)


class FrameMeshDataset(Dataset):
    """Load latent, demographic, and mesh data."""

    @staticmethod
    def auto_detect_latent_dim(csv_path: str) -> int:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path, nrows=1)
        z_cols = [
            col
            for col in df.columns
            if col.startswith("z_") and col[2:].isdigit()
        ]
        latent_dim = len(z_cols)
        if latent_dim == 0:
            raise ValueError(f"No z_* columns found in {csv_path}")

        logger.info(
            "Auto-detected latent_dim=%s from %s",
            latent_dim,
            os.path.basename(csv_path),
        )
        return latent_dim

    def __init__(
        self,
        memory_z_path: str,
        queries_path: str,
        patient_csv_path: str,
        target_seg_dir: str,
        config: MotionDecoderConfig,
        is_train: bool = True,
        load_meshes: bool = True,
    ):
        self.memory_z_path = memory_z_path
        self.queries_path = queries_path
        self.patient_csv_path = patient_csv_path
        self.target_seg_dir = target_seg_dir
        self.config = config
        self.is_train = is_train
        self.load_meshes = load_meshes
        self.demo_cols = list(config.demo_cols)

        logger.info("Initializing FrameMeshDataset (%s)", "train" if is_train else "val/test")
        logger.info("  memory_z: %s", memory_z_path)
        logger.info("  queries: %s", queries_path)
        logger.info("  patient_csv: %s", patient_csv_path)
        logger.info("  target_seg_dir: %s", target_seg_dir)

        self.memory_z_cols = [f"z_{i}" for i in range(1, config.latent_dim + 1)]
        self.queries_cols = [
            f"mesh_embed_{x}_t_{t}"
            for t in range(1, config.seq_len + 1)
            for x in range(1, config.latent_dim + 1)
        ]

        memory_usecols = ["eid_18545"] + self.memory_z_cols
        queries_usecols = ["eid_18545"] + self.queries_cols
        patient_usecols = ["eid_18545"] + self.demo_cols

        logger.info("Loading memory z columns...")
        self.memory_df = pd.read_csv(memory_z_path, usecols=memory_usecols)
        logger.info("Loaded %s memory samples", len(self.memory_df))

        logger.info("Loading query columns...")
        self.queries_df = pd.read_csv(queries_path, usecols=queries_usecols)
        logger.info("Loaded %s query samples", len(self.queries_df))

        logger.info("Loading patient demographics columns...")
        self.patient_df = pd.read_csv(
            patient_csv_path,
            usecols=patient_usecols,
        )
        logger.info("Loaded %s patient samples", len(self.patient_df))

        self._verify_required_columns()
        self._align_by_eid()
        self._extract_arrays()

        logger.info("Dataset statistics")
        logger.info("  samples: %s", len(self))
        logger.info("  demographics: %s", self.demographics.shape)
        logger.info("  memory_z: %s", self.memory_z.shape)
        logger.info("  queries: %s", self.queries.shape)

    def _verify_required_columns(self):
        missing_cols = []
        for col in ["eid_18545"] + self.memory_z_cols:
            if col not in self.memory_df.columns:
                missing_cols.append(f"memory_df:{col}")
        for col in ["eid_18545"] + self.queries_cols:
            if col not in self.queries_df.columns:
                missing_cols.append(f"queries_df:{col}")
        for col in ["eid_18545"] + self.demo_cols:
            if col not in self.patient_df.columns:
                missing_cols.append(f"patient_df:{col}")
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols[:20]}")

    def _align_by_eid(self):
        memory_eids = set(self.memory_df["eid_18545"].values)
        queries_eids = set(self.queries_df["eid_18545"].values)
        patient_eids = set(self.patient_df["eid_18545"].values)
        common_eids = sorted(memory_eids & queries_eids & patient_eids)

        logger.info(
            "EID counts: memory=%s queries=%s patient=%s common=%s",
            len(memory_eids),
            len(queries_eids),
            len(patient_eids),
            len(common_eids),
        )

        if not common_eids:
            raise ValueError("No matching eid_18545 found across datasets")

        self.memory_df = (
            self.memory_df[self.memory_df["eid_18545"].isin(common_eids)]
            .set_index("eid_18545")
            .loc[common_eids]
            .reset_index()
        )
        self.queries_df = (
            self.queries_df[self.queries_df["eid_18545"].isin(common_eids)]
            .set_index("eid_18545")
            .loc[common_eids]
            .reset_index()
        )
        self.patient_df = (
            self.patient_df[self.patient_df["eid_18545"].isin(common_eids)]
            .set_index("eid_18545")
            .loc[common_eids]
            .reset_index()
        )
        self.eids = [int(eid) for eid in common_eids]

    def _extract_arrays(self):
        num_samples = len(self.memory_df)

        self.demographics = self.patient_df[self.demo_cols].values.astype(np.float32)
        self.demographics = np.nan_to_num(self.demographics, nan=0.0)

        self.memory_z = self.memory_df[self.memory_z_cols].values.astype(np.float32)
        self.memory_z = np.nan_to_num(self.memory_z, nan=0.0)

        queries_flat = self.queries_df[self.queries_cols].values.astype(np.float32)
        self.queries = queries_flat.reshape(
            num_samples,
            self.config.seq_len,
            self.config.latent_dim,
        )
        self.queries = np.nan_to_num(self.queries, nan=0.0)

    def _load_mesh(self, subid: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.load_meshes:
            return (
                torch.zeros(
                    (self.config.seq_len, self.config.points, 3),
                    dtype=torch.float32,
                ),
                torch.zeros((self.config.seq_len, 100, 3), dtype=torch.long),
                torch.zeros((self.config.seq_len, 2, 100), dtype=torch.long),
            )

        mesh_path = f"{self.target_seg_dir}/{subid}/image_space_pipemesh"
        if self.config.surf_type == "sample":
            h5filepath = f"{mesh_path}/preprossed_decimate.hdf5"
        else:
            h5filepath = f"{mesh_path}/preprossed_vtk.hdf5"

        try:
            with h5py.File(h5filepath, "r") as file_obj:
                mesh_verts = torch.tensor(np.array(file_obj["heart_v"]), dtype=torch.float32)
                mesh_faces = torch.tensor(np.array(file_obj["heart_f"]), dtype=torch.long)
                mesh_edges = torch.tensor(np.array(file_obj["heart_e"]), dtype=torch.long)

            if torch.isnan(mesh_verts).any():
                logger.warning("Detected NaN mesh vertices for subid %s", subid)
            if torch.isinf(mesh_verts).any():
                logger.warning("Detected Inf mesh vertices for subid %s", subid)

            return mesh_verts, mesh_faces, mesh_edges
        except FileNotFoundError:
            logger.warning("Mesh not found for subid %s: %s", subid, h5filepath)
        except Exception as exc:
            logger.error("Error loading mesh for subid %s: %s", subid, exc)

        return (
            torch.zeros(
                (self.config.seq_len, self.config.points, 3),
                dtype=torch.float32,
            ),
            torch.zeros((self.config.seq_len, 100, 3), dtype=torch.long),
            torch.zeros((self.config.seq_len, 2, 100), dtype=torch.long),
        )

    def __len__(self):
        return len(self.memory_z)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        eid = self.eids[idx]
        mesh_verts, mesh_faces, mesh_edges = self._load_mesh(eid)
        return {
            "z": torch.tensor(self.memory_z[idx], dtype=torch.float32),
            "queries": torch.tensor(self.queries[idx], dtype=torch.float32),
            "heart_v": mesh_verts,
            "heart_f": mesh_faces,
            "heart_e": mesh_edges,
            "demographics": torch.tensor(self.demographics[idx], dtype=torch.float32),
            "eid": eid,
        }

    def get_eids(self):
        return self.eids


class FrameDataModule:
    """Small data module for train/validation datasets and loaders."""

    def __init__(self, config: MotionDecoderConfig):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = "fit"):
        if stage != "fit" and stage is not None:
            return

        if not hasattr(self.config, "_latent_dim_auto_detected"):
            detected_dim = FrameMeshDataset.auto_detect_latent_dim(
                self.config.memory_z_train
            )
            if self.config.latent_dim != detected_dim:
                old_dim = self.config.latent_dim
                old_hidden = self.config.residual_hidden_dim
                self.config.latent_dim = detected_dim
                self.config.residual_hidden_dim = detected_dim * 2
                logger.info(
                    "Auto-detected latent_dim %s (was %s); residual_hidden_dim %s -> %s",
                    detected_dim,
                    old_dim,
                    old_hidden,
                    self.config.residual_hidden_dim,
                )
            self.config._latent_dim_auto_detected = True

        self.train_dataset = FrameMeshDataset(
            memory_z_path=self.config.memory_z_train,
            queries_path=self.config.queries_train,
            patient_csv_path=self.config.train_csv_path,
            target_seg_dir=self.config.target_seg_dir,
            config=self.config,
            is_train=True,
        )

        self.val_dataset = FrameMeshDataset(
            memory_z_path=self.config.memory_z_val,
            queries_path=self.config.queries_val,
            patient_csv_path=self.config.val_csv_path,
            target_seg_dir=self.config.target_seg_dir,
            config=self.config,
            is_train=False,
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Training dataset not initialized. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("Validation dataset not initialized. Call setup('fit') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
