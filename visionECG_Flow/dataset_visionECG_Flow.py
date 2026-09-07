from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from config_visionECG_Flow import VisionECGFlowConfig


# DATASET
class ECGMotionDataset_VisionECGFlow(Dataset):
    def __init__(self, csv_path: str, config: VisionECGFlowConfig,
                 motion_scaler: Optional[StandardScaler] = None,
                 is_train: bool = True):
        self.config = config
        self.is_train = is_train

        # Read CSV
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples from {csv_path}")

        # Column lists
        self.demo_cols = list(config.demo_cols)
        self.ecg_cols = [f'ecg_embed_{i}' for i in range(config.ecg_embed_dim)]

        if config.is_frame_resolved:
            self.motion_cols = []
            for t in range(1, config.num_frames + 1):
                for x in range(1, config.motion_embed_dim + 1):
                    self.motion_cols.append(f'mesh_embed_{x}_t_{t}')
        else:
            self.motion_cols = [f'mesh_embed_{i}' for i in range(1, config.motion_embed_dim + 1)]

        # EID handling
        self.has_eid = 'eid_18545' in self.df.columns or 'eid' in self.df.columns
        if self.has_eid:
            self.eid_col = 'eid_18545' if 'eid_18545' in self.df.columns else 'eid'
            self.eids = self.df[self.eid_col].values
        else:
            self.eids = np.arange(len(self.df))

        # Required-column check
        missing_required = [c for c in self.demo_cols + self.ecg_cols if c not in self.df.columns]
        if missing_required:
            raise ValueError(f"Missing required (demographics/ECG) columns: {missing_required[:10]}...")

        missing_motion = [c for c in self.motion_cols if c not in self.df.columns]
        self.has_motion_data = (len(missing_motion) == 0)
        if not self.has_motion_data and is_train:
            raise ValueError(f"Missing motion columns in training set: {missing_motion[:5]}...")
        if not self.has_motion_data:
            print(f"Inference-only: {len(missing_motion)}/{len(self.motion_cols)} motion columns missing")

        # Extract arrays
        self.demographics = self.df[self.demo_cols].values.astype(np.float32)
        self.ecg_embeddings = self.df[self.ecg_cols].values.astype(np.float32)

        if self.has_motion_data:
            motion_flat = self.df[self.motion_cols].values.astype(np.float32)
            if config.is_frame_resolved:
                self.motion_embeddings = motion_flat.reshape(
                    len(self.df), config.num_frames, config.motion_embed_dim
                )
            else:
                self.motion_embeddings = motion_flat
        else:
            self.motion_embeddings = None

        # Replace NaNs
        self.demographics = np.nan_to_num(self.demographics, nan=0.0)
        self.ecg_embeddings = np.nan_to_num(self.ecg_embeddings, nan=0.0)
        if self.has_motion_data:
            self.motion_embeddings = np.nan_to_num(self.motion_embeddings, nan=0.0)

        # Normalise motion
        if config.normalize_data and self.has_motion_data:
            if is_train:
                self.motion_scaler = StandardScaler()
                self.motion_embeddings = self._fit_transform_motion(
                    self.motion_scaler, self.motion_embeddings
                )
            else:
                if motion_scaler is None:
                    raise ValueError("Motion scaler must be provided for non-training datasets")
                self.motion_scaler = motion_scaler
                self.motion_embeddings = self._transform_motion(
                    self.motion_scaler, self.motion_embeddings
                )
        else:
            self.motion_scaler = None

        print(f"Mode: {config.mode}")
        print(f"Demographics shape: {self.demographics.shape}")
        print(f"ECG embeddings shape: {self.ecg_embeddings.shape}")
        if self.has_motion_data:
            print(f"Motion embeddings shape: {self.motion_embeddings.shape}")

    def _fit_transform_motion(self, scaler: StandardScaler, motion: np.ndarray) -> np.ndarray:
        """Fit-transform on train motion."""
        if self.config.is_frame_resolved:
            shape = motion.shape
            flat = motion.reshape(-1, shape[-1])
            flat = scaler.fit_transform(flat)
            return flat.reshape(shape)
        return scaler.fit_transform(motion)

    def _transform_motion(self, scaler: StandardScaler, motion: np.ndarray) -> np.ndarray:
        """Transform val/test motion."""
        if self.config.is_frame_resolved:
            shape = motion.shape
            flat = motion.reshape(-1, shape[-1])
            flat = scaler.transform(flat)
            return flat.reshape(shape)
        return scaler.transform(motion)

    # Dataset protocol

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor,
                                        Optional[torch.Tensor], Dict]:
        """Return (demographics, ecg_embeddings, motion_embeddings, info)."""
        demographics = torch.FloatTensor(self.demographics[idx])
        ecg_embeddings = torch.FloatTensor(self.ecg_embeddings[idx])
        motion_embeddings = (
            torch.FloatTensor(self.motion_embeddings[idx]) if self.has_motion_data else None
        )
        info_dict = {'is_train': self.is_train, 'mode': self.config.mode}
        return demographics, ecg_embeddings, motion_embeddings, info_dict

    # Utilities

    def get_motion_scaler(self):
        return self.motion_scaler

    def get_eids(self):
        return self.eids

    def inverse_transform_motion(self, motion):
        """Inverse-transform motion."""
        if self.motion_scaler is None:
            return motion
        if isinstance(motion, torch.Tensor):
            motion = motion.cpu().numpy()
        if motion.ndim >= 2 and motion.shape[-1] == self.config.motion_embed_dim and motion.ndim != 1:
            shape = motion.shape
            flat = motion.reshape(-1, shape[-1])
            transformed = self.motion_scaler.inverse_transform(flat)
            return transformed.reshape(shape)
        return self.motion_scaler.inverse_transform(motion)


# COLLATE
def collate_fn_visionECG_Flow(batch):
    """Stack dataset samples into a batch."""
    demographics, ecg_embeddings, motion_embeddings, info_dicts = zip(*batch)
    demographics_b = torch.stack(demographics)
    ecg_b = torch.stack(ecg_embeddings)
    motion_b = torch.stack(motion_embeddings) if motion_embeddings[0] is not None else None
    info = {'is_train': info_dicts[0]['is_train'], 'mode': info_dicts[0]['mode']}
    return demographics_b, ecg_b, motion_b, info


# TEMPLATE COMPUTATION
def compute_template_visionECG_Flow(
    dataset: ECGMotionDataset_VisionECGFlow,
    mean_file: Optional[str] = None,
    std_file: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Empirical mean/std template."""
    motion = dataset.motion_embeddings
    mean_arr = np.load(mean_file).astype(np.float32) if mean_file else np.mean(motion, axis=0)
    std_arr = np.load(std_file).astype(np.float32) if std_file else np.std(motion, axis=0)
    expected_shape = motion.shape[1:]
    if mean_arr.shape != expected_shape or std_arr.shape != expected_shape:
        raise ValueError(
            f"Template shape mismatch: expected {expected_shape}, "
            f"got mean={mean_arr.shape}, std={std_arr.shape}"
        )
    return {
        'mean': torch.FloatTensor(mean_arr),
        'std': torch.FloatTensor(std_arr.astype(np.float32)),
    }
