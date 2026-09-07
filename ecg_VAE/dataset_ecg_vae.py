"""Data loading for preprocessed ECG VAE training tensors."""

from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset


EXPECTED_LEADS = 12
EXPECTED_SAMPLES = 600


def _load_torch_file(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ECGVAEDataset(Dataset):
    """Dataset of preprocessed 12-lead ECG tensors."""

    def __init__(self, pt_file: str, split: str):
        self.split = split
        self.data_path = Path(pt_file).expanduser()
        if not self.data_path.is_file():
            raise FileNotFoundError(
                f"{split} data file not found: {self.data_path}"
            )

        loaded_data = _load_torch_file(self.data_path)
        if not isinstance(loaded_data, (list, tuple)):
            raise TypeError(
                f"{self.data_path} must contain a list or tuple of ECG tensors, "
                f"got {type(loaded_data).__name__}"
            )

        self.ecg_data: List[torch.Tensor] = [
            signal for signal in loaded_data if signal is not None
        ]
        if not self.ecg_data:
            raise ValueError(
                f"{self.data_path} contains no usable ECG tensors"
            )

        for index, signal in enumerate(self.ecg_data):
            if not isinstance(signal, torch.Tensor):
                raise TypeError(
                    f"{self.data_path} item {index} must be a torch.Tensor, "
                    f"got {type(signal).__name__}"
                )
            if tuple(signal.shape) != (EXPECTED_LEADS, EXPECTED_SAMPLES):
                raise ValueError(
                    f"{self.data_path} item {index} has shape "
                    f"{tuple(signal.shape)}; expected "
                    f"({EXPECTED_LEADS}, {EXPECTED_SAMPLES})"
                )

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.ecg_data[index]

    def __len__(self) -> int:
        return len(self.ecg_data)


class ECGVAEDataModule:
    """Construct training and validation data loaders from preprocessed tensors."""

    def __init__(
        self,
        train_pt: str,
        val_pt: str,
        batch_size: int,
        num_workers: int,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_workers < 0:
            raise ValueError(
                f"num_workers must be non-negative, got {num_workers}"
            )

        self.train_pt = train_pt
        self.val_pt = val_pt
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_train = None
        self.data_val = None

    def setup(self) -> None:
        self.data_train = ECGVAEDataset(self.train_pt, split="train")
        self.data_val = ECGVAEDataset(self.val_pt, split="val")

    def train_dataloader(self) -> DataLoader:
        if self.data_train is None:
            raise RuntimeError("Call setup() before requesting the training loader")
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        if self.data_val is None:
            raise RuntimeError("Call setup() before requesting the validation loader")
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
