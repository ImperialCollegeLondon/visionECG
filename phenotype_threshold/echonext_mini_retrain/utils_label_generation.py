"""Binary-label + pos_weight utilities for threshold-based classification tasks."""

import numpy as np
import pandas as pd
import torch
from typing import Union, Tuple


def create_binary_label(
    value: Union[float, int],
    threshold: Union[float, int],
    direction: str,
) -> int:
    """Threshold one value while preserving NaN."""
    if pd.isna(value):
        return np.nan
    if direction == 'less_than':
        return 1 if value <= threshold else 0
    if direction == 'greater_than':
        return 1 if value >= threshold else 0
    raise ValueError(f"Invalid direction: '{direction}'. Must be 'less_than' or 'greater_than'")


def create_binary_labels_batch(
    values: Union[np.ndarray, pd.Series],
    threshold: Union[float, int],
    direction: str,
) -> np.ndarray:
    """Threshold arrays while preserving NaNs."""
    if isinstance(values, pd.Series):
        values = values.values

    if direction == 'less_than':
        labels = (values <= threshold).astype(float)
    elif direction == 'greater_than':
        labels = (values >= threshold).astype(float)
    else:
        raise ValueError(f"Invalid direction: '{direction}'. Must be 'less_than' or 'greater_than'")

    labels[pd.isna(values)] = np.nan
    return labels


def calculate_pos_weight(
    labels: Union[torch.Tensor, np.ndarray, pd.Series],
) -> torch.Tensor:
    """Return pos_weight = negatives / positives."""
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, pd.Series):
        labels = labels.values

    labels = labels[~pd.isna(labels)]
    num_positive = np.sum(labels == 1)
    num_negative = np.sum(labels == 0)
    pos_weight = num_negative / (num_positive + 1e-6)
    return torch.tensor(pos_weight, dtype=torch.float32)


def get_class_distribution(
    labels: Union[torch.Tensor, np.ndarray, pd.Series],
) -> Tuple[int, int, float]:
    """Return (num_healthy, num_diseased, imbalance_ratio); NaN values ignored."""
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, pd.Series):
        labels = labels.values

    labels = labels[~pd.isna(labels)]
    num_healthy = int(np.sum(labels == 0))
    num_diseased = int(np.sum(labels == 1))
    imbalance_ratio = (num_healthy / num_diseased) if num_diseased > 0 else float('inf')
    return num_healthy, num_diseased, imbalance_ratio


def validate_labels(
    labels: Union[torch.Tensor, np.ndarray, pd.Series],
    min_samples_per_class: int = 10,
) -> Tuple[bool, str]:
    """Validate binary labels and minimum class sizes."""
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, pd.Series):
        labels = labels.values

    valid_labels = labels[~pd.isna(labels)]
    if len(valid_labels) == 0:
        return False, "All labels are NaN"

    unique_values = np.unique(valid_labels)
    if not np.all(np.isin(unique_values, [0, 1])):
        return False, f"Labels contain non-binary values: {unique_values}"

    num_healthy, num_diseased, _ = get_class_distribution(labels)
    if num_healthy == 0:
        return False, "No healthy samples (label=0) found"
    if num_diseased == 0:
        return False, "No diseased samples (label=1) found"
    if num_healthy < min_samples_per_class:
        return False, f"Insufficient healthy samples: {num_healthy} < {min_samples_per_class}"
    if num_diseased < min_samples_per_class:
        return False, f"Insufficient diseased samples: {num_diseased} < {min_samples_per_class}"
    return True, ""
