from typing import Optional, Tuple

import torch


def calculate_atrial_rate(pp_interval_ms: torch.Tensor) -> torch.Tensor:
    # atrial_rate = 60000 / PPInterval; undefined intervals map to zero.
    return torch.where(
        (pp_interval_ms > 0) & (~torch.isnan(pp_interval_ms)),
        60000.0 / pp_interval_ms,
        torch.zeros_like(pp_interval_ms),
    )


def map_24_to_7_features(
    demographics: torch.Tensor,
    ecg_morphology: torch.Tensor,
) -> torch.Tensor:
    sex = demographics[:, 1]
    age_at_ecg = demographics[:, 0]
    ventricular_rate = ecg_morphology[:, 0]
    pp_interval = ecg_morphology[:, 7]
    pr_interval = ecg_morphology[:, 1]
    qrs_duration = ecg_morphology[:, 3]
    qt_corrected = ecg_morphology[:, 5]
    atrial_rate = calculate_atrial_rate(pp_interval)

    return torch.stack(
        [sex, age_at_ecg, ventricular_rate, atrial_rate, pr_interval, qrs_duration, qt_corrected],
        dim=1,
    )


def validate_feature_mapping(
    demographics: torch.Tensor,
    ecg_morphology: torch.Tensor,
    tabular_7: Optional[torch.Tensor] = None,
) -> Tuple[bool, dict]:
    """Silently verify that map_24_to_7_features produced the expected column ordering."""
    if tabular_7 is None:
        tabular_7 = map_24_to_7_features(demographics, ecg_morphology)

    batch_size = demographics.shape[0]
    assert demographics.shape == (batch_size, 8)
    assert ecg_morphology.shape == (batch_size, 16)
    assert tabular_7.shape == (batch_size, 7)

    expected_atrial_rate = calculate_atrial_rate(ecg_morphology[:, 7])
    checks = {
        "sex_match": torch.allclose(tabular_7[:, 0], demographics[:, 1]),
        "age_match": torch.allclose(tabular_7[:, 1], demographics[:, 0]),
        "ventricular_rate_match": torch.allclose(tabular_7[:, 2], ecg_morphology[:, 0]),
        "atrial_rate_match": torch.allclose(tabular_7[:, 3], expected_atrial_rate, equal_nan=True),
        "pr_interval_match": torch.allclose(tabular_7[:, 4], ecg_morphology[:, 1]),
        "qrs_duration_match": torch.allclose(tabular_7[:, 5], ecg_morphology[:, 3]),
        "qt_corrected_match": torch.allclose(tabular_7[:, 6], ecg_morphology[:, 5]),
    }
    is_valid = all(checks.values())
    stats = {"batch_size": batch_size, "all_valid": is_valid, **checks}
    return is_valid, stats


class FeatureMapper:
    """Wrap feature mapping as a callable module."""

    def __call__(self, demographics: torch.Tensor, ecg_morphology: torch.Tensor) -> torch.Tensor:
        return map_24_to_7_features(demographics, ecg_morphology)

    def validate(self, demographics: torch.Tensor, ecg_morphology: torch.Tensor) -> Tuple[bool, dict]:
        tabular_7 = self(demographics, ecg_morphology)
        return validate_feature_mapping(demographics, ecg_morphology, tabular_7)
