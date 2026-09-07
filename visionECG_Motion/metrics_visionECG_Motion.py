from typing import List, Tuple

import numpy as np
import torch

try:
    from pytorch3d.ops import knn_points
    _PYTORCH3D_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    knn_points = None
    _PYTORCH3D_IMPORT_ERROR = exc


def _compute_directed_nearest_distances(
    pred_verts: torch.Tensor,
    gt_verts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if knn_points is None:
        raise ModuleNotFoundError(
            "pytorch3d is required for mesh distance metrics"
        ) from _PYTORCH3D_IMPORT_ERROR
    if pred_verts.dim() == 2:
        pred_verts = pred_verts.unsqueeze(0)
    if gt_verts.dim() == 2:
        gt_verts = gt_verts.unsqueeze(0)

    dist_pred_to_gt = knn_points(pred_verts, gt_verts, K=1).dists.sqrt()
    dist_gt_to_pred = knn_points(gt_verts, pred_verts, K=1).dists.sqrt()
    return dist_pred_to_gt, dist_gt_to_pred


def compute_hausdorff_distance(
    pred_verts: torch.Tensor,
    gt_verts: torch.Tensor,
) -> torch.Tensor:
    dist_pred_to_gt, dist_gt_to_pred = _compute_directed_nearest_distances(
        pred_verts,
        gt_verts,
    )
    return torch.maximum(dist_pred_to_gt.max(), dist_gt_to_pred.max())


def compute_hd90(
    pred_verts: torch.Tensor,
    gt_verts: torch.Tensor,
) -> torch.Tensor:
    dist_pred_to_gt, dist_gt_to_pred = _compute_directed_nearest_distances(
        pred_verts,
        gt_verts,
    )
    return torch.maximum(
        torch.quantile(dist_pred_to_gt.flatten(), 0.9),
        torch.quantile(dist_gt_to_pred.flatten(), 0.9),
    )


def compute_assd(pred_verts: torch.Tensor, gt_verts: torch.Tensor) -> torch.Tensor:
    dist_pred_to_gt, dist_gt_to_pred = _compute_directed_nearest_distances(
        pred_verts,
        gt_verts,
    )
    return (dist_pred_to_gt.mean() + dist_gt_to_pred.mean()) / 2.0


def compute_mesh_metrics_for_sequence(
    pred_sequence: torch.Tensor,
    gt_sequence: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hd_values = []
    hd90_values = []
    assd_values = []
    for frame_idx in range(pred_sequence.shape[0]):
        pred_verts = pred_sequence[frame_idx]
        gt_verts = gt_sequence[frame_idx]
        dist_pred_to_gt, dist_gt_to_pred = _compute_directed_nearest_distances(
            pred_verts,
            gt_verts,
        )
        hd_values.append(
            torch.maximum(dist_pred_to_gt.max(), dist_gt_to_pred.max())
        )
        hd90_values.append(
            torch.maximum(
                torch.quantile(dist_pred_to_gt.flatten(), 0.9),
                torch.quantile(dist_gt_to_pred.flatten(), 0.9),
            )
        )
        assd_values.append(
            (dist_pred_to_gt.mean() + dist_gt_to_pred.mean()) / 2.0
        )
    return (
        torch.stack(hd_values).mean(),
        torch.stack(hd90_values).mean(),
        torch.stack(assd_values).mean(),
    )


def compute_batch_metrics_per_sample(
    pred_batch: torch.Tensor,
    gt_batch: torch.Tensor,
) -> List[dict]:
    batch_metrics = []
    for batch_idx in range(pred_batch.shape[0]):
        try:
            avg_hd, avg_hd90, avg_assd = compute_mesh_metrics_for_sequence(
                pred_batch[batch_idx],
                gt_batch[batch_idx],
            )
            mse = torch.mean((pred_batch[batch_idx] - gt_batch[batch_idx]) ** 2)
            batch_metrics.append(
                {
                    "hd": avg_hd.item(),
                    "hd90": avg_hd90.item(),
                    "assd": avg_assd.item(),
                    "mse": mse.item(),
                }
            )
        except Exception as exc:
            print(f"Warning: error computing metrics for sample {batch_idx}: {exc}")
            batch_metrics.append(
                {"hd": 0.0, "hd90": 0.0, "assd": 0.0, "mse": 0.0}
            )
    return batch_metrics


class EpochMetricsAccumulator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.hd_values = []
        self.hd90_values = []
        self.assd_values = []

    def update_raw_values(
        self,
        hd_values: List[float],
        hd90_values: List[float],
        assd_values: List[float],
    ):
        self.hd_values.extend(hd_values)
        self.hd90_values.extend(hd90_values)
        self.assd_values.extend(assd_values)

    def compute_epoch_stats(self) -> dict:
        if not self.hd_values:
            return {
                "hd_mean": 0.0,
                "hd_std": 0.0,
                "hd_median": 0.0,
                "hd_q1": 0.0,
                "hd_q3": 0.0,
                "hd90_mean": 0.0,
                "hd90_std": 0.0,
                "hd90_median": 0.0,
                "hd90_q1": 0.0,
                "hd90_q3": 0.0,
                "assd_mean": 0.0,
                "assd_std": 0.0,
                "assd_median": 0.0,
                "assd_q1": 0.0,
                "assd_q3": 0.0,
                "num_samples": 0,
            }

        hd_array = np.array(self.hd_values)
        hd90_array = np.array(self.hd90_values)
        assd_array = np.array(self.assd_values)
        return {
            "hd_mean": float(np.mean(hd_array)),
            "hd_std": float(np.std(hd_array)),
            "hd_median": float(np.median(hd_array)),
            "hd_q1": float(np.percentile(hd_array, 25)),
            "hd_q3": float(np.percentile(hd_array, 75)),
            "hd90_mean": float(np.mean(hd90_array)),
            "hd90_std": float(np.std(hd90_array)),
            "hd90_median": float(np.median(hd90_array)),
            "hd90_q1": float(np.percentile(hd90_array, 25)),
            "hd90_q3": float(np.percentile(hd90_array, 75)),
            "assd_mean": float(np.mean(assd_array)),
            "assd_std": float(np.std(assd_array)),
            "assd_median": float(np.median(assd_array)),
            "assd_q1": float(np.percentile(assd_array, 25)),
            "assd_q3": float(np.percentile(assd_array, 75)),
            "num_samples": len(self.hd_values),
        }


def print_metrics_summary(metrics: dict, prefix: str = "", logger=None):
    lines = [
        f"{prefix}Mesh Reconstruction Metrics:",
        (
            f"  Hausdorff Distance - Mean: {metrics['hd_mean']:.4f}, "
            f"Median: {metrics['hd_median']:.4f}"
        ),
        f"                     - Q1: {metrics['hd_q1']:.4f}, Q3: {metrics['hd_q3']:.4f}",
        f"                     - Std: {metrics['hd_std']:.4f}",
        (
            f"  HD90               - Mean: {metrics['hd90_mean']:.4f}, "
            f"Median: {metrics['hd90_median']:.4f}"
        ),
        (
            f"                     - Q1: {metrics['hd90_q1']:.4f}, "
            f"Q3: {metrics['hd90_q3']:.4f}"
        ),
        f"                     - Std: {metrics['hd90_std']:.4f}",
        (
            f"  ASSD               - Mean: {metrics['assd_mean']:.4f}, "
            f"Median: {metrics['assd_median']:.4f}"
        ),
        (
            f"                     - Q1: {metrics['assd_q1']:.4f}, "
            f"Q3: {metrics['assd_q3']:.4f}"
        ),
        f"                     - Std: {metrics['assd_std']:.4f}",
        f"  Valid Samples: {metrics.get('num_samples', metrics.get('num_valid_samples', 0))}",
    ]
    if logger:
        for line in lines:
            logger.info(line)
    else:
        for line in lines:
            print(line)
