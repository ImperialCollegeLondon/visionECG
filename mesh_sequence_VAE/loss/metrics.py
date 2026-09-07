import torch
import numpy as np
from pytorch3d.ops import knn_points
from typing import Tuple, List


def compute_hausdorff_distance(pred_verts: torch.Tensor, gt_verts: torch.Tensor) -> torch.Tensor:
    """Symmetric Hausdorff distance between two point sets."""
    if pred_verts.dim() == 2:
        pred_verts = pred_verts.unsqueeze(0)
    if gt_verts.dim() == 2:
        gt_verts = gt_verts.unsqueeze(0)

    dist_pred_to_gt = knn_points(pred_verts, gt_verts, K=1).dists.sqrt()
    dist_gt_to_pred = knn_points(gt_verts, pred_verts, K=1).dists.sqrt()
    return torch.max(dist_pred_to_gt.max(), dist_gt_to_pred.max())


def compute_assd(pred_verts: torch.Tensor, gt_verts: torch.Tensor) -> torch.Tensor:
    """Average symmetric surface distance."""
    if pred_verts.dim() == 2:
        pred_verts = pred_verts.unsqueeze(0)
    if gt_verts.dim() == 2:
        gt_verts = gt_verts.unsqueeze(0)

    dist_pred_to_gt = knn_points(pred_verts, gt_verts, K=1).dists.sqrt()
    dist_gt_to_pred = knn_points(gt_verts, pred_verts, K=1).dists.sqrt()
    return (dist_pred_to_gt.mean() + dist_gt_to_pred.mean()) / 2.0


def compute_mesh_metrics_for_sequence(pred_sequence: torch.Tensor,
                                      gt_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (avg HD, avg ASSD) over a mesh sequence."""
    seq_len = pred_sequence.shape[0]
    hd_values = []
    assd_values = []

    for t in range(seq_len):
        hd_values.append(compute_hausdorff_distance(pred_sequence[t], gt_sequence[t]))
        assd_values.append(compute_assd(pred_sequence[t], gt_sequence[t]))

    return torch.stack(hd_values).mean(), torch.stack(assd_values).mean()


def compute_batch_metrics(pred_batch: torch.Tensor,
                          gt_batch: torch.Tensor,
                          patient_ids: List[str] = None) -> dict:
    """Summarize batch mesh-sequence distances."""
    batch_size = pred_batch.shape[0]
    hd_values = []
    assd_values = []

    for b in range(batch_size):
        try:
            avg_hd, avg_assd = compute_mesh_metrics_for_sequence(pred_batch[b], gt_batch[b])
            hd_values.append(avg_hd.item())
            assd_values.append(avg_assd.item())
        except Exception as e:
            print(f"Error computing metrics for batch {b}: {e}")
            if patient_ids:
                print(f"Patient ID: {patient_ids[b]}")
            continue

    if not hd_values:
        return {
            'hd_mean': 0.0, 'hd_median': 0.0, 'hd_q1': 0.0, 'hd_q3': 0.0,
            'assd_mean': 0.0, 'assd_median': 0.0, 'assd_q1': 0.0, 'assd_q3': 0.0,
            'num_valid_samples': 0,
        }

    hd_array = np.array(hd_values)
    assd_array = np.array(assd_values)

    return {
        'hd_mean': float(np.mean(hd_array)),
        'hd_median': float(np.median(hd_array)),
        'hd_q1': float(np.percentile(hd_array, 25)),
        'hd_q3': float(np.percentile(hd_array, 75)),
        'assd_mean': float(np.mean(assd_array)),
        'assd_median': float(np.median(assd_array)),
        'assd_q1': float(np.percentile(assd_array, 25)),
        'assd_q3': float(np.percentile(assd_array, 75)),
        'num_valid_samples': len(hd_values),
    }


class EpochMetricsAccumulator:
    """Accumulate HD/ASSD across batches within an epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.hd_values = []
        self.assd_values = []

    def update_raw_values(self, hd_values: List[float], assd_values: List[float]):
        self.hd_values.extend(hd_values)
        self.assd_values.extend(assd_values)

    def compute_epoch_stats(self) -> dict:
        if not self.hd_values:
            return {
                'hd_mean': 0.0, 'hd_median': 0.0, 'hd_q1': 0.0, 'hd_q3': 0.0,
                'assd_mean': 0.0, 'assd_median': 0.0, 'assd_q1': 0.0, 'assd_q3': 0.0,
                'num_samples': 0,
            }

        hd_array = np.array(self.hd_values)
        assd_array = np.array(self.assd_values)

        return {
            'hd_mean': float(np.mean(hd_array)),
            'hd_median': float(np.median(hd_array)),
            'hd_q1': float(np.percentile(hd_array, 25)),
            'hd_q3': float(np.percentile(hd_array, 75)),
            'assd_mean': float(np.mean(assd_array)),
            'assd_median': float(np.median(assd_array)),
            'assd_q1': float(np.percentile(assd_array, 25)),
            'assd_q3': float(np.percentile(assd_array, 75)),
            'num_samples': len(self.hd_values),
        }


def print_metrics_summary(metrics: dict, prefix: str = "", logger=None):
    """Log a formatted summary of mesh reconstruction metrics."""
    lines = [
        f"{prefix}Mesh Reconstruction Metrics:",
        f"  Hausdorff Distance  - Mean: {metrics['hd_mean']:.4f}, Median: {metrics['hd_median']:.4f}",
        f"                      - Q1: {metrics['hd_q1']:.4f}, Q3: {metrics['hd_q3']:.4f}",
        f"  ASSD               - Mean: {metrics['assd_mean']:.4f}, Median: {metrics['assd_median']:.4f}",
        f"                      - Q1: {metrics['assd_q1']:.4f}, Q3: {metrics['assd_q3']:.4f}",
        f"  Valid Samples: {metrics.get('num_samples', metrics.get('num_valid_samples', 0))}",
    ]
    for line in lines:
        if logger:
            logger.info(line)
        else:
            print(line)
