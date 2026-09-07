#!/usr/bin/env python

import argparse
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model_echonext_mini_retrain import ECGEchoNextFinetune
from loader_ecg_mini_retrain import create_dataloaders
from utils_label_generation import create_binary_labels_batch


POSITIVE_OPERATOR_TO_DIRECTION = {"lt": "less_than", "gt": "greater_than"}


@dataclass(frozen=True)
class TaskConfig:
    task_key: str
    display_name: str
    label_column: str
    positive_operator: str  # 'lt' or 'gt'
    threshold: float
    sex_filter: Optional[int] = None

    @property
    def threshold_direction(self) -> str:
        return POSITIVE_OPERATOR_TO_DIRECTION[self.positive_operator]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes', 'y', 't'):
        return True
    if v.lower() in ('false', '0', 'no', 'n', 'f'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def stratified_split_and_write(
    train_csv_path: str,
    label_column: str,
    threshold: float,
    threshold_direction: str,
    val_frac: float,
    split_seed: int,
    sex_filter,
    out_dir: str,
    logger: logging.Logger,
) -> Tuple[str, str]:
    """Create and save a stratified 80/20 split."""
    df = pd.read_csv(train_csv_path)
    logger.info(f"Loaded full training CSV: {len(df)} rows")

    if sex_filter is not None:
        sex_name = 'male' if sex_filter == 1 else 'female'
        df = df[df['Sex'] == sex_filter].copy()
        logger.info(f"After sex filter ({sex_name}={sex_filter}): {len(df)} rows")

    raw = df[label_column].values.astype(np.float32)
    labels = create_binary_labels_batch(raw, threshold, threshold_direction)

    nan_mask = np.isnan(labels)
    if nan_mask.any():
        df = df.loc[~nan_mask].reset_index(drop=True)
        labels = labels[~nan_mask]
        logger.info(f"After dropping NaN labels: {len(df)} rows")

    logger.info(f"Full pos prevalence: pos={int(labels.sum())} / {len(labels)} "
                f"({labels.mean():.4f})")

    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_frac, stratify=labels, random_state=split_seed
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, 'train_split.csv')
    val_path = os.path.join(out_dir, 'val_split.csv')
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    logger.info(
        f"Split written: train={len(train_df)} "
        f"(pos={int(labels[train_idx].sum())}, prev={labels[train_idx].mean():.4f}); "
        f"val={len(val_df)} "
        f"(pos={int(labels[val_idx].sum())}, prev={labels[val_idx].mean():.4f})"
    )
    return train_path, val_path


def run_and_save_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    output_path: str,
    pos_weight: torch.Tensor,
    logger: logging.Logger,
) -> Tuple[float, float, int]:
    """Evaluate a loader and save predictions."""
    model.eval()
    all_eids, all_labels, all_probs = [], [], []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            ecg = batch['ecg_raw'].to(device)
            tabular = batch['tabular_7'].to(device)
            labels = batch['label'].to(device)
            eids = batch['eid']
            logits = model(ecg, tabular)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            total_loss += loss.item()
            n_batches += 1
            probs = torch.sigmoid(logits).cpu().numpy()
            all_eids.extend(eids.numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    avg_loss = total_loss / max(1, n_batches)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.5
    df = pd.DataFrame({
        'eid': all_eids,
        'true_label': all_labels,
        'predicted_probability': all_probs,
    })
    df.to_csv(output_path, index=False)
    logger.info(f"  Saved predictions: {output_path} "
                f"(n={len(df)}, loss={avg_loss:.4f}, auroc={auroc:.4f})")
    return avg_loss, auroc, len(df)


def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    log_file = os.path.join(output_dir, "logs", "training.log")

    logger = logging.getLogger("echonext_mini_retrain")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor,
    device: str,
    grad_clip_norm: Optional[float] = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch in train_loader:
        ecg = batch['ecg_raw'].to(device)
        tabular = batch['tabular_7'].to(device)
        labels = batch['label'].to(device)

        logits = model(ecg, tabular)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    try:
        avg_auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        avg_auroc = 0.5
    return avg_loss, avg_auroc


def validate_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    pos_weight: torch.Tensor,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in val_loader:
            ecg = batch['ecg_raw'].to(device)
            tabular = batch['tabular_7'].to(device)
            labels = batch['label'].to(device)

            logits = model(ecg, tabular)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    try:
        avg_auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        avg_auroc = 0.5
    return avg_loss, avg_auroc


def main():
    parser = argparse.ArgumentParser(description="EchoNext-Mini retrain — standard tasks")

    parser.add_argument("--task_key", type=str, required=True,
                        help="Task identifier, safe for filenames (e.g. LVEF_less_than_45)")
    parser.add_argument("--display_name", type=str, required=True,
                        help="Human-readable task label for logs (e.g. 'LVEF < 45')")
    parser.add_argument("--label_column", type=str, required=True,
                        help="Column in --train_csv / --test_csv to threshold into the binary label")
    parser.add_argument("--positive_operator", type=str, required=True,
                        choices=["lt", "gt"],
                        help="lt: label=1 iff value<threshold; gt: label=1 iff value>=threshold")
    parser.add_argument("--threshold", type=float, required=True,
                        help="Threshold applied to --label_column")

    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to pretrained EchoNext-Mini weights.pt")
    parser.add_argument("--freeze_backbone", action="store_true", default=True,
                        help="Freeze backbone (train only output layer)")

    parser.add_argument("--train_csv", type=str, required=True,
                        help="Training CSV (will be split 80/20 into train/val)")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="External test CSV")
    parser.add_argument("--val_frac", type=float, default=0.2,
                        help="Validation fraction of --train_csv (default: 0.2)")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Random seed for stratified split (default: 42)")
    parser.add_argument("--preprocessed_ecg", type=str, required=True,
                        help="Preprocessed ECG .pt file (12x2500)")
    parser.add_argument("--ecg_phenotypes_path", type=str, required=True,
                        help="ECG morphology CSV")
    parser.add_argument("--tabular_transform_path", type=str, required=True,
                        help="Reference tabular_transformer.joblib")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--use_timestamp_subdir", type=str2bool, default=True,
                        help="Append {task_key}_{timestamp} to --output_dir (default: True)")
    parser.add_argument("--snapshot_every", type=int, default=10,
                        help="Save prediction snapshots every N epochs (default: 10)")

    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5, EchoNext-Mini)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16, EchoNext-Mini)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01, EchoNext-Mini)")
    parser.add_argument("--early_stopping_patience", type=int, default=20,
                        help="Early-stopping patience on val_loss (default: 20)")
    parser.add_argument("--grad_clip_norm", type=float, default=None,
                        help="Optional gradient-clipping max-norm; default None = disabled "
                             "(EchoNext reference uses no clipping)")

    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use (-1 for CPU)")

    args = parser.parse_args()

    task_config = TaskConfig(
        task_key=args.task_key,
        display_name=args.display_name,
        label_column=args.label_column,
        positive_operator=args.positive_operator,
        threshold=args.threshold,
    )

    if args.use_timestamp_subdir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_output_dir = os.path.join(args.output_dir, f"{task_config.task_key}_{timestamp}")
    else:
        task_output_dir = args.output_dir
    os.makedirs(task_output_dir, exist_ok=True)

    logger = setup_logging(task_output_dir)
    logger.info(f"Task: {task_config.task_key} ({task_config.display_name})")
    logger.info(f"Label column: {task_config.label_column} "
                f"{task_config.threshold_direction} {task_config.threshold}")
    logger.info(f"Output directory: {task_output_dir}")

    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = f"cuda:{args.gpu_id}"
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(args.gpu_id)})")
    else:
        device = "cpu"
        logger.info("Using device: CPU")

    splits_dir = os.path.join(task_output_dir, 'splits')
    train_split_path, val_split_path = stratified_split_and_write(
        train_csv_path=args.train_csv,
        label_column=task_config.label_column,
        threshold=task_config.threshold,
        threshold_direction=task_config.threshold_direction,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
        sex_filter=None,
        out_dir=splits_dir,
        logger=logger,
    )

    train_loader, val_loader, _, pos_weight = create_dataloaders(
        train_csv=train_split_path,
        val_csv=val_split_path,
        preprocessed_ecg_path=args.preprocessed_ecg,
        ecg_phenotypes_path=args.ecg_phenotypes_path,
        label_column=task_config.label_column,
        threshold=task_config.threshold,
        threshold_direction=task_config.threshold_direction,
        tabular_transform_path=args.tabular_transform_path,
        sex_filter=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Discard the duplicate training loader.
    # Reuse the saved tabular transformer.
    _, test_loader, _, _ = create_dataloaders(
        train_csv=train_split_path,
        val_csv=args.test_csv,
        preprocessed_ecg_path=args.preprocessed_ecg,
        ecg_phenotypes_path=args.ecg_phenotypes_path,
        label_column=task_config.label_column,
        threshold=task_config.threshold,
        threshold_direction=task_config.threshold_direction,
        tabular_transform_path=args.tabular_transform_path,
        sex_filter=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    pos_weight = pos_weight.to(device)
    logger.info(f"pos_weight for BCE loss: {pos_weight.item():.2f}")

    model = ECGEchoNextFinetune(
        weights_path=args.weights_path,
        freeze_backbone=args.freeze_backbone,
        device=device,
        filter_size=16,
        dropout=0.5,
    )

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
    )

    train_pred_path = os.path.join(task_output_dir, "train_predictions.csv")
    val_pred_path = os.path.join(task_output_dir, "val_predictions.csv")
    test_pred_path = os.path.join(task_output_dir, "test_predictions.csv")
    best_ckpt_path = os.path.join(task_output_dir, "best_model.ckpt")

    best_val_loss = float('inf')
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0

    history = {
        'epoch': [], 'train_loss': [], 'train_auroc': [],
        'val_loss': [], 'val_auroc': [],
        'test_loss': [], 'test_auroc': [],
        'is_best': [], 'is_snapshot': [],
    }

    last_epoch = 0
    for epoch in range(args.epochs):
        last_epoch = epoch + 1
        logger.info(f"Epoch {last_epoch}/{args.epochs}")

        train_loss, train_auroc = train_epoch(
            model, train_loader, optimizer, pos_weight, device, args.grad_clip_norm
        )
        val_loss, val_auroc = validate_epoch(
            model, val_loader, pos_weight, device
        )
        logger.info(f"  Train loss={train_loss:.4f} auroc={train_auroc:.4f} | "
                    f"Val loss={val_loss:.4f} auroc={val_auroc:.4f}")

        is_best = val_loss < best_val_loss
        is_snapshot = (last_epoch % args.snapshot_every == 0)
        epoch_test_loss = float('nan')
        epoch_test_auroc = float('nan')

        if is_best:
            best_val_loss = val_loss
            best_epoch = last_epoch
            torch.save({
                'epoch': last_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_auroc': val_auroc,
                'task_config': asdict(task_config),
            }, best_ckpt_path)
            logger.info(f"  New best val_loss={val_loss:.4f} — writing best_model.ckpt + predictions")
            tl, ta, n_tr = run_and_save_predictions(model, train_loader, device, train_pred_path, pos_weight, logger)
            vl, va, n_va = run_and_save_predictions(model, val_loader,   device, val_pred_path,   pos_weight, logger)
            tel, tea, n_te = run_and_save_predictions(model, test_loader, device, test_pred_path, pos_weight, logger)
            epoch_test_loss, epoch_test_auroc = tel, tea
            best_metrics = {
                'best_epoch': best_epoch,
                'train_loss_pred': tl, 'train_auroc_pred': ta,
                'val_loss_pred': vl,   'val_auroc_pred': va,
                'test_loss': tel,      'test_auroc': tea,
                'n_train': n_tr, 'n_val': n_va, 'n_test': n_te,
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if is_snapshot:
            logger.info(f"  Snapshot @ epoch {last_epoch}")
            snap_train = os.path.join(task_output_dir, f"train_predictions_epoch{last_epoch}.csv")
            snap_val   = os.path.join(task_output_dir, f"val_predictions_epoch{last_epoch}.csv")
            snap_test  = os.path.join(task_output_dir, f"test_predictions_epoch{last_epoch}.csv")
            run_and_save_predictions(model, train_loader, device, snap_train, pos_weight, logger)
            run_and_save_predictions(model, val_loader,   device, snap_val,   pos_weight, logger)
            stel, stea, _ = run_and_save_predictions(model, test_loader, device, snap_test, pos_weight, logger)
            if not is_best:
                epoch_test_loss = stel
                epoch_test_auroc = stea

        history['epoch'].append(last_epoch)
        history['train_loss'].append(train_loss)
        history['train_auroc'].append(train_auroc)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)
        history['test_loss'].append(epoch_test_loss)
        history['test_auroc'].append(epoch_test_auroc)
        history['is_best'].append(bool(is_best))
        history['is_snapshot'].append(bool(is_snapshot))

        if patience_counter >= args.early_stopping_patience:
            logger.info(f"Early stopping after {last_epoch} epochs "
                        f"(no val_loss improvement for {args.early_stopping_patience} epochs)")
            break

    history_df = pd.DataFrame(history)
    history_path = os.path.join(task_output_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    logger.info(f"Training history saved: {history_path}")

    summary_row = {
        'task_key': task_config.task_key,
        'display_name': task_config.display_name,
        'label_column': task_config.label_column,
        'positive_operator': task_config.positive_operator,
        'threshold': task_config.threshold,
        'best_epoch': best_metrics.get('best_epoch', -1),
        'best_val_loss': best_val_loss,
        'train_loss': best_metrics.get('train_loss_pred', float('nan')),
        'train_auroc': best_metrics.get('train_auroc_pred', float('nan')),
        'val_loss': best_metrics.get('val_loss_pred', float('nan')),
        'val_auroc': best_metrics.get('val_auroc_pred', float('nan')),
        'test_loss': best_metrics.get('test_loss', float('nan')),
        'test_auroc': best_metrics.get('test_auroc', float('nan')),
        'n_train': best_metrics.get('n_train', -1),
        'n_val': best_metrics.get('n_val', -1),
        'n_test': best_metrics.get('n_test', -1),
        'pos_weight': float(pos_weight.item()),
        'total_epochs': last_epoch,
        'train_pred_path': os.path.abspath(train_pred_path),
        'val_pred_path': os.path.abspath(val_pred_path),
        'test_pred_path': os.path.abspath(test_pred_path),
    }
    summary_path = os.path.join(task_output_dir, "metrics_summary.csv")
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    logger.info(f"Metrics summary saved: {summary_path}")
    logger.info(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f}, "
                f"test_auroc={summary_row['test_auroc']})")


if __name__ == "__main__":
    main()
