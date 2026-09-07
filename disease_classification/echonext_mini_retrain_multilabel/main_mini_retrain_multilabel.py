#!/usr/bin/env python
"""EchoNext-Mini retrain multilabel training entry point."""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from loader_ecg_mini_retrain_multilabel import (
    create_dataloaders,
    prepare_train_internal_val_split,
)
from model_echonext_mini_retrain_multilabel import ECGEchoNextMiniRetrainMultiLabel


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, received {value!r}.")


def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    log_file = os.path.join(output_dir, "logs", "training.log")

    logger = logging.getLogger("echonext_mini_retrain_multilabel")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def compute_metrics_per_label(
    all_probs: np.ndarray, all_labels: np.ndarray, threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    num_labels = all_probs.shape[1]
    metrics = {
        "auroc": np.zeros(num_labels, dtype=np.float32),
        "auprc": np.zeros(num_labels, dtype=np.float32),
        "f1": np.zeros(num_labels, dtype=np.float32),
        "tpr": np.zeros(num_labels, dtype=np.float32),
        "tnr": np.zeros(num_labels, dtype=np.float32),
        "fpr": np.zeros(num_labels, dtype=np.float32),
        "fnr": np.zeros(num_labels, dtype=np.float32),
    }

    for idx in range(num_labels):
        y_true = all_labels[:, idx]
        y_prob = all_probs[:, idx]
        y_pred = (y_prob >= threshold).astype(int)

        if len(np.unique(y_true)) > 1:
            metrics["auroc"][idx] = roc_auc_score(y_true, y_prob)
            metrics["auprc"][idx] = average_precision_score(y_true, y_prob)

        metrics["f1"][idx] = f1_score(y_true, y_pred, zero_division=0.0)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        metrics["tpr"][idx] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics["tnr"][idx] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["fpr"][idx] = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["fnr"][idx] = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return metrics


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: Optional[torch.Tensor],
    device: torch.device,
) -> Dict[str, object]:
    model.train()
    total_loss = 0.0
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in train_loader:
        ecg = batch["ecg_raw"].to(device)
        tabular = batch["tabular_7"].to(device)
        labels = batch["disease_labels"].to(device)

        optimizer.zero_grad()
        logits = model(ecg, tabular)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_probs_np = np.vstack(all_probs)
    all_labels_np = np.vstack(all_labels)
    return {
        "loss": total_loss / max(len(train_loader), 1),
        "metrics": compute_metrics_per_label(all_probs_np, all_labels_np),
    }


def validate_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    pos_weight: Optional[torch.Tensor],
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            ecg = batch["ecg_raw"].to(device)
            tabular = batch["tabular_7"].to(device)
            labels = batch["disease_labels"].to(device)

            logits = model(ecg, tabular)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            total_loss += loss.item()

            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs_np = np.vstack(all_probs)
    all_labels_np = np.vstack(all_labels)
    return {
        "loss": total_loss / max(len(data_loader), 1),
        "metrics": compute_metrics_per_label(all_probs_np, all_labels_np),
    }


def save_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_path: str,
    disease_names: List[str],
) -> str:
    model.eval()
    all_eids: List[int] = []
    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            ecg = batch["ecg_raw"].to(device)
            tabular = batch["tabular_7"].to(device)
            logits = model(ecg, tabular)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_eids.extend(batch["eid"].cpu().numpy().tolist())
            all_labels.append(batch["disease_labels"].cpu().numpy())
            all_probs.append(probs)

    labels_np = np.vstack(all_labels)
    probs_np = np.vstack(all_probs)

    df_dict: Dict[str, object] = {"eid": np.asarray(all_eids, dtype=np.int64)}
    for idx, disease in enumerate(disease_names):
        df_dict[f"{disease}_true"] = labels_np[:, idx]
    for idx, disease in enumerate(disease_names):
        df_dict[f"{disease}_prob"] = probs_np[:, idx]

    predictions_df = pd.DataFrame(df_dict)
    expected_rows = len(data_loader.dataset)
    if len(predictions_df) != expected_rows:
        raise RuntimeError(
            "Prediction row count does not match dataset size: "
            f"predictions={len(predictions_df)}, dataset={expected_rows}, output={output_path}"
        )
    predictions_df.to_csv(output_path, index=False)
    return output_path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    epoch: int,
    selection_name: str,
    selection_value: float,
    disease_names: List[str],
    model_config: Dict[str, object],
) -> None:
    torch.save(
        {
            "epoch": epoch + 1,
            "selection_name": selection_name,
            "selection_value": selection_value,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "disease_labels": disease_names,
            "model_config": model_config,
        },
        checkpoint_path,
    )


def append_history_rows(
    history_rows: List[Dict[str, float]],
    epoch: int,
    train_results: Dict[str, object],
    internal_val_results: Dict[str, object],
    external_test_results: Dict[str, object],
    disease_names: List[str],
) -> None:
    row: Dict[str, float] = {
        "epoch": float(epoch + 1),
        "train_loss": float(train_results["loss"]),
        "internal_val_loss": float(internal_val_results["loss"]),
        "external_test_loss": float(external_test_results["loss"]),
        "train_auroc_mean": float(np.mean(train_results["metrics"]["auroc"])),
        "internal_val_auroc_mean": float(np.mean(internal_val_results["metrics"]["auroc"])),
        "external_test_auroc_mean": float(np.mean(external_test_results["metrics"]["auroc"])),
        "train_auprc_mean": float(np.mean(train_results["metrics"]["auprc"])),
        "internal_val_auprc_mean": float(np.mean(internal_val_results["metrics"]["auprc"])),
        "external_test_auprc_mean": float(np.mean(external_test_results["metrics"]["auprc"])),
    }

    for split_name, results in (
        ("train", train_results),
        ("internal_val", internal_val_results),
        ("external_test", external_test_results),
    ):
        for metric_name in ("auroc", "auprc", "f1", "tpr", "tnr", "fpr", "fnr"):
            values = results["metrics"][metric_name]
            for idx, disease in enumerate(disease_names):
                row[f"{split_name}_{metric_name}_{disease}"] = float(values[idx])

    history_rows.append(row)


def log_per_label_metrics(
    logger: logging.Logger,
    split_name: str,
    metrics: Dict[str, np.ndarray],
    disease_names: List[str],
) -> None:
    logger.info("  %s Per-Label Metrics:", split_name)
    logger.info(
        "  %-15s %6s %6s %6s %6s %6s %6s %6s", "Disease", "AUROC", "AUPRC", "F1", "TPR", "TNR", "FPR", "FNR"
    )
    for idx, disease_name in enumerate(disease_names):
        logger.info(
            "  %-15s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f",
            disease_name,
            metrics["auroc"][idx], metrics["auprc"][idx], metrics["f1"][idx],
            metrics["tpr"][idx], metrics["tnr"][idx], metrics["fpr"][idx], metrics["fnr"][idx],
        )
    logger.info(
        "  %-15s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f",
        "MEAN",
        float(np.mean(metrics["auroc"])), float(np.mean(metrics["auprc"])), float(np.mean(metrics["f1"])),
        float(np.mean(metrics["tpr"])), float(np.mean(metrics["tnr"])), float(np.mean(metrics["fpr"])),
        float(np.mean(metrics["fnr"])),
    )


def format_median_range(values: pd.Series) -> str:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        return "N/A"
    return (
        f"{float(np.median(numeric_values)):.4f} "
        f"[{float(np.min(numeric_values)):.4f}, {float(np.max(numeric_values)):.4f}]"
    )


def save_median_over_epochs_summary(
    history_rows: List[Dict[str, float]], output_dir: str, logger: logging.Logger,
) -> str:
    if not history_rows:
        raise ValueError("Cannot write summary before any training history exists.")

    history_df = pd.DataFrame(history_rows)
    summary_row: Dict[str, object] = {
        "summary": "median_over_epochs",
        "num_epochs": int(len(history_df)),
    }
    for column in history_df.columns:
        if column == "epoch":
            continue
        summary_row[column] = format_median_range(history_df[column])

    summary_path = os.path.join(output_dir, "median_over_epochs_summary.csv")
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    logger.info("Saved median performance summary: %s", summary_path)
    return summary_path


def save_split_predictions(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    internal_val_loader: torch.utils.data.DataLoader,
    external_test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    disease_names: List[str],
    suffix: str,
    logger: logging.Logger,
) -> None:
    tag = f"_{suffix}" if suffix else ""
    train_path = os.path.join(output_dir, f"train_predictions{tag}.csv")
    internal_val_path = os.path.join(output_dir, f"internal_val_predictions{tag}.csv")
    external_test_path = os.path.join(output_dir, f"external_test_predictions{tag}.csv")
    save_predictions(model, train_loader, device, train_path, disease_names)
    save_predictions(model, internal_val_loader, device, internal_val_path, disease_names)
    save_predictions(model, external_test_loader, device, external_test_path, disease_names)
    logger.info("Saved predictions: %s", train_path)
    logger.info("Saved predictions: %s", internal_val_path)
    logger.info("Saved predictions: %s", external_test_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EchoNext-Mini retrain multilabel training")

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--external_test_csv", type=str, required=True)
    parser.add_argument("--preprocessed_ecg", type=str, required=True)
    parser.add_argument("--ecg_phenotypes_path", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--tabular_transform_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--diseases", nargs="+", required=True)

    parser.add_argument("--internal_split_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID; use -1 for CPU.")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--filter_size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument(
        "--use_pos_weight",
        type=parse_bool,
        default=True,
        metavar="{true,false}",
        help="Apply positive-class weighting (default: true).",
    )
    parser.add_argument(
        "--save_val_best_auc_predictions",
        type=parse_bool,
        default=False,
        metavar="{true,false}",
        help="Additionally track best internal-val mean AUROC and save its predictions.",
    )
    parser.add_argument(
        "--no_dt_suffix",
        action="store_true",
        help="Write directly into --output_dir instead of a timestamped subdirectory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    disease_names = list(args.diseases)

    if args.no_dt_suffix:
        experiment_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(
            args.output_dir, f"mini_retrain_multilabel_{len(disease_names)}labels_intext_{timestamp}",
        )
    os.makedirs(experiment_dir, exist_ok=True)

    logger = setup_logging(experiment_dir)
    logger.info("=" * 80)
    logger.info("ECHONEXT-MINI RETRAIN MULTILABEL")
    logger.info("=" * 80)
    logger.info("Output directory: %s", experiment_dir)
    logger.info("Disease labels: %s", disease_names)
    logger.info("Train CSV: %s", args.train_csv)
    logger.info("External test CSV: %s", args.external_test_csv)
    logger.info("Weights path: %s", args.weights_path)
    logger.info("Tabular transform: %s", args.tabular_transform_path)
    logger.info("Optimizer: Adam")
    logger.info("Learning rate: %s", args.lr)
    logger.info("Batch size: %s", args.batch_size)
    logger.info("Weight decay: %s", args.weight_decay)
    logger.info("Max epochs: %s", args.epochs)
    logger.info("Early stopping patience: %s", args.early_stopping_patience)
    logger.info("LR scheduler: None")
    logger.info("Freeze backbone: True")
    logger.info("Use pos_weight: %s", args.use_pos_weight)
    logger.info("Save val-best-AUC predictions: %s", args.save_val_best_auc_predictions)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(args.gpu_id))
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")

    logger.info("\nPreparing train/internal validation split...")
    train_split_df, internal_val_split_df, split_statistics = prepare_train_internal_val_split(
        train_csv_path=args.train_csv,
        disease_label_columns=disease_names,
        internal_split_ratio=args.internal_split_ratio,
        seed=args.seed,
        output_dir=experiment_dir,
        split_logger=logger,
    )

    logger.info("\nCreating dataloaders...")
    train_loader, internal_val_loader, external_test_loader, pos_weight = create_dataloaders(
        train_csv_path=train_split_df,
        internal_val_csv_path=internal_val_split_df,
        external_test_csv_path=args.external_test_csv,
        preprocessed_ecg_path=args.preprocessed_ecg,
        ecg_phenotypes_path=args.ecg_phenotypes_path,
        disease_label_columns=disease_names,
        tabular_transform_path=args.tabular_transform_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    final_train_rows = int(len(train_loader.dataset))
    final_internal_val_rows = int(len(internal_val_loader.dataset))
    final_train_val_rows = final_train_rows + final_internal_val_rows
    external_test_dataset = external_test_loader.dataset
    external_test_input_rows = int(external_test_dataset.input_row_count)
    external_test_excluded_rows = int(external_test_dataset.excluded_row_count)
    external_test_retained_rows = int(external_test_dataset.retained_row_count)

    if split_statistics["input_rows"] != (
        split_statistics["excluded_rows"] + split_statistics["retained_rows"]
    ):
        raise RuntimeError(f"Training filtering statistics are inconsistent: {split_statistics}")
    if split_statistics["retained_rows"] != final_train_val_rows:
        raise RuntimeError(
            "Filtered training pool does not match final train/internal-validation datasets: "
            f"retained={split_statistics['retained_rows']}, final_train_val={final_train_val_rows}"
        )
    if split_statistics["train_rows"] != final_train_rows:
        raise RuntimeError(
            "Training split size changed during dataset construction: "
            f"split={split_statistics['train_rows']}, dataset={final_train_rows}"
        )
    if split_statistics["internal_val_rows"] != final_internal_val_rows:
        raise RuntimeError(
            "Internal-validation split size changed during dataset construction: "
            f"split={split_statistics['internal_val_rows']}, dataset={final_internal_val_rows}"
        )
    if external_test_input_rows != external_test_excluded_rows + external_test_retained_rows:
        raise RuntimeError(
            "External-test filtering counts are inconsistent: "
            f"input={external_test_input_rows}, excluded={external_test_excluded_rows}, "
            f"retained={external_test_retained_rows}"
        )
    if external_test_retained_rows != len(external_test_loader.dataset):
        raise RuntimeError(
            "External-test retained size does not match its dataset: "
            f"retained={external_test_retained_rows}, dataset={len(external_test_loader.dataset)}"
        )

    logger.info("\n%s", "=" * 80)
    logger.info("DATA FILTERING AND PREDICTION SIZE SUMMARY")
    logger.info("%s", "=" * 80)
    logger.info("Training source input rows: %d", split_statistics["input_rows"])
    logger.info("Training source rows excluded (all selected labels = 0): %d", split_statistics["excluded_rows"])
    logger.info("Training source rows retained after filtering: %d", split_statistics["retained_rows"])
    logger.info("Final training split rows: %d", final_train_rows)
    logger.info("Final internal-validation split rows: %d", final_internal_val_rows)
    logger.info("Training + internal-validation rows: %d", final_train_val_rows)
    logger.info("External-test input rows: %d", external_test_input_rows)
    logger.info("External-test rows excluded (all selected labels = 0): %d", external_test_excluded_rows)
    logger.info("External-test rows retained after filtering: %d", external_test_retained_rows)
    logger.info("%s", "=" * 80)

    if args.use_pos_weight:
        pos_weight = pos_weight.to(device)
        logger.info("pos_weight: %s", [round(float(x), 4) for x in pos_weight.cpu().numpy()])
    else:
        pos_weight = None
        logger.info("pos_weight: disabled")

    logger.info("\nInitializing model...")
    model = ECGEchoNextMiniRetrainMultiLabel(
        weights_path=args.weights_path,
        disease_labels=disease_names,
        freeze_backbone=True,
        device=str(device),
        filter_size=args.filter_size,
        dropout=args.dropout,
    )
    model.to(device)

    parameter_counts = model.get_parameter_count()
    logger.info(
        "Parameters: total=%s, trainable=%s, frozen=%s",
        f"{parameter_counts['total']:,}",
        f"{parameter_counts['trainable']:,}",
        f"{parameter_counts['frozen']:,}",
    )

    optimizer = optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
    )

    model_config = {
        "weights_path": args.weights_path,
        "tabular_transform_path": args.tabular_transform_path,
        "filter_size": args.filter_size,
        "dropout": args.dropout,
        "freeze_backbone": True,
        "len_tabular_feature_vector": 7,
        "disease_labels": disease_names,
        "use_pos_weight": bool(args.use_pos_weight),
    }

    history_rows: List[Dict[str, float]] = []
    best_val_loss = np.inf
    best_val_loss_epoch = -1
    best_val_auc = -np.inf
    best_val_auc_epoch = -1
    patience_counter = 0

    best_val_loss_checkpoint = os.path.join(experiment_dir, "best_checkpoint.ckpt")
    best_val_auc_checkpoint = os.path.join(experiment_dir, "best_checkpoint_valauc.ckpt")

    logger.info("\nStarting training...")
    for epoch in range(args.epochs):
        logger.info("\nEpoch %d/%d", epoch + 1, args.epochs)
        logger.info("-" * 40)

        train_results = train_epoch(model, train_loader, optimizer, pos_weight, device)
        internal_val_results = validate_epoch(model, internal_val_loader, pos_weight, device)
        external_test_results = validate_epoch(model, external_test_loader, pos_weight, device)

        logger.info("Train loss: %.4f", float(train_results["loss"]))
        log_per_label_metrics(logger, "TRAIN", train_results["metrics"], disease_names)
        logger.info("Internal val loss: %.4f", float(internal_val_results["loss"]))
        log_per_label_metrics(logger, "INTERNAL VAL", internal_val_results["metrics"], disease_names)
        logger.info("External test loss: %.4f", float(external_test_results["loss"]))
        log_per_label_metrics(logger, "EXTERNAL TEST", external_test_results["metrics"], disease_names)

        append_history_rows(
            history_rows, epoch, train_results, internal_val_results, external_test_results, disease_names,
        )

        current_val_loss = float(internal_val_results["loss"])
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_val_loss_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=best_val_loss_checkpoint,
                epoch=epoch,
                selection_name="internal_val_loss",
                selection_value=current_val_loss,
                disease_names=disease_names,
                model_config=model_config,
            )
            logger.info("Internal validation loss improved to %.4f; saved %s",
                        best_val_loss, best_val_loss_checkpoint)
        else:
            patience_counter += 1
            logger.info(
                "Internal validation loss did not improve. Patience %d/%d",
                patience_counter, args.early_stopping_patience,
            )

        if args.save_val_best_auc_predictions:
            current_val_auc = float(np.mean(internal_val_results["metrics"]["auroc"]))
            if current_val_auc > best_val_auc:
                best_val_auc = current_val_auc
                best_val_auc_epoch = epoch
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    checkpoint_path=best_val_auc_checkpoint,
                    epoch=epoch,
                    selection_name="internal_val_auroc_mean",
                    selection_value=current_val_auc,
                    disease_names=disease_names,
                    model_config=model_config,
                )
                save_split_predictions(
                    model=model,
                    train_loader=train_loader,
                    internal_val_loader=internal_val_loader,
                    external_test_loader=external_test_loader,
                    device=device,
                    output_dir=experiment_dir,
                    disease_names=disease_names,
                    suffix="valauc",
                    logger=logger,
                )
                logger.info("Internal val mean AUROC improved to %.4f; saved %s",
                            best_val_auc, best_val_auc_checkpoint)

        pd.DataFrame(history_rows).to_csv(os.path.join(experiment_dir, "training_history.csv"), index=False)

        if patience_counter >= args.early_stopping_patience:
            logger.info(
                "Early stopping triggered after %d epochs due to internal validation loss plateau.",
                epoch + 1,
            )
            break

    save_median_over_epochs_summary(history_rows, experiment_dir, logger)

    if not os.path.exists(best_val_loss_checkpoint):
        raise RuntimeError(f"Missing best-val-loss checkpoint: {best_val_loss_checkpoint}")
    logger.info("\nReloading best-val-loss checkpoint (epoch %d, val_loss %.4f)",
                best_val_loss_epoch + 1, best_val_loss)
    reloaded = torch.load(best_val_loss_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(reloaded["model_state_dict"])
    save_split_predictions(
        model=model,
        train_loader=train_loader,
        internal_val_loader=internal_val_loader,
        external_test_loader=external_test_loader,
        device=device,
        output_dir=experiment_dir,
        disease_names=disease_names,
        suffix="",
        logger=logger,
    )

    logger.info("\nTraining finished.")
    logger.info("Best internal validation loss: %.4f (epoch %d)", best_val_loss, best_val_loss_epoch + 1)
    if args.save_val_best_auc_predictions:
        logger.info("Best internal val mean AUROC: %.4f (epoch %d)", best_val_auc, best_val_auc_epoch + 1)


if __name__ == "__main__":
    main()
