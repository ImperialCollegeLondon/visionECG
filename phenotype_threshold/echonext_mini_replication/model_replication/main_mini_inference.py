import argparse
import logging
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model_echonext_mini import ECGEchoNextMini
from loader_ecg_mini import ECGEchoNextMiniDataModule


DISEASE_NAMES: List[str] = [
    "LVEF_le45",
    "LVWT_ge1.3",
    "AorticStenosis",
    "AorticRegurgitation",
    "MitralRegurgitation",
    "TricuspidRegurgitation",
    "PulmonaryRegurgitation",
    "RVDysfunction",
    "PericardialEffusion",
    "PASP_ge45",
    "TRmax_ge32",
    "SHD",
]


def setup_logging(output_dir: str) -> logging.Logger:
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "inference.log")

    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("EchoNext-Mini inference | log file: %s", log_file)
    return logger


@torch.no_grad()
def run_inference(
    model: ECGEchoNextMini,
    dataloader: torch.utils.data.DataLoader,
    device: str,
):
    model.eval()
    all_probs = []
    all_eids = []

    for batch in tqdm(dataloader, desc="Running inference"):
        ecg = batch["ecg_raw"].to(device)
        tabular_7 = batch["tabular_7"].to(device)
        eids = batch["eid"].numpy()

        logits = model(ecg, tabular_7)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_eids.append(eids)

    return np.concatenate(all_probs, axis=0), np.concatenate(all_eids, axis=0)


def save_predictions(eids: np.ndarray, probs: np.ndarray, output_path: str) -> None:
    if probs.shape[1] != 12:
        raise ValueError(f"Expected 12 disease columns, got {probs.shape[1]}")
    data = {"eid": eids}
    for i, disease_name in enumerate(DISEASE_NAMES):
        data[disease_name] = probs[:, i]
    pd.DataFrame(data).to_csv(output_path, index=False)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu"
    logger.info("Device: %s", device)
    logger.info("Weights: %s", args.weights_path)
    logger.info("Val CSV: %s", args.val_csv)
    logger.info("Preprocessed ECG: %s", args.preprocessed_ecg)
    logger.info("ECG phenotypes: %s", args.ecg_phenotypes_path)
    logger.info("Tabular transform: %s", args.tabular_transform_path)
    logger.info("Batch size: %d", args.batch_size)

    model = ECGEchoNextMini(weights_path=args.weights_path, device=device)

    data_module = ECGEchoNextMiniDataModule(
        train_csv_path=args.val_csv,
        val_csv_path=args.val_csv,
        preprocessed_ecg_path=args.preprocessed_ecg,
        ecg_phenotypes_path=args.ecg_phenotypes_path,
        label_column=args.label_column,
        threshold=args.threshold,
        threshold_direction=args.threshold_direction,
        tabular_transform_path=args.tabular_transform_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        inference_mode=True,
    )
    data_module.setup("fit")
    val_loader = data_module.val_dataloader()
    logger.info("Validation samples: %d", len(val_loader.dataset))

    all_probs, all_eids = run_inference(model, val_loader, device)

    nan_count = int(np.isnan(all_probs).sum())
    if nan_count > 0:
        logger.warning("Found %d NaN values in predictions (shape=%s)", nan_count, all_probs.shape)
    else:
        logger.info(
            "Predictions ok (shape=%s, min=%.6f, max=%.6f, mean=%.6f)",
            all_probs.shape, float(np.min(all_probs)), float(np.max(all_probs)), float(np.mean(all_probs)),
        )

    val_csv_name = os.path.splitext(os.path.basename(args.val_csv))[0]
    pred_path = os.path.join(args.output_dir, f"{val_csv_name}_echonext_prediction.csv")
    save_predictions(all_eids, all_probs, pred_path)
    logger.info("Predictions saved to: %s (n=%d)", pred_path, len(all_eids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoNext-Mini inference")

    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to pretrained weights.pt")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Path to validation CSV")
    parser.add_argument("--preprocessed_ecg", type=str, required=True,
                        help="Path to preprocessed ECG .pt file")
    parser.add_argument("--ecg_phenotypes_path", type=str, required=True,
                        help="Path to ECG morphology CSV")
    parser.add_argument("--tabular_transform_path", type=str, required=True,
                        help="Path to reference tabular_transformer.joblib (use 'none' for local scaler)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")

    parser.add_argument("--label_column", type=str, default="diseased",
                        help="Label column name (unused in inference mode; kept for loader signature)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold for continuous labels (unused in inference mode)")
    parser.add_argument("--threshold_direction", type=str, default="less_than",
                        choices=["less_than", "greater_than"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id (use -1 for CPU)")

    args = parser.parse_args()

    if args.tabular_transform_path and args.tabular_transform_path.lower() == "none":
        args.tabular_transform_path = None

    main(args)
