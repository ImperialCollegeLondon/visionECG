#!/usr/bin/env python
"""Train and validate the ECG VAE."""

import argparse
import datetime
import logging
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_ecg_vae import ECGVAEDataModule
from decoder_ecg_vae import CausalCNNVDecoder
from encoder_ecg_vae import CausalCNNVEncoder
from model_ecg_vae import ECGVAEModel


LOGGER_NAME = "ecg_VAE"
NUM_LEADS = 12
SIGNAL_WIDTH = 600


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_ecg_vae_model(config: Dict) -> ECGVAEModel:
    encoder = CausalCNNVEncoder(
        in_channels=NUM_LEADS,
        res_channels=config["res_channels"],
        reduced_size=config["reduced_size"],
        depth=config["depth"],
        out_channels=config["z_dim"],
        kernel_size=5,
        dropout=0.3,
        softplus_eps=1e-4,
        sd_output=True,
    )
    decoder = CausalCNNVDecoder(
        k=config["z_dim"],
        width=SIGNAL_WIDTH,
        in_channels=config["reduced_size"],
        res_channels=config["res_channels"],
        out_channels=NUM_LEADS,
        depth=config["depth"],
        kernel_size=5,
        gaussian_out=True,
        softplus_eps=1e-4,
        dropout=0.0,
    )
    return ECGVAEModel(
        encoder=encoder,
        decoder=decoder,
        std_is_log=False,
        lr=config["lr"],
        beta=config["beta"],
        checkpoint_dir=str(config["checkpoint_dir"]),
        plot_dir=str(config["plot_dir"]),
        num_epochs=config["epochs"],
    )


def train_ecg_vae_epoch(
    model: ECGVAEModel,
    train_loader,
    optimizer,
    device: torch.device,
    epoch: int,
):
    logger = logging.getLogger(LOGGER_NAME)
    model.train()
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    progress = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
    for batch_index, batch in enumerate(progress):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()

        output = model(batch)
        reconstruction_loss, kl_loss, loss = model.loss_function(output)
        if not torch.isfinite(loss):
            logger.warning(
                "Skipping non-finite loss at training batch %d",
                batch_index,
            )
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_reconstruction_loss += reconstruction_loss.item()
        total_kl_loss += kl_loss.item()
        num_batches += 1
        progress.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Recon": f"{reconstruction_loss.item():.4f}",
                "KL": f"{kl_loss.item():.4f}",
            }
        )

    if num_batches == 0:
        raise RuntimeError("No finite training batches were completed")
    return (
        total_loss / num_batches,
        total_reconstruction_loss / num_batches,
        total_kl_loss / num_batches,
    )


def validate_ecg_vae_epoch(
    model: ECGVAEModel,
    val_loader,
    device: torch.device,
    epoch: int,
):
    model.eval()
    model.outputs.clear()

    losses = []
    reconstruction_losses = []
    kl_losses = []
    correlations = []

    with torch.no_grad():
        progress = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]")
        for batch in progress:
            batch = batch.to(device, non_blocking=True)
            output = model(batch)
            reconstruction_loss, kl_loss, loss = model.loss_function(output)
            if not torch.isfinite(loss):
                raise RuntimeError("Validation produced a non-finite loss")

            losses.append(loss)
            reconstruction_losses.append(reconstruction_loss)
            kl_losses.append(kl_loss)

            batch_correlations = []
            for sample_index in range(batch.shape[0]):
                for lead_index in range(batch.shape[1]):
                    correlation = model.pearson_correlation(
                        batch[sample_index, lead_index],
                        output["reconstruction"][sample_index, lead_index],
                    )
                    if torch.isfinite(correlation):
                        batch_correlations.append(correlation)
            if batch_correlations:
                mean_correlation = torch.stack(batch_correlations).mean()
                correlations.append(mean_correlation)
            else:
                mean_correlation = torch.tensor(float("nan"), device=device)

            if len(model.outputs) < 5:
                model.outputs.append(
                    [
                        output["x"][0].detach(),
                        output["reconstruction"][0].detach(),
                    ]
                )

            progress.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Recon": f"{reconstruction_loss.item():.4f}",
                    "KL": f"{kl_loss.item():.4f}",
                    "Rho": (
                        f"{mean_correlation.item():.4f}"
                        if torch.isfinite(mean_correlation)
                        else "nan"
                    ),
                }
            )

    if not losses:
        raise RuntimeError("No validation batches were completed")

    average_loss = torch.stack(losses).mean()
    average_reconstruction_loss = torch.stack(reconstruction_losses).mean()
    average_kl_loss = torch.stack(kl_losses).mean()
    average_correlation = (
        torch.stack(correlations).mean()
        if correlations
        else torch.tensor(float("nan"), device=device)
    )
    return (
        float(average_loss.item()),
        float(average_reconstruction_loss.item()),
        float(average_kl_loss.item()),
        float(average_correlation.item()),
    )


def train_ecg_vae(
    model: ECGVAEModel,
    train_loader,
    val_loader,
    config: Dict,
    device: torch.device,
    start_epoch: int = 0,
    optimizer_state=None,
) -> ECGVAEModel:
    logger = logging.getLogger(LOGGER_NAME)
    model.set_n_data(len(train_loader.dataset))

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        logger.info("Loaded optimizer state from checkpoint")
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    writer = SummaryWriter(log_dir=str(config["tensorboard_dir"]))

    patience = config["early_stop_patience"]
    min_delta = config["early_stop_min_delta"]
    epochs_without_improvement = 0
    best_for_early_stopping = model.best_val_loss

    logger.info(
        "Starting training for %d epochs at epoch %d",
        config["epochs"],
        start_epoch + 1,
    )
    logger.info("Training samples: %d", len(train_loader.dataset))
    logger.info("Validation samples: %d", len(val_loader.dataset))
    logger.info("Batch size: %d", config["batch_size"])
    logger.info("Learning rate: %s", config["lr"])
    logger.info("Beta: %s", config["beta"])
    logger.info(
        "Early stopping: patience=%d, min_delta=%s",
        patience,
        min_delta,
    )

    try:
        for epoch in range(start_epoch, config["epochs"]):
            logger.info("Epoch %d/%d", epoch + 1, config["epochs"])
            train_loss, train_reconstruction_loss, train_kl_loss = (
                train_ecg_vae_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    epoch,
                )
            )
            val_loss, val_reconstruction_loss, val_kl_loss, val_rho = (
                validate_ecg_vae_epoch(model, val_loader, device, epoch)
            )
            # scheduler.step()

            model.epoch_loss.append(val_loss)
            model.epoch_recon_loss.append(val_reconstruction_loss)
            model.epoch_kld_loss.append(val_kl_loss)
            model.epoch_rho.append(val_rho)

            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar(
                "Loss/Train_Reconstruction",
                train_reconstruction_loss,
                epoch,
            )
            writer.add_scalar(
                "Loss/Validation_Reconstruction",
                val_reconstruction_loss,
                epoch,
            )
            writer.add_scalar("Loss/Train_KL", train_kl_loss, epoch)
            writer.add_scalar("Loss/Validation_KL", val_kl_loss, epoch)
            writer.add_scalar("Metrics/Validation_Rho", val_rho, epoch)
            writer.add_scalar(
                "Learning_Rate",
                optimizer.param_groups[0]["lr"],
                epoch,
            )

            logger.info(
                "Train Loss: %.6f, Recon Loss: %.6f, KL Loss: %.6f",
                train_loss,
                train_reconstruction_loss,
                train_kl_loss,
            )
            logger.info(
                "Val Loss: %.6f, Recon Loss: %.6f, KL Loss: %.6f, "
                "Rho: %.6f",
                val_loss,
                val_reconstruction_loss,
                val_kl_loss,
                val_rho,
            )

            if val_loss < model.best_val_loss:
                model.best_val_loss = val_loss
                model.best_val_rho = val_rho
                model.best_epoch = epoch
                checkpoint_path = model.save_checkpoint(
                    optimizer,
                    epoch,
                    val_loss,
                    val_rho,
                )
                model.plot_reconstructions(
                    epoch,
                    save_dir=str(config["best_plot_dir"]),
                )
                logger.info(
                    "Saved new best checkpoint to %s",
                    checkpoint_path,
                )

            if val_loss < best_for_early_stopping - min_delta:
                best_for_early_stopping = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                logger.info(
                    "No improvement for %d/%d epochs",
                    epochs_without_improvement,
                    patience,
                )

            model.epoch = epoch + 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping triggered after %d epochs without "
                    "sufficient improvement",
                    patience,
                )
                break
    finally:
        writer.close()

    if not model.epoch_loss:
        raise RuntimeError(
            "No epochs were run; --epochs must be greater than the resume epoch"
        )
    logger.info("Training completed.")
    logger.info(
        "Best validation loss: %.6f at epoch %d",
        model.best_val_loss,
        model.best_epoch + 1,
    )
    return model


def _load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def _setup_ecg_vae_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def run_ecg_vae_training(args: argparse.Namespace) -> Path:
    if args.seed < 0:
        raise ValueError(f"--seed must be non-negative, got {args.seed}")
    set_random_seed(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperparameters = (
        f"z{args.z_dim}_r{args.reduced_size}_b{args.beta}_lr{args.lr}"
        f"_rc{args.res_channels}_d{args.depth}"
    )
    run_dir = (
        Path(args.checkpoint_dir).expanduser()
        / f"ecg_vae_{timestamp}_{hyperparameters}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "beta": args.beta,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "res_channels": args.res_channels,
        "z_dim": args.z_dim,
        "reduced_size": args.reduced_size,
        "depth": args.depth,
        "checkpoint_dir": run_dir,
        "epochs": args.epochs,
        "plot_dir": run_dir / "ecg_vae_reconstruction_plots",
        "best_plot_dir": run_dir / "ecg_vae_best_epoch_plots",
        "tensorboard_dir": run_dir / "ecg_vae_tensorboard",
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
    }
    logger = _setup_ecg_vae_logger(run_dir / "ecg_vae_training.log")
    logger.info("Run directory: %s", run_dir)

    if torch.cuda.is_available():
        if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu {args.gpu} is unavailable; visible GPU count is "
                f"{torch.cuda.device_count()}"
            )
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info("  %s: %s", key, value)

    data_module = ECGVAEDataModule(
        train_pt=args.train_pt,
        val_pt=args.val_pt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    model = create_ecg_vae_model(config).to(device)
    logger.info(
        "Model parameters: %d",
        sum(parameter.numel() for parameter in model.parameters()),
    )

    start_epoch = 0
    optimizer_state = None
    if args.resume_from is not None:
        resume_path = Path(args.resume_from).expanduser()
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )
        logger.info("Loading checkpoint from %s", resume_path)
        checkpoint = _load_checkpoint(resume_path, device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer_state = checkpoint["optimizer"]
        start_epoch = int(checkpoint["epoch"]) + 1
        model.best_val_loss = float(checkpoint.get("val_loss", float("inf")))
        model.best_val_rho = float(checkpoint.get("rho", 0.0))
        model.best_epoch = int(checkpoint["epoch"])

    model = train_ecg_vae(
        model,
        train_loader,
        val_loader,
        config,
        device,
        start_epoch=start_epoch,
        optimizer_state=optimizer_state,
    )
    results_path = run_dir / "ecg_vae_loss.csv"
    model.results_table().to_csv(results_path, index=False)
    logger.info("Saved loss table to %s", results_path)
    return run_dir


def build_ecg_vae_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and validate the ECG VAE"
    )
    parser.add_argument("--train_pt", type=str, required=True)
    parser.add_argument("--val_pt", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--res_channels", type=int, required=True)
    parser.add_argument("--z_dim", type=int, required=True)
    parser.add_argument("--reduced_size", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--early_stop_patience", type=int, required=True)
    parser.add_argument("--early_stop_min_delta", type=float, required=True)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_ecg_vae_argument_parser().parse_args()
    run_ecg_vae_training(args)


if __name__ == "__main__":
    main()
