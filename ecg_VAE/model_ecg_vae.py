"""Pure-PyTorch variational autoencoder for 12-lead ECG signals."""

from pathlib import Path
from typing import Dict, Optional

import matplotlib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


class ECGVAEModel(nn.Module):
    """Gaussian variational autoencoder for 12-lead ECG data."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float,
        beta: float,
        std_is_log: bool,
        checkpoint_dir: str,
        plot_dir: str,
        num_epochs: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.beta_max = beta
        self.std_is_log = std_is_log
        self.checkpoint_dir = checkpoint_dir
        self.plot_dir = plot_dir
        self.num_epochs = num_epochs

        self.epoch_recon_loss = []
        self.epoch_kld_loss = []
        self.epoch_loss = []
        self.epoch_rho = []

        self.best_val_loss = float("inf")
        self.best_val_rho = 0.0
        self.best_epoch = 0
        self.epoch = 0
        self._n_data = 0
        self.outputs = []

    def set_n_data(self, n_data: int) -> None:
        self._n_data = n_data

    @property
    def n_data(self) -> int:
        return self._n_data

    def reparameterize(
        self,
        mu: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        if self.std_is_log:
            std = std.exp()
        eps = torch.normal(torch.zeros_like(mu), torch.ones_like(std))
        return mu + eps * std

    def forward(self, batch, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        x = batch["signal"] if isinstance(batch, dict) else batch
        if x.ndim == 2:
            x = x.unsqueeze(0)

        mu, std = self.encoder(x)
        z = mu if deterministic else self.reparameterize(mu=mu, std=std)
        reconstruction_mean, reconstruction_std = self.decoder(z)

        if x.shape != reconstruction_mean.shape:
            raise ValueError(
                f"Reconstruction mean shape {tuple(reconstruction_mean.shape)} "
                f"does not match input shape {tuple(x.shape)}"
            )
        if x.shape != reconstruction_std.shape:
            raise ValueError(
                f"Reconstruction standard-deviation shape "
                f"{tuple(reconstruction_std.shape)} does not match input shape "
                f"{tuple(x.shape)}"
            )

        return {
            "x": x,
            "reconstruction": reconstruction_mean,
            "reconstruction_mean": reconstruction_mean,
            "reconstruction_std": reconstruction_std,
            "z": z,
            "mu": mu,
            "std": std,
        }

    def loss_function(self, output: Dict[str, torch.Tensor]):
        std = output["std"].exp() if self.std_is_log else output["std"]
        kld_loss = -0.5 * torch.sum(
            1
            + std.pow(2).log()
            - output["mu"].pow(2)
            - std.pow(2),
            dim=1,
        ).mean()
        reconstruction_loss = F.mse_loss(
            output["reconstruction"],
            output["x"],
            reduction="mean",
        )
        total_loss = reconstruction_loss + self.beta_max * kld_loss
        return reconstruction_loss, kld_loss, total_loss

    @staticmethod
    def pearson_correlation(
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x = x - x.mean()
        y = y - y.mean()
        denominator = torch.sqrt((x**2).sum() * (y**2).sum())
        return (x * y).sum() / denominator

    def save_checkpoint(
        self,
        optimizer,
        epoch: int,
        val_loss,
        val_rho,
    ) -> Path:
        if self.checkpoint_dir is None:
            raise ValueError("Checkpoint directory is not configured")

        checkpoint_path = Path(self.checkpoint_dir) / "ecg_vae_best.ckpt"
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": float(val_loss),
            "rho": float(val_rho),
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def plot_reconstructions(
        self,
        epoch: int,
        save_dir: Optional[str] = None,
    ) -> None:
        if self.plot_dir is None and save_dir is None:
            raise ValueError("Plot directory is not configured")

        plot_save_dir = (
            Path(save_dir)
            if save_dir is not None
            else Path(self.plot_dir) / f"epoch_{epoch}"
        )
        plot_save_dir.mkdir(parents=True, exist_ok=True)

        for sample_index, (true_values, reconstructed_values) in enumerate(
            self.outputs,
            start=1,
        ):
            true_values = true_values.detach().cpu().numpy()
            reconstructed_values = reconstructed_values.detach().cpu().numpy()

            figure, axes = plt.subplots(
                nrows=3,
                ncols=4,
                figsize=(16, 12),
                dpi=300,
            )
            for lead_index, axis in enumerate(axes.flat):
                axis.plot(
                    true_values[lead_index].reshape(-1),
                    label="Original",
                )
                axis.plot(
                    reconstructed_values[lead_index].reshape(-1),
                    label="Reconstruction",
                )
                axis.set_title(LEAD_NAMES[lead_index])
                axis.legend()

            figure.tight_layout()
            figure.savefig(
                plot_save_dir
                / f"ecg_vae_reconstruction_{sample_index}.png"
            )
            plt.close(figure)

    def results_table(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {
                "loss": self.epoch_loss,
                "recon_loss": self.epoch_recon_loss,
                "kld_loss": self.epoch_kld_loss,
                "rho": self.epoch_rho,
            }
        )
