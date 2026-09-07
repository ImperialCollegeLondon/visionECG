"""Causal-CNN variational encoder for the ECG VAE."""

import torch
import torch.nn as nn

from modules_ecg_vae import CausalCNN, Softplus, SqueezeChannels


class CausalCNNVEncoder(nn.Module):
    """Encode ECGs into latent distribution parameters."""

    def __init__(
        self,
        in_channels: int,
        res_channels: int,
        depth: int,
        reduced_size: int,
        out_channels: int,
        kernel_size: int,
        softplus_eps: float,
        dropout: float,
        sd_output: bool,
    ):
        super().__init__()
        self.network = nn.Sequential(
            CausalCNN(
                in_channels,
                res_channels,
                depth,
                reduced_size,
                kernel_size,
            ),
            nn.AdaptiveMaxPool1d(1),
            SqueezeChannels(),
        )
        self.linear_mean = nn.Linear(reduced_size, out_channels)
        self.sd_output = sd_output
        if self.sd_output:
            self.linear_sd = nn.Sequential(
                nn.Linear(reduced_size, out_channels),
                Softplus(softplus_eps),
            )

    def forward(self, x: torch.Tensor):
        output = self.network(x)
        if self.sd_output:
            return self.linear_mean(output), self.linear_sd(output)
        return self.linear_mean(output).squeeze()
