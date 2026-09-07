"""Causal-CNN variational decoder for the ECG VAE."""

import torch
import torch.nn as nn

from modules_ecg_vae import CausalCNN, Softplus


class CausalCNNVDecoder(nn.Module):
    """Decode latents into ECG distribution parameters."""

    def __init__(
        self,
        k: int,
        width: int,
        in_channels: int,
        res_channels: int,
        depth: int,
        out_channels: int,
        kernel_size: int,
        gaussian_out: bool,
        softplus_eps: float,
        dropout: float,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.gaussian_out = gaussian_out
        self.linear1 = nn.Linear(k, in_channels)
        self.linear2 = nn.Linear(in_channels, in_channels * width)
        self.dropout_layer = nn.Dropout(dropout)
        self.causal_cnn = CausalCNN(
            in_channels=in_channels,
            res_channels=res_channels,
            depth=depth,
            out_channels=out_channels,
            kernel_size=kernel_size,
            forward=False,
        )
        if self.gaussian_out:
            flattened_size = out_channels * width
            self.linear_mean = nn.Linear(flattened_size, flattened_size)
            self.linear_sd = nn.Sequential(
                nn.Linear(flattened_size, flattened_size),
                Softplus(softplus_eps),
            )

    def forward(self, x: torch.Tensor):
        batch_size, _ = x.shape
        output = self.linear1(x)
        output = self.linear2(output)
        output = output.view(batch_size, self.in_channels, self.width)
        output = self.causal_cnn(output)

        if self.gaussian_out:
            output_shape = output.shape
            output = torch.flatten(output, start_dim=1)
            return (
                self.linear_mean(output).reshape(output_shape),
                self.linear_sd(output).reshape(output_shape),
            )
        return output
