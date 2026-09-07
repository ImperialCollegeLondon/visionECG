"""Causal convolution building blocks used by the ECG VAE."""

from typing import Callable, Optional

import torch
import torch.nn as nn


class Softplus(nn.Module):
    """Apply stable positive Softplus output."""

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus(x) + self.eps


class SqueezeChannels(nn.Module):
    """Remove the singleton temporal dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(2)


class CausalConvolutionBlock(nn.Module):
    """Dilated residual convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        num_residual_conv: int,
        act: Optional[Callable] = None,
        forward: bool = True,
    ):
        super().__init__()

        conv_op = nn.Conv1d if forward else nn.ConvTranspose1d
        padding = ((kernel_size - 1) * dilation) // 2

        layers = []
        for index in range(num_residual_conv):
            block_in_channels = in_channels if index == 0 else out_channels
            layers.extend(
                [
                    conv_op(
                        block_in_channels,
                        out_channels,
                        kernel_size,
                        padding=padding,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                ]
            )
            if index < num_residual_conv - 1:
                layers.append(nn.LeakyReLU())

        self.res_net = nn.Sequential(*layers)
        self.skip_conn = None
        if in_channels != out_channels:
            self.skip_conn = conv_op(in_channels, out_channels, 1, bias=False)
            self.skip_bnorm = nn.BatchNorm1d(out_channels)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip_conn is None:
            identity = x
        else:
            identity = self.skip_bnorm(self.skip_conn(x))

        output = self.res_net(x) + identity
        return output if self.act is None else self.act(output)


class CausalCNN(nn.Module):
    """Shared dilated residual causal network."""

    def __init__(
        self,
        in_channels: int,
        res_channels: int,
        depth: int,
        out_channels: int,
        kernel_size: int,
        forward: bool = True,
    ):
        super().__init__()

        conv_op = nn.Conv1d if forward else nn.ConvTranspose1d
        dilation_size = 1 if forward else 2**depth

        if in_channels != out_channels:
            self.final_skip_connection = conv_op(in_channels, out_channels, 1)
            self.final_batch_norm = nn.BatchNorm1d(out_channels)
        else:
            self.final_skip_connection = None

        layers = []
        for index in range(depth):
            block_in_channels = in_channels if index == 0 else res_channels
            layers.append(
                CausalConvolutionBlock(
                    in_channels=block_in_channels,
                    out_channels=res_channels,
                    kernel_size=kernel_size,
                    dilation=dilation_size,
                    num_residual_conv=2,
                    forward=forward,
                    act=nn.LeakyReLU(),
                )
            )
            dilation_size = 2 ** (index + 1) if forward else dilation_size // 2

        layers.append(
            CausalConvolutionBlock(
                in_channels=res_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation_size,
                num_residual_conv=2,
                act=None,
                forward=forward,
            )
        )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.final_skip_connection is None:
            identity = x
        else:
            identity = self.final_batch_norm(self.final_skip_connection(x))
        return nn.LeakyReLU()(self.network(x) + identity)
