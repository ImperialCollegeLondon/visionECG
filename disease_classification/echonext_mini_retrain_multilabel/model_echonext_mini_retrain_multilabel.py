#!/usr/bin/env python
"""EchoNext-Mini retrain model with tabular fusion."""

import os
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn


class BasicBlock1d(nn.Module):
    """1D residual block used by the EchoNext ResNet backbone."""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout: float = 0.5,
        kernel_size: int = 7,
        padding: int = 3,
        bias: bool = False,
        inplace: bool = True,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=inplace)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=bias
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet1dWithTabular(nn.Module):
    """EchoNext ResNet1d backbone concatenated with a tabular feature vector."""

    def __init__(
        self,
        len_tabular_feature_vector: int = 7,
        filter_size: int = 16,
        input_channels: int = 12,
        dropout_value: float = 0.5,
        num_classes: int = 12,
        conv1_kernel_size: int = 15,
        conv1_stride: int = 2,
        padding: int = 7,
        bias: bool = False,
    ):
        super().__init__()
        self.inplanes = filter_size
        self.layers = [3, 4, 6, 3]
        self.conv1 = nn.Conv1d(
            input_channels,
            self.inplanes,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=padding,
            bias=bias,
        )
        self.dropout_value = dropout_value
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock1d, filter_size, self.layers[0])
        self.layer2 = self._make_layer(BasicBlock1d, 2 * filter_size, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1d, 4 * filter_size, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1d, 8 * filter_size, self.layers[3], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_value)
        intermediate_dim = 8 * filter_size * BasicBlock1d.expansion * 2 + len_tabular_feature_vector
        self.output = nn.Linear(intermediate_dim, num_classes)

    def _make_layer(
        self, block: Type[nn.Module], planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, dropout=self.dropout_value)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout=self.dropout_value))
        return nn.Sequential(*layers)

    def forward(self, x_and_tabular: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, tabular = x_and_tabular
        x = torch.transpose(x, 2, 3)
        x = torch.squeeze(x, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, tabular), dim=1)
        return self.output(x)


class ECGEchoNextMiniRetrainMultiLabel(nn.Module):
    """Retrain the pretrained EchoNext-Mini output layer."""

    def __init__(
        self,
        weights_path: str,
        disease_labels: List[str],
        freeze_backbone: bool = True,
        device: str = "cpu",
        filter_size: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()

        if filter_size != 16:
            raise ValueError(f"filter_size must be 16 for pretrained EchoNext weights, got {filter_size}")

        self.weights_path = weights_path
        self.disease_labels = list(disease_labels)
        self.freeze_backbone = freeze_backbone
        self.device_name = device

        pretrained_model = ResNet1dWithTabular(
            len_tabular_feature_vector=7,
            filter_size=filter_size,
            input_channels=12,
            dropout_value=dropout,
            num_classes=12,
            conv1_kernel_size=15,
            conv1_stride=2,
            padding=7,
        )
        self._load_pretrained_weights(pretrained_model, weights_path, device)

        self.model = ResNet1dWithTabular(
            len_tabular_feature_vector=7,
            filter_size=filter_size,
            input_channels=12,
            dropout_value=dropout,
            num_classes=len(self.disease_labels),
            conv1_kernel_size=15,
            conv1_stride=2,
            padding=7,
        )

        self._transfer_backbone_weights(pretrained_model, self.model)
        self._configure_mode()
        self.model.to(device)

    def _load_pretrained_weights(self, model: nn.Module, weights_path: str, device: str) -> None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Unexpected pretrained weight mismatch. Missing={missing_keys}, unexpected={unexpected_keys}"
            )

    def _transfer_backbone_weights(
        self, pretrained_model: ResNet1dWithTabular, new_model: ResNet1dWithTabular
    ) -> None:
        pretrained_state = pretrained_model.state_dict()
        new_state = new_model.state_dict()

        for key, value in pretrained_state.items():
            if key.startswith("output."):
                continue
            if key not in new_state:
                raise KeyError(f"Backbone parameter '{key}' missing from new model")
            new_state[key] = value

        new_model.load_state_dict(new_state, strict=False)

    def _configure_mode(self) -> None:
        if self.freeze_backbone:
            for name, param in self.model.named_parameters():
                param.requires_grad = name.startswith("output.")
        else:
            for param in self.model.parameters():
                param.requires_grad = True
        self.model.train()

    def forward(self, ecg: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        return self.model((ecg, tabular))

    def get_parameter_count(self) -> Dict[str, int]:
        total = sum(param.numel() for param in self.model.parameters())
        trainable = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
