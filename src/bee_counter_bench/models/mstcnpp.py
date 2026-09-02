"""Acausal MS-TCN++ reference topology adapted to one-dimensional signals."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _mask_3d(mask: Tensor | None, reference: Tensor) -> Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 2:
        mask = mask.unsqueeze(1)
    if (
        mask.ndim != 3
        or mask.shape[0] != reference.shape[0]
        or mask.shape[-1] != reference.shape[-1]
    ):
        raise ValueError("mask must have shape [batch, time] or [batch, 1, time].")
    if mask.shape[1] != 1:
        raise ValueError("mask channel dimension must equal one.")
    return mask.to(device=reference.device, dtype=reference.dtype)


class PredictionGeneration(nn.Module):
    """MS-TCN++ Prediction Generation with complementary dilations."""

    def __init__(
        self,
        num_layers: int,
        feature_maps: int,
        input_channels: int,
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if min(num_layers, feature_maps, input_channels, num_classes) <= 0:
            raise ValueError("Prediction Generation dimensions must be positive.")
        self.num_layers = num_layers
        self.conv_1x1_in = nn.Conv1d(input_channels, feature_maps, 1)
        self.conv_dilated_1 = nn.ModuleList(
            nn.Conv1d(
                feature_maps,
                feature_maps,
                3,
                padding=2 ** (num_layers - 1 - index),
                dilation=2 ** (num_layers - 1 - index),
            )
            for index in range(num_layers)
        )
        self.conv_dilated_2 = nn.ModuleList(
            nn.Conv1d(
                feature_maps,
                feature_maps,
                3,
                padding=2**index,
                dilation=2**index,
            )
            for index in range(num_layers)
        )
        self.conv_fusion = nn.ModuleList(
            nn.Conv1d(2 * feature_maps, feature_maps, 1) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(feature_maps, num_classes, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.conv_1x1_in(inputs)
        for index in range(self.num_layers):
            residual = features
            features = self.conv_fusion[index](
                torch.cat(
                    (
                        self.conv_dilated_1[index](features),
                        self.conv_dilated_2[index](features),
                    ),
                    dim=1,
                )
            )
            features = self.dropout(F.relu(features))
            features = features + residual
        return self.conv_out(features)


class DilatedResidualLayer(nn.Module):
    """Single-dilation refinement layer from the pinned upstream model."""

    def __init__(self, dilation: int, feature_maps: int, dropout: float = 0.5) -> None:
        super().__init__()
        if dilation <= 0 or feature_maps <= 0:
            raise ValueError("Dilation and feature maps must be positive.")
        self.conv_dilated = nn.Conv1d(
            feature_maps,
            feature_maps,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv_1x1 = nn.Conv1d(feature_maps, feature_maps, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        output = F.relu(self.conv_dilated(inputs))
        output = self.dropout(self.conv_1x1(output))
        return inputs + output


class Refinement(nn.Module):
    def __init__(
        self,
        num_layers: int,
        feature_maps: int,
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if min(num_layers, feature_maps, num_classes) <= 0:
            raise ValueError("Refinement dimensions must be positive.")
        self.conv_1x1 = nn.Conv1d(num_classes, feature_maps, 1)
        self.layers = nn.ModuleList(
            DilatedResidualLayer(2**index, feature_maps, dropout)
            for index in range(num_layers)
        )
        self.conv_out = nn.Conv1d(feature_maps, num_classes, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.conv_1x1(inputs)
        for layer in self.layers:
            output = layer(output)
        return self.conv_out(output)


class MSTCNPP(nn.Module):
    """Multi-stage acausal reference model; production causality comes later."""

    def __init__(
        self,
        prediction_generation_layers: int,
        refinement_layers: int,
        refinement_stages: int,
        feature_maps: int,
        input_channels: int,
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if refinement_stages <= 0:
            raise ValueError("At least one refinement stage is required.")
        self.PG = PredictionGeneration(
            prediction_generation_layers,
            feature_maps,
            input_channels,
            num_classes,
            dropout,
        )
        self.Rs = nn.ModuleList(
            Refinement(refinement_layers, feature_maps, num_classes, dropout)
            for _ in range(refinement_stages)
        )

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, channels, time].")
        valid = _mask_3d(mask, inputs)
        model_input = inputs if valid is None else inputs * valid
        output = self.PG(model_input)
        if valid is not None:
            output = output * valid
        outputs = [output]
        for refinement in self.Rs:
            probabilities = F.softmax(output, dim=1)
            if valid is not None:
                probabilities = probabilities * valid
            output = refinement(probabilities)
            if valid is not None:
                output = output * valid
            outputs.append(output)
        return torch.stack(outputs, dim=0)
