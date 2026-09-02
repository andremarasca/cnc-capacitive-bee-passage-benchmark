"""Frame classification and truncated smoothing losses for MS-TCN++."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bee_counter_bench.constants import MASKED_LABEL


@dataclass(frozen=True)
class MultiStageLoss:
    total: Tensor
    cross_entropy: Tensor
    smoothing: Tensor
    cross_entropy_by_stage: Tensor
    smoothing_by_stage: Tensor


class MSTCNPPMultiStageLoss(nn.Module):
    """CE plus truncated temporal MSE, evaluated at every model stage."""

    def __init__(
        self,
        smoothing_weight: float = 0.15,
        smoothing_tau: float = 4.0,
        ignore_index: int = MASKED_LABEL,
    ) -> None:
        super().__init__()
        if smoothing_weight < 0 or smoothing_tau <= 0:
            raise ValueError("Invalid temporal smoothing settings.")
        self.smoothing_weight = float(smoothing_weight)
        self.smoothing_tau = float(smoothing_tau)
        self.ignore_index = int(ignore_index)

    def forward(
        self,
        stage_logits: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
    ) -> MultiStageLoss:
        if stage_logits.ndim != 4:
            raise ValueError(
                "stage_logits must have shape [stages, batch, classes, time]."
            )
        stages, batch, _classes, frames = stage_logits.shape
        if targets.shape != (batch, frames):
            raise ValueError("targets must have shape [batch, time].")
        valid = targets != self.ignore_index
        if mask is not None:
            if mask.ndim == 3 and mask.shape[1] == 1:
                mask = mask[:, 0]
            if mask.shape != targets.shape:
                raise ValueError(
                    "mask must have shape [batch, time] or [batch, 1, time]."
                )
            valid = valid & mask.to(device=targets.device, dtype=torch.bool)
        if not torch.any(valid):
            raise ValueError("Loss batch contains no valid target frames.")

        ce_values: list[Tensor] = []
        smoothing_values: list[Tensor] = []
        safe_targets = targets.masked_fill(~valid, self.ignore_index)
        transition_valid = valid[:, 1:] & valid[:, :-1]

        for logits in stage_logits:
            frame_ce = F.cross_entropy(
                logits,
                safe_targets,
                ignore_index=self.ignore_index,
                reduction="none",
            )
            ce_values.append(frame_ce[valid].mean())

            if frames < 2 or not torch.any(transition_valid):
                smoothing_values.append(logits.sum() * 0.0)
                continue
            log_probabilities = F.log_softmax(logits, dim=1)
            squared_difference = (
                log_probabilities[:, :, 1:] - log_probabilities.detach()[:, :, :-1]
            ).square()
            truncated = torch.clamp(
                squared_difference,
                min=0.0,
                max=self.smoothing_tau**2,
            )
            expanded_valid = transition_valid.unsqueeze(1).expand_as(truncated)
            smoothing_values.append(truncated[expanded_valid].mean())

        ce_by_stage = torch.stack(ce_values)
        smoothing_by_stage = torch.stack(smoothing_values)
        ce_total = ce_by_stage.sum()
        smoothing_total = smoothing_by_stage.sum()
        total = ce_total + self.smoothing_weight * smoothing_total
        if ce_by_stage.shape != (stages,):
            raise AssertionError("Unexpected loss aggregation shape.")
        return MultiStageLoss(
            total=total,
            cross_entropy=ce_total,
            smoothing=smoothing_total,
            cross_entropy_by_stage=ce_by_stage,
            smoothing_by_stage=smoothing_by_stage,
        )
