"""Analytical receptive-field calculation for the compact MS-TCN++."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReceptiveField:
    past_frames: int
    future_frames: int
    total_frames: int
    past_seconds: float
    future_seconds: float
    total_seconds: float


def mstcnpp_receptive_field(
    prediction_generation_layers: int,
    refinement_layers: int,
    refinement_stages: int,
    sample_rate_hz: float,
    *,
    causal: bool = False,
) -> ReceptiveField:
    if (
        min(
            prediction_generation_layers,
            refinement_layers,
            refinement_stages,
        )
        <= 0
    ):
        raise ValueError("Layer and stage counts must be positive.")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    pg_radius = sum(
        max(2 ** (prediction_generation_layers - 1 - index), 2**index)
        for index in range(prediction_generation_layers)
    )
    refinement_radius = sum(2**index for index in range(refinement_layers))
    symmetric_radius = pg_radius + refinement_stages * refinement_radius
    if causal:
        past = 2 * symmetric_radius
        future = 0
    else:
        past = symmetric_radius
        future = symmetric_radius
    total = past + 1 + future
    return ReceptiveField(
        past_frames=past,
        future_frames=future,
        total_frames=total,
        past_seconds=past / sample_rate_hz,
        future_seconds=future / sample_rate_hz,
        total_seconds=total / sample_rate_hz,
    )
