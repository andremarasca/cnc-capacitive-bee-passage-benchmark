"""Derive categorical passage labels from synchronized CNC position."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bee_counter_bench.constants import MASKED_LABEL, LEFT, OUT, RIGHT

from .schemas import Direction, LabelResult, PassageEvent


def _contiguous_true_runs(mask: NDArray[np.bool_]) -> tuple[tuple[int, int], ...]:
    changes = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def _speed_summary(
    position_mm: NDArray[np.float64],
    start_idx: int,
    end_idx: int,
    sample_rate_hz: float,
) -> tuple[float, float, float]:
    span = position_mm[start_idx - 1 : end_idx + 2]
    speeds = np.abs(np.diff(span)) * sample_rate_hz * 60.0
    if len(speeds) == 0:
        return 0.0, 0.0, 0.0
    return float(speeds.mean()), float(speeds.min()), float(speeds.max())


def build_passage_labels(
    position_mm: ArrayLike,
    *,
    zone_low_mm: float,
    zone_high_mm: float,
    sample_rate_hz: float,
) -> LabelResult:
    """Create OUT/LEFT/RIGHT labels from complete CNC zone crossings.

    Frames inside incomplete, reversing, or chunk-truncated zone visits are marked
    MASKED_LABEL. Event indices are inclusive. Position is used only for supervision.
    """
    position = np.asarray(position_mm, dtype=np.float64)
    if position.ndim != 1:
        raise ValueError("position_mm must be one-dimensional.")
    if len(position) == 0:
        return LabelResult(np.empty(0, dtype=np.int64), ())
    if not np.isfinite(position).all():
        raise ValueError("position_mm contains non-finite values.")
    if zone_low_mm >= zone_high_mm:
        raise ValueError("zone_low_mm must be lower than zone_high_mm.")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")

    labels = np.full(len(position), OUT, dtype=np.int64)
    inside_zone = (position >= zone_low_mm) & (position <= zone_high_mm)
    events: list[PassageEvent] = []

    for start_idx, end_idx in _contiguous_true_runs(inside_zone):
        if start_idx == 0 or end_idx >= len(position) - 1:
            labels[start_idx : end_idx + 1] = MASKED_LABEL
            continue

        before = float(position[start_idx - 1])
        after = float(position[end_idx + 1])
        if before < zone_low_mm and after > zone_high_mm:
            label = RIGHT
            direction = Direction.RIGHT
        elif before > zone_high_mm and after < zone_low_mm:
            label = LEFT
            direction = Direction.LEFT
        else:
            labels[start_idx : end_idx + 1] = MASKED_LABEL
            continue

        labels[start_idx : end_idx + 1] = label
        mean_speed, min_speed, max_speed = _speed_summary(
            position, start_idx, end_idx, sample_rate_hz
        )
        events.append(
            PassageEvent(
                start_idx=start_idx,
                end_idx=end_idx,
                direction=direction,
                mean_speed_mm_min=mean_speed,
                min_speed_mm_min=min_speed,
                max_speed_mm_min=max_speed,
            )
        )

    return LabelResult(labels=labels, events=tuple(events))
