"""Stable event schemas and extraction from frame labels/probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bee_counter_bench.constants import LEFT, RIGHT
from bee_counter_bench.data.schemas import Direction, PassageEvent


@dataclass(frozen=True)
class SegmentEvent:
    """One inclusive temporal event in local sample coordinates."""

    start_idx: int
    end_idx: int
    direction: Direction
    confidence: float = 1.0
    source_id: str = ""
    latency_frames: int = 0

    def __post_init__(self) -> None:
        if self.start_idx < 0 or self.end_idx < self.start_idx:
            raise ValueError("Event indices must be non-negative and ordered.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Event confidence must be in [0, 1].")
        if self.latency_frames < 0:
            raise ValueError("latency_frames cannot be negative.")

    @property
    def duration_frames(self) -> int:
        return self.end_idx - self.start_idx + 1


@dataclass(frozen=True)
class EventMatch:
    """One deterministic one-to-one target/prediction match."""

    target_index: int
    prediction_index: int
    iou: float
    direction_correct: bool


def target_events_from_passages(
    events: Iterable[PassageEvent],
    *,
    source_id: str = "",
) -> tuple[SegmentEvent, ...]:
    return tuple(
        SegmentEvent(
            start_idx=event.start_idx,
            end_idx=event.end_idx,
            direction=event.direction,
            confidence=1.0,
            source_id=source_id,
        )
        for event in events
    )


def events_from_labels(
    labels: ArrayLike,
    *,
    probabilities: NDArray[np.float64] | None = None,
    source_id: str = "",
    min_duration_frames: int = 1,
) -> tuple[SegmentEvent, ...]:
    """Extract contiguous LEFT/RIGHT events from categorical frame labels."""
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    if min_duration_frames <= 0:
        raise ValueError("min_duration_frames must be positive.")
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != (3, len(label_array)):
            raise ValueError("probabilities must have shape [3, time].")
    events: list[SegmentEvent] = []
    start: int | None = None
    current_label: int | None = None
    for index, label in enumerate(label_array):
        if label not in (LEFT, RIGHT):
            if start is not None and current_label is not None:
                _append_event(
                    events,
                    start,
                    index - 1,
                    current_label,
                    probabilities,
                    source_id,
                    min_duration_frames,
                )
            start = None
            current_label = None
            continue
        if start is None:
            start = index
            current_label = int(label)
            continue
        if int(label) != current_label:
            _append_event(
                events,
                start,
                index - 1,
                current_label,
                probabilities,
                source_id,
                min_duration_frames,
            )
            start = index
            current_label = int(label)
    if start is not None and current_label is not None:
        _append_event(
            events,
            start,
            len(label_array) - 1,
            current_label,
            probabilities,
            source_id,
            min_duration_frames,
        )
    return tuple(events)


def _append_event(
    events: list[SegmentEvent],
    start: int,
    end: int,
    label: int,
    probabilities: NDArray[np.float64] | None,
    source_id: str,
    min_duration_frames: int,
) -> None:
    if end - start + 1 < min_duration_frames:
        return
    direction = Direction.LEFT if label == LEFT else Direction.RIGHT
    confidence = 1.0
    if probabilities is not None:
        confidence = float(probabilities[label, start : end + 1].mean())
    events.append(
        SegmentEvent(
            start_idx=start,
            end_idx=end,
            direction=direction,
            confidence=confidence,
            source_id=source_id,
        )
    )


def temporal_iou(left: SegmentEvent, right: SegmentEvent) -> float:
    """Inclusive temporal IoU for two events."""
    intersection = max(
        0, min(left.end_idx, right.end_idx) - max(left.start_idx, right.start_idx) + 1
    )
    if intersection == 0:
        return 0.0
    union = max(left.end_idx, right.end_idx) - min(left.start_idx, right.start_idx) + 1
    return float(intersection / union)
