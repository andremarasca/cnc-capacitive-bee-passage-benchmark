"""Stable in-memory schemas for labeled temporal sequences."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from bee_counter_bench.constants import (
    MASKED_LABEL,
    LEFT as LEFT_ID,
    OUT,
    RIGHT as RIGHT_ID,
)


class Direction(IntEnum):
    """Physical direction for a complete passage."""

    LEFT = LEFT_ID
    RIGHT = RIGHT_ID


@dataclass(frozen=True)
class PassageEvent:
    """One complete ground-truth or predicted passage, using inclusive indices."""

    start_idx: int
    end_idx: int
    direction: Direction
    mean_speed_mm_min: float
    min_speed_mm_min: float
    max_speed_mm_min: float

    def __post_init__(self) -> None:
        if self.start_idx < 0 or self.end_idx < self.start_idx:
            raise ValueError("Passage indices must be non-negative and ordered.")
        if (
            min(
                self.mean_speed_mm_min,
                self.min_speed_mm_min,
                self.max_speed_mm_min,
            )
            < 0
        ):
            raise ValueError("Passage speed magnitudes cannot be negative.")

    @property
    def duration_frames(self) -> int:
        return self.end_idx - self.start_idx + 1


@dataclass(frozen=True)
class LabelResult:
    """Categorical labels and complete events for one position sequence."""

    labels: NDArray[np.int64]
    events: tuple[PassageEvent, ...]


@dataclass(frozen=True)
class PreprocessingMetadata:
    """Provenance required to reproduce one preprocessed sequence."""

    schema_version: int
    source_sample_rate_hz: int
    output_sample_rate_hz: int
    filter_family: str
    filter_mode: str
    filter_prototype_order: int
    filter_transformed_order: int
    low_cutoff_hz: float | None
    high_cutoff_hz: float
    filter_norm: str
    decimation_method: str
    filter_initialization: str = "steady_state_first_sample"
    state_scope: str = "chunk"


@dataclass(frozen=True)
class PreprocessedSequence:
    """Versioned, aligned and labeled sequence persisted for model training."""

    signal: NDArray[np.float32]
    position_mm: NDArray[np.float32]
    labels: NDArray[np.int64]
    valid_mask: NDArray[np.bool_]
    sample_rate_hz: int
    session_id: str
    chunk_id: str
    source_path: str
    source_sha256: str
    preprocessing: PreprocessingMetadata
    events: tuple[PassageEvent, ...] = ()

    def __post_init__(self) -> None:
        arrays = (self.signal, self.position_mm, self.labels, self.valid_mask)
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("Sequence arrays must be one-dimensional.")
        if len({len(array) for array in arrays}) != 1 or not len(self.signal):
            raise ValueError("Sequence arrays must be non-empty and equally sized.")
        if self.signal.dtype != np.float32 or self.position_mm.dtype != np.float32:
            raise ValueError("Signal and position must use float32.")
        if self.labels.dtype != np.int64 or self.valid_mask.dtype != np.bool_:
            raise ValueError("Labels must use int64 and valid_mask must use bool.")
        if (
            not np.isfinite(self.signal).all()
            or not np.isfinite(self.position_mm).all()
        ):
            raise ValueError("Signal and position must contain only finite values.")
        allowed_labels = np.array(
            (OUT, LEFT_ID, RIGHT_ID, MASKED_LABEL), dtype=np.int64
        )
        if not np.isin(self.labels, allowed_labels).all():
            raise ValueError("Sequence contains an unsupported categorical label.")
        if not np.array_equal(self.valid_mask, self.labels != MASKED_LABEL):
            raise ValueError("valid_mask must be exactly labels != MASKED_LABEL.")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        if self.sample_rate_hz != self.preprocessing.output_sample_rate_hz:
            raise ValueError("Sequence rate disagrees with preprocessing metadata.")
        if not self.session_id or not self.chunk_id or not self.source_path:
            raise ValueError("Session, chunk, and source path are required.")
        digest = self.source_sha256.lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("source_sha256 must be a 64-character hexadecimal digest.")
        if any(event.end_idx >= len(self.signal) for event in self.events):
            raise ValueError("An event extends beyond the sequence.")
