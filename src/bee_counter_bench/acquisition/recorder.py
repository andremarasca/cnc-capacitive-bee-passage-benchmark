"""Round-trip chunked bench recorder.

Records raw ai3 signal + computed CNC position from Step/Direction.
Chunks are defined by number of CNC round-trips (ida-e-volta) instead
of fixed time. Each chunk starts at the first sample of a positive
direction change (0 to maximum transition) and ends just before the next one.

Outputs NPZ files: raw_signal (100kHz), position_mm (100kHz), metadata.
"""

from __future__ import annotations

import numpy as np


class StreamingRisingEdgeDetector:
    """Detect validated rising edges without losing pulses between blocks."""

    def __init__(self, *, minimum_edge_gap: int, minimum_pulse_samples: int) -> None:
        self._minimum_edge_gap = max(1, minimum_edge_gap)
        self._minimum_pulse_samples = max(1, minimum_pulse_samples)
        self._previous_high = False
        self._samples_since_edge = 0
        self._pending_high_samples = 0

    def detect(self, step_high: np.ndarray) -> np.ndarray:
        """Return accepted edge indices in the current block."""
        values = np.asarray(step_high, dtype=np.bool_)
        sample_count = len(values)
        if sample_count == 0:
            return np.array([], dtype=np.intp)

        candidates: list[int] = []
        if self._pending_high_samples > 0:
            leading_high = self._leading_high_samples(values)
            if self._pending_high_samples + leading_high >= self._minimum_pulse_samples:
                candidates.append(0)
                self._pending_high_samples = 0
            elif leading_high == sample_count:
                self._pending_high_samples += sample_count
                self._previous_high = True
                self._samples_since_edge += sample_count
                return np.array([], dtype=np.intp)
            else:
                self._pending_high_samples = 0

        previous = np.empty(sample_count, dtype=np.bool_)
        previous[0] = self._previous_high
        previous[1:] = values[:-1]
        rising_edges = np.flatnonzero(values & ~previous)
        for index in rising_edges.tolist():
            high_samples = self._high_run_length(values, index)
            if high_samples >= self._minimum_pulse_samples:
                candidates.append(index)
            elif index + high_samples == sample_count:
                self._pending_high_samples = high_samples

        self._previous_high = bool(values[-1])
        accepted = self._apply_minimum_gap(sorted(set(candidates)))
        if accepted:
            self._samples_since_edge = sample_count - accepted[-1] - 1
        else:
            self._samples_since_edge += sample_count
        return np.asarray(accepted, dtype=np.intp)

    @staticmethod
    def _leading_high_samples(values: np.ndarray) -> int:
        low_indices = np.flatnonzero(~values)
        return len(values) if len(low_indices) == 0 else int(low_indices[0])

    @staticmethod
    def _high_run_length(values: np.ndarray, start: int) -> int:
        low_indices = np.flatnonzero(~values[start:])
        return len(values) - start if len(low_indices) == 0 else int(low_indices[0])

    def _apply_minimum_gap(self, candidates: list[int]) -> list[int]:
        accepted: list[int] = []
        for index in candidates:
            distance = (
                index - accepted[-1] if accepted else self._samples_since_edge + index
            )
            if distance >= self._minimum_edge_gap:
                accepted.append(index)
        return accepted


class RoundTripChunker:
    """Accumulates DAQ data and cuts chunks at round-trip boundaries.

    A round trip is one complete 0 to maximum to 0 cycle.
    Chunk boundary = first sample where direction changes to positive
    (i.e., CNC starts going from 0 toward travel_max).

    The tracker counts positive reversals (direction -1 to +1).
    Chunk starts at reversal #K, ends just before reversal #K+N.
    """

    def __init__(
        self,
        trips_per_chunk: int = 10,
        steps_per_mm: float = 100.0,
        travel_min_mm: float | None = None,
        travel_max_mm: float | None = None,
        voltage_threshold: float = 2.5,
        min_edge_gap: int = 10,
        min_pulse_samples: int = 2,
        dir_vote_samples: int = 3,
    ) -> None:
        if travel_min_mm is None or travel_max_mm is None:
            raise ValueError("travel_min_mm and travel_max_mm must be provided.")
        if travel_max_mm <= travel_min_mm:
            raise ValueError("travel_max_mm must be greater than travel_min_mm.")
        self._trips_per_chunk = trips_per_chunk
        self._mm_per_step = 1.0 / steps_per_mm
        self._travel_min = travel_min_mm
        self._travel_max = travel_max_mm
        self._threshold = voltage_threshold
        self._dir_vote_samples = dir_vote_samples

        # Step tracking state
        self._step_count: int = 0
        self._last_direction: int = 0
        self._edge_detector = StreamingRisingEdgeDetector(
            minimum_edge_gap=min_edge_gap,
            minimum_pulse_samples=min_pulse_samples,
        )
        self._total_edges: int = 0

        # Round-trip counting
        self._positive_reversals: int = 0
        self._chunk_start_reversal: int = 0
        self._recording_chunk: bool = False

        # Buffers for current chunk
        self._signal_buf: list[np.ndarray] = []
        self._position_buf: list[np.ndarray] = []
        self._chunk_samples: int = 0

        # Pre-chunk buffer (samples before first boundary)
        self._waiting_for_start: bool = True

        # One-time initial calibration flag
        self._calibrated: bool = False

    @property
    def position_mm(self) -> float:
        """Current CNC position in mm."""
        return self._step_count * self._mm_per_step

    @property
    def positive_reversals(self) -> int:
        """Total positive direction reversals (0 to maximum transitions)."""
        return self._positive_reversals

    @property
    def total_edges(self) -> int:
        return self._total_edges

    @property
    def chunk_samples(self) -> int:
        return self._chunk_samples

    @property
    def trips_in_current_chunk(self) -> int:
        """Round-trips accumulated in current chunk."""
        return self._positive_reversals - self._chunk_start_reversal

    def feed(
        self, dir_signal: np.ndarray, step_signal: np.ndarray, signal: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Feed a DAQ chunk and return any completed round-trip chunks.

        Properly slices data at reversal boundaries:
        - Discards all samples before first positive reversal.
        - Cuts chunk exactly at each trip-boundary reversal sample.

        Args:
            dir_signal: Direction voltage array.
            step_signal: Step voltage array (pulses).
            signal: Capacitive sensor signal (ai3).

        Returns:
            List of (signal_chunk, position_chunk) tuples for completed chunks.
            Each is a contiguous array covering exactly N round-trips.
        """
        n = len(step_signal)
        if n == 0:
            return []

        # Vectorized edge detection
        step_high = step_signal > self._threshold
        dir_high = dir_signal > self._threshold
        edge_indices = self._detect_edges(step_high)

        # Compute per-sample position
        position = self._compute_position(edge_indices, dir_high, n)

        # Find reversal points (direction changes to positive: starts of ida)
        # Does NOT increment _positive_reversals; we do it in the loop below.
        reversal_samples = self._find_positive_reversals(edge_indices, dir_high)

        completed: list[tuple[np.ndarray, np.ndarray]] = []
        sig_f32 = signal.astype(np.float32)

        if not reversal_samples:
            # No reversals: buffer everything if already recording
            if self._recording_chunk:
                self._signal_buf.append(sig_f32)
                self._position_buf.append(position)
                self._chunk_samples += n
            return completed

        # cursor: index from which data has not yet been assigned
        cursor = 0

        for rev_sample in reversal_samples:
            self._positive_reversals += 1

            if self._waiting_for_start:
                # First positive reversal ever: discard [0 .. rev_sample)
                cursor = rev_sample
                self._waiting_for_start = False
                self._recording_chunk = True
                self._chunk_start_reversal = self._positive_reversals
                continue

            trips_done = self._positive_reversals - self._chunk_start_reversal

            if trips_done >= self._trips_per_chunk and self._recording_chunk:
                # Buffer [cursor .. rev_sample) into current chunk, then flush
                if rev_sample > cursor:
                    self._signal_buf.append(sig_f32[cursor:rev_sample])
                    self._position_buf.append(position[cursor:rev_sample])
                    self._chunk_samples += rev_sample - cursor

                chunk = self._flush_chunk()
                if chunk is not None:
                    completed.append(chunk)

                # New chunk starts at rev_sample
                cursor = rev_sample
                self._recording_chunk = True
                self._chunk_start_reversal = self._positive_reversals

        # Buffer remaining [cursor .. n) for in-progress chunk
        if self._recording_chunk and cursor < n:
            self._signal_buf.append(sig_f32[cursor:])
            self._position_buf.append(position[cursor:])
            self._chunk_samples += n - cursor

        return completed

    def flush_partial(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Flush whatever is buffered (partial chunk on stop)."""
        return self._flush_chunk()

    def _flush_chunk(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Concatenate and clear buffers."""
        if not self._signal_buf:
            return None
        sig = np.concatenate(self._signal_buf)
        pos = np.concatenate(self._position_buf)
        self._signal_buf.clear()
        self._position_buf.clear()
        self._chunk_samples = 0
        return (sig, pos)

    def _detect_edges(self, step_high: np.ndarray) -> np.ndarray:
        """Delegate streaming pulse validation to the shared detector."""
        return self._edge_detector.detect(step_high)

    def _direction_at_edges(
        self, edge_indices: np.ndarray, dir_high: np.ndarray, n: int
    ) -> np.ndarray:
        """Read direction at each step edge using majority vote.

        Instead of reading the direction signal at a single sample
        (which can be noisy or caught mid-transition), averages
        the direction signal over +/- dir_vote_samples around each edge.
        """
        if self._dir_vote_samples <= 0:
            return np.where(dir_high[edge_indices], 1, -1)

        half = self._dir_vote_samples
        directions = np.empty(len(edge_indices), dtype=np.int8)
        for i, idx in enumerate(edge_indices):
            start = max(0, idx - half)
            end = min(n, idx + half + 1)
            vote = np.mean(dir_high[start:end])
            directions[i] = 1 if vote > 0.5 else -1
        return directions

    def _compute_position(
        self, edge_indices: np.ndarray, dir_high: np.ndarray, n: int
    ) -> np.ndarray:
        """Compute per-sample position from edges."""
        position = np.empty(n, dtype=np.float32)

        if len(edge_indices) == 0:
            position[:] = self.position_mm
            return position

        directions = self._direction_at_edges(edge_indices, dir_high, n)

        # One-time calibration: first positive reversal sets position to 0 mm.
        if not self._calibrated:
            for i in range(len(directions)):
                prev_d = self._last_direction if i == 0 else int(directions[i - 1])
                if prev_d == -1 and directions[i] == 1:
                    partial_sum = int(np.sum(directions[: i + 1]))
                    self._step_count = -partial_sum
                    self._calibrated = True
                    break

        cum_steps = np.cumsum(directions) + self._step_count
        edge_positions = cum_steps * self._mm_per_step

        # Fill position: constant between edges
        pre_pos = self.position_mm
        position[: edge_indices[0]] = pre_pos
        for i in range(len(edge_indices) - 1):
            position[edge_indices[i] : edge_indices[i + 1]] = edge_positions[i]
        position[edge_indices[-1] :] = edge_positions[-1]

        # Update persistent state
        self._step_count = int(cum_steps[-1])
        self._total_edges += len(edge_indices)

        return position

    def _find_positive_reversals(
        self, edge_indices: np.ndarray, dir_high: np.ndarray
    ) -> list[int]:
        """Find sample indices where direction changes to positive (+1).

        Uses the same majority-vote direction reading as _compute_position.
        Returns sample indices only. Does NOT increment _positive_reversals
        (the caller in feed() handles counting to keep it synchronised
        with the per-reversal slicing logic).
        """
        if len(edge_indices) == 0:
            return []

        n = len(dir_high)
        reversals: list[int] = []
        directions = self._direction_at_edges(edge_indices, dir_high, n)

        for i, d in enumerate(directions):
            prev_d = self._last_direction if i == 0 else int(directions[i - 1])
            if prev_d != 0 and d != prev_d and d == 1:
                reversals.append(int(edge_indices[i]))

        # Track last direction for next chunk
        self._last_direction = int(directions[-1])
        return reversals
