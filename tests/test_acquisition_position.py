import numpy as np
import pytest

from bee_counter_bench.acquisition.recorder import RoundTripChunker
from bee_counter_bench.acquisition.position_tracker import (
    StepDirectionPositionTracker,
)


def _pulse_train(length: int, edge_indices: tuple[int, ...]) -> np.ndarray:
    signal = np.zeros(length, dtype=np.float32)
    signal[list(edge_indices)] = 3.3
    return signal


def test_step_direction_updates_persistent_position() -> None:
    tracker = RoundTripChunker(
        trips_per_chunk=10,
        steps_per_mm=100.0,
        travel_min_mm=0.0,
        travel_max_mm=100.0,
        voltage_threshold=2.5,
        min_edge_gap=1,
        min_pulse_samples=1,
        dir_vote_samples=0,
    )
    capacitive_signal = np.zeros(8, dtype=np.float32)

    tracker.feed(
        dir_signal=np.full(8, 3.3, dtype=np.float32),
        step_signal=_pulse_train(8, (1, 3, 5)),
        signal=capacitive_signal,
    )

    assert tracker.total_edges == 3
    assert tracker.position_mm == pytest.approx(0.03)

    tracker.feed(
        dir_signal=np.zeros(8, dtype=np.float32),
        step_signal=_pulse_train(8, (1, 3)),
        signal=capacitive_signal,
    )

    assert tracker.total_edges == 5
    assert tracker.position_mm == pytest.approx(0.01)


def test_chunker_accepts_valid_pulse_split_between_blocks() -> None:
    tracker = RoundTripChunker(
        trips_per_chunk=10,
        steps_per_mm=100.0,
        travel_min_mm=0.0,
        travel_max_mm=100.0,
        voltage_threshold=2.5,
        min_edge_gap=1,
        min_pulse_samples=2,
        dir_vote_samples=0,
    )
    direction = np.full(4, 3.3, dtype=np.float32)
    capacitive_signal = np.zeros(4, dtype=np.float32)

    tracker.feed(
        direction,
        np.asarray([0.0, 0.0, 0.0, 3.3], dtype=np.float32),
        capacitive_signal,
    )
    tracker.feed(
        direction,
        np.asarray([3.3, 0.0, 0.0, 0.0], dtype=np.float32),
        capacitive_signal,
    )

    assert tracker.total_edges == 1
    assert tracker.position_mm == pytest.approx(0.01)


def test_live_tracker_accepts_valid_pulse_split_between_blocks() -> None:
    tracker = StepDirectionPositionTracker(
        steps_per_mm=100.0,
        travel_min_mm=0.0,
        travel_max_mm=100.0,
        voltage_threshold=2.5,
        min_edge_gap=1,
        min_pulse_samples=2,
        dir_vote_samples=0,
    )
    direction = np.full(4, 3.3, dtype=np.float32)

    tracker.feed(
        direction,
        np.asarray([0.0, 0.0, 0.0, 3.3], dtype=np.float32),
    )
    tracker.feed(
        direction,
        np.asarray([3.3, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert tracker.total_edges == 1
    assert tracker.position_mm == pytest.approx(0.01)
