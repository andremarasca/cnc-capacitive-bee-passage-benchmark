"""Tests for the physically interpretable threshold baseline."""

import numpy as np

from bee_counter_bench.baseline.threshold import (
    ThresholdParameters,
    close_short_gaps,
    infer_events,
    microaggregate_fold_metrics,
)
from bee_counter_bench.data.schemas import Direction


def test_close_short_gaps_only_closes_bounded_allowed_runs() -> None:
    active = np.asarray([False, True, True, False, False, True, False], dtype=np.bool_)

    closed = close_short_gaps(active, maximum_gap=2)

    assert closed.tolist() == [False, True, True, True, True, True, False]


def test_extrema_order_controls_direction() -> None:
    signal = np.asarray([0.0, 2.0, 1.0, -3.0, 0.0], dtype=np.float32)
    parameters = ThresholdParameters(
        threshold=0.5,
        merge_gap_frames=1,
        minimum_duration_frames=1,
        positive_first_direction=Direction.RIGHT,
    )

    events = infer_events(signal, parameters, source_id="session")

    assert len(events) == 1
    assert events[0].direction == Direction.RIGHT
    assert (events[0].start_idx, events[0].end_idx) == (1, 3)


def test_micro_summary_keeps_repeated_target_sessions_fold_scoped() -> None:
    fold_template = {
        "target_events": 10,
        "predicted_events": 12,
        "count_error": 2,
        "direction_counts": {
            "targets": {"LEFT": 5, "RIGHT": 5},
            "predictions": {"LEFT": 6, "RIGHT": 6},
        },
        "wrong_direction_overlaps_iou_01": 2,
        "fragmentation_candidates": 3,
        "merge_candidates": 4,
        "thresholds": {
            "iou_25": {
                "threshold": 0.25,
                "tp": 8,
                "fp": 4,
                "fn": 2,
                "precision": 8 / 12,
                "recall": 8 / 10,
                "f1": 8 / 11,
                "by_direction": {
                    "LEFT": {"tp": 4, "fp": 2, "fn": 1},
                    "RIGHT": {"tp": 4, "fp": 2, "fn": 1},
                },
            }
        },
    }
    folds = [
        {"test_metrics": fold_template},
        {"test_metrics": fold_template},
    ]

    result = microaggregate_fold_metrics(folds, (0.25,))

    assert result["target_events"] == 20
    assert result["predicted_events"] == 24
    assert result["thresholds"]["iou_25"]["tp"] == 16
    assert result["thresholds"]["iou_25"]["fp"] == 8
    assert result["thresholds"]["iou_25"]["fn"] == 4
    assert result["wrong_direction_overlaps_iou_01"] == 4
    assert result["fragmentation_candidates"] == 6
    assert result["merge_candidates"] == 8
