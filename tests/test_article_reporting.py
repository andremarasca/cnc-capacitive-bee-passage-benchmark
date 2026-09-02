from __future__ import annotations

from pathlib import Path

import pytest

from bee_counter_bench.data.schemas import Direction
from bee_counter_bench.evaluation.events import SegmentEvent
from bee_counter_bench.reporting.article_experiment import (
    TargetSession,
    _prepare_destination,
    evaluate_saved_sessions,
    summarize_across_seeds,
)


def _target(start: int, end: int, direction: Direction) -> SegmentEvent:
    return SegmentEvent(
        start_idx=start,
        end_idx=end,
        direction=direction,
        source_id="session_a",
    )


def _prediction(
    start: int,
    end: int,
    direction: Direction,
) -> dict[str, float | int | str]:
    return {
        "start_idx": start,
        "end_idx": end,
        "direction": direction.name,
        "confidence": 0.9,
    }


def test_saved_predictions_recalculate_presence_and_direction_metrics() -> None:
    evaluation = {
        "sessions": [
            {
                "session_id": "session_a",
                "frames": 100,
                "target_events": 2,
                "predicted_events": [
                    _prediction(0, 9, Direction.RIGHT),
                    _prediction(20, 29, Direction.RIGHT),
                    _prediction(40, 49, Direction.LEFT),
                ],
            }
        ],
        "event_metrics": {
            "thresholds": {
                "iou_25": {"tp": 1, "fp": 2, "fn": 1},
                "iou_50": {"tp": 1, "fp": 2, "fn": 1},
            }
        },
    }
    targets = {
        "session_a": TargetSession(
            events=(
                _target(0, 9, Direction.LEFT),
                _target(20, 29, Direction.RIGHT),
            ),
            frames=100,
        )
    }

    result = evaluate_saved_sessions(evaluation, targets)

    assert result.metrics[("directional", "iou_25")]["tp"] == 1
    assert result.metrics[("directional", "iou_25")]["fp"] == 2
    assert result.metrics[("presence", "iou_25")]["tp"] == 2
    assert result.metrics[("presence", "iou_25")]["fp"] == 1
    assert result.metrics[("presence", "any_overlap")]["recall"] == 1.0
    assert result.direction_confusion == ((0, 1), (0, 1))


def test_across_seed_summary_uses_sample_standard_deviation() -> None:
    rows = [
        {
            "normalization": "training_population",
            "protocol": "main_session_grouped",
            "seed": 42,
            "tp": 8,
            "fp": 2,
            "fn": 2,
            "precision": 0.8,
            "recall": 0.8,
            "f1": 0.8,
            "target_events": 10,
            "predicted_events": 10,
            "count_bias": 0,
            "count_bias_percent": 0.0,
        },
        {
            "normalization": "training_population",
            "protocol": "main_session_grouped",
            "seed": 43,
            "tp": 10,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "target_events": 10,
            "predicted_events": 10,
            "count_bias": 0,
            "count_bias_percent": 0.0,
        },
    ]

    summary = summarize_across_seeds(
        rows,
        group_fields=("normalization", "protocol"),
    )

    assert len(summary) == 1
    assert summary[0]["seed_count"] == 2
    assert summary[0]["f1_mean"] == pytest.approx(0.9)
    assert summary[0]["f1_sd"] == pytest.approx(2**0.5 / 10.0)


def test_report_destination_never_reuses_nonempty_directory() -> None:
    destination = Path(__file__).resolve().parent

    with pytest.raises(FileExistsError, match="non-empty output"):
        _prepare_destination(destination)
