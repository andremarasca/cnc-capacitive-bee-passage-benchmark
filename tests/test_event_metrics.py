"""Tests for event extraction and deterministic optimal matching."""

from bee_counter_bench.data.schemas import Direction
from bee_counter_bench.evaluation.events import SegmentEvent
from bee_counter_bench.evaluation.metrics import (
    event_detection_metrics,
    event_metrics,
    match_events,
)


def _event(start: int, end: int, source: str = "session") -> SegmentEvent:
    return SegmentEvent(start, end, Direction.LEFT, source_id=source)


def _right_event(start: int, end: int, source: str = "session") -> SegmentEvent:
    return SegmentEvent(start, end, Direction.RIGHT, source_id=source)


def test_matching_maximizes_cardinality_before_total_iou() -> None:
    targets = (_event(0, 9), _event(8, 17))
    predictions = (_event(0, 10), _event(0, 8))

    matches = match_events(targets, predictions, iou_threshold=0.10)

    assert {(item.target_index, item.prediction_index) for item in matches} == {
        (0, 1),
        (1, 0),
    }


def test_matching_never_crosses_session_boundaries() -> None:
    matches = match_events(
        (_event(0, 9, "session_a"),),
        (_event(0, 9, "session_b"),),
        iou_threshold=0.25,
    )

    assert matches == ()


def test_event_report_uses_counts_without_exposure_rate() -> None:
    report = event_metrics(
        (_event(0, 9),),
        (_event(0, 9), _event(20, 29)),
    )

    primary = report["thresholds"]["iou_25"]
    assert set(primary) == {
        "threshold",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "by_direction",
    }
    assert primary["fp"] == 1


def test_direction_mismatch_is_only_accepted_in_presence_mode() -> None:
    targets = (_event(0, 9),)
    predictions = (_right_event(0, 9),)

    directional = event_detection_metrics(
        targets,
        predictions,
        iou_threshold=0.25,
        require_direction=True,
    )
    presence = event_detection_metrics(
        targets,
        predictions,
        iou_threshold=0.25,
        require_direction=False,
    )

    assert (directional["tp"], directional["fp"], directional["fn"]) == (0, 1, 1)
    assert (presence["tp"], presence["fp"], presence["fn"]) == (1, 0, 0)


def test_iou_boundaries_are_inclusive() -> None:
    target = (_event(0, 3),)

    at_iou_25 = match_events(target, (_event(3, 3),), iou_threshold=0.25)
    at_iou_50 = match_events(target, (_event(2, 3),), iou_threshold=0.50)

    assert len(at_iou_25) == 1
    assert len(at_iou_50) == 1


def test_any_overlap_accepts_one_frame_but_not_zero_frames() -> None:
    targets = (_event(0, 9),)

    one_frame = match_events(targets, (_event(9, 20),), iou_threshold=None)
    no_overlap = match_events(targets, (_event(10, 20),), iou_threshold=None)

    assert len(one_frame) == 1
    assert no_overlap == ()


def test_fragmentation_still_uses_one_to_one_matching() -> None:
    targets = (_event(0, 20),)
    predictions = (_event(0, 9), _event(10, 20))

    matches = match_events(targets, predictions, iou_threshold=None)

    assert len(matches) == 1
