"""Frame and event metrics for passage segmentation."""

from __future__ import annotations

from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right
from dataclasses import asdict
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment

from bee_counter_bench.constants import MASKED_LABEL
from bee_counter_bench.evaluation.events import (
    EventMatch,
    SegmentEvent,
)


_DIRECTIONS = ("LEFT", "RIGHT")


def match_events(
    targets: tuple[SegmentEvent, ...],
    predictions: tuple[SegmentEvent, ...],
    *,
    iou_threshold: float | None,
    require_direction: bool = True,
    require_source: bool = True,
) -> tuple[EventMatch, ...]:
    """Match events by maximum cardinality, then maximum total IoU."""
    if iou_threshold is not None and not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1] or None.")
    target_groups = _event_groups(
        targets,
        require_source=require_source,
        require_direction=require_direction,
    )
    prediction_groups = _event_groups(
        predictions,
        require_source=require_source,
        require_direction=require_direction,
    )
    matches: list[EventMatch] = []
    for key in sorted(target_groups.keys() & prediction_groups.keys()):
        matches.extend(
            _match_group(
                targets,
                predictions,
                target_groups[key],
                prediction_groups[key],
                iou_threshold,
            )
        )
    matches.sort(key=lambda match: (match.target_index, match.prediction_index))
    return tuple(matches)


def _event_groups(
    events: tuple[SegmentEvent, ...],
    *,
    require_source: bool,
    require_direction: bool,
) -> dict[tuple[str, int], list[int]]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        source = event.source_id if require_source else ""
        direction = int(event.direction) if require_direction else 0
        groups[(source, direction)].append(index)
    return groups


def _match_group(
    targets: tuple[SegmentEvent, ...],
    predictions: tuple[SegmentEvent, ...],
    target_indices: list[int],
    prediction_indices: list[int],
    iou_threshold: float | None,
) -> list[EventMatch]:
    target_starts = np.asarray(
        [targets[index].start_idx for index in target_indices], dtype=np.int64
    )[:, None]
    target_ends = np.asarray(
        [targets[index].end_idx for index in target_indices], dtype=np.int64
    )[:, None]
    prediction_starts = np.asarray(
        [predictions[index].start_idx for index in prediction_indices],
        dtype=np.int64,
    )[None, :]
    prediction_ends = np.asarray(
        [predictions[index].end_idx for index in prediction_indices],
        dtype=np.int64,
    )[None, :]
    intersections = np.maximum(
        0,
        np.minimum(target_ends, prediction_ends)
        - np.maximum(target_starts, prediction_starts)
        + 1,
    )
    unions = (
        np.maximum(target_ends, prediction_ends)
        - np.minimum(target_starts, prediction_starts)
        + 1
    )
    ious = intersections / unions
    valid = intersections > 0 if iou_threshold is None else ious >= iou_threshold
    if not np.any(valid):
        return []

    maximum_matches = min(len(target_indices), len(prediction_indices))
    cardinality_weight = float(maximum_matches + 1)
    scores = np.where(valid, cardinality_weight + ious, 0.0)
    local_targets, local_predictions = linear_sum_assignment(-scores)
    matches = []
    for local_target, local_prediction in zip(
        local_targets.tolist(), local_predictions.tolist(), strict=True
    ):
        if not valid[local_target, local_prediction]:
            continue
        target_index = target_indices[local_target]
        prediction_index = prediction_indices[local_prediction]
        matches.append(
            EventMatch(
                target_index=target_index,
                prediction_index=prediction_index,
                iou=float(ious[local_target, local_prediction]),
                direction_correct=(
                    targets[target_index].direction
                    == predictions[prediction_index].direction
                ),
            )
        )
    return matches


def frame_metrics(
    targets: ArrayLike,
    predictions: ArrayLike,
    *,
    ignore_index: int = MASKED_LABEL,
) -> dict[str, Any]:
    """Frame accuracy and per-class F1 for labels 0/1/2."""
    target_array = np.asarray(targets, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    if target_array.shape != prediction_array.shape:
        raise ValueError("targets and predictions must have identical shape.")
    valid = target_array != ignore_index
    if not np.any(valid):
        raise ValueError("No valid frames for metrics.")
    target_valid = target_array[valid]
    prediction_valid = prediction_array[valid]
    confusion = np.zeros((3, 3), dtype=np.int64)
    for target_class in range(3):
        for prediction_class in range(3):
            confusion[target_class, prediction_class] = int(
                np.count_nonzero(
                    (target_valid == target_class)
                    & (prediction_valid == prediction_class)
                )
            )
    per_class: dict[str, Any] = {}
    f1_values = []
    for class_id, name in ((0, "OUT"), (1, "LEFT"), (2, "RIGHT")):
        tp = float(confusion[class_id, class_id])
        fp = float(confusion[:, class_id].sum() - confusion[class_id, class_id])
        fn = float(confusion[class_id, :].sum() - confusion[class_id, class_id])
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    return {
        "valid_frames": int(np.count_nonzero(valid)),
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "macro_f1": float(sum(f1_values) / len(f1_values)),
        "confusion": confusion.tolist(),
        "per_class": per_class,
    }


def event_metrics(
    targets: tuple[SegmentEvent, ...],
    predictions: tuple[SegmentEvent, ...],
    *,
    iou_thresholds: tuple[float, ...] = (0.10, 0.25, 0.50),
    include_matches: bool = False,
    require_direction: bool = True,
) -> dict[str, Any]:
    """Compute event, count, direction and segmentation diagnostics."""
    payload: dict[str, Any] = {}
    for threshold in iou_thresholds:
        matches = match_events(
            targets,
            predictions,
            iou_threshold=threshold,
            require_direction=require_direction,
        )
        matched_targets = {match.target_index for match in matches}
        matched_predictions = {match.prediction_index for match in matches}
        tp = len(matches)
        fp = len(predictions) - tp
        fn = len(targets) - tp
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        key = f"iou_{int(round(threshold * 100)):02d}"
        threshold_payload: dict[str, Any] = {
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "by_direction": _direction_metrics(
                targets,
                predictions,
                matches,
                matched_targets,
                matched_predictions,
            ),
        }
        if include_matches:
            threshold_payload["matches"] = [asdict(match) for match in matches]
        payload[key] = threshold_payload
    direction_counts = {
        "targets": dict(
            sorted(Counter(event.direction.name for event in targets).items())
        ),
        "predictions": dict(
            sorted(Counter(event.direction.name for event in predictions).items())
        ),
    }
    loose_matches = match_events(
        targets, predictions, iou_threshold=0.01, require_direction=False
    )
    wrong_direction = sum(
        1
        for match in loose_matches
        if targets[match.target_index].direction
        != predictions[match.prediction_index].direction
    )
    return {
        "thresholds": payload,
        "target_events": len(targets),
        "predicted_events": len(predictions),
        "count_error": len(predictions) - len(targets),
        "direction_counts": direction_counts,
        "wrong_direction_overlaps_iou_01": wrong_direction,
        "fragmentation_candidates": _fragmentation_candidates(targets, predictions),
        "merge_candidates": _merge_candidates(targets, predictions),
    }


def event_detection_metrics(
    targets: tuple[SegmentEvent, ...],
    predictions: tuple[SegmentEvent, ...],
    *,
    iou_threshold: float | None,
    require_direction: bool,
) -> dict[str, Any]:
    """Return auditable event counts for one matching criterion."""
    matches = match_events(
        targets,
        predictions,
        iou_threshold=iou_threshold,
        require_direction=require_direction,
    )
    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(targets) - true_positives
    precision = (
        0.0
        if true_positives + false_positives == 0
        else true_positives / (true_positives + false_positives)
    )
    recall = (
        0.0
        if true_positives + false_negatives == 0
        else true_positives / (true_positives + false_negatives)
    )
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "criterion": (
            "any_overlap"
            if iou_threshold is None
            else f"iou_{int(round(iou_threshold * 100)):02d}"
        ),
        "iou_threshold": iou_threshold,
        "require_direction": require_direction,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "target_events": len(targets),
        "predicted_events": len(predictions),
    }


def temporal_error_metrics(
    targets: tuple[SegmentEvent, ...],
    predictions: tuple[SegmentEvent, ...],
    *,
    iou_threshold: float,
    sample_rate_hz: float,
    include_values: bool = False,
) -> dict[str, Any]:
    """Summarize absolute boundary and center errors for matched events."""
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    matches = match_events(
        targets,
        predictions,
        iou_threshold=iou_threshold,
    )
    values: dict[str, list[float]] = {"start": [], "end": [], "center": []}
    for match in matches:
        target = targets[match.target_index]
        prediction = predictions[match.prediction_index]
        values["start"].append(abs(prediction.start_idx - target.start_idx))
        values["end"].append(abs(prediction.end_idx - target.end_idx))
        target_center = (target.start_idx + target.end_idx) / 2.0
        prediction_center = (prediction.start_idx + prediction.end_idx) / 2.0
        values["center"].append(abs(prediction_center - target_center))
    report = {
        "iou_threshold": iou_threshold,
        "matched_events": len(matches),
        "errors": {
            name: _absolute_error_summary(measurements, sample_rate_hz)
            for name, measurements in values.items()
        },
    }
    if include_values:
        report["absolute_errors_frames"] = values
    return report


def _absolute_error_summary(
    values: list[float],
    sample_rate_hz: float,
) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "median_frames": 0.0,
            "p95_frames": 0.0,
            "median_seconds": 0.0,
            "p95_seconds": 0.0,
        }
    array = np.asarray(values, dtype=np.float64)
    median_frames = float(np.median(array))
    p95_frames = float(np.percentile(array, 95.0))
    return {
        "count": int(array.size),
        "median_frames": median_frames,
        "p95_frames": p95_frames,
        "median_seconds": median_frames / sample_rate_hz,
        "p95_seconds": p95_frames / sample_rate_hz,
    }


def _direction_metrics(
    targets: tuple[SegmentEvent, ...],
    predictions: tuple[SegmentEvent, ...],
    matches: tuple[EventMatch, ...],
    matched_targets: set[int],
    matched_predictions: set[int],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for direction in _DIRECTIONS:
        tp = sum(
            1
            for match in matches
            if targets[match.target_index].direction.name == direction
            and predictions[match.prediction_index].direction.name == direction
        )
        fp = sum(
            1
            for index, prediction in enumerate(predictions)
            if prediction.direction.name == direction
            and index not in matched_predictions
        )
        fn = sum(
            1
            for index, target in enumerate(targets)
            if target.direction.name == direction and index not in matched_targets
        )
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        payload[direction] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return payload


def _same_source(left: SegmentEvent, right: SegmentEvent) -> bool:
    return left.source_id == right.source_id


def _fragmentation_candidates(
    targets: tuple[SegmentEvent, ...],
    predictions: tuple[SegmentEvent, ...],
) -> int:
    return _multiple_overlap_count(targets, predictions)


def _merge_candidates(
    targets: tuple[SegmentEvent, ...],
    predictions: tuple[SegmentEvent, ...],
) -> int:
    return _multiple_overlap_count(predictions, targets)


def _multiple_overlap_count(
    queries: tuple[SegmentEvent, ...],
    candidates: tuple[SegmentEvent, ...],
) -> int:
    by_source: dict[str, list[SegmentEvent]] = defaultdict(list)
    for candidate in candidates:
        by_source[candidate.source_id].append(candidate)
    indexes = {
        source: (
            sorted(event.start_idx for event in events),
            sorted(event.end_idx for event in events),
        )
        for source, events in by_source.items()
    }
    count = 0
    for query in queries:
        starts, ends = indexes.get(query.source_id, ([], []))
        overlaps = bisect_right(starts, query.end_idx) - bisect_left(
            ends, query.start_idx
        )
        if overlaps > 1:
            count += 1
    return count
