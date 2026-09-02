"""Training-only polarity and validation-selected threshold baseline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from bee_counter_bench.data.schemas import Direction, PreprocessedSequence
from bee_counter_bench.data.sequence_io import (
    file_sha256,
    load_preprocessed_sequence,
)
from bee_counter_bench.evaluation.events import (
    SegmentEvent,
    target_events_from_passages,
)
from bee_counter_bench.evaluation.metrics import event_metrics
from bee_counter_bench.evaluation.metrics import match_events
from bee_counter_bench.provenance import find_project_root, source_code_sha256


@dataclass(frozen=True)
class ThresholdParameters:
    """Parameters selected without access to the test split."""

    threshold: float
    merge_gap_frames: int
    minimum_duration_frames: int
    positive_first_direction: Direction

    @property
    def negative_first_direction(self) -> Direction:
        return (
            Direction.RIGHT
            if self.positive_first_direction == Direction.LEFT
            else Direction.LEFT
        )


@dataclass(frozen=True)
class BaselineSample:
    """One normalized physical session used by the threshold detector."""

    session_id: str
    signal: NDArray[np.float32]
    targets: tuple[SegmentEvent, ...]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def close_short_gaps(active: NDArray[np.bool_], maximum_gap: int) -> NDArray[np.bool_]:
    """Close inactive runs bounded by detections when their length is allowed."""
    if active.ndim != 1:
        raise ValueError("active must be one-dimensional.")
    if maximum_gap < 0:
        raise ValueError("maximum_gap cannot be negative.")
    closed = active.copy()
    if maximum_gap == 0 or len(closed) < 3:
        return closed
    starts, ends = _true_runs(closed)
    for left_end, right_start in zip(ends[:-1], starts[1:], strict=True):
        if right_start - left_end <= maximum_gap:
            closed[left_end:right_start] = True
    return closed


def infer_events(
    signal: NDArray[np.float32],
    parameters: ThresholdParameters,
    *,
    source_id: str,
) -> tuple[SegmentEvent, ...]:
    """Detect threshold runs and infer direction from the two pulse extrema."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    active = np.abs(signal) >= parameters.threshold
    active = close_short_gaps(active, parameters.merge_gap_frames)
    starts, ends = _true_runs(active)
    events = []
    for start, end_exclusive in zip(starts, ends, strict=True):
        duration = int(end_exclusive - start)
        if duration < parameters.minimum_duration_frames:
            continue
        segment = signal[start:end_exclusive]
        positive_first = int(np.argmax(segment)) < int(np.argmin(segment))
        direction = (
            parameters.positive_first_direction
            if positive_first
            else parameters.negative_first_direction
        )
        events.append(
            SegmentEvent(
                start_idx=int(start),
                end_idx=int(end_exclusive - 1),
                direction=direction,
                confidence=float(
                    min(np.max(np.abs(segment)) / parameters.threshold, 1.0)
                ),
                source_id=source_id,
            )
        )
    return tuple(events)


def _true_runs(
    active: NDArray[np.bool_],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    padded = np.concatenate(
        (
            np.zeros(1, dtype=np.int8),
            active.astype(np.int8, copy=False),
            np.zeros(1, dtype=np.int8),
        )
    )
    differences = np.diff(padded)
    starts = np.flatnonzero(differences == 1).astype(np.int64, copy=False)
    ends = np.flatnonzero(differences == -1).astype(np.int64, copy=False)
    return starts, ends


def learn_positive_first_direction(
    samples: Iterable[BaselineSample],
) -> tuple[Direction, dict[str, int]]:
    """Learn the physical pulse-order mapping from training events only."""
    votes = {
        "positive_first_left_correct": 0,
        "positive_first_right_correct": 0,
    }
    for sample in samples:
        for target in sample.targets:
            segment = sample.signal[target.start_idx : target.end_idx + 1]
            positive_first = int(np.argmax(segment)) < int(np.argmin(segment))
            left_mapping_is_correct = (
                positive_first and target.direction == Direction.LEFT
            ) or (not positive_first and target.direction == Direction.RIGHT)
            key = (
                "positive_first_left_correct"
                if left_mapping_is_correct
                else "positive_first_right_correct"
            )
            votes[key] += 1
    left_votes = votes["positive_first_left_correct"]
    right_votes = votes["positive_first_right_correct"]
    direction = Direction.LEFT if left_votes >= right_votes else Direction.RIGHT
    return direction, votes


def evaluate_samples(
    samples: Iterable[BaselineSample],
    parameters: ThresholdParameters,
    *,
    iou_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    targets = []
    predictions = []
    for sample in samples:
        targets.extend(sample.targets)
        predictions.extend(
            infer_events(sample.signal, parameters, source_id=sample.session_id)
        )
    return event_metrics(
        tuple(targets),
        tuple(predictions),
        iou_thresholds=iou_thresholds,
    )


def run_threshold_baseline(
    *,
    dataset_dir: str | Path,
    split_manifest_path: str | Path,
    normalization_manifest_path: str | Path,
    output_path: str | Path,
    thresholds: tuple[float, ...],
    merge_gap_frames: tuple[int, ...],
    minimum_duration_frames: tuple[int, ...],
    iou_thresholds: tuple[float, ...] = (0.10, 0.25, 0.50),
) -> dict[str, Any]:
    """Select baseline parameters on validation and evaluate each test fold."""
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite baseline report: {destination}")
    dataset_root = Path(dataset_dir).resolve()
    split_path = Path(split_manifest_path).resolve()
    normalization_path = Path(normalization_manifest_path).resolve()
    split_manifest = _read_json(split_path)
    normalization_manifest = _read_json(normalization_path)
    sequences = _load_sequences(dataset_root)
    normalization_by_fold = {
        str(fold["fold_id"]): fold["signal"] for fold in normalization_manifest["folds"]
    }
    fold_reports = []
    for fold in split_manifest["folds"]:
        fold_id = str(fold["fold_id"])
        statistics = normalization_by_fold[fold_id]
        samples = {
            split: _samples(
                fold[f"{split}_session_ids"],
                sequences,
                mean_value=float(statistics["mean"]),
                standard_deviation=float(statistics["standard_deviation"]),
            )
            for split in ("train", "validation", "test")
        }
        positive_first_direction, polarity_votes = learn_positive_first_direction(
            samples["train"]
        )
        selected, validation_metrics = _select_parameters(
            samples["validation"],
            positive_first_direction=positive_first_direction,
            thresholds=thresholds,
            merge_gap_frames=merge_gap_frames,
            minimum_duration_frames=minimum_duration_frames,
            iou_thresholds=iou_thresholds,
        )
        test_metrics = evaluate_samples(
            samples["test"], selected, iou_thresholds=iou_thresholds
        )
        fold_reports.append(
            {
                "fold_id": fold_id,
                "parameters": _parameter_payload(selected),
                "polarity_votes": polarity_votes,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
            }
        )

    report = {
        "schema_version": 1,
        "method": "absolute_threshold_on_fold_normalized_signal",
        "selection": "validation_event_f1_at_iou_0.25",
        "tie_breaks": [
            "validation_absolute_count_error",
            "validation_false_positive_count",
            "fixed_parameter_order",
        ],
        "protocol": split_manifest["protocol"],
        "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
        "split_manifest_sha256": file_sha256(split_path),
        "normalization_manifest_sha256": file_sha256(normalization_path),
        "source_code_sha256": source_code_sha256(find_project_root(dataset_root)),
        "grid": {
            "thresholds": list(thresholds),
            "merge_gap_frames": list(merge_gap_frames),
            "minimum_duration_frames": list(minimum_duration_frames),
        },
        "folds": fold_reports,
        "macro_summary": _macro_summary(fold_reports, iou_thresholds),
        "aggregate_metrics": microaggregate_fold_metrics(fold_reports, iou_thresholds),
    }
    _write_json(destination, report)
    return report


def _load_sequences(dataset_root: Path) -> dict[str, PreprocessedSequence]:
    manifest = _read_json(dataset_root / "manifest.json")
    return {
        str(record["session_id"]): load_preprocessed_sequence(
            dataset_root / str(record["output"])
        )
        for record in manifest["sequences"]
    }


def _samples(
    session_ids: Iterable[str],
    sequences: dict[str, PreprocessedSequence],
    *,
    mean_value: float,
    standard_deviation: float,
) -> tuple[BaselineSample, ...]:
    if standard_deviation <= 0.0:
        raise ValueError("Normalization standard deviation must be positive.")
    samples = []
    for session_id in sorted(session_ids):
        sequence = sequences[session_id]
        signal = np.asarray(
            (sequence.signal - mean_value) / standard_deviation,
            dtype=np.float32,
        )
        samples.append(
            BaselineSample(
                session_id=session_id,
                signal=signal,
                targets=target_events_from_passages(
                    sequence.events, source_id=session_id
                ),
            )
        )
    return tuple(samples)


def _select_parameters(
    validation_samples: tuple[BaselineSample, ...],
    *,
    positive_first_direction: Direction,
    thresholds: tuple[float, ...],
    merge_gap_frames: tuple[int, ...],
    minimum_duration_frames: tuple[int, ...],
    iou_thresholds: tuple[float, ...],
) -> tuple[ThresholdParameters, dict[str, Any]]:
    best_score: tuple[float, int, int] | None = None
    best_parameters: ThresholdParameters | None = None
    best_selection: dict[str, Any] | None = None
    for threshold in thresholds:
        for gap in merge_gap_frames:
            for duration in minimum_duration_frames:
                parameters = ThresholdParameters(
                    threshold=float(threshold),
                    merge_gap_frames=int(gap),
                    minimum_duration_frames=int(duration),
                    positive_first_direction=positive_first_direction,
                )
                selection = _selection_metrics(validation_samples, parameters)
                score = (
                    float(selection["f1"]),
                    -abs(int(selection["count_error"])),
                    -int(selection["fp"]),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_parameters = parameters
                    best_selection = selection
    if best_parameters is None or best_selection is None:
        raise ValueError("Baseline grid cannot be empty.")
    validation_metrics = evaluate_samples(
        validation_samples,
        best_parameters,
        iou_thresholds=iou_thresholds,
    )
    return best_parameters, validation_metrics


def _selection_metrics(
    samples: tuple[BaselineSample, ...],
    parameters: ThresholdParameters,
) -> dict[str, float | int]:
    targets = []
    predictions = []
    for sample in samples:
        targets.extend(sample.targets)
        predictions.extend(
            infer_events(sample.signal, parameters, source_id=sample.session_id)
        )
    targets_tuple = tuple(targets)
    predictions_tuple = tuple(predictions)
    matches = match_events(
        targets_tuple,
        predictions_tuple,
        iou_threshold=0.25,
    )
    true_positive = len(matches)
    false_positive = len(predictions_tuple) - true_positive
    false_negative = len(targets_tuple) - true_positive
    precision = (
        0.0
        if true_positive + false_positive == 0
        else true_positive / (true_positive + false_positive)
    )
    recall = (
        0.0
        if true_positive + false_negative == 0
        else true_positive / (true_positive + false_negative)
    )
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count_error": len(predictions_tuple) - len(targets_tuple),
    }


def _parameter_payload(parameters: ThresholdParameters) -> dict[str, Any]:
    payload = asdict(parameters)
    payload["positive_first_direction"] = parameters.positive_first_direction.name
    payload["negative_first_direction"] = parameters.negative_first_direction.name
    return payload


def microaggregate_fold_metrics(
    folds: list[dict[str, Any]],
    iou_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    reports = [fold["test_metrics"] for fold in folds]
    target_events = sum(int(report["target_events"]) for report in reports)
    predicted_events = sum(int(report["predicted_events"]) for report in reports)
    thresholds = {}
    for threshold in iou_thresholds:
        key = f"iou_{int(round(threshold * 100)):02d}"
        values = [report["thresholds"][key] for report in reports]
        threshold_summary = _summed_classification_metrics(values)
        threshold_summary["threshold"] = threshold
        threshold_summary["by_direction"] = {
            direction: _summed_classification_metrics(
                [value["by_direction"][direction] for value in values]
            )
            for direction in ("LEFT", "RIGHT")
        }
        thresholds[key] = threshold_summary
    return {
        "thresholds": thresholds,
        "target_events": target_events,
        "predicted_events": predicted_events,
        "count_error": predicted_events - target_events,
        "direction_counts": {
            category: {
                direction: sum(
                    int(report["direction_counts"][category].get(direction, 0))
                    for report in reports
                )
                for direction in ("LEFT", "RIGHT")
            }
            for category in ("targets", "predictions")
        },
        "wrong_direction_overlaps_iou_01": sum(
            int(report["wrong_direction_overlaps_iou_01"]) for report in reports
        ),
        "fragmentation_candidates": sum(
            int(report["fragmentation_candidates"]) for report in reports
        ),
        "merge_candidates": sum(int(report["merge_candidates"]) for report in reports),
    }


def _summed_classification_metrics(
    reports: list[dict[str, Any]],
) -> dict[str, int | float]:
    true_positive = sum(int(report["tp"]) for report in reports)
    false_positive = sum(int(report["fp"]) for report in reports)
    false_negative = sum(int(report["fn"]) for report in reports)
    precision = (
        0.0
        if true_positive + false_positive == 0
        else true_positive / (true_positive + false_positive)
    )
    recall = (
        0.0
        if true_positive + false_negative == 0
        else true_positive / (true_positive + false_negative)
    )
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _macro_summary(
    folds: list[dict[str, Any]],
    iou_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for threshold in iou_thresholds:
        key = f"iou_{int(round(threshold * 100)):02d}"
        for metric in ("precision", "recall", "f1"):
            values = [
                float(fold["test_metrics"]["thresholds"][key][metric]) for fold in folds
            ]
            summary[f"{metric}_{key}"] = {
                "mean": mean(values),
                "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
            }
    count_errors = [abs(int(fold["test_metrics"]["count_error"])) for fold in folds]
    summary["absolute_count_error"] = {
        "mean": mean(count_errors),
        "standard_deviation": (stdev(count_errors) if len(count_errors) > 1 else 0.0),
    }
    return summary
