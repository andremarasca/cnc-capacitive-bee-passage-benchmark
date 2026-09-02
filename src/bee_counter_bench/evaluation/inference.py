"""Full-session inference with halo discard and post-stitch event extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.special import softmax
from torch import nn

from bee_counter_bench.config import (
    AppConfig,
    NormalizationConfig,
    config_to_dict,
    normalization_config_to_dict,
)
from bee_counter_bench.data.normalization import normalize_session_signal
from bee_counter_bench.data.schemas import PreprocessedSequence
from bee_counter_bench.data.sequence_io import (
    file_sha256,
    load_preprocessed_sequence,
)
from bee_counter_bench.evaluation.events import (
    SegmentEvent,
    events_from_labels,
    target_events_from_passages,
)
from bee_counter_bench.evaluation.metrics import (
    event_metrics,
    frame_metrics,
    match_events,
    temporal_error_metrics,
)
from bee_counter_bench.provenance import find_project_root, source_code_sha256
from bee_counter_bench.training.checkpoint import load_checkpoint
from bee_counter_bench.training.trainer import build_model


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


def stitch_session_logits(
    *,
    model: nn.Module,
    signal: np.ndarray,
    mean_value: float,
    standard_deviation: float,
    core_frames: int,
    halo_frames: int,
    device: torch.device,
    normalization_config: NormalizationConfig | None = None,
) -> np.ndarray:
    """Infer overlapping inputs and retain each non-overlapping core once."""
    if signal.ndim != 1 or not len(signal):
        raise ValueError("signal must be a non-empty one-dimensional array.")
    if standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be positive.")
    if core_frames <= 0 or halo_frames < 0:
        raise ValueError("Invalid core or halo length.")
    normalized = normalize_session_signal(
        signal,
        training_mean=mean_value,
        training_standard_deviation=standard_deviation,
        config=normalization_config,
    )
    logits = np.empty((3, len(normalized)), dtype=np.float32)
    with torch.no_grad():
        for core_start in range(0, len(normalized), core_frames):
            core_end = min(len(normalized), core_start + core_frames)
            input_start = max(0, core_start - halo_frames)
            input_end = min(len(normalized), core_end + halo_frames)
            inputs = torch.from_numpy(
                normalized[input_start:input_end][None, None, :]
            ).to(device)
            mask = torch.ones((1, input_end - input_start), dtype=torch.bool)
            output = model(inputs, mask.to(device))[-1, 0]
            local_start = core_start - input_start
            local_end = local_start + core_end - core_start
            logits[:, core_start:core_end] = (
                output[:, local_start:local_end].detach().cpu().numpy()
            )
    return logits


def evaluate_checkpoint(
    *,
    config: AppConfig,
    checkpoint_path: str | Path,
    dataset_dir: str | Path,
    split_manifest_path: str | Path,
    normalization_manifest_path: str | Path,
    fold_id: str,
    output_path: str | Path,
    device_name: str = "auto",
    core_frames: int = 900,
    halo_frames: int = 142,
    normalization_config: NormalizationConfig | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint on untouched test sessions."""
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {destination}")
    checkpoint_file = Path(checkpoint_path).resolve()
    dataset_root = Path(dataset_dir).resolve()
    split_path = Path(split_manifest_path).resolve()
    normalization_path = Path(normalization_manifest_path).resolve()
    checkpoint = load_checkpoint(checkpoint_file)
    _validate_provenance(
        checkpoint,
        config=config,
        dataset_root=dataset_root,
        split_path=split_path,
        normalization_path=normalization_path,
        normalization_config=normalization_config,
        fold_id=fold_id,
    )
    split_manifest = _read_json(split_path)
    normalization_manifest = _read_json(normalization_path)
    fold = _find_fold(split_manifest, fold_id)
    statistics = _find_fold(normalization_manifest, fold_id)["signal"]
    sequences = _load_test_sequences(
        dataset_root,
        fold["test_session_ids"],
    )
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_target_labels = []
    all_prediction_labels = []
    all_targets: list[SegmentEvent] = []
    all_predictions: list[SegmentEvent] = []
    target_speeds = []
    session_reports = []
    for sequence in sequences:
        logits = stitch_session_logits(
            model=model,
            signal=sequence.signal,
            mean_value=float(statistics["mean"]),
            standard_deviation=float(statistics["standard_deviation"]),
            core_frames=core_frames,
            halo_frames=halo_frames,
            device=device,
            normalization_config=normalization_config,
        )
        probabilities = softmax(logits, axis=0)
        predicted_labels = np.asarray(logits.argmax(axis=0), dtype=np.int64)
        targets = target_events_from_passages(
            sequence.events,
            source_id=sequence.session_id,
        )
        predictions = events_from_labels(
            predicted_labels,
            probabilities=probabilities,
            source_id=sequence.session_id,
        )
        all_target_labels.append(sequence.labels)
        all_prediction_labels.append(predicted_labels)
        all_targets.extend(targets)
        all_predictions.extend(predictions)
        target_speeds.extend(event.mean_speed_mm_min for event in sequence.events)
        session_reports.append(
            {
                "session_id": sequence.session_id,
                "frames": len(sequence.signal),
                "target_events": len(targets),
                "predicted_events": [
                    {
                        "start_idx": event.start_idx,
                        "end_idx": event.end_idx,
                        "direction": event.direction.name,
                        "confidence": event.confidence,
                    }
                    for event in predictions
                ],
            }
        )

    targets_tuple = tuple(all_targets)
    predictions_tuple = tuple(all_predictions)
    iou_thresholds = config.evaluation.segment_iou_thresholds
    primary_iou = 0.25
    report = {
        "schema_version": 1,
        "fold_id": fold_id,
        "checkpoint_sha256": file_sha256(checkpoint_file),
        "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
        "split_manifest_sha256": file_sha256(split_path),
        "normalization_manifest_sha256": file_sha256(normalization_path),
        "normalization_runtime": normalization_config_to_dict(
            normalization_config
        ),
        "model_runtime": config_to_dict(config)["model"],
        "source_code_sha256": source_code_sha256(find_project_root(dataset_root)),
        "inference": {
            "device": str(device),
            "core_frames": core_frames,
            "halo_frames": halo_frames,
            "stitch": "discard_internal_halo_then_concatenate_logits",
            "event_extraction": "after_session_logit_stitching",
        },
        "frame_metrics": frame_metrics(
            np.concatenate(all_target_labels),
            np.concatenate(all_prediction_labels),
        ),
        "event_metrics": event_metrics(
            targets_tuple,
            predictions_tuple,
            iou_thresholds=iou_thresholds,
        ),
        "temporal_errors": temporal_error_metrics(
            targets_tuple,
            predictions_tuple,
            iou_threshold=primary_iou,
            sample_rate_hz=float(config.preprocessing.output_sample_rate_hz),
            include_values=True,
        ),
        "speed_stratified_target_recall": _speed_stratified_recall(
            targets_tuple,
            predictions_tuple,
            target_speeds,
            iou_threshold=primary_iou,
        ),
        "sessions": session_reports,
    }
    _write_json(destination, report)
    return report


def _load_test_sequences(
    dataset_root: Path,
    session_ids: Iterable[str],
) -> tuple[PreprocessedSequence, ...]:
    manifest = _read_json(dataset_root / "manifest.json")
    artifacts = {
        str(record["session_id"]): dataset_root / str(record["output"])
        for record in manifest["sequences"]
    }
    return tuple(
        load_preprocessed_sequence(artifacts[session_id])
        for session_id in sorted(session_ids)
    )


def _find_fold(manifest: dict[str, Any], fold_id: str) -> dict[str, Any]:
    for fold in manifest["folds"]:
        if str(fold["fold_id"]) == fold_id:
            return fold
    raise ValueError(f"Unknown fold_id: {fold_id}")


def _validate_provenance(
    checkpoint: dict[str, Any],
    *,
    config: AppConfig,
    dataset_root: Path,
    split_path: Path,
    normalization_path: Path,
    normalization_config: NormalizationConfig | None,
    fold_id: str,
) -> None:
    if str(checkpoint["fold_id"]) != fold_id:
        raise ValueError("Checkpoint fold does not match requested evaluation fold.")
    expected = {
        "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
        "split_manifest_sha256": file_sha256(split_path),
        "normalization_manifest_sha256": file_sha256(normalization_path),
        "normalization_runtime": normalization_config_to_dict(
            normalization_config
        ),
        "model": config_to_dict(config)["model"],
        "source_code_sha256": source_code_sha256(find_project_root(dataset_root)),
    }
    provenance = checkpoint["provenance"]
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint provenance mismatch: {mismatches}")


def _speed_stratified_recall(
    targets: tuple[SegmentEvent, ...],
    predictions: tuple[SegmentEvent, ...],
    speeds_mm_min: list[float],
    *,
    iou_threshold: float,
) -> dict[str, Any]:
    if len(targets) != len(speeds_mm_min):
        raise ValueError("Every target event must have one speed value.")
    matches = match_events(targets, predictions, iou_threshold=iou_threshold)
    matched_by_target = {match.target_index: match for match in matches}
    bins = (
        ("below_1500", None, 1500.0),
        ("1500_to_below_2500", 1500.0, 2500.0),
        ("2500_and_above", 2500.0, None),
    )
    report = {}
    for name, lower, upper in bins:
        indices = [
            index
            for index, speed in enumerate(speeds_mm_min)
            if (lower is None or speed >= lower) and (upper is None or speed < upper)
        ]
        matched = [
            matched_by_target[index] for index in indices if index in matched_by_target
        ]
        absolute_errors = {"start": [], "end": [], "center": []}
        for match in matched:
            target = targets[match.target_index]
            prediction = predictions[match.prediction_index]
            absolute_errors["start"].append(
                float(abs(prediction.start_idx - target.start_idx))
            )
            absolute_errors["end"].append(
                float(abs(prediction.end_idx - target.end_idx))
            )
            absolute_errors["center"].append(
                float(
                    abs(
                        (prediction.start_idx + prediction.end_idx) / 2.0
                        - (target.start_idx + target.end_idx) / 2.0
                    )
                )
            )
        report[name] = {
            "lower_mm_min_inclusive": lower,
            "upper_mm_min_exclusive": upper,
            "target_events": len(indices),
            "matched_events": len(matched),
            "recall": 0.0 if not indices else len(matched) / len(indices),
            "median_matched_iou": (
                0.0 if not matched else float(np.median([item.iou for item in matched]))
            ),
            "temporal_errors": {
                key: _error_summary(values) for key, values in absolute_errors.items()
            },
            "absolute_errors_frames": absolute_errors,
        }
    return report


def _error_summary(values: list[float]) -> dict[str, float | int]:
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
        "count": len(values),
        "median_frames": median_frames,
        "p95_frames": p95_frames,
        "median_seconds": median_frames / 30.0,
        "p95_seconds": p95_frames / 30.0,
    }
