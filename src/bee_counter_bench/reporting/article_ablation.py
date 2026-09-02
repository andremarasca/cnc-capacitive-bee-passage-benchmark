"""Build article-facing normalization and matching sensitivity reports."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

from bee_counter_bench.data.schemas import Direction
from bee_counter_bench.data.sequence_io import (
    file_sha256,
    load_preprocessed_sequence,
)
from bee_counter_bench.evaluation.events import (
    SegmentEvent,
    target_events_from_passages,
)
from bee_counter_bench.evaluation.metrics import event_detection_metrics

CRITERIA = (("iou_25", 0.25), ("iou_50", 0.50), ("any_overlap", None))
DIRECTION_MODES = (("directional", True), ("presence", False))
CAUSAL_PROTOCOLS = (
    "main_session_grouped",
    "leave_one_collection_out",
    "pairwise_collection_transfer",
)


@dataclass(frozen=True, order=True)
class EvaluationKey:
    protocol: str
    seed: int
    fold_id: str


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _discover_evaluations(root: Path) -> dict[EvaluationKey, dict[str, Any]]:
    evaluations: dict[EvaluationKey, dict[str, Any]] = {}
    for path in sorted(root.rglob("evaluation.json")):
        relative = path.relative_to(root)
        if len(relative.parts) != 4:
            continue
        protocol, seed_directory, fold_id, _ = relative.parts
        if not seed_directory.startswith("seed_"):
            continue
        key = EvaluationKey(
            protocol=protocol,
            seed=int(seed_directory.removeprefix("seed_")),
            fold_id=fold_id,
        )
        if key in evaluations:
            raise ValueError(f"Duplicate evaluation: {key}")
        evaluations[key] = _read_json(path)
    return evaluations


def _load_targets(project_root: Path) -> dict[str, tuple[SegmentEvent, ...]]:
    dataset_root = project_root / "outputs" / "processed" / "dataset"
    manifest = _read_json(dataset_root / "manifest.json")
    targets = {}
    for record in manifest["sequences"]:
        sequence = load_preprocessed_sequence(dataset_root / str(record["output"]))
        targets[sequence.session_id] = target_events_from_passages(
            sequence.events,
            source_id=sequence.session_id,
        )
    return targets


def _predictions(session: dict[str, Any]) -> tuple[SegmentEvent, ...]:
    session_id = str(session["session_id"])
    return tuple(
        SegmentEvent(
            start_idx=int(event["start_idx"]),
            end_idx=int(event["end_idx"]),
            direction=Direction[str(event["direction"])],
            confidence=float(event["confidence"]),
            source_id=session_id,
        )
        for event in session["predicted_events"]
    )


def _evaluate_matching(
    evaluation: dict[str, Any],
    targets_by_session: dict[str, tuple[SegmentEvent, ...]],
) -> dict[tuple[str, str], dict[str, float | int]]:
    counts = {
        (mode, criterion): {"tp": 0, "fp": 0, "fn": 0}
        for mode, _ in DIRECTION_MODES
        for criterion, _ in CRITERIA
    }
    for session in evaluation["sessions"]:
        session_id = str(session["session_id"])
        targets = targets_by_session[session_id]
        predictions = _predictions(session)
        for mode, require_direction in DIRECTION_MODES:
            for criterion, threshold in CRITERIA:
                metrics = event_detection_metrics(
                    targets,
                    predictions,
                    iou_threshold=threshold,
                    require_direction=require_direction,
                )
                for name in ("tp", "fp", "fn"):
                    counts[(mode, criterion)][name] += int(metrics[name])
    report = {}
    for key, values in counts.items():
        precision = _ratio(values["tp"], values["tp"] + values["fp"])
        recall = _ratio(values["tp"], values["tp"] + values["fn"])
        report[key] = {
            **values,
            "precision": precision,
            "recall": recall,
            "f1": _ratio(2.0 * precision * recall, precision + recall),
            "target_events": values["tp"] + values["fn"],
            "predicted_events": values["tp"] + values["fp"],
        }
    _validate_stored_directional_metrics(evaluation, report)
    return report


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _validate_stored_directional_metrics(
    evaluation: dict[str, Any],
    report: dict[tuple[str, str], dict[str, float | int]],
) -> None:
    for criterion in ("iou_25", "iou_50"):
        stored = evaluation["event_metrics"]["thresholds"][criterion]
        recalculated = report[("directional", criterion)]
        for name in ("tp", "fp", "fn"):
            if int(stored[name]) != int(recalculated[name]):
                raise ValueError(
                    f"Stored and recalculated {criterion} {name} differ."
                )


def _mean_sd(values: Iterable[float]) -> tuple[float, float]:
    materialized = [float(value) for value in values]
    return mean(materialized), stdev(materialized) if len(materialized) > 1 else 0.0


def _sensitivity_rows(
    normalization: str,
    evaluations: dict[EvaluationKey, dict[str, Any]],
    targets_by_session: dict[str, tuple[SegmentEvent, ...]],
) -> list[dict[str, Any]]:
    evaluated = {
        key: _evaluate_matching(value, targets_by_session)
        for key, value in evaluations.items()
    }
    grouped: dict[tuple[str, int], list[EvaluationKey]] = defaultdict(list)
    for key in evaluated:
        grouped[(key.protocol, key.seed)].append(key)
    rows = []
    protocols = sorted({key.protocol for key in evaluated})
    for protocol in protocols:
        seeds = sorted(seed for candidate, seed in grouped if candidate == protocol)
        for mode, require_direction in DIRECTION_MODES:
            for criterion, threshold in CRITERIA:
                seed_metrics = []
                for seed in seeds:
                    counts = {"tp": 0, "fp": 0, "fn": 0}
                    for key in grouped[(protocol, seed)]:
                        metrics = evaluated[key][(mode, criterion)]
                        for name in counts:
                            counts[name] += int(metrics[name])
                    precision = _ratio(counts["tp"], counts["tp"] + counts["fp"])
                    recall = _ratio(counts["tp"], counts["tp"] + counts["fn"])
                    seed_metrics.append(
                        {
                            **counts,
                            "precision": precision,
                            "recall": recall,
                            "f1": _ratio(
                                2.0 * precision * recall,
                                precision + recall,
                            ),
                            "target_events": counts["tp"] + counts["fn"],
                            "predicted_events": counts["tp"] + counts["fp"],
                        }
                    )
                row: dict[str, Any] = {
                    "normalization": normalization,
                    "protocol": protocol,
                    "direction_mode": mode,
                    "require_direction": require_direction,
                    "criterion": criterion,
                    "iou_threshold": threshold,
                    "seeds": len(seeds),
                }
                for metric in (
                    "tp",
                    "fp",
                    "fn",
                    "precision",
                    "recall",
                    "f1",
                    "target_events",
                    "predicted_events",
                ):
                    metric_mean, metric_sd = _mean_sd(
                        item[metric] for item in seed_metrics
                    )
                    row[f"{metric}_mean"] = metric_mean
                    row[f"{metric}_sd"] = metric_sd
                row["count_bias_percent_mean"] = 100.0 * _ratio(
                    row["predicted_events_mean"] - row["target_events_mean"],
                    row["target_events_mean"],
                )
                rows.append(row)
    return rows


def _comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (
            row["normalization"],
            row["protocol"],
            row["direction_mode"],
            row["criterion"],
        ): row
        for row in rows
    }
    comparisons = []
    for protocol in CAUSAL_PROTOCOLS:
        for mode, _ in DIRECTION_MODES:
            for criterion, _ in CRITERIA:
                before = indexed[("training_population", protocol, mode, criterion)]
                after = indexed[("causal_adaptive_tau60", protocol, mode, criterion)]
                comparison = {
                    "protocol": protocol,
                    "direction_mode": mode,
                    "criterion": criterion,
                    "seeds": before["seeds"],
                }
                for metric in (
                    "tp_mean",
                    "fp_mean",
                    "fn_mean",
                    "precision_mean",
                    "recall_mean",
                    "f1_mean",
                    "predicted_events_mean",
                    "count_bias_percent_mean",
                ):
                    comparison[f"before_{metric}"] = before[metric]
                    comparison[f"after_{metric}"] = after[metric]
                    comparison[f"delta_{metric}"] = after[metric] - before[metric]
                comparisons.append(comparison)
    return comparisons


def write_article_ablation_reports(
    *,
    project_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Recalculate all article sensitivity results from canonical artifacts."""
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output: {destination}")
    targets = _load_targets(root)
    baseline = _discover_evaluations(root / "outputs" / "runs")
    causal = _discover_evaluations(
        root / "outputs" / "ablations" / "causal_adaptive_tau60"
    )
    expected_causal = {
        key for key in baseline if key.protocol in CAUSAL_PROTOCOLS
    }
    if set(causal) != expected_causal:
        raise ValueError("Causal and baseline task matrices do not match.")
    baseline_rows = _sensitivity_rows("training_population", baseline, targets)
    causal_rows = _sensitivity_rows("causal_adaptive_tau60", causal, targets)
    sensitivity = baseline_rows + causal_rows
    comparisons = _comparison_rows(sensitivity)
    _write_csv(destination / "matching_sensitivity.csv", sensitivity)
    _write_csv(destination / "causal_normalization_comparison.csv", comparisons)
    summary = {
        "schema_version": 1,
        "normalization_configuration": {
            "path": "configs/normalization_causal_tau60.yaml",
            "sha256": file_sha256(
                root / "configs" / "normalization_causal_tau60.yaml"
            ),
        },
        "canonical_artifacts": {
            "baseline_evaluations": len(baseline),
            "causal_evaluations": len(causal),
            "causal_promotion_manifest_sha256": file_sha256(
                root
                / "outputs"
                / "ablations"
                / "causal_adaptive_tau60"
                / "promotion_manifest.json"
            ),
        },
        "matching_sensitivity": sensitivity,
        "normalization_comparison": comparisons,
    }
    _write_json(destination / "causal_normalization_summary.json", summary)
    return summary
