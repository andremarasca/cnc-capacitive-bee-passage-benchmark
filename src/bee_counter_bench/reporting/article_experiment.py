"""Traceable article reports for the frozen RF165 experiment campaign."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

from bee_counter_bench.config import (
    config_to_dict,
    load_config,
    load_normalization_config,
    normalization_config_to_dict,
)
from bee_counter_bench.data.schemas import Direction
from bee_counter_bench.data.sequence_io import (
    file_sha256,
    load_preprocessed_sequence,
)
from bee_counter_bench.evaluation.events import (
    SegmentEvent,
    target_events_from_passages,
)
from bee_counter_bench.evaluation.metrics import (
    event_detection_metrics,
    match_events,
)
from bee_counter_bench.provenance import source_code_sha256
from bee_counter_bench.training.article_experiment import (
    ArticleExperimentPlan,
    ArticleExperimentTask,
    build_article_tasks,
    ensure_campaign_manifest,
)

CRITERIA: tuple[tuple[str, float | None], ...] = (
    ("iou_25", 0.25),
    ("iou_50", 0.50),
    ("any_overlap", None),
)
DIRECTION_MODES: tuple[tuple[str, bool], ...] = (
    ("directional", True),
    ("presence", False),
)
DIRECTIONS = ("LEFT", "RIGHT")
LCO_PROTOCOL = "leave_one_collection_out"


@dataclass(frozen=True)
class TargetSession:
    """Canonical target events and frame count for one physical session."""

    events: tuple[SegmentEvent, ...]
    frames: int


@dataclass(frozen=True)
class MatchingResult:
    """Recalculated event metrics and presence-conditioned direction matrix."""

    metrics: dict[tuple[str, str], dict[str, float | int]]
    direction_confusion: tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class TaskArtifacts:
    """Validated artifacts and derived results for one campaign task."""

    task: ArticleExperimentTask
    run_dir: Path
    evaluation_path: Path
    run_manifest_path: Path
    checkpoint_path: Path
    evaluation: dict[str, Any]
    run_manifest: dict[str, Any]
    matching: MatchingResult
    collection_id: str | None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty report: {path}")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError(f"CSV rows have inconsistent columns: {path}")
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean_sd(values: Iterable[float]) -> tuple[float, float]:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("Cannot summarize an empty metric sequence.")
    return mean(materialized), stdev(materialized) if len(materialized) > 1 else 0.0


def _metric_from_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    target_events = tp + fn
    predicted_events = tp + fp
    count_bias = predicted_events - target_events
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2.0 * precision * recall, precision + recall),
        "target_events": target_events,
        "predicted_events": predicted_events,
        "count_bias": count_bias,
        "count_bias_percent": 100.0 * _ratio(count_bias, target_events),
    }


def _predictions(session: Mapping[str, Any]) -> tuple[SegmentEvent, ...]:
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


def evaluate_saved_sessions(
    evaluation: Mapping[str, Any],
    targets_by_session: Mapping[str, TargetSession],
    *,
    validate_stored_metrics: bool = True,
) -> MatchingResult:
    """Recalculate directional, presence, and direction-confusion results."""
    counts = {
        (mode, criterion): {"tp": 0, "fp": 0, "fn": 0}
        for mode, _ in DIRECTION_MODES
        for criterion, _ in CRITERIA
    }
    confusion = [[0, 0], [0, 0]]
    observed_session_ids: list[str] = []
    for session in evaluation["sessions"]:
        session_id = str(session["session_id"])
        if session_id in observed_session_ids:
            raise ValueError(f"Duplicate evaluation session: {session_id}")
        if session_id not in targets_by_session:
            raise ValueError(f"Unknown evaluation session: {session_id}")
        observed_session_ids.append(session_id)
        target_session = targets_by_session[session_id]
        if int(session["frames"]) != target_session.frames:
            raise ValueError(f"Frame count mismatch for session {session_id}.")
        if int(session["target_events"]) != len(target_session.events):
            raise ValueError(f"Target event count mismatch for session {session_id}.")
        predictions = _predictions(session)
        if any(event.end_idx >= target_session.frames for event in predictions):
            raise ValueError(f"Prediction exceeds session {session_id} bounds.")
        for mode, require_direction in DIRECTION_MODES:
            for criterion, threshold in CRITERIA:
                metrics = event_detection_metrics(
                    target_session.events,
                    predictions,
                    iou_threshold=threshold,
                    require_direction=require_direction,
                )
                for name in ("tp", "fp", "fn"):
                    counts[(mode, criterion)][name] += int(metrics[name])
        presence_matches = match_events(
            target_session.events,
            predictions,
            iou_threshold=0.25,
            require_direction=False,
        )
        for match in presence_matches:
            target = target_session.events[match.target_index]
            prediction = predictions[match.prediction_index]
            target_index = DIRECTIONS.index(target.direction.name)
            prediction_index = DIRECTIONS.index(prediction.direction.name)
            confusion[target_index][prediction_index] += 1

    result = MatchingResult(
        metrics={
            key: _metric_from_counts(
                int(value["tp"]),
                int(value["fp"]),
                int(value["fn"]),
            )
            for key, value in counts.items()
        },
        direction_confusion=(
            (confusion[0][0], confusion[0][1]),
            (confusion[1][0], confusion[1][1]),
        ),
    )
    if validate_stored_metrics:
        _validate_stored_directional_metrics(evaluation, result.metrics)
    return result


def _validate_stored_directional_metrics(
    evaluation: Mapping[str, Any],
    recalculated: Mapping[tuple[str, str], Mapping[str, float | int]],
) -> None:
    for criterion in ("iou_25", "iou_50"):
        stored = evaluation["event_metrics"]["thresholds"][criterion]
        current = recalculated[("directional", criterion)]
        for name in ("tp", "fp", "fn"):
            if int(stored[name]) != int(current[name]):
                raise ValueError(
                    f"Stored and recalculated {criterion} {name} differ."
                )


def _load_targets(project_root: Path) -> dict[str, TargetSession]:
    dataset_root = project_root / "outputs" / "processed" / "dataset"
    manifest = _read_json(dataset_root / "manifest.json")
    targets: dict[str, TargetSession] = {}
    for record in manifest["sequences"]:
        sequence = load_preprocessed_sequence(dataset_root / str(record["output"]))
        if sequence.session_id in targets:
            raise ValueError(f"Duplicate dataset session: {sequence.session_id}")
        targets[sequence.session_id] = TargetSession(
            events=target_events_from_passages(
                sequence.events,
                source_id=sequence.session_id,
            ),
            frames=len(sequence.signal),
        )
    return targets


def _normalization_contract(
    plan: ArticleExperimentPlan,
    variant: str,
) -> tuple[dict[str, Any], Path | None]:
    selected = next(
        (item for item in plan.normalizations if item.variant == variant),
        None,
    )
    if selected is None:
        raise ValueError(f"Unknown normalization variant: {variant}")
    config = (
        None
        if selected.config_path is None
        else load_normalization_config(selected.config_path)
    )
    contract = normalization_config_to_dict(config)
    if contract["variant"] != variant:
        raise ValueError(
            f"Plan variant {variant!r} differs from normalization contract."
        )
    return contract, selected.config_path


def _folds_by_protocol(plan: ArticleExperimentPlan) -> dict[str, dict[str, Any]]:
    return {
        protocol: _read_json(
            plan.project_root / "outputs" / "splits" / f"{protocol}.json"
        )
        for protocol in plan.protocols
    }


def _collection_by_fold(
    splits: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], str | None]:
    mapping: dict[tuple[str, str], str | None] = {}
    for protocol, split in splits.items():
        session_collections = {
            str(unit["session_id"]): str(unit["collection_id"])
            for unit in split["session_units"]
        }
        for fold in split["folds"]:
            fold_id = str(fold["fold_id"])
            collections = {
                session_collections[str(session_id)]
                for session_id in fold["test_session_ids"]
            }
            if protocol == LCO_PROTOCOL and len(collections) != 1:
                raise ValueError(
                    f"LCO fold {fold_id} does not test exactly one collection."
                )
            mapping[(protocol, fold_id)] = (
                next(iter(collections)) if len(collections) == 1 else None
            )
    return mapping


def _task_run_dir(
    plan: ArticleExperimentPlan,
    task: ArticleExperimentTask,
) -> Path:
    return (
        plan.output_root
        / task.normalization_variant
        / "runs"
        / task.protocol
        / f"seed_{task.seed}"
        / task.fold_id
    )


def _assert_equal(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"Provenance mismatch for {name}: observed={observed!r}, "
            f"expected={expected!r}."
        )


def _validate_task_artifacts(
    *,
    plan: ArticleExperimentPlan,
    task: ArticleExperimentTask,
    targets_by_session: Mapping[str, TargetSession],
    split: Mapping[str, Any],
    collection_id: str | None,
    expected_source_hash: str,
    expected_model_hash: str,
    expected_model_runtime: Mapping[str, Any],
    expected_halo_frames: int,
    expected_normalization: Mapping[str, Any],
) -> TaskArtifacts:
    run_dir = _task_run_dir(plan, task)
    evaluation_path = run_dir / "evaluation.json"
    run_manifest_path = run_dir / "run_manifest.json"
    checkpoint_path = run_dir / "training" / "best_checkpoint.pt"
    for path in (evaluation_path, run_manifest_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing campaign artifact: {path}")
    evaluation = _read_json(evaluation_path)
    run_manifest = _read_json(run_manifest_path)
    dataset_manifest = (
        plan.project_root / "outputs" / "processed" / "dataset" / "manifest.json"
    )
    split_path = plan.project_root / "outputs" / "splits" / f"{task.protocol}.json"
    normalization_path = (
        plan.project_root
        / "outputs"
        / "normalization"
        / f"{task.protocol}.json"
    )
    expected_evaluation = {
        "schema_version": 1,
        "fold_id": task.fold_id,
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
        "split_manifest_sha256": file_sha256(split_path),
        "normalization_manifest_sha256": file_sha256(normalization_path),
        "normalization_runtime": dict(expected_normalization),
        "model_runtime": dict(expected_model_runtime),
        "source_code_sha256": expected_source_hash,
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    for name, expected in expected_evaluation.items():
        _assert_equal(f"evaluation.{name}", evaluation.get(name), expected)
    inference = evaluation.get("inference", {})
    _assert_equal(
        "evaluation.inference.core_frames",
        inference.get("core_frames"),
        plan.core_frames,
    )
    _assert_equal(
        "evaluation.inference.halo_frames",
        inference.get("halo_frames"),
        expected_halo_frames,
    )
    expected_task = {
        "protocol": task.protocol,
        "fold_id": task.fold_id,
        "seed": task.seed,
    }
    _assert_equal(
        "run_manifest.experiment_id",
        run_manifest.get("experiment_id"),
        plan.experiment_id,
    )
    _assert_equal("run_manifest.task", run_manifest.get("task"), expected_task)
    _assert_equal(
        "run_manifest.source_code_sha256",
        run_manifest.get("source_code_sha256"),
        expected_source_hash,
    )
    _assert_equal(
        "run_manifest.model_configuration",
        run_manifest.get("model_configuration"),
        {
            "path": plan.model_path.relative_to(plan.project_root).as_posix(),
            "sha256": expected_model_hash,
        },
    )
    _assert_equal(
        "run_manifest.normalization_runtime",
        run_manifest.get("normalization_runtime"),
        dict(expected_normalization),
    )
    _assert_equal(
        "run_manifest.execution",
        run_manifest.get("execution"),
        {
            "core_frames": plan.core_frames,
            "halo_frames": expected_halo_frames,
        },
    )
    elapsed_seconds = run_manifest.get("elapsed_seconds")
    if not isinstance(elapsed_seconds, (int, float)) or not math.isfinite(
        float(elapsed_seconds)
    ) or float(elapsed_seconds) < 0.0:
        raise ValueError(f"Invalid elapsed time in {run_manifest_path}.")
    fold = next(
        (
            candidate
            for candidate in split["folds"]
            if str(candidate["fold_id"]) == task.fold_id
        ),
        None,
    )
    if fold is None:
        raise ValueError(f"Unknown split fold: {task.protocol}/{task.fold_id}")
    expected_sessions = {str(value) for value in fold["test_session_ids"]}
    observed_sessions = {
        str(session["session_id"]) for session in evaluation["sessions"]
    }
    _assert_equal(
        "evaluation test sessions",
        observed_sessions,
        expected_sessions,
    )
    matching = evaluate_saved_sessions(evaluation, targets_by_session)
    return TaskArtifacts(
        task=task,
        run_dir=run_dir,
        evaluation_path=evaluation_path,
        run_manifest_path=run_manifest_path,
        checkpoint_path=checkpoint_path,
        evaluation=evaluation,
        run_manifest=run_manifest,
        matching=matching,
        collection_id=collection_id,
    )


def _validate_exact_evaluation_matrix(
    plan: ArticleExperimentPlan,
    tasks: Sequence[ArticleExperimentTask],
) -> None:
    expected = {
        (_task_run_dir(plan, task) / "evaluation.json").resolve()
        for task in tasks
    }
    found = {
        path.resolve()
        for normalization in plan.normalizations
        for path in (
            plan.output_root / normalization.variant / "runs"
        ).rglob("evaluation.json")
    }
    missing = sorted(str(path) for path in expected - found)
    unexpected = sorted(str(path) for path in found - expected)
    if missing or unexpected:
        raise ValueError(
            "Campaign evaluation matrix differs from the frozen plan: "
            f"missing={missing}, unexpected={unexpected}."
        )


def _task_metric_rows(
    artifacts: Sequence[TaskArtifacts],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        task = artifact.task
        for mode, require_direction in DIRECTION_MODES:
            for criterion, threshold in CRITERIA:
                rows.append(
                    {
                        "normalization": task.normalization_variant,
                        "protocol": task.protocol,
                        "seed": task.seed,
                        "fold_id": task.fold_id,
                        "collection_id": artifact.collection_id,
                        "direction_mode": mode,
                        "require_direction": require_direction,
                        "criterion": criterion,
                        "iou_threshold": threshold,
                        **artifact.matching.metrics[(mode, criterion)],
                    }
                )
    return rows


def _pool_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    pooled: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        members = grouped[key]
        counts = {
            name: sum(int(member[name]) for member in members)
            for name in ("tp", "fp", "fn")
        }
        pooled.append(
            {
                **dict(zip(group_fields, key, strict=True)),
                "fold_count": len(members),
                **_metric_from_counts(**counts),
            }
        )
    return pooled


def summarize_across_seeds(
    seed_rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Calculate arithmetic mean and sample SD for every seed-level metric."""
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    summaries: list[dict[str, Any]] = []
    metric_names = (
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "target_events",
        "predicted_events",
        "count_bias",
        "count_bias_percent",
    )
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        members = grouped[key]
        seeds = [int(member["seed"]) for member in members]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seed rows for group {key}.")
        row: dict[str, Any] = {
            **dict(zip(group_fields, key, strict=True)),
            "seed_count": len(seeds),
        }
        for metric in metric_names:
            metric_mean, metric_sd = _mean_sd(member[metric] for member in members)
            row[f"{metric}_mean"] = metric_mean
            row[f"{metric}_sd"] = metric_sd
        summaries.append(row)
    return summaries


def _confusion_seed_rows(
    artifacts: Sequence[TaskArtifacts],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[TaskArtifacts]] = defaultdict(list)
    for artifact in artifacts:
        grouped[
            (
                artifact.task.normalization_variant,
                artifact.task.protocol,
                artifact.task.seed,
            )
        ].append(artifact)
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        matrix = [[0, 0], [0, 0]]
        for artifact in grouped[key]:
            for target_index in range(2):
                for prediction_index in range(2):
                    matrix[target_index][prediction_index] += (
                        artifact.matching.direction_confusion[target_index][
                            prediction_index
                        ]
                    )
        total = sum(sum(row) for row in matrix)
        accuracy = _ratio(matrix[0][0] + matrix[1][1], total)
        for target_index, target_direction in enumerate(DIRECTIONS):
            target_total = sum(matrix[target_index])
            for prediction_index, predicted_direction in enumerate(DIRECTIONS):
                matches = matrix[target_index][prediction_index]
                rows.append(
                    {
                        "normalization": key[0],
                        "protocol": key[1],
                        "seed": key[2],
                        "target_direction": target_direction,
                        "predicted_direction": predicted_direction,
                        "matches": matches,
                        "row_fraction": _ratio(matches, target_total),
                        "presence_matches": total,
                        "direction_accuracy": accuracy,
                    }
                )
    return rows


def _confusion_summary_rows(
    seed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    group_fields = (
        "normalization",
        "protocol",
        "target_direction",
        "predicted_direction",
    )
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        members = grouped[key]
        result: dict[str, Any] = {
            **dict(zip(group_fields, key, strict=True)),
            "seed_count": len(members),
        }
        for metric in (
            "matches",
            "row_fraction",
            "presence_matches",
            "direction_accuracy",
        ):
            metric_mean, metric_sd = _mean_sd(member[metric] for member in members)
            result[f"{metric}_mean"] = metric_mean
            result[f"{metric}_sd"] = metric_sd
        rows.append(result)
    return rows


def _model_run_rows(
    plan: ArticleExperimentPlan,
    artifacts: Sequence[TaskArtifacts],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        training = artifact.run_manifest["training"]
        task = artifact.task
        rows.append(
            {
                "normalization": task.normalization_variant,
                "protocol": task.protocol,
                "seed": task.seed,
                "fold_id": task.fold_id,
                "collection_id": artifact.collection_id,
                "elapsed_seconds": float(artifact.run_manifest["elapsed_seconds"]),
                "best_epoch": int(training["best_epoch"]),
                "epochs_completed": int(training["epochs_completed"]),
                "best_validation_loss": float(training["best_validation_loss"]),
                "evaluation_path": artifact.evaluation_path.relative_to(
                    plan.project_root
                ).as_posix(),
                "evaluation_sha256": file_sha256(artifact.evaluation_path),
                "run_manifest_path": artifact.run_manifest_path.relative_to(
                    plan.project_root
                ).as_posix(),
                "run_manifest_sha256": file_sha256(artifact.run_manifest_path),
                "checkpoint_path": artifact.checkpoint_path.relative_to(
                    plan.project_root
                ).as_posix(),
                "checkpoint_sha256": file_sha256(artifact.checkpoint_path),
                "source_code_sha256": artifact.evaluation["source_code_sha256"],
            }
        )
    return rows


def _timing_seed_rows(
    model_runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in model_runs:
        grouped[
            (str(row["normalization"]), str(row["protocol"]), int(row["seed"]))
        ].append(row)
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        elapsed = [float(row["elapsed_seconds"]) for row in grouped[key]]
        rows.append(
            {
                "normalization": key[0],
                "protocol": key[1],
                "seed": key[2],
                "fold_count": len(elapsed),
                "total_elapsed_seconds": sum(elapsed),
                "mean_task_elapsed_seconds": mean(elapsed),
                "min_task_elapsed_seconds": min(elapsed),
                "max_task_elapsed_seconds": max(elapsed),
            }
        )
    return rows


def _timing_summary_rows(
    seed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["normalization"]), str(row["protocol"]))].append(row)
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        members = grouped[key]
        result: dict[str, Any] = {
            "normalization": key[0],
            "protocol": key[1],
            "seed_count": len(members),
            "campaign_elapsed_seconds": sum(
                float(member["total_elapsed_seconds"]) for member in members
            ),
        }
        for metric in (
            "total_elapsed_seconds",
            "mean_task_elapsed_seconds",
            "min_task_elapsed_seconds",
            "max_task_elapsed_seconds",
        ):
            metric_mean, metric_sd = _mean_sd(member[metric] for member in members)
            result[f"{metric}_mean"] = metric_mean
            result[f"{metric}_sd"] = metric_sd
        rows.append(result)
    return rows


def _prepare_destination(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)


def write_article_experiment_reports(
    plan: ArticleExperimentPlan,
) -> dict[str, Any]:
    """Validate all planned RF165 runs and write article-facing reports."""
    campaign_manifest = ensure_campaign_manifest(plan)
    tasks = build_article_tasks(plan)
    _validate_exact_evaluation_matrix(plan, tasks)
    targets_by_session = _load_targets(plan.project_root)
    splits = _folds_by_protocol(plan)
    collections = _collection_by_fold(splits)
    current_source_hash = source_code_sha256(plan.project_root)
    model_hash = file_sha256(plan.model_path)
    model_config = load_config(
        plan.project_root / "configs" / "experiment.yaml",
        model_path=plan.model_path,
    )
    normalization_contracts = {
        normalization.variant: _normalization_contract(
            plan, normalization.variant
        )[0]
        for normalization in plan.normalizations
    }
    artifacts = [
        _validate_task_artifacts(
            plan=plan,
            task=task,
            targets_by_session=targets_by_session,
            split=splits[task.protocol],
            collection_id=collections[(task.protocol, task.fold_id)],
            expected_source_hash=current_source_hash,
            expected_model_hash=model_hash,
            expected_model_runtime=config_to_dict(model_config)["model"],
            expected_halo_frames=model_config.model.future_context_frames,
            expected_normalization=normalization_contracts[
                task.normalization_variant
            ],
        )
        for task in tasks
    ]
    task_metrics = _task_metric_rows(artifacts)
    seed_group_fields = (
        "normalization",
        "protocol",
        "seed",
        "direction_mode",
        "require_direction",
        "criterion",
        "iou_threshold",
    )
    seed_metrics = _pool_rows(task_metrics, group_fields=seed_group_fields)
    aggregate_group_fields = tuple(
        field for field in seed_group_fields if field != "seed"
    )
    aggregate_metrics = summarize_across_seeds(
        seed_metrics,
        group_fields=aggregate_group_fields,
    )
    lco_collection_seed_metrics = [
        {
            **{
                key: value
                for key, value in row.items()
                if key not in {"fold_id", "fold_count"}
            },
            "fold_count": 1,
        }
        for row in task_metrics
        if row["protocol"] == LCO_PROTOCOL
    ]
    lco_collection_summaries = summarize_across_seeds(
        lco_collection_seed_metrics,
        group_fields=(
            "normalization",
            "protocol",
            "collection_id",
            "direction_mode",
            "require_direction",
            "criterion",
            "iou_threshold",
        ),
    )
    confusion_seed = _confusion_seed_rows(artifacts)
    confusion_summary = _confusion_summary_rows(confusion_seed)
    model_runs = _model_run_rows(plan, artifacts)
    timing_seed = _timing_seed_rows(model_runs)
    timing_summary = _timing_summary_rows(timing_seed)
    provenance = {
        "schema_version": 1,
        "experiment_id": plan.experiment_id,
        "plan": {
            "path": plan.plan_path.relative_to(plan.project_root).as_posix(),
            "sha256": file_sha256(plan.plan_path),
        },
        "model_configuration": {
            "path": plan.model_path.relative_to(plan.project_root).as_posix(),
            "sha256": model_hash,
            "receptive_field_frames": model_config.model.receptive_field_frames,
            "context_frames_each_side": model_config.model.future_context_frames,
        },
        "source_code_sha256": current_source_hash,
        "campaign_manifest": {
            "path": plan.output_root.joinpath("campaign_manifest.json")
            .relative_to(plan.project_root)
            .as_posix(),
            "sha256": file_sha256(
                plan.output_root / "campaign_manifest.json"
            ),
            "contract": campaign_manifest,
        },
        "dataset_manifest": {
            "path": "outputs/processed/dataset/manifest.json",
            "sha256": file_sha256(
                plan.project_root
                / "outputs"
                / "processed"
                / "dataset"
                / "manifest.json"
            ),
        },
        "normalizations": [
            {
                "variant": normalization.variant,
                "configuration": (
                    None
                    if normalization.config_path is None
                    else {
                        "path": normalization.config_path.relative_to(
                            plan.project_root
                        ).as_posix(),
                        "sha256": file_sha256(normalization.config_path),
                    }
                ),
                "runtime_contract": normalization_contracts[
                    normalization.variant
                ],
            }
            for normalization in plan.normalizations
        ],
        "splits": [
            {
                "protocol": protocol,
                "path": f"outputs/splits/{protocol}.json",
                "sha256": file_sha256(
                    plan.project_root / "outputs" / "splits" / f"{protocol}.json"
                ),
                "normalization_manifest_path": (
                    f"outputs/normalization/{protocol}.json"
                ),
                "normalization_manifest_sha256": file_sha256(
                    plan.project_root
                    / "outputs"
                    / "normalization"
                    / f"{protocol}.json"
                ),
            }
            for protocol in plan.protocols
        ],
        "input_artifacts": model_runs,
    }
    summary = {
        "schema_version": 1,
        "experiment_id": plan.experiment_id,
        "expected_evaluations": len(tasks),
        "validated_evaluations": len(artifacts),
        "criteria": [
            {"name": name, "iou_threshold": threshold}
            for name, threshold in CRITERIA
        ],
        "metric_aggregation": (
            "Counts are pooled across folds within each seed. Precision, recall, "
            "and F1 are recomputed from pooled counts. Across-seed summaries are "
            "arithmetic means and sample standard deviations of seed-level values."
        ),
        "direction_confusion_condition": (
            "One-to-one presence matches at temporal IoU >= 0.25, with target "
            "direction as rows and predicted direction as columns."
        ),
        "seed_metrics": seed_metrics,
        "matching_sensitivity": aggregate_metrics,
        "lco_collection_seed_metrics": lco_collection_seed_metrics,
        "lco_collection_summaries": lco_collection_summaries,
        "direction_confusion_seed": confusion_seed,
        "direction_confusion": confusion_summary,
        "timing_seed": timing_seed,
        "timing": timing_summary,
        "provenance": provenance,
    }
    destination = plan.output_root / "reports"
    staging = plan.output_root / "reports.staging"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output: {destination}")
    if destination.exists():
        destination.rmdir()
    _prepare_destination(staging)
    _write_csv(staging / "task_metrics.csv", task_metrics)
    _write_csv(staging / "seed_summaries.csv", seed_metrics)
    _write_csv(staging / "matching_sensitivity.csv", aggregate_metrics)
    _write_csv(
        staging / "collection_seed_summaries.csv",
        lco_collection_seed_metrics,
    )
    _write_csv(
        staging / "collection_summaries.csv",
        lco_collection_summaries,
    )
    _write_csv(staging / "direction_confusion_seed.csv", confusion_seed)
    _write_csv(staging / "direction_confusion.csv", confusion_summary)
    _write_csv(staging / "model_runs.csv", model_runs)
    _write_csv(staging / "timing_seed.csv", timing_seed)
    _write_csv(staging / "timing.csv", timing_summary)
    _write_json(staging / "artifact_provenance.json", provenance)
    _write_json(staging / "final_summary.json", summary)
    staging.replace(destination)
    return summary
