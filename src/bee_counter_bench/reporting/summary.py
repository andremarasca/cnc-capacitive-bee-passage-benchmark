"""Aggregate completed model and baseline reports without hidden weighting."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import numpy as np

from bee_counter_bench.training.runner import build_suite_tasks


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


def summarize_results(
    *,
    project_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Require the complete task matrix and write JSON plus flat CSV results."""
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite summary directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    expected_tasks = build_suite_tasks(root)
    records = []
    missing = []
    for task in expected_tasks:
        run_dir = (
            root
            / "outputs"
            / "runs"
            / task.protocol
            / f"seed_{task.seed}"
            / task.fold_id
        )
        evaluation_path = run_dir / "evaluation.json"
        manifest_path = run_dir / "run_manifest.json"
        if not evaluation_path.is_file() or not manifest_path.is_file():
            missing.append(task.task_id)
            continue
        records.append(
            {
                "protocol": task.protocol,
                "seed": task.seed,
                "fold_id": task.fold_id,
                "evaluation": _read_json(evaluation_path),
                "run": _read_json(manifest_path),
            }
        )
    if missing:
        raise FileNotFoundError(f"Missing completed experiment tasks: {missing}")

    protocol_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        protocol_groups[record["protocol"]].append(record)
    protocols = {
        protocol: _protocol_summary(items)
        for protocol, items in sorted(protocol_groups.items())
    }
    baseline = _baseline_summary(root)
    report = {
        "schema_version": 1,
        "completed_tasks": len(records),
        "protocols": protocols,
        "baselines": baseline,
    }
    _write_json(destination / "final_summary.json", report)
    _write_csv(destination / "model_runs.csv", [_flat_row(item) for item in records])
    _write_csv(destination / "model_seed_summaries.csv", _seed_rows(protocols))
    return report


def _protocol_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_seed[int(record["seed"])].append(record)
    seed_summaries = {
        str(seed): _micro_summary(items) for seed, items in sorted(by_seed.items())
    }
    primary_f1 = [
        float(summary["event_metrics"]["iou_25"]["f1"])
        for summary in seed_summaries.values()
    ]
    return {
        "runs": len(records),
        "seeds": seed_summaries,
        "across_seed_primary_f1": {
            "mean": mean(primary_f1),
            "standard_deviation": stdev(primary_f1) if len(primary_f1) > 1 else 0.0,
        },
        "experiment_rows": [_flat_row(item) for item in records],
    }


def _micro_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    thresholds = ("iou_10", "iou_25", "iou_50")
    event_summary = {}
    for key in thresholds:
        tp = sum(
            int(item["evaluation"]["event_metrics"]["thresholds"][key]["tp"])
            for item in records
        )
        fp = sum(
            int(item["evaluation"]["event_metrics"]["thresholds"][key]["fp"])
            for item in records
        )
        fn = sum(
            int(item["evaluation"]["event_metrics"]["thresholds"][key]["fn"])
            for item in records
        )
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = (
            0.0
            if precision + recall == 0.0
            else 2 * precision * recall / (precision + recall)
        )
        event_summary[key] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    temporal = {}
    for name in ("start", "end", "center"):
        values = [
            float(value)
            for item in records
            for value in item["evaluation"]["temporal_errors"][
                "absolute_errors_frames"
            ][name]
        ]
        temporal[name] = _temporal_summary(values)
    speed = {}
    for bin_name in (
        "below_1500",
        "1500_to_below_2500",
        "2500_and_above",
    ):
        target_events = sum(
            int(
                item["evaluation"]["speed_stratified_target_recall"][bin_name][
                    "target_events"
                ]
            )
            for item in records
        )
        matched_events = sum(
            int(
                item["evaluation"]["speed_stratified_target_recall"][bin_name][
                    "matched_events"
                ]
            )
            for item in records
        )
        speed[bin_name] = {
            "target_events": target_events,
            "matched_events": matched_events,
            "recall": 0.0 if target_events == 0 else matched_events / target_events,
            "temporal_errors": {
                name: _temporal_summary(
                    [
                        float(value)
                        for item in records
                        for value in item["evaluation"][
                            "speed_stratified_target_recall"
                        ][bin_name]["absolute_errors_frames"][name]
                    ]
                )
                for name in ("start", "end", "center")
            },
        }
    target_events = sum(
        int(item["evaluation"]["event_metrics"]["target_events"]) for item in records
    )
    predicted_events = sum(
        int(item["evaluation"]["event_metrics"]["predicted_events"]) for item in records
    )
    return {
        "folds": len(records),
        "target_events": target_events,
        "predicted_events": predicted_events,
        "count_error": predicted_events - target_events,
        "event_metrics": event_summary,
        "temporal_errors": temporal,
        "speed_stratified_target_recall": speed,
    }


def _temporal_summary(values: list[float]) -> dict[str, float | int]:
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


def _flat_row(record: dict[str, Any]) -> dict[str, Any]:
    evaluation = record["evaluation"]
    events = evaluation["event_metrics"]
    row = {
        "protocol": record["protocol"],
        "seed": record["seed"],
        "fold_id": record["fold_id"],
        "target_events": events["target_events"],
        "predicted_events": events["predicted_events"],
        "count_error": events["count_error"],
        "best_epoch": record["run"]["training"]["best_epoch"],
        "elapsed_seconds": record["run"]["elapsed_seconds"],
    }
    for key in ("iou_10", "iou_25", "iou_50"):
        for metric in ("precision", "recall", "f1"):
            row[f"{metric}_{key}"] = events["thresholds"][key][metric]
    for name in ("start", "end", "center"):
        errors = evaluation["temporal_errors"]["errors"][name]
        row[f"{name}_median_frames"] = errors["median_frames"]
        row[f"{name}_p95_frames"] = errors["p95_frames"]
    return row


def _baseline_summary(root: Path) -> dict[str, Any]:
    baseline_root = root / "outputs" / "baseline"
    reports = {}
    for path in sorted(baseline_root.glob("*.json")):
        report = _read_json(path)
        reports[path.stem] = {
            "macro_summary": report["macro_summary"],
            "aggregate_metrics": report["aggregate_metrics"],
            "selected_parameters": [
                {
                    "fold_id": fold["fold_id"],
                    **fold["parameters"],
                }
                for fold in report["folds"]
            ],
        }
    return reports


def _seed_rows(protocols: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for protocol, protocol_report in protocols.items():
        for seed, summary in protocol_report["seeds"].items():
            row = {
                "protocol": protocol,
                "seed": int(seed),
                "folds": summary["folds"],
                "target_events": summary["target_events"],
                "predicted_events": summary["predicted_events"],
                "count_error": summary["count_error"],
            }
            for key, metrics in summary["event_metrics"].items():
                for metric in ("precision", "recall", "f1"):
                    row[f"{metric}_{key}"] = metrics[metric]
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = list(materialized[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
