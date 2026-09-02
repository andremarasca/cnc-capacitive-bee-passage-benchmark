"""Pre-training audit of provenance, dataset, splits and normalization."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from bee_counter_bench.config import AppConfig, load_yaml
from bee_counter_bench.constants import LEFT, MASKED_LABEL, OUT, RIGHT
from bee_counter_bench.data.normalization import build_normalization_manifest
from bee_counter_bench.data.sequence_io import (
    file_sha256,
    load_preprocessed_sequence,
)
from bee_counter_bench.provenance import source_code_sha256


def _read_json(path: Path, *, allow_bom: bool = False) -> dict[str, Any]:
    encoding = "utf-8-sig" if allow_bom else "utf-8"
    with path.open("r", encoding=encoding) as stream:
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


def audit_pretraining_artifacts(
    *,
    project_root: str | Path,
    config: AppConfig,
    dataset_dir: str | Path,
    split_dir: str | Path,
    normalization_dir: str | Path,
    output_path: str | Path,
    verify_raw_hashes: bool = True,
) -> dict[str, Any]:
    """Fail on any mismatch before model fitting and write a compact report."""
    root = Path(project_root).resolve()
    dataset_root = Path(dataset_dir).resolve()
    split_root = Path(split_dir).resolve()
    normalization_root = Path(normalization_dir).resolve()
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite audit report: {destination}")

    experiment = load_yaml(root / "configs" / "experiment.yaml")
    raw_manifest_path = root / "data" / "metadata" / "raw_manifest.json"
    raw_manifest = _read_json(raw_manifest_path, allow_bom=True)
    dataset_manifest_path = dataset_root / "manifest.json"
    dataset_manifest = _read_json(dataset_manifest_path)
    raw_report = _audit_raw_files(root, raw_manifest, verify_raw_hashes)
    dataset_report, units = _audit_dataset(
        config,
        experiment,
        raw_manifest,
        dataset_root,
        dataset_manifest,
    )
    gcode_report = _audit_gcode(root, raw_manifest)
    split_reports = {}
    for split_path in sorted(split_root.glob("*.json")):
        normalization_path = normalization_root / split_path.name
        split_reports[split_path.stem] = _audit_split_and_normalization(
            split_path,
            normalization_path,
            dataset_root,
            units,
        )
    expected_protocols = {
        "main_session_grouped",
        "leave_one_collection_out",
        "pairwise_collection_transfer",
        "gcode_blocked",
    }
    if set(split_reports) != expected_protocols:
        raise ValueError(
            f"Expected split protocols {sorted(expected_protocols)}, "
            f"found {sorted(split_reports)}."
        )

    report = {
        "schema_version": 1,
        "status": "passed",
        "raw_manifest_sha256": file_sha256(raw_manifest_path),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "source_code_sha256": source_code_sha256(root),
        "raw": raw_report,
        "dataset": dataset_report,
        "gcode": gcode_report,
        "protocols": split_reports,
    }
    _write_json(destination, report)
    return report


def _audit_raw_files(
    root: Path,
    raw_manifest: dict[str, Any],
    verify_hashes: bool,
) -> dict[str, Any]:
    files = raw_manifest["files"]
    paths = [str(record["relative_path"]) for record in files]
    if len(paths) != len(set(paths)):
        raise ValueError("Raw manifest contains duplicate paths.")
    total_bytes = 0
    total_samples = 0
    for record in files:
        path = root / str(record["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw file: {record['relative_path']}")
        size = path.stat().st_size
        if size != int(record["size_bytes"]):
            raise ValueError(f"Raw byte count mismatch: {record['relative_path']}")
        if verify_hashes and file_sha256(path) != str(record["sha256"]):
            raise ValueError(f"Raw SHA-256 mismatch: {record['relative_path']}")
        total_bytes += size
        total_samples += int(record["input_samples"])
    return {
        "files": len(files),
        "bytes": total_bytes,
        "samples": total_samples,
        "sha256_verified": verify_hashes,
    }


def _audit_dataset(
    config: AppConfig,
    experiment: dict[str, Any],
    raw_manifest: dict[str, Any],
    dataset_root: Path,
    dataset_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    expected = experiment["dataset"]
    summary = dataset_manifest["summary"]
    observed_events = sum(int(value) for value in summary["events"].values())
    checks = {
        "sessions": (int(summary["sessions"]), int(expected["expected_sessions"])),
        "source_chunks": (
            int(summary["source_chunks"]),
            int(expected["expected_chunks"]),
        ),
        "events": (observed_events, int(expected["expected_valid_events"])),
    }
    mismatches = {
        name: values for name, values in checks.items() if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(f"Dataset count mismatch: {mismatches}")
    if dataset_manifest["raw_manifest_sha256"] != file_sha256(
        dataset_root.parents[2] / "data" / "metadata" / "raw_manifest.json"
    ):
        raise ValueError("Dataset references a different raw manifest.")

    raw_sessions = {
        str(record["session_id"]): record for record in raw_manifest["sessions"]
    }
    units = {}
    label_totals: Counter[str] = Counter()
    event_totals: Counter[str] = Counter()
    for record in dataset_manifest["sequences"]:
        session_id = str(record["session_id"])
        if session_id in units:
            raise ValueError(f"Duplicate processed session: {session_id}")
        if session_id not in raw_sessions:
            raise ValueError(f"Unknown processed session: {session_id}")
        output_path = dataset_root / str(record["output"])
        if file_sha256(output_path) != str(record["output_sha256"]):
            raise ValueError(f"Processed SHA-256 mismatch: {record['output']}")
        sequence = load_preprocessed_sequence(output_path)
        _audit_sequence(config, sequence, record)
        label_counts = {
            "OUT": int(np.count_nonzero(sequence.labels == OUT)),
            "LEFT": int(np.count_nonzero(sequence.labels == LEFT)),
            "RIGHT": int(np.count_nonzero(sequence.labels == RIGHT)),
        }
        event_counts = Counter(event.direction.name for event in sequence.events)
        if label_counts != record["labels"] or dict(event_counts) != record["events"]:
            raise ValueError(f"Sequence summary mismatch: {session_id}")
        label_totals.update(label_counts)
        event_totals.update(event_counts)
        speed_counts = Counter(
            _speed_bin(event.mean_speed_mm_min) for event in sequence.events
        )
        units[session_id] = {
            "session_id": session_id,
            "collection_id": str(record["collection_id"]),
            "gcode_id": str(record["gcode_id"]),
            "motion_family": str(record["motion_family"]),
            "frames": len(sequence.signal),
            "events": len(sequence.events),
            "speed_bins": dict(speed_counts),
        }
    if set(units) != set(raw_sessions):
        raise ValueError("Processed and raw session sets differ.")
    if dict(sorted(label_totals.items())) != summary["labels"]:
        raise ValueError("Dataset label totals do not match sequence artifacts.")
    if dict(sorted(event_totals.items())) != summary["events"]:
        raise ValueError("Dataset event totals do not match sequence artifacts.")
    return {
        "sessions": len(units),
        "source_chunks": int(summary["source_chunks"]),
        "input_samples": int(summary["input_samples"]),
        "output_frames": int(summary["output_frames"]),
        "labels": summary["labels"],
        "events": summary["events"],
        "masked_frames": 0,
    }, units


def _audit_sequence(
    config: AppConfig,
    sequence: Any,
    record: dict[str, Any],
) -> None:
    metadata = sequence.preprocessing
    if sequence.session_id != str(record["session_id"]):
        raise ValueError("Sequence session identifier mismatch.")
    if len(sequence.signal) != int(record["output_frames"]):
        raise ValueError(f"Frame count mismatch: {sequence.session_id}")
    if np.any(sequence.labels == MASKED_LABEL):
        raise ValueError(
            f"Masked labels remain in final dataset: {sequence.session_id}"
        )
    expected_metadata = {
        "source_sample_rate_hz": config.preprocessing.source_sample_rate_hz,
        "output_sample_rate_hz": config.preprocessing.output_sample_rate_hz,
        "filter_family": "bessel",
        "filter_mode": "bandpass",
        "filter_prototype_order": config.preprocessing.bessel.order,
        "filter_transformed_order": 2 * config.preprocessing.bessel.order,
        "low_cutoff_hz": config.preprocessing.bessel.low_cutoff_hz,
        "high_cutoff_hz": config.preprocessing.bessel.high_cutoff_hz,
        "filter_norm": config.preprocessing.bessel.norm,
        "filter_initialization": config.preprocessing.initialization,
        "state_scope": "session",
    }
    mismatches = {
        key: (getattr(metadata, key), value)
        for key, value in expected_metadata.items()
        if getattr(metadata, key) != value
    }
    if mismatches:
        raise ValueError(f"Preprocessing metadata mismatch: {mismatches}")
    chunks = record["source_chunks"]
    cursor = 0
    for chunk in chunks:
        if int(chunk["output_start"]) != cursor:
            raise ValueError(f"Non-contiguous chunk map: {sequence.session_id}")
        cursor = int(chunk["output_end_exclusive"])
    if cursor != len(sequence.signal):
        raise ValueError(f"Chunk map does not cover session: {sequence.session_id}")


def _audit_gcode(root: Path, raw_manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_path = root / "data" / "metadata" / "gcode_manifest.json"
    manifest = _read_json(manifest_path, allow_bom=True)
    records = manifest.get("files", manifest.get("gcode_files"))
    if not isinstance(records, list):
        raise ValueError("G-code manifest has no file list.")
    gcode_ids = set()
    for record in records:
        relative = str(record.get("relative_path", record.get("path")))
        path = root / relative
        if not path.is_file() or file_sha256(path) != str(record["sha256"]):
            raise ValueError(f"G-code artifact mismatch: {relative}")
        gcode_ids.add(path.stem)
    session_gcodes = {str(record["gcode_id"]) for record in raw_manifest["sessions"]}
    if gcode_ids != session_gcodes:
        raise ValueError("G-code file set differs from session metadata.")
    for session in raw_manifest["sessions"]:
        session_id = str(session["session_id"])
        gcode_id = str(session["gcode_id"])
        if not session_id.startswith(f"{gcode_id}__"):
            raise ValueError(f"Session name does not identify its G-code: {session_id}")
    return {
        "manifest_sha256": file_sha256(manifest_path),
        "files": len(gcode_ids),
        "session_prefix_mapping": "passed",
    }


def _audit_split_and_normalization(
    split_path: Path,
    normalization_path: Path,
    dataset_root: Path,
    units: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    split = _read_json(split_path)
    normalization = _read_json(normalization_path)
    if split["dataset_manifest_sha256"] != file_sha256(dataset_root / "manifest.json"):
        raise ValueError(f"Split dataset hash mismatch: {split_path.name}")
    recomputed = build_normalization_manifest(dataset_root, split_path)
    if normalization != recomputed:
        raise ValueError(f"Normalization mismatch: {normalization_path.name}")
    all_sessions = set(units)
    test_counts: Counter[str] = Counter()
    fold_reports = []
    for fold in split["folds"]:
        sets = {
            name: set(fold[f"{name}_session_ids"])
            for name in ("train", "validation", "test", "excluded")
        }
        names = tuple(sets)
        overlaps = {
            f"{left}_{right}": sorted(sets[left] & sets[right])
            for index, left in enumerate(names)
            for right in names[index + 1 :]
            if sets[left] & sets[right]
        }
        if overlaps:
            raise ValueError(f"Session leakage in {fold['fold_id']}: {overlaps}")
        if set().union(*sets.values()) != all_sessions:
            raise ValueError(
                f"Fold does not account for every session: {fold['fold_id']}"
            )
        test_counts.update(sets["test"])
        if "test_gcode_ids" in fold:
            train_gcodes = {units[item]["gcode_id"] for item in sets["train"]}
            validation_gcodes = {units[item]["gcode_id"] for item in sets["validation"]}
            test_gcodes = {units[item]["gcode_id"] for item in sets["test"]}
            if (
                train_gcodes & validation_gcodes
                or train_gcodes & test_gcodes
                or validation_gcodes & test_gcodes
            ):
                raise ValueError(f"G-code leakage in {fold['fold_id']}.")
        fold_reports.append(
            {
                "fold_id": str(fold["fold_id"]),
                **{
                    name: _composition(sets[name], units)
                    for name in ("train", "validation", "test", "excluded")
                },
            }
        )
    expected_test_repetitions = (
        2 if split["protocol"] == "pairwise_collection_transfer" else 1
    )
    if set(test_counts) != all_sessions or any(
        count != expected_test_repetitions for count in test_counts.values()
    ):
        raise ValueError(f"Unexpected test coverage in {split_path.name}.")
    return {
        "protocol": split["protocol"],
        "split_manifest_sha256": file_sha256(split_path),
        "normalization_manifest_sha256": file_sha256(normalization_path),
        "folds": fold_reports,
        "test_repetitions_per_session": expected_test_repetitions,
    }


def _composition(
    session_ids: Iterable[str],
    units: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected = [units[session_id] for session_id in sorted(session_ids)]
    collections: Counter[str] = Counter()
    families: Counter[str] = Counter()
    speed_bins: Counter[str] = Counter()
    for unit in selected:
        collections[unit["collection_id"]] += unit["events"]
        families[unit["motion_family"]] += unit["events"]
        speed_bins.update(unit["speed_bins"])
    return {
        "sessions": len(selected),
        "frames": sum(unit["frames"] for unit in selected),
        "events": sum(unit["events"] for unit in selected),
        "events_by_collection": dict(sorted(collections.items())),
        "events_by_motion_family": dict(sorted(families.items())),
        "events_by_speed_bin": dict(sorted(speed_bins.items())),
    }


def _speed_bin(speed_mm_min: float) -> str:
    if speed_mm_min < 1500.0:
        return "below_1500"
    if speed_mm_min < 2500.0:
        return "1500_to_below_2500"
    return "2500_and_above"
