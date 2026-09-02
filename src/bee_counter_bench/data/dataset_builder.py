"""Build the canonical 30 Hz dataset from allowlisted raw sessions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from bee_counter_bench.config import AppConfig, save_resolved_config
from bee_counter_bench.constants import LEFT, MASKED_LABEL, OUT, RIGHT
from bee_counter_bench.data.labels import build_passage_labels
from bee_counter_bench.data.preprocessing import StatefulSessionPreprocessor
from bee_counter_bench.data.schemas import PreprocessedSequence
from bee_counter_bench.data.sequence_io import file_sha256, save_preprocessed_sequence


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
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


def _load_raw_chunk(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    required = {"raw_signal", "position_mm", "sample_rate", "chunk_index"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"Missing raw NPZ fields in {path.name}: {sorted(missing)}"
            )
        signal = np.asarray(archive["raw_signal"])
        position = np.asarray(archive["position_mm"])
        sample_rate = int(archive["sample_rate"].item())
        chunk_index = int(archive["chunk_index"].item())
    return signal, position, sample_rate, chunk_index


def _group_files(raw_manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in raw_manifest["files"]:
        grouped[str(record["session_id"])].append(record)
    for records in grouped.values():
        records.sort(key=lambda item: int(item["chunk_index"]))
    return dict(sorted(grouped.items()))


def _composite_source_hash(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["relative_path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(str(record["sha256"])))
    return digest.hexdigest()


def build_dataset(
    *,
    project_root: str | Path,
    config: AppConfig,
    output_dir: str | Path,
    max_sessions: int | None = None,
    verify_raw_hashes: bool = True,
) -> dict[str, Any]:
    """Create one deterministic session-level dataset manifest and NPZ set."""
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    raw_manifest_path = root / "data" / "metadata" / "raw_manifest.json"
    raw_manifest = _read_json(raw_manifest_path)
    sessions_by_id = {
        str(record["session_id"]): record for record in raw_manifest["sessions"]
    }
    files_by_session = _group_files(raw_manifest)
    session_ids = tuple(sorted(files_by_session))
    if max_sessions is not None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive.")
        session_ids = session_ids[:max_sessions]

    save_resolved_config(config, destination / "resolved_config.yaml")
    sequence_records: list[dict[str, Any]] = []
    label_totals: Counter[str] = Counter()
    event_totals: Counter[str] = Counter()
    input_samples_total = 0
    output_frames_total = 0

    for session_id in session_ids:
        session_metadata = sessions_by_id[session_id]
        source_records = files_by_session[session_id]
        processor = StatefulSessionPreprocessor(
            config.preprocessing,
            config.preprocessing.source_sample_rate_hz,
        )
        signal_parts: list[np.ndarray] = []
        position_parts: list[np.ndarray] = []
        chunk_records: list[dict[str, Any]] = []
        output_cursor = 0
        preprocessing_metadata = None

        for source_record in source_records:
            relative_path = Path(str(source_record["relative_path"]))
            source_path = root / relative_path
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Raw file not found: {relative_path.as_posix()}"
                )
            if verify_raw_hashes:
                actual_hash = file_sha256(source_path)
                if actual_hash != source_record["sha256"]:
                    raise ValueError(
                        f"Raw SHA-256 mismatch: {relative_path.as_posix()}"
                    )
            else:
                actual_hash = str(source_record["sha256"])

            raw_signal, position_mm, sample_rate, chunk_index = _load_raw_chunk(
                source_path
            )
            if sample_rate != config.preprocessing.source_sample_rate_hz:
                raise ValueError(
                    f"Unexpected sample rate in {relative_path.as_posix()}: "
                    f"{sample_rate}"
                )
            if chunk_index != int(source_record["chunk_index"]):
                raise ValueError(f"Chunk index mismatch in {relative_path.as_posix()}.")
            result = processor.process(raw_signal, position_mm)
            preprocessing_metadata = result.metadata
            signal_parts.append(result.signal)
            position_parts.append(result.position_mm)
            chunk_records.append(
                {
                    "chunk_id": str(source_record["chunk_id"]),
                    "chunk_index": chunk_index,
                    "relative_path": relative_path.as_posix(),
                    "sha256": actual_hash,
                    "input_samples": len(raw_signal),
                    "output_start": output_cursor,
                    "output_end_exclusive": output_cursor + len(result.signal),
                }
            )
            output_cursor += len(result.signal)
            input_samples_total += len(raw_signal)

        if preprocessing_metadata is None:
            raise ValueError(f"Session has no raw chunks: {session_id}")
        signal = np.concatenate(signal_parts).astype(np.float32, copy=False)
        position = np.concatenate(position_parts).astype(np.float32, copy=False)
        labeled = build_passage_labels(
            position,
            zone_low_mm=config.geometry.zone_low_mm,
            zone_high_mm=config.geometry.zone_high_mm,
            sample_rate_hz=config.preprocessing.output_sample_rate_hz,
        )
        masked_frames = int(np.count_nonzero(labeled.labels == MASKED_LABEL))
        if masked_frames:
            raise ValueError(
                f"Session {session_id} contains {masked_frames} frames from "
                "incomplete, reversing, or boundary-truncated ROI visits."
            )

        expected_events = int(session_metadata["expected_events"])
        if len(labeled.events) != expected_events:
            raise ValueError(
                f"Session {session_id} produced {len(labeled.events)} events, "
                f"expected {expected_events}."
            )

        source_hash = _composite_source_hash(source_records)
        relative_output = Path(session_id) / f"{session_id}__sequence.npz"
        sequence = PreprocessedSequence(
            signal=signal,
            position_mm=position,
            labels=labeled.labels,
            valid_mask=np.ones(len(signal), dtype=np.bool_),
            sample_rate_hz=config.preprocessing.output_sample_rate_hz,
            session_id=session_id,
            chunk_id="session_full",
            source_path=f"data/raw/{session_metadata['collection_id']}/{session_id}",
            source_sha256=source_hash,
            preprocessing=preprocessing_metadata,
            events=labeled.events,
        )
        save_preprocessed_sequence(destination / relative_output, sequence)
        output_sha256 = file_sha256(destination / relative_output)

        label_counts = {
            "OUT": int(np.count_nonzero(labeled.labels == OUT)),
            "LEFT": int(np.count_nonzero(labeled.labels == LEFT)),
            "RIGHT": int(np.count_nonzero(labeled.labels == RIGHT)),
        }
        direction_counts = Counter(event.direction.name for event in labeled.events)
        label_totals.update(label_counts)
        event_totals.update(direction_counts)
        output_frames_total += len(signal)
        sequence_records.append(
            {
                "session_id": session_id,
                "collection_id": str(session_metadata["collection_id"]),
                "session_date": str(session_metadata["session_date"]),
                "gcode_id": str(session_metadata["gcode_id"]),
                "motion_family": str(session_metadata["motion_family"]),
                "source_composite_sha256": source_hash,
                "source_chunks": chunk_records,
                "output": relative_output.as_posix(),
                "output_sha256": output_sha256,
                "input_samples": sum(
                    int(record["input_samples"]) for record in chunk_records
                ),
                "output_frames": len(signal),
                "labels": label_counts,
                "events": dict(sorted(direction_counts.items())),
            }
        )

    manifest = {
        "schema_version": 1,
        "raw_manifest_sha256": file_sha256(raw_manifest_path),
        "sequences": sequence_records,
        "summary": {
            "sessions": len(sequence_records),
            "source_chunks": sum(
                len(record["source_chunks"]) for record in sequence_records
            ),
            "input_samples": input_samples_total,
            "output_frames": output_frames_total,
            "labels": dict(sorted(label_totals.items())),
            "events": dict(sorted(event_totals.items())),
        },
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest
