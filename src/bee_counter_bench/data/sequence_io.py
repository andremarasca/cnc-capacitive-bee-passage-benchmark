"""Safe, versioned persistence for preprocessed training sequences."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import (
    Direction,
    PassageEvent,
    PreprocessedSequence,
    PreprocessingMetadata,
)

SCHEMA_VERSION = 1


def file_sha256(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest without loading the whole file in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _metadata_json(metadata: PreprocessingMetadata) -> str:
    return json.dumps(asdict(metadata), sort_keys=True, separators=(",", ":"))


def save_preprocessed_sequence(
    path: str | Path,
    sequence: PreprocessedSequence,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save a sequence as a non-pickle NPZ artifact."""
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("Preprocessed sequence path must end in .npz.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    event_count = len(sequence.events)
    event_start = np.fromiter(
        (event.start_idx for event in sequence.events),
        dtype=np.int64,
        count=event_count,
    )
    event_end = np.fromiter(
        (event.end_idx for event in sequence.events), dtype=np.int64, count=event_count
    )
    event_direction = np.fromiter(
        (int(event.direction) for event in sequence.events),
        dtype=np.int8,
        count=event_count,
    )
    event_speeds = np.asarray(
        [
            (event.mean_speed_mm_min, event.min_speed_mm_min, event.max_speed_mm_min)
            for event in sequence.events
        ],
        dtype=np.float64,
    ).reshape(event_count, 3)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.stem}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary_path,
            schema_version=np.int32(SCHEMA_VERSION),
            signal=sequence.signal,
            position_mm=sequence.position_mm,
            labels=sequence.labels,
            valid_mask=sequence.valid_mask,
            sample_rate_hz=np.int32(sequence.sample_rate_hz),
            session_id=np.str_(sequence.session_id),
            chunk_id=np.str_(sequence.chunk_id),
            source_path=np.str_(sequence.source_path),
            source_sha256=np.str_(sequence.source_sha256.lower()),
            preprocessing_json=np.str_(_metadata_json(sequence.preprocessing)),
            event_start_idx=event_start,
            event_end_idx=event_end,
            event_direction=event_direction,
            event_speeds_mm_min=event_speeds,
        )
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def _scalar(archive: Any, key: str) -> Any:
    value = archive[key]
    if value.shape != ():
        raise ValueError(f"Expected scalar NPZ field {key!r}.")
    return value.item()


def load_preprocessed_sequence(path: str | Path) -> PreprocessedSequence:
    """Load and validate a sequence without allowing pickled objects."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        version = int(_scalar(archive, "schema_version"))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported preprocessed sequence schema {version}; "
                f"expected {SCHEMA_VERSION}."
            )
        metadata_raw = json.loads(str(_scalar(archive, "preprocessing_json")))
        metadata = PreprocessingMetadata(**metadata_raw)

        event_start = np.asarray(archive["event_start_idx"], dtype=np.int64)
        event_end = np.asarray(archive["event_end_idx"], dtype=np.int64)
        event_direction = np.asarray(archive["event_direction"], dtype=np.int8)
        event_speeds = np.asarray(archive["event_speeds_mm_min"], dtype=np.float64)
        count = len(event_start)
        if event_end.shape != (count,) or event_direction.shape != (count,):
            raise ValueError("Event arrays have inconsistent lengths.")
        if event_speeds.shape != (count, 3):
            raise ValueError("Event speed array must have shape [events, 3].")
        events = tuple(
            PassageEvent(
                start_idx=int(event_start[index]),
                end_idx=int(event_end[index]),
                direction=Direction(int(event_direction[index])),
                mean_speed_mm_min=float(event_speeds[index, 0]),
                min_speed_mm_min=float(event_speeds[index, 1]),
                max_speed_mm_min=float(event_speeds[index, 2]),
            )
            for index in range(count)
        )

        return PreprocessedSequence(
            signal=np.asarray(archive["signal"], dtype=np.float32),
            position_mm=np.asarray(archive["position_mm"], dtype=np.float32),
            labels=np.asarray(archive["labels"], dtype=np.int64),
            valid_mask=np.asarray(archive["valid_mask"], dtype=np.bool_),
            sample_rate_hz=int(_scalar(archive, "sample_rate_hz")),
            session_id=str(_scalar(archive, "session_id")),
            chunk_id=str(_scalar(archive, "chunk_id")),
            source_path=str(_scalar(archive, "source_path")),
            source_sha256=str(_scalar(archive, "source_sha256")),
            preprocessing=metadata,
            events=events,
        )
