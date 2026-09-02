"""Session-level training samples with event-safe cores and temporal halos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from bee_counter_bench.config import NormalizationConfig
from bee_counter_bench.constants import MASKED_LABEL
from bee_counter_bench.data.normalization import normalize_session_signal
from bee_counter_bench.data.schemas import PassageEvent, PreprocessedSequence
from bee_counter_bench.data.sequence_io import load_preprocessed_sequence

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class SequenceSample:
    artifact: str
    session_id: str
    input_start: int
    input_end: int
    core_start: int
    core_end: int


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _session_artifacts(dataset_dir: Path) -> dict[str, str]:
    manifest = _read_json(dataset_dir / "manifest.json")
    return {
        str(record["session_id"]): str(record["output"])
        for record in manifest["sequences"]
    }


def safe_core_ranges(
    length: int,
    events: tuple[PassageEvent, ...],
    core_length: int | None,
) -> tuple[tuple[int, int], ...]:
    """Return contiguous [start, end) cores whose boundaries preserve events."""
    if length <= 0:
        raise ValueError("Sequence length must be positive.")
    if core_length is None:
        return ((0, length),)
    if core_length <= 0:
        raise ValueError("core_length must be positive.")
    if any(event.duration_frames > core_length for event in events):
        raise ValueError("One event is longer than the configured safe core.")
    valid_boundary = np.ones(length + 1, dtype=np.bool_)
    for event in events:
        valid_boundary[event.start_idx + 1 : event.end_idx + 1] = False
    minimum_cores = int(np.ceil(length / core_length))
    maximum_cores = minimum_cores + len(events) + 1
    for core_count in range(minimum_cores, maximum_cores + 1):
        boundaries = _balanced_boundaries(
            length,
            core_length,
            core_count,
            valid_boundary,
        )
        if boundaries is not None:
            return tuple(
                (start, end)
                for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
            )
    raise ValueError("Could not construct event-safe cores within the length limit.")


def _balanced_boundaries(
    length: int,
    core_length: int,
    core_count: int,
    valid_boundary: np.ndarray,
) -> tuple[int, ...] | None:
    boundaries = [0]
    for boundary_index in range(1, core_count):
        previous = boundaries[-1]
        remaining_cores = core_count - boundary_index
        lower = max(
            previous + 1,
            length - remaining_cores * core_length,
        )
        upper = min(
            previous + core_length,
            length - remaining_cores,
        )
        if lower > upper:
            return None
        candidates = np.flatnonzero(valid_boundary[lower : upper + 1]) + lower
        if not len(candidates):
            return None
        target = boundary_index * length / core_count
        chosen = min(
            candidates.tolist(), key=lambda value: (abs(value - target), value)
        )
        boundaries.append(int(chosen))
    boundaries.append(length)
    if any(
        end - start > core_length
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        return None
    return tuple(boundaries)


class PassageSequenceDataset(torch.utils.data.Dataset[dict[str, Any]]):
    """Load normalized session cores with context that is excluded from loss."""

    def __init__(
        self,
        *,
        dataset_dir: str | Path,
        split_manifest: str | Path,
        normalization_manifest: str | Path,
        fold_id: str,
        split: SplitName,
        crop_length: int | None = 900,
        halo_frames: int = 142,
        normalization_config: NormalizationConfig | None = None,
    ) -> None:
        if halo_frames < 0:
            raise ValueError("halo_frames cannot be negative.")
        self.dataset_dir = Path(dataset_dir).resolve()
        split_payload = _read_json(Path(split_manifest).resolve())
        normalization_payload = _read_json(Path(normalization_manifest).resolve())
        fold = _find_fold(split_payload, fold_id)
        normalization = _find_fold(normalization_payload, fold_id)["signal"]
        self.mean = float(normalization["mean"])
        self.standard_deviation = float(normalization["standard_deviation"])
        self.normalization_config = normalization_config
        if self.standard_deviation <= 0.0:
            raise ValueError("Normalization standard deviation must be positive.")

        artifacts = _session_artifacts(self.dataset_dir)
        session_key = f"{split}_session_ids"
        if session_key not in fold:
            raise ValueError(f"Split fold has no {session_key!r}.")
        self._sequences: dict[str, PreprocessedSequence] = {}
        self._normalized_signals: dict[str, np.ndarray] = {}
        self.samples: list[SequenceSample] = []
        for session_id in fold[session_key]:
            if session_id not in artifacts:
                raise ValueError(f"Unknown split session: {session_id}")
            artifact = artifacts[session_id]
            sequence = load_preprocessed_sequence(self.dataset_dir / artifact)
            self._sequences[artifact] = sequence
            if (
                self.normalization_config is not None
                and self.normalization_config.method == "causal_adaptive"
            ):
                self._normalized_signals[artifact] = normalize_session_signal(
                    sequence.signal,
                    training_mean=self.mean,
                    training_standard_deviation=self.standard_deviation,
                    config=self.normalization_config,
                )
            for core_start, core_end in safe_core_ranges(
                len(sequence.signal), sequence.events, crop_length
            ):
                self.samples.append(
                    SequenceSample(
                        artifact=artifact,
                        session_id=str(session_id),
                        input_start=max(0, core_start - halo_frames),
                        input_end=min(len(sequence.signal), core_end + halo_frames),
                        core_start=core_start,
                        core_end=core_end,
                    )
                )
        if not self.samples:
            raise ValueError("Dataset split produced no samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        sequence = self._sequences[sample.artifact]
        if sample.artifact in self._normalized_signals:
            signal = self._normalized_signals[sample.artifact][
                sample.input_start : sample.input_end
            ]
        else:
            signal = sequence.signal[sample.input_start : sample.input_end].astype(
                np.float32
            )
            signal = (signal - self.mean) / self.standard_deviation
        labels = sequence.labels[sample.input_start : sample.input_end]
        input_mask = np.ones(len(signal), dtype=np.bool_)
        loss_mask = np.zeros(len(signal), dtype=np.bool_)
        core_start = sample.core_start - sample.input_start
        core_end = sample.core_end - sample.input_start
        loss_mask[core_start:core_end] = sequence.valid_mask[
            sample.core_start : sample.core_end
        ]
        return {
            "features": torch.from_numpy(
                signal[None, :].astype(np.float32, copy=False)
            ),
            "labels": torch.from_numpy(labels.astype(np.int64, copy=False)),
            "input_mask": torch.from_numpy(input_mask),
            "loss_mask": torch.from_numpy(loss_mask),
            "session_id": sample.session_id,
            "artifact": sample.artifact,
            "input_start": sample.input_start,
            "input_end": sample.input_end,
            "core_start": sample.core_start,
            "core_end": sample.core_end,
        }


def _find_fold(manifest: dict[str, Any], fold_id: str) -> dict[str, Any]:
    for fold in manifest["folds"]:
        if fold["fold_id"] == fold_id:
            return fold
    raise ValueError(f"Unknown fold_id: {fold_id}")


def collate_padded(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Right-pad inputs while keeping model and loss masks independent."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    lengths = torch.tensor(
        [item["features"].shape[-1] for item in batch], dtype=torch.long
    )
    maximum_length = int(lengths.max().item())
    channels = int(batch[0]["features"].shape[0])
    features = torch.zeros((len(batch), channels, maximum_length), dtype=torch.float32)
    labels = torch.full((len(batch), maximum_length), MASKED_LABEL, dtype=torch.long)
    input_mask = torch.zeros((len(batch), maximum_length), dtype=torch.bool)
    loss_mask = torch.zeros((len(batch), maximum_length), dtype=torch.bool)
    for index, item in enumerate(batch):
        length = int(lengths[index].item())
        features[index, :, :length] = _as_tensor(item["features"], torch.float32)
        labels[index, :length] = _as_tensor(item["labels"], torch.long)
        input_mask[index, :length] = _as_tensor(item["input_mask"], torch.bool)
        loss_mask[index, :length] = _as_tensor(item["loss_mask"], torch.bool)
    return {
        "features": features,
        "labels": labels,
        "input_mask": input_mask,
        "loss_mask": loss_mask,
        "lengths": lengths,
        "session_id": [item["session_id"] for item in batch],
        "artifact": [item["artifact"] for item in batch],
        "input_start": torch.tensor(
            [item["input_start"] for item in batch], dtype=torch.long
        ),
        "input_end": torch.tensor(
            [item["input_end"] for item in batch], dtype=torch.long
        ),
        "core_start": torch.tensor(
            [item["core_start"] for item in batch], dtype=torch.long
        ),
        "core_end": torch.tensor(
            [item["core_end"] for item in batch], dtype=torch.long
        ),
    }


def _as_tensor(value: Tensor, dtype: torch.dtype) -> Tensor:
    return value if value.dtype == dtype else value.to(dtype=dtype)
