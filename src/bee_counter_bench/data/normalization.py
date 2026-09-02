"""Population normalization fitted only on training sessions."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from bee_counter_bench.config import NormalizationConfig
from bee_counter_bench.data.sequence_io import (
    file_sha256,
    load_preprocessed_sequence,
)


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


def _artifacts_by_session(dataset_dir: Path) -> dict[str, Path]:
    manifest = _read_json(dataset_dir / "manifest.json")
    return {
        str(record["session_id"]): dataset_dir / str(record["output"])
        for record in manifest["sequences"]
    }


def normalize_session_signal(
    signal: np.ndarray,
    *,
    training_mean: float,
    training_standard_deviation: float,
    config: NormalizationConfig | None = None,
) -> np.ndarray:
    """Normalize a complete session with the selected training-only setup."""
    values = np.asarray(signal)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array.")
    if not np.isfinite(values).all():
        raise ValueError("signal must contain only finite values.")
    if not math.isfinite(training_mean):
        raise ValueError("Training mean must be finite.")
    if not math.isfinite(training_standard_deviation) or (
        training_standard_deviation <= 0.0
    ):
        raise ValueError("Training standard deviation must be positive and finite.")
    if config is None or config.method == "training_population":
        return np.asarray(
            (values - training_mean) / training_standard_deviation,
            dtype=np.float32,
        )
    normalized, _ = causal_adaptive_normalize(
        values,
        initial_location=training_mean,
        initial_scale=training_standard_deviation,
        config=config,
    )
    return normalized


def causal_adaptive_normalize(
    signal: np.ndarray,
    *,
    initial_location: float,
    initial_scale: float,
    config: NormalizationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize one session using only state from preceding samples."""
    if config.method != "causal_adaptive":
        raise ValueError("Causal normalization requires method='causal_adaptive'.")
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array.")
    if not np.isfinite(values).all():
        raise ValueError("signal must contain only finite values.")
    if not math.isfinite(initial_location) or not math.isfinite(initial_scale):
        raise ValueError("Initial training statistics must be finite.")
    if initial_scale <= 0.0:
        raise ValueError("Initial training scale must be positive.")

    location_time_constant = float(config.location_time_constant_seconds)
    scale_time_constant = float(config.scale_time_constant_seconds)
    adaptation_clip = float(config.adaptation_clip_z)
    minimum_scale_ratio = float(config.minimum_scale_ratio)
    maximum_scale_ratio = float(config.maximum_scale_ratio)
    location_alpha = 1.0 - math.exp(
        -1.0 / (config.sample_rate_hz * location_time_constant)
    )
    scale_alpha = 1.0 - math.exp(
        -1.0 / (config.sample_rate_hz * scale_time_constant)
    )
    minimum_variance = (initial_scale * minimum_scale_ratio) ** 2
    maximum_variance = (initial_scale * maximum_scale_ratio) ** 2
    location = float(initial_location)
    variance = float(initial_scale**2)
    normalized = np.empty(values.size, dtype=np.float32)
    minimum_scale_seen = math.inf
    maximum_scale_seen = 0.0

    for index, value in enumerate(values):
        scale = math.sqrt(variance)
        z_value = (float(value) - location) / scale
        normalized[index] = np.float32(z_value)

        bounded_z = min(adaptation_clip, max(-adaptation_clip, z_value))
        bounded_residual = scale * bounded_z
        location += location_alpha * bounded_residual
        variance += scale_alpha * (bounded_residual**2 - variance)
        variance = min(maximum_variance, max(minimum_variance, variance))
        minimum_scale_seen = min(minimum_scale_seen, scale)
        maximum_scale_seen = max(maximum_scale_seen, scale)

    if not np.isfinite(normalized).all():
        raise ValueError("Causal normalization produced non-finite values.")
    return normalized, {
        "variant": config.variant,
        "method": config.method,
        "parameters": asdict(config),
        "initial_location": float(initial_location),
        "initial_scale": float(initial_scale),
        "final_location": location,
        "final_scale": math.sqrt(variance),
        "minimum_scale_seen": minimum_scale_seen,
        "maximum_scale_seen": maximum_scale_seen,
        "location_alpha": location_alpha,
        "scale_alpha": scale_alpha,
        "output_mean": float(np.mean(normalized, dtype=np.float64)),
        "output_standard_deviation": float(np.std(normalized, dtype=np.float64)),
        "output_minimum": float(np.min(normalized)),
        "output_maximum": float(np.max(normalized)),
    }


def _training_statistics(
    session_ids: Iterable[str],
    artifacts: dict[str, Path],
) -> dict[str, float | int | str]:
    count = 0
    total = 0.0
    total_squared = 0.0
    minimum = math.inf
    maximum = -math.inf
    for session_id in sorted(session_ids):
        if session_id not in artifacts:
            raise ValueError(f"Unknown training session: {session_id}")
        signal = np.asarray(
            load_preprocessed_sequence(artifacts[session_id]).signal,
            dtype=np.float64,
        )
        count += int(signal.size)
        total += float(signal.sum())
        total_squared += float(np.square(signal).sum())
        minimum = min(minimum, float(signal.min()))
        maximum = max(maximum, float(signal.max()))
    if count <= 1:
        raise ValueError("Training split must contain at least two frames.")
    mean = total / count
    variance = max(0.0, total_squared / count - mean * mean)
    standard_deviation = math.sqrt(variance)
    if standard_deviation <= 0.0:
        raise ValueError("Training signal has zero standard deviation.")
    return {
        "count": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "variance": variance,
        "minimum": minimum,
        "maximum": maximum,
        "formula": "(signal - mean) / standard_deviation",
    }


def build_normalization_manifest(
    dataset_dir: str | Path,
    split_manifest_path: str | Path,
) -> dict[str, Any]:
    """Compute one frame-weighted training-only record per fold."""
    dataset_root = Path(dataset_dir).resolve()
    split_path = Path(split_manifest_path).resolve()
    split_manifest = _read_json(split_path)
    artifacts = _artifacts_by_session(dataset_root)
    folds = []
    for fold in split_manifest["folds"]:
        train_session_ids = sorted(fold["train_session_ids"])
        folds.append(
            {
                "fold_id": str(fold["fold_id"]),
                "train_session_ids": train_session_ids,
                "signal": _training_statistics(train_session_ids, artifacts),
            }
        )
    return {
        "schema_version": 1,
        "protocol": str(split_manifest["protocol"]),
        "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
        "split_manifest_sha256": file_sha256(split_path),
        "scope": "training_sessions_only_frame_weighted_population_statistics",
        "folds": folds,
    }


def build_all_normalizations(
    *,
    dataset_dir: str | Path,
    split_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Create normalization manifests for every split protocol."""
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split_path in sorted(Path(split_dir).resolve().glob("*.json")):
        payload = build_normalization_manifest(dataset_dir, split_path)
        output_path = destination / f"{split_path.stem}.json"
        _write_json(output_path, payload)
        paths[split_path.stem] = output_path
    return paths
