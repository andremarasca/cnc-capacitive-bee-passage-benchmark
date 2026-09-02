"""Self-contained training checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from bee_counter_bench.config import (
    AppConfig,
    NormalizationConfig,
    config_to_dict,
    normalization_config_to_dict,
)
from bee_counter_bench.constants import CLASS_NAMES

CHECKPOINT_SCHEMA_VERSION = 4
UPSTREAM_MS_TCNPP_COMMIT = "sj-li/MS-TCN2@f423a9e65f4ccb1cd7322eb9f94946a19e787993"


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    config: AppConfig,
    epoch: int,
    fold_id: str,
    metrics: dict[str, Any],
    dataset_manifest_sha256: str,
    split_manifest_sha256: str,
    normalization_manifest_sha256: str,
    normalization_config: NormalizationConfig | None,
    source_code_sha256: str,
) -> dict[str, Any]:
    """Build a serializable checkpoint dictionary."""
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "upstream_reference": UPSTREAM_MS_TCNPP_COMMIT,
        "epoch": int(epoch),
        "fold_id": fold_id,
        "metrics": metrics,
        "config": config_to_dict(config),
        "classes": {str(key): value for key, value in CLASS_NAMES.items()},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "provenance": {
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "normalization_manifest_sha256": normalization_manifest_sha256,
            "normalization_runtime": normalization_config_to_dict(
                normalization_config
            ),
            "source_code_sha256": source_code_sha256,
            "preprocessing": config_to_dict(config)["preprocessing"],
            "model": config_to_dict(config)["model"],
        },
    }


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Save a checkpoint atomically enough for local training runs."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(
    path: str | Path, *, expected_schema_version: int = CHECKPOINT_SCHEMA_VERSION
) -> dict[str, Any]:
    """Load a checkpoint and validate the stable schema contract."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    schema = payload.get("schema_version")
    if schema != expected_schema_version:
        raise ValueError(
            f"Unsupported checkpoint schema {schema}; expected {expected_schema_version}."
        )
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "config",
        "classes",
        "provenance",
        "fold_id",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    return payload
