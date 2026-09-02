"""Resumable single-task and bounded-parallel experiment execution."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
import yaml

from bee_counter_bench.config import (
    NormalizationConfig,
    config_to_dict,
    load_config,
    load_normalization_config,
    normalization_config_to_dict,
    with_seed,
)
from bee_counter_bench.data.sequence_io import file_sha256
from bee_counter_bench.evaluation.inference import evaluate_checkpoint
from bee_counter_bench.provenance import source_code_sha256
from bee_counter_bench.training.trainer import train_fold


@dataclass(frozen=True)
class ExperimentTask:
    protocol: str
    fold_id: str
    seed: int

    @property
    def task_id(self) -> str:
        return f"{self.protocol}__seed_{self.seed}__{self.fold_id}"


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


def run_training_task(
    *,
    project_root: str | Path,
    task: ExperimentTask,
    device: str = "auto",
    resume: bool = True,
    progress: bool = True,
    runs_root: str | Path | None = None,
    model_path: str | Path | None = None,
    normalization_config_path: str | Path | None = None,
    core_frames: int = 900,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Train and evaluate one protocol/fold/seed task."""
    root = Path(project_root).resolve()
    if core_frames <= 0:
        raise ValueError("core_frames must be positive.")
    resolved_runs_root = (
        root / "outputs" / "runs"
        if runs_root is None
        else Path(runs_root).resolve()
    )
    resolved_model_path = (
        root / "configs" / "model.yaml"
        if model_path is None
        else Path(model_path).resolve()
    )
    normalization_config = (
        None
        if normalization_config_path is None
        else load_normalization_config(normalization_config_path)
    )
    config = with_seed(
        load_config(
            root / "configs" / "experiment.yaml",
            model_path=resolved_model_path,
        ),
        task.seed,
    )
    halo_frames = config.model.future_context_frames
    run_dir = (
        resolved_runs_root / task.protocol / f"seed_{task.seed}" / task.fold_id
    )
    status_path = run_dir / "status.json"
    evaluation_path = run_dir / "evaluation.json"
    split_path = root / "outputs" / "splits" / f"{task.protocol}.json"
    normalization_path = root / "outputs" / "normalization" / f"{task.protocol}.json"
    dataset_dir = root / "outputs" / "processed" / "dataset"
    if not split_path.is_file() or not normalization_path.is_file():
        raise FileNotFoundError(f"Missing derived artifacts for {task.protocol}.")
    split = _read_json(split_path)
    if task.fold_id not in {str(fold["fold_id"]) for fold in split["folds"]}:
        raise ValueError(f"Unknown fold {task.fold_id!r} for {task.protocol}.")
    if evaluation_path.is_file():
        report = _read_json(evaluation_path)
        _validate_completed_evaluation(
            report,
            root=root,
            dataset_dir=dataset_dir,
            split_path=split_path,
            normalization_path=normalization_path,
            normalization_config=normalization_config,
            model_runtime=config_to_dict(config)["model"],
            fold_id=task.fold_id,
        )
        _write_status(status_path, task, "complete", "already_evaluated")
        return report

    started = time.monotonic()
    _write_status(status_path, task, "running", "training")
    try:
        training_result = train_fold(
            config=config,
            dataset_dir=dataset_dir,
            split_manifest=split_path,
            normalization_manifest=normalization_path,
            fold_id=task.fold_id,
            output_dir=run_dir / "training",
            crop_length=core_frames,
            halo_frames=halo_frames,
            device=device,
            num_workers=0,
            resume=resume,
            progress=progress,
            progress_every_batches=100,
            normalization_config=normalization_config,
        )
        _write_status(status_path, task, "running", "evaluation")
        evaluation = evaluate_checkpoint(
            config=config,
            checkpoint_path=run_dir / "training" / "best_checkpoint.pt",
            dataset_dir=dataset_dir,
            split_manifest_path=split_path,
            normalization_manifest_path=normalization_path,
            fold_id=task.fold_id,
            output_path=evaluation_path,
            device_name=device,
            core_frames=core_frames,
            halo_frames=halo_frames,
            normalization_config=normalization_config,
        )
        run_manifest = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "task": {
                "protocol": task.protocol,
                "fold_id": task.fold_id,
                "seed": task.seed,
            },
            "elapsed_seconds": time.monotonic() - started,
            "source_code_sha256": source_code_sha256(root),
            "model_configuration": {
                "path": resolved_model_path.relative_to(root).as_posix(),
                "sha256": file_sha256(resolved_model_path),
            },
            "normalization_runtime": normalization_config_to_dict(
                normalization_config
            ),
            "execution": {
                "core_frames": core_frames,
                "halo_frames": halo_frames,
            },
            "training": training_result,
            "environment": _environment(),
        }
        _write_json(run_dir / "run_manifest.json", run_manifest)
        _write_status(status_path, task, "complete", "evaluated")
        return evaluation
    except Exception as error:
        _write_status(status_path, task, "failed", repr(error))
        raise


def build_suite_tasks(project_root: str | Path) -> tuple[ExperimentTask, ...]:
    """Build the frozen clean-run task matrix from split manifests."""
    root = Path(project_root).resolve()
    experiment = _read_yaml_like_json(root / "configs" / "experiment.yaml")
    main_seeds = tuple(int(value) for value in experiment["training"]["main_seeds"])
    protocol_seeds = {
        "main_session_grouped": main_seeds,
        "leave_one_collection_out": main_seeds,
        "pairwise_collection_transfer": main_seeds,
        "gcode_blocked": main_seeds,
    }
    tasks = []
    for protocol, seeds in protocol_seeds.items():
        split = _read_json(root / "outputs" / "splits" / f"{protocol}.json")
        for seed in seeds:
            for fold in split["folds"]:
                tasks.append(
                    ExperimentTask(
                        protocol=protocol,
                        fold_id=str(fold["fold_id"]),
                        seed=seed,
                    )
                )
    return tuple(tasks)


def run_suite(
    *,
    project_root: str | Path,
    parallel_tasks: int = 2,
    device: str = "cuda",
) -> dict[str, Any]:
    """Launch the frozen task matrix with bounded parallel processes."""
    if parallel_tasks <= 0:
        raise ValueError("parallel_tasks must be positive.")
    root = Path(project_root).resolve()
    tasks = build_suite_tasks(root)
    logs_dir = root / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pending = list(tasks)
    running: dict[subprocess.Popen[str], tuple[ExperimentTask, Any]] = {}
    completed = []
    failed = []
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["PYTHONHASHSEED"] = "0"

    while pending or running:
        while pending and len(running) < parallel_tasks:
            task = pending.pop(0)
            log_path = logs_dir / f"{task.task_id}.log"
            log_stream = log_path.open("a", encoding="utf-8", buffering=1)
            command = [
                sys.executable,
                "-m",
                "bee_counter_bench.cli",
                "train-task",
                "--protocol",
                task.protocol,
                "--fold-id",
                task.fold_id,
                "--seed",
                str(task.seed),
                "--device",
                device,
                "--resume",
            ]
            process = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[process] = (task, log_stream)
        time.sleep(2.0)
        for process, (task, log_stream) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_stream.close()
            del running[process]
            if return_code == 0:
                completed.append(task.task_id)
            else:
                failed.append({"task_id": task.task_id, "return_code": return_code})
        _write_json(
            logs_dir / "suite_status.json",
            {
                "schema_version": 1,
                "total": len(tasks),
                "pending": len(pending),
                "running": [task.task_id for task, _ in running.values()],
                "completed": completed,
                "failed": failed,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )
    if failed:
        raise RuntimeError(f"Experiment suite failed tasks: {failed}")
    return {"total": len(tasks), "completed": completed, "failed": failed}


def _write_status(
    path: Path,
    task: ExperimentTask,
    state: str,
    stage: str,
) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "task_id": task.task_id,
            "state": state,
            "stage": stage,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def _validate_completed_evaluation(
    report: dict[str, Any],
    *,
    root: Path,
    dataset_dir: Path,
    split_path: Path,
    normalization_path: Path,
    normalization_config: NormalizationConfig | None,
    model_runtime: dict[str, Any],
    fold_id: str,
) -> None:
    expected = {
        "fold_id": fold_id,
        "dataset_manifest_sha256": file_sha256(dataset_dir / "manifest.json"),
        "split_manifest_sha256": file_sha256(split_path),
        "normalization_manifest_sha256": file_sha256(normalization_path),
        "normalization_runtime": normalization_config_to_dict(
            normalization_config
        ),
        "model_runtime": model_runtime,
        "source_code_sha256": source_code_sha256(root),
    }
    mismatches = {
        key: (report.get(key), value)
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Completed evaluation provenance mismatch: {mismatches}")


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pyyaml": yaml.__version__,
        "torch": torch.__version__,
        "torch_installation": _distribution_source("torch"),
        "cuda_compiled": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_driver": _nvidia_driver_version(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def _distribution_source(package_name: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(package_name)
    direct_url_text = distribution.read_text("direct_url.json")
    return {
        "version": distribution.version,
        "direct_url": json.loads(direct_url_text) if direct_url_text else None,
    }


def _nvidia_driver_version() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    versions = sorted(
        {line.strip() for line in result.stdout.splitlines() if line.strip()}
    )
    return ",".join(versions) or None


def _read_yaml_like_json(path: Path) -> dict[str, Any]:
    from bee_counter_bench.config import load_yaml

    return load_yaml(path)
