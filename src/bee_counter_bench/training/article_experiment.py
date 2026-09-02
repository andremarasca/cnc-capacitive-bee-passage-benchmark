"""Frozen execution plan for the context-safe article experiment."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from bee_counter_bench.config import load_yaml
from bee_counter_bench.data.sequence_io import file_sha256
from bee_counter_bench.provenance import source_code_sha256
from bee_counter_bench.training.runner import ExperimentTask, run_training_task


@dataclass(frozen=True)
class NormalizationVariant:
    variant: str
    config_path: Path | None


@dataclass(frozen=True)
class ArticleExperimentPlan:
    experiment_id: str
    project_root: Path
    plan_path: Path
    model_path: Path
    output_root: Path
    protocols: tuple[str, ...]
    seeds: tuple[int, ...]
    core_frames: int
    expected_task_count: int
    maximum_parallel_gpu_tasks: int
    normalizations: tuple[NormalizationVariant, ...]


@dataclass(frozen=True)
class ArticleExperimentTask:
    normalization_variant: str
    protocol: str
    fold_id: str
    seed: int

    @property
    def task_id(self) -> str:
        return (
            f"{self.normalization_variant}__{self.protocol}__"
            f"seed_{self.seed}__{self.fold_id}"
        )

    @property
    def training_task(self) -> ExperimentTask:
        return ExperimentTask(
            protocol=self.protocol,
            fold_id=self.fold_id,
            seed=self.seed,
        )


def _resolve(project_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


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


def _git_state(project_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "The scientific source tree must be committed and clean before "
            "the article campaign starts."
        )
    return {"commit": commit, "dirty": False}


def load_article_experiment_plan(path: str | Path) -> ArticleExperimentPlan:
    """Load and validate one immutable article experiment plan."""
    plan_path = Path(path).resolve()
    payload = load_yaml(plan_path)
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported article experiment plan schema.")
    project_root = plan_path.parent.parent
    protocols = tuple(str(value) for value in payload.get("protocols", ()))
    seeds = tuple(int(value) for value in payload.get("seeds", ()))
    normalizations = tuple(
        NormalizationVariant(
            variant=str(item["variant"]),
            config_path=(
                None
                if item.get("config") is None
                else _resolve(project_root, item["config"])
            ),
        )
        for item in payload.get("normalizations", ())
    )
    plan = ArticleExperimentPlan(
        experiment_id=str(payload.get("experiment_id", "")),
        project_root=project_root,
        plan_path=plan_path,
        model_path=_resolve(project_root, payload.get("model_config")),
        output_root=_resolve(project_root, payload.get("output_root")),
        protocols=protocols,
        seeds=seeds,
        core_frames=int(payload.get("core_frames", 0)),
        expected_task_count=int(payload.get("expected_task_count", 0)),
        maximum_parallel_gpu_tasks=int(
            payload.get("maximum_parallel_gpu_tasks", 0)
        ),
        normalizations=normalizations,
    )
    _validate_plan(plan)
    return plan


def _validate_plan(plan: ArticleExperimentPlan) -> None:
    if not plan.experiment_id:
        raise ValueError("experiment_id cannot be empty.")
    if not plan.model_path.is_file():
        raise FileNotFoundError(f"Missing model configuration: {plan.model_path}")
    if not plan.protocols or len(set(plan.protocols)) != len(plan.protocols):
        raise ValueError("Protocols must be non-empty and unique.")
    if not plan.seeds or len(set(plan.seeds)) != len(plan.seeds):
        raise ValueError("Seeds must be non-empty and unique.")
    if plan.core_frames <= 0:
        raise ValueError("core_frames must be positive.")
    if plan.expected_task_count <= 0:
        raise ValueError("expected_task_count must be positive.")
    if plan.maximum_parallel_gpu_tasks <= 0:
        raise ValueError("maximum_parallel_gpu_tasks must be positive.")
    variants = [item.variant for item in plan.normalizations]
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("Normalization variants must be non-empty and unique.")
    for normalization in plan.normalizations:
        if (
            normalization.config_path is not None
            and not normalization.config_path.is_file()
        ):
            raise FileNotFoundError(
                f"Missing normalization configuration: {normalization.config_path}"
            )
    expected_parent = (plan.project_root / "outputs" / "experiments").resolve()
    if expected_parent not in plan.output_root.parents:
        raise ValueError("output_root must be inside outputs/experiments.")


def build_article_tasks(
    plan: ArticleExperimentPlan,
    *,
    seeds: Iterable[int] | None = None,
    normalization_variants: Iterable[str] | None = None,
) -> tuple[ArticleExperimentTask, ...]:
    """Build the exact task matrix selected from the frozen plan."""
    selected_seeds = plan.seeds if seeds is None else tuple(int(seed) for seed in seeds)
    selected_variants = (
        tuple(item.variant for item in plan.normalizations)
        if normalization_variants is None
        else tuple(str(value) for value in normalization_variants)
    )
    unknown_seeds = set(selected_seeds).difference(plan.seeds)
    if unknown_seeds:
        raise ValueError(f"Seeds are outside the frozen plan: {sorted(unknown_seeds)}")
    known_variants = {item.variant for item in plan.normalizations}
    unknown_variants = set(selected_variants).difference(known_variants)
    if unknown_variants:
        raise ValueError(
            f"Normalization variants are outside the frozen plan: "
            f"{sorted(unknown_variants)}"
        )
    tasks = []
    for seed in selected_seeds:
        for normalization_variant in selected_variants:
            for protocol in plan.protocols:
                split = _read_json(
                    plan.project_root / "outputs" / "splits" / f"{protocol}.json"
                )
                for fold in split["folds"]:
                    tasks.append(
                        ArticleExperimentTask(
                            normalization_variant=normalization_variant,
                            protocol=protocol,
                            fold_id=str(fold["fold_id"]),
                            seed=seed,
                        )
                    )
    return tuple(tasks)


def build_campaign_manifest(plan: ArticleExperimentPlan) -> dict[str, Any]:
    """Build the deterministic contract shared by every campaign task."""
    root = plan.project_root
    tasks = build_article_tasks(plan)
    if len(tasks) != plan.expected_task_count:
        raise ValueError(
            f"Frozen task count differs: {len(tasks)} != "
            f"{plan.expected_task_count}."
        )
    inputs: dict[str, Any] = {
        "dataset_manifest": {
            "path": "outputs/processed/dataset/manifest.json",
            "sha256": file_sha256(
                root / "outputs" / "processed" / "dataset" / "manifest.json"
            ),
        },
        "splits": {},
        "normalization_manifests": {},
        "normalization_runtime_configs": {},
    }
    for protocol in plan.protocols:
        split_path = root / "outputs" / "splits" / f"{protocol}.json"
        normalization_path = (
            root / "outputs" / "normalization" / f"{protocol}.json"
        )
        inputs["splits"][protocol] = {
            "path": split_path.relative_to(root).as_posix(),
            "sha256": file_sha256(split_path),
        }
        inputs["normalization_manifests"][protocol] = {
            "path": normalization_path.relative_to(root).as_posix(),
            "sha256": file_sha256(normalization_path),
        }
    for normalization in plan.normalizations:
        inputs["normalization_runtime_configs"][normalization.variant] = (
            None
            if normalization.config_path is None
            else {
                "path": normalization.config_path.relative_to(root).as_posix(),
                "sha256": file_sha256(normalization.config_path),
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": plan.experiment_id,
        "git": _git_state(root),
        "source_code_sha256": source_code_sha256(root),
        "plan": {
            "path": plan.plan_path.relative_to(root).as_posix(),
            "sha256": file_sha256(plan.plan_path),
        },
        "model": {
            "path": plan.model_path.relative_to(root).as_posix(),
            "sha256": file_sha256(plan.model_path),
        },
        "inputs": inputs,
        "execution": {
            "core_frames": plan.core_frames,
            "task_count": plan.expected_task_count,
            "maximum_parallel_gpu_tasks": plan.maximum_parallel_gpu_tasks,
            "task_ids": [task.task_id for task in tasks],
        },
    }


def ensure_campaign_manifest(plan: ArticleExperimentPlan) -> dict[str, Any]:
    """Create or validate the immutable campaign manifest."""
    expected = build_campaign_manifest(plan)
    path = plan.output_root / "campaign_manifest.json"
    if path.exists():
        existing = _read_json(path)
        if existing != expected:
            raise RuntimeError(
                "Existing campaign manifest differs from the current source "
                "tree or frozen inputs."
            )
        return existing
    _write_json(path, expected)
    return expected


@contextmanager
def _exclusive_campaign_lock(plan: ArticleExperimentPlan) -> Iterable[None]:
    lock_path = plan.output_root / "campaign.lock"
    plan.output_root.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            f"Another campaign owns the execution lock: {lock_path}"
        ) from error
    _write_json(
        lock_path / "owner.json",
        {
            "schema_version": 1,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    try:
        yield
    finally:
        owner_path = lock_path / "owner.json"
        if owner_path.exists():
            owner_path.unlink()
        lock_path.rmdir()


def _normalization(
    plan: ArticleExperimentPlan,
    variant: str,
) -> NormalizationVariant:
    for normalization in plan.normalizations:
        if normalization.variant == variant:
            return normalization
    raise ValueError(f"Unknown normalization variant: {variant}")


def run_article_task(
    *,
    plan: ArticleExperimentPlan,
    task: ArticleExperimentTask,
    device: str,
    resume: bool,
    progress: bool,
) -> dict[str, Any]:
    """Run one task in its isolated architecture and normalization namespace."""
    ensure_campaign_manifest(plan)
    if task not in build_article_tasks(plan):
        raise ValueError(f"Task is outside the frozen campaign: {task.task_id}")
    normalization = _normalization(plan, task.normalization_variant)
    runs_root = plan.output_root / task.normalization_variant / "runs"
    return run_training_task(
        project_root=plan.project_root,
        task=task.training_task,
        device=device,
        resume=resume,
        progress=progress,
        runs_root=runs_root,
        model_path=plan.model_path,
        normalization_config_path=normalization.config_path,
        core_frames=plan.core_frames,
        experiment_id=plan.experiment_id,
    )


def run_article_suite(
    *,
    plan: ArticleExperimentPlan,
    parallel_tasks: int,
    device: str,
    seeds: Iterable[int] | None = None,
    normalization_variants: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Execute the selected article task matrix with resumable child processes."""
    if parallel_tasks <= 0:
        raise ValueError("parallel_tasks must be positive.")
    if (
        device.startswith("cuda")
        and parallel_tasks > plan.maximum_parallel_gpu_tasks
    ):
        raise ValueError(
            "CUDA parallelism exceeds the frozen campaign limit of "
            f"{plan.maximum_parallel_gpu_tasks}."
        )
    tasks = list(
        build_article_tasks(
            plan,
            seeds=seeds,
            normalization_variants=normalization_variants,
        )
    )
    with _exclusive_campaign_lock(plan):
        ensure_campaign_manifest(plan)
        return _run_article_suite_unlocked(
            plan=plan,
            tasks=tasks,
            parallel_tasks=parallel_tasks,
            device=device,
        )


def _run_article_suite_unlocked(
    *,
    plan: ArticleExperimentPlan,
    tasks: list[ArticleExperimentTask],
    parallel_tasks: int,
    device: str,
) -> dict[str, Any]:
    """Run tasks while the caller owns the campaign lock."""
    logs_dir = plan.output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pending = list(tasks)
    running: dict[
        subprocess.Popen[str], tuple[ArticleExperimentTask, Any]
    ] = {}
    completed: list[str] = []
    failed: list[dict[str, Any]] = []
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
                "train-article-task",
                "--plan",
                str(plan.plan_path),
                "--normalization",
                task.normalization_variant,
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
                cwd=plan.project_root,
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
                failed.append(
                    {"task_id": task.task_id, "return_code": return_code}
                )
        _write_json(
            logs_dir / "suite_status.json",
            {
                "schema_version": 1,
                "experiment_id": plan.experiment_id,
                "total": len(tasks),
                "pending": len(pending),
                "running": [task.task_id for task, _ in running.values()],
                "completed": completed,
                "failed": failed,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )
    if failed:
        raise RuntimeError(f"Article experiment failed tasks: {failed}")
    return {"total": len(tasks), "completed": completed, "failed": failed}
