"""Unified command-line interface for the reproducibility workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from bee_counter_bench.baseline.threshold import run_threshold_baseline
from bee_counter_bench.config import load_config, load_yaml
from bee_counter_bench.data.dataset_builder import build_dataset
from bee_counter_bench.data.normalization import build_all_normalizations
from bee_counter_bench.data.splits import build_split_manifests
from bee_counter_bench.evaluation.audit import audit_pretraining_artifacts
from bee_counter_bench.evaluation.context_safety import (
    write_context_safety_audit,
)
from bee_counter_bench.training.dataset import PassageSequenceDataset, collate_padded
from bee_counter_bench.training.losses import MSTCNPPMultiStageLoss
from bee_counter_bench.training.runner import (
    ExperimentTask,
    run_suite,
    run_training_task,
)
from bee_counter_bench.training.article_experiment import (
    ArticleExperimentTask,
    load_article_experiment_plan,
    run_article_suite,
    run_article_task,
)
from bee_counter_bench.training.trainer import build_model
from bee_counter_bench.reporting.summary import summarize_results
from bee_counter_bench.reporting.article_ablation import (
    write_article_ablation_reports,
)
from bee_counter_bench.reporting.article_experiment import (
    write_article_experiment_reports,
)
from bee_counter_bench.reporting.temporal_examples import write_temporal_examples
from bee_counter_bench.reporting.filter_response import write_filter_response
from bee_counter_bench.reporting.dataset_figures import write_dataset_figures


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bee-counter-bench",
        description="Reproduce the controlled bee-counter bench experiments.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current directory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gcode = subparsers.add_parser("generate-gcode")
    gcode.add_argument("--output-dir", type=Path, default=Path("outputs/gcode"))
    verify_gcode = subparsers.add_parser("verify-gcode")
    verify_gcode.add_argument(
        "--generated-dir", type=Path, default=Path("outputs/gcode")
    )

    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument(
        "--output-dir", type=Path, default=Path("outputs/processed/dataset")
    )
    preprocess.add_argument("--max-sessions", type=int)
    preprocess.add_argument("--skip-raw-hash-verification", action="store_true")

    subparsers.add_parser("make-splits")
    subparsers.add_parser("normalize")
    audit = subparsers.add_parser("audit")
    audit.add_argument("--skip-raw-hash-verification", action="store_true")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--skip-raw-hash-verification", action="store_true")

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument(
        "--protocol",
        choices=(
            "main_session_grouped",
            "leave_one_collection_out",
            "pairwise_collection_transfer",
            "gcode_blocked",
            "all",
        ),
        default="all",
    )

    train_task = subparsers.add_parser("train-task")
    train_task.add_argument("--protocol", required=True)
    train_task.add_argument("--fold-id", required=True)
    train_task.add_argument("--seed", type=int, required=True)
    train_task.add_argument("--device", default="auto")
    train_task.add_argument("--resume", action="store_true")

    suite = subparsers.add_parser("run-suite")
    suite.add_argument("--parallel", type=int, default=2)
    suite.add_argument("--device", default="cuda")

    article_task = subparsers.add_parser("train-article-task")
    article_task.add_argument("--plan", type=Path, required=True)
    article_task.add_argument("--normalization", required=True)
    article_task.add_argument("--protocol", required=True)
    article_task.add_argument("--fold-id", required=True)
    article_task.add_argument("--seed", type=int, required=True)
    article_task.add_argument("--device", default="auto")
    article_task.add_argument("--resume", action="store_true")

    article_suite = subparsers.add_parser("run-article-suite")
    article_suite.add_argument("--plan", type=Path, required=True)
    article_suite.add_argument("--parallel", type=int, default=1)
    article_suite.add_argument("--device", default="cuda")
    article_suite.add_argument("--seed", type=int, action="append")
    article_suite.add_argument("--normalization", action="append")

    context_audit = subparsers.add_parser("audit-article-context")
    context_audit.add_argument("--plan", type=Path, required=True)

    article_report = subparsers.add_parser("summarize-article-experiment")
    article_report.add_argument("--plan", type=Path, required=True)

    subparsers.add_parser("summarize")
    article_ablation = subparsers.add_parser("summarize-article-ablation")
    article_ablation.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/article_ablation"),
    )
    subparsers.add_parser("make-filter-response")
    subparsers.add_parser("make-dataset-figures")
    subparsers.add_parser("make-temporal-examples")

    smoke = subparsers.add_parser("smoke-test")
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--model-config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one reproducibility command."""
    arguments = _parser().parse_args(argv)
    root = arguments.project_root.resolve()
    command = arguments.command
    if command == "generate-gcode":
        _generate_gcode(root, arguments.output_dir)
    elif command == "verify-gcode":
        _print_json(_verify_generated_gcode(root, arguments.generated_dir))
    elif command == "preprocess":
        _preprocess(
            root,
            output_dir=arguments.output_dir,
            max_sessions=arguments.max_sessions,
            verify_hashes=not arguments.skip_raw_hash_verification,
        )
    elif command == "make-splits":
        _make_splits(root)
    elif command == "normalize":
        _normalize(root)
    elif command == "audit":
        _audit(root, not arguments.skip_raw_hash_verification)
    elif command == "prepare":
        _prepare(root, not arguments.skip_raw_hash_verification)
    elif command == "baseline":
        _baseline(root, arguments.protocol)
    elif command == "train-task":
        report = run_training_task(
            project_root=root,
            task=ExperimentTask(
                protocol=arguments.protocol,
                fold_id=arguments.fold_id,
                seed=arguments.seed,
            ),
            device=arguments.device,
            resume=arguments.resume,
            progress=True,
        )
        _print_json({"event_metrics": report["event_metrics"]})
    elif command == "run-suite":
        _print_json(
            run_suite(
                project_root=root,
                parallel_tasks=arguments.parallel,
                device=arguments.device,
            )
        )
    elif command == "train-article-task":
        plan = load_article_experiment_plan(arguments.plan)
        report = run_article_task(
            plan=plan,
            task=ArticleExperimentTask(
                normalization_variant=arguments.normalization,
                protocol=arguments.protocol,
                fold_id=arguments.fold_id,
                seed=arguments.seed,
            ),
            device=arguments.device,
            resume=arguments.resume,
            progress=True,
        )
        _print_json({"event_metrics": report["event_metrics"]})
    elif command == "run-article-suite":
        plan = load_article_experiment_plan(arguments.plan)
        _print_json(
            run_article_suite(
                plan=plan,
                parallel_tasks=arguments.parallel,
                device=arguments.device,
                seeds=arguments.seed,
                normalization_variants=arguments.normalization,
            )
        )
    elif command == "audit-article-context":
        plan = load_article_experiment_plan(arguments.plan)
        _print_json(write_context_safety_audit(plan))
    elif command == "summarize-article-experiment":
        plan = load_article_experiment_plan(arguments.plan)
        summary = write_article_experiment_reports(plan)
        _print_json(
            {
                "experiment_id": summary["experiment_id"],
                "validated_evaluations": summary["validated_evaluations"],
            }
        )
    elif command == "summarize":
        _print_json(
            summarize_results(
                project_root=root,
                output_dir=root / "outputs" / "reports" / "final",
            )
        )
    elif command == "summarize-article-ablation":
        _print_json(
            write_article_ablation_reports(
                project_root=root,
                output_dir=root / arguments.output_dir,
            )
        )
    elif command == "make-filter-response":
        _print_json(
            write_filter_response(
                config=load_config(root / "configs" / "experiment.yaml"),
                output_dir=root / "outputs" / "reports" / "filter_response",
            )
        )
    elif command == "make-dataset-figures":
        _print_json(
            write_dataset_figures(
                dataset_dir=root / "outputs" / "processed" / "dataset",
                output_dir=root / "outputs" / "reports" / "dataset_figures",
            )
        )
    elif command == "make-temporal-examples":
        _print_json(
            write_temporal_examples(
                dataset_dir=root / "outputs" / "processed" / "dataset",
                raw_manifest_path=root / "data" / "metadata" / "raw_manifest.json",
                selection_path=root / "data" / "metadata" / "temporal_examples.csv",
                output_dir=root / "outputs" / "reports" / "temporal_examples",
            )
        )
    elif command == "smoke-test":
        _print_json(
            _smoke_test(
                root,
                arguments.device,
                model_path=arguments.model_config,
            )
        )
    else:
        raise AssertionError(f"Unsupported command: {command}")


def _generate_gcode(root: Path, output_dir: Path) -> None:
    destination = output_dir if output_dir.is_absolute() else root / output_dir
    subprocess.run(
        [
            sys.executable,
            "-m",
            "bee_counter_bench.gcode.generator",
            "--output-dir",
            str(destination),
        ],
        cwd=root,
        check=True,
    )


def _verify_generated_gcode(root: Path, generated_dir: Path) -> dict[str, Any]:
    directory = generated_dir if generated_dir.is_absolute() else root / generated_dir
    manifest = json.loads(
        (root / "data" / "metadata" / "gcode_manifest.json").read_text(
            encoding="utf-8-sig"
        )
    )
    expected = {
        Path(record["relative_path"]).name: str(record["sha256"])
        for record in manifest["files"]
    }
    generated = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.gcode"))
    }
    if generated != expected:
        missing = sorted(set(expected) - set(generated))
        extra = sorted(set(generated) - set(expected))
        mismatched = sorted(
            name
            for name in expected.keys() & generated.keys()
            if expected[name] != generated[name]
        )
        raise ValueError(
            f"Generated G-code mismatch: missing={missing}, extra={extra}, "
            f"hash_mismatch={mismatched}"
        )
    return {"status": "passed", "files": len(generated)}


def _preprocess(
    root: Path,
    *,
    output_dir: Path,
    max_sessions: int | None,
    verify_hashes: bool,
) -> None:
    destination = output_dir if output_dir.is_absolute() else root / output_dir
    report = build_dataset(
        project_root=root,
        config=load_config(root / "configs" / "experiment.yaml"),
        output_dir=destination,
        max_sessions=max_sessions,
        verify_raw_hashes=verify_hashes,
    )
    _print_json(report["summary"])


def _make_splits(root: Path) -> None:
    paths = build_split_manifests(
        project_root=root,
        dataset_dir=root / "outputs" / "processed" / "dataset",
        output_dir=root / "outputs" / "splits",
    )
    _print_json({name: str(path.relative_to(root)) for name, path in paths.items()})


def _normalize(root: Path) -> None:
    paths = build_all_normalizations(
        dataset_dir=root / "outputs" / "processed" / "dataset",
        split_dir=root / "outputs" / "splits",
        output_dir=root / "outputs" / "normalization",
    )
    _print_json({name: str(path.relative_to(root)) for name, path in paths.items()})


def _audit(root: Path, verify_hashes: bool) -> None:
    report = audit_pretraining_artifacts(
        project_root=root,
        config=load_config(root / "configs" / "experiment.yaml"),
        dataset_dir=root / "outputs" / "processed" / "dataset",
        split_dir=root / "outputs" / "splits",
        normalization_dir=root / "outputs" / "normalization",
        output_path=root / "outputs" / "audit" / "pretraining_audit.json",
        verify_raw_hashes=verify_hashes,
    )
    _print_json({"status": report["status"], "dataset": report["dataset"]})


def _prepare(root: Path, verify_hashes: bool) -> None:
    _preprocess(
        root,
        output_dir=Path("outputs/processed/dataset"),
        max_sessions=None,
        verify_hashes=verify_hashes,
    )
    _make_splits(root)
    _normalize(root)
    _audit(root, verify_hashes=False)


def _baseline(root: Path, requested_protocol: str) -> None:
    experiment = load_yaml(root / "configs" / "experiment.yaml")
    baseline = experiment["baseline"]
    evaluation = experiment["evaluation"]
    protocols = (
        (
            "main_session_grouped",
            "leave_one_collection_out",
            "pairwise_collection_transfer",
            "gcode_blocked",
        )
        if requested_protocol == "all"
        else (requested_protocol,)
    )
    reports = {}
    for protocol in protocols:
        report = run_threshold_baseline(
            dataset_dir=root / "outputs" / "processed" / "dataset",
            split_manifest_path=root / "outputs" / "splits" / f"{protocol}.json",
            normalization_manifest_path=(
                root / "outputs" / "normalization" / f"{protocol}.json"
            ),
            output_path=root / "outputs" / "baseline" / f"{protocol}.json",
            thresholds=tuple(float(value) for value in baseline["thresholds"]),
            merge_gap_frames=tuple(
                int(value) for value in baseline["merge_gap_frames"]
            ),
            minimum_duration_frames=tuple(
                int(value) for value in baseline["minimum_duration_frames"]
            ),
            iou_thresholds=tuple(
                float(value) for value in evaluation["event_iou_thresholds"]
            ),
        )
        reports[protocol] = report["macro_summary"]
    _print_json(reports)


def _smoke_test(
    root: Path,
    device_name: str,
    *,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Run one real batch through model and loss without an optimizer step."""
    resolved_model_path = (
        root / "configs" / "model.yaml"
        if model_path is None
        else (model_path if model_path.is_absolute() else root / model_path)
    )
    config = load_config(
        root / "configs" / "experiment.yaml",
        model_path=resolved_model_path,
    )
    split_path = root / "outputs" / "splits" / "main_session_grouped.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    fold_id = str(split["folds"][0]["fold_id"])
    dataset = PassageSequenceDataset(
        dataset_dir=root / "outputs" / "processed" / "dataset",
        split_manifest=split_path,
        normalization_manifest=(
            root / "outputs" / "normalization" / "main_session_grouped.json"
        ),
        fold_id=fold_id,
        split="train",
        crop_length=900,
        halo_frames=config.model.future_context_frames,
    )
    batch = collate_padded([dataset[0]])
    device = torch.device(device_name)
    model = build_model(config).to(device)
    outputs = model(
        batch["features"].to(device),
        batch["input_mask"].to(device),
    )
    loss = MSTCNPPMultiStageLoss(
        smoothing_weight=config.training.smoothing_weight,
        smoothing_tau=config.training.smoothing_tau,
    )(
        outputs,
        batch["labels"].to(device),
        batch["loss_mask"].to(device),
    )
    loss.total.backward()
    return {
        "status": "passed",
        "device": str(device),
        "fold_id": fold_id,
        "input_frames": int(batch["features"].shape[-1]),
        "loss_frames": int(batch["loss_mask"].sum().item()),
        "stages": int(outputs.shape[0]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "receptive_field_frames": config.model.receptive_field_frames,
        "loss": float(loss.total.detach().cpu().item()),
    }


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
