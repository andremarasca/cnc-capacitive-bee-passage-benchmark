from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml


MANIFEST_NAME = "release_manifest.json"
EXPECTED = {
    "article_tasks": 48,
    "dataset_events": 22_571,
    "dataset_frames": 3_387_695,
    "dataset_sessions": 36,
    "raw_chunks": 1_142,
    "receptive_field_frames": 165,
    "source_code_sha256": (
        "9a7c7e8efa5ca4475622fc9386cd7522e582de419b436e3e478b3be6c7a5558e"
    ),
    "trainable_parameters": 75_337,
}
IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
}
REQUIRED_SEQUENCE_KEYS = {
    "event_direction",
    "event_end_idx",
    "event_speeds_mm_min",
    "event_start_idx",
    "labels",
    "position_mm",
    "preprocessing_json",
    "sample_rate_hz",
    "schema_version",
    "session_id",
    "signal",
    "source_path",
    "source_sha256",
    "valid_mask",
}


class ReleaseAuditError(RuntimeError):
    """Raised when the curated release violates its frozen contract."""


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scientific_source_sha256(root: Path) -> str:
    """Reproduce the source hash embedded in the article artifacts."""
    paths = [
        path
        for path in (root / "src").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    paths.extend(path for path in (root / "configs").glob("*.yaml") if path.is_file())
    paths.extend(
        path for path in (root / "pyproject.toml", root / "uv.lock") if path.is_file()
    )

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative_path = path.relative_to(root).as_posix().encode("utf-8")
        content_digest = hashlib.sha256(path.read_bytes()).digest()
        digest.update(len(relative_path).to_bytes(4, "big"))
        digest.update(relative_path)
        digest.update(content_digest)
    return digest.hexdigest()


def relative_files(root: Path) -> list[Path]:
    """List stable release files while ignoring local Git and tool caches."""
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        has_ignored_part = any(
            part in IGNORED_PARTS or part.endswith(".egg-info")
            for part in relative.parts
        )
        if not path.is_file() or has_ignored_part:
            continue
        if relative.as_posix() == MANIFEST_NAME or path.suffix == ".pyc":
            continue
        files.append(relative)
    return sorted(files, key=lambda item: item.as_posix())


def build_manifest(root: Path) -> dict[str, Any]:
    """Build the deterministic checksum manifest for the curated snapshot."""
    entries: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for relative in relative_files(root):
        path = root / relative
        size = path.stat().st_size
        total_bytes += size
        entries[relative.as_posix()] = {
            "sha256": sha256(path),
            "size_bytes": size,
        }
    return {
        "schema_version": 1,
        "generated_by": "scripts/release_audit.py",
        "scope": {
            "license_granted": False,
            "legacy_rf285_results_included": False,
            "raw_recordings_included": False,
            "source_archive_commit": (
                "7e66ee07dc5d4c59a7e7cce566909d0c38ef376f"
            ),
        },
        "expected": EXPECTED,
        "summary": {
            "file_count": len(entries),
            "total_bytes": total_bytes,
        },
        "files": entries,
    }


def verify_manifest(root: Path) -> None:
    """Verify all release files against the frozen checksum manifest."""
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ReleaseAuditError(
            f"Missing {MANIFEST_NAME}. Run with --write-manifest after review."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_entries = manifest.get("files", {})
    actual_files = {path.as_posix() for path in relative_files(root)}
    expected_files = set(expected_entries)

    missing = sorted(expected_files - actual_files)
    unexpected = sorted(actual_files - expected_files)
    if missing:
        raise ReleaseAuditError(f"Missing release files: {missing}")
    if unexpected:
        raise ReleaseAuditError(f"Unexpected release files: {unexpected}")

    for relative, expected in expected_entries.items():
        path = root / relative
        actual_size = path.stat().st_size
        if actual_size != expected["size_bytes"]:
            raise ReleaseAuditError(f"Size mismatch: {relative}")
        if sha256(path) != expected["sha256"]:
            raise ReleaseAuditError(f"SHA-256 mismatch: {relative}")


def verify_scope(root: Path) -> None:
    """Reject raw, transient, historical, or accidental release content."""
    forbidden_directories = [
        root / "data" / "raw",
        root / "outputs" / "ablations",
        root / "outputs" / "runs",
    ]
    existing = [str(path.relative_to(root)) for path in forbidden_directories if path.exists()]
    if existing:
        raise ReleaseAuditError(f"Forbidden directories present: {existing}")

    forbidden_names = {"last_checkpoint.pt", "status.json"}
    forbidden_suffixes = {".tar", ".tgz"}
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.relative_to(root).parts:
            continue
        relative = path.relative_to(root)
        if path.name in forbidden_names:
            raise ReleaseAuditError(f"Transient artifact present: {relative}")
        if path.suffix in forbidden_suffixes or path.name.endswith(".tar.gz"):
            raise ReleaseAuditError(f"Archive artifact present: {relative}")
        if path.suffix == ".npz" and not relative.as_posix().startswith(
            "outputs/processed/dataset/"
        ):
            raise ReleaseAuditError(f"NPZ outside the processed dataset: {relative}")
        if any(part.endswith(".tmp") for part in relative.parts):
            raise ReleaseAuditError(f"Temporary path present: {relative}")

    campaign = root / "outputs" / "experiments" / "article_rf165"
    for transient in (campaign / "logs", campaign / "remote_logs"):
        if transient.exists():
            raise ReleaseAuditError(
                f"Transient campaign directory present: {transient.relative_to(root)}"
            )


def verify_structured_files(root: Path) -> dict[str, int]:
    """Parse every versioned JSON, YAML, and CSV artifact."""
    counts = {"csv": 0, "json": 0, "yaml": 0}
    for relative in relative_files(root):
        path = root / relative
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8-sig"))
            counts["json"] += 1
        elif path.suffix in {".yaml", ".yml"}:
            yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            counts["yaml"] += 1
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as file_handle:
                reader = csv.reader(file_handle)
                rows = [row for row in reader if row]
            if not rows or not rows[0]:
                raise ReleaseAuditError(f"Empty CSV artifact: {relative}")
            width = len(rows[0])
            if any(len(row) != width for row in rows[1:]):
                raise ReleaseAuditError(f"Inconsistent CSV width: {relative}")
            counts["csv"] += 1
    return counts


def verify_text_safety(root: Path) -> None:
    """Reject common credentials and private local paths in text artifacts."""
    patterns = {
        "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
        "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "private Windows path": re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
        "private project path": re.compile(r"C:\\doutorado\\", re.IGNORECASE),
        "secret assignment": re.compile(
            r"(?:api[_-]?key|password|passwd|secret|token)\s*[:=]\s*"
            r"['\"][^'\"]+['\"]",
            re.IGNORECASE,
        ),
    }
    text_suffixes = {
        "",
        ".csv",
        ".gcode",
        ".gitignore",
        ".json",
        ".md",
        ".py",
        ".settings",
        ".sh",
        ".toml",
        ".yaml",
        ".yml",
    }
    for relative in relative_files(root):
        path = root / relative
        if path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for description, pattern in patterns.items():
            if pattern.search(text):
                raise ReleaseAuditError(f"Detected {description} in {relative}")


def verify_checkpoints(root: Path) -> int:
    """Load checksummed checkpoints and validate their RF165 state dictionaries."""
    import torch

    campaign_root = root / "outputs" / "experiments" / "article_rf165"
    checkpoints = sorted(campaign_root.rglob("best_checkpoint.pt"))
    for path in checkpoints:
        # These repository-owned files are loaded only after checksum verification.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 4:
            raise ReleaseAuditError(f"Checkpoint schema mismatch: {path}")
        state = payload.get("model_state_dict")
        if not isinstance(state, dict):
            raise ReleaseAuditError(f"Missing model state: {path}")
        parameter_count = sum(tensor.numel() for tensor in state.values())
        if parameter_count != EXPECTED["trainable_parameters"]:
            raise ReleaseAuditError(f"Checkpoint parameter mismatch: {path}")
        model_block = payload.get("config", {}).get("model", {})
        model_config = model_block.get("architecture", model_block)
        if model_config.get("receptive_field_frames") != EXPECTED[
            "receptive_field_frames"
        ]:
            raise ReleaseAuditError(f"Checkpoint RF165 mismatch: {path}")
    return len(checkpoints)


def verify_dataset(root: Path) -> dict[str, int]:
    """Verify processed sequence hashes, schemas, and published totals."""
    dataset_root = root / "outputs" / "processed" / "dataset"
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sessions = manifest.get("sequences", [])
    summary = manifest.get("summary", {})

    if len(sessions) != EXPECTED["dataset_sessions"]:
        raise ReleaseAuditError("Unexpected number of dataset sessions.")
    if summary.get("sessions") != EXPECTED["dataset_sessions"]:
        raise ReleaseAuditError("Dataset manifest session total mismatch.")
    if summary.get("source_chunks") != EXPECTED["raw_chunks"]:
        raise ReleaseAuditError("Dataset manifest raw-chunk total mismatch.")
    if summary.get("output_frames") != EXPECTED["dataset_frames"]:
        raise ReleaseAuditError("Dataset manifest frame total mismatch.")
    if sum(summary.get("events", {}).values()) != EXPECTED["dataset_events"]:
        raise ReleaseAuditError("Dataset manifest event total mismatch.")

    total_frames = 0
    total_events = 0
    listed_outputs: set[str] = set()
    for session in sessions:
        relative_output = Path(session["output"])
        listed_outputs.add(relative_output.as_posix())
        sequence_path = dataset_root / relative_output
        if not sequence_path.is_file():
            raise ReleaseAuditError(f"Missing processed sequence: {relative_output}")
        if sha256(sequence_path) != session["output_sha256"]:
            raise ReleaseAuditError(f"Processed sequence hash mismatch: {relative_output}")

        with np.load(sequence_path, allow_pickle=False) as sequence:
            missing_keys = REQUIRED_SEQUENCE_KEYS - set(sequence.files)
            if missing_keys:
                raise ReleaseAuditError(
                    f"Missing sequence keys in {relative_output}: {sorted(missing_keys)}"
                )
            if any(sequence[key].dtype.hasobject for key in sequence.files):
                raise ReleaseAuditError(f"Object array present in {relative_output}")
            frames = int(sequence["signal"].shape[0])
            if any(
                sequence[key].shape[0] != frames
                for key in ("position_mm", "labels", "valid_mask")
            ):
                raise ReleaseAuditError(f"Frame-array length mismatch: {relative_output}")
            if int(sequence["sample_rate_hz"]) != 30:
                raise ReleaseAuditError(f"Unexpected sample rate: {relative_output}")
            labels = set(np.unique(sequence["labels"]).tolist())
            if not labels.issubset({0, 1, 2}):
                raise ReleaseAuditError(f"Unexpected label value: {relative_output}")
            event_count = int(sequence["event_start_idx"].shape[0])
            if any(
                sequence[key].shape[0] != event_count
                for key in ("event_end_idx", "event_direction", "event_speeds_mm_min")
            ):
                raise ReleaseAuditError(f"Event-array length mismatch: {relative_output}")
            total_frames += frames
            total_events += event_count

    actual_outputs = {
        path.relative_to(dataset_root).as_posix()
        for path in dataset_root.glob("*/*__sequence.npz")
    }
    if actual_outputs != listed_outputs:
        raise ReleaseAuditError("Processed dataset files differ from the dataset manifest.")
    if total_frames != EXPECTED["dataset_frames"]:
        raise ReleaseAuditError("Processed sequence frame total mismatch.")
    if total_events != EXPECTED["dataset_events"]:
        raise ReleaseAuditError("Processed sequence event total mismatch.")

    return {
        "events": total_events,
        "frames": total_frames,
        "sessions": len(sessions),
    }


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and reject non-mapping documents."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReleaseAuditError(f"Expected a YAML mapping: {path}")
    return payload


def verify_rf165_contract(root: Path) -> dict[str, Any]:
    """Verify the frozen model, campaign plan, and curated run inventory."""
    model_path = root / "configs" / "model_rf165.yaml"
    model = load_yaml(model_path)
    architecture = model["architecture"]
    if architecture["receptive_field_frames"] != EXPECTED["receptive_field_frames"]:
        raise ReleaseAuditError("RF165 receptive-field mismatch.")
    if architecture["expected_trainable_parameters"] != EXPECTED["trainable_parameters"]:
        raise ReleaseAuditError("RF165 parameter-count mismatch.")
    if architecture["context_frames_each_side"] != 82:
        raise ReleaseAuditError("RF165 context mismatch.")

    plan = load_yaml(root / "configs" / "article_rf165.yaml")
    if plan["model_config"] != "configs/model_rf165.yaml":
        raise ReleaseAuditError("Article plan does not select the RF165 configuration.")
    if plan["expected_task_count"] != EXPECTED["article_tasks"]:
        raise ReleaseAuditError("Article plan task-count mismatch.")
    if plan["protocols"] != ["main_session_grouped", "leave_one_collection_out"]:
        raise ReleaseAuditError("Unexpected article protocol set.")
    if plan["seeds"] != [42, 43, 44]:
        raise ReleaseAuditError("Unexpected article seed set.")

    campaign_root = root / "outputs" / "experiments" / "article_rf165"
    campaign = json.loads(
        (campaign_root / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    execution = campaign["execution"]
    if execution["task_count"] != EXPECTED["article_tasks"]:
        raise ReleaseAuditError("Campaign task-count mismatch.")
    if len(set(execution["task_ids"])) != EXPECTED["article_tasks"]:
        raise ReleaseAuditError("Campaign task identifiers are incomplete or duplicated.")

    source_hash = scientific_source_sha256(root)
    if source_hash != EXPECTED["source_code_sha256"]:
        raise ReleaseAuditError("Scientific source hash differs from the article artifacts.")

    run_manifests = sorted(campaign_root.rglob("run_manifest.json"))
    for run_manifest_path in run_manifests:
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest["source_code_sha256"] != source_hash:
            raise ReleaseAuditError(
                f"Run source hash mismatch: {run_manifest_path.relative_to(root)}"
            )
        if run_manifest["execution"]["halo_frames"] != 82:
            raise ReleaseAuditError(
                f"Run RF165 halo mismatch: {run_manifest_path.relative_to(root)}"
            )

    expected_counts = {
        "evaluation.json": EXPECTED["article_tasks"],
        "run_manifest.json": EXPECTED["article_tasks"],
        "training_log.json": EXPECTED["article_tasks"],
        "best_checkpoint.pt": EXPECTED["article_tasks"],
        "resolved_config.yaml": EXPECTED["article_tasks"],
        "resolved_normalization.json": EXPECTED["article_tasks"],
    }
    for name, expected_count in expected_counts.items():
        actual_count = sum(1 for _ in campaign_root.rglob(name))
        if actual_count != expected_count:
            raise ReleaseAuditError(
                f"Unexpected {name} count: {actual_count}, expected {expected_count}."
            )

    return {
        "parameters": architecture["expected_trainable_parameters"],
        "receptive_field_frames": architecture["receptive_field_frames"],
        "source_code_sha256": source_hash,
        "tasks": execution["task_count"],
    }


def find_metric(
    rows: list[dict[str, str]],
    **criteria: str,
) -> float:
    """Find one metric row by an exact set of categorical fields."""
    matches = [
        row
        for row in rows
        if all(row.get(field) == value for field, value in criteria.items())
    ]
    if len(matches) != 1:
        raise ReleaseAuditError(f"Expected one metric row for {criteria}, found {len(matches)}.")
    return float(matches[0]["f1_mean"])


def verify_article_metrics(root: Path) -> dict[str, float]:
    """Verify the principal article-facing aggregate metrics."""
    report_root = root / "outputs" / "experiments" / "article_rf165" / "reports"
    with (report_root / "matching_sensitivity.csv").open(
        encoding="utf-8", newline=""
    ) as file_handle:
        rows = list(csv.DictReader(file_handle))

    observed = {
        "fixed_5fold_directional_iou25": find_metric(
            rows,
            normalization="training_population",
            protocol="main_session_grouped",
            direction_mode="directional",
            criterion="iou_25",
        ),
        "adaptive_5fold_directional_iou25": find_metric(
            rows,
            normalization="causal_adaptive_tau60",
            protocol="main_session_grouped",
            direction_mode="directional",
            criterion="iou_25",
        ),
        "fixed_lco_directional_iou25": find_metric(
            rows,
            normalization="training_population",
            protocol="leave_one_collection_out",
            direction_mode="directional",
            criterion="iou_25",
        ),
        "adaptive_lco_directional_iou25": find_metric(
            rows,
            normalization="causal_adaptive_tau60",
            protocol="leave_one_collection_out",
            direction_mode="directional",
            criterion="iou_25",
        ),
        "adaptive_5fold_presence_any_overlap": find_metric(
            rows,
            normalization="causal_adaptive_tau60",
            protocol="main_session_grouped",
            direction_mode="presence",
            criterion="any_overlap",
        ),
    }
    expected = {
        "fixed_5fold_directional_iou25": 0.7694913198574055,
        "adaptive_5fold_directional_iou25": 0.7956509348962928,
        "fixed_lco_directional_iou25": 0.5610489182587053,
        "adaptive_lco_directional_iou25": 0.6630743965096004,
        "adaptive_5fold_presence_any_overlap": 0.8948308975360422,
    }
    for name, expected_value in expected.items():
        if not math.isclose(observed[name], expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise ReleaseAuditError(f"Article metric mismatch: {name}")

    baseline = json.loads(
        (root / "outputs" / "baseline" / "main_session_grouped.json").read_text(
            encoding="utf-8"
        )
    )
    observed["heuristic_directional_iou25"] = baseline["aggregate_metrics"][
        "thresholds"
    ]["iou_25"]["f1"]
    if not math.isclose(
        observed["heuristic_directional_iou25"],
        0.2693770739471767,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ReleaseAuditError("Heuristic baseline metric mismatch.")
    return observed


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit the curated RF165 code-and-data release."
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write a deterministic checksum manifest after all domain checks pass.",
    )
    return parser.parse_args()


def main() -> None:
    """Run domain checks and either write or verify the release manifest."""
    arguments = parse_arguments()
    root = Path(__file__).resolve().parents[1]

    if not arguments.write_manifest:
        verify_manifest(root)
    verify_scope(root)
    structured_files = verify_structured_files(root)
    verify_text_safety(root)
    dataset = verify_dataset(root)
    rf165 = verify_rf165_contract(root)
    metrics = verify_article_metrics(root)
    checkpoints = verify_checkpoints(root)

    if arguments.write_manifest:
        manifest = build_manifest(root)
        (root / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        manifest_status = "written"
    else:
        manifest_status = "verified"

    print(
        json.dumps(
            {
                "dataset": dataset,
                "checkpoints": checkpoints,
                "manifest": manifest_status,
                "metrics": metrics,
                "rf165": rf165,
                "status": "passed",
                "structured_files": structured_files,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
