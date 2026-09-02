"""Reproducible MS-TCN++ training loop."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from bee_counter_bench.config import (
    AppConfig,
    NormalizationConfig,
    config_to_dict,
    normalization_config_to_dict,
    save_resolved_config,
)
from bee_counter_bench.data.sequence_io import file_sha256
from bee_counter_bench.models.mstcnpp import MSTCNPP
from bee_counter_bench.provenance import find_project_root, source_code_sha256
from bee_counter_bench.training.checkpoint import (
    checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)
from bee_counter_bench.training.dataset import PassageSequenceDataset, collate_padded
from bee_counter_bench.training.losses import MSTCNPPMultiStageLoss


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_cross_entropy: float
    train_smoothing: float
    train_frame_accuracy: float
    train_macro_f1: float
    train_left_f1: float
    train_right_f1: float
    validation_loss: float
    validation_cross_entropy: float
    validation_smoothing: float
    validation_frame_accuracy: float
    validation_macro_f1: float
    validation_left_f1: float
    validation_right_f1: float
    improved: bool


def set_reproducible_seed(seed: int) -> None:
    """Seed every training RNG and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_model(config: AppConfig) -> MSTCNPP:
    """Instantiate the configured MS-TCN++ reference model."""
    model_cfg = config.model
    model = MSTCNPP(
        prediction_generation_layers=model_cfg.prediction_generation_layers,
        refinement_layers=model_cfg.refinement_layers,
        refinement_stages=model_cfg.refinement_stages,
        feature_maps=model_cfg.feature_maps,
        input_channels=model_cfg.input_channels,
        num_classes=model_cfg.num_classes,
        dropout=model_cfg.dropout,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != model_cfg.expected_trainable_parameters:
        raise ValueError(
            "Configured and instantiated trainable parameter counts differ."
        )
    return model


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("features", "labels", "input_mask", "loss_mask"):
        moved[key] = batch[key].to(device)
    return moved


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: MSTCNPPMultiStageLoss,
    device: torch.device,
    optimizer: Adam | None,
    phase: str,
    epoch: int,
    total_epochs: int,
    progress: bool,
    progress_every_batches: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "cross_entropy": 0.0, "smoothing": 0.0, "frames": 0.0}
    confusion = torch.zeros((3, 3), dtype=torch.long)
    batch_total = len(loader)
    started = time.monotonic()
    if progress:
        print(
            f"[epoch {epoch:03d}/{total_epochs:03d}] {phase}: "
            f"{batch_total} batches",
            flush=True,
        )

    for batch_index, batch in enumerate(loader, start=1):
        batch = _move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(batch["features"], batch["input_mask"])
        loss = criterion(logits, batch["labels"], batch["loss_mask"])
        if training:
            loss.total.backward()
            optimizer.step()

        frames = float(batch["loss_mask"].sum().item())
        totals["loss"] += float(loss.total.detach().cpu().item()) * frames
        totals["cross_entropy"] += (
            float(loss.cross_entropy.detach().cpu().item()) * frames
        )
        totals["smoothing"] += float(loss.smoothing.detach().cpu().item()) * frames
        totals["frames"] += frames

        predictions = logits[-1].argmax(dim=1)
        valid = batch["loss_mask"].detach().cpu().bool()
        targets = batch["labels"].detach().cpu()
        predicted_cpu = predictions.detach().cpu()
        for target_class in range(3):
            target_mask = valid & (targets == target_class)
            for predicted_class in range(3):
                confusion[target_class, predicted_class] += int(
                    torch.count_nonzero(
                        target_mask & (predicted_cpu == predicted_class)
                    )
                )

        should_report = (
            batch_index == batch_total or batch_index % progress_every_batches == 0
        )
        if progress and should_report:
            elapsed = max(time.monotonic() - started, 1e-9)
            average_loss = totals["loss"] / max(totals["frames"], 1.0)
            batches_per_second = batch_index / elapsed
            print(
                f"[epoch {epoch:03d}/{total_epochs:03d}] {phase}: "
                f"{batch_index}/{batch_total} batches | "
                f"loss={average_loss:.6f} | {batches_per_second:.2f} batch/s",
                flush=True,
            )

    if totals["frames"] <= 0:
        raise ValueError("Epoch contained no valid frames.")
    metric_payload = _confusion_metrics(confusion)
    return {
        "loss": totals["loss"] / totals["frames"],
        "cross_entropy": totals["cross_entropy"] / totals["frames"],
        "smoothing": totals["smoothing"] / totals["frames"],
        **metric_payload,
    }


def _confusion_metrics(confusion: torch.Tensor) -> dict[str, float]:
    total = float(confusion.sum().item())
    accuracy = 0.0 if total == 0 else float(confusion.diag().sum().item()) / total
    f1_values = []
    metrics: dict[str, float] = {"frame_accuracy": accuracy}
    names = ("out", "left", "right")
    for index, name in enumerate(names):
        true_positive = float(confusion[index, index].item())
        false_positive = float(confusion[:, index].sum().item()) - true_positive
        false_negative = float(confusion[index, :].sum().item()) - true_positive
        precision = (
            0.0
            if true_positive + false_positive == 0
            else true_positive / (true_positive + false_positive)
        )
        recall = (
            0.0
            if true_positive + false_negative == 0
            else true_positive / (true_positive + false_negative)
        )
        f1 = (
            0.0
            if precision + recall == 0
            else 2.0 * precision * recall / (precision + recall)
        )
        metrics[f"{name}_f1"] = f1
        f1_values.append(f1)
    metrics["macro_f1"] = float(sum(f1_values) / len(f1_values))
    return metrics


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def train_fold(
    *,
    config: AppConfig,
    dataset_dir: str | Path,
    split_manifest: str | Path,
    normalization_manifest: str | Path,
    fold_id: str,
    output_dir: str | Path,
    crop_length: int | None = 900,
    halo_frames: int = 142,
    device: str = "auto",
    num_workers: int = 0,
    resume: bool = False,
    progress: bool = False,
    progress_every_batches: int = 100,
    normalization_config: NormalizationConfig | None = None,
) -> dict[str, Any]:
    """Train one fold and save best/last checkpoints plus compact logs."""
    if progress_every_batches <= 0:
        raise ValueError("progress_every_batches must be positive.")

    destination = Path(output_dir)
    destination_has_content = destination.exists() and any(destination.iterdir())
    if destination_has_content and not resume:
        raise FileExistsError(f"Refusing to overwrite training run: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    resolved_config_path = destination / "resolved_config.yaml"
    if not resolved_config_path.exists():
        save_resolved_config(config, resolved_config_path)
    resolved_normalization_path = destination / "resolved_normalization.json"
    if not resolved_normalization_path.exists():
        _write_json(
            resolved_normalization_path,
            normalization_config_to_dict(normalization_config),
        )

    set_reproducible_seed(config.training.seed)
    torch_device = _device(device)
    train_dataset = PassageSequenceDataset(
        dataset_dir=dataset_dir,
        split_manifest=split_manifest,
        normalization_manifest=normalization_manifest,
        fold_id=fold_id,
        split="train",
        crop_length=crop_length,
        halo_frames=halo_frames,
        normalization_config=normalization_config,
    )
    validation_dataset = PassageSequenceDataset(
        dataset_dir=dataset_dir,
        split_manifest=split_manifest,
        normalization_manifest=normalization_manifest,
        fold_id=fold_id,
        split="validation",
        crop_length=crop_length,
        halo_frames=halo_frames,
        normalization_config=normalization_config,
    )
    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(config.training.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_padded,
        generator=data_loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_padded,
    )

    model = build_model(config).to(torch_device)
    criterion = MSTCNPPMultiStageLoss(
        smoothing_weight=config.training.smoothing_weight,
        smoothing_tau=config.training.smoothing_tau,
    )
    optimizer = Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=config.training.adam_betas,
        eps=config.training.adam_epsilon,
        weight_decay=config.training.weight_decay,
    )

    best_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    history: list[EpochMetrics] = []
    start_epoch = 1
    dataset_root = Path(dataset_dir)
    split_path = Path(split_manifest)
    normalization_path = Path(normalization_manifest)
    code_sha256 = source_code_sha256(find_project_root(dataset_root))

    if resume:
        checkpoint_path = _resume_checkpoint_path(destination)
        if checkpoint_path is not None:
            checkpoint = load_checkpoint(checkpoint_path)
            _validate_resume_checkpoint(
                checkpoint,
                config=config,
                fold_id=fold_id,
                dataset_manifest_sha256=file_sha256(dataset_root / "manifest.json"),
                split_manifest_sha256=file_sha256(split_path),
                normalization_manifest_sha256=file_sha256(normalization_path),
                normalization_config=normalization_config,
                source_code_sha256=code_sha256,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            _move_optimizer_state(optimizer, torch_device)
            resume_state = checkpoint.get("resume_state", {})
            _restore_random_state(
                resume_state.get("random_state"), data_loader_generator
            )
            history = _history_from_resume_state_or_log(
                resume_state,
                destination / "training_log.json",
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            if history:
                start_epoch = max(start_epoch, history[-1].epoch + 1)
            best_loss = float(
                resume_state.get(
                    "best_loss",
                    checkpoint.get("metrics", {}).get("validation_loss", float("inf")),
                )
            )
            best_epoch = int(resume_state.get("best_epoch", checkpoint["epoch"]))
            stale_epochs = int(resume_state.get("stale_epochs", 0))
            if progress:
                print(
                    f"[resume] {destination} from {checkpoint_path.name} "
                    f"at epoch {start_epoch}",
                    flush=True,
                )
        elif destination_has_content:
            raise FileExistsError(
                f"Cannot resume {destination}: no last_checkpoint.pt or "
                "best_checkpoint.pt was found."
            )

    if start_epoch > config.training.epochs:
        result = _training_result(
            fold_id=fold_id,
            device=torch_device,
            train_samples=len(train_dataset),
            validation_samples=len(validation_dataset),
            checkpoint="best_checkpoint.pt",
            best_epoch=best_epoch,
            best_loss=best_loss,
            history=history,
        )
        _write_json(destination / "training_log.json", result)
        if progress:
            print(
                f"[complete] {destination} already reached requested epochs", flush=True
            )
        return result

    if progress:
        print(
            f"[train] fold={fold_id} device={torch_device} "
            f"train_samples={len(train_dataset)} "
            f"validation_samples={len(validation_dataset)} "
            f"epochs={start_epoch}-{config.training.epochs}",
            flush=True,
        )

    for epoch in range(start_epoch, config.training.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=torch_device,
            optimizer=optimizer,
            phase="train",
            epoch=epoch,
            total_epochs=config.training.epochs,
            progress=progress,
            progress_every_batches=progress_every_batches,
        )
        with torch.no_grad():
            validation_metrics = _run_epoch(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=torch_device,
                optimizer=None,
                phase="validation",
                epoch=epoch,
                total_epochs=config.training.epochs,
                progress=progress,
                progress_every_batches=progress_every_batches,
            )

        improved = validation_metrics["loss"] < best_loss
        if improved:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        epoch_record = EpochMetrics(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            train_cross_entropy=train_metrics["cross_entropy"],
            train_smoothing=train_metrics["smoothing"],
            train_frame_accuracy=train_metrics["frame_accuracy"],
            train_macro_f1=train_metrics["macro_f1"],
            train_left_f1=train_metrics["left_f1"],
            train_right_f1=train_metrics["right_f1"],
            validation_loss=validation_metrics["loss"],
            validation_cross_entropy=validation_metrics["cross_entropy"],
            validation_smoothing=validation_metrics["smoothing"],
            validation_frame_accuracy=validation_metrics["frame_accuracy"],
            validation_macro_f1=validation_metrics["macro_f1"],
            validation_left_f1=validation_metrics["left_f1"],
            validation_right_f1=validation_metrics["right_f1"],
            improved=improved,
        )
        history.append(epoch_record)

        result = _training_result(
            fold_id=fold_id,
            device=torch_device,
            train_samples=len(train_dataset),
            validation_samples=len(validation_dataset),
            checkpoint="best_checkpoint.pt",
            best_epoch=best_epoch,
            best_loss=best_loss,
            history=history,
        )
        _write_json(destination / "training_log.json", result)

        payload = _checkpoint_for_epoch(
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            fold_id=fold_id,
            validation_loss=validation_metrics["loss"],
            best_loss=best_loss,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            history=history,
            dataset_manifest_sha256=file_sha256(dataset_root / "manifest.json"),
            split_manifest_sha256=file_sha256(split_path),
            normalization_manifest_sha256=file_sha256(normalization_path),
            normalization_config=normalization_config,
            source_code_sha256=code_sha256,
            data_loader_generator=data_loader_generator,
        )
        save_checkpoint(destination / "last_checkpoint.pt", payload)
        if improved:
            save_checkpoint(destination / "best_checkpoint.pt", payload)

        if progress:
            marker = " *best*" if improved else ""
            print(
                f"[epoch {epoch:03d}/{config.training.epochs:03d}] done | "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={validation_metrics['loss']:.6f} "
                f"val_macro_f1={validation_metrics['macro_f1']:.6f}"
                f"{marker}",
                flush=True,
            )
        if stale_epochs >= config.training.patience:
            if progress:
                print(
                    f"[early-stop] no validation-loss improvement for "
                    f"{stale_epochs} epochs",
                    flush=True,
                )
            break

    result = _training_result(
        fold_id=fold_id,
        device=torch_device,
        train_samples=len(train_dataset),
        validation_samples=len(validation_dataset),
        checkpoint="best_checkpoint.pt",
        best_epoch=best_epoch,
        best_loss=best_loss,
        history=history,
    )
    _write_json(destination / "training_log.json", result)
    return result


def _checkpoint_for_epoch(
    *,
    model: nn.Module,
    optimizer: Adam,
    config: AppConfig,
    epoch: int,
    fold_id: str,
    validation_loss: float,
    best_loss: float,
    best_epoch: int,
    stale_epochs: int,
    history: list[EpochMetrics],
    dataset_manifest_sha256: str,
    split_manifest_sha256: str,
    normalization_manifest_sha256: str,
    normalization_config: NormalizationConfig | None,
    source_code_sha256: str,
    data_loader_generator: torch.Generator,
) -> dict[str, Any]:
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=epoch,
        fold_id=fold_id,
        metrics={"validation_loss": validation_loss, "best_validation_loss": best_loss},
        dataset_manifest_sha256=dataset_manifest_sha256,
        split_manifest_sha256=split_manifest_sha256,
        normalization_manifest_sha256=normalization_manifest_sha256,
        normalization_config=normalization_config,
        source_code_sha256=source_code_sha256,
    )
    payload["resume_state"] = {
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": [asdict(item) for item in history],
        "random_state": _capture_random_state(data_loader_generator),
    }
    return payload


def _resume_checkpoint_path(destination: Path) -> Path | None:
    last = destination / "last_checkpoint.pt"
    if last.is_file():
        return last
    best = destination / "best_checkpoint.pt"
    if best.is_file():
        return best
    return None


def _history_from_resume_state_or_log(
    resume_state: Any,
    training_log_path: Path,
) -> list[EpochMetrics]:
    raw_history = []
    if isinstance(resume_state, dict) and isinstance(resume_state.get("history"), list):
        raw_history = resume_state["history"]
    elif training_log_path.is_file():
        with training_log_path.open("r", encoding="utf-8") as stream:
            log_payload = json.load(stream)
        if isinstance(log_payload, dict) and isinstance(
            log_payload.get("history"), list
        ):
            raw_history = log_payload["history"]
    return [EpochMetrics(**item) for item in raw_history]


def _move_optimizer_state(optimizer: Adam, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    *,
    config: AppConfig,
    fold_id: str,
    dataset_manifest_sha256: str,
    split_manifest_sha256: str,
    normalization_manifest_sha256: str,
    normalization_config: NormalizationConfig | None,
    source_code_sha256: str,
) -> None:
    if str(checkpoint["fold_id"]) != fold_id:
        raise ValueError("Resume checkpoint belongs to a different fold.")
    if checkpoint["config"] != config_to_dict(config):
        raise ValueError("Resume checkpoint configuration does not match this run.")
    expected = {
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "normalization_manifest_sha256": normalization_manifest_sha256,
        "normalization_runtime": normalization_config_to_dict(
            normalization_config
        ),
        "source_code_sha256": source_code_sha256,
    }
    provenance = checkpoint["provenance"]
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Resume checkpoint provenance mismatch: {mismatches}")


def _capture_random_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "data_loader": generator.get_state(),
    }


def _restore_random_state(
    payload: Any,
    generator: torch.Generator,
) -> None:
    if not isinstance(payload, dict):
        return
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and payload.get("torch_cuda"):
        torch.cuda.set_rng_state_all(payload["torch_cuda"])
    generator.set_state(payload["data_loader"])


def _training_result(
    *,
    fold_id: str,
    device: torch.device,
    train_samples: int,
    validation_samples: int,
    checkpoint: str,
    best_epoch: int,
    best_loss: float,
    history: list[EpochMetrics],
) -> dict[str, Any]:
    return {
        "fold_id": fold_id,
        "device": str(device),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "checkpoint": checkpoint,
        "history": [asdict(item) for item in history],
    }
