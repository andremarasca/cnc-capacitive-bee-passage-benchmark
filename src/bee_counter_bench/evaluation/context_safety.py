"""Audit neural event isolation and residual preprocessing memory."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import sosfilt

from bee_counter_bench.config import (
    load_config,
    load_normalization_config,
    normalization_config_to_dict,
)
from bee_counter_bench.data.preprocessing import design_bessel_sos
from bee_counter_bench.data.sequence_io import file_sha256
from bee_counter_bench.models.receptive_field import mstcnpp_receptive_field
from bee_counter_bench.provenance import source_code_sha256
from bee_counter_bench.training.article_experiment import ArticleExperimentPlan


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite context audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def build_context_safety_audit(plan: ArticleExperimentPlan) -> dict[str, Any]:
    """Build evidence that event frames cannot access labeled neighbors."""
    root = plan.project_root
    dataset_root = root / "outputs" / "processed" / "dataset"
    dataset_manifest_path = dataset_root / "manifest.json"
    manifest = _read_json(dataset_manifest_path)
    config = load_config(
        root / "configs" / "experiment.yaml",
        model_path=plan.model_path,
    )
    receptive_field = mstcnpp_receptive_field(
        config.model.prediction_generation_layers,
        config.model.refinement_layers,
        config.model.refinement_stages,
        float(config.preprocessing.output_sample_rate_hz),
        causal=config.model.causal,
    )

    gaps: list[int] = []
    durations: list[int] = []
    event_peak_to_peak: list[float] = []
    safe_out_samples: list[float] = []
    transitions: Counter[str] = Counter()
    violations: list[dict[str, Any]] = []
    first_event_padding = 0
    last_event_padding = 0

    for record in manifest["sequences"]:
        path = dataset_root / str(record["output"])
        with np.load(path, allow_pickle=False) as sequence:
            starts = sequence["event_start_idx"].astype(np.int64)
            ends = sequence["event_end_idx"].astype(np.int64)
            directions = sequence["event_direction"].astype(np.int64)
            signal = sequence["signal"].astype(np.float64)
            labels = sequence["labels"].astype(np.int64)
        if not (len(starts) == len(ends) == len(directions)):
            raise ValueError(f"Event arrays differ in {record['session_id']}.")
        if len(starts) and (
            int(starts[0]) < receptive_field.past_frames
        ):
            first_event_padding += 1
        if len(ends) and (
            len(signal) - int(ends[-1]) - 1 < receptive_field.future_frames
        ):
            last_event_padding += 1
        safe_out = labels == 0
        for start, end in zip(starts, ends, strict=True):
            start_index = int(start)
            end_index = int(end)
            durations.append(end_index - start_index + 1)
            event_peak_to_peak.append(
                float(np.ptp(signal[start_index : end_index + 1]))
            )
            safe_out[
                max(0, start_index - receptive_field.past_frames) : min(
                    len(safe_out),
                    end_index + receptive_field.future_frames + 1,
                )
            ] = False
        safe_out_samples.extend(signal[safe_out][::30].tolist())
        for index in range(max(0, len(starts) - 1)):
            gap = int(starts[index + 1] - ends[index] - 1)
            gaps.append(gap)
            transition = f"{int(directions[index])}->{int(directions[index + 1])}"
            transitions[transition] += 1
            if gap < receptive_field.future_frames:
                violations.append(
                    {
                        "session_id": str(record["session_id"]),
                        "event_index": index,
                        "side": "future",
                        "gap_frames": gap,
                    }
                )
            if gap < receptive_field.past_frames:
                violations.append(
                    {
                        "session_id": str(record["session_id"]),
                        "event_index": index + 1,
                        "side": "past",
                        "gap_frames": gap,
                    }
                )

    if not gaps or not durations:
        raise ValueError("Dataset contains no auditable event transitions.")
    gap_array = np.asarray(gaps, dtype=np.int64)
    duration_array = np.asarray(durations, dtype=np.int64)
    event_amplitudes = np.asarray(event_peak_to_peak, dtype=np.float64)
    out_array = np.asarray(safe_out_samples, dtype=np.float64)
    out_center = float(np.median(out_array))
    robust_out_standard_deviation = float(
        1.4826 * np.median(np.abs(out_array - out_center))
    )
    minimum_gap = int(gap_array.min())
    normalization_memory = []
    for normalization in plan.normalizations:
        runtime = (
            None
            if normalization.config_path is None
            else load_normalization_config(normalization.config_path)
        )
        normalization_memory.append(
            {
                "runtime": normalization_config_to_dict(runtime),
                "strict_finite_memory": runtime is None,
                "state_scope": (
                    "training_sessions_only_fixed_statistics"
                    if runtime is None
                    else runtime.state_scope
                ),
                "interpretation": (
                    "Fixed training-population statistics add no temporal "
                    "state during inference."
                    if runtime is None
                    else "The causal adaptive normalizer retains decaying "
                    "past-session state and does not establish complete "
                    "event independence."
                ),
            }
        )

    report = {
        "schema_version": 1,
        "experiment_id": plan.experiment_id,
        "source_code_sha256": source_code_sha256(root),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "model_configuration": {
            "path": plan.model_path.relative_to(root).as_posix(),
            "sha256": file_sha256(plan.model_path),
            "trainable_parameters": config.model.expected_trainable_parameters,
        },
        "receptive_field": {
            "total_frames": receptive_field.total_frames,
            "past_frames": receptive_field.past_frames,
            "future_frames": receptive_field.future_frames,
            "total_seconds": receptive_field.total_seconds,
            "past_seconds": receptive_field.past_seconds,
            "future_seconds": receptive_field.future_seconds,
        },
        "event_isolation": {
            "sessions": len(manifest["sequences"]),
            "events": len(durations),
            "adjacent_pairs": len(gaps),
            "minimum_out_gap_frames": minimum_gap,
            "minimum_out_gap_seconds": minimum_gap
            / config.preprocessing.output_sample_rate_hz,
            "minimum_margin_frames": minimum_gap
            - max(receptive_field.past_frames, receptive_field.future_frames),
            "minimum_margin_seconds": (
                minimum_gap
                - max(
                    receptive_field.past_frames,
                    receptive_field.future_frames,
                )
            )
            / config.preprocessing.output_sample_rate_hz,
            "event_frame_neighbor_violations": len(violations),
            "violations": violations,
            "strict_neural_event_isolation": not violations,
            "gap_frames_quantiles": _quantiles(gap_array),
            "event_duration_frames_quantiles": _quantiles(duration_array),
            "events_longer_than_receptive_field": int(
                np.count_nonzero(
                    duration_array > receptive_field.total_frames
                )
            ),
            "sessions_with_first_event_padding": first_event_padding,
            "sessions_with_last_event_padding": last_event_padding,
        },
        "direction_sequence": {
            "transitions": dict(sorted(transitions.items())),
            "alternating_transitions": sum(
                count
                for transition, count in transitions.items()
                if transition in {"1->2", "2->1"}
            ),
            "same_direction_transitions": sum(
                count
                for transition, count in transitions.items()
                if transition in {"1->1", "2->2"}
            ),
        },
        "preprocessing_memory": _filter_memory_report(
            config=config,
            minimum_gap_frames=minimum_gap,
            event_duration_frames=duration_array,
            median_event_peak_to_peak=float(np.median(event_amplitudes)),
            robust_out_standard_deviation=robust_out_standard_deviation,
        ),
        "normalization_memory": normalization_memory,
        "claim_boundary": (
            "No labeled neighboring-event interval is contained in the "
            "nominal neural receptive field of a frame from the focal event. "
            "This does not claim complete statistical independence because "
            "the causal Bessel IIR and, when selected, the causal adaptive "
            "normalizer retain decaying past state."
        ),
    }
    if violations:
        raise ValueError(
            "Configured receptive field reaches at least one neighboring event."
        )
    return report


def write_context_safety_audit(plan: ArticleExperimentPlan) -> dict[str, Any]:
    """Write the immutable context-safety gate for an article campaign."""
    report = build_context_safety_audit(plan)
    destination = plan.output_root / "audits" / "context_safety.json"
    _write_json(destination, report)
    return report


def _quantiles(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, (0.0, 0.5, 0.9, 0.95, 0.99, 1.0))
    return {
        name: float(value)
        for name, value in zip(
            ("minimum", "median", "p90", "p95", "p99", "maximum"),
            quantiles,
            strict=True,
        )
    }


def _filter_memory_report(
    *,
    config: Any,
    minimum_gap_frames: int,
    event_duration_frames: np.ndarray,
    median_event_peak_to_peak: float,
    robust_out_standard_deviation: float,
) -> dict[str, Any]:
    source_rate = config.preprocessing.source_sample_rate_hz
    output_rate = config.preprocessing.output_sample_rate_hz
    sos = design_bessel_sos(
        config.preprocessing,
        source_sample_rate_hz=source_rate,
    )
    probe_durations = {
        "median": int(np.median(event_duration_frames)),
        "p95": int(np.quantile(event_duration_frames, 0.95)),
        "maximum": int(event_duration_frames.max()),
    }
    probes = {}
    for name, duration_frames in probe_durations.items():
        duration_seconds = duration_frames / output_rate
        probe_samples = max(1, int(round(duration_seconds * source_rate)))
        total_frames = duration_frames + minimum_gap_frames + 900
        signal = np.zeros(
            int(np.ceil(total_frames * source_rate / output_rate)),
            dtype=np.float64,
        )
        phase = np.arange(probe_samples, dtype=np.float64) / probe_samples
        signal[:probe_samples] = np.sin(2.0 * np.pi * phase)
        filtered = sosfilt(sos, signal)
        frame_count = len(signal) * output_rate // source_rate
        edges = np.floor(
            np.arange(frame_count + 1, dtype=np.float64)
            * source_rate
            / output_rate
        ).astype(np.int64)
        downsampled = np.asarray(
            [
                filtered[edges[index] : edges[index + 1]].mean()
                for index in range(frame_count)
            ],
            dtype=np.float64,
        )
        peak = float(np.max(np.abs(downsampled)))
        gap_end_index = duration_frames + minimum_gap_frames
        tail = downsampled[gap_end_index:]
        point_ratio = (
            0.0
            if peak == 0.0 or gap_end_index >= len(downsampled)
            else float(abs(downsampled[gap_end_index]) / peak)
        )
        maximum_tail_ratio = (
            0.0
            if peak == 0.0 or not len(tail)
            else float(np.max(np.abs(tail)) / peak)
        )
        empirical_noise_ratio = (
            None
            if robust_out_standard_deviation <= 0.0
            else maximum_tail_ratio
            * median_event_peak_to_peak
            / robust_out_standard_deviation
        )
        probes[name] = {
            "duration_frames": duration_frames,
            "duration_seconds": duration_seconds,
            "point_tail_to_probe_peak_ratio": point_ratio,
            "maximum_later_tail_to_probe_peak_ratio": maximum_tail_ratio,
            "heuristic_tail_to_robust_out_noise_ratio": empirical_noise_ratio,
        }
    return {
        "filter_family": config.preprocessing.bessel.mode,
        "state_scope": config.preprocessing.state_scope,
        "strict_finite_memory": False,
        "median_event_peak_to_peak": median_event_peak_to_peak,
        "robust_out_standard_deviation": robust_out_standard_deviation,
        "probes": probes,
        "interpretation": (
            "The neural field excludes labeled neighboring-event frames. "
            "The causal IIR preprocessing retains decaying past state, so the "
            "audit does not claim complete statistical independence."
        ),
    }
