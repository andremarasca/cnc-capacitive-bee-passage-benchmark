"""Reproducible raw and preprocessed temporal passage examples."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bee_counter_bench.data.schemas import PreprocessedSequence
from bee_counter_bench.data.sequence_io import file_sha256, load_preprocessed_sequence


@dataclass(frozen=True)
class ExampleSelection:
    panel: str
    direction: str
    collection_id: str
    session_id: str
    event_index: int
    relative_raw_path: str


@dataclass(frozen=True)
class TemporalExample:
    selection: ExampleSelection
    event_start_frame: int
    event_end_frame: int
    mean_speed_mm_min: float
    raw_time_s: np.ndarray
    raw_signal: np.ndarray
    raw_position_mm: np.ndarray
    processed_time_s: np.ndarray
    processed_signal: np.ndarray
    processed_position_mm: np.ndarray
    event_start_s: float
    event_end_s: float
    raw_sha256: str


def write_temporal_examples(
    *,
    dataset_dir: str | Path,
    raw_manifest_path: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
    context_s: float = 1.2,
) -> dict[str, Any]:
    """Generate the article temporal examples from traced raw inputs."""
    if context_s <= 0.0:
        raise ValueError("context_s must be positive.")
    dataset_root = Path(dataset_dir).resolve()
    raw_manifest_file = Path(raw_manifest_path).resolve()
    selection_file = Path(selection_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite temporal examples: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    dataset_manifest = _read_json(dataset_root / "manifest.json")
    raw_manifest = _read_json(raw_manifest_file, encoding="utf-8-sig")
    selections = _read_selections(selection_file)
    sequences = _load_sequences(dataset_root, dataset_manifest)
    raw_files = {
        str(record["relative_path"]).replace("\\", "/"): record
        for record in raw_manifest["files"]
    }
    project_root = raw_manifest_file.parents[2]
    examples = [
        _load_example(
            selection,
            sequence=sequences[selection.session_id],
            raw_files=raw_files,
            project_root=project_root,
            context_s=context_s,
        )
        for selection in selections
    ]
    _plot_examples(
        examples,
        destination / "temporal_passage_examples.svg",
        destination / "temporal_passage_examples.pdf",
        destination / "temporal_passage_examples.png",
    )
    report = {
        "schema_version": 1,
        "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
        "raw_manifest_sha256": file_sha256(raw_manifest_file),
        "selection_sha256": file_sha256(selection_file),
        "selection_path": selection_file.relative_to(project_root).as_posix(),
        "context_s": context_s,
        "raw_display_downsampling": "bin_mean_for_vector_rendering_only",
        "examples": [
            {
                "panel": example.selection.panel,
                "direction": example.selection.direction,
                "collection_id": example.selection.collection_id,
                "session_id": example.selection.session_id,
                "event_index": example.selection.event_index,
                "event_start_frame": example.event_start_frame,
                "event_end_frame": example.event_end_frame,
                "mean_speed_mm_min": example.mean_speed_mm_min,
                "relative_raw_path": example.selection.relative_raw_path,
                "raw_sha256": example.raw_sha256,
            }
            for example in examples
        ],
    }
    _write_json(destination / "temporal_passage_examples.json", report)
    return report


def _load_example(
    selection: ExampleSelection,
    *,
    sequence: PreprocessedSequence,
    raw_files: dict[str, dict[str, Any]],
    project_root: Path,
    context_s: float,
) -> TemporalExample:
    event = sequence.events[selection.event_index]
    if event.direction.name != selection.direction:
        raise ValueError(
            f"Direction mismatch for {selection.session_id} event "
            f"{selection.event_index}: {event.direction.name}"
        )
    raw_record = raw_files.get(selection.relative_raw_path)
    if raw_record is None:
        raise KeyError(
            f"Raw selection is absent from manifest: {selection.relative_raw_path}"
        )
    if str(raw_record["session_id"]) != selection.session_id:
        raise ValueError("Selected raw chunk belongs to a different session.")
    raw_path = project_root / selection.relative_raw_path
    if file_sha256(raw_path) != str(raw_record["sha256"]):
        raise ValueError(f"Raw SHA-256 mismatch: {selection.relative_raw_path}")

    raw_rate_hz = 100_000.0
    processed_rate_hz = 30.0
    raw_start_global = int(
        (event.start_idx / processed_rate_hz - context_s) * raw_rate_hz
    )
    raw_end_global = int(
        ((event.end_idx + 1) / processed_rate_hz + context_s) * raw_rate_hz
    )
    chunk_offset = sum(
        int(record["input_samples"])
        for record in raw_files.values()
        if str(record["session_id"]) == selection.session_id
        and int(record["chunk_index"]) < int(raw_record["chunk_index"])
    )
    local_start = raw_start_global - chunk_offset
    local_end = raw_end_global - chunk_offset
    with np.load(raw_path, allow_pickle=False) as payload:
        sample_rate = float(payload["sample_rate"])
        if sample_rate != raw_rate_hz:
            raise ValueError(f"Unexpected raw sample rate: {sample_rate}")
        raw_signal = np.asarray(payload["raw_signal"][local_start:local_end])
        raw_position = np.asarray(payload["position_mm"][local_start:local_end])
    if len(raw_signal) != raw_end_global - raw_start_global:
        raise ValueError("Selected temporal window crosses a raw chunk boundary.")

    processed_margin = int(round(context_s * processed_rate_hz))
    processed_start = max(0, event.start_idx - processed_margin)
    processed_end = min(len(sequence.signal), event.end_idx + processed_margin + 1)
    event_center_s = (event.start_idx + event.end_idx + 1) / 2.0 / processed_rate_hz
    raw_time = (
        np.arange(raw_start_global, raw_end_global) / raw_rate_hz - event_center_s
    )
    processed_indices = np.arange(processed_start, processed_end)
    processed_time = processed_indices / processed_rate_hz - event_center_s
    raw_time, raw_signal, raw_position = _bin_mean(
        raw_time,
        raw_signal,
        raw_position,
        maximum_points=1200,
    )
    return TemporalExample(
        selection=selection,
        event_start_frame=event.start_idx,
        event_end_frame=event.end_idx,
        mean_speed_mm_min=float(event.mean_speed_mm_min),
        raw_time_s=raw_time,
        raw_signal=raw_signal,
        raw_position_mm=raw_position,
        processed_time_s=processed_time,
        processed_signal=np.asarray(sequence.signal[processed_start:processed_end]),
        processed_position_mm=np.asarray(
            sequence.position_mm[processed_start:processed_end]
        ),
        event_start_s=event.start_idx / processed_rate_hz - event_center_s,
        event_end_s=(event.end_idx + 1) / processed_rate_hz - event_center_s,
        raw_sha256=str(raw_record["sha256"]),
    )


def _bin_mean(
    time_s: np.ndarray,
    signal: np.ndarray,
    position_mm: np.ndarray,
    *,
    maximum_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_count = min(maximum_points, len(signal))
    edges = np.linspace(0, len(signal), point_count + 1, dtype=np.int64)
    valid = edges[1:] > edges[:-1]

    def means(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                values[start:end].mean()
                for start, end in zip(edges[:-1][valid], edges[1:][valid], strict=True)
            ],
            dtype=np.float64,
        )

    return means(time_s), means(signal), means(position_mm)


def _plot_examples(
    examples: list[TemporalExample],
    svg_path: Path,
    pdf_path: Path,
    png_path: Path,
) -> None:
    if len(examples) != 2:
        raise ValueError("The article layout requires exactly two temporal examples.")
    plt = _pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex="col")
    panel_labels = (("A", "B"), ("C", "D"))
    for column, example in enumerate(examples):
        raw_axis = axes[0, column]
        processed_axis = axes[1, column]
        _plot_signal_and_position(
            raw_axis,
            example.raw_time_s,
            example.raw_signal,
            example.raw_position_mm,
            event_start_s=example.event_start_s,
            event_end_s=example.event_end_s,
            signal_label="Raw signal (V)",
        )
        _plot_signal_and_position(
            processed_axis,
            example.processed_time_s,
            example.processed_signal,
            example.processed_position_mm,
            event_start_s=example.event_start_s,
            event_end_s=example.event_end_s,
            signal_label="Filtered signal (V)",
        )
        raw_axis.set_title(
            f"{example.selection.direction}, {example.mean_speed_mm_min:.0f} mm/min"
        )
        processed_axis.set_xlabel("Time relative to event center (s)")
        for row, axis in enumerate((raw_axis, processed_axis)):
            axis.text(
                0.02,
                0.94,
                panel_labels[row][column],
                transform=axis.transAxes,
                va="top",
                fontweight="bold",
            )
    figure.tight_layout()
    figure.savefig(svg_path, format="svg")
    _strip_trailing_whitespace(svg_path)
    figure.savefig(pdf_path, format="pdf")
    figure.savefig(png_path, format="png", dpi=200)
    plt.close(figure)


def _strip_trailing_whitespace(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    normalized = "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
    path.write_text(normalized, encoding="utf-8", newline="\n")


def _plot_signal_and_position(
    axis: Any,
    time_s: np.ndarray,
    signal: np.ndarray,
    position_mm: np.ndarray,
    *,
    event_start_s: float,
    event_end_s: float,
    signal_label: str,
) -> None:
    axis.axvspan(event_start_s, event_end_s, color="#E9C46A", alpha=0.28)
    axis.plot(time_s, signal, color="#202020", linewidth=0.8)
    axis.set_ylabel(signal_label)
    axis.grid(True, color="#DDDDDD", linewidth=0.5)
    position_axis = axis.twinx()
    position_axis.plot(time_s, position_mm, color="#2A6FBB", linewidth=0.9)
    position_axis.set_ylabel("CNC position (mm)", color="#2A6FBB")
    position_axis.tick_params(axis="y", colors="#2A6FBB")


def _load_sequences(
    dataset_root: Path,
    manifest: dict[str, Any],
) -> dict[str, PreprocessedSequence]:
    return {
        str(record["session_id"]): load_preprocessed_sequence(
            dataset_root / str(record["output"])
        )
        for record in manifest["sequences"]
    }


def _read_selections(path: Path) -> list[ExampleSelection]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [
            ExampleSelection(
                panel=str(row["panel"]),
                direction=str(row["direction"]),
                collection_id=str(row["collection_id"]),
                session_id=str(row["session_id"]),
                event_index=int(row["event_index"]),
                relative_raw_path=str(row["relative_raw_path"]),
            )
            for row in csv.DictReader(stream)
        ]


def _read_json(path: Path, *, encoding: str = "utf-8") -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding=encoding))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
