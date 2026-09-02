"""Dataset-only vector figures that do not use model predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bee_counter_bench.data.schemas import Direction, PreprocessedSequence
from bee_counter_bench.data.sequence_io import load_preprocessed_sequence


def write_dataset_figures(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write the C2 position profile and complete event-speed distribution."""
    dataset_root = Path(dataset_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite dataset figures: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    sequences = _load_sequences(dataset_root)
    profile_report = _write_roi_profile(
        [sequence for collection, sequence in sequences if collection == "c2"],
        destination / "roi_profile_c2.svg",
        destination / "roi_profile_c2_statistics.npz",
    )
    speed_report = _write_speed_distribution(
        [sequence for _, sequence in sequences],
        destination / "event_speed_distribution.svg",
    )
    report = {
        "schema_version": 1,
        "roi_profile": profile_report,
        "speed_distribution": speed_report,
    }
    _write_json(destination / "dataset_figures.json", report)
    return report


def _load_sequences(
    dataset_root: Path,
) -> list[tuple[str, PreprocessedSequence]]:
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    return [
        (
            str(record["collection_id"]),
            load_preprocessed_sequence(dataset_root / str(record["output"])),
        )
        for record in manifest["sequences"]
    ]


def _write_roi_profile(
    sequences: list[PreprocessedSequence],
    svg_path: Path,
    statistics_path: Path,
) -> dict[str, Any]:
    grid = np.arange(16.0, 96.0 + 0.125, 0.25, dtype=np.float64)
    right_matrix = _passage_matrix(
        sequences, grid, direction=Direction.RIGHT, margin_frames=45
    )
    left_matrix = _passage_matrix(
        sequences, grid, direction=Direction.LEFT, margin_frames=45
    )
    right = _profile_statistics(right_matrix)
    left = _profile_statistics(left_matrix)
    np.savez_compressed(
        statistics_path,
        position_mm=grid,
        right_mean=right["mean"],
        right_standard_deviation=right["standard_deviation"],
        right_median=right["median"],
        right_p25=right["p25"],
        right_p75=right["p75"],
        right_count=right["count"],
        left_mean=left["mean"],
        left_standard_deviation=left["standard_deviation"],
        left_median=left["median"],
        left_p25=left["p25"],
        left_p75=left["p75"],
        left_count=left["count"],
    )
    _plot_roi_profile(
        grid,
        right,
        left,
        right_matrix.shape[0],
        left_matrix.shape[0],
        svg_path,
    )
    return {
        "collection": "c2",
        "sessions": len(sequences),
        "right_passages": int(right_matrix.shape[0]),
        "left_passages": int(left_matrix.shape[0]),
        "grid_mm": {"minimum": 16.0, "maximum": 96.0, "step": 0.25},
        "roi_mm": [36.0, 76.0],
        "margin_frames": 45,
        "interpretation": "post_hoc_physical_support_not_roi_selection",
    }


def _passage_matrix(
    sequences: list[PreprocessedSequence],
    grid: np.ndarray,
    *,
    direction: Direction,
    margin_frames: int,
) -> np.ndarray:
    rows = []
    for sequence in sequences:
        for event in sequence.events:
            if event.direction != direction:
                continue
            start = max(0, event.start_idx - margin_frames)
            end = min(len(sequence.signal), event.end_idx + margin_frames + 1)
            position = np.asarray(sequence.position_mm[start:end], dtype=np.float64)
            signal = np.asarray(sequence.signal[start:end], dtype=np.float64)
            order = np.argsort(position)
            unique_position, indices = np.unique(position[order], return_index=True)
            unique_signal = signal[order][indices]
            if len(unique_position) < 4:
                continue
            inside = (grid >= unique_position[0]) & (grid <= unique_position[-1])
            if np.count_nonzero(inside) < 10:
                continue
            row = np.full(grid.shape, np.nan, dtype=np.float64)
            row[inside] = np.interp(grid[inside], unique_position, unique_signal)
            rows.append(row)
    if not rows:
        raise ValueError(f"No {direction.name} passages available for ROI profile.")
    return np.vstack(rows)


def _profile_statistics(matrix: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": np.nanmean(matrix, axis=0),
        "standard_deviation": np.nanstd(matrix, axis=0),
        "median": np.nanmedian(matrix, axis=0),
        "p25": np.nanpercentile(matrix, 25.0, axis=0),
        "p75": np.nanpercentile(matrix, 75.0, axis=0),
        "count": np.sum(np.isfinite(matrix), axis=0),
    }


def _plot_roi_profile(
    grid: np.ndarray,
    right: dict[str, np.ndarray],
    left: dict[str, np.ndarray],
    right_passages: int,
    left_passages: int,
    output_path: Path,
) -> None:
    plt = _pyplot()
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    panels = (
        (axes[0], right, "RIGHT", "#16697A", right_passages),
        (axes[1], left, "LEFT", "#D1495B", left_passages),
    )
    for axis, statistics, direction, color, count in panels:
        average = statistics["mean"]
        deviation = statistics["standard_deviation"]
        axis.fill_between(
            grid,
            average - deviation,
            average + deviation,
            color=color,
            alpha=0.18,
            linewidth=0,
            label="Mean +/- 1 SD",
        )
        axis.fill_between(
            grid,
            statistics["p25"],
            statistics["p75"],
            color=color,
            alpha=0.12,
            linewidth=0,
            label="IQR",
        )
        axis.plot(grid, average, color=color, linewidth=1.5, label="Mean")
        axis.plot(
            grid,
            statistics["median"],
            color="#202020",
            linewidth=0.9,
            linestyle="--",
            label="Median",
        )
        for boundary in (36.0, 76.0):
            axis.axvline(boundary, color="#202020", linewidth=0.9, linestyle=":")
        axis.axhline(0.0, color="#777777", linewidth=0.6)
        axis.set_ylabel("Filtered signal (V)")
        axis.set_title(f"{direction}, n={count}")
        axis.grid(True, color="#DDDDDD", linewidth=0.6)
        axis.legend(loc="upper right", ncol=4, frameon=False)
    axes[-1].set_xlabel("CNC position (mm)")
    figure.tight_layout()
    figure.savefig(output_path, format="svg")
    plt.close(figure)


def _write_speed_distribution(
    sequences: list[PreprocessedSequence],
    output_path: Path,
) -> dict[str, Any]:
    speeds = np.asarray(
        [
            event.mean_speed_mm_min
            for sequence in sequences
            for event in sequence.events
        ],
        dtype=np.float64,
    )
    counts = {
        "below_1500": int(np.count_nonzero(speeds < 1500.0)),
        "1500_to_below_2500": int(
            np.count_nonzero((speeds >= 1500.0) & (speeds < 2500.0))
        ),
        "2500_and_above": int(np.count_nonzero(speeds >= 2500.0)),
    }
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.2, 3.7))
    bins = np.arange(400.0, 3600.0 + 100.0, 100.0)
    axis.hist(speeds, bins=bins, color="#16697A", edgecolor="white", linewidth=0.4)
    for boundary in (1500.0, 2500.0):
        axis.axvline(boundary, color="#D1495B", linewidth=1.2, linestyle="--")
    axis.set_xlabel("Mean event speed (mm/min)")
    axis.set_ylabel("Events")
    axis.grid(True, axis="y", color="#DDDDDD", linewidth=0.6)
    figure.tight_layout()
    figure.savefig(output_path, format="svg")
    plt.close(figure)
    return {
        "events": int(len(speeds)),
        "minimum_mm_min": float(speeds.min()),
        "maximum_mm_min": float(speeds.max()),
        "median_mm_min": float(np.median(speeds)),
        "bins": counts,
        "bin_rule": "three_equal_width_intervals_of_programmed_500_to_3500_mm_min_envelope",
    }


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)
