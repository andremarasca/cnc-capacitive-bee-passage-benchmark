"""Analytical frequency response for the frozen preprocessing filter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import sosfreqz

from bee_counter_bench.config import AppConfig
from bee_counter_bench.data.preprocessing import design_bessel_sos


def response_statistics(config: AppConfig) -> dict[str, Any]:
    """Calculate physical marker attenuation and the numerical -3 dB band."""
    frequencies = np.geomspace(0.01, 20.0, 200_000)
    sos = design_bessel_sos(config.preprocessing)
    _, response = sosfreqz(
        sos,
        worN=frequencies,
        fs=config.preprocessing.source_sample_rate_hz,
    )
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-300))
    marker_frequencies = {
        "500_mm_min_characteristic": 500.0 / 2400.0,
        "3500_mm_min_characteristic": 3500.0 / 2400.0,
        "3500_mm_min_second_multiple": 2.0 * 3500.0 / 2400.0,
        "3500_mm_min_third_multiple": 3.0 * 3500.0 / 2400.0,
    }
    above_minus_three = frequencies[magnitude_db >= -3.0]
    if not len(above_minus_three):
        raise ValueError("Filter response never reaches -3 dB.")
    return {
        "schema_version": 1,
        "filter": {
            "family": "bessel",
            "mode": "bandpass",
            "prototype_order": config.preprocessing.bessel.order,
            "transformed_order": 2 * config.preprocessing.bessel.order,
            "critical_frequencies_hz": [
                config.preprocessing.bessel.low_cutoff_hz,
                config.preprocessing.bessel.high_cutoff_hz,
            ],
            "norm": config.preprocessing.bessel.norm,
            "source_sample_rate_hz": config.preprocessing.source_sample_rate_hz,
        },
        "physical_scale": {
            "roi_width_mm": 40.0,
            "formula": "velocity_mm_min / (60 * roi_width_mm)",
        },
        "markers": {
            name: {
                "frequency_hz": frequency,
                "magnitude_db": float(np.interp(frequency, frequencies, magnitude_db)),
                "attenuation_db": float(
                    -np.interp(frequency, frequencies, magnitude_db)
                ),
            }
            for name, frequency in marker_frequencies.items()
        },
        "minus_3_db_band_hz": [
            float(above_minus_three[0]),
            float(above_minus_three[-1]),
        ],
        "peak": {
            "frequency_hz": float(frequencies[int(np.argmax(magnitude_db))]),
            "magnitude_db": float(np.max(magnitude_db)),
        },
    }


def write_filter_response(
    *,
    config: AppConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write an audit JSON and publication-ready SVG response plot."""
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite filter response: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    report = response_statistics(config)
    _write_json(destination / "filter_response.json", report)
    _write_svg(config, report, destination / "filter_response.svg")
    return report


def _write_svg(config: AppConfig, report: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frequencies = np.geomspace(0.03, 20.0, 20_000)
    sos = design_bessel_sos(config.preprocessing)
    _, response = sosfreqz(
        sos,
        worN=frequencies,
        fs=config.preprocessing.source_sample_rate_hz,
    )
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-300))
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    axis.semilogx(frequencies, magnitude_db, color="#16697A", linewidth=2.0)
    axis.axhline(-3.0, color="#555555", linewidth=1.0, linestyle="--")
    for marker in report["markers"].values():
        axis.scatter(
            marker["frequency_hz"],
            marker["magnitude_db"],
            color="#D1495B",
            s=24,
            zorder=3,
        )
    axis.set_xlim(0.03, 20.0)
    axis.set_ylim(-45.0, 2.0)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Magnitude (dB)")
    axis.grid(True, which="both", color="#D9D9D9", linewidth=0.7)
    figure.savefig(path, format="svg")
    plt.close(figure)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)
