"""Causal signal preprocessing shared by offline data and streaming audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import bessel, sosfilt, sosfilt_zi

from bee_counter_bench.config import PreprocessingConfig

from .schemas import PreprocessingMetadata


@dataclass(frozen=True)
class PreprocessingResult:
    """Aligned preprocessed signal, CNC position, and provenance."""

    signal: NDArray[np.float32]
    position_mm: NDArray[np.float32]
    metadata: PreprocessingMetadata


class StreamingBlockMean:
    """Streaming time-bin mean equivalent to resampling a concatenated session."""

    def __init__(self, source_sample_rate_hz: int, output_sample_rate_hz: int) -> None:
        if not 0 < output_sample_rate_hz < source_sample_rate_hz:
            raise ValueError(
                "Streaming resampler requires 0 < output rate < source rate."
            )
        self.source_rate = int(source_sample_rate_hz)
        self.output_rate = int(output_sample_rate_hz)
        self._buffer = np.empty(0, dtype=np.float64)
        self._buffer_start = 0
        self._total_samples = 0
        self._next_output_index = 0

    def process(self, values: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError("Streaming resampler input must be one-dimensional.")
        if len(array):
            self._buffer = np.concatenate((self._buffer, array))
            self._total_samples += len(array)
        available_outputs = self._total_samples * self.output_rate // self.source_rate
        count = available_outputs - self._next_output_index
        if count <= 0:
            return np.empty(0, dtype=np.float64)
        indices = self._next_output_index + np.arange(count + 1, dtype=np.int64)
        global_edges = np.floor(
            indices.astype(np.float64) * self.source_rate / self.output_rate
        ).astype(np.int64)
        local_edges = global_edges - self._buffer_start
        cumulative = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(self._buffer, dtype=np.float64))
        )
        widths = np.diff(local_edges)
        output = (cumulative[local_edges[1:]] - cumulative[local_edges[:-1]]) / widths
        self._next_output_index = int(available_outputs)
        consumed = int(global_edges[-1] - self._buffer_start)
        self._buffer = self._buffer[consumed:]
        self._buffer_start = int(global_edges[-1])
        return output


class StatefulSessionPreprocessor:
    """Causal filter and resampler whose states persist across raw session chunks."""

    def __init__(self, config: PreprocessingConfig, source_sample_rate_hz: int) -> None:
        self.config = config
        self.source_rate = int(source_sample_rate_hz)
        self.sos = design_bessel_sos(config, source_sample_rate_hz=self.source_rate)
        self.filter_state: NDArray[np.float64] | None = None
        self.signal_resampler = StreamingBlockMean(
            self.source_rate, config.output_sample_rate_hz
        )
        self.position_resampler = StreamingBlockMean(
            self.source_rate, config.output_sample_rate_hz
        )

    def process(
        self, raw_signal: ArrayLike, position_mm: ArrayLike
    ) -> PreprocessingResult:
        signal = np.asarray(raw_signal, dtype=np.float64)
        position = np.asarray(position_mm, dtype=np.float64)
        if signal.ndim != 1 or position.ndim != 1 or len(signal) != len(position):
            raise ValueError(
                "Session chunk signal and position must be aligned 1-D arrays."
            )
        if (
            not len(signal)
            or not np.isfinite(signal).all()
            or not np.isfinite(position).all()
        ):
            raise ValueError("Session chunk arrays must be non-empty and finite.")
        if self.filter_state is None:
            if self.config.initialization != "steady_state_first_sample":
                raise ValueError(
                    f"Unsupported initialization: {self.config.initialization}"
                )
            self.filter_state = sosfilt_zi(self.sos) * signal[0]
        filtered, self.filter_state = sosfilt(self.sos, signal, zi=self.filter_state)
        output_signal = self.signal_resampler.process(filtered)
        output_position = self.position_resampler.process(position)
        if len(output_signal) != len(output_position):
            raise AssertionError("Streaming signal and position resamplers diverged.")
        bessel_config = self.config.bessel
        metadata = PreprocessingMetadata(
            schema_version=1,
            source_sample_rate_hz=self.source_rate,
            output_sample_rate_hz=self.config.output_sample_rate_hz,
            filter_family="bessel",
            filter_mode=bessel_config.mode,
            filter_prototype_order=bessel_config.order,
            filter_transformed_order=(
                2 * bessel_config.order
                if bessel_config.mode == "bandpass"
                else bessel_config.order
            ),
            low_cutoff_hz=bessel_config.low_cutoff_hz,
            high_cutoff_hz=bessel_config.high_cutoff_hz,
            filter_norm=bessel_config.norm,
            decimation_method="streaming_time_bin_mean_v1",
            filter_initialization=self.config.initialization,
            state_scope="session",
        )
        return PreprocessingResult(
            signal=output_signal.astype(np.float32),
            position_mm=output_position.astype(np.float32),
            metadata=metadata,
        )


def design_bessel_sos(
    config: PreprocessingConfig,
    *,
    source_sample_rate_hz: int | None = None,
) -> NDArray[np.float64]:
    """Design the configured digital Bessel filter as SOS coefficients."""
    sample_rate_hz = (
        config.source_sample_rate_hz
        if source_sample_rate_hz is None
        else int(source_sample_rate_hz)
    )
    if sample_rate_hz <= 0:
        raise ValueError("source_sample_rate_hz must be positive.")
    nyquist_hz = sample_rate_hz / 2.0
    bessel_config = config.bessel
    if not 0 < bessel_config.high_cutoff_hz < nyquist_hz:
        raise ValueError("Bessel high cutoff must be between zero and Nyquist.")

    if bessel_config.mode == "bandpass":
        if bessel_config.low_cutoff_hz is None:
            raise ValueError("Band-pass mode requires a low cutoff.")
        if not 0 < bessel_config.low_cutoff_hz < bessel_config.high_cutoff_hz:
            raise ValueError("Bessel low cutoff must be below the high cutoff.")
        cutoff: float | tuple[float, float] = (
            bessel_config.low_cutoff_hz,
            bessel_config.high_cutoff_hz,
        )
        btype = "bandpass"
    elif bessel_config.mode == "lowpass":
        cutoff = bessel_config.high_cutoff_hz
        btype = "lowpass"
    else:
        raise ValueError(f"Unsupported Bessel mode: {bessel_config.mode!r}.")

    return bessel(
        bessel_config.order,
        cutoff,
        btype=btype,
        fs=sample_rate_hz,
        output="sos",
        norm=bessel_config.norm,
    ).astype(np.float64, copy=False)
