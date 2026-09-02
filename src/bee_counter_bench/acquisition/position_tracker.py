"""Independent X-axis tracker driven by hardware Step/Direction signals."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from bee_counter_bench.bench_config import SETTINGS_PATH, load_app_config
from bee_counter_bench.acquisition.daq import DaqCaptureMulti
from bee_counter_bench.acquisition.recorder import (
    RoundTripChunker,
    StreamingRisingEdgeDetector,
)


TRACKER_IDLE_SLEEP_S = 0.01
logger = logging.getLogger(__name__)
SAFE_SESSION_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


class PositionTrackerError(Exception):
    """Raised when the hardware tracker cannot provide a trustworthy position."""


@dataclass(frozen=True)
class TrackerSnapshot:
    """Thread-safe snapshot of the latest known X position."""

    position_mm: float
    total_samples: int
    total_edges: int
    live_data_seen: bool
    error: str | None
    updated_at_monotonic: float


class BenchRecordingSession:
    """Persist bench recordings while the tracker consumes the same DAQ stream."""

    def __init__(
        self,
        *,
        output_dir: Path,
        session_prefix: str = "bench_raw",
        trips_per_chunk: int,
        steps_per_mm: float,
        travel_min_mm: float,
        travel_max_mm: float,
        voltage_threshold: float,
        min_edge_gap: int,
        min_pulse_samples: int,
        dir_vote_samples: int,
        sample_rate_hz: int,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prefix = sanitize_recording_session_prefix(session_prefix)
        self.session_name = f"{safe_prefix}__{timestamp}"
        self.session_dir = output_dir / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._sample_rate_hz = sample_rate_hz
        self._chunker = RoundTripChunker(
            trips_per_chunk=trips_per_chunk,
            steps_per_mm=steps_per_mm,
            travel_min_mm=travel_min_mm,
            travel_max_mm=travel_max_mm,
            voltage_threshold=voltage_threshold,
            min_edge_gap=min_edge_gap,
            min_pulse_samples=min_pulse_samples,
            dir_vote_samples=dir_vote_samples,
        )
        self.chunk_index = 0
        self.total_samples = 0
        self.total_trips = 0

    def feed(self, raw_chunk: np.ndarray) -> None:
        """Feed one DAQ chunk and persist any completed round-trip chunks."""
        if raw_chunk.ndim != 2 or raw_chunk.shape[0] < 3:
            return
        completed = self._chunker.feed(raw_chunk[0], raw_chunk[1], raw_chunk[2])
        for signal, position in completed:
            self.chunk_index += 1
            self.total_samples += len(signal)
            self._save_chunk(signal, position, partial=False)
        self.total_trips = self._chunker.positive_reversals

    def close(self) -> None:
        """Flush the partial recording chunk when acquisition stops."""
        partial = self._chunker.flush_partial()
        if partial is None:
            return
        signal, position = partial
        if len(signal) == 0:
            return
        self.chunk_index += 1
        self.total_samples += len(signal)
        self._save_chunk(signal, position, partial=True)
        self.total_trips = self._chunker.positive_reversals

    def _save_chunk(
        self,
        signal: np.ndarray,
        position: np.ndarray,
        *,
        partial: bool,
    ) -> None:
        suffix = "_partial" if partial else ""
        filename = f"{self.session_name}_chunk{self.chunk_index:03d}{suffix}.npz"
        filepath = self.session_dir / filename
        np.savez_compressed(
            filepath,
            raw_signal=signal,
            position_mm=position,
            sample_rate=np.array(self._sample_rate_hz, dtype=np.int32),
            chunk_index=np.array(self.chunk_index, dtype=np.int32),
        )
        duration_s = len(signal) / self._sample_rate_hz
        size_mb = filepath.stat().st_size / (1024 * 1024)
        tag = " (PARTIAL)" if partial else ""
        logger.info(
            "Chunk %03d saved%s: %s (%.1fs, %.1f MB)",
            self.chunk_index,
            tag,
            filename,
            duration_s,
            size_mb,
        )


class StepDirectionPositionTracker:
    """Reconstruct absolute X position from direction and step waveforms."""

    def __init__(
        self,
        *,
        steps_per_mm: float,
        travel_min_mm: float,
        travel_max_mm: float,
        voltage_threshold: float,
        min_edge_gap: int,
        min_pulse_samples: int,
        dir_vote_samples: int,
        initial_position_mm: float = 0.0,
    ) -> None:
        if steps_per_mm <= 0:
            raise ValueError("steps_per_mm must be greater than zero")
        if travel_max_mm <= travel_min_mm:
            raise ValueError("travel_max_mm must be greater than travel_min_mm")
        self._steps_per_mm = steps_per_mm
        self._travel_min_steps = int(round(travel_min_mm * steps_per_mm))
        self._travel_max_steps = int(round(travel_max_mm * steps_per_mm))
        self._threshold = voltage_threshold
        self._dir_vote_samples = max(0, dir_vote_samples)

        initial_steps = int(round(initial_position_mm * steps_per_mm))
        self._step_count = self._clamp_steps(initial_steps)
        self._edge_detector = StreamingRisingEdgeDetector(
            minimum_edge_gap=min_edge_gap,
            minimum_pulse_samples=min_pulse_samples,
        )
        self._total_edges = 0
        self._total_samples = 0

    @property
    def position_mm(self) -> float:
        """Return the current reconstructed X position in millimetres."""
        return self._step_count / self._steps_per_mm

    @property
    def total_edges(self) -> int:
        """Return the total accepted rising edges."""
        return self._total_edges

    @property
    def total_samples(self) -> int:
        """Return the total processed samples."""
        return self._total_samples

    def set_position_mm(self, position_mm: float) -> None:
        """Seed the tracker with a known absolute X position."""
        self._step_count = self._clamp_steps(
            int(round(position_mm * self._steps_per_mm))
        )

    def feed(self, dir_signal: np.ndarray, step_signal: np.ndarray) -> float:
        """Consume one DAQ chunk and return the updated X position."""
        sample_count = int(len(step_signal))
        if len(dir_signal) != sample_count:
            raise ValueError("dir_signal and step_signal must have identical lengths")
        if sample_count == 0:
            return self.position_mm

        step_high = step_signal > self._threshold
        dir_high = dir_signal > self._threshold
        edge_indices = self._detect_edges(step_high)
        if len(edge_indices) > 0:
            directions = self._direction_at_edges(edge_indices, dir_high, sample_count)
            self._step_count = self._clamp_steps(
                self._step_count + int(np.sum(directions))
            )
            self._total_edges += int(len(edge_indices))

        self._total_samples += sample_count
        return self.position_mm

    def _detect_edges(self, step_high: np.ndarray) -> np.ndarray:
        return self._edge_detector.detect(step_high)

    def _direction_at_edges(
        self,
        edge_indices: np.ndarray,
        dir_high: np.ndarray,
        sample_count: int,
    ) -> np.ndarray:
        if self._dir_vote_samples <= 0:
            return np.where(dir_high[edge_indices], 1, -1)

        half_window = self._dir_vote_samples
        directions = np.empty(len(edge_indices), dtype=np.int8)
        for index, edge_sample in enumerate(edge_indices):
            start = max(0, int(edge_sample) - half_window)
            end = min(sample_count, int(edge_sample) + half_window + 1)
            vote = float(np.mean(dir_high[start:end]))
            directions[index] = 1 if vote > 0.5 else -1
        return directions

    def _clamp_steps(self, step_count: int) -> int:
        return min(self._travel_max_steps, max(self._travel_min_steps, step_count))


class BenchDaqPositionTracker:
    """Background NI-DAQ reader that keeps an independent X position snapshot."""

    def __init__(
        self,
        *,
        settings_path: Path = SETTINGS_PATH,
        initial_position_mm: float = 0.0,
        recordings_dir: Path | None = None,
        recording_session_prefix: str = "bench_raw",
        trips_per_chunk: int | None = None,
        enable_recording: bool = True,
    ) -> None:
        self._settings_path = settings_path
        self._initial_position_mm = initial_position_mm
        self._recordings_dir = recordings_dir
        self._recording_session_prefix = recording_session_prefix
        self._trips_per_chunk = trips_per_chunk
        self._enable_recording = enable_recording
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._daq: DaqCaptureMulti | None = None
        self._tracker: StepDirectionPositionTracker | None = None
        self._recording_session: BenchRecordingSession | None = None
        self._snapshot = TrackerSnapshot(
            position_mm=initial_position_mm,
            total_samples=0,
            total_edges=0,
            live_data_seen=False,
            error=None,
            updated_at_monotonic=time.monotonic(),
        )

    def start(self) -> None:
        """Start background sampling from the external Step/Direction hardware."""
        if self._thread is not None and self._thread.is_alive():
            return

        app_config = load_app_config(self._settings_path)
        daq_channels = list(app_config.daq.channels)
        if len(daq_channels) < 2:
            raise PositionTrackerError(
                "The bench DAQ configuration must expose direction and step channels"
            )
        if self._enable_recording and len(daq_channels) < 3:
            raise PositionTrackerError(
                "Recording requires direction, step, and signal channels in the bench DAQ configuration"
            )

        self._daq = DaqCaptureMulti()
        self._daq.configure(
            device_name=app_config.daq.device_name,
            module=app_config.daq.module,
            channels=daq_channels,
            sample_rate_hz=app_config.daq.sample_rate_hz,
            terminal_config=app_config.daq.terminal_config,
            voltage_min=app_config.daq.voltage_min,
            voltage_max=app_config.daq.voltage_max,
        )
        self._tracker = StepDirectionPositionTracker(
            steps_per_mm=app_config.step_position.steps_per_mm,
            travel_min_mm=app_config.geometry.travel_min_mm,
            travel_max_mm=app_config.geometry.travel_max_mm,
            voltage_threshold=app_config.step_position.voltage_threshold,
            min_edge_gap=app_config.step_position.min_edge_gap,
            min_pulse_samples=app_config.step_position.min_pulse_samples,
            dir_vote_samples=app_config.step_position.dir_vote_samples,
            initial_position_mm=self._initial_position_mm,
        )
        if self._enable_recording:
            output_dir = self._recordings_dir or app_config.paths.recordings_dir
            if not output_dir.is_absolute():
                output_dir = (Path.cwd() / output_dir).resolve()
            self._recording_session = BenchRecordingSession(
                output_dir=output_dir,
                session_prefix=self._recording_session_prefix,
                trips_per_chunk=(
                    app_config.step_position.trips_per_chunk
                    if self._trips_per_chunk is None
                    else self._trips_per_chunk
                ),
                steps_per_mm=app_config.step_position.steps_per_mm,
                travel_min_mm=app_config.geometry.travel_min_mm,
                travel_max_mm=app_config.geometry.travel_max_mm,
                voltage_threshold=app_config.step_position.voltage_threshold,
                min_edge_gap=app_config.step_position.min_edge_gap,
                min_pulse_samples=app_config.step_position.min_pulse_samples,
                dir_vote_samples=app_config.step_position.dir_vote_samples,
                sample_rate_hz=app_config.daq.sample_rate_hz,
            )
        else:
            self._recording_session = None

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="bench-x-axis-tracker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop background acquisition and release NI-DAQ resources."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def snapshot(self) -> TrackerSnapshot:
        """Return the latest tracker snapshot."""
        with self._lock:
            return self._snapshot

    def get_position_mm(self) -> float:
        """Return the latest known X position."""
        return self.snapshot().position_mm

    @property
    def recording_session_dir(self) -> Path | None:
        """Return the active recording session directory, when enabled."""
        if self._recording_session is None:
            return None
        return self._recording_session.session_dir

    @property
    def recording_chunk_count(self) -> int | None:
        """Return the number of saved bench chunks, when recording is enabled."""
        if self._recording_session is None:
            return None
        return self._recording_session.chunk_index

    @property
    def recording_total_samples(self) -> int | None:
        """Return the total saved bench samples, when recording is enabled."""
        if self._recording_session is None:
            return None
        return self._recording_session.total_samples

    def wait_until_live(self, timeout_s: float) -> bool:
        """Wait until at least one hardware chunk has been processed."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.snapshot().live_data_seen:
                return True
            time.sleep(TRACKER_IDLE_SLEEP_S)
        return self.snapshot().live_data_seen

    def _run_loop(self) -> None:
        assert self._daq is not None
        assert self._tracker is not None

        try:
            self._daq.start()
            while not self._stop_event.is_set():
                chunks = self._daq.get_chunks()
                if not chunks:
                    time.sleep(TRACKER_IDLE_SLEEP_S)
                    continue
                for chunk in chunks:
                    if chunk.ndim != 2 or chunk.shape[0] < 2:
                        continue
                    position_mm = self._tracker.feed(chunk[0], chunk[1])
                    if self._recording_session is not None:
                        self._recording_session.feed(chunk)
                    with self._lock:
                        self._snapshot = TrackerSnapshot(
                            position_mm=position_mm,
                            total_samples=self._tracker.total_samples,
                            total_edges=self._tracker.total_edges,
                            live_data_seen=True,
                            error=None,
                            updated_at_monotonic=time.monotonic(),
                        )
        except Exception as exc:  # noqa: BLE001 - DAQ drivers raise platform-specific errors.
            with self._lock:
                self._snapshot = TrackerSnapshot(
                    position_mm=self._snapshot.position_mm,
                    total_samples=self._snapshot.total_samples,
                    total_edges=self._snapshot.total_edges,
                    live_data_seen=self._snapshot.live_data_seen,
                    error=str(exc),
                    updated_at_monotonic=time.monotonic(),
                )
        finally:
            if self._recording_session is not None:
                self._recording_session.close()
            if self._daq is not None:
                self._daq.stop()
                self._daq = None


def sanitize_recording_session_prefix(value: str) -> str:
    """Return a filesystem-safe recording session prefix."""
    sanitized = SAFE_SESSION_NAME_PATTERN.sub("_", value).strip("._-")
    return sanitized or "bench_raw"
