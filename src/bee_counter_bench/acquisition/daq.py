"""Multi-channel NI DAQ acquisition for counter stimulus bench.

Captures 3 analog channels simultaneously: Direction, Step, and Signal.
Returns 2D arrays [3, N] per chunk for position tracking + recording.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

MAX_QUEUE_SIZE = 500
MAX_CONSECUTIVE_ERRORS = 3
MAX_RESTARTS = 5
RESTART_SETTLE_DELAY_S = 0.5


class DaqCaptureMulti:
    """NI DAQmx multi-channel continuous acquisition in background thread.

    Reads Direction (ai1), Step (ai2), and Signal (ai3) in a single task.
    Returns 2D [3, N] float32 chunks via get_chunks().
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._running: bool = False
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=MAX_QUEUE_SIZE)

        self._device_name: str = "cDAQ9185-1D0D747"
        self._module: str = "Mod4"
        self._channels: list[str] = ["Mod3/ai1", "Mod3/ai2", "Mod4/ai3"]
        self._sample_rate: int = 100000
        self._terminal_config: str = "diff"
        self._voltage_min: float = -10.0
        self._voltage_max: float = 10.0

        self._start_timestamp: float = 0.0
        self._total_samples: int = 0
        self._errors: int = 0

    def configure(
        self,
        device_name: str = "cDAQ9185-1D0D747",
        module: str = "Mod4",
        channels: list[str] | None = None,
        sample_rate_hz: int = 100000,
        terminal_config: str = "diff",
        voltage_min: float = -10.0,
        voltage_max: float = 10.0,
        **_kwargs: object,
    ) -> None:
        """Set acquisition parameters before start()."""
        self._device_name = device_name
        self._module = module
        if channels is not None:
            self._channels = channels
        self._sample_rate = sample_rate_hz
        self._terminal_config = terminal_config
        self._voltage_min = voltage_min
        self._voltage_max = voltage_max

    @property
    def channel_paths(self) -> list[str]:
        """Full NI-DAQmx paths for all channels.

        Supports two formats in self._channels:
          - "Mod3/ai1" uses the declared module and channel as-is.
          - "ai1" uses self._module as the module prefix.
        """
        paths = []
        for ch in self._channels:
            if "/" in ch:
                paths.append(f"{self._device_name}{ch}")
            else:
                paths.append(f"{self._device_name}{self._module}/{ch}")
        return paths

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def start_timestamp(self) -> float:
        return self._start_timestamp

    @property
    def total_samples(self) -> int:
        return self._total_samples

    def start(self) -> None:
        """Start background acquisition thread."""
        if self._running:
            return
        self._running = True
        self._total_samples = 0
        self._errors = 0
        self._start_timestamp = time.time()
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name="DAQ-Multi"
        )
        self._thread.start()
        logger.info(
            "DAQ multi-ch started: %s @ %d Hz", self.channel_paths, self._sample_rate
        )

    def stop(self) -> None:
        """Stop acquisition."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info(
            "DAQ stopped: %d samples, %d errors", self._total_samples, self._errors
        )

    def get_chunks(self) -> list[np.ndarray]:
        """Drain all queued 2D chunks [n_channels, n_samples]."""
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    def _create_and_run_task(self) -> bool:
        """Create DAQ task, read until error or stop."""
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, TerminalConfiguration

        terminal_map = {
            "diff": TerminalConfiguration.DIFF,
            "rse": TerminalConfiguration.RSE,
            "nrse": TerminalConfiguration.NRSE,
            "pseudo_diff": TerminalConfiguration.PSEUDO_DIFF,
        }
        terminal = terminal_map.get(self._terminal_config, TerminalConfiguration.DIFF)
        read_chunk = max(100, self._sample_rate // 10)

        try:
            with nidaqmx.Task() as task:
                for ch_path in self.channel_paths:
                    task.ai_channels.add_ai_voltage_chan(
                        ch_path,
                        min_val=self._voltage_min,
                        max_val=self._voltage_max,
                        terminal_config=terminal,
                    )
                try:
                    task.timing.cfg_samp_clk_timing(
                        self._sample_rate,
                        sample_mode=AcquisitionType.CONTINUOUS,
                        samps_per_chan=read_chunk,
                    )
                except nidaqmx.DaqError as e:
                    logger.error("DAQ config error: %s", e)
                    return False

                input_buf_size = max(read_chunk * 10, self._sample_rate * 5)
                task.in_stream.input_buf_size = input_buf_size
                logger.info(
                    "DAQ task: %s (chunk=%d, buf=%.1fs)",
                    self.channel_paths,
                    read_chunk,
                    input_buf_size / self._sample_rate,
                )

                consecutive_errors = 0
                while self._running:
                    try:
                        data = task.read(
                            number_of_samples_per_channel=read_chunk, timeout=5.0
                        )
                        # Multi-channel: list of lists [n_ch][n_samp]
                        chunk = np.array(data, dtype=np.float32)
                        if chunk.ndim == 1:
                            chunk = chunk.reshape(1, -1)
                        self._total_samples += chunk.shape[1]
                        consecutive_errors = 0
                        try:
                            self._queue.put_nowait(chunk)
                        except queue.Full:
                            try:
                                self._queue.get_nowait()
                            except queue.Empty:
                                pass
                            self._queue.put_nowait(chunk)

                    except nidaqmx.DaqError as e:
                        self._errors += 1
                        consecutive_errors += 1
                        is_overflow = "-200279" in str(e)
                        if is_overflow:
                            logger.warning(
                                "DAQ overflow (%d/%d)",
                                consecutive_errors,
                                MAX_CONSECUTIVE_ERRORS,
                            )
                        else:
                            logger.warning("DAQ read error: %s", e)
                        if not self._running:
                            return False
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            if is_overflow:
                                return True
                            logger.error(
                                "DAQ: %d consecutive errors", consecutive_errors
                            )
                            return False
                        time.sleep(0.05)

                    except Exception as e:
                        logger.error("DAQ unexpected: %s", e)
                        self._errors += 1
                        return False

        except Exception as e:
            logger.error("DAQ task creation failed: %s", e)
            return False

        return False

    def _read_loop(self) -> None:
        """Background thread with auto-recovery."""
        restart_count = 0
        while self._running and restart_count < MAX_RESTARTS:
            should_retry = self._create_and_run_task()
            if not should_retry or not self._running:
                break
            restart_count += 1
            logger.info("DAQ restart %d/%d", restart_count, MAX_RESTARTS)
            time.sleep(RESTART_SETTLE_DELAY_S)

        if restart_count >= MAX_RESTARTS:
            logger.error("DAQ gave up after %d restarts", MAX_RESTARTS)
        self._running = False
