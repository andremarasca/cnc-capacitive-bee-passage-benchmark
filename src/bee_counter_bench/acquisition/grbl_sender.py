"""Stream G-code files to GRBL 0.9j-compatible controllers.

The module implements the official GRBL serial streaming patterns with
explicit fault handling. It stops sending new commands when the controller
reports an error or alarm, and it never clears alarms automatically.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Deque, Iterable, Literal, Protocol, Sequence, cast

from bee_counter_bench.bench_config import PROJECT_ROOT

try:
    SERIAL_MODULE: Any | None = importlib.import_module("serial")
    LIST_PORTS_MODULE: Any | None = importlib.import_module("serial.tools.list_ports")
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    SERIAL_MODULE = None
    LIST_PORTS_MODULE = None


BAUDRATE_DEFAULT = 115200
RX_BUFFER_SIZE_DEFAULT = 128
MAX_GRBL_LINE_CHARS_DEFAULT = 78
SERIAL_TIMEOUT_S = 0.1
WRITE_TIMEOUT_S = 5.0
RESET_DELAY_S = 2.0
WAKE_DELAY_S = 0.5
RESPONSE_TIMEOUT_S_DEFAULT = 8.0
STATUS_POLL_INTERVAL_S_DEFAULT = 0.2
PROGRESS_INTERVAL_DEFAULT = 100
FEED_HOLD = b"!"
STATUS_QUERY = b"?"
CTRL_X = b"\x18"

COMMENT_PATTERN = re.compile(r"\([^)]*\)")
STATUS_PATTERN = re.compile(r"<([^,|>]+)")
FEEDRATE_PATTERN = re.compile(r"F(?P<feed>[0-9]+(?:\.[0-9]+)?)")
ARDUINO_VID = 0x2341
ARDUINO_UNO_PIDS = {0x0043, 0x0001}
ARDUINO_MEGA_PIDS = {0x0042, 0x0010}
CH340_VID = 0x1A86
CH340_PID = 0x7523
FTDI_VID = 0x0403
FTDI_SERIAL_PID = 0x6001

GcodeCommand = tuple[int, str]
StreamMode = Literal["simple", "buffered"]
CheckModeToggle = Literal["enabled", "disabled", "startup", "unknown"]
StreamCommandCallback = Callable[[int, str], None]


class SerialLike(Protocol):
    """Port contract required by the GRBL sender."""

    @property
    def in_waiting(self) -> int:
        """Return pending bytes in the receive buffer."""

    def write(self, data: bytes) -> int:
        """Write bytes to the serial link."""

    def read(self, size: int = 1) -> bytes:
        """Read bytes from the serial link."""

    def readline(self) -> bytes:
        """Read one line from the serial link."""

    def reset_input_buffer(self) -> None:
        """Discard pending input bytes."""

    def reset_output_buffer(self) -> None:
        """Discard pending output bytes."""

    def close(self) -> None:
        """Close the serial link."""


@dataclass(frozen=True)
class StreamSettings:
    """Runtime settings for one GRBL streaming session."""

    rx_buffer_size: int = RX_BUFFER_SIZE_DEFAULT
    response_timeout_s: float = RESPONSE_TIMEOUT_S_DEFAULT
    status_poll_interval_s: float = STATUS_POLL_INTERVAL_S_DEFAULT
    progress_interval: int = PROGRESS_INTERVAL_DEFAULT


@dataclass
class StreamStats:
    """Summary of a streaming operation."""

    total_commands: int
    sent_commands: int = 0
    confirmed_commands: int = 0
    start_time_monotonic: float = 0.0
    end_time_monotonic: float = 0.0

    @property
    def elapsed_s(self) -> float:
        """Return elapsed time in seconds."""
        end_time = self.end_time_monotonic or time.monotonic()
        return max(0.0, end_time - self.start_time_monotonic)


@dataclass(frozen=True)
class SerialPortCandidate:
    """Serial port candidate for GRBL auto-detection."""

    device: str
    description: str
    hwid: str
    score: int


class GrblSenderError(Exception):
    """Base class for GRBL sender failures."""


class GrblAlarmError(GrblSenderError):
    """Raised when GRBL reports an alarm state."""


class GrblCommandError(GrblSenderError):
    """Raised when GRBL rejects a command."""

    def __init__(self, response: str, source_line: int | None, command: str | None):
        self.response = response
        self.source_line = source_line
        self.command = command
        location = f"source line {source_line}" if source_line else "unknown line"
        super().__init__(f"GRBL rejected {location}: {response} ({command})")


class GrblTimeoutError(GrblSenderError):
    """Raised when GRBL does not answer within the configured timeout."""


class GrblResetError(GrblSenderError):
    """Raised when GRBL restarts while a stream is active."""

    def __init__(self, response: str, source_line: int | None, command: str | None):
        self.response = response
        self.source_line = source_line
        self.command = command
        location = f"source line {source_line}" if source_line else "unknown line"
        super().__init__(
            f"GRBL reset during stream at {location}: {response} ({command})"
        )


class JsonRunLogger:
    """Write JSON Lines logs for serial auditability."""

    def __init__(self, path: Path, echo: bool) -> None:
        self.path = path
        self.echo = echo
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def close(self) -> None:
        """Close the log file."""
        self._handle.close()

    def emit(
        self,
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR"],
        context: str,
        message: str,
        *,
        error_code: str | None = None,
        **fields: Any,
    ) -> None:
        """Write one canonical structured log event."""
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "context": context,
            "error_code": error_code,
            "message": message,
        }
        event.update(fields)
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        self._handle.write(line + "\n")
        self._handle.flush()
        if self.echo and level in {"INFO", "WARNING", "ERROR"}:
            print(f"[{level}] {message}")


class GrblFileSender:
    """Reliable GRBL serial streamer with alarm-aware halt behavior."""

    def __init__(
        self,
        serial_port: SerialLike,
        logger: JsonRunLogger,
        settings: StreamSettings,
        on_command_sent: StreamCommandCallback | None = None,
        on_command_acknowledged: StreamCommandCallback | None = None,
    ) -> None:
        self._serial = serial_port
        self._logger = logger
        self._settings = settings
        self._on_command_sent = on_command_sent
        self._on_command_acknowledged = on_command_acknowledged
        self.last_state = "Unknown"

    def wake(self) -> None:
        """Wake GRBL and flush startup text."""
        time.sleep(RESET_DELAY_S)
        self._write_bytes(b"\r\n\r\n", context="wake")
        time.sleep(WAKE_DELAY_S)
        self._read_available(context="wake")

    def soft_reset(self) -> None:
        """Send Ctrl-X and wait for startup text."""
        self._write_bytes(CTRL_X, context="soft_reset")
        deadline = time.monotonic() + RESET_DELAY_S + self._settings.response_timeout_s
        while time.monotonic() < deadline:
            line = self._read_line(deadline=deadline, context="soft_reset")
            if line is None:
                continue
            if "Grbl" in line:
                self._logger.emit("INFO", "soft_reset", "GRBL restarted", line=line)
                self._read_available(context="soft_reset")
                return
        raise GrblTimeoutError("GRBL did not restart after Ctrl-X")

    def send_command(self, command: str, context: str = "command") -> None:
        """Send one GRBL command and wait for ok/error."""
        clean_command = command.strip()
        self._write_ascii(clean_command + "\n", context=context)
        self._read_until_ack(context=context, source_line=None, command=clean_command)

    def read_settings(self) -> dict[str, str]:
        """Read GRBL $$ settings as key/value strings."""
        self._write_ascii("$$\n", context="settings_read")
        deadline = time.monotonic() + self._settings.response_timeout_s
        observed: dict[str, str] = {}
        while time.monotonic() < deadline:
            line = self._read_line(deadline=deadline, context="settings_read")
            if line is None:
                continue
            if line == "ok":
                return observed
            setting_match = re.match(r"^(\$\d+)=(.+?)(?:\s|$)", line)
            if setting_match is not None:
                observed[setting_match.group(1)] = setting_match.group(2).strip()
                continue
            if line.lower().startswith("error"):
                raise GrblCommandError(line, None, "$$")
            if line.startswith("ALARM:"):
                raise GrblAlarmError(line)
            self._handle_feedback(line, context="settings_read")
        raise GrblTimeoutError("Timed out reading GRBL settings")

    def send_stream_command(self, source_line: int, command: str) -> None:
        """Send one stream command and raise on mid-stream resets."""
        clean_command = command.strip()
        self._write_ascii(clean_command + "\n", context="tx", source_line=source_line)
        self._read_until_ack(
            context="stream", source_line=source_line, command=clean_command
        )

    def enable_check_mode(self) -> None:
        """Enable GRBL check mode so streaming validates without motion."""
        self._logger.emit("INFO", "check_mode", "Enabling GRBL check mode")
        state = self._toggle_check_mode(context="check_mode_enable")
        if state == "enabled":
            return
        self._logger.emit(
            "WARNING",
            "check_mode",
            "Check mode was not enabled after first toggle; toggling again",
            error_code="CHECK_MODE_ENABLE_RETRY",
            state=state,
        )
        state = self._toggle_check_mode(context="check_mode_enable_retry")
        if state != "enabled":
            raise GrblSenderError(
                f"Could not enable GRBL check mode; last toggle state was {state}"
            )

    def disable_check_mode(self) -> None:
        """Disable GRBL check mode and wait for reset or ok."""
        self._logger.emit("INFO", "check_mode", "Disabling GRBL check mode")
        state = self._toggle_check_mode(context="check_mode_disable")
        if state in {"disabled", "startup"}:
            return
        if state == "enabled":
            self._logger.emit(
                "WARNING",
                "check_mode",
                "Check mode was enabled during cleanup; sending soft reset",
                error_code="CHECK_MODE_CLEANUP_RESET",
            )
            self.soft_reset()
            return
        raise GrblSenderError(
            f"Could not disable GRBL check mode; last toggle state was {state}"
        )

    def _toggle_check_mode(self, context: str) -> CheckModeToggle:
        """Toggle GRBL check mode and return the reported state."""
        self._write_ascii("$C\n", context=context)
        deadline = time.monotonic() + RESET_DELAY_S + self._settings.response_timeout_s
        toggle_state: CheckModeToggle = "unknown"
        while time.monotonic() < deadline:
            line = self._read_line(deadline=deadline, context=context)
            if line is None:
                continue
            if line == "[Enabled]":
                toggle_state = "enabled"
                self._logger.emit(
                    "INFO", "check_mode", "Check mode toggle response", response=line
                )
                continue
            if line == "[Disabled]":
                toggle_state = "disabled"
                self._logger.emit(
                    "INFO", "check_mode", "Check mode toggle response", response=line
                )
                continue
            if line == "ok":
                return toggle_state
            if "Grbl" in line:
                self._read_available(context=context)
                return "startup"
            if line.startswith("ALARM:"):
                raise GrblAlarmError(line)
            if line.lower().startswith("error"):
                raise GrblCommandError(line, None, "$C")
        raise GrblTimeoutError("GRBL did not respond to check mode toggle")

    def query_status(self) -> str:
        """Request one GRBL status report and return the parsed state."""
        self._write_bytes(STATUS_QUERY, context="status")
        deadline = time.monotonic() + self._settings.response_timeout_s
        while time.monotonic() < deadline:
            line = self._read_line(deadline=deadline, context="status")
            if line is None:
                continue
            if line.startswith("<"):
                self._update_state_from_status(line)
                return self.last_state
            self._handle_feedback(line, context="status")
        raise GrblTimeoutError("GRBL did not return a status report")

    def wait_for_idle(self, timeout_s: float) -> None:
        """Poll GRBL until the controller reports Idle."""
        self._logger.emit("INFO", "wait_idle", "Waiting for GRBL Idle state")
        started = time.monotonic()
        while True:
            state = self.query_status()
            if state == "Idle":
                self._logger.emit("INFO", "wait_idle", "GRBL is Idle")
                return
            if state == "Alarm":
                raise GrblAlarmError("GRBL entered Alarm while waiting for Idle")
            if timeout_s > 0 and time.monotonic() - started > timeout_s:
                raise GrblTimeoutError(
                    f"GRBL did not become Idle within {timeout_s:.1f}s"
                )
            time.sleep(self._settings.status_poll_interval_s)

    def stream(self, commands: Sequence[GcodeCommand], mode: StreamMode) -> StreamStats:
        """Stream commands using the selected GRBL protocol."""
        if mode == "simple":
            return self._stream_simple(commands)
        return self._stream_buffered(commands)

    def _stream_simple(self, commands: Sequence[GcodeCommand]) -> StreamStats:
        stats = StreamStats(
            total_commands=len(commands), start_time_monotonic=time.monotonic()
        )
        self._logger.emit(
            "INFO",
            "stream",
            "Starting simple send-response stream",
            total=len(commands),
        )
        for index, (source_line, command) in enumerate(commands, start=1):
            self._write_ascii(command + "\n", context="tx", source_line=source_line)
            self._notify_command_sent(source_line, command)
            stats.sent_commands += 1
            self._read_until_ack(
                context="stream", source_line=source_line, command=command
            )
            stats.confirmed_commands += 1
            self._notify_command_acknowledged(source_line, command)
            self._emit_progress(index, stats.total_commands, source_line)
        stats.end_time_monotonic = time.monotonic()
        self._emit_stream_finished("Simple stream finished", stats)
        return stats

    def _stream_buffered(self, commands: Sequence[GcodeCommand]) -> StreamStats:
        stats = StreamStats(
            total_commands=len(commands), start_time_monotonic=time.monotonic()
        )
        inflight: Deque[tuple[int, str, int]] = deque()
        self._logger.emit(
            "INFO",
            "stream",
            "Starting buffered character-counting stream",
            total=len(commands),
        )
        for source_line, command in commands:
            line_length = len(command) + 1
            while (
                self._inflight_size(inflight) + line_length
                >= self._settings.rx_buffer_size - 1
            ):
                self._drain_one_response(inflight, stats)
            self._write_ascii(command + "\n", context="tx", source_line=source_line)
            self._notify_command_sent(source_line, command)
            inflight.append((source_line, command, line_length))
            stats.sent_commands += 1
            self._emit_progress(stats.sent_commands, stats.total_commands, source_line)
        while inflight:
            self._drain_one_response(inflight, stats)
        stats.end_time_monotonic = time.monotonic()
        self._emit_stream_finished("Buffered stream finished", stats)
        return stats

    def _drain_one_response(
        self,
        inflight: Deque[tuple[int, str, int]],
        stats: StreamStats,
    ) -> None:
        deadline = time.monotonic() + self._settings.response_timeout_s
        while time.monotonic() < deadline:
            line = self._read_line(deadline=deadline, context="stream")
            if line is None:
                continue
            if line == "ok":
                if inflight:
                    source_line, command, _line_length = inflight.popleft()
                    stats.confirmed_commands += 1
                    self._notify_command_acknowledged(source_line, command)
                    self._logger.emit(
                        "DEBUG",
                        "ack",
                        "Command confirmed",
                        source_line=source_line,
                        command=command,
                        confirmed=stats.confirmed_commands,
                    )
                return
            if line.lower().startswith("error"):
                fault_source_line, fault_command = self._first_inflight_command(
                    inflight
                )
                self._halt_after_fault(line, fault_source_line, fault_command)
                raise GrblCommandError(line, fault_source_line, fault_command)
            if line.startswith("ALARM:"):
                fault_source_line, fault_command = self._first_inflight_command(
                    inflight
                )
                self._halt_after_fault(line, fault_source_line, fault_command)
                raise GrblAlarmError(line)
            if self._is_reset_feedback(line):
                fault_source_line, fault_command = self._first_inflight_command(
                    inflight
                )
                self._raise_reset_during_stream(line, fault_source_line, fault_command)
            self._handle_feedback(line, context="stream")
        raise GrblTimeoutError("Timed out waiting for GRBL stream response")

    def _read_until_ack(
        self, context: str, source_line: int | None, command: str | None
    ) -> None:
        deadline = time.monotonic() + self._settings.response_timeout_s
        while time.monotonic() < deadline:
            line = self._read_line(deadline=deadline, context=context)
            if line is None:
                continue
            if line == "ok":
                return
            if line.lower().startswith("error"):
                self._halt_after_fault(line, source_line, command)
                raise GrblCommandError(line, source_line, command)
            if line.startswith("ALARM:"):
                self._halt_after_fault(line, source_line, command)
                raise GrblAlarmError(line)
            if context == "stream" and self._is_reset_feedback(line):
                self._raise_reset_during_stream(line, source_line, command)
            self._handle_feedback(line, context=context)
        raise GrblTimeoutError(f"Timed out waiting for GRBL response in {context}")

    def _handle_feedback(self, line: str, context: str) -> None:
        if line.startswith("<"):
            self._update_state_from_status(line)
            return
        if self._is_reset_feedback(line):
            self._logger.emit("INFO", context, "GRBL startup", line=line)
            return
        self._logger.emit("DEBUG", context, "GRBL feedback", line=line)

    def _notify_command_sent(self, source_line: int, command: str) -> None:
        if self._on_command_sent is not None:
            self._on_command_sent(source_line, command)

    def _notify_command_acknowledged(self, source_line: int, command: str) -> None:
        if self._on_command_acknowledged is not None:
            self._on_command_acknowledged(source_line, command)

    @staticmethod
    def _is_reset_feedback(line: str) -> bool:
        lowered = line.strip().lower()
        return (
            lowered.startswith("grbl")
            or lowered == "init"
            or lowered.startswith("init ")
            or " reset" in lowered
            or lowered.startswith("reset")
        )

    def _update_state_from_status(self, line: str) -> None:
        status_match = STATUS_PATTERN.match(line)
        if status_match is None:
            self._logger.emit("WARNING", "status", "Could not parse status", line=line)
            return
        self.last_state = status_match.group(1)
        self._logger.emit(
            "DEBUG", "status", "Status updated", state=self.last_state, line=line
        )

    def _halt_after_fault(
        self, response: str, source_line: int | None, command: str | None
    ) -> None:
        self._logger.emit(
            "ERROR",
            "fault",
            "Fault received; halting stream and sending feed hold",
            error_code="GRBL_FAULT",
            response=response,
            source_line=source_line,
            command=command,
        )
        self._write_bytes(FEED_HOLD, context="fault_feed_hold")

    def _raise_reset_during_stream(
        self, response: str, source_line: int | None, command: str | None
    ) -> None:
        self._logger.emit(
            "ERROR",
            "fault",
            "GRBL restarted while a stream was active; stopping immediately",
            error_code="GRBL_RESET_DURING_STREAM",
            response=response,
            source_line=source_line,
            command=command,
        )
        raise GrblResetError(response, source_line, command)

    def _emit_progress(self, current: int, total: int, source_line: int) -> None:
        interval = max(1, self._settings.progress_interval)
        if current == total or current == 1 or current % interval == 0:
            percent = 100.0 * current / max(1, total)
            self._logger.emit(
                "INFO",
                "progress",
                f"Progress {current}/{total} ({percent:.1f}%)",
                current=current,
                total=total,
                percent=round(percent, 3),
                source_line=source_line,
            )

    def _emit_stream_finished(self, message: str, stats: StreamStats) -> None:
        self._logger.emit(
            "INFO",
            "stream",
            message,
            total_commands=stats.total_commands,
            sent_commands=stats.sent_commands,
            confirmed_commands=stats.confirmed_commands,
            elapsed_s=round(stats.elapsed_s, 3),
        )

    def _read_available(self, context: str) -> None:
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            line = self._read_line(deadline=deadline, context=context)
            if line is None:
                continue
            self._handle_feedback(line, context=context)

    def _read_line(self, deadline: float, context: str) -> str | None:
        while time.monotonic() < deadline:
            raw = self._serial.readline()
            if raw == b"":
                return None
            line = raw.decode("ascii", errors="replace").strip()
            if not line:
                continue
            self._logger.emit("DEBUG", "rx", "RX", line=line, parent_context=context)
            return line
        return None

    def _write_ascii(self, text: str, context: str, **fields: Any) -> None:
        self._logger.emit(
            "DEBUG",
            "tx",
            "TX",
            data=text.rstrip("\n"),
            parent_context=context,
            **fields,
        )
        self._serial.write(text.encode("ascii"))

    def _write_bytes(self, data: bytes, context: str) -> None:
        self._logger.emit(
            "DEBUG", "tx", "TX bytes", data=list(data), parent_context=context
        )
        self._serial.write(data)

    @staticmethod
    def _inflight_size(inflight: Deque[tuple[int, str, int]]) -> int:
        return sum(line_length for _source_line, _command, line_length in inflight)

    @staticmethod
    def _first_inflight_command(
        inflight: Deque[tuple[int, str, int]],
    ) -> tuple[int | None, str | None]:
        if not inflight:
            return None, None
        source_line, command, _line_length = inflight[0]
        return source_line, command


def stats_to_fields(stats: StreamStats) -> dict[str, int | float]:
    """Convert stream stats to JSON-serializable log fields."""
    return {
        "total_commands": stats.total_commands,
        "sent_commands": stats.sent_commands,
        "confirmed_commands": stats.confirmed_commands,
        "elapsed_s": round(stats.elapsed_s, 3),
    }


def normalize_gcode_line(raw_line: str) -> str:
    """Return one GRBL-ready line without comments or whitespace."""
    without_parentheses = COMMENT_PATTERN.sub("", raw_line)
    without_semicolon = without_parentheses.split(";", 1)[0]
    compact = re.sub(r"\s+", "", without_semicolon).upper()
    return compact.strip()


def load_gcode_commands(
    gcode_path: Path,
    max_lines: int | None,
    max_line_chars: int,
    max_feedrate: float | None = None,
) -> list[GcodeCommand]:
    """Load and normalize commands from a G-code file."""
    commands: list[GcodeCommand] = []
    with gcode_path.open("r", encoding="utf-8") as handle:
        for source_line, raw_line in enumerate(handle, start=1):
            command = normalize_gcode_line(raw_line)
            if not command:
                continue
            command = cap_gcode_feedrate(command, max_feedrate)
            if len(command) > max_line_chars:
                raise ValueError(
                    f"Line {source_line} has {len(command)} characters after cleanup; "
                    f"limit is {max_line_chars}: {command}"
                )
            commands.append((source_line, command))
            if max_lines is not None and len(commands) >= max_lines:
                break
    if not commands:
        raise ValueError(f"No executable G-code commands found in {gcode_path}")
    return commands


def cap_gcode_feedrate(command: str, max_feedrate: float | None) -> str:
    """Return command with F words capped to the configured maximum."""
    if max_feedrate is None:
        return command
    if max_feedrate <= 0:
        raise ValueError("max_feedrate must be greater than zero")

    def replace_feedrate(match: re.Match[str]) -> str:
        feedrate = float(match.group("feed"))
        if feedrate <= max_feedrate:
            return match.group(0)
        return f"F{format_feedrate(max_feedrate)}"

    return FEEDRATE_PATTERN.sub(replace_feedrate, command)


def format_feedrate(feedrate: float) -> str:
    """Format a capped feedrate without unnecessary decimal zeros."""
    if feedrate.is_integer():
        return str(int(feedrate))
    return f"{feedrate:.3f}".rstrip("0").rstrip(".")


def score_port(description: str, hwid: str, vid: int | None, pid: int | None) -> int:
    """Score a serial port for GRBL probing priority."""
    text = f"{description} {hwid}".lower()
    score = 0
    if vid == ARDUINO_VID and pid in ARDUINO_UNO_PIDS | ARDUINO_MEGA_PIDS:
        score += 100
    if vid == CH340_VID and pid == CH340_PID:
        score += 70
    if vid == FTDI_VID and pid == FTDI_SERIAL_PID:
        score += 70
    if "arduino" in text:
        score += 60
    if "ch340" in text or "wch" in text:
        score += 40
    if "ftdi" in text:
        score += 40
    if "usb serial" in text or "usb-serial" in text:
        score += 20
    return score


def list_serial_candidates() -> list[SerialPortCandidate]:
    """List serial ports ordered by likely GRBL relevance."""
    _serial_module, list_ports_module = require_pyserial()
    candidates: list[SerialPortCandidate] = []
    for port_info in cast(Iterable[Any], list_ports_module.comports()):
        score = score_port(
            port_info.description or "",
            port_info.hwid or "",
            port_info.vid,
            port_info.pid,
        )
        candidates.append(
            SerialPortCandidate(
                device=str(port_info.device),
                description=str(port_info.description or ""),
                hwid=str(port_info.hwid or ""),
                score=score,
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def detect_grbl_port(baudrate: int, logger: JsonRunLogger) -> str:
    """Probe likely serial ports and return the first GRBL controller."""
    require_pyserial()
    candidates = [
        candidate for candidate in list_serial_candidates() if candidate.score > 0
    ]
    if not candidates:
        candidates = list_serial_candidates()
    for candidate in candidates:
        logger.emit(
            "INFO",
            "auto_port",
            "Probing serial port",
            port=candidate.device,
            score=candidate.score,
        )
        probe = open_probe_port_with_retries(candidate.device, baudrate, logger)
        if probe is None:
            continue
        try:
            if probe_port_for_grbl(probe, logger):
                logger.emit("INFO", "auto_port", "GRBL detected", port=candidate.device)
                return candidate.device
        finally:
            probe.close()
    raise GrblSenderError("No GRBL controller detected on available serial ports")


def open_probe_port_with_retries(
    port: str, baudrate: int, logger: JsonRunLogger
) -> SerialLike | None:
    """Open a candidate port with short retries for transient Windows locks."""
    serial_module, _list_ports_module = require_pyserial()
    last_error = ""
    for attempt in range(1, 4):
        try:
            return cast(
                SerialLike,
                serial_module.Serial(
                    port,
                    baudrate,
                    timeout=SERIAL_TIMEOUT_S,
                    write_timeout=WRITE_TIMEOUT_S,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - serial drivers raise platform-specific errors.
            last_error = str(exc)
            logger.emit(
                "WARNING",
                "auto_port",
                "Could not open candidate port",
                port=port,
                attempt=attempt,
                error=last_error,
            )
            time.sleep(0.5)
    logger.emit(
        "ERROR",
        "auto_port",
        "Candidate port unavailable after retries",
        port=port,
        error=last_error,
    )
    return None


def probe_port_for_grbl(serial_port: SerialLike, logger: JsonRunLogger) -> bool:
    """Return True when a serial port looks like GRBL."""
    time.sleep(RESET_DELAY_S)
    serial_port.reset_input_buffer()
    serial_port.write(b"\r\n\r\n")
    time.sleep(WAKE_DELAY_S)
    serial_port.write(b"$I\n")
    serial_port.write(STATUS_QUERY)
    deadline = time.monotonic() + RESPONSE_TIMEOUT_S_DEFAULT
    observed: list[str] = []
    while time.monotonic() < deadline:
        raw = serial_port.readline()
        if raw == b"":
            continue
        line = raw.decode("ascii", errors="replace").strip()
        if not line:
            continue
        observed.append(line)
        if "Grbl" in line or (line.startswith("<") and "MPos:" in line):
            logger.emit("INFO", "auto_port", "GRBL probe matched", line=line)
            return True
    logger.emit(
        "DEBUG", "auto_port", "GRBL probe did not match", observed=observed[:10]
    )
    return False


def open_serial_port(port: str, baudrate: int, logger: JsonRunLogger) -> SerialLike:
    """Open the serial port used by the GRBL controller."""
    serial_module, _list_ports_module = require_pyserial()
    last_error = ""
    for attempt in range(1, 4):
        try:
            return cast(
                SerialLike,
                serial_module.Serial(
                    port=port,
                    baudrate=baudrate,
                    timeout=SERIAL_TIMEOUT_S,
                    write_timeout=WRITE_TIMEOUT_S,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - serial drivers raise platform-specific errors.
            last_error = str(exc)
            logger.emit(
                "WARNING",
                "connect",
                "Could not open GRBL serial port",
                port=port,
                attempt=attempt,
                error=last_error,
            )
            time.sleep(0.5)
    raise GrblSenderError(f"Could not open GRBL serial port {port}: {last_error}")


def require_pyserial() -> tuple[Any, Any]:
    """Fail early when pyserial is unavailable."""
    if SERIAL_MODULE is None or LIST_PORTS_MODULE is None:
        raise RuntimeError(
            "pyserial is required. Install it with: python -m pip install pyserial"
        )
    return SERIAL_MODULE, LIST_PORTS_MODULE


def resolve_mode(requested_mode: str, check_only: bool) -> StreamMode:
    """Resolve the effective streaming mode."""
    if requested_mode == "simple" or check_only:
        return "simple"
    return "buffered"


def run_check_mode(
    sender: GrblFileSender, commands: Sequence[GcodeCommand]
) -> StreamStats:
    """Validate G-code in GRBL check mode and always leave check mode."""
    sender.enable_check_mode()
    reset_detected = False
    try:
        return sender.stream(commands, mode="simple")
    except GrblResetError:
        reset_detected = True
        raise
    finally:
        if not reset_detected:
            sender.disable_check_mode()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Reliable GRBL 0.9j G-code file sender"
    )
    parser.add_argument(
        "gcode_file", nargs="?", type=Path, help="G-code file to stream"
    )
    parser.add_argument("--port", help="Serial port, e.g. COM10")
    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="Probe serial ports and select the first GRBL controller",
    )
    parser.add_argument(
        "--list-ports", action="store_true", help="List serial ports and exit"
    )
    parser.add_argument(
        "--baudrate", type=int, default=BAUDRATE_DEFAULT, help="Serial baudrate"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Use GRBL check mode only; no motion is executed",
    )
    parser.add_argument(
        "--precheck", action="store_true", help="Run check mode before real streaming"
    )
    parser.add_argument(
        "--status-only", action="store_true", help="Connect, query status, and exit"
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "simple", "buffered"),
        default="auto",
        help="Streaming protocol",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        help="Limit executable G-code commands for controlled tests",
    )
    parser.add_argument(
        "--max-line-chars",
        type=int,
        default=MAX_GRBL_LINE_CHARS_DEFAULT,
        help="Maximum cleaned GRBL line length",
    )
    parser.add_argument(
        "--max-feedrate",
        type=float,
        help="Cap every G-code F word to this feedrate before streaming",
    )
    parser.add_argument(
        "--rx-buffer-size",
        type=int,
        default=RX_BUFFER_SIZE_DEFAULT,
        help="GRBL serial RX buffer size",
    )
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=RESPONSE_TIMEOUT_S_DEFAULT,
        help="Seconds to wait for ok/error",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=0.0,
        help="Seconds to wait for Idle; 0 means no timeout",
    )
    parser.add_argument(
        "--no-wait-idle",
        action="store_true",
        help="Do not wait for final Idle after real streaming",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=PROGRESS_INTERVAL_DEFAULT,
        help="Commands between progress logs",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "acquisition" / "logs" / "last_run.log",
        help="JSONL log path",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Do not echo INFO logs to console"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the file and exit without opening serial",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the GRBL sender CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = JsonRunLogger(args.log_path, echo=not args.quiet)
    try:
        if args.list_ports:
            for candidate in list_serial_candidates():
                print(
                    f"{candidate.device}\t score={candidate.score}\t {candidate.description}\t {candidate.hwid}"
                )
            return 0

        commands: list[GcodeCommand] = []
        if args.gcode_file is not None:
            commands = load_gcode_commands(
                args.gcode_file,
                args.max_lines,
                args.max_line_chars,
                args.max_feedrate,
            )
            logger.emit(
                "INFO",
                "gcode",
                "G-code loaded",
                file=str(args.gcode_file),
                commands=len(commands),
            )

        if args.dry_run:
            print(f"Loaded {len(commands)} executable commands from {args.gcode_file}")
            return 0

        if not args.status_only and not commands:
            parser.error(
                "gcode_file is required unless --status-only or --list-ports is used"
            )

        if not args.port and not args.auto_port:
            parser.error("Use --port COMx or --auto-port")

        port = args.port or detect_grbl_port(args.baudrate, logger)
        logger.emit(
            "INFO",
            "connect",
            "Opening GRBL serial port",
            port=port,
            baudrate=args.baudrate,
        )

        serial_port = open_serial_port(port, args.baudrate, logger)
        try:
            settings = StreamSettings(
                rx_buffer_size=args.rx_buffer_size,
                response_timeout_s=args.response_timeout,
                progress_interval=args.progress_interval,
            )
            sender = GrblFileSender(serial_port, logger, settings)
            sender.wake()
            sender.send_command("$I", context="version")
            state = sender.query_status()
            logger.emit("INFO", "status", "Initial GRBL state", state=state)

            if args.status_only:
                print(f"GRBL status on {port}: {state}")
                return 0

            if args.check:
                stats = run_check_mode(sender, commands)
                print(format_stats("Check mode finished", stats, logger.path))
                return 0

            if args.precheck:
                run_check_mode(sender, commands)
                sender.wake()

            mode = resolve_mode(args.mode, check_only=False)
            stats = sender.stream(commands, mode=mode)
            if not args.no_wait_idle:
                sender.wait_for_idle(timeout_s=args.idle_timeout)
            print(format_stats("Streaming finished", stats, logger.path))
            return 0
        finally:
            serial_port.close()
    except (GrblSenderError, ValueError, RuntimeError) as exc:
        logger.emit(
            "ERROR",
            "main",
            "Sender failed",
            error_code=exc.__class__.__name__,
            error=str(exc),
        )
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Log: {logger.path}", file=sys.stderr)
        return 2
    finally:
        logger.close()


def format_stats(prefix: str, stats: StreamStats, log_path: Path) -> str:
    """Format a short human-readable execution summary."""
    return (
        f"{prefix}: sent={stats.sent_commands}, confirmed={stats.confirmed_commands}, "
        f"elapsed={stats.elapsed_s:.2f}s, log={log_path}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
