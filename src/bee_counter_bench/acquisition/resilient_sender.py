"""Fault-tolerant GRBL sender with independent X-axis recovery."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from bee_counter_bench.bench_config import PROJECT_ROOT, SETTINGS_PATH
from bee_counter_bench.acquisition.grbl_sender import (
    BAUDRATE_DEFAULT,
    MAX_GRBL_LINE_CHARS_DEFAULT,
    PROGRESS_INTERVAL_DEFAULT,
    RESPONSE_TIMEOUT_S_DEFAULT,
    GrblCommandError,
    GrblFileSender,
    GrblResetError,
    GrblSenderError,
    GrblTimeoutError,
    JsonRunLogger,
    StreamSettings,
    cap_gcode_feedrate,
    detect_grbl_port,
    normalize_gcode_line,
    open_serial_port,
    require_pyserial,
)
from bee_counter_bench.acquisition.position_tracker import (
    BenchDaqPositionTracker,
    PositionTrackerError,
    TrackerSnapshot,
)


WORD_PATTERN = re.compile(r"([A-Z])([-+]?(?:\d+(?:\.\d*)?|\.\d+))")
MOTION_CODES = {0, 1, 2, 3}
UNSUPPORTED_AXES = {"Y", "Z", "A", "B", "C"}
ACQUISITION_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "acquisition"
CHECKPOINT_PATH_DEFAULT = ACQUISITION_OUTPUT_DIR / "cnc_checkpoint.json"
LOG_PATH_DEFAULT = ACQUISITION_OUTPUT_DIR / "logs" / "resilient_last_run.log"
FAULT_STATS_PATH_DEFAULT = (
    ACQUISITION_OUTPUT_DIR / "logs" / "resilient_fault_stats_history.jsonl"
)
FIRMWARE_SETTINGS_PATH_DEFAULT = Path(__file__).with_name("firmware.settings")
TRACKER_READY_TIMEOUT_S_DEFAULT = 3.0
MAX_RECOVERY_ATTEMPTS_DEFAULT = 100
CHECKPOINT_REPLACE_ATTEMPTS_DEFAULT = 10
CHECKPOINT_REPLACE_RETRY_S_DEFAULT = 0.05
RETURN_ZERO_TOLERANCE_MM = 0.01


class ResilientSenderError(GrblSenderError):
    """Raised when the job cannot be resumed safely."""


class PositionTracker(Protocol):
    """Minimal tracker contract used by the resilient sender."""

    def start(self) -> None:
        """Start background tracking."""

    def stop(self) -> None:
        """Stop background tracking."""

    def snapshot(self) -> TrackerSnapshot:
        """Return the latest X-axis tracker snapshot."""

    def wait_until_live(self, timeout_s: float) -> bool:
        """Wait until the tracker has consumed live hardware data."""


@dataclass(frozen=True)
class ResumableCommand:
    """One executable G-code command plus its source file position."""

    command_index: int
    source_line: int
    char_offset: int
    command: str


@dataclass(frozen=True)
class FirmwareSettingMismatch:
    """One GRBL firmware setting that differs from the expected value."""

    key: str
    expected: str
    actual: str | None


@dataclass(frozen=True)
class FirmwareSettingsGuard:
    """Verify and restore controller settings after reconnects."""

    expected: dict[str, str]
    source_path: Path

    def enforce(self, sender: GrblFileSender, logger: JsonRunLogger) -> None:
        """Read, correct, and verify GRBL settings against the export."""
        observed = sender.read_settings()
        mismatches = self._find_mismatches(observed)
        if not mismatches:
            logger.emit(
                "INFO",
                "firmware_settings",
                "GRBL firmware settings match expected export",
                settings_path=str(self.source_path),
                checked_settings=len(self.expected),
            )
            return

        logger.emit(
            "WARNING",
            "firmware_settings",
            "Correcting divergent GRBL firmware settings",
            error_code="GRBL_SETTINGS_DIVERGED",
            settings_path=str(self.source_path),
            mismatches=[asdict(item) for item in mismatches],
        )
        for mismatch in mismatches:
            sender.send_command(
                f"{mismatch.key}={mismatch.expected}",
                context="settings_restore",
            )

        verified = sender.read_settings()
        remaining = self._find_mismatches(verified)
        if remaining:
            raise ResilientSenderError(
                "GRBL firmware settings still diverge after restore: "
                + ", ".join(
                    f"{item.key} expected {item.expected} got {item.actual}"
                    for item in remaining
                )
            )
        logger.emit(
            "INFO",
            "firmware_settings",
            "GRBL firmware settings restored and verified",
            settings_path=str(self.source_path),
            restored_settings=len(mismatches),
        )

    def _find_mismatches(
        self, observed: dict[str, str]
    ) -> list[FirmwareSettingMismatch]:
        mismatches: list[FirmwareSettingMismatch] = []
        for key, expected in sorted(self.expected.items(), key=_setting_sort_key):
            actual = observed.get(key)
            if actual is None or not firmware_setting_values_match(actual, expected):
                mismatches.append(
                    FirmwareSettingMismatch(
                        key=key,
                        expected=expected,
                        actual=actual,
                    )
                )
        return mismatches


@dataclass(frozen=True)
class ResumeModalSnapshot:
    """Safe modal state that can be replayed after a GRBL reset."""

    coordinate_system: str | None = None
    plane: str | None = None
    units: str | None = None
    distance_mode: str | None = None
    feed_mode: str | None = None
    path_mode: str | None = None
    feedrate: str | None = None
    spindle_speed: str | None = None
    spindle_state: str | None = None
    coolant_state: str | None = None

    def to_commands(self) -> list[str]:
        """Return replayable modal commands in a stable order."""
        commands: list[str] = []
        for value in (
            self.coordinate_system,
            self.plane,
            self.units,
            self.distance_mode,
            self.feed_mode,
            self.path_mode,
            self.feedrate,
            self.spindle_speed,
            self.spindle_state,
            self.coolant_state,
        ):
            if value is not None:
                commands.append(value)
        return commands


@dataclass(frozen=True)
class CheckpointState:
    """Durable state required to resume the job after a reset or crash."""

    gcode_file: str
    last_acknowledged_command_index: int
    last_acknowledged_line_number: int | None
    last_acknowledged_char_offset: int | None
    inflight_command_index: int | None
    inflight_line_number: int | None
    inflight_char_offset: int | None
    last_known_x_mm: float | None
    recovery_count: int
    completed: bool
    updated_at: str


@dataclass(frozen=True)
class RecoveryRunStats:
    """Human-readable execution summary."""

    total_commands: int
    confirmed_commands: int
    recovery_count: int
    elapsed_s: float


@dataclass(frozen=True)
class CommandFaultEvent:
    """One stream interruption attributed to a specific G-code command."""

    timestamp: str
    error_code: str
    source_line: int
    command_index: int
    char_offset: int
    command: str
    error: str
    recovery_count: int
    tracked_x_mm: float | None


class CommandFaultStatsCollector:
    """Collect per-command stream failures for postmortem analysis."""

    def __init__(self, *, gcode_file: Path, total_commands: int) -> None:
        self._gcode_file = str(gcode_file)
        self._total_commands = total_commands
        self._events: list[CommandFaultEvent] = []

    def record_fault(
        self,
        *,
        error_code: str,
        command: ResumableCommand,
        error: str,
        recovery_count: int,
        tracked_x_mm: float | None,
    ) -> None:
        self._events.append(
            CommandFaultEvent(
                timestamp=datetime.now(UTC).isoformat(),
                error_code=error_code,
                source_line=command.source_line,
                command_index=command.command_index,
                char_offset=command.char_offset,
                command=command.command,
                error=error,
                recovery_count=recovery_count,
                tracked_x_mm=tracked_x_mm,
            )
        )

    def build_report(
        self,
        *,
        run_status: str,
        confirmed_commands: int | None,
        recovery_count: int | None,
        failure_error_code: str | None,
        failure_error: str | None,
    ) -> dict[str, Any]:
        grouped: dict[tuple[int, int, int, str], dict[str, Any]] = {}
        for event in self._events:
            key = (
                event.command_index,
                event.source_line,
                event.char_offset,
                event.command,
            )
            entry = grouped.get(key)
            if entry is None:
                entry = {
                    "command_index": event.command_index,
                    "source_line": event.source_line,
                    "char_offset": event.char_offset,
                    "command": event.command,
                    "occurrences": 0,
                    "error_counts": {},
                    "first_seen_at": event.timestamp,
                    "last_seen_at": event.timestamp,
                }
                grouped[key] = entry
            entry["occurrences"] = int(entry["occurrences"]) + 1
            error_counts = entry["error_counts"]
            error_counts[event.error_code] = (
                int(error_counts.get(event.error_code, 0)) + 1
            )
            entry["last_seen_at"] = event.timestamp

        faulty_commands = sorted(
            grouped.values(),
            key=lambda item: (
                -int(item["occurrences"]),
                int(item["source_line"]),
                int(item["command_index"]),
            ),
        )
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_status": run_status,
            "gcode_file": self._gcode_file,
            "total_commands": self._total_commands,
            "confirmed_commands": confirmed_commands,
            "recovery_count": recovery_count,
            "fault_event_count": len(self._events),
            "unique_faulty_commands": len(faulty_commands),
            "faulty_commands": faulty_commands,
            "fault_events": [asdict(event) for event in self._events],
            "failure_error_code": failure_error_code,
            "failure_error": failure_error,
        }


def append_fault_stats_report(
    path: Path,
    report: dict[str, Any],
    *,
    logger: JsonRunLogger,
) -> None:
    """Append one run fault report to a JSONL history file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
    except OSError as exc:
        logger.emit(
            "WARNING",
            "fault_stats",
            "Could not persist command fault statistics",
            error_code="FAULT_STATS_WRITE_ERROR",
            path=str(path),
            error=str(exc),
        )
        return

    logger.emit(
        "INFO",
        "fault_stats",
        "Command fault statistics updated",
        path=str(path),
        run_status=report.get("run_status"),
        fault_event_count=report.get("fault_event_count"),
        unique_faulty_commands=report.get("unique_faulty_commands"),
    )


class AtomicCheckpointStore:
    """Persist checkpoint state using atomic file replacement."""

    def __init__(
        self,
        path: Path,
        logger: JsonRunLogger,
        *,
        replace_attempts: int = CHECKPOINT_REPLACE_ATTEMPTS_DEFAULT,
        replace_retry_s: float = CHECKPOINT_REPLACE_RETRY_S_DEFAULT,
    ) -> None:
        self._path = path
        self._logger = logger
        self._replace_attempts = max(1, replace_attempts)
        self._replace_retry_s = max(0.0, replace_retry_s)

    @property
    def path(self) -> Path:
        """Return the checkpoint file path."""
        return self._path

    def load(self) -> CheckpointState | None:
        """Load checkpoint state, ignoring corrupt files safely."""
        for candidate_path in self._candidate_paths():
            checkpoint = self._load_candidate(candidate_path)
            if checkpoint is not None:
                if candidate_path != self._path:
                    self._logger.emit(
                        "WARNING",
                        "checkpoint",
                        "Recovered checkpoint state from a pending temp file",
                        error_code="CHECKPOINT_RECOVERED_FROM_TEMP",
                        path=str(candidate_path),
                    )
                return checkpoint
        return None

    def _load_candidate(self, candidate_path: Path) -> CheckpointState | None:
        try:
            raw = json.loads(candidate_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            self._logger.emit(
                "WARNING",
                "checkpoint",
                "Checkpoint file is invalid JSON; ignoring it",
                error_code="CHECKPOINT_INVALID_JSON",
                path=str(candidate_path),
                error=str(exc),
            )
            return None
        except OSError as exc:
            self._logger.emit(
                "WARNING",
                "checkpoint",
                "Checkpoint file could not be read; ignoring it",
                error_code="CHECKPOINT_READ_ERROR",
                path=str(candidate_path),
                error=str(exc),
            )
            return None

        try:
            return CheckpointState(**raw)
        except TypeError as exc:
            self._logger.emit(
                "WARNING",
                "checkpoint",
                "Checkpoint file schema is invalid; ignoring it",
                error_code="CHECKPOINT_SCHEMA_ERROR",
                path=str(candidate_path),
                error=str(exc),
            )
            return None

    def save(self, state: CheckpointState) -> None:
        """Persist the latest checkpoint state with Windows-friendly retries."""
        payload = json.dumps(asdict(state), indent=2, ensure_ascii=False) + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._path.with_suffix(self._path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._replace_with_retries(temporary_path)

    def _candidate_paths(self) -> list[Path]:
        candidates: list[tuple[int, Path]] = []
        for candidate_path in (
            self._path,
            self._path.with_suffix(self._path.suffix + ".tmp"),
        ):
            if not candidate_path.exists():
                continue
            try:
                modified_ns = candidate_path.stat().st_mtime_ns
            except OSError:
                modified_ns = -1
            candidates.append((modified_ns, candidate_path))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [candidate_path for _modified_ns, candidate_path in candidates]

    def _replace_with_retries(self, temporary_path: Path) -> None:
        last_error: PermissionError | None = None
        for attempt in range(1, self._replace_attempts + 1):
            try:
                os.replace(temporary_path, self._path)
                return
            except PermissionError as exc:
                last_error = exc
                self._clear_readonly(self._path)
                self._clear_readonly(temporary_path)
                if attempt < self._replace_attempts:
                    time.sleep(self._replace_retry_s * attempt)
                    continue

        assert last_error is not None
        self._logger.emit(
            "WARNING",
            "checkpoint",
            "Checkpoint target is locked on Windows; keeping the fresh .tmp file for recovery and continuing the job",
            error_code="CHECKPOINT_REPLACE_DEFERRED",
            path=str(self._path),
            temporary_path=str(temporary_path),
            error=str(last_error),
            attempts=self._replace_attempts,
        )

    @staticmethod
    def _clear_readonly(path: Path) -> None:
        try:
            if path.exists():
                os.chmod(path, 0o666)
        except OSError:
            return


def iter_words(command: str) -> list[tuple[str, str]]:
    """Return parsed G-code words from a normalized command string."""
    return [(letter, value) for letter, value in WORD_PATTERN.findall(command)]


def load_firmware_settings_guard(settings_path: Path) -> FirmwareSettingsGuard:
    """Load the exported GRBL settings used as the controller safety baseline."""
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    raw_settings = payload.get("settings")
    if not isinstance(raw_settings, list):
        raise ValueError(
            f"Firmware settings file has no settings list: {settings_path}"
        )

    expected: dict[str, str] = {}
    for item in raw_settings:
        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid firmware setting entry in {settings_path}: {item}"
            )
        key = item.get("key")
        value = item.get("value")
        if not isinstance(key, str) or not re.fullmatch(r"\$\d+", key):
            raise ValueError(f"Invalid firmware setting key in {settings_path}: {key}")
        if not isinstance(value, str):
            raise ValueError(f"Invalid firmware setting value for {key}: {value}")
        expected[key] = value.strip()

    if not expected:
        raise ValueError(f"Firmware settings file is empty: {settings_path}")
    return FirmwareSettingsGuard(expected=expected, source_path=settings_path)


def firmware_setting_values_match(actual: str, expected: str) -> bool:
    """Compare GRBL setting values without treating decimal formatting as drift."""
    actual_clean = actual.strip()
    expected_clean = expected.strip()
    try:
        return Decimal(actual_clean) == Decimal(expected_clean)
    except InvalidOperation:
        return actual_clean == expected_clean


def _setting_sort_key(item: tuple[str, str]) -> int:
    return int(item[0].lstrip("$"))


def load_resumable_commands(
    gcode_path: Path,
    *,
    max_line_chars: int,
    max_feedrate: float | None,
) -> list[ResumableCommand]:
    """Load normalized commands with source offsets for exact resume."""
    commands: list[ResumableCommand] = []
    char_offset = 0
    with gcode_path.open("r", encoding="utf-8") as handle:
        for source_line, raw_line in enumerate(handle, start=1):
            command = normalize_gcode_line(raw_line)
            source_offset = char_offset
            char_offset += len(raw_line)
            if not command:
                continue
            command = cap_gcode_feedrate(command, max_feedrate)
            if len(command) > max_line_chars:
                raise ValueError(
                    f"Line {source_line} has {len(command)} characters after cleanup; "
                    f"limit is {max_line_chars}: {command}"
                )
            commands.append(
                ResumableCommand(
                    command_index=len(commands),
                    source_line=source_line,
                    char_offset=source_offset,
                    command=command,
                )
            )
    if not commands:
        raise ValueError(f"No executable G-code commands found in {gcode_path}")
    return commands


def validate_resumable_commands(commands: Sequence[ResumableCommand]) -> None:
    """Reject G-code patterns that cannot be resumed safely with X-only recovery."""
    distance_mode: str | None = None
    for item in commands:
        words = iter_words(item.command)
        axes = {
            letter for letter, _value in words if letter in {"X", *UNSUPPORTED_AXES}
        }
        g_codes = [int(float(value)) for letter, value in words if letter == "G"]

        if 91 in g_codes:
            raise ResilientSenderError(
                f"Automatic resume does not support incremental mode (G91) at source line {item.source_line}"
            )
        if 92 in g_codes:
            raise ResilientSenderError(
                f"Automatic resume does not support user G92 offsets inside the input file (source line {item.source_line})"
            )
        if 90 in g_codes:
            distance_mode = "G90"
        motion_codes = MOTION_CODES.intersection(g_codes)
        if {2, 3}.intersection(motion_codes):
            raise ResilientSenderError(
                f"Automatic resume does not support arc moves at source line {item.source_line}"
            )
        if motion_codes:
            if distance_mode != "G90":
                raise ResilientSenderError(
                    f"Explicit G90 absolute mode is required before motion resumes safely (source line {item.source_line})"
                )
            if axes & UNSUPPORTED_AXES:
                unsupported = ", ".join(sorted(axes & UNSUPPORTED_AXES))
                raise ResilientSenderError(
                    f"Automatic resume only tracks X-axis motion; unsupported axis word(s) {unsupported} found at source line {item.source_line}"
                )


def collect_modal_snapshot(
    prefix_commands: Sequence[ResumableCommand],
) -> ResumeModalSnapshot:
    """Collect safe modal words that must survive a GRBL reset."""
    snapshot = ResumeModalSnapshot()
    for item in prefix_commands:
        values = dict(iter_words(item.command))
        g_codes = [
            int(float(value))
            for letter, value in iter_words(item.command)
            if letter == "G"
        ]
        m_codes = [
            int(float(value))
            for letter, value in iter_words(item.command)
            if letter == "M"
        ]

        coordinate_system = next((code for code in g_codes if 54 <= code <= 59), None)
        plane = next((code for code in g_codes if code in {17, 18, 19}), None)
        units = next((code for code in g_codes if code in {20, 21}), None)
        distance_mode = next((code for code in g_codes if code in {90, 91}), None)
        feed_mode = next((code for code in g_codes if code in {93, 94}), None)
        path_mode = next((code for code in g_codes if code in {61, 64}), None)
        spindle_state = next((code for code in m_codes if code in {3, 4, 5}), None)
        coolant_state = next((code for code in m_codes if code in {7, 8, 9}), None)

        if coordinate_system is not None:
            snapshot = ResumeModalSnapshot(
                coordinate_system=f"G{coordinate_system}",
                plane=snapshot.plane,
                units=snapshot.units,
                distance_mode=snapshot.distance_mode,
                feed_mode=snapshot.feed_mode,
                path_mode=snapshot.path_mode,
                feedrate=snapshot.feedrate,
                spindle_speed=snapshot.spindle_speed,
                spindle_state=snapshot.spindle_state,
                coolant_state=snapshot.coolant_state,
            )
        if plane is not None:
            snapshot = ResumeModalSnapshot(
                coordinate_system=snapshot.coordinate_system,
                plane=f"G{plane}",
                units=snapshot.units,
                distance_mode=snapshot.distance_mode,
                feed_mode=snapshot.feed_mode,
                path_mode=snapshot.path_mode,
                feedrate=snapshot.feedrate,
                spindle_speed=snapshot.spindle_speed,
                spindle_state=snapshot.spindle_state,
                coolant_state=snapshot.coolant_state,
            )
        if units is not None:
            snapshot = ResumeModalSnapshot(
                coordinate_system=snapshot.coordinate_system,
                plane=snapshot.plane,
                units=f"G{units}",
                distance_mode=snapshot.distance_mode,
                feed_mode=snapshot.feed_mode,
                path_mode=snapshot.path_mode,
                feedrate=snapshot.feedrate,
                spindle_speed=snapshot.spindle_speed,
                spindle_state=snapshot.spindle_state,
                coolant_state=snapshot.coolant_state,
            )
        if distance_mode is not None:
            snapshot = ResumeModalSnapshot(
                coordinate_system=snapshot.coordinate_system,
                plane=snapshot.plane,
                units=snapshot.units,
                distance_mode=f"G{distance_mode}",
                feed_mode=snapshot.feed_mode,
                path_mode=snapshot.path_mode,
                feedrate=snapshot.feedrate,
                spindle_speed=snapshot.spindle_speed,
                spindle_state=snapshot.spindle_state,
                coolant_state=snapshot.coolant_state,
            )
        if feed_mode is not None:
            snapshot = ResumeModalSnapshot(
                coordinate_system=snapshot.coordinate_system,
                plane=snapshot.plane,
                units=snapshot.units,
                distance_mode=snapshot.distance_mode,
                feed_mode=f"G{feed_mode}",
                path_mode=snapshot.path_mode,
                feedrate=snapshot.feedrate,
                spindle_speed=snapshot.spindle_speed,
                spindle_state=snapshot.spindle_state,
                coolant_state=snapshot.coolant_state,
            )
        if path_mode is not None:
            snapshot = ResumeModalSnapshot(
                coordinate_system=snapshot.coordinate_system,
                plane=snapshot.plane,
                units=snapshot.units,
                distance_mode=snapshot.distance_mode,
                feed_mode=snapshot.feed_mode,
                path_mode=f"G{path_mode}",
                feedrate=snapshot.feedrate,
                spindle_speed=snapshot.spindle_speed,
                spindle_state=snapshot.spindle_state,
                coolant_state=snapshot.coolant_state,
            )
        if "F" in values:
            snapshot = ResumeModalSnapshot(
                coordinate_system=snapshot.coordinate_system,
                plane=snapshot.plane,
                units=snapshot.units,
                distance_mode=snapshot.distance_mode,
                feed_mode=snapshot.feed_mode,
                path_mode=snapshot.path_mode,
                feedrate=f"F{values['F']}",
                spindle_speed=snapshot.spindle_speed,
                spindle_state=snapshot.spindle_state,
                coolant_state=snapshot.coolant_state,
            )
        if "S" in values:
            snapshot = ResumeModalSnapshot(
                coordinate_system=snapshot.coordinate_system,
                plane=snapshot.plane,
                units=snapshot.units,
                distance_mode=snapshot.distance_mode,
                feed_mode=snapshot.feed_mode,
                path_mode=snapshot.path_mode,
                feedrate=snapshot.feedrate,
                spindle_speed=f"S{values['S']}",
                spindle_state=snapshot.spindle_state,
                coolant_state=snapshot.coolant_state,
            )
        if spindle_state is not None:
            snapshot = ResumeModalSnapshot(
                coordinate_system=snapshot.coordinate_system,
                plane=snapshot.plane,
                units=snapshot.units,
                distance_mode=snapshot.distance_mode,
                feed_mode=snapshot.feed_mode,
                path_mode=snapshot.path_mode,
                feedrate=snapshot.feedrate,
                spindle_speed=snapshot.spindle_speed,
                spindle_state=f"M{spindle_state}",
                coolant_state=snapshot.coolant_state,
            )
        if coolant_state is not None:
            snapshot = ResumeModalSnapshot(
                coordinate_system=snapshot.coordinate_system,
                plane=snapshot.plane,
                units=snapshot.units,
                distance_mode=snapshot.distance_mode,
                feed_mode=snapshot.feed_mode,
                path_mode=snapshot.path_mode,
                feedrate=snapshot.feedrate,
                spindle_speed=snapshot.spindle_speed,
                spindle_state=snapshot.spindle_state,
                coolant_state=f"M{coolant_state}",
            )
    return snapshot


def format_axis_value(value: float) -> str:
    """Format a GRBL axis value without scientific notation."""
    if abs(value) < 0.0000005:
        return "0"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def serial_exception_types() -> tuple[type[BaseException], ...]:
    """Return the serial exception types used by pyserial on this host."""
    try:
        serial_module, _list_ports_module = require_pyserial()
    except RuntimeError:
        return (OSError,)
    serial_exception = getattr(serial_module, "SerialException", OSError)
    timeout_exception = getattr(serial_module, "SerialTimeoutException", OSError)
    return (serial_exception, timeout_exception, OSError)


class ResilientGrblJobRunner:
    """Coordinate streaming, checkpointing, and reset recovery."""

    def __init__(
        self,
        *,
        gcode_file: Path,
        port: str,
        baudrate: int,
        commands: Sequence[ResumableCommand],
        checkpoint_store: AtomicCheckpointStore,
        tracker: PositionTracker,
        logger: JsonRunLogger,
        settings: StreamSettings,
        idle_timeout_s: float,
        wait_for_idle: bool,
        tracker_ready_timeout_s: float,
        progress_interval: int,
        max_recovery_attempts: int,
        unlock_after_reset: bool,
        firmware_settings_guard: FirmwareSettingsGuard | None = None,
        fault_stats_collector: CommandFaultStatsCollector | None = None,
        open_serial: Callable[[str, int, JsonRunLogger], Any] = open_serial_port,
    ) -> None:
        self._gcode_file = gcode_file
        self._port = port
        self._baudrate = baudrate
        self._commands = list(commands)
        self._checkpoint_store = checkpoint_store
        self._tracker = tracker
        self._logger = logger
        self._settings = settings
        self._idle_timeout_s = idle_timeout_s
        self._wait_for_idle = wait_for_idle
        self._tracker_ready_timeout_s = tracker_ready_timeout_s
        self._progress_interval = max(1, progress_interval)
        self._max_recovery_attempts = max_recovery_attempts
        self._unlock_after_reset = unlock_after_reset
        self._firmware_settings_guard = firmware_settings_guard
        self._fault_stats_collector = fault_stats_collector
        self._open_serial = open_serial

    def run(self, checkpoint: CheckpointState | None = None) -> RecoveryRunStats:
        """Run or resume the job until completion or a non-recoverable fault."""
        confirmed_commands = 0
        recovery_count = 0
        checkpoint_state = checkpoint or self._fresh_checkpoint()
        next_command_index = checkpoint_state.last_acknowledged_command_index + 1
        if checkpoint_state.last_acknowledged_command_index >= 0:
            confirmed_commands = checkpoint_state.last_acknowledged_command_index + 1
            recovery_count = checkpoint_state.recovery_count

        serial_port: Any | None = None
        sender: GrblFileSender | None = None
        recording_session_dir: Any | None = None
        start_time = time.monotonic()
        self._tracker.start()
        try:
            recording_session_dir = getattr(
                self._tracker, "recording_session_dir", None
            )
            if recording_session_dir is not None:
                self._logger.emit(
                    "INFO",
                    "recording",
                    "Bench recording session started",
                    output_dir=str(recording_session_dir),
                )
            if not self._tracker.wait_until_live(self._tracker_ready_timeout_s):
                tracked_x_mm = self._tracked_position(
                    fallback_position_mm=checkpoint_state.last_known_x_mm,
                    require_live=False,
                )
                self._logger.emit(
                    "WARNING",
                    "tracker",
                    "X-axis tracker did not report live data before streaming; using the latest seeded position until samples arrive",
                    error_code="TRACKER_NOT_LIVE",
                    tracker_ready_timeout_s=self._tracker_ready_timeout_s,
                    fallback_x_mm=tracked_x_mm,
                )
            serial_port, sender = self._connect_sender()
            if next_command_index > 0:
                sender = self._recover_sender(
                    sender=sender,
                    next_command_index=next_command_index,
                    recovery_count=recovery_count,
                    fallback_position_mm=checkpoint_state.last_known_x_mm,
                    resume_reason="checkpoint resume",
                )
            else:
                sender = self._recover_sender(
                    sender=sender,
                    next_command_index=0,
                    recovery_count=recovery_count,
                    fallback_position_mm=checkpoint_state.last_known_x_mm,
                    resume_reason="initial alignment",
                )

            while next_command_index < len(self._commands):
                command = self._commands[next_command_index]
                checkpoint_state = self._mark_inflight(
                    checkpoint_state,
                    command,
                    recovery_count,
                )
                self._checkpoint_store.save(checkpoint_state)
                try:
                    assert sender is not None
                    sender.send_stream_command(command.source_line, command.command)
                except GrblResetError as exc:
                    recovery_count += 1
                    self._logger.emit(
                        "WARNING",
                        "stream",
                        "GRBL reset detected mid-stream; attempting recovery",
                        error_code="GRBL_RESET_DURING_STREAM",
                        error=str(exc),
                        source_line=command.source_line,
                        command_index=command.command_index,
                        command=command.command,
                    )
                    self._record_fault(
                        error_code="GRBL_RESET_DURING_STREAM",
                        command=command,
                        error=str(exc),
                        recovery_count=recovery_count,
                        tracked_x_mm=checkpoint_state.last_known_x_mm,
                    )
                    self._ensure_recoverable(recovery_count, command)
                    if serial_port is not None:
                        serial_port.close()
                    serial_port, sender = self._connect_sender()
                    sender = self._recover_sender(
                        sender=sender,
                        next_command_index=next_command_index,
                        recovery_count=recovery_count,
                        fallback_position_mm=checkpoint_state.last_known_x_mm,
                        resume_reason="controller reset",
                    )
                    checkpoint_state = self._with_recovery_count(
                        checkpoint_state, recovery_count
                    )
                    self._checkpoint_store.save(checkpoint_state)
                    continue
                except serial_exception_types() as exc:
                    recovery_count += 1
                    self._logger.emit(
                        "WARNING",
                        "stream",
                        "Serial transport dropped; attempting recovery",
                        error_code="SERIAL_DROP",
                        error=str(exc),
                        source_line=command.source_line,
                        command_index=command.command_index,
                        command=command.command,
                    )
                    self._record_fault(
                        error_code="SERIAL_DROP",
                        command=command,
                        error=str(exc),
                        recovery_count=recovery_count,
                        tracked_x_mm=checkpoint_state.last_known_x_mm,
                    )
                    self._ensure_recoverable(recovery_count, command)
                    if serial_port is not None:
                        serial_port.close()
                    serial_port, sender = self._connect_sender()
                    sender = self._recover_sender(
                        sender=sender,
                        next_command_index=next_command_index,
                        recovery_count=recovery_count,
                        fallback_position_mm=checkpoint_state.last_known_x_mm,
                        resume_reason="serial drop",
                    )
                    checkpoint_state = self._with_recovery_count(
                        checkpoint_state, recovery_count
                    )
                    self._checkpoint_store.save(checkpoint_state)
                    continue
                except GrblTimeoutError as exc:
                    recovery_count += 1
                    self._logger.emit(
                        "WARNING",
                        "stream",
                        "GRBL stopped acknowledging mid-stream; attempting recovery",
                        error_code="GRBL_RESPONSE_TIMEOUT",
                        error=str(exc),
                        source_line=command.source_line,
                        command_index=command.command_index,
                        command=command.command,
                    )
                    self._record_fault(
                        error_code="GRBL_RESPONSE_TIMEOUT",
                        command=command,
                        error=str(exc),
                        recovery_count=recovery_count,
                        tracked_x_mm=checkpoint_state.last_known_x_mm,
                    )
                    self._ensure_recoverable(recovery_count, command)
                    if serial_port is not None:
                        serial_port.close()
                    serial_port, sender = self._connect_sender()
                    sender = self._recover_sender(
                        sender=sender,
                        next_command_index=next_command_index,
                        recovery_count=recovery_count,
                        fallback_position_mm=checkpoint_state.last_known_x_mm,
                        resume_reason="response timeout",
                    )
                    checkpoint_state = self._with_recovery_count(
                        checkpoint_state, recovery_count
                    )
                    self._checkpoint_store.save(checkpoint_state)
                    continue

                confirmed_commands += 1
                checkpoint_state = self._mark_acknowledged(
                    checkpoint_state,
                    command,
                    recovery_count,
                )
                self._checkpoint_store.save(checkpoint_state)
                next_command_index += 1
                if (
                    confirmed_commands == 1
                    or confirmed_commands == len(self._commands)
                    or confirmed_commands % self._progress_interval == 0
                ):
                    percent = 100.0 * confirmed_commands / max(1, len(self._commands))
                    self._logger.emit(
                        "INFO",
                        "progress",
                        f"Progress {confirmed_commands}/{len(self._commands)} ({percent:.1f}%)",
                        confirmed_commands=confirmed_commands,
                        total_commands=len(self._commands),
                        percent=round(percent, 3),
                        source_line=command.source_line,
                    )

            if self._wait_for_idle and sender is not None:
                sender.wait_for_idle(timeout_s=self._idle_timeout_s)
            if sender is not None:
                self._return_to_zero_if_needed(
                    sender,
                    fallback_position_mm=checkpoint_state.last_known_x_mm,
                )
                if self._wait_for_idle:
                    sender.wait_for_idle(timeout_s=self._idle_timeout_s)

            final_state = CheckpointState(
                gcode_file=checkpoint_state.gcode_file,
                last_acknowledged_command_index=checkpoint_state.last_acknowledged_command_index,
                last_acknowledged_line_number=checkpoint_state.last_acknowledged_line_number,
                last_acknowledged_char_offset=checkpoint_state.last_acknowledged_char_offset,
                inflight_command_index=None,
                inflight_line_number=None,
                inflight_char_offset=None,
                last_known_x_mm=self._tracked_position(
                    fallback_position_mm=checkpoint_state.last_known_x_mm,
                    require_live=False,
                ),
                recovery_count=recovery_count,
                completed=True,
                updated_at=self._timestamp(),
            )
            self._checkpoint_store.save(final_state)
            elapsed_s = time.monotonic() - start_time
            return RecoveryRunStats(
                total_commands=len(self._commands),
                confirmed_commands=confirmed_commands,
                recovery_count=recovery_count,
                elapsed_s=elapsed_s,
            )
        finally:
            if serial_port is not None:
                serial_port.close()
            self._tracker.stop()
            recording_chunk_count = getattr(
                self._tracker, "recording_chunk_count", None
            )
            recording_total_samples = getattr(
                self._tracker, "recording_total_samples", None
            )
            if recording_session_dir is not None:
                self._logger.emit(
                    "INFO",
                    "recording",
                    "Bench recording session finished",
                    output_dir=str(recording_session_dir),
                    chunks=recording_chunk_count,
                    total_samples=recording_total_samples,
                )

    def _fresh_checkpoint(self) -> CheckpointState:
        return CheckpointState(
            gcode_file=str(self._gcode_file),
            last_acknowledged_command_index=-1,
            last_acknowledged_line_number=None,
            last_acknowledged_char_offset=None,
            inflight_command_index=None,
            inflight_line_number=None,
            inflight_char_offset=None,
            last_known_x_mm=None,
            recovery_count=0,
            completed=False,
            updated_at=self._timestamp(),
        )

    def _connect_sender(self) -> tuple[Any, GrblFileSender]:
        self._logger.emit(
            "INFO",
            "connect",
            "Opening GRBL serial port",
            port=self._port,
            baudrate=self._baudrate,
        )
        serial_port = self._open_serial(self._port, self._baudrate, self._logger)
        sender = GrblFileSender(serial_port, self._logger, self._settings)
        sender.wake()
        sender.send_command("$I", context="version")
        state = sender.query_status()
        self._logger.emit("INFO", "status", "Initial GRBL state", state=state)
        if state == "Alarm" and self._unlock_after_reset:
            sender.send_command("$X", context="unlock")
            state = sender.query_status()
            self._logger.emit("INFO", "status", "GRBL unlocked", state=state)
        if self._firmware_settings_guard is not None:
            self._firmware_settings_guard.enforce(sender, self._logger)
        return serial_port, sender

    def _recover_sender(
        self,
        *,
        sender: GrblFileSender,
        next_command_index: int,
        recovery_count: int,
        fallback_position_mm: float | None,
        resume_reason: str,
    ) -> GrblFileSender:
        tracked_x_mm = self._tracked_position(
            fallback_position_mm=fallback_position_mm,
            require_live=False,
        )
        snapshot = collect_modal_snapshot(self._commands[:next_command_index])
        self._logger.emit(
            "WARNING",
            "recovery",
            "Recovering GRBL stream after interruption",
            error_code="GRBL_RECOVERY",
            reason=resume_reason,
            next_command_index=next_command_index,
            tracked_x_mm=tracked_x_mm,
            recovery_count=recovery_count,
        )
        for command in snapshot.to_commands():
            sender.send_command(command, context="recovery")
        sender.send_command(
            f"G92X{format_axis_value(tracked_x_mm)}",
            context="recovery",
        )
        return sender

    def _return_to_zero_if_needed(
        self,
        sender: GrblFileSender,
        *,
        fallback_position_mm: float | None,
    ) -> None:
        tracked_x_mm = self._tracked_position(
            fallback_position_mm=fallback_position_mm,
            require_live=False,
        )
        if abs(tracked_x_mm) <= RETURN_ZERO_TOLERANCE_MM:
            self._logger.emit(
                "INFO",
                "return_zero",
                "Skipping final return to zero because tracked X is already zero",
                tracked_x_mm=tracked_x_mm,
                tolerance_mm=RETURN_ZERO_TOLERANCE_MM,
            )
            return
        self._logger.emit(
            "INFO",
            "return_zero",
            "Returning CNC X axis to zero after streaming",
            tracked_x_mm=tracked_x_mm,
            tolerance_mm=RETURN_ZERO_TOLERANCE_MM,
        )
        sender.send_command("G90", context="return_zero")
        sender.send_command("G0X0", context="return_zero")

    def _mark_inflight(
        self,
        state: CheckpointState,
        command: ResumableCommand,
        recovery_count: int,
    ) -> CheckpointState:
        return CheckpointState(
            gcode_file=state.gcode_file,
            last_acknowledged_command_index=state.last_acknowledged_command_index,
            last_acknowledged_line_number=state.last_acknowledged_line_number,
            last_acknowledged_char_offset=state.last_acknowledged_char_offset,
            inflight_command_index=command.command_index,
            inflight_line_number=command.source_line,
            inflight_char_offset=command.char_offset,
            last_known_x_mm=self._tracked_position(
                fallback_position_mm=state.last_known_x_mm,
                require_live=False,
            ),
            recovery_count=recovery_count,
            completed=False,
            updated_at=self._timestamp(),
        )

    def _mark_acknowledged(
        self,
        state: CheckpointState,
        command: ResumableCommand,
        recovery_count: int,
    ) -> CheckpointState:
        return CheckpointState(
            gcode_file=state.gcode_file,
            last_acknowledged_command_index=command.command_index,
            last_acknowledged_line_number=command.source_line,
            last_acknowledged_char_offset=command.char_offset,
            inflight_command_index=None,
            inflight_line_number=None,
            inflight_char_offset=None,
            last_known_x_mm=self._tracked_position(
                fallback_position_mm=state.last_known_x_mm,
                require_live=False,
            ),
            recovery_count=recovery_count,
            completed=False,
            updated_at=self._timestamp(),
        )

    def _with_recovery_count(
        self,
        state: CheckpointState,
        recovery_count: int,
    ) -> CheckpointState:
        return CheckpointState(
            gcode_file=state.gcode_file,
            last_acknowledged_command_index=state.last_acknowledged_command_index,
            last_acknowledged_line_number=state.last_acknowledged_line_number,
            last_acknowledged_char_offset=state.last_acknowledged_char_offset,
            inflight_command_index=state.inflight_command_index,
            inflight_line_number=state.inflight_line_number,
            inflight_char_offset=state.inflight_char_offset,
            last_known_x_mm=self._tracked_position(
                fallback_position_mm=state.last_known_x_mm,
                require_live=False,
            ),
            recovery_count=recovery_count,
            completed=state.completed,
            updated_at=self._timestamp(),
        )

    def _tracked_position(
        self,
        *,
        fallback_position_mm: float | None,
        require_live: bool,
    ) -> float:
        snapshot = self._tracker.snapshot()
        if snapshot.error is not None:
            raise PositionTrackerError(snapshot.error)
        if snapshot.live_data_seen or not require_live:
            return snapshot.position_mm
        if fallback_position_mm is not None:
            return fallback_position_mm
        raise PositionTrackerError(
            "The X-axis tracker has not produced any live position yet"
        )

    def _ensure_recoverable(
        self,
        recovery_count: int,
        command: ResumableCommand,
    ) -> None:
        if recovery_count > self._max_recovery_attempts:
            raise ResilientSenderError(
                f"Exceeded maximum recovery attempts while retrying source line {command.source_line}"
            )

    def _record_fault(
        self,
        *,
        error_code: str,
        command: ResumableCommand,
        error: str,
        recovery_count: int,
        tracked_x_mm: float | None,
    ) -> None:
        if self._fault_stats_collector is None:
            return
        self._fault_stats_collector.record_fault(
            error_code=error_code,
            command=command,
            error=error,
            recovery_count=recovery_count,
            tracked_x_mm=tracked_x_mm,
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat()


def load_resume_checkpoint(
    checkpoint_store: AtomicCheckpointStore,
    *,
    gcode_path: Path,
    logger: JsonRunLogger,
) -> CheckpointState | None:
    """Return a compatible checkpoint for the current input file, if any."""
    checkpoint = checkpoint_store.load()
    if checkpoint is None:
        return None
    resolved_path = str(gcode_path.resolve())
    if checkpoint.gcode_file != resolved_path:
        logger.emit(
            "INFO",
            "checkpoint",
            "Ignoring checkpoint from another G-code file",
            checkpoint_file=checkpoint.gcode_file,
            requested_file=resolved_path,
        )
        return None
    if checkpoint.completed:
        logger.emit(
            "INFO",
            "checkpoint",
            "Ignoring completed checkpoint for a fresh run",
            checkpoint_file=checkpoint.gcode_file,
        )
        return None
    return checkpoint


def format_recovery_stats(
    stats: RecoveryRunStats,
    log_path: Path,
    checkpoint_path: Path,
    fault_stats_path: Path,
) -> str:
    """Return a compact summary for CLI users."""
    return (
        f"Streaming finished: confirmed={stats.confirmed_commands}/{stats.total_commands}, "
        f"recoveries={stats.recovery_count}, elapsed={stats.elapsed_s:.2f}s, "
        f"log={log_path}, checkpoint={checkpoint_path}, fault_stats={fault_stats_path}"
    )


def estimate_checkpoint_progress(
    checkpoint_store: AtomicCheckpointStore | None,
) -> tuple[int | None, int | None]:
    """Estimate confirmed commands and recovery count from saved checkpoint state."""
    if checkpoint_store is None:
        return None, None
    checkpoint = checkpoint_store.load()
    if checkpoint is None:
        return None, None
    return max(
        0, checkpoint.last_acknowledged_command_index + 1
    ), checkpoint.recovery_count


def build_parser() -> argparse.ArgumentParser:
    """Build the resilient sender CLI parser."""
    parser = argparse.ArgumentParser(
        description="Fault-tolerant GRBL sender with X-axis hardware tracking"
    )
    parser.add_argument("gcode_file", type=Path, help="G-code file to stream")
    parser.add_argument("--port", help="Serial port, e.g. COM10")
    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="Probe serial ports and select the first GRBL controller",
    )
    parser.add_argument(
        "--baudrate", type=int, default=BAUDRATE_DEFAULT, help="Serial baudrate"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=CHECKPOINT_PATH_DEFAULT,
        help="JSON checkpoint file used for resume",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=LOG_PATH_DEFAULT,
        help="JSONL execution log path",
    )
    parser.add_argument(
        "--fault-stats-path",
        type=Path,
        default=FAULT_STATS_PATH_DEFAULT,
        help="Append-only JSONL history with per-command fault statistics",
    )
    parser.add_argument(
        "--settings-path",
        type=Path,
        default=SETTINGS_PATH,
        help="Public experiment configuration used by the bench tracker",
    )
    parser.add_argument(
        "--firmware-settings-path",
        type=Path,
        default=FIRMWARE_SETTINGS_PATH_DEFAULT,
        help="Exported GRBL firmware settings JSON used to verify and restore controller settings",
    )
    parser.add_argument(
        "--no-firmware-settings-guard",
        action="store_true",
        help="Disable automatic GRBL firmware settings verification and restore on connect/reconnect",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Bench recording output directory, defaulting to the public configuration",
    )
    parser.add_argument(
        "--trips",
        type=int,
        help="Round-trips per NPZ chunk, defaulting to the public configuration",
    )
    parser.add_argument(
        "--tracker-ready-timeout",
        type=float,
        default=TRACKER_READY_TIMEOUT_S_DEFAULT,
        help="Seconds to wait for live tracker data before streaming",
    )
    parser.add_argument(
        "--initial-x-mm",
        type=float,
        default=0.0,
        help="Seed X position when no checkpoint exists yet",
    )
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=RESPONSE_TIMEOUT_S_DEFAULT,
        help="Seconds to wait for each GRBL acknowledgement",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=0.0,
        help="Seconds to wait for Idle after the last command; 0 means no timeout",
    )
    parser.add_argument(
        "--no-wait-idle",
        action="store_true",
        help="Skip the final Idle wait after the last command",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=PROGRESS_INTERVAL_DEFAULT,
        help="Commands between progress log entries",
    )
    parser.add_argument(
        "--max-recovery-attempts",
        type=int,
        default=MAX_RECOVERY_ATTEMPTS_DEFAULT,
        help="Maximum automatic recovery attempts before aborting",
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
        "--ignore-checkpoint",
        action="store_true",
        help="Ignore any existing checkpoint and start from the beginning",
    )
    parser.add_argument(
        "--unlock-after-reset",
        action="store_true",
        help="Send $X after reconnect if GRBL comes back in Alarm",
    )
    parser.add_argument(
        "--no-recording",
        action="store_true",
        help="Disable NPZ recording and keep only the X-axis tracker active",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate G-code and checkpoint compatibility without opening serial",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Do not echo INFO logs to console"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the resilient sender CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = JsonRunLogger(args.log_path, echo=not args.quiet)
    checkpoint_store: AtomicCheckpointStore | None = None
    fault_stats_collector: CommandFaultStatsCollector | None = None
    run_stats: RecoveryRunStats | None = None
    run_error: Exception | None = None
    try:
        gcode_path = args.gcode_file.resolve()
        commands = load_resumable_commands(
            gcode_path,
            max_line_chars=args.max_line_chars,
            max_feedrate=args.max_feedrate,
        )
        validate_resumable_commands(commands)
        fault_stats_collector = CommandFaultStatsCollector(
            gcode_file=gcode_path,
            total_commands=len(commands),
        )
        logger.emit(
            "INFO",
            "gcode",
            "G-code loaded for resilient streaming",
            file=str(gcode_path),
            commands=len(commands),
        )

        checkpoint_store = AtomicCheckpointStore(args.checkpoint_path, logger)
        checkpoint = None
        if not args.ignore_checkpoint:
            checkpoint = load_resume_checkpoint(
                checkpoint_store,
                gcode_path=gcode_path,
                logger=logger,
            )

        initial_x_mm = args.initial_x_mm
        if checkpoint is not None and checkpoint.last_known_x_mm is not None:
            initial_x_mm = checkpoint.last_known_x_mm

        if args.dry_run:
            print(
                f"Validated {len(commands)} resumable commands from {gcode_path}; checkpoint={args.checkpoint_path}"
            )
            return 0

        if not args.port and not args.auto_port:
            parser.error("Use --port COMx or --auto-port")
        port = args.port or detect_grbl_port(args.baudrate, logger)
        settings = StreamSettings(
            response_timeout_s=args.response_timeout,
            progress_interval=args.progress_interval,
        )
        firmware_settings_guard = None
        if not args.no_firmware_settings_guard:
            firmware_settings_guard = load_firmware_settings_guard(
                args.firmware_settings_path.resolve()
            )
            logger.emit(
                "INFO",
                "firmware_settings",
                "Loaded GRBL firmware settings guard",
                settings_path=str(firmware_settings_guard.source_path),
                expected_settings=len(firmware_settings_guard.expected),
            )
        tracker = BenchDaqPositionTracker(
            settings_path=args.settings_path.resolve(),
            initial_position_mm=initial_x_mm,
            recordings_dir=args.output_dir,
            recording_session_prefix=gcode_path.stem,
            trips_per_chunk=args.trips,
            enable_recording=not args.no_recording,
        )
        runner = ResilientGrblJobRunner(
            gcode_file=gcode_path,
            port=port,
            baudrate=args.baudrate,
            commands=commands,
            checkpoint_store=checkpoint_store,
            tracker=tracker,
            logger=logger,
            settings=settings,
            idle_timeout_s=args.idle_timeout,
            wait_for_idle=not args.no_wait_idle,
            tracker_ready_timeout_s=args.tracker_ready_timeout,
            progress_interval=args.progress_interval,
            max_recovery_attempts=args.max_recovery_attempts,
            unlock_after_reset=args.unlock_after_reset,
            firmware_settings_guard=firmware_settings_guard,
            fault_stats_collector=fault_stats_collector,
        )
        run_stats = runner.run(
            checkpoint=CheckpointState(
                gcode_file=str(gcode_path),
                last_acknowledged_command_index=-1
                if checkpoint is None
                else checkpoint.last_acknowledged_command_index,
                last_acknowledged_line_number=None
                if checkpoint is None
                else checkpoint.last_acknowledged_line_number,
                last_acknowledged_char_offset=None
                if checkpoint is None
                else checkpoint.last_acknowledged_char_offset,
                inflight_command_index=None
                if checkpoint is None
                else checkpoint.inflight_command_index,
                inflight_line_number=None
                if checkpoint is None
                else checkpoint.inflight_line_number,
                inflight_char_offset=None
                if checkpoint is None
                else checkpoint.inflight_char_offset,
                last_known_x_mm=None
                if checkpoint is None
                else checkpoint.last_known_x_mm,
                recovery_count=0 if checkpoint is None else checkpoint.recovery_count,
                completed=False if checkpoint is None else checkpoint.completed,
                updated_at=datetime.now(UTC).isoformat(),
            )
        )
        print(
            format_recovery_stats(
                run_stats,
                log_path=logger.path,
                checkpoint_path=checkpoint_store.path,
                fault_stats_path=args.fault_stats_path,
            )
        )
        return 0
    except (
        FileNotFoundError,
        PositionTrackerError,
        ResilientSenderError,
        GrblCommandError,
        GrblResetError,
        GrblTimeoutError,
        GrblSenderError,
        RuntimeError,
        ValueError,
    ) as exc:
        run_error = exc
        logger.emit(
            "ERROR",
            "main",
            "Resilient sender failed",
            error_code=exc.__class__.__name__,
            error=str(exc),
        )
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Log: {logger.path}", file=sys.stderr)
        return 2
    finally:
        if fault_stats_collector is not None and not args.dry_run:
            if run_stats is None:
                confirmed_commands, recovery_count = estimate_checkpoint_progress(
                    checkpoint_store
                )
            else:
                confirmed_commands = run_stats.confirmed_commands
                recovery_count = run_stats.recovery_count

            report = fault_stats_collector.build_report(
                run_status="completed"
                if run_error is None and run_stats is not None
                else "failed",
                confirmed_commands=confirmed_commands,
                recovery_count=recovery_count,
                failure_error_code=None
                if run_error is None
                else run_error.__class__.__name__,
                failure_error=None if run_error is None else str(run_error),
            )
            append_fault_stats_report(
                args.fault_stats_path,
                report,
                logger=logger,
            )
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
