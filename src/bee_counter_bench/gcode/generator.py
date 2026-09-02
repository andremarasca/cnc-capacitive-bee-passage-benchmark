"""Generate self-contained CNC G-code for the counter stimulus bench."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bee_counter_bench.bench_config import load_app_config


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GcodeRunConfig:
    """Runtime configuration for local G-code generation."""

    output_dir: Path
    travel_min_mm: float
    travel_max_mm: float
    step_mm: float
    min_segment_time_s: float
    dt_s: float
    v_min_mm_min: float
    v_max_mm_min: float
    leg_pause_s: float
    v_step_mm_min: float
    sweep_decrement_mm_min: float
    desired_hours: float
    seed: int
    max_file_duration_hours: float
    max_file_line_count: int
    monte_carlo_leg_count: int
    cnc_overhead_factor: float
    ou_alpha_v0: float
    ou_lambda_range: tuple[float, float]
    ou_sigma_range: tuple[float, float]
    ou_kappa: float
    ou_margin_frac: float
    ou_bee_a_max_mm_s2: float
    ou_accel_lpf_cutoff_hz: float

    @property
    def travel_distance_mm(self) -> float:
        """Return the absolute travel distance of one leg."""
        return self.travel_max_mm - self.travel_min_mm

    @property
    def max_file_duration_s(self) -> float:
        """Return the configured maximum G-code file duration in seconds."""
        return self.max_file_duration_hours * 3600.0


@dataclass(frozen=True)
class GcodeBlock:
    """One indivisible G-code block used for file fragmentation."""

    lines: tuple[str, ...]
    duration_s: float
    start_label: float | None = None
    end_label: float | None = None


@dataclass(frozen=True)
class GcodeFragment:
    """A complete G-code file body assembled from indivisible blocks."""

    blocks: tuple[GcodeBlock, ...]
    duration_s: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments using YAML defaults."""
    app_config = load_app_config()
    parser = argparse.ArgumentParser(description="Generate bench G-code locally.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=app_config.paths.gcode_output_dir,
        help="Directory for generated G-code files.",
    )
    parser.add_argument(
        "--v-min",
        type=float,
        default=app_config.gcode.v_min_mm_min,
        help="Minimum feedrate in mm/min.",
    )
    parser.add_argument(
        "--v-max",
        type=float,
        default=app_config.gcode.v_max_mm_min,
        help="Maximum feedrate in mm/min.",
    )
    parser.add_argument(
        "--leg-pause",
        type=float,
        default=app_config.gcode.leg_pause_s,
        help="Pause between legs in seconds.",
    )
    parser.add_argument(
        "--v-step",
        type=float,
        default=app_config.gcode.v_step_mm_min,
        help="Velocity step for ramp combinations in mm/min.",
    )
    parser.add_argument(
        "--sweep-decrement",
        type=float,
        default=app_config.gcode.sweep_decrement_mm_min,
        help="Velocity decrement for constant sweeps in mm/min.",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=app_config.gcode.desired_hours,
        help="Target duration for the random trajectory program.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=app_config.gcode.seed,
        help="Random seed for OU trajectory sampling.",
    )
    parser.add_argument(
        "--monte-carlo-leg-count",
        type=int,
        default=app_config.gcode.monte_carlo_leg_count,
        help="Number of simulated legs used to estimate random program duration.",
    )
    parser.add_argument(
        "--max-file-hours",
        type=float,
        default=app_config.gcode.max_file_duration_hours,
        help="Maximum estimated duration per generated G-code file.",
    )
    parser.add_argument(
        "--max-file-lines",
        type=int,
        default=app_config.gcode.max_file_line_count,
        help="Maximum lines per generated G-code file when block boundaries allow.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> GcodeRunConfig:
    """Build the runtime configuration from YAML defaults and CLI overrides."""
    app_config = load_app_config()
    geometry = app_config.geometry
    gcode = app_config.gcode
    synthetic = app_config.synthetic_generation
    output_dir = args.output_dir.resolve()
    return GcodeRunConfig(
        output_dir=output_dir,
        travel_min_mm=geometry.travel_min_mm,
        travel_max_mm=geometry.travel_max_mm,
        step_mm=gcode.step_mm,
        min_segment_time_s=gcode.min_segment_time_s,
        dt_s=gcode.dt_s,
        v_min_mm_min=args.v_min,
        v_max_mm_min=args.v_max,
        leg_pause_s=args.leg_pause,
        v_step_mm_min=args.v_step,
        sweep_decrement_mm_min=args.sweep_decrement,
        desired_hours=args.hours,
        seed=args.seed,
        max_file_duration_hours=args.max_file_hours,
        max_file_line_count=args.max_file_lines,
        monte_carlo_leg_count=args.monte_carlo_leg_count,
        cnc_overhead_factor=gcode.cnc_overhead_factor,
        ou_alpha_v0=synthetic.ou_alpha_v0,
        ou_lambda_range=synthetic.ou_lambda_range,
        ou_sigma_range=synthetic.ou_sigma_range,
        ou_kappa=synthetic.ou_kappa,
        ou_margin_frac=synthetic.ou_margin_frac,
        ou_bee_a_max_mm_s2=synthetic.ou_bee_a_max_mm_s2,
        ou_accel_lpf_cutoff_hz=synthetic.ou_accel_lpf_cutoff_hz,
    )


def gcode_header(description: str, config: GcodeRunConfig) -> list[str]:
    """Return the G-code preamble lines."""
    return [
        f"; {description}",
        (
            f"; Travel range: [{config.travel_min_mm:.2f}, {config.travel_max_mm:.2f}] mm"
            f" | Distance: {config.travel_distance_mm:.2f} mm"
            f" | Step: {config.step_mm:.2f} mm"
        ),
        (
            f"; Feedrate range: [{config.v_min_mm_min:.0f}, {config.v_max_mm_min:.0f}] mm/min"
        ),
        f"; Leg pause: G4 P{config.leg_pause_s:.1f}",
        f"; Seed: {config.seed}",
        "G21 ; Metric mode",
        "G90 ; Absolute positioning",
        "",
    ]


def simulate_ou_trajectory(
    rng: np.random.Generator,
    config: GcodeRunConfig,
) -> dict[str, np.ndarray]:
    """Simulate one OU-driven trajectory over the configured travel distance."""
    v_min_s = config.v_min_mm_min / 60.0
    v_max_s = config.v_max_mm_min / 60.0
    initial_blend = rng.uniform() ** config.ou_alpha_v0
    velocity = v_min_s + (v_max_s - v_min_s) * initial_blend
    lambda_value = rng.uniform(*config.ou_lambda_range)
    sigma_value = rng.uniform(*config.ou_sigma_range)

    margin = config.ou_margin_frac * (v_max_s - v_min_s)
    v_hi_thresh = v_max_s - margin
    v_lo_thresh = v_min_s + margin
    alpha_lpf = config.dt_s / (
        config.dt_s + 1.0 / (2.0 * np.pi * config.ou_accel_lpf_cutoff_hz)
    )

    time_value = 0.0
    position_value = 0.0
    ou_acceleration = 0.0
    filtered_acceleration = 0.0
    times = [time_value]
    positions = [position_value]
    velocities = [velocity]

    max_steps = int(60.0 / config.dt_s)
    for _ in range(max_steps):
        noise = rng.standard_normal()
        ou_acceleration = (
            ou_acceleration * (1.0 - lambda_value * config.dt_s)
            + sigma_value * np.sqrt(config.dt_s) * noise
        )

        boundary_force = 0.0
        if velocity > v_hi_thresh and margin > 0:
            boundary_force -= config.ou_kappa * ((velocity - v_hi_thresh) / margin) ** 2
        if velocity < v_lo_thresh and margin > 0:
            boundary_force += config.ou_kappa * ((v_lo_thresh - velocity) / margin) ** 2

        filtered_acceleration = (
            alpha_lpf * (ou_acceleration + boundary_force)
            + (1.0 - alpha_lpf) * filtered_acceleration
        )
        acceleration = float(
            np.clip(
                filtered_acceleration,
                -config.ou_bee_a_max_mm_s2,
                config.ou_bee_a_max_mm_s2,
            )
        )
        velocity = float(
            np.clip(velocity + acceleration * config.dt_s, v_min_s, v_max_s)
        )
        position_value += max(velocity, 0.0) * config.dt_s
        time_value += config.dt_s

        times.append(time_value)
        positions.append(position_value)
        velocities.append(velocity)

        if position_value >= config.travel_distance_mm:
            break

    return {
        "time": np.array(times, dtype=np.float64),
        "position": np.array(positions, dtype=np.float64),
        "velocity": np.array(velocities, dtype=np.float64),
    }


def compute_ramp_trajectory(
    v_start_mm_min: float,
    v_end_mm_min: float,
    config: GcodeRunConfig,
) -> dict[str, np.ndarray]:
    """Compute a deterministic ramp trajectory over the configured distance."""
    sample_x = np.arange(0.0, config.travel_distance_mm + 1e-9, config.step_mm)
    frac = sample_x / config.travel_distance_mm
    v_mm_min = v_start_mm_min + (v_end_mm_min - v_start_mm_min) * frac
    v_mm_s = v_mm_min / 60.0

    dt_arr = np.zeros(len(sample_x), dtype=np.float64)
    dt_arr[1:] = config.step_mm / v_mm_s[1:]
    return {
        "position": sample_x,
        "velocity": v_mm_s,
        "time": np.cumsum(dt_arr),
    }


def _adaptive_sample_points(
    traj: dict[str, np.ndarray],
    config: GcodeRunConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample trajectory waypoints with spacing adapted to feedrate."""
    positions: list[float] = []
    feedrates: list[float] = []
    sample_position = config.step_mm

    while sample_position <= config.travel_distance_mm + 1e-6:
        velocity_mm_s = float(
            np.interp(sample_position, traj["position"], traj["velocity"])
        )
        feedrate = float(
            np.clip(velocity_mm_s * 60.0, config.v_min_mm_min, config.v_max_mm_min)
        )
        positions.append(min(sample_position, config.travel_distance_mm))
        feedrates.append(feedrate)

        step = max(config.step_mm, (feedrate / 60.0) * config.min_segment_time_s)
        step = max(config.step_mm, round(step * 2.0) / 2.0)
        sample_position += step

    if not positions or positions[-1] < config.travel_distance_mm - 1e-6:
        velocity_mm_s = float(
            np.interp(config.travel_distance_mm, traj["position"], traj["velocity"])
        )
        feedrates.append(
            float(
                np.clip(velocity_mm_s * 60.0, config.v_min_mm_min, config.v_max_mm_min)
            )
        )
        positions.append(config.travel_distance_mm)

    return np.array(positions, dtype=np.float64), np.array(feedrates, dtype=np.float64)


def trajectory_to_gcode_leg(
    traj: dict[str, np.ndarray],
    leg_id: int,
    direction: str,
    leg_type: str,
    config: GcodeRunConfig,
    v_start_label: float = 0.0,
    v_end_label: float = 0.0,
) -> list[str]:
    """Convert a sampled trajectory into one semantic G-code leg."""
    lines: list[str] = []
    sample_x, feedrates = _adaptive_sample_points(traj, config)
    actual_v_start = float(feedrates[0]) if len(feedrates) > 0 else 0.0
    actual_v_end = float(feedrates[-1]) if len(feedrates) > 0 else 0.0
    if leg_type == "ramp":
        actual_v_start = v_start_label
        actual_v_end = v_end_label

    lines.append(
        f"; LEG_START id={leg_id} type={leg_type} direction={direction}"
        f" v_start={actual_v_start:.0f} v_end={actual_v_end:.0f}"
    )
    for index, sample_distance in enumerate(sample_x):
        if direction == "fwd":
            x_cmd = config.travel_min_mm + sample_distance
        else:
            x_cmd = config.travel_max_mm - sample_distance
        lines.append(f"G1 X{x_cmd:.2f} F{feedrates[index]:.0f}")

    end_x = config.travel_max_mm if direction == "fwd" else config.travel_min_mm
    lines.append(f"; LEG_END id={leg_id}")
    lines.append(
        f"G4 P{config.leg_pause_s:.1f} ; Dwell {config.leg_pause_s:.1f}s @ X={end_x:.2f}"
    )
    return lines


def estimate_trajectory_duration_s(
    traj: dict[str, np.ndarray],
    config: GcodeRunConfig,
) -> float:
    """Estimate physical execution time for one leg, including dwell."""
    return float(traj["time"][-1]) + config.leg_pause_s


def footer_lines() -> list[str]:
    """Return the G-code program footer."""
    return ["", "M2 ; End of program"]


def split_blocks(
    blocks: tuple[GcodeBlock, ...],
    description: str,
    config: GcodeRunConfig,
) -> tuple[GcodeFragment, ...]:
    """Split blocks into complete G-code files without cutting a movement leg."""
    if config.max_file_duration_hours <= 0:
        raise ValueError("max_file_duration_hours must be positive.")
    if config.max_file_line_count <= 0:
        raise ValueError("max_file_line_count must be positive.")

    header_line_count = len(gcode_header(description, config)) + 5
    footer_line_count = len(footer_lines())
    fragments: list[GcodeFragment] = []
    current_blocks: list[GcodeBlock] = []
    current_duration = 0.0
    current_body_lines = 0

    for block in blocks:
        projected_duration = current_duration + block.duration_s
        projected_lines = (
            header_line_count
            + current_body_lines
            + len(block.lines)
            + footer_line_count
        )
        exceeds_duration = projected_duration > config.max_file_duration_s
        exceeds_lines = projected_lines > config.max_file_line_count
        if current_blocks and (exceeds_duration or exceeds_lines):
            fragments.append(
                GcodeFragment(
                    blocks=tuple(current_blocks),
                    duration_s=current_duration,
                )
            )
            current_blocks = []
            current_duration = 0.0
            current_body_lines = 0

        current_blocks.append(block)
        current_duration += block.duration_s
        current_body_lines += len(block.lines)

    if current_blocks:
        fragments.append(
            GcodeFragment(blocks=tuple(current_blocks), duration_s=current_duration)
        )
    return tuple(fragments)


def render_fragment(
    description: str,
    fragment: GcodeFragment,
    part_index: int,
    total_parts: int,
    config: GcodeRunConfig,
) -> str:
    """Render one G-code fragment as a complete program."""
    lines = gcode_header(description, config)
    lines.extend(
        [
            f"; Fragment: {part_index:03d}/{total_parts:03d}",
            f"; Estimated duration: {fragment.duration_s / 3600.0:.3f} h",
            f"; Max fragment duration: {config.max_file_duration_hours:.3f} h",
            f"; Max fragment lines: {config.max_file_line_count}",
            "",
        ]
    )
    for block in fragment.blocks:
        lines.extend(block.lines)
    lines.extend(footer_lines())
    return "\n".join(lines)


def fragment_path(stem: str, fragment: GcodeFragment, index: int, total: int) -> str:
    """Build deterministic output filename for one fragment."""
    if total == 1:
        return f"{stem}.gcode"
    if stem == "bee_constant_velocity_sweep":
        first = fragment.blocks[0].start_label
        last = fragment.blocks[-1].end_label
        if first is not None and last is not None:
            return f"{stem}_part{index:03d}_{first:.0f}_{last:.0f}.gcode"
    return f"{stem}_part{index:03d}.gcode"


def write_fragmented_program(
    output_dir: Path,
    stem: str,
    description: str,
    blocks: tuple[GcodeBlock, ...],
    config: GcodeRunConfig,
) -> tuple[Path, ...]:
    """Write one logical program as one or more complete G-code files."""
    fragments = split_blocks(blocks, description, config)
    output_paths: list[Path] = []
    for index, fragment in enumerate(fragments, start=1):
        path = output_dir / fragment_path(stem, fragment, index, len(fragments))
        path.write_text(
            render_fragment(description, fragment, index, len(fragments), config),
            encoding="utf-8",
        )
        output_paths.append(path)
    return tuple(output_paths)


def remove_stale_generated_files(output_dir: Path) -> None:
    """Remove local files from previous generator runs before writing fragments."""
    patterns = (
        "bee_random_accel*.gcode",
        "bee_velocity_combos*.gcode",
        "bee_constant_velocity_sweep*.gcode",
    )
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink()


def estimate_roundtrips(config: GcodeRunConfig) -> int:
    """Estimate round-trips needed to approximate the requested random-run duration."""
    rng = np.random.default_rng(99)
    leg_times: list[float] = []
    for _ in range(config.monte_carlo_leg_count):
        leg_times.append(float(simulate_ou_trajectory(rng, config)["time"][-1]))
    mean_leg = float(np.mean(leg_times)) if leg_times else 0.0
    mean_roundtrip = mean_leg * 2.0 * config.cnc_overhead_factor
    target_seconds = config.desired_hours * 3600.0
    return max(1, int(np.ceil(target_seconds / max(mean_roundtrip, 1e-9))))


def build_random_roundtrip_blocks(config: GcodeRunConfig) -> tuple[GcodeBlock, ...]:
    """Build random acceleration blocks, keeping each round-trip indivisible."""
    n_roundtrips = estimate_roundtrips(config)
    rng = np.random.default_rng(config.seed)
    blocks: list[GcodeBlock] = []
    leg_id = 0
    for trip in range(n_roundtrips):
        block_lines = [f"; === ROUNDTRIP {trip + 1}/{n_roundtrips} ==="]
        block_duration = 0.0
        for direction in ("fwd", "rev"):
            traj = simulate_ou_trajectory(rng, config)
            leg_id += 1
            block_lines.extend(
                trajectory_to_gcode_leg(
                    traj,
                    leg_id,
                    direction,
                    "random",
                    config,
                )
            )
            block_duration += estimate_trajectory_duration_s(traj, config)
        block_lines.append("")
        blocks.append(GcodeBlock(lines=tuple(block_lines), duration_s=block_duration))
    return tuple(blocks)


def generate_random_gcode(config: GcodeRunConfig) -> str:
    """Generate the OU random-trajectory G-code program."""
    blocks = build_random_roundtrip_blocks(config)
    description = f"Bench random acceleration G-code - {len(blocks)} roundtrips"
    fragment = GcodeFragment(
        blocks=blocks,
        duration_s=sum(block.duration_s for block in blocks),
    )
    return render_fragment(description, fragment, 1, 1, config)


def generate_combo_gcode(config: GcodeRunConfig) -> str:
    """Generate the ramp-combination G-code program."""
    blocks = build_combo_leg_blocks(config)
    v_count = (
        int(round((config.v_max_mm_min - config.v_min_mm_min) / config.v_step_mm_min))
        + 1
    )
    description = (
        f"Bench velocity combos G-code - {len(blocks)} legs"
        f" ({v_count} velocities, step {config.v_step_mm_min:.0f} mm/min)"
    )
    fragment = GcodeFragment(
        blocks=blocks,
        duration_s=sum(block.duration_s for block in blocks),
    )
    return render_fragment(description, fragment, 1, 1, config)


def build_combo_leg_blocks(config: GcodeRunConfig) -> tuple[GcodeBlock, ...]:
    """Build ramp-combination leg blocks."""
    v_values = np.arange(
        config.v_min_mm_min,
        config.v_max_mm_min + 1e-6,
        config.v_step_mm_min,
    )
    combos = [(v_start, v_end) for v_start in v_values for v_end in v_values]
    blocks: list[GcodeBlock] = []
    for index, (v_start, v_end) in enumerate(combos, start=1):
        direction = "fwd" if index % 2 else "rev"
        traj = compute_ramp_trajectory(v_start, v_end, config)
        blocks.append(
            GcodeBlock(
                lines=tuple(
                    trajectory_to_gcode_leg(
                        traj,
                        index,
                        direction,
                        "ramp",
                        config,
                        v_start_label=v_start,
                        v_end_label=v_end,
                    )
                ),
                duration_s=estimate_trajectory_duration_s(traj, config),
            )
        )
    return tuple(blocks)


def generate_constant_velocity_sweep_gcode(config: GcodeRunConfig) -> str:
    """Generate the constant-velocity sweep G-code program."""
    blocks = build_constant_velocity_blocks(config)
    description = (
        f"Bench constant velocity sweep - {len(blocks)} legs"
        f" ({config.v_max_mm_min:.0f} to {config.v_min_mm_min:.0f} mm/min"
        f", decrement {config.sweep_decrement_mm_min:.0f} mm/min)"
    )
    fragment = GcodeFragment(
        blocks=blocks,
        duration_s=sum(block.duration_s for block in blocks),
    )
    return render_fragment(description, fragment, 1, 1, config)


def build_constant_velocity_blocks(
    config: GcodeRunConfig,
    v_from: float | None = None,
    v_to: float | None = None,
) -> tuple[GcodeBlock, ...]:
    """Build constant-velocity sweep leg blocks."""
    v_start = v_from if v_from is not None else config.v_max_mm_min
    v_end = v_to if v_to is not None else config.v_min_mm_min
    n_steps = int(round((v_start - v_end) / config.sweep_decrement_mm_min))
    v_values = [
        v_start - step_index * config.sweep_decrement_mm_min
        for step_index in range(n_steps + 1)
    ]
    blocks: list[GcodeBlock] = []
    for index, velocity in enumerate(v_values, start=1):
        direction = "fwd" if index % 2 else "rev"
        traj = compute_ramp_trajectory(velocity, velocity, config)
        blocks.append(
            GcodeBlock(
                lines=tuple(
                    trajectory_to_gcode_leg(
                        traj,
                        index,
                        direction,
                        "constant",
                        config,
                        v_start_label=velocity,
                        v_end_label=velocity,
                    )
                ),
                duration_s=estimate_trajectory_duration_s(traj, config),
                start_label=velocity,
                end_label=velocity,
            )
        )
    return tuple(blocks)


def write_gcode_files(config: GcodeRunConfig) -> tuple[Path, ...]:
    """Write all local G-code programs and return their generated paths."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_generated_files(config.output_dir)

    random_blocks = build_random_roundtrip_blocks(config)
    combo_blocks = build_combo_leg_blocks(config)
    constant_blocks = build_constant_velocity_blocks(config)

    random_paths = write_fragmented_program(
        config.output_dir,
        "bee_random_accel",
        f"Bench random acceleration G-code - {len(random_blocks)} roundtrips",
        random_blocks,
        config,
    )
    combo_paths = write_fragmented_program(
        config.output_dir,
        "bee_velocity_combos",
        (
            f"Bench velocity combos G-code - {len(combo_blocks)} legs"
            f" (step {config.v_step_mm_min:.0f} mm/min)"
        ),
        combo_blocks,
        config,
    )
    constant_paths = write_fragmented_program(
        config.output_dir,
        "bee_constant_velocity_sweep",
        (
            f"Bench constant velocity sweep - {len(constant_blocks)} legs"
            f" ({config.v_max_mm_min:.0f} to {config.v_min_mm_min:.0f} mm/min"
            f", decrement {config.sweep_decrement_mm_min:.0f} mm/min)"
        ),
        constant_blocks,
        config,
    )
    return random_paths + combo_paths + constant_paths


def main() -> None:
    """Generate all local G-code artifacts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    config = build_config(parse_args())
    output_paths = write_gcode_files(config)
    LOGGER.info(
        "Travel range: %.1f-%.1f mm | Center: %.1f mm",
        config.travel_min_mm,
        config.travel_max_mm,
        (config.travel_min_mm + config.travel_max_mm) / 2.0,
    )
    LOGGER.info(
        "Fragmentation: max %.2fh or %d lines per file",
        config.max_file_duration_hours,
        config.max_file_line_count,
    )
    for output_path in output_paths:
        LOGGER.info("Saved %s", output_path)


if __name__ == "__main__":
    main()
