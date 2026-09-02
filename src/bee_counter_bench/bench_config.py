"""Configuration adapter for G-code generation and optional acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bee_counter_bench.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"
HARDWARE_PATH = PROJECT_ROOT / "configs" / "hardware.yaml"
SETTINGS_PATH = EXPERIMENT_PATH


@dataclass(frozen=True)
class PathConfig:
    recordings_dir: Path
    gcode_output_dir: Path


@dataclass(frozen=True)
class GeometryConfig:
    travel_min_mm: float
    travel_max_mm: float
    sensor_center_mm: float | None
    zone_half_width_mm: float


@dataclass(frozen=True)
class DaqConfig:
    device_name: str
    module: str
    channel: str
    channels: tuple[str, ...]
    sample_rate_hz: int
    voltage_min: float
    voltage_max: float
    terminal_config: str


@dataclass(frozen=True)
class StepPositionConfig:
    steps_per_mm: float
    voltage_threshold: float
    min_edge_gap: int
    min_pulse_samples: int
    dir_vote_samples: int
    trips_per_chunk: int


@dataclass(frozen=True)
class GcodeConfig:
    machine_v_max_mm_min: float
    machine_a_max_mm_s2: float
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


@dataclass(frozen=True)
class RandomProcessConfig:
    ou_alpha_v0: float
    ou_lambda_range: tuple[float, float]
    ou_sigma_range: tuple[float, float]
    ou_kappa: float
    ou_margin_frac: float
    ou_bee_a_max_mm_s2: float
    ou_accel_lpf_cutoff_hz: float


@dataclass(frozen=True)
class AppConfig:
    paths: PathConfig
    geometry: GeometryConfig
    daq: DaqConfig
    step_position: StepPositionConfig
    gcode: GcodeConfig
    synthetic_generation: RandomProcessConfig


def load_app_config(settings_path: Path = EXPERIMENT_PATH) -> AppConfig:
    """Load the bench configuration from the public experiment files."""
    experiment_path = Path(settings_path).resolve()
    experiment = load_yaml(experiment_path)
    hardware = load_yaml(experiment_path.parent / "hardware.yaml")
    paths = experiment["paths"]
    geometry = experiment["geometry"]
    generation = experiment["gcode_generation"]
    machine = generation["machine"]
    path = generation["path"]
    motion = generation["motion"]
    output = generation["output"]
    estimation = generation["duration_estimation"]
    random_process = generation["random_process"]
    acquisition = hardware["acquisition"]
    cnc = hardware["cnc"]
    project_root = experiment_path.parent.parent

    return AppConfig(
        paths=PathConfig(
            recordings_dir=(project_root / paths["raw_data"]).resolve(),
            gcode_output_dir=(project_root / paths["gcode"]).resolve(),
        ),
        geometry=GeometryConfig(
            travel_min_mm=float(geometry["travel_min_mm"]),
            travel_max_mm=float(geometry["travel_max_mm"]),
            sensor_center_mm=(
                float(geometry["roi_low_mm"] + geometry["roi_high_mm"]) / 2.0
            ),
            zone_half_width_mm=(
                float(geometry["roi_high_mm"] - geometry["roi_low_mm"]) / 2.0
            ),
        ),
        daq=DaqConfig(
            device_name=str(acquisition["device_name"]),
            module=str(acquisition["module_slot"]),
            channel=str(acquisition["channel"]),
            channels=tuple(str(value) for value in acquisition["channels"]),
            sample_rate_hz=int(acquisition["sample_rate_hz"]),
            voltage_min=float(acquisition["input_range_v"][0]),
            voltage_max=float(acquisition["input_range_v"][1]),
            terminal_config=str(acquisition["terminal_configuration"]),
        ),
        step_position=StepPositionConfig(
            steps_per_mm=float(cnc["steps_per_mm"]),
            voltage_threshold=float(cnc["step_detection_threshold_v"]),
            min_edge_gap=int(cnc["minimum_edge_gap_samples"]),
            min_pulse_samples=int(cnc["minimum_pulse_samples"]),
            dir_vote_samples=int(cnc["direction_vote_samples"]),
            trips_per_chunk=int(cnc["trips_per_chunk"]),
        ),
        gcode=GcodeConfig(
            machine_v_max_mm_min=float(machine["maximum_velocity_mm_min"]),
            machine_a_max_mm_s2=float(machine["maximum_acceleration_mm_s2"]),
            step_mm=float(path["step_mm"]),
            min_segment_time_s=float(path["minimum_segment_time_s"]),
            dt_s=float(path["integration_step_s"]),
            v_min_mm_min=float(motion["minimum_velocity_mm_min"]),
            v_max_mm_min=float(motion["maximum_velocity_mm_min"]),
            leg_pause_s=float(motion["leg_pause_s"]),
            v_step_mm_min=float(motion["velocity_step_mm_min"]),
            sweep_decrement_mm_min=float(motion["sweep_decrement_mm_min"]),
            desired_hours=float(motion["random_program_hours"]),
            seed=int(motion["seed"]),
            max_file_duration_hours=float(output["maximum_file_hours"]),
            max_file_line_count=int(output["maximum_file_lines"]),
            monte_carlo_leg_count=int(estimation["monte_carlo_legs"]),
            cnc_overhead_factor=float(estimation["cnc_overhead_factor"]),
        ),
        synthetic_generation=RandomProcessConfig(
            ou_alpha_v0=float(random_process["initial_velocity_exponent"]),
            ou_lambda_range=tuple(
                float(value) for value in random_process["reversion_rate_range"]
            ),
            ou_sigma_range=tuple(
                float(value) for value in random_process["noise_scale_range"]
            ),
            ou_kappa=float(random_process["boundary_gain"]),
            ou_margin_frac=float(random_process["boundary_margin_fraction"]),
            ou_bee_a_max_mm_s2=float(random_process["maximum_acceleration_mm_s2"]),
            ou_accel_lpf_cutoff_hz=float(random_process["acceleration_low_pass_hz"]),
        ),
    )
