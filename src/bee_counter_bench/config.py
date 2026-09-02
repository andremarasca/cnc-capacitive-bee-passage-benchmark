"""Typed loading of the frozen experiment and model configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml


@dataclass(frozen=True)
class PathsConfig:
    recordings_dir: Path
    datasets_dir: Path
    models_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class GeometryConfig:
    zone_low_mm: float
    zone_high_mm: float


@dataclass(frozen=True)
class BesselConfig:
    mode: str
    order: int
    low_cutoff_hz: float | None
    high_cutoff_hz: float
    norm: str


@dataclass(frozen=True)
class PreprocessingConfig:
    source_sample_rate_hz: int
    output_sample_rate_hz: int
    initialization: str
    state_scope: str
    bessel: BesselConfig


@dataclass(frozen=True)
class LabelsConfig:
    out_id: int
    left_id: int
    right_id: int
    ignore_id: int = -100


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int
    num_classes: int
    feature_maps: int
    prediction_generation_layers: int
    refinement_layers: int
    refinement_stages: int
    dropout: float
    causal: bool
    expected_trainable_parameters: int
    receptive_field_frames: int
    future_context_frames: int


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    adam_betas: tuple[float, float]
    adam_epsilon: float
    weight_decay: float
    smoothing_weight: float
    smoothing_tau: float
    seed: int
    patience: int


@dataclass(frozen=True)
class EvaluationConfig:
    segment_iou_thresholds: tuple[float, ...]


@dataclass(frozen=True)
class StreamingConfig:
    block_size_frames: int
    persist_state: bool


@dataclass(frozen=True)
class NormalizationConfig:
    """Signal normalization applied independently to each physical session."""

    schema_version: int
    variant: str
    method: Literal["training_population", "causal_adaptive"]
    sample_rate_hz: float
    location_time_constant_seconds: float | None = None
    scale_time_constant_seconds: float | None = None
    adaptation_clip_z: float | None = None
    minimum_scale_ratio: float | None = None
    maximum_scale_ratio: float | None = None
    state_scope: Literal["session"] = "session"


@dataclass(frozen=True)
class AppConfig:
    schema_version: int
    paths: PathsConfig
    geometry: GeometryConfig
    preprocessing: PreprocessingConfig
    labels: LabelsConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    streaming: StreamingConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping without applying implicit defaults."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    with source.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a mapping: {source}")
    return payload


def _section(root: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = root.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section {key!r} must be a mapping.")
    return value


def _required(root: Mapping[str, Any], key: str) -> Any:
    if key not in root:
        raise ValueError(f"Missing required configuration key {key!r}.")
    return root[key]


def _resolve(project_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_config(
    path: str | Path = "configs/experiment.yaml",
    *,
    model_path: str | Path | None = None,
    seed: int | None = None,
) -> AppConfig:
    """Resolve the public configuration into the compact runtime contract."""
    experiment_path = Path(path).resolve()
    experiment = load_yaml(experiment_path)
    project_root = experiment_path.parent.parent
    resolved_model_path = (
        experiment_path.parent / "model.yaml"
        if model_path is None
        else Path(model_path).resolve()
    )
    model_document = load_yaml(resolved_model_path)

    paths = _section(experiment, "paths")
    geometry = _section(experiment, "geometry")
    preprocessing = _section(experiment, "preprocessing")
    filter_config = _section(preprocessing, "filter")
    labels = _section(_section(experiment, "labels"), "classes")
    evaluation = _section(experiment, "evaluation")
    architecture = _section(model_document, "architecture")
    loss = _section(model_document, "loss")
    optimizer = _section(model_document, "optimizer")
    training = _section(model_document, "training")
    seeds = _section(experiment, "training")

    config = AppConfig(
        schema_version=int(_required(experiment, "schema_version")),
        paths=PathsConfig(
            recordings_dir=_resolve(project_root, _required(paths, "raw_data")),
            datasets_dir=_resolve(project_root, _required(paths, "outputs"))
            / "processed",
            models_dir=_resolve(project_root, _required(paths, "outputs")) / "models",
            reports_dir=_resolve(project_root, _required(paths, "outputs")) / "reports",
        ),
        geometry=GeometryConfig(
            zone_low_mm=float(_required(geometry, "roi_low_mm")),
            zone_high_mm=float(_required(geometry, "roi_high_mm")),
        ),
        preprocessing=PreprocessingConfig(
            source_sample_rate_hz=int(
                _required(preprocessing, "source_sample_rate_hz")
            ),
            output_sample_rate_hz=int(
                _required(preprocessing, "output_sample_rate_hz")
            ),
            initialization=str(_required(filter_config, "initialization")),
            state_scope=str(_required(filter_config, "state_scope")).replace(
                "physical_", ""
            ),
            bessel=BesselConfig(
                mode=str(_required(filter_config, "mode")),
                order=int(_required(filter_config, "prototype_order")),
                low_cutoff_hz=float(_required(filter_config, "low_critical_hz")),
                high_cutoff_hz=float(_required(filter_config, "high_critical_hz")),
                norm=str(_required(filter_config, "norm")),
            ),
        ),
        labels=LabelsConfig(
            out_id=int(_required(labels, "OUT")),
            left_id=int(_required(labels, "LEFT")),
            right_id=int(_required(labels, "RIGHT")),
        ),
        model=ModelConfig(
            input_channels=int(_required(architecture, "input_channels")),
            num_classes=len(_required(architecture, "classes")),
            feature_maps=int(_required(architecture, "feature_maps")),
            prediction_generation_layers=int(
                _required(architecture, "prediction_generation_layers")
            ),
            refinement_layers=int(
                _required(architecture, "refinement_layers_per_stage")
            ),
            refinement_stages=int(_required(architecture, "refinement_stages")),
            dropout=float(_required(architecture, "dropout")),
            causal=bool(_required(architecture, "causal")),
            expected_trainable_parameters=int(
                _required(architecture, "expected_trainable_parameters")
            ),
            receptive_field_frames=int(
                _required(architecture, "receptive_field_frames")
            ),
            future_context_frames=int(
                _required(architecture, "context_frames_each_side")
            ),
        ),
        training=TrainingConfig(
            epochs=int(_required(training, "maximum_epochs")),
            batch_size=int(_required(training, "batch_size")),
            learning_rate=float(_required(optimizer, "learning_rate")),
            adam_betas=tuple(float(value) for value in _required(optimizer, "betas")),  # type: ignore[arg-type]
            adam_epsilon=float(_required(optimizer, "epsilon")),
            weight_decay=float(_required(optimizer, "weight_decay")),
            smoothing_weight=float(_required(loss, "smoothing_weight")),
            smoothing_tau=float(_required(loss, "smoothing_tau")),
            seed=int(seed if seed is not None else _required(seeds, "default_seed")),
            patience=int(_required(training, "early_stopping_patience")),
        ),
        evaluation=EvaluationConfig(
            segment_iou_thresholds=tuple(
                float(value) for value in _required(evaluation, "event_iou_thresholds")
            )
        ),
        streaming=StreamingConfig(
            block_size_frames=int(_required(preprocessing, "output_sample_rate_hz")),
            persist_state=True,
        ),
    )
    return _validate(config)


def load_normalization_config(path: str | Path) -> NormalizationConfig:
    """Load one explicit normalization variant without changing defaults."""
    payload = load_yaml(path)
    config = NormalizationConfig(
        schema_version=int(_required(payload, "schema_version")),
        variant=str(_required(payload, "variant")),
        method=str(_required(payload, "method")),  # type: ignore[arg-type]
        sample_rate_hz=float(_required(payload, "sample_rate_hz")),
        location_time_constant_seconds=_optional_float(
            payload, "location_time_constant_seconds"
        ),
        scale_time_constant_seconds=_optional_float(
            payload, "scale_time_constant_seconds"
        ),
        adaptation_clip_z=_optional_float(payload, "adaptation_clip_z"),
        minimum_scale_ratio=_optional_float(payload, "minimum_scale_ratio"),
        maximum_scale_ratio=_optional_float(payload, "maximum_scale_ratio"),
        state_scope=str(_required(payload, "state_scope")),  # type: ignore[arg-type]
    )
    return _validate_normalization(config)


def _optional_float(root: Mapping[str, Any], key: str) -> float | None:
    value = root.get(key)
    return None if value is None else float(value)


def _validate_normalization(config: NormalizationConfig) -> NormalizationConfig:
    if config.schema_version != 1:
        raise ValueError(
            f"Unsupported normalization schema: {config.schema_version}."
        )
    if config.method not in {"training_population", "causal_adaptive"}:
        raise ValueError(f"Unsupported normalization method: {config.method}.")
    if config.sample_rate_hz <= 0.0:
        raise ValueError("Normalization sample rate must be positive.")
    if config.state_scope != "session":
        raise ValueError("Normalization state must be reset for every session.")
    if config.method == "training_population":
        return config
    required = (
        config.location_time_constant_seconds,
        config.scale_time_constant_seconds,
        config.adaptation_clip_z,
        config.minimum_scale_ratio,
        config.maximum_scale_ratio,
    )
    if any(value is None or value <= 0.0 for value in required):
        raise ValueError("Causal normalization parameters must be positive.")
    if float(config.maximum_scale_ratio) <= float(config.minimum_scale_ratio):
        raise ValueError("Maximum scale ratio must exceed minimum scale ratio.")
    return config


def _validate(config: AppConfig) -> AppConfig:
    if config.schema_version != 1:
        raise ValueError(f"Unsupported configuration schema: {config.schema_version}.")
    if config.geometry.zone_low_mm >= config.geometry.zone_high_mm:
        raise ValueError("ROI lower bound must be below its upper bound.")
    preprocessing = config.preprocessing
    if (
        not 0
        < preprocessing.output_sample_rate_hz
        < preprocessing.source_sample_rate_hz
    ):
        raise ValueError("Output sample rate must be below the source sample rate.")
    if preprocessing.state_scope != "session":
        raise ValueError("The final protocol requires session-scoped filter state.")
    if preprocessing.bessel.mode != "bandpass":
        raise ValueError("The frozen filter must be a band-pass filter.")
    if not (
        0
        < float(preprocessing.bessel.low_cutoff_hz or 0.0)
        < preprocessing.bessel.high_cutoff_hz
        < preprocessing.source_sample_rate_hz / 2.0
    ):
        raise ValueError("Invalid band-pass critical frequencies.")
    if (config.labels.out_id, config.labels.left_id, config.labels.right_id) != (
        0,
        1,
        2,
    ):
        raise ValueError("Class identifiers are fixed as OUT=0, LEFT=1, RIGHT=2.")
    if config.model.num_classes != 3 or config.model.input_channels != 1:
        raise ValueError("The model requires one input channel and three classes.")
    from bee_counter_bench.models.receptive_field import mstcnpp_receptive_field

    receptive_field = mstcnpp_receptive_field(
        config.model.prediction_generation_layers,
        config.model.refinement_layers,
        config.model.refinement_stages,
        float(config.preprocessing.output_sample_rate_hz),
        causal=config.model.causal,
    )
    if receptive_field.total_frames != config.model.receptive_field_frames:
        raise ValueError("Configured and analytical receptive fields differ.")
    if receptive_field.future_frames != config.model.future_context_frames:
        raise ValueError("Configured and analytical future contexts differ.")
    if config.model.causal:
        raise ValueError("The implemented MS-TCN++ requires symmetric padding.")
    if len(config.training.adam_betas) != 2:
        raise ValueError("Adam requires exactly two beta coefficients.")
    thresholds = config.evaluation.segment_iou_thresholds
    if tuple(sorted(set(thresholds))) != thresholds:
        raise ValueError("Event IoU thresholds must be sorted and unique.")
    return config


def with_seed(config: AppConfig, seed: int) -> AppConfig:
    """Return an immutable runtime configuration with a different train seed."""
    return replace(config, training=replace(config.training, seed=int(seed)))


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """Convert a resolved runtime configuration to YAML-safe primitives."""

    project_root = config.paths.recordings_dir.parents[1]

    def normalize(value: Any) -> Any:
        if isinstance(value, Path):
            try:
                return value.relative_to(project_root).as_posix()
            except ValueError:
                return value.as_posix()
        if isinstance(value, tuple):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        return value

    return normalize(asdict(config))


def normalization_config_to_dict(
    config: NormalizationConfig | None,
) -> dict[str, Any]:
    """Return the exact runtime normalization contract for provenance."""
    if config is None:
        return {
            "schema_version": 1,
            "variant": "training_population",
            "method": "training_population",
        }
    return asdict(config)


def save_resolved_config(config: AppConfig, path: str | Path) -> Path:
    """Persist the exact compact runtime configuration used by one run."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(
            config_to_dict(config),
            stream,
            sort_keys=False,
            allow_unicode=False,
        )
    return destination
