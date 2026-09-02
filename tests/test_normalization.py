"""Tests for population and strictly causal adaptive normalization."""

from pathlib import Path

import numpy as np

from bee_counter_bench.config import NormalizationConfig, load_normalization_config
from bee_counter_bench.data.normalization import (
    causal_adaptive_normalize,
    normalize_session_signal,
)


def _causal_config(**overrides: float) -> NormalizationConfig:
    values = {
        "schema_version": 1,
        "variant": "test_causal",
        "method": "causal_adaptive",
        "sample_rate_hz": 30.0,
        "location_time_constant_seconds": 60.0,
        "scale_time_constant_seconds": 60.0,
        "adaptation_clip_z": 3.0,
        "minimum_scale_ratio": 0.25,
        "maximum_scale_ratio": 8.0,
        "state_scope": "session",
    }
    values.update(overrides)
    return NormalizationConfig(**values)  # type: ignore[arg-type]


def test_versioned_causal_configuration_is_complete() -> None:
    path = Path("configs/normalization_causal_tau60.yaml")

    config = load_normalization_config(path)

    assert config.method == "causal_adaptive"
    assert config.sample_rate_hz == 30.0
    assert config.location_time_constant_seconds == 60.0
    assert config.scale_time_constant_seconds == 60.0
    assert config.state_scope == "session"


def test_future_values_do_not_change_causal_prefix() -> None:
    prefix = np.linspace(-1.0, 1.0, 120, dtype=np.float64)
    first = np.concatenate((prefix, np.full(60, 100.0)))
    second = np.concatenate((prefix, np.full(60, -100.0)))

    first_output, _ = causal_adaptive_normalize(
        first,
        initial_location=0.0,
        initial_scale=1.0,
        config=_causal_config(),
    )
    second_output, _ = causal_adaptive_normalize(
        second,
        initial_location=0.0,
        initial_scale=1.0,
        config=_causal_config(),
    )

    np.testing.assert_array_equal(first_output[: len(prefix)], second_output[: len(prefix)])


def test_current_sample_uses_previous_state() -> None:
    output, _ = causal_adaptive_normalize(
        np.asarray([12.0, 12.0]),
        initial_location=10.0,
        initial_scale=2.0,
        config=_causal_config(),
    )

    assert output[0] == 1.0
    assert output[1] < output[0]


def test_each_session_restarts_from_training_statistics() -> None:
    signal = np.asarray([3.0, 4.0, 5.0])
    first, _ = causal_adaptive_normalize(
        signal,
        initial_location=1.0,
        initial_scale=2.0,
        config=_causal_config(),
    )
    second, _ = causal_adaptive_normalize(
        signal,
        initial_location=1.0,
        initial_scale=2.0,
        config=_causal_config(),
    )

    np.testing.assert_array_equal(first, second)


def test_scale_limits_and_clipping_keep_outputs_finite() -> None:
    signal = np.concatenate(
        (np.zeros(20_000, dtype=np.float64), np.full(20_000, 1.0e12))
    )

    output, diagnostics = causal_adaptive_normalize(
        signal,
        initial_location=0.0,
        initial_scale=2.0,
        config=_causal_config(
            location_time_constant_seconds=0.1,
            scale_time_constant_seconds=0.1,
        ),
    )

    assert np.isfinite(output).all()
    assert diagnostics["final_scale"] >= 0.5
    assert diagnostics["final_scale"] <= 16.0
    assert diagnostics["maximum_scale_seen"] <= 16.0


def test_population_normalization_matches_original_expression() -> None:
    signal = np.asarray([-3.0, 1.0, 5.0], dtype=np.float32)
    expected = (signal - 1.0) / 2.0

    actual = normalize_session_signal(
        signal,
        training_mean=1.0,
        training_standard_deviation=2.0,
    )

    np.testing.assert_array_equal(actual, expected)
