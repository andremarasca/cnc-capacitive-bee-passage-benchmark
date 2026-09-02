"""Numerical contract for the frozen physically justified filter."""

import pytest

from bee_counter_bench.config import load_config
from bee_counter_bench.reporting.filter_response import response_statistics


def test_filter_response_matches_frozen_engineering_values() -> None:
    report = response_statistics(load_config())

    assert report["minus_3_db_band_hz"] == pytest.approx([0.1483, 3.7087], abs=0.0002)
    expected_attenuation = {
        "500_mm_min_characteristic": 1.334,
        "3500_mm_min_characteristic": 0.256,
        "3500_mm_min_second_multiple": 1.696,
        "3500_mm_min_third_multiple": 4.438,
    }
    for marker, expected in expected_attenuation.items():
        assert report["markers"][marker]["attenuation_db"] == pytest.approx(
            expected, abs=0.002
        )
