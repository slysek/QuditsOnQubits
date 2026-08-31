from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from pathlib import Path
import sys

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.modules.pop("qudits_on_qubits", None)


import qudits_on_qubits.bell_measurements as bell_measurements
from qudits_on_qubits.bell_measurements import postprocessing
from qudits_on_qubits.bell_measurements.postprocessing import (
    compute_bell_value_from_counts,
    compute_complex_expectation,
    leakage_rate,
)


_SETTING = ("A0",)
_QUTRIT_BITS_BY_SETTING = {_SETTING: ((0, 1),)}
_OUTCOME_MAP = {0: 0, 1: 1, 2: 2, 3: None}


def test_complex_expectation_accepts_signed_float_weights():
    value = compute_complex_expectation(
        {"00": 2.5, "01": -0.5},
        powers=(1,),
        qutrit_bit_indices=((0, 1),),
        bit_order="left-to-right",
    )

    expected = (2.5 + (-0.5) * complex(-0.5, math.sqrt(3) / 2)) / 2.0
    assert value == pytest.approx(expected)


def test_complex_expectation_returns_zero_for_zero_signed_denominator():
    assert compute_complex_expectation(
        {"00": 1.0, "01": -1.0},
        powers=(1,),
        qutrit_bit_indices=((0, 1),),
        bit_order="left-to-right",
    ) == 0j


def test_bell_value_accepts_weighted_counts():
    value = compute_bell_value_from_counts(
        {("A",): {"00": 3.0, "01": -1.0}},
        terms=({"settings": ("A",), "powers": (1,), "coeff": 2},),
        qutrit_bit_indices_by_setting={("A",): ((0, 1),)},
        bit_order="left-to-right",
    )

    expected = 2 * (3.0 - complex(-0.5, math.sqrt(3) / 2)) / 2.0
    assert value == pytest.approx(expected)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf")])
def test_complex_expectation_rejects_non_finite_weights(weight):
    with pytest.raises(ValueError, match="finite"):
        compute_complex_expectation(
            {"00": weight},
            powers=(1,),
            qutrit_bit_indices=((0, 1),),
            bit_order="left-to-right",
        )


def test_leakage_rate_rejects_negative_weighted_counts():
    with pytest.raises(ValueError, match="non-negative"):
        leakage_rate({"11": -0.5}, ((0, 1),), bit_order="left-to-right")


def test_invalid_codeword_statistics_reports_raw_count_weights():
    result = postprocessing.invalid_codeword_statistics(
        {_SETTING: {"00": 6, "01": 2, "11": 2}},
        _QUTRIT_BITS_BY_SETTING,
        bit_order="left-to-right",
        outcome_map=_OUTCOME_MAP,
        require_non_negative=True,
    )

    assert result == postprocessing.InvalidCodewordStatistics(
        total_weight=10.0,
        accepted_weight=8.0,
        invalid_weight=2.0,
        invalid_fraction=0.2,
    )


def test_invalid_codeword_statistics_aggregates_partial_invalidity_across_settings():
    two_qutrit_setting = ("A0", "B0")
    one_qutrit_setting = ("A1",)
    result = postprocessing.invalid_codeword_statistics(
        {
            two_qutrit_setting: {"0000": 3, "1100": 2},
            one_qutrit_setting: {"00": 5},
        },
        {
            two_qutrit_setting: ((0, 1), (2, 3)),
            one_qutrit_setting: ((0, 1),),
        },
        bit_order="left-to-right",
        outcome_map=_OUTCOME_MAP,
        require_non_negative=True,
    )

    assert result.total_weight == 10.0
    assert result.accepted_weight == 8.0
    assert result.invalid_weight == 2.0
    assert result.invalid_fraction == 0.2


def test_invalid_codeword_statistics_is_immutable():
    result = postprocessing.InvalidCodewordStatistics(1.0, 1.0, 0.0, 0.0)

    with pytest.raises(FrozenInstanceError):
        result.total_weight = 2.0


def test_invalid_codeword_statistics_accepts_signed_quasi_weights():
    result = postprocessing.invalid_codeword_statistics(
        {_SETTING: {"00": 0.9, "11": -0.1, "10": 0.2}},
        _QUTRIT_BITS_BY_SETTING,
        bit_order="left-to-right",
        outcome_map=_OUTCOME_MAP,
        require_non_negative=False,
    )

    assert result.total_weight == pytest.approx(1.0)
    assert result.invalid_weight == pytest.approx(-0.1)
    assert result.accepted_weight == pytest.approx(1.1)
    assert result.invalid_fraction == pytest.approx(-0.1)


def test_invalid_codeword_statistics_rejects_signed_weights_when_required():
    with pytest.raises(ValueError, match="non-negative"):
        postprocessing.invalid_codeword_statistics(
            {_SETTING: {"11": -0.1}},
            _QUTRIT_BITS_BY_SETTING,
            bit_order="left-to-right",
            outcome_map=_OUTCOME_MAP,
            require_non_negative=True,
        )


def test_invalid_codeword_statistics_defines_zero_fraction_for_zero_total():
    result = postprocessing.invalid_codeword_statistics(
        {_SETTING: {"00": 1.0, "11": -1.0}},
        _QUTRIT_BITS_BY_SETTING,
        bit_order="left-to-right",
        outcome_map=_OUTCOME_MAP,
        require_non_negative=False,
    )

    assert result.total_weight == 0.0
    assert result.accepted_weight == 1.0
    assert result.invalid_weight == -1.0
    assert result.invalid_fraction == 0.0


@pytest.mark.parametrize(
    "weight",
    [True, 1 + 0j, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_codeword_statistics_rejects_invalid_weights(weight):
    with pytest.raises(ValueError, match="finite real"):
        postprocessing.invalid_codeword_statistics(
            {_SETTING: {"00": weight}},
            _QUTRIT_BITS_BY_SETTING,
            bit_order="left-to-right",
            outcome_map=_OUTCOME_MAP,
            require_non_negative=False,
        )


@pytest.mark.parametrize("require_non_negative", [None, 0, 1, "false"])
def test_invalid_codeword_statistics_requires_boolean_policy(require_non_negative):
    with pytest.raises(ValueError, match="require_non_negative"):
        postprocessing.invalid_codeword_statistics(
            {_SETTING: {"00": 1}},
            _QUTRIT_BITS_BY_SETTING,
            bit_order="left-to-right",
            outcome_map=_OUTCOME_MAP,
            require_non_negative=require_non_negative,
        )


def test_invalid_codeword_statistics_is_exported_from_public_package():
    assert (
        bell_measurements.InvalidCodewordStatistics
        is postprocessing.InvalidCodewordStatistics
    )
    assert (
        bell_measurements.invalid_codeword_statistics
        is postprocessing.invalid_codeword_statistics
    )


@pytest.mark.parametrize("count", [-1, 1.0, True])
def test_leakage_rate_preserves_non_negative_integer_count_contract(count):
    with pytest.raises(ValueError, match="non-negative integers"):
        leakage_rate({"11": count}, ((0, 1),), bit_order="left-to-right")
